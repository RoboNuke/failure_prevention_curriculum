from __future__ import annotations

import collections
import glob
import itertools
import os
import re
from typing import Any

import gymnasium
from packaging import version

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.utils import ScopedTimer
from skrl.utils.tensorboard import SummaryWriter

from configs.manager.preprocessor_registry import resolve_preprocessor
from configs.manager.sac_cfg import SAC_CFG
from models.block_simba import (
    assign_block_slice,
    merge_optimizer_states,
    slice_block_state_dict,
    slice_optimizer_state,
)
from models.preprocessor_wrapper import PerAgentPreprocessorWrapper


class SAC(Agent):
    def __init__(
        self,
        *,
        models: dict[str, Model],
        memory: Memory | None = None,
        observation_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg: SAC_CFG | dict = {},
        num_agents: int = 1,
    ) -> None:
        """Soft Actor-Critic (SAC) with per-agent block-parallel independence.

        Each agent owns a fixed env partition (envs ``[i*epa, (i+1)*epa)``); each has
        its own learnable entropy coefficient and its own tensorboard writer. No
        metrics are aggregated across agents.

        :param models: Agent's models.
        :param memory: Memory to storage agent's data and environment transitions.
            For ``num_agents > 1`` this should be a ``MultiRandomMemory`` so that
            sampled mini-batches preserve the per-agent env partitioning.
        :param observation_space: Observation space.
        :param action_space: Action space.
        :param device: Data allocation and computation device.
        :param cfg: Agent's configuration.
        :param num_agents: Number of block-parallel agents.
        """
        self.cfg: SAC_CFG
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=SAC_CFG(**cfg) if isinstance(cfg, dict) else cfg,
        )

        self.num_agents = num_agents

        # models — all five required, no silent fallback to None
        required = ("policy", "critic_1", "critic_2", "target_critic_1", "target_critic_2")
        missing = [k for k in required if k not in self.models or self.models[k] is None]
        if missing:
            raise ValueError(f"SAC requires models {required}; missing or None: {missing}")
        self.policy = self.models["policy"]
        self.critic_1 = self.models["critic_1"]
        self.critic_2 = self.models["critic_2"]
        self.target_critic_1 = self.models["target_critic_1"]
        self.target_critic_2 = self.models["target_critic_2"]

        # checkpointing is handled per-agent by write_checkpoint()/load() — we don't
        # populate self.checkpoint_modules so the base Agent's bundled save path stays out.

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
            if self.critic_1 is not None:
                self.critic_1.broadcast_parameters()
            if self.critic_2 is not None:
                self.critic_2.broadcast_parameters()

        # set up automatic mixed precision
        self._device_type = torch.device(self.device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self.cfg.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        # entropy — per-agent (N, 1) coefficient. Adam state is element-wise so a single
        # optimizer over the (N, 1) parameter is fully independent across agents.
        self._entropy_coefficient = torch.full(
            (num_agents, 1), float(self.cfg.initial_entropy_value), device=self.device
        )
        if self.cfg.learn_entropy:
            # target_entropy is action-space dependent; same scalar across agents.
            self._target_entropy = self.cfg.target_entropy
            if self._target_entropy is None:
                if issubclass(type(self.action_space), gymnasium.spaces.Box):
                    self._target_entropy = -np.prod(self.action_space.shape).astype(np.float32)
                elif issubclass(type(self.action_space), gymnasium.spaces.Discrete):
                    self._target_entropy = -self.action_space.n
                else:
                    self._target_entropy = 0

            self.log_entropy_coefficient = torch.log(self._entropy_coefficient.clone()).requires_grad_(True)
            self.entropy_optimizer = torch.optim.Adam(
                [self.log_entropy_coefficient], lr=self.cfg.learning_rate[2]
            )

        # set up optimizers and learning rate schedulers
        if self.policy is not None and self.critic_1 is not None and self.critic_2 is not None:
            self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.learning_rate[0])
            self.critic_optimizer = torch.optim.Adam(
                itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()),
                lr=self.cfg.learning_rate[1],
            )
            self.policy_scheduler = self.cfg.learning_rate_scheduler[0]
            self.critic_scheduler = self.cfg.learning_rate_scheduler[1]
            if self.policy_scheduler is not None:
                self.policy_scheduler = self.cfg.learning_rate_scheduler[0](
                    self.policy_optimizer, **self.cfg.learning_rate_scheduler_kwargs[0]
                )
            if self.critic_scheduler is not None:
                self.critic_scheduler = self.cfg.learning_rate_scheduler[1](
                    self.critic_optimizer, **self.cfg.learning_rate_scheduler_kwargs[1]
                )

        # set up target networks
        if self.target_critic_1 is not None and self.target_critic_2 is not None:
            self.target_critic_1.freeze_parameters(True)
            self.target_critic_2.freeze_parameters(True)
            self.target_critic_1.update_parameters(self.critic_1, polyak=1)
            self.target_critic_2.update_parameters(self.critic_2, polyak=1)

        # set up observation preprocessor.
        # `cfg.observation_preprocessor` may be a class, a registered string name (from
        # YAML), or None. Resolve to a class first; then build N independent instances
        # and wrap them so per-agent batch slices route to per-agent preprocessors.
        preproc_cls = resolve_preprocessor(self.cfg.observation_preprocessor)
        if preproc_cls is not None:
            preproc_list = [
                preproc_cls(**self.cfg.observation_preprocessor_kwargs)
                for _ in range(num_agents)
            ]
            self._observation_preprocessor = PerAgentPreprocessorWrapper(num_agents, preproc_list)
        else:
            self._observation_preprocessor = self._empty_preprocessor

        # per-agent tracking buffers (writers created in init() once experiment_dir is set)
        self.per_agent_writers: list[SummaryWriter] = []
        self.per_agent_tracking: list[collections.defaultdict] = []
        self._per_agent_track_rewards: list[collections.deque] = []
        self._per_agent_track_timesteps: list[collections.deque] = []
        self._per_agent_track_success: list[collections.deque] = []

        # Success-prediction config (read once into instance attrs for fast access).
        self.predict_success: bool = bool(getattr(self.cfg, "predict_success", False))
        self.success_bce_weight: float = float(getattr(self.cfg, "success_bce_weight", 0.0))
        self.success_info_key: str = str(getattr(self.cfg, "success_info_key", "is_success"))

        # Per-env "success seen this trajectory so far" — OR-accumulated each step,
        # reset on episode end. Allocated lazily in init() once we know the env count
        # via the memory's num_envs attribute.
        self._traj_success_so_far: torch.Tensor | None = None

    # --------------------------------------------------------------
    # Per-agent helpers
    # --------------------------------------------------------------
    def _expand_per_agent(self, x_n1: torch.Tensor, batch_per_agent: int) -> torch.Tensor:
        """``(N, 1) -> (N*B, 1)`` to broadcast against flat batch tensors."""
        return x_n1.repeat_interleave(batch_per_agent, dim=0)

    def track_per_agent(self, tag: str, values_per_agent) -> None:
        """Buffer a scalar per agent under ``tag``; ``values_per_agent`` is iterable of length N."""
        if not self.per_agent_tracking:
            return
        for i in range(self.num_agents):
            v = values_per_agent[i]
            self.per_agent_tracking[i][tag].append(v.item() if torch.is_tensor(v) else float(v))

    # --------------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------------
    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        """Initialize per-agent writers and memory tensors.

        Drops the inherited single ``self.writer`` — every metric is published per-agent.

        Idempotent: ``trainer.train()`` calls ``init()`` internally, so the runner is
        free to call it manually first (e.g. to materialize per-agent folders for a
        config-dump). Subsequent calls are no-ops to avoid duplicating writers.
        """
        if getattr(self, "_init_done", False):
            return
        super().init(trainer_cfg=trainer_cfg)
        self.enable_models_training_mode(False)

        # tear down the shared writer the base class created (we publish per-agent only).
        # Base only sets self.writer when write_interval > 0; otherwise the attribute
        # may not exist at all.
        writer = getattr(self, "writer", None)
        if writer is not None:
            writer.close()
            self.writer = None

        # per-agent writers + per-agent reward/episode deques.
        # Layout: <experiment_dir>/<i>/ holds tensorboard events AND checkpoints for agent i,
        # so each agent's folder is fully self-contained.
        if self.write_interval > 0:
            for i in range(self.num_agents):
                self.per_agent_writers.append(
                    SummaryWriter(log_dir=os.path.join(self.experiment_dir, str(i)))
                )
                self.per_agent_tracking.append(collections.defaultdict(list))
                self._per_agent_track_rewards.append(collections.deque(maxlen=100))
                self._per_agent_track_timesteps.append(collections.deque(maxlen=100))
                self._per_agent_track_success.append(collections.deque(maxlen=100))

        # memory tensors (observation-only; states intentionally absent)
        if self.memory is not None:
            self.memory.create_tensor(name="observations", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_observations", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)

            self._tensors_names = [
                "observations",
                "actions",
                "rewards",
                "next_observations",
                "terminated",
            ]

            # Success-prediction extras: register the per-trajectory label tensor
            # and add it to the sampled tensor list so it appears in batches.
            if self.predict_success:
                self.memory.create_tensor(
                    name="trajectory_succeeded", size=1, dtype=torch.float32
                )
                self._tensors_names.append("trajectory_succeeded")

            # Per-env trajectory-success accumulator. Sized from the memory's num_envs
            # (which the runner sets to total_envs = num_envs_per_agent * num_agents).
            self._traj_success_so_far = torch.zeros(
                self.memory.num_envs, dtype=torch.bool, device=self.device
            )

        self._init_done = True

    def write_tracking_data(self, *, timestep: int, timesteps: int) -> None:
        """Flush per-agent tracking buckets to per-agent writers."""
        for i, writer in enumerate(self.per_agent_writers):
            for tag, values in self.per_agent_tracking[i].items():
                if not values:
                    continue
                if tag.endswith("(min)"):
                    writer.add_scalar(tag=tag, value=float(np.min(values)), timestep=timestep)
                elif tag.endswith("(max)"):
                    writer.add_scalar(tag=tag, value=float(np.max(values)), timestep=timestep)
                else:
                    writer.add_scalar(tag=tag, value=float(np.mean(values)), timestep=timestep)
            self.per_agent_tracking[i].clear()

    # --------------------------------------------------------------
    # Interaction
    # --------------------------------------------------------------
    def act(
        self, observations: torch.Tensor, states: torch.Tensor | None, *, timestep: int, timesteps: int
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample actions from the policy. ``states`` is accepted for trainer compatibility but ignored."""
        inputs = {"observations": self._observation_preprocessor(observations)}
        if self.training and timestep < self.cfg.random_timesteps:
            return self.policy.random_act(inputs, role="policy")
        with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            actions, outputs = self.policy.act(inputs, role="policy")
        return actions, outputs

    def record_transition(
        self,
        *,
        observations: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Per-agent reward/episode bookkeeping + memory write.

        Skips ``super().record_transition`` because the base implementation accumulates
        a single global reward stream; we publish per-agent rewards instead.
        """
        if self.write_interval > 0:
            if self._cumulative_rewards is None:
                self._cumulative_rewards = torch.zeros_like(rewards, dtype=torch.float32)
                self._cumulative_timesteps = torch.zeros_like(rewards, dtype=torch.int32)

            self._cumulative_rewards.add_(rewards)
            self._cumulative_timesteps.add_(1)

            total_envs = rewards.shape[0]
            epa = total_envs // self.num_agents
            rewards_per_agent = rewards.view(self.num_agents, epa, 1)

            self.track_per_agent("Reward / Instantaneous reward (max)",
                                 rewards_per_agent.amax(dim=(1, 2)))
            self.track_per_agent("Reward / Instantaneous reward (min)",
                                 rewards_per_agent.amin(dim=(1, 2)))
            self.track_per_agent("Reward / Instantaneous reward (mean)",
                                 rewards_per_agent.mean(dim=(1, 2)))

            # per-env episode finishes; partition by agent index
            done = (terminated + truncated).bool().view(-1)
            cum_r_flat = self._cumulative_rewards.view(-1)
            cum_t_flat = self._cumulative_timesteps.view(-1)
            for i in range(self.num_agents):
                env_lo, env_hi = i * epa, (i + 1) * epa
                done_slice = done[env_lo:env_hi]
                if done_slice.any():
                    finished_envs = done_slice.nonzero(as_tuple=False).view(-1) + env_lo
                    self._per_agent_track_rewards[i].extend(cum_r_flat[finished_envs].tolist())
                    self._per_agent_track_timesteps[i].extend(cum_t_flat[finished_envs].tolist())
                    self._cumulative_rewards.view(-1)[finished_envs] = 0
                    self._cumulative_timesteps.view(-1)[finished_envs] = 0

                if len(self._per_agent_track_rewards[i]):
                    tr = np.array(self._per_agent_track_rewards[i])
                    tt = np.array(self._per_agent_track_timesteps[i])
                    self.per_agent_tracking[i]["Reward / Total reward (max)"].append(float(tr.max()))
                    self.per_agent_tracking[i]["Reward / Total reward (min)"].append(float(tr.min()))
                    self.per_agent_tracking[i]["Reward / Total reward (mean)"].append(float(tr.mean()))
                    self.per_agent_tracking[i]["Episode / Total timesteps (max)"].append(float(tt.max()))
                    self.per_agent_tracking[i]["Episode / Total timesteps (min)"].append(float(tt.min()))
                    self.per_agent_tracking[i]["Episode / Total timesteps (mean)"].append(float(tt.mean()))

        if self.training:
            if self.cfg.rewards_shaper is not None:
                rewards = self.cfg.rewards_shaper(rewards, timestep, timesteps)
            self.memory.add_samples(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                terminated=terminated,
            )

            # ----- Success-prediction bookkeeping -----
            # Per-step OR over the configured info key, finalize on episode end.
            if self.predict_success:
                if not isinstance(infos, dict) or self.success_info_key not in infos:
                    keys = list(infos.keys()) if isinstance(infos, dict) else type(infos).__name__
                    raise KeyError(
                        f"infos missing required key '{self.success_info_key}' for "
                        f"success prediction (predict_success=True). Got keys: {keys}. "
                        f"Wrap the env to emit it, or set sac_cfg.success_info_key, or "
                        f"set predict_success=false to disable the success head."
                    )
                step_success = infos[self.success_info_key]
                if not torch.is_tensor(step_success):
                    step_success = torch.as_tensor(step_success, device=self.device)
                step_success = step_success.to(self.device).bool().view(-1)
                if step_success.shape[0] != self._traj_success_so_far.shape[0]:
                    raise ValueError(
                        f"infos['{self.success_info_key}'] has shape {tuple(step_success.shape)} "
                        f"but expected per-env tensor of length {self._traj_success_so_far.shape[0]}"
                    )
                self._traj_success_so_far |= step_success

                done_mask = (terminated.bool() | truncated.bool()).view(-1)
                if done_mask.any():
                    finished = done_mask.nonzero(as_tuple=False).view(-1)
                    labels = self._traj_success_so_far[finished].float()
                    self.memory.finalize_trajectory(
                        env_indices=finished, success_labels=labels
                    )
                    # Per-agent success-rate deque
                    epa = self.memory.num_envs // self.num_agents
                    for env_i, lbl in zip(finished.tolist(), labels.tolist()):
                        if self._per_agent_track_success:
                            self._per_agent_track_success[env_i // epa].append(int(lbl))
                    # Reset accumulator for finished envs
                    self._traj_success_so_far[finished] = False

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        pass

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        if self.training:
            if timestep >= self.cfg.learning_starts:
                with ScopedTimer() as timer:
                    self.enable_models_training_mode(True)
                    self.update(timestep=timestep, timesteps=timesteps)
                    self.enable_models_training_mode(False)
                    # algorithm wall-clock duplicated to every per-agent log
                    self.track_per_agent(
                        "Stats / Algorithm update time (ms)",
                        [timer.elapsed_time_ms] * self.num_agents,
                    )

        # base.post_interaction handles checkpointing + calls write_tracking_data on interval
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    # --------------------------------------------------------------
    # Update
    # --------------------------------------------------------------
    def update(self, *, timestep: int, timesteps: int) -> None:
        N = self.num_agents
        B = self.cfg.batch_size // N  # samples per agent

        for gradient_step in range(self.cfg.gradient_steps):
            sampled_list = self.memory.sample(
                names=self._tensors_names, batch_size=self.cfg.batch_size
            )[0]
            sampled = dict(zip(self._tensors_names, sampled_list))
            sampled_observations = sampled["observations"]
            sampled_actions = sampled["actions"]
            sampled_rewards = sampled["rewards"]
            sampled_next_observations = sampled["next_observations"]
            sampled_terminated = sampled["terminated"]
            sampled_traj_succeeded = sampled.get("trajectory_succeeded")  # None when not predicting

            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                inputs = {
                    "observations": self._observation_preprocessor(sampled_observations, train=True),
                }
                next_inputs = {
                    "observations": self._observation_preprocessor(sampled_next_observations, train=True),
                }

                with torch.no_grad():
                    next_actions, outputs = self.policy.act(next_inputs, role="policy")
                    next_log_prob = outputs["log_prob"]

                    target_q1_values, _ = self.target_critic_1.act(
                        {**next_inputs, "taken_actions": next_actions}, role="target_critic_1"
                    )
                    target_q2_values, _ = self.target_critic_2.act(
                        {**next_inputs, "taken_actions": next_actions}, role="target_critic_2"
                    )
                    ent_flat = self._expand_per_agent(self._entropy_coefficient, B)  # (N*B, 1)
                    target_q_values = torch.min(target_q1_values, target_q2_values) - ent_flat * next_log_prob
                    target_values = (
                        sampled_rewards
                        + self.cfg.discount_factor * sampled_terminated.logical_not() * target_q_values
                    )

                critic_1_values, _ = self.critic_1.act({**inputs, "taken_actions": sampled_actions}, role="critic_1")
                critic_2_values, _ = self.critic_2.act({**inputs, "taken_actions": sampled_actions}, role="critic_2")

                critic_loss = (
                    F.mse_loss(critic_1_values, target_values) + F.mse_loss(critic_2_values, target_values)
                ) / 2

            # critic step
            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()
            if config.torch.is_distributed:
                self.critic_1.reduce_parameters()
                self.critic_2.reduce_parameters()
            if self.cfg.grad_norm_clip > 0:
                self.scaler.unscale_(self.critic_optimizer)
                nn.utils.clip_grad_norm_(
                    itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()),
                    self.cfg.grad_norm_clip,
                )
            self.scaler.step(self.critic_optimizer)

            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                actions, outputs = self.policy.act(inputs, role="policy")
                log_prob = outputs["log_prob"]
                critic_1_pi, _ = self.critic_1.act({**inputs, "taken_actions": actions}, role="critic_1")
                critic_2_pi, _ = self.critic_2.act({**inputs, "taken_actions": actions}, role="critic_2")

                ent_flat = self._expand_per_agent(self._entropy_coefficient, B)  # detached, no grad
                policy_loss = (ent_flat * log_prob - torch.min(critic_1_pi, critic_2_pi)).mean()

                # Optional success-prediction BCE term, added to the policy loss so the
                # backbone gradient comes from both objectives.
                bce_per_sample = None
                if self.predict_success:
                    if "success_logit" not in outputs:
                        raise RuntimeError(
                            "predict_success=True but policy.act() did not emit 'success_logit'. "
                            "Confirm BlockSimBaActor was constructed with predict_success=True."
                        )
                    if sampled_traj_succeeded is None:
                        raise RuntimeError(
                            "predict_success=True but memory did not return 'trajectory_succeeded'. "
                            "Confirm SAC.init() registered the tensor and the memory is "
                            "TrajectoryBufferedMemory."
                        )
                    success_logit = outputs["success_logit"]  # (N*B, 1)
                    bce_per_sample = F.binary_cross_entropy_with_logits(
                        success_logit, sampled_traj_succeeded, reduction="none"
                    )
                    policy_loss = policy_loss + self.success_bce_weight * bce_per_sample.mean()

            self.policy_optimizer.zero_grad()
            self.scaler.scale(policy_loss).backward()
            if config.torch.is_distributed:
                self.policy.reduce_parameters()
            if self.cfg.grad_norm_clip > 0:
                self.scaler.unscale_(self.policy_optimizer)
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_norm_clip)
            self.scaler.step(self.policy_optimizer)

            # per-agent entropy step
            if self.cfg.learn_entropy:
                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    log_prob_per_agent = log_prob.view(N, B, 1).mean(dim=1)  # (N, 1)
                    entropy_loss_per_agent = -(
                        self.log_entropy_coefficient
                        * (log_prob_per_agent + self._target_entropy).detach()
                    )  # (N, 1)
                    entropy_loss = entropy_loss_per_agent.sum()

                self.entropy_optimizer.zero_grad()
                self.scaler.scale(entropy_loss).backward()
                self.scaler.step(self.entropy_optimizer)

                self._entropy_coefficient = torch.exp(self.log_entropy_coefficient.detach())  # (N, 1)

            self.scaler.update()

            # target networks
            self.target_critic_1.update_parameters(self.critic_1, polyak=self.cfg.polyak)
            self.target_critic_2.update_parameters(self.critic_2, polyak=self.cfg.polyak)

            if self.policy_scheduler:
                self.policy_scheduler.step()
            if self.critic_scheduler:
                self.critic_scheduler.step()

            # per-agent metric tracking
            if self.write_interval > 0:
                def split(t):  # (N*B, *) -> (N, B, -1)
                    return t.view(N, B, -1)

                policy_terms = (ent_flat * log_prob - torch.min(critic_1_pi, critic_2_pi))
                self.track_per_agent("Loss / Policy loss",
                                     split(policy_terms).mean(dim=(1, 2)))
                critic_loss_per_agent = 0.5 * (
                    F.mse_loss(split(critic_1_values), split(target_values), reduction="none").mean(dim=(1, 2))
                    + F.mse_loss(split(critic_2_values), split(target_values), reduction="none").mean(dim=(1, 2))
                )
                self.track_per_agent("Loss / Critic loss", critic_loss_per_agent)

                self.track_per_agent("Q-network / Q1 (max)",  split(critic_1_values).amax(dim=(1, 2)))
                self.track_per_agent("Q-network / Q1 (min)",  split(critic_1_values).amin(dim=(1, 2)))
                self.track_per_agent("Q-network / Q1 (mean)", split(critic_1_values).mean(dim=(1, 2)))
                self.track_per_agent("Q-network / Q2 (max)",  split(critic_2_values).amax(dim=(1, 2)))
                self.track_per_agent("Q-network / Q2 (min)",  split(critic_2_values).amin(dim=(1, 2)))
                self.track_per_agent("Q-network / Q2 (mean)", split(critic_2_values).mean(dim=(1, 2)))

                self.track_per_agent("Target / Target (max)",  split(target_values).amax(dim=(1, 2)))
                self.track_per_agent("Target / Target (min)",  split(target_values).amin(dim=(1, 2)))
                self.track_per_agent("Target / Target (mean)", split(target_values).mean(dim=(1, 2)))

                # Action diagnostics — surface tanh saturation and log_prob collapse.
                with torch.no_grad():
                    abs_a = actions.abs()
                    saturation = (abs_a > 0.99).float()
                self.track_per_agent("Action / |a| max",       split(abs_a).amax(dim=(1, 2)))
                self.track_per_agent("Action / |a| mean",      split(abs_a).mean(dim=(1, 2)))
                self.track_per_agent("Action / saturation rate", split(saturation).mean(dim=(1, 2)))
                self.track_per_agent("Action / log_prob (mean)", split(log_prob).mean(dim=(1, 2)))

                if self.cfg.learn_entropy:
                    self.track_per_agent("Loss / Entropy loss", entropy_loss_per_agent.squeeze(-1))
                    self.track_per_agent("Coefficient / Entropy coefficient",
                                         self._entropy_coefficient.squeeze(-1))

                # Per-agent BCE loss (sampled batch) and rolling success rate (over
                # finished trajectories) when success prediction is enabled.
                if self.predict_success and bce_per_sample is not None:
                    self.track_per_agent(
                        "Loss / BCE success loss",
                        split(bce_per_sample).mean(dim=(1, 2)),
                    )
                    self.track_per_agent(
                        "Q-network / Success prob (mean)",
                        split(outputs["success_prob"]).mean(dim=(1, 2)),
                    )
                    if self._per_agent_track_success:
                        per_agent_rate = []
                        for i in range(N):
                            d = self._per_agent_track_success[i]
                            per_agent_rate.append(float(np.mean(d)) if d else 0.0)
                        self.track_per_agent(
                            "Episode / Success rate", per_agent_rate
                        )

                if self.policy_scheduler:
                    lr = self.policy_scheduler.get_last_lr()[0]
                    self.track_per_agent("Learning / Policy learning rate", [lr] * N)
                if self.critic_scheduler:
                    lr = self.critic_scheduler.get_last_lr()[0]
                    self.track_per_agent("Learning / Critic learning rate", [lr] * N)

    # --------------------------------------------------------------
    # Per-agent checkpoint save/load
    # --------------------------------------------------------------
    def _build_per_agent_checkpoint(self, i: int, step: int) -> dict:
        """Build the per-agent checkpoint dict for slot ``i`` at training ``step``."""
        ckpt = {
            "step": int(step),
            "num_agents": int(self.num_agents),
            "agent_idx": int(i),
            "policy":           slice_block_state_dict(self.policy,           i, self.num_agents),
            "critic_1":         slice_block_state_dict(self.critic_1,         i, self.num_agents),
            "critic_2":         slice_block_state_dict(self.critic_2,         i, self.num_agents),
            "target_critic_1":  slice_block_state_dict(self.target_critic_1,  i, self.num_agents),
            "target_critic_2":  slice_block_state_dict(self.target_critic_2,  i, self.num_agents),
            "entropy_coefficient":     self._entropy_coefficient[i].detach().clone().cpu(),
            "log_entropy_coefficient": (
                self.log_entropy_coefficient.detach()[i].clone().cpu()
                if self.cfg.learn_entropy else None
            ),
            "policy_optimizer":  slice_optimizer_state(
                self.policy_optimizer.state_dict(), i, self.num_agents
            ),
            "critic_optimizer":  slice_optimizer_state(
                self.critic_optimizer.state_dict(), i, self.num_agents
            ),
            "entropy_optimizer": (
                slice_optimizer_state(self.entropy_optimizer.state_dict(), i, self.num_agents)
                if self.cfg.learn_entropy else None
            ),
            "observation_preprocessor": self._build_preprocessor_state_for(i),
        }
        return ckpt

    def _build_preprocessor_state_for(self, i: int):
        """Return the preprocessor state dict for agent ``i``, or None if no preprocessor.

        If a ``PerAgentPreprocessorWrapper`` is configured, the per-agent state for slot
        ``i`` MUST be present — anything else is a configuration error.
        """
        if not isinstance(self._observation_preprocessor, PerAgentPreprocessorWrapper):
            return None
        full = self._observation_preprocessor.state_dict()
        key = f"agent_{i}"
        if key not in full:
            raise KeyError(
                f"PerAgentPreprocessorWrapper has no state for {key}; "
                f"got keys {sorted(full.keys())}"
            )
        return full[key]

    def write_checkpoint(self, timestep: int, timesteps: int) -> None:
        """Save one ``ckpt_{timestep}.pt`` file per agent, each in its own folder.

        Replaces the base Agent's bundled checkpoint write; we save sliced state
        per-agent so each agent's folder is fully independent.
        """
        tag = str(timestep)
        for i in range(self.num_agents):
            ckpt_dir = os.path.join(self.experiment_dir, str(i), "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            path = os.path.join(ckpt_dir, f"ckpt_{tag}.pt")
            torch.save(self._build_per_agent_checkpoint(i, timestep), path)

    # --- load helpers ---
    @staticmethod
    def _is_single_agent_dir(path: str) -> bool:
        """A folder is a 'single-agent' folder if it contains checkpoints/ckpt_*.pt directly."""
        return bool(glob.glob(os.path.join(path, "checkpoints", "ckpt_*.pt")))

    @staticmethod
    def _resolve_ckpt_file(agent_dir: str, step: int | None) -> str:
        """Return the path to ckpt_{step}.pt, or the latest if step is None."""
        ckpt_dir = os.path.join(agent_dir, "checkpoints")
        if step is not None:
            path = os.path.join(ckpt_dir, f"ckpt_{step}.pt")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Checkpoint file not found: {path}")
            return path
        candidates = glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No ckpt_*.pt files in {ckpt_dir}")

        def _step_of(p: str) -> int:
            m = re.search(r"ckpt_(\d+)\.pt$", os.path.basename(p))
            return int(m.group(1)) if m else -1

        return max(candidates, key=_step_of)

    def _load_one_into_slot(self, agent_dir: str, target_slot: int, step: int | None) -> dict:
        """Load a single per-agent ckpt file from ``agent_dir`` into block slot ``target_slot``.

        Loads weights, per-slot entropy coefficient, and per-slot preprocessor state.
        Optimizer state is NOT loaded here (caller stitches optimizer states in bulk).
        Returns the raw checkpoint dict for follow-up handling.
        """
        path = self._resolve_ckpt_file(agent_dir, step)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # Validate required top-level keys are present (no silent fallback).
        required_keys = {
            "step", "num_agents", "agent_idx",
            "policy", "critic_1", "critic_2", "target_critic_1", "target_critic_2",
            "entropy_coefficient", "log_entropy_coefficient",
            "policy_optimizer", "critic_optimizer", "entropy_optimizer",
            "observation_preprocessor",
        }
        missing = required_keys - set(ckpt.keys())
        if missing:
            raise KeyError(f"Checkpoint at {path} is missing required keys: {sorted(missing)}")

        assign_block_slice(self.policy,          target_slot, self.num_agents, ckpt["policy"])
        assign_block_slice(self.critic_1,        target_slot, self.num_agents, ckpt["critic_1"])
        assign_block_slice(self.critic_2,        target_slot, self.num_agents, ckpt["critic_2"])
        assign_block_slice(self.target_critic_1, target_slot, self.num_agents, ckpt["target_critic_1"])
        assign_block_slice(self.target_critic_2, target_slot, self.num_agents, ckpt["target_critic_2"])

        with torch.no_grad():
            self._entropy_coefficient[target_slot].copy_(
                ckpt["entropy_coefficient"].to(self.device)
            )
            # cfg.learn_entropy and saved log_entropy_coefficient must agree.
            saved_log_ent = ckpt["log_entropy_coefficient"]
            if self.cfg.learn_entropy and saved_log_ent is None:
                raise ValueError(
                    f"cfg.learn_entropy=True but checkpoint at {path} has "
                    f"log_entropy_coefficient=None (saved with learn_entropy=False)."
                )
            if not self.cfg.learn_entropy and saved_log_ent is not None:
                raise ValueError(
                    f"cfg.learn_entropy=False but checkpoint at {path} contains a "
                    f"log_entropy_coefficient (saved with learn_entropy=True)."
                )
            if self.cfg.learn_entropy:
                self.log_entropy_coefficient.data[target_slot].copy_(
                    saved_log_ent.to(self.device)
                )

        # Preprocessor state: cfg and saved must agree on presence.
        wrapper_configured = isinstance(self._observation_preprocessor, PerAgentPreprocessorWrapper)
        saved_preproc = ckpt["observation_preprocessor"]
        if wrapper_configured and saved_preproc is None:
            raise ValueError(
                f"PerAgentPreprocessorWrapper is configured but checkpoint at {path} "
                f"has no observation_preprocessor state."
            )
        if not wrapper_configured and saved_preproc is not None:
            raise ValueError(
                f"Checkpoint at {path} contains observation_preprocessor state but the "
                f"current run has no PerAgentPreprocessorWrapper configured."
            )
        if wrapper_configured:
            preproc = self._observation_preprocessor.preprocessor_list[target_slot]
            if preproc is None:
                raise ValueError(
                    f"PerAgentPreprocessorWrapper slot {target_slot} is None; cannot "
                    f"load preprocessor state from {path}."
                )
            preproc.load_state_dict(saved_preproc)

        return ckpt

    def load(self, path: str, *, step: int | None = None) -> None:
        """Load weights/state from a checkpoint folder.

        Two modes auto-detected from ``path``:

        * **Single-agent**: ``path/checkpoints/ckpt_*.pt`` exists directly. Requires
          the current run to have ``num_agents == 1``; loads into slot 0.
        * **Multi-agent**: ``path`` contains subfolders ``0/``, ``1/``, ..., each with
          its own ``checkpoints/ckpt_*.pt``. Strict ``num_agents`` match required.

        :param path: Folder path (per the modes above).
        :param step: Optional specific training step to load. If ``None`` (default),
            the latest ``ckpt_<step>.pt`` found in each folder is used.
        """
        if self._is_single_agent_dir(path):
            if self.num_agents != 1:
                raise ValueError(
                    f"Single-agent checkpoint at {path} requires num_agents=1, "
                    f"but this run has num_agents={self.num_agents}"
                )
            ckpt = self._load_one_into_slot(path, target_slot=0, step=step)
            if "num_agents" not in ckpt:
                raise KeyError(f"Checkpoint at {path} is missing required key 'num_agents'")
            src_n = int(ckpt["num_agents"])
            # A slice taken from an N>1 file carries optimizer state with param_groups that
            # don't fit a 1-agent optimizer. We refuse to load — silently dropping it would
            # give the user a 1-agent run with fresh Adam moments under the same name.
            if src_n != 1:
                raise ValueError(
                    f"Refusing to load single-agent checkpoint at {path}: it was sliced "
                    f"from a num_agents={src_n} run, so its optimizer state cannot be "
                    f"restored into a 1-agent optimizer. Use multi-agent load mode "
                    f"(point --checkpoint at the parent run dir) or train fresh."
                )
            self.policy_optimizer.load_state_dict(
                merge_optimizer_states([ckpt["policy_optimizer"]], 1)
            )
            self.critic_optimizer.load_state_dict(
                merge_optimizer_states([ckpt["critic_optimizer"]], 1)
            )
            saved_ent_opt = ckpt["entropy_optimizer"]
            if self.cfg.learn_entropy and saved_ent_opt is None:
                raise ValueError(
                    f"cfg.learn_entropy=True but checkpoint at {path} has "
                    f"entropy_optimizer=None (saved with learn_entropy=False)."
                )
            if not self.cfg.learn_entropy and saved_ent_opt is not None:
                raise ValueError(
                    f"cfg.learn_entropy=False but checkpoint at {path} contains an "
                    f"entropy_optimizer (saved with learn_entropy=True)."
                )
            if self.cfg.learn_entropy:
                self.entropy_optimizer.load_state_dict(
                    merge_optimizer_states([saved_ent_opt], 1)
                )
            return

        # Multi-agent: expect path/0, path/1, ..., path/(N-1) all present.
        per_agent_ckpts: list[dict] = []
        for i in range(self.num_agents):
            agent_dir = os.path.join(path, str(i))
            if not os.path.isdir(agent_dir):
                raise FileNotFoundError(
                    f"Expected per-agent dir {agent_dir} for num_agents={self.num_agents}"
                )
            ckpt = self._load_one_into_slot(agent_dir, target_slot=i, step=step)
            if "num_agents" not in ckpt:
                raise KeyError(f"Checkpoint at {agent_dir} is missing required key 'num_agents'")
            if ckpt["num_agents"] != self.num_agents:
                raise ValueError(
                    f"Checkpoint at {agent_dir} has num_agents={ckpt['num_agents']} but "
                    f"current run has num_agents={self.num_agents}"
                )
            per_agent_ckpts.append(ckpt)

        # Stitch optimizer state across agents.
        self.policy_optimizer.load_state_dict(
            merge_optimizer_states([c["policy_optimizer"] for c in per_agent_ckpts], self.num_agents)
        )
        self.critic_optimizer.load_state_dict(
            merge_optimizer_states([c["critic_optimizer"] for c in per_agent_ckpts], self.num_agents)
        )
        # cfg.learn_entropy must agree with all saved files (no silent skip on mismatch).
        ent_opt_present = [c["entropy_optimizer"] is not None for c in per_agent_ckpts]
        if any(ent_opt_present) != all(ent_opt_present):
            raise ValueError(
                f"Inconsistent entropy_optimizer presence across per-agent checkpoints: "
                f"{ent_opt_present}. All agents must have been saved with the same "
                f"learn_entropy setting."
            )
        all_have = all(ent_opt_present)
        if self.cfg.learn_entropy and not all_have:
            raise ValueError(
                f"cfg.learn_entropy=True but per-agent checkpoints under {path} have "
                f"entropy_optimizer=None (saved with learn_entropy=False)."
            )
        if not self.cfg.learn_entropy and all_have:
            raise ValueError(
                f"cfg.learn_entropy=False but per-agent checkpoints under {path} contain "
                f"entropy_optimizer state (saved with learn_entropy=True)."
            )
        if self.cfg.learn_entropy:
            self.entropy_optimizer.load_state_dict(
                merge_optimizer_states([c["entropy_optimizer"] for c in per_agent_ckpts], self.num_agents)
            )
