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
        state_space: gymnasium.Space | None = None,
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

        # Asymmetric actor-critic: when ``state_space`` is provided, the critic
        # consumes the (typically larger) state vector while the actor still uses
        # the policy observation. Memory then stores both ``observations`` and
        # ``states``. When ``state_space is None``, the critic uses observations
        # (symmetric — backward-compatible).
        self.state_space = state_space
        self._asymmetric: bool = state_space is not None

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
            # Entropy gets AdamW with weight_decay=0 — pulling log_alpha toward 0 has no
            # principled meaning, so the user-configured weight_decay is intentionally
            # NOT applied here.
            self.entropy_optimizer = torch.optim.AdamW(
                [self.log_entropy_coefficient],
                lr=self.cfg.entropy_lr,
                weight_decay=0.0,
            )

        # set up optimizers and learning rate schedulers (AdamW with decoupled weight decay)
        if self.policy is not None and self.critic_1 is not None and self.critic_2 is not None:
            self.policy_optimizer = torch.optim.AdamW(
                self.policy.parameters(),
                lr=self.cfg.actor_lr,
                weight_decay=self.cfg.weight_decay,
            )
            self.critic_optimizer = torch.optim.AdamW(
                itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()),
                lr=self.cfg.critic_lr,
                weight_decay=self.cfg.weight_decay,
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

        # State preprocessor (asymmetric setup only). Same class as obs preprocessor,
        # sized to state_space. The runner injects size+device into preprocessor_kwargs;
        # for state we re-use the same kwargs but override `size` to state_space.
        if self._asymmetric and preproc_cls is not None:
            state_kwargs = dict(self.cfg.observation_preprocessor_kwargs)
            state_kwargs["size"] = state_space
            state_preproc_list = [preproc_cls(**state_kwargs) for _ in range(num_agents)]
            self._state_preprocessor = PerAgentPreprocessorWrapper(num_agents, state_preproc_list)
        else:
            self._state_preprocessor = self._empty_preprocessor

        # per-agent tracking buffers (writers created in init() once experiment_dir is set)
        self.per_agent_writers: list[SummaryWriter] = []
        # Separate torch.utils.tensorboard.SummaryWriter per agent dedicated to
        # image events. skrl's SummaryWriter (used for scalars) doesn't
        # implement add_image; both writers point at the same log dir so TB
        # picks up both event streams together.
        self.per_agent_image_writers: list = []
        self.per_agent_tracking: list[collections.defaultdict] = []
        # Episode totals / lengths accumulated since the last write_interval flush.
        # write_tracking_data computes max/min/mean over the interval and then clears,
        # so the published values reflect ONLY the episodes that finished in the
        # current interval — no rolling-window lag.
        self._per_agent_track_rewards: list[list[float]] = []
        self._per_agent_track_timesteps: list[list[int]] = []
        # Finished-trajectory success labels accumulated since the last write_interval
        # flush. Cleared in write_tracking_data after the per-agent emit, so the published
        # rate always reflects only this-interval episodes (no rolling-window lag).
        self._per_agent_episodes_this_interval: list[list[int]] = []
        # Per-agent per-trajectory forward-distance accumulator (filled when a
        # task-specific wrapper publishes `info["per_env_episode_distance"]`),
        # cleared in write_tracking_data after emit. Currently the AntSuccess
        # wrapper publishes this; absent for other tasks (consumer skips).
        self._per_agent_distances_this_interval: list[list[float]] = []
        # Same pattern for per-trajectory average forward velocity (m/s).
        self._per_agent_velocities_this_interval: list[list[float]] = []
        # One-shot stdout warning when info[success_info_key] is absent.
        self._warned_no_success_key: bool = False

        # Success-prediction config (read once into instance attrs for fast access).
        self.predict_success: bool = bool(getattr(self.cfg, "predict_success", False))
        self.success_td_weight: float = float(getattr(self.cfg, "success_td_weight", 0.0))
        self.success_td_discount: float = float(getattr(self.cfg, "success_td_discount", 0.99))
        self.success_info_key: str = str(getattr(self.cfg, "success_info_key", "is_success"))

        # Per-env "success seen this trajectory so far" — OR-accumulated each step,
        # reset on episode end. Allocated lazily in init() once we know the env count
        # via the memory's num_envs attribute.
        self._traj_success_so_far: torch.Tensor | None = None

        # Per-step actor success probability stash from the most recent act()
        # call. Used by record_transition to feed the SuccessPredMetricsTracker
        # so we get Forge-style early_term_* metrics on Factory (and any other
        # task with predict_success=true). Tracker itself is allocated lazily
        # on the first record_transition since we need num_envs from the data.
        self._latest_success_prob: torch.Tensor | None = None
        self._success_pred_tracker = None  # type: ignore[assignment]
        # Per-rollout-trajectory predictive-quality tracker (AUC / ECE /
        # per-class BCE / monotonicity / heatmaps). Lazily allocated on first
        # record_transition since num_envs comes from the data shape.
        self._pred_quality_tracker = None  # type: ignore[assignment]
        # Last update's gradient-norm slices, captured pre-optimizer-step
        # and surfaced in write_tracking_data so the TB scalars don't have
        # to be re-derived after the step has already been applied.
        self._last_action_head_grad_norm: torch.Tensor | None = None
        self._last_success_head_grad_norm: torch.Tensor | None = None

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

    def _compute_actor_head_grad_norms(self):
        """Return per-agent (action_head_grad_norm, success_head_grad_norm) tensors.

        The actor's ``fc_out`` BlockLinear is the only place where action-vs-
        success outputs have private parameters: ``weight`` is (N, total_out,
        hidden) and ``bias`` is (N, total_out), with the action slice at the
        first ``_policy_out_dim`` rows and the success slice at the last
        ``success_out_dim`` rows. Backbone params (fc_in, resblocks, ln_out,
        and the std rows of fc_out if state-dependent std is on) are shared,
        so attributing them to one head or the other isn't well-defined; we
        report only the head-private slice norms.

        Returns ``(None, None)`` (or one None) if the corresponding slice has
        no parameters (e.g. predict_success=False ⇒ no success slice).
        """
        actor_mean = self.policy.actor_mean
        fc_out = actor_mean.fc_out
        N = self.num_agents
        out_dim = fc_out.weight.shape[1]
        policy_out = getattr(self.policy, "_policy_out_dim", out_dim - actor_mean.success_out_dim - actor_mean.std_out_dim)
        success_out = actor_mean.success_out_dim

        # Grads may be None if backward didn't reach this param (e.g. on the
        # very first call before optimizer has stepped at all). Guard.
        wg = fc_out.weight.grad
        bg = fc_out.bias.grad
        if wg is None or bg is None:
            return None, None

        action_gn: torch.Tensor | None = None
        if policy_out > 0:
            wa = wg[:, :policy_out, :].reshape(N, -1)   # (N, policy_out * hidden)
            ba = bg[:, :policy_out].reshape(N, -1)      # (N, policy_out)
            action_gn = torch.cat([wa, ba], dim=1).norm(dim=1)  # (N,)

        success_gn: torch.Tensor | None = None
        if success_out > 0:
            ws = wg[:, -success_out:, :].reshape(N, -1)
            bs = bg[:, -success_out:].reshape(N, -1)
            success_gn = torch.cat([ws, bs], dim=1).norm(dim=1)  # (N,)

        return action_gn, success_gn

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

        # Skrl's base init() also writes an empty tfevents file to <experiment_dir>/
        # (from the now-closed shared writer) and creates an empty <experiment_dir>/
        # checkpoints/ directory. Both belong to the per-agent subfolders only —
        # remove the top-level orphans so the experiment dir is clean. Use os.rmdir
        # (not rmtree) on the checkpoints folder so any unexpected file blocks the
        # deletion loudly rather than getting silently nuked.
        for events_file in glob.glob(os.path.join(self.experiment_dir, "events.out.tfevents.*")):
            try:
                os.remove(events_file)
            except OSError:
                pass
        ckpt_dir = os.path.join(self.experiment_dir, "checkpoints")
        if os.path.isdir(ckpt_dir):
            try:
                os.rmdir(ckpt_dir)
            except OSError:
                pass

        # per-agent writers + per-agent reward/episode deques.
        # Layout: <experiment_dir>/<i>/ holds tensorboard events AND checkpoints for agent i,
        # so each agent's folder is fully self-contained.
        if self.write_interval > 0:
            from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter
            for i in range(self.num_agents):
                agent_log_dir = os.path.join(self.experiment_dir, str(i))
                self.per_agent_writers.append(
                    SummaryWriter(log_dir=agent_log_dir)
                )
                # Image writer (torch SummaryWriter) — same log dir so TB
                # aggregates scalar + image events under the same agent run.
                self.per_agent_image_writers.append(
                    TorchSummaryWriter(log_dir=agent_log_dir)
                )
                self.per_agent_tracking.append(collections.defaultdict(list))
                self._per_agent_track_rewards.append([])
                self._per_agent_track_timesteps.append([])
                self._per_agent_episodes_this_interval.append([])
                self._per_agent_distances_this_interval.append([])
                self._per_agent_velocities_this_interval.append([])

        # memory tensors. In symmetric mode (state_space=None) only obs are stored.
        # In asymmetric mode, both obs (for actor) and states (for critic) are stored.
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

            if self._asymmetric:
                self.memory.create_tensor(name="states", size=self.state_space, dtype=torch.float32)
                self.memory.create_tensor(name="next_states", size=self.state_space, dtype=torch.float32)
                self._tensors_names.extend(["states", "next_states"])

            # Success-prediction extras: stage the per-step ``is_success_step``
            # flag (used by the memory at finalize-time to find first-success
            # step), and register the three derived per-step tensors used to
            # build TD bootstrap targets during update():
            #   * ``is_first_success_step``: 1 at first-success step, else 0.
            #   * ``success_terminal``: 1 at success-terminal or failure-
            #     terminal step (no bootstrap past these), else 0.
            #   * ``success_loss_mask``: 1 if this transition contributes to
            #     the loss, 0 if masked (post-success states).
            if self.predict_success:
                self.memory.create_tensor(name="is_success_step", size=1, dtype=torch.float32)
                self.memory.create_tensor(name="is_first_success_step", size=1, dtype=torch.float32)
                self.memory.create_tensor(name="success_terminal", size=1, dtype=torch.float32)
                self.memory.create_tensor(name="success_loss_mask", size=1, dtype=torch.float32)
                # Sampled (and used by update()): only the three derived ones.
                # is_success_step lives in staging only; once finalized it has
                # no further role.
                self._tensors_names.extend([
                    "is_first_success_step",
                    "success_terminal",
                    "success_loss_mask",
                ])

            # Per-env trajectory-success accumulator. Sized from the memory's num_envs
            # (which the runner sets to total_envs = num_envs_per_agent * num_agents).
            self._traj_success_so_far = torch.zeros(
                self.memory.num_envs, dtype=torch.bool, device=self.device
            )

        self._init_done = True

    def write_tracking_data(self, *, timestep: int, timesteps: int) -> None:
        """Flush per-agent tracking buckets to per-agent writers."""
        # Compute predictive-quality interval metrics (AUC, ECE, per-class
        # BCE, monotonicity) and emit per-class heatmap PNGs to TB. Done
        # BEFORE the scalar flush below so the new scalars land in this
        # interval's write. Tracker also clears its own per-interval buffer.
        if self._pred_quality_tracker is not None:
            self._pred_quality_tracker.flush_per_agent(
                per_agent_tracking=self.per_agent_tracking,
                per_agent_writers=self.per_agent_image_writers,
                timestep=timestep,
            )

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

            # Episode totals + lengths over episodes that finished since the last
            # flush. Cleared here so the next interval starts fresh — no rolling-
            # window lag against `Episode_Termination/<term>` (which Isaac Lab logs
            # as the steady-state distribution over termination terms).
            rewards_list = self._per_agent_track_rewards[i]
            if rewards_list:
                arr = np.array(rewards_list, dtype=np.float64)
                writer.add_scalar(tag="Reward / Total reward (max)",  value=float(arr.max()),  timestep=timestep)
                writer.add_scalar(tag="Reward / Total reward (min)",  value=float(arr.min()),  timestep=timestep)
                writer.add_scalar(tag="Reward / Total reward (mean)", value=float(arr.mean()), timestep=timestep)
                rewards_list.clear()

            timesteps_list = self._per_agent_track_timesteps[i]
            if timesteps_list:
                arr = np.array(timesteps_list, dtype=np.float64)
                writer.add_scalar(tag="Episode / Total timesteps (max)",  value=float(arr.max()),  timestep=timestep)
                writer.add_scalar(tag="Episode / Total timesteps (min)",  value=float(arr.min()),  timestep=timestep)
                writer.add_scalar(tag="Episode / Total timesteps (mean)", value=float(arr.mean()), timestep=timestep)
                timesteps_list.clear()

            # Success rate over trajectories that finished since the last flush.
            ep = self._per_agent_episodes_this_interval[i]
            if ep:
                writer.add_scalar(
                    tag="Episode / Success rate",
                    value=float(np.mean(ep)),
                    timestep=timestep,
                )
                ep.clear()

            # Per-trajectory forward distance (max/min/mean) — populated only when
            # a task-specific wrapper publishes per_env_episode_distance.
            dist_list = self._per_agent_distances_this_interval[i]
            if dist_list:
                arr = np.array(dist_list, dtype=np.float64)
                writer.add_scalar(tag="Episode / Distance traveled (max)",  value=float(arr.max()),  timestep=timestep)
                writer.add_scalar(tag="Episode / Distance traveled (min)",  value=float(arr.min()),  timestep=timestep)
                writer.add_scalar(tag="Episode / Distance traveled (mean)", value=float(arr.mean()), timestep=timestep)
                dist_list.clear()

            # Per-trajectory average velocity (max/min/mean) — same conditional
            # population as distance.
            vel_list = self._per_agent_velocities_this_interval[i]
            if vel_list:
                arr = np.array(vel_list, dtype=np.float64)
                writer.add_scalar(tag="Episode / Velocity (max)",  value=float(arr.max()),  timestep=timestep)
                writer.add_scalar(tag="Episode / Velocity (min)",  value=float(arr.min()),  timestep=timestep)
                writer.add_scalar(tag="Episode / Velocity (mean)", value=float(arr.mean()), timestep=timestep)
                vel_list.clear()

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
            # Uniform on [-1, 1] — matches the tanh-squashed policy's support.
            # Skips skrl's default random_act which calls Box.sample() on the env's
            # action_space; Isaac Lab advertises Box(-inf, +inf), so that fallback
            # samples N(0, 1) and pumps the replay buffer with actions outside the
            # policy's reachable range, producing a misleading reward cliff at the
            # random→policy hand-off.
            n = observations.shape[0]
            actions = torch.rand(n, *self.action_space.shape, device=self.device) * 2.0 - 1.0
            return actions, {}
        with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            actions, outputs = self.policy.act(inputs, role="policy")
        # Stash per-env success probability for record_transition's metrics
        # tracker. Cheap clone — same shape as observations[:, 0].
        if self.predict_success and "success_prob" in outputs:
            self._latest_success_prob = outputs["success_prob"].detach().view(-1).clone()
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

            # Per-env episode finishes; partition by agent index. Stats over the
            # accumulated episodes (max/min/mean) are emitted in write_tracking_data
            # at write_interval boundaries — see that method for the flush + clear.
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

            # Per-agent reward decomposition (preferred): a wrapper publishes
            # already-normalized per-env per-term values in `infos["per_env_rew"]`
            # plus a bool mask in `infos["per_env_rew_mask"]`. We partition by
            # agent (env i belongs to agent i // epa) and mean over the resetting
            # envs in that agent's slice. Tag matches Isaac Lab's convention so
            # old + new tensorboards plot continuously.
            per_env_rew_terms: set[str] = set()
            if (
                isinstance(infos, dict)
                and "per_env_rew" in infos
                and "per_env_rew_mask" in infos
            ):
                per_env = infos["per_env_rew"]
                mask = infos["per_env_rew_mask"]
                if torch.is_tensor(mask) and mask.any():
                    for term, per_env_vals in per_env.items():
                        per_env_rew_terms.add(f"Episode_Reward/{term}")
                        for i in range(self.num_agents):
                            env_lo, env_hi = i * epa, (i + 1) * epa
                            agent_mask = mask[env_lo:env_hi]
                            if not agent_mask.any():
                                continue
                            vals = per_env_vals[env_lo:env_hi][agent_mask]
                            self.per_agent_tracking[i][f"Episode_Reward/{term}"].append(
                                float(vals.mean().item())
                            )

            # Per-trajectory forward-distance ingestion (task-specific wrapper).
            # Wrappers like AntSuccessWrapper publish `info["per_env_episode_distance"]`
            # (the final per-env displacement for the just-ended episode) plus a
            # mask. We accumulate per-agent into a per-interval list; max/min/mean
            # are emitted in write_tracking_data, then cleared.
            if (
                isinstance(infos, dict)
                and "per_env_episode_distance" in infos
                and "per_env_episode_distance_mask" in infos
                and self._per_agent_distances_this_interval
            ):
                distances = infos["per_env_episode_distance"]
                dist_mask = infos["per_env_episode_distance_mask"]
                if torch.is_tensor(dist_mask) and dist_mask.any():
                    for i in range(self.num_agents):
                        env_lo, env_hi = i * epa, (i + 1) * epa
                        agent_mask = dist_mask[env_lo:env_hi]
                        if not agent_mask.any():
                            continue
                        vals = distances[env_lo:env_hi][agent_mask]
                        self._per_agent_distances_this_interval[i].extend(
                            vals.tolist()
                        )

            # Per-trajectory average-velocity ingestion (task-specific wrapper).
            # Same pattern as distance — accumulate per-agent into a per-interval
            # list, flush + clear in write_tracking_data.
            if (
                isinstance(infos, dict)
                and "per_env_episode_velocity" in infos
                and "per_env_episode_velocity_mask" in infos
                and self._per_agent_velocities_this_interval
            ):
                velocities = infos["per_env_episode_velocity"]
                vel_mask = infos["per_env_episode_velocity_mask"]
                if torch.is_tensor(vel_mask) and vel_mask.any():
                    for i in range(self.num_agents):
                        env_lo, env_hi = i * epa, (i + 1) * epa
                        agent_mask = vel_mask[env_lo:env_hi]
                        if not agent_mask.any():
                            continue
                        vals = velocities[env_lo:env_hi][agent_mask]
                        self._per_agent_velocities_this_interval[i].extend(
                            vals.tolist()
                        )

            # Per-step per-agent partition for Factory/Forge diagnostic tensors.
            # The wrappers publish raw per-env tensors under `per_env_*` keys;
            # we slice each agent's [env_lo:env_hi] portion, compute the right
            # aggregate (env-mean, mean-over-non-zero, etc.), and push to that
            # agent's tracking bucket. Without this, the env-aggregated scalars
            # in info["log"]["logs_rew_*"] / "successes" / "success_times"
            # would be the same value across all agents.
            per_env_logs_terms_handled: set[str] = set()
            if isinstance(infos, dict):
                # logs_rew/<term>: env-mean of unscaled per-step reward components.
                per_env_logs = infos.get("per_env_logs_rew")
                if isinstance(per_env_logs, dict):
                    for term, vals in per_env_logs.items():
                        # Wrappers MUST publish per-env tensors here (filtered
                        # at source). Anything else is a wrapper-side bug — fail
                        # loud so we don't silently drop a metric.
                        if not torch.is_tensor(vals):
                            raise TypeError(
                                f"per_env_logs_rew[{term!r}] is {type(vals).__name__}, "
                                f"expected torch.Tensor of shape ({total_envs},)"
                            )
                        if vals.dim() == 0 or vals.shape[0] != total_envs:
                            raise ValueError(
                                f"per_env_logs_rew[{term!r}] has shape {tuple(vals.shape)}, "
                                f"expected first dim == total_envs ({total_envs}). "
                                f"Filter scalar/aggregated terms in the wrapper before publishing."
                            )
                        per_env_logs_terms_handled.add(f"logs_rew_{term}")
                        for i in range(self.num_agents):
                            env_lo, env_hi = i * epa, (i + 1) * epa
                            agent_vals = vals[env_lo:env_hi].float()
                            self.per_agent_tracking[i][f"logs_rew/{term}"].append(
                                float(agent_vals.mean().item())
                            )

                # Per-agent `successes` and `success_times` mirror upstream
                # Factory's `_log_factory_metrics` semantics, just sliced to each
                # agent's env partition. Factory only writes these to extras on
                # specific steps; we replicate that timing per-agent so the TB
                # values are directly comparable to a single-agent run.
                #
                #   `successes`     : at any step where the agent has a resetting
                #                     env, count_nonzero(curr_successes[slice]) /
                #                     epa  (fraction of agent's envs at goal at
                #                     reset moment).
                #   `success_times` : at any step where any env in the slice has
                #                     a nonzero ep_success_time, mean step-of-
                #                     first-success across those envs.
                curr_succ = infos.get("per_env_curr_successes")
                if curr_succ is not None:
                    if not (torch.is_tensor(curr_succ) and curr_succ.dim() > 0
                            and curr_succ.shape[0] == total_envs):
                        raise ValueError(
                            f"per_env_curr_successes has shape "
                            f"{tuple(curr_succ.shape) if torch.is_tensor(curr_succ) else type(curr_succ).__name__}, "
                            f"expected ({total_envs},)"
                        )
                    done_flat = (terminated + truncated).bool().view(-1)
                    for i in range(self.num_agents):
                        env_lo, env_hi = i * epa, (i + 1) * epa
                        if done_flat[env_lo:env_hi].any():
                            self.per_agent_tracking[i]["successes"].append(
                                float(curr_succ[env_lo:env_hi].float().mean().item())
                            )

                ep_succ_times = infos.get("per_env_ep_success_times")
                if ep_succ_times is not None:
                    if not (torch.is_tensor(ep_succ_times) and ep_succ_times.dim() > 0
                            and ep_succ_times.shape[0] == total_envs):
                        raise ValueError(
                            f"per_env_ep_success_times has shape "
                            f"{tuple(ep_succ_times.shape) if torch.is_tensor(ep_succ_times) else type(ep_succ_times).__name__}, "
                            f"expected ({total_envs},)"
                        )
                    for i in range(self.num_agents):
                        env_lo, env_hi = i * epa, (i + 1) * epa
                        agent_times = ep_succ_times[env_lo:env_hi]
                        nonzero = agent_times > 0
                        if nonzero.any():
                            self.per_agent_tracking[i]["success_times"].append(
                                float(agent_times[nonzero].float().mean().item())
                            )

                # Forge early_term_*: per-agent versions of Forge's success-
                # prediction quality metrics, fed by the env's published
                # per-env first_pred_success_tx (per threshold) + ep_success_times.
                # Tag layout: `<thresh>/early_term_*` so each TB tab is one τ.
                first_pred = infos.get("per_env_first_pred_success_tx")
                if isinstance(first_pred, dict) and torch.is_tensor(ep_succ_times):
                    for thresh, fst in first_pred.items():
                        if not (torch.is_tensor(fst) and fst.dim() > 0
                                and fst.shape[0] == total_envs):
                            raise ValueError(
                                f"per_env_first_pred_success_tx[{thresh!r}] has shape "
                                f"{tuple(fst.shape) if torch.is_tensor(fst) else type(fst).__name__}, "
                                f"expected ({total_envs},)"
                            )
                        tag_root = f"{float(thresh):.1f}"
                        for i in range(self.num_agents):
                            env_lo, env_hi = i * epa, (i + 1) * epa
                            agent_fst = fst[env_lo:env_hi]
                            agent_est = ep_succ_times[env_lo:env_hi]

                            delay_mask = (agent_est != 0) & (agent_fst != 0)
                            if delay_mask.any():
                                delay = (agent_fst[delay_mask] - agent_est[delay_mask]).float().mean()
                                self.per_agent_tracking[i][f"{tag_root}/early_term_delay_all"].append(
                                    float(delay.item())
                                )
                                correct_mask = delay_mask & (agent_fst > agent_est)
                                if correct_mask.any():
                                    cd = (agent_fst[correct_mask] - agent_est[correct_mask]).float().mean()
                                    self.per_agent_tracking[i][f"{tag_root}/early_term_delay_correct"].append(
                                        float(cd.item())
                                    )
                            pred_mask = agent_fst != 0
                            if pred_mask.any():
                                tps = (agent_est[pred_mask] > 0) & (agent_est[pred_mask] < agent_fst[pred_mask])
                                self.per_agent_tracking[i][f"{tag_root}/early_term_precision"].append(
                                    float(tps.float().sum().item() / pred_mask.float().sum().item())
                                )
                            true_mask = agent_est > 0
                            if true_mask.any() and pred_mask.any():
                                tps = (agent_est[pred_mask] > 0) & (agent_est[pred_mask] < agent_fst[pred_mask])
                                self.per_agent_tracking[i][f"{tag_root}/early_term_recall"].append(
                                    float(tps.float().sum().item() / true_mask.float().sum().item())
                                )

                # predict_success=true path (Factory & friends): feed the
                # SuccessPredMetricsTracker with the actor's per-step success
                # probability stashed by act(), using the wrapper-published
                # ep_success_times as the true-success step. Same metrics as
                # Forge above, same `<thresh>/early_term_*` layout — different
                # source. Skips silently when no success_prob is available
                # (e.g. Forge runs, where the env's first_pred_success_tx path
                # above handles things).
                if (
                    self.predict_success
                    and self._latest_success_prob is not None
                    and torch.is_tensor(ep_succ_times)
                ):
                    if self._latest_success_prob.shape[0] != total_envs:
                        raise ValueError(
                            f"stashed success_prob shape "
                            f"{tuple(self._latest_success_prob.shape)} doesn't match "
                            f"total_envs={total_envs}"
                        )
                    if self._success_pred_tracker is None:
                        from wrappers.success_pred_metrics import SuccessPredMetricsTracker
                        self._success_pred_tracker = SuccessPredMetricsTracker(
                            num_envs=total_envs,
                            num_agents=self.num_agents,
                            device=self.device,
                        )
                    done_mask = (terminated + truncated).bool().view(-1).to(self.device)
                    # Update first_pred_success_tx from this step's prob.
                    self._success_pred_tracker.update(self._latest_success_prob, done_mask)
                    # Flush metrics using the env's still-current ep_success_times
                    # (matches Forge's order: capture metrics, THEN reset).
                    self._success_pred_tracker.flush_per_agent(
                        self.per_agent_tracking, ep_succ_times
                    )
                    # Reset per-env tracker state for envs that finished.
                    self._success_pred_tracker.reset_envs(done_mask)
                    # NOTE: do NOT clear self._latest_success_prob here. The
                    # PredictionQualityTracker further down also reads it; the
                    # next act() refills it before the next record_transition,
                    # so stale-data leak isn't possible.

            # Isaac Lab managers publish mean-across-resetting-envs scalars under
            # infos["log"] on reset steps (e.g. "Episode_Reward/lifting_object",
            # "Episode_Termination/<name>"). The env is shared across agents so we
            # mirror each scalar into every per-agent writer — except for
            # `Episode_Reward/<term>` keys already handled per-agent above.
            if isinstance(infos, dict):
                env_log = infos.get("log")
                if isinstance(env_log, dict) and env_log:
                    for k, v in env_log.items():
                        if k in per_env_rew_terms:
                            continue  # superseded by the per-agent decomposition
                        # Skip env-aggregated logs_rew_/successes/success_times
                        # if a wrapper is publishing per-env tensors for them
                        # (they get logged per-agent above instead).
                        if k in per_env_logs_terms_handled:
                            continue
                        if "per_env_curr_successes" in infos and k == "successes":
                            continue
                        if "per_env_ep_success_times" in infos and k == "success_times":
                            continue
                        if "per_env_first_pred_success_tx" in infos and (
                            k.startswith("early_term_delay_all/")
                            or k.startswith("early_term_delay_correct/")
                            or k.startswith("early_term_precision/")
                            or k.startswith("early_term_recall/")
                        ):
                            continue
                        if torch.is_tensor(v):
                            if v.numel() != 1:
                                continue
                            scalar = float(v.item())
                        elif isinstance(v, (int, float, np.integer, np.floating)):
                            scalar = float(v)
                        else:
                            continue
                        # TB tag rewrite: turn `logs_rew_<term>` into
                        # `logs_rew/<term>` so the dashboard groups all
                        # reward-component scalars under a single section.
                        tag = (
                            "logs_rew/" + k[len("logs_rew_"):]
                            if k.startswith("logs_rew_")
                            else k
                        )
                        for i in range(self.num_agents):
                            self.per_agent_tracking[i][tag].append(scalar)

        # Per-step is_success indicator — needed both for the training-side
        # add_samples (so the memory can later locate t* during finalize) and
        # for the diagnostic OR-accumulator below. Compute once up-front.
        has_key = isinstance(infos, dict) and self.success_info_key in infos
        step_success: torch.Tensor | None = None
        if has_key:
            raw = infos[self.success_info_key]
            if not torch.is_tensor(raw):
                raw = torch.as_tensor(raw, device=self.device)
            step_success = raw.to(self.device).bool().view(-1)

        if self.training:
            if self.cfg.rewards_shaper is not None:
                rewards = self.cfg.rewards_shaper(rewards, timestep, timesteps)
            extra_kwargs: dict[str, torch.Tensor] = {}
            if self._asymmetric:
                # Strict: in asymmetric mode the trainer MUST pass per-step states.
                if states is None or next_states is None:
                    raise RuntimeError(
                        "asymmetric SAC requires states and next_states from the "
                        "trainer (env.state() must return non-None). Got "
                        f"states={states is not None}, next_states={next_states is not None}."
                    )
                extra_kwargs["states"] = states
                extra_kwargs["next_states"] = next_states
            if self.predict_success:
                if step_success is None:
                    # Mirrors the predict_success branch below — same error
                    # so configs that fail to wrap the env fail the same way
                    # regardless of which path triggers first.
                    keys = list(infos.keys()) if isinstance(infos, dict) else type(infos).__name__
                    raise KeyError(
                        f"infos missing required key '{self.success_info_key}' for "
                        f"success prediction (predict_success=True). Got keys: {keys}."
                    )
                extra_kwargs["is_success_step"] = step_success.float().unsqueeze(-1)
            self.memory.add_samples(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                terminated=terminated,
                **extra_kwargs,
            )

        # ----- Success bookkeeping (diagnostic always; finalize when predict_success) -----
        # OR-accumulator drives the per-trajectory label used for the diagnostic
        # `Episode / Success rate` (always logged when info[success_info_key] is
        # present). Decoupled from `if self.training:` so eval rollouts also
        # publish that metric. The memory.finalize_trajectory call below is
        # gated on predict_success + training (it's what materializes the TD
        # target ingredients in the main buffer).
        if not has_key:
            if self.predict_success:
                keys = list(infos.keys()) if isinstance(infos, dict) else type(infos).__name__
                raise KeyError(
                    f"infos missing required key '{self.success_info_key}' for "
                    f"success prediction (predict_success=True). Got keys: {keys}. "
                    f"Wrap the env to emit it, or set sac_cfg.success_info_key, or "
                    f"set predict_success=false to disable the success head."
                )
            if not self._warned_no_success_key:
                print(
                    f"[SAC] info['{self.success_info_key}'] not provided; "
                    f"'Episode / Success rate' will not be logged. Wrap the env to enable."
                )
                self._warned_no_success_key = True
        elif self._traj_success_so_far is not None:
            if step_success.shape[0] != self._traj_success_so_far.shape[0]:
                raise ValueError(
                    f"infos['{self.success_info_key}'] has shape {tuple(step_success.shape)} "
                    f"but expected per-env tensor of length {self._traj_success_so_far.shape[0]}"
                )
            self._traj_success_so_far |= step_success

            done_mask = (terminated.bool() | truncated.bool()).view(-1)

            # Predictive-quality tracker: stage this step's (P, is_success) and
            # finalize trajectories on done. Lazily allocated on first call so
            # we can read num_envs from observed data instead of needing it at
            # __init__. Only fires when predict_success=True and the actor
            # actually emitted success_prob this step.
            if (
                self.predict_success
                and self._latest_success_prob is not None
                and self.write_interval > 0
            ):
                if self._pred_quality_tracker is None:
                    from learning.pred_quality import PredictionQualityTracker
                    max_ep_len = int(getattr(
                        self.memory, "max_episode_length", self.memory.num_envs
                    ))
                    self._pred_quality_tracker = PredictionQualityTracker(
                        num_envs=self.memory.num_envs,
                        num_agents=self.num_agents,
                        max_episode_length=max_ep_len,
                        device=self.device,
                    )
                # Pass per_env_curr_successes (instantaneous goal flag) when
                # the wrapper publishes it — gives the tracker a strict
                # at-end outcome label (matches the env's ``successes``
                # semantics). Falls back to "ever touched" when absent.
                curr_succ_for_tracker = infos.get("per_env_curr_successes") if isinstance(infos, dict) else None
                self._pred_quality_tracker.update(
                    success_prob=self._latest_success_prob,
                    is_success_step=step_success,
                    done_mask=done_mask,
                    curr_success=curr_succ_for_tracker,
                )

            if done_mask.any():
                finished = done_mask.nonzero(as_tuple=False).view(-1)
                labels = self._traj_success_so_far[finished].float()
                epa = self.memory.num_envs // self.num_agents
                # Diagnostic per-agent list (only populated when write_interval>0).
                if self._per_agent_episodes_this_interval:
                    for env_i, lbl in zip(finished.tolist(), labels.tolist()):
                        self._per_agent_episodes_this_interval[env_i // epa].append(int(lbl))
                # Materialize TD target ingredients in the main buffer — only
                # when we're training the success head.
                if self.predict_success and self.training:
                    self.memory.finalize_trajectory(env_indices=finished)
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
        # cfg.batch_size is PER AGENT. The memory's sample() interprets the
        # batch_size argument as per-agent and internally returns N * batch_size
        # rows partitioned [agent0 | agent1 | ...].
        B = self.cfg.batch_size

        # One-shot init-time action diagnostic: snapshot tanh saturation BEFORE any
        # gradient step touches the policy. Writes "Action / |a| ... (init)" tags so
        # init behavior is visible separately from the running average produced by
        # the in-loop tracking below (which is post-update by construction).
        if not getattr(self, "_logged_init_action_diag", False) and self.write_interval > 0:
            with torch.no_grad():
                init_sampled = self.memory.sample(
                    names=["observations"], batch_size=B
                )[0][0]
                init_inputs = {"observations": self._observation_preprocessor(init_sampled, train=False)}
                init_actions, _ = self.policy.act(init_inputs, role="policy")
                init_abs_a = init_actions.abs()
                init_sat = (init_abs_a > 0.99).float()
            init_split = lambda t: t.view(N, B, -1)
            self.track_per_agent("Action / |a| max (init)",  init_split(init_abs_a).amax(dim=(1, 2)))
            self.track_per_agent("Action / |a| mean (init)", init_split(init_abs_a).mean(dim=(1, 2)))
            self.track_per_agent("Action / saturation rate (init)", init_split(init_sat).mean(dim=(1, 2)))
            self._logged_init_action_diag = True

        for gradient_step in range(self.cfg.gradient_steps):
            sampled_list = self.memory.sample(
                names=self._tensors_names, batch_size=B
            )[0]
            sampled = dict(zip(self._tensors_names, sampled_list))
            sampled_observations = sampled["observations"]
            sampled_actions = sampled["actions"]
            sampled_rewards = sampled["rewards"]
            sampled_next_observations = sampled["next_observations"]
            sampled_terminated = sampled["terminated"]
            # Success-head TD ingredients (None when not predicting).
            sampled_is_first_succ = sampled.get("is_first_success_step")
            sampled_succ_terminal = sampled.get("success_terminal")
            sampled_succ_loss_mask = sampled.get("success_loss_mask")

            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                inputs = {
                    "observations": self._observation_preprocessor(sampled_observations, train=True),
                }
                next_inputs = {
                    "observations": self._observation_preprocessor(sampled_next_observations, train=True),
                }
                # In asymmetric mode, the critic consumes states (not obs). Build
                # separate input dicts for the critic networks; the actor still
                # uses the policy obs above.
                if self._asymmetric:
                    critic_inputs = {
                        "observations": self._state_preprocessor(sampled["states"], train=True),
                    }
                    critic_next_inputs = {
                        "observations": self._state_preprocessor(sampled["next_states"], train=True),
                    }
                else:
                    critic_inputs = inputs
                    critic_next_inputs = next_inputs

                with torch.no_grad():
                    next_actions, outputs = self.policy.act(next_inputs, role="policy")
                    next_log_prob = outputs["log_prob"]
                    # Success-head bootstrap value at s_{t+1} (used below to build
                    # the TD target for the success-prediction loss). Falls back
                    # to None when predict_success is off.
                    next_success_prob = outputs.get("success_prob")

                    target_q1_values, _ = self.target_critic_1.act(
                        {**critic_next_inputs, "taken_actions": next_actions}, role="target_critic_1"
                    )
                    target_q2_values, _ = self.target_critic_2.act(
                        {**critic_next_inputs, "taken_actions": next_actions}, role="target_critic_2"
                    )
                    ent_flat = self._expand_per_agent(self._entropy_coefficient, B)  # (N*B, 1)
                    target_q_values = torch.min(target_q1_values, target_q2_values) - ent_flat * next_log_prob
                    target_values = (
                        sampled_rewards
                        + self.cfg.discount_factor * sampled_terminated.logical_not() * target_q_values
                    )

                critic_1_values, _ = self.critic_1.act({**critic_inputs, "taken_actions": sampled_actions}, role="critic_1")
                critic_2_values, _ = self.critic_2.act({**critic_inputs, "taken_actions": sampled_actions}, role="critic_2")

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
                # Critic Q for the policy gradient: actor uses obs, critic uses state.
                critic_1_pi, _ = self.critic_1.act({**critic_inputs, "taken_actions": actions}, role="critic_1")
                critic_2_pi, _ = self.critic_2.act({**critic_inputs, "taken_actions": actions}, role="critic_2")

                ent_flat = self._expand_per_agent(self._entropy_coefficient, B)  # detached, no grad
                policy_loss = (ent_flat * log_prob - torch.min(critic_1_pi, critic_2_pi)).mean()

                # Success-prediction TD loss. Per-step bootstrap target:
                #   target = r + γ · (1 − terminal) · V(s_{t+1}).detach()
                # with r = is_first_success_step (1 at first-success, else 0)
                # and terminal = success_terminal (1 at first-success OR
                # failed-trajectory-end, else 0). At success-terminal the
                # target collapses to 1; at failure-terminal it collapses to 0
                # (no bootstrap, no reward); pre-terminal it bootstraps from
                # the actor's own next-obs success-prob estimate.
                # Loss is BCE between the live logit and the (soft) target,
                # masked by success_loss_mask (post-success transitions are
                # excluded since their label is undefined by design).
                # Backbone gradient comes from both this and the policy loss.
                bce_per_sample = None
                bce_masked_mean: torch.Tensor | None = None
                if self.predict_success:
                    if "success_logit" not in outputs:
                        raise RuntimeError(
                            "predict_success=True but policy.act() did not emit 'success_logit'. "
                            "Confirm BlockSimBaActor was constructed with predict_success=True."
                        )
                    if (
                        sampled_is_first_succ is None
                        or sampled_succ_terminal is None
                        or sampled_succ_loss_mask is None
                    ):
                        raise RuntimeError(
                            "predict_success=True but memory did not return success-head TD "
                            "tensors. Confirm SAC.init() registered "
                            "is_first_success_step / success_terminal / success_loss_mask "
                            "and the memory is TrajectoryBufferedMemory."
                        )
                    if next_success_prob is None:
                        raise RuntimeError(
                            "predict_success=True but next-obs forward pass did not emit "
                            "'success_prob'. Check BlockSimBaActor.act()."
                        )
                    success_logit = outputs["success_logit"]  # (N*B, 1), grad on
                    # TD target. ``next_success_prob`` is already detached
                    # (computed under torch.no_grad()).
                    td_target = (
                        sampled_is_first_succ
                        + self.success_td_discount
                        * (1.0 - sampled_succ_terminal)
                        * next_success_prob
                    ).clamp_(0.0, 1.0)
                    bce_per_sample = F.binary_cross_entropy_with_logits(
                        success_logit, td_target, reduction="none"
                    )
                    # Masked mean: average only over contributing rows. Guard
                    # against an all-masked batch (rare, but possible if every
                    # sampled row is a post-success state of a long trajectory).
                    mask_sum = sampled_succ_loss_mask.sum().clamp_min(1.0)
                    bce_masked_mean = (sampled_succ_loss_mask * bce_per_sample).sum() / mask_sum
                    policy_loss = policy_loss + self.success_td_weight * bce_masked_mean

            self.policy_optimizer.zero_grad()
            self.scaler.scale(policy_loss).backward()
            if config.torch.is_distributed:
                self.policy.reduce_parameters()
            # Unscale unconditionally so the per-agent grad-norm slices we
            # capture below reflect true (un-amped) magnitudes. If mixed
            # precision is off the scaler is a no-op; if on, scaler.step()
            # detects the prior unscale and skips a redundant pass.
            self.scaler.unscale_(self.policy_optimizer)
            if self.write_interval > 0:
                act_gn, succ_gn = self._compute_actor_head_grad_norms()
                self._last_action_head_grad_norm = act_gn
                self._last_success_head_grad_norm = succ_gn
            if self.cfg.grad_norm_clip > 0:
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

                # Continuous-action L2 norm — surfaces "do nothing" collapse.
                # If the policy parks all continuous dims near 0 (e.g. when the
                # entropy term dominates and the critic gradient is tiny), L2
                # norm trends toward 0. Excluding Bernoulli dims keeps {-1,+1}
                # gripper outputs from inflating the norm artificially.
                cont_idx = getattr(self.policy, "_cont_action_idx", None)
                if cont_idx is not None and cont_idx.numel() > 0:
                    with torch.no_grad():
                        cont_actions = actions.index_select(-1, cont_idx)   # (N*B, num_cont)
                        cont_l2 = cont_actions.norm(dim=-1)                 # (N*B,)
                        cont_l2_per_agent = cont_l2.view(N, -1)             # (N, B)
                    self.track_per_agent("Action / continuous L2 (max)",  cont_l2_per_agent.amax(dim=1))
                    self.track_per_agent("Action / continuous L2 (min)",  cont_l2_per_agent.amin(dim=1))
                    self.track_per_agent("Action / continuous L2 (mean)", cont_l2_per_agent.mean(dim=1))
                    self.track_per_agent("Action / continuous L2 (std)",  cont_l2_per_agent.std(dim=1))

                # Gripper diagnostic — open rate is the headline metric. If it's
                # stuck near 0 or 1 the gripper is locked and the agent can't grasp.
                gidx = self.cfg.gripper_action_idx
                if gidx is not None:
                    with torch.no_grad():
                        g = actions[..., gidx].unsqueeze(-1)         # (N*B, 1)
                        g_open = (g >= 0).float()
                    self.track_per_agent("Gripper / open rate",   split(g_open).mean(dim=(1, 2)))
                    self.track_per_agent("Gripper / action mean", split(g).mean(dim=(1, 2)))
                    self.track_per_agent("Gripper / action std",  split(g).flatten(1).std(dim=1))

                if self.cfg.learn_entropy:
                    self.track_per_agent("Loss / Entropy loss", entropy_loss_per_agent.squeeze(-1))
                    self.track_per_agent("Coefficient / Entropy coefficient",
                                         self._entropy_coefficient.squeeze(-1))

                # Per-agent success-head TD loss + diagnostic stats (training
                # head only). Reports the **masked** BCE loss (post-success
                # rows excluded) so the value matches what the optimizer sees.
                # The rolling "Episode / Success rate" diagnostic is emitted
                # from write_tracking_data, independent of predict_success.
                # All success-head training-time scalars share a single TB tab
                # `Success Prediction Quality / *` (single-slash tags only).
                # Values are masked-mean over non-post-success rows so they
                # match what the optimizer actually sees; the rolling
                # `Episode / Success rate` diagnostic is emitted from
                # write_tracking_data independently.
                if self.predict_success and bce_per_sample is not None:
                    bce_p = split(bce_per_sample)            # (N, B, 1)
                    mask_p = split(sampled_succ_loss_mask)   # (N, B, 1)
                    mask_sum_p = mask_p.sum(dim=(1, 2)).clamp_min(1.0)  # (N,)
                    self.track_per_agent(
                        "Success Prediction Quality / BCE success loss",
                        (mask_p * bce_p).sum(dim=(1, 2)) / mask_sum_p,
                    )
                    succ_prob_p = split(outputs["success_prob"])  # (N, B, 1)
                    self.track_per_agent(
                        "Success Prediction Quality / Success prob (mean)",
                        (mask_p * succ_prob_p).sum(dim=(1, 2)) / mask_sum_p,
                    )
                    target_p = split(td_target)
                    self.track_per_agent(
                        "Success Prediction Quality / Success TD target (mean)",
                        (mask_p * target_p).sum(dim=(1, 2)) / mask_sum_p,
                    )
                    self.track_per_agent(
                        "Success Prediction Quality / Success loss-mask rate",
                        mask_p.mean(dim=(1, 2)),
                    )
                    if self._last_success_head_grad_norm is not None:
                        self.track_per_agent(
                            "Success Prediction Quality / success head grad norm",
                            self._last_success_head_grad_norm,
                        )

                # Action-head grad norm: lives under the actor diagnostics
                # tab (Action / *) since it's an actor health metric, not a
                # success-prediction quality one. Always tracked when
                # available (does not require predict_success).
                if self._last_action_head_grad_norm is not None:
                    self.track_per_agent(
                        "Action / action head grad norm",
                        self._last_action_head_grad_norm,
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
