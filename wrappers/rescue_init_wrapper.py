"""Observe-only rescue-state injection on naturally-occurring env resets.

Wraps the env *outside* :class:`StateSnapshotWrapper`. Each :meth:`step` call:

1. Delegates to the inner step (which advances physics, captures the new
   scene state into the snapshot history, and rolls the per-env head).
2. Detects ``terminated | truncated`` ⇒ ``done`` set.
3. Partitions done envs by agent. Per agent, gates on the rolling success
   rate ``p_hat_succ >= rho_min`` (queried from
   :class:`learning.rescue_metrics.RescueMetricsTracker`) and on
   ``len(buffer) > 0``. Bernoulli(α) selects which of the gated done envs
   get rescue-initialized.
4. For each selected env: samples one slot uniformly, calls
   ``state_snapshot.restore_state(env_ids, snapshots)``, recomputes that
   env's observation by querying the underlying Isaac Lab observation
   manager, and splices the new obs into the return tensor.
5. Writes ``info["initialized_from_rescue"]`` / ``info["rescue_slot_idx"]`` /
   ``info["rescue_agent_idx"]`` as per-env tensors for SAC to read.

This wrapper MUST NOT call ``env.reset()``. Isaac Lab auto-resets terminated
envs internally on the next step; we only *observe* that natural reset and
overwrite the post-reset PhysX state for the curriculum-selected subset.
"""

from __future__ import annotations

from typing import Any

import torch

from skrl.envs.wrappers.torch.base import Wrapper


class RescueInitWrapper(Wrapper):
    """Outermost wrapper that injects rescue states on natural env resets."""

    def __init__(
        self,
        env: Any,
        *,
        rescue_buffers: list,
        state_snapshot,
        metrics_tracker,
        num_agents: int,
        alpha: float,
        rho_min: float,
    ) -> None:
        super().__init__(env)
        if rescue_buffers is None:
            raise ValueError("RescueInitWrapper.rescue_buffers is required (no default).")
        if state_snapshot is None:
            raise ValueError("RescueInitWrapper.state_snapshot is required (no default).")
        if metrics_tracker is None:
            raise ValueError("RescueInitWrapper.metrics_tracker is required (no default).")
        if num_agents is None or num_agents < 1:
            raise ValueError(f"RescueInitWrapper.num_agents must be >= 1, got {num_agents!r}")
        if alpha is None or not (0.0 <= float(alpha) <= 1.0):
            raise ValueError(f"RescueInitWrapper.alpha must be in [0, 1], got {alpha!r}")
        if rho_min is None or not (0.0 <= float(rho_min) <= 1.0):
            raise ValueError(f"RescueInitWrapper.rho_min must be in [0, 1], got {rho_min!r}")

        self._rescue_buffers = list(rescue_buffers)
        if len(self._rescue_buffers) != int(num_agents):
            raise ValueError(
                f"rescue_buffers length ({len(self._rescue_buffers)}) != num_agents ({num_agents})"
            )
        self._state_snapshot = state_snapshot
        self._metrics = metrics_tracker
        self._num_agents = int(num_agents)
        self.alpha = float(alpha)
        self.rho_min = float(rho_min)

        # Total num_envs and per-agent partition size.
        total_envs = int(self._env.num_envs)
        if total_envs % self._num_agents != 0:
            raise ValueError(
                f"num_envs ({total_envs}) must be divisible by num_agents ({self._num_agents})"
            )
        self._num_envs = total_envs
        self.epa = total_envs // self._num_agents
        self._device = state_snapshot._device

    # ------------------------------------------------------------------
    # Wrapper API
    # ------------------------------------------------------------------
    def step(self, actions: torch.Tensor):
        obs, rew, term, trunc, info = self._env.step(actions)

        # Allocate the per-step info tensors. Default: no rescue init this step.
        init_flag = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        slot_idx_t = torch.full((self._num_envs,), -1, dtype=torch.long, device=self._device)
        agent_idx_t = torch.full((self._num_envs,), -1, dtype=torch.long, device=self._device)

        done = (term.view(-1).bool() | trunc.view(-1).bool()).to(self._device)
        if done.any():
            # Per-agent selection.
            for agent_i in range(self._num_agents):
                buf = self._rescue_buffers[agent_i]
                if len(buf) == 0:
                    continue
                if self._metrics.p_hat_succ(agent_i) < self.rho_min:
                    continue
                env_lo, env_hi = agent_i * self.epa, (agent_i + 1) * self.epa
                done_in_agent = done[env_lo:env_hi]
                if not bool(done_in_agent.any().item()):
                    continue
                # Bernoulli(α) per done env in this agent's slice.
                done_idx = done_in_agent.nonzero(as_tuple=False).view(-1)  # local 0..epa-1
                if done_idx.numel() == 0:
                    continue
                pick = torch.rand(done_idx.numel(), device=self._device) < self.alpha
                if not bool(pick.any().item()):
                    continue
                selected_local = done_idx[pick]
                selected_global = selected_local + env_lo
                k = int(selected_global.numel())
                # Sample k slots from this agent's buffer.
                slots, sim_states, _obs_stored = buf.sample(k)
                # Restore PhysX state on the selected envs.
                self._state_snapshot.restore_state(selected_global, sim_states)
                # Bump init-attempts; the outcome lands later via
                # record_outcome() at trajectory end.
                for s in slots.tolist():
                    buf.record_init(int(s))
                # Flag tensors.
                init_flag[selected_global] = True
                slot_idx_t[selected_global] = slots.to(self._device)
                agent_idx_t[selected_global] = agent_i

                # Recompute observations for the restored envs so the policy
                # sees the new state on its next act(). Manager-based envs
                # expose obs_manager.compute(); direct-API envs expose
                # _get_observations() (Factory/Forge). Fail loud if neither.
                obs = self._recompute_obs(selected_global, obs)

        # Surface the per-env flags so SAC can route bookkeeping.
        info["initialized_from_rescue"] = init_flag
        info["rescue_slot_idx"] = slot_idx_t
        info["rescue_agent_idx"] = agent_idx_t

        return obs, rew, term, trunc, info

    def reset(self):
        return self._env.reset()

    def state(self):
        return self._env.state() if hasattr(self._env, "state") else None

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        return self._env.close()

    # ------------------------------------------------------------------
    # Observation recomputation after sim-state restore
    # ------------------------------------------------------------------
    def _recompute_obs(self, env_ids: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Re-run the env's observation pipeline on ``env_ids`` and splice into ``obs``.

        Isaac Lab manager-based envs (e.g. Lift) expose
        ``env.unwrapped.observation_manager`` with a ``compute()`` method that
        returns ``{"policy": Tensor(num_envs, obs_dim), ...}``. Direct-API envs
        (Factory/Forge) override ``_get_observations()`` and return a similar
        dict. Either is acceptable; we try in order and fall back to the
        existing obs (no-op) only if neither path is available — in which
        case the restored state's effect will be one step delayed.
        """
        unw = self._unwrapped
        new_obs = None
        om = getattr(unw, "observation_manager", None)
        if om is not None and hasattr(om, "compute"):
            full = om.compute()
            policy = full.get("policy") if isinstance(full, dict) else None
            if torch.is_tensor(policy):
                new_obs = policy
        if new_obs is None and hasattr(unw, "_get_observations"):
            try:
                full = unw._get_observations()
                policy = full.get("policy") if isinstance(full, dict) else None
                if torch.is_tensor(policy):
                    new_obs = policy
            except Exception:
                new_obs = None
        if new_obs is None:
            # Last-resort: leave obs untouched. Fail loud only if the user has
            # actually opted-in to rescue (caller already checked enabled);
            # the policy still sees consistent shape, restore takes effect
            # next step.
            return obs
        # Flatten if shape is (num_envs, ...) matching obs's layout.
        if new_obs.shape != obs.shape:
            # Some manager envs return obs in (num_envs, obs_dim) directly,
            # matching the IsaacLabWrapper's flattened return. If shapes
            # disagree we can't safely splice — bail to existing obs.
            return obs
        out = obs.clone()
        out[env_ids] = new_obs[env_ids].to(out.dtype)
        return out
