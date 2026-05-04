"""Per-env reward decomposition + success injection for ``Isaac-Factory-*`` tasks.

Factory is a direct-API env (``DirectRLEnv``) — no ``reward_manager``, so the parent
:class:`RewardDecompositionWrapper`'s monkey-patch on ``RewardManager.reset``
gracefully no-ops. Per-env per-term decomposition is recovered here by hooking
``_log_factory_metrics``, which receives the full per-env ``rew_dict`` before
it's collapsed to a scalar in ``extras``.

Mirrors :class:`ForgeWrapper` but **without** the Forge-specific
``_log_forge_metrics`` hook (Factory has no `success_pred_error` reward term —
success prediction is handled by our own SAC ``predict_success`` head, trained
via BCE against the per-trajectory ``ep_succeeded`` label that we inject as
``info["is_success"]``).

Action space is 6 (no Forge-style success-prediction action dim). All 6 dims
are continuous, so YAML should set ``bernoulli_action_dims: null`` and
``force_zero_action_dims: null``.
"""

from __future__ import annotations

from typing import Any

import torch

from wrappers.reward_decomposition import RewardDecompositionWrapper


class FactoryWrapper(RewardDecompositionWrapper):
    """Per-env reward decomposition + per-step success flag for Factory tasks."""

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        # Per-env per-term running episode sums (in scaled units, same as the
        # env-returned reward). Lazily populated when the first reward log fires.
        self._episode_sums: dict[str, torch.Tensor] = {}
        # Latest per-env curr_successes captured by the reward-log hook (per-step
        # geometric success indicator).
        self._latest_curr_successes: torch.Tensor | None = None
        # Latest per-env unscaled rew_dict (per-step) captured by the reward-log
        # hook. Used to emit per-agent `logs_rew/<term>` to TB by partitioning
        # the per-env values across agent slices in the SAC consumer.
        self._latest_rew_dict: dict[str, torch.Tensor] = {}
        self._install_factory_reward_hook()

    # ------------------------------------------------------------------
    # Hook installation
    # ------------------------------------------------------------------
    def _install_factory_reward_hook(self) -> None:
        unwrapped = self._unwrapped
        original_factory_log = unwrapped._log_factory_metrics

        def hooked_factory_log(rew_dict, curr_successes):
            self._latest_curr_successes = curr_successes.clone()
            # Capture per-env unscaled rew_dict for per-agent logs_rew/<term>.
            # Only per-env tensors are publishable per-agent; upstream Factory
            # has at least one global scalar term (`action_penalty_ee =
            # torch.norm(actions, p=2)` — no dim arg, collapses to 0-dim).
            # Drop those here so SAC's per-agent partition path doesn't see
            # un-splittable values; the global mirror in info["log"] still emits.
            num_envs = self._unwrapped.num_envs
            self._latest_rew_dict = {
                term: val.clone()
                for term, val in rew_dict.items()
                if isinstance(val, torch.Tensor) and val.dim() > 0 and val.shape[0] == num_envs
            }
            self._accumulate_per_env_term(rew_dict, self._factory_scales())
            return original_factory_log(rew_dict, curr_successes)

        unwrapped._log_factory_metrics = hooked_factory_log

    # ------------------------------------------------------------------
    # Reward scale lookup
    # ------------------------------------------------------------------
    def _factory_scales(self) -> dict[str, float]:
        """Static factory reward scales from ``cfg_task``. Mirrors the rew_scales
        dict constructed in ``FactoryEnv._get_factory_rew_dict`` (keypoint and
        engagement terms have scale 1.0; action penalties are negative)."""
        cfg = self._unwrapped.cfg_task
        return {
            "kp_baseline": 1.0,
            "kp_coarse": 1.0,
            "kp_fine": 1.0,
            "action_penalty_ee": -float(cfg.action_penalty_ee_scale),
            "action_grad_penalty": -float(cfg.action_grad_penalty_scale),
            "curr_engaged": 1.0,
            "curr_success": 1.0,
        }

    def _accumulate_per_env_term(
        self, rew_dict: dict[str, torch.Tensor], scales: dict[str, float]
    ) -> None:
        """Add this step's scaled per-env per-term reward into ``_episode_sums``."""
        device = self._unwrapped.device
        num_envs = self._unwrapped.num_envs
        for term, val in rew_dict.items():
            if not isinstance(val, torch.Tensor):
                continue
            v = val.view(num_envs, -1).sum(dim=-1) if val.dim() > 1 else val
            scaled = v * scales.get(term, 0.0)
            if term not in self._episode_sums:
                self._episode_sums[term] = torch.zeros(num_envs, device=device)
            self._episode_sums[term] += scaled

    # ------------------------------------------------------------------
    # step() — publish per_env_rew on episode end + per-step is_success
    # ------------------------------------------------------------------
    def step(self, actions):
        obs, reward, terminated, truncated, info = super().step(actions)

        # Per-step success flag. ``ep_succeeded[i]`` is True for any env that
        # reached the success_threshold at any point in the current episode.
        # SAC's OR-accumulator latches True; even though _reset_idx zeroes
        # ep_succeeded on terminating envs DURING this step, prior steps
        # already latched True in the accumulator, so trajectory labels are
        # captured correctly at finalize_trajectory.
        info["is_success"] = self._unwrapped.ep_succeeded.bool().clone()

        # Publish per-env tensors so SAC can partition by agent and emit
        # per-agent metrics: `logs_rew/<term>` (unscaled per-step reward
        # components), `successes` (fraction of agent's envs at goal at reset
        # moment, mirrors upstream Factory semantics), and `success_times`
        # (mean step-of-first-success across agent's envs that have succeeded).
        info["per_env_logs_rew"] = self._latest_rew_dict
        if self._latest_curr_successes is not None:
            info["per_env_curr_successes"] = self._latest_curr_successes
        info["per_env_ep_success_times"] = self._unwrapped.ep_success_times.clone()

        # Per-env episode-end capture: at this point, _accumulate_per_env_term
        # has already added the terminating step's contribution to
        # _episode_sums (the hook fires inside super().step()). Snapshot the
        # values for resetting envs, then clear those slots for the next
        # episode.
        done = (terminated | truncated).view(-1).to(self._unwrapped.device)
        if done.any():
            num_envs = self._unwrapped.num_envs
            mask = torch.zeros(num_envs, dtype=torch.bool, device=self._unwrapped.device)
            mask[done.nonzero(as_tuple=False).view(-1)] = True

            per_env: dict[str, torch.Tensor] = {}
            for term, sums in self._episode_sums.items():
                full = torch.zeros(num_envs, dtype=sums.dtype, device=sums.device)
                full[mask] = sums[mask]
                per_env[term] = full
                # Reset accumulators for resetting envs (start fresh next episode).
                sums[mask] = 0.0

            info["per_env_rew"] = per_env
            info["per_env_rew_mask"] = mask

        return obs, reward, terminated, truncated, info
