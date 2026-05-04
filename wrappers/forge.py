"""Per-env reward decomposition + success injection for ``Isaac-Forge-*`` tasks.

Forge envs are direct (not manager-based), so they don't have a ``reward_manager``
with ``_episode_sums`` — the parent :class:`RewardDecompositionWrapper`'s hook
gracefully no-ops for direct envs. We replace it here with hooks on Factory's
``_log_factory_metrics`` and Forge's ``_log_forge_metrics``, which receive the
full per-env-per-term ``rew_dict`` before it's mean'd into ``extras``.

Each per-step capture is multiplied by the term's reward scale (read from
``cfg_task`` for static scales and from instance attributes for dynamic ones
like ``success_pred_scale``), then accumulated into a per-env per-term episode
sum. On episode end, the accumulated values are published in
``info["per_env_rew"]`` (same convention as our manager-based path), so SAC's
existing consumer partitions by agent and emits ``Episode_Reward/<term>`` per
agent in per-episode units.

We also inject ``info["is_success"]`` per step from ``env.ep_succeeded`` (the
Factory env's per-env "did this episode reach success at any point" buffer).
SAC's existing OR-accumulator + ``finalize_trajectory`` path picks it up so
``Episode / Success rate`` reflects the fraction of recent episodes that
reached the ``success_threshold``.
"""

from __future__ import annotations

from typing import Any

import torch

from wrappers.reward_decomposition import RewardDecompositionWrapper


class ForgeWrapper(RewardDecompositionWrapper):
    """Per-env reward decomposition + per-step success flag for Forge tasks."""

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        # Per-env per-term running episode sums (in scaled units, same as the
        # env-returned reward). Lazily populated when the first reward log fires.
        self._episode_sums: dict[str, torch.Tensor] = {}
        # Latest per-env curr_successes captured by the reward-log hook (per-step
        # geometric success indicator).
        self._latest_curr_successes: torch.Tensor | None = None
        # Latest per-env unscaled rew_dict (per-step). Combined across both the
        # factory and forge log hooks for per-agent logs_rew/<term> metrics.
        self._latest_rew_dict: dict[str, torch.Tensor] = {}
        self._install_forge_reward_hooks()

    # ------------------------------------------------------------------
    # Hook installation
    # ------------------------------------------------------------------
    def _install_forge_reward_hooks(self) -> None:
        unwrapped = self._unwrapped

        # Both methods are called from inside _get_rewards each step. They
        # receive rew_dict (per-env per-term tensors) BEFORE it's reduced to a
        # scalar in extras, so we can capture the per-env values here.
        original_factory_log = unwrapped._log_factory_metrics

        def hooked_factory_log(rew_dict, curr_successes):
            self._latest_curr_successes = curr_successes.clone()
            # Capture per-env unscaled rew_dict (factory terms) for per-agent
            # logs_rew/<term>. Only per-env tensors (shape (num_envs,)) are
            # publishable per-agent; upstream Factory has at least one global
            # scalar term (`action_penalty_ee = torch.norm(actions, p=2)`),
            # which collapses across all envs and can't be split — keep it
            # off this path so the global mirror in info["log"] still emits it.
            num_envs = self._unwrapped.num_envs
            for term, val in rew_dict.items():
                if isinstance(val, torch.Tensor) and val.dim() > 0 and val.shape[0] == num_envs:
                    self._latest_rew_dict[term] = val.clone()
            self._accumulate_per_env_term(rew_dict, self._factory_scales())
            return original_factory_log(rew_dict, curr_successes)

        unwrapped._log_factory_metrics = hooked_factory_log

        original_forge_log = unwrapped._log_forge_metrics

        def hooked_forge_log(rew_dict, policy_success_pred):
            # Capture per-env unscaled rew_dict (forge terms) too; merge with
            # factory terms captured above. Same per-env shape filter — global
            # scalar terms get dropped here so SAC's per-agent partition only
            # sees splittable values.
            num_envs = self._unwrapped.num_envs
            for term, val in rew_dict.items():
                if isinstance(val, torch.Tensor) and val.dim() > 0 and val.shape[0] == num_envs:
                    self._latest_rew_dict[term] = val.clone()
            self._accumulate_per_env_term(rew_dict, self._forge_scales())
            return original_forge_log(rew_dict, policy_success_pred)

        unwrapped._log_forge_metrics = hooked_forge_log

    # ------------------------------------------------------------------
    # Reward scale lookup
    # ------------------------------------------------------------------
    def _factory_scales(self) -> dict[str, float]:
        """Static factory reward scales from cfg_task. Mirrors the rew_scales dict
        constructed in ``FactoryEnv._get_factory_rew_dict`` (keypoint and engagement
        terms have scale 1.0; action penalties are negative)."""
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

    def _forge_scales(self) -> dict[str, float]:
        """Forge-specific reward scales. ``success_pred_error`` is dynamic — it
        ramps from 0 to 1 once a fraction of envs have demonstrated true success
        (see ``forge_env.py``), so we read it from the env each step."""
        cfg = self._unwrapped.cfg_task
        return {
            "action_penalty_asset": -float(cfg.action_penalty_asset_scale),
            "contact_penalty": -float(cfg.contact_penalty_scale),
            "success_pred_error": -float(self._unwrapped.success_pred_scale),
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
            # rew_dict values can be (num_envs,) or (num_envs, ...) — collapse to (num_envs,)
            v = val.view(num_envs, -1).sum(dim=-1) if val.dim() > 1 else val
            scaled = v * scales.get(term, 0.0)
            if term not in self._episode_sums:
                self._episode_sums[term] = torch.zeros(num_envs, device=device)
            self._episode_sums[term] += scaled

    # ------------------------------------------------------------------
    # step() — publish per_env_rew on episode end + per-step is_success
    # ------------------------------------------------------------------
    def step(self, actions):
        # super().step() in RewardDecompositionWrapper would normally inject
        # per_env_rew via the reward_manager hook — but Forge has no reward
        # manager so that hook is a no-op. We override its inject below.
        obs, reward, terminated, truncated, info = super().step(actions)

        # Per-step success flag: env.ep_succeeded is True for any env that has
        # reached success at any point in the current (still-ongoing OR just-
        # ended-this-step) episode. SAC's OR-accumulator latches on this; even
        # though _reset_idx zeroes ep_succeeded on terminating envs *during*
        # this step, prior steps already latched True in the accumulator, so
        # the trajectory label remains correct.
        info["is_success"] = self._unwrapped.ep_succeeded.bool().clone()

        # Publish per-env tensors so SAC can partition by agent and emit
        # per-agent versions of `logs_rew/<term>`, `successes`, `success_times`
        # (mirrors upstream Factory's extras semantics, per-agent). For Forge,
        # also publish first_pred_success_tx (one tensor per threshold) so SAC
        # can compute per-agent early_term_*/<thresh> metrics.
        info["per_env_logs_rew"] = self._latest_rew_dict
        if self._latest_curr_successes is not None:
            info["per_env_curr_successes"] = self._latest_curr_successes
        info["per_env_ep_success_times"] = self._unwrapped.ep_success_times.clone()
        # Forge-specific: per-threshold first-prediction-success step.
        first_pred = getattr(self._unwrapped, "first_pred_success_tx", None)
        if first_pred is not None:
            info["per_env_first_pred_success_tx"] = {
                thresh: tx.clone() for thresh, tx in first_pred.items()
            }

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
