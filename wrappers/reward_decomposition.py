"""Generic per-env reward decomposition for Isaac Lab manager-based envs.

Hooks ``RewardManager.reset`` to capture per-env per-term episode sums BEFORE
they're zeroed, then publishes them in ``info["per_env_rew"]`` plus a
``info["per_env_rew_mask"]`` bool tensor marking which envs reset this step.
SAC consumes those to log ``Episode_Reward/<term>`` per agent in per-episode
units (raw episode sums, summing across terms equals the per-env episode return).

Task-agnostic: works for any env that has a manager-based ``reward_manager``.
For envs without one (e.g. direct-API envs), the hook silently no-ops and the
wrapper behaves like a plain ``IsaacLabWrapper``.
"""

from __future__ import annotations

from typing import Any

import torch

from skrl.envs.wrappers.torch.isaaclab_envs import IsaacLabWrapper


class RewardDecompositionWrapper(IsaacLabWrapper):
    """Captures Isaac Lab's RewardManager per-env per-term episode sums on reset
    and emits them in ``info["per_env_rew"]`` for SAC to partition + log."""

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        # Set BEFORE attempting to install the hook in case _install fails the check.
        self._captured_per_env: dict[str, torch.Tensor] | None = None
        self._captured_env_ids: torch.Tensor | None = None
        # The first ``env.reset()`` (called by the trainer before env.step() loops)
        # also fires ``reward_manager.reset(all_envs)`` even though no real episode
        # has run; ``_episode_sums`` are all zeros at that point. Capturing it would
        # publish a spurious all-zero per_env_rew on the first step() call, dragging
        # down the first TB write (e.g. Episode_Reward/lifting_object reads 0
        # instead of the spawn-settle floor of ~0.6). Flip this flag inside step()
        # and skip the capture until then.
        self._step_fired: bool = False
        self._install_reward_manager_hook()

    def _install_reward_manager_hook(self) -> None:
        rm = getattr(self._unwrapped, "reward_manager", None)
        if rm is None:
            # Direct-API envs / non-manager-based envs: gracefully skip.
            # The wrapper still works (just doesn't publish per_env_rew).
            return

        original_reset = rm.reset

        def hooked_reset(env_ids=None):
            # Skip captures that happen BEFORE the first env.step() — those are
            # the trainer's initial env.reset(), where _episode_sums are all
            # zeros (no episodes have run yet) and capturing would publish a
            # spurious all-zero per_env_rew event.
            if env_ids is not None and self._step_fired:
                if isinstance(env_ids, torch.Tensor):
                    ids = env_ids
                else:
                    ids = torch.as_tensor(env_ids, device=self._unwrapped.device)
                if ids.numel() > 0:
                    self._captured_per_env = {
                        term: rm._episode_sums[term][ids].clone()
                        for term in rm._episode_sums
                    }
                    self._captured_env_ids = ids.clone()
            return original_reset(env_ids)

        rm.reset = hooked_reset

    def step(self, actions):
        # Mark BEFORE delegating: if super().step() triggers _reset_idx for
        # terminated envs (the real per-step reset path), the hook should
        # capture those.
        self._step_fired = True
        obs, reward, terminated, truncated, info = super().step(actions)
        self._inject_per_env_rew(info)
        return obs, reward, terminated, truncated, info

    def _inject_per_env_rew(self, info: dict) -> None:
        """Publish captured per-env per-term episode sums into info, then clear."""
        if self._captured_per_env is None or self._captured_env_ids is None:
            return

        num_envs = self._unwrapped.num_envs
        device = self._unwrapped.device
        mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        mask[self._captured_env_ids] = True

        per_env: dict[str, torch.Tensor] = {}
        for term, captured_vals in self._captured_per_env.items():
            full = torch.zeros(num_envs, dtype=captured_vals.dtype, device=device)
            full[self._captured_env_ids] = captured_vals
            per_env[term] = full

        info["per_env_rew"] = per_env
        info["per_env_rew_mask"] = mask

        self._captured_per_env = None
        self._captured_env_ids = None
