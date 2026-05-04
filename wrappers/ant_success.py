"""Per-step success injection for ``Isaac-Ant-v0`` (and any Ant-prefixed task).

Inherits per-env reward decomposition from :class:`RewardDecompositionWrapper`
and adds three pieces of bookkeeping:

1. ``info["is_success"]`` per step, defined as:

       is_success[i] = (current_root_x[i] - initial_root_x[i]) >= success_threshold

   SAC's existing OR-accumulator latches True if this is ever satisfied during an
   episode, so the trajectory-level success label = "agent walked >= N meters this
   episode," and ``Episode / Success rate`` reports the fraction of recent episodes
   that crossed the threshold.

2. ``info["per_env_episode_distance"]`` + ``..._mask`` on episode-end steps:
   per-env final forward displacement (m) of the just-ended trajectory. SAC
   aggregates these per write_interval and publishes
   ``Episode / Distance traveled (max/min/mean)``.

3. ``info["per_env_episode_velocity"]`` + ``..._mask`` on episode-end steps:
   per-env average forward velocity (m/s) over the just-ended trajectory
   (= total displacement / total duration). SAC aggregates these per
   write_interval and publishes ``Episode / Velocity (max/min/mean)``.

Default threshold is 25 m — for an Ant policy that has actually learned to walk
(~1 m/s sustained over a 16-second episode → ~16 m), a 25 m threshold provides
useful gradient on success rate even as the policy improves.
"""

from __future__ import annotations

from typing import Any

import torch

from wrappers.reward_decomposition import RewardDecompositionWrapper


class AntSuccessWrapper(RewardDecompositionWrapper):
    """Adds ``info["is_success"]`` + per-trajectory distance on top of
    generic reward decomposition."""

    def __init__(
        self,
        env: Any,
        *,
        success_threshold: float = 70.0,
        info_key: str = "is_success",
    ) -> None:
        super().__init__(env)
        self._success_threshold = float(success_threshold)
        self._success_info_key = str(info_key)
        # Lazily allocated on the first step() call so we know num_envs/device.
        self._initial_x: torch.Tensor | None = None
        # Per-env "current episode forward displacement" — refreshed every non-
        # reset step. On the step where an env resets, this still holds the
        # previous step's value, which we treat as the "final displacement" of
        # the just-ended episode. (We can't read pre-reset position any later;
        # Isaac Lab's _reset_idx fires inside env.step() and cur_x is post-reset.)
        self._last_displacement: torch.Tensor | None = None
        # Per-env step counter for the current episode, refreshed in lockstep
        # with _last_displacement. On the terminating step it freezes at the
        # count up through the previous (non-reset) step, so dividing
        # _last_displacement by _episode_steps × dt gives a consistent
        # average-velocity estimate over the same windowed steps.
        self._episode_steps: torch.Tensor | None = None
        print(f"[AntSuccessWrapper] success_threshold = {self._success_threshold}")

    def step(self, actions):
        # super().step() runs the env (including any internal _reset_idx) and
        # injects per_env_rew + per_env_rew_mask via the inherited path.
        obs, reward, terminated, truncated, info = super().step(actions)

        asset = self._unwrapped.scene["robot"]
        cur_x = asset.data.root_pos_w[:, 0]  # (num_envs,) world-frame x

        if self._initial_x is None:
            # First call after env init — current pose IS the initial pose.
            self._initial_x = cur_x.clone()
            self._last_displacement = torch.zeros_like(cur_x)
            self._episode_steps = torch.zeros(
                cur_x.shape[0], dtype=torch.long, device=cur_x.device
            )

        done = (terminated | truncated).view(-1).to(self._initial_x.device)

        # Update per-env episode displacement + step counter ONLY for envs that
        # did NOT just reset. For resetting envs, both retain their previous
        # step's value (≈ final state of the just-ended episode).
        not_done = ~done
        if not_done.any():
            self._last_displacement[not_done] = (
                cur_x[not_done] - self._initial_x[not_done]
            )
            self._episode_steps[not_done] += 1

        # Compute per-env average velocity for the (just-ended OR ongoing)
        # episode using consistent windowed values. Clamp denominator to dt to
        # avoid div-by-zero on the very first step before any non-reset update.
        dt = float(self._unwrapped.step_dt)
        duration = torch.clamp(self._episode_steps.float() * dt, min=dt)
        avg_velocity = self._last_displacement / duration

        # Publish per-trajectory distance + velocity for envs that just ended.
        # SAC consumes these at write_interval boundaries (max/min/mean). Other
        # envs' values in the tensors are ignored via the mask.
        info["per_env_episode_distance"] = self._last_displacement.clone()
        info["per_env_episode_distance_mask"] = done
        info["per_env_episode_velocity"] = avg_velocity.clone()
        info["per_env_episode_velocity_mask"] = done

        # Now (after capturing) update _initial_x and step counter for resetting
        # envs. cur_x for those envs is the new episode's starting position.
        if done.any():
            self._initial_x[done] = cur_x[done]
            self._episode_steps[done] = 0

        # Per-step success flag uses the up-to-date diff. For envs that just
        # reset, diff is 0 (initial == current), so is_success starts False.
        info[self._success_info_key] = (cur_x - self._initial_x) >= self._success_threshold
        return obs, reward, terminated, truncated, info
