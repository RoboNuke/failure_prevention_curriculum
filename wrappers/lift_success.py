"""Per-step success injection for ``Isaac-Lift-Cube-Franka-v0``.

Inherits per-env reward decomposition from :class:`RewardDecompositionWrapper`
and adds:

* ``info["is_success"]`` per step — computed via Isaac Lab's stock
  ``object_reached_goal`` termination function (object position within
  ``threshold`` meters of the commanded goal pose). SAC consumes this for the
  rolling success-rate diagnostic and for the optional success-prediction head.

The success-condition function and its threshold default match what Isaac Lab
ships in
``isaaclab_tasks.manager_based.manipulation.lift.mdp.terminations.object_reached_goal``;
in the stock Lift cfg this term exists but is **not** registered as an active
termination, so it has no effect on episode boundaries — only on our injected
``info`` flag.
"""

from __future__ import annotations

from typing import Any

from isaaclab_tasks.manager_based.manipulation.lift.mdp.terminations import (
    object_reached_goal,
)

from wrappers.reward_decomposition import RewardDecompositionWrapper


class LiftSuccessWrapper(RewardDecompositionWrapper):
    """Adds ``info["is_success"]`` per step on top of generic reward decomposition."""

    def __init__(
        self,
        env: Any,
        *,
        threshold: float = 0.02,
        command_name: str = "object_pose",
        info_key: str = "is_success",
    ) -> None:
        super().__init__(env)
        self._success_threshold = float(threshold)
        self._success_command_name = str(command_name)
        self._success_info_key = str(info_key)

    def step(self, actions):
        # super().step() already injects per_env_rew + per_env_rew_mask.
        obs, reward, terminated, truncated, info = super().step(actions)
        is_success = object_reached_goal(
            self._unwrapped,
            command_name=self._success_command_name,
            threshold=self._success_threshold,
        )
        info[self._success_info_key] = is_success
        return obs, reward, terminated, truncated, info
