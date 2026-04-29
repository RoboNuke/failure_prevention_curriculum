"""Per-step success injection for ``Isaac-Lift-Cube-Franka-v0``.

Subclasses skrl's ``IsaacLabWrapper``. After every step, computes the canonical
Isaac Lab success condition (``object_reached_goal``: object position within
``threshold`` meters of the commanded goal pose) and writes the resulting per-env
boolean tensor into ``info[info_key]`` so SAC's success-prediction head can train
on it.

The success condition function and its threshold default match what Isaac Lab
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
from skrl.envs.wrappers.torch.isaaclab_envs import IsaacLabWrapper


class LiftSuccessWrapper(IsaacLabWrapper):
    """Adds ``info["is_success"]`` per step for the Lift task."""

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
        obs, reward, terminated, truncated, info = super().step(actions)
        is_success = object_reached_goal(
            self._unwrapped,
            command_name=self._success_command_name,
            threshold=self._success_threshold,
        )
        # ``info`` and ``self._info`` reference the same dict (skrl stores the
        # latest one on the wrapper); a single write is visible everywhere.
        info[self._success_info_key] = is_success
        return obs, reward, terminated, truncated, info
