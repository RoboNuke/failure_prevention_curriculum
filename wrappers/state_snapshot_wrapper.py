"""Per-env Isaac Lab scene state snapshot + restore.

Maintains, for every env, a ring of per-step scene snapshots covering up to
``max_episode_length`` steps of the most recent trajectory. Exposes a
:meth:`restore_state` method that writes back into PhysX *without* calling
``env.reset()`` — the writes take effect on the next ``env.step()``, which is
exactly when the downstream policy will see them.

Snapshot schema (locked at ctor time, fails loud on any mismatch):

* For every :class:`Articulation` in ``env.unwrapped.scene``: ``joint_pos``
  then ``joint_vel`` (each ``(num_envs, num_joints)``), flat-concatenated per
  env.
* For every :class:`RigidObject` in ``env.unwrapped.scene``: ``root_state_w``
  (``(num_envs, 13)`` — pos[3] + quat[4] + lin_vel[3] + ang_vel[3]).

The schema is captured once at construction; subsequent step()s assume the
scene shape is stable (Isaac Lab guarantees this for manager-based envs after
scene clone). Any unexpected articulation/rigid-object missing or shape change
will surface as a hard exception in ``_capture`` or ``restore_state``.

Models its restore path on
https://github.com/RoboNuke/Continuous_Force_RL/blob/main/wrappers/mechanics/efficient_reset_wrapper.py
— uses the same ``write_joint_state_to_sim`` / ``write_root_pose_to_sim`` +
``write_root_velocity_to_sim`` API on per-env index slices.

This wrapper MUST NOT call ``env.reset()``. It only observes the env's natural
step cycle and writes back to PhysX state buffers on demand.
"""

from __future__ import annotations

from typing import Any

import torch

from skrl.envs.wrappers.torch.base import Wrapper


class StateSnapshotWrapper(Wrapper):
    """Snapshot + restore per-env Isaac Lab scene state.

    History layout:
        ``_history: (num_envs, max_episode_length, snapshot_dim)`` GPU float32.
        ``_head:    (num_envs,) long`` — next index to write per env.

    After each :meth:`step`, the post-step scene is captured into
    ``_history[arange, _head]``, then ``_head`` is rolled — done envs reset
    to 0 (so the new episode's s_0 lands at slot 0 next call); others
    advance by 1. On a step that did NOT trigger a done, the previous
    trajectory's slots above the current head are still readable — the
    backward rescue scan reads them BEFORE the next step overwrites them.
    """

    def __init__(self, env: Any, *, max_episode_length: int, device: torch.device | str) -> None:
        super().__init__(env)
        if max_episode_length is None or max_episode_length <= 0:
            raise ValueError(
                f"StateSnapshotWrapper.max_episode_length must be > 0, got {max_episode_length!r}"
            )
        if device is None:
            raise ValueError("StateSnapshotWrapper.device is required (no default).")
        self.max_episode_length = int(max_episode_length)
        self._device = torch.device(device)

        # Introspect the scene once. Subsequent steps assume schema stability.
        scene = getattr(self._unwrapped, "scene", None)
        if scene is None:
            raise RuntimeError(
                "StateSnapshotWrapper requires env.unwrapped.scene; the wrapped env "
                f"({self._unwrapped.__class__.__name__}) does not expose one."
            )

        # _slices: list of (kind, name, slice) describing how to pack/unpack
        # the flat snapshot vector. kind in {"art", "rigid"}.
        self._slices: list[tuple[str, str, slice]] = []
        cursor = 0
        # Articulations: joint_pos + joint_vel concatenated.
        arts = dict(getattr(scene, "articulations", {}))
        if not arts:
            raise RuntimeError(
                "StateSnapshotWrapper found no articulations in env.unwrapped.scene; "
                "rescue-state restore requires at least one (e.g. the robot)."
            )
        for name, art in sorted(arts.items()):
            data = art.data
            jp = data.joint_pos
            jv = data.joint_vel
            if jp.shape != jv.shape:
                raise RuntimeError(
                    f"StateSnapshotWrapper: articulation {name!r} has inconsistent "
                    f"joint_pos/joint_vel shapes {tuple(jp.shape)} vs {tuple(jv.shape)}"
                )
            nj = int(jp.shape[1])
            self._slices.append(("art", name, slice(cursor, cursor + 2 * nj)))
            cursor += 2 * nj
        # Rigid objects: root_state_w (13).
        rigids = dict(getattr(scene, "rigid_objects", {}))
        for name, rb in sorted(rigids.items()):
            rs = rb.data.root_state_w
            if rs.shape[1] != 13:
                raise RuntimeError(
                    f"StateSnapshotWrapper: rigid object {name!r} root_state_w has "
                    f"width {rs.shape[1]}, expected 13."
                )
            self._slices.append(("rigid", name, slice(cursor, cursor + 13)))
            cursor += 13
        self.snapshot_dim = int(cursor)

        # num_envs from the wrapped env (skrl's IsaacLabWrapper exposes it via __getattr__).
        # Note: skrl's Wrapper base defines `num_envs` as a read-only property that
        # delegates to the wrapped env, so we mirror it into a private field for hot-
        # path indexing instead of shadowing the property.
        num_envs = int(self._env.num_envs)
        self._num_envs = num_envs
        self._history = torch.zeros(
            (num_envs, self.max_episode_length, self.snapshot_dim),
            dtype=torch.float32,
            device=self._device,
        )
        self._head = torch.zeros((num_envs,), dtype=torch.long, device=self._device)
        # Cache the scene refs for the hot path.
        self._scene = scene
        self._arts = arts
        self._rigids = rigids
        self._env_index_arange = torch.arange(num_envs, device=self._device)

    # ------------------------------------------------------------------
    # Snapshot capture / restore primitives
    # ------------------------------------------------------------------
    def _capture(self) -> torch.Tensor:
        """Return ``(num_envs, snapshot_dim)`` snapshot of the current scene state."""
        out = torch.empty((self._num_envs, self.snapshot_dim), dtype=torch.float32, device=self._device)
        for kind, name, sl in self._slices:
            if kind == "art":
                art = self._arts[name]
                nj = (sl.stop - sl.start) // 2
                out[:, sl.start : sl.start + nj] = art.data.joint_pos.to(self._device, dtype=torch.float32)
                out[:, sl.start + nj : sl.stop] = art.data.joint_vel.to(self._device, dtype=torch.float32)
            elif kind == "rigid":
                rb = self._rigids[name]
                out[:, sl] = rb.data.root_state_w.to(self._device, dtype=torch.float32)
            else:
                raise RuntimeError(f"StateSnapshotWrapper: unknown slice kind {kind!r}")
        return out

    def restore_state(self, env_ids: torch.Tensor, snapshots: torch.Tensor) -> None:
        """Write packed ``snapshots`` back into PhysX for the listed ``env_ids``.

        ``env_ids``  : (k,) long tensor of env indices to overwrite.
        ``snapshots``: (k, snapshot_dim) tensor matching the locked schema.

        Effects take hold on the next :meth:`step` — this method NEVER calls
        ``env.reset()``.
        """
        if env_ids.ndim != 1:
            raise ValueError(f"restore_state: env_ids must be 1-D, got shape {tuple(env_ids.shape)}")
        if snapshots.ndim != 2 or snapshots.shape[1] != self.snapshot_dim:
            raise ValueError(
                f"restore_state: snapshots must be (k, {self.snapshot_dim}), got "
                f"{tuple(snapshots.shape)}"
            )
        if env_ids.shape[0] != snapshots.shape[0]:
            raise ValueError(
                f"restore_state: env_ids and snapshots disagree on k "
                f"({env_ids.shape[0]} vs {snapshots.shape[0]})"
            )
        if env_ids.numel() == 0:
            return
        env_ids_dev = env_ids.to(self._device, dtype=torch.long)
        snaps_dev = snapshots.to(self._device, dtype=torch.float32)
        for kind, name, sl in self._slices:
            if kind == "art":
                art = self._arts[name]
                nj = (sl.stop - sl.start) // 2
                jp = snaps_dev[:, sl.start : sl.start + nj]
                jv = snaps_dev[:, sl.start + nj : sl.stop]
                # Isaac Lab API: write_joint_state_to_sim(position, velocity, env_ids=...)
                art.write_joint_state_to_sim(jp, jv, env_ids=env_ids_dev)
                # Also align controller setpoints to the restored state so the
                # next physics tick doesn't pull the joints back toward stale
                # action targets from the perturbed trajectory. Some Isaac Lab
                # tasks (e.g. Lift) wire articulations through PD controllers
                # whose setpoints live in a separate buffer.
                if hasattr(art, "set_joint_position_target"):
                    art.set_joint_position_target(jp, env_ids=env_ids_dev)
                if hasattr(art, "set_joint_velocity_target"):
                    art.set_joint_velocity_target(jv, env_ids=env_ids_dev)
            elif kind == "rigid":
                rb = self._rigids[name]
                rs = snaps_dev[:, sl]  # (k, 13)
                pose = rs[:, :7]
                vel = rs[:, 7:]
                # Isaac Lab API: split pose / velocity writes.
                rb.write_root_pose_to_sim(pose, env_ids=env_ids_dev)
                rb.write_root_velocity_to_sim(vel, env_ids=env_ids_dev)
            else:
                raise RuntimeError(f"restore_state: unknown slice kind {kind!r}")

        # Overlay the just-restored states onto the history at slot 0 of the
        # corresponding envs. The rescue init wrapper only restores into envs
        # that just hit a natural done (so their _head has already rolled to 0
        # in the same step() — see step() below). Slot 0 is the new episode's
        # s_0 from the rescue buffer's perspective.
        self._history[env_ids_dev, 0] = snaps_dev

    # ------------------------------------------------------------------
    # Public read accessors
    # ------------------------------------------------------------------
    def history_for_env_step(self, env_i: int, step_i: int) -> torch.Tensor:
        """Return ``_history[env_i, step_i]`` — the packed snapshot at that step."""
        return self._history[int(env_i), int(step_i)]

    # ------------------------------------------------------------------
    # Wrapper API
    # ------------------------------------------------------------------
    def step(self, actions: torch.Tensor):
        obs, rew, term, trunc, info = self._env.step(actions)
        # Capture POST-step scene state into the current head slot for each env,
        # then roll the head per env: done -> 0, else +1 (clamped).
        snap = self._capture()
        # Bounds check: head must be in [0, max_episode_length). Episodes
        # exceeding max_episode_length would overflow; in Isaac Lab the env
        # auto-truncates at max_episode_length so this is a hard error if hit.
        if bool((self._head >= self.max_episode_length).any().item()):
            bad = (self._head >= self.max_episode_length).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"StateSnapshotWrapper history overflow on envs {bad}: head "
                f"exceeded max_episode_length={self.max_episode_length}."
            )
        self._history[self._env_index_arange, self._head] = snap
        # Compute done mask. term/trunc come back as (num_envs, 1) from IsaacLabWrapper.
        done = (term.view(-1).bool() | trunc.view(-1).bool())
        self._head = torch.where(
            done.to(self._device),
            torch.zeros_like(self._head),
            self._head + 1,
        )
        # Surface the snapshot wrapper's view on info for downstream consumers.
        info.setdefault("_state_snapshot_step_capture", True)
        return obs, rew, term, trunc, info

    def reset(self):
        out = self._env.reset()
        # Re-zero head on any explicit external reset (the trainer's initial reset).
        self._head.zero_()
        return out

    def state(self):
        return self._env.state() if hasattr(self._env, "state") else None

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        return self._env.close()
