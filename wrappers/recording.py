"""Periodic 3x4-grid GIF + TB-video recorder wrapper for vectorized Isaac Lab envs.

Composes around an already-wrapped skrl ``IsaacLabWrapper`` (e.g. ``ForgeWrapper``,
``RewardDecompositionWrapper``) by delegating ``step`` / ``reset`` / attribute
access. Counts naturally-occurring *global* resets — steps where every env
reports ``terminated | truncated`` — and opens a recording session every K-th
global reset. A session captures up to ``max_episode_length`` per-env frames
plus per-env accumulated reward and per-step ``min(Q1, Q2)(s_t, a_t)``, then
hands off to ``recording_grid.build_grid_video`` to compose the 3x4 grid.

Critics (``critic_1``, ``critic_2``) and the SAC state preprocessor are
**required** when ``recorder.enabled=True`` — passed in as ``__init__`` args.
Missing critics raise rather than silently displaying ``N/A``.

Rendering is suppressed between sessions by setting the recorder camera's
``update_period`` to a large value, then restoring it to 0.0 (every-step) for
the duration of a session. So when no session is active, the camera primitive
exists but is not re-rasterized.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import torch

from wrappers.recording_grid import build_grid_video, write_gif, write_tb_video


CAMERA_KEY = "recorder_camera"
"""Scene-sensor name under which the runner attaches the recorder ``TiledCamera``."""


def _coerce_done(t: torch.Tensor) -> torch.Tensor:
    """Squeeze trailing dim and cast to bool. ``terminated`` / ``truncated`` may
    arrive as ``(N, 1)`` or ``(N,)`` from skrl wrappers."""
    return t.view(-1).bool()


class RecordingWrapper:
    """Wraps an already-wrapped skrl IsaacLab env. Adds frame capture and grid GIF."""

    def __init__(
        self,
        env: Any,
        recorder_cfg: Any,
        critic_1: Any,
        critic_2: Any,
        state_preprocessor: Callable[[torch.Tensor], torch.Tensor] | None,
        max_episode_length: int,
        output_dir: str,
        image_writer: Any,
        agent_timestep_fn: Callable[[], int] | None = None,
    ) -> None:
        if not recorder_cfg.enabled:
            raise ValueError(
                "RecordingWrapper constructed with recorder.enabled=False — "
                "the runner should skip wrapping in that case."
            )
        if critic_1 is None or critic_2 is None:
            raise ValueError(
                "RecordingWrapper requires non-None critic_1 and critic_2 (used to "
                "compute min(Q1, Q2)(s_t, a_t) for the value overlay). Construct "
                "the wrapper after the SAC agent so the critics are available."
            )
        if max_episode_length <= 0:
            raise ValueError(
                f"RecordingWrapper requires max_episode_length > 0, got {max_episode_length!r}. "
                "Set ``runner_cfg.env_cfg_overrides`` so the env exposes a finite episode length."
            )

        self._env = env
        self._cfg = recorder_cfg
        self._critic_1 = critic_1
        self._critic_2 = critic_2
        self._state_preprocessor = state_preprocessor or (lambda s: s)
        self._max_ep_len = int(max_episode_length)
        self._output_dir = output_dir
        self._image_writer = image_writer
        self._agent_timestep_fn = agent_timestep_fn or (lambda: 0)

        os.makedirs(self._output_dir, exist_ok=True)

        # Resolve the camera sensor once. If the runner didn't attach one, the
        # recorder is broken — fail loudly.
        scene = self._env.unwrapped.scene
        if not hasattr(scene, "sensors") or CAMERA_KEY not in scene.sensors:
            raise RuntimeError(
                f"RecordingWrapper: expected a TiledCamera at scene.sensors[{CAMERA_KEY!r}] "
                "but none was found. The runner must inject the camera into env_cfg.scene "
                "before gym.make() when sac_cfg.recorder.enabled=True."
            )
        self._camera = scene.sensors[CAMERA_KEY]
        self._device = self._env.unwrapped.device
        self._num_envs = int(self._env.num_envs)
        H = int(recorder_cfg.height)
        W = int(recorder_cfg.width)

        # Pre-allocate per-env CPU frame buffer once. uint8 RGB.
        self._frames = torch.zeros(
            (self._num_envs, self._max_ep_len, H, W, 3), dtype=torch.uint8
        )
        self._returns = torch.zeros(self._num_envs, dtype=torch.float32)
        self._term_step = torch.full((self._num_envs,), self._max_ep_len, dtype=torch.int64)
        self._success = torch.zeros(self._num_envs, dtype=torch.bool)
        self._values = torch.zeros((self._num_envs, self._max_ep_len), dtype=torch.float32)
        self._env_done = torch.zeros(self._num_envs, dtype=torch.bool)

        # Session state.
        self._state = "idle"  # one of: "idle", "starting", "active"
        self._t_in_session = 0
        self._global_reset_count = 0
        self._total_steps = 0  # used as TB global_step when no agent timestep fn is provided

        # Pre-step caches (set in step() *before* delegating, used to compute
        # Q(s_t, a_t) where s_t is the policy's input observation).
        self._cached_pre_obs: torch.Tensor | None = None
        self._cached_pre_state: torch.Tensor | None = None

        # Disable camera rendering until first session opens.
        self._set_camera_active(False)

    # ------------------------------------------------------------------
    # Delegation glue (skrl trainer talks to this object directly)
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        # Called only when ``name`` not found on self — forward to inner env.
        return getattr(self._env, name)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> Any:
        return self._device

    @property
    def observation_space(self):  # type: ignore[no-untyped-def]
        return self._env.observation_space

    @property
    def action_space(self):  # type: ignore[no-untyped-def]
        return self._env.action_space

    @property
    def state_space(self):  # type: ignore[no-untyped-def]
        return getattr(self._env, "state_space", None)

    def state(self):  # type: ignore[no-untyped-def]
        return self._env.state()

    def reset(self):  # type: ignore[no-untyped-def]
        obs, info = self._env.reset()
        self._cached_pre_obs = obs
        # state() may not be valid before first step in some envs; guard.
        try:
            st = self._env.state()
            self._cached_pre_state = st if st is not None else obs
        except Exception:
            self._cached_pre_state = obs
        return obs, info

    def close(self) -> None:
        self._env.close()

    def render(self, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        return self._env.render(*args, **kwargs)

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------
    def step(self, actions: torch.Tensor):  # type: ignore[no-untyped-def]
        # Snapshot pre-step inputs for the critic. The action was sampled from
        # the policy using ``self._cached_pre_obs`` from the previous step (or
        # reset). For asymmetric AC, the critic uses the matching state.
        pre_obs = self._cached_pre_obs
        pre_state = self._cached_pre_state if self._cached_pre_state is not None else pre_obs

        # Open session if scheduled.
        if self._state == "starting":
            self._open_session()

        obs, reward, terminated, truncated, info = self._env.step(actions)

        # Cache for the *next* step.
        self._cached_pre_obs = obs
        try:
            st = self._env.state()
            self._cached_pre_state = st if st is not None else obs
        except Exception:
            self._cached_pre_state = obs

        self._total_steps += 1
        if self._state == "active":
            self._capture_step(actions, pre_state, reward, terminated, truncated, info)

        # Detect global reset (every env done this step). This may also close
        # the active session at exactly its natural boundary.
        done = _coerce_done(terminated) | _coerce_done(truncated)
        if bool(done.all()):
            self._global_reset_count += 1
            if self._state == "active":
                self._close_and_emit_session()
            # Schedule the next session if cadence hits. (May open immediately
            # next step.)
            if self._global_reset_count % max(1, int(self._cfg.record_every_k_resets)) == 0:
                self._state = "starting"

        # Safety: close session if we hit the per-env-frame buffer limit before
        # a global reset arrives (e.g. unusually long max_ep_len mismatch).
        if self._state == "active" and self._t_in_session >= self._max_ep_len:
            self._close_and_emit_session()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------
    def _open_session(self) -> None:
        self._frames.zero_()
        self._returns.zero_()
        self._term_step.fill_(self._max_ep_len)
        self._success.zero_()
        self._values.zero_()
        self._env_done.zero_()
        self._t_in_session = 0
        self._state = "active"
        self._set_camera_active(True)

    def _capture_step(
        self,
        actions: torch.Tensor,
        pre_state: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        info: dict,
    ) -> None:
        t = self._t_in_session
        if t >= self._max_ep_len:
            return  # safety; outer step() will close session

        # Frames: only update tiles whose env hasn't already terminated this session.
        rgb = self._read_camera_rgb()  # (num_envs, H, W, 3) uint8 on CPU
        alive_mask = ~self._env_done  # (num_envs,) cpu bool
        alive_idx = alive_mask.nonzero(as_tuple=False).view(-1)
        if alive_idx.numel() > 0:
            self._frames[alive_idx, t] = rgb[alive_idx]

        # Reward accumulation (only alive envs).
        rew_cpu = reward.detach().view(-1).float().cpu()
        if alive_idx.numel() > 0:
            self._returns[alive_idx] += rew_cpu[alive_idx]

        # Critic value: min(Q1, Q2)(s_t, a_t). Only alive envs.
        if alive_idx.numel() > 0:
            v = self._compute_min_q(pre_state, actions)  # (num_envs,) cpu float
            self._values[alive_idx, t] = v[alive_idx]

        # Mark new terminations for envs alive this step.
        term_now = _coerce_done(terminated).cpu()
        trunc_now = _coerce_done(truncated).cpu()
        new_done = (term_now | trunc_now) & alive_mask  # (num_envs,) cpu bool
        if bool(new_done.any()):
            new_done_idx = new_done.nonzero(as_tuple=False).view(-1)
            self._term_step[new_done_idx] = t
            # Per-env success flag from info; default False if missing.
            succ = info.get("is_success", None)
            if isinstance(succ, torch.Tensor):
                succ_cpu = succ.view(-1).bool().cpu()
                self._success[new_done_idx] = succ_cpu[new_done_idx]
            self._env_done[new_done_idx] = True

        self._t_in_session += 1

    def _close_and_emit_session(self) -> None:
        # Compose grid + write outputs.
        global_step = int(self._agent_timestep_fn() or self._total_steps)
        idx = self._global_reset_count
        try:
            grid = build_grid_video(
                frames=self._frames[:, : self._t_in_session],
                returns=self._returns,
                term_step=self._term_step.clamp(max=self._t_in_session),
                is_success=self._success,
                values=self._values[:, : self._t_in_session],
            )
            gif_path = os.path.join(
                self._output_dir, f"recording_{idx:06d}.gif"
            )
            write_gif(grid, gif_path, fps=int(self._cfg.fps))
            print(f"[recorder] wrote {gif_path} ({grid.shape[0]} frames)", flush=True)
            if self._image_writer is not None:
                write_tb_video(
                    self._image_writer,
                    tag="Video / grid_3x4",
                    grid=grid,
                    fps=int(self._cfg.fps),
                    global_step=global_step,
                )
        except Exception as e:
            # Never let recorder failure tank training.
            print(f"[recorder] WARNING: grid emission failed: {e!r}", flush=True)
        finally:
            self._state = "idle"
            self._t_in_session = 0
            self._set_camera_active(False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _read_camera_rgb(self) -> torch.Tensor:
        """Pull the latest TiledCamera RGB tensor and move to CPU as uint8 ``(N, H, W, 3)``."""
        rgb = self._camera.data.output["rgb"]  # device tensor; layout per IsaacLab is (N, H, W, C)
        if rgb.dim() != 4:
            raise RuntimeError(
                f"recorder camera returned unexpected shape {tuple(rgb.shape)}; expected (N, H, W, C)"
            )
        # Drop alpha if present.
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if rgb.dtype != torch.uint8:
            # Float frames are typically [0, 1]; scale.
            rgb = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        return rgb.detach().cpu()

    def _compute_min_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute ``min(Q1, Q2)(state, action)`` for every env and return a CPU
        ``(num_envs,)`` float tensor."""
        with torch.no_grad():
            processed = self._state_preprocessor(state)
            inputs = {"observations": processed, "taken_actions": action}
            q1, _, _ = _unpack_act(self._critic_1.act(inputs, role="critic_1"))
            q2, _, _ = _unpack_act(self._critic_2.act(inputs, role="critic_2"))
            v = torch.minimum(q1, q2).view(-1).float().cpu()
        return v

    def _set_camera_active(self, active: bool) -> None:
        """Toggle camera rasterization. We mutate ``update_period`` in place so
        IsaacLab skips the sensor entirely between sessions."""
        target = 0.0 if active else 1.0e9
        # Try a few common locations — IsaacLab's API surface has shifted.
        # Prefer the live sensor attribute; fall back to its cfg.
        for owner in (self._camera, getattr(self._camera, "cfg", None)):
            if owner is None:
                continue
            if hasattr(owner, "update_period"):
                try:
                    owner.update_period = target
                except Exception:
                    pass


def _unpack_act(ret: Any) -> tuple[torch.Tensor, Any, Any]:
    """skrl model ``act`` returns ``(actions, log_prob, outputs)`` — tolerate
    2-tuples too. Returns ``(value, _, _)`` shape-normalized."""
    if isinstance(ret, tuple):
        if len(ret) >= 3:
            return ret[0], ret[1], ret[2]
        if len(ret) == 2:
            return ret[0], ret[1], None
    return ret, None, None
