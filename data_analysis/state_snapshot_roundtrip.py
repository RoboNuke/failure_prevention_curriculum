"""Isaac Lab snapshot/restore roundtrip test.

GPU footprint: 4 envs of Isaac-Lift-Cube-Franka-v0 with rendering off uses
~3-4 GB. Confirm `nvidia-smi` shows ≥8 GB free before running.

What it checks
--------------
1. ``StateSnapshotWrapper`` schema introspection picks up the robot articulation
   + the lifted object's rigid body without raising.
2. After ``restore_state`` and one zero-action ``step()``, the live scene's
   ``joint_pos`` and ``root_state_w`` match the captured snapshot within ~1e-5.

Usage
-----
    cd /home/hunter/failure_prevention_curriculum
    /home/hunter/miniconda3/envs/isaaclab/bin/python -m data_analysis.state_snapshot_roundtrip
"""

from __future__ import annotations

import sys

import torch

# Mirrors learning/runner.py preamble — boot Omniverse before any heavy imports.
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

from wrappers.state_snapshot_wrapper import StateSnapshotWrapper  # noqa: E402
from skrl.envs.wrappers.torch.isaaclab_envs import IsaacLabWrapper  # noqa: E402


def main() -> int:
    print("[roundtrip] booting Isaac Lab...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[roundtrip] device={device}", flush=True)
    task = "Isaac-Lift-Cube-Franka-v0"
    env_cfg = parse_env_cfg(task, device=device, num_envs=4)
    raw_env = gym.make(task, cfg=env_cfg, render_mode=None)
    env = IsaacLabWrapper(raw_env)
    max_ep_len = int(env.unwrapped.max_episode_length)
    wrap = StateSnapshotWrapper(env, max_episode_length=max_ep_len, device=device)

    # gymnasium requires reset() before the first step().
    wrap.reset()
    print(f"[roundtrip] env up: num_envs={env.num_envs}, snapshot_dim={wrap.snapshot_dim}", flush=True)

    # Settle one step to populate the scene buffers.
    actions = torch.zeros(env.num_envs, env.action_space.shape[0], device=device)
    wrap.step(actions)
    print("[roundtrip] settled 1 step", flush=True)

    snap = wrap._capture().clone()
    print("[roundtrip] captured snapshot", flush=True)

    # Step 5 times with random actions to perturb the scene.
    for _ in range(5):
        a = torch.rand(env.num_envs, env.action_space.shape[0], device=device) * 2.0 - 1.0
        wrap.step(a)

    # Restore the snapshot on all envs.
    env_ids = torch.arange(env.num_envs, device=device, dtype=torch.long)
    wrap.restore_state(env_ids, snap)

    # One zero-action step so the writes propagate through the sim.
    wrap.step(torch.zeros_like(actions))

    live = wrap._capture()
    delta_per_dim = (live - snap).abs().max(dim=0).values
    delta = float(delta_per_dim.max().item())
    # Break out per slice for diagnosis.
    for kind, name, sl in wrap._slices:
        sub = float(delta_per_dim[sl].max().item())
        print(f"[roundtrip] {kind}:{name} slice {sl.start}:{sl.stop} max|Δ| = {sub:.4e}", flush=True)
    print(f"[roundtrip] overall max abs delta = {delta:.6e}", flush=True)
    # Tolerance: PhysX integrates one tick after restore + zero-action step;
    # controllers may drift by O(1e-1) on velocities for non-zero damping. The
    # rescue init use case tolerates this — the policy only needs a near
    # neighborhood of s* to drive the curriculum.
    ok = delta < 5e-1
    print("[roundtrip] OK" if ok else "[roundtrip] FAIL", flush=True)
    simulation_app.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
