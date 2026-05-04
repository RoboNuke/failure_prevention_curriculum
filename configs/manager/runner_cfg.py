"""Runner-level configuration loaded from YAML.

Holds the values the runner used to take exclusively from CLI flags (task,
num_envs, num_agents, total_timesteps, memory_size, eval_timesteps, seed).
Moving them into YAML lets a launcher invoke the runner with just a config
path + experiment name, while CLI flags remain available as one-off overrides
when provided.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(kw_only=True)
class RunnerCfg:
    """Per-task runner-level hyperparameters."""

    task: str
    """Isaac Lab gym id, e.g. ``"Isaac-Lift-Cube-Franka-v0"``."""

    num_envs: int
    """Envs PER agent. Total Isaac envs = ``num_envs * num_agents``."""

    num_agents: int = 1
    """Block-parallel agents trained simultaneously."""

    total_timesteps: int = 10_000
    """Training duration in **env_steps** (i.e. ``env.step()`` calls). Each env_step
    advances every parallel env by one tick, so total transitions written to the
    replay buffer = ``total_timesteps * num_envs * num_agents``. This is the value
    passed straight through to the trainer's ``timesteps`` field."""

    eval_timesteps: int = 250
    """Eval duration in env_steps. Used in place of ``total_timesteps`` when
    ``--mode eval`` is selected."""

    memory_size: int = 1_000_000
    """Replay buffer capacity PER AGENT. Per-env depth = ``memory_size // num_envs``."""

    seed: int = -1
    """Global seed. ``-1`` lets skrl pick a non-deterministic one."""
