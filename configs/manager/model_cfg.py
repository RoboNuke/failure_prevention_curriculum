"""Model hyperparameters loaded from YAML and forwarded to ``BlockSimBaActor`` /
``BlockSimBaQCritic`` constructors.

Defaults mirror the constructor defaults in ``models/block_simba.py`` so that
``ModelCfg()`` (with no overrides) reproduces today's hardcoded behavior.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(kw_only=True)
class ActorCfg:
    """Kwargs forwarded to ``BlockSimBaActor`` (in addition to obs/act/device/num_agents)."""

    act_init_std: float = 0.60653066
    actor_n: int = 2
    actor_latent: int = 512
    last_layer_scale: float = 1.0
    clip_log_std: bool = True
    min_log_std: float = -20.0
    max_log_std: float = 2.0
    reduction: str = "sum"
    use_state_dependent_std: bool = False
    bernoulli_action_dims: list[int] | None = None
    """Action dims that should be sampled from a Bernoulli (binary) distribution
    instead of a squashed Gaussian. Output is mapped {0,1} -> {-1,+1} for env
    compatibility (Isaac Lab's BinaryJointAction reads <0 as close, >=0 as open).
    Uses a straight-through estimator so the critic Q-gradient flows back into
    the policy via the soft probability. ``None`` or ``[]`` keeps every dim
    continuous. For Lift Franka set to ``[7]`` (gripper)."""

    force_zero_action_dims: list[int] | None = None
    """Action dims that should be hard-coded to 0 in the policy output (no
    learnable parameters allocated, no log_prob contribution). Useful for envs
    that internally ignore certain action dims — e.g., Forge sets actions[:, 3:5]
    to zero inside ``_apply_action``, so we can match by setting
    ``force_zero_action_dims: [3, 4]`` to avoid wasting actor capacity on
    predictions the env will discard. ``None`` or ``[]`` disables. Must not
    overlap with ``bernoulli_action_dims``."""


@dataclasses.dataclass(kw_only=True)
class CriticCfg:
    """Kwargs forwarded to ``BlockSimBaQCritic`` (used for both Q-critics + targets)."""

    critic_output_init_mean: float = 0.0
    critic_n: int = 2
    critic_latent: int = 512
    clip_actions: bool = False


@dataclasses.dataclass(kw_only=True)
class ModelCfg:
    actor: ActorCfg = dataclasses.field(default_factory=ActorCfg)
    critic: CriticCfg = dataclasses.field(default_factory=CriticCfg)
