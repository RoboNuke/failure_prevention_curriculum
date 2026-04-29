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
