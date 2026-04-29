"""Public surface of the config-manager package.

Re-exports the registry-driven loader and the registered dataclass types so callers
can keep writing ``from configs.manager import ConfigManager, SAC_CFG, ModelCfg``.
"""

from configs.manager.manager import ConfigManager
from configs.manager.model_cfg import ActorCfg, CriticCfg, ModelCfg
from configs.manager.preprocessor_registry import (
    available_preprocessors,
    resolve_preprocessor,
)
from configs.manager.sac_cfg import SAC_CFG

__all__ = [
    "ConfigManager",
    "SAC_CFG",
    "ModelCfg",
    "ActorCfg",
    "CriticCfg",
    "resolve_preprocessor",
    "available_preprocessors",
]
