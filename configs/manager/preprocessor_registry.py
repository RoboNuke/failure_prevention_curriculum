"""Name → class registry for observation preprocessors.

YAML configs can only carry strings, but the SAC ``observation_preprocessor`` field
expects a class to be instantiated at runtime. This registry bridges the two: a
short string in YAML (``RunningStandardScaler``) resolves here to the actual
class. Mirrors what ``skrl.utils.runner.torch.runner`` does for its own configs.
"""

from __future__ import annotations

from typing import Any


def _running_standard_scaler() -> type:
    """Lazy import — keeps this module cheap to import even before skrl is loaded."""
    from skrl.resources.preprocessors.torch import RunningStandardScaler
    return RunningStandardScaler


_REGISTRY: dict[str, "callable[[], type]"] = {
    "RunningStandardScaler": _running_standard_scaler,
}


def available_preprocessors() -> list[str]:
    """Return the registered preprocessor names (sorted)."""
    return sorted(_REGISTRY.keys())


def resolve_preprocessor(value: Any) -> Any:
    """Translate a YAML-loaded preprocessor field into a class (or None).

    * ``None`` → ``None`` (no preprocessor; SAC falls back to identity).
    * ``str`` → looked up in the registry; raises ``KeyError`` if unknown.
    * already a class (``type``) → returned unchanged.

    Anything else raises ``TypeError``.
    """
    if value is None:
        return None
    if isinstance(value, type):
        return value
    if isinstance(value, str):
        if value not in _REGISTRY:
            raise KeyError(
                f"Unknown observation_preprocessor {value!r}; expected one of "
                f"{available_preprocessors()} or null"
            )
        return _REGISTRY[value]()
    raise TypeError(
        f"observation_preprocessor must be None, a class, or a registered string; "
        f"got {type(value).__name__}"
    )
