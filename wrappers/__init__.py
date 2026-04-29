"""Env wrapper registry.

Maps short config-friendly names (e.g. ``"lift"``) to wrapper classes that take an
unwrapped Isaac Lab env and return a stepable env compatible with the rest of the
training loop. Imports of Isaac-Lab-touching modules are lazy so importing this
package does not require Isaac Sim to be initialized.
"""

from __future__ import annotations

from typing import Any, Callable

# name -> "module.ClassName". Resolved lazily by ``make_wrapper``.
_REGISTRY: dict[str, str] = {
    "lift": "wrappers.lift_success.LiftSuccessWrapper",
}


def available_wrappers() -> list[str]:
    """Return the registered wrapper names (sorted)."""
    return sorted(_REGISTRY.keys())


def make_wrapper(name: str, env: Any, **kwargs: Any) -> Any:
    """Instantiate the wrapper registered under ``name`` around ``env``.

    :param name: Registered wrapper key (see :func:`available_wrappers`).
    :param env: Unwrapped Isaac Lab env (the result of ``gym.make``); the wrapper
        is responsible for any further wrapping (e.g. ``IsaacLabWrapper`` behavior).
    :param kwargs: Extra keyword arguments forwarded to the wrapper's constructor.

    :raises KeyError: ``name`` is not in the registry.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown env wrapper {name!r}; expected one of {available_wrappers()}"
        )
    qualname = _REGISTRY[name]
    mod_name, cls_name = qualname.rsplit(".", 1)
    mod = __import__(mod_name, fromlist=[cls_name])
    cls: Callable[..., Any] = getattr(mod, cls_name)
    return cls(env, **kwargs)
