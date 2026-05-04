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
    "ant_success": "wrappers.ant_success.AntSuccessWrapper",
    "factory": "wrappers.factory.FactoryWrapper",
    "forge": "wrappers.forge.ForgeWrapper",
    "reward_decomposition": "wrappers.reward_decomposition.RewardDecompositionWrapper",
}

# Default wrapper applied when no task-specific or YAML-specified wrapper is set.
# `RewardDecompositionWrapper` is task-agnostic — it provides per-env per-term
# reward logging for any manager-based env, and gracefully no-ops for direct envs.
_DEFAULT_WRAPPER_NAME: str = "reward_decomposition"

# Default wrapper for a given Isaac Lab task id, matched by prefix. Lets the runner
# auto-apply success wrappers for known tasks without forcing every YAML to set
# ``sac_cfg.success_wrapper``. First matching prefix wins.
_TASK_DEFAULTS: dict[str, str] = {
    "Isaac-Lift-Cube-": "lift",
    "Isaac-Ant-": "ant_success",
    "Isaac-Factory-": "factory",
    "Isaac-Forge-": "forge",
}


def available_wrappers() -> list[str]:
    """Return the registered wrapper names (sorted)."""
    return sorted(_REGISTRY.keys())


def default_wrapper_for_task(task: str) -> str | None:
    """Return the registered wrapper name whose prefix matches ``task``, or ``None``.

    Used by the runner to auto-select a success wrapper based on the task id when
    the user hasn't set one explicitly in the YAML.
    """
    for prefix, name in _TASK_DEFAULTS.items():
        if task.startswith(prefix):
            return name
    return None


def fallback_wrapper_name() -> str:
    """Return the default wrapper applied when no task-specific or YAML-specified
    wrapper is set. Currently :class:`RewardDecompositionWrapper`."""
    return _DEFAULT_WRAPPER_NAME


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
