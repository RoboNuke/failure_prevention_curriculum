"""YAML-driven config manager.

Loads a single YAML file that contains one section per registered config dataclass
(top-level header == header name in :attr:`ConfigManager.REGISTRY`). Each section's
keys must match dataclass fields exactly; unknown keys raise. Nested dataclass-typed
fields (e.g. ``SAC_CFG.experiment``, ``ModelCfg.actor``) recurse into their own dict.

Strict by design — any divergence between the YAML and the registered dataclasses
is a hard error, never a silent fallback.
"""

from __future__ import annotations

import dataclasses
import typing
from pathlib import Path
from typing import Any

import yaml

from configs.manager.model_cfg import ModelCfg
from configs.manager.sac_cfg import SAC_CFG


def _resolved_field_types(dataclass_type: type) -> dict[str, Any]:
    """Resolve a dataclass's annotations against the defining module's globals.

    ``from __future__ import annotations`` makes ``field.type`` a string;
    ``typing.get_type_hints`` evaluates those strings in the module they were
    defined in, which has the right imports in scope (e.g. ``ExperimentCfg``
    inside ``skrl.agents.torch.base``).
    """
    return typing.get_type_hints(dataclass_type)


def _to_yaml_safe(obj: Any) -> Any:
    """Recursively convert a Python value to YAML-safe primitives.

    Most config payloads are pure data (numbers, strings, bools, lists, dicts) and
    pass through unchanged. The cases worth calling out:
      * **Dataclasses** are deep-converted via :func:`dataclasses.asdict`, so a
        round-trip through ``dump()`` -> ``load()`` reproduces the same instances
        when fields hold only data values.
      * **Class objects** (e.g. ``learning_rate_scheduler`` set to a scheduler
        class) are written as their fully-qualified import path. This is portable
        and self-documenting but won't auto-instantiate on reload — strict-load
        will treat them as strings unless the dataclass is widened to accept str.
      * **Callables** (e.g. ``rewards_shaper``) are written similarly.
      * Anything else falls back to ``repr``. The dump file is then a record
        rather than a strict-loadable artifact, but always succeeds.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _to_yaml_safe(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_yaml_safe(x) for x in obj]
    if isinstance(obj, type):
        return f"{obj.__module__}.{obj.__qualname__}"
    if callable(obj):
        mod = getattr(obj, "__module__", "<unknown>")
        name = getattr(obj, "__qualname__", repr(obj))
        return f"{mod}.{name}"
    return repr(obj)


class ConfigManager:
    """Header → dataclass dispatch table.

    Add new project configs by registering them here. The runner loads all
    registered headers in a single ``ConfigManager.load(path)`` call.
    """

    REGISTRY: dict[str, type] = {
        "sac_cfg": SAC_CFG,
        "model_cfg": ModelCfg,
    }

    @classmethod
    def load(cls, yaml_path: str | Path | None) -> dict:
        """Return ``{header: populated_instance}`` for every registered header.

        * ``yaml_path is None`` returns one default-constructed instance per header
          (each dataclass with its own field defaults). The runner always passes a
          path; this branch exists for tests / programmatic use.
        * Otherwise, every registered header MUST appear in the YAML. Unknown
          headers, unknown fields within a section, and missing files all raise.
        """
        if yaml_path is None:
            return {header: type_() for header, type_ in cls.REGISTRY.items()}

        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(
                f"Config root in {path} must be a mapping, got {type(raw).__name__}"
            )

        unknown_headers = set(raw) - set(cls.REGISTRY)
        if unknown_headers:
            raise KeyError(
                f"Unknown config header(s) {sorted(unknown_headers)} in {path}; "
                f"expected one of {sorted(cls.REGISTRY)}"
            )
        missing_headers = set(cls.REGISTRY) - set(raw)
        if missing_headers:
            raise KeyError(
                f"Missing config section(s) {sorted(missing_headers)} in {path}"
            )

        out = {}
        for header, type_ in cls.REGISTRY.items():
            out[header] = cls._build(type_, raw[header], context=header)
        return out

    @classmethod
    def dump(cls, configs: dict, path: str | Path) -> None:
        """Write a loaded-configs dict to a YAML file at ``path``.

        Used to record the *actual* values fed to training (after ``load()`` and any
        runtime CLI overrides). The output's top-level shape mirrors the input
        accepted by ``load()``: one header per registered config dataclass.

        Strict: every registered header must be present in ``configs``; extra
        headers raise. Parent directories are created on demand.

        :param configs: Dict of ``{header: dataclass_instance}`` (the same shape
            ``ConfigManager.load()`` returns).
        :param path: Destination YAML path.

        :raises TypeError: ``configs`` is not a dict.
        :raises KeyError: ``configs`` is missing a registered header or has extras.
        """
        if not isinstance(configs, dict):
            raise TypeError(
                f"configs must be a dict, got {type(configs).__name__}"
            )
        missing = set(cls.REGISTRY) - set(configs)
        if missing:
            raise KeyError(
                f"Configs missing required headers: {sorted(missing)}; "
                f"expected all of {sorted(cls.REGISTRY)}"
            )
        extra = set(configs) - set(cls.REGISTRY)
        if extra:
            raise KeyError(
                f"Configs has unexpected headers: {sorted(extra)}; "
                f"expected only {sorted(cls.REGISTRY)}"
            )

        # Preserve REGISTRY order in the output for deterministic, readable diffs.
        serializable = {header: _to_yaml_safe(configs[header]) for header in cls.REGISTRY}
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            yaml.safe_dump(serializable, f, sort_keys=False, default_flow_style=False)

    @classmethod
    def _build(cls, dataclass_type: type, data: Any, *, context: str):
        """Recursively populate ``dataclass_type`` from a YAML-derived dict.

        Strict: every key in ``data`` must be a declared field; any extras raise.
        Nested dataclass-typed fields recurse into their own dicts.
        """
        if not dataclasses.is_dataclass(dataclass_type):
            raise TypeError(
                f"{context}: registered type {dataclass_type!r} is not a dataclass"
            )
        if not isinstance(data, dict):
            raise ValueError(
                f"{context}: expected mapping, got {type(data).__name__}"
            )

        valid_names = {f.name for f in dataclasses.fields(dataclass_type)}
        unknown = set(data) - valid_names
        if unknown:
            raise KeyError(
                f"Unknown field(s) {sorted(unknown)} under '{context}'; "
                f"expected one of {sorted(valid_names)}"
            )

        type_hints = _resolved_field_types(dataclass_type)
        kwargs = {}
        for name, value in data.items():
            field_type = type_hints.get(name)
            if dataclasses.is_dataclass(field_type) and isinstance(value, dict):
                kwargs[name] = cls._build(field_type, value, context=f"{context}.{name}")
            else:
                kwargs[name] = value
        return dataclass_type(**kwargs)
