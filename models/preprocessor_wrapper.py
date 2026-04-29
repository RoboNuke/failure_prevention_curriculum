"""Per-agent preprocessor wrapper (mirrors RoboNuke's pattern).

Adapts a list of ``num_agents`` independent preprocessors (e.g. one
``RunningStandardScaler`` per agent) to skrl's single-preprocessor call interface.
The wrapper slices the input tensor along its leading batch dim by the per-agent
env partition, applies each agent's preprocessor to its slice, and concatenates the
results back together.

Today our SAC uses no preprocessor (effectively a no-op pass-through); this wrapper
is in place so dropping a real preprocessor in later — and the per-agent checkpoint
format already provisions a slot — works without further plumbing.
"""

from __future__ import annotations

from typing import Any

import torch


class PerAgentPreprocessorWrapper:
    """Routes batch slices to per-agent preprocessors.

    The input tensor is assumed to be flat along dim 0, organized as
    ``[agent_0_envs..., agent_1_envs..., ..., agent_{N-1}_envs...]``. The leading
    dim must be divisible by ``num_agents``.
    """

    def __init__(self, num_agents: int, preprocessor_list: list) -> None:
        if len(preprocessor_list) != num_agents:
            raise ValueError(
                f"Expected {num_agents} preprocessors, got {len(preprocessor_list)}"
            )
        self.num_agents = num_agents
        self.preprocessor_list = preprocessor_list

    def __call__(
        self, tensor: torch.Tensor, train: bool = False, inverse: bool = False
    ) -> torch.Tensor:
        if tensor is None:
            return tensor
        total = tensor.shape[0]
        if total % self.num_agents != 0:
            raise ValueError(
                f"Batch dim ({total}) not divisible by num_agents ({self.num_agents})"
            )
        per_agent = total // self.num_agents

        out_chunks = []
        for i, preproc in enumerate(self.preprocessor_list):
            chunk = tensor[i * per_agent : (i + 1) * per_agent]
            if preproc is None:
                out_chunks.append(chunk)
            else:
                # Preprocessors must accept (tensor, train=..., inverse=...) — the same
                # signature skrl uses internally. Anything else is a configuration error.
                out_chunks.append(preproc(chunk, train=train, inverse=inverse))
        return torch.cat(out_chunks, dim=0)

    def state_dict(self) -> dict:
        """Return state dicts for all per-agent preprocessors keyed by ``agent_<i>``."""
        out = {}
        for i, preproc in enumerate(self.preprocessor_list):
            if preproc is not None and hasattr(preproc, "state_dict"):
                out[f"agent_{i}"] = preproc.state_dict()
        return out

    def load_state_dict(self, state_dict: dict) -> None:
        for i, preproc in enumerate(self.preprocessor_list):
            key = f"agent_{i}"
            if key in state_dict and preproc is not None and hasattr(preproc, "load_state_dict"):
                preproc.load_state_dict(state_dict[key])
