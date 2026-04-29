"""Block-parallel-aware replay memory.

Ported from RoboNuke/Continuous_Force_RL/memories/multi_random.py with an added
`sample()` method for SAC-style off-policy replay. The key invariant: returned
batches are ordered such that block-parallel layers reshaping
`(num_agents * batch_per_agent, dim) -> (num_agents, batch_per_agent, dim)`
route each agent's data to the correct block.
"""

from typing import List, Optional, Tuple, Union

import torch

from skrl.memories.torch import Memory


class MultiRandomMemory(Memory):
    def __init__(
        self,
        *,
        memory_size: int,
        num_envs: int = 1,
        num_agents: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        export: bool = False,
        export_format: str = "pt",
        export_directory: str = "",
        replacement: bool = True,
    ) -> None:
        super().__init__(
            memory_size=memory_size,
            num_envs=num_envs,
            device=device,
            export=export,
            export_format=export_format,
            export_directory=export_directory,
        )
        if num_envs % num_agents != 0:
            raise ValueError(
                f"num_envs ({num_envs}) must be divisible by num_agents ({num_agents})"
            )
        self.num_agents = num_agents
        self.envs_per_agent = num_envs // num_agents
        self.replacement = replacement

    # -----------------------------
    # SAC-style replay sampling
    # -----------------------------
    def sample(
        self,
        names: Tuple[str],
        batch_size: int,
        mini_batches: int = 1,
        sequence_length: int = 1,
    ) -> List[List[torch.Tensor]]:
        """Sample a single mini-batch of size `batch_size`, partitioned by agent.

        Per-agent env partitioning: rows ``[i*per_agent, (i+1)*per_agent)`` of
        the returned batch come exclusively from envs ``[i*envs_per_agent,
        (i+1)*envs_per_agent)``. This matches how block-parallel layers reshape
        the flat batch back to ``(num_agents, per_agent, dim)``.
        """
        if batch_size % self.num_agents != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by num_agents ({self.num_agents})"
            )
        per_agent = batch_size // self.num_agents

        # Cap timestep range to filled portion of the memory to avoid sampling NaN slots.
        timestep_high = self.memory_size if self.filled else max(self.memory_index, 1)

        flat_indices_per_agent = []
        for a in range(self.num_agents):
            env_lo = a * self.envs_per_agent
            t = torch.randint(0, timestep_high, (per_agent,), device=self.device)
            e = torch.randint(env_lo, env_lo + self.envs_per_agent, (per_agent,), device=self.device)
            flat_indices_per_agent.append(t * self.num_envs + e)

        indices = torch.cat(flat_indices_per_agent, dim=0)
        return [self.sample_by_index(names, indexes=indices)[0]]

    # -----------------------------
    # On-policy / full-rollout sampling (kept for future PPO-style use)
    # -----------------------------
    def sample_all(
        self,
        names: Tuple[str],
        mini_batches: int = 1,
        sequence_length: int = 1,
        shuffle: bool = True,
    ) -> List[List[torch.Tensor]]:
        """Return all data as ``mini_batches`` partitions, each preserving per-agent ordering.

        Within each mini-batch, samples are concatenated as ``[agent0_chunk, agent1_chunk, ...]``
        so a block-parallel reshape recovers per-agent slices.
        """
        agent_batch_size = self.memory_size * self.envs_per_agent // mini_batches

        if shuffle:
            timestep_indices = torch.stack(
                [torch.randperm(self.memory_size, device=self.device) for _ in range(self.num_envs)]
            )
        else:
            timestep_indices = (
                torch.arange(self.memory_size, device=self.device)
                .unsqueeze(0)
                .expand(self.num_envs, -1)
            )

        agent_data_idxs = []
        for a_idx in range(self.num_agents):
            agent_env_start = a_idx * self.envs_per_agent
            agent_shuffles = timestep_indices[agent_env_start : agent_env_start + self.envs_per_agent, :]

            env_indices_list = []
            for env_offset in range(self.envs_per_agent):
                global_env_idx = agent_env_start + env_offset
                env_timesteps = agent_shuffles[env_offset, :]
                env_memory_indices = env_timesteps * self.num_envs + global_env_idx
                env_indices_list.append(env_memory_indices)

            stacked = torch.stack(env_indices_list, dim=0)
            indices = stacked.t().flatten()
            agent_data_idxs.append(indices)

        idxs = [[] for _ in range(mini_batches)]
        for b_idx in range(mini_batches):
            for a_idx in range(self.num_agents):
                a = b_idx * agent_batch_size
                b = a + agent_batch_size
                idxs[b_idx].append(agent_data_idxs[a_idx][a:b])
            idxs[b_idx] = torch.cat(idxs[b_idx], dim=0)

        return [self.sample_by_index(names, indexes=idxs[j])[0] for j in range(mini_batches)]
