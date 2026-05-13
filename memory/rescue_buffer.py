"""Per-agent rescue-state buffer ``B_c``.

Implements the storage half of Algorithm 1 (Rescue Recovery by Detection of
Trajectory Divergence). Each agent gets one ``RescueBuffer`` instance.

Storage layout (preallocated, GPU-resident):

* ``sim_state (C, snapshot_dim) float32`` — packed scene state for env restore.
* ``obs       (C, obs_dim)      float32`` — raw observation at the rescue point.
  Kept raw (un-normalized) so the running standard scaler can keep updating
  without invalidating buffer entries; consumers normalize at metric time.
* ``add_step              (C,)  long``    — env-step at which the slot was filled.
* ``add_p_value           (C,)  float32`` — predictor P at insertion time.
* ``source_trajectory_step(C,)  long``    — step index within the source trajectory.
* ``init_attempts         (C,)  long``    — number of times this slot has been used.
* ``init_successes        (C,)  long``    — number of those attempts that succeeded.
* ``_filled               (C,)  bool``    — occupancy mask.
* ``_insertion_order      (C,)  long``    — monotonic counter for FIFO tie-break.

Ragged per-init outcome lists (``init_episode_lengths``,
``init_times_to_success``, ``init_action_entropies``) live on the CPU as
``list[list[...]]``. Their consumers (Section 2 metrics) aggregate via NumPy
anyway; keeping them ragged saves GPU memory and avoids a fixed per-slot cap.

Eviction: when full, evict the dead slot with the smallest ``_insertion_order``;
if no slot is dead (``init_attempts >= dead_point_min_attempts`` AND
``init_successes == 0``), evict the overall smallest ``_insertion_order``
(FIFO).

Fail-loud: required collaborators are ctor arguments. Missing/None raises.
"""

from __future__ import annotations

from typing import Tuple

import torch


class RescueBuffer:
    """Per-agent fixed-capacity rescue-state buffer."""

    def __init__(
        self,
        *,
        capacity: int,
        snapshot_dim: int,
        obs_dim: int,
        device: torch.device | str,
        dead_point_min_attempts: int,
    ) -> None:
        if capacity is None or capacity <= 0:
            raise ValueError(f"RescueBuffer.capacity must be > 0, got {capacity!r}")
        if snapshot_dim is None or snapshot_dim <= 0:
            raise ValueError(f"RescueBuffer.snapshot_dim must be > 0, got {snapshot_dim!r}")
        if obs_dim is None or obs_dim <= 0:
            raise ValueError(f"RescueBuffer.obs_dim must be > 0, got {obs_dim!r}")
        if device is None:
            raise ValueError("RescueBuffer.device is required (no default).")
        if dead_point_min_attempts is None or dead_point_min_attempts < 1:
            raise ValueError(
                f"RescueBuffer.dead_point_min_attempts must be >= 1, got {dead_point_min_attempts!r}"
            )

        self.capacity = int(capacity)
        self.snapshot_dim = int(snapshot_dim)
        self.obs_dim = int(obs_dim)
        self.device = torch.device(device)
        self.dead_point_min_attempts = int(dead_point_min_attempts)

        C = self.capacity
        self.sim_state = torch.zeros((C, self.snapshot_dim), dtype=torch.float32, device=self.device)
        self.obs = torch.zeros((C, self.obs_dim), dtype=torch.float32, device=self.device)
        self.add_step = torch.zeros((C,), dtype=torch.long, device=self.device)
        self.add_p_value = torch.zeros((C,), dtype=torch.float32, device=self.device)
        self.source_trajectory_step = torch.zeros((C,), dtype=torch.long, device=self.device)
        self.init_attempts = torch.zeros((C,), dtype=torch.long, device=self.device)
        self.init_successes = torch.zeros((C,), dtype=torch.long, device=self.device)
        self._filled = torch.zeros((C,), dtype=torch.bool, device=self.device)
        # _insertion_order: monotonically increasing counter assigned at add() time.
        # FIFO eviction picks the smallest value among the candidate set.
        self._insertion_order = torch.zeros((C,), dtype=torch.long, device=self.device)
        self._next_insertion: int = 0

        # CPU-side ragged outcome lists, indexed by slot.
        self.init_episode_lengths: list[list[int]] = [[] for _ in range(C)]
        self.init_times_to_success: list[list[int]] = [[] for _ in range(C)]
        self.init_action_entropies: list[list[float]] = [[] for _ in range(C)]

    # ------------------------------------------------------------------
    # Capacity / occupancy
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return int(self._filled.sum().item())

    def is_full(self) -> bool:
        return int(self._filled.sum().item()) >= self.capacity

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------
    def _evict_slot(self) -> int:
        """Pick a slot to overwrite. Dead-first; ties broken by oldest insertion.

        Falls back to overall-oldest insertion (FIFO) if no dead slot exists.
        Asserts the buffer is currently full — callers should only invoke this
        path after ``is_full()``.
        """
        if not self.is_full():
            raise RuntimeError("RescueBuffer._evict_slot called while not full.")
        # Dead = filled AND init_attempts >= threshold AND init_successes == 0.
        dead_mask = (
            self._filled
            & (self.init_attempts >= self.dead_point_min_attempts)
            & (self.init_successes == 0)
        )
        if bool(dead_mask.any().item()):
            # Among dead slots, pick the smallest _insertion_order.
            big = torch.iinfo(self._insertion_order.dtype).max
            scores = torch.where(
                dead_mask,
                self._insertion_order,
                torch.full_like(self._insertion_order, big),
            )
            return int(torch.argmin(scores).item())
        # Pure FIFO over all filled slots.
        big = torch.iinfo(self._insertion_order.dtype).max
        scores = torch.where(
            self._filled,
            self._insertion_order,
            torch.full_like(self._insertion_order, big),
        )
        return int(torch.argmin(scores).item())

    def _next_free_or_evict(self) -> int:
        """Return a slot index ready to be (over)written."""
        if self.is_full():
            slot = self._evict_slot()
            self._clear_slot_metadata(slot)
            return slot
        # First unfilled slot.
        free = (~self._filled).nonzero(as_tuple=False).flatten()
        return int(free[0].item())

    def _clear_slot_metadata(self, slot: int) -> None:
        self.init_attempts[slot] = 0
        self.init_successes[slot] = 0
        self.init_episode_lengths[slot] = []
        self.init_times_to_success[slot] = []
        self.init_action_entropies[slot] = []

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------
    def add(
        self,
        *,
        sim_state: torch.Tensor,
        obs: torch.Tensor,
        add_step: int,
        add_p_value: float,
        source_trajectory_step: int,
    ) -> int:
        """Insert one rescue point; returns the slot index it occupies."""
        if sim_state.shape != (self.snapshot_dim,):
            raise ValueError(
                f"add: sim_state shape {tuple(sim_state.shape)} != ({self.snapshot_dim},)"
            )
        if obs.shape != (self.obs_dim,):
            raise ValueError(f"add: obs shape {tuple(obs.shape)} != ({self.obs_dim},)")
        slot = self._next_free_or_evict()
        self.sim_state[slot].copy_(sim_state.to(self.device, dtype=torch.float32))
        self.obs[slot].copy_(obs.to(self.device, dtype=torch.float32))
        self.add_step[slot] = int(add_step)
        self.add_p_value[slot] = float(add_p_value)
        self.source_trajectory_step[slot] = int(source_trajectory_step)
        self.init_attempts[slot] = 0
        self.init_successes[slot] = 0
        self._filled[slot] = True
        self._insertion_order[slot] = self._next_insertion
        self._next_insertion += 1
        return slot

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Uniform sample with replacement of ``n`` filled slots.

        Returns ``(slot_idx (n,), sim_state (n, snapshot_dim), obs (n, obs_dim))``.
        Raises if the buffer is empty (fail-loud — callers must gate on ``len``).
        """
        if n is None or n <= 0:
            raise ValueError(f"sample: n must be > 0, got {n!r}")
        filled = self._filled.nonzero(as_tuple=False).flatten()
        if filled.numel() == 0:
            raise RuntimeError("RescueBuffer.sample called on empty buffer.")
        pick = torch.randint(
            low=0, high=int(filled.numel()), size=(int(n),), device=self.device
        )
        slot_idx = filled[pick]
        return slot_idx, self.sim_state[slot_idx], self.obs[slot_idx]

    # ------------------------------------------------------------------
    # Outcome bookkeeping
    # ------------------------------------------------------------------
    def record_init(self, slot_idx: int) -> None:
        """Bump ``init_attempts`` for a slot that's about to be used as an init."""
        s = int(slot_idx)
        if s < 0 or s >= self.capacity:
            raise IndexError(f"record_init: slot_idx {s} out of [0, {self.capacity})")
        if not bool(self._filled[s].item()):
            raise RuntimeError(f"record_init: slot {s} is not filled.")
        self.init_attempts[s] += 1

    def record_outcome(
        self,
        slot_idx: int,
        *,
        success: bool,
        length: int,
        time_to_success: int | None,
        action_entropy: float,
    ) -> None:
        """Record the outcome of a trajectory that was initialized from this slot."""
        s = int(slot_idx)
        if s < 0 or s >= self.capacity:
            raise IndexError(f"record_outcome: slot_idx {s} out of [0, {self.capacity})")
        if not bool(self._filled[s].item()):
            # Slot was evicted between init and outcome — silently drop. This is
            # the only "soft" failure: eviction is legitimate and racing it
            # against outstanding outcomes is unavoidable with FIFO. We don't
            # raise here so the rollout loop isn't punished for buffer churn.
            return
        if success:
            self.init_successes[s] += 1
            if time_to_success is None:
                raise ValueError(
                    f"record_outcome: success=True requires time_to_success, got None (slot {s})"
                )
            self.init_times_to_success[s].append(int(time_to_success))
        self.init_episode_lengths[s].append(int(length))
        self.init_action_entropies[s].append(float(action_entropy))
