"""Trajectory-staged replay memory for SAC + success-prediction TD targets.

Subclass of :class:`MultiRandomMemory` that solves the "label availability lags
transition writing" problem inherent to per-step success-prediction targets:

* Transitions are NOT written to the main buffer when ``add_samples`` is called.
  They go into per-env *staging* tensors of length ``max_episode_length``.
* When :meth:`finalize_trajectory` is called for an env (typically on episode
  termination), the staged transitions are scanned for the first step where
  ``is_success_step`` was True (the trajectory's first-success step t*) and
  three derived per-step fields are computed:
    - ``is_first_success_step[t]``: 1.0 at t == t*, else 0.0.
    - ``success_terminal[t]``: 1.0 at t == t* (success-terminal) OR at the
      final staged step if the trajectory failed (failure-terminal); else 0.0.
    - ``success_loss_mask[t]``: 1.0 for steps that contribute to the success-
      head loss (= all pre-and-up-to-success-step steps for successful
      trajectories; all steps for failed ones); 0.0 for post-success steps
      (label is undefined past first-success per design).
  These plus the regular transition tensors are committed to the main buffer.
* The main buffer therefore only ever contains finalized rows — SAC never
  trains against stale labels and there's no in-place mutation.

Each env's main-buffer column has its own write head (``_env_main_index``) so
different envs can be at different write positions; the buffer is asynchronous
across envs. :meth:`sample` accounts for this by sampling timesteps within each
env's own valid range.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch

from skrl.utils.spaces.torch import compute_space_size

from memory.multi_random import MultiRandomMemory


def _find_first_streak(flags: torch.Tensor, n: int) -> tuple[int, int]:
    """Return the (start, end) indices of the first window of ``n`` consecutive
    True entries in the 1-D boolean tensor ``flags``. Returns ``(-1, -1)`` if
    no such window exists.

    ``end`` is inclusive (= start + n − 1). For n=1, this is just the first
    True index; for n>1, this enforces the consecutive-streak criterion used
    to gate trajectories as positive.
    """
    if n <= 0:
        raise ValueError(f"streak length must be >= 1, got {n}")
    if flags.numel() < n:
        return -1, -1
    # Run-length scan: count consecutive Trues; emit first index whose count
    # reaches n. Implemented in Python on CPU since trajectories are short
    # (≤ max_episode_length, typically a few hundred) and per-env.
    flags_cpu = flags.detach().cpu().tolist()
    run = 0
    for i, v in enumerate(flags_cpu):
        if v:
            run += 1
            if run >= n:
                return i - n + 1, i
        else:
            run = 0
    return -1, -1


class TrajectoryBufferedMemory(MultiRandomMemory):
    def __init__(
        self,
        *,
        memory_size: int,
        num_envs: int = 1,
        num_agents: int = 1,
        max_episode_length: int,
        success_streak_len: int = 1,
        success_use_streak: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        export: bool = False,
        export_format: str = "pt",
        export_directory: str = "",
        replacement: bool = True,
    ) -> None:
        super().__init__(
            memory_size=memory_size,
            num_envs=num_envs,
            num_agents=num_agents,
            device=device,
            export=export,
            export_format=export_format,
            export_directory=export_directory,
            replacement=replacement,
        )
        if max_episode_length <= 0:
            raise ValueError(
                f"max_episode_length must be > 0, got {max_episode_length}"
            )
        if success_streak_len < 1:
            raise ValueError(
                f"success_streak_len must be >= 1, got {success_streak_len}"
            )
        self.max_episode_length = int(max_episode_length)
        self.success_streak_len = int(success_streak_len)
        self.success_use_streak = bool(success_use_streak)

        # Per-env staging area: one named tensor of shape
        # (num_envs, max_episode_length, size) per registered tensor name.
        # Allocated lazily in create_tensor().
        self._staging: dict[str, torch.Tensor] = {}

        # Per-env current step pointer into staging.
        self._stage_t = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Per-env main-buffer write head + filled flag — replaces base Memory's
        # single global memory_index/filled. We keep the base attrs untouched (and
        # essentially unused) so existing skrl-side code that reads them doesn't
        # crash, but our sample() override consults these per-env fields instead.
        self._env_main_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._env_filled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------
    # Tensor lifecycle
    # ------------------------------------------------------------------
    def create_tensor(
        self,
        name: str,
        *,
        size,
        dtype: Optional[torch.dtype] = None,
        keep_dimensions: bool = False,
    ) -> bool:
        created = super().create_tensor(
            name=name, size=size, dtype=dtype, keep_dimensions=keep_dimensions
        )
        if not created:
            return False  # tensor already existed; staging is presumably already in place
        # mirror the size resolution that the base does for the main tensor
        flat_size = (
            size if isinstance(size, int) and not keep_dimensions
            else compute_space_size(size, occupied_size=True)
        )
        self._staging[name] = torch.zeros(
            (self.num_envs, self.max_episode_length, flat_size),
            dtype=dtype if dtype is not None else torch.float32,
            device=self.device,
        )
        return True

    # ------------------------------------------------------------------
    # Writes go to staging, not main
    # ------------------------------------------------------------------
    def add_samples(self, **tensors: torch.Tensor) -> None:
        """Append one transition per env into the staging buffer.

        The per-env step pointer ``_stage_t`` advances by 1 across all envs (the
        standard skrl semantic — every env steps in lockstep). Transitions are NOT
        committed to the main buffer here; that happens in :meth:`finalize_trajectory`
        when an env's episode ends.
        """
        if (self._stage_t >= self.max_episode_length).any():
            bad = (self._stage_t >= self.max_episode_length).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"Staging overflow on envs {bad}: episode exceeded "
                f"max_episode_length={self.max_episode_length}. Increase the cap or "
                f"check the env's time-out termination term."
            )
        env_idx = torch.arange(self.num_envs, device=self.device)
        for name, t in tensors.items():
            if name not in self._staging:
                # tensors not registered for staging (e.g. trajectory_succeeded —
                # written at finalize time) are ignored here.
                continue
            if t.shape[0] != self.num_envs:
                raise ValueError(
                    f"add_samples expected '{name}' shape[0]={self.num_envs}, got {t.shape[0]}"
                )
            self._staging[name][env_idx, self._stage_t] = t
        self._stage_t = self._stage_t + 1

    # ------------------------------------------------------------------
    # Episode-end commit
    # ------------------------------------------------------------------
    def finalize_trajectory(self, env_indices: torch.Tensor) -> None:
        """Commit each finishing env's staged trajectory to the main buffer.

        Reads each env's staged ``is_success_step`` (per-step *instantaneous*
        success flag — True iff the geometric success criterion holds right
        now). Scans for the first window of ``success_streak_len`` consecutive
        True flags ([t_start, t_end] with t_end = t_start + N − 1) and stamps
        the three success-head TD-target ingredients across that env's
        committed slots:

          * ``is_first_success_step``: 1 at every step t in [t_start, t_end],
            else 0. This is the "reward" term of the TD target.
          * ``success_terminal``: 1 at every step t in [t_start, t_end], so
            the bootstrap term collapses to 0 inside the streak — each
            streak step has hard target = 1. For trajectories with no
            qualifying streak: 1 at the final staged step (failure-terminal,
            target = 0), else 0.
          * ``success_loss_mask``: 1 for [0, t_end] (pre-streak + streak),
            0 for (t_end, n−1] (post-streak rows masked out — agent may slip
            out, label is undefined). For failed trajectories: all 1.

        Pre-streak step at t < t_start has is_first=0 and terminal=0, so its
        TD target is γ · V(s_{t+1}). Step t_start − 1 thus bootstraps from a
        next-state whose target is 1, giving target ≈ γ.

        :param env_indices: 1D LongTensor of env indices whose trajectories just ended.
        """
        for required in (
            "is_success_step",
            "is_first_success_step",
            "success_terminal",
            "success_loss_mask",
        ):
            if required not in self.tensors:
                raise KeyError(
                    f"finalize_trajectory requires the '{required}' tensor to be "
                    f"registered (call create_tensor first — SAC.init() does this)."
                )
        env_indices = env_indices.flatten().to(torch.long)

        # Process envs sequentially — trajectory length varies per env so vectorising
        # the slice writes is awkward. With a few envs finishing per step this is fast.
        for env_i in env_indices.tolist():
            n = int(self._stage_t[env_i].item())
            if n == 0:
                continue  # nothing staged (e.g. env reset on the very first step)

            base = int(self._env_main_index[env_i].item())
            slot_offsets = torch.arange(n, device=self.device)
            slot_idxs = (base + slot_offsets) % self.memory_size

            # Copy the regular staged tensors first (observations, actions, etc.).
            for name, staged in self._staging.items():
                self.tensors[name][slot_idxs, env_i] = staged[env_i, :n]

            # Resolve trajectory-level success criterion.
            #   streak mode (default): first window of N consecutive Trues
            #     anywhere in the trajectory; success anchors over the window,
            #     mask covers [0, t_end], post-streak rows excluded.
            #   terminal mode: success iff the final staged step's flag is True;
            #     anchor at n-1 only, mask covers all steps. This mirrors the
            #     "success at end" semantic and removes the touch-and-slip
            #     definitional ambiguity entirely.
            is_succ = self._staging["is_success_step"][env_i, :n, 0].bool()
            if self.success_use_streak:
                t_start, t_end = _find_first_streak(is_succ, self.success_streak_len)
            else:
                if bool(is_succ[n - 1].item()):
                    t_start, t_end = n - 1, n - 1
                else:
                    t_start, t_end = -1, -1
            if t_start >= 0:
                # is_first_success_step: 1 across the streak, else 0. Drives
                # the "reward" term of the TD target.
                first_succ = torch.zeros(n, device=self.device, dtype=torch.float32)
                first_succ[t_start : t_end + 1] = 1.0
                # success_terminal: 1 across the streak, so each streak step
                # collapses to a hard target of 1 (no bootstrap).
                terminal = torch.zeros(n, device=self.device, dtype=torch.float32)
                terminal[t_start : t_end + 1] = 1.0
                # success_loss_mask: 1 for [0, t_end] (pre-streak bootstrap +
                # streak anchors), 0 for (t_end, n-1] (post-streak rows
                # excluded — agent may slip out, label undefined).
                mask = torch.zeros(n, device=self.device, dtype=torch.float32)
                mask[: t_end + 1] = 1.0
            else:
                # Failed trajectory — final staged step is the failure-terminal.
                first_succ = torch.zeros(n, device=self.device, dtype=torch.float32)
                terminal = torch.zeros(n, device=self.device, dtype=torch.float32)
                terminal[n - 1] = 1.0
                # All steps contribute to the loss (target propagates 0 backward
                # through bootstrap: γ · 0 = 0).
                mask = torch.ones(n, device=self.device, dtype=torch.float32)

            self.tensors["is_first_success_step"][slot_idxs, env_i, 0] = first_succ
            self.tensors["success_terminal"][slot_idxs, env_i, 0] = terminal
            self.tensors["success_loss_mask"][slot_idxs, env_i, 0] = mask

            # Bookkeeping: advance the per-env head, set filled if we crossed.
            new_index = base + n
            if new_index >= self.memory_size:
                self._env_filled[env_i] = True
            self._env_main_index[env_i] = new_index % self.memory_size
            self._stage_t[env_i] = 0

    # ------------------------------------------------------------------
    # Sampling: per-env valid range
    # ------------------------------------------------------------------
    def sample(
        self,
        names: Tuple[str],
        batch_size: int,
        mini_batches: int = 1,
        sequence_length: int = 1,
    ) -> List[List[torch.Tensor]]:
        """Sample a batch from finalized trajectories only.

        ``batch_size`` is **per-agent** (matching :class:`MultiRandomMemory`'s
        contract — SAC.update relies on the same semantic across both memory
        types so it can ``view(num_agents, batch_size, -1)`` over the result).
        Each agent draws ``batch_size`` rows from its own env partition; for
        each sampled env, the timestep is drawn uniformly from that env's
        valid range (``[0, _env_main_index)`` if it hasn't wrapped, else
        ``[0, memory_size)``). Total returned rows = ``batch_size * num_agents``.
        Raises if any env has zero finalized rows — ``learning_starts`` should
        be ≥ ``max_episode_length`` to guarantee every env has at least one
        finished episode in the buffer.
        """
        per_agent = batch_size

        t_max_per_env = torch.where(
            self._env_filled,
            torch.full_like(self._env_main_index, self.memory_size),
            self._env_main_index,
        )
        if (t_max_per_env == 0).any():
            empty = (t_max_per_env == 0).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"Cannot sample: envs {empty} have no finalized trajectories. "
                f"Increase learning_starts so every env finishes at least one episode "
                f"before SAC.update() fires (learning_starts >= max_episode_length)."
            )

        chunks = []
        for a in range(self.num_agents):
            env_lo = a * self.envs_per_agent
            env_hi = env_lo + self.envs_per_agent
            e = torch.randint(env_lo, env_hi, (per_agent,), device=self.device)
            ub = t_max_per_env[e].to(torch.float32)
            t = (torch.rand(per_agent, device=self.device) * ub).long()
            chunks.append(t * self.num_envs + e)
        indices = torch.cat(chunks, dim=0)
        return [self.sample_by_index(names, indexes=indices)[0]]

    # NOTE: sample_all() inherited from MultiRandomMemory uses the base's single
    # global memory_index/filled and would NOT give correct results with the
    # per-env asynchronous layout used here. SAC only calls sample(), so we leave
    # sample_all() unaudited — do not use it with this class without first
    # adapting it to per-env indexing.
