"""Per-write-interval predictive-quality metrics for the success head.

Captures the actor's online success-probability prediction at each rollout
step, finalizes per-trajectory data on episode end (with the same masking
rules as :class:`memory.trajectory_buffered.TrajectoryBufferedMemory` —
post-success steps masked out, pre-success and failed steps included), then
on TB flush computes:

    * AUC(ROC) of (P_t, traj_outcome) over non-masked transitions.
    * Per-class BCE — successful trajs vs failed trajs.
    * ECE with 10 equal-width P bins.
    * Outcome-relative trajectory monotonicity, split per class:
        - "monotonicity success": for successful trajs, fraction of (t, t+1)
          step pairs where P rises (i.e. moves toward eventual success).
        - "monotonicity fail":    for failed trajs, fraction of step pairs
          where P falls (moves toward eventual failure).
    * Heatmaps (per class, per agent): Y = step bin (30 bins along episode),
      X = TB-write iteration, color = mean P. Each ``flush`` appends one
      column per class. Rendered with matplotlib's ``RdYlGn`` cmap, written
      as PNG image to TB at ``global_step=0`` so the dashboard always shows
      only the latest.

All metrics use rollout-time predictions (the actor's ``success_prob`` at
``act()`` time), restricted to non-masked transitions. The interval window
is whatever set of trajectories completed since the last flush.
"""

from __future__ import annotations

import io
from collections import defaultdict
from typing import Any

import numpy as np
import torch

# matplotlib is part of the Isaac Lab env. Lazy-import inside the render
# function so import time of this module stays cheap when predict_success
# is False (the tracker is never instantiated in that case anyway).
_MPL = None


def _ensure_mpl():
    global _MPL
    if _MPL is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.figure as mpl_figure
        import matplotlib.cm as mpl_cm

        _MPL = (mpl_figure, mpl_cm)
    return _MPL


# Number of step-bins along the heatmap Y axis. Independent of max_episode_length
# so heatmaps are visually comparable across tasks of different episode lengths.
_HEATMAP_STEP_BINS = 30
# ECE bins: 10 equal-width bins of P over [0, 1].
_ECE_BINS = 10


class PredictionQualityTracker:
    """Per-rollout-trajectory predictive quality bookkeeping for the success head.

    Stores per-env staged P values during the rollout, finalizes each
    trajectory with the same masking convention as the trajectory-buffered
    memory (post-success rows masked out for successful trajs; all rows
    kept for failed trajs), and at flush time produces the scalar + image
    metrics listed in the module docstring.
    """

    def __init__(
        self,
        num_envs: int,
        num_agents: int,
        max_episode_length: int,
        device: torch.device | str,
    ) -> None:
        if num_envs % num_agents != 0:
            raise ValueError(
                f"num_envs ({num_envs}) must be divisible by num_agents ({num_agents})"
            )
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.epa = num_envs // num_agents
        self.device = torch.device(device)
        self.max_episode_length = int(max_episode_length)
        self._n_bins = _HEATMAP_STEP_BINS
        self._ece_bins = _ECE_BINS

        # Per-env rollout staging.
        self._stage_P = torch.zeros(
            (num_envs, self.max_episode_length), dtype=torch.float32, device=self.device
        )
        # ``_stage_succ`` is the per-step "ever touched goal" indicator (e.g.
        # Factory's ``ep_succeeded``, which latches True from the first
        # geometric success onward). Used to find the trajectory's first-
        # success step t* — the same notion used for TD-target masking.
        self._stage_succ = torch.zeros(
            (num_envs, self.max_episode_length), dtype=torch.bool, device=self.device
        )
        # ``_stage_curr`` is the per-step *instantaneous* success indicator
        # (e.g. Factory's ``curr_successes`` — True iff at goal pose right
        # now). Optional — populated only when the wrapper publishes it. The
        # truncation-step value is the trajectory's strict-outcome label
        # (matches ``successes`` semantics) and is what we use to classify
        # success vs failure for the pred-quality metrics. Without this we
        # fall back to "ever touched" via _stage_succ, which on tasks where
        # the agent often touches-and-slips makes every trajectory look like
        # a success and silences the failure-class metrics.
        self._stage_curr = torch.zeros(
            (num_envs, self.max_episode_length), dtype=torch.bool, device=self.device
        )
        self._has_curr_data = False
        self._stage_t = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        # Completed trajectories accumulated since the last flush, partitioned
        # per agent. Each entry is a dict with numpy arrays — converting to
        # numpy at finalize-time avoids holding GPU tensors across the whole
        # interval.
        self._completed: list[list[dict]] = [[] for _ in range(num_agents)]

        # Per-(agent, class) heatmap history. Each list grows by one column
        # (length _n_bins) per flush that observed at least one trajectory
        # of that class. Stored as numpy arrays of length _n_bins (NaN where
        # no data).
        self._hist_succ: list[list[np.ndarray]] = [[] for _ in range(num_agents)]
        self._hist_fail: list[list[np.ndarray]] = [[] for _ in range(num_agents)]

    # ------------------------------------------------------------------
    # Per-step ingestion
    # ------------------------------------------------------------------
    def update(
        self,
        success_prob: torch.Tensor,
        is_success_step: torch.Tensor,
        done_mask: torch.Tensor,
        curr_success: torch.Tensor | None = None,
    ) -> None:
        """Stage one step per env, then finalize trajectories for envs whose
        ``done_mask`` is True.

        ``success_prob``    : (num_envs,) float in [0, 1] — actor's online prediction.
        ``is_success_step`` : (num_envs,) bool — "ever touched goal" sticky indicator
                              (e.g. Factory's ``ep_succeeded``). Used to find t*.
        ``done_mask``       : (num_envs,) bool.
        ``curr_success``    : (num_envs,) bool, optional — instantaneous geometric
                              success indicator (e.g. Factory's ``curr_successes``).
                              When provided, the trajectory's outcome label uses the
                              truncation-step value (matches the env's ``successes``
                              metric); when absent, falls back to "ever touched".
        """
        if success_prob.shape[0] != self.num_envs:
            raise ValueError(
                f"success_prob shape {tuple(success_prob.shape)} != num_envs={self.num_envs}"
            )
        if (self._stage_t >= self.max_episode_length).any():
            bad = (self._stage_t >= self.max_episode_length).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"PredictionQualityTracker staging overflow on envs {bad}: episode "
                f"exceeded max_episode_length={self.max_episode_length}."
            )
        env_idx = torch.arange(self.num_envs, device=self.device)
        self._stage_P[env_idx, self._stage_t] = success_prob.detach().to(self.device).float().view(-1)
        self._stage_succ[env_idx, self._stage_t] = is_success_step.to(self.device).bool().view(-1)
        if curr_success is not None:
            self._stage_curr[env_idx, self._stage_t] = curr_success.to(self.device).bool().view(-1)
            self._has_curr_data = True
        self._stage_t = self._stage_t + 1

        if done_mask.any():
            self._finalize(done_mask.nonzero(as_tuple=False).flatten())

    def _finalize(self, env_indices: torch.Tensor) -> None:
        """Snapshot each finishing env's trajectory into the per-agent buffer and
        reset its staging.

        Two distinct flags are derived from the staged data:
        * ``t*`` (mask boundary) — first step of ``is_success_step`` (= ever-
          touched-goal). Same notion the TD-target memory uses for masking.
          Determines which steps contribute to the loss / metrics: pre-and-at
          for trajectories with a t*, all steps for those without.
        * ``outcome`` (per-trajectory label for AUC / per-class BCE / heatmap
          classification) — when ``curr_success`` was staged, uses the value at
          the truncation step (= what the env's ``successes`` metric measures);
          otherwise falls back to "ever touched goal" (= ``t* exists``).

        These differ on tasks like Factory where the agent can briefly touch
        the goal pose then slip out: such trajectories have a real ``t*`` (so
        post-touch steps get masked) but the strict outcome label is *failure*
        (geometric criterion not met at end). Without this distinction every
        touch-and-slip trajectory was being labelled success-class, which
        silenced the failure-side metrics.
        """
        for env_i in env_indices.tolist():
            n = int(self._stage_t[env_i].item())
            if n == 0:
                continue
            P = self._stage_P[env_i, :n].detach().cpu().numpy().astype(np.float32, copy=True)
            succ = self._stage_succ[env_i, :n].detach().cpu().numpy()
            success_idx = np.flatnonzero(succ)
            if success_idx.size > 0:
                t_star = int(success_idx[0])
                mask = np.zeros(n, dtype=bool)
                mask[: t_star + 1] = True
            else:
                mask = np.ones(n, dtype=bool)

            if self._has_curr_data:
                # Strict outcome: at goal pose at the truncation moment.
                outcome = bool(self._stage_curr[env_i, n - 1].item())
            else:
                # Loose outcome fallback: ever touched goal.
                outcome = success_idx.size > 0

            agent = env_i // self.epa
            self._completed[agent].append(
                {"P": P, "mask": mask, "outcome": outcome, "n": n}
            )
            self._stage_t[env_i] = 0

    # ------------------------------------------------------------------
    # Per-flush metric computation
    # ------------------------------------------------------------------
    def flush_per_agent(
        self,
        per_agent_tracking: list[dict],
        per_agent_writers: list,
        timestep: int,
    ) -> None:
        """Compute interval metrics from completed trajectories, append to per-agent
        scalar buckets, render and emit per-agent heatmaps to TB at step=0
        (overwrites previous in the TB UI), then clear the interval buffer.
        """
        for i in range(self.num_agents):
            trajs = self._completed[i]
            if not trajs:
                # No finished trajectories this interval — leave heatmaps as-is
                # (don't append a blank column; the next non-empty flush will
                # produce a contiguous one).
                continue

            # Per-class trajectory counts surfaced unconditionally so a missing
            # AUC / BCE-failure / heatmap-failure can be diagnosed immediately:
            # if ``num fail trajs`` is 0 every interval, the issue is the
            # outcome label (e.g. on Factory ``info["is_success"]`` reflects
            # ``ep_succeeded`` which latches on *any* touch of the goal pose,
            # so trajectories that touch-and-slip are still labelled success).
            n_succ_trajs = sum(1 for t in trajs if t["outcome"])
            n_fail_trajs = len(trajs) - n_succ_trajs
            per_agent_tracking[i]["Success Prediction Quality / num success trajs"].append(
                float(n_succ_trajs)
            )
            per_agent_tracking[i]["Success Prediction Quality / num fail trajs"].append(
                float(n_fail_trajs)
            )

            # Collect non-masked rows across all trajectories.
            P_list, label_list, traj_idx_list, step_idx_list = [], [], [], []
            # And per-trajectory mask-applied P for monotonicity.
            traj_P_masked_succ: list[np.ndarray] = []
            traj_P_masked_fail: list[np.ndarray] = []
            for ti, t in enumerate(trajs):
                m = t["mask"]
                if not m.any():
                    continue
                P = t["P"]
                outcome = t["outcome"]
                masked_P = P[m]
                if outcome:
                    traj_P_masked_succ.append(masked_P)
                else:
                    traj_P_masked_fail.append(masked_P)
                P_list.append(masked_P)
                label_list.append(np.full(masked_P.shape[0], 1.0 if outcome else 0.0, dtype=np.float32))
                step_idx_list.append(np.flatnonzero(m).astype(np.int32))
                traj_idx_list.append(np.full(masked_P.shape[0], ti, dtype=np.int32))

            if not P_list:
                continue

            P_all = np.concatenate(P_list)
            y_all = np.concatenate(label_list)
            step_all = np.concatenate(step_idx_list)

            # ------ AUC(ROC) ------
            # Defined only when both classes are represented.
            if y_all.min() < y_all.max():
                from sklearn.metrics import roc_auc_score

                auc = float(roc_auc_score(y_all, P_all))
                per_agent_tracking[i]["Success Prediction Quality / AUC ROC"].append(auc)

            # ------ Per-class BCE (calibration-style, vs trajectory label) ------
            # Use a small eps to avoid log(0). Exclude NaN-on-empty cases by
            # gating on .any().
            eps = 1e-7
            succ_rows = y_all > 0.5
            if succ_rows.any():
                bce_succ = float(-np.log(np.clip(P_all[succ_rows], eps, 1.0)).mean())
                per_agent_tracking[i]["Success Prediction Quality / BCE success class"].append(bce_succ)
            fail_rows = ~succ_rows
            if fail_rows.any():
                bce_fail = float(-np.log(np.clip(1.0 - P_all[fail_rows], eps, 1.0)).mean())
                per_agent_tracking[i]["Success Prediction Quality / BCE failure class"].append(bce_fail)

            # ------ ECE (10 equal-width P bins) ------
            ece = 0.0
            N = P_all.shape[0]
            edges = np.linspace(0.0, 1.0, self._ece_bins + 1)
            for b in range(self._ece_bins):
                lo, hi = edges[b], edges[b + 1]
                if b == self._ece_bins - 1:
                    in_bin = (P_all >= lo) & (P_all <= hi)  # include 1.0
                else:
                    in_bin = (P_all >= lo) & (P_all < hi)
                n_b = int(in_bin.sum())
                if n_b == 0:
                    continue
                conf_b = float(P_all[in_bin].mean())
                acc_b = float(y_all[in_bin].mean())
                ece += (n_b / N) * abs(conf_b - acc_b)
            per_agent_tracking[i]["Success Prediction Quality / ECE"].append(float(ece))

            # ------ Per-class trajectory monotonicity (outcome-relative) ------
            # For each non-masked traj-segment (length k_j), count step pairs
            # where dP has the right sign for that traj's outcome. Per-traj
            # fraction, then average across trajs in this interval.
            mono_succ_list: list[float] = []
            for masked_P in traj_P_masked_succ:
                if masked_P.size < 2:
                    continue
                d = np.diff(masked_P)
                mono_succ_list.append(float((d > 0).mean()))
            if mono_succ_list:
                per_agent_tracking[i]["Success Prediction Quality / monotonicity success"].append(
                    float(np.mean(mono_succ_list))
                )

            mono_fail_list: list[float] = []
            for masked_P in traj_P_masked_fail:
                if masked_P.size < 2:
                    continue
                d = np.diff(masked_P)
                mono_fail_list.append(float((d < 0).mean()))
            if mono_fail_list:
                per_agent_tracking[i]["Success Prediction Quality / monotonicity fail"].append(
                    float(np.mean(mono_fail_list))
                )

            # ------ Heatmap column update + render ------
            # Step bins: floor(step / max_ep_len * n_bins), clamped.
            bin_idx_all = np.minimum(
                (step_all.astype(np.float32) / max(1, self.max_episode_length) * self._n_bins).astype(np.int32),
                self._n_bins - 1,
            )

            # Build a column for each class (mean P per bin, NaN if empty).
            def _build_column(rows_mask: np.ndarray) -> np.ndarray | None:
                if not rows_mask.any():
                    return None
                bins = bin_idx_all[rows_mask]
                Ps = P_all[rows_mask]
                col = np.full(self._n_bins, np.nan, dtype=np.float32)
                # vectorized mean per bin via bincount
                counts = np.bincount(bins, minlength=self._n_bins)
                sums = np.bincount(bins, weights=Ps, minlength=self._n_bins)
                non_empty = counts > 0
                col[non_empty] = sums[non_empty] / counts[non_empty]
                return col

            col_succ = _build_column(y_all > 0.5)
            col_fail = _build_column(y_all <= 0.5)

            # Append a column to BOTH classes every interval so the two
            # heatmaps stay X-axis-aligned. Missing-class intervals get a
            # zeros column (renders solid red under the RdYlGn cmap, visually
            # signalling "no data of this class in this interval"). Both
            # heatmaps re-render every flush.
            zeros_col = np.zeros(self._n_bins, dtype=np.float32)
            self._hist_succ[i].append(col_succ if col_succ is not None else zeros_col)
            self._hist_fail[i].append(col_fail if col_fail is not None else zeros_col)
            self._render_and_log_heatmap(
                history=self._hist_succ[i],
                writer=per_agent_writers[i],
                tag="Success Prediction Quality / heatmap success",
                title=f"Success-trajectory mean P per step bin (agent {i})",
            )
            self._render_and_log_heatmap(
                history=self._hist_fail[i],
                writer=per_agent_writers[i],
                tag="Success Prediction Quality / heatmap failure",
                title=f"Failure-trajectory mean P per step bin (agent {i})",
            )

            # Clear this agent's interval buffer.
            self._completed[i].clear()

    # ------------------------------------------------------------------
    # Heatmap rendering
    # ------------------------------------------------------------------
    def _render_and_log_heatmap(
        self, *, history: list[np.ndarray], writer: Any, tag: str, title: str
    ) -> None:
        mpl_figure, mpl_cm = _ensure_mpl()
        H = np.stack(history, axis=1)  # (n_bins, n_iters)
        fig = mpl_figure.Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        im = ax.imshow(
            H,
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            origin="lower",
            cmap="RdYlGn",
            interpolation="nearest",
        )
        ax.set_xlabel("flush iteration")
        ax.set_ylabel(f"step bin (0..{self._n_bins - 1}, episode-relative)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="mean P(success)")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)

        # Decode PNG → HWC uint8 → CHW float tensor for SummaryWriter.add_image.
        # Avoid pulling PIL: use matplotlib's image.imread which accepts a buffer.
        import matplotlib.image as mpl_image

        arr = mpl_image.imread(buf)  # HWC float in [0, 1] or HWC uint8
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]  # drop alpha
        chw = np.transpose(arr, (2, 0, 1))  # CHW

        # global_step=0 → TB's image dashboard shows the latest event at this
        # tag/step. Newer writes overwrite the displayed image even though
        # underlying events accumulate on disk.
        writer.add_image(tag, chw, global_step=0, dataformats="CHW")
