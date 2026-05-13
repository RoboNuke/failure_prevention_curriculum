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


# Default number of step-bins along the heatmap Y axis. Override per-run via
# `sac_cfg.success_heatmap_step_bins`. Independent of max_episode_length so
# heatmaps are visually comparable across tasks of different episode lengths.
_DEFAULT_HEATMAP_STEP_BINS = 30
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
        success_streak_len: int = 1,
        success_use_streak: bool = True,
        heatmap_step_bins: int = _DEFAULT_HEATMAP_STEP_BINS,
        on_finalize=None,
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
        if success_streak_len < 1:
            raise ValueError(
                f"success_streak_len must be >= 1, got {success_streak_len}"
            )
        self.success_streak_len = int(success_streak_len)
        self.success_use_streak = bool(success_use_streak)
        if heatmap_step_bins < 1:
            raise ValueError(
                f"heatmap_step_bins must be >= 1, got {heatmap_step_bins}"
            )
        self._n_bins = int(heatmap_step_bins)
        self._ece_bins = _ECE_BINS

        # Optional callback invoked inside _finalize for each finishing env.
        # Signature: on_finalize(env_i, P_np, succ_np, outcome, n, extras) where
        # ``extras`` is a dict of per-step tensors (sliced to [0:n] on the
        # ``device``) staged via update(extra=...). Used by the rescue-buffer
        # subsystem to run a backward scan + slot insertion without coupling
        # tracker internals to that module.
        self._on_finalize = on_finalize
        # Lazy per-extra-tensor staging. Allocated on first update(..., extra=...)
        # call so this stays free for predict_success-only configs.
        self._stage_extras: dict[str, torch.Tensor] = {}

        # Per-env rollout staging.
        self._stage_P = torch.zeros(
            (num_envs, self.max_episode_length), dtype=torch.float32, device=self.device
        )
        # ``_stage_succ`` is the per-step *instantaneous* success indicator
        # (True iff the geometric success criterion holds right now). At
        # finalize time we scan it for the first window of
        # ``success_streak_len`` consecutive Trues — same criterion the
        # memory uses to stamp the training TD targets. Outcome label is
        # "trajectory contained a qualifying streak"; mask is [0, t_end].
        self._stage_succ = torch.zeros(
            (num_envs, self.max_episode_length), dtype=torch.bool, device=self.device
        )
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
        # Per-agent calibration history: one column per flush, length
        # ``_ece_bins``, value = signed ``conf_b − acc_b`` per ECE bin (NaN
        # where the bin had no rows this interval). Rendered with a diverging
        # cmap centered on 0 — see _render_and_log_heatmap.
        self._hist_calib: list[list[np.ndarray]] = [[] for _ in range(num_agents)]

    # ------------------------------------------------------------------
    # Per-step ingestion
    # ------------------------------------------------------------------
    def update(
        self,
        success_prob: torch.Tensor,
        is_success_step: torch.Tensor,
        done_mask: torch.Tensor,
        extra: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Stage one step per env, then finalize trajectories for envs whose
        ``done_mask`` is True.

        ``success_prob``    : (num_envs,) float in [0, 1] — actor's online prediction.
        ``is_success_step`` : (num_envs,) bool — *instantaneous* per-step success
                              flag (geometric criterion holds right now). At
                              finalize-time, scanned for the first window of
                              ``success_streak_len`` consecutive Trues.
        ``done_mask``       : (num_envs,) bool.
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
        # Stage optional per-step extras alongside P / is_success_step. First
        # encounter of a key allocates the storage tensor sized to match the
        # extra's per-env shape: (num_envs, max_ep_len, *trailing).
        if extra is not None:
            for k, v in extra.items():
                if v.shape[0] != self.num_envs:
                    raise ValueError(
                        f"extras[{k!r}] leading dim {v.shape[0]} != num_envs {self.num_envs}"
                    )
                tail = tuple(v.shape[1:])
                buf = self._stage_extras.get(k)
                if buf is None:
                    buf = torch.zeros(
                        (self.num_envs, self.max_episode_length, *tail),
                        dtype=v.dtype if v.is_floating_point() else torch.float32,
                        device=self.device,
                    )
                    self._stage_extras[k] = buf
                buf[env_idx, self._stage_t] = v.detach().to(self.device).to(buf.dtype)
        self._stage_t = self._stage_t + 1

        if done_mask.any():
            self._finalize(done_mask.nonzero(as_tuple=False).flatten())

    def _finalize(self, env_indices: torch.Tensor) -> None:
        """Snapshot each finishing env's trajectory into the per-agent buffer and
        reset its staging.

        Outcome and mask are derived from the same N-streak criterion the
        memory uses to stamp TD targets:
        * Scan staged ``is_success_step`` for the first window of
          ``success_streak_len`` consecutive Trues. The window ends at
          ``t_end = t_start + N − 1``.
        * If found: ``outcome=success``, mask covers ``[0, t_end]``
          (pre-streak bootstrap steps + streak anchors). Post-streak rows
          are excluded — agent may slip out, label is undefined there.
        * If not found: ``outcome=failure``, mask covers all steps.

        This guarantees the rollout-time predictive-quality metrics (AUC,
        per-class BCE, ECE, monotonicity, heatmaps) classify trajectories
        and weight steps the same way the training loss does — they
        measure exactly what the head is being optimized to predict.
        """
        n_streak = self.success_streak_len
        for env_i in env_indices.tolist():
            n = int(self._stage_t[env_i].item())
            if n == 0:
                continue
            P = self._stage_P[env_i, :n].detach().cpu().numpy().astype(np.float32, copy=True)
            succ = self._stage_succ[env_i, :n].detach().cpu().numpy()

            # Apply the same trajectory-level criterion the memory uses, so
            # outcome labels and masks match the training TD targets.
            t_start, t_end = -1, -1
            if self.success_use_streak:
                # First run of n_streak consecutive True.
                if succ.size >= n_streak:
                    run = 0
                    for i in range(succ.size):
                        if succ[i]:
                            run += 1
                            if run >= n_streak:
                                t_start = i - n_streak + 1
                                t_end = i
                                break
                        else:
                            run = 0
            else:
                # Terminal mode: success iff the final staged step is True.
                if succ.size > 0 and bool(succ[-1]):
                    t_start, t_end = succ.size - 1, succ.size - 1

            if t_start >= 0:
                mask = np.zeros(n, dtype=bool)
                mask[: t_end + 1] = True
                outcome = True
            else:
                mask = np.ones(n, dtype=bool)
                outcome = False

            agent = env_i // self.epa
            self._completed[agent].append(
                {"P": P, "mask": mask, "outcome": outcome, "n": n}
            )
            # Fire rescue-side callback BEFORE zeroing _stage_t so any consumer
            # that needs the trailing trajectory length sees the post-increment
            # value. Extras are sliced to [0:n] on-device; callback is free to
            # copy to CPU / push to its own buffers.
            if self._on_finalize is not None:
                extras_slice: dict[str, torch.Tensor] = {}
                for k, buf in self._stage_extras.items():
                    extras_slice[k] = buf[env_i, :n].clone()
                # P_np and mask are already CPU NumPy; outcome is python bool.
                self._on_finalize(int(env_i), P, succ, bool(outcome), int(n), extras_slice)
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
            # if ``num fail trajs`` is 0 every interval, every trajectory
            # contains a qualifying N-streak — bump ``success_streak_len`` if
            # the head is over-confidently calling everything a success.
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
            from learning.calibration_utils import roc_auc as _roc_auc
            auc = _roc_auc(P_all, y_all)
            if auc is not None:
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

            # ------ ECE (10 equal-width P bins) + signed-gap heatmap column ------
            # `calib_col[b]` = conf_b − acc_b (signed; negative = overconfident,
            # positive = underconfident). NaN for empty bins so the heatmap
            # renders them black via the diverging cmap's `set_bad`.
            from learning.calibration_utils import ece as _ece
            ece_val, calib_col = _ece(P_all, y_all, self._ece_bins)
            per_agent_tracking[i]["Success Prediction Quality / ECE"].append(float(ece_val))
            self._hist_calib[i].append(calib_col)

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
            # NaN column, which renders BLACK under our cmap (cmap.set_bad)
            # — visually distinct from any value in the [0,1] RdYlGn range,
            # so "no data of this class in this interval" is unambiguous.
            # Within a column, individual bins with no rows are also NaN
            # (see `_build_column`), and likewise render black.
            nan_col = np.full(self._n_bins, np.nan, dtype=np.float32)
            self._hist_succ[i].append(col_succ if col_succ is not None else nan_col)
            self._hist_fail[i].append(col_fail if col_fail is not None else nan_col)
            # Step-binned heatmaps render with the y-axis labelled in actual
            # episode-step units (0..max_episode_length) instead of bin
            # indices. With `heatmap_step_bins` rows of pixel data, each row
            # visually spans `max_episode_length / n_bins` steps — set
            # `success_heatmap_step_bins = max_episode_length` for one row per
            # step (no aggregation), keep it small (e.g. 30) to compare
            # heatmaps across tasks of different episode lengths.
            steps_per_bin = self.max_episode_length / max(1, self._n_bins)
            step_ylabel = (
                f"episode step (0..{self.max_episode_length - 1}, "
                f"{steps_per_bin:g} step(s) per row)"
            )
            self._render_and_log_heatmap(
                history=self._hist_succ[i],
                writer=per_agent_writers[i],
                tag="Success Prediction Quality / heatmap success",
                title=f"Success-trajectory mean P per episode step (agent {i})",
                env_step=timestep,
                y_extent=(0.0, float(self.max_episode_length)),
                ylabel=step_ylabel,
            )
            self._render_and_log_heatmap(
                history=self._hist_fail[i],
                writer=per_agent_writers[i],
                tag="Success Prediction Quality / heatmap failure",
                title=f"Failure-trajectory mean P per episode step (agent {i})",
                env_step=timestep,
                y_extent=(0.0, float(self.max_episode_length)),
                ylabel=step_ylabel,
            )

            # Calibration heatmap: signed gap conf_b − acc_b per ECE bin over
            # time. Diverging colormap red → green → blue, centered at 0:
            #   negative = overconfident (P too high vs. realized accuracy)
            #   ~zero    = well calibrated
            #   positive = underconfident (P too low vs. realized accuracy)
            # NaN bins (no rows in this interval) render black, same UX as
            # the per-class heatmaps. vmin/vmax = ±1 covers the full range
            # of conf−acc; in practice values cluster well inside this.
            mpl_figure, mpl_cm = _ensure_mpl()
            from matplotlib.colors import LinearSegmentedColormap
            calib_cmap = LinearSegmentedColormap.from_list(
                "calib_rgb", ["red", "green", "blue"], N=256
            )
            calib_cmap.set_bad(color="black")
            self._render_and_log_heatmap(
                history=self._hist_calib[i],
                writer=per_agent_writers[i],
                tag="Success Prediction Quality / heatmap calibration",
                title=f"Calibration gap (conf − acc) per predicted-P (agent {i})",
                env_step=timestep,
                cmap=calib_cmap,
                vmin=-1.0,
                vmax=1.0,
                y_extent=(0.0, 1.0),
                ylabel=f"predicted P (0..1, {self._ece_bins} bins)",
                cbar_label="conf − acc  (− overconfident, + underconfident)",
                yticks=[b / self._ece_bins for b in range(self._ece_bins + 1)],
            )

            # Clear this agent's interval buffer.
            self._completed[i].clear()

    # ------------------------------------------------------------------
    # Heatmap rendering
    # ------------------------------------------------------------------
    def _render_and_log_heatmap(
        self,
        *,
        history: list[np.ndarray],
        writer: Any,
        tag: str,
        title: str,
        env_step: int,
        cmap: Any | None = None,
        vmin: float = 0.0,
        vmax: float = 1.0,
        ylabel: str | None = None,
        cbar_label: str = "mean P(success)",
        y_extent: tuple[float, float] | None = None,
        yticks: list[float] | None = None,
    ) -> None:
        mpl_figure, mpl_cm = _ensure_mpl()
        H = np.stack(history, axis=1)  # (n_bins, n_iters)
        n_y = H.shape[0]
        # Axis-label range; pixel data is unaffected.
        y_lo, y_hi = y_extent if y_extent is not None else (0.0, float(n_y))
        fig = mpl_figure.Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        if cmap is None:
            # Default: RdYlGn with NaN→black for the per-class P-mean heatmaps.
            cmap = mpl_cm.get_cmap("RdYlGn").copy()
            cmap.set_bad(color="black")
        # X axis maps to env steps: each column corresponds to one
        # write_interval, and we append a column every flush — so the columns
        # span [0, env_step]. Y axis is step bin index.
        im = ax.imshow(
            H,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            cmap=cmap,
            interpolation="nearest",
            extent=[0.0, float(env_step), y_lo, y_hi],
        )
        ax.set_xlabel("env steps")
        ax.set_ylabel(
            ylabel if ylabel is not None
            else f"row (0..{n_y - 1})"
        )
        if yticks is not None:
            ax.set_yticks(yticks)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label=cbar_label)
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
