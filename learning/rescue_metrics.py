"""Rescue-buffer evaluation metrics.

Implements Sections 1–6 of
``/home/hunter/Downloads/rescue_buffer_metrics_spec.md`` plus the rolling
success-rate accumulator that Algorithm 1 reads to gate buffer adds and
rescue inits.

Design mirrors :class:`learning.pred_quality.PredictionQualityTracker`:
per-agent scalar buckets in ``per_agent_tracking[i][tag]`` are appended during
the rollout and consumed by ``flush_per_agent`` on a write_interval boundary;
histograms/images go directly to the per-agent SummaryWriter.

GPU-resident rolling-window storage (per agent):

* ``_ring_success            (W,) bool``
* ``_ring_return             (W,) float32``
* ``_ring_length             (W,) long``
* ``_ring_time_to_success    (W,) long   (-1 sentinel when not successful)``
* ``_ring_init_flag          (W,) bool``  ← True ⇒ initialized from rescue point
* ``_ring_slot_idx           (W,) long   (-1 sentinel when default init)``
* ``_ring_action_entropy_k   (W,) float32`` — mean entropy over first K steps
* ``_ring_states             (W, max_ep_len, obs_dim) float32`` — raw obs per step
* ``_ring_state_len          (W,) long`` — actual length of each stored trajectory

Required ctor collaborators (None ⇒ raise):
``cfg, summary_writers, observation_preprocessor, success_prob_query,
rescue_buffers, num_agents, epa, obs_dim, max_episode_length,
write_interval, experiment_dir, device``.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from configs.manager.rescue_buffer_cfg import RescueBufferCfg
from learning.calibration_utils import ece as _ece
from learning.calibration_utils import roc_auc as _roc_auc
from memory.rescue_buffer import RescueBuffer


def _require(name: str, value: Any) -> Any:
    if value is None:
        raise ValueError(f"RescueMetricsTracker.{name} is required (no default).")
    return value


class RescueMetricsTracker:
    """Per-agent rescue-buffer metric accumulator + publisher."""

    def __init__(
        self,
        *,
        cfg: RescueBufferCfg,
        summary_writers: list,
        observation_preprocessor: Callable[[torch.Tensor], torch.Tensor],
        success_prob_query: Callable[[torch.Tensor, int], torch.Tensor],
        rescue_buffers: list[RescueBuffer],
        num_agents: int,
        epa: int,
        obs_dim: int,
        max_episode_length: int,
        write_interval: int,
        experiment_dir: str | os.PathLike,
        device: torch.device | str,
    ) -> None:
        self.cfg = _require("cfg", cfg)
        self.summary_writers = _require("summary_writers", summary_writers)
        self.observation_preprocessor = _require(
            "observation_preprocessor", observation_preprocessor
        )
        self.success_prob_query = _require("success_prob_query", success_prob_query)
        self.rescue_buffers = _require("rescue_buffers", rescue_buffers)
        self.num_agents = int(_require("num_agents", num_agents))
        self.epa = int(_require("epa", epa))
        self.obs_dim = int(_require("obs_dim", obs_dim))
        self.max_episode_length = int(_require("max_episode_length", max_episode_length))
        self.write_interval = int(_require("write_interval", write_interval))
        self.experiment_dir = Path(_require("experiment_dir", experiment_dir))
        self.device = torch.device(_require("device", device))

        if len(rescue_buffers) != self.num_agents:
            raise ValueError(
                f"rescue_buffers length ({len(rescue_buffers)}) != num_agents ({self.num_agents})"
            )
        if len(summary_writers) != self.num_agents:
            raise ValueError(
                f"summary_writers length ({len(summary_writers)}) != num_agents ({self.num_agents})"
            )
        if cfg.metric_compute_interval % self.write_interval != 0:
            raise ValueError(
                f"rescue_buffer_cfg.metric_compute_interval ({cfg.metric_compute_interval}) "
                f"must be a multiple of write_interval ({self.write_interval}); recommended equal."
            )

        self.W = int(cfg.window_size)
        self.K = int(cfg.action_entropy_first_k_steps)

        # Per-agent rolling window state.
        self._ring_head: list[int] = [0] * self.num_agents
        self._ring_filled: list[int] = [0] * self.num_agents

        def _zeros(*shape, dtype=torch.float32):
            return [torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(self.num_agents)]

        def _full(value, *shape, dtype=torch.long):
            return [
                torch.full(shape, value, dtype=dtype, device=self.device)
                for _ in range(self.num_agents)
            ]

        self._ring_success = _zeros(self.W, dtype=torch.bool)
        self._ring_return = _zeros(self.W, dtype=torch.float32)
        self._ring_length = _zeros(self.W, dtype=torch.long)
        self._ring_time_to_success = _full(-1, self.W, dtype=torch.long)
        self._ring_init_flag = _zeros(self.W, dtype=torch.bool)
        self._ring_slot_idx = _full(-1, self.W, dtype=torch.long)
        self._ring_action_entropy_k = _zeros(self.W, dtype=torch.float32)
        self._ring_states = _zeros(self.W, self.max_episode_length, self.obs_dim, dtype=torch.float32)
        self._ring_state_len = _full(0, self.W, dtype=torch.long)

        # Per-interval / running counters.
        self._added_per_interval: list[int] = [0] * self.num_agents
        self._added_total: list[int] = [0] * self.num_agents
        self._adds_since_cluster: list[int] = [0] * self.num_agents
        self._adds_since_projection: list[int] = [0] * self.num_agents

        # Cached clustering labels (per agent) — refreshed on threshold; held
        # between recomputes so we can keep emitting last-known cluster stats.
        self._cluster_labels: list[np.ndarray | None] = [None] * self.num_agents

        # Snapshot directory for §5.5 projection PNG + §-disk dumps.
        self._snapshot_dir = self.experiment_dir / cfg.snapshot_dir
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Tracks the last env_step at which we dumped the buffer to disk per agent.
        self._last_disk_dump_step: list[int] = [-1] * self.num_agents

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------
    def p_hat_succ(self, agent_i: int) -> float:
        """Rolling success rate over the agent's recent ``W`` trajectories.

        Returns 0.0 when no trajectories have been committed yet, which keeps
        the curriculum dormant (the rescue-init wrapper gates on ``>= rho_min``).
        """
        n = self._ring_filled[agent_i]
        if n == 0:
            return 0.0
        return float(self._ring_success[agent_i][:n].float().mean().item())

    def bump_added(self, agent_i: int) -> None:
        self._added_per_interval[agent_i] += 1
        self._added_total[agent_i] += 1
        self._adds_since_cluster[agent_i] += 1
        self._adds_since_projection[agent_i] += 1

    # ------------------------------------------------------------------
    # Trajectory commit
    # ------------------------------------------------------------------
    def commit_trajectory(
        self,
        *,
        agent_i: int,
        success: bool,
        ret: float,
        length: int,
        time_to_success: int | None,
        init_flag: bool,
        slot_idx: int,
        action_entropy_first_k: float,
        states: torch.Tensor,
    ) -> None:
        """Append one trajectory summary into the agent's rolling window.

        ``states`` must be ``(length, obs_dim)`` of raw (un-normalized) observations.
        """
        if agent_i < 0 or agent_i >= self.num_agents:
            raise IndexError(f"commit_trajectory: agent_i {agent_i} out of range")
        if states.ndim != 2 or states.shape[1] != self.obs_dim:
            raise ValueError(
                f"commit_trajectory: states must be (length, {self.obs_dim}); got {tuple(states.shape)}"
            )
        n = min(int(length), self.max_episode_length)
        if int(states.shape[0]) < n:
            raise ValueError(
                f"commit_trajectory: states.shape[0] ({states.shape[0]}) < length ({n})"
            )

        h = self._ring_head[agent_i]
        self._ring_success[agent_i][h] = bool(success)
        self._ring_return[agent_i][h] = float(ret)
        self._ring_length[agent_i][h] = int(length)
        self._ring_time_to_success[agent_i][h] = int(time_to_success) if (success and time_to_success is not None) else -1
        self._ring_init_flag[agent_i][h] = bool(init_flag)
        self._ring_slot_idx[agent_i][h] = int(slot_idx) if init_flag else -1
        self._ring_action_entropy_k[agent_i][h] = float(action_entropy_first_k)
        self._ring_states[agent_i][h, :n] = states[:n].to(self.device, dtype=torch.float32)
        # Zero the trailing rows so stale data from a prior trajectory at this
        # ring slot can't leak into Section-6 distance computations.
        if n < self.max_episode_length:
            self._ring_states[agent_i][h, n:] = 0.0
        self._ring_state_len[agent_i][h] = n

        # Update slot-side outcome bookkeeping if this trajectory was a rescue init.
        if init_flag and 0 <= int(slot_idx):
            self.rescue_buffers[agent_i].record_outcome(
                int(slot_idx),
                success=bool(success),
                length=int(length),
                time_to_success=int(time_to_success) if (success and time_to_success is not None) else None,
                action_entropy=float(action_entropy_first_k),
            )

        # Ring advance.
        self._ring_head[agent_i] = (h + 1) % self.W
        if self._ring_filled[agent_i] < self.W:
            self._ring_filled[agent_i] += 1

    # ------------------------------------------------------------------
    # Flush — write Section 1-6 metrics to TB
    # ------------------------------------------------------------------
    def flush_per_agent(
        self,
        per_agent_tracking: list[dict],
        per_agent_writers: list,
        timestep: int,
    ) -> None:
        """Compute and emit all rescue metrics for every agent."""
        for i in range(self.num_agents):
            self._flush_one_agent(i, per_agent_tracking, per_agent_writers, timestep)

    def _flush_one_agent(
        self,
        i: int,
        per_agent_tracking: list[dict],
        per_agent_writers: list,
        timestep: int,
    ) -> None:
        buf = self.rescue_buffers[i]
        tracking = per_agent_tracking[i]
        writer = per_agent_writers[i]

        # ----- §1 Composition -----
        tracking["rescue_buffer/size"].append(float(len(buf)))
        tracking["rescue_buffer/added_per_interval"].append(float(self._added_per_interval[i]))
        tracking["rescue_buffer/added_total"].append(float(self._added_total[i]))
        self._added_per_interval[i] = 0

        # ----- §4 Health -----
        filled_mask = buf._filled
        n_filled = int(filled_mask.sum().item())
        if n_filled > 0:
            dead_mask = (
                filled_mask
                & (buf.init_attempts >= buf.dead_point_min_attempts)
                & (buf.init_successes == 0)
            )
            dead_count = int(dead_mask.sum().item())
            tracking["rescue_buffer/dead_count"].append(float(dead_count))
            tracking["rescue_buffer/dead_fraction"].append(float(dead_count) / n_filled)
            src_steps = buf.source_trajectory_step[filled_mask].float()
            tracking["rescue_buffer/source_step_mean"].append(float(src_steps.mean().item()))
            tracking["rescue_buffer/source_step_std"].append(
                float(src_steps.std(unbiased=False).item()) if src_steps.numel() > 1 else 0.0
            )
            writer.add_histogram(
                "rescue_buffer/source_step_hist", src_steps.cpu().numpy(), global_step=timestep
            )
        else:
            tracking["rescue_buffer/dead_count"].append(0.0)
            tracking["rescue_buffer/dead_fraction"].append(0.0)

        # ----- §2 Performance split by init type -----
        n_w = self._ring_filled[i]
        if n_w > 0:
            succ = self._ring_success[i][:n_w]
            ret = self._ring_return[i][:n_w]
            ln = self._ring_length[i][:n_w].float()
            tts = self._ring_time_to_success[i][:n_w]
            init = self._ring_init_flag[i][:n_w]
            ae_k = self._ring_action_entropy_k[i][:n_w]

            rescue_mask = init
            default_mask = ~init

            def _mean_or_skip(values: torch.Tensor, mask: torch.Tensor, tag: str) -> None:
                sel = values[mask]
                if sel.numel() == 0:
                    return
                tracking[tag].append(float(sel.float().mean().item()))

            _mean_or_skip(succ.float(), rescue_mask, "rescue_init_performance/success_rate")
            _mean_or_skip(succ.float(), default_mask, "default_init_performance/success_rate")
            _mean_or_skip(ret, rescue_mask, "rescue_init_performance/return")
            _mean_or_skip(ret, default_mask, "default_init_performance/return")
            _mean_or_skip(ln, rescue_mask, "rescue_init_performance/episode_length")
            _mean_or_skip(ln, default_mask, "default_init_performance/episode_length")

            # time_to_success: only over successful trajectories.
            for grp_mask, tag in (
                (rescue_mask, "rescue_init_performance/time_to_success"),
                (default_mask, "default_init_performance/time_to_success"),
            ):
                sel_mask = grp_mask & succ
                if sel_mask.any():
                    tracking[tag].append(float(tts[sel_mask].float().mean().item()))

            _mean_or_skip(ae_k, rescue_mask, "rescue_init_performance/action_entropy")

        # ----- §3 Predictor quality on rescue states -----
        if n_filled > 0:
            # Normalize stored raw obs through the (per-agent) running scaler,
            # then query the actor for current P. observation_preprocessor is a
            # PerAgentPreprocessorWrapper — it shards by agent partition, so we
            # need to query per-agent obs via the success_prob_query closure
            # which accepts (obs, agent_i).
            buf_obs = buf.obs[filled_mask]  # (n_filled, obs_dim)
            with torch.no_grad():
                current_P = self.success_prob_query(buf_obs, i).detach().to("cpu").numpy().astype(np.float32)
            init_attempts_np = buf.init_attempts[filled_mask].cpu().numpy()
            init_successes_np = buf.init_successes[filled_mask].cpu().numpy()
            add_p_np = buf.add_p_value[filled_mask].cpu().numpy()

            # 3.3 P-drift always emit when buffer non-empty.
            drift = current_P - add_p_np
            tracking["predictor/p_drift_mean"].append(float(np.mean(drift)))
            tracking["predictor/p_drift_std"].append(float(np.std(drift)) if drift.size > 1 else 0.0)
            writer.add_histogram("predictor/p_drift_hist", drift, global_step=timestep)

            # 3.1 ECE: include only points with init_attempts >= 1; per-attempt label.
            has_attempts = init_attempts_np >= 1
            if has_attempts.any():
                # Expand per-point P over attempts; per-attempt label = success-rate of that point
                # (each attempt is i.i.d. Bernoulli; using the per-point empirical success rate
                # as the binary label set gives the same ECE bin counts as expanding to attempts).
                # To be faithful to the spec's "per-attempt outcomes as labels", we expand.
                Ps = []
                ys = []
                for idx_in_filled, ok in enumerate(has_attempts):
                    if not ok:
                        continue
                    n_a = int(init_attempts_np[idx_in_filled])
                    n_s = int(init_successes_np[idx_in_filled])
                    Ps.append(np.full(n_a, current_P[idx_in_filled], dtype=np.float32))
                    ys.append(np.concatenate([np.ones(n_s, dtype=np.float32), np.zeros(n_a - n_s, dtype=np.float32)]))
                P_attempts = np.concatenate(Ps) if Ps else np.zeros(0, dtype=np.float32)
                y_attempts = np.concatenate(ys) if ys else np.zeros(0, dtype=np.float32)
                if P_attempts.size > 0:
                    ece_val, _ = _ece(P_attempts, y_attempts, int(self.cfg.predictor_ece_n_bins))
                    tracking["predictor/ece_on_rescue_states"].append(float(ece_val))

            # 3.2 ROC-AUC: per-point label = ever-succeeded across its attempts.
            has_any = init_attempts_np >= 1
            if int(has_any.sum()) >= 10:
                y_point = (init_successes_np[has_any] > 0).astype(np.float32)
                P_point = current_P[has_any].astype(np.float32)
                # Spec: skip if fewer than 10 with each class.
                if (y_point > 0.5).sum() >= 10 and (y_point < 0.5).sum() >= 10:
                    auc = _roc_auc(P_point, y_point)
                    if auc is not None:
                        tracking["predictor/auc_on_rescue_states"].append(float(auc))

        # ----- §5 Diversity + §6 Visitation -----
        if n_filled >= 2:
            # Normalize buffer obs for distance computations. The
            # observation_preprocessor is a PerAgentPreprocessorWrapper; we need
            # to encode just this agent's slice. The closure exposed via
            # success_prob_query already operates per-agent; mirror that here
            # with a direct preprocessor call on the agent partition.
            with torch.no_grad():
                buf_obs = buf.obs[filled_mask]
                # PerAgentPreprocessorWrapper's __call__ expects (B, obs_dim)
                # batched across agents in agent-partition layout. For metric
                # purposes we just normalize *this* agent's rows through
                # *this* agent's scaler — bypass the wrapper by reaching its
                # per-agent member if available; otherwise apply the wrapper
                # naively. We require the wrapper to expose .preprocessors
                # (list per agent) or .modules (PerAgent dict). Fail loud.
                scaler = self._per_agent_scaler(i)
                buf_obs_norm = scaler(buf_obs).detach()  # (n_filled, obs_dim)

            # §5 — pairwise distances within buffer
            with torch.no_grad():
                pdist = torch.cdist(buf_obs_norm, buf_obs_norm)
                # Mask the diagonal (zero self-distances) for the percentile calc.
                eye = torch.eye(n_filled, device=self.device, dtype=torch.bool)
                off_diag = pdist[~eye]
            self._maybe_recompute_clustering(i, buf_obs_norm.cpu().numpy(), off_diag, tracking, writer, timestep)

            # §6 — visitation: default-init trajectories from the rolling window
            self._compute_visitation(i, buf_obs_norm, off_diag, tracking, writer, timestep, scaler)

            # §5.5 — 2-D projection (PNG) — gated by recompute threshold
            self._maybe_recompute_projection(i, buf_obs_norm.cpu().numpy(), tracking, writer, timestep)

        # ----- Disk snapshot (every buffer_snapshot_interval env steps) -----
        last = self._last_disk_dump_step[i]
        if timestep - max(last, 0) >= int(self.cfg.buffer_snapshot_interval) and n_filled > 0:
            self._dump_to_disk(i, timestep)
            self._last_disk_dump_step[i] = timestep

    # ------------------------------------------------------------------
    # Per-agent scaler accessor (fail-loud)
    # ------------------------------------------------------------------
    def _per_agent_scaler(self, agent_i: int) -> Callable[[torch.Tensor], torch.Tensor]:
        wrapper = self.observation_preprocessor
        # PerAgentPreprocessorWrapper exposes a .preprocessor_list per agent.
        preps = getattr(wrapper, "preprocessor_list", None)
        if preps is None or len(preps) <= agent_i:
            raise RuntimeError(
                "RescueMetricsTracker requires observation_preprocessor to expose a "
                "per-agent .preprocessor_list of length >= num_agents; got "
                f"{type(wrapper).__name__} (preps={preps!r})."
            )
        prep = preps[agent_i]
        if prep is None:
            # No-op preprocessor — return identity. SAC's preprocessor slot can be
            # legitimately empty (see PerAgentPreprocessorWrapper docstring); distance
            # metrics simply work on raw obs in that case.
            return lambda x: x
        return prep

    # ------------------------------------------------------------------
    # §5 — clustering
    # ------------------------------------------------------------------
    def _maybe_recompute_clustering(
        self,
        i: int,
        buf_obs_np: np.ndarray,
        off_diag_distances: torch.Tensor,
        tracking: dict,
        writer,
        timestep: int,
    ) -> None:
        # Gate
        if self._adds_since_cluster[i] < int(self.cfg.clustering_recompute_threshold):
            return
        if buf_obs_np.shape[0] < int(self.cfg.min_buffer_size_for_clustering):
            return

        from sklearn.cluster import DBSCAN

        # Pairwise-distance diagnostics — log alongside the eps actually used by
        # DBSCAN so it's possible to tell whether `auto-eps = 5th percentile` is
        # straddling a bridge between visually distinct blobs (in which case
        # n_clusters collapses to 1). Cheap: off_diag_distances is already on
        # GPU from the caller.
        if off_diag_distances.numel() > 0:
            q = torch.quantile(
                off_diag_distances,
                torch.tensor([0.01, 0.05, 0.25, 0.50], device=off_diag_distances.device),
            ).tolist()
            tracking["diversity/pairwise_p1"].append(float(q[0]))
            tracking["diversity/pairwise_p5"].append(float(q[1]))
            tracking["diversity/pairwise_p25"].append(float(q[2]))
            tracking["diversity/pairwise_p50"].append(float(q[3]))

        eps_cfg = self.cfg.dbscan_eps
        if eps_cfg is None:
            # 5th-percentile of pairwise distances.
            eps_val = float(torch.quantile(off_diag_distances, 0.05).item())
        else:
            eps_val = float(eps_cfg)
        # Edge case: zero eps would collapse everything to noise.
        eps_val = max(eps_val, 1e-8)
        tracking["diversity/dbscan_eps_used"].append(float(eps_val))
        labels = DBSCAN(eps=eps_val, min_samples=int(self.cfg.dbscan_min_samples)).fit_predict(buf_obs_np)
        self._cluster_labels[i] = labels

        non_noise = labels[labels != -1]
        n_clusters = int(np.unique(non_noise).size) if non_noise.size else 0
        tracking["diversity/n_clusters"].append(float(n_clusters))

        # Silhouette: at least 2 clusters, non-noise points only.
        if n_clusters >= 2 and non_noise.size >= 2:
            from sklearn.metrics import silhouette_score

            mask = labels != -1
            try:
                sil = float(silhouette_score(buf_obs_np[mask], labels[mask]))
                tracking["diversity/silhouette_score"].append(sil)
            except Exception:
                pass

        # Cluster entropy.
        if non_noise.size > 0:
            _, counts = np.unique(non_noise, return_counts=True)
            p = counts.astype(np.float64) / counts.sum()
            entropy = float(-(p * np.log(np.clip(p, 1e-12, None))).sum())
            tracking["diversity/cluster_entropy"].append(entropy)

        noise_frac = float((labels == -1).sum()) / float(labels.size)
        tracking["diversity/noise_fraction"].append(noise_frac)
        self._adds_since_cluster[i] = 0

    # ------------------------------------------------------------------
    # §5.5 — projection PNG
    # ------------------------------------------------------------------
    def _maybe_recompute_projection(
        self,
        i: int,
        buf_obs_np: np.ndarray,
        tracking: dict,
        writer,
        timestep: int,
    ) -> None:
        if self._adds_since_projection[i] < int(self.cfg.projection_recompute_threshold):
            return
        if buf_obs_np.shape[0] < max(2, int(self.cfg.projection_n_neighbors) + 1):
            return

        method = self.cfg.projection_method.lower()
        if method == "umap":
            try:
                import umap  # type: ignore

                reducer = umap.UMAP(
                    n_neighbors=int(self.cfg.projection_n_neighbors),
                    min_dist=float(self.cfg.projection_min_dist),
                    n_components=2,
                )
                proj = reducer.fit_transform(buf_obs_np)
            except Exception:
                # Fall back to PCA if umap is not installed.
                from sklearn.decomposition import PCA

                proj = PCA(n_components=2).fit_transform(buf_obs_np)
        elif method == "pca":
            from sklearn.decomposition import PCA

            proj = PCA(n_components=2).fit_transform(buf_obs_np)
        else:
            raise ValueError(f"rescue_buffer_cfg.projection_method must be 'umap' or 'pca', got {method!r}")

        # Render PNG, save to disk, log to TB at global_step=0.
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.figure as mpl_figure

        # Cluster labels are positional in the buffer; any eviction or addition
        # since the last clustering invalidates the cached labels w.r.t. the
        # current buffer rows. To keep the projection PNG semantically meaningful
        # — including a distinct color for DBSCAN noise — recompute clustering
        # inline here whenever the cached labels don't line up with the
        # current buffer size. DBSCAN on <few-thousand points is cheap.
        cached = self._cluster_labels[i]
        if cached is None or cached.shape[0] != buf_obs_np.shape[0]:
            from sklearn.cluster import DBSCAN
            # Mirror clustering eps logic from _maybe_recompute_clustering:
            # null cfg ⇒ 5th-percentile pairwise distance.
            if self.cfg.dbscan_eps is None:
                with torch.no_grad():
                    bx = torch.as_tensor(buf_obs_np, device=self.device)
                    pd = torch.cdist(bx, bx)
                    eye = torch.eye(bx.shape[0], device=self.device, dtype=torch.bool)
                    eps_val = float(torch.quantile(pd[~eye], 0.05).item())
            else:
                eps_val = float(self.cfg.dbscan_eps)
            eps_val = max(eps_val, 1e-8)
            labels = DBSCAN(eps=eps_val, min_samples=int(self.cfg.dbscan_min_samples)).fit_predict(buf_obs_np)
            self._cluster_labels[i] = labels
        else:
            labels = cached

        fig = mpl_figure.Figure(figsize=(6, 6), dpi=100)
        ax = fig.add_subplot(111)
        # Plot noise (-1) and clusters separately so noise gets its own visually
        # distinct treatment (gray, smaller, semi-transparent) and clusters get
        # the tab10 categorical palette.
        noise_mask = labels < 0
        if noise_mask.any():
            ax.scatter(
                proj[noise_mask, 0], proj[noise_mask, 1],
                c="lightgray", s=8, alpha=0.5, label="noise",
            )
        cluster_mask = ~noise_mask
        if cluster_mask.any():
            ax.scatter(
                proj[cluster_mask, 0], proj[cluster_mask, 1],
                c=labels[cluster_mask], cmap="tab10", s=10, vmin=0, vmax=9,
                label="clusters",
            )
        ax.legend(loc="best", fontsize=8)
        ax.set_title(f"Rescue buffer projection (agent {i}, n={proj.shape[0]})")
        ax.set_xlabel("dim 0")
        ax.set_ylabel("dim 1")
        fig.tight_layout()
        png_path = self._snapshot_dir / f"projection_agent{i}.png"
        fig.savefig(png_path)

        # Encode for TB
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        import matplotlib.image as mpl_image

        arr = mpl_image.imread(buf)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        chw = np.transpose(arr, (2, 0, 1))
        writer.add_image(f"diversity/projection_2d", chw, global_step=0, dataformats="CHW")
        self._adds_since_projection[i] = 0

    # ------------------------------------------------------------------
    # §6 — visitation
    # ------------------------------------------------------------------
    def _compute_visitation(
        self,
        i: int,
        buf_obs_norm: torch.Tensor,
        off_diag_distances: torch.Tensor,
        tracking: dict,
        writer,
        timestep: int,
        scaler: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        n_w = self._ring_filled[i]
        if n_w == 0:
            return
        init = self._ring_init_flag[i][:n_w]
        # Default-init rows only (Section 6 explicitly).
        default_idx = (~init).nonzero(as_tuple=False).flatten()
        if default_idx.numel() == 0:
            return
        # Subsample.
        frac = float(self.cfg.nn_query_subsample_rollout)
        if frac < 1.0:
            keep = max(1, int(default_idx.numel() * frac))
            perm = torch.randperm(default_idx.numel(), device=self.device)[:keep]
            default_idx = default_idx[perm]

        # For each selected trajectory, compute min distance from any of its
        # states to the buffer obs (normalized).
        min_dists: list[float] = []
        for j in default_idx.tolist():
            ln = int(self._ring_state_len[i][j].item())
            if ln <= 0:
                continue
            traj_states = self._ring_states[i][j, :ln]
            with torch.no_grad():
                traj_norm = scaler(traj_states).detach()
                d = torch.cdist(traj_norm, buf_obs_norm)  # (ln, n_filled)
                m = float(d.min().item())
            min_dists.append(m)
        if not min_dists:
            return
        min_dists_arr = np.array(min_dists, dtype=np.float32)

        # 6.1 — visitation fraction at percentile epsilons.
        off_diag_np = off_diag_distances.cpu().numpy() if off_diag_distances.numel() else np.zeros(0)
        for p in self.cfg.nn_distance_percentiles:
            if off_diag_np.size == 0:
                continue
            eps = float(np.percentile(off_diag_np, p))
            tracking[f"visitation/eps_p{p}"].append(eps)
            frac_below = float((min_dists_arr <= eps).mean())
            tracking[f"visitation/fraction_eps_p{p}"].append(frac_below)

        # 6.2 — per-trajectory min-distance histogram + percentiles.
        writer.add_histogram("visitation/min_distance_hist", min_dists_arr, global_step=timestep)
        tracking["visitation/min_distance_p25"].append(float(np.percentile(min_dists_arr, 25)))
        tracking["visitation/min_distance_p50"].append(float(np.percentile(min_dists_arr, 50)))
        tracking["visitation/min_distance_p75"].append(float(np.percentile(min_dists_arr, 75)))

    # ------------------------------------------------------------------
    # Disk snapshot
    # ------------------------------------------------------------------
    def _dump_to_disk(self, i: int, timestep: int) -> None:
        buf = self.rescue_buffers[i]
        step_dir = self._snapshot_dir / f"step_{timestep}_agent{i}"
        step_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "sim_state": buf.sim_state.detach().cpu(),
            "obs": buf.obs.detach().cpu(),
            "add_step": buf.add_step.detach().cpu(),
            "add_p_value": buf.add_p_value.detach().cpu(),
            "source_trajectory_step": buf.source_trajectory_step.detach().cpu(),
            "init_attempts": buf.init_attempts.detach().cpu(),
            "init_successes": buf.init_successes.detach().cpu(),
            "filled": buf._filled.detach().cpu(),
            "insertion_order": buf._insertion_order.detach().cpu(),
        }
        torch.save(payload, step_dir / "buffer.pt")
        meta = {
            "timestep": int(timestep),
            "agent": int(i),
            "capacity": int(buf.capacity),
            "snapshot_dim": int(buf.snapshot_dim),
            "obs_dim": int(buf.obs_dim),
            "n_filled": int(len(buf)),
            "init_episode_lengths": buf.init_episode_lengths,
            "init_times_to_success": buf.init_times_to_success,
            "init_action_entropies": buf.init_action_entropies,
        }
        with open(step_dir / "meta.json", "w") as f:
            json.dump(meta, f)
