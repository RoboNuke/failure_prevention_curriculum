"""Rescue-point buffer + metrics configuration.

Implements the hyperparameters of Algorithm 1 (Rescue Recovery by Detection of
Trajectory Divergence) — see ``/home/hunter/Downloads/Rescue_from_Failure (1).pdf``
— together with the metric-suite knobs from
``/home/hunter/Downloads/rescue_buffer_metrics_spec.md``. One combined dataclass
because the algorithm and its metrics share buffer state (dead-point thresholds,
buffer-size gates for clustering, etc.) and the YAML registry is flat.

Strict-load convention: required algorithm fields have no defaults; missing them
in YAML raises. Metric-side knobs default to the spec's recommended values so a
minimal opt-in YAML need only set the algorithm half.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(kw_only=True)
class RescueBufferCfg:
    """Rescue-state buffer + metric publication knobs."""

    enabled: bool = False
    """Master toggle. When False, runner skips buffer/wrapper/tracker
    construction entirely and SAC behaves identically to its non-rescue path."""

    # ---- Algorithm 1 hyperparameters (no defaults — fail loud) ----
    tau: float
    """``f_phi(s_t) >= tau`` ⇒ ``s_t`` is a rescue-point candidate.
    Backward scan returns the latest qualifying state."""

    delta: float
    """Failure detector: only consider adding rescue points from trajectories
    whose terminal success probability ``f_phi(s_T) <= delta``."""

    alpha: float
    """Per-done-env Bernoulli prob of overwriting a natural reset with a sample
    drawn uniformly from the rescue buffer (curriculum injection rate)."""

    rho_min: float
    """Minimum rolling success rate required to (a) add new rescue points and
    (b) use the buffer for initialization. Keeps the curriculum dormant during
    warm-up before the predictor is meaningful."""

    window_size: int
    """``W`` — number of recent finished trajectories per agent used to compute
    the rolling success rate ``p_hat_succ`` and the Section-2 init-split metrics."""

    max_buffer_size: int
    """Per-agent capacity ``|B_c|``. Eviction is dead-first (``init_attempts >=
    dead_point_min_attempts`` AND ``init_successes == 0``), then FIFO by insertion
    order. Preallocated as fixed-shape GPU tensors."""

    # ---- Algorithm extras ----
    dead_point_min_attempts: int = 3
    """Min ``init_attempts`` before a zero-success point is flagged as dead."""

    action_entropy_first_k_steps: int = 10
    """``K`` — number of steps from each rollout's start over which mean action
    entropy is averaged (Section 2.5)."""

    # ---- Metric publication ----
    metric_compute_interval: int = 1000
    """Env-steps between metric computations. Must equal sac_cfg write_interval
    (raises at tracker ctor otherwise) so the scalar buckets emit in lockstep."""

    buffer_snapshot_interval: int = 50_000
    """Env-steps between full buffer dumps to disk."""

    snapshot_dir: str = "rescue_snapshots"
    """Subdirectory under the experiment directory for buffer + projection dumps."""

    # ---- Section 5 — diversity / clustering ----
    nn_query_subsample_rollout: float = 1.0
    """Fraction of rollout trajectories to query for Section-6 visitation."""

    nn_distance_percentiles: list[int] = dataclasses.field(
        default_factory=lambda: [5, 25, 50]
    )
    """Percentiles of within-buffer pairwise distance used as Section-6 epsilons."""

    dbscan_eps: float | None = None
    """If null, auto-set to the 5th-percentile pairwise distance at clustering time."""

    dbscan_min_samples: int = 5
    clustering_recompute_threshold: int = 25
    min_buffer_size_for_clustering: int = 20

    predictor_ece_n_bins: int = 10
    """ECE bin count for Section 3.1."""

    projection_method: str = "umap"
    """``"umap"`` or ``"pca"`` for the Section 5.5 2-D projection."""

    projection_recompute_threshold: int = 50
    projection_n_neighbors: int = 15
    projection_min_dist: float = 0.1
