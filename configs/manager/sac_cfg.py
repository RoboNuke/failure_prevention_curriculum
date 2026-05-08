from __future__ import annotations

from typing import Callable

import dataclasses

from skrl.agents.torch import AgentCfg


@dataclasses.dataclass(kw_only=True)
class RecorderCfg:
    """Configures the optional ``RecordingWrapper`` that produces 3x4-grid GIFs and
    TB videos of vectorized rollouts during training. See ``wrappers/recording.py``.

    The recorder injects a single ``TiledCameraCfg`` into the env scene before
    ``gym.make()`` (so the camera prim is created per-env). When ``enabled`` is
    False, the wrapper is not constructed and no camera is added — zero overhead.
    """

    enabled: bool = False
    """Master switch. When False, no camera is added and the wrapper is not built."""

    record_every_k_resets: int = 20
    """Open a recording session every K-th observed *global* reset (a step where
    ``(terminated | truncated).all()`` is True). Per-env async resets in between
    do not count. The first session opens at the first global reset."""

    width: int = 240
    """Per-tile camera width in pixels."""

    height: int = 180
    """Per-tile camera height in pixels."""

    camera_pos: tuple[float, float, float] = (1.0, 0.0, 0.35)
    """Camera position offset relative to each env's local origin. Default
    matches the RoboNuke/Continuous_Force_RL eval setup."""

    camera_quat: tuple[float, float, float, float] = (-0.3535534, 0.6123724, 0.6123724, -0.3535534)
    """Camera orientation as a (w, x, y, z) quaternion under the ROS
    convention. Default matches RoboNuke/Continuous_Force_RL."""

    focal_length: float = 24.0
    """PinholeCamera focal length (mm-equivalent). RoboNuke default."""

    focus_distance: float = 0.05
    """PinholeCamera focus distance. RoboNuke default."""

    horizontal_aperture: float = 20.955
    """PinholeCamera horizontal aperture (mm). RoboNuke default."""

    clipping_range: tuple[float, float] = (0.1, 20.0)
    """PinholeCamera near/far clip planes (m). RoboNuke default."""

    fps: int = 30
    """Playback rate for the saved GIF and TB video."""

    output_subdir: str = "videos"
    """Subdirectory under the SAC experiment dir where GIFs are written."""


@dataclasses.dataclass(kw_only=True)
class SAC_CFG(AgentCfg):
    """Configuration for the SAC agent."""

    gradient_steps: int = 1
    """Number of gradient steps to perform for each update."""

    batch_size: int = 64
    """Batch size **per agent** for sampling transitions from memory during training.
    With ``num_agents`` block-parallel agents, the total memory sample size is
    ``batch_size * num_agents`` (each agent draws ``batch_size`` transitions from
    its own env partition)."""

    discount_factor: float = 0.99
    """Parameter that balances the importance of future rewards (close to 1.0) versus immediate rewards (close to 0.0).

    Range: ``[0.0, 1.0]``.
    """

    polyak: float = 0.005
    """Parameter to control the update of the target networks by polyak averaging.

    Range: ``[0.0, 1.0]``. See :py:meth:`~skrl.models.torch.base.Model.update_parameters` for more details.
    """

    actor_lr: float = 1e-3
    """Learning rate for the actor (policy) optimizer."""

    critic_lr: float = 1e-3
    """Learning rate for the critic (Q-network) optimizer."""

    entropy_lr: float = 1e-3
    """Learning rate for the entropy coefficient (log_alpha) optimizer."""

    weight_decay: float = 0.0
    """Decoupled weight decay applied by AdamW to the actor and critic optimizers.
    The entropy optimizer is intentionally excluded — pulling log_alpha toward 0
    has no principled meaning. Set to 0.0 to disable (equivalent to Adam)."""

    learning_rate_scheduler: type | tuple[type | None, type | None, type | None] | None = None
    """Learning rate scheduler class for the actor and critic networks, and entropy coefficient.

    See :ref:`learning_rate_schedulers` for more details.

    * If a class is provided, the same learning rate scheduler will be used for the networks/coefficient.
    * If a tuple is provided, its elements will be used for each network/coefficient in order.
    """

    learning_rate_scheduler_kwargs: dict | tuple[dict, dict, dict] = dataclasses.field(default_factory=dict)
    """Keyword arguments for the learning rate scheduler's constructor.

    See :ref:`learning_rate_schedulers` for more details.

    .. warning::

        The ``optimizer`` argument is automatically passed to the learning rate scheduler's constructor.
        Therefore, it must not be provided in the keyword arguments.

    * If a dictionary is provided, the same keyword arguments will be used for the networks/coefficient.
    * If a tuple is provided, its elements will be used for each network/coefficient in order.
    """

    observation_preprocessor: type | None = None
    """Preprocessor class to process the environment's observations.

    See :ref:`preprocessors` for more details.
    """

    observation_preprocessor_kwargs: dict = dataclasses.field(default_factory=dict)
    """Keyword arguments for the observation preprocessor's constructor.

    See :ref:`preprocessors` for more details.
    """

    random_timesteps: int = 0
    """Number of random exploration (sampling random actions) steps to perform before sampling actions from the policy."""

    learning_starts: int = 0
    """Number of steps to perform before calling the algorithm update function."""

    grad_norm_clip: float = 0
    """Clipping coefficient for the gradients by their global norm.

    If less than or equal to 0, the gradients will not be clipped.
    """

    learn_entropy: bool = True
    """Whether to learn the entropy coefficient."""

    initial_entropy_value: float = 0.2
    """Initial value for the entropy coefficient."""

    target_entropy: float | None = None
    """Target value for computing the entropy loss."""

    rewards_shaper: Callable | None = None
    """Rewards shaping function. YAML can't carry callables — set
    ``rewards_shaper_scale`` instead and the runner builds a multiplicative
    lambda. Direct assignment is still supported for programmatic configs."""

    rewards_shaper_scale: float | None = None
    """Scalar reward multiplier. When set, the runner installs
    ``rewards_shaper = lambda r, *a, **k: r * scale`` before constructing SAC.
    Mirrors skrl's runner convention; null disables shaping."""

    gripper_action_idx: int | None = None
    """Index of the binary gripper action dim, if any. When set, SAC logs
    ``Gripper / open rate`` (fraction of `actions[..., idx] >= 0`) plus mean and
    std of the raw value, so a stuck/locked gripper is visible in tensorboard.
    For Lift Franka, set to 7 (act_dim=8: arm 0-6, gripper 7). Diagnostic only;
    works regardless of whether the gripper dim is sampled from a Gaussian or
    a Bernoulli (see ``model_cfg.actor.bernoulli_action_dims``)."""

    mixed_precision: bool = False
    """Whether to enable automatic mixed precision for higher performance."""

    predict_success: bool = True
    """Whether the policy emits a success-probability head trained with BCE against
    per-trajectory binary labels. When ``True``, the runner uses
    ``TrajectoryBufferedMemory`` so transitions are staged per env and only
    pushed to the main replay buffer (with a frozen success label) on episode end."""

    success_td_weight: float = 1.0
    """Scalar weight on the success-prediction TD loss added to the SAC policy
    loss when ``predict_success`` is True. The loss is BCE between the actor's
    sigmoid'd ``success_logit`` and a discounted-success target computed via
    1-step bootstrap with terminal anchors at first-success (target=1) and
    failed-trajectory end (target=0); see ``success_td_discount``."""

    success_td_discount: float = 0.99
    """Discount factor γ for the success-prediction TD target. The target at
    a non-terminal step is ``γ · sigmoid(success_logit(s_{t+1})).detach()``.
    For an L-step episode, γ ≈ 1 − 1/L makes the head represent "probability
    of success within roughly L steps" — so 0.99 fits 100-step horizons,
    0.995 fits ~200-step, etc. Independent from ``discount_factor`` (which
    is the SAC RL discount); they encode different time horizons."""

    success_info_key: str = "is_success"
    """Key in the env's per-step ``infos`` dict whose value is a per-env boolean
    tensor that flags 'success holds *right now*' (instantaneous, NOT latching).
    A trajectory is labeled success only if ``is_success`` was True for at least
    ``success_streak_len`` consecutive steps somewhere in the trajectory.
    Required when ``predict_success`` is True."""

    success_use_streak: bool = True
    """Selects how a trajectory is labeled positive for success-head training and
    diagnostics:
      * ``True`` (default): a trajectory is positive iff ``info[success_info_key]``
        was True for at least ``success_streak_len`` consecutive steps somewhere
        in the trajectory. Each streak step gets TD target=1; pre-streak
        bootstraps with γ; post-streak masked out. (See ``success_streak_len``.)
      * ``False``: a trajectory is positive iff ``info[success_info_key]`` is
        True at the *terminal* step (the final staged step before reset). The
        terminal step gets TD target=1 (success-terminal) or 0 (failure-
        terminal); all earlier steps bootstrap from V(s_{t+1}). No streak
        scanning, no post-success masking — every transition contributes to
        the loss. ``success_streak_len`` is ignored in this mode.
    Both modes use the same TD/loss machinery; only the target-stamping
    function (in ``TrajectoryBufferedMemory.finalize_trajectory``) and the
    matching ``PredictionQualityTracker`` outcome-labelling change."""

    success_streak_len: int = 1
    """Number of consecutive steps on which ``info[success_info_key]`` must be
    True for a trajectory to count as a success. The first qualifying window
    [t_start, t_end] (with t_end = t_start + success_streak_len − 1) defines the
    success-anchor block:
      * Each of the streak's `success_streak_len` steps gets TD target = 1
        (terminal anchor, no bootstrap).
      * Pre-streak steps (0..t_start−1) bootstrap normally:
          target = γ · V(s_{t+1})
        so step t_start−1 has target ≈ γ (since V(s_{t_start}) is anchored to 1).
      * Post-streak steps (t_end+1..n−1) are masked out of the loss (their label
        is undefined by design — the agent may slip out, that's fine).
    `success_streak_len=1` reproduces the old "first instantaneous success step
    is the anchor" behavior. Larger values count touch-and-slip as failure: the
    peg must remain seated for N steps before the trajectory is positive. Both
    the training target AND the prediction-quality metrics (AUC / per-class BCE
    / heatmaps / Episode-Success-rate) use this same streak criterion, so they
    measure the same thing."""

    success_train_min_successes: int = 0
    """Minimum number of successful trajectories (label=1, OR'd over the
    trajectory's per-step `is_success` flag) that must have been observed in
    the replay buffer before the success-prediction TD loss is added to the
    policy loss. While the count is below this threshold, the success head
    receives no gradient updates — preventing the head from saturating to
    `logit ≈ −∞` on long stretches of all-failure trajectories early in
    training (which is otherwise amplified by `success_td_weight`). 0 means
    no gating (default behavior). The cumulative count is published to TB as
    `Success Prediction Quality / cum success trajs` so the gate state is
    visible. Counted globally across all envs / agents."""

    success_heatmap_step_bins: int = 30
    """Number of vertical bins on the success/failure prediction-quality
    heatmaps. Each bin aggregates ``mean P_head`` across all transitions whose
    episode-relative step falls in that bin's range. Bins span
    ``[0, max_episode_length)`` uniformly, so a value of 30 on a 150-step task
    means 5 steps per bin. Set this to ``max_episode_length`` for one row per
    step (no aggregation); keep it small (e.g. 30) for visually-comparable
    heatmaps across tasks of different episode lengths. Affects only the
    success/failure heatmaps — the calibration heatmap's y-axis is the
    fixed-size ECE bin grid (10 bins). Min value 1."""

    success_wrapper: str | None = None
    """Name of an env wrapper (registered in ``wrappers/__init__.py``) that emits the
    per-step success flag into ``infos``. Currently registered: ``"lift"``. When
    ``predict_success`` is True, a non-null wrapper name is required — the runner
    raises before booting Isaac Sim if the combination is inconsistent."""

    recorder: RecorderCfg = dataclasses.field(default_factory=RecorderCfg)
    """Configures the optional ``RecordingWrapper`` (3x4-grid GIFs + TB videos).
    Default is disabled. See :class:`RecorderCfg` and ``wrappers/recording.py``."""

    def expand(self) -> None:
        """Expand the configuration."""
        super().expand()
        # learning rate scheduler
        if self.learning_rate_scheduler is None:
            self.learning_rate_scheduler = (None, None, None)
        elif not isinstance(self.learning_rate_scheduler, (tuple, list)):
            self.learning_rate_scheduler = (
                self.learning_rate_scheduler,
                self.learning_rate_scheduler,
                self.learning_rate_scheduler,
            )
        # learning rate scheduler kwargs
        if not isinstance(self.learning_rate_scheduler_kwargs, (tuple, list)):
            self.learning_rate_scheduler_kwargs = (
                self.learning_rate_scheduler_kwargs,
                self.learning_rate_scheduler_kwargs,
                self.learning_rate_scheduler_kwargs,
            )
