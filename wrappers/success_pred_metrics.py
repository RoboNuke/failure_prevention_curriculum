"""Per-threshold success-prediction quality tracker.

Mirrors the metrics Forge's env computes for its action-dim success predictor —
delay between predicted and true success, plus precision/recall of "would
early-termination at threshold τ have been correct" — but works from any
per-step per-env success-probability stream (e.g. our actor's `success_prob`
head when ``predict_success=true``). Source-agnostic: feed it a (num_envs,)
probability tensor every step and a done-mask, and it owns the per-env
"first step where prob ≥ τ" bookkeeping internally.

Tag layout (intentionally different from the Forge env's flat tags): each tag
is prefixed with the threshold so TB groups one tab per τ. Inside each tab
you get ``early_term_delay_all``, ``early_term_delay_correct``,
``early_term_precision``, ``early_term_recall``.

This is used for ``predict_success=true`` envs (e.g. Factory) where the
predictor is our actor's BCE head. Forge's existing per_env_first_pred_success_tx
path in SAC stays separate (Forge's env owns its own bookkeeping driven by the
action-dim predictor); the tag layout there has been converged to match.
"""

from __future__ import annotations

from typing import Iterable

import torch


class SuccessPredMetricsTracker:
    """Tracks per-threshold first-prediction-crossing step per env.

    Call ``update`` every step with the per-env success probability and the
    per-env done-mask (terminated|truncated). Call ``flush_per_agent`` to push
    the four metrics into per-agent tracking buckets — that should fire every
    step too, since the metrics are "running over still-open episodes" (matches
    Forge env's semantics, which logs them every step).
    """

    def __init__(
        self,
        num_envs: int,
        num_agents: int,
        device: torch.device | str,
        thresholds: Iterable[float] = (0.5, 0.6, 0.7, 0.8, 0.9),
    ) -> None:
        if num_envs % num_agents != 0:
            raise ValueError(
                f"num_envs ({num_envs}) must be divisible by num_agents ({num_agents})"
            )
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.epa = num_envs // num_agents
        self.device = torch.device(device)
        self.thresholds = tuple(thresholds)

        # Per-env episode-step counter (reset to 0 on done, increments before
        # we check the threshold so step 1 is the first actionable step).
        self._step = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        # Per-threshold per-env "first step where prob ≥ τ" (0 = not yet).
        self._first_pred: dict[float, torch.Tensor] = {
            t: torch.zeros(num_envs, dtype=torch.long, device=self.device)
            for t in self.thresholds
        }

    def update(self, success_prob: torch.Tensor, done_mask: torch.Tensor) -> None:
        """Advance the internal episode-step counter and record threshold crossings.

        ``success_prob`` is (num_envs,) in [0, 1]. ``done_mask`` is (num_envs,)
        bool — envs that terminated/truncated this step. We reset their step
        counter and first_pred entries AFTER recording crossings, so a final
        step crossing still gets captured for the trajectory that just ended
        (consumers that read first_pred should call ``flush_per_agent`` BEFORE
        ``update`` resets, OR rely on the post-update ``ep_success_times``
        snapshot; we choose the latter — flush is called from the SAC consumer
        with the env's still-current ep_success_times before reset zeroes them
        out, just like Forge does).
        """
        if success_prob.shape[0] != self.num_envs:
            raise ValueError(
                f"success_prob shape {tuple(success_prob.shape)} doesn't match "
                f"num_envs={self.num_envs}"
            )
        if done_mask.shape[0] != self.num_envs:
            raise ValueError(
                f"done_mask shape {tuple(done_mask.shape)} doesn't match "
                f"num_envs={self.num_envs}"
            )
        self._step += 1
        for t in self.thresholds:
            crossed = (success_prob >= t) & (self._first_pred[t] == 0)
            if crossed.any():
                self._first_pred[t][crossed] = self._step[crossed]

    def reset_envs(self, done_mask: torch.Tensor) -> None:
        """Zero per-env state for envs that finished this step.

        Call AFTER ``flush_per_agent`` so the flush sees the final-step values
        (matching Forge env's ``_reset_buffers`` ordering).
        """
        if done_mask.any():
            self._step[done_mask] = 0
            for t in self.thresholds:
                self._first_pred[t][done_mask] = 0

    def flush_per_agent(
        self,
        per_agent_tracking: list[dict],
        ep_success_times: torch.Tensor,
    ) -> None:
        """Append per-agent early_term metrics to the tracking buckets.

        ``ep_success_times`` is (num_envs,) — per-env step-of-first-true-success
        in the still-open episode (0 = not yet). Same convention as Forge.

        Tags: ``<τ>/early_term_delay_all``, ``<τ>/early_term_delay_correct``,
        ``<τ>/early_term_precision``, ``<τ>/early_term_recall``. The threshold
        is rendered with one decimal (e.g. ``0.5``) so TB groups one tab per τ.
        """
        if ep_success_times.shape[0] != self.num_envs:
            raise ValueError(
                f"ep_success_times shape {tuple(ep_success_times.shape)} "
                f"doesn't match num_envs={self.num_envs}"
            )
        for t in self.thresholds:
            fst = self._first_pred[t]
            tag_root = f"{t:.1f}"
            for i in range(self.num_agents):
                lo, hi = i * self.epa, (i + 1) * self.epa
                a_fst = fst[lo:hi]
                a_est = ep_success_times[lo:hi]

                # Signed delay (pred - true) over envs where both have fired.
                # Positive = prediction lagged truth; negative = prediction was
                # premature.
                delay_mask = (a_est != 0) & (a_fst != 0)
                if delay_mask.any():
                    delay = (a_fst[delay_mask] - a_est[delay_mask]).float().mean()
                    per_agent_tracking[i][f"{tag_root}/early_term_delay_all"].append(
                        float(delay.item())
                    )
                    correct_mask = delay_mask & (a_fst > a_est)
                    if correct_mask.any():
                        cd = (a_fst[correct_mask] - a_est[correct_mask]).float().mean()
                        per_agent_tracking[i][f"{tag_root}/early_term_delay_correct"].append(
                            float(cd.item())
                        )

                # Of envs that predicted, fraction where truth had already fired.
                pred_mask = a_fst != 0
                if pred_mask.any():
                    tps = (a_est[pred_mask] > 0) & (a_est[pred_mask] < a_fst[pred_mask])
                    per_agent_tracking[i][f"{tag_root}/early_term_precision"].append(
                        float(tps.float().sum().item() / pred_mask.float().sum().item())
                    )
                    # Of envs that truly succeeded, fraction the predictor caught
                    # (with truth before prediction).
                    true_mask = a_est > 0
                    if true_mask.any():
                        per_agent_tracking[i][f"{tag_root}/early_term_recall"].append(
                            float(tps.float().sum().item() / true_mask.float().sum().item())
                        )
