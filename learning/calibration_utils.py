"""Small reusable calibration helpers.

Extracted from :mod:`learning.pred_quality` so the rescue-metrics tracker can
compute ECE / ROC-AUC over rescue-point predictions without duplicating code.
Pure NumPy + sklearn — no torch dependency, no per-agent partitioning. Callers
own the partitioning and pass per-group flat arrays.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def roc_auc(P: np.ndarray, y: np.ndarray) -> float | None:
    """ROC-AUC of (P, y). Returns None when only one class is present.

    :param P: predicted probabilities, shape ``(N,)``, values in ``[0, 1]``.
    :param y: binary labels, shape ``(N,)``, values in ``{0, 1}``.
    """
    if P.shape != y.shape:
        raise ValueError(f"roc_auc: shape mismatch P {P.shape} vs y {y.shape}")
    if P.size == 0:
        return None
    # Defined only when both classes are represented.
    if y.min() >= y.max():
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, P))


def ece(
    P: np.ndarray,
    y: np.ndarray,
    n_bins: int,
) -> Tuple[float, np.ndarray]:
    """Expected Calibration Error with ``n_bins`` equal-width P bins over [0, 1].

    Returns ``(ece, calib_col)`` where ``calib_col[b] = conf_b − acc_b`` (signed;
    negative ⇒ overconfident, positive ⇒ underconfident, NaN where the bin had
    no rows). The signed column is the same one ``pred_quality`` plots in its
    calibration heatmap; rescue-side callers can ignore it.
    """
    if P.shape != y.shape:
        raise ValueError(f"ece: shape mismatch P {P.shape} vs y {y.shape}")
    if n_bins < 1:
        raise ValueError(f"ece: n_bins must be >= 1, got {n_bins}")
    N = P.shape[0]
    calib_col = np.full(n_bins, np.nan, dtype=np.float32)
    if N == 0:
        return 0.0, calib_col
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == n_bins - 1:
            in_bin = (P >= lo) & (P <= hi)  # include 1.0 in the last bin
        else:
            in_bin = (P >= lo) & (P < hi)
        n_b = int(in_bin.sum())
        if n_b == 0:
            continue
        conf_b = float(P[in_bin].mean())
        acc_b = float(y[in_bin].mean())
        calib_col[b] = conf_b - acc_b
        total += (n_b / N) * abs(conf_b - acc_b)
    return float(total), calib_col
