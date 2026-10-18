"""
metrics.py
==========
Multi-objective evaluation metrics for the Shariah-compliant NSGA-II framework.

Implemented metrics
-------------------
• Hypervolume (HV)      — area of objective space dominated by the Pareto front
• Schott's Spacing (S)  — uniformity of solution distribution along the front
• Constraint Feasibility Rate (CFR) — fraction of feasible solutions (%)
• Compromise Solution   — TOPSIS-based balanced single-point selection

All functions operate purely on numpy arrays for speed; no pymoo dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.optimization.engine import OptimizationResult


# ---------------------------------------------------------------------------
# 1. Hypervolume Indicator
# ---------------------------------------------------------------------------

def calculate_hypervolume(
    net_returns: np.ndarray,
    liquidity_scores: np.ndarray,
    ref_point: tuple[float, float] = (0.0, 0.0),
) -> float:
    """Compute the exact 2-D hypervolume dominated by the Pareto front.

    Uses a sweep-line algorithm over non-overlapping rectangular strips:

    .. math::

        HV = \\sum_{i=0}^{K-1} (f_1^{(i)} - r_1)\\;(f_2^{(i)} - f_2^{(i-1)})

    where points are sorted by ascending :math:`f_2` (liquidity) and
    :math:`f_2^{(-1)} = r_2` (lower reference boundary).

    Parameters
    ----------
    net_returns : np.ndarray of shape (K,)
        Net annualised yield values :math:`f_1` for each Pareto solution.
    liquidity_scores : np.ndarray of shape (K,)
        Aggregate liquidity index values :math:`f_2`.
    ref_point : tuple (r1, r2), optional
        Nadir / worst-case reference point (default ``(0.0, 0.0)``).
        Must satisfy :math:`r_1 \\le \\min(f_1)` and :math:`r_2 \\le \\min(f_2)`.

    Returns
    -------
    float
        Non-negative dominated hypervolume area.

    Examples
    --------
    >>> calculate_hypervolume(np.array([3., 2., 1.]),
    ...                       np.array([1., 2., 3.]),
    ...                       ref_point=(0., 0.))
    6.0  # 3*1 + 2*1 + 1*1
    """
    f1 = np.asarray(net_returns,      dtype=float)
    f2 = np.asarray(liquidity_scores, dtype=float)
    r1, r2 = float(ref_point[0]), float(ref_point[1])

    if len(f1) == 0:
        return 0.0

    # Sort by f2 ascending (sweep bottom → top)
    sort_idx = np.argsort(f2)
    f1 = f1[sort_idx]
    f2 = f2[sort_idx]

    hv = 0.0
    prev_f2 = r2
    for i in range(len(f1)):
        width  = f1[i] - r1           # horizontal extent from ref
        height = f2[i] - prev_f2      # strip height
        if width > 0 and height > 0:
            hv += width * height
        prev_f2 = f2[i]

    return hv


# ---------------------------------------------------------------------------
# 2. Schott's Spacing Metric
# ---------------------------------------------------------------------------

def calculate_spacing(
    net_returns: np.ndarray,
    liquidity_scores: np.ndarray,
) -> float:
    """Compute Schott's spacing metric S for the Pareto front.

    Measures the uniformity of solution distribution:

    .. math::

        S = \\sqrt{\\frac{1}{K-1} \\sum_{i=1}^{K}\\left(d_i - \\bar{d}\\right)^2}

    where :math:`d_i = \\min_{j \\neq i} \\left(|f_1^{(i)}-f_1^{(j)}| +
    |f_2^{(i)}-f_2^{(j)}|\\right)` is the nearest-neighbour Manhattan distance.

    A value of 0 indicates perfectly uniform spacing.

    Parameters
    ----------
    net_returns : np.ndarray of shape (K,)
        Net return values :math:`f_1`.
    liquidity_scores : np.ndarray of shape (K,)
        Liquidity score values :math:`f_2`.

    Returns
    -------
    float
        Spacing metric *S* ≥ 0.  Returns ``0.0`` if *K* ≤ 1.
    """
    f1 = np.asarray(net_returns,      dtype=float)
    f2 = np.asarray(liquidity_scores, dtype=float)
    K  = len(f1)

    if K <= 1:
        return 0.0

    # Compute pairwise Manhattan distances between all solutions
    d_min = np.empty(K)
    for i in range(K):
        dists = np.abs(f1 - f1[i]) + np.abs(f2 - f2[i])
        dists[i] = np.inf          # exclude self-distance
        d_min[i] = dists.min()

    d_bar = d_min.mean()
    spacing = float(np.sqrt(np.sum((d_min - d_bar) ** 2) / (K - 1)))
    return spacing


# ---------------------------------------------------------------------------
# 3. Constraint Feasibility Rate
# ---------------------------------------------------------------------------

def calculate_cfr(violations: np.ndarray) -> float:
    """Compute the Constraint Feasibility Rate (CFR) as a percentage.

    .. math::

        \\text{CFR} = 100 \\times
        \\frac{|\\{i : v_i \\le 10^{-5}\\}|}{K}

    Parameters
    ----------
    violations : np.ndarray of shape (K,)
        Total constraint violation magnitude per solution (e.g.
        ``asb_violations + go_plus_violations`` in MYR, or the
        dimensionless ``G`` residuals).  Feasible iff ``v_i ≤ 1e-5``.

    Returns
    -------
    float
        Percentage of feasible solutions in [0.0, 100.0].
    """
    v = np.asarray(violations, dtype=float)
    if len(v) == 0:
        return 0.0
    feasible = v <= 1e-5
    return 100.0 * float(feasible.sum()) / len(v)


# ---------------------------------------------------------------------------
# 4. Compromise Solution Extractor (TOPSIS)
# ---------------------------------------------------------------------------

def extract_compromise_solution(
    result: "OptimizationResult",
    weights_preference: tuple[float, float] = (0.5, 0.5),
) -> dict:
    """Select the best balanced single solution from the Pareto front via TOPSIS.

    TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
    normalises both objectives to [0, 1], applies preference weights, computes
    distances to the ideal (1, 1) and nadir (0, 0) corners, and picks the
    solution maximising the relative closeness coefficient:

    .. math::

        C_i = \\frac{d_i^-}{d_i^+ + d_i^-}

    Parameters
    ----------
    result : OptimizationResult
        Completed NSGA-II result from :func:`~src.optimization.engine.run_nsga2`.
    weights_preference : tuple (w1, w2), optional
        Preference weights for (net_return, liquidity_score).
        Default (0.5, 0.5) gives equal importance.

    Returns
    -------
    dict with keys:
        index, weights, allocations_rm, net_return, liquidity_score,
        tax_relief_rm, zakat_rm, topsis_score
    """
    w1, w2 = float(weights_preference[0]), float(weights_preference[1])

    f1 = result.net_returns       # shape (K,)
    f2 = result.liquidity_scores  # shape (K,)
    K  = len(f1)

    if K == 1:
        idx = 0
        topsis_score = 1.0
    else:
        # Step 1: Normalise to [0, 1] range (min-max)
        f1_min, f1_max = f1.min(), f1.max()
        f2_min, f2_max = f2.min(), f2.max()

        f1_n = (f1 - f1_min) / (f1_max - f1_min + 1e-30)
        f2_n = (f2 - f2_min) / (f2_max - f2_min + 1e-30)

        # Step 2: Apply preference weights
        v1 = w1 * f1_n
        v2 = w2 * f2_n

        # Step 3: Ideal A+ = (w1, w2), Nadir A- = (0, 0)
        a_plus  = np.array([w1, w2])
        a_minus = np.array([0.0, 0.0])

        V = np.column_stack([v1, v2])  # (K, 2)

        # Step 4: Distances
        d_plus  = np.sqrt(((V - a_plus)  ** 2).sum(axis=1))
        d_minus = np.sqrt(((V - a_minus) ** 2).sum(axis=1))

        # Step 5: Closeness coefficient (higher = better balance)
        topsis_scores = d_minus / (d_plus + d_minus + 1e-30)
        idx = int(np.argmax(topsis_scores))
        topsis_score = float(topsis_scores[idx])

    return {
        "index":            idx,
        "weights":          result.weights[idx],
        "allocations_rm":   result.allocations_rm[idx],
        "net_return":       float(result.net_returns[idx]),
        "liquidity_score":  float(result.liquidity_scores[idx]),
        "tax_relief_rm":    float(result.tax_relief_rm[idx]),
        "zakat_rm":         float(result.zakat_rm[idx]),
        "topsis_score":     topsis_score,
    }
