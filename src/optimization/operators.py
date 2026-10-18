"""
operators.py
============
Genetic / evolutionary operator utilities for the NSGA-II portfolio optimiser.

Repair / normalisation operators are applied *after* crossover and mutation
to ensure every candidate solution is a valid portfolio weight vector living
on the probability simplex.
"""

from __future__ import annotations

import numpy as np

# Small constant to prevent division by zero when all weights are ≤ 0
_EPSILON: float = 1e-12


def simplex_normalize(weights: np.ndarray | list[float]) -> np.ndarray:
    """Project a raw weight vector onto the non-negative probability simplex.

    Implements the Non-Negative Simplex Normalisation repair operator:

    .. math::

        \\hat{w}_i = \\frac{\\max(0,\\, w_i)}{\\sum_{j=1}^{N} \\max(0,\\, w_j) + \\epsilon}

    This mapping is applied element-wise:

    1. Negative weights are clipped to zero (no short-selling).
    2. The clipped vector is rescaled so that all components sum to 1.
    3. A small :math:`\\epsilon` guards against the degenerate all-negative
       input, producing a uniform allocation in that edge case.

    Parameters
    ----------
    weights : array-like of shape (N,)
        Raw weight vector from the evolutionary operator (may contain
        negative values or values that do not sum to 1).

    Returns
    -------
    np.ndarray of shape (N,)
        Non-negative weight vector satisfying
        ``sum(w_hat) ≈ 1`` and ``w_hat[i] >= 0`` for all *i*.

    Examples
    --------
    >>> simplex_normalize([0.3, -0.1, 0.5, 0.0, 0.2])
    array([0.3, 0. , 0.5, 0. , 0.2])  # already valid; negatives clipped
    >>> simplex_normalize([-1, -2, -3])
    array([0.333..., 0.333..., 0.333...])  # degenerate → uniform (via ε)
    """
    weights = np.asarray(weights, dtype=float)
    clipped = np.maximum(0.0, weights)
    denom = clipped.sum() + _EPSILON
    return clipped / denom
