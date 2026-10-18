"""
repair.py
=========
Custom pymoo repair operator for the Shariah-compliant portfolio optimiser.

The SimplexRepair operator is applied after each crossover / mutation step to
ensure every candidate solution is a valid portfolio weight vector living on
the non-negative probability simplex:

    w̃_i = max(0, w_i) / (Σ_j max(0, w_j) + ε)

This prevents negative weights (no short-selling) and guarantees the budget
constraint Σ w_i = 1 is satisfied before the objective is evaluated.
"""

from __future__ import annotations

import numpy as np
from pymoo.core.repair import Repair

from src.optimization.operators import simplex_normalize


class SimplexRepair(Repair):
    """Non-negative simplex normalisation repair operator for pymoo.

    Inherits from :class:`pymoo.core.repair.Repair` and overrides ``_do``
    to map any raw decision-variable matrix onto the probability simplex.

    Applied automatically by pymoo after crossover and mutation when passed
    to the algorithm via the ``repair`` keyword argument.

    Parameters
    ----------
    None — stateless operator; no constructor arguments required.

    Examples
    --------
    >>> repair = SimplexRepair()
    >>> X_raw = np.array([[-0.5, 1.2, 0.3, -0.1, 0.8],
    ...                   [ 0.2, 0.2, 0.2,  0.2, 0.2]])
    >>> X_fixed = repair._do(problem=None, X=X_raw)
    >>> np.allclose(X_fixed.sum(axis=1), 1.0)
    True
    """

    def _do(self, problem, X: np.ndarray, **kwargs) -> np.ndarray:
        """Repair a population matrix so every row lies on the simplex.

        Parameters
        ----------
        problem : pymoo Problem (may be None in unit tests)
            The optimisation problem instance (unused; kept for API
            compatibility with pymoo's :class:`Repair` base class).
        X : np.ndarray of shape (pop_size, n_var)
            Raw decision-variable matrix produced by crossover / mutation.
            Values may be negative or exceed 1.0.
        **kwargs
            Additional keyword arguments forwarded by pymoo (ignored).

        Returns
        -------
        np.ndarray of shape (pop_size, n_var)
            Repaired matrix where:
            • Every element is ≥ 0.0
            • Every row sums to 1.0 (within floating-point precision)
        """
        X = np.asarray(X, dtype=float)
        return np.apply_along_axis(simplex_normalize, axis=1, arr=X)
