"""
engine.py
=========
High-level NSGA-II execution engine for the Shariah-compliant bi-objective
retail portfolio optimiser.

Entry point
-----------
    from src.optimization.engine import run_nsga2
    result = run_nsga2(problem, pop_size=100, n_gen=200, seed=42)
    df = result.to_dataframe()

The returned :class:`OptimizationResult` gives direct access to Pareto-front
weights, monetary allocations, net returns, liquidity scores, tax-relief and
Zakat amounts, and constraint-violation residuals for all non-dominated
solutions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.optimize import minimize

from src.config import (
    ASSETS,
    ASSET_INDEX,
    ZAKAT_NISAB,
    ZAKAT_RATE,
    SSPN_RELIEF_CAP,
)
from src.optimization.repair import SimplexRepair

if TYPE_CHECKING:
    from src.optimization.problem import ShariahPortfolioProblem

# Convenience index aliases
_ASB  = ASSET_INDEX["ASB"]
_TH   = ASSET_INDEX["TH"]
_KWSP = ASSET_INDEX["KWSP"]
_SSPN = ASSET_INDEX["SSPN"]
_GOP  = ASSET_INDEX["GO+"]

# Output column headers for to_dataframe()
_WEIGHT_COLS  = [f"w_{a.replace('+', 'plus')}" for a in ASSETS]
_RM_COLS      = [f"RM_{a.replace('+', 'plus')}" for a in ASSETS]
_SCALAR_COLS  = [
    "Net_Return",
    "Liquidity_Score",
    "Tax_Relief_RM",
    "Zakat_RM",
]


@dataclass
class OptimizationResult:
    """Structured container for NSGA-II Pareto-front results.

    All arrays are sorted in *ascending order of liquidity_scores* so that
    the caller can directly trace a continuous trade-off curve from
    yield-maximising to liquidity-maximising portfolios.

    Parameters
    ----------
    weights : np.ndarray, shape (K, 5)
        Non-dominated Pareto portfolio weight allocations.
        Each row sums to 1.0 with non-negative elements.
    net_returns : np.ndarray, shape (K,)
        Net annualised yield f1 = portfolio return + SSPN relief − Zakat.
    liquidity_scores : np.ndarray, shape (K,)
        Aggregate liquidity index f2 = Σ wᵢ Lᵢ.
    allocations_rm : np.ndarray, shape (K, 5)
        Monetary allocations in MYR: allocations_rm[k, i] = weights[k, i] * C.
    tax_relief_rm : np.ndarray, shape (K,)
        Absolute annual SSPN tax saving in MYR:
        min(w_SSPN * C, 8000) * tau.
    zakat_rm : np.ndarray, shape (K,)
        Absolute annual Zakat liability in MYR:
        0.025 * I{C >= 25000} * (w_ASB + w_SSPN + w_GO+) * C.
    asb_violations : np.ndarray, shape (K,)
        Excess over ASB ceiling in MYR: max(0, w_ASB * C - 300000).
        Ideally all zeros on the true Pareto front.
    go_plus_violations : np.ndarray, shape (K,)
        Excess over GO+ ceiling in MYR: max(0, w_GO+ * C - 20000).
        Ideally all zeros on the true Pareto front.
    raw_result : pymoo Result
        The raw pymoo Result object for advanced post-processing.
    """

    weights:          np.ndarray
    net_returns:      np.ndarray
    liquidity_scores: np.ndarray
    allocations_rm:   np.ndarray
    tax_relief_rm:    np.ndarray
    zakat_rm:         np.ndarray
    asb_violations:   np.ndarray
    go_plus_violations: np.ndarray
    raw_result:       object = field(repr=False)

    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Export all Pareto solutions to a tidy :class:`pandas.DataFrame`.

        Returns
        -------
        pd.DataFrame
            One row per non-dominated solution, columns:

            ``w_ASB, w_TH, w_KWSP, w_SSPN, w_GOplus``
                Portfolio weights (sum to 1.0).
            ``RM_ASB, RM_TH, RM_KWSP, RM_SSPN, RM_GOplus``
                Monetary allocations in MYR.
            ``Net_Return``
                Net annualised yield (decimal).
            ``Liquidity_Score``
                Aggregate liquidity index (decimal).
            ``Tax_Relief_RM``
                Annual SSPN tax saving in MYR.
            ``Zakat_RM``
                Annual Zakat liability in MYR.

            Rows are sorted by ascending ``Liquidity_Score``.
        """
        data = np.hstack([
            self.weights,                           # (K, 5)
            self.allocations_rm,                    # (K, 5)
            self.net_returns[:, None],              # (K, 1)
            self.liquidity_scores[:, None],         # (K, 1)
            self.tax_relief_rm[:, None],            # (K, 1)
            self.zakat_rm[:, None],                 # (K, 1)
        ])
        cols = _WEIGHT_COLS + _RM_COLS + _SCALAR_COLS
        return pd.DataFrame(data, columns=cols).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main execution function
# ---------------------------------------------------------------------------

def run_nsga2(
    problem: "ShariahPortfolioProblem",
    pop_size: int = 100,
    n_gen: int = 200,
    seed: int = 42,
    verbose: bool = False,
) -> OptimizationResult:
    """Run NSGA-II on a :class:`~src.optimization.problem.ShariahPortfolioProblem`.

    Parameters
    ----------
    problem : ShariahPortfolioProblem
        A fully configured bi-objective portfolio optimisation problem
        instance (capital, tax rate, returns, liquidity, caps).
    pop_size : int, optional
        Population size (default 100).  Larger values yield denser Pareto
        fronts but increase wall-clock time.
    n_gen : int, optional
        Number of generations (default 200).
    seed : int, optional
        Random seed for full reproducibility (default 42).
    verbose : bool, optional
        If True, pymoo prints per-generation progress (default False).

    Returns
    -------
    OptimizationResult
        Structured result container with Pareto weights, objectives,
        monetary allocations, tax/Zakat amounts, and violation residuals.
        Solutions are sorted in ascending order of ``liquidity_scores``.

    Raises
    ------
    RuntimeError
        If pymoo returns no feasible Pareto solutions (``res.X is None``).

    Notes
    -----
    Operator configuration:

    * **Crossover** — Simulated Binary Crossover (SBX): ``prob_var=0.90``,
      ``eta=20.0``, with :class:`~src.optimization.repair.SimplexRepair`.
    * **Mutation**  — Polynomial Mutation (PM): ``prob=0.20``, ``eta=20.0``,
      with :class:`~src.optimization.repair.SimplexRepair`.
    * **Repair**    — :class:`~src.optimization.repair.SimplexRepair` also
      attached to the algorithm level (selection / mating pool).
    * **Survival**  — Default NSGA-II rank-and-crowding.
    """
    repair = SimplexRepair()

    algorithm = NSGA2(
        pop_size=pop_size,
        crossover=SBX(prob_var=0.90, eta=20.0, repair=repair),
        mutation=PM(prob=0.20, eta=20.0, repair=repair),
        repair=repair,
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", n_gen)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=verbose,
    )

    if res.X is None:
        raise RuntimeError(
            "NSGA-II returned no feasible solutions. "
            "Check problem constraints and population size."
        )

    # ── Extract Pareto data ──────────────────────────────────────────────
    C   = problem.capital
    tau = problem.marginal_tax_rate

    weights = np.asarray(res.X, dtype=float)          # (K, 5)

    # Objectives (un-negate from pymoo minimisation convention)
    net_returns      = -res.F[:, 0]                   # (K,)
    liquidity_scores = -res.F[:, 1]                   # (K,)

    # Monetary allocations (MYR)
    allocations_rm = weights * C                      # (K, 5)

    # Absolute SSPN tax relief in MYR: min(w_SSPN * C, 8000) * tau
    tax_relief_rm = np.minimum(
        weights[:, _SSPN] * C, SSPN_RELIEF_CAP
    ) * tau                                           # (K,)

    # Absolute Zakat liability in MYR
    zakat_indicator = float(C >= ZAKAT_NISAB)
    zakatable_weight_sum = (
        weights[:, _ASB] + weights[:, _SSPN] + weights[:, _GOP]
    )
    zakat_rm = ZAKAT_RATE * zakat_indicator * zakatable_weight_sum * C  # (K,)

    # Constraint violation residuals in MYR (0 if feasible)
    asb_violations      = np.maximum(0.0, weights[:, _ASB] * C - problem.asb_cap)
    go_plus_violations  = np.maximum(0.0, weights[:, _GOP] * C - problem.go_plus_cap)

    # ── Sort by ascending liquidity for continuous trade-off trace ───────
    sort_idx = np.argsort(liquidity_scores)

    return OptimizationResult(
        weights            = weights[sort_idx],
        net_returns        = net_returns[sort_idx],
        liquidity_scores   = liquidity_scores[sort_idx],
        allocations_rm     = allocations_rm[sort_idx],
        tax_relief_rm      = tax_relief_rm[sort_idx],
        zakat_rm           = zakat_rm[sort_idx],
        asb_violations     = asb_violations[sort_idx],
        go_plus_violations = go_plus_violations[sort_idx],
        raw_result         = res,
    )
