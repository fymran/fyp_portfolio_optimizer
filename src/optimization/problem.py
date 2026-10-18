"""
problem.py
==========
pymoo problem definition for the Shariah-compliant bi-objective
retail portfolio optimiser.

Objectives (both maximised → negated for pymoo's minimisation convention)
--------------------------------------------------------------------------
f1 : Net Annualised Yield
     = Σ wᵢ Rᵢ
       + min(w_SSPN·C, 8000)·τ / C          ← SSPN tax-relief bonus
       − 0.025·I{C≥Nisab}·(w_ASB+w_SSPN+w_GO+) ← Zakat liability

f2 : Aggregate Liquidity Index
     = Σ wᵢ Lᵢ

Inequality constraints (g ≤ 0, scale-invariant form)
-----------------------------------------------------
g1 = (w_ASB·C − asb_cap)   / C   ← ASB statutory ceiling
g2 = (w_GO+·C − go_plus_cap) / C ← TNG GO+ e-wallet ceiling

Decision variables
------------------
x = [w_ASB, w_TH, w_KWSP, w_SSPN, w_GO+]  ∈ [0, 1]^5
(simplex constraint enforced by SimplexRepair before each evaluation)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from src.config import (
    ASSET_INDEX,
    RETURNS,
    LIQUIDITY,
    ASB_CEILING,
    GOPLUS_CEILING,
    ZAKAT_NISAB,
    SSPN_RELIEF_CAP,
)
from src.domain.tax_zakat import calculate_sspn_tax_relief, calculate_zakat
from src.optimization.operators import simplex_normalize

# Convenience index aliases
_ASB  = ASSET_INDEX["ASB"]
_TH   = ASSET_INDEX["TH"]
_KWSP = ASSET_INDEX["KWSP"]
_SSPN = ASSET_INDEX["SSPN"]
_GOP  = ASSET_INDEX["GO+"]


class ShariahPortfolioProblem(ElementwiseProblem):
    """Bi-objective Shariah-compliant retail portfolio optimisation problem.

    Formulated for pymoo's NSGA-II (or any Pareto-front solver).
    Each decision vector ``x`` represents a portfolio weight allocation over
    the five-asset universe defined in :mod:`src.config`.

    Parameters
    ----------
    capital : float
        Total investable capital *C* in MYR.  Must be > 0.
    marginal_tax_rate : float
        Investor's marginal income-tax rate *τ* (decimal, e.g. 0.24 for 24 %).
        Used to compute the SSPN tax-relief bonus.
    expected_returns : np.ndarray of shape (5,), optional
        Override the default annual returns vector **R** from ``config.py``.
    liquidity_scores : np.ndarray of shape (5,), optional
        Override the default liquidity vector **L** from ``config.py``.
    asb_cap : float, optional
        Statutory ASB investment ceiling in MYR (default: RM 300,000).
    go_plus_cap : float, optional
        TNG GO+ e-wallet ceiling in MYR (default: RM 20,000).
    nisab : float, optional
        Zakat Nisab threshold in MYR (default: RM 25,000).

    Attributes
    ----------
    capital, marginal_tax_rate, asb_cap, go_plus_cap, nisab : float
        Problem parameters.
    R : np.ndarray
        Expected annual returns vector.
    L : np.ndarray
        Normalised liquidity scores vector.
    """

    def __init__(
        self,
        capital: float,
        marginal_tax_rate: float,
        expected_returns: Optional[np.ndarray] = None,
        liquidity_scores: Optional[np.ndarray] = None,
        asb_cap: float = ASB_CEILING,
        go_plus_cap: float = GOPLUS_CEILING,
        nisab: float = ZAKAT_NISAB,
    ) -> None:
        if capital <= 0:
            raise ValueError(f"capital must be > 0; got {capital}")
        if not (0.0 <= marginal_tax_rate <= 0.30):
            raise ValueError(
                f"marginal_tax_rate must be in [0, 0.30]; got {marginal_tax_rate}"
            )

        super().__init__(
            n_var=5,
            n_obj=2,
            n_ieq_constr=2,   # pymoo 0.6.x uses n_ieq_constr for inequality constraints
            xl=0.0,
            xu=1.0,
        )

        self.capital = float(capital)
        self.marginal_tax_rate = float(marginal_tax_rate)
        self.R = np.asarray(
            expected_returns if expected_returns is not None else RETURNS,
            dtype=float,
        )
        self.L = np.asarray(
            liquidity_scores if liquidity_scores is not None else LIQUIDITY,
            dtype=float,
        )
        self.asb_cap = float(asb_cap)
        self.go_plus_cap = float(go_plus_cap)
        self.nisab = float(nisab)

    # ------------------------------------------------------------------
    # pymoo ElementwiseProblem interface
    # ------------------------------------------------------------------

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        """Evaluate objectives and constraints for a single portfolio vector.

        Parameters
        ----------
        x : np.ndarray of shape (5,)
            Raw decision vector ``[w_ASB, w_TH, w_KWSP, w_SSPN, w_GO+]``.
            Simplex normalisation is applied internally before evaluation so
            that the function is well-defined even for out-of-simplex inputs.
        out : dict
            pymoo output dictionary.  Populated with:
            • ``out["F"]`` : array of shape (2,) — negated objectives for
              minimisation: ``[-f1, -f2]``.
            • ``out["G"]`` : array of shape (2,) — inequality constraint
              residuals ``[g1, g2]``; feasible when ≤ 0.
        """
        C   = self.capital
        tau = self.marginal_tax_rate

        # ── Repair: project onto simplex ──────────────────────────────
        w = simplex_normalize(x)

        # ── Objective 1: Net Annualised Yield (f1) ────────────────────
        # Base expected return
        base_return = float(w @ self.R)

        # SSPN income-tax relief (per-capital rate)
        sspn_relief = calculate_sspn_tax_relief(
            w_sspn=w[_SSPN],
            capital=C,
            tau=tau,
        )

        # Zakat liability (per-capital rate, deducted from yield)
        zakat_charge = calculate_zakat(w, capital=C, nisab=self.nisab)

        f1 = base_return + sspn_relief - zakat_charge

        # ── Objective 2: Aggregate Liquidity Index (f2) ───────────────
        f2 = float(w @ self.L)

        # Both objectives are maximised → negate for pymoo minimisation
        out["F"] = [-f1, -f2]

        # ── Inequality Constraints (g ≤ 0) ────────────────────────────
        # Scale-invariant: divide by C so units are dimensionless
        g1 = (w[_ASB] * C - self.asb_cap)  / C   # ASB ceiling
        g2 = (w[_GOP]  * C - self.go_plus_cap) / C   # GO+ ceiling

        out["G"] = [g1, g2]
