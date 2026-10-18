"""
equal_weight.py
===============
Naïve 1/N equal-weight baseline for the Shariah-compliant portfolio comparison.

Allocates exactly 1/5 = 20 % to every asset regardless of capital or tax
status.  This is the simplest possible diversification heuristic and serves as
the lower-bound benchmark that the NSGA-II front is expected to dominate on at
least one objective.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.config import (
    ASSET_INDEX,
    RETURNS,
    LIQUIDITY,
    ASB_CEILING,
    GOPLUS_CEILING,
    ZAKAT_NISAB,
    ZAKAT_RATE,
    SSPN_RELIEF_CAP,
)

_ASB  = ASSET_INDEX["ASB"]
_SSPN = ASSET_INDEX["SSPN"]
_GOP  = ASSET_INDEX["GO+"]

_EQUAL_WEIGHT = np.full(5, 0.20)


def evaluate_equal_weight(
    capital: float,
    marginal_tax_rate: float,
    expected_returns: Optional[np.ndarray] = None,
    liquidity_scores: Optional[np.ndarray] = None,
    asb_cap: float = ASB_CEILING,
    go_plus_cap: float = GOPLUS_CEILING,
    nisab: float = ZAKAT_NISAB,
) -> dict:
    """Evaluate the naïve 1/N equal-weight portfolio allocation.

    Parameters
    ----------
    capital : float
        Total investable capital *C* in MYR (> 0).
    marginal_tax_rate : float
        Investor's marginal income-tax rate *τ* (decimal).
    expected_returns : np.ndarray of shape (5,), optional
        Override default annual returns from ``config.py``.
    liquidity_scores : np.ndarray of shape (5,), optional
        Override default liquidity scores from ``config.py``.
    asb_cap, go_plus_cap : float
        Statutory ceilings in MYR.
    nisab : float
        Zakat Nisab threshold in MYR.

    Returns
    -------
    dict with keys:
        weights, allocations_rm, gross_return, tax_relief_rm, tax_relief_rate,
        zakat_rm, zakat_rate, net_return, liquidity_score,
        asb_violation_rm, go_plus_violation_rm, is_feasible
    """
    R = np.asarray(expected_returns if expected_returns is not None else RETURNS, dtype=float)
    L = np.asarray(liquidity_scores if liquidity_scores is not None else LIQUIDITY, dtype=float)

    w = _EQUAL_WEIGHT.copy()
    C = float(capital)
    tau = float(marginal_tax_rate)

    allocations_rm = w * C

    # ── Gross base return ─────────────────────────────────────────────────
    gross_return = float(w @ R)

    # ── SSPN tax relief ───────────────────────────────────────────────────
    tax_relief_rm   = min(w[_SSPN] * C, SSPN_RELIEF_CAP) * tau
    tax_relief_rate = tax_relief_rm / C if C > 0 else 0.0

    # ── Zakat liability ───────────────────────────────────────────────────
    zakat_indicator = 1.0 if C >= nisab else 0.0
    zakatable_weight = w[_ASB] + w[_SSPN] + w[_GOP]
    zakat_rm   = ZAKAT_RATE * zakat_indicator * zakatable_weight * C
    zakat_rate = zakat_rm / C if C > 0 else 0.0

    # ── Net return ────────────────────────────────────────────────────────
    net_return = gross_return + tax_relief_rate - zakat_rate

    # ── Liquidity score ───────────────────────────────────────────────────
    liquidity_score = float(w @ L)

    # ── Statutory violations (MYR) ────────────────────────────────────────
    asb_violation_rm      = max(0.0, w[_ASB] * C - asb_cap)
    go_plus_violation_rm  = max(0.0, w[_GOP] * C - go_plus_cap)

    is_feasible = (asb_violation_rm == 0.0) and (go_plus_violation_rm == 0.0)

    return {
        "weights":             w,
        "allocations_rm":      allocations_rm,
        "gross_return":        gross_return,
        "tax_relief_rm":       tax_relief_rm,
        "tax_relief_rate":     tax_relief_rate,
        "zakat_rm":            zakat_rm,
        "zakat_rate":          zakat_rate,
        "net_return":          net_return,
        "liquidity_score":     liquidity_score,
        "asb_violation_rm":    asb_violation_rm,
        "go_plus_violation_rm": go_plus_violation_rm,
        "is_feasible":         is_feasible,
    }
