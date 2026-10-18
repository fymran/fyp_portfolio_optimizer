"""
rule_based.py
=============
AKPK (Agensi Kaunseling dan Pengurusan Kredit) priority-cascade rule-based
heuristic baseline for the Shariah-compliant portfolio comparison.

Priority cascade (in order)
----------------------------
1. Liquidity buffer   → TNG GO+  (15 % of C, capped at RM 20,000)
2. Fiscal tax capture → SSPN     (RM 8,000 if τ > 0, else RM 0)
3. Sovereign trust    → ASB      (residual C, capped at RM 300,000)
4. Overflow reserve   → TH       (absorbs all remaining capital)
5. Retirement lockup  → KWSP     (held at 0; voluntary EPF not triggered)

This deterministic heuristic encodes standard Malaysian retail financial
planning advice and acts as the "expert rule" single-point baseline.
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
_TH   = ASSET_INDEX["TH"]
_KWSP = ASSET_INDEX["KWSP"]
_SSPN = ASSET_INDEX["SSPN"]
_GOP  = ASSET_INDEX["GO+"]


def evaluate_rule_based(
    capital: float,
    marginal_tax_rate: float,
    expected_returns: Optional[np.ndarray] = None,
    liquidity_scores: Optional[np.ndarray] = None,
    asb_cap: float = ASB_CEILING,
    go_plus_cap: float = GOPLUS_CEILING,
    nisab: float = ZAKAT_NISAB,
) -> dict:
    """Evaluate the AKPK priority-cascade rule-based portfolio allocation.

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

    C   = float(capital)
    tau = float(marginal_tax_rate)

    alloc = np.zeros(5, dtype=float)  # [ASB, TH, KWSP, SSPN, GO+]

    # ── Step 1: Liquidity buffer → TNG GO+ ───────────────────────────────
    c_gop = min(C * 0.15, go_plus_cap)
    alloc[_GOP] = c_gop
    c1 = C - c_gop

    # ── Step 2: Fiscal tax capture → SSPN ────────────────────────────────
    c_sspn = min(c1, SSPN_RELIEF_CAP) if tau > 0.0 else 0.0
    alloc[_SSPN] = c_sspn
    c2 = c1 - c_sspn

    # ── Step 3: Sovereign fixed-price trust → ASB ─────────────────────────
    c_asb = min(c2, asb_cap)
    alloc[_ASB] = c_asb
    c3 = c2 - c_asb

    # ── Step 4: Overflow reserve → TH ────────────────────────────────────
    alloc[_TH] = c3

    # ── Step 5: Retirement → KWSP (held in reserve, 0 voluntary alloc) ───
    alloc[_KWSP] = 0.0

    # ── Normalise to weights ──────────────────────────────────────────────
    w = alloc / C

    # ── Gross base return ─────────────────────────────────────────────────
    gross_return = float(w @ R)

    # ── SSPN tax relief ───────────────────────────────────────────────────
    tax_relief_rm   = min(alloc[_SSPN], SSPN_RELIEF_CAP) * tau
    tax_relief_rate = tax_relief_rm / C if C > 0 else 0.0

    # ── Zakat liability ───────────────────────────────────────────────────
    zakat_indicator  = 1.0 if C >= nisab else 0.0
    zakatable_weight = w[_ASB] + w[_SSPN] + w[_GOP]
    zakat_rm   = ZAKAT_RATE * zakat_indicator * zakatable_weight * C
    zakat_rate = zakat_rm / C if C > 0 else 0.0

    # ── Net return ────────────────────────────────────────────────────────
    net_return = gross_return + tax_relief_rate - zakat_rate

    # ── Liquidity score ───────────────────────────────────────────────────
    liquidity_score = float(w @ L)

    # ── Statutory violations (MYR) ────────────────────────────────────────
    # By construction both should be 0 — included for API symmetry / testing
    asb_violation_rm     = max(0.0, alloc[_ASB] - asb_cap)
    go_plus_violation_rm = max(0.0, alloc[_GOP] - go_plus_cap)

    is_feasible = (asb_violation_rm == 0.0) and (go_plus_violation_rm == 0.0)

    return {
        "weights":             w,
        "allocations_rm":      alloc,
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
