"""
tax_zakat.py
============
Shariah-compliant tax-relief and Zakat computation functions.

Mathematical definitions
------------------------

SSPN Tax-Relief Benefit (per-unit-capital, after-tax basis):

    relief(w_sspn, C, tau) = min(w_sspn * C, SSPN_CAP) * tau / C

    where
        w_sspn  : weight allocated to SSPN  (scalar, [0, 1])
        C       : total invested capital (MYR)
        tau     : marginal income-tax rate of the investor (decimal)
        SSPN_CAP: RM 8,000 statutory ceiling

    Interpretation: annual tax saved, normalised to a per-Ringgit-of-capital
    rate, so it can be added directly to the expected-return vector.

Zakat Liability Rate:

    zakat(w, C) = 0.025 * I{C >= Nisab} * (w_ASB + w_SSPN + w_GO+)

    where
        w       : weight vector (array of length 5)
        C       : total capital (MYR)
        Nisab   : RM 25,000 threshold
        I{.}    : indicator function (1 if condition holds, else 0)

    Note: Tabung Haji (TH) and KWSP are *excluded* from the Zakat base.
    TH deducts Zakat internally; KWSP contributions are not zakat-eligible
    until withdrawn.
"""

from __future__ import annotations

import numpy as np

from src.config import (
    ASSET_INDEX,
    ZAKAT_NISAB,
    ZAKAT_RATE,
    ZAKAT_ELIGIBLE_INDICES,
    SSPN_RELIEF_CAP,
)


def calculate_sspn_tax_relief(
    w_sspn: float,
    capital: float,
    tau: float,
    sspn_cap: float = SSPN_RELIEF_CAP,
) -> float:
    """Return the SSPN tax-relief as a *per-capital* rate.

    Parameters
    ----------
    w_sspn : float
        Portfolio weight allocated to SSPN (must be in [0, 1]).
    capital : float
        Total invested capital C in MYR.  Must be > 0.
    tau : float
        Investor's marginal income-tax rate (decimal, e.g. 0.24 for 24 %).
    sspn_cap : float, optional
        Statutory SSPN relief ceiling in MYR (default: RM 8,000).

    Returns
    -------
    float
        Annualised tax saving per Ringgit of capital, i.e.
        ``min(w_sspn * C, sspn_cap) * tau / C``.
        Returns 0.0 if ``tau == 0`` or ``capital <= 0``.

    Examples
    --------
    >>> calculate_sspn_tax_relief(0.5, 10_000, 0.24)   # SSPN alloc = RM 5 000
    0.012  # = min(5000, 8000) * 0.24 / 10000
    >>> calculate_sspn_tax_relief(1.0, 100_000, 0.24)  # alloc exceeds cap
    0.0192 # = 8000 * 0.24 / 100000
    """
    if capital <= 0 or tau == 0.0:
        return 0.0

    eligible_amount = min(w_sspn * capital, sspn_cap)
    return (eligible_amount * tau) / capital


def calculate_zakat(
    weights: np.ndarray | list[float],
    capital: float,
    nisab: float = ZAKAT_NISAB,
) -> float:
    """Return the Zakat liability as a *per-capital* rate.

    Zakat is levied at 2.5 % on the zakatable portion of the portfolio
    (ASB + SSPN + GO+) *only when* total capital meets or exceeds the Nisab
    threshold.

    Parameters
    ----------
    weights : array-like of shape (5,)
        Portfolio weight vector ``[w_ASB, w_TH, w_KWSP, w_SSPN, w_GO+]``.
        Weights need not sum to exactly 1 (post-normalisation slack is fine),
        but each element must be non-negative.
    capital : float
        Total invested capital C in MYR.
    nisab : float, optional
        Nisab threshold in MYR (default: RM 25,000).

    Returns
    -------
    float
        ``ZAKAT_RATE * (w_ASB + w_SSPN + w_GO+)`` if ``capital >= nisab``,
        otherwise ``0.0``.

    Examples
    --------
    >>> calculate_zakat([0.4, 0.2, 0.2, 0.1, 0.1], 30_000)
    0.015  # 2.5 % * (0.4 + 0.1 + 0.1) = 2.5 % * 0.6
    >>> calculate_zakat([0.4, 0.2, 0.2, 0.1, 0.1], 10_000)
    0.0    # capital < Nisab
    """
    weights = np.asarray(weights, dtype=float)

    if capital < nisab:
        return 0.0

    zakatable_weight = float(weights[ZAKAT_ELIGIBLE_INDICES].sum())
    return ZAKAT_RATE * zakatable_weight
