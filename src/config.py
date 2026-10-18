"""
config.py
=========
Statutory boundaries and baseline parameters for the Shariah-compliant
retail portfolio optimisation framework.

All monetary amounts are in Malaysian Ringgit (MYR).
All rate/return values are expressed as decimals (e.g. 5.25 % → 0.0525).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Asset universe
# ---------------------------------------------------------------------------
ASSETS: list[str] = ["ASB", "TH", "KWSP", "SSPN", "GO+"]
N_ASSETS: int = len(ASSETS)

# Index map for convenient look-up
ASSET_INDEX: dict[str, int] = {name: i for i, name in enumerate(ASSETS)}

# ---------------------------------------------------------------------------
# Expected annual returns vector R  (decimal form)
# Source: placeholder / illustrative values — replace with live data
# ---------------------------------------------------------------------------
#   ASB  : 5.25 %   (Amanah Saham Bumiputera dividend + bonus, 2023 est.)
#   TH   : 3.10 %   (Tabung Haji hibah, illustrative)
#   KWSP : 5.40 %   (EPF dividend, FY2023)
#   SSPN : 3.60 %   (SSPN-i dividend, illustrative)
#   GO+  : 3.45 %   (TNG GO+ annual equivalent, illustrative)
RETURNS: np.ndarray = np.array([0.0525, 0.0310, 0.0540, 0.0360, 0.0345])

# ---------------------------------------------------------------------------
# Normalised liquidity scores  L ∈ [0, 1]^5
# Higher score → more liquid
#   ASB  : 0.7  (redeemable on demand, but subject to counter availability)
#   TH   : 0.7  (withdrawal allowed with notice)
#   KWSP : 0.2  (locked until retirement; limited early withdrawal)
#   SSPN : 0.2  (longer lock-in for tax benefit; partial early withdrawal)
#   GO+  : 1.0  (instant e-wallet redemption)
# ---------------------------------------------------------------------------
LIQUIDITY: np.ndarray = np.array([0.7, 0.7, 0.2, 0.2, 1.0])

# ---------------------------------------------------------------------------
# Statutory ceilings (MYR)
# ---------------------------------------------------------------------------
ASB_CEILING: float = 300_000.0   # Maximum permissible ASB investment per unit holder
GOPLUS_CEILING: float = 20_000.0  # TNG GO+ e-money wallet ceiling

# ---------------------------------------------------------------------------
# Tax-relief parameters
# ---------------------------------------------------------------------------
SSPN_RELIEF_CAP: float = 8_000.0  # Annual SSPN tax-relief statutory ceiling (MYR)

# ---------------------------------------------------------------------------
# Zakat parameters
# ---------------------------------------------------------------------------
ZAKAT_NISAB: float = 25_000.0   # Nisab threshold (MYR); below this, no Zakat liability
ZAKAT_RATE: float = 0.025       # 2.5 % of qualifying holdings

# Assets subject to Zakat (savings/investment instruments that are zakatable).
# Tabung Haji (TH) and KWSP are excluded:
#   • TH  — institution deducts Zakat internally; double-counting risk.
#   • KWSP — EPF contributions are not zakat-eligible until withdrawal.
ZAKAT_ELIGIBLE_ASSETS: list[str] = ["ASB", "SSPN", "GO+"]
ZAKAT_ELIGIBLE_INDICES: list[int] = [ASSET_INDEX[a] for a in ZAKAT_ELIGIBLE_ASSETS]
