"""
personas.py
===========
Benchmark investor personas for the Shariah-compliant portfolio evaluation.

Three personas capture the full spectrum of the Malaysian retail investor
population targeted by the AKPK / BNM financial inclusion agenda:

  • Aiman      — Young, low-income, zero tax liability (B40 segment)
  • Farah      — Mid-career professional, moderate tax bracket (M40 segment)
  • Encik Razak — High-net-worth retiree/investor, top bracket (T20 segment)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    """Immutable description of a benchmark investor persona.

    Attributes
    ----------
    name : str
        Short display name (e.g. "Aiman").
    archetype : str
        Socio-economic archetype label (e.g. "Young Urban Earner").
    capital : float
        Total investable capital *C* in MYR.
    marginal_tax_rate : float
        Investor's marginal income-tax rate *τ* (decimal, 0 – 0.30).
    monthly_income : float
        Gross monthly income in MYR (used for contextual reporting only).
    description : str
        Free-text narrative about the persona's financial situation.
    """

    name:              str
    archetype:         str
    capital:           float
    marginal_tax_rate: float
    monthly_income:    float
    description:       str


# ---------------------------------------------------------------------------
# Benchmark persona instances
# ---------------------------------------------------------------------------

AIMAN = Persona(
    name              = "Aiman",
    archetype         = "Young Urban Earner (B40)",
    capital           = 5_000.0,
    marginal_tax_rate = 0.00,
    monthly_income    = 1_200.0,
    description       = (
        "Recent graduate working in Kuala Lumpur. Monthly income of RM 1,200 "
        "places him firmly in the B40 bracket with zero income-tax liability. "
        "Investable capital of RM 5,000 represents his Hari Raya savings. "
        "Priority: capital preservation and liquidity (emergency fund). "
        "Zakat not applicable (C < Nisab RM 25,000)."
    ),
)

FARAH = Persona(
    name              = "Farah",
    archetype         = "Mid-Career Professional (M40)",
    capital           = 35_000.0,
    marginal_tax_rate = 0.06,
    monthly_income    = 4_500.0,
    description       = (
        "Accountant with 8 years of experience. Monthly income RM 4,500 "
        "puts her in the 6 % marginal tax bracket. Investable capital of "
        "RM 35,000 exceeds the Nisab threshold, so Zakat applies. "
        "Priority: SSPN tax optimisation for her children's education, "
        "balanced with moderate liquidity needs."
    ),
)

ENCIK_RAZAK = Persona(
    name              = "Encik Razak",
    archetype         = "High-Net-Worth Investor (T20)",
    capital           = 450_000.0,
    marginal_tax_rate = 0.25,
    monthly_income    = 18_000.0,
    description       = (
        "Senior engineer / part-time entrepreneur. Monthly income RM 18,000 "
        "places him in the top 25 % marginal tax bracket. Capital of "
        "RM 450,000 triggers both ASB and GO+ statutory ceilings and incurs "
        "full Zakat liability. Priority: maximise after-tax yield while "
        "satisfying all statutory constraints and Shariah compliance."
    ),
)

# ---------------------------------------------------------------------------
# Exported registry
# ---------------------------------------------------------------------------

PERSONAS: dict[str, Persona] = {
    "Aiman":        AIMAN,
    "Farah":        FARAH,
    "Encik Razak":  ENCIK_RAZAK,
}
