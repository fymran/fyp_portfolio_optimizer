"""
test_baselines_personas.py
==========================
Unit tests for Milestone 4:
    • Naïve 1/N equal-weight baseline  (src/baselines/equal_weight.py)
    • Rule-based AKPK baseline         (src/baselines/rule_based.py)
    • Multi-objective metrics          (src/evaluation/metrics.py)
    • Persona registry                 (src/evaluation/personas.py)

Run with:
    pytest tests/test_baselines_personas.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.baselines.equal_weight import evaluate_equal_weight
from src.baselines.rule_based import evaluate_rule_based
from src.evaluation.personas import PERSONAS, AIMAN, FARAH, ENCIK_RAZAK
from src.evaluation.metrics import (
    calculate_hypervolume,
    calculate_spacing,
    calculate_cfr,
    extract_compromise_solution,
)
from src.config import ASSET_INDEX, SSPN_RELIEF_CAP, ZAKAT_RATE

_ASB  = ASSET_INDEX["ASB"]
_TH   = ASSET_INDEX["TH"]
_KWSP = ASSET_INDEX["KWSP"]
_SSPN = ASSET_INDEX["SSPN"]
_GOP  = ASSET_INDEX["GO+"]


# ============================================================================
# 1. Naïve 1/N Equal-Weight Baseline
# ============================================================================

class TestEqualWeight:
    """Verify equal-weight baseline metrics and violation detection."""

    # ── Encik Razak: high-capital, all caps breached ──

    def test_encik_razak_goplus_violation_amount(self):
        """w_GO+ * C = 0.2 * 450000 = RM90,000 → violation = RM 70,000 exactly."""
        res = evaluate_equal_weight(
            capital=ENCIK_RAZAK.capital,
            marginal_tax_rate=ENCIK_RAZAK.marginal_tax_rate,
        )
        expected_violation = 0.20 * 450_000.0 - 20_000.0  # = 70_000
        assert pytest.approx(res["go_plus_violation_rm"], abs=1e-6) == expected_violation, (
            f"Expected GO+ violation = RM70,000; got RM{res['go_plus_violation_rm']:.2f}"
        )

    def test_encik_razak_goplus_350pct_excess(self):
        """RM 90,000 is 350% of the RM 20,000 cap → violation = 3.5 × cap."""
        res = evaluate_equal_weight(
            capital=ENCIK_RAZAK.capital,
            marginal_tax_rate=ENCIK_RAZAK.marginal_tax_rate,
        )
        # 70,000 / 20,000 = 3.5  (350% excess)
        ratio = res["go_plus_violation_rm"] / 20_000.0
        assert pytest.approx(ratio, rel=1e-9) == 3.50

    def test_encik_razak_not_feasible(self):
        """Equal-weight is_feasible must be False for Encik Razak."""
        res = evaluate_equal_weight(
            capital=ENCIK_RAZAK.capital,
            marginal_tax_rate=ENCIK_RAZAK.marginal_tax_rate,
        )
        assert not res["is_feasible"]

    def test_encik_razak_asb_no_violation(self):
        """w_ASB * C = 0.2 * 450000 = RM90,000 < RM300,000 → ASB violation = 0."""
        res = evaluate_equal_weight(
            capital=ENCIK_RAZAK.capital,
            marginal_tax_rate=ENCIK_RAZAK.marginal_tax_rate,
        )
        assert res["asb_violation_rm"] == 0.0

    # ── Aiman: low-capital, zero tax ──

    def test_aiman_zero_tax_relief(self):
        """Aiman has τ = 0 → tax_relief_rm must be exactly RM 0.00."""
        res = evaluate_equal_weight(
            capital=AIMAN.capital,
            marginal_tax_rate=AIMAN.marginal_tax_rate,
        )
        assert res["tax_relief_rm"] == 0.0, (
            f"Expected tax_relief_rm = 0.0 for τ=0; got {res['tax_relief_rm']}"
        )

    def test_aiman_feasible(self):
        """Aiman's C = RM 5,000 → both allocations well within ceilings."""
        res = evaluate_equal_weight(
            capital=AIMAN.capital,
            marginal_tax_rate=AIMAN.marginal_tax_rate,
        )
        assert res["is_feasible"]

    def test_aiman_no_zakat(self):
        """Aiman's C = RM 5,000 < Nisab RM 25,000 → zakat_rm = RM 0.00."""
        res = evaluate_equal_weight(
            capital=AIMAN.capital,
            marginal_tax_rate=AIMAN.marginal_tax_rate,
        )
        assert res["zakat_rm"] == 0.0

    # ── Farah: medium-capital, moderate tax ──

    def test_farah_tax_relief_capped(self):
        """w_SSPN * C = 0.2 * 35000 = RM7,000 < RM8,000 cap → relief = 7000 * 0.06."""
        res = evaluate_equal_weight(
            capital=FARAH.capital,
            marginal_tax_rate=FARAH.marginal_tax_rate,
        )
        expected = min(0.20 * 35_000.0, SSPN_RELIEF_CAP) * FARAH.marginal_tax_rate
        assert pytest.approx(res["tax_relief_rm"], rel=1e-9) == expected

    def test_weights_sum_to_one(self):
        """Equal-weight vector must sum to 1.0."""
        res = evaluate_equal_weight(capital=10_000.0, marginal_tax_rate=0.0)
        assert pytest.approx(res["weights"].sum(), abs=1e-10) == 1.0

    def test_all_weights_equal(self):
        """All weights must be 0.20."""
        res = evaluate_equal_weight(capital=10_000.0, marginal_tax_rate=0.0)
        np.testing.assert_allclose(res["weights"], np.full(5, 0.20), atol=1e-12)


# ============================================================================
# 2. Rule-Based AKPK Baseline
# ============================================================================

class TestRuleBased:
    """Verify AKPK cascade logic, constraint feasibility, and monetary amounts."""

    def test_all_personas_feasible(self):
        """Rule-based allocation must be feasible (zero violations) for ALL personas."""
        for name, persona in PERSONAS.items():
            res = evaluate_rule_based(
                capital=persona.capital,
                marginal_tax_rate=persona.marginal_tax_rate,
            )
            assert res["is_feasible"] is True, (
                f"Rule-based infeasible for {name}: "
                f"ASB_viol={res['asb_violation_rm']:.2f}, "
                f"GOP_viol={res['go_plus_violation_rm']:.2f}"
            )

    def test_all_personas_zero_asb_violation(self):
        """asb_violation_rm must be 0 for all three personas."""
        for name, persona in PERSONAS.items():
            res = evaluate_rule_based(
                capital=persona.capital,
                marginal_tax_rate=persona.marginal_tax_rate,
            )
            assert res["asb_violation_rm"] == 0.0, (
                f"ASB violation RM{res['asb_violation_rm']:.2f} for {name}"
            )

    def test_all_personas_zero_goplus_violation(self):
        """go_plus_violation_rm must be 0 for all three personas."""
        for name, persona in PERSONAS.items():
            res = evaluate_rule_based(
                capital=persona.capital,
                marginal_tax_rate=persona.marginal_tax_rate,
            )
            assert res["go_plus_violation_rm"] == 0.0, (
                f"GO+ violation RM{res['go_plus_violation_rm']:.2f} for {name}"
            )

    # ── Encik Razak: all four steps trigger their ceiling ──

    def test_encik_razak_asb_capped_at_300k(self):
        """ASB allocation must be exactly RM 300,000 (ceiling binding)."""
        res = evaluate_rule_based(
            capital=ENCIK_RAZAK.capital,
            marginal_tax_rate=ENCIK_RAZAK.marginal_tax_rate,
        )
        assert pytest.approx(res["allocations_rm"][_ASB], abs=1e-6) == 300_000.0

    def test_encik_razak_sspn_captures_8000(self):
        """SSPN allocation must be exactly RM 8,000 (tax-capture step)."""
        res = evaluate_rule_based(
            capital=ENCIK_RAZAK.capital,
            marginal_tax_rate=ENCIK_RAZAK.marginal_tax_rate,
        )
        assert pytest.approx(res["allocations_rm"][_SSPN], abs=1e-6) == 8_000.0

    def test_encik_razak_goplus_at_cap(self):
        """GO+ allocation must be exactly RM 20,000 (go_plus_cap binding)."""
        res = evaluate_rule_based(
            capital=ENCIK_RAZAK.capital,
            marginal_tax_rate=ENCIK_RAZAK.marginal_tax_rate,
        )
        assert pytest.approx(res["allocations_rm"][_GOP], abs=1e-6) == 20_000.0

    def test_encik_razak_th_receives_122k(self):
        """Remaining C = 450000 - 20000 - 8000 - 300000 = RM 122,000 → TH."""
        res = evaluate_rule_based(
            capital=ENCIK_RAZAK.capital,
            marginal_tax_rate=ENCIK_RAZAK.marginal_tax_rate,
        )
        # 450000 - 20000 (GO+) - 8000 (SSPN) - 300000 (ASB) = 122000
        assert pytest.approx(res["allocations_rm"][_TH], abs=1e-6) == 122_000.0

    def test_encik_razak_kwsp_zero(self):
        """KWSP must remain RM 0.00 (discretionary, held in reserve)."""
        res = evaluate_rule_based(
            capital=ENCIK_RAZAK.capital,
            marginal_tax_rate=ENCIK_RAZAK.marginal_tax_rate,
        )
        assert res["allocations_rm"][_KWSP] == 0.0

    def test_encik_razak_allocations_sum_to_capital(self):
        """Total monetary allocations must equal C = RM 450,000."""
        res = evaluate_rule_based(
            capital=ENCIK_RAZAK.capital,
            marginal_tax_rate=ENCIK_RAZAK.marginal_tax_rate,
        )
        assert pytest.approx(res["allocations_rm"].sum(), abs=1e-4) == ENCIK_RAZAK.capital

    # ── Aiman: small capital, zero tax ──

    def test_aiman_zero_sspn(self):
        """τ = 0 → SSPN allocation must be RM 0.00."""
        res = evaluate_rule_based(
            capital=AIMAN.capital,
            marginal_tax_rate=AIMAN.marginal_tax_rate,
        )
        assert res["allocations_rm"][_SSPN] == 0.0

    def test_aiman_goplus_15pct(self):
        """C = 5000 → GO+ = min(5000*0.15, 20000) = RM 750."""
        res = evaluate_rule_based(
            capital=AIMAN.capital,
            marginal_tax_rate=AIMAN.marginal_tax_rate,
        )
        assert pytest.approx(res["allocations_rm"][_GOP], abs=1e-6) == 750.0

    def test_weights_sum_to_one(self):
        """Rule-based weights must sum to 1.0 for all personas."""
        for name, persona in PERSONAS.items():
            res = evaluate_rule_based(
                capital=persona.capital,
                marginal_tax_rate=persona.marginal_tax_rate,
            )
            assert pytest.approx(res["weights"].sum(), abs=1e-9) == 1.0, (
                f"Weights don't sum to 1 for {name}: {res['weights'].sum()}"
            )


# ============================================================================
# 3. Multi-Objective Metrics
# ============================================================================

class TestHypervolume:
    """Verify calculate_hypervolume with known synthetic fronts."""

    def test_strictly_positive_for_valid_front(self):
        """HV must be > 0 for any front with f1, f2 > ref_point."""
        f1 = np.array([0.05, 0.04, 0.03])
        f2 = np.array([0.60, 0.70, 0.80])
        hv = calculate_hypervolume(f1, f2, ref_point=(0.0, 0.0))
        assert hv > 0.0, f"Expected HV > 0; got {hv}"

    def test_exact_staircase_value(self):
        """3-point staircase front: HV = 3*1 + 2*1 + 1*1 = 6."""
        f1 = np.array([3.0, 2.0, 1.0])
        f2 = np.array([1.0, 2.0, 3.0])
        hv = calculate_hypervolume(f1, f2, ref_point=(0.0, 0.0))
        assert pytest.approx(hv, rel=1e-9) == 6.0

    def test_single_point(self):
        """Single point (a, b): HV = a * b."""
        hv = calculate_hypervolume(
            np.array([0.05]), np.array([0.7]), ref_point=(0.0, 0.0)
        )
        assert pytest.approx(hv, rel=1e-9) == 0.05 * 0.7

    def test_empty_front_returns_zero(self):
        """Empty front → HV = 0."""
        hv = calculate_hypervolume(np.array([]), np.array([])  )
        assert hv == 0.0

    def test_hv_increases_with_better_front(self):
        """A Pareto front with higher f1 and f2 values dominates more space."""
        f1_weak   = np.array([0.03, 0.02, 0.01])
        f1_strong = np.array([0.06, 0.04, 0.02])
        f2 = np.array([0.4, 0.6, 0.8])
        hv_weak   = calculate_hypervolume(f1_weak,   f2)
        hv_strong = calculate_hypervolume(f1_strong, f2)
        assert hv_strong > hv_weak

    def test_unsorted_input_same_result(self):
        """HV must be identical regardless of input ordering."""
        f1 = np.array([3.0, 1.0, 2.0])
        f2 = np.array([1.0, 3.0, 2.0])
        hv = calculate_hypervolume(f1, f2, ref_point=(0.0, 0.0))
        assert pytest.approx(hv, rel=1e-9) == 6.0


class TestSpacing:
    """Verify calculate_spacing with known distance patterns."""

    def test_perfectly_equidistant_spacing_is_zero(self):
        """Equidistant points along a line → d_i all equal → S = 0."""
        # 5 equally-spaced points on a diagonal
        f1 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        f2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s = calculate_spacing(f1, f2)
        assert pytest.approx(s, abs=1e-9) == 0.0

    def test_single_point_returns_zero(self):
        """K = 1 → S = 0 (undefined spacing, returns 0 by convention)."""
        s = calculate_spacing(np.array([0.05]), np.array([0.7]))
        assert s == 0.0

    def test_two_points_returns_zero(self):
        """K = 2 → both points are nearest to each other → std = 0 → S = 0."""
        s = calculate_spacing(np.array([0.05, 0.03]), np.array([0.6, 0.8]))
        assert pytest.approx(s, abs=1e-9) == 0.0

    def test_non_equidistant_spacing_positive(self):
        """Non-uniform distribution → S > 0."""
        # Clustered at both ends, gap in the middle
        f1 = np.array([5.0, 4.9, 4.8, 1.2, 1.1, 1.0])
        f2 = np.array([1.0, 1.1, 1.2, 4.8, 4.9, 5.0])
        s = calculate_spacing(f1, f2)
        assert s > 0.0

    def test_spacing_non_negative(self):
        """Spacing metric must always be ≥ 0."""
        rng = np.random.default_rng(7)
        f1 = rng.uniform(0.01, 0.08, 20)
        f2 = rng.uniform(0.20, 0.95, 20)
        s = calculate_spacing(f1, f2)
        assert s >= 0.0


class TestCFR:
    """Verify calculate_cfr (Constraint Feasibility Rate)."""

    def test_all_feasible_gives_100_pct(self):
        """All violations = 0 → CFR = 100.0%."""
        v = np.zeros(50)
        assert calculate_cfr(v) == 100.0

    def test_all_infeasible_gives_0_pct(self):
        """All violations > 1e-5 → CFR = 0.0%."""
        v = np.ones(50) * 1.0  # RM 1.0 violation each
        assert calculate_cfr(v) == 0.0

    def test_half_feasible_gives_50_pct(self):
        """50 feasible out of 100 → CFR = 50%."""
        v = np.concatenate([np.zeros(50), np.ones(50)])
        assert pytest.approx(calculate_cfr(v), abs=1e-9) == 50.0

    def test_at_threshold_is_feasible(self):
        """Violation exactly at 1e-5 is considered feasible."""
        v = np.array([0.0, 1e-5, 2e-5])
        # 2 feasible out of 3 → 66.666...%
        assert pytest.approx(calculate_cfr(v), rel=1e-6) == 200.0 / 3.0

    def test_empty_array_returns_zero(self):
        """Empty violations array → CFR = 0.0."""
        assert calculate_cfr(np.array([])) == 0.0


# ============================================================================
# 4. Persona Registry
# ============================================================================

class TestPersonas:
    """Smoke-tests for the PERSONAS registry."""

    def test_all_three_personas_present(self):
        """PERSONAS dict must contain Aiman, Farah, and Encik Razak."""
        assert set(PERSONAS.keys()) == {"Aiman", "Farah", "Encik Razak"}

    def test_persona_capital_values(self):
        assert PERSONAS["Aiman"].capital       == 5_000.0
        assert PERSONAS["Farah"].capital       == 35_000.0
        assert PERSONAS["Encik Razak"].capital == 450_000.0

    def test_persona_tax_rates(self):
        assert PERSONAS["Aiman"].marginal_tax_rate       == 0.00
        assert PERSONAS["Farah"].marginal_tax_rate       == 0.06
        assert PERSONAS["Encik Razak"].marginal_tax_rate == 0.25

    def test_persona_is_frozen(self):
        """Persona must be immutable (frozen dataclass)."""
        with pytest.raises((AttributeError, TypeError)):
            PERSONAS["Aiman"].capital = 9_999.0  # type: ignore[misc]

    def test_persona_monthly_incomes(self):
        assert PERSONAS["Aiman"].monthly_income       == 1_200.0
        assert PERSONAS["Farah"].monthly_income       == 4_500.0
        assert PERSONAS["Encik Razak"].monthly_income == 18_000.0
