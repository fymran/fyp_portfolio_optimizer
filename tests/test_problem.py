"""
test_problem.py
===============
Unit tests for Milestone 2:
    • SimplexRepair  — src/optimization/repair.py
    • ShariahPortfolioProblem — src/optimization/problem.py

Run with:
    pytest tests/test_problem.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.optimization.repair import SimplexRepair
from src.optimization.problem import ShariahPortfolioProblem
from src.config import ASSET_INDEX, RETURNS, LIQUIDITY, ZAKAT_RATE, SSPN_RELIEF_CAP

# Convenience index aliases
_ASB  = ASSET_INDEX["ASB"]
_TH   = ASSET_INDEX["TH"]
_KWSP = ASSET_INDEX["KWSP"]
_SSPN = ASSET_INDEX["SSPN"]
_GOP  = ASSET_INDEX["GO+"]

TOLERANCE = 1e-6  # sum-to-one tolerance for simplex assertions


# ============================================================================
# 1. SimplexRepair tests
# ============================================================================

class TestSimplexRepair:
    """Verify that SimplexRepair produces valid simplex rows for any input."""

    def _repaired(self, X_raw: np.ndarray) -> np.ndarray:
        """Helper: apply repair and return result."""
        return SimplexRepair()._do(problem=None, X=X_raw)

    def test_rows_sum_to_one(self):
        """Every repaired row must sum to 1.0 ± 1e-6."""
        rng = np.random.default_rng(seed=0)
        X_raw = rng.uniform(-3.0, 3.0, size=(10, 5))
        X_fixed = self._repaired(X_raw)
        row_sums = X_fixed.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(10), atol=TOLERANCE,
                                   err_msg="Row sums must equal 1.0")

    def test_no_negative_elements(self):
        """All repaired weights must be ≥ 0 (no short-selling)."""
        rng = np.random.default_rng(seed=1)
        X_raw = rng.uniform(-5.0, 5.0, size=(10, 5))
        X_fixed = self._repaired(X_raw)
        assert np.all(X_fixed >= -TOLERANCE), (
            f"Negative weights found: min={X_fixed.min():.6f}"
        )

    def test_output_shape_preserved(self):
        """Output shape must match input shape (10, 5)."""
        X_raw = np.ones((10, 5))
        X_fixed = self._repaired(X_raw)
        assert X_fixed.shape == (10, 5)

    def test_mixed_negative_and_positive(self):
        """Matrix with mix of negatives, zeros, and large positives → valid rows."""
        X_raw = np.array([
            [-0.5,  1.2,  0.3, -0.1,  0.8],
            [ 0.0,  0.0,  0.0,  0.0,  1.0],
            [ 0.2,  0.2,  0.2,  0.2,  0.2],
            [-3.0, -1.0, -2.0, -0.5, -0.1],  # all negative → near-zero output
            [ 5.0,  0.0,  0.0,  0.0,  0.0],  # single dominant → w≈1 for idx 0
        ])
        X_fixed = self._repaired(X_raw)
        # Rows 0-2 and 4 should sum to 1
        for i in [0, 1, 2, 4]:
            assert abs(X_fixed[i].sum() - 1.0) < TOLERANCE, (
                f"Row {i} sums to {X_fixed[i].sum():.8f}"
            )
        # All-negative row: all weights clipped to 0, sum ≈ 0 (degenerate)
        assert np.all(X_fixed[3] >= -TOLERANCE)

    def test_already_valid_simplex_unchanged(self):
        """A population already on the simplex passes through unchanged."""
        X_raw = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]] * 10, dtype=float)
        X_fixed = self._repaired(X_raw)
        np.testing.assert_allclose(X_fixed, X_raw, atol=TOLERANCE)

    def test_large_population(self):
        """Stress test: 500-individual population all produces valid rows."""
        rng = np.random.default_rng(seed=42)
        X_raw = rng.uniform(-10.0, 10.0, size=(500, 5))
        X_fixed = self._repaired(X_raw)
        # Rows with at least one positive entry must sum to 1
        has_positive = (X_raw > 0).any(axis=1)
        row_sums = X_fixed[has_positive].sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(has_positive.sum()),
                                   atol=TOLERANCE)


# ============================================================================
# 2. Objective Calculation tests
# ============================================================================

def _evaluate(problem: ShariahPortfolioProblem, x: np.ndarray) -> dict:
    """Helper: call _evaluate and return the out dict."""
    out: dict = {}
    problem._evaluate(x, out)
    return out


class TestObjectiveCalculation:
    """Verify f1 (net yield) and f2 (liquidity) under different scenarios."""

    # Baseline weight for these tests: equal allocation across all 5 assets
    w_equal = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    def test_no_tax_no_zakat_below_nisab(self):
        """C=5,000 (< Nisab), τ=0 → SSPN relief=0 and Zakat=0.

        f1 must equal the plain weighted return Σ wᵢ Rᵢ only.
        """
        prob = ShariahPortfolioProblem(capital=5_000.0, marginal_tax_rate=0.0)
        out = _evaluate(prob, self.w_equal)

        f1_neg, f2_neg = out["F"]
        f1 = -f1_neg  # un-negate

        expected_base = float(self.w_equal @ RETURNS)
        # No SSPN relief (tau=0), no Zakat (C < Nisab)
        assert pytest.approx(f1, rel=1e-9) == expected_base, (
            f"Expected f1={expected_base:.6f} (base only), got {f1:.6f}"
        )

    def test_liquidity_objective_correct(self):
        """f2 must equal Σ wᵢ Lᵢ for any valid weight vector."""
        prob = ShariahPortfolioProblem(capital=10_000.0, marginal_tax_rate=0.0)
        out = _evaluate(prob, self.w_equal)
        f2 = -out["F"][1]  # un-negate
        expected = float(self.w_equal @ LIQUIDITY)
        assert pytest.approx(f2, rel=1e-9) == expected

    def test_zakat_active_above_nisab(self):
        """C=35,000 (>= Nisab), τ=0.06 → Zakat deducted and SSPN relief added.

        We verify:
        1. Zakat deduction is positive (lowers f1 vs. base).
        2. SSPN relief is positive (raises f1 vs. base before Zakat).
        3. f1 = base_return + sspn_relief - zakat_charge.
        """
        C, tau = 35_000.0, 0.06
        w = np.array([0.30, 0.20, 0.20, 0.15, 0.15])  # simplex-valid

        prob = ShariahPortfolioProblem(capital=C, marginal_tax_rate=tau)
        out = _evaluate(prob, w)
        f1 = -out["F"][0]

        base_return = float(w @ RETURNS)

        # SSPN relief: min(0.15 * 35000, 8000) * 0.06 / 35000
        sspn_amount = min(w[_SSPN] * C, SSPN_RELIEF_CAP)
        expected_relief = (sspn_amount * tau) / C

        # Zakat: 2.5% * (w_ASB + w_SSPN + w_GO+)
        expected_zakat = ZAKAT_RATE * (w[_ASB] + w[_SSPN] + w[_GOP])

        expected_f1 = base_return + expected_relief - expected_zakat

        assert pytest.approx(f1, rel=1e-9) == expected_f1, (
            f"Expected f1={expected_f1:.8f}, got {f1:.8f}"
        )
        # Sanity: relief is positive, Zakat is positive
        assert expected_relief > 0.0
        assert expected_zakat > 0.0

    def test_sspn_relief_caps_at_8000(self):
        """SSPN relief term must not exceed 8000*tau/C regardless of w_SSPN."""
        C, tau = 100_000.0, 0.24
        # Push all weight into SSPN
        w_heavy_sspn = np.array([0.0, 0.0, 0.0, 1.0, 0.0])

        prob = ShariahPortfolioProblem(capital=C, marginal_tax_rate=tau)
        out = _evaluate(prob, w_heavy_sspn)
        f1 = -out["F"][0]

        # Relief must be capped: min(100000, 8000) * 0.24 / 100000 = 0.0192
        max_relief = (SSPN_RELIEF_CAP * tau) / C
        base_return = float(w_heavy_sspn @ RETURNS)
        # Zakat: SSPN is zakatable, C >= Nisab
        zakat = ZAKAT_RATE * w_heavy_sspn[_SSPN]
        expected_f1 = base_return + max_relief - zakat

        assert pytest.approx(f1, rel=1e-9) == expected_f1

    def test_objectives_are_negated_in_out(self):
        """pymoo minimises, so out['F'] must contain [-f1, -f2] (both ≤ 0 for
        positive objectives)."""
        prob = ShariahPortfolioProblem(capital=10_000.0, marginal_tax_rate=0.0)
        out = _evaluate(prob, self.w_equal)
        # f1 and f2 are both positive yield/liquidity → negated values ≤ 0
        assert out["F"][0] <= 0.0, "Negated f1 should be ≤ 0"
        assert out["F"][1] <= 0.0, "Negated f2 should be ≤ 0"


# ============================================================================
# 3. Constraint Violation tests
# ============================================================================

class TestConstraintViolations:
    """Verify g1 (ASB ceiling) and g2 (GO+ ceiling) constraint logic."""

    C = 450_000.0  # high capital to trigger ceiling violations easily

    def _prob(self) -> ShariahPortfolioProblem:
        return ShariahPortfolioProblem(capital=self.C, marginal_tax_rate=0.0)

    def test_asb_violation_g1_positive(self):
        """w_ASB=0.80 → ASB amount = RM360,000 > RM300,000 cap → g1 > 0."""
        # Build a simplex-valid weight with w_ASB=0.80
        w = np.array([0.80, 0.05, 0.05, 0.05, 0.05])
        prob = self._prob()
        out = _evaluate(prob, w)
        g1, g2 = out["G"]
        assert g1 > 0.0, (
            f"Expected g1 > 0 (ASB violation), got g1={g1:.6f}"
        )

    def test_goplus_violation_g2_positive(self):
        """w_GO+=0.10 → GO+ amount = RM45,000 > RM20,000 cap → g2 > 0."""
        w = np.array([0.30, 0.30, 0.20, 0.10, 0.10])
        prob = self._prob()
        out = _evaluate(prob, w)
        g1, g2 = out["G"]
        assert g2 > 0.0, (
            f"Expected g2 > 0 (GO+ violation), got g2={g2:.6f}"
        )

    def test_feasible_allocation_no_violation(self):
        """Allocation satisfying both caps must yield g1 ≤ 0 and g2 ≤ 0."""
        # w_ASB  ≤ 300000/450000 ≈ 0.6667
        # w_GO+  ≤  20000/450000 ≈ 0.0444
        w_asb_max  = 300_000.0 / self.C   # ≈ 0.6667
        w_gop_max  =  20_000.0 / self.C   # ≈ 0.0444
        # Remaining weight distributed to TH, KWSP, SSPN
        remainder  = 1.0 - w_asb_max - w_gop_max
        w = np.array([
            w_asb_max,
            remainder / 3,
            remainder / 3,
            remainder / 3,
            w_gop_max,
        ])
        # Re-normalise to account for float rounding
        w = w / w.sum()

        prob = self._prob()
        out = _evaluate(prob, w)
        g1, g2 = out["G"]
        assert g1 <= 1e-9, (
            f"Expected g1 ≤ 0 (feasible ASB), got g1={g1:.8f}"
        )
        assert g2 <= 1e-9, (
            f"Expected g2 ≤ 0 (feasible GO+), got g2={g2:.8f}"
        )

    def test_both_violations_simultaneously(self):
        """Simultaneously exceeding both caps → both g1 > 0 and g2 > 0."""
        # w_ASB=0.75 → RM337,500 > RM300,000 cap
        # w_GO+=0.15 → RM67,500  > RM20,000 cap
        w = np.array([0.75, 0.05, 0.05, 0.0, 0.15])
        prob = self._prob()
        out = _evaluate(prob, w)
        g1, g2 = out["G"]
        assert g1 > 0.0, f"Expected g1 > 0, got {g1:.6f}"
        assert g2 > 0.0, f"Expected g2 > 0, got {g2:.6f}"

    def test_exactly_at_asb_cap_feasible(self):
        """Allocation exactly at ASB cap (w_ASB = 300000/C) → g1 ≈ 0 (feasible)."""
        w_asb = 300_000.0 / self.C
        w_gop = 0.01  # well below GO+ cap
        rem = 1.0 - w_asb - w_gop
        w = np.array([w_asb, rem / 3, rem / 3, rem / 3, w_gop])
        w = w / w.sum()

        prob = self._prob()
        out = _evaluate(prob, w)
        g1, _ = out["G"]
        assert g1 <= 1e-8, (
            f"Expected g1 ≈ 0 at ASB cap boundary, got g1={g1:.8f}"
        )

    def test_constraint_scale_invariance(self):
        """g1 and g2 values are dimensionless (divided by C), not in MYR."""
        C = 450_000.0
        prob = ShariahPortfolioProblem(capital=C, marginal_tax_rate=0.0)
        w = np.array([0.80, 0.05, 0.05, 0.05, 0.05])
        out = _evaluate(prob, w)
        g1, _ = out["G"]
        # g1 = (0.80*450000 - 300000) / 450000 = 60000/450000 ≈ 0.1333
        expected_g1 = (0.80 * C - 300_000.0) / C
        assert pytest.approx(g1, rel=1e-9) == expected_g1


# ============================================================================
# 4. ShariahPortfolioProblem constructor validation tests
# ============================================================================

class TestProblemInit:
    """Verify that invalid constructor arguments raise appropriate errors."""

    def test_negative_capital_raises(self):
        with pytest.raises(ValueError, match="capital must be > 0"):
            ShariahPortfolioProblem(capital=-1.0, marginal_tax_rate=0.0)

    def test_zero_capital_raises(self):
        with pytest.raises(ValueError, match="capital must be > 0"):
            ShariahPortfolioProblem(capital=0.0, marginal_tax_rate=0.0)

    def test_tax_rate_above_30pct_raises(self):
        with pytest.raises(ValueError, match="marginal_tax_rate"):
            ShariahPortfolioProblem(capital=10_000.0, marginal_tax_rate=0.31)

    def test_custom_returns_and_liquidity(self):
        """Custom R and L vectors should override config defaults."""
        R_custom = np.array([0.10, 0.08, 0.07, 0.06, 0.05])
        L_custom = np.array([1.0,  0.9,  0.8,  0.7,  0.6])
        prob = ShariahPortfolioProblem(
            capital=10_000.0,
            marginal_tax_rate=0.0,
            expected_returns=R_custom,
            liquidity_scores=L_custom,
        )
        np.testing.assert_array_equal(prob.R, R_custom)
        np.testing.assert_array_equal(prob.L, L_custom)

    def test_problem_dimensions(self):
        """pymoo problem metadata must match specification."""
        prob = ShariahPortfolioProblem(capital=10_000.0, marginal_tax_rate=0.0)
        assert prob.n_var == 5
        assert prob.n_obj == 2
        assert prob.n_ieq_constr == 2
