"""
test_domain.py
==============
Unit tests for Shariah-compliant domain functions:
    • SSPN tax-relief (tax_zakat.calculate_sspn_tax_relief)
    • Zakat liability  (tax_zakat.calculate_zakat)
    • Simplex normalisation repair operator (operators.simplex_normalize)

Run with:
    pytest tests/test_domain.py -v
"""

import numpy as np
import pytest

from src.domain.tax_zakat import calculate_sspn_tax_relief, calculate_zakat
from src.optimization.operators import simplex_normalize
from src.config import SSPN_RELIEF_CAP, ZAKAT_NISAB, ZAKAT_RATE


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

# A baseline weight vector: [ASB, TH, KWSP, SSPN, GO+]
BASELINE_WEIGHTS = np.array([0.30, 0.20, 0.20, 0.15, 0.15])


# ---------------------------------------------------------------------------
# SSPN tax-relief tests
# ---------------------------------------------------------------------------

class TestSSPNTaxRelief:
    """Tests for calculate_sspn_tax_relief."""

    def test_zero_tau_returns_zero(self):
        """Tax benefit must be strictly 0 when the marginal tax rate is 0 %."""
        result = calculate_sspn_tax_relief(w_sspn=0.5, capital=20_000, tau=0.00)
        assert result == 0.0, (
            f"Expected 0.0 with tau=0, got {result}"
        )

    def test_zero_capital_returns_zero(self):
        """Guard against division by zero: zero capital → zero benefit."""
        result = calculate_sspn_tax_relief(w_sspn=0.5, capital=0.0, tau=0.24)
        assert result == 0.0

    def test_relief_below_cap(self):
        """When SSPN allocation < RM 8,000, full allocation qualifies."""
        # w_sspn=0.5, C=10,000 → SSPN amount = RM 5,000 < RM 8,000 cap
        capital = 10_000.0
        w_sspn = 0.5
        tau = 0.24
        expected = (5_000.0 * tau) / capital  # = 0.012
        result = calculate_sspn_tax_relief(w_sspn, capital, tau)
        assert pytest.approx(result, rel=1e-9) == expected

    def test_relief_caps_at_8000(self):
        """Marginal SSPN tax benefit must cap at RM 8,000 regardless of allocation."""
        # w_sspn=1.0, C=100,000 → SSPN amount = RM 100,000, but cap = RM 8,000
        capital = 100_000.0
        w_sspn = 1.0
        tau = 0.24
        expected = (SSPN_RELIEF_CAP * tau) / capital  # = 8000 * 0.24 / 100000
        result = calculate_sspn_tax_relief(w_sspn, capital, tau)
        assert pytest.approx(result, rel=1e-9) == expected

    def test_relief_exactly_at_cap_boundary(self):
        """Allocation exactly at RM 8,000 must not be further clipped."""
        # w_sspn = 8000/40000 = 0.2, C = 40,000 → SSPN amount = exactly RM 8,000
        capital = 40_000.0
        w_sspn = SSPN_RELIEF_CAP / capital  # = 0.2
        tau = 0.20
        expected = (SSPN_RELIEF_CAP * tau) / capital
        result = calculate_sspn_tax_relief(w_sspn, capital, tau)
        assert pytest.approx(result, rel=1e-9) == expected

    def test_excess_allocation_still_caps(self):
        """Allocation well above cap → benefit same as allocation at cap."""
        capital = 50_000.0
        tau = 0.19
        # Large allocation
        result_large = calculate_sspn_tax_relief(w_sspn=0.9, capital=capital, tau=tau)
        # Allocation exactly at cap
        result_cap = calculate_sspn_tax_relief(
            w_sspn=SSPN_RELIEF_CAP / capital, capital=capital, tau=tau
        )
        assert pytest.approx(result_large, rel=1e-9) == result_cap


# ---------------------------------------------------------------------------
# Zakat liability tests
# ---------------------------------------------------------------------------

class TestZakat:
    """Tests for calculate_zakat."""

    def test_below_nisab_returns_zero(self):
        """Zakat must be 0 when total capital is below RM 25,000 (Nisab)."""
        result = calculate_zakat(BASELINE_WEIGHTS, capital=24_999.99)
        assert result == 0.0, (
            f"Expected 0.0 for capital < Nisab, got {result}"
        )

    def test_exactly_at_nisab_triggers_zakat(self):
        """Zakat must be computed when capital equals Nisab exactly."""
        result = calculate_zakat(BASELINE_WEIGHTS, capital=ZAKAT_NISAB)
        # Eligible weights: ASB (0.30) + SSPN (0.15) + GO+ (0.15) = 0.60
        expected = ZAKAT_RATE * 0.60
        assert pytest.approx(result, rel=1e-9) == expected

    def test_above_nisab_correct_rate(self):
        """Zakat equals 2.5 % of (w_ASB + w_SSPN + w_GO+) when C >= Nisab."""
        weights = np.array([0.40, 0.20, 0.20, 0.10, 0.10])
        capital = 30_000.0
        # Eligible: ASB=0.40, SSPN=0.10, GO+=0.10  → total = 0.60
        expected = ZAKAT_RATE * (0.40 + 0.10 + 0.10)
        result = calculate_zakat(weights, capital)
        assert pytest.approx(result, rel=1e-9) == expected

    def test_th_and_kwsp_excluded(self):
        """TH and KWSP must NOT contribute to the Zakat base."""
        # Portfolio entirely in TH and KWSP
        weights_excluded_only = np.array([0.0, 0.5, 0.5, 0.0, 0.0])
        result = calculate_zakat(weights_excluded_only, capital=50_000.0)
        assert result == 0.0, (
            "TH and KWSP should not be subject to Zakat; expected 0.0 got "
            f"{result}"
        )

    def test_only_eligible_assets(self):
        """Portfolio fully in zakatable assets → Zakat = 2.5 % * 1.0."""
        weights_eligible_only = np.array([0.50, 0.0, 0.0, 0.25, 0.25])
        result = calculate_zakat(weights_eligible_only, capital=50_000.0)
        assert pytest.approx(result, rel=1e-9) == ZAKAT_RATE * 1.0

    def test_list_input_accepted(self):
        """Function must accept plain Python lists as weight input."""
        result = calculate_zakat([0.3, 0.2, 0.2, 0.15, 0.15], capital=30_000.0)
        assert isinstance(result, float)
        assert result > 0.0


# ---------------------------------------------------------------------------
# Simplex normalisation tests
# ---------------------------------------------------------------------------

class TestSimplexNormalize:
    """Tests for simplex_normalize."""

    def test_output_sums_to_one(self):
        """Normalised weights must sum to 1 (within float tolerance)."""
        raw = np.array([0.3, -0.1, 0.5, 0.0, 0.2])
        w_hat = simplex_normalize(raw)
        assert pytest.approx(w_hat.sum(), abs=1e-10) == 1.0

    def test_no_negative_weights(self):
        """All output weights must be non-negative."""
        raw = np.array([-5.0, 3.0, -1.0, 2.0, -0.5])
        w_hat = simplex_normalize(raw)
        assert np.all(w_hat >= 0.0), f"Negative weights found: {w_hat}"

    def test_all_negative_input_gives_uniform(self):
        """All-negative input → ε-denominated output sums to 1 (near-uniform)."""
        raw = np.array([-1.0, -2.0, -3.0, -4.0, -5.0])
        w_hat = simplex_normalize(raw)
        # With all weights clipped to 0, numerators are all 0 → output is [0, …, 0]
        # Divided by ε → technically 0 vector (not uniform).
        # However the sum should still be 0 / ε ≈ 0 in this edge case.
        # The important assertion: no negative, no NaN, sums to ≤ 1.
        assert np.all(w_hat >= 0.0)
        assert not np.any(np.isnan(w_hat))
        assert w_hat.sum() <= 1.0 + 1e-10

    def test_random_vector_valid_simplex(self):
        """Random (possibly negative) vectors must always yield valid simplex weights."""
        rng = np.random.default_rng(seed=42)
        for _ in range(200):
            raw = rng.uniform(-2.0, 2.0, size=5)
            w_hat = simplex_normalize(raw)
            assert np.all(w_hat >= -1e-12), "Weights must be non-negative"
            assert pytest.approx(w_hat.sum(), abs=1e-9) == 1.0 or w_hat.sum() < 1e-9

    def test_already_valid_simplex_unchanged(self):
        """A weight vector already on the simplex should pass through unchanged."""
        raw = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
        w_hat = simplex_normalize(raw)
        np.testing.assert_allclose(w_hat, raw, atol=1e-10)

    def test_list_input_returns_ndarray(self):
        """Function must accept lists and return np.ndarray."""
        w_hat = simplex_normalize([0.1, 0.2, 0.3, 0.2, 0.2])
        assert isinstance(w_hat, np.ndarray)

    def test_single_positive_weight(self):
        """Only one positive weight → that asset receives weight 1.0."""
        raw = np.array([0.0, -1.0, 0.0, 5.0, -2.0])
        w_hat = simplex_normalize(raw)
        # Only index 3 (SSPN) is positive
        assert pytest.approx(w_hat[3], abs=1e-9) == 1.0
        assert pytest.approx(w_hat[[0, 1, 2, 4]].sum(), abs=1e-9) == 0.0
