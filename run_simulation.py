"""
run_simulation.py
=================
Top-level simulation pipeline for the Shariah-compliant retail portfolio
optimisation framework.

Iterates across all three benchmark personas (Aiman, Farah, Encik Razak),
runs NSGA-II, evaluates deterministic baselines, and prints a structured
Markdown comparison table for each persona.

Usage
-----
    python run_simulation.py
"""

from __future__ import annotations

import sys
import numpy as np

from src.evaluation.personas import PERSONAS
from src.optimization.problem import ShariahPortfolioProblem
from src.optimization.engine import run_nsga2
from src.baselines.equal_weight import evaluate_equal_weight
from src.baselines.rule_based import evaluate_rule_based
from src.evaluation.metrics import (
    calculate_hypervolume,
    calculate_spacing,
    calculate_cfr,
    extract_compromise_solution,
)
from src.config import ASSETS

# ── NSGA-II hyper-parameters ──────────────────────────────────────────────
POP_SIZE = 100
N_GEN    = 200
SEED     = 42

_ASSET_COLS = [f"w_{a}" for a in ASSETS]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _pct(x: float) -> str:
    """Format decimal as percentage string (e.g. 0.0525 → '5.250%')."""
    return f"{x * 100:.3f}%"


def _rm(x: float) -> str:
    """Format float as RM amount (e.g. 1234.56 → 'RM 1,234.56')."""
    return f"RM {x:,.2f}"


def _w(x: float) -> str:
    """Format weight as percentage (e.g. 0.20 → '20.00%')."""
    return f"{x * 100:.2f}%"


def _weights_str(w: np.ndarray) -> str:
    """Format weight vector as pipe-delimited string."""
    return " | ".join(_w(wi) for wi in w)


def _viol_str(asb_v: float, gop_v: float) -> str:
    """Format violation pair compactly."""
    if asb_v == 0.0 and gop_v == 0.0:
        return "None ✓"
    parts = []
    if asb_v > 0:
        parts.append(f"ASB +{_rm(asb_v)}")
    if gop_v > 0:
        parts.append(f"GO+ +{_rm(gop_v)}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Per-persona simulation
# ---------------------------------------------------------------------------

def run_persona(persona_name: str) -> None:
    """Run full NSGA-II + baseline evaluation for one persona and print tables."""
    persona = PERSONAS[persona_name]
    C   = persona.capital
    tau = persona.marginal_tax_rate

    print(f"\n{'='*80}")
    print(f"## Persona: {persona.name}  |  {persona.archetype}")
    print(f"   Capital: {_rm(C)}  |  Tax Rate: {_pct(tau)}  |  "
          f"Monthly Income: {_rm(persona.monthly_income)}")
    print(f"   {persona.description}")
    print(f"{'='*80}\n")

    # ── Run NSGA-II ───────────────────────────────────────────────────────
    print(f"Running NSGA-II (pop={POP_SIZE}, gen={N_GEN}, seed={SEED}) …")
    problem = ShariahPortfolioProblem(capital=C, marginal_tax_rate=tau)
    result  = run_nsga2(problem, pop_size=POP_SIZE, n_gen=N_GEN, seed=SEED,
                        verbose=False)

    K = result.weights.shape[0]
    print(f"  → {K} non-dominated Pareto solutions found.\n")

    # ── Pareto-front quality metrics ──────────────────────────────────────
    total_violations = result.asb_violations + result.go_plus_violations
    hv  = calculate_hypervolume(result.net_returns, result.liquidity_scores)
    sp  = calculate_spacing(result.net_returns, result.liquidity_scores)
    cfr = calculate_cfr(total_violations)

    # ── Extract anchor solutions ──────────────────────────────────────────
    compromise  = extract_compromise_solution(result)
    idx_max_ret = int(np.argmax(result.net_returns))
    idx_max_liq = int(np.argmax(result.liquidity_scores))

    def _nsga2_row(label: str, idx: int) -> dict:
        return {
            "strategy":        label,
            "weights":         result.weights[idx],
            "net_return":      result.net_returns[idx],
            "liquidity_score": result.liquidity_scores[idx],
            "tax_relief_rm":   result.tax_relief_rm[idx],
            "zakat_rm":        result.zakat_rm[idx],
            "asb_viol":        result.asb_violations[idx],
            "gop_viol":        result.go_plus_violations[idx],
            "hv":              hv,
            "spacing":         sp,
            "cfr":             cfr,
        }

    # ── Baselines ─────────────────────────────────────────────────────────
    ew = evaluate_equal_weight(C, tau)
    rb = evaluate_rule_based(C, tau)

    def _baseline_row(label: str, b: dict) -> dict:
        return {
            "strategy":        label,
            "weights":         b["weights"],
            "net_return":      b["net_return"],
            "liquidity_score": b["liquidity_score"],
            "tax_relief_rm":   b["tax_relief_rm"],
            "zakat_rm":        b["zakat_rm"],
            "asb_viol":        b["asb_violation_rm"],
            "gop_viol":        b["go_plus_violation_rm"],
            "hv":              None,
            "spacing":         None,
            "cfr":             None,
        }

    rows = [
        _nsga2_row("NSGA-II Compromise",    compromise["index"]),
        _nsga2_row("NSGA-II Max Return",    idx_max_ret),
        _nsga2_row("NSGA-II Max Liquidity", idx_max_liq),
        _baseline_row("Rule-Based (AKPK)",  rb),
        _baseline_row("Naïve 1/N",          ew),
    ]

    # ── Print Markdown table ──────────────────────────────────────────────
    _print_strategy_table(rows)
    _print_front_metrics(hv, sp, cfr, K)


def _print_strategy_table(rows: list[dict]) -> None:
    """Print the per-strategy comparison table in Markdown."""
    # Header
    header = (
        "|           Strategy           |  w_ASB  |   w_TH  | w__KWSP | w__SSPN |  w_GO+  | "
        "Net Return | Liquidity |   Tax Relief   |      Zakat     | Violations |"
    )
    sep = "|" + "|".join(["-" * max(3, len(h)) for h in header.split("|")[1:-1]]) + "|"
    print(header)
    print(sep)
    for r in rows:
        w = r["weights"]
        viol = _viol_str(r["asb_viol"], r["gop_viol"])
        print(
            f"| {r['strategy']:<28} "
            f"| {_w(w[0]):>7} "
            f"| {_w(w[1]):>7} "
            f"| {_w(w[2]):>7} "
            f"| {_w(w[3]):>7} "
            f"| {_w(w[4]):>7} "
            f"| {_pct(r['net_return']):>10} "
            f"|   {r['liquidity_score']:.4f}  "
            f"| {_rm(r['tax_relief_rm']):>14} "
            f"| {_rm(r['zakat_rm']):>14} "
            f"|   {viol}   |"
        )
    print()


def _print_front_metrics(hv: float, sp: float, cfr: float, K: int) -> None:
    """Print NSGA-II Pareto-front quality metrics."""
    print("### NSGA-II Front Quality Metrics\n")
    print(f"|              Metric               |     Value     |")
    print(f"|-----------------------------------|---------------|")
    print(f"| Pareto Solutions (K)              |      {K}      |")
    print(f"| Hypervolume (HV)                  |    {hv:.6f}   |")
    print(f"| Spacing (S)                       |   {sp:.6f}    |")
    print(f"| Constraint Feasibility Rate (CFR) |   {cfr:.1f}%  |")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "#" * 80)
    print("# Shariah-Compliant Retail Portfolio Optimisation — Simulation Report")
    print("# NSGA-II  |  pop=100  |  gen=200  |  seed=42")
    print("#" * 80)

    for persona_name in PERSONAS:
        run_persona(persona_name)

    print("\n" + "#" * 80)
    print("# Simulation complete.")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    main()
