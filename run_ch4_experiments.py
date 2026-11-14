"""
run_ch4_experiments.py
======================
Chapter 4 empirical parameter-tuning experiments and RO3 functional test-case
verification suite for the Shariah-compliant retail portfolio optimiser.

Experiments
-----------
1. Effect of Population Size Initialization (N_pop in [20, 50, 100, 150, 200])
   on Farah's benchmark profile (C = RM 35,000, tau = 0.06).
2. Effect of Maximum Generations (G_max in [25, 50, 100, 200, 300])
   on Farah's benchmark profile (C = RM 35,000, tau = 0.06).
3. Effect of Genetic Operator Rates (p_c, p_m) across four regimes on both
   Farah (C = RM 35,000, tau = 0.06) and Encik Razak (C = RM 450,000, tau = 0.25).
4. Functional Boundary & Constraint Test-Case Log (TC-01 through TC-06)
   covering Aiman, Farah, and Encik Razak.

Usage
-----
    python run_ch4_experiments.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

# Ensure project root is importable regardless of current working directory
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import (
    ASB_CEILING,
    ASSET_INDEX,
    GOPLUS_CEILING,
    SSPN_RELIEF_CAP,
    ZAKAT_NISAB,
)
from src.evaluation.metrics import (
    calculate_cfr,
    calculate_hypervolume,
    calculate_spacing,
)
from src.evaluation.personas import PERSONAS
from src.optimization.engine import OptimizationResult, run_nsga2
from src.optimization.problem import ShariahPortfolioProblem

_ASB = ASSET_INDEX["ASB"]
_SSPN = ASSET_INDEX["SSPN"]
_GOP = ASSET_INDEX["GO+"]

BASELINE_POP_SIZE = 100
BASELINE_N_GEN = 200
BASELINE_PC = 0.90
BASELINE_PM = 0.20
SEED = 42


def _run_benchmark(
    capital: float,
    marginal_tax_rate: float,
    pop_size: int = BASELINE_POP_SIZE,
    n_gen: int = BASELINE_N_GEN,
    prob_crossover: float = BASELINE_PC,
    prob_mutation: float = BASELINE_PM,
    seed: int = SEED,
) -> dict:
    """Instantiate problem, run timed NSGA-II, and compute front metrics."""
    problem = ShariahPortfolioProblem(
        capital=capital,
        marginal_tax_rate=marginal_tax_rate,
    )
    t0 = time.perf_counter()
    result = run_nsga2(
        problem,
        pop_size=pop_size,
        n_gen=n_gen,
        seed=seed,
        verbose=False,
        prob_crossover=prob_crossover,
        prob_mutation=prob_mutation,
    )
    elapsed_s = time.perf_counter() - t0

    total_violations = result.asb_violations + result.go_plus_violations
    k = int(result.weights.shape[0])
    hv = calculate_hypervolume(result.net_returns, result.liquidity_scores)
    spacing = calculate_spacing(result.net_returns, result.liquidity_scores)
    cfr = calculate_cfr(total_violations)

    return {
        "k": k,
        "hv": hv,
        "spacing": spacing,
        "cfr": cfr,
        "elapsed_s": elapsed_s,
        "result": result,
    }


def _print_markdown_table(headers: list[str], rows: list[list[str]], alignments: list[str] | None = None) -> None:
    """Print a cleanly aligned GitHub-flavored Markdown table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _format_cell(val: str, width: int, align: str) -> str:
        if align == "right":
            return val.rjust(width)
        if align == "center":
            return val.center(width)
        return val.ljust(width)

    if alignments is None:
        alignments = ["left"] * len(headers)

    header_line = "| " + " | ".join(_format_cell(h, widths[i], "center") for i, h in enumerate(headers)) + " |"
    sep_parts = []
    for i, w in enumerate(widths):
        if alignments[i] == "right":
            sep_parts.append("-" * (w - 1) + ":")
        elif alignments[i] == "center":
            sep_parts.append(":" + "-" * max(1, w - 2) + ":")
        else:
            sep_parts.append("-" * w)
    sep_line = "| " + " | ".join(sep_parts) + " |"

    print(header_line)
    print(sep_line)
    for row in rows:
        print("| " + " | ".join(_format_cell(row[i], widths[i], alignments[i]) for i in range(len(headers))) + " |")
    print()


def run_experiment_1() -> None:
    """Experiment 1: Effect of Population Size Initialization (N_pop)."""
    farah = PERSONAS["Farah"]
    pop_sizes = [20, 50, 100, 150, 200]

    print("## Experiment 1: Effect of Population Size Initialization ($N_{\\text{pop}}$)")
    print(
        f"- **Benchmark Profile:** {farah.name} (C = RM {farah.capital:,.2f}, "
        f"tau = {farah.marginal_tax_rate:.2f})"
    )
    print(
        f"- **Fixed Parameters:** n_gen = {BASELINE_N_GEN}, p_c = {BASELINE_PC:.2f}, "
        f"p_m = {BASELINE_PM:.2f}, seed = {SEED}\n"
    )

    rows: list[list[str]] = []
    for pop_size in pop_sizes:
        metrics = _run_benchmark(
            capital=farah.capital,
            marginal_tax_rate=farah.marginal_tax_rate,
            pop_size=pop_size,
            n_gen=BASELINE_N_GEN,
            prob_crossover=BASELINE_PC,
            prob_mutation=BASELINE_PM,
            seed=SEED,
        )
        rows.append([
            str(pop_size),
            str(metrics["k"]),
            f"{metrics['hv']:.6f}",
            f"{metrics['spacing']:.6f}",
            f"{metrics['cfr']:.2f}%",
            f"{metrics['elapsed_s']:.4f}",
        ])

    _print_markdown_table(
        headers=[
            "Population Size",
            "Pareto Solutions (K)",
            "Hypervolume (HV)",
            "Spacing (S)",
            "CFR (%)",
            "Modeling Time (seconds)",
        ],
        rows=rows,
        alignments=["right", "right", "right", "right", "right", "right"],
    )


def run_experiment_2() -> None:
    """Experiment 2: Effect of Maximum Generations (G_max)."""
    farah = PERSONAS["Farah"]
    generations = [25, 50, 100, 200, 300]

    print("## Experiment 2: Effect of Maximum Generations ($G_{\\max}$)")
    print(
        f"- **Benchmark Profile:** {farah.name} (C = RM {farah.capital:,.2f}, "
        f"tau = {farah.marginal_tax_rate:.2f})"
    )
    print(
        f"- **Fixed Parameters:** pop_size = {BASELINE_POP_SIZE}, p_c = {BASELINE_PC:.2f}, "
        f"p_m = {BASELINE_PM:.2f}, seed = {SEED}\n"
    )

    rows: list[list[str]] = []
    for n_gen in generations:
        metrics = _run_benchmark(
            capital=farah.capital,
            marginal_tax_rate=farah.marginal_tax_rate,
            pop_size=BASELINE_POP_SIZE,
            n_gen=n_gen,
            prob_crossover=BASELINE_PC,
            prob_mutation=BASELINE_PM,
            seed=SEED,
        )
        rows.append([
            str(n_gen),
            f"{metrics['hv']:.6f}",
            f"{metrics['spacing']:.6f}",
            f"{metrics['cfr']:.2f}%",
            f"{metrics['elapsed_s']:.4f}",
        ])

    _print_markdown_table(
        headers=[
            "Generations",
            "Hypervolume (HV)",
            "Spacing (S)",
            "CFR (%)",
            "Modeling Time (seconds)",
        ],
        rows=rows,
        alignments=["right", "right", "right", "right", "right"],
    )


def run_experiment_3() -> None:
    """Experiment 3: Effect of Genetic Operator Rates (p_c and p_m)."""
    operator_configs = [
        ("Low Exploration (p_c = 0.60, p_m = 0.05)", 0.60, 0.05),
        ("Moderate (p_c = 0.80, p_m = 0.10)", 0.80, 0.10),
        ("Proposed Baseline (p_c = 0.90, p_m = 0.20)", 0.90, 0.20),
        ("High Perturbation (p_c = 0.95, p_m = 0.40)", 0.95, 0.40),
    ]

    benchmarks = [
        ("3A", PERSONAS["Farah"]),
        ("3B", PERSONAS["Encik Razak"]),
    ]

    print("## Experiment 3: Effect of Genetic Operator Rates ($p_c$ and $p_m$)")
    print(
        f"- **Fixed Parameters:** pop_size = {BASELINE_POP_SIZE}, "
        f"n_gen = {BASELINE_N_GEN}, seed = {SEED}\n"
    )

    for label, persona in benchmarks:
        print(
            f"### Experiment {label} — {persona.name} "
            f"(C = RM {persona.capital:,.2f}, tau = {persona.marginal_tax_rate:.2f})\n"
        )
        rows: list[list[str]] = []
        for config_name, pc, pm in operator_configs:
            metrics = _run_benchmark(
                capital=persona.capital,
                marginal_tax_rate=persona.marginal_tax_rate,
                pop_size=BASELINE_POP_SIZE,
                n_gen=BASELINE_N_GEN,
                prob_crossover=pc,
                prob_mutation=pm,
                seed=SEED,
            )
            rows.append([
                config_name,
                f"{metrics['hv']:.6f}",
                f"{metrics['spacing']:.6f}",
                f"{metrics['cfr']:.2f}%",
                f"{metrics['elapsed_s']:.4f}",
            ])

        _print_markdown_table(
            headers=[
                "Operator Config",
                "Hypervolume (HV)",
                "Spacing (S)",
                "CFR (%)",
                "Modeling Time (seconds)",
            ],
            rows=rows,
            alignments=["left", "right", "right", "right", "right"],
        )


def run_experiment_4() -> bool:
    """Experiment 4: Functional Boundary & Constraint Test-Case Log (RO3 Verification)."""
    print("## Experiment 4: Functional Boundary & Constraint Test-Case Log (RO3 Verification)")
    print(
        f"- **Engine Configuration:** pop_size = {BASELINE_POP_SIZE}, "
        f"n_gen = {BASELINE_N_GEN}, p_c = {BASELINE_PC:.2f}, "
        f"p_m = {BASELINE_PM:.2f}, seed = {SEED}\n"
    )

    aiman = PERSONAS["Aiman"]
    farah = PERSONAS["Farah"]
    razak = PERSONAS["Encik Razak"]

    res_aiman: OptimizationResult = _run_benchmark(
        capital=aiman.capital, marginal_tax_rate=aiman.marginal_tax_rate
    )["result"]
    res_farah: OptimizationResult = _run_benchmark(
        capital=farah.capital, marginal_tax_rate=farah.marginal_tax_rate
    )["result"]
    res_razak: OptimizationResult = _run_benchmark(
        capital=razak.capital, marginal_tax_rate=razak.marginal_tax_rate
    )["result"]

    all_weights = np.vstack([res_aiman.weights, res_farah.weights, res_razak.weights])
    max_sum_dev = float(np.max(np.abs(all_weights.sum(axis=1) - 1.0)))
    min_weight = float(np.min(all_weights))
    tc01_pass = (max_sum_dev <= 1e-6) and (min_weight >= -1e-12)

    max_relief_aiman = float(np.max(res_aiman.tax_relief_rm))
    max_sspn_aiman = float(np.max(res_aiman.allocations_rm[:, _SSPN]))
    tc02_pass = abs(max_relief_aiman) <= 1e-9

    max_zakat_aiman = float(np.max(res_aiman.zakat_rm))
    tc03_pass = abs(max_zakat_aiman) <= 1e-9

    max_relief_farah = float(np.max(res_farah.tax_relief_rm))
    expected_farah_cap = SSPN_RELIEF_CAP * farah.marginal_tax_rate
    zakatable_farah = (
        res_farah.weights[:, _ASB]
        + res_farah.weights[:, _SSPN]
        + res_farah.weights[:, _GOP]
    ) > 1e-6
    zakat_active_farah = bool(
        zakatable_farah.any() and np.all(res_farah.zakat_rm[zakatable_farah] > 0.0)
    )
    min_zakat_farah_nz = (
        float(np.min(res_farah.zakat_rm[zakatable_farah])) if zakatable_farah.any() else 0.0
    )
    max_zakat_farah = float(np.max(res_farah.zakat_rm))
    tc04_pass = (max_relief_farah <= expected_farah_cap + 1e-6) and zakat_active_farah

    max_gop_razak = float(np.max(res_razak.allocations_rm[:, _GOP]))
    max_gop_viol_razak = float(np.max(res_razak.go_plus_violations))
    tc05_pass = (max_gop_razak <= GOPLUS_CEILING + 1e-4) and (max_gop_viol_razak <= 1e-5)

    max_asb_razak = float(np.max(res_razak.allocations_rm[:, _ASB]))
    max_asb_viol_razak = float(np.max(res_razak.asb_violations))
    tc06_pass = (max_asb_razak <= ASB_CEILING + 1e-4) and (max_asb_viol_razak <= 1e-5)

    test_cases = [
        [
            "TC-01",
            "Simplex Weight Summation (sum w_i = 1.0, w_i >= 0)",
            f"All Pareto fronts (K_total = {all_weights.shape[0]})",
            "|sum(w_i) - 1.0| <= 1e-6 and min(w_i) >= 0",
            f"max|sum(w)-1| = {max_sum_dev:.2e}, min(w) = {min_weight:.2e}",
            "PASS" if tc01_pass else "FAIL",
        ],
        [
            "TC-02",
            "Zero-Tax SSPN Avoidance",
            f"Aiman: C = RM {aiman.capital:,.0f}, tau = {aiman.marginal_tax_rate:.2f}",
            "Tax Relief = RM 0.00",
            f"Max Tax Relief = RM {max_relief_aiman:.2f} (Max RM_SSPN = RM {max_sspn_aiman:.2f})",
            "PASS" if tc02_pass else "FAIL",
        ],
        [
            "TC-03",
            "Nisab Zakat Exemption",
            f"Aiman: C = RM {aiman.capital:,.0f} < RM {ZAKAT_NISAB:,.0f}",
            "Zakat = RM 0.00",
            f"Max Zakat = RM {max_zakat_aiman:.2f}",
            "PASS" if tc03_pass else "FAIL",
        ],
        [
            "TC-04",
            "Active Nisab & SSPN Relief Cap",
            f"Farah: C = RM {farah.capital:,.0f}, tau = {farah.marginal_tax_rate:.2f}",
            f"Tax Relief <= RM {expected_farah_cap:,.2f}, Zakat > RM 0.00",
            (
                f"Max Relief = RM {max_relief_farah:.2f}, "
                f"Zakat in [RM {min_zakat_farah_nz:.2f}, RM {max_zakat_farah:.2f}]"
            ),
            "PASS" if tc04_pass else "FAIL",
        ],
        [
            "TC-05",
            "TNG GO+ Statutory Wallet Ceiling",
            f"Encik Razak: C = RM {razak.capital:,.0f}, tau = {razak.marginal_tax_rate:.2f}",
            f"RM_GO+ <= RM {GOPLUS_CEILING:,.2f}",
            f"Max RM_GO+ = RM {max_gop_razak:,.2f} (Viol = RM {max_gop_viol_razak:.4f})",
            "PASS" if tc05_pass else "FAIL",
        ],
        [
            "TC-06",
            "ASB Statutory Holding Limit",
            f"Encik Razak: C = RM {razak.capital:,.0f}, tau = {razak.marginal_tax_rate:.2f}",
            f"RM_ASB <= RM {ASB_CEILING:,.2f}",
            f"Max RM_ASB = RM {max_asb_razak:,.2f} (Viol = RM {max_asb_viol_razak:.4f})",
            "PASS" if tc06_pass else "FAIL",
        ],
    ]

    _print_markdown_table(
        headers=[
            "Test ID",
            "Test Case Description",
            "Input Parameters",
            "Expected Output",
            "Actual Output",
            "Status",
        ],
        rows=test_cases,
        alignments=["center", "left", "left", "left", "left", "center"],
    )

    all_passed = all(row[-1] == "PASS" for row in test_cases)
    return all_passed


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("# Chapter 4 Empirical Results: Parameter Tuning & Functional Test-Case Suite\n")

    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
    all_passed = run_experiment_4()

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
