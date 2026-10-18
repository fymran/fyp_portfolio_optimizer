"""
streamlit_app.py
================
Interactive Streamlit decision-support dashboard for the Shariah-compliant
retail portfolio optimisation framework.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys

# ── Ensure project root is on sys.path so src.* imports work ──────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import ASSETS, RETURNS, LIQUIDITY, ZAKAT_NISAB
from src.optimization.problem import ShariahPortfolioProblem
from src.optimization.engine import run_nsga2, OptimizationResult
from src.baselines.equal_weight import evaluate_equal_weight
from src.baselines.rule_based import evaluate_rule_based
from src.evaluation.personas import PERSONAS
from src.evaluation.metrics import (
    calculate_hypervolume,
    calculate_spacing,
    calculate_cfr,
    extract_compromise_solution,
)

# ── Page configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shariah Portfolio Optimizer",
    layout="wide",
    page_icon="📈",
)

# ── Constants ──────────────────────────────────────────────────────────────
_ASSET_LABELS = ["ASB", "TH", "KWSP", "SSPN", "GO+"]
_ASSET_COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]

_PERSONA_META = {
    "Custom Input": None,
    "Aiman (Student / B40)":        ("Aiman",       5_000.0,   0.00, 1_200.0),
    "Farah (Executive / M40)":      ("Farah",       35_000.0,  0.06, 4_500.0),
    "Encik Razak (Senior Mgr / T20)": ("Encik Razak", 450_000.0, 0.25, 18_000.0),
}

# Malaysian LHDN marginal tax brackets (YA 2024, chargeable income)
_LHDN_BRACKETS = [
    (5_000,     0.00),
    (20_000,    0.01),
    (35_000,    0.03),
    (50_000,    0.08),
    (70_000,    0.13),
    (100_000,   0.21),
    (250_000,   0.24),
    (400_000,   0.245),
    (600_000,   0.25),
    (1_000_000, 0.26),
    (float("inf"), 0.28),
]


def _suggest_tax_rate(monthly_income: float) -> float:
    """Return the LHDN marginal rate for a given monthly gross income."""
    annual = monthly_income * 12
    for ceiling, rate in _LHDN_BRACKETS:
        if annual <= ceiling:
            return rate
    return 0.28


# ── Cached optimisation ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _run_cached(
    capital: float,
    tau: float,
    pop_size: int,
    n_gen: int,
    seed: int,
    nisab: float,
) -> dict:
    """Run NSGA-II and both baselines; return plain Python/numpy dict."""
    prob   = ShariahPortfolioProblem(
        capital=capital, marginal_tax_rate=tau, nisab=nisab
    )
    result = run_nsga2(prob, pop_size=pop_size, n_gen=n_gen, seed=seed,
                       verbose=False)
    ew = evaluate_equal_weight(capital, tau, nisab=nisab)
    rb = evaluate_rule_based(capital, tau, nisab=nisab)
    return {"result": result, "ew": ew, "rb": rb}


# ── Formatting helpers ─────────────────────────────────────────────────────

def _pct(x: float, dp: int = 3) -> str:
    return f"{x * 100:.{dp}f}%"

def _rm(x: float) -> str:
    return f"RM {x:,.2f}"

def _feasible_badge(ok: bool) -> str:
    return "✅ Feasible" if ok else "❌ Infeasible"


# ===========================================================================
# HEADER
# ===========================================================================

st.markdown(
    """
    <h1 style='margin-bottom:0'>📈 Shariah-Compliant Portfolio Optimizer</h1>
    <p style='color:#666; font-size:1.05rem; margin-top:4px'>
        Multi-objective NSGA-II · BNM & ASNB Statutory Compliance · Zakat-Purified Yield
    </p>
    """,
    unsafe_allow_html=True,
)

badge_cols = st.columns([1,1,1])
badge_cols[0].markdown(
    "<span style='background:#1f77b4;color:white;padding:3px 10px;"
    "border-radius:12px;font-size:0.8rem'>⚙ NSGA-II Powered</span>",
    unsafe_allow_html=True,
)
badge_cols[1].markdown(
    "<span style='background:#2ca02c;color:white;padding:3px 10px;"
    "border-radius:12px;font-size:0.8rem'>🏦 BNM &amp; ASNB Compliant</span>",
    unsafe_allow_html=True,
)
badge_cols[2].markdown(
    "<span style='background:#9467bd;color:white;padding:3px 10px;"
    "border-radius:12px;font-size:0.8rem'>☪ Zakat-Purified</span>",
    unsafe_allow_html=True,
)
st.divider()


# ===========================================================================
# SIDEBAR
# ===========================================================================

with st.sidebar:
    st.header("⚙️ Configuration")

    # ── Persona preset ──────────────────────────────────────────────────────
    st.subheader("Investor Persona Preset")
    preset_key = st.selectbox(
        "Select Preset",
        options=list(_PERSONA_META.keys()),
        index=0,
    )
    preset = _PERSONA_META[preset_key]

    st.divider()

    # ── Financial parameters ────────────────────────────────────────────────
    st.subheader("Financial Parameters")

    default_capital = float(preset[1]) if preset else 50_000.0
    default_income  = float(preset[3]) if preset else 5_000.0

    capital_col, _ = st.columns([3, 1])
    capital = st.number_input(
        "Investable Capital C (RM)",
        min_value=1_000.0,
        max_value=1_000_000.0,
        value=default_capital,
        step=1_000.0,
        format="%.0f",
        help="Total amount to invest across all 5 instruments.",
    )
    capital = float(
        st.slider(
            "Capital slider (RM)",
            min_value=1_000,
            max_value=1_000_000,
            value=int(capital),
            step=1_000,
            label_visibility="collapsed",
        )
    )

    monthly_income = st.number_input(
        "Monthly Gross Income (RM)",
        min_value=0.0,
        max_value=100_000.0,
        value=default_income,
        step=100.0,
        format="%.0f",
    )
    suggested_tau = _suggest_tax_rate(monthly_income)

    default_tau = float(preset[2]) if preset else suggested_tau
    tau_pct = st.slider(
        f"Marginal Tax Rate τ (%) — LHDN suggested: {_pct(suggested_tau, 1)}",
        min_value=0.0,
        max_value=30.0,
        value=round(default_tau * 100, 1),
        step=0.5,
        help="Override the auto-suggested LHDN marginal rate.",
    )
    tau = tau_pct / 100.0

    nisab = st.number_input(
        "Zakat Nisab Threshold (RM)",
        min_value=1_000.0,
        max_value=100_000.0,
        value=float(ZAKAT_NISAB),
        step=500.0,
        format="%.0f",
    )

    st.divider()

    # ── TOPSIS preference ───────────────────────────────────────────────────
    st.subheader("Investor Preference (TOPSIS)")
    w_liq = st.slider(
        "Liquidity Preference wL",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="wL = weight on liquidity; wR = 1 − wL = weight on net return.",
    )
    w_ret = 1.0 - w_liq
    st.caption(f"Net Return preference wR = **{w_ret:.2f}**  |  Liquidity preference wL = **{w_liq:.2f}**")

    st.divider()

    # ── Evolutionary hyperparameters ────────────────────────────────────────
    with st.expander("🧬 Evolutionary Hyperparameters"):
        pop_size = st.number_input("Population Size", min_value=20, max_value=500,
                                   value=100, step=10)
        n_gen    = st.number_input("Generations",     min_value=50, max_value=1000,
                                   value=200, step=50)
        seed     = st.number_input("Random Seed",     min_value=0,  max_value=9999,
                                   value=42,  step=1)

    st.divider()

    # ── Run button ──────────────────────────────────────────────────────────
    run_clicked = st.button("🚀 Run Optimization", type="primary", use_container_width=True)


# ===========================================================================
# OPTIMIZATION (triggered by button OR first load)
# ===========================================================================

# Use session_state to persist results across widget interactions
if "opt_data" not in st.session_state:
    st.session_state["opt_data"] = None

if run_clicked:
    with st.spinner("Running NSGA-II optimisation… this may take ~15–30 s"):
        st.session_state["opt_data"] = _run_cached(
            capital=capital,
            tau=tau,
            pop_size=int(pop_size),
            n_gen=int(n_gen),
            seed=int(seed),
            nisab=nisab,
        )
        st.session_state["opt_params"] = dict(
            capital=capital, tau=tau, w_liq=w_liq, w_ret=w_ret
        )

if st.session_state["opt_data"] is None:
    st.info(
        "👈 Configure your parameters in the sidebar and click **Run Optimization** to begin.",
        icon="📊",
    )
    st.stop()

# ── Unpack cached results ──────────────────────────────────────────────────
data:   dict            = st.session_state["opt_data"]
result: OptimizationResult = data["result"]
ew:     dict            = data["ew"]
rb:     dict            = data["rb"]

# Retrieve params used for this run
run_params = st.session_state.get("opt_params", {})
_C   = run_params.get("capital", capital)
_tau = run_params.get("tau", tau)

# ── Extract key solutions ─────────────────────────────────────────────────
compromise  = extract_compromise_solution(result, weights_preference=(w_ret, w_liq))
idx_max_ret = int(np.argmax(result.net_returns))
idx_max_liq = int(np.argmax(result.liquidity_scores))

comp_w   = compromise["weights"]
comp_f1  = compromise["net_return"]
comp_f2  = compromise["liquidity_score"]
comp_tax = compromise["tax_relief_rm"]
comp_zak = compromise["zakat_rm"]
comp_asb_v = float(result.asb_violations[compromise["index"]])
comp_gop_v = float(result.go_plus_violations[compromise["index"]])
comp_feasible = (comp_asb_v < 1e-4) and (comp_gop_v < 1e-4)

# Pareto-front quality metrics
total_viol = result.asb_violations + result.go_plus_violations
hv  = calculate_hypervolume(result.net_returns, result.liquidity_scores)
sp  = calculate_spacing(result.net_returns, result.liquidity_scores)
cfr = calculate_cfr(total_viol)
K   = result.weights.shape[0]


# ===========================================================================
# TOP KPI CARDS
# ===========================================================================

st.subheader("🏆 TOPSIS-Selected Portfolio — Key Metrics")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

kpi1.metric(
    "Net Return f₁ (% p.a.)",
    f"{comp_f1 * 100:.3f}%",
    help="Net annualised yield after SSPN tax relief and Zakat deduction.",
)
kpi2.metric(
    "Liquidity Score f₂",
    f"{comp_f2:.4f}",
    help="Weighted average liquidity index across the 5 instruments.",
)
kpi3.metric(
    "Annual Tax Relief",
    f"RM {comp_tax:,.2f}",
    help="Absolute SSPN income-tax saving for the year.",
)
kpi4.metric(
    "Annual Zakat Liability",
    f"RM {comp_zak:,.2f}",
    help="2.5% Zakat on ASB + SSPN + GO+ holdings (if C ≥ Nisab).",
)
kpi5.metric(
    "Statutory Compliance",
    _feasible_badge(comp_feasible),
    help="Both ASB ≤ RM300k and GO+ ≤ RM20k must be satisfied.",
)

st.divider()


# ===========================================================================
# PANEL A — Interactive Pareto Frontier
# ===========================================================================

st.subheader("Panel A · Pareto Frontier (Return vs Liquidity)")

def _hover_text(w: np.ndarray, f1: float, f2: float) -> str:
    lines = [f"<b>Net Yield:</b> {f1*100:.3f}%  |  <b>Liquidity:</b> {f2:.4f}"]
    for i, label in enumerate(_ASSET_LABELS):
        lines.append(f"  {label}: {w[i]*100:.1f}%")
    return "<br>".join(lines)

fig_pareto = go.Figure()

# ── NSGA-II front points ───────────────────────────────────────────────────
hover_texts = [
    _hover_text(result.weights[i], result.net_returns[i], result.liquidity_scores[i])
    for i in range(K)
]
fig_pareto.add_trace(go.Scatter(
    x=result.liquidity_scores,
    y=result.net_returns * 100,
    mode="lines+markers",
    name="NSGA-II Pareto Front",
    marker=dict(size=7, color="#1f77b4", opacity=0.7),
    line=dict(color="#1f77b4", width=1.5),
    hovertemplate="%{customdata}<extra></extra>",
    customdata=hover_texts,
))

# ── TOPSIS compromise (gold star) ─────────────────────────────────────────
fig_pareto.add_trace(go.Scatter(
    x=[comp_f2],
    y=[comp_f1 * 100],
    mode="markers",
    name="TOPSIS Compromise",
    marker=dict(size=18, symbol="star", color="gold",
                line=dict(color="#333", width=1.5)),
    hovertemplate=(
        f"<b>TOPSIS Compromise</b><br>"
        f"Net Yield: {comp_f1*100:.3f}%<br>"
        f"Liquidity: {comp_f2:.4f}<extra></extra>"
    ),
))

# ── Rule-based (orange diamond) ───────────────────────────────────────────
fig_pareto.add_trace(go.Scatter(
    x=[rb["liquidity_score"]],
    y=[rb["net_return"] * 100],
    mode="markers",
    name="Rule-Based (AKPK)",
    marker=dict(size=14, symbol="diamond", color="orange",
                line=dict(color="#333", width=1.5)),
    hovertemplate=(
        f"<b>Rule-Based (AKPK)</b><br>"
        f"Net Yield: {rb['net_return']*100:.3f}%<br>"
        f"Liquidity: {rb['liquidity_score']:.4f}"
        + ("<br>✅ Feasible" if rb["is_feasible"] else "<br>❌ Infeasible")
        + "<extra></extra>"
    ),
))

# ── Naïve 1/N (red cross) ─────────────────────────────────────────────────
viol_note = ""
if not ew["is_feasible"]:
    viol_note = (
        f"<br>⚠ GO+ excess: RM {ew['go_plus_violation_rm']:,.0f}"
        if ew["go_plus_violation_rm"] > 0 else ""
    )
    viol_note += (
        f"<br>⚠ ASB excess: RM {ew['asb_violation_rm']:,.0f}"
        if ew["asb_violation_rm"] > 0 else ""
    )

fig_pareto.add_trace(go.Scatter(
    x=[ew["liquidity_score"]],
    y=[ew["net_return"] * 100],
    mode="markers",
    name="Naïve 1/N",
    marker=dict(size=14, symbol="x", color="red",
                line=dict(color="#900", width=2.5)),
    hovertemplate=(
        f"<b>Naïve 1/N</b><br>"
        f"Net Yield: {ew['net_return']*100:.3f}%<br>"
        f"Liquidity: {ew['liquidity_score']:.4f}"
        + viol_note
        + ("<br>❌ Infeasible" if not ew["is_feasible"] else "<br>✅ Feasible")
        + "<extra></extra>"
    ),
))

fig_pareto.update_layout(
    xaxis_title="Aggregate Liquidity Score f₂",
    yaxis_title="Net Annualised Return f₁ (% p.a.)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="closest",
    height=480,
    margin=dict(l=60, r=30, t=40, b=60),
    plot_bgcolor="#fafafa",
)

st.plotly_chart(fig_pareto, use_container_width=True)

st.divider()


# ===========================================================================
# PANEL B — Portfolio Breakdown & Asset Allocation
# ===========================================================================

st.subheader("Panel B · TOPSIS Portfolio Breakdown")
col_donut, col_bar = st.columns(2)

# ── Donut chart ────────────────────────────────────────────────────────────
with col_donut:
    st.markdown("**Asset Weight Allocation (%)**")
    fig_donut = go.Figure(go.Pie(
        labels=_ASSET_LABELS,
        values=comp_w * 100,
        hole=0.55,
        marker=dict(colors=_ASSET_COLORS),
        texttemplate="%{label}<br>%{percent:.1%}",
        hovertemplate="%{label}: %{value:.2f}%<extra></extra>",
    ))
    fig_donut.add_annotation(
        text=f"<b>RM {_C:,.0f}</b><br>Total Capital",
        x=0.5, y=0.5,
        font=dict(size=13, color="#333"),
        showarrow=False,
    )
    fig_donut.update_layout(
        showlegend=True,
        legend=dict(orientation="v", x=1.0, y=0.5),
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_donut, use_container_width=True)

# ── Horizontal bar chart vs statutory limits ───────────────────────────────
with col_bar:
    st.markdown("**Ringgit Allocation vs Statutory Limits**")
    alloc_rm   = comp_w * _C
    limit_vals = [300_000.0, None, None, 8_000.0, 20_000.0]  # None = no cap
    limit_labels = ["ASB cap", None, None, "SSPN tax zone", "GO+ cap"]

    bar_fig = go.Figure()

    bar_fig.add_trace(go.Bar(
        y=_ASSET_LABELS,
        x=alloc_rm,
        orientation="h",
        name="Allocated (RM)",
        marker_color=_ASSET_COLORS,
        text=[f"RM {v:,.0f}" for v in alloc_rm],
        textposition="outside",
        hovertemplate="%{y}: RM %{x:,.0f}<extra></extra>",
    ))

    # Overlay limit lines as scatter markers
    for i, (lv, ll) in enumerate(zip(limit_vals, limit_labels)):
        if lv is not None:
            bar_fig.add_shape(
                type="line",
                x0=lv, x1=lv, y0=i - 0.4, y1=i + 0.4,
                line=dict(color="red", width=2.5, dash="dash"),
            )
            bar_fig.add_annotation(
                x=lv, y=i,
                text=f" {ll}<br>RM {lv:,.0f}",
                showarrow=False,
                xanchor="left",
                font=dict(size=10, color="red"),
            )

    bar_fig.update_layout(
        xaxis_title="Allocation (RM)",
        yaxis=dict(autorange="reversed"),
        height=380,
        margin=dict(l=60, r=120, t=30, b=50),
        plot_bgcolor="#fafafa",
        showlegend=False,
    )
    st.plotly_chart(bar_fig, use_container_width=True)

st.divider()


# ===========================================================================
# PANEL C — Comprehensive Strategy Benchmarking Table
# ===========================================================================

st.subheader("Panel C · Strategy Comparison Table")

def _build_row(label: str, w: np.ndarray, f1: float, f2: float,
               tax_rm: float, zak_rm: float, asb_v: float, gop_v: float) -> dict:
    row = {"Strategy": label}
    alloc = w * _C
    for i, a in enumerate(_ASSET_LABELS):
        row[f"w_{a} (%)"] = f"{w[i]*100:.1f}%"
        row[f"RM_{a}"]    = f"RM {alloc[i]:,.0f}"
    row["Net Yield (% p.a.)"]   = f"{f1*100:.3f}%"
    row["Liquidity Index"]       = f"{f2:.4f}"
    row["Tax Relief (RM)"]       = f"RM {tax_rm:,.2f}"
    row["Zakat (RM)"]            = f"RM {zak_rm:,.2f}"
    feasible = (asb_v < 1e-4) and (gop_v < 1e-4)
    row["Statutory Status"]      = "✅" if feasible else "❌"
    return row

rows = []

# NSGA-II solutions
for label, idx in [
    ("NSGA-II Compromise ⭐",    compromise["index"]),
    ("NSGA-II Max Return",       idx_max_ret),
    ("NSGA-II Max Liquidity",    idx_max_liq),
]:
    rows.append(_build_row(
        label,
        result.weights[idx],
        result.net_returns[idx],
        result.liquidity_scores[idx],
        result.tax_relief_rm[idx],
        result.zakat_rm[idx],
        result.asb_violations[idx],
        result.go_plus_violations[idx],
    ))

# Baselines
for label, b in [("Rule-Based (AKPK)", rb), ("Naïve 1/N", ew)]:
    rows.append(_build_row(
        label,
        b["weights"],
        b["net_return"],
        b["liquidity_score"],
        b["tax_relief_rm"],
        b["zakat_rm"],
        b["asb_violation_rm"],
        b["go_plus_violation_rm"],
    ))

bench_df = pd.DataFrame(rows).set_index("Strategy")
st.dataframe(bench_df, use_container_width=True)

st.divider()


# ===========================================================================
# PANEL D — Front Quality Metrics & Export
# ===========================================================================

st.subheader("Panel D · Pareto Front Quality & Data Export")

qcol1, qcol2, qcol3, qcol4 = st.columns(4)
qcol1.metric("Pareto Solutions (K)", K)
qcol2.metric("Hypervolume (HV)", f"{hv:.6f}",
             help="Area of objective space dominated by the front. Higher is better.")
qcol3.metric("Spacing (S)", f"{sp:.6f}",
             help="Schott's spacing metric. Lower = more uniformly distributed front.")
qcol4.metric("Constraint Feasibility Rate", f"{cfr:.1f}%",
             help="Fraction of Pareto front satisfying all statutory constraints.")

st.markdown(" ")

# Export Pareto front as CSV
pareto_df   = result.to_dataframe()
csv_bytes   = pareto_df.to_csv(index=False).encode("utf-8")

dl_col, _ = st.columns([2, 5])
dl_col.download_button(
    label="📥 Export Pareto Front CSV",
    data=csv_bytes,
    file_name=f"pareto_front_C{int(_C)}_tau{int(_tau*100)}.csv",
    mime="text/csv",
    use_container_width=True,
)

with st.expander("🔍 Preview Pareto Front Data"):
    st.dataframe(pareto_df.style.format({
        c: "{:.4f}" for c in pareto_df.select_dtypes("float").columns
    }), use_container_width=True)

st.divider()
st.caption(
    "Shariah Portfolio Optimizer · NSGA-II Multi-Objective Framework · "
    "pymoo 0.6 · Streamlit · © 2025 FYP Research"
)
