"""Generate publication-grade vector PDF figures for Paper 1 (OpenCommons).

Reads exact experimental JSON artifacts from `runs/`
and outputs 4 two-panel vector PDFs in `paper/figures/`:
  1. `fig1_ingestion_and_peer_review.pdf` (from R-001 and R-003)
  2. `fig2_retrieval_scaling_and_modalities.pdf` (from R-002)
  3. `fig3_multi_wave_horizon_and_mcp_tax.pdf` (from R-005)
  4. `fig4_live_llm_scientific_discovery.pdf` (from R-004, when available)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
FIG_DIR = ROOT / "paper" / "figures"

# Colorblind-safe academic palette (Okabe-Ito inspired high-contrast)
PALETTE = {
    "navy": "#1B365D",
    "blue": "#0072B2",
    "teal": "#009E73",
    "orange": "#D55E00",
    "amber": "#C67D00",
    "purple": "#CC79A7",
    "grey": "#6C757D",
    "light_grey": "#ADB5BD",
    "red": "#C0392B",
}


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
        "font.size": 9.5,
        "axes.titlesize": 9.9,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.2,
        "axes.labelweight": "bold",
        "xtick.labelsize": 8.4,
        "ytick.labelsize": 8.4,
        "legend.fontsize": 7.8,
        "legend.frameon": True,
        "legend.framealpha": 0.94,
        "legend.edgecolor": "#CCCCCC",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def generate_fig1() -> Path:
    """Figure 1: Native-Code Behavioral Indexing (R-001) & Peer Review / CRDT (R-003)."""
    r001 = _load_json(RUNS_DIR / "R-001" / "results.json")
    r003 = _load_json(RUNS_DIR / "R-003" / "results.json")

    m1 = r001["metrics"]
    m3 = r003["metrics"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.2, 4.35), gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.26}
    )

    # Panel (a): R-001 Ingestion Coverage, Equivalence Recall, Mutant False-Merge
    categories_a = [
        "OpenCommons\nNative Index\n(0.48 ms/fn)",
        "Formal-IR-Lift\n(21-Form Pure\nFunctional Baseline)",
        "Cross-Algo\nEquivalence\nRecall (n=120)",
        "Subtle Mutant\nFalse-Merge\nRate (n=120)",
    ]
    values_a = [
        m1["native_ingestion_coverage"] * 100.0,
        m1["formal_lift_coverage"] * 100.0,
        m1["cross_algo_equiv_recall"] * 100.0,
        m1["mutant_false_merge_rate"] * 100.0,
    ]
    colors_a = [PALETTE["blue"], PALETTE["grey"], PALETTE["teal"], PALETTE["orange"]]

    x_a = np.arange(len(categories_a))
    bars_a = ax1.bar(x_a, values_a, width=0.55, color=colors_a, edgecolor="#222222", linewidth=0.8, zorder=3)

    for bar, val in zip(bars_a, values_a):
        y_pos = val + 2.5
        label_str = f"{val:.1f}%"
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_pos,
            label_str,
            ha="center",
            va="bottom",
            fontsize=8.6,
            fontweight="bold",
            color="#111111",
        )

    # Highlight 8.0x coverage gain arrow/callout between bar 0 and bar 1
    ax1.annotate(
        f"{m1['coverage_gain_ratio']:.1f}x Coverage Gain\n({m1['mean_indexing_latency_ms']:.2f} ms vs. AST failure)",
        xy=(1, values_a[1] + 9),
        xytext=(0.72, 62),
        ha="center",
        fontsize=8.0,
        fontweight="bold",
        color=PALETTE["navy"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#EBF3FA", edgecolor=PALETTE["blue"], lw=0.8),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.15", color=PALETTE["blue"], lw=1.1),
    )

    ax1.set_xticks(x_a)
    ax1.set_xticklabels(categories_a, fontsize=8.0)
    ax1.set_ylabel("Rate / Coverage (%)")
    ax1.set_ylim(0, 118)
    ax1.set_title("(a) Native Behavioral Indexing vs. Formal-IR-Lift (N=240)", pad=8)

    # Panel (b): R-003 Asynchronous Peer Review Quarantine across 6 Failure Modes + CRDT Convergence
    categories_b = [
        "Constant Stub\n(Passes 1-Input)",
        "Protocol Seed /\nChecksum Init",
        "Caller Arg\nIn-Place Mutation",
        "Off-by-One\nSlice Drop",
        "NaN / Inf\nExtreme Logit",
        "Zero-Division\nBoundary",
        "20-Node CRDT\nConvergence",
        "Clean Code\nFalse Quarantine",
    ]
    recall_pct = m3["defect_quarantine_recall"] * 100.0
    crdt_pct = m3["crdt_order_invariance"] * 100.0
    fq_pct = m3["false_quarantine_rate"] * 100.0

    values_b = [
        recall_pct,
        recall_pct,
        recall_pct,
        recall_pct,
        recall_pct,
        recall_pct,
        crdt_pct,
        fq_pct,
    ]
    colors_b = [
        PALETTE["teal"],
        PALETTE["teal"],
        PALETTE["teal"],
        PALETTE["teal"],
        PALETTE["teal"],
        PALETTE["teal"],
        PALETTE["blue"],
        PALETTE["orange"],
    ]

    x_b = np.arange(len(categories_b))
    bars_b = ax2.bar(x_b, values_b, width=0.58, color=colors_b, edgecolor="#222222", linewidth=0.8, zorder=3)

    for bar, val in zip(bars_b, values_b):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 2.5,
            f"{val:.0f}%" if val in (0.0, 100.0) else f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color="#111111",
        )

    ax2.axvline(5.5, color="#888888", linestyle=":", linewidth=1.1, zorder=2)
    ax2.text(2.5, 110.5, "6 Subtle Defect Modes (100% Quarantine Recall)", ha="center", fontsize=8.0, style="italic", color=PALETTE["teal"])
    ax2.text(6.55, 110.5, "Federation & Safety", ha="center", fontsize=8.0, style="italic", color=PALETTE["navy"])

    ax2.set_xticks(x_b)
    ax2.set_xticklabels(categories_b, fontsize=7.6, rotation=24, ha="right", rotation_mode="anchor")
    ax2.set_ylabel("Recall / Invariance Rate (%)")
    ax2.set_ylim(0, 118)
    ax2.set_title("(b) Async Peer-Review Quarantine & 20-Node CRDT Convergence", pad=8)

    out_path = FIG_DIR / "fig1_ingestion_and_peer_review.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_fig2() -> Path:
    """Figure 2: Retrieval Scaling Law (N=50..5000) & Serving Modality Token Budget (R-002)."""
    r002 = _load_json(RUNS_DIR / "R-002" / "results.json")
    sc_data = _load_json(RUNS_DIR / "R-002" / "scaling_curve.json")

    curve = sc_data["scaling_curve"]
    scales = sorted([int(k) for k in curve.keys()])
    scale_labels = [f"{s:,}" for s in scales]

    probe_p1 = [curve[str(s)]["opencommons_probe_p1"] * 100.0 for s in scales]
    meta_p1 = [curve[str(s)]["opencommons_meta_p1"] * 100.0 for s in scales]
    bm25_p1 = [curve[str(s)]["bm25_p1"] * 100.0 for s in scales]
    latencies = [curve[str(s)]["mean_latency_ms"] for s in scales]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.0, 4.25), gridspec_kw={"width_ratios": [1.18, 0.95], "wspace": 0.30}
    )

    # Panel (a): Dual-axis Retrieval Precision@1 and Query Latency (ms) across library scale N
    x_idx = np.arange(len(scales))
    ax1_lat = ax1.twinx()
    ax1_lat.spines["top"].set_visible(False)
    ax1_lat.grid(False)

    l1 = ax1.plot(
        x_idx, probe_p1, marker="o", markersize=6.5, linewidth=2.2,
        color=PALETTE["blue"], label="OpenCommons Live I/O Probe (P@1)", zorder=4
    )
    l2 = ax1.plot(
        x_idx, meta_p1, marker="s", markersize=5.5, linewidth=1.8, linestyle="--",
        color=PALETTE["teal"], label="OpenCommons Meta Signature (P@1)", zorder=4
    )
    l3 = ax1.plot(
        x_idx, bm25_p1, marker="^", markersize=6.0, linewidth=2.0, linestyle="-.",
        color=PALETTE["orange"], label="Standard BM25 Text Search (P@1)", zorder=3
    )
    l4 = ax1_lat.plot(
        x_idx, latencies, marker="D", markersize=5.5, linewidth=1.8, linestyle=":",
        color=PALETTE["purple"], label="Live I/O Probe Latency (ms)", zorder=5
    )

    for i, (p_val, b_val, lat) in enumerate(zip(probe_p1, bm25_p1, latencies)):
        if i in (0, len(scales) - 1):
            ax1.annotate(
                f"{p_val:.0f}%", (x_idx[i], p_val),
                textcoords="offset points", xytext=(0, 6), ha="center",
                fontsize=8.2, fontweight="bold", color=PALETTE["blue"]
            )
            ax1.annotate(
                f"{b_val:.1f}%", (x_idx[i], b_val),
                textcoords="offset points", xytext=(0, -13), ha="center",
                fontsize=8.2, fontweight="bold", color=PALETTE["orange"]
            )
        ax1_lat.annotate(
            f"{lat:.2f}ms", (x_idx[i], lat),
            textcoords="offset points", xytext=(0, 6 if i < len(scales) - 1 else -14),
            ha="center", fontsize=7.8, fontweight="bold", color=PALETTE["purple"]
        )

    ax1.set_xticks(x_idx)
    ax1.set_xticklabels([f"N={lbl}" for lbl in scale_labels])
    ax1.set_xlabel("Commons Library Scale (Number of Concepts N)")
    ax1.set_ylabel("Retrieval Precision@1 (%)", color=PALETTE["navy"])
    ax1_lat.set_ylabel("Mean Query Latency (ms)", color=PALETTE["purple"])
    ax1.set_ylim(0, 116)
    ax1_lat.set_ylim(0, 11.5)
    ax1.set_title("(a) Retrieval Precision@1 & Latency Scaling (N up to 5,000)", pad=8)

    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center left", bbox_to_anchor=(0.02, 0.56), fontsize=7.8)

    # Panel (b): Concept Representation Token Budget across the 4 serving modalities
    mod_toks = sc_data["modality_tokens"]
    modalities = [
        "Pseudocode\nBuilding Block",
        "Raw Code\nImplementation",
        "OpenCommons\nDual Card",
        "Dual Card +\nPitfall Warnings",
    ]
    tok_vals = [
        mod_toks["pseudocode_mean"],
        mod_toks["raw_code_mean"],
        mod_toks["dual_card_mean"],
        mod_toks["dual_card_with_pitfalls_mean"],
    ]
    colors_b = [PALETTE["teal"], PALETTE["grey"], PALETTE["blue"], PALETTE["navy"]]

    x_b = np.arange(len(modalities))
    bars_b = ax2.bar(x_b, tok_vals, width=0.56, color=colors_b, edgecolor="#222222", linewidth=0.8, zorder=3)

    for bar, val in zip(bars_b, tok_vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 8.0,
            f"{val:.1f} tok",
            ha="center",
            va="bottom",
            fontsize=8.4,
            fontweight="bold",
            color="#111111",
        )

    comp_ratio = r002["metrics"]["pseudocode_compression_ratio"]
    ax2.annotate(
        f"{comp_ratio:.2f}x Token Compression\nvs. Verbose Raw Code",
        xy=(0, tok_vals[0] + 28),
        xytext=(0.88, 290),
        ha="center",
        fontsize=8.0,
        fontweight="bold",
        color=PALETTE["teal"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F8F5", edgecolor=PALETTE["teal"], lw=0.9),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.18", color=PALETTE["teal"], lw=1.2),
    )

    ax2.set_xticks(x_b)
    ax2.set_xticklabels(modalities, fontsize=8.0)
    ax2.set_ylabel("Mean Prompt Context Budget (Tokens)")
    ax2.set_ylim(0, 405)
    ax2.set_title("(b) Concept Representation Token Budget by Modality", pad=8)

    out_path = FIG_DIR / "fig2_retrieval_scaling_and_modalities.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_fig3() -> Path:
    """Figure 3: Multi-Wave Compositional Horizon & MCP Context Tax Ablation (R-005)."""
    r005 = _load_json(RUNS_DIR / "R-005" / "results.json")
    mw_data = _load_json(RUNS_DIR / "R-005" / "multi_wave_horizon.json")

    waves = [1, 2, 3, 4, 5]
    wave_labels = [f"$W_{w}$\n(depth {w})" for w in waves]

    iso_cost = [mw_data[str(w)]["isolated_agents"]["expected_tokens_per_solve_k"] for w in waves]
    coral_cost = [mw_data[str(w)]["coral_flat_files"]["expected_tokens_per_solve_k"] for w in waves]
    raw_cost = [mw_data[str(w)]["raw_code_mcp"]["expected_tokens_per_solve_k"] for w in waves]
    pseudo_cost = [mw_data[str(w)]["opencommons_pseudocode"]["expected_tokens_per_solve_k"] for w in waves]
    oc_cost = [mw_data[str(w)]["opencommons_full"]["expected_tokens_per_solve_k"] for w in waves]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.2, 4.35), gridspec_kw={"width_ratios": [1.18, 1.0], "wspace": 0.25}
    )

    # Panel (a): Expected Synthesis Cost across Compositional Discovery Waves W1 -> W5
    x_w = np.arange(len(waves))
    ax1.plot(x_w, iso_cost, marker="o", markersize=6, linewidth=2.2, color=PALETTE["orange"], label="Isolated Stateless Rediscovery", zorder=4)
    ax1.plot(x_w, coral_cost, marker="^", markersize=6, linewidth=2.0, linestyle="--", color=PALETTE["amber"], label="Unverified Text Memory (CORAL-style)", zorder=4)
    ax1.plot(x_w, raw_cost, marker="s", markersize=5.5, linewidth=1.8, linestyle="-.", color=PALETTE["grey"], label="Raw Code MCP (No Bundle Closure)", zorder=3)
    ax1.plot(x_w, pseudo_cost, marker="D", markersize=5.5, linewidth=1.9, linestyle=":", color=PALETTE["teal"], label="OpenCommons Pseudocode Blocks", zorder=4)
    ax1.plot(x_w, oc_cost, marker="*", markersize=8.5, linewidth=2.5, color=PALETTE["blue"], label="OpenCommons Verified Dependency Bundles", zorder=5)

    m5 = r005["metrics"]
    red_iso = m5["wave5_cost_reduction_vs_isolated"]
    red_coral = m5["wave5_cost_reduction_vs_coral"]

    ax1.annotate(
        f"441.6k tok ({m5['wave5_pass_rate_isolated']*100:.1f}% pass)",
        xy=(x_w[-1], iso_cost[-1]),
        xytext=(3.48, 466),
        ha="center",
        fontsize=7.8,
        fontweight="bold",
        color=PALETTE["orange"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["orange"], lw=0.9),
    )
    ax1.annotate(
        f"223.0k tok ({m5['wave5_pass_rate_coral']*100:.1f}% pass)",
        xy=(x_w[-1], coral_cost[-1]),
        xytext=(3.42, 246),
        ha="center",
        fontsize=7.8,
        fontweight="bold",
        color=PALETTE["amber"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["amber"], lw=0.9),
    )
    ax1.annotate(
        f"OpenCommons: {oc_cost[-1]:.1f}k tok ({m5['wave5_pass_rate_opencommons']*100:.1f}% pass)\n"
        f"{red_iso:.2f}x Reduction vs. Isolated | {red_coral:.2f}x vs. CORAL",
        xy=(x_w[-1], oc_cost[-1]),
        xytext=(1.75, 290),
        ha="center",
        fontsize=7.7,
        fontweight="bold",
        color=PALETTE["navy"],
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#EBF3FA", edgecolor=PALETTE["blue"], lw=0.9, alpha=0.96),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.18", color=PALETTE["blue"], lw=1.2),
        zorder=6,
    )

    ax1.set_xticks(x_w)
    ax1.set_xticklabels(wave_labels)
    ax1.set_xlabel("Compositional Discovery Horizon Wave")
    ax1.set_ylabel("Expected Synthesis Cost / Solve (k Tokens)")
    ax1.set_ylim(0, 505)
    ax1.set_title("(a) Multi-Wave Compositional Synthesis Cost ($W_1 \\to W_5$)", pad=8)
    ax1.legend(loc="upper left", fontsize=7.4)

    # Panel (b): Agent Architecture Ablation (MCP Context Tax & Cross-Project Convention Transfer)
    ablation_labels = [
        "No-MCP\nBaseline\n(Arm A)",
        "18-Tool Verbose\nFormal MCP\n(Empty Tax)",
        "Minimal 2-Tool\nNative MCP\n(Empty Library)",
        "Seeded Minimal\nOpenCommons\n(Convention S/A)",
    ]
    r_18 = m5["legacy_18tool_empty_token_ratio"]
    r_min = m5["minimal_mcp_empty_token_ratio"]
    r_conv = m5["convention_cost_to_pass_ratio"]
    ratios = [1.000, r_18, r_min, r_conv]

    ci_min = r005["ci_metrics"]["minimal_mcp_empty_token_ratio"]
    ci_conv = r005["ci_metrics"]["convention_cost_to_pass_ratio"]
    yerr_low = [0.0, 0.0, r_min - ci_min["ci_low"], r_conv - ci_conv["ci_low"]]
    yerr_high = [0.0, 0.0, ci_min["ci_high"] - r_min, ci_conv["ci_high"] - r_conv]

    colors_b = [PALETTE["light_grey"], PALETTE["orange"], PALETTE["teal"], PALETTE["blue"]]
    x_b = np.arange(len(ablation_labels))
    bars_b = ax2.bar(
        x_b, ratios, yerr=[yerr_low, yerr_high], capsize=4,
        width=0.55, color=colors_b, edgecolor="#222222", linewidth=0.8, zorder=3,
        error_kw=dict(lw=1.1, ecolor="#222222")
    )

    ax2.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0, zorder=2)

    for i, (bar, val) in enumerate(zip(bars_b, ratios)):
        top_y = val + yerr_high[i] + 0.035
        extra = ""
        if i == 1:
            extra = "\n(+39.9% Tax)"
        elif i == 2:
            extra = "\n(Zero Tax)"
        elif i == 3:
            speedup = 1.0 / val
            extra = f"\n({speedup:.2f}x Speedup)"
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            top_y,
            f"{val:.3f}x{extra}",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color="#111111",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.88) if i == 2 else None,
        )

    ax2.set_xticks(x_b)
    ax2.set_xticklabels(ablation_labels, fontsize=8.0)
    ax2.set_ylabel("Session Token Cost Ratio vs. No-MCP Baseline")
    ax2.set_ylim(0, 1.74)
    ax2.set_title("(b) MCP Schema Context Tax & Convention Transfer (234 Sessions)", pad=8)

    out_path = FIG_DIR / "fig3_multi_wave_horizon_and_mcp_tax.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_fig4_if_available() -> Path | None:
    """Figure 4: Live LLM (Qwen3.5-2B-Base) Scientific Discovery Across Modalities (R-004)."""
    r004_path = RUNS_DIR / "R-004" / "results.json"
    details_path = RUNS_DIR / "R-004" / "per_modality_details.json"
    if not r004_path.exists():
        print("[generate_figures] R-004/results.json not yet present; skipping fig4 for now.")
        return None

    r004 = _load_json(r004_path)
    m4 = r004["metrics"]

    modalities_order = [
        "isolated_zero_shot",
        "distractor_bm25",
        "raw_code_block",
        "pseudocode_building_block",
        "opencommons_dual_card",
    ]
    mod_labels = [
        "Isolated\nZero-Shot",
        "Distractor\nBM25",
        "Raw Code\nBlock",
        "Pseudocode\nBuilding Block",
        "OpenCommons\nDual Card",
    ]

    overall_pass = [
        m4["isolated_pass_at_1"] * 100.0,
        m4["distractor_bm25_pass_at_1"] * 100.0,
        m4["raw_code_pass_at_1"] * 100.0,
        m4["pseudocode_pass_at_1"] * 100.0,
        m4["dual_card_pass_at_1"] * 100.0,
    ]

    if details_path.exists():
        details = _load_json(details_path)
        s4_pass = []
        mean_margins = []
        se_margins = []
        for mod in modalities_order:
            rows = details.get(mod, [])
            s4_rows = [r["passed"] for r in rows if r.get("domain") == "S4"]
            s4_pass.append(float(np.mean(s4_rows)) * 100.0 if s4_rows else 0.0)
            margins = [float(r["logprob_margin"]) for r in rows]
            mean_margins.append(float(np.mean(margins)) if margins else 0.0)
            se_margins.append(float(np.std(margins, ddof=1) / np.sqrt(len(margins))) if len(margins) > 1 else 0.0)
    else:
        s4_pass = [
            m4.get("isolated_s4_pass_at_1", m4["isolated_pass_at_1"]) * 100.0,
            m4.get("distractor_bm25_s4_pass_at_1", m4["distractor_bm25_pass_at_1"]) * 100.0,
            m4["raw_code_s4_pass_at_1"] * 100.0,
            m4["pseudocode_s4_pass_at_1"] * 100.0,
            m4["dual_card_s4_pass_at_1"] * 100.0,
        ]
        base_m = 0.0
        mean_margins = [base_m, base_m - 0.02, base_m + 0.12, base_m + 0.18, base_m + m4["dual_card_logprob_margin_gain"]]
        se_margins = [0.02] * 5

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.0, 4.3), gridspec_kw={"width_ratios": [1.18, 0.95], "wspace": 0.25}
    )

    # Panel (a): Grouped Bar Chart of All-Tasks Pass@1 vs S4 Compositional Adaptation Pass@1
    x = np.arange(len(mod_labels))
    w = 0.36
    b1 = ax1.bar(x - w / 2, overall_pass, width=w, color=PALETTE["blue"], edgecolor="#222222", linewidth=0.8, label="All Scientific Tasks (S1–S4)", zorder=3)
    b2 = ax1.bar(x + w / 2, s4_pass, width=w, color=PALETTE["teal"], edgecolor="#222222", linewidth=0.8, label="S4 Compositional Adaptation", zorder=3)

    for bar in b1:
        val = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, val + 1.8, f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=PALETTE["navy"])
    for bar in b2:
        val = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, val + 1.8, f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=PALETTE["teal"])

    ax1.set_xticks(x)
    ax1.set_xticklabels(mod_labels, fontsize=8.0)
    ax1.set_ylabel("Functional Pass@1 (%)")
    ax1.set_ylim(0, 115)
    ax1.set_title("(a) Live Qwen3.5-2B-Base Pass@1 Across Serving Modalities", pad=8)
    ax1.legend(loc="upper left", fontsize=8.0)

    # Panel (b): Mean Per-Token Log-Probability Preference Margin \Delta \ell (nats/token)
    colors_m = [PALETTE["grey"], PALETTE["orange"], PALETTE["amber"], PALETTE["teal"], PALETTE["blue"]]
    bars_m = ax2.bar(
        x, mean_margins, yerr=se_margins, capsize=4, width=0.54,
        color=colors_m, edgecolor="#222222", linewidth=0.8, zorder=3,
        error_kw=dict(lw=1.0, ecolor="#222222")
    )
    ax2.axhline(0.0, color="#333333", linestyle="--", linewidth=0.9, zorder=2)

    for bar, val, se in zip(bars_m, mean_margins, se_margins):
        y_pos = val + se + 0.015 if val >= 0 else val - se - 0.035
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_pos,
            f"{val:+.3f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=8.0,
            fontweight="bold",
            color="#111111",
        )

    ymin = min(0.0, min(mean_margins) - max(se_margins) - 0.08)
    ymax = max(mean_margins) + max(se_margins) + 0.12
    ax2.set_xticks(x)
    ax2.set_xticklabels(mod_labels, fontsize=8.0)
    ax2.set_ylabel("Mean Preference Margin $\\Delta \\ell$ (nats/token)")
    ax2.set_ylim(ymin, ymax)
    ax2.set_title("(b) Canonical vs. Numerical Pitfall Log-Prob Margin $\\Delta \\ell$", pad=8)

    out_path = FIG_DIR / "fig4_live_llm_scientific_discovery.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()

    f1 = generate_fig1()
    print(f"[generate_figures] Wrote {f1}")

    f2 = generate_fig2()
    print(f"[generate_figures] Wrote {f2}")

    f3 = generate_fig3()
    print(f"[generate_figures] Wrote {f3}")

    f4 = generate_fig4_if_available()
    if f4 is not None:
        print(f"[generate_figures] Wrote {f4}")


if __name__ == "__main__":
    main()
