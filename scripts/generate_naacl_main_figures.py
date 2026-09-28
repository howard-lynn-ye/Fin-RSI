#!/usr/bin/env python3
"""Generate publication-grade vector PDF & 300-DPI PNG figures for the NAACL paper.

Follows `academic-paper-writing` (Law 4 & Law 5) and `scientific-visualization-designer`
(WSJ / Dona Wong / Edward Tufte principles):
  - Zero overlapping text, generous padding inside boxes, crisp vector typography
  - Minimalist 2-4 item legends, direct data labeling, high data-to-ink ratio
  - Exports both `.pdf` (vector Type-42 TrueType fonts) and `.png` (300 DPI) to:
      * /usr/local/google/home/shwaihe/fin-skills/paper/latex_naacl/figures/
      * /usr/local/google/home/shwaihe/stock_prediction/paper/latex_naacl/figures/
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Ensure Type 42 TrueType fonts in PDF for ACL/NAACL camera-ready compliance
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#334155",
    "axes.linewidth": 0.8,
})

FIN_SKILLS_FIG_DIR = Path("/usr/local/google/home/shwaihe/fin-skills/paper/latex_naacl/figures")
STOCK_PRED_FIG_DIR = Path("/usr/local/google/home/shwaihe/stock_prediction/paper/latex_naacl/figures")


def draw_rounded_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    facecolor: str,
    edgecolor: str,
    linewidth: float = 1.4,
    linestyle: str = "-",
    pad: float = 0.010,
    zorder: int = 2,
):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad={pad},rounding_size=0.016",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(box)
    return box


def draw_arrow(
    ax,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str = "#334155",
    lw: float = 1.5,
    style: str = "-|>",
    rad: float = 0.0,
    ls: str = "-",
    zorder: int = 4,
):
    arrow = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=style,
        mutation_scale=12,
        linewidth=lw,
        linestyle=ls,
        color=color,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def generate_fig1_architecture() -> list[Path]:
    """Generate Figure 1: Dual-Layer Fin-RSI Closed-Loop Inference & Evolution Architecture."""
    fig, ax = plt.subplots(figsize=(13.4, 4.65), dpi=300)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # -------------------------------------------------------------------------
    # Outer container band 1: Online Point-in-Time Closed-Loop Inference (Top)
    # -------------------------------------------------------------------------
    draw_rounded_box(
        ax, 0.012, 0.565, 0.976, 0.415,
        facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.2, pad=0.008, zorder=1
    )
    ax.text(
        0.024, 0.952,
        "ONLINE PATH: Point-in-Time Closed-Loop Quantitative Inference & Counterfactual Verification (t -> t + H)",
        fontsize=9.0, fontweight="bold", color="#0F172A", va="center", ha="left", zorder=5
    )

    # Box 1: Query & PIT State
    b1_x, b1_y, b1_w, b1_h = 0.024, 0.595, 0.195, 0.315
    draw_rounded_box(ax, b1_x, b1_y, b1_w, b1_h, "#EFF6FF", "#2563EB", linewidth=1.4)
    ax.text(b1_x + b1_w / 2, b1_y + 0.265, "1. PIT Query & State", fontsize=8.6, fontweight="bold", color="#1E3A8A", ha="center")
    ax.text(b1_x + b1_w / 2, b1_y + 0.200, r"Task $q_t$, Data $\mathcal{D}_{\leq t}$", fontsize=8.1, color="#1E293B", ha="center")
    ax.text(b1_x + b1_w / 2, b1_y + 0.132, "US / China A-Share / HK", fontsize=7.5, color="#475569", ha="center")
    ax.text(b1_x + b1_w / 2, b1_y + 0.078, "Crypto Perps / Options", fontsize=7.5, color="#475569", ha="center")
    ax.text(b1_x + b1_w / 2, b1_y + 0.024, "Strict Bitemporal Cutoff", fontsize=7.2, fontweight="bold", color="#2563EB", ha="center")

    # Box 2: RAG-Jev Two-Stage Progressive Router
    b2_x, b2_y, b2_w, b2_h = 0.246, 0.595, 0.232, 0.315
    draw_rounded_box(ax, b2_x, b2_y, b2_w, b2_h, "#F0FDF4", "#16A34A", linewidth=1.5)
    ax.text(b2_x + b2_w / 2, b2_y + 0.265, "2. RAG-Jev Progressive Router", fontsize=8.6, fontweight="bold", color="#14532D", ha="center")
    ax.text(b2_x + b2_w / 2, b2_y + 0.198, r"$s_{\mathrm{route}} = \lambda s_{\mathrm{lex}} + (1-\lambda)\langle\phi(q_t),\phi(d_i)\rangle$", fontsize=7.6, color="#1E293B", ha="center")
    ax.text(b2_x + b2_w / 2, b2_y + 0.130, "129 Verified Skills (15 Plugins)", fontsize=7.5, color="#334155", ha="center")
    ax.text(b2_x + b2_w / 2, b2_y + 0.078, "Stage-1 Card + Stage-2 Lazy Ref", fontsize=7.4, color="#475569", ha="center")
    ax.text(b2_x + b2_w / 2, b2_y + 0.024, "4,039 Tokens (-98.91% vs. Full)", fontsize=7.2, fontweight="bold", color="#15803D", ha="center")

    # Box 3: Atomic Skill + 64-KC Associative Operator
    b3_x, b3_y, b3_w, b3_h = 0.505, 0.595, 0.235, 0.315
    draw_rounded_box(ax, b3_x, b3_y, b3_w, b3_h, "#FEFCE8", "#CA8A04", linewidth=1.5)
    ax.text(b3_x + b3_w / 2, b3_y + 0.265, "3. 64-KC Associative Readout", fontsize=8.6, fontweight="bold", color="#713F12", ha="center")
    ax.text(b3_x + b3_w / 2, b3_y + 0.198, r"$z_t = \mathrm{TopK}_k(W_{\mathrm{JEV}}\tilde{x}_t), \; k=6$", fontsize=7.9, color="#1E293B", ha="center")
    ax.text(b3_x + b3_w / 2, b3_y + 0.130, r"$\hat{r}_t = (1-\gamma(n))\hat{y}_{\mathrm{KC}} + \gamma(n)s_{\mathrm{skill}}$", fontsize=7.6, color="#1E293B", ha="center")
    ax.text(b3_x + b3_w / 2, b3_y + 0.078, "64-KC Mushroom-Body + Prior Blend", fontsize=7.3, color="#475569", ha="center")
    ax.text(b3_x + b3_w / 2, b3_y + 0.024, "Zero Backprop / Sparse WTA", fontsize=7.2, fontweight="bold", color="#A16207", ha="center")

    # Box 4: 37 Counterfactual Guards & Settlement
    b4_x, b4_y, b4_w, b4_h = 0.767, 0.595, 0.208, 0.315
    draw_rounded_box(ax, b4_x, b4_y, b4_w, b4_h, "#FEF2F2", "#DC2626", linewidth=1.5)
    ax.text(b4_x + b4_w / 2, b4_y + 0.265, "4. 37 Guards & Settlement", fontsize=8.6, fontweight="bold", color="#7F1D1D", ha="center")
    ax.text(b4_x + b4_w / 2, b4_y + 0.198, r"$g_j(\tau) = \mathbf{1}[\mathcal{M}(\tau) \equiv \mathcal{M}(\tilde{\tau}_j)]$", fontsize=7.8, color="#1E293B", ha="center")
    ax.text(b4_x + b4_w / 2, b4_y + 0.130, "AST + Future-Perturbation Probe", fontsize=7.4, color="#334155", ha="center")
    ax.text(b4_x + b4_w / 2, b4_y + 0.078, r"Dopamine $G = \log(W_T / W_0)$", fontsize=7.6, color="#1E293B", ha="center")
    ax.text(b4_x + b4_w / 2, b4_y + 0.024, "Fail-Closed SHA-256 Receipt", fontsize=7.2, fontweight="bold", color="#B91C1C", ha="center")

    # Horizontal arrows in Online Path
    draw_arrow(ax, b1_x + b1_w + 0.004, b1_y + b1_h / 2, b2_x - 0.004, b2_y + b2_h / 2, color="#1E40AF", lw=1.7)
    draw_arrow(ax, b2_x + b2_w + 0.004, b2_y + b2_h / 2, b3_x - 0.004, b3_y + b3_h / 2, color="#15803D", lw=1.7)
    draw_arrow(ax, b3_x + b3_w + 0.004, b3_y + b3_h / 2, b4_x - 0.004, b4_y + b4_h / 2, color="#B45309", lw=1.7)

    # -------------------------------------------------------------------------
    # Outer container band 2: Offline Dual-Layer Governed Fin-RSI (Bottom)
    # -------------------------------------------------------------------------
    draw_rounded_box(
        ax, 0.012, 0.020, 0.976, 0.420,
        facecolor="#F1F5F9", edgecolor="#64748B", linewidth=1.3, linestyle="--", pad=0.008, zorder=1
    )
    ax.text(
        0.024, 0.414,
        "OFFLINE DUAL-LAYER FIN-RSI: Cryptographically Locked Co-Evolution of Discrete Skill Cards & Numeric Operators",
        fontsize=8.8, fontweight="bold", color="#0F172A", va="center", ha="left", zorder=5
    )

    # Layer A Card (Left): Skill-Space RSI
    la_x, la_y, la_w, la_h = 0.024, 0.042, 0.460, 0.340
    draw_rounded_box(ax, la_x, la_y, la_w, la_h, "#ECFDF5", "#059669", linewidth=1.6)
    ax.text(
        la_x + 0.014, la_y + 0.300,
        "LAYER A: Skill-Space RSI (Discrete Card Evolution)",
        fontsize=8.2, fontweight="bold", color="#065F46", ha="left"
    )
    ax.text(
        la_x + la_w - 0.012, la_y + 0.300,
        "[SKILL_HARNESS_LOCK]",
        fontsize=7.1, fontweight="bold", color="#047857", ha="right"
    )
    # 3 sub-steps inside Layer A
    s1_w = 0.136
    draw_rounded_box(ax, la_x + 0.012, la_y + 0.064, s1_w, 0.202, "#FFFFFF", "#10B981", linewidth=1.1, pad=0.006, zorder=3)
    ax.text(la_x + 0.012 + s1_w / 2, la_y + 0.222, "A1. Margin Audit", fontsize=7.7, fontweight="bold", color="#065F46", ha="center")
    ax.text(la_x + 0.012 + s1_w / 2, la_y + 0.156, r"$\Delta(q) = s_{y^*} - \max_{j\neq y^*} s_j$", fontsize=7.1, color="#1E293B", ha="center")
    ax.text(la_x + 0.012 + s1_w / 2, la_y + 0.095, "36 Misses + 5 Thin (<15%)", fontsize=6.9, color="#475569", ha="center")

    draw_rounded_box(ax, la_x + 0.162, la_y + 0.064, s1_w, 0.202, "#FFFFFF", "#10B981", linewidth=1.1, pad=0.006, zorder=3)
    ax.text(la_x + 0.162 + s1_w / 2, la_y + 0.222, "A2. Card Mutation", fontsize=7.7, fontweight="bold", color="#065F46", ha="center")
    ax.text(la_x + 0.162 + s1_w / 2, la_y + 0.156, "+TRIGGER Anchors", fontsize=7.2, color="#1E293B", ha="center")
    ax.text(la_x + 0.162 + s1_w / 2, la_y + 0.095, "+Sibling SKIP Repel", fontsize=7.0, color="#475569", ha="center")

    draw_rounded_box(ax, la_x + 0.312, la_y + 0.064, s1_w, 0.202, "#FFFFFF", "#10B981", linewidth=1.1, pad=0.006, zorder=3)
    ax.text(la_x + 0.312 + s1_w / 2, la_y + 0.222, "A3. 4-Gate Promote", fontsize=7.7, fontweight="bold", color="#065F46", ha="center")
    ax.text(la_x + 0.312 + s1_w / 2, la_y + 0.156, "Top-1: 66.7% -> 100%", fontsize=7.1, fontweight="bold", color="#047857", ha="center")
    ax.text(la_x + 0.312 + s1_w / 2, la_y + 0.095, "Blind Holdout: 100%", fontsize=7.0, color="#475569", ha="center")

    draw_arrow(ax, la_x + 0.150, la_y + 0.165, la_x + 0.160, la_y + 0.165, color="#059669", lw=1.3)
    draw_arrow(ax, la_x + 0.300, la_y + 0.165, la_x + 0.310, la_y + 0.165, color="#059669", lw=1.3)
    ax.text(
        la_x + la_w / 2, la_y + 0.018,
        "Immutable Guard Invariant: Zero edits to evals/queries.jsonl, eval_triggers.py, eval_blind.py, validate.py",
        fontsize=6.9, color="#065F46", ha="center", style="italic"
    )

    # Layer B Card (Right): Operator-Space RSI
    lb_x, lb_y, lb_w, lb_h = 0.514, 0.042, 0.460, 0.340
    draw_rounded_box(ax, lb_x, lb_y, lb_w, lb_h, "#EFF6FF", "#1D4ED8", linewidth=1.6)
    ax.text(
        lb_x + 0.014, lb_y + 0.300,
        "LAYER B: Operator-Space RSI (Readout Evolution)",
        fontsize=8.2, fontweight="bold", color="#1E3A8A", ha="left"
    )
    ax.text(
        lb_x + lb_w - 0.012, lb_y + 0.300,
        "[HARNESS_LOCK]",
        fontsize=7.1, fontweight="bold", color="#1D4ED8", ha="right"
    )
    # 3 sub-steps inside Layer B
    draw_rounded_box(ax, lb_x + 0.012, lb_y + 0.064, s1_w, 0.202, "#FFFFFF", "#3B82F6", linewidth=1.1, pad=0.006, zorder=3)
    ax.text(lb_x + 0.012 + s1_w / 2, lb_y + 0.222, "B1. ESS & SNR Probe", fontsize=7.7, fontweight="bold", color="#1E3A8A", ha="center")
    ax.text(lb_x + 0.012 + s1_w / 2, lb_y + 0.156, "Sparse Cold-Start &", fontsize=7.1, color="#1E293B", ha="center")
    ax.text(lb_x + 0.012 + s1_w / 2, lb_y + 0.095, "Crisis Drift Diagnosis", fontsize=7.0, color="#475569", ha="center")

    draw_rounded_box(ax, lb_x + 0.162, lb_y + 0.064, s1_w, 0.202, "#FFFFFF", "#3B82F6", linewidth=1.1, pad=0.006, zorder=3)
    ax.text(lb_x + 0.162 + s1_w / 2, lb_y + 0.222, "B2. Woodbury-Fisher", fontsize=7.7, fontweight="bold", color="#1E3A8A", ha="center")
    ax.text(lb_x + 0.162 + s1_w / 2, lb_y + 0.156, r"Rank-1 $\Sigma_t^{-1}$ Update +", fontsize=7.1, color="#1E293B", ha="center")
    ax.text(lb_x + 0.162 + s1_w / 2, lb_y + 0.095, "Fisher Ridge + VolGate", fontsize=7.0, color="#475569", ha="center")

    draw_rounded_box(ax, lb_x + 0.312, lb_y + 0.064, s1_w, 0.202, "#FFFFFF", "#3B82F6", linewidth=1.1, pad=0.006, zorder=3)
    ax.text(lb_x + 0.312 + s1_w / 2, lb_y + 0.222, "B3. Pareto Ledger", fontsize=7.7, fontweight="bold", color="#1E3A8A", ha="center")
    ax.text(lb_x + 0.312 + s1_w / 2, lb_y + 0.156, "Sharpe: 1.23 -> 1.91", fontsize=7.1, fontweight="bold", color="#1D4ED8", ha="center")
    ax.text(lb_x + 0.312 + s1_w / 2, lb_y + 0.095, "MaxDD: -18.2% -> -12.8%", fontsize=7.0, color="#475569", ha="center")

    draw_arrow(ax, lb_x + 0.150, lb_y + 0.165, lb_x + 0.160, lb_y + 0.165, color="#1D4ED8", lw=1.3)
    draw_arrow(ax, lb_x + 0.300, lb_y + 0.165, lb_x + 0.310, lb_y + 0.165, color="#1D4ED8", lw=1.3)
    ax.text(
        lb_x + lb_w / 2, lb_y + 0.018,
        "Super-Additive Synergy with Layer A: +0.54 Sharpe (t = 3.06, p = 0.0375) & +0.0134 Rank IC (p = 0.0028)",
        fontsize=6.9, color="#1E3A8A", ha="center", style="italic"
    )

    # Cross-layer coupling arrow between Layer A and Layer B
    draw_arrow(ax, la_x + la_w + 0.003, la_y + 0.165, lb_x - 0.003, lb_y + 0.165, color="#0F172A", lw=1.5, style="<->")

    # Vertical feedback arrows in the clean inter-band channel (y = 0.448 .. 0.562)
    # 1. Router margin telemetry -> Layer A
    draw_arrow(ax, 0.295, 0.562, 0.295, 0.446, color="#059669", lw=1.5, ls="--")
    ax.text(0.285, 0.504, "Margin Telemetry", fontsize=7.1, color="#065F46", ha="right", va="center", fontweight="bold")

    # 2. Layer A Evolved Gen-3 Skill Cards -> Online Router
    draw_arrow(ax, 0.420, 0.446, 0.420, 0.562, color="#059669", lw=1.7)
    ax.text(0.430, 0.504, "Gen-3 Cards (100% Top-1)", fontsize=7.1, color="#065F46", ha="left", va="center", fontweight="bold")

    # 3. Layer B Evolved Gen-3 Operator -> Online 64-KC Associative Readout
    draw_arrow(ax, 0.635, 0.446, 0.635, 0.562, color="#1D4ED8", lw=1.7)
    ax.text(0.645, 0.504, "Gen-3 Woodbury+Fisher Op", fontsize=7.1, color="#1E3A8A", ha="left", va="center", fontweight="bold")

    # 4. Online 37-Guard & Dopamine Return -> Layer B
    draw_arrow(ax, 0.865, 0.562, 0.865, 0.446, color="#DC2626", lw=1.5, ls="--")
    ax.text(0.875, 0.504, "OOS Return & Guard Trace", fontsize=7.1, color="#991B1B", ha="left", va="center", fontweight="bold")

    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)

    outputs: list[Path] = []
    for target_dir in (FIN_SKILLS_FIG_DIR, STOCK_PRED_FIG_DIR):
        target_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = target_dir / "fig1_dual_layer_fin_rsi_arch.pdf"
        png_path = target_dir / "fig1_dual_layer_fin_rsi_arch.png"
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
        fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.04)
        outputs.extend([pdf_path, png_path])
    plt.close(fig)
    return outputs


def generate_fig2_empirical_results() -> list[Path]:
    """Generate Figure 2: 3-Panel Empirical Results (Skill-Space RSI, Operator Pareto, 2x2 Synergy)."""
    import json
    root_dir = Path("/usr/local/google/home/shwaihe/fin-skills")
    skill_rsi = json.loads((root_dir / "benchmarks/SKILL_RSI_EVOLUTION_REPORT.json").read_text(encoding="utf-8"))
    fin_rsi = json.loads((root_dir / "benchmarks/FIN_RSI_PARETO_LEDGER_REPORT.json").read_text(encoding="utf-8"))
    dual_syn = json.loads((root_dir / "benchmarks/DUAL_LAYER_RSI_SYNERGY_RESULTS.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.05), dpi=300)
    ax_a, ax_b, ax_c = axes

    # =========================================================================
    # Panel (a): Layer A — Skill-Space RSI Across 5 Retrieval Architectures
    # =========================================================================
    encoders = ["Lexical\nTrigger", "BM25S\nRetriever", "FinBERT\nDense", "BGE-m3\nDense", "JEV-64KC\nRouter"]
    g0 = skill_rsi["generations"]["Gen-0_Unoptimized_Wave2_Catalog"]
    g1 = skill_rsi["generations"]["Gen-1_Contrastive_TRIGGER_Amplification"]
    g3 = skill_rsi["generations"]["Gen-3_Champion_1Hop_Xref_Graph_and_Package_Sync"]

    def _get_enc_accs(gen_dict: dict) -> list[float]:
        me = gen_dict["multi_encoder_routing"]
        return [
            round(gen_dict["eval_triggers"]["top1_accuracy"] * 100.0, 1),
            round(me["bm25s_lexical_baseline"]["top1_shuffled_rate"] * 100.0, 1),
            round(me["finbert_financial_encoder"]["top1_shuffled_rate"] * 100.0, 1),
            round(me["bge_reranker_v2_m3"]["top1_shuffled_rate"] * 100.0, 1),
            round(me["jev_system_one_calibrated_router_ours"]["top1_shuffled_rate"] * 100.0, 1),
        ]

    gen0_acc = _get_enc_accs(g0)
    gen1_acc = _get_enc_accs(g1)
    gen3_acc = _get_enc_accs(g3)

    x = np.arange(len(encoders))
    width = 0.25

    bars0 = ax_a.bar(x - width, gen0_acc, width, label="Gen-0 Seed Corpus", color="#94A3B8", edgecolor="#475569", linewidth=0.7, zorder=3)
    bars1 = ax_a.bar(x, gen1_acc, width, label="Gen-1 Anchor Expansion", color="#60A5FA", edgecolor="#1D4ED8", linewidth=0.7, zorder=3)
    bars3 = ax_a.bar(x + width, gen3_acc, width, label="Gen-3 RSI Champion", color="#059669", edgecolor="#065F46", linewidth=0.8, zorder=3)

    ax_a.set_ylim(20.0, 113.5)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(encoders, fontsize=7.8)
    ax_a.set_ylabel("Top-1 Routing Accuracy (%)", fontsize=8.5, fontweight="bold", color="#0F172A")
    ax_a.set_title("(a) Layer A: Skill-Space RSI Across Encoders (N=108)", fontsize=8.8, fontweight="bold", color="#0F172A", pad=8)
    ax_a.grid(axis="y", linestyle="--", linewidth=0.5, color="#CBD5E1", alpha=0.8, zorder=0)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    for b in bars0:
        h = b.get_height()
        ax_a.text(b.get_x() + b.get_width() / 2, h + 1.0, f"{h:.0f}%", ha="center", va="bottom", fontsize=6.4, color="#475569")
    for b in bars3:
        h = b.get_height()
        ax_a.text(b.get_x() + b.get_width() / 2, h + 1.0, f"{h:.0f}%", ha="center", va="bottom", fontsize=6.7, fontweight="bold", color="#065F46")

    ax_a.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=7.0)

    # =========================================================================
    # Panel (b): Layer B — Operator-Space RSI Pareto Frontier (M=5 Seeds)
    # =========================================================================
    arms = fin_rsi["arms"]
    r1 = arms["Row_1_Full_Dense_Multimodal_Ref"]
    r2 = arms["Row_2_Prod_Baseline_Verbal_Reflexion_RSI"]
    r3 = arms["Row_3_Prod_Baseline_MMAN_Barra_Dual"]
    r4 = arms["Row_4_Prod_Baseline_JEV_SystemOne_Static_64KC"]
    r5 = arms["Row_5_RSI_Gen1_ValueSpace_BoundedESS"]
    r6 = arms["Row_6_RSI_Gen2_Subspace_Precision_Stein"]
    r7 = arms["Row_7_RSI_Gen3_Streaming_Woodbury_Fisher_JEV_64KC"]

    methods = [
        ("Verbal Reflexion", abs(r2["max_drawdown_pct"]), r2["annualized_net_sharpe"], r2["annualized_net_sharpe_std"], "#DC2626", "X", 75),
        ("MMAN Barra Dual", abs(r3["max_drawdown_pct"]), r3["annualized_net_sharpe"], r3["annualized_net_sharpe_std"], "#64748B", "s", 55),
        ("Full Dense Ref", abs(r1["max_drawdown_pct"]), r1["annualized_net_sharpe"], r1["annualized_net_sharpe_std"], "#475569", "D", 58),
        ("Gen-0 Static JEV-64KC", abs(r4["max_drawdown_pct"]), r4["annualized_net_sharpe"], r4["annualized_net_sharpe_std"], "#2563EB", "o", 65),
        ("Gen-1 Bounded-ESS", abs(r5["max_drawdown_pct"]), r5["annualized_net_sharpe"], r5["annualized_net_sharpe_std"], "#0284C7", "^", 68),
        ("Gen-2 Precision-Stein", abs(r6["max_drawdown_pct"]), r6["annualized_net_sharpe"], r6["annualized_net_sharpe_std"], "#0D9488", "v", 72),
        ("Gen-3 Woodbury+Fisher", abs(r7["max_drawdown_pct"]), r7["annualized_net_sharpe"], r7["annualized_net_sharpe_std"], "#059669", "*", 145),
    ]

    gen_dd = [m[1] for m in methods[3:]]
    gen_sh = [m[2] for m in methods[3:]]
    ax_b.plot(gen_dd, gen_sh, linestyle="-", linewidth=1.8, color="#059669", zorder=2, label="Operator-RSI Trajectory")

    for label, dd, sh, sh_std, col, marker, sz in methods:
        ax_b.errorbar(dd, sh, yerr=sh_std, fmt="none", ecolor=col, elinewidth=1.0, capsize=2.5, alpha=0.65, zorder=3)
        ax_b.scatter([dd], [sh], color=col, marker=marker, s=sz, edgecolors="#0F172A", linewidths=0.7, zorder=4)

    # Direct non-overlapping labels
    ax_b.annotate(f"Verbal Reflexion\n({r2['annualized_net_sharpe']:+.2f}, {r2['max_drawdown_pct']:.1f}% DD)",
                  xy=(abs(r2["max_drawdown_pct"]), r2["annualized_net_sharpe"]), xytext=(31.2, -0.12),
                  fontsize=6.8, color="#991B1B", fontweight="bold", ha="center",
                  arrowprops=dict(arrowstyle="->", color="#DC2626", lw=0.8))
    ax_b.annotate(f"MMAN Barra\n({r3['annualized_net_sharpe']:+.2f})",
                  xy=(abs(r3["max_drawdown_pct"]), r3["annualized_net_sharpe"]), xytext=(25.5, 0.52),
                  fontsize=6.8, color="#475569", ha="center", va="center",
                  arrowprops=dict(arrowstyle="->", color="#64748B", lw=0.7))
    ax_b.annotate(f"Gen-0 Static\n({r4['annualized_net_sharpe']:+.2f})",
                  xy=(abs(r4["max_drawdown_pct"]), r4["annualized_net_sharpe"]), xytext=(26.2, 1.18),
                  fontsize=7.0, color="#1E40AF", fontweight="bold", ha="center",
                  arrowprops=dict(arrowstyle="->", color="#2563EB", lw=0.8))
    ax_b.annotate(f"Gen-1 ({r5['annualized_net_sharpe']:+.2f})",
                  xy=(abs(r5["max_drawdown_pct"]), r5["annualized_net_sharpe"]), xytext=(23.8, 1.56),
                  fontsize=6.8, color="#0369A1", ha="center",
                  arrowprops=dict(arrowstyle="->", color="#0284C7", lw=0.7))
    ax_b.annotate(f"Full Dense ({r1['annualized_net_sharpe']:+.2f})",
                  xy=(abs(r1["max_drawdown_pct"]), r1["annualized_net_sharpe"]), xytext=(22.5, 1.92),
                  fontsize=6.8, color="#334155", ha="center",
                  arrowprops=dict(arrowstyle="->", color="#475569", lw=0.7))
    ax_b.annotate(f"Gen-2 ({r6['annualized_net_sharpe']:+.2f})",
                  xy=(abs(r6["max_drawdown_pct"]), r6["annualized_net_sharpe"]), xytext=(25.2, 0.85),
                  fontsize=6.8, color="#0F766E", ha="center",
                  arrowprops=dict(arrowstyle="->", color="#0D9488", lw=0.7))
    ax_b.annotate(f"Gen-3 Champion\n({r7['annualized_net_sharpe']:+.2f}, {r7['max_drawdown_pct']:.1f}% DD)",
                  xy=(abs(r7["max_drawdown_pct"]), r7["annualized_net_sharpe"]), xytext=(15.2, 2.22),
                  fontsize=7.2, color="#065F46", fontweight="bold", ha="center",
                  arrowprops=dict(arrowstyle="->", color="#059669", lw=1.0))

    ax_b.set_xlim(36.5, 9.5)
    ax_b.set_ylim(-1.05, 2.55)
    ax_b.set_xlabel("Max Drawdown Magnitude (%) [Lower / Right is Safer]", fontsize=8.1, fontweight="bold", color="#0F172A")
    ax_b.set_ylabel("OOS Annualized Net Sharpe (M=5)", fontsize=8.5, fontweight="bold", color="#0F172A")
    ax_b.set_title("(b) Layer B: Operator-Space RSI Pareto Frontier", fontsize=8.8, fontweight="bold", color="#0F172A", pad=8)
    ax_b.grid(True, linestyle="--", linewidth=0.5, color="#CBD5E1", alpha=0.8, zorder=0)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # =========================================================================
    # Panel (c): 2x2 Dual-Layer Fin-RSI Synergy Across 4 Architectures (M=5)
    # =========================================================================
    bb_eval = dual_syn["multi_backbone_2x2_evaluation"]
    arch_keys = [
        ("BM25S-\nLexical", "BM25S-Lexical"),
        ("FinBERT-\n110M", "ProsusAI/finbert (110M)"),
        ("BGE-v2-m3\n(568M)", "BAAI/bge-reranker-v2-m3 (568M)"),
        ("JEV+DeepSeek-\nR1-1.5B", "JEV System-One + DeepSeek-R1-Distill-1.5B"),
    ]
    backbones = [k[0] for k in arch_keys]
    s00 = [bb_eval[k[1]]["Gen0_Skill_Gen0_Op"]["net_sharpe"] for k in arch_keys]
    s03 = [bb_eval[k[1]]["Gen0_Skill_Gen3_Op"]["net_sharpe"] for k in arch_keys]
    s30 = [bb_eval[k[1]]["Gen3_Skill_Gen0_Op"]["net_sharpe"] for k in arch_keys]
    s33 = [bb_eval[k[1]]["Gen3_Skill_Gen3_Op"]["net_sharpe"] for k in arch_keys]
    s33_err = [bb_eval[k[1]]["Gen3_Skill_Gen3_Op"]["net_sharpe_std"] for k in arch_keys]
    syn_deltas = [bb_eval[k[1]]["synergy_sharpe"] for k in arch_keys]

    xb = np.arange(len(backbones))
    w4 = 0.185

    ax_c.bar(xb - 1.5 * w4, s00, w4, label="(Gen-0 Skill, Gen-0 Op)", color="#CBD5E1", edgecolor="#475569", linewidth=0.7, zorder=3)
    ax_c.bar(xb - 0.5 * w4, s03, w4, label="(Gen-0 Skill, Gen-3 Op)", color="#93C5FD", edgecolor="#1D4ED8", linewidth=0.7, zorder=3)
    ax_c.bar(xb + 0.5 * w4, s30, w4, label="(Gen-3 Skill, Gen-0 Op)", color="#6EE7B7", edgecolor="#047857", linewidth=0.7, zorder=3)
    b33 = ax_c.bar(
        xb + 1.5 * w4, s33, w4, yerr=s33_err, capsize=2.2,
        error_kw=dict(elinewidth=0.9, ecolor="#064E3B"),
        label="(Gen-3, Gen-3) Champion", color="#059669", edgecolor="#064E3B", linewidth=0.8, zorder=4
    )

    for idx, b in enumerate(b33):
        h = b.get_height()
        ax_c.text(
            b.get_x() + b.get_width() / 2, h + s33_err[idx] + 0.04,
            f"{h:.2f}\n(+{syn_deltas[idx]:.2f})",
            ha="center", va="bottom", fontsize=6.3, fontweight="bold", color="#065F46"
        )

    ax_c.set_ylim(0.0, 2.75)
    ax_c.set_xticks(xb)
    ax_c.set_xticklabels(backbones, fontsize=7.6)
    ax_c.set_ylabel("OOS Annualized Net Sharpe", fontsize=8.5, fontweight="bold", color="#0F172A")
    ax_c.set_title("(c) 2x2 Dual-Layer Fin-RSI Synergy (p = 0.0375)", fontsize=8.8, fontweight="bold", color="#0F172A", pad=8)
    ax_c.grid(axis="y", linestyle="--", linewidth=0.5, color="#CBD5E1", alpha=0.8, zorder=0)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.legend(loc="upper center", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=6.6, ncol=2)

    fig.tight_layout(pad=1.15)

    outputs: list[Path] = []
    for target_dir in (FIN_SKILLS_FIG_DIR, STOCK_PRED_FIG_DIR):
        target_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = target_dir / "fig2_dual_layer_empirical_results.pdf"
        png_path = target_dir / "fig2_dual_layer_empirical_results.png"
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
        fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.04)
        outputs.extend([pdf_path, png_path])
    plt.close(fig)
    return outputs


def main() -> int:
    out1 = generate_fig1_architecture()
    out2 = generate_fig2_empirical_results()
    print("Generated Figure 1 files:", [str(p) for p in out1])
    print("Generated Figure 2 files:", [str(p) for p in out2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
