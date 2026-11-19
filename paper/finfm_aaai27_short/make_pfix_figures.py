"""Data figures of the short paper, drawn from the same pinned files as make_pfix_results.py.

    python make_pfix_figures.py      # writes figures/pfix_*.pdf and figures/frozen_noLib.pdf

Colours and fonts come from ../latex_acl2027/figstyle.py (read from figstyle.tex). Series colour
is the model-size group throughout: blue for models of at least 14B parameters, magenta for the
smaller ones (validated pair). Orange is reserved for checks, aqua for self-improvement.
"""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "latex_acl2027"))
import figstyle as fs  # noqa: E402
from make_pfix_results import ANALYSIS, ANALYSIS_SHA256, CORE, EXTENSION, METHODS, SUMMARY, \
    SUMMARY_SHA256  # noqa: E402

OUT = HERE / "figures"
META = {"CreationDate": None, "ModDate": None, "Creator": None, "Producer": None}
COL = 3.3      # AAAI column width, inches
LARGE, SMALL = fs.BLUE, fs.COLOR["finmagenta"]
ARMS = ["F0", "F1", "F2", "F3", "F4"]
ARM_LABEL = {"F0": "F0\nnone", "F1": "F1\ninterface", "F2": "F2\n+ call",
             "F3": "F3\n+ rules", "F4": "F4\n+ route()"}
CHECKS = [("execution_lag", "execution lag"), ("warmup_rows_zero", "warm-up rows zero"),
          ("takes_positions", "takes positions"), ("position_scale", "position scale"),
          ("split_representation", "split representation"), ("finite_output", "finite output")]
METHOD_LABEL = {"no_improve": "NoImprove", "self_debug": "SelfDebug", "check_feedback": "CheckFeedback",
                "reflexion": "Reflexion", "team": "Team", "evolve": "Evolve",
                "playbook": "PlaybookLearning"}


def load():
    raw = ANALYSIS.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ANALYSIS_SHA256, "not the pinned P-fix analysis"
    summary = SUMMARY.read_bytes()
    assert hashlib.sha256(summary).hexdigest() == SUMMARY_SHA256, "not the pinned frozen summary"
    return json.loads(raw), json.loads(summary)["models"]


def style(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left" if grid_axis == "y" else "bottom"].set_visible(False)
    ax.tick_params(length=0, pad=2)
    ax.grid(axis=grid_axis, color=fs.GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    # Constrained layout at the exact column width: the PDF is placed at 100%, never scaled.
    fig.savefig(OUT / f"{name}.pdf", metadata=META)
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)


def main_figure(fix):
    """(a) valid rate by arm and size group; (b) every model's valid episodes, F0 against F4."""
    big, small = fix["by_size"]["ge14b"], fix["by_size"]["lt14b"]
    sizes = fix["sizes_b"]
    per = fix["per_model"]
    key = lambda m: (-per[m]["F4"]["valid"], -per[m]["F0"]["valid"], m)  # noqa: E731
    rows_big = sorted(big["models"], key=key)
    rows_small = sorted(small["models"], key=key)
    zero = [m for m in rows_small if per[m]["F0"]["valid"] == per[m]["F4"]["valid"] == 0]
    rows_small = [m for m in rows_small if m not in zero]
    n_rows = len(rows_big) + len(rows_small)
    h_a, h_b = 1.18, 0.112 * (n_rows + 2.2) + 0.62
    fig = plt.figure(figsize=(COL, h_a + h_b), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.015, w_pad=0.01)
    top, bottom = fig.subfigures(2, 1, height_ratios=[h_a, h_b], hspace=0.04)

    ax = top.subplots()
    style(ax)
    w, gap = 0.36, 0.04
    for i, arm in enumerate(ARMS):
        for j, (group, colour) in enumerate(((big, LARGE), (small, SMALL))):
            p = group["pooled"][arm]
            v = 100 * p["valid"] / p["n"]
            x = i + (j - 0.5) * (w + gap)
            ax.bar(x, v, width=w, color=colour, linewidth=0)
            ax.text(x, v + 2.2, f"{v:.0f}" if v >= 1 or v == 0 else f"{v:.1f}", ha="center",
                    va="bottom", fontsize=7, color=fs.INK)
    ax.set_xticks(range(len(ARMS)), [ARM_LABEL[a] for a in ARMS], fontsize=7, linespacing=1.0)
    ax.set_ylim(0, 75)
    ax.set_yticks([0, 25, 50, 75], ["0", "25", "50", "75%"], fontsize=7)
    ax.set_xlim(-0.6, len(ARMS) - 0.4)
    ax.legend(handles=[Patch(color=LARGE, label=f"{len(big['models'])} models of 14B or more"),
                       Patch(color=SMALL, label=f"{len(small['models'])} smaller models")],
              loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, frameon=False, fontsize=7,
              handlelength=0.9, handleheight=0.7, borderaxespad=0.1, columnspacing=1.0)
    top.suptitle("(a) Valid episodes by arm", x=0.0, ha="left", fontsize=7.5, fontweight="bold",
                 color=fs.INK)

    ax = bottom.subplots()
    style(ax, grid_axis="x")
    y, ticks, labels = 0, [], []
    for group, colour in ((rows_big, LARGE), (rows_small, SMALL)):
        for m in group:
            f0, f4 = per[m]["F0"]["valid"], per[m]["F4"]["valid"]
            ax.plot([f0, f4], [y, y], color=fs.mix(colour, "#ffffff", 0.45), linewidth=1.6,
                    solid_capstyle="round", zorder=1)
            ax.scatter([f0], [y], s=16, facecolor="white", edgecolor=colour, linewidth=1.0, zorder=3)
            ax.scatter([f4], [y], s=18, facecolor=colour, edgecolor="white", linewidth=0.6, zorder=4)
            ticks.append(y)
            labels.append(f"{m} ({sizes[m]:g}B)")
            y += 1
        y += 0.4
    ax.text(-0.3, y, f"and {len(zero)} smaller models at 0 in both", fontsize=7, color=fs.INK2,
            va="center", ha="left")
    ax.set_yticks(ticks, labels, fontsize=7)
    ax.set_xlim(-0.6, 18.6)
    ax.set_ylim(y + 0.7, -0.7)
    ax.set_xticks([0, 6, 12, 18], ["0", "6", "12", "18"], fontsize=7)
    ax.set_xlabel("valid episodes of 18", fontsize=7, labelpad=2)
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
                              markeredgecolor=fs.INK2, markersize=4.2, label="F0, no library"),
                       Line2D([], [], marker="o", linestyle="", markerfacecolor=fs.INK2,
                              markeredgecolor="white", markersize=4.6, label="F4, after route()")],
              loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, frameon=False, fontsize=7,
              handletextpad=0.2, borderaxespad=0.1, columnspacing=1.0)
    bottom.suptitle("(b) Each model, F0 and F4", x=0.0, ha="left", fontsize=7.5, fontweight="bold",
                    color=fs.INK)
    save(fig, "pfix_main")


def checks_figure(fix):
    """Share of the programs that ran which failed each visible check, per arm."""
    fc = fix["failed_checks"]
    fig, ax = plt.subplots(figsize=(COL, 1.55), layout="constrained")
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list(
        "checks", [fs.mix(fs.ORANGE, "#ffffff", 0.06), fs.ORANGE])
    for i, arm in enumerate(ARMS):
        ran = fc[arm]["programs_ran"]
        for j, (key, _) in enumerate(CHECKS):
            share = fc[arm][key] / ran
            colour = ramp(min(share / 0.6, 1.0))
            ax.add_patch(matplotlib.patches.FancyBboxPatch(
                (i + 0.04, j + 0.06), 0.92, 0.88, boxstyle="round,pad=0,rounding_size=0.06",
                facecolor=colour, edgecolor="none"))
            ink = "white" if share > 0.33 else fs.INK
            ax.text(i + 0.5, j + 0.5, f"{100 * share:.0f}", ha="center", va="center", fontsize=7,
                    color=ink)
    ax.set_xlim(0, len(ARMS))
    ax.set_ylim(len(CHECKS), 0)
    ax.set_xticks([i + 0.5 for i in range(len(ARMS))],
                  [f"{a}\n{fc[a]['programs_ran']} ran" for a in ARMS], fontsize=7, linespacing=1.0)
    ax.xaxis.tick_top()
    ax.set_yticks([j + 0.5 for j in range(len(CHECKS))], [c[1] for c in CHECKS], fontsize=7)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0, pad=2)
    save(fig, "pfix_checks")


def routing_figure(fix):
    """F4: how often route() returned the task's rule, nothing or another rule, per model."""
    per, routing = fix["per_model"], fix["routing_by_model"]
    sizes = fix["sizes_b"]
    models = sorted(per, key=lambda m: (-sizes[m], m))
    fig, ax = plt.subplots(figsize=(COL, 0.112 * len(models) + 0.75), layout="constrained")
    style(ax, grid_axis="x")
    for y, m in enumerate(models):
        r = routing.get(m, {})
        right, wrong, none = r.get("right", 0), r.get("wrong", 0), r.get("none", 0)
        assert right + wrong + none == per[m]["F4"]["n"], m
        colour = LARGE if sizes[m] >= 14 else SMALL
        ax.barh(y, 18, height=0.62, color=fs.UNMOVED, linewidth=0)
        ax.barh(y, right, height=0.62, color=colour, linewidth=0)
        if wrong:
            ax.barh(y, wrong, left=right + 0.15, height=0.62, color=fs.ORANGE, linewidth=0)
        ax.text(18.6, y, f"{per[m]['F4']['valid']}", va="center", ha="left", fontsize=7,
                color=fs.INK)
    ax.text(18.6, -1.15, "valid", va="center", ha="left", fontsize=7, color=fs.INK2)
    ax.set_yticks(range(len(models)), [f"{m} ({sizes[m]:g}B)" for m in models], fontsize=7)
    ax.set_ylim(len(models) - 0.4, -1.6)
    ax.set_xlim(0, 18)
    ax.set_xticks([0, 6, 12, 18], ["0", "6", "12", "18"], fontsize=7)
    ax.set_xlabel("F4 episodes of 18", fontsize=7, labelpad=2)
    ax.legend(handles=[Patch(color=LARGE, label="task's rule (14B or more)"),
                       Patch(color=SMALL, label="task's rule (smaller)"),
                       Patch(color=fs.ORANGE, label="another rule"),
                       Patch(color=fs.UNMOVED, label="none")],
              loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=2, frameon=False, fontsize=7,
              handlelength=0.9, handleheight=0.7, columnspacing=0.8, labelspacing=0.25)
    save(fig, "pfix_routing")


def frozen_figure(frozen):
    """The study of 8 November without the library: feedback arms and the seven methods."""
    models = CORE + EXTENSION
    arms = [("repeat", "Repeat"), ("check", "Check"), ("explain", "Explain")]
    arm_v = [sum(frozen[m]["P"][f"{a}|0"][0] for m in models) for a, _ in arms]
    arm_n = sum(frozen[m]["P"]["repeat|0"][1] for m in models)
    s_models = [m for m in CORE if "S" in frozen[m]]
    meths = ["no_improve"] + METHODS
    meth_v = [sum(frozen[m]["S"][f"{k}|0"][0] for m in s_models) for k in meths]
    meth_n = sum(frozen[m]["S"]["no_improve|0"][1] for m in s_models)
    order = sorted(range(len(meths)), key=lambda i: (meths[i] != "no_improve", -meth_v[i]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL, 1.45), width_ratios=[1, 2.1],
                                 layout="constrained")
    for ax, labels, values, n, title in (
            (a1, [a[1] for a in arms], arm_v, arm_n, f"(a) Arms, {len(models)} models"),
            (a2, [METHOD_LABEL[meths[i]] for i in order], [meth_v[i] for i in order], meth_n,
             f"(b) Methods, {len(s_models)} core models")):
        style(ax, grid_axis="x")
        for y, (lab, v) in enumerate(zip(labels, values)):
            share = 100 * v / n
            ax.barh(y, share, height=0.6, color=fs.AQUA if lab not in ("Repeat", "NoImprove")
                    else fs.mix(fs.AQUA, "#ffffff", 0.45), linewidth=0)
            ax.text(share + 0.6, y, f"{v}", va="center", ha="left", fontsize=7, color=fs.INK)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.invert_yaxis()
        top, ticks = (25, [0, 10, 20]) if ax is a2 else (15, [0, 5, 10, 15])
        ax.set_xlim(0, top)
        ax.set_xticks(ticks, [str(t) for t in ticks], fontsize=7)
        ax.set_xlabel(f"% valid of {n}", fontsize=7, labelpad=2)
        ax.set_title(title, loc="left", fontsize=7.5, fontweight="bold", color=fs.INK, pad=4)
    save(fig, "frozen_nolib")


def main():
    plt.rcParams.update(fs.RC)
    fix, frozen = load()
    main_figure(fix)
    checks_figure(fix)
    routing_figure(fix)
    frozen_figure(frozen)
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.pdf"))))


if __name__ == "__main__":
    main()
