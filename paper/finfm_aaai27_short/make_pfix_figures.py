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


TEXTW = 7.0    # AAAI text width, inches


def nature(ax, x=True, y=True):
    """Journal style: left and bottom spines only, thin, outward ticks, no grid."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(y)
    ax.spines["bottom"].set_visible(x)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.5)
        ax.spines[side].set_color(fs.MUTED)
    ax.tick_params(axis="both", direction="out", length=2.5, width=0.5, pad=1.5, labelsize=7,
                   colors=fs.INK2)
    if not y:
        ax.tick_params(axis="y", length=0)
    if not x:
        ax.tick_params(axis="x", length=0)


def panel(ax, letter, title, x=-0.02, y=1.0):
    """Bold lower-case panel letter and a short title, left-aligned above the axes."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            color=fs.INK, ha="right", va="bottom")
    ax.text(x + 0.03, y, title, transform=ax.transAxes, fontsize=7.5, color=fs.INK,
            ha="left", va="bottom")


def headline_figure(fix):
    """Page 1: every model's valid rate without the library, with its interface, and after
    route(); the call named by us (F2) as a reference."""
    per, big, small = fix["per_model"], fix["by_size"]["ge14b"], fix["by_size"]["lt14b"]
    xs = {"F0": 0, "F1": 1, "F4": 2, "F2": 3.25}
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.72), sharey=True, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.06)
    for ax, group, colour, letter, title in (
            (axes[0], big, LARGE, "a", f"{len(big['models'])} models of 14B or more"),
            (axes[1], small, SMALL, "b", f"{len(small['models'])} smaller models")):
        nature(ax)
        ax.axvspan(1.6, 2.4, color=fs.mix(colour, "#ffffff", 0.08), linewidth=0, zorder=0)
        ax.axvline(2.75, color=fs.GRID, linewidth=0.6, linestyle=(0, (2, 2)), zorder=0)
        light = fs.mix(colour, "#ffffff", 0.32)
        for m in group["models"]:
            ys = [100 * per[m][a]["valid"] / per[m][a]["n"] for a in ("F0", "F1", "F4")]
            ax.plot([0, 1, 2], ys, color=light, linewidth=0.7, zorder=1, solid_capstyle="round")
            ax.scatter([xs["F2"]], [100 * per[m]["F2"]["valid"] / per[m]["F2"]["n"]], s=6,
                       color=light, linewidth=0, zorder=1)
        pooled = group["pooled"]
        mean = [100 * pooled[a]["valid"] / pooled[a]["n"] for a in ("F0", "F1", "F4")]
        ax.plot([0, 1, 2], mean, color=colour, linewidth=2.0, zorder=3, solid_capstyle="round")
        ax.scatter([0, 1, 2], mean, s=22, color=colour, edgecolor="white", linewidth=0.8, zorder=4)
        f2 = 100 * pooled["F2"]["valid"] / pooled["F2"]["n"]
        ax.plot([xs["F2"] - 0.22, xs["F2"] + 0.22], [f2, f2], color=fs.INK2, linewidth=1.4, zorder=3)
        ax.annotate(f"{mean[2]:.1f}%", (2, mean[2]), xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, color=fs.INK, fontweight="bold")
        ax.annotate(f"{mean[0]:.1f}%", (0, mean[0]), xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, color=fs.INK)
        ax.set_xticks([0, 1, 2, xs["F2"]], ["F0\nnone", "F1\nAPI", "F4\nroute", "F2\nnamed"],
                      fontsize=7, linespacing=1.0)
        ax.set_xlim(-0.45, 3.65)
        ax.set_ylim(-3, 103)
        ax.set_yticks([0, 50, 100], ["0", "50", "100"])
        panel(ax, letter, title, x=0.0, y=1.03)
    axes[0].set_ylabel("valid (%)", fontsize=7, labelpad=1, color=fs.INK2)
    save(fig, "pfix_headline")


def analysis_figure(fix):
    """Page 4: a valid rate by arm with every model; b after the right rule was named; c checks
    failed without and with the library; d flagged programs that were then fixed."""
    per, sizes = fix["per_model"], fix["sizes_b"]
    big, small = fix["by_size"]["ge14b"], fix["by_size"]["lt14b"]
    fig = plt.figure(figsize=(TEXTW, 1.82), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.01, wspace=0.05)
    ax_a, ax_b, ax_c, ax_d = fig.subplots(1, 4, width_ratios=[1.3, 1.28, 1.22, 1.1])

    # a: bars per arm and size group, each model as a dot
    nature(ax_a)
    w = 0.34
    for i, arm in enumerate(ARMS):
        for j, (group, colour) in enumerate(((big, LARGE), (small, SMALL))):
            p = group["pooled"][arm]
            x = i + (j - 0.5) * (w + 0.05)
            ax_a.bar(x, 100 * p["valid"] / p["n"], width=w, color=fs.mix(colour, "#ffffff", 0.28),
                     linewidth=0, zorder=1)
            vals = sorted(100 * per[m][arm]["valid"] / per[m][arm]["n"] for m in group["models"])
            off = [((k % 3) - 1) * 0.07 for k in range(len(vals))]
            ax_a.scatter([x + o for o in off], vals, s=5, color=colour, linewidth=0, zorder=2)
    ax_a.set_xticks(range(len(ARMS)), [f"{a}" for a in ARMS])
    ax_a.set_ylim(0, 119)
    ax_a.spines["left"].set_bounds(0, 100)
    ax_a.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100"])
    ax_a.set_ylabel("valid (%)", fontsize=7, labelpad=1, color=fs.INK2)
    ax_a.legend(handles=[Patch(color=fs.mix(LARGE, "#ffffff", 0.28), label="14B or more"),
                         Patch(color=fs.mix(SMALL, "#ffffff", 0.28), label="smaller"),
                         Line2D([], [], marker="o", linestyle="", color=fs.INK2, markersize=2.5,
                                label="a model")],
                loc="upper left", ncol=3, frameon=False, fontsize=7, handlelength=0.8,
                handleheight=0.7, borderaxespad=0.0, columnspacing=0.6, handletextpad=0.3)
    panel(ax_a, "a", "Valid episodes by arm", x=0.0, y=1.03)

    # b: F4 after a right route: did the last program call positions(), and was it valid?
    nature(ax_b, y=False)
    y, ticks, labels = 0, [], []
    for name, t, colour in (("14B or more", big["told"]["F4"], LARGE),
                            ("smaller", small["told"]["F4"], SMALL)):
        ax_b.text(0, y - 0.66, f"{name}: {t['episodes']} right routes", ha="left", va="center",
                  fontsize=7, color=fs.INK)
        for what, total_key, valid_key in (("called positions()", "last_calls_positions",
                                            "last_calls_positions_valid"),
                                           ("own rule", "last_own_rule", "last_own_rule_valid")):
            total, valid = t.get(total_key, 0), t.get(valid_key, 0)
            ax_b.barh(y, total, height=0.62, color=fs.mix(colour, "#ffffff", 0.3), linewidth=0)
            if valid:
                ax_b.barh(y, valid, height=0.62, color=colour, linewidth=0)
            ax_b.text(total + 2, y, f"{valid}/{total} valid", ha="left", va="center", fontsize=7,
                      color=fs.INK)
            ticks.append(y)
            labels.append(what)
            y += 1
        y += 0.95
    ax_b.set_yticks(ticks, labels)
    ax_b.tick_params(axis="y", length=0, pad=2)
    ax_b.set_ylim(y - 0.75, -1.2)
    ax_b.set_xlim(0, 150)
    ax_b.set_xticks([0, 50, 100], ["0", "50", "100"])
    ax_b.spines["bottom"].set_bounds(0, 100)
    ax_b.set_xlabel("F4 episodes", fontsize=7, labelpad=1, color=fs.INK2)
    panel(ax_b, "b", "Told the right rule", x=-0.55, y=1.03)

    # c: share of programs that ran failing each check, F0 against F4
    nature(ax_c, y=False)
    fc = fix["failed_checks"]
    for yy, (key, label) in enumerate(CHECKS):
        a0 = 100 * fc["F0"][key] / fc["F0"]["programs_ran"]
        a4 = 100 * fc["F4"][key] / fc["F4"]["programs_ran"]
        ax_c.plot([a0, a4], [yy, yy], color=fs.mix(fs.ORANGE, "#ffffff", 0.35), linewidth=1.4,
                  solid_capstyle="round", zorder=1)
        ax_c.scatter([a0], [yy], s=16, facecolor="white", edgecolor=fs.ORANGE, linewidth=1.0, zorder=3)
        ax_c.scatter([a4], [yy], s=18, facecolor=fs.ORANGE, edgecolor="white", linewidth=0.6, zorder=4)
        ax_c.text(-2, yy, label, ha="right", va="center", fontsize=7, color=fs.INK2)
    ax_c.set_ylim(len(CHECKS) + 0.35, -0.8)
    ax_c.set_yticks([])
    ax_c.set_xlim(0, 62)
    ax_c.set_xticks([0, 20, 40, 60], ["0", "20", "40", "60%"])
    ax_c.legend(handles=[Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
                                markeredgecolor=fs.ORANGE, markersize=4, label="F0"),
                         Line2D([], [], marker="o", linestyle="", markerfacecolor=fs.ORANGE,
                                markeredgecolor="white", markersize=4.4, label="F4")],
                loc="lower right", bbox_to_anchor=(1.0, 0.0), frameon=False, fontsize=7,
                handletextpad=0.1, ncol=2, columnspacing=0.6, borderaxespad=0.0)
    ax_c.set_xlabel("of the programs that ran", fontsize=7, labelpad=1, color=fs.INK2)
    panel(ax_c, "c", "Checks failed", x=-0.62, y=1.03)

    # d: episodes in which a new check flagged a program and an attempt was left: ended valid?
    nature(ax_d)
    rp = fix["repair"]
    for i, arm in enumerate(ARMS):
        r = rp[arm]
        share = 100 * r["then_valid"] / r["with_another_attempt"]
        ax_d.bar(i, share, width=0.6, color=fs.mix(fs.ORANGE, "#ffffff", 0.75), linewidth=0)
        ax_d.text(i, share + 1.2, f"{r['then_valid']}/{r['with_another_attempt']}", ha="center",
                  va="bottom", fontsize=7, color=fs.INK, rotation=90)
    ax_d.set_xticks(range(len(ARMS)), ARMS)
    ax_d.set_ylim(0, 40)
    ax_d.set_yticks([0, 10, 20, 30, 40], ["0", "10", "20", "30", "40"])
    ax_d.set_ylabel("ended valid (%)", fontsize=7, labelpad=1, color=fs.INK2)
    panel(ax_d, "d", "Flagged, then fixed", x=-0.3, y=1.03)
    save(fig, "pfix_analysis")


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
    headline_figure(fix)
    analysis_figure(fix)
    checks_figure(fix)
    routing_figure(fix)
    frozen_figure(frozen)
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.pdf"))))


if __name__ == "__main__":
    main()
