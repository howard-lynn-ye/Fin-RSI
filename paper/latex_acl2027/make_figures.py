"""Draw the data figures for the Fin-RSI paper from the task suite's own data.

    python paper/latex_acl2027/make_figures.py

figures/task_checks.pdf:
(a) One asset of the seed-11 panel: the raw quotes a program receives and the adjusted closes
    it should compute from them. (b) The execution_lag check on two inverse-volatility
    programs: the library's lagged positions and the same positions read on the close they
    trade over. Prices from row k onward are perturbed exactly as the guard perturbs them;
    a cell is filled where a position moved.

figures/real_panel.pdf: the real-price panel (fin_skills.tasks.real), one small multiple per
    stock: raw quote and adjusted close, both indexed to 100 at the first close, on a log
    scale. Needs the fetched panel; the figure shows no vendor price level.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
SEED, ASSET, K, WINDOW = 11, "C", 64, 20
SHOW = range(56, 73)                       # rows drawn in panel (b)
# Reference palette, categorical slots 1-3 (validated all-pairs, light surface).
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, UNMOVED = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#f1f0ed"


def perturb(frame, k: int):
    """The guard's second perturbation: rows >= k times a per-column factor (seed 0)."""
    out = frame.copy()
    factors = np.random.default_rng(0).uniform(0.5, 1.5, size=frame.shape[1])
    out.iloc[k:] = out.iloc[k:].to_numpy() * factors
    return out


STYLE = {
    "font.family": "TeX Gyre Termes", "mathtext.fontset": "stix", "font.size": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "pdf.fonttype": 42}


def real_panel_figure() -> None:
    import matplotlib.dates as mdates
    from matplotlib.ticker import FixedLocator, NullLocator
    from fin_skills.tasks import real

    raw, _, adjusted = real.panel()
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.55), sharey=True,
                             gridspec_kw=dict(wspace=0.12))
    names = {"AMZN": "Amazon", "SHOP": "Shopify", "GOOGL": "Alphabet (class A)",
             "TSLA": "Tesla"}
    for ax, (ticker, date, ratio, _) in zip(axes, real.SPLITS):
        r = 100 * raw[ticker] / raw[ticker].iloc[0]
        a = 100 * adjusted[ticker] / adjusted[ticker].iloc[0]
        ax.plot(a.index, a, color=BLUE, lw=1.2, solid_capstyle="round", zorder=3)
        ax.plot(r.index, r, color=ORANGE, lw=1.2, solid_capstyle="round", zorder=2)
        when = np.datetime64(date)
        ax.axvline(when, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
        day = np.datetime64(date).astype(object)
        ax.text(when, 185, f" {ratio:g}-for-1, {day.day} {day.strftime('%b')}", color=INK2,
                fontsize=6.5, va="top", ha="left")
        ax.set_yscale("log")
        ax.set_ylim(2.0, 210)
        ax.yaxis.set_major_locator(FixedLocator([3, 10, 30, 100]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_yticklabels(["3", "10", "30", "100"])
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(4, 6, 8)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.set_xlim(raw.index[0], raw.index[-1])
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(names[ticker], fontsize=8, loc="left", color=INK, pad=3)
    first = axes[0]                        # direct labels once, where there is room
    first.text(np.datetime64("2022-08-15"), 62, "adjusted", color=INK, fontsize=6.5,
               ha="center", va="top")
    first.text(np.datetime64("2022-08-15"), 5.9, "raw", color=INK, fontsize=6.5,
               ha="center", va="bottom")
    axes[0].set_ylabel("close, first = 100", labelpad=2)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=BLUE, lw=1.2, label="adjusted close"),
                        Line2D([], [], color=ORANGE, lw=1.2, label="raw quote (as traded)")],
               loc="upper right", bbox_to_anchor=(0.995, 1.13), ncol=2, frameon=False,
               fontsize=7, handlelength=1.6, labelcolor=INK)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "real_panel.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "real_panel.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    print("real panel", raw.index[0].date(), raw.index[-1].date())


def main() -> None:
    from fin_skills.algorithms import positions
    from fin_skills.tasks import panel

    raw, factors, adjusted = panel(SEED)
    split_row = int(np.flatnonzero(np.diff(factors[ASSET].to_numpy()))[0]) + 1

    def lagged(prices):
        return positions("inverse_volatility", prices, factors, window=WINDOW)

    def same_close(prices):                # decides row t with the close of row t
        return lagged(prices).shift(-1).fillna(0.0)

    moved = {}
    for name, fn in (("lagged (correct)", lagged), ("same close (defect)", same_close)):
        delta = (fn(raw) - fn(perturb(raw, K))).abs().max(axis=1).to_numpy()
        moved[name] = delta > 1e-8

    plt.rcParams.update(STYLE)
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.03, 2.75),
                               gridspec_kw=dict(height_ratios=[1.7, 1.0], hspace=0.85))

    rows = np.arange(len(raw))
    a.plot(rows, adjusted[ASSET], color=BLUE, lw=1.2, label="adjusted close")
    a.plot(rows, raw[ASSET], color=ORANGE, lw=1.2, label="raw quote")
    a.axvline(split_row, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    a.text(split_row + 1.5, a.get_ylim()[1] * 0.97, "3-for-1 split", color=INK2, va="top",
           fontsize=7)
    a.text(rows[-1] + 1.5, adjusted[ASSET].iloc[-1], "adjusted", color=INK, va="center",
           fontsize=7)
    a.text(rows[-1] + 1.5, raw[ASSET].iloc[-1], "raw", color=INK, va="center", fontsize=7)
    jump = raw[ASSET].iloc[split_row] / raw[ASSET].iloc[split_row - 1] - 1
    pct = f"{jump:.0%}".replace("-", "\u2212")         # a true minus sign, not a hyphen
    a.annotate(f"ignore the split and this\nis a {pct} one-day return",
               xy=(split_row + 0.6, (raw[ASSET].iloc[split_row - 1] + raw[ASSET].iloc[split_row]) / 2),
               xytext=(split_row + 9, 82), fontsize=6.5, color=INK, va="center",
               arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.6, shrinkA=1, shrinkB=1,
                               mutation_scale=6, connectionstyle="arc3,rad=0.25"))
    a.set_xlim(0, rows[-1])
    a.set_xlabel("row", labelpad=1)
    a.set_ylabel("price of asset C", labelpad=2)
    a.grid(axis="y", color=GRID, lw=0.5)
    a.set_axisbelow(True)
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
    a.legend(loc="lower left", frameon=False, fontsize=7, handlelength=1.6,
             borderaxespad=0.2, labelcolor=INK)
    a.set_title("(a) what the program receives, and what it must compute", fontsize=8,
                loc="left", color=INK, pad=4)

    names = list(moved)[::-1]              # correct program on top
    b.add_patch(Rectangle((K - 0.5, -0.55), 1.0, 2.1, facecolor=ORANGE, alpha=0.12,
                          edgecolor="none", zorder=0))
    for y, name in enumerate(names):
        for x in SHOW:
            hit = moved[name][x]
            color = (ORANGE if x <= K else BLUE) if hit else UNMOVED
            b.add_patch(Rectangle((x - 0.45, y - 0.4), 0.9, 0.8, facecolor=color,
                                  edgecolor="none"))
    b.plot([K - 0.5, K - 0.5], [-0.55, 1.5], color=INK2, lw=0.8)
    b.text(K - 0.5, 1.75, f"prices perturbed from row k = {K}", ha="left", va="center",
           color=INK2, fontsize=7)
    for y, name in enumerate(names):
        verdict = "fails" if moved[name][K] else "passes"
        b.text(SHOW.stop - 0.2, y, verdict, va="center", ha="left", color=INK, fontsize=7,
               fontweight="bold" if verdict == "fails" else "normal")
    b.set_xlim(SHOW.start - 0.6, SHOW.stop - 0.4)
    b.set_ylim(-0.6, 2.05)
    b.set_yticks(range(len(names)), names)
    b.set_xticks([x for x in SHOW if x % 4 == 0])
    b.tick_params(axis="y", length=0)
    b.set_xlabel("row", labelpad=1)
    for side in ("top", "right", "left"):
        b.spines[side].set_visible(False)
    b.legend(handles=[Patch(color=ORANGE, label="moved at or before k: check fails"),
                      Patch(color=BLUE, label="moved after k: allowed"),
                      Patch(color=UNMOVED, label="unchanged")],
             loc="upper center", bbox_to_anchor=(0.42, -0.5), ncol=2, frameon=False,
             fontsize=6.5, handlelength=1.0, columnspacing=0.8, labelcolor=INK)
    b.set_title("(b) execution_lag on two inverse-volatility programs", fontsize=8,
                loc="left", color=INK, pad=12)

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "task_checks.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "task_checks.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    first = {n: int(np.flatnonzero(moved[n])[0]) for n in names}
    print("split row", split_row, "first moved row", first)
    real_panel_figure()


if __name__ == "__main__":
    main()
