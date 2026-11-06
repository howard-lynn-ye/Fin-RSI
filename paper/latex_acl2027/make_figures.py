"""Draw the data figures for the Fin-RSI paper from the task suite's own data.

    python paper/latex_acl2027/make_figures.py

figures/episode_split.pdf: Figure 2 (a), see episode_split(). Needs the fetched real panel.

figures/episode_lag.tex: Figure 2 (c) as TikZ data, see episode_lag(). Synthetic panel only.

figures/real_panel.pdf: the real-price panel (fin_skills.tasks.real), one small multiple per
    stock: raw quote and adjusted close, both indexed to 100 at the first close, on a log
    scale. Needs the fetched panel; the figure shows no vendor price level.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
SEED, ASSET, K, WINDOW = 11, "C", 64, 20
SHOW = range(56, 73)                       # rows drawn in Figure 2 (c)
# Colours: figures/figstyle.tex, read by figstyle.py (reference palette, validated).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import BLUE, GRID, INK, INK2, MUTED, ORANGE, RC, SHORT, UNMOVED  # noqa: E402,F401


def perturb(frame, k: int):
    """The guard's second perturbation: rows >= k times a per-column factor (seed 0)."""
    out = frame.copy()
    factors = np.random.default_rng(0).uniform(0.5, 1.5, size=frame.shape[1])
    out.iloc[k:] = out.iloc[k:].to_numpy() * factors
    return out


STYLE = RC                                 # figstyle.py: the shared figure style
META = {"CreationDate": None}               # no timestamp, so a rerun does not change the PDFs


def real_panel_figure() -> None:
    import matplotlib.dates as mdates
    from matplotlib.ticker import FixedLocator, NullLocator
    from fin_skills.tasks import real

    raw, _, adjusted = real.panel()
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 4, figsize=(6.2, 1.5), sharey=True,
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
        late = day.month >= 8                # label on the left where the right has no room
        pad = np.timedelta64(2, "D") * (-1 if late else 1)
        ax.text(when + pad, 520, f"{ratio:g}-for-1\n{day.day} {day.strftime('%b')}",
                color=INK2, fontsize=7, va="top", ha="right" if late else "left",
                linespacing=1.05)
        ax.set_yscale("log")
        ax.set_ylim(2.0, 560)
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
    first.text(np.datetime64("2022-08-15"), 62, "adjusted", color=INK, fontsize=7,
               ha="center", va="top")
    first.text(np.datetime64("2022-08-15"), 5.9, "raw", color=INK, fontsize=7,
               ha="center", va="bottom")
    axes[0].set_ylabel("close, first = 100", labelpad=2)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=BLUE, lw=1.2, label="adjusted close"),
                        Line2D([], [], color=ORANGE, lw=1.2, label="raw quote (as traded)")],
               loc="upper right", bbox_to_anchor=(0.995, 1.13), ncol=2, frameon=False,
               fontsize=7, handlelength=1.6, labelcolor=INK)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "real_panel.pdf", bbox_inches="tight", pad_inches=0.02, metadata=META)
    fig.savefig(OUT / "real_panel.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    print("real panel", raw.index[0].date(), raw.index[-1].date())


EPISODE = Path(__file__).resolve().parent / "figures" / "episode_src"


def episode_split() -> None:
    """Figure 2 (a): Amazon around its 20-for-1 split, and what run 1 does with it.

    Top: raw quote and adjusted close, both indexed to 100 at the first row drawn, log scale.
    Bottom: the sign of Amazon's position from run 1 (episode_src/run1.py, which reads raw
    quotes) and from the task's independent reference, on task momentum_a and the real panel.
    Needs the fetched panel; the figure shows no vendor price level.
    """
    from fin_skills.tasks import panel, suite
    from fin_skills.tasks.judge import reference

    raw, factors, adjusted = panel("real")
    task = next(t for t in suite().tasks if t.id == "momentum_a")
    ns: dict = {}
    exec((EPISODE / "run1.py").read_text(), ns)
    run1 = np.sign(ns["solve"](raw, factors)["AMZN"].to_numpy())
    ref = np.sign(np.asarray(reference(task, "real"))[:, list(raw.columns).index("AMZN")])
    split = int(np.flatnonzero(np.diff(factors["AMZN"].to_numpy()))[0]) + 1
    lo, hi = split - 14, split + 31                    # rows drawn
    rows = np.arange(lo, hi)
    wrong = (run1 != ref) & (np.arange(len(raw)) > split)
    jump = raw["AMZN"].iloc[split] / raw["AMZN"].iloc[split - 1] - 1

    plt.rcParams.update(STYLE)
    fig, (a, b) = plt.subplots(2, 1, figsize=(1.80, 1.78), sharex=True,
                               gridspec_kw=dict(height_ratios=[2.5, 1.0], hspace=0.16))
    r = 100 * raw["AMZN"].iloc[lo:hi] / raw["AMZN"].iloc[lo]
    d = 100 * adjusted["AMZN"].iloc[lo:hi] / adjusted["AMZN"].iloc[lo]
    a.plot(rows, d.to_numpy(), color=BLUE, lw=1.2, solid_capstyle="round", zorder=3)
    a.plot(rows, r.to_numpy(), color=ORANGE, lw=1.2, solid_capstyle="round", zorder=2)
    a.axvline(split - 0.5, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    a.set_yscale("log")
    a.set_ylim(2.6, 260)
    from matplotlib.ticker import FixedLocator, NullLocator
    a.yaxis.set_major_locator(FixedLocator([5, 20, 100]))
    a.yaxis.set_minor_locator(NullLocator())
    a.set_yticklabels(["5", "20", "100"])
    a.text(split + 0.6, 205, "20-for-1 split", color=INK2, fontsize=7, va="top", ha="left")
    a.text(hi - 0.5, d.iloc[-1] * 0.72, "adjusted", color=INK, fontsize=7, ha="right",
           va="top")
    a.text(hi - 0.5, r.iloc[-1] * 1.35, "raw quote", color=INK, fontsize=7, ha="right",
           va="bottom")
    pct = f"{jump:.0%}".replace("-", "\u2212")         # a true minus sign
    a.annotate(f"{pct} in one day", xy=(split + 0.3, 22), xytext=(split + 3.2, 22),
               fontsize=7, color=INK, ha="left", va="center",
               arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.6, shrinkA=1, shrinkB=0,
                               mutation_scale=6))
    a.grid(axis="y", color=GRID, lw=0.5)
    a.set_axisbelow(True)
    a.tick_params(axis="x", length=0)
    a.set_ylabel("first day = 100", labelpad=1.5, fontsize=7)

    for x in rows[wrong[lo:hi]]:
        b.add_patch(Rectangle((x - 0.5, -0.62), 1.0, 2.24, facecolor=ORANGE, alpha=0.16,
                              edgecolor="none", zorder=0))
    for y, sign in ((1, run1), (0, ref)):
        for x in rows:
            color = BLUE if sign[x] > 0 else (SHORT if sign[x] < 0 else UNMOVED)
            b.add_patch(Rectangle((x - 0.42, y - 0.36), 0.84, 0.72, facecolor=color,
                                  edgecolor="none", zorder=1))
    b.set_ylim(-0.62, 1.62)
    b.set_yticks([1, 0], ["run 1", "reference"])
    b.tick_params(axis="y", length=0, pad=2)
    b.set_xlim(lo - 0.6, hi - 0.4)
    months = [i for i in rows if raw.index[i].day <= 7 and
              (i == lo or raw.index[i].month != raw.index[i - 1].month)]
    b.set_xticks(months, [raw.index[i].strftime("%b") for i in months])
    b.tick_params(axis="x", length=2, pad=1.5)
    for ax in (a, b):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    b.spines["left"].set_visible(False)
    short = int((run1[split + 1:hi] < 0).sum())
    print("episode: split row", split, raw.index[split].date(), "one-day", pct,
          "run 1 short rows", short, "wrong side", int(wrong.sum()))
    (OUT / "episode_numbers.tex").write_text(
        "% GENERATED by make_figures.py (episode_split) - do not edit by hand.\n"
        f"\\newcommand{{\\EpisodeDrop}}{{{-round(100 * jump)}}}% percent, one day, raw quotes\n"
        f"\\newcommand{{\\EpisodeShort}}{{{short}}}% sessions run 1 is short after the split\n"
        f"\\newcommand{{\\EpisodeWrong}}{{{int(wrong.sum())}}}% of them, the reference is long\n")
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "episode_split.pdf", bbox_inches="tight", pad_inches=0.01, metadata=META)
    fig.savefig(OUT / "episode_split.png", dpi=300, bbox_inches="tight", pad_inches=0.01)


def episode_lag() -> None:
    """Figure 2 (c): what the execution_lag check sees, written as TikZ data for episode.tex.

    Two inverse-volatility programs on the seed-11 panel: the library's lagged positions and
    the same positions read on the close they trade over. Prices from row K on are perturbed
    as the guard perturbs them (both of assert_lagged's perturbations); a cell is marked where
    a position moved. The verdicts are the guard's own, asserted against the cells.
    """
    from fin_skills.algorithms import positions
    from fin_skills.api import get
    from fin_skills.tasks import panel

    raw, factors, _ = panel(SEED)

    def lagged(prices):
        return positions("inverse_volatility", prices, factors, window=WINDOW)

    def same_close(prices):                # decides row t with the close of row t
        return lagged(prices).shift(-1).fillna(0.0)

    scaled = raw.copy()
    scaled.iloc[K:] = scaled.iloc[K:].to_numpy() * 2.0
    rows = {}
    for name, fn in (("A", lagged), ("B", same_close)):
        moved = np.zeros(len(raw), dtype=bool)
        for other in (scaled, perturb(raw, K)):
            moved |= (fn(raw) - fn(other)).abs().max(axis=1).to_numpy() > 1e-8
        passed = bool(get("execution_lag").run(fn=fn, df=raw.copy(), k=K, tol=1e-8).passed)
        assert passed == (not moved[:K + 1].any()), (name, passed)
        rows[name] = (moved, passed)
    assert rows["A"][1] and not rows["B"][1]
    first = {n: int(np.flatnonzero(m)[0]) for n, (m, _) in rows.items()}
    assert first == {"A": K + 1, "B": K}, first      # the figure's labels: k + 1 and k
    print("episode (c): k", K, "first moved row", first)

    def cells(moved):                      # 0 unchanged, 1 moved after k, 2 moved by row k
        return ",".join(str(0 if not moved[x] else (2 if x <= K else 1)) for x in SHOW)

    (OUT / "episode_lag.tex").write_text(
        "% GENERATED by make_figures.py (episode_lag) - do not edit by hand.\n"
        f"% Seed {SEED} panel, inverse_volatility (window {WINDOW}), rows {SHOW.start}.."
        f"{SHOW.stop - 1}.\n"
        f"\\def\\LagK{{{K - SHOW.start}}}% first perturbed row, counted from the first drawn\n"
        f"\\def\\LagRows{{{len(SHOW)}}}%\n"
        f"\\def\\LagCellsA{{{cells(rows['A'][0])}}}% lagged: 0 same, 1 moved after k, "
        "2 moved by k\n"
        f"\\def\\LagCellsB{{{cells(rows['B'][0])}}}% same close\n")


def main() -> None:
    episode_split()
    episode_lag()
    real_panel_figure()


if __name__ == "__main__":
    main()
