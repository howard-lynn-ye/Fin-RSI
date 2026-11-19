"""Appendix figure of the short paper: the real-price panel, in the style of the main figures.

    FIN_TASKS_REAL_PANEL=/path/to/real_panel_2022.csv python3 make_real_panel.py [--png]

Writes figures/real_panel_short.pdf. Needs the fetched panel (fin_skills.tasks.real; Yahoo's terms
forbid redistributing it, so it is not in the repository); the loader checks its SHA-256. The
figure shows no vendor price level: both series are indexed to 100 at the first close. The
one-day changes printed at each split are computed here from the panel, as is every other number.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))        # the repository root, for fin_skills
import make_pfix_figures as F  # noqa: E402  helpers, colours and the label checks

import matplotlib.dates as mdates  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

from fin_skills.tasks import real  # noqa: E402

def signed(v, digits):
    """A signed percentage with a true minus sign."""
    return f"{'+' if v >= 0 else '−'}{abs(v):.{digits}f}%"


NAMES = {"AMZN": "Amazon", "SHOP": "Shopify", "GOOGL": "Alphabet (class A)", "TSLA": "Tesla"}


def main():
    path = real.default_path()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == real.SHA256, "not the pinned panel"
    raw, _, adjusted = real.panel()
    F.setup()
    fig = F.plt.figure(figsize=(F.TEXTW, 1.55))
    W, left, gap, bottom, height = F.TEXTW, 0.44, 0.13, 0.27, 1.02
    w = (W - left - 0.04 - 3 * gap) / 4
    axes = []
    for k, (ticker, date, ratio, _) in enumerate(real.SPLITS):
        ax = F.axes_in(fig, left + k * (w + gap), bottom, w, height)
        axes.append(ax)
        F.journal(ax, left=k == 0)
        r = 100 * raw[ticker] / raw[ticker].iloc[0]
        a = 100 * adjusted[ticker] / adjusted[ticker].iloc[0]
        ax.plot(a.index, a, color=F.BLUE, lw=1.0, solid_capstyle="round", zorder=3)
        ax.plot(r.index, r, color=F.ORANGE, lw=1.0, solid_capstyle="round", zorder=2)
        day = np.datetime64(date)
        i = raw.index.get_loc(np.datetime64(date, "ns"))
        assert raw.index[i] == np.datetime64(date, "ns") and i > 0
        drop = 100 * (raw[ticker].iloc[i] / raw[ticker].iloc[i - 1] - 1)
        move = 100 * (adjusted[ticker].iloc[i] / adjusted[ticker].iloc[i - 1] - 1)
        assert abs(drop - 100 * (1 / ratio - 1)) < 4    # the split, not the market, moved it
        ax.axvline(day, color=F.MUTED, lw=0.5, ls=(0, (2.5, 2)), zorder=1)
        when = day.astype(object)
        late = when.month >= 8                # text on the left where the right has no room
        pad = np.timedelta64(3, "D") * (-1 if late else 1)
        ha = "right" if late else "left"
        ax.text(day + pad, 420, f"{ratio:g}-for-1, {when.day} {when.strftime('%b')}",
                color=F.INK2, fontsize=F.FS, ha=ha, va="center")
        ax.set_yscale("log")
        ax.set_ylim(2.2, 520)
        ax.spines["left"].set_bounds(3, 100)
        ax.yaxis.set_major_locator(FixedLocator([3, 10, 30, 100]))
        ax.yaxis.set_minor_locator(NullLocator())
        if k == 0:
            ax.set_yticklabels(["3", "10", "30", "100"])
            ax.set_ylabel("first close = 100", labelpad=1.5, color=F.INK2)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(4, 6, 8)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.set_xlim(raw.index[0], raw.index[-1])
        # under the split's name: what the raw quote and the adjusted close did that day
        parts = [(signed(drop, 0), F.ORANGE, "bold"), ("  ", F.INK, "normal"),
                 (signed(move, 1), F.BLUE, "normal")]
        x0, y0 = F.to_inches(fig, ax, mdates.date2num(day + pad), 210)
        if late:
            x0 -= sum(F._width(fig, t, fontweight=wt) for t, _, wt in parts)
        for t, c, wt in parts:
            x0 = F.rich(fig, x0, y0, [t], colour=c, weight=wt)
        F.letter(fig, left + k * (w + gap) - (0.40 if k == 0 else 0.02), 1.43,
                 "abcd"[k], [NAMES[ticker]])
    first = axes[0]                            # direct labels once, where there is room
    first.text(np.datetime64("2022-08-20"), 62, "adjusted", color=F.BLUE, fontsize=F.FS,
               ha="center", va="top")
    first.text(np.datetime64("2022-08-20"), 6.4, "as traded", color=F.ORANGE, fontsize=F.FS,
               ha="center", va="bottom")
    print("real panel", raw.index[0].date(), raw.index[-1].date(), len(raw), "sessions")
    F.check_and_save(fig, "real_panel_short")
    for p in F.PROBLEMS:
        print("PROBLEM", *p)
    if F.PROBLEMS:
        sys.exit(1)


if __name__ == "__main__":
    main()
