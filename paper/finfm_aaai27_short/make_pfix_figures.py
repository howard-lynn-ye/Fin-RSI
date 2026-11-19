"""Main-text data figures of the short paper, drawn from the pinned data.

    python3 make_pfix_figures.py [--png]   # writes figures/pfix_models.pdf (Figure 2),
                                           # figures/pfix_arms.pdf (Figure 3) and
                                           # figures/pfix_analysis.pdf (Figure 4)

Designed by an independent review on 2026-11-19 (journal style, every model shown). Reads the
paper's pinned analysis (its hash is the one make_pfix_results.py pins) and the figure style
../latex_acl2027/figstyle.py (colours, TeX Gyre Heros). Every number drawn or printed is a field
of pfix_analysis.json or a sum or ratio of its fields; the captions take their numbers from
pfix_numbers.tex. Each figure is drawn at its print size (3.3 in or 7.0 in wide) and checked
before it is saved: no text below 7 pt, no text outside the figure, no two labels overlapping.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True          # never leave a __pycache__ in the paper's tree
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "latex_acl2027"))
import figstyle as fs  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
sys.path.insert(0, str(HERE))
from make_pfix_results import ANALYSIS, ANALYSIS_SHA256  # noqa: E402  one pin for both scripts
PNG = "--png" in sys.argv
META = {"CreationDate": None, "ModDate": None, "Creator": None, "Producer": None}

COL, TEXTW = 3.3, 7.0                    # AAAI column and text width, inches
ARMS = ["F0", "F1", "F2", "F3", "F4"]
BLUE, MAG, ORANGE = fs.BLUE, fs.COLOR["finmagenta"], fs.ORANGE
INK, INK2, MUTED = fs.INK, fs.INK2, fs.MUTED
LINE, FAINT = fs.COLOR["figline"], fs.COLOR["figfaint"]
WHITE = "#ffffff"
EMPTY = fs.mix(fs.COLOR["figfaint"], WHITE, 0.75)    # an empty cell of a design matrix
GROUP_COLOUR = {"ge14b": BLUE, "lt14b": MAG}
FS = 7.0                                 # every label; panel letters 8.5 bold, titles 7.5


def tint(c, t):
    """t of colour c on white."""
    return fs.mix(c, WHITE, t)


# --------------------------------------------------------------------------------------------
# data

def load():
    raw = ANALYSIS.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ANALYSIS_SHA256, "not the pinned analysis"
    d = json.loads(raw)
    per, sizes = d["per_model"], d["sizes_b"]
    assert len(per) == 20 and all(per[m][a]["n"] == 18 for m in per for a in ARMS)
    for g in ("ge14b", "lt14b"):
        for a in ARMS:   # the group pools are the sums of their models
            p = d["by_size"][g]["pooled"][a]
            ms = d["by_size"][g]["models"]
            assert p["valid"] == sum(per[m][a]["valid"] for m in ms)
            assert p["first_valid"] == sum(per[m][a]["first_valid"] for m in ms)
            assert p["accepted_wrong"] == sum(per[m][a]["accepted_wrong"] for m in ms)
    assert set(d["by_size"]["ge14b"]["models"]) == {m for m in per if sizes[m] >= 14}
    return d


def pct(v, n):
    return 100.0 * v / n


# --------------------------------------------------------------------------------------------
# drawing helpers

def setup():
    plt.rcParams.update(fs.RC)
    plt.rcParams.update({"font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
                         "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
                         "hatch.linewidth": 0.6, "lines.solid_capstyle": "butt",
                         "svg.fonttype": "none"})


def axes_in(fig, x, y, w, h):
    """Axes placed in inches from the lower-left corner of the figure."""
    W, H = fig.get_size_inches()
    return fig.add_axes([x / W, y / H, w / W, h / H])


def journal(ax, left=True, bottom=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.5)
        ax.spines[side].set_color(MUTED)
    ax.tick_params(axis="both", direction="out", length=2.2, width=0.5, pad=1.5, labelsize=FS,
                   colors=INK2)
    if not left:
        ax.tick_params(axis="y", length=0)
    if not bottom:
        ax.tick_params(axis="x", length=0)
    ax.patch.set_alpha(0)


PROBLEMS: list = []


def check_and_save(fig, name):
    """Report text below 7 pt, text outside the figure and overlapping labels; save anyway so
    the drawing can be inspected, and let main() fail at the end if anything was reported."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    boxes = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if t.get_fontsize() < 7.0 - 1e-9:
            PROBLEMS.append((name, "below 7 pt", t.get_text(), t.get_fontsize()))
        bb = t.get_window_extent(r)
        if bb.width == 0:
            continue
        if not (bb.x0 >= fb.x0 - 0.5 and bb.x1 <= fb.x1 + 0.5 and bb.y0 >= fb.y0 - 0.5 and
                bb.y1 <= fb.y1 + 0.5):
            PROBLEMS.append((name, "outside the figure", t.get_text()))
        boxes.append((t.get_text(), bb))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            ox = min(a.x1, b.x1) - max(a.x0, b.x0)
            oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ox > 0.8 and oy > 0.8:          # more than ~0.25 pt at 220 dpi
                PROBLEMS.append((name, "overlapping labels", boxes[i][0], boxes[j][0]))
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", metadata=META)
    if PNG:
        fig.savefig(OUT / f"{name}.png", dpi=220)
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f"{name}: {w:.2f} x {h:.2f} in, {len(boxes)} labels checked")


# --------------------------------------------------------------------------------------------
# inch-placed text and keys

MONO = "TeX Gyre Cursor"


def _width(fig, s, **kw):
    t = fig.text(0, 0, s, fontsize=kw.pop("fontsize", FS), **kw)
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


def rich(fig, x, y, parts, va="center", colour=INK, size=FS, weight="normal"):
    """Place text pieces one after another from (x, y) inches; a piece in backquotes is code."""
    for s in parts:
        code = s.startswith("`")
        s = s.strip("`")
        kw = dict(fontsize=size, color=colour, fontweight=weight)
        if code:
            kw["family"] = MONO
        fig.text(x, y, s, transform=fig.dpi_scale_trans, ha="left", va=va, **kw)
        x += _width(fig, s, **kw)
    return x


def mark(fig, x, y, kind, colour, ms=4.0):
    """A key mark centred at (x, y) inches."""
    tr = fig.dpi_scale_trans
    if kind == "open":
        fig.add_artist(Line2D([x], [y], transform=tr, marker="o", ms=ms * 0.85, mfc=WHITE,
                              mec=colour, mew=0.8, ls=""))
    elif kind == "dot":
        fig.add_artist(Line2D([x], [y], transform=tr, marker="o", ms=ms, mfc=colour, mec=WHITE,
                              mew=0.5, ls=""))
    elif kind == "diamond":
        fig.add_artist(Line2D([x], [y], transform=tr, marker="D", ms=ms * 0.8, mfc=colour,
                              mec=WHITE, mew=0.5, ls=""))
    elif kind == "dash":
        fig.add_artist(Line2D([x - 0.045, x + 0.045], [y, y], transform=tr, color=colour, lw=1.0))
    elif kind == "tick":
        fig.add_artist(Line2D([x, x], [y - 0.035, y + 0.035], transform=tr, color=colour, lw=0.6))
    elif kind == "small":
        fig.add_artist(Line2D([x], [y], transform=tr, marker="o", ms=2.6, mfc=colour, mec="none",
                              ls=""))
    else:                                   # ("box", face, hatch, edge)
        _, face, hatch, edge = kind
        fig.add_artist(Rectangle((x - 0.045, y - 0.035), 0.09, 0.07, transform=tr,
                                 facecolor=face, hatch=hatch, edgecolor=edge, linewidth=0))


def key(fig, x, y, items, gap=0.11, wrap=None, line=0.12):
    """A row of (mark, colour, [text pieces]) entries; wraps at x + wrap inches."""
    x0 = x
    for kind, colour, parts in items:
        w = 0.11 + sum(_width(fig, p.strip("`"), family=MONO if p.startswith("`") else None)
                       for p in parts)
        if wrap is not None and x > x0 and x + w > x0 + wrap:
            x, y = x0, y - line
        mark(fig, x + 0.045, y, kind, colour)
        x = rich(fig, x + 0.11, y, parts) + gap
    return x


def letter(fig, x, y, char, parts=()):
    """Bold panel letter at (x, y) inches (baseline), then its title."""
    fig.text(x, y, char, transform=fig.dpi_scale_trans, fontsize=8.5, fontweight="bold",
             color=INK, ha="left", va="baseline")
    if parts:
        rich(fig, x + 0.13, y, parts, va="baseline", size=7.5)


def to_inches(fig, ax, xd, yd):
    """Data coordinates of ax to inches."""
    px = ax.transData.transform((xd, yd))
    return px[0] / fig.dpi, px[1] / fig.dpi


# --------------------------------------------------------------------------------------------
# Figure 1 (page 1): every model, from no library (F0) to route() (F4), against the call we name (F2)

def fig_headline(d, name="pfix_models"):
    per, sizes = d["per_model"], d["sizes_b"]
    big, small = d["by_size"]["ge14b"], d["by_size"]["lt14b"]
    models = sorted(per, key=lambda m: (sizes[m], m))
    x = {m: i for i, m in enumerate(models)}
    cut = len(small["models"]) - 0.5            # boundary between the size groups
    assert all(sizes[m] < 14 for m in models[:len(small["models"])])

    fig = plt.figure(figsize=(COL, 1.76))
    ax = axes_in(fig, 0.335, 0.33, 2.925, 1.08)
    journal(ax, bottom=False)
    xl = (-0.6, len(models) - 0.4)
    ax.set_xlim(*xl)
    ax.set_ylim(-4, 104)
    ax.set_yticks([0, 25, 50, 75, 100], ["0", "", "50", "", "100"])
    ax.set_ylabel("valid (%)", labelpad=1.5, color=INK2)
    ax.spines["left"].set_bounds(0, 100)
    ax.set_xticks([])
    ax.axvspan(cut, xl[1], color=tint(BLUE, 0.07), linewidth=0, zorder=0)
    for m in models:
        c = BLUE if sizes[m] >= 14 else MAG
        v0, v2, v4 = (pct(per[m][a]["valid"], per[m][a]["n"]) for a in ("F0", "F2", "F4"))
        xi = x[m]
        ax.plot([xi - 0.36, xi + 0.36], [v2, v2], color=INK, linewidth=1.0, zorder=2)
        if abs(v4 - v0) > 0.1:
            ax.plot([xi, xi], [v0, v4], color=tint(c, 0.5), linewidth=1.2, zorder=1)
        ax.scatter([xi], [v0], s=11, facecolor=WHITE, edgecolor=c, linewidth=0.8, zorder=3)
        ax.scatter([xi], [v4], s=17, facecolor=c, edgecolor=WHITE, linewidth=0.5, zorder=4)
    # x axis: model size, one column per model; a bracket joins models of equal size
    runs, i = [], 0
    while i < len(models):
        j = i
        while j + 1 < len(models) and sizes[models[j + 1]] == sizes[models[i]]:
            j += 1
        runs.append((i, j, sizes[models[i]]))
        i = j + 1
    for i, j, s in runs:
        ax.text((i + j) / 2, -10, f"{s:g}", ha="center", va="top", fontsize=FS, color=INK2,
                clip_on=False)
        ax.plot([i - 0.3, j + 0.3], [-7, -7], color=MUTED, linewidth=0.5, clip_on=False,
                solid_capstyle="butt")
    ax.text((len(models) - 1) / 2, -22.5, "parameters (billions), one column per model",
            ha="center", va="top", fontsize=FS, color=INK2, clip_on=False)
    # direct labels: the smaller model that gained, and the one valid in every episode
    for m, dx, ha in (("qwen3-4b", 0.55, "left"), ("qwen3-30b-a3b", -0.6, "right")):
        v4 = pct(per[m]["F4"]["valid"], per[m]["F4"]["n"])
        ax.text(x[m] + dx, v4, m, ha=ha, va="center", fontsize=FS, color=INK2)
    # group headers above the plot, with the pooled rates (F0 -> F4)
    xc_in, _ = to_inches(fig, ax, cut, 0)
    fig.add_artist(Line2D([xc_in, xc_in], [0.33 + 0.02, 1.73], transform=fig.dpi_scale_trans,
                          color=LINE, linewidth=0.5))
    for group, colour, label, x0 in ((small, MAG, f"<14B: {len(small['models'])} models",
                                      0.36),
                                     (big, BLUE, f"≥14B: {len(big['models'])} models",
                                      xc_in + 0.06)):
        p = group["pooled"]
        mark(fig, x0 + 0.03, 1.665, "dot", colour)
        rich(fig, x0 + 0.09, 1.665, [label], weight="bold")
        rich(fig, x0 + 0.09, 1.54, [f"pooled {pct(p['F0']['valid'], p['F0']['n']):.1f}% → "
                                    f"{pct(p['F4']['valid'], p['F4']['n']):.1f}%"], colour=INK2)
    # key, in the empty upper middle of the smaller models' columns
    kx, _ = to_inches(fig, ax, 4.45, 0)
    for k, (kind, colour, parts) in enumerate((("open", INK2, ["F0 no library"]),
                                               ("dot", INK2, ["F4 ", "`route()`", " first"]),
                                               ("dash", INK, ["F2 call named by us"]))):
        _, ky = to_inches(fig, ax, 0, 96 - 13 * k)
        key(fig, kx, ky, [(kind, colour, parts)])
    check_and_save(fig, name)


# --------------------------------------------------------------------------------------------
# Figure 3 (page 3): the five arms, what each adds, and what each scored, per model and group

def fig_arms(d, name="pfix_arms"):
    per = d["per_model"]
    big, small = d["by_size"]["ge14b"], d["by_size"]["lt14b"]
    pooled = d["pooled"]
    fig = plt.figure(figsize=(COL, 1.06))
    tr = fig.dpi_scale_trans
    row_h, top = 0.142, 0.825                    # row pitch and the y of row F0, inches
    ys = {a: top - i * row_h for i, a in enumerate(ARMS)}
    head_y = top + 0.105
    # design matrix: the pieces each arm adds to the messages
    cols = [("first", 0.385, 0.40), ("system prompt", 0.83, 0.42),
            ("listing", 1.305, 0.42), ("task text", 1.79, 0.50)]
    on = {("F1", 1): "interface", ("F2", 1): "interface", ("F3", 1): "interface",
          ("F4", 1): "interface", ("F3", 2): "rule lines", ("F2", 3): "id from us",
          ("F4", 3): "id it named", ("F4", 0): "`route()`"}
    for a in ARMS:
        fig.text(0.03, ys[a], a, transform=tr, fontsize=FS, color=INK, va="center", ha="left",
                 fontweight="bold")
    for k, (head, cx, w) in enumerate(cols):
        fig.text(cx, head_y, head, transform=tr, fontsize=FS, color=MUTED, ha="center",
                 va="bottom")
        for a in ARMS:
            text = on.get((a, k))
            fig.add_artist(FancyBboxPatch((cx - w / 2, ys[a] - 0.05), w, 0.1,
                                          boxstyle="round,pad=0,rounding_size=0.014",
                                          transform=tr, linewidth=0.5,
                                          facecolor=tint(BLUE, 0.13) if text else EMPTY,
                                          edgecolor=tint(BLUE, 0.65) if text else EMPTY))
            if text:
                code = text.startswith("`")
                fig.text(cx, ys[a], text.strip("`"), transform=tr, fontsize=FS, color=INK,
                         ha="center", va="center_baseline", family=MONO if code else None)
    fig.add_artist(FancyArrowPatch((cols[0][1], ys["F4"] - 0.055), (cols[3][1], ys["F4"] - 0.055),
                                   transform=tr, connectionstyle="arc3,rad=0.085",
                                   arrowstyle="-|>,head_length=2.4,head_width=1.4",
                                   linewidth=0.6, color=MUTED, shrinkA=0, shrinkB=0))
    r = d["routing"]
    fig.text((cols[0][1] + cols[3][1]) / 2, 0.02,
             f"names the task's rule in {r['right']} of {sum(r[k] for k in ('right', 'none', 'wrong'))}",
             transform=tr, fontsize=FS, color=MUTED, ha="center", va="bottom")
    # outcome strip: each model (tick), each size group pooled (marker), all models (number)
    x0, w = 2.11, 0.84
    bottom = ys["F4"] - row_h / 2
    ax = axes_in(fig, x0, bottom, w, top + row_h / 2 - bottom)
    journal(ax, left=False)
    ax.set_xlim(-4, 104)
    ax.set_ylim(bottom, top + row_h / 2)
    ax.spines["bottom"].set_bounds(0, 100)
    ax.set_xticks([0, 50, 100], ["0", "50", "100%"])
    ax.set_yticks([])
    for a in ARMS:
        ax.axhline(ys[a], color=FAINT, linewidth=0.6, zorder=0)
        for g, group in (("lt14b", small), ("ge14b", big)):
            for m in group["models"]:
                v = pct(per[m][a]["valid"], per[m][a]["n"])
                ax.plot([v, v], [ys[a] - 0.042, ys[a] + 0.042], color=tint(GROUP_COLOUR[g], 0.5),
                        linewidth=0.6, zorder=1)
        vb = pct(big["pooled"][a]["valid"], big["pooled"][a]["n"])
        vs = pct(small["pooled"][a]["valid"], small["pooled"][a]["n"])
        ax.plot([vs, vb], [ys[a], ys[a]], color=MUTED, linewidth=0.6, zorder=2)
        ax.scatter([vs], [ys[a]], s=14, marker="D", facecolor=MAG, edgecolor=WHITE,
                   linewidth=0.5, zorder=3)
        ax.scatter([vb], [ys[a]], s=20, marker="o", facecolor=BLUE, edgecolor=WHITE,
                   linewidth=0.5, zorder=3)
        fig.text(x0 + w + 0.07, ys[a], f"{pct(pooled[a]['valid'], pooled[a]['n']):.1f}",
                 transform=tr, fontsize=FS, color=INK, ha="left", va="center")
    fig.text(x0 + w + 0.07, head_y, "all", transform=tr, fontsize=FS, color=MUTED, ha="left",
             va="bottom")
    fig.canvas.draw()                       # "valid" after the last tick label
    last = ax.get_xticklabels()[-1].get_window_extent(fig.canvas.get_renderer())
    fig.text(last.x1 / fig.dpi + 0.025, last.y0 / fig.dpi, "valid", transform=tr, fontsize=FS,
             color=INK2, ha="left", va="bottom")
    key(fig, x0 - 0.03, head_y + 0.045, [("dot", BLUE, ["≥14B"]), ("diamond", MAG, ["<14B"])],
        gap=0.1)
    check_and_save(fig, name)


# --------------------------------------------------------------------------------------------
# Figure 4 (page 4): how episodes ended, where route() leads, what the checks flag

def band(ax, x0, a0, a1, x1, b0, b1, colour, alpha, z=1):
    """A flow from [a0, a1] at x0 to [b0, b1] at x1 (data coordinates, y grows downward)."""
    xm = (x0 + x1) / 2
    verts = [(x0, a0), (xm, a0), (xm, b0), (x1, b0), (x1, b1), (xm, b1), (xm, a1), (x0, a1),
             (x0, a0)]
    codes = [MPath.MOVETO] + [MPath.CURVE4] * 3 + [MPath.LINETO] + [MPath.CURVE4] * 3 + \
        [MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=colour, alpha=alpha, linewidth=0,
                           zorder=z))


def stack(ax, x, yc, lines, colour=INK, pitch=0.108, z=5):
    """Lines of text centred on (x, yc) in data coordinates, `pitch` inches apart; a line in
    backquotes is code."""
    fig = ax.figure
    _, y0 = ax.transData.transform((x, yc))
    for k, s in enumerate(lines):
        dy = (k - (len(lines) - 1) / 2) * pitch * fig.dpi
        _, yy = ax.transData.inverted().transform((0, y0 - dy))
        code = s.startswith("`")
        ax.text(x, yy, s.strip("`"), ha="center", va="center_baseline", fontsize=FS,
                color=colour, zorder=z, family=MONO if code else None)


def fig_analysis(d, name="pfix_analysis"):
    big, small = d["by_size"]["ge14b"], d["by_size"]["lt14b"]
    fig = plt.figure(figsize=(TEXTW, 1.86))
    tr = fig.dpi_scale_trans
    title_y, plot_b, plot_h = 1.75, 0.30, 1.09

    # a: how each episode ended, per arm and size group ----------------------------------
    ax = axes_in(fig, 0.36, plot_b, 1.2, 1.10)
    journal(ax, left=False)
    rows, y = [], 0
    for g in ("ge14b", "lt14b"):
        for a in ARMS:
            rows.append((g, a, y))
            y += 1
        y += 0.8
    VALID_LATER = 0.42
    REST = tint(LINE, 0.6)
    for g, a, y in rows:
        p = d["by_size"][g]["pooled"][a]
        n = p["n"]
        parts = [(p["first_valid"], GROUP_COLOUR[g], None, "none"),
                 (p["valid"] - p["first_valid"], tint(GROUP_COLOUR[g], VALID_LATER), None, "none"),
                 (p["accepted_wrong"], WHITE, "//////", ORANGE),
                 (n - p["valid"] - p["accepted_wrong"], REST, None, "none")]
        assert all(v >= 0 for v, *_ in parts)
        left = 0
        for val, face, hatch, edge in parts:
            if val:
                ax.barh(y, pct(val, n), left=pct(left, n), height=0.72, facecolor=face,
                        hatch=hatch, edgecolor=edge, linewidth=0, zorder=2)
                if left:             # a hairline of the surface colour between segments
                    ax.plot([pct(left, n)] * 2, [y - 0.36, y + 0.36], color=WHITE,
                            linewidth=0.8, zorder=3, solid_capstyle="butt")
            left += val
        ax.text(-2.5, y, a, ha="right", va="center", fontsize=FS, color=INK)
    ax.set_ylim(rows[-1][2] + 0.6, -0.6)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xticks([0, 50, 100], ["0", "50", "100%"])
    ax.spines["bottom"].set_bounds(0, 100)
    for g, label, i in (("ge14b", "≥14B", 2), ("lt14b", "<14B", 7)):
        _, yy = to_inches(fig, ax, 0, rows[i][2])
        fig.text(0.07, yy, label, transform=tr, rotation=90, ha="center", va="center",
                 fontsize=FS, color=INK, fontweight="bold")
        fig.add_artist(Line2D([0.135, 0.135], [yy - 0.25, yy + 0.25], transform=tr,
                              color=GROUP_COLOUR[g], linewidth=1.6, solid_capstyle="butt"))
    ax.set_xlabel("episodes of the size group", labelpad=1.5, color=INK2)
    letter(fig, 0.02, title_y, "a", ["How the episodes ended"])
    key(fig, 0.06, 1.585, [
        (("box", BLUE, None, "none"), None, ["valid first try"]),
        (("box", tint(BLUE, VALID_LATER), None, "none"), None, ["valid later"]),
        (("box", WHITE, "//////", ORANGE), None, ["accepted but wrong"]),
        (("box", REST, None, "none"), None, ["rejected"])], gap=0.12, wrap=1.62, line=0.12)

    # b: F4 -- what route() named, what the last program did, how the episode ended -------
    r, tb, ts = d["routing"], big["told"]["F4"], small["told"]["F4"]
    rb, rs = big["routing"], small["routing"]
    assert rb.get("none", 0) == 0 and rb.get("wrong", 0) == 0     # only smaller models missed
    assert r["right"] == tb["episodes"] + ts["episodes"] == rb["right"] + rs["right"]
    ax = axes_in(fig, 1.84, 0.12, 2.48, 1.42)
    ax.set_xlim(0, 1)
    ax.axis("off")
    gap = 15
    cx = {1: (0.0, 0.17), 2: (0.425, 0.665), 3: (0.87, 1.0)}
    right, none, wrong = r["right"], r.get("none", 0), r.get("wrong", 0)
    nb, ns = tb["episodes"], ts["episodes"]
    # column 1, coloured by size group: the task's rule (>=14B, <14B), none, another rule
    c1 = {"rb": (0, nb), "rs": (nb, nb + ns), "none": (right + gap, right + gap + none),
          "wrong": (right + none + 2 * gap, right + none + 2 * gap + wrong)}
    calls = tb["last_calls_positions"] + ts["last_calls_positions"]
    own = tb["last_own_rule"] + ts["last_own_rule"]
    c2 = {"calls": (0, calls), "own": (calls + gap, calls + gap + own)}
    lane = (c2["own"][1] + gap, c2["own"][1] + gap + none + wrong)   # no rule named: no node
    vt, nt = d["pooled"]["F4"]["valid"], d["pooled"]["F4"]["n"]
    c3 = {"valid": (0, vt), "not": (vt + gap, nt + gap)}
    ymax = max(c1["wrong"][1], lane[1], c3["not"][1])
    ax.set_ylim(ymax + 2, -30)
    # flows 1 -> 2: each size group's right routes, split by what the last program did
    fill = {k: v[0] for k, v in c2.items()}
    for src, t, colour in (("rb", tb, BLUE), ("rs", ts, MAG)):
        y1 = c1[src][0]
        for node, n in (("calls", t["last_calls_positions"]), ("own", t["last_own_rule"])):
            band(ax, cx[1][1], y1, y1 + n, cx[2][0], fill[node], fill[node] + n, colour, 0.35)
            y1 += n
            fill[node] += n
    # no rule named, or another rule: a lane under column 2 (the episode continues as F1)
    lane_fill, lane_parts = lane[0], []
    for k, (a0, a1) in (("none", c1["none"]), ("wrong", c1["wrong"])):
        n = a1 - a0
        band(ax, cx[1][1], a0, a1, cx[2][0], lane_fill, lane_fill + n, MAG, 0.35)
        ax.add_patch(Rectangle((cx[2][0], lane_fill), cx[2][1] - cx[2][0], n, facecolor=MAG,
                               alpha=0.35, linewidth=0, zorder=1))
        lane_parts.append((lane_fill, n, r.get(k + "_valid", 0)))
        lane_fill += n
    # flows 2 -> 3: valid episodes first (dark), then the rest (light)
    out = {"calls": c2["calls"][0], "own": c2["own"][0]}
    into = {k: v[0] for k, v in c3.items()}
    flows = [("calls", BLUE, tb["last_calls_positions_valid"], "valid"),
             ("calls", MAG, ts["last_calls_positions_valid"], "valid"),
             ("own", BLUE, tb["last_own_rule_valid"], "valid"),
             ("own", MAG, ts.get("last_own_rule_valid", 0), "valid"),
             ("calls", BLUE, tb["last_calls_positions"] - tb["last_calls_positions_valid"], "not"),
             ("calls", MAG, ts["last_calls_positions"] - ts["last_calls_positions_valid"], "not"),
             ("own", BLUE, tb["last_own_rule"] - tb["last_own_rule_valid"], "not"),
             ("own", MAG, ts["last_own_rule"] - ts.get("last_own_rule_valid", 0), "not")]
    for node, colour, n, dest in flows[:4]:
        if n:
            band(ax, cx[2][1], out[node], out[node] + n, cx[3][0], into[dest], into[dest] + n,
                 colour, 0.8, z=2)
            out[node] += n
            into[dest] += n
    for (a0, n, nv) in lane_parts:          # the lane's valid episodes
        if nv:
            band(ax, cx[2][1], a0, a0 + nv, cx[3][0], into["valid"], into["valid"] + nv, MAG,
                 0.8, z=2)
            into["valid"] += nv
    for node, colour, n, dest in flows[4:]:
        if n:
            band(ax, cx[2][1], out[node], out[node] + n, cx[3][0], into[dest], into[dest] + n,
                 colour, 0.2)
            out[node] += n
            into[dest] += n
    for (a0, n, nv) in lane_parts:
        band(ax, cx[2][1], a0 + nv, a0 + n, cx[3][0], into["not"], into["not"] + n - nv, MAG,
             0.2)
        into["not"] += n - nv
    assert into["valid"] == c3["valid"][1] and into["not"] == c3["not"][1]
    assert vt == tb["valid"] + ts["valid"] + r.get("none_valid", 0) + r.get("wrong_valid", 0)

    def node(col, a, b, face, lines=(), ink=INK):
        x0, x1 = cx[col]
        ax.add_patch(Rectangle((x0, a), x1 - x0, b - a, facecolor=face, edgecolor="none",
                               zorder=4))
        if lines:
            stack(ax, (x0 + x1) / 2, (a + b) / 2, lines, colour=ink)
    grey = tint(LINE, 0.75)
    node(1, *c1["rb"], BLUE, ["≥14B", f"{nb}"], WHITE)
    node(1, *c1["rs"], MAG, ["<14B", f"{ns}"], WHITE)
    node(1, *c1["none"], MAG, ["none", f"{none}"], WHITE)
    node(1, *c1["wrong"], MAG)
    node(2, *c2["calls"], grey, ["calls", "`positions()`", f"{calls}"])
    node(2, *c2["own"], grey, ["own rule", f"{own}"])
    node(3, *c3["valid"], INK, ["valid", f"{vt}"], WHITE)
    node(3, *c3["not"], grey, ["not", "valid", f"{nt - vt}"])
    stack(ax, sum(cx[2]) / 2, lane[0] + none / 2, ["no right rule"], colour=INK)
    ax.text(cx[1][1] + 0.015, sum(c1["wrong"]) / 2, f"another rule {wrong}", ha="left",
            va="center", fontsize=FS, color=INK, zorder=5)
    for col, parts in ((1, ["`route()`", " named"]), (2, ["last program"]), (3, ["episode"])):
        xx, yy = to_inches(fig, ax, sum(cx[col]) / 2, -9)
        w = sum(_width(fig, p.strip("`"), family=MONO if p.startswith("`") else None)
                for p in parts)
        rich(fig, xx - w / 2, yy, parts, va="bottom", colour=MUTED)
    _, ya = to_inches(fig, ax, 0, c1["rb"][0])
    _, yb = to_inches(fig, ax, 0, c1["rs"][1])
    fig.text(1.775, (ya + yb) / 2, "the task's rule", transform=tr, rotation=90, ha="center",
             va="center", fontsize=FS, color=INK)
    fig.add_artist(Line2D([1.825, 1.825], [yb + 0.01, ya - 0.01], transform=tr, color=MUTED,
                          linewidth=0.5))
    letter(fig, 1.73, title_y, "b", ["F4: what ", "`route()`", " named, what the model wrote"])

    # c: share of programs that ran failing each check, per arm ---------------------------
    checks = [("execution_lag", "execution lag"), ("warmup_rows_zero", "warm-up rows zero"),
              ("takes_positions", "takes positions"), ("position_scale", "position scale"),
              ("split_representation", "split representation"), ("finite_output", "finite output")]
    fc = d["failed_checks"]
    ax = axes_in(fig, 5.22, plot_b, 0.80, plot_h + 0.04)
    journal(ax, left=False)
    for yy, (k, label) in enumerate(checks):
        vals = {a: pct(fc[a][k], fc[a]["programs_ran"]) for a in ARMS}
        ax.plot([min(vals.values()), max(vals.values())], [yy, yy], color=tint(ORANGE, 0.3),
                linewidth=1.3, zorder=1, solid_capstyle="round")
        for a in ("F1", "F2", "F3"):
            ax.scatter([vals[a]], [yy], s=6, color=INK2, linewidth=0, zorder=2)
        ax.scatter([vals["F0"]], [yy], s=15, facecolor=WHITE, edgecolor=ORANGE, linewidth=0.9,
                   zorder=3)
        ax.scatter([vals["F4"]], [yy], s=17, facecolor=ORANGE, edgecolor=WHITE, linewidth=0.5,
                   zorder=4)
        ax.text(-3, yy, label, ha="right", va="center", fontsize=FS, color=INK)
    ax.set_ylim(len(checks) - 0.45, -0.55)
    ax.set_xlim(0, 60)
    ax.set_yticks([])
    ax.set_xticks([0, 20, 40, 60], ["0", "20", "40", "60%"])
    ax.set_xlabel("of the programs that ran", labelpad=1.5, color=INK2)
    letter(fig, 4.40, title_y, "c", ["What the checks flagged"])
    key(fig, 4.42, 1.585, [("open", ORANGE, ["F0"]), ("small", INK2, ["F1–F3"]),
                           ("dot", ORANGE, ["F4"])], gap=0.1)

    # d: a check flagged a program and an attempt was left: did the episode end valid? -----
    rp = d["repair"]
    ax = axes_in(fig, 6.24, plot_b, 0.40, plot_h - 0.06)
    journal(ax, left=False)
    for yy, a in enumerate(ARMS):
        t, v = rp[a]["with_another_attempt"], rp[a]["then_valid"]
        ax.barh(yy, t, height=0.6, color=tint(ORANGE, 0.28), linewidth=0)
        ax.barh(yy, v, height=0.6, color=ORANGE, linewidth=0)
        ax.text(-5, yy, a, ha="right", va="center", fontsize=FS, color=INK)
        ax.text(t + 4, yy, f"{v}/{t}", ha="left", va="center", fontsize=FS, color=INK)
    ax.set_ylim(len(ARMS) - 0.45, -0.55)
    ax.set_xlim(0, 140)
    ax.set_yticks([])
    ax.set_xticks([0, 100], ["0", "100"])
    ax.spines["bottom"].set_bounds(0, 140)
    ax.set_xlabel("episodes", labelpad=1.5, color=INK2)
    letter(fig, 6.06, title_y, "d", ["Repairs"])
    key(fig, 6.08, 1.585, [(("box", tint(ORANGE, 0.28), None, "none"), None, ["flagged"])])
    key(fig, 6.08, 1.465, [(("box", ORANGE, None, "none"), None, ["then valid"])])
    check_and_save(fig, name)


def captions(d):
    """The suggested captions, with every number filled in from the analysis."""
    per, big, small, pooled = d["per_model"], d["by_size"]["ge14b"], d["by_size"]["lt14b"], d["pooled"]
    P = lambda p: f"{pct(p['valid'], p['n']):.1f}"
    bp, sp = big["pooled"], small["pooled"]
    f4f2 = next(c for c in d["exploratory_contrasts"] if c["contrast"] == "F4-F2")
    f4f0_big = next(c for c in big["exploratory_contrasts"] if c["contrast"] == "F4-F0")
    gain = {m: per[m]["F4"]["valid"] - per[m]["F0"]["valid"] for m in small["models"]}
    top_small = max(gain, key=gain.get)
    assert sorted(gain.values())[-2] <= 1
    max_dev = max(abs(v) for v in f4f2["per_model"].values())
    assert f4f0_big["better"] == len(big["models"]) and f4f0_big["worse"] == 0
    assert d["routing"].get("wrong_valid", 0) == 0
    assert pooled["F1"]["valid"] == pooled["F3"]["valid"]
    r, tb, ts = d["routing"], big["told"]["F4"], small["told"]["F4"]
    fc, rp = d["failed_checks"], d["repair"]
    share = lambda a, k: f"{pct(fc[a][k], fc[a]['programs_ran']):.0f}%"
    flagged = sum(rp[a]["with_another_attempt"] for a in ARMS)
    fixed = sum(rp[a]["then_valid"] for a in ARMS)
    n_routes = r["right"] + r.get("none", 0) + r.get("wrong", 0)
    text = f"""P1 (replaces pfix_headline, page 1)
route() lets the larger models find the library call. Each of the {len(big['models'])} models of 14B parameters or more was valid in more episodes when route() first asked it which library rule the task describes and then named that call (F4, filled dots) than without the library (F0, open circles): pooled, {P(bp['F0'])}% to {P(bp['F4'])}%, as high as when we named the call ourselves (F2, black dashes; {P(bp['F2'])}%). Pooled, the {len(small['models'])} smaller models went from {P(sp['F0'])}% to {P(sp['F4'])}% (F2 {P(sp['F2'])}%); only {top_small} gained more than one episode ({per[top_small]['F0']['valid']} to {per[top_small]['F4']['valid']} of 18). In every model F4 and F2 differ by at most {max_dev} episodes ({f4f2['same']} of {f4f2['models']} identical). One column per model, ordered by size; 18 episodes per model and arm.

P3 (replaces the TikZ arms diagram, page 3)
Naming the call is what lifts validity, whether we name it (F2) or route() does (F4); documenting every rule (F3) adds nothing over the interface alone (F1). Left: what each arm of the library study (§5) adds to the system prompt, the algorithm listing and the task text; all share the task text and Check feedback. In F4, route() is asked first and its answer is named in the task text; it named the task's rule in {r['right']} of {n_routes} episodes. Right: valid episodes per model (ticks), pooled over the {len(big['models'])} models of 14B or more (dots) and the {len(small['models'])} smaller ones (diamonds); numbers: all {len(per)} models (F0 {P(pooled['F0'])}%, F1 {P(pooled['F1'])}%, F2 {P(pooled['F2'])}%, F3 {P(pooled['F3'])}%, F4 {P(pooled['F4'])}%).

P4 (replaces pfix_analysis, page 4)
Why the library helps and where it stops. (a) How each episode ended, per size group ({bp['F0']['n']} and {sp['F0']['n']} episodes per arm): valid with the first program, valid after feedback, accepted but wrong (passed every visible check, wrong output), or rejected (no program passed the checks). Naming the call raised the larger models' first-program successes from {bp['F0']['first_valid']} (F0) to {bp['F2']['first_valid']} (F2) and {bp['F4']['first_valid']} (F4); accepted-but-wrong programs were most common in F1 ({bp['F1']['accepted_wrong']}) and F3 ({bp['F3']['accepted_wrong']}), where models mostly wrapped a rule of their own. (b) The {pooled['F4']['n']} F4 episodes, from what route() named to how the episode ended (band width: episodes). The larger models always got the task's rule; their last program called positions() in {tb['last_calls_positions']} of {tb['episodes']} episodes ({tb['last_calls_positions_valid']} valid) and kept a rule of its own in {tb['last_own_rule']} ({tb['last_own_rule_valid']} valid). The smaller models got the rule in {ts['episodes']} episodes but kept their own rule in {ts['last_own_rule']} ({ts.get('last_own_rule_valid', 0)} valid), and no program ran in {ts['no_program_ran']}; no wrong route ended valid. (c) Share of the programs that ran that failed each visible check: without the library (F0, open) {share('F0', 'execution_lag')} traded on the close they read and {share('F0', 'warmup_rows_zero')} held positions in warm-up rows; after route() (F4, filled) {share('F4', 'execution_lag')} and {share('F4', 'warmup_rows_zero')}; position scale did not fall ({share('F0', 'position_scale')} to {share('F4', 'position_scale')}). Grey: F1 to F3. (d) Episodes in which a check flagged a program and another attempt was left: few ended valid ({fixed} of the {flagged} such episodes over the five arms; {rp['F0']['then_valid']} of {rp['F0']['with_another_attempt']} in F0, {rp['F4']['then_valid']} of {rp['F4']['with_another_attempt']} in F4).
"""
    return text  # the paper's captions use macros from pfix_numbers.tex


def main():
    setup()
    d = load()
    fig_headline(d)
    fig_arms(d)
    fig_analysis(d)
    captions(d)
    for p in PROBLEMS:
        print("PROBLEM", *p)
    if PROBLEMS:
        sys.exit(1)


if __name__ == "__main__":
    main()
