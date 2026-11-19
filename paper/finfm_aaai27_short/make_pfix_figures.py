"""Main-text data figures of the short paper, drawn from the pinned data.

    python3 make_pfix_figures.py [--png]   # writes figures/pfix_models.pdf (Figure 2),
                                           # figures/pfix_matrix.pdf (Figure 3) and
                                           # figures/pfix_analysis.pdf (Figure 4)

Designed after an independent review on 2026-11-19 (journal style, every model shown). Reads the
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
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle  # noqa: E402
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
# Figure 2 (page 2): every model, grouped by model series and ordered by size within a series

# model id -> series: the id without its size (the study's eight families split Qwen into three
# series). The analysis has no such field, so this table is checked against every model id.
SERIES = {"qwen-coder-7b": "qwen-coder", "qwen-coder-14b": "qwen-coder",
          "qwen-coder-32b": "qwen-coder", "qwen-general-7b": "qwen-general",
          "qwen-general-14b": "qwen-general", "qwen-general-32b": "qwen-general",
          "qwen3-4b": "qwen3", "qwen3-30b-a3b": "qwen3", "phi4-mini": "phi4", "phi4-14b": "phi4",
          "mistral-7b": "mistral", "mistral-nemo-12b": "mistral",
          "mistral-small-24b": "mistral", "falcon3-7b": "falcon3", "falcon3-10b": "falcon3",
          "granite-2b": "granite", "granite-8b": "granite",
          "deepseek-r1-distill-14b": "deepseek-r1-distill", "olmo3-7b": "olmo3",
          "smollm3-3b": "smollm3"}


def fig_families(d, name="pfix_models"):
    per, sizes = d["per_model"], d["sizes_b"]
    big, small = d["by_size"]["ge14b"], d["by_size"]["lt14b"]
    assert set(SERIES) == set(per)
    n = per["qwen3-4b"]["F0"]["n"]
    fams = {}
    for m in per:
        fams.setdefault(SERIES[m], []).append(m)
    for f in fams:
        fams[f].sort(key=lambda m: (sizes[m], m))
    # series from the one with the largest model, a series' models from small to large; the
    # models alone in their series last, under one bracket
    alone = sorted((fams.pop(f)[0] for f in [f for f in fams if len(fams[f]) == 1]),
                   key=lambda m: (-sizes[m], m))
    order = sorted(fams, key=lambda f: (-sizes[fams[f][-1]], -len(fams[f]), f))
    fams["others"] = alone
    order.append("others")
    FGAP = 0.55
    x, xpos, spans = 0.0, {}, {}
    for f in order:
        x0 = x
        for m in fams[f]:
            xpos[m] = x
            x += 1
        spans[f] = (x0, x - 1)
        x += FGAP
    xmax = x - FGAP - 1

    fig = plt.figure(figsize=(COL, 1.96))
    ax = axes_in(fig, 0.335, 0.40, 2.945, 1.22)
    journal(ax, bottom=False)
    xl = (-0.55, xmax + 0.55)
    ax.set_xlim(*xl)
    ax.set_ylim(-4, 104)
    ax.set_yticks([0, 25, 50, 75, 100], ["0", "", "50", "", "100"])
    ax.set_ylabel("valid (%)", labelpad=0.5, color=INK2)
    ax.spines["left"].set_bounds(0, 100)
    ax.set_xticks([])
    for m in per:                                 # a model of 14B or more: a blue column
        if sizes[m] >= 14:
            ax.axvspan(xpos[m] - 0.5, xpos[m] + 0.5, color=tint(BLUE, 0.08), linewidth=0,
                       zorder=0)
    for f in order[:-1]:                          # F4 along the series, small to large
        ms = fams[f]
        if len(ms) > 1:
            ax.plot([xpos[m] for m in ms], [pct(per[m]["F4"]["valid"], n) for m in ms],
                    color=tint(INK2, 0.5), linewidth=0.6, zorder=1)
    for m in per:
        c = BLUE if sizes[m] >= 14 else MAG
        v0, v2, v4 = (pct(per[m][a]["valid"], n) for a in ("F0", "F2", "F4"))
        xi = xpos[m]
        if abs(v4 - v0) > 0.1:
            ax.plot([xi, xi], [v0, v4], color=tint(c, 0.45), linewidth=1.1, zorder=2)
        ax.plot([xi - 0.36, xi + 0.36], [v2, v2], color=INK, linewidth=0.9, zorder=3,
                solid_capstyle="butt")
        ax.scatter([xi], [v0], s=10, facecolor=WHITE, edgecolor=c, linewidth=0.75, zorder=4)
        ax.scatter([xi], [v4], s=16, facecolor=c, edgecolor=WHITE, linewidth=0.5, zorder=5)
    # under each column its size; under each series a bracket and the series' name
    for m in per:
        ax.text(xpos[m], -9, f"{sizes[m]:g}", ha="center", va="top", fontsize=FS, color=INK2,
                clip_on=False)
    for k, f in enumerate(order):
        a, b = spans[f]
        ax.plot([a - 0.32, b + 0.32], [-6.5, -6.5], color=MUTED, linewidth=0.5, clip_on=False,
                solid_capstyle="butt")
        ax.text((a + b) / 2, -19.5 - 9.5 * (k % 2), f, ha="center", va="top", fontsize=FS,
                color=INK if f != "others" else INK2, clip_on=False)
    m = alone[0]                                  # the one model alone in its series that rose
    assert sizes[m] >= 14 and all(sizes[o] < 14 for o in alone[1:])
    v4 = pct(per[m]["F4"]["valid"], n)
    ax.text(xpos[m] + 0.35, v4 + 17, m, ha="right", va="bottom", fontsize=FS, color=INK2)
    ax.plot([xpos[m], xpos[m]], [v4 + 5, v4 + 16], color=MUTED, linewidth=0.5, zorder=1)
    # group headers with the pooled rates (F0 -> F4)
    x0 = 0.05
    for group, colour, label in ((big, BLUE, f"≥14B ({len(big['models'])})"),
                                 (small, MAG, f"<14B ({len(small['models'])})")):
        p = group["pooled"]
        mark(fig, x0 + 0.03, 1.885, "dot", colour)
        x1 = rich(fig, x0 + 0.09, 1.885, [label], weight="bold")
        x0 = rich(fig, x1 + 0.05, 1.885, [f"pooled {pct(p['F0']['valid'], p['F0']['n']):.1f}% → "
                                          f"{pct(p['F4']['valid'], p['F4']['n']):.1f}%"],
                  colour=INK2) + 0.16
    # key, in the empty upper right
    kx, _ = to_inches(fig, ax, spans[order[-3]][0] - 0.3, 0)
    for k, (kind, colour, parts) in enumerate((("open", INK2, ["F0 no library"]),
                                               ("dot", INK2, ["F4 ", "`route()`", " first"]),
                                               ("dash", INK, ["F2 call named by us"]))):
        _, ky = to_inches(fig, ax, 0, 96 - 12.5 * k)
        key(fig, kx, ky, [(kind, colour, parts)])
    check_and_save(fig, name)


# --------------------------------------------------------------------------------------------
# Figure 3 (page 3): every model under every arm, and what route() named for it

ROUTE_KINDS = (("right", "the task's rule"), ("wrong", "another rule"), ("none", "no rule"))


def fig_matrix(d, name="pfix_matrix"):
    per, sizes, rbm = d["per_model"], d["sizes_b"], d["routing_by_model"]
    big, small, pooled = d["by_size"]["ge14b"], d["by_size"]["lt14b"], d["pooled"]
    blocks = [("ge14b", sorted(big["models"], key=lambda m: (-sizes[m], m))),
              ("lt14b", sorted(small["models"], key=lambda m: (-sizes[m], m)))]
    n = per[blocks[0][1][0]]["F0"]["n"]
    pitch, gap = 0.106, 0.07                      # row pitch, gap between the size groups
    rows_h = len(per) * pitch + gap
    top_pad, bottom_pad = 0.36, 0.36              # header above; "all models" row, key below
    H = top_pad + rows_h + bottom_pad
    fig = plt.figure(figsize=(COL, H))
    tr = fig.dpi_scale_trans
    x_name, x_size = 0.19, 1.39                   # name left edge, size right edge
    cw, x_cells = 0.226, 1.43                     # cell pitch, left edge of F0
    x_bar = x_cells + 5 * cw + 0.09               # route() bar
    w_bar = COL - x_bar - 0.03
    y = H - top_pad                               # top of the first row
    ys = {}
    for g, ms in blocks:
        y0 = y
        for m in ms:
            ys[m] = y - pitch / 2
            y -= pitch
        # size group: a bar in its colour and its label, as in Figure 4a
        fig.add_artist(Line2D([0.135, 0.135], [y + 0.015, y0 - 0.015], transform=tr,
                              color=GROUP_COLOUR[g], linewidth=1.6, solid_capstyle="butt"))
        fig.text(0.07, (y + y0) / 2, "≥14B" if g == "ge14b" else "<14B", transform=tr,
                 rotation=90, ha="center", va="center", fontsize=FS, color=INK,
                 fontweight="bold")
        y -= gap
    for g, ms in blocks:
        colour = GROUP_COLOUR[g]
        for m in ms:
            yc = ys[m]
            fig.text(x_name, yc, m, transform=tr, fontsize=FS, color=INK, ha="left",
                     va="center_baseline")
            fig.text(x_size, yc, f"{sizes[m]:g}", transform=tr, fontsize=FS, color=INK2,
                     ha="right", va="center_baseline")
            for k, a in enumerate(ARMS):
                v = per[m][a]["valid"]
                t = v / n
                face = tint(colour, 0.10 + 0.90 * t) if v else EMPTY
                fig.add_artist(FancyBboxPatch((x_cells + k * cw + 0.012, yc - pitch / 2 + 0.01),
                                              cw - 0.024, pitch - 0.02, transform=tr,
                                              boxstyle="round,pad=0,rounding_size=0.012",
                                              facecolor=face, edgecolor="none"))
                fig.text(x_cells + (k + 0.5) * cw, yc, f"{v}", transform=tr, fontsize=FS,
                         ha="center", va="center_baseline",
                         color=WHITE if t > 0.45 else (INK if v else MUTED))
            # what route() named in the model's 18 F4 episodes
            r = rbm[m]
            assert sum(r.get(k, 0) for k, _ in ROUTE_KINDS) == n
            left = x_bar
            for k, face in zip(("right", "wrong", "none"), (colour, INK2, tint(LINE, 0.75))):
                w = w_bar * r.get(k, 0) / n
                if w:
                    fig.add_artist(Rectangle((left, yc - 0.033), w, 0.066, transform=tr,
                                             facecolor=face, edgecolor="none"))
                    if left > x_bar:          # surface hairline between segments
                        fig.add_artist(Line2D([left, left], [yc - 0.033, yc + 0.033],
                                              transform=tr, color=WHITE, linewidth=0.8))
                left += w
    # header: arms over the cells, route() over the bars
    hy = H - top_pad + 0.035
    for k, a in enumerate(ARMS):
        fig.text(x_cells + (k + 0.5) * cw, hy, a, transform=tr, fontsize=FS, color=INK,
                 ha="center", va="bottom", fontweight="bold")
    fig.text(x_size, hy, "B", transform=tr, fontsize=FS, color=INK2, ha="right", va="bottom")
    fig.text(x_cells + 2.5 * cw, hy + 0.14, f"valid episodes of {n}", transform=tr,
             fontsize=FS, color=MUTED, ha="center", va="bottom")
    w = _width(fig, "route()", family=MONO) + _width(fig, " named")
    rich(fig, x_bar + w_bar - w, hy + 0.14 + 0.045, ["`route()`", " named"], colour=MUTED)
    for xx, s, ha in ((x_bar, "0", "left"), (x_bar + w_bar, f"{n}", "right")):
        fig.text(xx, hy, s, transform=tr, fontsize=FS, color=INK2, ha=ha, va="bottom")
    # bottom row: all 20 models, as percentages, and every route() answer
    yb = y + gap - 0.13
    fig.add_artist(Line2D([x_name, COL - 0.03], [yb + 0.075, yb + 0.075], transform=tr,
                          color=LINE, linewidth=0.5))
    fig.text(x_name, yb, f"all {len(per)} models, %", transform=tr, fontsize=FS, color=INK,
             ha="left", va="center_baseline")
    for k, a in enumerate(ARMS):
        fig.text(x_cells + (k + 0.5) * cw, yb, f"{pct(pooled[a]['valid'], pooled[a]['n']):.1f}",
                 transform=tr, fontsize=FS, color=INK, ha="center", va="center_baseline")
    r = d["routing"]
    total = sum(r.get(k, 0) for k, _ in ROUTE_KINDS)
    fig.text(x_bar + w_bar / 2, yb, f"{r['right']} of {total}", transform=tr, fontsize=FS,
             color=INK, ha="center", va="center_baseline")
    # key of the bars: the task's rule in the colour of the size group, another rule, no rule
    ky = yb - 0.15
    kx = rich(fig, x_name, ky, ["In F4, ", "`route()`", " named"], colour=INK2) + 0.07
    fig.add_artist(Rectangle((kx, ky), 0.09, 0.035, transform=tr, facecolor=BLUE,
                             edgecolor="none"))
    fig.add_artist(Rectangle((kx, ky - 0.035), 0.09, 0.035, transform=tr, facecolor=MAG,
                             edgecolor="none"))
    kx = rich(fig, kx + 0.11, ky, ["the task's rule"]) + 0.09
    key(fig, kx, ky, [(("box", INK2, None, "none"), None, ["another rule"]),
                      (("box", tint(LINE, 0.75), None, "none"), None, ["none"])], gap=0.09)
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
    H = 2.0
    fig = plt.figure(figsize=(TEXTW, H))
    tr = fig.dpi_scale_trans
    title_y, plot_b, plot_h, key_y = H - 0.11, 0.30, 1.27, H - 0.275

    # a: how each episode ended, per arm and size group ----------------------------------
    ax = axes_in(fig, 0.36, plot_b, 1.2, plot_h - 0.03)
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
    key(fig, 0.06, key_y, [
        (("box", BLUE, None, "none"), None, ["valid first try"]),
        (("box", tint(BLUE, VALID_LATER), None, "none"), None, ["valid later"]),
        (("box", WHITE, "//////", ORANGE), None, ["accepted but wrong"]),
        (("box", REST, None, "none"), None, ["rejected"])], gap=0.12, wrap=1.62, line=0.12)

    # b: F4 -- what route() named, what the last program did, how the episode ended -------
    r, tb, ts = d["routing"], big["told"]["F4"], small["told"]["F4"]
    rb, rs = big["routing"], small["routing"]
    assert rb.get("none", 0) == 0 and rb.get("wrong", 0) == 0     # only smaller models missed
    assert r["right"] == tb["episodes"] + ts["episodes"] == rb["right"] + rs["right"]
    ax = axes_in(fig, 1.84, 0.12, 2.48, plot_h + 0.29)
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
    checks = [("execution_lag", "execution lag", "trades on the close it reads"),
              ("warmup_rows_zero", "warm-up rows zero", "a position before the lookback"),
              ("takes_positions", "takes positions", "no position on any row"),
              ("position_scale", "position scale", "not ±1/N, or over the cap"),
              ("split_representation", "split representation", "differs if splits pre-applied"),
              ("finite_output", "finite output", "missing or infinite values")]
    fc = d["failed_checks"]
    share = {k: {a: pct(fc[a][k], fc[a]["programs_ran"]) for a in ARMS} for k, *_ in checks}
    assert [k for k, *_ in checks] == sorted(share, key=lambda k: -share[k]["F0"])
    x_lab, x_plot, w_plot = 4.47, 5.62, 0.86
    ax = axes_in(fig, x_plot, plot_b, w_plot, plot_h + 0.04)
    journal(ax, left=False)
    ax.set_ylim(len(checks) - 0.5, -0.5)
    ax.set_xlim(0, 60)
    for yy, (k, label, note) in enumerate(checks):
        vals = share[k]
        ax.plot([min(vals.values()), max(vals.values())], [yy, yy], color=tint(ORANGE, 0.3),
                linewidth=1.3, zorder=1, solid_capstyle="round")
        for a in ("F1", "F2", "F3"):
            ax.scatter([vals[a]], [yy], s=6, color=INK2, linewidth=0, zorder=2)
        ax.scatter([vals["F0"]], [yy], s=15, facecolor=WHITE, edgecolor=ORANGE, linewidth=0.9,
                   zorder=3)
        ax.scatter([vals["F4"]], [yy], s=17, facecolor=ORANGE, edgecolor=WHITE, linewidth=0.5,
                   zorder=4)
        _, yc = to_inches(fig, ax, 0, yy)
        fig.text(x_lab, yc + 0.048, label, transform=tr, fontsize=FS, color=INK, ha="left",
                 va="center_baseline")
        fig.text(x_lab, yc - 0.052, note, transform=tr, fontsize=FS, color=MUTED, ha="left",
                 va="center_baseline")
        fig.text(x_plot + w_plot + 0.08, yc, f"{vals['F0']:.0f} → {vals['F4']:.0f}",
                 transform=tr, fontsize=FS, color=INK, ha="left", va="center_baseline")
    ax.set_yticks([])
    ax.set_xticks([0, 20, 40, 60], ["0", "20", "40", "60%"])
    ax.set_xlabel("of the programs that ran", labelpad=1.5, color=INK2)
    fig.text(x_plot + w_plot + 0.08, key_y - 0.115, "F0 → F4", transform=tr, fontsize=FS,
             color=MUTED, ha="left", va="center")
    letter(fig, 4.40, title_y, "c", ["What the checks flagged"])
    key(fig, x_plot - 0.02, key_y, [("open", ORANGE, ["F0"]), ("small", INK2, ["F1–F3"]),
                                    ("dot", ORANGE, ["F4"])], gap=0.1)
    check_and_save(fig, name)


def main():
    setup()
    d = load()
    fig_families(d)
    fig_matrix(d)
    fig_analysis(d)
    for p in PROBLEMS:
        print("PROBLEM", *p)
    if PROBLEMS:
        sys.exit(1)


if __name__ == "__main__":
    main()
