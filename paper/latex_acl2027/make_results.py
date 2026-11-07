"""Draw the result figures of the Fin-RSI paper from the benchmark's episode records.

    python paper/latex_acl2027/make_results.py runs/rsi-bench/*/episodes.jsonl \
        [--fits runs/rsi-bench/*/fits.jsonl]
    python paper/latex_acl2027/make_results.py --placeholder

Input: the JSON lines fin_skills.rsi.benchmark() writes, one per episode, for any number of
models. Only synthetic seeds enter these figures; real-panel episodes are reported on their own.
--fits reads Report.fits records (method, arm, model_id, usage) for the development cost.

figures/results_effects.pdf (one column)
  (a) Each method minus NoImprove in valid rate, per arm, paired on model, task and seed.
  (b) The library: cards+library minus cards, per method, paired the same way.
      Intervals: 95% bootstrap, resampling held-out tasks (all seeds and models of a task move
      together). Filled markers: Holm-adjusted p < 0.05 over every comparison in (a) and (b).
figures/results_cost.pdf (one column)
  Valid rate against generated tokens per episode (log scale): one point per method and arm,
  the two arms of a method joined by a line and labelled directly, with the Pareto frontier.
  With --fits, a dotted extension adds each learning method's development tokens spread over
  its held-out episodes.
figures/results_outcomes.pdf (one column)
  Where every episode ends: valid; accepted but wrong (passed every check, wrong answer: what the
  checks miss); rejected by a check; no runnable program (no program, or it did not run).

--placeholder draws the same figures from generated records. With "null" (the default) every
method and arm has the same success probabilities and costs; with "spread" they differ by
invented amounts, so the layout can be checked on data that is spread out the way results will
be. Either way the figures are stamped PLACEHOLDER; their numbers mean nothing and must never
reach a submission.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
sys.path.insert(0, str(HERE))
from make_figures import (BLUE, GRID, INK, INK2, META, MUTED, ORANGE, SHORT,  # noqa: E402
                          STYLE)

METHODS = ("no_improve", "self_debug", "check_feedback", "reflexion", "playbook", "team", "evolve")
LABEL = dict(no_improve="NoImprove", self_debug="SelfDebug", check_feedback="CheckFeedback",
             reflexion="Reflexion", playbook="PlaybookLearning", team="Team", evolve="Evolve")
ARMS = ("cards", "cards+library")
ARM_LABEL = {"cards": "no library", "cards+library": "with library"}
ARM_COLOR = {"cards": INK2, "cards+library": BLUE}         # the library is the skills' blue
ARM_MARK = {"cards": "o", "cards+library": "s"}
OUTCOMES = ("valid", "accepted but wrong", "rejected by a check", "no runnable program")
OUTCOME_COLOR = ("#2f3a4f", ORANGE, "#949cab", "#d9dde3")  # checks' blind spot in checks orange
DRAWS = 4000


# ---------------------------------------------------------------- records
def outcome(row: dict) -> str:
    steps = row.get("steps") or []
    if not steps or not steps[-1].get("executed"):
        return OUTCOMES[3]
    if not row.get("accepted"):
        return OUTCOMES[2]
    return OUTCOMES[0] if row.get("valid") else OUTCOMES[1]


def load(patterns) -> list[dict]:
    rows = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            with open(path, encoding="utf-8") as fh:
                rows += [json.loads(line) for line in fh if line.strip()]
    return [r for r in rows if r.get("seed") != "real"]


# Invented for --placeholder spread only (layout checks): no method has these properties.
SPREAD_SHIFT = dict(zip(METHODS, (0.0, 0.03, 0.08, 0.10, 0.13, 0.06, 0.04)))
SPREAD_LIB = dict(zip(METHODS, (0.02, 0.04, 0.06, 0.05, 0.07, 0.03, 0.05)))
SPREAD_TOKENS = dict(zip(METHODS, (1500, 2600, 4200, 7000, 4800, 11000, 6000)))


def placeholder(kind: str = "null", seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Records in the benchmark's schema. null: no method, arm or model differs in anything.
    spread: methods and arms differ by the invented SPREAD_* amounts above."""
    from fin_skills.tasks import suite

    spread = kind == "spread"
    rng = np.random.default_rng(seed)
    tasks = [t.id for t in suite().heldout]
    models = [f"model-{i}" for i in range(1, 6)]
    p_task = dict(zip(tasks, rng.beta(2.0, 2.0, size=len(tasks))))
    rows, fits = [], []
    for model in models:
        for method in METHODS:
            for arm in ARMS:
                lib = arm == "cards+library"
                shift = (SPREAD_SHIFT[method] + (SPREAD_LIB[method] if lib else 0.0)
                         if spread else 0.0)
                tokens = SPREAD_TOKENS[method] * (0.85 if lib else 1.0) if spread else 2000
                if method in ("playbook", "evolve"):
                    dev = (400_000 if method == "playbook" else 900_000) if spread else 400_000
                    fits.append(dict(method=method, arm=arm, model_id=model,
                                     usage=dict(calls=300, generated_tokens=dev)))
                for task in tasks:
                    for s in (11, 23, 37):
                        u = rng.random()
                        p = float(np.clip(p_task[task] + shift, 0.02, 0.95))
                        valid = u < p
                        wrong = p <= u < p + 0.05
                        rejected = p + 0.05 <= u < p + 0.05 + 0.7 * (1 - p - 0.05)
                        ran = valid or wrong or rejected
                        rows.append(dict(
                            method=method, arm=arm, model_id=model, task_id=task, seed=s,
                            valid=bool(valid), accepted=bool(valid or wrong),
                            steps=[dict(executed=bool(ran))],
                            usage=dict(calls=int(rng.integers(1, 7)),
                                       generated_tokens=int(rng.lognormal(np.log(tokens), 0.5)))))
    return rows, fits


# ---------------------------------------------------------------- statistics
def cell(rows):
    """{(method, arm): {(model, task, seed): row}}"""
    out: dict = {}
    for r in rows:
        out.setdefault((r["method"], r["arm"]), {})[(r["model_id"], r["task_id"], r["seed"])] = r
    return out


def paired(a: dict, b: dict, tasks, rng) -> dict:
    """Mean of valid(a) - valid(b) over matched episodes, with a task-clustered bootstrap."""
    keys = sorted(set(a) & set(b))
    by_task: dict = {}
    for k in keys:
        by_task.setdefault(k[1], []).append(float(a[k]["valid"]) - float(b[k]["valid"]))
    sums = np.array([sum(by_task.get(t, [])) for t in tasks])
    counts = np.array([len(by_task.get(t, [])) for t in tasks])
    est = sums.sum() / max(1, counts.sum())
    idx = rng.integers(0, len(tasks), size=(DRAWS, len(tasks)))
    boot = sums[idx].sum(1) / np.maximum(1, counts[idx].sum(1))
    p = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()) + 1 / DRAWS)
    return dict(est=est, lo=np.quantile(boot, 0.025), hi=np.quantile(boot, 0.975), p=p,
                n=len(keys))


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, running)
    return list(adj)


def rate(rows_by_key: dict, tasks, rng) -> dict:
    by_task: dict = {}
    for k, r in rows_by_key.items():
        by_task.setdefault(k[1], []).append(float(r["valid"]))
    sums = np.array([sum(by_task.get(t, [])) for t in tasks])
    counts = np.array([len(by_task.get(t, [])) for t in tasks])
    idx = rng.integers(0, len(tasks), size=(DRAWS, len(tasks)))
    boot = sums[idx].sum(1) / np.maximum(1, counts[idx].sum(1))
    return dict(est=sums.sum() / max(1, counts.sum()), lo=np.quantile(boot, 0.025),
                hi=np.quantile(boot, 0.975))


# ---------------------------------------------------------------- drawing
def stamp(ax, size: float = 13) -> None:
    ax.text(0.5, 0.5, "PLACEHOLDER\nnot results", transform=ax.transAxes, ha="center",
            va="center", rotation=22, fontsize=size, fontweight="bold", color="#c0392b",
            alpha=0.28, zorder=10, linespacing=1.0)


def tidy(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.5, pad=1.5)


def comparisons(cells, tasks) -> list:
    """(panel, method, arm, result) for every paired comparison, Holm-adjusted together."""
    rng = np.random.default_rng(0)
    comps = []
    for arm in ARMS:
        base = cells.get(("no_improve", arm), {})
        for m in METHODS[1:]:
            if (m, arm) in cells and base:
                comps.append(("a", m, arm, paired(cells[(m, arm)], base, tasks, rng)))
    for m in METHODS:
        if (m, "cards") in cells and (m, "cards+library") in cells:
            comps.append(("b", m, "cards+library",
                          paired(cells[(m, "cards+library")], cells[(m, "cards")], tasks, rng)))
    for c, p in zip(comps, holm([c[3]["p"] for c in comps])):
        c[3]["p_holm"] = p
    return comps


def arm_handles(extra=()):
    from matplotlib.lines import Line2D
    return [Line2D([], [], color=ARM_COLOR[a], marker=ARM_MARK[a], ms=4.2, lw=1.0,
                   mfc=ARM_COLOR[a], label=ARM_LABEL[a]) for a in ARMS] + list(extra)


def effects_figure(comps, placeholder_data: bool) -> None:
    plt.rcParams.update(STYLE)
    W, H = 3.0, 2.42                                     # inches; ACL column is 3.03 in
    fig = plt.figure(figsize=(W, H))
    bottom, top = 0.36, H - 0.56

    def axes(left, width):
        return fig.add_axes([left / W, bottom / H, width / W, (top - bottom) / H])

    axa, axb = axes(1.0, 1.06), axes(2.2, 0.76)

    def forest(ax, panel, offsets, labels):
        for _, m, arm, r in (c for c in comps if c[0] == panel):
            y = METHODS.index(m) + offsets[arm]
            filled = r["p_holm"] < 0.05
            ax.plot([100 * r["lo"], 100 * r["hi"]], [y, y], color=ARM_COLOR[arm], lw=1.0,
                    solid_capstyle="round", zorder=2)
            ax.plot(100 * r["est"], y, ARM_MARK[arm], ms=4.2, mew=1.0, zorder=3,
                    mec=ARM_COLOR[arm], mfc=ARM_COLOR[arm] if filled else "white")
        ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
        ax.set_yticks(range(len(METHODS)), [LABEL[m] for m in METHODS])
        ax.tick_params(axis="y", labelleft=labels)
        ax.set_ylim(len(METHODS) - 0.5, -0.6)
        tidy(ax)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)

    forest(axa, "a", {"cards": -0.2, "cards+library": 0.2}, True)
    axa.text(0, 0, "reference", color=MUTED, fontsize=7, ha="center", va="center",
             bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    axa.set_xlabel("minus NoImprove (pp)", labelpad=1.5)
    forest(axb, "b", {"cards+library": 0.0}, False)
    axb.set_xlabel("library effect (pp)", labelpad=1.5)
    for x, text in ((0.02, "(a) Each method vs. NoImprove"), (2.2, "(b) The library")):
        fig.text(x / W, (top + 0.07) / H, text, fontsize=7.5, fontweight="bold", color=INK,
                 ha="left", va="bottom")
    from matplotlib.lines import Line2D
    fig.legend(handles=arm_handles([Line2D([], [], color=INK2, marker="o", ms=4.2, lw=0,
                                           mfc="white", label="open: n.s. after Holm")]),
               loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, fontsize=7,
               handlelength=1.5, columnspacing=1.0, handletextpad=0.5, labelcolor=INK,
               borderaxespad=0.1)
    if placeholder_data:
        stamp(axa)
        stamp(axb, 9)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "results_effects.pdf", metadata=META)
    fig.savefig(OUT / "results_effects.png", dpi=250)


def place_labels(fig, ax, pts, dev) -> None:
    """Name each method once, next to one of its two points, where the name covers the least
    of the points, lines and names already placed. Greedy, in method order; no leader lines."""
    from matplotlib.patheffects import withStroke
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    to_px = ax.transData.transform
    frame = ax.get_window_extent(r)
    obstacles = []                                       # (owner, box) in display pixels

    def box(x0, y0, x1, y1):
        return Bbox([[min(x0, x1), min(y0, y1)], [max(x0, x1), max(y0, y1)]])

    def segment(owner, a, b, half=1.5):
        (x0, y0), (x1, y1) = to_px(a), to_px(b)
        n = max(2, int(np.hypot(x1 - x0, y1 - y0) / 3))
        for t in np.linspace(0, 1, n):
            x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            obstacles.append((owner, box(x - half, y - half, x + half, y + half)))

    for (m, arm), (tok, v) in pts.items():
        x, y = to_px((tok, v))
        obstacles.append((m, box(x - 5, y - 5, x + 5, y + 5)))
        if (m, arm) in dev:
            segment(m, (tok, v), (tok + dev[(m, arm)], v))
    for m in METHODS:
        if all((m, a) in pts for a in ARMS):
            segment(m, pts[(m, ARMS[0])], pts[(m, ARMS[1])])

    def overlap(b, me):
        def grow(o, owner):                              # keep clear of other methods' marks
            g = 0 if owner in (me, None) else 6
            return Bbox([[o.x0 - g, o.y0 - g], [o.x1 + g, o.y1 + g]])
        area = sum(max(0, min(b.x1, o.x1) - max(b.x0, o.x0)) *
                   max(0, min(b.y1, o.y1) - max(b.y0, o.y0))
                   for o in (grow(o, owner) for owner, o in obstacles))
        outside = (max(0, frame.x0 - b.x0) + max(0, b.x1 - frame.x1) +
                   max(0, frame.y0 - b.y0) + max(0, b.y1 - frame.y1))
        return area + 400 * outside

    inv = ax.transData.inverted()
    for m in METHODS:
        mine = [(a, pts[(m, a)]) for a in ARMS if (m, a) in pts]
        if not mine:
            continue
        t = ax.text(0, 0, LABEL[m], fontsize=7, color=INK, ha="left", va="bottom", zorder=4,
                    path_effects=[withStroke(linewidth=2.2, foreground="white")])
        ext = t.get_window_extent(r)
        w, h = ext.width, ext.height
        cands = []                                       # (preference, x0, y0) in pixels
        for rank, (a, (tok, v)) in enumerate(reversed(mine)):   # the library point first
            x, y = to_px((tok, v))
            xr = to_px((tok + dev.get((m, a), 0.0), v))[0]
            g = 4.5
            cands += [(0 + rank, x - g - w, y - h / 2), (1 + rank, xr + g, y - h / 2),
                      (2 + rank, x - w / 2, y + g), (3 + rank, x - w / 2, y - g - h),
                      (4 + rank, x - g - w, y + 1), (4 + rank, x - g - w, y - h - 1),
                      (5 + rank, xr + g, y + 1), (5 + rank, xr + g, y - h - 1)]
        best = min(cands, key=lambda c: (overlap(box(c[1], c[2], c[1] + w, c[2] + h), m),
                                         c[0]))
        t.set_position(inv.transform((best[1], best[2])))
        obstacles.append((None, box(best[1] - 2, best[2] - 1, best[1] + w + 2,
                                    best[2] + h + 1)))


def cost_figure(cells, fits, tasks, placeholder_data: bool) -> None:
    rng = np.random.default_rng(1)
    plt.rcParams.update(STYLE)
    W, H = 3.0, 2.3
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.42 / W, 0.36 / H, 2.52 / W, (H - 0.36 - 0.42) / H])
    pts = {}                                             # (method, arm) -> (tokens, valid %)
    dev = {}                                             # (method, arm) -> tokens per episode
    for (m, arm), rows in cells.items():
        tok = float(np.mean([r["usage"]["generated_tokens"] for r in rows.values()]))
        pts[(m, arm)] = (tok, 100 * rate(rows, tasks, rng)["est"])
        spent = [f["usage"]["generated_tokens"] for f in fits
                 if f["method"] == m and f["arm"] == arm]
        if spent:
            dev[(m, arm)] = sum(spent) / max(1, len(rows))
    for m in METHODS:                                    # the two arms of a method, joined
        pair = [pts[(m, a)] for a in ARMS if (m, a) in pts]
        if len(pair) == 2:
            ax.plot(*zip(*pair), color=SHORT, lw=1.3, solid_capstyle="round", zorder=2)
    for (m, arm), (tok, v) in pts.items():
        ax.plot(tok, v, ARM_MARK[arm], ms=4.2, mew=1.0, mec=ARM_COLOR[arm],
                mfc=ARM_COLOR[arm], zorder=3)
        if (m, arm) in dev:                              # development tokens per episode
            end = tok + dev[(m, arm)]
            ax.plot([tok, end], [v, v], color=ARM_COLOR[arm], lw=0.7, ls=(0, (1, 1.5)),
                    zorder=1)
            ax.plot(end, v, "|", ms=4.5, mew=1.0, color=ARM_COLOR[arm], zorder=3)
    xmax = max(tok + dev.get(k, 0.0) for k, (tok, _) in pts.items())
    front, best = [], -1.0
    for tok, v in sorted(pts.values(), key=lambda p: (p[0], -p[1])):
        if v > best:
            front.append((tok, v))
            best = v
    xs, ys = zip(*front)
    ax.step(xs, ys, where="post", color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.set_xscale("log")
    xmin = min(tok for tok, _ in pts.values())
    vals = [v for _, v in pts.values()]
    ax.set_ylim(min(vals) - 4, max(vals) + 4)
    ax.set_xlim(xmin / 1.6, xmax * 1.25)
    ax.set_xlabel("generated tokens per episode", labelpad=1.5)
    ax.set_ylabel("valid rate (%)", labelpad=1.5)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    tidy(ax)

    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    lo_x, hi_x = ax.get_xlim()
    ticks = [t for t in (300, 500, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5) if lo_x <= t <= hi_x]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xticklabels([f"{t / 1e3:g}k" if t >= 1e3 else f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    place_labels(fig, ax, pts, dev)
    from matplotlib.lines import Line2D
    extra = [Line2D([], [], color=MUTED, lw=0.7, ls=(0, (3, 2)), label="Pareto frontier")]
    if fits:
        extra.append(Line2D([], [], color=INK2, marker="|", ms=4.5, lw=0.7, ls=(0, (1, 1.5)),
                            label="+ development tokens"))
    fig.legend(handles=arm_handles(extra), loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2,
               frameon=False, fontsize=7, handlelength=1.5, columnspacing=1.0,
               handletextpad=0.5, labelcolor=INK, borderaxespad=0.1)
    if placeholder_data:
        stamp(ax)
    fig.savefig(OUT / "results_cost.pdf", metadata=META)
    fig.savefig(OUT / "results_cost.png", dpi=250)


def outcomes_figure(cells, placeholder_data: bool) -> None:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(2.62, 2.45))       # with its labels, one ACL column
    labels, y = [], 0.0
    ticks = []
    for m in METHODS:
        for arm in ARMS:
            if (m, arm) not in cells:
                continue
            rows = list(cells[(m, arm)].values())
            shares = [sum(outcome(r) == o for r in rows) / len(rows) for o in OUTCOMES]
            left = 0.0
            for share, color in zip(shares, OUTCOME_COLOR):
                ax.barh(y, 100 * share, left=100 * left, height=0.78, color=color,
                        edgecolor="white", linewidth=0.8)
                left += share
            ticks.append(y)
            labels.append(f"{LABEL[m]}" if arm == "cards" else "+ library")
            y += 1.0
        y += 0.45
    ax.set_yticks(ticks, labels)
    for t, lab in zip(ax.get_yticklabels(), labels):
        if lab == "+ library":
            t.set_color(BLUE)
    ax.set_ylim(y - 0.3, -0.75)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of episodes (%)", labelpad=1.5)
    tidy(ax)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=o) for o, c in zip(OUTCOMES, OUTCOME_COLOR)],
              loc="lower center", bbox_to_anchor=(0.3, 1.0), ncol=2, frameon=False,
              fontsize=7, handlelength=1.0, columnspacing=0.9, labelcolor=INK)
    if placeholder_data:
        stamp(ax)
    fig.savefig(OUT / "results_outcomes.pdf", bbox_inches="tight", pad_inches=0.02,
                metadata=META)
    fig.savefig(OUT / "results_outcomes.png", dpi=250, bbox_inches="tight", pad_inches=0.02)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("episodes", nargs="*", help="episodes.jsonl files or glob patterns")
    ap.add_argument("--fits", nargs="*", default=[], help="fits.jsonl files or glob patterns")
    ap.add_argument("--placeholder", nargs="?", const="null", choices=("null", "spread"),
                    help="draw from generated records, stamped PLACEHOLDER: null (no method "
                         "differs) or spread (invented differences, for layout checks)")
    a = ap.parse_args(argv)
    if a.placeholder == bool(a.episodes):
        ap.error("give episode files, or --placeholder, not both")
    if a.placeholder:
        rows, fits = placeholder(a.placeholder)
        print(f"PLACEHOLDER ({a.placeholder}): generated records; these figures are not results")
    else:
        rows, fits = load(a.episodes), load(a.fits) if a.fits else []
    from fin_skills.tasks import suite
    tasks = sorted({r["task_id"] for r in rows} | {t.id for t in suite().heldout})
    cells = cell(rows)
    print(f"{len(rows)} synthetic-seed episodes, {len({r['model_id'] for r in rows})} models, "
          f"{len(cells)} method x arm cells")
    comps = comparisons(cells, tasks)
    for c in comps:
        print(f"{c[0]} {LABEL[c[1]]:16s} {c[2]:13s} {100 * c[3]['est']:+6.1f} pp "
              f"[{100 * c[3]['lo']:+6.1f}, {100 * c[3]['hi']:+6.1f}]  p={c[3]['p']:.4f} "
              f"holm={c[3]['p_holm']:.4f}  n={c[3]['n']}")
    effects_figure(comps, bool(a.placeholder))
    cost_figure(cells, fits, tasks, bool(a.placeholder))
    outcomes_figure(cells, bool(a.placeholder))
    source = {"null": "null generated records: no method or arm differs",
              "spread": "generated records with invented differences (SPREAD_* in "
                        "make_results.py), for layout checks only"}
    (OUT / "results_source.txt").write_text(
        (f"PLACEHOLDER ({source[a.placeholder]})\n" if a.placeholder else
         "episodes: " + " ".join(a.episodes) + "\nfits: " + " ".join(a.fits) + "\n"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
