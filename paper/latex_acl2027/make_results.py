"""Draw the result figures of the Fin-RSI paper from the benchmark's episode records.

    python paper/latex_acl2027/make_results.py runs/rsi-bench/*/episodes.jsonl \
        [--fits runs/rsi-bench/*/fits.jsonl]
    python paper/latex_acl2027/make_results.py --placeholder

Input: the JSON lines fin_skills.rsi.benchmark() writes, one per episode, for any number of
models. Only synthetic seeds enter these figures; real-panel episodes are reported on their own.
--fits reads Report.fits records (method, arm, model_id, usage) for the development cost.

figures/results_main.pdf (full width)
  (a) Each method minus NoImprove in valid rate, per arm, paired on model, task and seed.
  (b) The library: cards+library minus cards, per method, paired the same way.
      Intervals: 95% bootstrap, resampling held-out tasks (all seeds and models of a task move
      together). Filled markers: Holm-adjusted p < 0.05 over every comparison in (a) and (b).
  (c) Valid rate against generated tokens per episode (log scale), per method and arm, with the
      Pareto frontier. With --fits, an open marker adds each learning method's development tokens
      spread over its held-out episodes.
figures/results_outcomes.pdf (one column)
  Where every episode ends: valid; accepted but wrong (passed every check, wrong answer: what the
  checks miss); rejected by a check; no runnable program (no program, or it did not run).

--placeholder draws the same figures from generated records in which every method and arm has
the same success probabilities and the same costs. The figures are stamped PLACEHOLDER; their
numbers mean nothing and must never reach a submission.
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
from make_figures import BLUE, GRID, INK, INK2, MUTED, ORANGE, STYLE  # noqa: E402

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


def placeholder(seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Null records in the benchmark's schema: no method, arm or model differs in anything."""
    from fin_skills.tasks import suite

    rng = np.random.default_rng(seed)
    tasks = [t.id for t in suite().heldout]
    models = [f"model-{i}" for i in range(1, 6)]
    p_task = dict(zip(tasks, rng.beta(2.0, 2.0, size=len(tasks))))
    rows, fits = [], []
    for model in models:
        for method in METHODS:
            for arm in ARMS:
                if method in ("playbook", "evolve"):
                    fits.append(dict(method=method, arm=arm, model_id=model,
                                     usage=dict(calls=300, generated_tokens=400_000)))
                for task in tasks:
                    for s in (11, 23, 37):
                        u = rng.random()
                        p = p_task[task]
                        valid = u < p
                        wrong = p <= u < p + 0.08
                        rejected = p + 0.08 <= u < p + 0.08 + 0.7 * (1 - p - 0.08)
                        ran = valid or wrong or rejected
                        rows.append(dict(
                            method=method, arm=arm, model_id=model, task_id=task, seed=s,
                            valid=bool(valid), accepted=bool(valid or wrong),
                            steps=[dict(executed=bool(ran))],
                            usage=dict(calls=int(rng.integers(1, 7)),
                                       generated_tokens=int(rng.lognormal(7.6, 0.5)))))
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
def stamp(ax) -> None:
    ax.text(0.5, 0.5, "PLACEHOLDER\nnot results", transform=ax.transAxes, ha="center",
            va="center", rotation=22, fontsize=13, fontweight="bold", color="#c0392b",
            alpha=0.28, zorder=10, linespacing=1.0)


def tidy(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.5, pad=1.5)


def main_figure(cells, fits, tasks, placeholder_data: bool) -> list[str]:
    rng = np.random.default_rng(0)
    comps = []                                   # (panel, method, arm, result)
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

    plt.rcParams.update(STYLE)
    W, H = 6.28, 2.3                                     # inches; ACL text width is 6.30 in
    fig = plt.figure(figsize=(W, H))
    bottom, top = 0.4, H - 0.5

    def axes(left, width):
        return fig.add_axes([left / W, bottom / H, width / W, (top - bottom) / H])

    axa, axb, axc = axes(0.98, 1.42), axes(2.52, 1.1), axes(4.12, 1.18)

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
        ax.set_ylim(len(METHODS) - 0.45, -0.55)
        tidy(ax)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)

    forest(axa, "a", {"cards": -0.16, "cards+library": 0.16}, True)
    axa.text(0, 0, "reference", color=MUTED, fontsize=7, ha="center", va="center",
             bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    axa.set_xlabel("minus NoImprove (pp)", labelpad=1.5)
    forest(axb, "b", {"cards+library": 0.0}, False)
    axb.set_xlabel("with minus without (pp)", labelpad=1.5)
    for x, text in ((0.02, "(a) Each method vs. NoImprove"), (2.52, "(b) The library"),
                    (3.72, "(c) Validity against cost")):
        fig.text(x / W, (top + 0.08) / H, text, fontsize=7.5, fontweight="bold", color=INK,
                 ha="left", va="bottom")

    # (c) cost and validity: one point per method and arm
    pts = []
    for m in METHODS:
        for arm in ARMS:
            if (m, arm) not in cells:
                continue
            rows = cells[(m, arm)]
            tok = float(np.mean([r["usage"]["generated_tokens"] for r in rows.values()]))
            v = rate(rows, tasks, rng)
            pts.append((m, arm, tok, v))
            axc.plot([tok, tok], [100 * v["lo"], 100 * v["hi"]], color=ARM_COLOR[arm],
                     lw=0.8, alpha=0.5, zorder=2)
            axc.plot(tok, 100 * v["est"], ARM_MARK[arm], ms=4.2, mew=1.0, mec=ARM_COLOR[arm],
                     mfc=ARM_COLOR[arm], zorder=3)
            spent = [f["usage"]["generated_tokens"] for f in fits
                     if f["method"] == m and f["arm"] == arm]
            if spent:                                    # development tokens per episode
                per = sum(spent) / max(1, len(rows))
                axc.plot([tok, tok + per], [100 * v["est"]] * 2, color=ARM_COLOR[arm], lw=0.7,
                         ls=(0, (2, 1.5)), zorder=1)
                axc.plot(tok + per, 100 * v["est"], "|", ms=5, mew=1.0, color=ARM_COLOR[arm],
                         zorder=3)
    xmax = max([p[2] for p in pts] + [p[2] + sum(f["usage"]["generated_tokens"] for f in fits
               if f["method"] == p[0] and f["arm"] == p[1]) / max(1, len(cells[p[:2]]))
               for p in pts])
    front, best = [], -1.0
    for m, arm, tok, v in sorted(pts, key=lambda p: (p[2], -p[3]["est"])):
        if v["est"] > best:
            front.append((tok, 100 * v["est"]))
            best = v["est"]
    xs, ys = zip(*front)
    axc.step(list(xs) + [xmax], list(ys) + [ys[-1]], where="post", color=MUTED, lw=0.7,
             ls=(0, (3, 2)), zorder=1)
    axc.set_xscale("log")
    xmin = min(p[2] for p in pts)
    axc.set_xlim(xmin / 1.6, xmax * 1.6)
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    lo_x, hi_x = axc.get_xlim()
    ticks = [t for t in (100, 300, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6) if lo_x <= t <= hi_x]
    axc.xaxis.set_major_locator(FixedLocator(ticks))
    axc.set_xticklabels([f"{t / 1e3:g}k" if t >= 1e3 else f"{t:g}" for t in ticks])
    axc.xaxis.set_minor_locator(NullLocator())
    axc.xaxis.set_minor_formatter(NullFormatter())
    axc.set_xlabel("generated tokens per episode", labelpad=1.5)
    axc.set_ylabel("valid rate (%)", labelpad=1.5)
    axc.grid(color=GRID, lw=0.5)
    axc.set_axisbelow(True)
    tidy(axc)
    # labels in a column right of the axes, in the order of the library points, with leaders
    lib = sorted((p for p in pts if p[1] == "cards+library"), key=lambda p: -p[3]["est"])
    y0, y1 = axc.get_ylim()
    to_frac = lambda y: (y - y0) / (y1 - y0)
    slots = np.linspace(0.97, 0.03, len(lib))
    for (m, _, tok, v), slot in zip(lib, slots):
        axc.annotate(LABEL[m], xy=(tok, 100 * v["est"]), xycoords="data",
                     xytext=(1.06, slot), textcoords="axes fraction", fontsize=7, color=INK,
                     va="center", ha="left", annotation_clip=False,
                     arrowprops=dict(arrowstyle="-", color=GRID, lw=0.5, shrinkA=1.5,
                                     shrinkB=2.5))

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=ARM_COLOR[a], marker=ARM_MARK[a], ms=4.2, lw=1.0,
                      mfc=ARM_COLOR[a], label=ARM_LABEL[a]) for a in ARMS]
    handles += [Line2D([], [], color=INK2, marker="o", ms=4.2, lw=0, mfc="white",
                       label="open: not significant after Holm")]
    if fits:
        handles += [Line2D([], [], color=INK2, marker="|", ms=5, lw=0.7, ls=(0, (2, 1.5)),
                           label="plus development tokens")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=4,
               frameon=False, fontsize=7, handlelength=1.8, columnspacing=1.4,
               labelcolor=INK, borderaxespad=0.1)
    if placeholder_data:
        for ax in (axa, axb, axc):
            stamp(ax)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "results_main.pdf")
    fig.savefig(OUT / "results_main.png", dpi=250)
    return [f"{c[0]} {LABEL[c[1]]:16s} {c[2]:13s} {100 * c[3]['est']:+6.1f} pp "
            f"[{100 * c[3]['lo']:+6.1f}, {100 * c[3]['hi']:+6.1f}]  p={c[3]['p']:.4f} "
            f"holm={c[3]['p_holm']:.4f}  n={c[3]['n']}" for c in comps]


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
    fig.savefig(OUT / "results_outcomes.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "results_outcomes.png", dpi=250, bbox_inches="tight", pad_inches=0.02)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("episodes", nargs="*", help="episodes.jsonl files or glob patterns")
    ap.add_argument("--fits", nargs="*", default=[], help="fits.jsonl files or glob patterns")
    ap.add_argument("--placeholder", action="store_true",
                    help="draw from null generated records, stamped PLACEHOLDER")
    a = ap.parse_args(argv)
    if a.placeholder == bool(a.episodes):
        ap.error("give episode files, or --placeholder, not both")
    if a.placeholder:
        rows, fits = placeholder()
        print("PLACEHOLDER: null generated records; these figures are not results")
    else:
        rows, fits = load(a.episodes), load(a.fits) if a.fits else []
    from fin_skills.tasks import suite
    tasks = sorted({r["task_id"] for r in rows} | {t.id for t in suite().heldout})
    cells = cell(rows)
    print(f"{len(rows)} synthetic-seed episodes, {len({r['model_id'] for r in rows})} models, "
          f"{len(cells)} method x arm cells")
    for line in main_figure(cells, fits, tasks, a.placeholder):
        print(line)
    outcomes_figure(cells, a.placeholder)
    (OUT / "results_source.txt").write_text(
        ("PLACEHOLDER (null generated records)\n" if a.placeholder else
         "episodes: " + " ".join(a.episodes) + "\nfits: " + " ".join(a.fits) + "\n"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
