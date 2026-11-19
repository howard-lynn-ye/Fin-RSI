"""P-confirm analysis: valid rates per arm with bootstrap intervals, the primary test, routing.

    python scripts/analyze_p_confirm.py <runs/main> <out.json> [--models a,b,...]

Design: docs/P_CONFIRM_STUDY_DESIGN_20261119.md, fixed before any confirmatory run. A model is
included when all three units (c0, route, lib) have a complete run; the first complete run by
job number is used. Every episode stays in the denominator.

Populations: primary (the eight models of at least 14B named in the design), small (the other
twelve), all. Task groups: catalogued (positions() has the rule), other (it does not).

Intervals: 95% percentile intervals from a bootstrap that resamples rules (a rule's two tasks
and both seeds stay together) and models, 10,000 draws, seed 20261119. Primary test: exact
sign-flip over the rules of the catalogued tasks on C2 minus C0 in the primary population.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

UNITS = ("c0", "route", "lib")
ARMS = ("C0", "C1", "C2", "C3", "C4")
CONTRASTS = (("C2", "C0"), ("C2", "C1"), ("C2", "C4"), ("C3", "C2"), ("C1", "C0"), ("C4", "C0"))
PRIMARY = ("C2", "C0", "primary", "catalogued")
NO_HARM = ("C2", "C1", "primary", "other")
NO_HARM_MARGIN = -10.0          # percentage points: the lower bound must stay above it
DRAWS, SEED = 10_000, 20261119
POSITIONS = r"(?<![\w.])positions\s*\("
PROPERTY_CHECKS = ("takes_positions", "position_scale", "warmup_rows_zero")


def first_complete(unit_dir: Path):
    for terminal in sorted(unit_dir.glob("job-*/*.jsonl.terminal.json"),
                           key=lambda p: int(p.parent.name.split("-")[1])):
        if json.loads(terminal.read_text(encoding="utf-8")).get("status") == "complete":
            rows = Path(str(terminal)[:-len(".terminal.json")]).read_text(encoding="utf-8")
            return terminal.parent.name, [json.loads(line) for line in rows.splitlines()]
    return None, None


def arm_of(method: str) -> str:
    m = re.fullmatch(r"confirm_(C[0-4])(_cause)?", method)
    if not m:
        raise ValueError(f"not a P-confirm method: {method}")
    return m.group(1)


def pct(valid, n):
    return round(100 * valid / n, 1) if n else None


def interval(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [round(float(lo), 1), round(float(hi), 1)]


def sign_flip(per_rule):
    """Exact two- and one-sided p for the mean of per-rule differences under sign flips."""
    d = np.asarray(per_rule, dtype=float)
    observed = d.mean()
    signs = np.array(list(itertools.product((1, -1), repeat=len(d))))
    means = (signs * d).mean(axis=1)
    return dict(rules=len(d), mean_difference_points=round(100 * float(observed), 1),
                p_two_sided=float((np.abs(means) >= abs(observed) - 1e-12).mean()),
                p_one_sided=float((means >= observed - 1e-12).mean()))


def analyze(root: Path, models=None, primary_models=None, draws=DRAWS):
    from fin_skills.algorithms import routing
    from fin_skills.rsi.pconfirm import PRIMARY_MODELS
    primary_models = tuple(primary_models or PRIMARY_MODELS)
    rows, runs, missing = {}, {}, []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        model = model_dir.name
        if models and model not in models:
            continue
        got, collected = {}, []
        for unit in UNITS:
            run, unit_rows = first_complete(model_dir / unit)
            if unit_rows is None:
                break
            got[unit] = run
            collected.extend(unit_rows)
        else:
            rows[model], runs[model] = collected, got
            continue
        missing.append(model)
    if not rows:
        raise ValueError("no model has all units complete")
    names = sorted(rows)
    tasks = sorted({(r["task_id"], r["task_algorithm"]) for rs in rows.values() for r in rs})
    task_ids = [t for t, _ in tasks]
    rule_of = dict(tasks)
    catalogued = {t: rule in routing.RULES for t, rule in tasks}
    feedback = {("cause" if r["method"].endswith("_cause") else "check")
                for rs in rows.values() for r in rs}
    if len(feedback) != 1:
        raise ValueError(f"episodes mix feedback modes: {sorted(feedback)}")
    # valid[arm][model, task] and n[arm][model, task] over seeds
    valid = {a: np.zeros((len(names), len(task_ids))) for a in ARMS}
    count = {a: np.zeros((len(names), len(task_ids))) for a in ARMS}
    routing_counts = {m: collections.Counter() for m in names}
    route_ok = np.zeros((len(names), len(task_ids)))
    retrieval = {}
    diag = {a: collections.Counter() for a in ARMS}
    failed = {a: collections.Counter() for a in ARMS}
    for i, model in enumerate(names):
        for r in rows[model]:
            arm, j = arm_of(r["method"]), task_ids.index(r["task_id"])
            valid[arm][i, j] += bool(r["valid"])
            count[arm][i, j] += 1
            steps = r["steps"]
            last = steps[-1] if steps else None
            named = (r.get("extra") or {}).get("routed")
            d = diag[arm]
            d["episodes"] += 1
            d["accepted_wrong"] += bool(r.get("accepted")) and not r["valid"]
            d["no_program_ran"] += not any(s["executed"] for s in steps)
            called = bool(last and re.search(POSITIONS, last.get("code") or "", re.M))
            d["last_calls_positions"] += called
            d["last_calls_positions_valid"] += called and bool(r["valid"])
            caught = [k for k, s in enumerate(steps) if s["executed"] and any(
                (s.get("checks") or {}).get(c) is False for c in PROPERTY_CHECKS)]
            if caught:
                d["flagged"] += 1
                d["flagged_with_another_attempt"] += caught[0] < len(steps) - 1
                d["flagged_then_valid"] += caught[0] < len(steps) - 1 and bool(r["valid"])
            for s in steps:
                if s["executed"]:
                    failed[arm]["programs_ran"] += 1
                    for name, ok in (s.get("checks") or {}).items():
                        failed[arm][name] += not ok
            if arm == "C2":
                rule, has = r["task_algorithm"], catalogued[r["task_id"]]
                how = (("right" if named == rule else "none" if named is None else "wrong")
                       if has else ("abstained" if named is None else "false_route"))
                routing_counts[model][how] += 1
                routing_counts[model][how + "_valid"] += bool(r["valid"])
                route_ok[i, j] += how in ("right", "abstained")
            if arm == "C4":
                retrieval[r["task_id"]] = named
    expected = {a: count[a] for a in ARMS}
    seeds = int(count["C0"].max())
    for a in ARMS:
        if not (count[a] == seeds).all():
            raise ValueError(f"arm {a}: episodes per model and task are not all {seeds}")
    populations = {"primary": [i for i, m in enumerate(names) if m in primary_models],
                   "small": [i for i, m in enumerate(names) if m not in primary_models],
                   "all": list(range(len(names)))}
    groups = {"catalogued": [j for j, t in enumerate(task_ids) if catalogued[t]],
              "other": [j for j, t in enumerate(task_ids) if not catalogued[t]],
              "all": list(range(len(task_ids)))}
    rng = np.random.default_rng(SEED)

    def clusters(task_index):
        by_rule = collections.defaultdict(list)
        for j in task_index:
            by_rule[rule_of[task_ids[j]]].append(j)
        return [by_rule[k] for k in sorted(by_rule)]

    def resample(model_index, task_index):
        """Weights for one bootstrap draw: models and rules with replacement."""
        cl = clusters(task_index)
        wm = np.bincount(rng.integers(0, len(model_index), len(model_index)),
                         minlength=len(model_index))
        wc = np.bincount(rng.integers(0, len(cl), len(cl)), minlength=len(cl))
        wt = np.zeros(len(task_ids))
        for weight, members in zip(wc, cl):
            wt[members] = weight
        return wm[:, None] * wt[None, task_index]

    def rate_block(model_index, task_index):
        out = {}
        for a in ARMS:
            v = valid[a][np.ix_(model_index, task_index)]
            n = count[a][np.ix_(model_index, task_index)]
            boots = []
            for _ in range(draws):
                w = resample(model_index, task_index)
                total = (w * n).sum()
                if total:
                    boots.append(100 * (w * v).sum() / total)
            out[a] = dict(valid=int(v.sum()), n=int(n.sum()), valid_pct=pct(v.sum(), n.sum()),
                          ci95=interval(boots) if boots else None)
        return out

    def contrast_block(a, b, model_index, task_index):
        va, vb = (valid[x][np.ix_(model_index, task_index)] for x in (a, b))
        n = count[a][np.ix_(model_index, task_index)]
        boots = []
        for _ in range(draws):
            w = resample(model_index, task_index)
            total = (w * n).sum()
            if total:
                boots.append(100 * (w * (va - vb)).sum() / total)
        per_model = (va - vb).sum(axis=1)
        return dict(contrast=f"{a}-{b}", points=round(100 * float((va - vb).sum() / n.sum()), 1),
                    ci95=interval(boots) if boots else None,
                    models_better=int((per_model > 0).sum()), models_worse=int((per_model < 0).sum()),
                    models_same=int((per_model == 0).sum()))

    out = dict(models=names, missing=missing, runs=runs, feedback=feedback.pop(),
               tasks={t: dict(rule=rule_of[t], catalogued=catalogued[t]) for t in task_ids},
               seeds_per_task=seeds, primary_models=[m for m in names if m in primary_models],
               bootstrap=dict(draws=draws, seed=SEED, resampled="rules and models"),
               rates={}, contrasts={})
    for pop, model_index in populations.items():
        if not model_index:
            continue
        for group, task_index in groups.items():
            if not task_index:
                continue
            key = f"{pop}|{group}"
            out["rates"][key] = rate_block(model_index, task_index)
            out["contrasts"][key] = [contrast_block(a, b, model_index, task_index)
                                     for a, b in CONTRASTS]
    # Primary test: sign flips over the rules of the catalogued tasks, primary population.
    a, b, pop, group = PRIMARY
    if populations[pop] and groups[group]:
        d = []
        for members in clusters(groups[group]):
            ix = np.ix_(populations[pop], members)
            d.append(float((valid[a][ix] - valid[b][ix]).sum() / count[a][ix].sum()))
        primary = next(c for c in out["contrasts"][f"{pop}|{group}"] if c["contrast"] == f"{a}-{b}")
        out["primary"] = dict(hypothesis="C2 valid more often than C0 on catalogued tasks, primary "
                                         "population", **primary, sign_flip=sign_flip(d),
                              supported=bool(primary["ci95"] and primary["ci95"][0] > 0))
    a, b, pop, group = NO_HARM
    if populations[pop] and groups[group]:
        harm = next(c for c in out["contrasts"][f"{pop}|{group}"] if c["contrast"] == f"{a}-{b}")
        out["no_harm"] = dict(margin_points=NO_HARM_MARGIN, **harm,
                              within_margin=bool(harm["ci95"] and harm["ci95"][0] > NO_HARM_MARGIN))
    # Routing: what C2's model named, and how often that was right (the id, or none).
    out["routing"] = {}
    for pop, model_index in populations.items():
        if not model_index:
            continue
        total = sum((routing_counts[names[i]] for i in model_index), collections.Counter())
        block = dict(counts=dict(total))
        for group in ("catalogued", "other", "all"):
            task_index = groups[group]
            if not task_index:
                continue
            ok = route_ok[np.ix_(model_index, task_index)]
            n = count["C2"][np.ix_(model_index, task_index)]
            boots = []
            for _ in range(draws):
                w = resample(model_index, task_index)
                if (w * n).sum():
                    boots.append(100 * (w * ok).sum() / (w * n).sum())
            block[group] = dict(right=int(ok.sum()), n=int(n.sum()), right_pct=pct(ok.sum(), n.sum()),
                                ci95=interval(boots) if boots else None)
        out["routing"][pop] = block
    out["routing_by_model"] = {m: dict(c) for m, c in routing_counts.items()}
    out["retrieval"] = dict(
        named=retrieval,
        right_catalogued=sum(retrieval.get(t) == rule_of[t] for t in task_ids if catalogued[t]),
        catalogued=sum(catalogued.values()),
        note="BM25 never abstains, so it is wrong on every task whose rule the library lacks")
    out["per_model"] = {m: {a: dict(valid=int(valid[a][i].sum()), n=int(count[a][i].sum()),
                                    catalogued_valid=int(valid[a][i, groups["catalogued"]].sum()),
                                    other_valid=int(valid[a][i, groups["other"]].sum()))
                            for a in ARMS} for i, m in enumerate(names)}
    out["per_task"] = {t: {a: dict(valid=int(valid[a][:, j].sum()), n=int(count[a][:, j].sum()),
                                   primary_valid=int(valid[a][populations["primary"], j].sum()))
                           for a in ARMS} for j, t in enumerate(task_ids)}
    out["diagnostics"] = {a: dict(c) for a, c in diag.items()}
    out["failed_checks"] = {a: dict(c) for a, c in failed.items()}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--models", default="")
    args = parser.parse_args(argv)
    out = analyze(args.root, set(filter(None, args.models.split(","))) or None)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    print(f"models {len(out['models'])} (primary {len(out['primary_models'])}); missing "
          f"{out['missing']}; feedback {out['feedback']}; tasks {len(out['tasks'])}")
    for key, block in out["rates"].items():
        print(f"{key:20} " + "  ".join(f"{a} {v['valid_pct']}% {v['ci95']}" for a, v in block.items()))
    for key, block in out["contrasts"].items():
        print(f"{key:20} " + "  ".join(f"{c['contrast']} {c['points']:+} {c['ci95']}" for c in block))
    for name in ("primary", "no_harm"):
        if name in out:
            print(name, json.dumps(out[name]))
    for pop, block in out["routing"].items():
        print("routing", pop, block)
    print("retrieval right on catalogued tasks:", out["retrieval"]["right_catalogued"], "/",
          out["retrieval"]["catalogued"])
    return out


if __name__ == "__main__":
    main()
