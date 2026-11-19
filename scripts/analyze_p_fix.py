"""P-fix analysis (exploratory, descriptive): docs/P_FIX_STUDY_DESIGN_20261118.md.

    python scripts/analyze_p_fix.py <runs/main> <out.json> [--models a,b,...]
        [--fix-root <runs-v2/main>] [--doc-root <runs-v3/main>] [--route-root <runs/main>]

For each model, the first complete run of each unit (lowest Slurm job id with a complete terminal
record) is read; every refusal, context stop, truncated reply and failed program stays in the
denominator. Reports valid episodes per arm and model, the three primary contrasts per model and
pooled (by algorithm family and variant too), the library funnel, library-call errors by group,
and a post-hoc scan of every program for file, network or process access (the no-Landlock
deviation on Zaratan). Nothing here is a significance test.

Amendment 2 adds arm F3 (documented interface, unit check_doc), read from --doc-root. Its
contrasts are reported apart from the declared ones, over the models whose F3 run is complete;
models without one keep their F0-F2 results. The split by model size (total parameters >= 14B
or smaller) is post hoc and descriptive.

Amendment 3 (v4) reruns every arm in one root with the declared-property checks and adds arm F4
(unit check_route): the model is first asked which catalogue rule computes the task's rule
(fin_skills.algorithms.routing). Pass that root as root, --doc-root and --route-root. The report
adds what the router named per episode and how often each visible check failed.
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import re
from pathlib import Path

ARMS = {"check_F0": "F0", "check_F1": "F1", "check_F2": "F2", "check_F3": "F3",
        "check_F4": "F4"}
UNITS = {"check_F0": ("check_F0",), "check_fix": ("check_F1", "check_F2")}
OPTIONAL_UNITS = {"check_doc": ("check_F3",),      # Amendment 2
                  "check_route": ("check_F4",)}    # Amendment 3
CONTRASTS = (("F1", "F0"), ("F2", "F0"), ("F2", "F1"))
EXPLORATORY_CONTRASTS = (("F3", "F1"), ("F3", "F2"), ("F3", "F0"),
                         ("F4", "F1"), ("F4", "F2"), ("F4", "F0"))
LIBRARY_ARMS = ("F1", "F2", "F3", "F4")
# Total parameters in billions: the number in the model's name (qwen3-30b-a3b counts its 30B
# total, not its 3B active), and phi4-mini's 3.8B from its model card. Post-hoc split at 14B.
SIZE_OVERRIDES = {"phi4-mini": 3.8}
LARGE_B = 14
EPISODES_PER_ARM = 18
SCAN = {
    "file": r"\bopen\s*\(|\bpathlib\b|\bPath\s*\(|\bos\.(?:listdir|walk|scandir|remove|unlink|"
            r"rename|makedirs|mkdir|chdir|path)\b|\bshutil\b|\bglob\b|\.read_(?:csv|parquet|"
            r"pickle|json|excel)\s*\(|\bnp\.(?:load|save|fromfile)\b|\.to_(?:csv|pickle|parquet)\(",
    "network": r"\bsocket\b|\brequests\b|\burllib\b|\bhttp\.client\b|\bhttpx\b|\baiohttp\b",
    "process": r"^\s*import\s+(?:subprocess|multiprocessing|threading|signal)\b|"
               r"^\s*from\s+(?:subprocess|multiprocessing|threading|signal)\s+import\b|"
               r"\bsubprocess\.|\bos\.(?:system|popen|exec\w*|spawn\w*|fork|kill)\b|"
               r"\bmultiprocessing\.|\bthreading\.|\bsignal\.(?:signal|alarm|SIG\w+)",
    "introspection": r"\b__import__\b|\bimportlib\b|\bsys\.modules\b|\binspect\b|\bglobals\s*\(|"
                     r"\b__builtins__\b|\beval\s*\(|\bexec\s*\(|\bcompile\s*\(",
    "private_package": r"fin_skills\.(?:tasks|rsi|agents)\b",
}
# Library functions a program calls, read from its text: the worker's trace counts positions(),
# run() and rolling() (library_calls) but not signal_positions(), and stores no trace in the row.
USES = {"imports_fin_skills": r"^\s*(?:from|import)\s+fin_skills\b",
        "positions": r"(?<![\w.])positions\s*\(", "signal_positions": r"\bsignal_positions\s*\(",
        "adjust_prices": r"\badjust_prices\s*\(", "run": r"(?<![\w.])run\s*\("}
# A program error that is really the cluster's: a runtime file the filesystem failed to return
# (Zaratan scratch returned ENOENT for venv files under load in two jobs on 2026-11-18).
INFRASTRUCTURE = r"No such file or directory: '/scratch/|site-packages/[^']*' not found"


def first_complete(unit_dir: Path):
    for terminal in sorted(unit_dir.glob("job-*/*.jsonl.terminal.json"),
                           key=lambda p: int(p.parent.name.split("-")[1])):
        if json.loads(terminal.read_text(encoding="utf-8")).get("status") == "complete":
            rows = Path(str(terminal)[:-len(".terminal.json")]).read_text(encoding="utf-8")
            return terminal.parent.name, [json.loads(line) for line in rows.splitlines()]
    return None, None


def error_group(error: str) -> str:
    lines = [line for line in (error or "").strip().splitlines() if line.strip()]
    head = lines[0] if lines else ""
    if "unknown algorithm" in head:
        return "algorithm id not in the catalogue"
    if "not a function" in head:
        return "own function passed as the algorithm"
    if "no states= argument" in head or "not returns or states" in head:
        return "states passed to positions()"
    if "closes, not returns" in head or "prices must be positive" in head:
        return "returns passed as prices"
    if "portfolio algorithms only" in head or "signal algorithms only" in head:
        return "argument that does not apply to the algorithm"
    if "look like event ratios" in head or "look cumulative" in head:
        return "split factors in the other convention"
    if "cannot tell from the data whether split_factors" in head:
        return "split factor convention not shown by the data"
    if "1-D array" in head:
        return "run() given a table"
    if re.search(r"requires >= \d+ observations|made no decision on any", head):
        return "too few rows (called on a slice)"
    if head.startswith("TypeError") and ("argument" in head or "parameter" in head):
        return "wrong arguments"
    if head.startswith(("KeyError", "ValueError")) and "parameter" in head:
        return "unknown or invalid parameter"
    return "other: " + re.sub(r"0x[0-9a-f]+", "0x?", head)[:100]


# The library's result is final (lagged, filled, scaled); dividing by the number of assets again
# scales it twice. Read from program text (a line dividing by the asset count that does not pass
# scale= to the library), for accepted-but-wrong episodes whose last program called the library.
RESCALE = r"/\s*(?:len\(\s*\w+(?:\.columns)?\s*\)|num_assets|n_assets|n_cols|ncols|n\b|4\b)"


def rate(valid, n):
    return round(100 * valid / n, 1) if n else None


def size_b(model: str) -> float | None:
    """Total parameters in billions, or None when the name carries none (listed as unsized)."""
    if model in SIZE_OVERRIDES:
        return SIZE_OVERRIDES[model]
    m = re.search(r"(\d+(?:\.\d+)?)b(?![a-z])", model)
    return float(m.group(1)) if m else None


def contrast(per_model, a, b):
    diffs = {m: c[a]["valid"] - c[b]["valid"] for m, c in per_model.items()
             if c[a]["n"] and c[b]["n"]}
    k = len(diffs)
    return dict(
        contrast=f"{a}-{b}", models=k,
        pooled_pct_points=round(100 * sum(diffs.values()) / (k * EPISODES_PER_ARM), 1) if k else None,
        better=sum(d > 0 for d in diffs.values()), worse=sum(d < 0 for d in diffs.values()),
        same=sum(d == 0 for d in diffs.values()), per_model=diffs)


def pool(per_model, arm):
    rows = [m[arm] for m in per_model.values() if m[arm]["n"]]
    n, v = sum(c["n"] for c in rows), sum(c["valid"] for c in rows)
    return dict(n=n, valid=v, valid_pct=rate(v, n), models=len(rows),
                first_valid=sum(c["first_valid"] for c in rows),
                accepted_wrong=sum(c["accepted_wrong"] for c in rows),
                models_none_valid=sum(c["valid"] == 0 for c in rows))


def analyze(root: Path, models=None, fix_root: Path | None = None, doc_root: Path | None = None,
            route_root: Path | None = None):
    """fix_root: read check_fix from another runs/main (a later library version with identical
    prompts); check_F0 never imports the library and is read from root. doc_root, route_root:
    read check_doc (F3) and check_route (F4) from their runs/main; a model without one keeps
    its other arms."""
    optional = {"check_doc": doc_root, "check_route": route_root}
    out = dict(models={}, runs={}, missing=[], fix_root=str(fix_root) if fix_root else None,
               doc_root=str(doc_root) if doc_root else None, missing_doc=[],
               route_root=str(route_root) if route_root else None, missing_route=[])
    rows_by = collections.defaultdict(list)
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        model = model_dir.name
        if models and model not in models:
            continue
        got = {}
        for unit, methods in UNITS.items():
            base = fix_root / model if fix_root is not None and unit == "check_fix" else model_dir
            run, rows = first_complete(base / unit)
            if rows is None:
                break
            got[unit] = run
            for r in rows:
                if r["method"] in methods:
                    rows_by[model].append(r)
        if len(got) != len(UNITS):
            out["missing"].append(model)
            rows_by.pop(model, None)
            continue
        for unit, methods in OPTIONAL_UNITS.items():
            if optional[unit] is None:
                continue
            run, rows = first_complete(optional[unit] / model / unit)
            if rows is None:
                out["missing_" + unit.split("_")[1]].append(model)
            else:
                got[unit] = run
                rows_by[model].extend(r for r in rows if r["method"] in methods)
        out["runs"][model] = got
    per_model, families, variants = {}, collections.defaultdict(collections.Counter), \
        collections.defaultdict(collections.Counter)
    funnel = {a: collections.Counter() for a in LIBRARY_ARMS}
    errors = {a: collections.Counter() for a in LIBRARY_ARMS}
    scan = collections.Counter()
    flagged, infrastructure = [], []
    programs = {a: collections.Counter() for a in ARMS.values()}
    failed_checks = {a: collections.Counter() for a in ARMS.values()}
    scan_not_python = 0
    routing = collections.Counter()
    routing_by_model = {}
    for model, rows in sorted(rows_by.items()):
        counts = {a: dict(n=0, valid=0, first_valid=0, accepted_wrong=0, calling=0, calling_ran=0)
                  for a in ARMS.values()}
        for r in rows:
            arm = ARMS[r["method"]]
            c = counts[arm]
            c["n"] += 1
            c["valid"] += bool(r["valid"])
            c["first_valid"] += bool(r["first_valid"])
            c["accepted_wrong"] += bool(r["accepted"]) and not r["valid"]
            fam, var = r["task_algorithm"], r["task_id"].rsplit("_", 1)[-1]
            families[(fam, arm)]["n"] += 1
            families[(fam, arm)]["valid"] += bool(r["valid"])
            variants[(var, arm)]["n"] += 1
            variants[(var, arm)]["valid"] += bool(r["valid"])
            steps = r["steps"]
            if arm == "F4":
                # What the model named when asked which catalogue rule computes the task's rule.
                named = (r.get("extra") or {}).get("routed")
                how = "right" if named == fam else "none" if named is None else "wrong"
                for k in (how, how + "_valid" if r["valid"] else None):
                    if k:
                        routing[k] += 1
                        routing_by_model.setdefault(model, collections.Counter())[k] += 1
            for i, s in enumerate(steps):
                p = programs[arm]
                if s["executed"]:
                    failed_checks[arm]["programs_ran"] += 1
                    for name, ok in (s.get("checks") or {}).items():
                        failed_checks[arm][name] += not ok
                p["replies"] += 1
                p["programs"] += bool(s.get("code"))
                p["executed"] += bool(s["executed"])
                calls = s["library_calls"] > 0
                c["calling"] += calls
                c["calling_ran"] += calls and bool(s["executed"])
                if arm != "F0":
                    p["calling"] += calls
                    p["calling_ran"] += calls and bool(s["executed"])
                    if calls and not s["executed"]:
                        errors[arm][error_group(s.get("error"))] += 1
                if re.search(INFRASTRUCTURE, s.get("error") or ""):
                    infrastructure.append(dict(model=model, arm=arm, task=r["task_id"],
                                               seed=r["seed"], attempt=i + 1))
                code = s.get("code") or ""
                if re.search(USES["imports_fin_skills"], code, re.M):
                    for k, rx in USES.items():
                        p["uses_" + k] += bool(re.search(rx, code, re.M))
                        if i == len(steps) - 1:
                            p["last_uses_" + k] += bool(re.search(rx, code, re.M))
                            p["last_uses_" + k + "_valid"] += bool(re.search(rx, code, re.M)) and bool(r["valid"])
                # Full-line comments are not code: models often leave commented-out example I/O.
                # Text that does not parse is not a program either (a reply with no code, for
                # example a model's reasoning): the worker's compile() raises before any
                # statement runs, so its words are counted, not scanned.
                live = "\n".join(x for x in code.splitlines() if not x.lstrip().startswith("#"))
                try:
                    ast.parse(code)
                except (SyntaxError, ValueError):
                    scan_not_python += 1
                    live = ""
                hits = [k for k, rx in SCAN.items() if re.search(rx, live, re.M)]
                for k in hits:
                    scan[k] += 1
                if hits:
                    flagged.append(dict(model=model, arm=arm, task=r["task_id"], seed=r["seed"],
                                        attempt=i + 1, executed=bool(s["executed"]),
                                        valid_episode=bool(r["valid"]), matched=hits))
            if arm != "F0":
                last = steps[-1] if steps else None
                f = funnel[arm]
                f["episodes"] += 1
                f["library_used_any"] += bool(r.get("library_used"))
                if last and last["library_calls"] > 0:
                    f["last_calls_library"] += 1
                    f["last_calls_library_ran"] += bool(last["executed"])
                    f["last_calls_library_passed_checks"] += bool(last["accepted"])
                    f["last_calls_library_valid"] += bool(r["valid"])
                    if last["accepted"] and not r["valid"]:
                        f["last_calls_library_accepted_wrong"] += 1
                        f["last_calls_library_accepted_wrong_rescaled"] += any(
                            re.search(RESCALE, x) and "scale=" not in x
                            for x in (last.get("code") or "").splitlines())
                elif last:
                    f["last_no_library"] += 1
                    f["last_no_library_valid"] += bool(r["valid"])
        expected = {a: EPISODES_PER_ARM for a in ("F0", "F1", "F2")}
        if doc_root is not None and model not in out["missing_doc"]:
            expected["F3"] = EPISODES_PER_ARM
        if route_root is not None and model not in out["missing_route"]:
            expected["F4"] = EPISODES_PER_ARM
        got_n = {a: c["n"] for a, c in counts.items() if c["n"]}
        if got_n != expected:
            raise ValueError(f"{model}: episodes per arm {got_n}, expected {expected}")
        per_model[model] = counts
    arms = [a for a in ARMS.values() if any(m[a]["n"] for m in per_model.values())]
    pooled = {arm: pool(per_model, arm) for arm in arms}
    contrasts = [contrast(per_model, a, b) for a, b in CONTRASTS]
    exploratory = [contrast(per_model, a, b) for a, b in EXPLORATORY_CONTRASTS if a in arms]
    by_size = {}
    sizes = {m: size_b(m) for m in per_model}
    for label, keep in ((f"ge{LARGE_B}b", lambda m: sizes[m] is not None and sizes[m] >= LARGE_B),
                        (f"lt{LARGE_B}b", lambda m: sizes[m] is not None and sizes[m] < LARGE_B)):
        group = {m: c for m, c in per_model.items() if keep(m)}
        by_size[label] = dict(
            models=sorted(group), pooled={arm: pool(group, arm) for arm in arms},
            contrasts=[contrast(group, a, b) for a, b in CONTRASTS],
            exploratory_contrasts=[contrast(group, a, b) for a, b in EXPLORATORY_CONTRASTS
                                   if a in arms],
            routing=dict(sum((routing_by_model.get(m, collections.Counter()) for m in group),
                             collections.Counter())))
    out.update(
        per_model=per_model, pooled=pooled, contrasts=contrasts,
        exploratory_contrasts=exploratory,
        sizes_b=sizes, unsized=sorted(m for m, b in sizes.items() if b is None), by_size=by_size,
        by_family={f"{f}|{a}": dict(c, valid_pct=rate(c["valid"], c["n"]))
                   for (f, a), c in sorted(families.items())},
        by_variant={f"{v}|{a}": dict(c, valid_pct=rate(c["valid"], c["n"]))
                    for (v, a), c in sorted(variants.items())},
        funnel={a: dict(c) for a, c in funnel.items()},
        programs={a: dict(c) for a, c in programs.items()},
        library_errors={a: dict(c.most_common()) for a, c in errors.items()},
        failed_checks={a: dict(c) for a, c in failed_checks.items()},
        routing=dict(routing), routing_by_model={m: dict(c) for m, c in routing_by_model.items()},
        scan=dict(patterns=SCAN, counts=dict(scan), flagged=flagged,
                  not_python=scan_not_python),
        infrastructure_suspect=infrastructure)
    return out


def report(out):
    print(f"models complete: {len(out['per_model'])}; missing: {out['missing']}; "
          f"without F3: {out['missing_doc']}; without F4: {out['missing_route']}")
    arms = list(out["pooled"])
    print("model                    " + "".join(f"{a:>4}" for a in arms) + "   (valid of 18)")
    for m, c in out["per_model"].items():
        print(f"  {m:24}" + "".join(f"{c[a]['valid']:4}" if c[a]["n"] else "   -" for a in arms))
    for a, p in out["pooled"].items():
        print(f"pooled {a}: {p['valid']}/{p['n']} = {p['valid_pct']}%  ({p['models']} models; "
              f"first attempt {p['first_valid']})")
    for kind in ("contrasts", "exploratory_contrasts"):
        for c in out[kind]:
            if c["models"]:
                print(f"{c['contrast']}: {c['pooled_pct_points']:+} points over {c['models']} "
                      f"models; better {c['better']}, worse {c['worse']}, same {c['same']}"
                      + ("  [Amendment 2]" if kind != "contrasts" else ""))
    for label, g in out["by_size"].items():
        print(f"{label} ({len(g['models'])} models): " + ", ".join(
            f"{a} {p['valid']}/{p['n']}" for a, p in g["pooled"].items()))
    for a, f in out["funnel"].items():
        print(f"funnel {a}: {f}")
    for a, e in out["library_errors"].items():
        print(f"library calls that raised, {a}: {e}")
    for a, c in out["failed_checks"].items():
        print(f"checks failed by programs that ran, {a}: {c}")
    print(f"F4 routing (episodes; _valid: of those, valid): {out['routing']}")
    for label, g in out["by_size"].items():
        print(f"  {label}: {g['routing']}")
    print(f"programs: {out['programs']}")
    print(f"scan: {out['scan']['counts']}; flagged programs {len(out['scan']['flagged'])}")
    print(f"steps whose error is a runtime file the filesystem failed to return: "
          f"{len(out['infrastructure_suspect'])}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--models", default="")
    parser.add_argument("--fix-root", type=Path, default=None,
                        help="runs/main of a later library version, for check_fix only")
    parser.add_argument("--doc-root", type=Path, default=None,
                        help="runs/main holding check_doc (arm F3, Amendment 2)")
    parser.add_argument("--route-root", type=Path, default=None,
                        help="runs/main holding check_route (arm F4, Amendment 3)")
    args = parser.parse_args(argv)
    out = analyze(args.root, set(filter(None, args.models.split(","))) or None, args.fix_root,
                  args.doc_root, args.route_root)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    report(out)


if __name__ == "__main__":
    main()
