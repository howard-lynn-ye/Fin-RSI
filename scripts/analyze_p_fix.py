"""P-fix analysis (exploratory, descriptive): docs/P_FIX_STUDY_DESIGN_20261118.md.

    python scripts/analyze_p_fix.py <runs/main> <out.json> [--models a,b,...]

For each model, the first complete run of each unit (lowest Slurm job id with a complete terminal
record) is read; every refusal, context stop, truncated reply and failed program stays in the
denominator. Reports valid episodes per arm and model, the three primary contrasts per model and
pooled (by algorithm family and variant too), the library funnel, library-call errors by group,
and a post-hoc scan of every program for file, network or process access (the no-Landlock
deviation on Zaratan). Nothing here is a significance test.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

ARMS = {"check_F0": "F0", "check_F1": "F1", "check_F2": "F2"}
UNITS = {"check_F0": ("check_F0",), "check_fix": ("check_F1", "check_F2")}
CONTRASTS = (("F1", "F0"), ("F2", "F0"), ("F2", "F1"))
EPISODES_PER_ARM = 18
SCAN = {
    "file": r"\bopen\s*\(|\bpathlib\b|\bPath\s*\(|\bos\.(?:listdir|walk|scandir|remove|unlink|"
            r"rename|makedirs|mkdir|chdir|path)\b|\bshutil\b|\bglob\b|\.read_(?:csv|parquet|"
            r"pickle|json|excel)\s*\(|\bnp\.(?:load|save|fromfile)\b|\.to_(?:csv|pickle|parquet)\(",
    "network": r"\bsocket\b|\brequests\b|\burllib\b|\bhttp\.client\b|\bhttpx\b|\baiohttp\b",
    "process": r"\bsubprocess\b|\bos\.(?:system|popen|exec\w*|spawn\w*|fork|kill)\b|"
               r"\bmultiprocessing\b|\bimport threading\b|\bthreading\.|\bimport signal\b|"
               r"\bsignal\.(?:signal|alarm|SIG\w+)",
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
    if "signal_positions" in head or "callable" in head:
        return "own function passed as the algorithm"
    if "closes, not returns" in head or "prices must be positive" in head:
        return "returns passed as prices"
    if head.startswith("TypeError") and ("argument" in head or "parameter" in head):
        return "wrong arguments"
    if head.startswith(("KeyError", "ValueError")) and "parameter" in head:
        return "unknown or invalid parameter"
    return "other: " + re.sub(r"0x[0-9a-f]+", "0x?", head)[:100]


def rate(valid, n):
    return round(100 * valid / n, 1) if n else None


def analyze(root: Path, models=None):
    out = dict(models={}, runs={}, missing=[])
    rows_by = collections.defaultdict(list)
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        model = model_dir.name
        if models and model not in models:
            continue
        got = {}
        for unit, methods in UNITS.items():
            run, rows = first_complete(model_dir / unit)
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
        out["runs"][model] = got
    per_model, families, variants = {}, collections.defaultdict(collections.Counter), \
        collections.defaultdict(collections.Counter)
    funnel = {a: collections.Counter() for a in ("F1", "F2")}
    errors = {a: collections.Counter() for a in ("F1", "F2")}
    scan = collections.Counter()
    flagged, infrastructure = [], []
    programs = {a: collections.Counter() for a in ARMS.values()}
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
            for i, s in enumerate(steps):
                p = programs[arm]
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
                hits = [k for k, rx in SCAN.items() if re.search(rx, code)]
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
                elif last:
                    f["last_no_library"] += 1
                    f["last_no_library_valid"] += bool(r["valid"])
        if any(c["n"] != EPISODES_PER_ARM for c in counts.values()):
            raise ValueError(f"{model}: episodes per arm {[c['n'] for c in counts.values()]}")
        per_model[model] = counts
    pooled = {}
    for arm in ARMS.values():
        n = sum(m[arm]["n"] for m in per_model.values())
        v = sum(m[arm]["valid"] for m in per_model.values())
        fv = sum(m[arm]["first_valid"] for m in per_model.values())
        aw = sum(m[arm]["accepted_wrong"] for m in per_model.values())
        pooled[arm] = dict(n=n, valid=v, valid_pct=rate(v, n), first_valid=fv, accepted_wrong=aw,
                           models_none_valid=sum(m[arm]["valid"] == 0 for m in per_model.values()))
    contrasts = []
    for a, b in CONTRASTS:
        diffs = {m: c[a]["valid"] - c[b]["valid"] for m, c in per_model.items()}
        k = len(diffs)
        contrasts.append(dict(
            contrast=f"{a}-{b}", models=k,
            pooled_pct_points=round(100 * sum(diffs.values()) / (k * EPISODES_PER_ARM), 1) if k else None,
            better=sum(d > 0 for d in diffs.values()), worse=sum(d < 0 for d in diffs.values()),
            same=sum(d == 0 for d in diffs.values()), per_model=diffs))
    out.update(
        per_model=per_model, pooled=pooled, contrasts=contrasts,
        by_family={f"{f}|{a}": dict(c, valid_pct=rate(c["valid"], c["n"]))
                   for (f, a), c in sorted(families.items())},
        by_variant={f"{v}|{a}": dict(c, valid_pct=rate(c["valid"], c["n"]))
                    for (v, a), c in sorted(variants.items())},
        funnel={a: dict(c) for a, c in funnel.items()},
        programs={a: dict(c) for a, c in programs.items()},
        library_errors={a: dict(c.most_common()) for a, c in errors.items()},
        scan=dict(patterns=SCAN, counts=dict(scan), flagged=flagged),
        infrastructure_suspect=infrastructure)
    return out


def report(out):
    print(f"models complete: {len(out['per_model'])}; missing: {out['missing']}")
    print("model                      F0  F1  F2   (valid of 18)")
    for m, c in out["per_model"].items():
        print(f"  {m:24} {c['F0']['valid']:3} {c['F1']['valid']:3} {c['F2']['valid']:3}")
    for a, p in out["pooled"].items():
        print(f"pooled {a}: {p['valid']}/{p['n']} = {p['valid_pct']}%  (first attempt {p['first_valid']})")
    for c in out["contrasts"]:
        print(f"{c['contrast']}: {c['pooled_pct_points']:+} points; models better {c['better']}, "
              f"worse {c['worse']}, same {c['same']}")
    for a, f in out["funnel"].items():
        print(f"funnel {a}: {f}")
    for a, e in out["library_errors"].items():
        print(f"library calls that raised, {a}: {e}")
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
    args = parser.parse_args(argv)
    out = analyze(args.root, set(filter(None, args.models.split(","))) or None)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    report(out)


if __name__ == "__main__":
    main()
