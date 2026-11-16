"""Compact per-model summary for the paper from the read-only Slurm outputs on Beacon.

    python extract_summary.py <dir> <verification-job> <triage-job> <library-calls-job>

<dir> holds permodel-PE-<v>.json and permodel-S-<v>.json (frozen verification job <v>),
triage-<t>.json (reply/failure triage job <t>) and pzeros-<z>.json (P library calls that ran,
job <z>), copied from Beacon with matching SHA-256. No grade is recomputed here; counts are
copied, sums are formed and checked. Writes study_summary.min.json next to this script and
prints its SHA-256 for make_results.py.

A model's P fields are filled once its track P is complete, its E fields once track E is;
"complete" means both, as in the first version of 2026-11-12.
"""
import hashlib
import json
import sys
from pathlib import Path

CORE = ["qwen-coder-32b", "mistral-small-24b", "granite-8b", "olmo3-7b", "phi4-mini"]
NOCODE = ("fail_no_fence", "fail_truncated_no_code", "fail_other_fence_only", "fail_unclosed_think")


def main(src_dir, v, t, z):
    d = Path(src_dir)
    src = {"permodel_pe": d / f"permodel-PE-{v}.json", "permodel_s": d / f"permodel-S-{v}.json",
           "triage": d / f"triage-{t}.json", "pzeros": d / f"pzeros-{z}.json"}
    sha = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in src.items()}
    pe = json.loads(src["permodel_pe"].read_text(encoding="utf-8"))["models"]
    s = json.loads(src["permodel_s"].read_text(encoding="utf-8"))["models"]
    tri = json.loads(src["triage"].read_text(encoding="utf-8"))["counts"]
    pz = json.loads(src["pzeros"].read_text(encoding="utf-8"))
    models = {}
    for m, val in sorted(pe.items()):
        p_done, e_done = "summary" in val["P"], "summary" in val["E"]
        row = dict(core=m in CORE, complete=p_done and e_done, P_complete=p_done, E_complete=e_done)
        if p_done:
            groups = {(g["method"], g["library"]): g for g in val["P"]["summary"]["groups"]}
            row["P"] = {f"{meth}|{int(lib)}": [g["valid"], g["episodes"]] for (meth, lib), g in groups.items()}
            row["P_contrasts"] = {f'{c["left"][0]}{int(c["left"][1])}-{c["right"][0]}{int(c["right"][1])}':
                                  round(c["effect"], 6) for c in val["P"]["summary"]["contrasts"]}
            row["P_accepted_wrong"] = sum(g["accepted_wrong"] for g in groups.values())
            row["P_library_calls_ran"] = pz.get(f"{m}|True|libcall", {}).get("executed_programs", 0)
            row["P_valid_with_library_used"] = sum(c.get("valid_and_library_used", 0)
                                                   for k, c in tri.items() if k.startswith(f"P|{m}|"))
        if e_done:
            cond = val["E"]["summary"]["conditions"]
            row["E"] = {lib: dict(
                episodes=cond[lib]["episodes"], exact=cond[lib]["valid"],
                sb_exact=round(cond[lib]["source_balanced_exact"], 6),
                sb_1pct=round(cond[lib]["source_balanced_relative_1pct"], 6),
                no_program=sum(tri[f"E|{m}|{lib}"].get(k, 0) for k in NOCODE),
                valid_with_library_used=tri[f"E|{m}|{lib}"].get("valid_and_library_used", 0))
                for lib in ("False", "True")}
            assert all(row["E"][lib]["episodes"] == tri[f"E|{m}|{lib}"]["episodes"] for lib in row["E"])
        if "summary" in s.get(m, {}):
            row["S"] = {f'{g["method"]}|{int(g["library"])}': [g["valid"], g["episodes"]]
                        for g in s[m]["summary"]["groups"]}
        models[m] = row
    out = dict(sources={k: dict(file=p.name, sha256=sha[k]) for k, p in src.items()},
               jobs=dict(verification=int(v), triage=int(t), library_calls=int(z)),
               note="Per-model description only; the frozen protocol's pooled analysis waits for all 20 models.",
               models=models)
    raw = json.dumps(out, sort_keys=True, separators=(",", ":")).encode()
    (Path(__file__).parent / "study_summary.min.json").write_bytes(raw)
    print(hashlib.sha256(raw).hexdigest(), len(raw), "bytes;",
          sum(r["P_complete"] for r in models.values()), "with P,",
          sum(r["E_complete"] for r in models.values()), "with E,",
          sum("S" in r for r in models.values()), "with S")


if __name__ == "__main__":
    main(*sys.argv[1:])
