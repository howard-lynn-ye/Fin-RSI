"""Results numbers and table of the short paper: the library study (P-fix) and the no-library parts
of the pre-registered study.

    python make_pfix_results.py [analysis.json]   # writes pfix_numbers.tex and results_table.tex

pfix_analysis.json is the output of scripts/analyze_p_fix.py (branch claude/pfix) on the Zaratan
P-fix runs: for each model, the first complete run of each unit. Its SHA-256 is pinned in
ANALYSIS_SHA256 once the runs are final; any other file only builds with --draft, and the PDF then
says DRAFT on every number. study_summary.min.json is the frozen study's per-model digest (see
make_results.py); only its no-library cells are read here: the feedback arms and the seven methods
without the library.
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
SUMMARY = HERE / "study_summary.min.json"
SUMMARY_SHA256 = "4918cd8be77a42925aad63540a69a38f9bc4febb982b2681e7765923783824a5"
ANALYSIS = HERE / "pfix_analysis.json"
ANALYSIS_SHA256 = "646228b09557d8d117d71b984018f8e8e43c077abfd731b1a0d99960cde8cb96"
CORE = ["qwen-coder-32b", "mistral-small-24b", "granite-8b", "olmo3-7b", "phi4-mini"]
EXTENSION = ["qwen-coder-14b", "qwen-coder-7b", "qwen-general-32b", "qwen-general-14b",
             "qwen-general-7b", "qwen3-30b-a3b", "qwen3-4b", "deepseek-r1-distill-14b",
             "phi4-14b", "mistral-nemo-12b", "mistral-7b", "falcon3-10b", "falcon3-7b",
             "granite-2b", "smollm3-3b"]
METHODS = ["self_debug", "check_feedback", "reflexion", "team", "evolve", "playbook"]
TAG = {"F0": "FZero", "F1": "FOne", "F2": "FTwo"}
# Fixed by the P-fix design (branch claude/pfix, commit bf782d72): fin_skills.rsi.pfix.interface_text()
# is 5,312 characters and lists 12 algorithms that positions() accepts; adopt.OUTPUT_CAP is 4,096
# tokens; 6 held-out tasks x 3 seeds per arm.
DESIGN = dict(FixInterfaceChars="5,312", FixInterfaceAlgorithms=12, FixOutputCap="4,096",
              FixEpisodesPerArm=18)
# How each library-error group of analyze_p_fix.py reads in a sentence.
PROSE = {"argument that does not apply to the algorithm": "an argument that does not apply to the algorithm",
         "too few rows (called on a slice)": r"a call on a slice of the rows",
         "states passed to positions()": r"states passed to \texttt{positions()}",
         "wrong arguments": "wrong arguments",
         "run() given a table": r"a table passed to \texttt{run()}",
         "algorithm id not in the catalogue": "an id not in the catalogue",
         "own function passed as the algorithm": "the model's own function passed as the algorithm",
         "split factors in the other convention": "split factors in the other convention",
         "returns passed as prices": "returns passed as prices"}
FAMILY = {"bollinger_reversion": "Bollinger", "rsi_reversion": "Rsi", "donchian_breakout": "Donchian"}


def tex(text):
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#"),
                 ("$", r"\$"), ("{", r"\{"), ("}", r"\}")):
        text = text.replace(a, b)
    return text


def pct(x, n):
    return f"{100 * x / n:.1f}"


def signed(x):
    return f"+{x:.1f}" if x > 0 else (f"$-${-x:.1f}" if x < 0 else "0.0")


def main(argv):
    draft = "--draft" in argv
    path = Path(next((a for a in argv if not a.startswith("--")), ANALYSIS))
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ANALYSIS_SHA256 and not draft:
        raise SystemExit(f"{path} is not the pinned P-fix analysis; use --draft for a draft build")
    fix = json.loads(raw)
    summary = SUMMARY.read_bytes()
    if hashlib.sha256(summary).hexdigest() != SUMMARY_SHA256:
        raise SystemExit("study_summary.min.json differs from the pinned digest")
    frozen = json.loads(summary)["models"]
    per = fix["per_model"]
    models = [m for m in CORE + EXTENSION if m in per]
    if not draft:
        assert models == CORE + EXTENSION and not fix["missing"], "P-fix must cover all 20 models"
    assert not fix["infrastructure_suspect"], "a program error names a missing runtime file"
    pooled, num = fix["pooled"], dict(DESIGN)
    num["FixModels"] = len(models)
    num["FixArm"] = f'{pooled["F0"]["n"]:,}'
    for arm, tag in TAG.items():
        p = pooled[arm]
        num[f"Fix{tag}Valid"] = p["valid"]
        num[f"Fix{tag}Pct"] = pct(p["valid"], p["n"])
        num[f"Fix{tag}First"] = p["first_valid"]
        num[f"Fix{tag}AcceptedWrong"] = p["accepted_wrong"]
        num[f"Fix{tag}NoneValid"] = p["models_none_valid"]
        best = max(models, key=lambda m: per[m][arm]["valid"])
        num[f"Fix{tag}Best"], num[f"Fix{tag}BestModel"] = per[best][arm]["valid"], best
    for c in fix["contrasts"]:
        a, b = c["contrast"].split("-")
        key = f"Fix{TAG[a]}Minus{TAG[b]}"
        num[key] = signed(c["pooled_pct_points"])
        num[key + "Better"], num[key + "Worse"], num[key + "Same"] = c["better"], c["worse"], c["same"]
    for arm in ("F1", "F2"):
        f, pr, tag = fix["funnel"][arm], fix["programs"][arm], TAG[arm]
        num[f"Fix{tag}UsedAny"] = f.get("library_used_any", 0)
        num[f"Fix{tag}LastCalls"] = f.get("last_calls_library", 0)
        num[f"Fix{tag}LastCallsRan"] = f.get("last_calls_library_ran", 0)
        num[f"Fix{tag}LastCallsChecked"] = f.get("last_calls_library_passed_checks", 0)
        num[f"Fix{tag}LastCallsValid"] = f.get("last_calls_library_valid", 0)
        num[f"Fix{tag}LastOwnValid"] = f.get("last_no_library_valid", 0)
        num[f"Fix{tag}LastOwn"] = f.get("last_no_library", 0)
        num[f"Fix{tag}LastCallsAcceptedWrong"] = f.get("last_calls_library_accepted_wrong", 0)
        num[f"Fix{tag}LastCallsRescaled"] = f.get("last_calls_library_accepted_wrong_rescaled", 0)
        num[f"Fix{tag}Calling"] = pr.get("calling", 0)
        num[f"Fix{tag}CallingRan"] = pr.get("calling_ran", 0)
        num[f"Fix{tag}Programs"] = f'{pr.get("programs", 0):,}'
        for use in ("imports_fin_skills", "positions", "signal_positions", "adjust_prices", "run"):
            ut = "".join(w.capitalize() for w in use.split("_"))
            num[f"Fix{tag}Uses{ut}"] = pr.get("uses_" + use, 0)
            num[f"Fix{tag}LastUses{ut}"] = pr.get("last_uses_" + use, 0)
            num[f"Fix{tag}LastUses{ut}Valid"] = pr.get("last_uses_" + use + "_valid", 0)
    errors = {}
    for arm in ("F1", "F2"):
        for k, v in fix["library_errors"][arm].items():
            errors[k] = errors.get(k, 0) + v
    top = sorted(errors.items(), key=lambda kv: -kv[1])
    num["FixLibErrors"] = sum(errors.values())
    for i, word in enumerate(("One", "Two", "Three", "Four")):
        name, count = top[i] if i < len(top) else ("none", 0)
        num[f"FixLibError{word}"], num[f"FixLibError{word}Count"] = PROSE.get(name, tex(name)), count
    for fam, tag in FAMILY.items():
        for arm, atag in TAG.items():
            cell = fix["by_family"].get(f"{fam}|{arm}")
            num[f"Fix{tag}{atag}"] = cell["valid_pct"] if cell else "--"
    for var in ("a", "b"):
        for arm, atag in TAG.items():
            cell = fix["by_variant"].get(f"{var}|{arm}")
            num[f"FixVar{var.upper()}{atag}"] = cell["valid_pct"] if cell else "--"
    num["FixScanFlagged"] = len(fix["scan"]["flagged"])
    num["FixScanFlaggedRan"] = sum(x["executed"] for x in fix["scan"]["flagged"])
    # The pre-registered study, no-library cells only: feedback arms and the seven methods.
    arms = {a: sum(frozen[m]["P"][f"{a}|0"][0] for m in CORE + EXTENSION)
            for a in ("repeat", "check", "explain")}
    n_arm = sum(frozen[m]["P"]["repeat|0"][1] for m in CORE + EXTENSION)
    num.update(FrozenRepeat=arms["repeat"], FrozenCheck=arms["check"], FrozenExplain=arms["explain"],
               FrozenArm=n_arm,
               FrozenCheckRepeat=signed(100 * (arms["check"] - arms["repeat"]) / n_arm),
               FrozenExplainCheck=signed(100 * (arms["explain"] - arms["check"]) / n_arm))
    s_models = [m for m in CORE if "S" in frozen[m]]
    s_n = sum(frozen[m]["S"]["no_improve|0"][1] for m in s_models)
    base = sum(frozen[m]["S"]["no_improve|0"][0] for m in s_models)
    num.update(SNoLibModels=len(s_models), SNoLibEpisodes=s_n, SNoLibNoImprove=base)
    for meth in METHODS:
        v = sum(frozen[m]["S"][f"{meth}|0"][0] for m in s_models)
        tag = "".join(w.capitalize() for w in meth.split("_"))
        num[f"SNoLib{tag}"] = v
        num[f"SNoLib{tag}Gain"] = signed(100 * (v - base) / s_n)
    # The prose states these directions; stop the build if the numbers no longer support them.
    if not draft:
        f2 = fix["funnel"]["F2"]
        assert pooled["F2"]["valid"] > pooled["F0"]["valid"], "F2 is not above F0"
        assert (f2["last_calls_library_valid"] / f2["last_calls_library"] >
                f2["last_no_library_valid"] / f2["last_no_library"]), "library calls not more often valid"
        assert 2 * fix["programs"]["F1"]["last_uses_signal_positions_valid"] > pooled["F1"]["valid"], \
            "most F1 valid episodes did not use signal_positions()"
        assert fix["programs"]["F1"]["uses_signal_positions"] > fix["programs"]["F1"]["uses_positions"]
        assert fix["programs"]["F2"]["uses_positions"] > fix["programs"]["F1"]["uses_positions"]
        c = {x["contrast"]: x for x in fix["contrasts"]}
        assert c["F2-F0"]["worse"] == 0, "abstract: no model did worse with the sentence"
        assert 2 * f2["last_calls_library_valid"] > f2["last_calls_library_ran"], "most calls that ran valid"
        assert not fix["scan"]["flagged"], "limitations: the scan found no file, network or process access"
    mark = " (DRAFT)" if draft else ""
    lines = [f"% Generated by make_pfix_results.py from {path.name} (sha256 "
             f"{hashlib.sha256(raw).hexdigest()[:12]}) and study_summary.min.json; do not edit.{mark}"]
    for k, v in num.items():
        text = str(v).replace("_", "\\_") if k.endswith("Model") else str(v)
        if draft and not k.startswith(("Frozen", "SNoLib")) and k not in DESIGN:
            text = "\\pending{" + text + "}"
        lines.append(f"\\newcommand{{\\{k}}}{{{text}}}")
    (HERE / "pfix_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = []
    for m in CORE + EXTENSION:
        name = f"\\texttt{{{m}}}" + ("$^\\ast$" if m in CORE else "")
        fr = frozen[m]["P"]
        rce = " & ".join(str(fr[f"{a}|0"][0]) for a in ("repeat", "check", "explain"))
        if m in per:
            c = per[m]
            calls = c["F1"]["calling"] + c["F2"]["calling"]
            ran = c["F1"]["calling_ran"] + c["F2"]["calling_ran"]
            cells = [str(c[a]["valid"]) for a in ("F0", "F1", "F2")] + [f"{ran}/{calls}"]
        else:
            cells = ["\\multicolumn{4}{c}{\\emph{running}}"]
        rows.append(" & ".join([name, *cells, rce]) + " \\\\" + (" \\midrule" if m == CORE[-1] else ""))
    table = ["% Generated by make_pfix_results.py; do not edit." + mark,
             "\\begin{tabular}{@{}lrrrrrrr@{}}", "\\toprule",
             " & \\multicolumn{4}{c}{Library study} & \\multicolumn{3}{c}{No library} \\\\",
             "\\cmidrule(lr){2-5}\\cmidrule(l){6-8}",
             "Model & F0 & F1 & F2 & ran & Rep. & Check & Expl. \\\\",
             "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]
    (HERE / "results_table.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    print(f"{len(models)} P-fix models{mark}; wrote pfix_numbers.tex and results_table.tex")


if __name__ == "__main__":
    main(sys.argv[1:])
