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
ANALYSIS_SHA256 = "f9962b897bb93626bbbf8647c65b4fb506848930078ddc31ef9cd6cd07e36923"
CORE = ["qwen-coder-32b", "mistral-small-24b", "granite-8b", "olmo3-7b", "phi4-mini"]
EXTENSION = ["qwen-coder-14b", "qwen-coder-7b", "qwen-general-32b", "qwen-general-14b",
             "qwen-general-7b", "qwen3-30b-a3b", "qwen3-4b", "deepseek-r1-distill-14b",
             "phi4-14b", "mistral-nemo-12b", "mistral-7b", "falcon3-10b", "falcon3-7b",
             "granite-2b", "smollm3-3b"]
METHODS = ["self_debug", "check_feedback", "reflexion", "team", "evolve", "playbook"]
TAG = {"F0": "FZero", "F1": "FOne", "F2": "FTwo", "F3": "FThree", "F4": "FFour"}
# The declared-property checks of fin_skills/tasks/properties.py (Amendment 3).
CHECK_TAG = {"takes_positions": "TakesPositions", "position_scale": "PositionScale",
             "warmup_rows_zero": "WarmupRowsZero", "finite_output": "FiniteOutput",
             "split_representation": "SplitRepresentation", "execution_lag": "ExecutionLag"}
# The post-hoc size split of analyze_p_fix.py (by_size): total parameters >= 14B, or fewer.
SIZE = {"ge14b": "Large", "lt14b": "Small"}
# Fixed by the P-fix design (branch claude/pfix, docs/P_FIX_STUDY_DESIGN_20261118.md, amendment 3 at
# b8417dcc): the v4 interface text is 5,657 characters and lists 12 algorithms that positions()
# accepts; 4,096 generated tokens per reply; at most 3 programs, 6 calls and 12,288 generated tokens
# per episode; route() replies are capped at 1,024 tokens; 6 held-out tasks x 3 seeds per arm.
DESIGN = dict(FixInterfaceChars="5,657", FixInterfaceAlgorithms=12, FixOutputCap="4,096",
              FixEpisodesPerArm=18, FixVisibleChecks=6, FixRouteCap="1,024", FixAttempts=3,
              FixCalls=6, FixTokens="12,288")
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
    arms_run = [a for a in TAG if a in pooled]  # F3 (Amendment 2) once its runs are in
    num["FixDocumented"] = int("F3" in pooled)
    for arm in arms_run:
        tag, p = TAG[arm], pooled[arm]
        num[f"Fix{tag}Valid"] = p["valid"]
        num[f"Fix{tag}Pct"] = pct(p["valid"], p["n"])
        num[f"Fix{tag}First"] = p["first_valid"]
        num[f"Fix{tag}AcceptedWrong"] = p["accepted_wrong"]
        num[f"Fix{tag}NoneValid"] = p["models_none_valid"]
        best = max(models, key=lambda m: per[m][arm]["valid"])
        num[f"Fix{tag}Best"], num[f"Fix{tag}BestModel"] = per[best][arm]["valid"], best
    for c in fix["contrasts"] + fix.get("exploratory_contrasts", []):
        a, b = c["contrast"].split("-")
        key = f"Fix{TAG[a]}Minus{TAG[b]}"
        num[key] = signed(c["pooled_pct_points"])
        num[key + "Better"], num[key + "Worse"], num[key + "Same"] = c["better"], c["worse"], c["same"]
    for arm in [a for a in arms_run if a != "F0"]:
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
    for arm in ("F1", "F2"):  # the declared arms; F3's errors are reported on their own
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
    num["FixSizeCut"] = "14B"
    for label, size in SIZE.items():
        g = fix["by_size"][label]
        num[f"Fix{size}Models"] = len(g["models"])
        for arm in arms_run:
            p = g["pooled"][arm]
            num[f"Fix{size}{TAG[arm]}Valid"], num[f"Fix{size}{TAG[arm]}N"] = p["valid"], p["n"]
            num[f"Fix{size}{TAG[arm]}Pct"] = pct(p["valid"], p["n"])
            num[f"Fix{size}{TAG[arm]}NoneValid"] = p["models_none_valid"]
        for c in g["contrasts"] + g.get("exploratory_contrasts", []):
            a, b = c["contrast"].split("-")
            key = f"Fix{size}{TAG[a]}Minus{TAG[b]}"
            num[key] = signed(c["pooled_pct_points"])
            num[key + "Better"], num[key + "Worse"] = c["better"], c["worse"]
    assert not fix["unsized"], f"models without a parameter count: {fix['unsized']}"
    # F4: what the model named when route() asked which catalogue rule computes the task's rule.
    for scope, routing in (("", fix.get("routing", {})),
                           *((size, fix["by_size"][label].get("routing", {}))
                             for label, size in SIZE.items())):
        for how in ("right", "none", "wrong"):
            num[f"Fix{scope}Route{how.capitalize()}"] = routing.get(how, 0)
            num[f"Fix{scope}Route{how.capitalize()}Valid"] = routing.get(how + "_valid", 0)
    # The best arm of the models under 14B (abstract and results: small models).
    small = fix["by_size"]["lt14b"]["pooled"]
    num["FixSmallMaxPct"] = max(pct(small[a]["valid"], small[a]["n"]) for a in arms_run)
    # Episodes in which a property check flagged a program that ran: another attempt? valid?
    total = {"episodes_flagged": 0, "with_another_attempt": 0, "then_valid": 0}
    for arm in arms_run:
        r = fix.get("repair", {}).get(arm, {})
        for key, name in (("episodes_flagged", "Flagged"), ("with_another_attempt", "FlaggedRetry"),
                          ("then_valid", "FlaggedThenValid")):
            num[f"Fix{TAG[arm]}{name}"] = r.get(key, 0)
            total[key] += r.get(key, 0)
    num["FixFlagged"], num["FixFlaggedRetry"] = total["episodes_flagged"], total["with_another_attempt"]
    num["FixFlaggedThenValid"] = total["then_valid"]
    # F2, and F4 after a right route: the sentence named the task's own rule. Did the last
    # program call positions() or keep a rule of its own?
    for scope, told in (("", fix.get("told", {})),
                        *((size, fix["by_size"][label].get("told", {})) for label, size in SIZE.items())):
        for arm in ("F2", "F4"):
            t = told.get(arm, {})
            for key, name in (("episodes", "Told"), ("valid", "ToldValid"),
                              ("last_calls_positions", "ToldCalls"),
                              ("last_calls_positions_valid", "ToldCallsValid"),
                              ("last_own_rule", "ToldOwn"), ("last_own_rule_valid", "ToldOwnValid"),
                              ("no_program_ran", "ToldNoProgram")):
                num[f"Fix{scope}{TAG[arm]}{name}"] = t.get(key, 0)
    # Programs that ran and failed each visible check, per arm and over all arms.
    for name, tag in CHECK_TAG.items():
        total = 0
        for arm in arms_run:
            n = fix.get("failed_checks", {}).get(arm, {}).get(name, 0)
            num[f"Fix{TAG[arm]}Failed{tag}"] = n
            total += n
        num[f"FixFailed{tag}"] = total
    for arm in arms_run:
        num[f"Fix{TAG[arm]}Ran"] = fix["failed_checks"][arm]["programs_ran"]
        num[f"Fix{TAG[arm]}Programs"] = f'{fix["programs"][arm]["programs"]:,}'
    # F4 against F0 per model: larger models that gained exactly one episode; models unchanged at 0.
    large = fix["by_size"]["ge14b"]["models"]
    num["FixLargeFFourByOne"] = sum(per[m]["F4"]["valid"] - per[m]["F0"]["valid"] == 1 for m in large)
    num["FixFFourSameAtZero"] = sum(per[m]["F4"]["valid"] == per[m]["F0"]["valid"] == 0 for m in models)
    # Caption numbers of Figure 2: the one smaller model that gained, and how close F4 is to F2.
    small_models = fix["by_size"]["lt14b"]["models"]
    gain = {m: per[m]["F4"]["valid"] - per[m]["F0"]["valid"] for m in small_models}
    top = max(gain, key=gain.get)
    num["FixSmallGainModel"] = top
    num["FixSmallGainFZero"], num["FixSmallGainFFour"] = per[top]["F0"]["valid"], per[top]["F4"]["valid"]
    f4f2 = next(c for c in fix["exploratory_contrasts"] if c["contrast"] == "F4-F2")
    num["FixFourTwoMaxDiff"] = max(abs(v) for v in f4f2["per_model"].values())
    num["FixFourTwoSame"] = f4f2["same"]
    num["FixScanFlagged"] = len(fix["scan"]["flagged"])
    num["FixScanNotPython"] = fix["scan"]["not_python"]
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
        ex = {x["contrast"]: x for x in fix["exploratory_contrasts"]}
        assert abs(pooled["F4"]["valid"] - pooled["F2"]["valid"]) <= 3, "F4 no longer matches F2"
        assert ex["F4-F0"]["worse"] == 0, "no model did worse with route()"
        assert pooled["F3"]["valid"] == pooled["F1"]["valid"], "F3 is no longer equal to F1"
        big, small = fix["by_size"]["ge14b"], fix["by_size"]["lt14b"]
        assert big["routing"]["right"] == big["pooled"]["F4"]["n"], "a large model misrouted"
        assert {x["contrast"]: x for x in big["exploratory_contrasts"]}["F4-F0"]["better"] == \
            len(big["models"]), "not every large model did better with route()"
        sp = small["pooled"]
        assert num["FixSmallMaxPct"] == max(pct(sp[a]["valid"], sp[a]["n"]) for a in sp), "small max"
        rp = fix["repair"]
        assert all(2 * rp[a]["then_valid"] < rp[a]["with_another_attempt"] for a in rp), "repair rare"
        st = small["told"]["F4"]
        assert st["episodes"] == small["routing"]["right"] and \
            st["last_calls_positions"] < st["episodes"] / 2, "small models told the rule"
        assert fix["routing"].get("wrong_valid", 0) == 0, "a wrong route was valid"
        assert f2["last_calls_library_accepted_wrong_rescaled"] == 0 == \
            fix["funnel"]["F4"]["last_calls_library_accepted_wrong_rescaled"], "a call scaled twice"
        f4 = fix["funnel"]["F4"]
        assert sorted(gain.values())[-2] <= 1, "a second smaller model gained more than one episode"
        same = {x["contrast"]: x for x in fix["exploratory_contrasts"]}["F4-F0"]["same"]
        assert num["FixFFourSameAtZero"] == same, "a model unchanged in F4 is not at 0"
        assert per["qwen3-30b-a3b"]["F2"]["valid"] == per["qwen3-30b-a3b"]["F4"]["valid"] == 18, "best"
        assert num["FixFFourBestModel"] == num["FixFTwoBestModel"] == "qwen3-30b-a3b"
        assert sum(frozen[m]["S"]["no_improve|0"][0] > 0 for m in s_models) == 2, "two core models"
        gains = {k: num["SNoLib" + "".join(w.capitalize() for w in k.split("_"))] for k in METHODS}
        assert min(gains.values()) == gains["self_debug"] and \
            max(gains.values()) == gains["check_feedback"] == gains["evolve"], "range of the methods"
        assert 2 * f4["last_calls_library_valid"] > f4["last_calls_library_ran"], "F4 calls mostly valid"
        for m in s_models:  # the seven methods gained only where NoImprove was already valid
            if frozen[m]["S"]["no_improve|0"][0] == 0:
                assert all(frozen[m]["S"][f"{k}|0"][0] == 0 for k in METHODS), m
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
            right = fix["routing_by_model"].get(m, {}).get("right", 0)  # the caption's "Rt."
            cells = [str(c[a]["valid"]) if c[a]["n"] else "--" for a in arms_run] + [str(right)]
        else:
            cells = [f"\\multicolumn{{{len(arms_run) + 1}}}{{c}}{{\\emph{{running}}}}"]
        rows.append(" & ".join([name, *cells, rce]) + " \\\\" + (" \\midrule" if m == CORE[-1] else ""))
    k = len(arms_run) + 1  # arm columns plus "ran"
    table = ["% Generated by make_pfix_results.py; do not edit." + mark,
             "\\begin{tabular}{@{}l" + "r" * (k + 3) + "@{}}", "\\toprule",
             f" & \\multicolumn{{{k}}}{{c}}{{Library study}} & \\multicolumn{{3}}{{c}}{{No library}} \\\\",
             f"\\cmidrule(lr){{2-{k + 1}}}\\cmidrule(l){{{k + 2}-{k + 4}}}",
             "Model & " + " & ".join(arms_run) + " & Rt. & Rep. & Check & Expl. \\\\",
             "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]
    (HERE / "results_table.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    names = {"no_improve": "NoImprove", "self_debug": "SelfDebug", "check_feedback": "CheckFeedback",
             "reflexion": "Reflexion", "team": "Team", "evolve": "Evolve", "playbook": "PlaybookLearning"}
    s_core = [m for m in CORE if "S" in frozen[m]]
    lines = ["% Generated by make_pfix_results.py from study_summary.min.json; do not edit.",
             "\\begin{tabular}{@{}l" + "r" * (len(s_core) + 1) + "@{}}", "\\toprule",
             "Method & " + " & ".join(f"\\texttt{{{m}}}" for m in s_core) + " & all \\\\", "\\midrule"]
    for k in ["no_improve"] + METHODS:
        vals = [frozen[m]["S"][f"{k}|0"][0] for m in s_core]
        lines.append(f"\\texttt{{{names[k]}}} & " + " & ".join(str(v) for v in vals) + f" & {sum(vals)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (HERE / "frozen_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(models)} P-fix models{mark}; wrote pfix_numbers.tex and results_table.tex")


if __name__ == "__main__":
    main(sys.argv[1:])
