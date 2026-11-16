"""Write the preliminary-results numbers and table of the short paper from study_summary.min.json.

    python make_results.py            # writes results_numbers.tex and results_table.tex

study_summary.min.json is a per-model digest (extract_summary.py) of three read-only Slurm jobs
on Beacon, last run 2026-11-16: 3655749 re-verified every finished allocation with the frozen
coordinator's own checks and summarised each model's complete tracks; 3655750 classified every
recorded reply (no program, program failed, wrong, valid); 3655751 counted P programs that called
the library and ran. Its sources and their SHA-256 are listed inside it. The same scripts ran on
2026-11-11 as 3620741, 3620783 and 3620797; every field of the 17 models complete then is
unchanged. The protocol allows only a
per-model description until all 20 models finish, so nothing here is pooled across models
except counts of models and sums of events.

FACTS are numbers from two further read-only jobs, kept here with their provenance:
  3620857: the 115 CodeFinQA labels stored as float32 leftovers (e.g. 41.0999984741 for 41.1)
           were matched by 0 of 5,865 episodes in each library condition;
  3620862: in the no-library PlaybookLearning fits, 7 of 497 proposer prompts carried a
           library hint (granite-8b 1/33, phi4-mini 6/180, olmo3-7b 0/266, qwen-coder-32b 0/18);
           no no-library test prompt did.
  interface.json of the frozen E enrollment (enrollment-3607864): the library text added to the
           E prompt lists 48 algorithms (17 of them missing a dependency) in 25,658 characters.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
SUMMARY = HERE / "study_summary.min.json"
SUMMARY_SHA256 = "61a0c481d16c9124a2cdc96eb8e9b2e8b5c37f809bc2341fd547ccd928e683f5"
CORE = ["qwen-coder-32b", "mistral-small-24b", "granite-8b", "olmo3-7b", "phi4-mini"]
EXTENSION = ["qwen-coder-14b", "qwen-coder-7b", "qwen-general-32b", "qwen-general-14b",
             "qwen-general-7b", "qwen3-30b-a3b", "qwen3-4b", "deepseek-r1-distill-14b",
             "phi4-14b", "mistral-nemo-12b", "mistral-7b", "falcon3-10b", "falcon3-7b",
             "granite-2b", "smollm3-3b"]
FACTS = dict(FloatLabels=115, FloatEpisodes="5,865", FloatValid=0, LeakPrompts=7,
             LeakOf=497, InterfaceAlgorithms=48, InterfaceChars="25,658")
METHODS = ["no_improve", "self_debug", "check_feedback", "reflexion", "playbook", "team", "evolve"]
S_WRITERS = ["qwen-coder-32b", "mistral-small-24b"]


def pct(x):
    return f"{100 * x:.1f}"


def main():
    raw = SUMMARY.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SUMMARY_SHA256:
        raise SystemExit("study_summary.min.json differs from the digest these numbers were checked on")
    models = json.loads(raw)["models"]
    assert sorted(models) == sorted(CORE + EXTENSION), "model list differs from the protocol's 20"
    done = [m for m in CORE + EXTENSION if models[m]["complete"]]
    running = [m for m in CORE + EXTENSION if not models[m]["complete"]]
    # Track P is described for every model whose P has finished, track E for those whose E has.
    p_done = [m for m in CORE + EXTENSION if models[m].get("P_complete", models[m]["complete"])]

    def p_valid(m, lib):
        return sum(models[m]["P"][f"{a}|{lib}"][0] for a in ("repeat", "check", "explain"))

    def p_total(m):
        return p_valid(m, 0) + p_valid(m, 1)

    e = {m: models[m]["E"] for m in done}
    lib_minus = {m: e[m]["True"]["sb_1pct"] - e[m]["False"]["sb_1pct"] for m in done}
    contrast = lambda m, k: models[m]["P_contrasts"][k]
    s_models = [m for m in CORE if "S" in models[m]]
    s_total = {m: sum(v[0] for v in models[m]["S"].values()) for m in s_models}
    best = max(p_done, key=p_total)
    others = [m for m in s_models if m not in S_WRITERS]
    worst_e = min(done, key=lambda m: lib_minus[m])
    q = models["qwen-coder-32b"]
    num = dict(
        ResModelsP=len(p_done), ResModelsDone=len(done), ResModelsRunning=len(running),
        ResCoreS=len(s_models),
        ResZeroP=sum(p_total(m) == 0 for m in p_done),
        ResBestP=p_total(best), ResBestPModel=best,
        ResLibCallsRan=sum(models[m]["P_library_calls_ran"] for m in p_done),
        ResLibCallModels=sum(models[m]["P_library_calls_ran"] > 0 for m in p_done),
        ResValidWithLib=sum(models[m]["P_valid_with_library_used"] for m in p_done),
        ResExplainCheckMin=pct(min(contrast(m, "explain0-check0") for m in p_done)),
        ResExplainCheckMax=pct(max(contrast(m, "explain0-check0") for m in p_done)),
        ResExplainCheckNeg=sum(contrast(m, "explain0-check0") < 0 for m in p_done),
        ResLibExplainMin=pct(min(contrast(m, "explain1-explain0") for m in p_done)),
        ResLibExplainMax=pct(max(contrast(m, "explain1-explain0") for m in p_done)),
        ResCheckRepeatMin=pct(min(contrast(m, "check0-repeat0") for m in p_done)),
        ResCheckRepeatMax=pct(max(contrast(m, "check0-repeat0") for m in p_done)),
        ResQwenAccepted=p_total("qwen-coder-32b") + q["P_accepted_wrong"],
        ResQwenAcceptedWrong=q["P_accepted_wrong"],
        ResELibLower=sum(lib_minus[m] < 0 for m in done),
        ResELibHigher=sum(lib_minus[m] > 0 for m in done),
        ResEWorstModel=worst_e, ResEWorstNo=pct(e[worst_e]["False"]["sb_1pct"]),
        ResEWorstLib=pct(e[worst_e]["True"]["sb_1pct"]),
        ResEWorstNoProgNo=e[worst_e]["False"]["no_program"],
        ResEWorstNoProgLib=f'{e[worst_e]["True"]["no_program"]:,}',
        ResEValidWithLib=sum(e[m][c]["valid_with_library_used"] for m in done for c in e[m]),
        ResSQwen=s_total["qwen-coder-32b"], ResSMistral=s_total["mistral-small-24b"],
        ResSEpisodes=sum(v[1] for v in q["S"].values()),
        ResSOthers=" and ".join([", ".join(str(s_total[m]) for m in others[:-1]), str(s_total[others[-1]])]),
        **{f"Fact{k}": v for k, v in FACTS.items()})
    for m, tag in (("qwen-coder-32b", "QwenCoderBig"), ("qwen-coder-14b", "QwenCoderMid")):
        num[f"ResE{tag}No"] = pct(e[m]["False"]["sb_1pct"])
        num[f"ResE{tag}Lib"] = pct(e[m]["True"]["sb_1pct"])
    # Track S: the text names the core models that wrote valid programs and states two patterns;
    # these asserts stop the build if a refreshed summary no longer supports the wording.
    writers = [m for m in s_models if s_total[m] >= 10]
    assert writers == S_WRITERS, writers
    assert all(s_total[m] <= 1 for m in s_models if m not in writers)
    sv = lambda m, meth, lib: models[m]["S"][f"{meth}|{lib}"][0]
    for m in writers:   # without the library every method but SelfDebug beats NoImprove; SelfDebug ties
        assert all(sv(m, k, 0) > sv(m, "no_improve", 0) for k in METHODS if k not in ("no_improve", "self_debug"))
        assert sv(m, "self_debug", 0) == sv(m, "no_improve", 0)
    mi = "mistral-small-24b"   # with the library mistral-small-24b falls under every method but Team
    assert all(sv(mi, k, 1) < sv(mi, k, 0) for k in METHODS if k != "team") and sv(mi, "team", 1) > sv(mi, "team", 0)
    for meth in METHODS:
        tag = "".join(w.capitalize() for w in meth.split("_"))
        num[f"ResSQwen{tag}"] = (f'{q["S"][f"{meth}|0"][0]} and {q["S"][f"{meth}|1"][0]}')
        num[f"ResS{tag}"] = "; ".join(f"{sv(m, meth, 0)}/{sv(m, meth, 1)}" for m in writers)
    lines = ["% Generated by make_results.py from study_summary.min.json "
             f"(sha256 {SUMMARY_SHA256[:12]}); do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in num.items()]
    (HERE / "results_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = []
    for m in CORE + EXTENSION:
        name = f"\\texttt{{{m}}}" + ("$^\\ast$" if m in CORE else "")
        r = models[m]
        if m not in p_done:
            rows.append(f"{name} & \\multicolumn{{4}}{{c}}{{\\emph{{running}}}} \\\\")
        else:
            cells = [name, f"{p_valid(m, 0)} / {p_valid(m, 1)}", str(r["P_library_calls_ran"])]
            if m in done:
                cells += [f'{pct(r["E"]["False"]["sb_1pct"])} $\\to$ {pct(r["E"]["True"]["sb_1pct"])}',
                          f'{r["E"]["False"]["no_program"]:,} $\\to$ {r["E"]["True"]["no_program"]:,}']
            else:
                cells += ["\\multicolumn{2}{c}{\\emph{running}}"]
            rows.append(" & ".join(cells) + " \\\\")
        if m == CORE[-1]:
            rows[-1] += " \\midrule"
    table = [
        "% Generated by make_results.py; do not edit.",
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\toprule",
        " & \\multicolumn{2}{c}{P} & \\multicolumn{2}{c}{E} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}",
        "Model & valid & calls & 1\\,\\% (\\%) & no program \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]
    (HERE / "results_table.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    print(f"{len(p_done)} models with P, {len(done)} with P and E, {len(running)} running; "
          "wrote results_numbers.tex, results_table.tex")


if __name__ == "__main__":
    main()
