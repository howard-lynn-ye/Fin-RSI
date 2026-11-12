"""Write the preliminary-results numbers and table of the short paper from study_summary.min.json.

    python make_results.py            # writes results_numbers.tex and results_table.tex

study_summary.min.json is a per-model digest of four read-only Slurm jobs on Beacon
(2026-11-11): 3620741 re-verified every finished allocation with the frozen coordinator's own
checks and summarised each model whose track was complete; 3620783 classified every recorded
reply (no program, program failed, wrong, valid); 3620797 counted P programs that called the
library and ran. Its sources and their SHA-256 are listed inside it. The protocol allows only a
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
SUMMARY_SHA256 = "f26f764387391600f6fd16903493c0b09bc38f9fdd633a32ee647ebd9985d072"
CORE = ["qwen-coder-32b", "mistral-small-24b", "granite-8b", "olmo3-7b", "phi4-mini"]
EXTENSION = ["qwen-coder-14b", "qwen-coder-7b", "qwen-general-32b", "qwen-general-14b",
             "qwen-general-7b", "qwen3-30b-a3b", "qwen3-4b", "deepseek-r1-distill-14b",
             "phi4-14b", "mistral-nemo-12b", "mistral-7b", "falcon3-10b", "falcon3-7b",
             "granite-2b", "smollm3-3b"]
FACTS = dict(FloatLabels=115, FloatEpisodes="5,865", FloatValid=0, LeakPrompts=7,
             LeakOf=497, InterfaceAlgorithms=48, InterfaceChars="25,658")
METHODS = ["no_improve", "self_debug", "check_feedback", "reflexion", "playbook", "team", "evolve"]


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

    def p_valid(m, lib):
        return sum(models[m]["P"][f"{a}|{lib}"][0] for a in ("repeat", "check", "explain"))

    def p_total(m):
        return p_valid(m, 0) + p_valid(m, 1)

    e = {m: models[m]["E"] for m in done}
    lib_minus = {m: e[m]["True"]["sb_1pct"] - e[m]["False"]["sb_1pct"] for m in done}
    contrast = lambda m, k: models[m]["P_contrasts"][k]
    s_models = [m for m in CORE if "S" in models[m]]
    s_total = {m: sum(v[0] for v in models[m]["S"].values()) for m in s_models}
    best = max(done, key=p_total)
    worst_e = min(done, key=lambda m: lib_minus[m])
    q = models["qwen-coder-32b"]
    num = dict(
        ResModelsDone=len(done), ResModelsRunning=len(running),
        ResCoreS=len(s_models),
        ResZeroP=sum(p_total(m) == 0 for m in done),
        ResBestP=p_total(best), ResBestPModel=best,
        ResLibCallsRan=sum(models[m]["P_library_calls_ran"] for m in done),
        ResLibCallModels=sum(models[m]["P_library_calls_ran"] > 0 for m in done),
        ResValidWithLib=sum(models[m]["P_valid_with_library_used"] for m in done),
        ResExplainCheckMax=pct(max(contrast(m, "explain0-check0") for m in done)),
        ResExplainCheckNeg=sum(contrast(m, "explain0-check0") < 0 for m in done),
        ResLibExplainMin=pct(min(contrast(m, "explain1-explain0") for m in done)),
        ResLibExplainMax=pct(max(contrast(m, "explain1-explain0") for m in done)),
        ResCheckRepeatMin=pct(min(contrast(m, "check0-repeat0") for m in done)),
        ResCheckRepeatMax=pct(max(contrast(m, "check0-repeat0") for m in done)),
        ResQwenAccepted=p_total("qwen-coder-32b") + q["P_accepted_wrong"],
        ResQwenAcceptedWrong=q["P_accepted_wrong"],
        ResELibLower=sum(lib_minus[m] < 0 for m in done),
        ResELibHigher=sum(lib_minus[m] > 0 for m in done),
        ResEWorstModel=worst_e, ResEWorstNo=pct(e[worst_e]["False"]["sb_1pct"]),
        ResEWorstLib=pct(e[worst_e]["True"]["sb_1pct"]),
        ResEWorstNoProgNo=e[worst_e]["False"]["no_program"],
        ResEWorstNoProgLib=f'{e[worst_e]["True"]["no_program"]:,}',
        ResEValidWithLib=sum(e[m][c]["valid_with_library_used"] for m in done for c in e[m]),
        ResSQwen=s_total["qwen-coder-32b"],
        ResSEpisodes=sum(v[1] for v in q["S"].values()),
        ResSOthers=", ".join(str(s_total[m]) for m in s_models if m != "qwen-coder-32b"),
        **{f"Fact{k}": v for k, v in FACTS.items()})
    for m, tag in (("qwen-coder-32b", "QwenCoderBig"), ("qwen-coder-14b", "QwenCoderMid")):
        num[f"ResE{tag}No"] = pct(e[m]["False"]["sb_1pct"])
        num[f"ResE{tag}Lib"] = pct(e[m]["True"]["sb_1pct"])
    for meth in METHODS:
        tag = "".join(w.capitalize() for w in meth.split("_"))
        num[f"ResSQwen{tag}"] = (f'{q["S"][f"{meth}|0"][0]} and {q["S"][f"{meth}|1"][0]}')
    lines = ["% Generated by make_results.py from study_summary.min.json "
             f"(sha256 {SUMMARY_SHA256[:12]}); do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in num.items()]
    (HERE / "results_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = []
    for m in CORE + EXTENSION:
        name = f"\\texttt{{{m}}}" + ("$^\\ast$" if m in CORE else "")
        if not models[m]["complete"]:
            rows.append(f"{name} & \\multicolumn{{4}}{{c}}{{\\emph{{running}}}} \\\\")
            continue
        r = models[m]
        rows.append(" & ".join([
            name, f"{p_valid(m, 0)} / {p_valid(m, 1)}", str(r["P_library_calls_ran"]),
            f'{pct(r["E"]["False"]["sb_1pct"])} $\\to$ {pct(r["E"]["True"]["sb_1pct"])}',
            f'{r["E"]["False"]["no_program"]:,} $\\to$ {r["E"]["True"]["no_program"]:,}']) + " \\\\")
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
    print(f"{len(done)} models done, {len(running)} running; wrote results_numbers.tex, results_table.tex")


if __name__ == "__main__":
    main()
