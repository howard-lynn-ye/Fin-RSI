# FinFM @ AAAI-27 submission: short paper (benchmark track)

This is the version to submit (decided 2026-11-05). The 7-page long version in
`../finfm_aaai27/` stays as a fallback in case the main results are complete for three or more
models by about 18 November 2026.

Workshop: FinFM, Foundation Models and Generative AI for Finance, at AAAI-27 (Montréal,
22 or 23 February 2027), https://finfm.finhorizons.org/. Deadline 20 November 2026, 23:59 AoE;
notification 2 December 2026. Rules checked on the workshop site on 2026-11-05:

- short papers: up to 4 pages of technical content, unlimited pages for references and
  appendices (long papers: up to 7);
- tracks: research papers and benchmark papers;
- AAAI-27 template, anonymized for double-blind review;
- non-archival: no proceedings from AAAI or FinFM, so the work can go to ACL 2027 via ARR;
- at least one author presents in person.

Portal: https://openreview.net/group?id=AAAI.org%2F2027%2FWorkshop%2FFinFM

## Layout

Since 2026-11-18 the paper reports experiments on the library as it now is (decided with the
author): section 5 "Study" has two parts, and section 6 reports them.

- **The library on held-out tasks (P-fix).** Arms F0 (no library), F1 (the position interface)
  and F2 (F1 plus one sentence naming the call), Check feedback, 3 programs, 4,096 tokens per
  reply, 6 held-out tasks x 3 seeds, 20 models, on Zaratan H100s. Design and amendment:
  `docs/P_FIX_STUDY_DESIGN_20261118.md` on branch `claude/pfix`. `pfix_analysis.json` is the
  output of `scripts/analyze_p_fix.py runs/main out.json --fix-root runs-v2/main` (F0 from the
  first run; F1/F2 from the rerun with the library at 001ea472, identical prompts), sha256
  646228b0...; `make_pfix_results.py` pins that hash, writes `pfix_numbers.tex` and
  `results_table.tex`, and asserts every direction the prose states.
- **Feedback and self-improvement without the library.** The no-library cells of the study frozen
  on 2026-11-08 (`study_summary.min.json`, pinned in both generators). Its library conditions
  and track E used an earlier interface text; they are released, not reported.

One positions() call with each task's parameters and factor convention is valid on all 18
held-out task x seed pairs (checked with the judge on 2026-11-18). Technical content ends on
page 4. `make_results.py` and `results_numbers.tex` still build the earlier frozen-study draft's
macros; the text no longer uses its numbers.

## Template and build

`aaai2027.sty` and `aaai2027.bst` are the official AAAI-27 author-kit files, unmodified, with
the same SHA-256 values as in `../finfm_aaai27/README.md`.

```bash
cd paper/finfm_aaai27_short
latexmk -pdf main.tex
```

## Citations of the organisers' work (added 2026-11-10)

The co-chairs named in the call are Yichen Li, Stefan Zohren, Yoontae Hwang and Qingsong Wen.
Nine of their papers are cited where they bear on the text, each checked against arXiv, the
publisher or the ACL/ICLR/KDD record: two surveys of language models in finance and the
position paper on evaluation bias (introduction), Time-LLM, the time-series foundation-model
survey and the LLM portfolio paper (introduction), and three Oxford papers on momentum,
changepoints and portfolio optimisation (task suite).
