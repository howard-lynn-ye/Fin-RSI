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

- **The library on held-out tasks (P-fix v4).** Arms F0 (no library), F1 (the position
  interface), F2 (F1 plus one sentence naming the call, id from the harness), F3 (F1 with each
  algorithm's rule documented) and F4 (`route()` first, then F2's sentence with the id the model
  named), Check feedback, six visible checks, 3 programs, 6 calls, 12,288 tokens per episode,
  4,096 per reply, 6 held-out tasks x 3 seeds, 20 models, on Zaratan H100s. F1-F0, F2-F0 and
  F2-F1 were declared first; F3, F4 and the 14B split are exploratory. Design and amendments 1-3:
  `docs/P_FIX_STUDY_DESIGN_20261118.md` on branch `claude/pfix` (library source b8417dcc).
  `pfix_analysis.json` is `scripts/analyze_p_fix.py runs-v4/main out.json --doc-root
  runs-v4/main --route-root runs-v4/main` (4830421c on `claude/pfix`);
  `make_pfix_results.py` pins that hash, writes `pfix_numbers.tex` and `results_table.tex`, and
  asserts every direction the prose states. The analysis was re-pinned at f9962b897bb9... when
  it gained `repair` (episodes in which a new property check flagged a program) and `told`
  (F2, and F4 after a right route: did the last program call positions()); nothing else changed.
  One number comes from the analysis before amendment 3 (commit 49891186, sha256
  1b5ac441666a...) and is a documented constant in the script: 81 accepted-but-wrong F0
  programs under the first three checks.
- **Feedback and self-improvement without the library.** The no-library cells of the study frozen
  on 2026-11-08 (`study_summary.min.json`, pinned in both generators). Its library conditions
  and track E used an earlier interface text; they are released, not reported.

One positions() call with each task's parameters and factor convention is valid on all 18
held-out task x seed pairs (checked with the judge on 2026-11-18). Technical content ends at
the bottom of page 4 (2026-11-19, with no spare line): any addition needs a cut. The study's six
held-out tasks are those of the suite of 8 November; in the current 34-task split Bollinger is a
development task, and section 4 says so. `make_results.py` and `results_numbers.tex` still build the earlier frozen-study draft's
macros; the text no longer uses its numbers.

## Figures (2026-11-19, third design)

Every figure is in one journal style (bold lower-case panel letters, thin axes, outward ticks,
no grid, TeX Gyre Heros, nothing below 7 pt). `make_pfix_figures.py` draws Figures 2-4 from the
pinned analysis (it imports the hash from `make_pfix_results.py`) at the exact column or text
width, and refuses to save a figure with text below 7 pt, text outside the figure or overlapping
labels; `--png` also writes previews (not committed). Background: project docs
`claude/PAPER_FIGURES.md` and `claude/FINFM_FIGURE_RESEARCH_20261119.md`.

- Figure 1 `figures/overview_short.tex`: a copy of the shared `overview3d.tex` with `route()` on
  the pick arrow and the hidden grader behind a wall after deliver; its caption is
  `overview_caption_short.tex`. The shared files are untouched.
- Figure 2 `pfix_models.pdf`: every model, grouped by series (the id without its size; the
  study's eight families split Qwen into three series) and ordered by size; F0 open, F4 filled,
  F2 dash, a line through F4 along each series, shaded columns for 14B or more.
- Figure 3 `pfix_matrix.pdf`: model x arm matrix of valid episodes (of 18), shaded by size
  group, with what `route()` named per model and the pooled row.
- Figure 4 `pfix_analysis.pdf` (full width): (a) how episodes ended by arm and size group; (b) the
  360 F4 episodes from what `route()` named to the last program to the outcome; (c) the share of
  programs that ran failing each check, with what a failure means and F0 -> F4. The repair
  numbers that were panel (d) are in the results text.
- Appendix Figure 7 `real_panel_short.pdf`, drawn by `make_real_panel.py` from the fetched real
  panel (not in the repository; pass its path in `FIN_TASKS_REAL_PANEL`, the SHA-256 is
  checked): the four splits, with the first-day change of the raw quote and of the adjusted
  close. Figures 5 (one episode) and 6 (the playbook gate) are the shared TikZ figures.
- Appendix tables: per-model table (`results_table.tex`) and the no-library study
  (`frozen_table.tex`), both written by `make_pfix_results.py`.

Track: Benchmark (Howard, 2026-11-19).

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
