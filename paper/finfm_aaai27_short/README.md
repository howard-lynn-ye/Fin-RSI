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

Since 2026-11-10 the text describes the study that actually ran (section 5, "Study
protocol": tracks P, E and S, frozen on 2026-11-08), not the earlier four-arm design. Its
numbers are in `study_numbers.tex`, read from the frozen plan. Technical content (sections
1-6, with Figures 1-2) now ends a quarter of the way down the right column of page 4, which
leaves about a third of a page for the results. They will need more, so plan to move
Figure 2 to the appendix (about 0.4 page). The method table, rule cards and task table are in the appendix; the
skill-arm figure is commented out here because that ablation was not run.

Numbers, figures and tables come from `../latex_acl2027/` (`make_numbers.py`), so the three
versions cannot disagree. The condensed text is in `sections/`.

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
