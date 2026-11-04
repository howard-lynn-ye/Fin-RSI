# Fin-RSI paper (ACL 2027 draft)

Title: **Fin-RSI: An Agent-Native Library of Financial Skills, Checks, and Self-Improvement**.
The earlier NAACL draft stays in `../latex_naacl/` until the coauthors agree what moves here.

| File | Status |
|---|---|
| `sections/library.tex` | drafted 2026-11-04: skills, checks, agent interfaces |
| `sections/methods.tex` | drafted 2026-11-04: the RSI interface, budgets, 7 methods, gated loop, skill arms |
| `sections/tasks.tex` | drafted 2026-11-04: task suite, references, judge and grader, split, protocol |
| `sections/related.tex` | drafted 2026-11-04: self-improvement, library-plus-benchmark, financial LLMs, metamorphic testing |
| `figures/overview.tex` | Figure 1 (TikZ): the library and one episode; counts are macros from `numbers.tex` |
| `figures/task_checks.pdf` | Figure 2: task data and the `execution_lag` check, drawn by `make_figures.py` from the suite |
| `main.tex` | abstract, introduction, results, discussion: placeholders in red |

## Numbers come from the code

`numbers.tex`, `cards_table.tex` and `tasks_table.tex` are generated. Do not edit them:

```bash
python paper/latex_acl2027/make_numbers.py   # from the repository root
```

`make_figures.py` draws Figure 2 the same way, from the seed-11 panel and the library's own
`positions()`, perturbing prices exactly as the `execution_lag` guard does.

`make_numbers.py` reads the catalog, the guard registry, the tool exporter, the rule cards, the task suite
and the RSI defaults, and it recomputes the reference-versus-library agreement on every task
and seed. Rerun both after any change to those, then rebuild.

## Build

```bash
cd paper/latex_acl2027
latexmk -pdf main.tex
```

`\pending{...}` marks text that waits for the experiments; it prints in red and must be gone
before submission.

Page budget: with the introduction and results still to come, the body already reaches page 6.
To stay within 8 pages, the cards, methods and arms tables are candidates for the appendix.
