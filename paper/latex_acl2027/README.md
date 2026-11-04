# Fin-RSI paper (ACL 2027 draft)

Title: **Fin-RSI: An Agent-Native Library of Financial Skills, Checks, and Self-Improvement**.
The earlier NAACL draft stays in `../latex_naacl/` until the coauthors agree what moves here.

| File | Status |
|---|---|
| `sections/library.tex` | drafted 2026-11-04: skills, checks, agent interfaces |
| `sections/methods.tex` | drafted 2026-11-04: the RSI interface, budgets, 7 methods, gated loop, skill arms |
| `sections/tasks.tex` | drafted 2026-11-04: task suite, references, judge and grader, split, protocol |
| `main.tex` | abstract, introduction, related work, results, discussion: placeholders in red |

## Numbers come from the code

`numbers.tex`, `cards_table.tex` and `tasks_table.tex` are generated. Do not edit them:

```bash
python paper/latex_acl2027/make_numbers.py   # from the repository root
```

It reads the catalog, the guard registry, the tool exporter, the rule cards, the task suite
and the RSI defaults, and it recomputes the reference-versus-library agreement on every task
and seed. Rerun it after any change to those, then rebuild.

## Build

```bash
cd paper/latex_acl2027
latexmk -pdf main.tex
```

`\pending{...}` marks text that waits for the experiments; it prints in red and must be gone
before submission.
