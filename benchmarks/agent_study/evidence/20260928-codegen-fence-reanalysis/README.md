# Post-hoc code-extraction sensitivity analysis (2026-09-28)

Source batch: `../20260928-codegen-utility/` (Beacon job 1733007, hash-verified).
Script: `benchmarks/agent_study/codegen_fence_reanalysis.py`.

1. `freeze` verifies the batch, takes the first complete Markdown fence of each response and
   writes `extractions.json` plus `extraction-receipt.json` before any program runs.
2. `score` executes each extraction with the frozen worker, sandbox and grader and writes
   `scores.json`; identical code that the official run executed must reproduce its grades.
3. `diagnose` locates the first moved row under each failed timing probe and counts
   re-cumulated split factors (`diagnostics.json`).

The official, pre-registered scores in the source batch are unchanged. See
`paper/CODEGEN_UTILITY_RESULTS_20260928.md` for results and limits.
