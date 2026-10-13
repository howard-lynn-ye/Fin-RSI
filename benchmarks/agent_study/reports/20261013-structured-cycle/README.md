# Repaired structured-output full-cycle results

All three planned seed-11 pairs completed the 2025-01-03 to 2026-09-25 replay.
Each arm starts with USD 100,000 and has 44 decision dates. Return Rate is cumulative,
after modeled trading costs: `100 * (ending capital / initial capital - 1)`; it is not annualized.
Slurm array 2124818 produced decisions and scores; independent export audit 2124823 rechecked
every original record, model receipt, decoder gate and cash-and-units replay. No decisions were replaced.

| Model | No library Return Rate | Library Return Rate | Difference (pp) | Failed dates: raw / library | Library execution dates |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-Coder-7B-Instruct | 33.64% | 40.32% | +6.68 | 0/44 / 0/44 | 0/44 |
| granite-3.3-8b-instruct | 17.67% | 25.84% | +8.18 | 0/44 / 0/44 | 43/44 |
| Qwen3-4B-Instruct-2507 | 40.85% | 36.12% | -4.73 | 0/44 / 0/44 | 44/44 |

## What these results show

- Both positive and negative differences are retained; no model, date or failure was dropped.
- A library receipt establishes tool/knowledge execution, not understanding or causal attribution.
  In particular, 7B made no research/library calls, so its return difference cannot establish tool benefit.
- Research-request errors remain in the records even when the final decision succeeds.
  A failed decision is separate from explicit HOLD and never receives a fallback portfolio.
- This is one seed on one development market path for three distinct models, not the required >=20-model study.
  Old protocol runs and other model sizes are not silently added to this repaired cohort.
- Historical archive availability and model training exposure remain unverified. Reference skills are
  current installed knowledge. These are retrospective case results, not unseen-market evidence.

## Reproducibility and public records

`summary.json` contains full-precision Return Rates, ending capital, completion counts and independent audits.
`ledgers.json` contains every dated target/action, prior holdings, research/receipt observations, daily NAV
and trades. Original-record SHA256 values bind those summaries to private Beacon originals. The export
omits raw model responses, full prompts and evidence text. No original dataset or response was edited.

Executable source: `v9-context-20261013-candidate6/source`, code committed at
`9faf01fda23b04ca077d502e36f71c2fee5b05d3`, merged through PR #31. Interaction revision:
`20261013-structured-final-v1`. Frozen run roots remain in
`/beacon-projects/radfm/wy891/fin-multisource-models-20261005/v9-context-model-preflight-20261013-r6`.
Earlier failed interface attempts remain in the sibling `20261013-context-repair` report.
