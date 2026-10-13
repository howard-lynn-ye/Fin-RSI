# Reviewed-interface model preflight, October 12, 2026

These are real-model interface checks, not full-period return experiments.
Each model saw the first, middle and last scheduled date independently in both
arms (seed 11, initially all cash). No Return Rate is defined for these six
independent scenarios. The model chose every submitted allocation.

| Model | No-library completed decisions | Library completed decisions | Gate |
|---|---:|---:|---|
| Qwen2.5-Coder-7B-Instruct | 3/3 | 3/3 | Pass for decision submission |
| Granite-3.3-8B-Instruct | 1/3 | 3/3 | Failed; full-period pair blocked |
| Qwen3-4B-Instruct-2507 | 3/3 | 3/3 | Pass for decision submission |

Preflight job: `2124509`. Independent record/source verification and diagnostic
job: `2124529`. The three public model reports preserve all cases, including
failed cases, research errors, tool calls, accepted targets and original record
hashes. `verified-summary.json` binds those public reports to the verified
original records. Original prompts and full responses remain on Beacon.

## Remaining problems and what the evidence means

- Qwen 7B repeatedly requested market data through the investment deadline,
  which was later than the current decision date. The as-of guard rejected
  those requests correctly. On the first and middle no-library cases, all six
  research calls failed even though the final allocation was accepted. A
  successful submission is therefore not sufficient evidence of usable research.
- Granite simulated several future dialogue turns inside one response. Its
  failed final decisions combined a new asset allocation with `action: hold`,
  sometimes with trailing prose. Those instructions conflict for an all-cash
  account. The executor did not guess a trade or convert the failure into HOLD.
- Granite also guessed algorithm identifiers and confused package tools with
  the study interface. The record contains both successful library queries and
  these failed calls. This remains a model/interface usability limitation.
- Qwen3 made no successful tool calls in either arm in these scenarios. Missing
  explicit CASH entries were corrected after feedback. Its successful decisions
  do not demonstrate actual tool use or the library's value.
- No full-period expansion was launched from these checks. Better decision
  formatting alone does not establish that all library capabilities were used,
  eliminate model failures on other dates, or prove higher Return Rate.

## Version and validation boundaries

The frozen source is `v9-review-20261012-candidate6/source`, interaction revision
`20261012-reviewed-decision-contract`, matching the experiment code in `b0411fa`.
It predates the separate V2 component integration in `f29f5da`; these are not
results for those new components. PR #29 merged the reviewed repair and both old
full-period ledgers after all 21 CI jobs passed (run `38330156568`).

The first preflight attempt, job `2124502`, failed before model decisions because
of a newly introduced workspace-call bug. Its records were retained. The fix and
actual-workspace regression passed Slurm `2124508` before this fresh run started.
The successful second run is not a rewrite of that failed attempt.

Historical evidence still uses retrospective eligibility assumptions rather than
verified point-in-time archive vintages. This is a limitation of the replay data,
separate from decision formatting and from the correctly functioning future-data
read restriction. No causal return improvement is inferred from these checks.

Publication checks (Slurm `2124533`) regenerated the catalog and package against
current master including the separate V2 integration, passed `validate.py` for
131 skills, and verified all three public report hashes. This publication changes
only the five report files; it does not rerun or relabel the frozen model experiment.
