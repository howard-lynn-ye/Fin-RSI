# Utility pilot evidence, 2026-10-01

This directory retains development failures as well as completed measurements. It does not
replace the frozen v3 trading results. No manuscript or slide result is generated here.

## Component measurements

The existing numerical-parity suite passed 8/8 checks in `component-recheck/parity.json`.
These are reference-computation checks, not model task-completion outcomes.

The existing synthetic defect benchmark was run with its fixed seeds 17, 29 and 43.
Before repair, other guards caught all 36 planted cases, but `pit_universe` could not run:
the benchmark's `BQE` frequency alias raised on pandas 2.1.4, producing 39 execution errors.
The original failed measurements remain under `component-recheck/`.

Replacing the alias with `pd.offsets.BQuarterEnd()` preserved quarter-end semantics and
produced 36/36 cases caught by at least one guard, zero clean-data false alarms, and zero
execution errors. All three `--check` commands exited 0. See `component-recheck-fixed/`.
The individual `cost_curve` gate still did not bind on `cost_too_low` in each world; another
guard caught that defect. Thus 36/36 describes suite coverage, not perfect per-guard sensitivity.
These are known synthetic regression fixtures, not unseen financial data or proof that all
real-world leakage will be detected. Receipts record source hashes and environment details.

## Model experiments

The protocol is described in `../../V4_UTILITY_PILOT.md`. Both arms receive the same three
calculation tasks and two seeds. Six task/seed pairs share one history; they do not establish
statistical significance or investment alpha. The treatment includes executable algorithms,
an adjusted-data adapter, and library documentation; this pilot does not isolate those parts.

`aborted-r1/` retains the first 7B attempt, job 1810695. It was cancelled before scoring
after five complete decisions revealed a controller error for omitted empty arguments.
Its partial outcomes must not be combined with the replacement run or treated as scores.

The replacement 7B job is 1810898. A same-task 14B diagnostic, job 1811294, was selected
after observing basic-control non-submission by 7B, before inspecting numerical grades.
This selection is adaptive and both model outcomes must be retained. Its first scheduler
request failed due to a GPU-type spelling error; the corrected request is explicitly recorded.

### Completed 7B result

Job 1810898 completed all 12 episodes in 22m38s. Its bundle is in `7b/` and the independent
CSV-based audit agrees with every verdict. Tampered scores, changed decisions, and a missing
decision were each rejected by negative checks of the audit script.

| Task | Raw correct | Library correct |
| --- | ---: | ---: |
| Equal-weight control | 0/2 | 1/2 |
| Inverse volatility, 60 returns | 0/2 | 0/2 |
| Inverse volatility, 252 returns | 0/2 | 0/2 |
| Total | 0/6 | 1/6 |

Raw submitted 0/6; library submitted 2/6. The correct library episode called `equal_weight`
and submitted its result. The other submission put everything in SPY and failed the numeric
criterion (maximum weight error 0.8866028268304387). This is not persuasive evidence of
reliable agent utility: even the elementary controls mostly failed. Neither inverse-volatility
task was solved by either arm. Common failures were malformed responses, invented API names,
incorrect date/array handling, and attempts to write in the read-only environment. Report the
result as a failed development gate rather than claiming that the library reliably improves
model performance. The 14B diagnostic remains pending and is not part of this result.

The resource ceiling is 8m08s already consumed by the aborted job, plus a 45-minute cap on
the replacement 7B job and a 65-minute cap on 14B: at most 118m08s of single-GPU allocation.
The existing v3 32B job is separate and unchanged. There is no authorization here for the
full v4 trading campaign or an additional model sweep.

Completed model bundles are independently checked by `scripts/audit_utility_pilot.py`, which
reads CSV with the standard library and calculates inverse volatility using sample standard
deviations, without importing the experiment scorer or fin-skills. It verifies frozen hashes,
complete pairing, and every numerical verdict. Frozen source snapshots support replay after
later development changes.
