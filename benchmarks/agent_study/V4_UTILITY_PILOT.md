# Financial computation utility pilot, 2026-10-01

This is a bounded development experiment, not the full proposed v4 trading study.
It tests whether executable financial tools improve accurate task completion after repairing
the study interface. It does not test or imply increased investment returns.

- Pinned Qwen2.5-Coder-7B, seeds 11 and 23; raw/library arms; 12 episodes total.
- Equal-weight control, 60-return inverse-volatility allocation, and 252-return allocation.
- Historical data through 2024-07-15 includes actual NVDA/AVGO splits. Both arms get identical
  tasks, visible files, generic tool instructions, turn limits and numerical tolerance.
- Library access and its method documentation are the treatment. The independent scoring
  reference uses vendor adjusted closes, never the library's submitted weights.
- Primary: correct target with maximum absolute weight error <= 0.001. Missing or invalid
  submissions fail. Report all episodes, errors, token usage and tool traces.
- No significance or out-of-sample alpha claim: six paired tasks/seeds share one history;
  historical data can overlap model training. No outcome-dependent task replacement.

`library_utility_pilot_v4 freeze NEW_DIR` freezes inputs, visible data, protocol and source
hashes. `run` requires the CPU qualification to pass before inference, records its receipt,
and never overwrites completed decisions. `score` requires every planned decision and checks
hashes. All commands are Python module invocations under `benchmarks.agent_study`.

The v3 scorer and existing evidence stay unchanged. The v4 runtime independently fixes the
initial-capital anchor, allows <=0.001 rounding excess only after normalization, supports
Python submit and identical JSON/Python tool signatures, and applies an explicit single-linkage
default in the HRP study adapter (the underlying library contract is unchanged).

Linux qualification must deny real future-source and hidden-ledger reads in both arms,
deny network and writes, block raw imports of fin_skills, and exercise HRP/inverse-volatility
through both interfaces. A Windows skip is not a passed sandbox test.

Full trading v4, manuscript changes and slide changes are outside this pilot.

## Transport amendment before scoring

Initial job 1810695 was stopped before numerical grading when its traces exposed an adapter
regression: `{"tool":"list_algorithms"}` was rejected because the empty `arguments` key
was omitted. The new protocol accepts omitted/null arguments as an empty object in both
dispatch paths and explicitly qualifies this case. Inputs, model, seeds, tasks, budgets and
scoring tolerance are unchanged. Partial first-attempt records remain in their original
directory; the replacement runs in a new directory and is reported separately.

## Same-task model diagnostic

Before inspecting numerical scores, 7B failed to submit on both seeds of the raw equal-weight
control and on one library control. Traces showed malformed code, date-column errors, and
attempts to write files despite a read-only contract. A single 14B diagnostic is prepared with
the same tasks, seeds, prompts, tools, limits and scoring. This is an adaptive development
choice, not an independently preregistered confirmatory replication. Both models and every
failure must be reported. The original source snapshot remains pinned for the active 7B run.

The 14B run may consume only the remainder of the original two GPU-hour pilot ceiling,
including the aborted attempt. No full trading campaign or further model sweep is included.
Select the model only at freeze time with `freeze NEW_DIR --family 14b`.
