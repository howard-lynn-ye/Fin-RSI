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

Full trading v4, additional GPU batches, manuscript changes and slide changes are outside
this pilot. Any expansion follows the complete pilot report and an explicit resource decision.
