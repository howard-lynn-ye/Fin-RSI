# Research-call repair and 32B return audit, 2026-10-17

The completed, unchanged 32B seed-11 pair returned **38.003847% with fin-skills**
and **18.976928% without it**, a difference of **19.026919 percentage points**.
Both arms started with USD 100,000. These are cumulative returns over
2025-01-03 through 2026-09-25, not annual rates or a complete economic cycle.
The original results and all derived decisions, trades and daily NAVs were merged
in PR #58. The new interface below has no completed return result yet.

## Why the positive return is not itself evidence of exceptional alpha

An independent units-and-cash replay exactly reproduced both original NAV paths.
Using the same dates, instruments, execution close and 5-bps entry cost:

| Path | Cumulative Return Rate |
|---|---:|
| Original 32B with library | 38.003847% |
| Original 32B without library | 18.976928% |
| Equal-weight buy and hold, all 12 instruments | 39.445545% |
| SPY buy and hold | 32.800125% |
| QQQ buy and hold | 44.712273% |

These baseline calculations are post-hoc descriptive checks, not newly selected
primary endpoints. `diagnostic.json` includes buy-and-hold results for **every**
instrument, including negative TLT performance, plus complete asset contributions
and trading fees for both model arms.

The library path gained 7.91 percentage points of initial capital from SPY,
7.63 from QQQ, 6.78 from GLD and 6.37 from EEM. Trading fees charged totaled 1.39 points of initial capital. The raw path lost 5.92 points on AVGO and 3.44 on ORLY;
fees cost 1.99 points. These describe the realized holdings; they do not identify
which tool, model error or knowledge passage caused the difference. The library
path did not beat equal-weight buy and hold on this market path.

## Information boundaries and the cheating question

The accounting identity passed. That is not proof of a leakage-free experiment.
The replay explicitly uses **retrospective availability assumptions**, not verified
archival vintages. Current installed skill knowledge can postdate observations;
the selected market path is a development path, not an unseen holdout. Preflights
also sample dates from this path. Do not label this result a strict historical
blind test or use it alone to claim robust alpha.

The additional boundary audit reconstructs each frozen dated evidence snapshot,
checks declared eligibility, and compares its hash against each original decision
receipt in both arms. This checks supplied dates and record bindings, not original
publication time, revised historical content, model memorization or hindsight in
current skill text. The original records are retained; no decision is rewritten.

## Observed failures and the new interface

There were zero failed final decisions among the 44 decisions per arm. Research
turns still contained 21 library failures and 33 raw failures. The raw path had
one date on which every attempted research call failed. `parse=json` indicates
that JSON was recognized; it does **not** mean a JSON syntax error.

Observed errors include missing scientific imports, indexing market data with
CASH, referencing prior-response variables, mixing date strings with datetime
objects, an obsolete Pandas method, optimizer dimension mismatch and truncated
model programs. One inspected failed raw program also attempted to label optimized
values with a ticker order different from its pivoted matrix. This is a potential
model-authored alignment mistake; the failed program does not establish that its
mislabelled weights were executed. No counterfactual corrected return is reported.

Revision `20261017-research-environment-v11`:

- Prebinds `np`, `pd`, `sco` and `json` in both arms' fresh Python environments.
- Describes CASH separately from traded assets, ISO date indexes, matrix/ticker
  alignment and current Pandas concatenation syntax in the common contract.
- Gives specific error feedback for cash indexing, date comparisons, removed
  `DataFrame.append` and inconsistent optimizer dimensions.
- Keeps tool receipts, failures, strategy choice, cash choice, response budgets,
  historical-data restrictions and final-action validation intact. It does not
  inject a portfolio, preserve undefined variables or silently repair model code.

These repairs address observed interface friction. They cannot guarantee that
arbitrary model-generated Python or investment reasoning will always be correct.
Frozen v9 results remain separate from new v11 experiments.

## Twenty-model execution and publication

The existing pinned matrix contains 20 enabled open-weight models, covering
Qwen, Mistral, Phi, Granite, OLMo, Falcon, SmolLM and a DeepSeek-distilled model
(whose base architecture is Qwen). All declared paired seeds 11, 23 and 37 are
required to count one model as complete. Aliases, failed preflights and partial
cycles do not count. Closed or unverified paid/subscription routes remain off.

`serial_model_cycle.py` can run one complete paired cycle per allocation and keeps every
existing acceptance gate. It verifies available pinned weight shards before
starting; it performs no downloads, paid calls, scheduler submissions or cache
deletions. It refuses unreviewed incomplete pairs instead of blindly retrying.
The tested recovery writer publishes complete JSON atomically. Each finished pair
exports both returns and all derived decisions, holdings, trades and daily NAVs,
with hashes binding them to the private original conversations.

Window owner: `01a0d6bf-630f-7930-ae86-c2a9841d940f`. This window permits at most
one running allocation; other windows' jobs are not counted against that limit.
Scheduling and actual new-model progress are recorded in `execution.json`.
The completed diagnostic CPU job was 2365211. First validation job 2365476
passed builds but never ran tests because its source copy lacked test files;
the corrected package includes the selected tests and their helper imports.
Source verification also caught an older skill document in the remote build copy
and a local CRLF-only difference in the generated index. The document was synced
from its upstream plugin and rebuilt; the index was compared against canonical
LF build/Git content. All 602 expected source files then matched exactly. No hash
check was disabled, and all 88 original evidence/decision bindings passed.

Thirteen pinned models have complete weight files in the known project caches.
Seven require staging. The shared volume had about 49 GiB available at inspection;
that is insufficient to download another complete BF16 32B model with a finishing
margin. Cached models can progress while this storage constraint is resolved.
Scavenger capacity was considered. The new matrix's interrupted-cycle recovery
has not been qualified across its model families, so the first stage uses ordinary
capacity with no automatic requeue. GPUs are untyped, not pinned to H200 or a node.

Only derived experiment artifacts are public. Private source/news text, market
archives, credentials and machine-specific cache configuration stay off GitHub.

Validation: 132 related regressions passed in job 2365768; the subsequent split-stage driver passed 11 checks in job 2366235. Both required build sequences passed. Job 2366254 was submitted for the first v11 7B pair and was pending with reason Priority at submission. There is no completed v11 return yet.
