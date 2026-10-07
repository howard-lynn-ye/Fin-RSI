# Additional frozen v7 records

This publication adds the personally authored comparison's full decision ledger and the
Mistral Small 24B seed 11 pair completed after the preceding snapshot's cutoff. These are
frozen v7 records. Neither is a new v8/v9 repair result.

| Case | Without library Return Rate | With library Return Rate | Library minus raw |
|---|---:|---:|---:|
| Personal v7, one non-blind case | 38.334637% | 38.255686% | -0.078951 percentage points |
| Mistral Small 24B, seed 11 only | 57.535909% | 14.029114% | -43.506795 percentage points |

Each pair contains 44 decisions per arm over the reported January 3, 2025 to September 25,
2026 window. Orders execute at the following session's close, with five basis points charged
per traded side and zero cash interest. The original budgets, failures and allocations are
preserved. Three seeds are not independent market periods.

The personal case was already reported in [PERSONAL_V7_20261005.md](../../PERSONAL_V7_20261005.md).
Its 88 decisions were authored in the conversation; a script transported responses and did
the accounting. This case is non-blind and does not count toward the twenty-model target.
Mistral seed 11 is one completed pair. Seeds 23 and 37 are incomplete, and the old batch
stopped with a context-budget error; this publication does not count it as a completed model.

| Files | Contents |
|---|---|
| [Personal JSON](personal-v7-ledger.json) / [CSV](personal-v7-ledger.csv) | All 88 decisions; JSON also includes authored action notes and daily NAV |
| [Mistral seed 11 JSON](mistral-small-24b-seed11.json) / [CSV](mistral-small-24b-seed11.csv) | All 88 decisions, including failures; JSON also includes daily NAV |
| [Export script](export.py) | Read-only receipt checks and independent cash/unit replay using original source |

For every decision the ledger gives the decision and execution dates, submitted weights,
turn counts, failures, turnover, fee fraction and NAV. A null target (blank CSV weights)
means the recorded failed decision kept the existing holdings; it is not an all-cash order.
The personal JSON also preserves the explicit action notes recorded during the experiment.
These are the externally recorded notes, not reconstructed explanations.

Capital columns are presentation values: original unit NAV multiplied by a USD 100,000
reference account. The v7 protocol did not freeze this dollar amount. Return Rate is
`100 * (ending NAV / initial NAV - 1)` and is unchanged by the display scaling. The fee
fraction is charged against equity before that rebalance, not a dollar amount.

The export rechecked the frozen source and input hashes, all decision receipt bindings,
personal request/response bindings, recorded audits, daily NAV, and terminal Return Rate.
Original decisions were not regenerated. The user explicitly authorized local publication
checks on October 7 because Beacon's submitted-job quota was full. Source documents,
initial prompts and full per-turn responses are excluded from the public export.

The earlier [11-model, 33-pair snapshot](../20261007-multisource/README.md) remains unchanged.
This adds one completed model seed pair, taking those reports together to 34 verified model
pairs and still 11 fully completed models, plus the separately reported personal case.
Source coverage and historical archive vintages remain incomplete; accounting consistency
does not establish unseen performance or a causal library effect.

Local publication validation: both exports passed the original archived-source/input and
receipt checks and read-only independent replay. Two public-ledger regression checks passed,
covering JSON hashes, all 176 decision rows, CSV agreement, fees and terminal returns.
The repository's index build, package build and validator passed in the required order.
