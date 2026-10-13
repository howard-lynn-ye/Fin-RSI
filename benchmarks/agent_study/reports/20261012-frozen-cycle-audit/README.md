# Frozen full-period results independently audited on October 12, 2026

Both batches completed 15 model/seed pairs: the same five models, seeds 11/23/37,
44 dates in each arm, USD 100,000 initial capital and unchanged original decisions.
These are two protocols on five distinct models, not thirty distinct models.
All listed pairs passed source/input/decision receipt validation and an independent
cash-and-units replay against daily NAV and ending capital. Audit job: `2124480`.

The primary metric is `100 * (ending capital / initial capital - 1)`.
The tables show the arithmetic mean of three cumulative period Return Rates;
they are not annualized returns. Failure counts pool the three seeds (132 dates).

## October 7 final-stage protocol

| Model | No library Return Rate | Library Return Rate | Difference (pp) | Failed dates: no library / library |
|---|---:|---:|---:|---:|
| 7b | 46.30% | 41.50% | -4.80 | 72/132 / 75/132 |
| falcon3-10b | 31.34% | 35.65% | +4.31 | 44/132 / 14/132 |
| granite-8b | 44.16% | 41.52% | -2.64 | 40/132 / 21/132 |
| qwen3-4b | 43.67% | 32.22% | -11.45 | 22/132 / 50/132 |
| smollm3-3b | 6.27% | 0.00% | -6.27 | 12/132 / 101/132 |

Files: [final-contract/summary.json](final-contract/summary.json) and [final-contract/ledgers.json](final-contract/ledgers.json).

## October 8 explicit-allocation protocol

| Model | No library Return Rate | Library Return Rate | Difference (pp) | Failed dates: no library / library |
|---|---:|---:|---:|---:|
| 7b | 36.24% | 34.55% | -1.69 | 2/132 / 2/132 |
| falcon3-10b | 28.27% | 21.79% | -6.48 | 91/132 / 95/132 |
| granite-8b | 28.99% | 21.28% | -7.71 | 116/132 / 117/132 |
| qwen3-4b | 20.16% | 20.99% | +0.83 | 0/132 / 20/132 |
| smollm3-3b | 2.43% | 8.50% | +6.07 | 73/132 / 115/132 |

Files: [explicit-allocation/summary.json](explicit-allocation/summary.json) and [explicit-allocation/ledgers.json](explicit-allocation/ledgers.json).

## What this establishes

- The published returns match the original execution ledger. This audit did not find a cash, units, NAV or return-summary discrepancy.
- Interface failures remain substantial and model-dependent. Failure leaves existing units unchanged; it is separately labeled and is not a successful HOLD.
- These are outcomes of the old end-to-end systems. They cannot establish the isolated value of a numerical algorithm or of all library capabilities.
- Lower failure counts do not guarantee higher returns. Both asset selection and exposure still come from the model.
- No loss amount is attributed to a bug without a controlled counterfactual. Rejected instructions were not converted into trades.
- Both positive and negative differences are retained. No failed models, seeds or dated decisions were dropped.
- Historical source eligibility uses declared backfill assumptions, not verified point-in-time vintages. This remains a data-validity limitation.
- Neither batch uses the October 12 three-way-review repair. That revision needs separate real-model qualification and new full-period results.

The original raw responses, prompts and evidence text remain on Beacon. Public ledgers contain original targets,
decision dispositions, daily NAV/trades and source fingerprints, without publishing those private full texts.
The read-only audit script is included for reproducibility. It checks the original stored audit instead of overwriting it.
