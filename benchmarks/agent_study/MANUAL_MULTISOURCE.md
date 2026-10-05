# Assistant decisions with public financial evidence

This retrospective case compares decisions made personally by the chat assistant with
and without new calls to fin-skills. The program prepares information, computes requested
statistics, validates explicit submissions and keeps the account. It contains no trading
policy, automatic allocation or model call.

## Execution status

All 44 dates and both arms are complete: 88 explicitly authored decisions, locked before
the first final score was opened. The valuation period is 2025-01-03 through 2026-09-25.

| Personal decision arm | Net Return Rate | Maximum drawdown | Mean daily equity weight |
| --- | ---: | ---: | ---: |
| No new library calls | 30.8924% | -10.8824% | 69.8912% |
| Library tools available | 31.0371% | -10.6185% | 69.6726% |

The library arm finished **0.1446 percentage points higher**. This is a small descriptive
difference in one non-blind retrospective case, not evidence of general return superiority.
An independent dollar/units account reproduced all 433 daily valuations, with maximum NAV
difference below 1.1e-14. Its Return Rates agree to much better than the reported precision.
The two arms made 25 and 26 rebalances, respectively. Actual fee deductions summed to
0.26197 and 0.26205 percentage points of initial capital; this arithmetic fee sum is not a
counterfactual compounded return loss. Cash earns zero in both accounts.

The 16 protocol tests passed on Beacon in job 1904516. Repository regeneration, validation
and 71 related tests passed in job 1905516; two independent accounting tests and the full
reconciliation passed in job 1905610. An independent code review confirmed
the fixes that make displayed holdings use the accounting price series and validate the
frozen protocol on every subsequent packet and final score. This is a bounded check of
these mechanisms, not a claim that every possible defect has been excluded.

## Common information

Both arms receive the same eligible records and market summaries, with source URLs,
availability assumptions and record ages. The underlying archive separately retains content
hashes and actual observation timestamps. Acquisition runs through Slurm. The source collector
and normalizer preserve original bytes on Beacon. Observed coverage
in the frozen evidence manifest is:

| Channel | Included records | Meaning and limits |
| --- | ---: | --- |
| News | 17 | Official Federal Reserve monetary releases; not a complete corporate or breaking-news feed |
| Psychology | 8 | Public Schwab survey reports; respondent expectations, not the entire investor population |
| Behavior | 832 | CFTC market-week records for eight futures markets; not ETF flows or unambiguous directional conviction |
| Officials | 50 | Text-readable House filings for Nancy Pelosi, Rohit Khanna and Michael T. McCaul; a convenience sample, not a complete official portfolio |

Thirty-three filings with insufficient text were excluded pending OCR; two amendments
were excluded instead of silently replacing originals. A missing ticker match in extracted
text does not establish the absence of holdings or trades. Option purchases, exercises,
stock sales and annual reporting-year holdings must be interpreted separately.

Sources: [CFTC archives](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm),
[CFTC delayed-release notice](https://www.cftc.gov/PressRoom/PressReleases/9147-25),
[Federal Reserve calendar](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm),
[Schwab press releases](https://pressroom.aboutschwab.com/press-releases/), and
[House disclosure search](https://disclosures-clerk.house.gov/FinancialDisclosure/ViewSearch).
Exact document URLs and checksums are retained with each normalized record.

## Availability assumptions

This archive was collected in October 2026. Its historical versions have not been
independently verified. The case therefore uses an explicitly labeled retrospective mode:

- Dated Fed and Schwab pages become eligible the following calendar day.
- CFTC report dates receive a 14-day lag, extended for the documented shutdown delays.
  The release override is a schedule, not a verified posting timestamp.
- House filing dates receive a three-day lag. The actual public posting time is unknown.
- Survey publication dates and fieldwork dates can differ substantially. The latter may
  appear in the retained source text but are not a required structured field in this version.

`asof_records(..., strict=True)` rejects these unverified historical backfills. They must
not be relabeled as prospectively archived observations. All 44 dates passed the four-channel
coverage check in retrospective mode; passing does not imply that every channel is recent.

The August 2025 Fed framework-review release shares the collector's monetary-series label
with policy statements. In this frozen case the assistant explicitly treated it as a
framework review, not a rate decision. Future collectors should separate these series;
the current evidence must not be silently rewritten during this case.

## Decisions and accounting

The universe is SPY, QQQ, EFA, EEM, TLT, IEF, GLD, DBC, NVDA, AVGO, ORLY and FAST. The
calendar starts on 2025-01-02, advances every ten market sessions for 44 decisions and ends
with valuation on 2026-09-25. These are the existing study's frozen market files.

At each date the assistant reads the common packet, explicitly locks the raw-arm decision,
requests library calculations and then explicitly locks the library-arm decision. Only then
can the next packet be opened. `null` means hold drifted positions, not sell to cash. Each
decision includes the assistant's rationale, evidence identifiers, packet hash and preceding
decision hash. Duplicate submissions and partial-path scoring are refused.

Library calculations include adjusted history, an HRP candidate, historical VaR/expected
shortfall, covariance, correlations and variance contributions. Diagnostics are computed for
the raw proposed allocation, not automatically for the eventual library allocation. Additional
21/63-day volatility calculations were explicitly requested at two decision dates. The
assistant interpreted the results and chose every final target; HRP was never an automatic
submit rule.

Trades execute at the next session's close, after that day's return. Both arms use the
same accounting series and charge five basis points per traded notional, including the
initial purchase. The account starts at 1.0; cash earns zero and no additional market-impact
model is included. The reported outcome is:

`Return Rate (%) = 100 * (final net NAV / initial NAV - 1)`

The comparison is library Return Rate minus raw Return Rate in **percentage points**.
Lower stock exposure, differing hedge weights and costs must accompany its interpretation.

## Reproduction and continuation

Install the benchmark dependencies with `python -m pip install -e ".[study,dev]"`.
The modules require numpy, pandas, scipy, beautifulsoup4 and pypdf. On a compute
allocation, use `collect_manual_sources.py ROOT`, then
`manual_multisource.py normalize ROOT`, `prepare ROOT --data-dir DATA`, and
`serve ROOT --data-dir DATA`. Each output directory must be new; collection/normalization
must not replace the evidence of a started run. The server consumes explicit request JSON
files with `action` in `packet`, `commit`, `library`, `score`, or `stop`; it never makes a
decision while idle. `commit` arguments are `arm`, `weights`, `rationale`, and `evidence_ids`.

Current private run root:
`/beacon-projects/radfm/wy891/fin-manual-multisource-20261005`.
The executing module is the frozen `manual_multisource_v3.py`, identical to this repository
module when the protocol was frozen. Earlier preflight versions contain no trading decisions.
Retain `protocol.json`, evidence, packets, decisions, library calls and hashes together.

The completed case must not be resumed, altered or rescored as a new experiment. The
workstation's shared connection pause interrupted execution at 34 completed dates. The user
explicitly authorized one read-only recovery check; it succeeded at 19:57:44 UTC, and the
remaining ten dates continued from the existing records. The preceding `srun` allocation
communication failures were handled with the already-running worker's inbox; no decisions
were regenerated. The pause and recovery are retained in the workstation incident archive.

The last decision hash is
`980ecd5130dac38397da26c907ff54b4b5df84775c3e6585dc8aea57f0062f79`.
`results.json` and `independent-audit.json` remain with the frozen private run. No source
articles, disclosure PDFs, historical market files or full decision packets are included
in the public repository.

## Separate model study

`multisource_model_study.py` supplies the same dated evidence directly in both arms' initial
user messages. It retains the original 7B/14B models, three seeds (11, 23, 37), market window,
costs and eight-call budget; the library arm also receives a dedicated library guide and
the v6 tools. The system instructions explicitly permit both the visible market files and
the supplied evidence. Every decision stores the actual initial messages, guide receipt
and evidence-packet hash. The
worker must fail attempts to open the complete future-evidence archive before GPU execution
is allowed. Failed model submissions hold existing positions and remain in the denominator.

Its separate run root is
`/beacon-projects/radfm/wy891/fin-multisource-models-20261005/r2`.
The initial CPU preflight (1905670) was stopped before model inference after review found
contradictory inherited price-only instructions and an aggregate function that could accept
older experiments. Both were fixed. The corrected CPU job 1905777 passed regeneration,
validation and 37 related tests, including both arms' actual initial messages. A separate
review confirmed those two fixes. Qualification runs separately for all six pairs.

Aggregate reporting requires all six model/seed pairs with matching evidence, market data,
code and protocol, plus score-to-inference-to-decision hash checks. It refuses older study
versions or a favorable completed subset. This model study is not the personally authored
case; no new model Return Rate is available until inference and scoring finish.

On a compute allocation, use the module commands `freeze PAIR --family 7b --seed 11
--evidence EVIDENCE --allow-retrospective`, `qualify PAIR`, `run PAIR`, and `score PAIR`.
Repeat with the declared models and seeds, then `aggregate BATCH`. Each pair directory
must be new. The explicit retrospective flag acknowledges the availability limitations;
it does not establish verified historical vintages.

## What this case cannot establish

The assistant had already seen library knowledge and old aggregate results; its two arms
share one chat context. This is a non-blind retrospective case with different reasoning and
tool budgets from the 7B/14B studies. It cannot isolate a causal library effect, establish
out-of-sample alpha or replace new model runs with the richer inputs. The model reruns have
not yet been completed with this bundle. Papers and slides are outside this work item.
