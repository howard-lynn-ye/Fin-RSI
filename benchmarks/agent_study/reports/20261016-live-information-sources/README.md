# Four-category live information collection — 2026-10-16

The library previously offered generic page transport for investor psychology and behavior,
without structured observations from those pages. This change adds public Michigan/AAII surveys and CFTC
TFF positioning adapters, wires them into the same public tool registry and persistent runner,
and provides `examples/collection-live.json` for the four required information categories.

## Installed-wheel acceptance

The candidate source is `74789f3827294bc221e1ab65081893c9813026d7`. A freshly built wheel was installed into a separate
environment and run from outside the checkout with `python -I`. The five changed runtime/tool
files were byte-compared across the committed source, wheel and installed package;
`installed-binding-v2.json` records their hashes, the wheel hash and dependency versions.

The real-source run started at 2026-10-16T17:55:18.971260+00:00 and finished at 2026-10-16T17:56:25.078104+00:00.
It saved **68 records** initially and **68 records** after
reopening SQLite and letting every required watch repoll on its 60-second acceptance schedule.
All source reports were OK. The bounded collector then stopped; no persistent service was
installed by this verification. Normal example intervals are 5 minutes for news, 1 hour for
Michigan/CFTC, and 30 minutes for House.

| Watch | Adapter | Saved records | Final source status |
|---|---|---:|---|
| fed-news | rss | 20 | ok |
| ecb-news | rss | 15 | ok |
| survey-michigan | umich | 3 | ok |
| positions-cftc | cftc | 20 | ok |
| house-disclosures | house | 10 | ok |

| Required category | Eligible records | Records using first-observed availability |
|---|---:|---:|
| news | 4 | 0 |
| psychology | 1 | 1 |
| behavior | 5 | 5 |
| official_disclosures | 8 | 8 |

These are actual public HTTP responses parsed by the installed library. They are not trading
decisions or Return Rate measurements. Unchanged data on a second poll is a valid deduplication
result; this short run does not prove detection of a newly published release. Source failure,
coverage gaps or partial parsing would make the acceptance fail rather than silently pass.

The first candidate passed its initial fetch but failed its scheduled AAII refresh. Inspection
found an HTTP-200 automated-access challenge, not a survey layout. That failed receipt is
preserved as `live-initial-failed.json`; it is not counted as a pass. AAII remains optional and
reports that restriction with a 24-hour retry delay, without bypass. The final default profile
uses the university's explicitly published CSV, measuring consumer psychology rather than
silently substituting an AAII investor reading.

## Implemented contracts

- Michigan parses the public current-table CSV into consumer sentiment index points, retaining
  month and preliminary/final status. Re-fetching detects revisions while preserving historical
  cutoff isolation. This measures US consumers; it is not the AAII investor population.

- AAII extracts the latest public bullish, neutral and bearish shares, preserving units, survey
  week and population. Historical averages are not mistaken for current readings. Missing panels,
  impossible percentages and future survey periods fail. Robots checks and a ten-second request
  floor are respected. This is a weekly respondent survey, not a measurement of every investor.
- CFTC queries the public TFF futures-only API for a selected contract and bounded recent window.
  Long/short counts, net contracts, open interest, contract units and trader groups are retained.
  Missing fields never become zero. Refreshing the window captures revisions in that window;
  it does not create a complete historical vintage archive.
- House supports the current filing year and a rolling lookback, with newest-first retrieval.
  Unparsed PDFs retain their evidence and retry with backoff instead of becoming permanently
  checkpointed. The current-year setting rolls forward; late updates to an older year's index
  require an additional explicit watch. Parsed PDFs remain checkpointed; reparsing after a parser
  upgrade uses a new watch ID. The example watches Pelosi, not every official.
- Polling, retry state, revision deduplication, restart persistence, and RAG snapshot refresh
  reuse the existing collector and database. Source/tool discovery and model-visible skill
  instructions include the new adapters. Importing/installing the package starts no background
  activity; `run` or an explicitly scheduled `collect_once` keeps data current.

## Availability and coverage correction

AAII, Michigan, CFTC and House do not supply reliable exact publication timestamps in these responses.
`published_at` therefore remains null. Their actual observation time establishes availability,
while survey week/month, positions-as-of date and filing date conservatively measure evidence age.
The coverage check accepts this explicitly labeled, source-specific contract with a matching
public host. Generic undated records remain ineligible. It reports
`publication_time_complete=false`, observed-time counts and null unknown observation lags.
Reference dates are never used to backdate access. A report first fetched today cannot enter
a historical decision before today, and re-fetching an old report cannot make it fresh.

The default behavioral age bound is now 14 days because weekly CFTC Tuesday snapshots normally
arrive Friday: the prior snapshot can already be ten days old just before the next release.
The psychology bound is 45 days because a month-start reference can remain current until the
next month's preliminary release. These are explicit freshness policies, not trading-score adjustments. Source publication
delays can still produce stale or missing coverage; polling does not remove those delays.

## Verification and limitations

Local CPU acceptance: **244 passed, 2 Linux-confinement cases skipped**. This includes **28 new
source regressions** plus existing collector, health, coverage, RAG, model-tool and MCP checks.
Tests inject changing responses and verify scheduled refresh, changed-record revisions,
restart deduplication, future-cutoff isolation, unparsed retries and registered configuration.
The required index/package/validation sequence passed (131 skills, one existing discovery-budget
warning). A no-isolation wheel build initially found old local setuptools and missing wheel;
the successful build used declared dependencies in an isolated build environment instead.

The receipt publishes counts, dates, provenance links and health. Downloaded article bodies,
PDF text and the local SQLite store are not committed. SEC remains an optional, separately
configured source and was skipped without a caller identity. No paid feed or model was used.

No Beacon connection, Slurm job submission or trading run was made. The frozen 32B experiment's
source, inputs, parameters and ledger are untouched. There is no new Return Rate in this patch.
Continuous collection on a production host is a separate deployment; this bounded verification
does not leave a daemon running or consume another Beacon job slot.

For the paper, finish and audit the existing frozen 32B pair first. Any comparison using these
new inputs must be a separately frozen protocol with matched information availability, trading
rules and costs, publishing both final capital and full decisions/ledger. New live observations
do not retroactively supply point-in-time evidence for an older backtest. Dataset/tool/RAG/RSI
ablations should isolate contributions; higher library Return Rate remains an empirical claim.

Primary sources inspected: [Michigan tables/CSVs](https://www.sca.isr.umich.edu/tables.html), [AAII](https://www.aaii.com/sentimentsurvey),
[CFTC API guide](https://publicreporting.cftc.gov/stories/s/COT-Help/p2fg-u73y/),
[TFF API](https://publicreporting.cftc.gov/resource/gpe5-46if.json),
[House](https://disclosures-clerk.house.gov/FinancialDisclosure).
