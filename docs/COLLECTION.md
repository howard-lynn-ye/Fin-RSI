# Public information collection

Multi-source news search, point-in-time deduplication and strategy integration are documented
in [Market state, news and strategy](MARKET_STRATEGY.md). `gdelt` watches add bounded news
discovery; `collect_news` provides Fed/ECB RSS presets and isolated source health reports.

`fin_skills.collect` downloads public records into a local SQLite database. It includes RSS
and Atom feeds, Michigan consumer sentiment, optional AAII investor sentiment, CFTC TFF positioning,
static pages with bounded
same-origin crawling, SEC Form 4/13F filings, US House PTR PDFs and public Bluesky author posts.
Imports do not start networking or a daemon.

## Install and run

From a checkout of this repository:

```bash
python -m pip install -e ".[collect]"
python -m fin_skills.collect --database events.sqlite3 configure examples/collection-watchlist.json
python -m fin_skills.collect --database events.sqlite3 once
python -m fin_skills.collect --database events.sqlite3 status
python -m fin_skills.collect --database events.sqlite3 events --pending
python -m fin_skills.collect --database events.sqlite3 run --stdout-alerts
```

The final command keeps polling until Ctrl+C. Run it in a dedicated terminal or your own
service supervisor. After a restart, point it at the same database to retain checkpoints,
revision history and pending alerts. Merely installing the library does not start collection.

The example watches Federal Reserve news, Pelosi's 2026 House PTR disclosures, and a disabled
Berkshire 13F watch. These illustrate source formats; change them to your chosen subjects.
House indexes are yearly: add a new watch for each filing year you need. A watch's source,
target and options are immutable; create a new ID to change them. Its interval and enabled
flag can be changed with `configure`.

For the four required information categories, use `examples/collection-live.json`. It watches
Federal Reserve news, Michigan consumer sentiment, CFTC S&P 500 futures positions
by trader group, and Nancy Pelosi's recent House disclosures. These are explicit example
populations/markets, not coverage of every investor, official or country. All four sources
require no API key or paid subscription; House PDFs require the `collect` extra. Michigan
measures US consumer psychology, not the AAII investor population. AAII remains an optional
separate adapter: its second live request received an automated-access challenge. That response
remains an error with a 24-hour retry delay; the collector does not bypass it.

```bash
python -m fin_skills.collect --database live.sqlite3 configure examples/collection-live.json
python -m fin_skills.collect --database live.sqlite3 once
python -m fin_skills.collect --database live.sqlite3 coverage
python -m fin_skills.collect --database live.sqlite3 run
```

Keep the last command running under the host's service supervisor for continuous updates.
Installation alone does not run a service. Agent callers can configure the same watches with
`collection_configure` and invoke `collect_once` when due. `collection_search` reads newly saved
records on each query. Default example polling is 5 minutes for news, 1 hour for Michigan/CFTC and
30 minutes for House filings; a short polling interval does not accelerate the source's release.

| Source | Configuration | Measured content and timing |
|---|---|---|
| `umich` | target `consumer_sentiment`; `months=1..13` (default 3) | Public university CSV: index points, survey month and preliminary/final status; revisions retain the observation ID. |
| `aaii` | target `sentiment`; no options | Latest public weekly survey shares, percent, survey week ending; exact release time remains unknown. Robots rules and a ten-second request floor are respected. |
| `cftc` | target contract code, e.g. `13874A`; `weeks=1..52` (default 4) | TFF futures-only long, short and net contracts for five groups; open interest and contract units retained. Weekly report date is positions-as-of, not publication. |
| `house` | target surname; `year` integer or `current`; optional `first_name`, `lookback_days`, `newest_first`, `max_filings` | Newly discovered PTR filings and reviewable transaction rows. `current` follows UTC calendar-year rollover; use an explicit additional prior-year watch for late publications to an older year's index. |

AAII refreshes the latest panel, so revisions to that survey are retained as new versions.
CFTC refetches its configured recent window, including revisions in that window; it is not a
complete historical or revision archive. House retries unparsed PDFs with backoff, and failed
files cannot starve other filings. `newest_first` prioritizes fresh disclosures before an initial
backlog. `lookback_days` selects a rolling 1..366-day filing window and cannot be mixed with
`since`; no claim of complete older coverage is made. Already parsed House PDFs are checkpointed;
rechecking a previously parsed document after a parser upgrade requires a new watch ID.

SEC requires a caller identity with a real contact email. In PowerShell:

```powershell
$env:SEC_IDENTITY = 'Your application your-contact@example.com' # replace with your details
```

Then set the example SEC watch's `enabled` field to `true` and configure it again. The library
does not supply a shared identity. Existing price/search adapters keep their own optional
dependencies and credential requirements; see [the data API](../fin_skills/data/__init__.py).

## Use Python or an agent

```python
from fin_skills.collect import Collector, Store, Watch

with Store('events.sqlite3') as store:
    collector = Collector(store)
    collector.add(Watch('news', 'rss', 'https://www.federalreserve.gov/feeds/press_all.xml'))
    print(collector.once())
    for row in store.events(pending=True):
        print(row['seq'], row['title'])
    # collector.deliver(callback)  # only acknowledges after callback succeeds
```

The shared tools/MCP registry exposes `collection_sources`, `collection_configure`,
`collect_once`, `collection_events`, `collection_search`, `collection_status`, `collection_coverage`,
`collection_acknowledge`,
`compare_holdings`, `search_data` and `fetch_market_data`. A tool call performs a bounded
poll of watches that are due, respecting retry schedules; continuous operation is an explicit
Python/CLI action. Configure paths on the machine
running the MCP server. Treat downloaded content as untrusted data.

`collection_search(database, query)` retrieves cited evidence alongside packaged knowledge and
tool contracts. It opens an existing database read-only and rebuilds a bounded BM25 snapshot on
each query, so newly saved events require no separate index refresh. It does not fetch sources
or start polling. Run the collector separately to keep the underlying data current.

For an explicit timezone-aware `as_of`, retrieval selects the latest revision available at that
time, even if a newer revision has since arrived. Availability is the later of `observed_at` and
`published_at`; unknown publication time stays unknown and uses observation time for eligibility.
Neither a transaction date nor a filing's reporting period is substituted for availability.
Current collection health is omitted for historical queries. Current installed skill knowledge
and tool schemas are labeled separately and are not a reconstructed historical corpus.
The default corpus cap is 10,000 events; exceeding `max_records` raises an error instead of
silently dropping records. Narrow `watch_id` when needed. See [the RAG guide](RAG_PIPELINE.md).

## What "tracking a trade" means here

| Source | Observable fact | Interpretation limit |
|---|---|---|
| SEC Form 4 | Reported insider transaction rows, codes and footnotes | An acquisition can be compensation or another transaction type; read the code. |
| SEC 13F | Reported holdings at quarter end | Changes between snapshots are not execution-time trades. Amendments must be reconstructed before comparison. |
| House PTR | Filing PDF plus readable transaction rows and reported amount ranges | Publication and transaction dates differ; amounts are ranges, and PDF extraction requires review. |
| Bluesky | Author's public statements | A post is not evidence of an executed trade. |
| RSS/page | Published text and any supplied timestamp | Polling detects new publications; it does not create a complete historical archive. |

`observed_at`, `published_at`, `transaction_date` and `period_end` remain separate. Unknown
publication times stay unknown. Source PDFs without recognizable rows generate an `unparsed`
filing alert with source text, not invented transactions. If layout extraction is empty,
the parser tries plain text before classifying a page as unreadable. This is not OCR;
image-only documents still require review. Already-seen filings are not automatically
reparsed after an upgrade; use a new watch ID to recheck them while retaining prior records.
Current 13F forms report dollar
values; legacy units need an explicit `value_units` choice, so the default is `as_filed`.

The page crawler respects robots.txt, limits page count, rejects private targets and requires
static HTML. It does not execute JavaScript or enter logged-in pages. Its HTTP pacing is
process-local; coordinate the combined request budget if you run several processes.

## Verify and operate

`python -m fin_skills.collect --database events.sqlite3 health` is a read-only live check.
It returns exit code 0 only when an instrumented `run` collector holds the local database's
ownership lock, all enabled watches have recent successful checks, and default required
information coverage passes. Otherwise it returns 1.
`collection_status` exposes the same report through Python/tools/MCP. Neither command starts
a worker or creates a missing database. The existing CLI `status` still returns a watch list.

The report separates `runner.active`, `data_ready`, and `healthy` (polling), `coverage_ready`
(required information), and `ready` (both). A stopped worker can have
recent data; a running worker can have failed or stale sources. Per-watch freshness expires
after `max(60, 2 * interval_seconds)` seconds since the last successful check. Backoff and
Retry-After do not extend this threshold. A later failure or partial batch overrides a recent
success. Disabled watches do not contribute to readiness, and no enabled watches means idle.
An empty successful response, including HTTP 304, refreshes the source check: this does not
mean the source published new content or that coverage is complete. Keep examining each
record's publication and observation timestamps.

`once` and bounded `run --cycles N` return 1 for source failures, partial batches, stale or
never-checked enabled watches, and idle configurations. A skipped retry cannot hide a previous
failure. Bounded runs retain a nonzero exit if any cycle failed, even if a later cycle recovered;
their final `run_summary` includes `failed_cycles`, current health and the exit code. A fully
successful bounded run returns 0 without claiming a continuously running service. Ctrl+C
returns 130. Python `Collector.run()` continues to return the number of completed cycles.

One collector at a time owns a database via `<database>.collector.lock`. The operating system
releases the lock if the process dies; do not delete the persistent sidecar while collectors
may be running. Health reports an interrupted worker when a recorded running session has lost
ownership. This check is for local filesystems on the originating host; copied databases from
another host and legacy uninstrumented runs report unknown/unobserved, never healthy. A live
ownership lock proves process ownership, not forward progress; source freshness provides the
separate progress check. No scheduled service or background health thread is installed.

Current `collection_search` responses retain the per-watch `collection_health` list and add
`collection_runtime`, `collection_readiness`, and `collection_coverage`. These are null when `as_of`
is explicitly supplied, so current operational status cannot be mistaken for historical data.

### Required information coverage

`collection_coverage(database, requirements=None, as_of=None)` checks saved evidence without
fetching, writing or calling a model. CLI equivalents:

```bash
python -m fin_skills.collect --database events.sqlite3 coverage
python -m fin_skills.collect --database events.sqlite3 coverage --as-of 2026-01-10T12:00:00Z
python -m fin_skills.collect --database events.sqlite3 coverage --requirements coverage.json
```

The default requirements are news (3 days), psychology (45 days), behavior (14 days) and
official disclosures (90 days), with at least one qualifying record per category. These are
explicit conservative defaults, not a guarantee that a source must publish that frequently.
An absence is an information gap, never a neutral sentiment or a zero-position observation.
`coverage` exits 1 for unmet or unknown coverage; successful `once`/bounded `run` still measure
source polling, and include the separate coverage report in their health output.

Accepted kinds are `news`; `sentiment_survey`/`sentiment_observation`;
`positioning_observation`/`fund_flow`/`market_behavior`; and `congressional_transaction`,
respectively. Generic pages and individual social posts cannot fill the psychology/behavior
requirements. Non-news records need `data.parse_status` of `structured` or `extracted_reviewable`.
Psychology/behavior measurements also need nonempty `metric` and `unit` fields and a finite
numeric `value`. These contracts let a caller supply measured observations; they do not add
new live survey/position adapters or validate the truth of source-provided measurements.

SEC company-insider and institution records are not government-official disclosures.
They can satisfy the separately selectable `institutional_disclosures` category (120 days),
using `insider_transaction`/`institutional_holding`. For example, `coverage.json` may contain:

```json
{"requirements": [{"category": "news", "watch_ids": ["news-fed", "news-ecb"],
                   "max_age_seconds": 259200, "min_records": 2}]}
```

The report always exposes the selected requirements. An empty requirement list is rejected.
This explicit news-only check does not change the four-category default in `health`. A
record must have a source URL and known publication time, or be an explicitly marked observation
from the supported AAII/Michigan/CFTC/House public adapter and matching source host. For the latter,
`published_at` stays null, availability begins at actual `observed_at`, and age is conservatively
measured from the survey week, positions report date or filing date. These reference dates are
never treated as public availability. A newly fetched old report stays stale. Unknown-time
generic records remain ineligible. `publication_time_complete` and per-category
`observed_time_records` expose the limitation; observation lag stays null when release time is
unknown. The 14-day behavioral age bound accommodates weekly Tuesday CFTC positions normally
released on Friday (the prior snapshot can be ten days old before the next release).
The 45-day psychology age bound allows a month-start reference to remain current until the
following month's preliminary release; callers can set stricter source-specific requirements.
Future publications/observations and newer revisions are excluded at `as_of`.
Current disabled watches do not satisfy requirements;
historical queries ignore present-day watch enablement. An oversized snapshot returns unknown
rather than reporting coverage from a silently truncated subset. Passing means the declared
minimums were met, not exhaustive category, asset or population coverage.

```bash
python -m pytest tests/test_collect.py tests/test_collection_health.py -q
python scripts/check_collection_live.py --house-year 2026 --house-name Pelosi --house-since 2026-08-01
```

The first command uses injected official-format responses and needs no network. The second
uses real public sources with a temporary database. Add `--sec-cik 1067983` only after setting
your identity. A passing fixture test does not establish current source availability.

For a persistent acceptance receipt, build a wheel, install it with the `collect` extra into
a fresh virtual environment, and run `scripts/check_installed_collection.py` using that
environment's Python with `-I`, from outside the checkout. Pass a new `--output` directory
and the wheel's source `--commit`. This check exercises the registered collection tools,
reopens the database, and waits for a scheduled news poll. It records source errors and
publication/observation dates; raw records remain in the output directory. SEC is skipped
when the caller has not configured `SEC_IDENTITY`.

The installed check exercises structured Michigan/CFTC observations, news and House disclosures,
requires the four coverage categories, and waits for all configured required watches to repoll.
A short successful poll cannot establish exhaustive coverage or prove that a new publication
was detected if the source did not publish anything during the test.
`receipt.json` separates tool discovery from tools actually exercised and reports partial
source failures as `needs_review`. It contains no trading performance measurement.

Source contracts checked on 2026-10-16: [Michigan public tables](https://www.sca.isr.umich.edu/tables.html), [AAII public survey](https://www.aaii.com/sentimentsurvey),
[CFTC public API guide](https://publicreporting.cftc.gov/stories/s/COT-Help/p2fg-u73y/),
[TFF public API](https://publicreporting.cftc.gov/resource/gpe5-46if.json), and
[House disclosure portal](https://disclosures-clerk.house.gov/FinancialDisclosure).

Check `status` for errors, partial results, remaining filings and next retry times. HTTP
Retry-After reaches the scheduler; failed filings retain individual retries so one old
failure cannot block later disclosures. Pending outbox records survive restarts. Delivery
callbacks are at-least-once; use the record's `seq` as a sink idempotency key. CLI stdout
alerts are for terminal inspection, not a guaranteed external notification service.

For source references and parsing details, see the
[collection skill](../plugins/fin-alt-data/skills/public-information-collection/SKILL.md).
