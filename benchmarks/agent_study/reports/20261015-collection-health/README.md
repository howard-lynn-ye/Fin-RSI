# Collector failure reporting and live health repair — 2026-10-15

The bounded collection CLI reported success even when its source reports contained errors or
partial batches. Before changing production code, the new regression test reproduced both
cases: `run --cycles 1` returned 0 instead of 1. See `before-fix.txt`.

## Changes

- Bounded runs return nonzero for any failed/partial cycle and retain those failures even after
  recovery. Their final JSON summary reports the cycle count, failed cycles and current health.
- A retry skipped during backoff cannot erase the last failure. CLI `once`, bounded `run` and
  the `collect_once` tool expose current database health. Empty or disabled configurations do
  not pass an operational check.
- Per-watch status includes enabled state, last-success age, a freshness threshold and current
  error/partial state. Freshness expires after `max(60, 2 * interval_seconds)`; retry delays do
  not extend it. A later failure takes precedence over a recent success.
- CLI `health` and the existing `collection_status` tool check local collector ownership
  separately from source readiness. An OS lock prevents overlapping collectors and releases
  on process death; interrupted workers can restart without deleting a stale marker.
- Current RAG responses expose runtime/readiness alongside the existing health list. Explicit
  historical queries still omit present-day health. Health reads do not create databases,
  migrate legacy schemas, fetch sources or start services.

## Verification

Local CPU acceptance passed: **171 tests**, including **19 new health regressions**. The run
covered `test_collect.py`, `test_collection_health.py`, `test_research_context.py`, `test_tools.py`
and `test_mcp.py`. `targeted-tests.txt` preserves the output; `receipt.json` binds source hashes
to the observed JUnit result.

The crash test launches a real fixture collector, verifies its running health from a separate
connection, kills only that owned interpreter, checks interrupted status, and restarts it on
the same database. A Windows venv launcher initially made this test target the launcher rather
than the interpreter. A separate PID probe confirmed the difference (launcher 58228,
interpreter 21616). The repaired fixture invokes the base interpreter with the test import
paths and asserts its recorded PID equals the process being killed. It passed independently
and in the final acceptance run.

The required `build_index.py`, `build_package.py`, and `validate.py` completed successfully.
Validation reports 131 skills and the existing discovery-budget warning for the 24-skill
`fin-libraries` plugin. Generated catalog/package content has no semantic change.

## Scope and limits

These are injected-source CPU tests, including the real local SQLite, process locking and MCP
transport. They do not establish current external-source availability or complete news,
sentiment, behavioral or disclosure coverage. A successful empty response refreshes source
poll health, not the publication date of its records. Liveness is supported for local filesystems
on the originating host; other-host/legacy observations are unknown rather than healthy.

No Beacon connection, Slurm submission, job cancellation, model call or trading experiment was
performed for this patch. No frozen 32B code, inputs, parameters or ledger files were edited.
There is **no new Return Rate result** in this repair receipt. The existing one-job experiment
limit is unchanged, and this patch does not claim deployment into the frozen experiment.

The base is `f2438d22078977d985349f63c4ea835cd7ea5082` on the repository's default branch,
`master`. GitHub CI and merge status are recorded by the pull request for this change.
