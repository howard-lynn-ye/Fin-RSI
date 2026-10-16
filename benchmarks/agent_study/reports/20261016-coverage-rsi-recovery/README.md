# Required data coverage and RSI interruption recovery — 2026-10-16

Successful source polling did not establish that the required news, psychology, behavior and
government-official disclosure information was present. Separately, an interrupted RSI writer
could leave a persistent lock marker or a partial final journal record and prevent continuation.
This patch makes these conditions observable and adds explicit, evidence-preserving recovery.

## Data coverage

- The new read-only `collection_coverage` tool and `coverage` CLI check the four required
  categories independently. Rules can specify source watch IDs, minimum counts and publication
  age. Institutional/insider disclosures are an optional separate category; they do not fill
  the government-official disclosure requirement.
- Unknown publication times, stale content, missing provenance and unparsed structured data
  cannot pass. Psychology and behavior require a named metric, numeric value and unit. A
  generic page or social post does not establish an aggregate psychological/behavioral series.
- Historical coverage uses records observed and published by the cutoff, retaining the eligible
  revision. Current checks also reject disabled sources. An oversized snapshot returns unknown
  rather than claiming coverage from a truncated sample.
- Source polling freshness, collector liveness and coverage remain separate fields. Overall
  `ready` and the `health` CLI now require both healthy continuous collection and required
  coverage. Current RAG results carry coverage; historical RAG queries omit current health.
- The tool is classified as external to the historical trading sandbox because it reads a
  mutable local collection database. The frozen experiment is not updated by this classification.

## RSI recovery

- An OS-owned writer lock releases when the process dies. Concurrent writers remain excluded;
  an empty persistent lock file alone no longer prevents continuation.
- `RSILoop.recover()` validates the journal chain and its seed/model/policy/protocol before
  changing anything. Only an unterminated malformed final fragment can be quarantined. Exact
  original bytes are backed up before atomic replacement. A valid hashed final record lacking
  its newline is retained, including any call-budget reservation.
- Complete malformed records, hash corruption and mismatched protocols remain errors. Legacy
  PID-only lock markers require explicit host confirmation that the old writer stopped; the
  marker is archived, not discarded. Stop old-version writers before upgrading an archive.
- Recovery does not invoke models, replay unfinished callbacks, refund spent budget, promote a
  candidate or clear the frozen state. Repeated recovery of a clean journal changes no records.

## Verification

Local CPU acceptance passed **246 tests**, including **22 new coverage regressions** and
**10 new RSI recovery regressions**. Integration checks passed **22 tests**, with **2 Linux
confinement tests skipped on Windows**. The test logs and JUnit XML accompany this report.
`receipt.json` records the observed counts and binds the changed source/test files by SHA-256.

Crash tests start a real fixture interpreter, assert its PID, verify that a second writer is
rejected, terminate only that owned child inside a proposer or evaluator, and recover the same
archive. They verify that already-started calls still consume budget and no callback is replayed.
Other tests exercise incomplete UTF-8/JSON tails, missing newlines, corruption, legacy locks,
frozen state, read-only database access, cutoff isolation and the registered tool interface.

The first integration attempt found a missing sandbox classification for the new tool. It also
overlapped package regeneration, temporarily removing package data used by two tests. The tool
classification was added, and integration tests were rerun after generation finished: all
applicable cases passed. The first validator also caught the README's old tool count; it now
matches the 67 registered tools. `integration-before-fix.txt` preserves that failed integration
run. An earlier attempt against old source could not import the new API; it is not presented as
a semantic before/after reproduction.

The required `build_index.py`, `build_package.py`, and `validate.py` completed sequentially.
Validation reports **131 skills** and the existing discovery-budget warning for the 24-skill
`fin-libraries` plugin. GitHub's cross-platform CI and merge status are tracked on this patch's PR.

## Scope and limits

This adds coverage validation, not the missing external source adapters. Passing declared
category/count/age checks is not exhaustive market coverage or semantic certification. These
local fixture tests do not establish current live-source availability or a financial improvement.
Lock recovery is supported on local filesystems; shared-filesystem semantics are not certified.

No Beacon connection, Slurm submission, cancellation, model call or trading experiment was
performed for this patch. The current 32B frozen source, parameters, inputs and ledger are
unchanged. The one-Beacon-job limit remains in force. There is **no new Return Rate result** in
this repair record, and these changes are not deployed into the frozen 32B experiment.

Base: `70c7f62c29d39ae64aba9741c4104e0522ba6e8d` on the repository default branch, `master`.
Published text logs preserve command output with encoding converted to UTF-8, line endings
normalized and trailing whitespace removed.
