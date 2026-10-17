## [HEAL-20261013-F24] explicit reader recovery and no-paid model selection

**Logged**: 2026-10-13T20:56:21.0162291Z
**Status**: pending-verify
**Trigger**: model research failures observed in real interface qualification
**Area**: experiment interface

### Failure and diagnosis
The candidate21 7B/32B canary completed 6/6 decisions for each model and successfully executed
RAG on 2/3 autonomous library dates each. Coder32B nevertheless produced nine truncated
research replies: original records show copied market rows inside generated Python. General7B
repeatedly iterated a reader dictionary as if it were the records list. Python variables from
previous tool replies also did not exist in the fresh worker. These are not failures to deliver
RAG results and do not establish a cause for lower full-period Return Rate.

### Fix and verification
Candidate24 adds explicit file/reader examples and neutral recovery feedback shared by both
arms. It changes no selected asset, trade, execution rule, output budget, or prior record.
The original regression run had 176 passes and one stale expected-model-set assertion. After
that test was updated to respect the explicit disabled flag, all 38 policy/model tests passed.
Build/index/package validation passed. New actual-model job 2131136 remains pending; model
behavioral improvement is not yet verified. Original receipt: qualification-7b-32b.json in
benchmarks/agent_study/reports/20261013-free-campaign/.

The same source disables all external/native models for new freezes and declares twenty
active, pinned, public HF IDs. Tests prove disabled models fail before creating a run directory.

### Metadata
- Pattern-Key: experiment.reader_contract_recovery
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-10-13

## [HEAL-20261013-S25] stage new public model weights within shared storage

**Logged**: 2026-10-13T20:56:21.0162291Z
**Status**: pending-verify
**Trigger**: storage guard stopped bulk weight download
**Area**: Beacon storage

### Failure and diagnosis
The initial seven-model downloader stopped before downloading weights: only 82 GiB remained
on the shared radfm filesystem (99% occupied). Repeating the same bulk download is unsuitable.

### Fix and verification
Prepared serial new-model groups, a 12 GiB free-space reserve, exclusive cache/lock, and cleanup
of only newly created weight blobs after all three independently scored cycles. Tokenizers,
all decisions, account ledgers and audit receipts remain. Syntax/roster checks passed on Slurm.
Actual staged download and cleanup have not run: the new-large submission was rejected with
AssocMaxSubmitJobLimit. Preserve the failed submission receipt; wait for a slot before retrying.

### Metadata
- Pattern-Key: beacon.serial_public_weight_storage
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-10-13

## [HEAL-20261013-PS1] literal script avoids nested PowerShell quoting

**Logged**: 2026-10-13T20:56:21.0162291Z
**Status**: verified
**Trigger**: shell parser error before any mutation
**Area**: local publication editing

### Fix and verification
A nested command string raised Missing ')' in method call before execution. Replaced it with
literal here-strings in reader_recovery24_edit.ps1. Re-running exited 0, created the intended
isolated branch, and produced only the intended 16-line runtime diff.

### Metadata
- Pattern-Key: powershell.literal_edit_script
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-10-13

## [HEAL-20261013-CI4] keep model roster test independent of optional study packages

**Status**: verified
**Trigger**: minimum-dependency CI failure
**Area**: tests

### Failure and diagnosis
CI job 115535830481 failed with ModuleNotFoundError: No module named 'bs4'. The new
roster test imported multisource_model_study, which imports the optional evidence collector.
The basic library environment intentionally does not install the study extra. Its other
3974 tests passed. This was a test dependency mistake, not a Pandas or trading-accounting bug.

### Fix and verification
Keep the twenty-model/no-paid roster assertions in the dependency-free model-matrix tests.
Move the actual disabled-freeze integration into the existing optional study module, which
already declares pytest.importorskip('bs4'). In an isolated Beacon publication copy, the
relevant tests passed with bs4 (39 passed) and with bs4 deliberately unavailable (25 passed,
one optional module skipped). All generation/validation scripts also passed. The active
candidate24 runtime and GPU jobs were not modified by this test-only correction.

### Metadata
- Pattern-Key: ci.optional_study_dependency_scope
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-10-13

## [HEAL-20261013-G26] decision completion is insufficient to release a campaign

**Logged**: 2026-10-13T22:03:00Z
**Status**: verified (release gate; model behavior remains unverified)
**Trigger**: complete candidate24 pilot had a false-positive campaign release
**Area**: experiment qualification

### Failure and diagnosis
Both models passed the instructed access probe and completed all six final decisions. General7B
used RAG on all three library dates. Coder32B used no substantive library tool in those dates,
and all six research attempts on the final library date failed. The original release script
checked decision completion only. Thus its `passed: true` did not prove research readiness.
Original sanitized receipts are preserved in the research-readiness report directory.

### Fix and verification
New freezes identify the changed interaction revision. Common instructions explain long-format
prices and Series access; the optional library entry point precedes low-level Python contracts.
The full-cycle CLI now verifies original preflight records and rejects all-failed research,
malformed/truncated replies and, for the two pilot families, absence of autonomous library
execution. It does not rewrite any response, force a trade, or select on Return Rate.
Final CLI/preflight integration tests: 29 passed in 8.75s on the CPU allocation 2132885. Index, package, and repository validation passed. The prior reader/protocol candidate passed 182 tests in 221.33s. Replaying the original pilot observations through the new review accepts 7B and rejects the defective 32B case. The sole CPU job completed; no repaired-model inference has run.

The user limited this project to one Slurm job total. All seven previous project jobs were
cancelled by exact ID after ownership checks. The prepared model drivers were not deployed.
The original 96G launch scripts are historical records, not valid restart instructions.

### Metadata
- Pattern-Key: experiment.decision_only_release_gate
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-10-13

## [HEAL-20261013-N29] audit nested tool delivery before releasing inference

**Status**: verified (CPU regression; repaired model behavior remains unmeasured)
**Trigger**: Python completion concealed failed nested research calls
**Area**: experiment tool contract and qualification

### Failure and diagnosis
The wrapper returned Python execution success even when every bound tool returned an error,
and research qualification counted that turn as successful. Existing failed probes were checked
after backend allocation. Pagination also silently coerced boolean, fractional and string inputs.

### Fix and verification
Keep raw Python status and model decisions, but expose nested receipts and reject all-failed
research. Negative Guard findings remain successful deliveries. Reject retained failed probes
before loading a model; require integer pagination arguments. CPU-only Slurm job 2139315 first
ran the new regression module against the prior source (12 failing cases), then applied the
patch: 165 focused tests passed in 194.03s. Index, package and repository validation passed.
Local source SHA256 values match the tested copy. Reproduction, validation and fingerprints are
in benchmarks/agent_study/reports/20261013-contract-audit/. The sole job completed; no GPU or
model inference was used, and no new Return Rate is claimed.

### Metadata
- Pattern-Key: experiment.nested_tool_delivery_and_early_rejection
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-10-13

## [HEAL-20261015-D31] distinguish Pandas shape failures from tool arguments

**Status**: verified (CPU recovery; model behavior on the patch is unmeasured)
**Area**: paired agent interface diagnostics

The preserved source29 7B pilot called Series.cov() with no second Series and repeatedly indexed
the scalar produced by Series.mean(). The TypeError hint incorrectly blamed bound-tool arguments.
Use shape-specific diagnostics and a neutral raw-matrix inspection example equally in both arms;
do not execute recovery or choose investments automatically. Slurm 2210699 reproduced 7 cases
on the old source, then passed 101 tests in 295.82s after the patch. Builds/validation passed and
local hashes match the tested source. Reports include all 12 previous pilot decisions and failures.
32B passed the previous pilot and proceeds independently on its unchanged source29 in job2232777.

- Pattern-Key: experiment.python_typeerror_misdiagnosis
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-10-15

## [HEAL-20261015-RSI] compare persistent configuration after JSON normalization

**Status**: verified
**Trigger**: first-development regression tests failed when reopening the RSI journal
**Area**: RSI persistence

The in-memory seed contained tuples while JSON restored lists, so unchanged configuration was
incorrectly rejected. Canonical JSON comparison fixes the round trip without relaxing model,
protocol, seed or budget checks. The original run had 26 failures and 12 passes; the corrected
38 cases passed, then the final expanded suite passed 42/42. See the source-verification report
and hashes under benchmarks/agent_study/reports/20261015-optional-rsi/.

- Pattern-Key: rsi.journal_json_roundtrip
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-10-15

## [HEAL-20261015-SSH] reuse a newly recovered shared master after ownership changes

**Status**: verified
**Trigger**: SharedOnly reported no available upstream connection
**Area**: Beacon monitoring

The prior fin-skills master had ended. Before creating a replacement, the recovery helper
detected a new existing plink process and stopped. Read-only ownership inspection identified
NPJ v13's newly verified shared master (parent 71932). Reusing that connection succeeded at
2026-10-15T17:47:56Z and showed only fin-skills job 2232777, pending Priority. No additional
login, job, cancellation or modification to the existing 32B experiment was performed.

- Pattern-Key: beacon.shared_master_owner_changed
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-10-15

## [HEAL-20261015-WT] read verified code from the owned publication worktree

**Status**: verified
**Trigger**: missing published report and sandbox Git worktree error
**Area**: local publication workspace

A report read incorrectly used the older primary checkout; Git status in the owned worktree
also required access outside the writable roots. Reading the exact owned worktree with the
approved filesystem scope found the report and returned clean Git status. No experiment code
was changed to work around this local path/access issue.

- Pattern-Key: workspace.owned_checkout_and_git_access
- Recurrence-Count: 2
- First-Seen: 2026-10-15
- Last-Seen: 2026-10-16

## [HEAL-20261015-LOG] normalize terminal trailing spaces for publication

**Status**: verified
**Area**: evidence publication

The selected Slurm node fields contained trailing spaces emitted by awk, and staged diff checking
rejected them. Trimmed only end-of-line spaces, documented the normalization, and reran
`git diff --cached --check` successfully. Values and ordering are unchanged.

- Pattern-Key: publication.terminal_trailing_spaces
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-10-15

## [HEAL-20261015-RESUME] continue a frozen pair after interrupted scoring

**Status**: verified (CPU fault injection); Slurm recovery not yet exercised
**Area**: experiment persistence

The original launcher rejects started experiments, and the frozen exclusive JSON writer raises
FileExistsError when restarting a partially completed score. The v8/v9 fault-injection tests
reproduce that FileExistsError on the retained NAV. A separate host recovery driver validates
frozen bindings and chronological decisions, preserves existing bytes, atomically publishes new
JSON and reuses identical deterministic score files. Its 57 CPU tests passed, including the real
ledger and independent accounting audit. Deployed files match their recorded hashes; original
pending job2232777 and its frozen source remain untouched. No model return is claimed by tests.

- Pattern-Key: experiment.frozen_pair_partial_score_resume
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-10-15

## [HEAL-20261015-OPTIONAL-STUDY] defer the optional replay import in recovery tests

**Status**: verified locally; new CI run pending
**Area**: test collection

CI run38796771156 failed during collection because the new recovery tests imported the study's
optional bs4 dependency even in minimum-dependency and extended-validation jobs. Importing the
study only inside the integration fixture preserves 23 unconditional core checks; 34 integration
cases explicitly skip without the study extra. Verification passed both configurations: 57/57
with the installed study dependencies; 23 passed and 34 skipped with a subprocess import blocker
simulating missing bs4. The reproduction helper and updated test hash are committed in the
safe-resume report. The production driver and Beacon deployment were unchanged.

- Pattern-Key: ci.optional_study_collection
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-10-15

## [HEAL-20261015-COLLECTION-EXIT] preserve source failures through bounded CLI exit

**Status**: verified with local CPU fixtures
**Area**: public-information collection

Before the patch, two regression cases showed `run --cycles 1` returning 0 for error and
partial source reports. The CLI now accumulates failed cycles, checks current source readiness,
and emits a final summary with a nonzero exit for failures, stale or uninitialized data, and
idle watchlists. Backoff skips preserve prior failures in CLI/tool health. Live ownership is
reported separately from source freshness. The final targeted suite passed 171 tests, including
19 new regressions; build/index/package validation passed. Evidence is in
`benchmarks/agent_study/reports/20261015-collection-health/`.

- Pattern-Key: collection.bounded_run_false_success
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-10-15

## [HEAL-20261015-WINDOWS-CHILD] terminate the actual fixture interpreter

**Status**: verified
**Area**: process-crash regression test

The first full test run observed a collector still holding its lock after terminating the
Windows venv launch process. A PID probe confirmed that this venv creates a different actual
interpreter PID (58228 versus 21616). The fixture now launches `sys._base_executable` on Windows,
passes the current import paths, and asserts runtime PID equality before terminating its own
child. The crash/restart case passed independently (1 passed, 18 deselected), then the complete
171-test acceptance passed. The fixture's former recorded worker PID was absent on cleanup
inspection; no unrelated process was stopped.

- Pattern-Key: tests.windows_venv_launcher_child_ownership
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-10-15

## [HEAL-20261016-COVERAGE-INTEGRATION] register the coverage tool and serialize regeneration

**Status**: verified with local CPU fixtures and repository validation
**Area**: tool integration and source verification

Integration tests found that the new collection_coverage tool lacked an explicit historical
sandbox classification. Two other failures came from running package-data consumers while
build_package was replacing generated files. Added the new mutable-database tool to EXTERNAL,
completed package generation, then reran integration checks: 22 passed, 2 Linux-only skips.
Validation also caught the README's old 66-tool count; updating it to the registered 67 tools
restored the required build/index/package validation. The independent 246-test acceptance had
already passed. Logs and source hashes are under
`benchmarks/agent_study/reports/20261016-coverage-rsi-recovery/`.

Build generation and its consumer tests must run sequentially. Do not interpret missing files
during regeneration as a production defect. No frozen 32B source or Beacon job changed.

- Pattern-Key: collection.coverage_registration_and_generated_test_order
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-10-16

## [HEAL-20261016-ISOLATED-BUILD] use the declared build backend in isolation

**Status**: verified
**Area**: installed-wheel verification

The no-isolation build failed because the shared local environment had setuptools 65.5.0
instead of the declared >=68 and no wheel package. Re-running the wheel build with default
build isolation installed the declared backend into a temporary environment and succeeded.
The wheel then installed in a dedicated environment; source/wheel/installed bytes matched,
and the actual source polling/coverage acceptance passed. No shared environment was upgraded.
Evidence: `benchmarks/agent_study/reports/20261016-live-information-sources/`.

- Pattern-Key: build.declared_backend_isolation
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-10-16


## [HEAL-20261016-AAII-CHALLENGE] expose blocked survey transport and verify a public CSV source

**Status**: verified for default-profile recovery; AAII live access remains restricted
**Area**: scheduled public-source collection

The first installed-wheel run fetched AAII readings successfully, but the next scheduled poll
returned an HTTP-200 automated-access challenge. The receipt correctly failed even though
previously collected category data remained present. One diagnostic read confirmed the challenge;
no bypass or repeated retries were attempted. The adapter now explicitly reports it with a
24-hour retry delay and preserves prior readings. The default profile uses the university's
explicit public consumer-sentiment CSV, with a separately labeled population and units.
A regression reproduces challenge/failure/backoff; another verifies preliminary-to-final
revisions without retroactive historical visibility. The corrected installed run is recorded
alongside the original failed receipt in the live-information-sources report.

- Pattern-Key: collection.aaii_http200_automated_access_challenge
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-10-16

## [HEAL-20261016-STAGED-TEST-IMPORT] preserve sibling test imports on deployment

**Status**: verified
**Area**: Slurm startup validation

The first flexible allocation exited during collection because a staged research
fixture imported a helper from the source tests directory. Run the selected tests
from the complete source tests directory and put that directory on PYTHONPATH.
The corrected allocation passed 66 checks in 22.83 seconds, then index/package/spec
validation. This establishes startup recovery only, not model or financial success.

- Pattern-Key: deployment.sibling_test_import_path
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-10-16
- Evidence: benchmarks/agent_study/reports/20261016-flexible-32b/

## [HEAL-20261016-PEER-TRANSFER] qualify host-staged transfers before decisions

**Status**: pending-verify on real dual-GPU generation
**Area**: model execution

The RTX 6000 Ada allocation failed an exact BF16 copy from GPU 0 to GPU 1 before
model loading. The guard correctly rejected it. Added an explicit blocking CPU
staging route, a generation-scoped Accelerate hook, and nested-input/restore tests.
The new allocation must pass the same exact-copy, full-context and real-tool gates;
no failed transfer is tolerated or replaced with invented numeric values.

- Pattern-Key: inference.cross_device_transfer_corruption
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-10-16
- Evidence: benchmarks/agent_study/reports/20261016-flexible-32b/
