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
- Recurrence-Count: 2
- First-Seen / Last-Seen: 2026-10-13

Verified recurrence 2026-11-09: an unused double-quoted remote command with backslash-escaped quotes failed the local parser before connection. Removed it and used a literal command/here-string; SharedOnly status read succeeded.

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

### October 17 recurrence
The new serial-driver selection tests repeated this dependency-scope mistake: minimum CI
job 117482929459 failed four cases on missing bs4 while 4173 other tests passed. Declare
pytest.importorskip('bs4') only inside the two study-integration test functions; keep the
seven dependency-free serial-driver cases active in minimum installs. The separate free-model
roster checks remain dependency-free. Minimum CI job 117485361979 passed at commit a3d5760;
the installed-study paths had also passed in Beacon jobs 2365768 and 2366235. The production
runner and frozen source of the active v11 experiment were not changed by this test fix.

### Metadata
- Pattern-Key: ci.optional_study_dependency_scope
- Recurrence-Count: 2
- First-Seen / Last-Seen: 2026-10-13 / 2026-10-17

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
- Recurrence-Count: 4
- First-Seen / Last-Seen: 2026-10-15 / 2026-11-09
- Latest verification: TACGEN intentionally closed healthy master3848 at20:38Z. A local
  shareexists guard stopped before upload/authentication. After checking ownership and zero
  SSH clients, one standard-helper recovery created master18016(parent68268); SharedOnly
  reads and the prepared source upload both exited0. No other owner's job was changed.

Verified recurrence 2026-11-09: previous master18016/parent68268 had ended. After zero-client and share checks, the standard hidden helper using the observed current PowerShell executable created master74308/parent32460. SharedOnly read and immutable v4 upload succeeded. A guessed Program Files executable had failed locally before any SSH attempt; no authentication retry loop or other owner connection was touched.

## [HEAL-20261015-WT] read verified code from the owned publication worktree

**Status**: verified
**Trigger**: missing published report and sandbox Git worktree error
**Area**: local publication workspace

A report read incorrectly used the older primary checkout; Git status in the owned worktree
also required access outside the writable roots. Reading the exact owned worktree with the
approved filesystem scope found the report and returned clean Git status. No experiment code
was changed to work around this local path/access issue.

- Pattern-Key: workspace.owned_checkout_and_git_access
- Recurrence-Count: 3
- First-Seen: 2026-10-15
- Last-Seen: 2026-10-17

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

**Status**: verified on L40S for host-staged transfers and full-context generation
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

Verification update: job 2317005 passed all 36 exact host-staged copy cases and
32,768-token generation with the complete CUDA BF16 model on two L40S GPUs.
The 68 selected tests and both build validations also passed. This does not assert
that direct RTX peer copies were repaired, or that the financial cycle completed.

Verification update, 2026-10-17: the complete 32B pair finished with exit 0,
44/44 decisions per arm and zero failed final decisions. Independent accounting
passed for both returns. See benchmarks/agent_study/reports/20261017-32b-full-cycle for the immutable
receipts and complete derived ledger. This confirms full-cycle L40S execution;
it still does not validate direct RTX peer transfers.

## [HEAL-20261017-RG] use ripgrep globs instead of shell-expanded Windows paths

**Status**: verified
**Area**: source inspection

A search passed a wildcard filename as a positional argument on Windows and
returned OS error 123. Searching the parent directory with `rg -g` instead found
the intended ledger definitions and trade append sites, exit 0. No code or run
was changed.

- Pattern-Key: powershell.ripgrep_glob_path
- Recurrence-Count: 2
- First-Seen / Last-Seen: 2026-10-17 / 2026-11-08

Verification update, 2026-11-08: manuscript review repeated the positional
wildcard mistake. Searching paper/latex_naacl with -g '*.tex' -g '*.bib'
completed successfully; the targeted terms had no matches. No manuscript changed.

## [HEAL-20261017-VALIDATION-BUNDLE] deploy complete canonical source for remote checks

**Status**: verified (deployment identity and build checks)
**Area**: Slurm source validation

A reused build-only snapshot omitted current tests (job 2365476), then strict
comparison found an older plugin document (2365644) and a CRLF-only generated
index difference (2365687). No model was run in these failed validation stages.
The source package now includes selected tests and their helper imports. The
upstream skill document was synced and the catalog/package regenerated. The
index was compared with canonical Git/LF generated content; no hash check was
disabled. Job 2365768 matched all 602 expected source files, passed the required
build/spec checks, and verified all 88 original evidence snapshot bindings.
The subsequent regression-test result is recorded separately in the report.

- Pattern-Key: deployment.partial_source_snapshot
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-10-17
- Evidence: benchmarks/agent_study/reports/20261017-research-reliability/

## [HEAL-20261108-REVIEW-PATHS] resolve review sources before reading

**Status**: verified
**Area**: source inspection
**Trigger**: missing local execution record and inaccessible guessed workshop URL

The report's relative execution.json was incorrectly resolved under runs/.
File discovery found it under benchmarks/agent_study/reports/20261017-research-reliability/;
reading that file succeeded and confirmed it is an October 17 historical snapshot.
The guessed ICAIF /workshops/ URL failed in the web reader. The official /news/
page opened successfully and supplied the workshop deadline notice. The failed
URL was not treated as evidence that the site itself was unavailable.

- Pattern-Key: review.unverified_source_location
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-11-08
- Verification: corrected local Get-Content and official news-page open both succeeded.
- Scope: read-only review recovery; no experiment, manuscript or remote state changed.

## [HEAL-20261108-RECEIPT-STDOUT] distinguish committed receipt from console failure

**Status**: verified
**Area**: local progress reporting
**Trigger**: publish_status.py wrote its receipt, then cp1252 stdout rejected Chinese text

Read the publisher and confirmed the atomic write precedes printing. Verified the
existing event 2026-11-08-9ae5e51c4b534cc39dcff282d598a2c9.json has the intended
run_id and partial result. Reprinted the success message with Python -X utf8;
the Unicode output succeeded with exit 0. Did not republish or create duplicates.
Use -X utf8 for future publisher invocations; no shared script was modified.

- Pattern-Key: progress.stdout_encoding_after_commit
- Recurrence-Count: 3
- First-Seen / Last-Seen: 2026-11-08 / 2026-11-09
- Verification: existing receipt assertions and UTF-8 success-message output passed.
- Later recurrence: production-runtime-20261108 was already committed before stdout failed.
  UTF-8 inspection found exactly one matching event; no second publication was made.

2026-11-09 recurrence: the parent Python used UTF-8 but its publisher subprocess did
not inherit Python's -X setting. Run original-study-20261109-0407 committed event
2026-11-09-c281654f19b54d66bb5966b95d00c1e4.json before stdout failed. Exact unique
run-id, item/result and evidence hash assertions passed; UTF-8 readback printed
success with exit0. No duplicate publication. Put -X utf8 on the child interpreter
argument list as well as the parent; capture encoding alone does not configure it.

## [HEAL-20261108-FACTOR-CARD] honor event-ratio as well as cumulative split inputs

**Status**: verified
**Area**: agent instructions
**Trigger**: new task prompts declared event ratios while the shared card assumed cumulative factors

The split_once rule and its feedback recipe applied raw * split_factors unconditionally
and offered positions() with its cumulative default. Ratio tasks require a running product
or factors='ratio'. The shared card now states both conventions explicitly, retaining the
double-adjustment rejection guidance. Frozen experiment files were not edited.

Verification: the new rule regression failed before the fix. Afterward the six-module
focused suite passed: 118 passed, 3 Linux-confinement tests skipped, 2 slow tests deselected.
Four new numerical tests compare the event-ratio recipe with independent reference equations.

- Pattern-Key: agents.rule_card_factor_convention
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-11-08
- Related Files: fin_skills/api/rules.py, tests/test_api_rules.py, tests/test_algorithm_panel.py

## [HEAL-20261108-TASK-WORKER] use the caller's package in child task execution

**Status**: verified
**Area**: task evaluator
**Trigger**: seven existing task-suite integration tests failed with ModuleNotFoundError

The parent imported the source checkout, but the autouse test fixture moved cwd to a
temporary directory. The child python -m invocation could not find the uninstalled package.
The worker environment now prepends the parent package root to PYTHONPATH and preserves
existing entries. No timeout, grading rule, import restriction or confinement hook changed.

Verification: the same focused suite changed from 7 failed / 111 passed to 118 passed,
3 skipped, 2 deselected. Correct programs are accepted and independently graded; planted
timing, split, warm-up, forbidden-import and output-shape defects retain their expected results.

- Pattern-Key: tasks.worker_source_checkout_import
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-11-08
- Related Files: fin_skills/tasks/judge.py, tests/test_tasks_suite.py


## [HEAL-20261108-V2-TELEMETRY] distinguish final calls from any-attempt calls

**Status**: verified
**Area**: manuscript evidence generator
**Trigger**: the new v2 recount asserted that any-attempt use equaled library_users

The frozen v2 scorer defines library_users from the final submission; v1 manuscript use
counts any attempt. Direct recount found 9 final callers and 10 any-attempt episodes.
The generator now computes both, checks the final count against scores.json, and the table
caption states its denominator and selection rule. No frozen evidence changed.

Verification: python scripts/build_codegen_evidence.py and --check both passed; the v2 table
rendered with 9/24 final callers and the text separately reports 10 any-attempt episodes.

- Pattern-Key: evidence.final_vs_any_attempt_telemetry
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-11-08

## [HEAL-20261108-LATEX-BUILD] compile in an isolated output directory

**Status**: verified
**Area**: manuscript build
**Trigger**: latexmk failed because its Perl engine was absent; repeated pdflatex passes
from the source directory then kept reading the old source-directory main.aux

Use the installed pdflatex directly, with the working directory set to the isolated build
directory and -include-directory pointing to the source. This avoids changing system tools
and makes the current aux file authoritative. No bibliographic content was edited.

Verification: .progress-audit/paper-build-20261108/pass-4.log and main.log report 24 pages,
with no undefined references or overfull boxes. The changed v2 table page was rendered and
visually inspected; references resolve and no text is clipped.

- Pattern-Key: latex.miktex_output_aux_precedence
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-11-08


## [HEAL-20261108-STRUCTURAL-EVAL] separate execution, attempt identity and trustworthy accounting

**Status**: verified locally; production/model qualification not performed
**Area**: task execution and experiment records

Independent review reproduced shared checker/reference access, mutable task identity,
post-start failure misclassification and lost known costs on malformed/refused responses.
The executor now stages a public runtime, confines candidates, bounds streams/files, and
computes checks in the parent. Readiness travels through a pipe, not a candidate-writable
marker. Complete task contracts are frozen/rechecked; source bundles, request/response/failure
records and terminal digests are retained. Bounded responses validate exact usage before
content/refusal parsing. Each attempt keeps separate numerical evidence, so repeated code
cannot overwrite the first attempt's grade.

Verification: local Linux/WSL related suite 84 passed; final attempt-index/contract/study delta
65 passed. The suites overlap. Independent closure receipts are
.progress-audit/independent-structure-closure-20261108.json and
.progress-audit/independent-controlled-study-review-20261108.json. No historical model outputs
were rerun and no scientific success rate was created from these scripted fixtures.

- Pattern-Key: evaluation.execution_identity_cost_boundary
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-11-08
- Supersedes current implementation approach of HEAL-20261108-TASK-WORKER; preserves its history.

## [HEAL-20261108-READ-STDOUT] UTF-8 for Chinese read-only inspection output

**Status**: verified
**Trigger**: Windows Python cp1252 stdout failed while printing Chinese protocol/state JSON

Both implementation and reviewer windows observed this read-only formatting failure. Running
Python with -X utf8 successfully printed the same content. No source JSON or progress receipt
needed to be recreated. This shares the console cause of HEAL-20261108-RECEIPT-STDOUT, but the
failed operations here were reads; they did not commit a receipt before failing.

- Pattern-Key: windows.python_read_stdout_cp1252
- Recurrence-Count: 4
- First-Seen / Last-Seen: 2026-11-08
- Verification: protocol inspection with python -X utf8 exited 0; reviewer reported the same.


Verified recurrence, 2026-11-08: reviewer S coordinator criteria inspection
printed Chinese through a PowerShell Python pipeline and hit UnicodeEncodeError
under cp1252. The same read-only command with python -X utf8 succeeded.
No scientific source or data changed; reported by the independent reviewer.
Verified recurrence 2026-11-09: protocol read_text without explicit UTF-8 failed decoding Chinese under cp1252. The same read-only inspection with python -X utf8 succeeded and returned the exact five model IDs. No protocol was altered.

## [HEAL-20261108-SCALAR-INTEGER] preserve exact integer output across the worker boundary

**Status**: verified
**Trigger**: independent reviewer reproduced 2^53+1 becoming 2^53 through float(output)

Scalar transport now preserves Python and NumPy integers as JSON integers; only floating
outputs convert to float. Parent validation and exact scoring avoid converting large integers
to float for finiteness. Linux tests cover both signs of 2^53+1, 10^400 and NumPy int64;
scalar/packet tests passed21 in17.92s. The independent window re-ran the four numerical
counterexamples successfully. This is local typed-contract qualification, not model evidence.

- Pattern-Key: evaluation.scalar_integer_float_roundtrip
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-11-08
- Verification: .progress-audit/scalar-exactness-linux.log; independent review chat
  01a1bca7-537c-70b0-b077-f422924260f1.

## [HEAL-20261108-PATH-DISCOVERY] resolve actual implementation and job-output paths

**Status**: verified
**Trigger**: review guessed report.py/records.py; implementation tried a GPU stderr file
before its dependent job had started. These were missing paths, not transport failures.

Use rg --files for repository discovery (Report lives in rsi/bench.py). For pending Slurm
jobs inspect state first and guard optional log reads with test -f. The actual files and CPU
log were successfully read through the same healthy connection. No authentication retry
or job restart was needed.

- Pattern-Key: inspection.existing_path_discovery
- Recurrence-Count: 7
- First-Seen / Last-Seen: 2026-11-08

2026-11-09 follow-up: a guessed self_debug.py and literal Windows wildcard failed;
rg --files located self_debug.sh, then exact-path reads succeeded. Remote directory listing
located worker.stderr (not worker-stderr.txt). The SSH transport was healthy; no retry occurred.

2026-11-08 follow-up: guessed `manifest.json`/`submit.ps1` and a prefixed local
qualification folder were absent; PowerShell did not expand the rg path globs.
`rg --files` resolved `core-qualification-20261108/launch.py`, `launch.sh` and
`benchmarks/agent_study/rsi_qualify.py`; exact-path reads then succeeded. No
transport retry, model run or source modification was needed.

Same-session recurrence: a prefixed coordinator folder and wildcard path operands
were guessed again. Corrected reads use the actual `coordinator-cpu-20261108-v2`
folder and `rg PATTERN .progress-audit -g '*provenance*.json'`; exact source/release
reads then passed. The matching independent-review handoff reported the same
missing-path category, not an infrastructure failure. Stop using globs as path
operands in PowerShell; filter with rg's own `-g` argument after directory discovery.

## [HEAL-20261108-PARQUET-ENGINE] isolate a missing production data dependency

**Status**: verified
**Trigger**: CPU intake job3607586 downloaded and hash-verified the pinned public training
file, then pandas could not read it because neither PyArrow nor fastparquet was installed.

The recovery installs pinned PyArrow20.0.0 into the new release's data-tools directory inside
Slurm, preserving the shared Python environment. It reuses and rechecks the already downloaded
file instead of downloading another copy. Job3607628 read all14377 training rows with
pandas2.2.3/PyArrow20.0.0 and wrote a verified source-intake receipt. No model or upstream
reference program was executed, and no final test file was downloaded.

- Pattern-Key: beacon.parquet_engine_missing
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-11-08
- Verification: fin-rsi-repair-20261108-v2/bizbench-train-3607628/terminal.json on Beacon;
  source SHA256 d0af1868ac46b082f153981b813b97b78707153b7a5cd872b30490bec47b3cc1.

## [HEAL-20261108-CPU-QUALIFICATION] complete the bounded production test plan

**Status**: verified
**Trigger**: serial CPU job3607508 exceeded its 1500-second process cap before all planned
tests completed; dependent GPU3607545 was cancelled without calling a model.

The recovery enumerates the complete test-node plan, runs one isolated pytest process per
node with two allocated CPU workers, validates exactly one unskipped JUnit result per node,
and enforces a55-minute global process deadline inside a65-minute allocation. Explicit
pytest root selection also prevents ancestor configuration from changing collected node IDs.
No test or assertion was removed. Job3607628 passed129/129 with no failures/errors/skips in
1112.229seconds; its source manifest is3b09f075a95a4d71102334e433a75463eac76fc822ca917b6d377152fc295d41.

- Pattern-Key: beacon.qualification_serial_timeout
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-11-08
- Verification: .progress-audit/fin-rsi-repair-20261108-v2/cpu-terminal-metadata.json.

## [HEAL-20261108-FIT-OUTCOMES] preserve learning-stage model outcomes and state

**Status**: verified locally; separate production fit qualification pending
**Trigger**: a synthetic writer refusal raised ModelRefusal out of PlaybookLearning.fit.
Evolve also treated failed zero-program episodes as an empty failure list.

Development refusals now remain counted model outcomes. Model proposal schema/size failures
have a distinct InvalidProposal type; provider, transport and evaluator faults still propagate.
Refusal, malformed proposals and exhausted budgets freeze the last accepted state and retain
all charged requests/tokens. Evolve includes failed zero-program episodes and rejects blank
or overlength instructions rather than silently truncating them.

Verification: twelve targeted cases plus existing core/wiring tests passed60/60 in3.49seconds,
including accepted-state preservation and transport-failure propagation. The initial failure
and an incomplete synthetic Grader fixture were diagnosed; the completed fixture supplies
the required check capability. Remote v2 source is unchanged and does not qualify this delta.

- Pattern-Key: evaluation.fit_refusal_outcome_and_state
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-11-08

Follow-up independent counterexample: dropping a refused zero-score task raised the old
tie-break mean from (0,1/3) to (0,2/3) without a correct program. The fixed scorer uses all
predeclared development episodes, assigning zero to missing/unexecuted/no-check results.
The same counterexample no longer improves the score. Final local core/wiring/fit suite:
61 passed in3.37seconds (tool receipt6ff5d4); independent acceptance and exact source hashes
are in .progress-audit/fit-outcome-acceptance-20261108.json. Production fit remains pending.

## [HEAL-20261108-SUPERVISION] enforce the wallclock outside worker cleanup

**Status**: verified on local Linux; production coordinator verification pending
**Trigger**: independent review reproduced a0.1-second SIGALRM taking1.513seconds because
the worker Popen context manager waited for a1.5-second child during exception cleanup.

The coordinator now supervises an independently owned process group from a parent process.
On deadline it sends TERM and then KILL only to that group, retains worker logs/accepted
chunks, and writes a separate authoritative terminal. The immutable numerical v2 runtime is
unchanged. WSL stdlib tests passed3/3 in1.484seconds: a60-second cleanup trap stopped at
0.602seconds under a0.6-second deadline; TERM-ignoring child/parent stopped at0.802seconds;
normal success was retained. An initial exit-state assertion observed transient R after
SIGKILL delivery; the test now confirms exit within a bounded0.5-second kernel-state window.
Independent replay also passed3/3. Actual imported module paths and runtime source hashes
are bound to the qualified tree before model loading; a different nearby tree is rejected.

- Pattern-Key: runtime.sigalarms_defeated_by_popen_cleanup
- Recurrence-Count: 1
- Verification: tests/test_study_supervisor.py; tool77c1b6; independent closure
  .progress-audit/frozen-coordination-repair-closure-20261108.json.
- Review probe correction: reviewer first used WSL system Python without pandas; it then
  loaded the original function AST using stdlib for the focused counterexample. No package
  was installed and this is not a production dependency failure.

## [HEAL-20261108-UPSTREAM-LABELS] keep ambiguous source tables unresolved

**Status**: verified in production
**Trigger**: enrollment3607809 failed on upstream TAT-QA labels that became identical after
NFKC/whitespace normalization. The strict context normalizer correctly refused a lossy map.

Source-table conversion now checks normalized row/column label uniqueness before constructing
a JSON variant. Ambiguous variants supply no source identity; original questions and cells
remain unchanged and use the existing unresolved fallback. No answer or model outcome
affects selection. Nine focused local tests passed, and the independent reviewer reproduced
four row/column and whitespace/NFKC cases. Recovery3607864 completed0:0 in2:08, freezing
367 questions and the entire planned cell list with zero model calls. Failed3607809 is retained.

- Pattern-Key: data.normalized_upstream_label_collision
- Recurrence-Count: 1
- Verification: .progress-audit/enrollment-20261108-v2/terminal-metadata.json.
- Review test-path correction: a guessed test_external_intake.py did not exist; rg located
  test_source_intake.py, and the corrected enrollment/intake command passed9 tests. The root
  also replaced a PowerShell-invalid rg path glob with discovered literal paths. Neither read
  failure changed source or experiment state; use discovery before constructing test paths.


## [HEAL-20261108-SUPERVISION] production verification follow-up

Production Slurm3607880 completed0:0 in14seconds. All15 coordinator tests passed without
skips, including actual v2 import binding and owned process-tree deadlines. This closes the
production-pending status above. Evidence: .progress-audit/coordinator-cpu-20261108/terminal-metadata.json.

## [HEAL-20261108-RELEASE-IMPORT] package from the resolved repository root

**Status**: verified
Direct execution from .progress-audit failed to import benchmarks. The helper now inserts its
resolved repository root before importing the declared test plan. The failed run made only an
empty local directory, verified by exact absolute path before nonrecursive removal; no remote
v3 or job existed. Corrected upload contained728 files, and remote archive SHA matched
611a629a93129df3c913aeb006ce551ce55729c4da593b1a7c79bbe640ecfa6b. This is packaging verification,
not scientific acceptance.
- Pattern-Key: packaging.script_directory_import_root
- Recurrence-Count: 1
- Evidence: .progress-audit/fin-rsi-repair-20261108-v3/upload-receipt.json.

## [HEAL-20261108-EXPIRED-DEPENDENCY] verify an expired completed prerequisite

**Status**: verified submission recovery; computation still pending
sbatch rejected afterok:3607880 with Job dependency problem after this successful CPU job had
aged out of the controller. No new job ID or computation existed. Confirmed sacct COMPLETED0:0,
the immutable terminal, missing submission marker and complete live queue, then removed only
that already-satisfied dependency. Submitted3607895 once and serialized the remaining own jobs.
This was not authentication failure, compute retry or authority to bypass unfinished gates.
- Pattern-Key: slurm.completed_dependency_controller_expiry
- Recurrence-Count: 1
- Evidence: .progress-audit/fin-rsi-repair-20261108-v3/submission-cpu-recovered.txt.

Operational follow-up to inspection.existing_path_discovery: guessed preparation-helper and
metadata paths returned missing; rg --files located the actual terminal-metadata.json before
reading. Reviewer corrected ExitStack import from unittest.mock to contextlib, then its
synthetic worker transport-failure replay succeeded. No production source fault followed.
An apply_patch write to HEALS.md failed despite normal non-readonly attributes; direct UTF-8
append below succeeded and was read back. No permission or policy setting was changed.

## [HEAL-20261108-TEAM] team_execution_identity

**Logged**: 2026-11-08T23:15:00Z
**Status**: verified
**Trigger**: tool-failure
**Area**: evaluation
**Priority**: high

### Failure
Adding Team to the full benchmark reproduced ValueError: unknown execution attempt
(index -1), 1 failed in 71.33 seconds. Team bypassed BaseRSI recording and its Step
kept the invalid default attempt index.

### Fix
BaseRSI._attempt and Team.run_checks now share _check_program, recording the actual
Judge attempt index after one charged check. Preserve the pre-writer budget gate,
original fields and Team roles/prompts. An unrelated dev grading change was reverted.

### Verification
Final full benchmark plus fit-outcome and learning-qualification regressions:
18 passed in 70.03s, exit 0. Team failure then success records indices [0,1],
first_valid=False and final valid=True. XML and exact source hashes are recorded in
.progress-audit/team-execution-repair-20261108.json. Production v4 CPU job3607936
is still pending; local verification does not certify the production snapshot.

### Metadata
- Related Files: fin_skills/rsi/methods.py, tests/test_rsi_methods.py
- Pattern-Key: evaluation.team_execution_identity
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-11-08

## [HEAL-20261108-SIT] interrupted_terminal_json

**Logged**: 2026-11-08T23:29:00Z
**Status**: verified
**Trigger**: independent fault-injection failure
**Area**: experiment ledger
**Priority**: high

### Failure
The independent reviewer reproduced two synthetic S parent failures: a truncated
worker terminal or accepted marker raised JSONDecodeError and prevented the parent
terminal from being written. No real model or Slurm job was used in those probes.

### Fix
The S parent records worker/marker read errors, marks invalid, verifies complete
accepted units against original reports and provider events, continues partial
cost indexing and writes its authoritative terminal. The analyzer uses the same
accepted-unit verification. Old P/E code is unchanged.

### Verification
Both parent truncation regressions now require an invalid terminal, no accepted
unit and one unknown-cost call. Complete S helper tests: 14 passed in5.07s.
JUnit: .progress-audit/seven-study-interruption-repair-20261108.xml. Production
S coordinator qualification remains pending.

### Metadata
- Related Files: scripts/run_seven_study.py, scripts/seven_study.py, scripts/analyze_seven_study.py, tests/test_seven_study.py
- Pattern-Key: evaluation.interrupted_terminal_json
- Recurrence-Count: 2
- First-Seen / Last-Seen: 2026-11-08

## [HEAL-20261108-SFP] slurm_literal_field_parser

**Logged**: 2026-11-08T23:45:00Z
**Status**: verified
**Trigger**: remote CLI stderr
**Area**: scheduler metadata
**Priority**: medium

### Failure
The type-comparison helper inserted CPUs/Task into slash-delimited sed syntax,
producing unknown-option errors. Command substitutions yielded empty strings, so
the shell comparison did not validate that field. This was not an SSH/auth failure.

### Fix
Stopped using a regex substitution for scheduler keys. Verified all recorded
before/H100/restored snapshots and fresh readback with escaped literal field
matching. Subsequent route comparison parses whitespace-delimited key=value fields
in Python, preserving the literal CPUs/Task key.

### Verification
All ten resource/dependency fields, including CPUs/Task=2, were equal across four
snapshots. Same owned job remained pending, returned to H200 with45min/2CPU/32G,
no requeue and no duplicate submission. The repaired route comparison then checked
both ordinary compatible GPU types and restored the original route, exit0.
Evidence: .progress-audit/mistral-type-invariants-verified-20261108.json and
.progress-audit/mistral-owned-route-recompare-20261108T2345.jsonl.

### Metadata
- Pattern-Key: slurm.literal_field_parser_slash
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-11-08

## Verified production follow-up for HEAL-20261108-TEAM and HEAL-20261108-SIT

Immutable v4 CPU job 3607936 passed the complete146-test declaration with zero
failures/errors/skips. The independent S coordinator job3607977 passed all18
declared checks under actual v4 imports, including three Linux process-group
checks. Exact source/composition/CPU evidence was independently reviewed before
the full70-unit S plan was frozen. First formal job3607999 then completed both
NoImprove library units (36 test episodes), with36 provider requests,20651 generated
tokens and no unknown/unfinished costs. These are finite engineering/first-unit
closure evidence; they do not claim full scientific acceptance or a completed
Team model comparison. See .progress-audit/seven-production-first-progress-20261108.json
and the immutable independent production-prerequisite review.

## [HEAL-20261109-R5] admit declared extension models without replacing a frozen plan

**Status**: verified locally, independently and in production CPU allocation

The old P/E runner generated all20 model cells but rejected non-core chunks and
looked up qualifications only in the initial plan. Independent execution of that
old runner reproduced the first rejection; removing it exposed the missing
qualification entry. New allocation-level admission pins a later extension's
receipt path/hash while keeping the full population and every already frozen
qualification unchanged. Parent, worker and analyzer verify source, model,
revision, full declared generation configuration and the same admission identity.
P/E interrupted terminals now reuse the existing S provider-accounting parser and
verify complete hashed reports instead of trusting completion markers.

Verification: final local related suite62passed/3Linux-only skipped,6.65s,
`.progress-audit/PE-admission-strict-contract-local-20261109.xml`. Independent
review exercised real20-model synthetic admission/analyzer composition, extension
worker load-boundary sentinel, mutated receipt/backend/inventory and old-claim
refusal; three Linux process-group tests passed separately. A final exact six-line
strict-generation delta passed27 independent tests and ten bad/missing-field probes.
No model or dataset score was generated for this verification. Deployment awaits
resolution of the separate16k/32k scientific contract conflict and new bounded CPU
acceptance; immutable v2/v4 and the active S deployment remain unchanged.

- Pattern-Key: evaluation.frozen_plan_extension_admission
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-11-09
- Evidence: `.progress-audit/r5-extension-admission-structural-closure-20261108.json`;
  `.progress-audit/PE-strict-generation-gate-independent-review-20261108.json`.

Production follow-up: user-delegated native-ceiling amendment resolved the explicit
contract conflict prospectively. New immutable P/E coordinator job3608226 passed69/69
checks under actual v2 imports, including Linux supervision, extension admission,
context boundaries and raw/derived cost accounting. No model calls were made for
these checks; complete fixed20 tokenizer preflight is a separate stage in that job.

## [HEAL-20261109-CUDA-INIT] diagnose allocated GPU startup without retrying experiment

**Status**: verified startup recovery; scientific job completion remains separate

S job3608064 received one L40S on ihccs418, but torch CUDA initialization returned
unknown error before model loading. Frozen worker and parent both record invalid;
no loaded.json or provider outputs were created, while exclusive claims remain.
Five afterok dependent jobs were canceled by Slurm without running. Preserve those
receipts and claims. One finite5-minute untyped GPU diagnosis3608228 will inspect
driver/Slurm visibility and compare batch versus srun initialization and an exact
small BF16 check. This diagnosis does not itself restart the scientific experiment.

- Pattern-Key: beacon.cuda_initialization_before_model
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-11-09
- Evidence: .progress-audit/fin-cuda-recovery-20261109/release.json;
  .progress-audit/context-production-and-recovery-progress-20261109.json

Diagnosis3608228 reproduced the failure on the same ihccs418 node in both batch and
srun, with driver615.71.09/CUDA13.0. Step mapping did not repair it. Recovery3608259
therefore excludes only that demonstrated incompatible node, keeps untyped gpu:1,
and checks current-allocation CUDA before moving either failed claim into immutable
history. Four file-only recovery tests passed. Actual recovered model execution
was pending at that point. Recovery3608259 subsequently passed CUDA on ihccs601,
loaded the unchanged Phi4 model and passed the full-context probe. Both old claims
were archived byte-for-byte before resumed execution. This verifies startup recovery,
not completion or quality of its original36 scientific cells.

## [HEAL-20261109-TOKENIZER-DEPS] isolate optional tokenizer conversion dependencies

**Status**: verified in production

CPU3608226 passed all69 source/coordinator checks and produced16 fixed-model profiles,
then pinned Mistral7b tokenizer conversion required missing SentencePiece and Protobuf.
The new finite CPU3608248 installs sentencepiece0.2.1/protobuf5.29.5 into its own
directory, reuses cached public metadata and unchanged immutable profile code, and
retains every old partial result. Shared environment, models, precision and weights
are untouched. Full20 preflight completion, not installation, verifies this repair.

Verification:3608248 COMPLETED in172seconds, all20 profiles complete with zero
initial overflow, each covering P108/E2202 fixed cells. Terminal SHA
d2496a65019ae1c21f27f2dd01e0ac098c17fd6eb51a53974726f42981325e91.
No weights or model outputs were generated. Actual multi-turn completion is separate.

- Pattern-Key: beacon.tokenizer_conversion_dependencies
- Recurrence-Count: 1
- Evidence: .progress-audit/context-profile-dependency-failure-20261109.txt;
  .progress-audit/fin-context-dependencies-20261109/release.json

## [HEAL-20261109-SLURM-LAUNCH] preserve LF and whole-minute allocation budgets

**Status**: verified submission and resource admission; model execution separate

Windows write_text translated a new shell wrapper to CRLF; sbatch rejected it before
submission. An append-only byte-written LF wrapper preserved all command lines and
was accepted. Slurm rounded04:53:25 up to04:54:00, which would fail exact released
resource admission. Held the still-pending3608253, prepared a lower293-minute release,
verified validate_allocation against04:53:00, then canceled the0-second pending job
and submitted3608259 with an exact matching04:53:00 controller value. Original files
and archive remain intact. No model call, scientific retry, or resource increase occurred.

- Pattern-Key: slurm.generated_launcher_and_minute_rounding
- Recurrence-Count: 1
- Evidence: .progress-audit/fin-rsi-seven-startup-recovery-20261109/launcher-line-endings-addendum.json;
  .progress-audit/fin-rsi-seven-startup-recovery-20261109/minute-budget-addendum.json;
  .progress-audit/fin-rsi-seven-startup-recovery-20261109/submission-minute.txt

## [HEAL-20261109-HF-METADATA] read publisher revision metadata after web-reader failure

**Status**: verified

The web reader returned Internal Error for the two official pinned Phi4 blob JSON
pages. This was not evidence that the publisher or revision was unavailable.
The same official revision's resolve/config.json and resolve/tokenizer_config.json
were fetched with Invoke-RestMethod, each completing successfully. The extracted
metadata showed16384 native/tokenizer context and no RoPE scaling. No credentials,
weight downloads or model calls were involved. The independent reviewer separately
verified the pinned official README rather than claiming to have read these JSONs.

- Pattern-Key: web.pinned_hf_metadata_reader_failure
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-11-09
- Evidence: `.progress-audit/phi4-native-context-conflict-20261109.json`.

## [HEAL-20261109-METADATA-BYTES] distinguish local serialization from remote evidence bytes

**Status**: verified

The local enrollment terminal copy used CRLF plus a terminal newline; its1dc2 hash
was incorrectly described in messages as the remote original hash. Byte-level
normalization of exactly those separators (no JSON parsing/reformatting) produces
the remote2048-byte c30b hash exactly. All8 payload file hashes are unchanged.
Preserve both receipts, name which byte representation each hash identifies, and
bind runtime plans to the actual remote original. No scientific rerun is needed.

- Pattern-Key: provenance.metadata_text_serialization
- Recurrence-Count: 1
- Evidence: .progress-audit/enrollment-serialization-identity-20261109.json

Operational note: two combined documentation patches were atomically rejected because
one context fragment omitted its line prefix and an empty later hunk pointed backward.
After reading the exact target, removing the unrelated hunks produced the intended
receipts/docs; no partial or duplicate files were present. Keep patch hunks in file order.

## [HEAL-20261109-CONTINUATION-FIXTURE] test recovery with the declared synthetic mode

**Status**: verified locally; deployment qualification recorded separately
**Trigger**: four new fault-injection cases used execution_mode=model with a scripted backend and no Linux confinement hook. The production guard correctly refused before the intended interruption.

The tests now identify themselves as trusted fixtures while exercising the real Meter, benchmark, immutable records and original P/E entry point. No production confinement gate changed. Fifteen fault-injection cases passed locally; they include durable response before Meter receipt, completed fit and episode reuse, original judge execution replay without early grading, checkpoint publication interruption, changed identities and unresolved provider requests.

- Pattern-Key: testing.synthetic_recovery_mode
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-11-09
- Verification: python -X utf8 -m pytest -q tests/test_study_continuation.py — 15 passed; no actual model calls.

Production verification: scavenger CPU job 3608744 completed 0:0 in 104 seconds,
with all 15 cases passing on each immutable v2 and v4 runtime, no skipped cases
and no actual model calls. Evidence: .progress-audit/continuation-CPU-result-20261109.txt.
This qualifies the component, not allocation ownership or a formal GPU continuation.

## [HEAL-20261109-STAGING-BRACES] quote generated dictionary literals before staging

**Status**: verified
**Trigger**: a nested f-string interpreted a dictionary in the Phi4 E-tail preparer
as the outer Python expression, raising NameError before any remote upload.

Escaped the inner dictionary braces. Verified the failed preparation directories
were empty before reuse, parsed the generated Python, checked exact disjoint
chunk counts and resource fields, then uploaded each archive once. Submissions
3608822 and 3608823 bind 1050 and 1002 original cells respectively; 3608822
subsequently started. This failure caused no model call or duplicate submission.

- Pattern-Key: staging.nested_fstring_dictionary
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-11-09
- Evidence: .progress-audit/stage-remaining-E-phi4-a-20261109.py;
  .progress-audit/stage-remaining-E-phi4-b-20261109.py;
  .progress-audit/remaining-E-phi4-submission-20261109.txt

## [HEAL-20261109-LEGACY-OWNER] bind the legacy job's verified owning release

**Status**: verified
**Trigger**: the retarget guard refused legacy job 3607734 because its Comment
did not contain the owner chat, before any scheduler mutation.

Verified the original remote release owner, exact job name and launch command;
the local copy is .progress-audit/core-qualification-20261108/release.json.
Added the matching owner Comment while held and verified it in readback. This
establishes ownership only; the later administrative hold is unresolved below.

- Pattern-Key: slurm.legacy_owner_comment
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-11-09

## [HEAL-20261109-ADMIN-HOLD] preserve a denied release after route comparison

**Status**: pending-verify
**Trigger**: Slurm 26.05.5 denied release of owned pending job 3607734 after
its third held resource-route update; readback is JobHeldAdmin, Priority=0.

Two earlier routes released successfully. The transition followed the update
back to beacon/default/H200; the exact controller cause is unproven. The job
still has zero runtime and unchanged 45-minute, 2-CPU, 32-GiB, no-requeue limits.
This is a scheduler authorization failure, not an SSH or authentication failure.

Fix: none applied. Stopped mutations, retained the original job and evidence,
and prepared a support message. Ordinary job-owner release cannot remove an
administrative hold. Do not bypass it with priority changes, cancellation and
resubmission, privileged commands, or repeated equivalent release attempts.
Other healthy jobs continue. Support mail requires explicit user authorization.

- Pattern-Key: slurm.pending_retarget_admin_hold
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-11-09
- Evidence: .progress-audit/mistral-release-error-state-readback-20261109.txt;
  .progress-audit/mistral-admin-hold-diagnosis-20261109.txt;
  .progress-audit/mistral-admin-hold-support-draft-20261109.txt
- Primary reference: https://slurm.schedmd.com/scontrol.html (hold and release).

## [HEAL-20261109-REPLAY-CONTEXT] preserve a unique child context request sequence

**Status**: verified locally and by independent targeted reproduction; new runtime
qualification recorded separately under CPU job 3608908.
**Trigger**: independent review reproduced a completed E prefix followed by fresh
generation with duplicate context request_index=0. The report completed but the
unchanged context-ledger validator correctly refused it.

Replayed checkpoint events now reserve indices from the actual child context guard;
new requests continue that sequence. Replay metadata retains each original event
hash and original index; parent bytes are unchanged. Missing or different guards
fail closed. The new fault case interrupts after one cell, resumes through three,
then resumes all four; physical requests and program executions remain four, all
logical context identities are unique, and old evidence stays byte-identical.

- Pattern-Key: continuation.context_sequence_after_completed_prefix
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-11-09
- Verification: full local 18-case intermediate suite passed; independent targeted
  mixed-prefix case passed. Subsequent original external checkpoint-gap case passed
  separately (1 passed,18 deselected); combined new deployment contains 19 cases.
- Files: scripts/study_continuation.py; tests/test_study_continuation.py

## [HEAL-20261109-CHECKPOINT-INTENT] reconstruct a deterministic checkpoint commit gap

**Status**: verified locally and by independent targeted reproduction; new runtime
qualification recorded separately under CPU job 3608908.
**Trigger**: interruption after fit/episode intent fsync but before writing complete
bytes prevented recovery, despite all preceding provider/program receipts existing.

Only fit/episode checkpoint intents may reconstruct through strict existing operation
replay after binding/request verification. The same original intent is retained while
the deterministic value is atomically completed. Existing pending complete bytes
still require explicit exact-byte recovery. Provider and program intents lacking a
complete response remain unresolved and are never implicitly reissued.

Tests cover both checkpoint kinds, pending-byte preservation, unresolved provider
refusal, and the original E entry point: recovery produces two cells using exactly
two total provider calls and two original scalar executions. The independent reviewer
also reproduced the corrected cases without changing source or accessing Beacon.

- Pattern-Key: continuation.deterministic_checkpoint_intent_gap
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-11-09
- Files: scripts/study_continuation.py; tests/test_study_continuation.py

Production verification for both continuation fixes: CPU job 3608908 completed
0:0 in194 seconds; all19 cases passed with zero skips on each immutable v2/v4
runtime. Component SHA20447afa2f1ae83a4f260050bec0c5ce1ebfaddfc7d96b406c6cbeac7ca0a1ca.
Evidence: .progress-audit/continuation-v2-CPU-terminal-20261109.txt. No real model
calls, formal allocation takeover, cumulative budget or full S continuation is
claimed by this component receipt.

2026-11-09 ETA recovery note (beacon.shared_master_owner_changed): prior master74308
and parent32460 were absent and zero SSH clients existed. After reading shared pause
history, one hidden standard-helper invocation created master19780(parent15284).
SharedOnly queue/receipt read succeeded with no remote pause; no blind retry. The
connection is retained for the now-authorized continuing original wave. Other owners
and the administrative job hold were not altered.

Read-command note: a PowerShell -ErrorAction option was accidentally appended to rg,
which parsed it as its -E encoding flag. Re-ran the exact existing design-file search
without that shell-specific argument; exit0 and matching cache/scope lines verified.
No source, remote job or evidence changed because of that failed read. A later
documentation patch contained an empty trailing hunk and was rejected atomically;
removing that unused hunk allowed the intended documentation update to apply.
