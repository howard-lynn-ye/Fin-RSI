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
**Status**: pending-verify
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
Final CLI/preflight integration tests: 29 passed in 8.75s on the CPU allocation 2132885. Index, package, and repository validation passed. Broader reader/protocol tests remain in progress; no repaired-model inference has run.

The user limited this project to one Slurm job total. All seven previous project jobs were
cancelled by exact ID after ownership checks. The prepared model drivers were not deployed.
The original 96G launch scripts are historical records, not valid restart instructions.

### Metadata
- Pattern-Key: experiment.decision_only_release_gate
- Recurrence-Count: 1
- First-Seen / Last-Seen: 2026-10-13
