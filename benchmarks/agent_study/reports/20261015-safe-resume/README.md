# Safe continuation of the frozen 32B pair

The original launcher refuses an existing output directory or inference-started receipt.
The study can skip existing decisions, but its exclusive JSON writer also prevents finishing
a partially written scoring stage. `scripts/frozen_cycle_resume.py` is a separate host driver
for these interruption cases. It does not modify the frozen study or submit jobs.

## Preserved experiment

Qwen2.5-Coder-32B-Instruct, revision `381fc969f78efac66bc87ff7ddeadb7e73c218a7`, seed 11,
44 decisions per arm, USD 100,000 initial capital, 2025-01-03 through 2026-09-25. The pinned
protocol SHA256 is in `cycle32-recovery.json`. Model precision, generation budgets, prompts,
data, order, targets, failed decisions and scoring remain those of the frozen source29.
RSI is not enabled. No paid model or new trading experiment is introduced.

On 2026-10-15 at 19:36:54 UTC, original job 2232777 remained pending with one H200, 8 CPUs,
64 GiB host RAM and an 8-hour limit. No full-cycle Return Rate is available. The scheduler's
2026-10-16 21:56:16 EDT start estimate is not a guarantee. Recovery files are preparation,
not evidence that recovery or the full model experiment has run on Beacon.

## Recovery behavior

1. Require a new allocated Slurm job, the ended original/recorded previous owner, and exactly
   one running fin-skills job. Reject a still-queued previous job or another fin-skills job.
2. Pin source-verification manifest, frozen protocol, model identity and independent exporter.
   Run the original complete source/data verification and campaign-readiness checks.
3. Require a contiguous prefix of decisions. Validate each retained response, model identity,
   evidence packet/snapshot, date and holdings before any new inference. Infrastructure-failure
   files, truncated JSON, nonfinite values, duplicates and inconsistent scoring state stop recovery.
4. Retain existing decisions byte for byte, including explicit holds and genuine failed model
   decisions. New JSON is published atomically without overwriting existing files. Deterministic
   scoring may verify and reuse identical existing NAV/score/audit files.
5. If every decision exists, finalize receipts and scoring without loading or calling the model.
   Otherwise invoke the original frozen runner for missing decisions only.
6. Hold a kernel lock and record each attempt, allocation/device metadata, driver hash and
   before/after artifact hashes. A killed process releases the lock automatically. Recheck every
   previously sealed artifact; independently audit before publishing both complete ledgers.

Legacy records are first cryptographically sealed when the recovery driver sees them. Validation
cannot retrospectively attest that a record was untouched before that first seal. A kill during a
legacy non-atomic JSON write stops recovery for investigation; it never triggers deletion or a
replacement trade. A model call interrupted before any complete decision was saved may need to be
repeated with its original per-decision seed. Different hardware need not be bitwise deterministic.

## Deployment and use

Copy the tested driver to `fin-context-safe-resume34.py`, the configuration to
`fin-context-safe-resume34.json`, and the launcher to `fin-context-safe-resume34.sbatch` under
`/beacon-projects/radfm/wy891/fin-manual-multisource-20261005/`. Verify their SHA256 values against
the source-verification record. Do not replace the original launcher or source files.

Leave job 2232777 alone while pending/running. After it ends, inspect its exit status and output.
If it completed, audit/publish its original results without submitting a recovery. If interrupted,
read the current connection guide, inspect the remaining records and confirm no fin-skills job
exists before submitting exactly one recovery allocation. Do not automatically retry decoder
failures or resubmit in a loop. No heartbeat or duplicate fallback job is created.

Reinspect Slurm GRES, account/QOS limits and eligible capacity at that time. Supply the smallest
compatible GPU constraint as a submission option; the launcher deliberately has no GPU default.
The frozen backend uses one device and bf16 32B weights; 48GB single cards cannot hold the complete
workload. The driver requires one visible GPU with at least 78 GiB reported VRAM and records the
actual layout. Use an untyped request only when its eligible pool meets this requirement. Current
cluster resources include incompatible 48GB cards and no capability features; a compatible type
can therefore be necessary. Choose between compatible types using then-current availability.
Retain the 8 CPU / 64G / 8h budget. If all decisions are already complete and only scoring/export
remains, request CPU only: the driver skips GPU inspection and model loading. Do not quantize,
change context budgets, pin nodes or add jobs.

The launcher's positional argument is the ended job ID. Completion is confirmed by a completed
recovery journal, original `completed.json`, independent audit and both public ledgers, not by
allocation or model startup. `output/recovery/` retains all attempt records and separate exports.

## Verification and reporting

CPU tests inject inference, scoring and export interruptions, compare resumed accounting with
the unchanged one-pass ledger, preserve failed decisions, check v8/v9 contracts and reject corrupt
records or duplicate allocations. Real accounting and independent audit execute; fixture model
answers/source verification are synthetic. These tests do not validate GPU loading or returns.

Publish actual final results as raw/library initial capital, ending capital and
`Return Rate (%) = (ending / initial - 1) * 100`, including costs and all failed decisions.
Commit both full decision/valuation/trade ledgers and the frozen receipts/audit. This is one seed
on historical backfills with declared availability assumptions, not proof of causal library alpha
or a historically verified point-in-time dataset. Report either sign of the return difference.
