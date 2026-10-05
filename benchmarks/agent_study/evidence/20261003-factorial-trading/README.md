# Frozen four-condition trading study

The full 2,112-decision batch completed on 2026-10-04 at 07:21 UTC. The study varies
optional skill-document access and numerical-tool access for Qwen2.5-Coder 7B/14B,
seeds 11/23/37, on original and predeclared transfer assets. Read PROTOCOL.md and
ANALYSIS_PLAN.md before interpreting RESULTS.md or analysis.json.

## What the release contains

- `analysis.json`: every seed/cell, paired effects, risk, exposure, transaction-cost
  sensitivity, failure denominators, successful use/adoption, tokens and generation time;
  all fixed baselines and paired calendar-block intervals.
- `completed.json`: the original finalizer's hashes of analysis and report.
- `receipts/`: all twelve original protocols, inputs, inference receipt maps, scores,
  and completion receipts. Decision contents are not distributed.
- `pipeline/`: exact frozen operational scripts. These record original Beacon paths;
  they are an archival reproducibility record, not a portable launch command.
- `SHA256SUMS.json`: hashes of the released frozen files, verified by the paper generator.
- `verification.json` and `verification_source.py`: publication verification on Slurm,
  including source/data/pipeline/decision receipts and independent dollar-holdings replay.

The release is based on master commit `bcf42c1de08458abace3b0b1e26b93ab689854d1`.
Its tree equals the frozen source base `ff7605eaa05c260491c4d060f1301b5f7f1cf54e`;
the five `factorial_*.py` modules are published byte-for-byte from the frozen run.
Existing package and study dependencies are identified in each protocol's source map.

## Reproduce the published tables

From the repository root:

```bash
python scripts/build_factorial_evidence.py
python scripts/build_factorial_evidence.py --check
python -m pytest -q tests/test_factorial_evidence.py
```

These commands require no model inference or private market files. They check the
published receipts and aggregate consistency, then regenerate the LaTeX tables. Original
receipt validation and independent NAV replay require the controlled experiment directory;
`verification_source.py` records precisely the verification performed there. Its export
step creates a new directory and intentionally refuses to overwrite an existing export.

Full inference requires separately obtained quotes and corporate actions, the recorded
model revisions and a Linux sandbox, then the frozen source/package, corpus and protocol
files at their recorded paths (or a separately declared relocation with new receipts).
Do not claim to have reproduced the frozen run by silently replacing those inputs.
Market datasets, weights and raw model outputs remain on Beacon and are not included here.
Hashes establish identity, not unrestricted access or independent custody.

## Interpretation

The original basket is development-exposed. Transfer changes assets on the same calendar,
with overlapping economic exposures. Current skill documents are not historical knowledge,
and pretraining contamination is not excluded. Seeds vary sampling on one market path.
The experiment estimates offered access, including prompt/calling-contract changes; zero
successful skill reads prevent identifying a benefit of consumed domain knowledge.
Calendar-block intervals are descriptive. No cell or failed decision was dropped.
The earlier currency and v5 studies use different protocols and must not be pooled with
this factorial batch. No live trades were placed.

All four GPU waves and final scoring completed successfully. Earlier scheduler-only
failures (submission limits, memory request and the first dispatcher state write) preceded
formal inference and are not model failures. The successful separate dispatch extension
changed scheduling only. The smoke gate retained its non-submission and selected no run
by return. Source and scientific protocol were unchanged during full inference.
