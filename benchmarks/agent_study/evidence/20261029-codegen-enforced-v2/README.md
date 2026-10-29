# Enforced library study v2 (frozen and submitted 2026-10-29, Beacon job 2906547)

Rerun of `../20260929-codegen-enforced/` with the repaired library surface. The user asked
for the run on 2026-10-29. Frozen before inference: `inputs.json` (72 episodes),
`protocol.json`, `job.sh`, `SHA256SUMS`, `manifest.json` (source commit and hashes);
`local-qualification.json` is the qualification of the extracted bundle on a local Linux
VM (seed 991, no model, pandas 2.2.3); the job repeats it on Beacon before inference.
`submission.json` records the single submission. Results are added only after the
complete batch is fetched and hash-verified.

Harness: `benchmarks/agent_study/codegen_enforced_v2.py`, with
`codegen_controller_worker_v2.py` (controller) and `codegen_worker_v2.py` (grader). The v1
files and evidence are unchanged.

| | v1 | v2 |
| --- | --- | --- |
| Tasks, seeds, models, attempts, tokens, temperature | E4 | same |
| Raw-arm prompts, checks, decisions | v1 | prompts byte-identical; checks and accept/reject decisions identical on all 97 v1 raw attempts under pandas 2.x; feedback identical up to paths, timings and the worker file name in tracebacks |
| Library interface text | `run(algorithm_id, data_dictionary, ...)`, no data keys | real signatures, data keys named, `positions()`/`adjust_prices()` documented |
| Library-arm prompt | interface only | plus the task's rule cards (`fin_skills.api.rules`) |
| Library-arm feedback | "check failed" | plus cause and rule card per failed check, never reference weights |
| Call tracer (controller and grader) | `run(name, data, **kwargs)`: correct keyword calls raised TypeError | `run(algorithm_id, data, **parameters)`; `positions()` traced |
| Grading | E4 equations and probes | same equations and probes; secondary strict lag audit (execution_lag guard, per-asset factors) |
| Controller inputs | one object reused by solve() and its checks | fresh copies per call (a program that edits its arguments can no longer fail its own checks) |
| Worker limits | none | printed output capped at 64 KiB per stream, address space 16 GiB |

Optional arm `rules_guards` (rule cards and explained feedback without the library) is off by
default; adding it makes 96 episodes and needs its own confirmation.

Results depend on the pandas version: under pandas 3.0.5, 4 of the 97 v1 raw attempts change decision (stricter in-place dtype rules), so the run uses the v1 environment (Python 3.12, pandas 2.2.3) and records versions in `inference-started.json`.

Local verification on 2026-10-29 (CPU, no model): qualification passes on seed 991 under
pandas 3.0.5 and under pandas 2.2.3; a scripted three-arm run is scored end to end
(`tests/test_codegen_enforced_v2.py`, slow tests included). The Beacon job repeats the
qualification before inference and stops if it fails.

To package: freeze locally with the final source (`python -m benchmarks.agent_study.
codegen_enforced_v2 freeze <root>`), bundle the source as in v1, record SHA-256 in a
`manifest.json`, then stage and submit `job.sh` only after confirmation.
