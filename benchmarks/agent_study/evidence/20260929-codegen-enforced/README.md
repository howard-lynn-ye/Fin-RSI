# Enforced library and guard study (frozen 2026-09-29)

Protocol: `paper/CODEGEN_ENFORCED_PROTOCOL_20260929.md` (also `protocol.md` here).
Frozen before inference: `inputs.json` (72 episodes), `protocol.json`, `job.sh`, `plan.json`,
`manifest.json` (SHA-256 of the uploaded bundle, including `library-and-runner.tar.gz`).
`local-qualification.json` is the controller/sandbox qualification of the extracted bundle on
a local Linux VM (seed 991, no model); the job repeats it on Beacon before inference.
Results are added only after the complete batch is fetched and hash-verified.
