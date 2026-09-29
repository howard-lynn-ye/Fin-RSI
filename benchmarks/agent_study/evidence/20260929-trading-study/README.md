# Out-of-sample trading study (frozen 2026-09-29)

Protocol: `paper/TRADING_STUDY_PROTOCOL_20260929.md`. One bundle per model (`7b/`, `14b/`,
`32b/`): frozen `inputs.json`, `protocol.json`, `job.sh`, `plan.json`, `manifest.json`
(SHA-256 of every uploaded file, including the data files and library snapshot) and the
scheduler receipt. Jobs 1767302 (7B), 1767303 (14B), 1767304 (32B), one submission each.
`local-qualification-7b-bundle.json`: controller/sandbox/ledger qualification of the extracted
bundle on a local Linux VM. Results are added only after each complete batch is fetched and
hash-verified.
