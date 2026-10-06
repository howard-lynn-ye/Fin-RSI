# Additional completed legacy v6 pairs, 2026-10-05

These pairs began before the complete v7 capability repair. They are kept as legacy results,
not relabelled as repaired-interface runs or counted as a completed twenty-model study.
The reporting window is 2025-01-03 through 2026-09-25, with the frozen transaction costs.

| Model | Seed | Without library Return Rate | With library Return Rate | Difference (percentage points) |
|---|---:|---:|---:|---:|
| Qwen2.5-Coder-7B-Instruct | 11 | 25.10133177447047% | 81.80117020034436% | +56.69983842587389 |
| Qwen2.5-Coder-7B-Instruct | 23 | 30.782318967332788% | 42.220125427151835% | +11.437806459819047 |
| Qwen2.5-Coder-14B-Instruct | 11 | 48.075004484437336% | 38.50646992357068% | -9.56853456086666 |

Independent cash-and-units replay passed for all three under Slurm job 1907698. Each archived
source tree verified its own original protocol and decision receipts before the separate
accounting audit. The audit did not change decisions, scores, source hashes or the completion
files. It establishes accounting consistency, not the validity of historical source vintages.

Successful submissions out of 44 (without/with library) were 30/37, 34/33 and 40/37. Failed
execution turns were 147/122, 123/149 and 47/85. The 7B seed-23 raw arm also had one failed
delivery. These are incomplete repeat sets with substantial execution failures. Do not average
only the positive results, infer that a particular repair caused these returns, or combine
these seeds with the non-blind personally authored case.

Archived runs are under
`/beacon-projects/radfm/wy891/fin-multisource-models-20261005/r2/batch/`.

| Directory | Scores SHA256 |
|---|---|
| `7b-11` | `ad27781f192c668c1f51ac544eddb872fb39be3dbb49a4e2c329ac3abc56c437` |
| `7b-23` | `342dc2061245d523135b94d770069dad45e8f145b252bb34218a81eb2d9f2de4` |
| `14b-11` | `f6dc22341c83d4f29f7bb038ec31348115116f4a2ac82c66dbb007907a99366d` |

The auditor source SHA256 was
`4210099d28ce19e42522c58a577a4668107b7709122479ce17106f49f9f54e06`.
Each run now has an additional `independent-model-audit.json` receipt. The three older pending
array tasks (1905959_3/4/5) and their dependent aggregate (1905960) were cancelled before
starting when the new v7 batch replaced them, freeing this project's submission slots.
Future experiments use a fresh source snapshot and protocol.
