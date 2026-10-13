# Access-qualified replay: first completed pair

This is an intermediate snapshot of the new 13-model, three-seed HF campaign.
One seed-11 Qwen2.5-Coder-7B pair is complete and independently audited. The other
planned pairs are explicitly listed as incomplete; this is not the 20-model result.

| Model / seed | No library Return Rate | Library Return Rate | Difference |
| --- | ---: | ---: | ---: |
| Qwen2.5-Coder-7B / 11 | 35.23% | 40.32% | +5.09 percentage points |

Both arms start with USD 100,000, cover 2025-01-03 through 2026-09-25, and complete
44 dated decisions. Return Rate is cumulative after modeled trading costs, not
annualized: `100 * (ending capital / initial capital - 1)`. Full precision and
ending capital are in `summary.json`. Final decision failures are zero in both
arms; intermediate invalid responses remain recorded.

The model passed separate real tool-access checks, but made zero research/library
calls during this full cycle. Its return difference therefore does not establish
benefit from executing library tools. Library usage is autonomous, not forced.

`ledgers.json` contains all 88 dated decisions, prior holdings, selected targets,
research observations, daily NAV and trades. Original-record hashes bind the
export to the private Beacon records. The export omits raw prompts, source prose
and model response text. No decision or return was edited.

Independent export/audit ran as a CPU-only Slurm step in allocation 2124996, using
`v9-access-20261013-candidate4/source`. Accounting was recomputed and required to
match the already stored independent audit. All planned-but-unfinished cases and
any invalid cases are reported; the audit found no invalid completed pair in this
snapshot. Current historical-availability assumptions and model training exposure
remain unverified, so these are retrospective development results.

`evidence-id-audit.json` checks six rejected document requests from the previous
Qwen3 seed-11 run against the exact original input packets and worker snapshots.
All six requested IDs were absent from both; the correct records used a different
source prefix. The worker did not lose an ID that the prompt had supplied. This
finding does not establish a causal effect on Return Rate.
