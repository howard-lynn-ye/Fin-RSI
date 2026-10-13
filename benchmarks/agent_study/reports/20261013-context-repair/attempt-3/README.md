# Structured-final preflight: complete, independently audited

Slurm array 2124804 completed all three frozen models at seed 11. Audit 2124808 verified every
original record and the three separately stored real decoder checks. All 18 calendar/arm cases
completed valid decisions; there were no parsed-response failures or truncated responses.

| Model | Raw completed | Library completed | Library execution observed |
| --- | --- | --- | --- |
| Qwen2.5-Coder-7B | 3/3 | 3/3 | 0/3 library cases |
| Granite3.3-8B | 3/3 | 3/3 | 3/3 library cases |
| Qwen3-4B-Instruct-2507 | 3/3 | 3/3 | 3/3 library cases |

The two CASH-nesting failures retained in attempt-2 no longer occurred. Every model also passed
one explicit real final-stage generation; early submissions cannot bypass this transport check.
Strict model/revision/token-budget receipts and frozen input messages were checked. These findings
establish this batch's interface completion, not profitability or absence of all future failures.

Three research requests were still rejected: Granite requested an unknown package tool once, and
raw Qwen3 requested an unknown visible evidence ID twice. All three episodes subsequently produced
valid decisions. These were not discarded or reported as successful tool calls. The independent
audit log is retained on Beacon. No library receipt was observed for 7B, so its passing cases cannot
be described as evidence that it used library capabilities.

Preflight accounts are independent all-cash scenarios at the first, middle and last decision dates.
No Return Rate is computed here. The full-period pair is a separate, subsequently launched run
using the same frozen source, data and protocol; see full-cycle-launch.json for its launch receipt.
Historical vintage verification and training-exposure limitations remain unchanged.

All public reports are byte-identical to verified-summary.json's hashes. Private prompts, original
responses and evidence stay in `/beacon-projects/radfm/wy891/fin-multisource-models-20261005/v9-context-model-preflight-20261013-r6`.
Source is `v9-context-20261013-candidate6/source`, corresponding executable code committed at
`9faf01fda23b04ca077d502e36f71c2fee5b05d3`. Interaction: `20261013-structured-final-v1`.
