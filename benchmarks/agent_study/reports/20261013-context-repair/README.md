# Current-observation and tool-route repair

This repair follows the retained [three-model preflight](../20261012-reviewed-preflight/README.md).
Its source starts at master `037cc0c`, including V2 integration `f29f5da`.
New freezes use interaction revision `20261013-single-action-envelope`.
The first attempted revision is retained below with its failures.

The task distinguishes the visible observation cutoff from the future valuation deadline. It shows
the current CASH fraction, explains that allocation entries share one unit, and asks for one current
reply. Conflicting HOLD and allocation instructions remain rejected. Recovery feedback lists the
valid research and decision choices without selecting an investment.

RAG's package call templates previously advertised direct tool names that this study cannot call.
The replay now renders those templates through the real `call_library_tool` route, preserving the
package argument schema and offline policy. Wrong-route errors explain discovery and retry; they
do not execute a substitute call. `research_context` documents its automatic visible-evidence input.

Decision validation rejects abandoned failures, altered turn order/phase, and an earlier accepted
decision hidden by later turns. Preflight observations are recomputed from retained originals and
separate successful submission, delivered research, failed/truncated responses and library receipts.
A model may choose no tools. Its result is reported as access offered, with no observed library use.
`passed` remains a decision-completion gate, not proof of profitable or comprehensive library use.

Claude Fable 5.1 (xhigh) and Gemini 3.8 Flash High (high) independently reviewed the supplied original
sources. Both returned successful, nonempty, exact-model-verified responses. See [review.json](review.json)
for scope, fingerprints and disposition of findings. They did not certify the final patch.

Historical evidence is still a current-archive backfill with declared eligibility assumptions.
New protocol, score and aggregate files explicitly carry `historical_availability_verified=false`;
verification rejects relabeling this dataset as a verified historical vintage. This prevents an
unsupported claim; it does not repair missing archives. Training exposure to the development market
path remains unverified. Old results and failed attempts remain intact.

Validation and fixed-date model replay are recorded separately as they complete. Preflight cases are
independent all-cash accounts at the first, middle and last decision dates, not a Return Rate path.
The sampling seed equals the corresponding full-run decision seed; these are development checks,
not independent holdouts. Neither an interface fix nor passing tests establishes positive returns.


## Retained first attempt and correction

Slurm array 2124644 completed the fixed-date cases; audit 2124667 verified all 18 original records.
This attempt passed 110 targeted tests and all 21 GitHub CI jobs at `81abb8d`, but it did not pass
the model decision gate. Passing software tests did not establish model interface compliance.

| Model | Raw completed | Library completed | Library execution observed |
| --- | --- | --- | --- |
| Qwen2.5-Coder-7B | 2/3 | 2/3 | 2/3 library cases |
| Granite3.3-8B | 3/3 | 0/3 | 3/3 library cases |
| Qwen3-4B-Instruct-2507 | 3/3 | 3/3 | 2/3 library cases |

The new human-readable `REBALANCE` heading was copied as a tool name or JSON key. This was a
regression introduced by the attempted prompt repair. Other failures included final-stage research
requests and valid allocation JSON followed by prose. Nothing was silently executed or converted
into an intentional HOLD. See [the verified summary](attempt-1/verified-summary.json) and the three
public per-case reports beside it. Original model messages remain in the frozen Beacon run.

The correction advertises actual `submit`/`hold` JSON tool envelopes in both stages and directs
optional submit explanations into `arguments.rationale`. Exact legacy whole-response forms remain
accepted for compatibility. `hold` takes empty arguments; submit may carry a string rationale.
No new alias, text-mining parser or default investment was added. Both reviewers checked the
canonical path; details and bounded remaining choices are in [envelope-review.json](envelope-review.json).

Candidate4 passed build/index/spec checks and 112 targeted tests (Slurm 2124699). Two additional
canonical-envelope regression cases and a fresh identical three-model replay are tracked in
[validation.json](validation.json). The previous failed attempt is not relabeled as a pass.
