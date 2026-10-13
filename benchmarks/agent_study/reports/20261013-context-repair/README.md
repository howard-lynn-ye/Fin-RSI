# Current-observation and tool-route repair

This repair follows the retained [three-model preflight](../20261012-reviewed-preflight/README.md).
Its source starts at master `037cc0c`, including V2 integration `f29f5da`.
New freezes use interaction revision `20261013-structured-final-v1`.
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


## Structured final output

The [second attempt](attempt-2/README.md) completed 16/18 fixed-date cases. Its two remaining
failures put CASH beside allocation. Both failed attempts remain published; neither launched a
full-cycle comparison. The new repair constrains final-stage JSON structure with the pinned
[lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer) 0.11.3 transport. It does not
choose HOLD, assets, allocation amounts or a default cash value. Research and early decisions
remain in the first six responses. Both arms get the same final two-response transport and budget.

Strict JSON and financial validation still apply. A grammar-accepted string or number that strict
JSON rejects is retained as an interface error and consumes the existing correction budget. A
response the grammar could not produce stops the run with a private infrastructure diagnostic.
Neither class is silently salvaged, scored as an intentional HOLD, or replaced by a policy.

The protocol binds decoder version, schema hash and parser configuration. Per-response receipts
bind the actual backend model ID, revision and token budget. The preflight additionally requires
one real final-stage generation per model, even when all calendar cases submit early. That extra
response is stored separately, never executed, and never passed into the paired experiment's
history. Each real case resets seed/call counters. This is an additional interface qualification
cost, not an equal-inference-budget comparison with the older protocols.

Claude and Gemini reviewed the supplied implementation independently. Their verified receipts,
accepted repairs and source-checked objections are in [schema-review.json](schema-review.json).
The personal conversation path declares text transport separately. Unqualified model families,
reasoning templates and closed API transports are not silently substituted into the HF protocol.


The [structured-final preflight](attempt-3/README.md) completed 18/18 decisions and 3/3 real decoder
checks, independently verified by Slurm audit 2124808. Full-cycle array 2124818 was then submitted;
this launch is not a completed Return Rate result.
