# Decision-interface review, October 8, 2026

Codex inspected the source and original Beacon records. Claude Fable 5.1
(`xhigh`) and Gemini 3.8 Flash High (`high`) independently reviewed the same
source packet at commit `7f436eb`. Both returned nonempty successful reviews
with the requested model verified by the backend receipt. The consultation
was read-only; reviewer suggestions were checked against source before adoption.
`review.json` records provenance without publishing raw conversations.

## Findings and changes

The demonstrated defects concern the experimental interface. They do not
establish an error in a portfolio algorithm or explain every return difference.

| Finding | Change for new freezes |
|---|---|
| The legacy parser could execute one instruction while discarding a competing malformed decision or code fence. | Parse a complete top-level response. Reject extra fields, nested instructions and competing actions. A single complete JSON fence has the same meaning regardless of its language label. |
| Query signatures were absent from the initial prompt; the paired probe guessed unsupported `start_date` and `ticker` arguments. | Generate signatures from the actual tool methods: common readers in both arms and library tools in the library arm, before the first call. Worker failures return the exact contract. No heuristic parameter renaming. |
| Relative allocation worked in JSON but not in bound Python. | Both routes use the same allocation/weights validator and conversion receipts. |
| Research parse errors pushed the model toward a terminal decision. | Give phase-specific feedback; truncated responses explicitly request shorter output. |
| Examples contained a real asset and a numeric portfolio. | Explain the allocation fields without a recommended ticker or amount. A causal anchoring effect remains unmeasured. |
| Absolute-weight rounding was described too broadly as never scaling. | Preserve the existing 0.001 rounding tolerance, disclose it in help and record original weights plus whether rounding occurred. Over-budget orders still fail. |
| Success receipts could outlive a changed final response or violate the final-stage restriction. | Recheck model instructions, conversion receipts and the prohibition on final-stage Python. |
| One model on one date was insufficient qualification for a full batch. | Each new model/seed pair must complete independent first, middle and last scheduled-date scenarios in both arms before full-period decisions begin. |
| Progress printed only `submitted`, making HOLD indistinguishable from failure. | Also report `decision_action` and `decision_completed`. |

The completed follow-up repair uses revision `20261012-reviewed-decision-contract`. The model still
chooses all assets and amounts, including cash. Relative parts avoid asking it
to calculate an exact sum. Allocation replaces the whole portfolio; missing
assets are liquidated. HOLD preserves units and incurs no rebalance fee. The
six research / two decision-response budget is unchanged and identical in both
arms. Valid decisions can finish before the research budget is exhausted.

The preflight tests interface completion, never profit. It accepts genuine HOLD
and retains failed responses. A failed case blocks the new full-period pair;
it is not converted to HOLD or silently removed from the model roster. The
preflight has its own records and is not scored as a Return Rate experiment.
Its scope is independent all-cash accounts at three calendar positions, not a
guarantee of success at every portfolio state or date. Research-error counts
are retained even when the final decision succeeds.

## Reviewer claims that were not adopted

- Gemini attributed the old bare-fence behavior to an optional Python label.
  The actual `trading_study.FENCE` requires `python` or `py`. That attribution
  was incorrect. JSON mislabeled as Python remains a useful regression case.
- Claude flagged templates that prefill `<think>`. The existing backend
  already restores that opening marker in `decode_response`; the parser must
  not unconditionally reinterpret unmatched closing tags as valid reasoning.
- The original failure at index 14 was also over-budget. Accepting its shorter
  envelope would not make its 1.40 total a valid cash-only order.
- Neither matching portfolios in one paired probe nor a concrete prompt
  example proves a causal bias or explains the return gap. The example was
  removed as a preventive improvement.
- No default trade, target exposure, forced library call, performance-based
  qualification or successful-HOLD replacement was added.

## Validation and limits

Regression coverage includes conflicting or nested instructions, duplicate
keys, reasoning boundaries, fence labels, numeric overflow and booleans,
Python/JSON parity, raw-arm isolation, worker-delivered error contracts,
receipt tampering, failed preflight preservation and independent cash/units
accounting. Validation results and the exact tested patch are recorded in
`validation.json` when complete.

The earlier `20261008-explicit-allocation` batch and its source remain separate.
Its Granite first-date record still failed after correction: two code fences,
missing explicit CASH, and percentage-like numbers sent as absolute weights.
That record is not a result of this review repair. Existing immutable results
must not be rescored or relabeled as repaired outcomes.

Unit tests and reviewer agreement cannot guarantee that every model will follow
the interface. Real-model preflight and a subsequent complete return period
are still necessary. Library-specific tool use must be reported separately;
an accepted allocation alone does not demonstrate library use or utility.

## October 12 follow-up

Both reviewers also completed a second read-only review of the candidate repair.
The adopted follow-up fixes cover CRLF/case variations in fence labels, optional
string rationale on HOLD, deeply nested JSON, Python error hints, library
signatures before first use, real confined-worker allocation tests, and exact
Python receipt validation without applying portfolio normalization twice.
The two remaining October 8 test failures were fixture mistakes: tampering
checks tried to overwrite immutable files through the exclusive-create helper.
Those fixtures now edit only their temporary test files; the production
exclusive-create behavior is retained.

The HF backend's sampling seed and call counter are reset for every main study
decision. No seed was inferred for unsupported API adapters. Personal-chat runs
remain explicitly outside the automatic model preflight because their replies
are manually delivered through a separate provenance protocol. They must not be
reported as having passed this automatic gate. A stale gate requires a new frozen
run directory; operators must preserve the old records rather than delete them.
### Integration failures found during the first launch

The first real-model preflight (Slurm 2124502) stopped before any model decision:
`decision_workspace` was called without its required `personal` argument. This was a
bug in the newly added preflight, not a model failure or an explanation for older
returns. The first launch and frozen source are preserved. The fix explicitly selects
the non-personal workspace and creates its temporary parent directory first.

The regression now runs the actual workspace, date cutoff, evidence snapshot,
Controller, prompt, decision parsing and record verification for both v8 and v9,
with a synthetic backend and synthetic prices. Slurm 2124508 passed 66 related tests.
These are integration results, not real-model performance. The test also collects
cleanly without the optional study dependency (1 passed, 6 skipped); CI had exposed
an unconditional `bs4` import in the earlier test module.

A fresh three-model preflight was submitted as 2124509 using the corrected frozen
source and a separate output root. `preflight-r2.sbatch` preserves its launcher.
No successful model qualification or new full-period Return Rate is claimed here.
The review passes and prior unit tests did not catch this integration failure;
source review alone is not evidence of a bug-free experiment.
