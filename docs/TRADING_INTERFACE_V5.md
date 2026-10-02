# Trading interface v5

This version repairs the study adapter through which an agent uses fin-skills.
It does not change allocation algorithms or the investment objective. Frozen v4
source and recorded decisions are retained. Existing v4 scores are not v5 results.

## Inputs and calls

Both arms can call read_market for recent-first raw rows, with explicit as_of,
ticker/date filters and row pagination. The raw CSV and corporate actions remain
available. This convenience reader does not adjust prices or recommend an asset.
A page is not an estimation window: read the full visible CSV in Python for fitting.

The library arm can discover all installed skill documents via list_skills(query=...),
read all pages with read_skill, and read the Markdown references listed by each skill.
No skill text is automatically injected into every decision.

All tool results are dictionaries. For a portfolio method chosen by the agent:

    card = describe_algorithm(algorithm_id="hrp")
    print(card["python_example"])

The returned example runs the chosen adapter, checks reply["ok"], and calls
submit(reply["result"]["weights"]). These study functions already exist in the
Python namespace; importing run_algorithm from fin_skills.algorithms is incorrect.
The direct package function run(algorithm_id, data, **parameters) is a separate API.

The catalog describes internally prepared inputs separately from the adapter's
accepted arguments. Unknown arguments fail with the expected signature; no strategy
or malformed code is silently rewritten.

read_file/read_skill advance by next_offset, measured in original text characters.
read_market/list_skills advance by row offsets. A null next_offset ends pagination.
Responses stay within the 4000-character budget as complete JSON. Oversized
structured outputs return an explicit error; numeric weights are never partially
returned. Decision records include the exact model_feedback string.

For Guards, ok means execution succeeded; passed is the audit verdict. Warnings in
summary still matter when passed is true. The adapter exposes adjustment_check and
data_quality; this is not a mandatory complete audit pipeline.

## Validation and scope

The regression suite reproduces the old attribute-access failure, executes the
documented HRP/inverse-volatility/equal-weight calls and submissions in the Linux
sandbox, checks old isolation probes, tests a successfully executed failed Guard,
recovers full Unicode documents/references through pagination, and verifies both
arms see the same recent raw rows. No model performance gain follows from these tests.

The v5 runner is benchmarks.agent_study.trading_study_v5, with freeze/qualify/run/score
commands matching v4. Its source receipt includes the new modules and inherited v4
dependencies. Before GPU inference, freeze a new protocol, qualify that exact source,
and run a bounded model usability check. Do not overwrite or resume a v4 pair using v5.

The investment objective remains net risk-adjusted return. Cumulative net Return Rate
is the reported primary endpoint. Changing this objective or making Guards mandatory
would require a separately declared experiment, rather than an undocumented repair.

## r2 follow-up from independent review and real-model smoke

The original v5 real-model smoke submitted no valid targets in four development
decisions. It is retained under the original model-smoke directory, not relabeled
as a passed test or an investment-return comparison.

r2 adds separate execution/delivery/combined failure counts, preserves original
Python output in traces, budgets the serialized feedback while retaining stdout
head/tail and exception text, distinguishes normal document pagination from lost
output, and reports algorithm defaults/effective parameters separately from the
adapter's history length. Invalid history or method windows are rejected.

The common prompt now documents the actual read_market rows key and raw CSV pivot
shape. The controller supplies the current call number and remaining call budget
equally to both arms, with a final-call reminder. It never inserts weights, chooses
an algorithm or repairs model code automatically.

These are development fixes after observing failures. The prior snapshots remain
frozen. A new bounded smoke must report all decisions and successful library calls;
submission success alone is not proof of library use or investment benefit.

Generation with finish_reason=length is never executed, including a parseable prefix. Feedback explicitly names the output-token limit and asks for one shorter call. Raw generation and finish_reason remain in the decision record.

## Development check observed on 2026-10-02

Beacon job 1825106 completed the same four development decisions (7B/14B, raw/library).
All four submitted valid targets, compared with zero of four in the earlier v5 smoke.
The declared combined gate still failed: neither library arm recorded a successful
adapter algorithm call. Direct package use requires separate trajectory inspection.
These adaptively reused cases establish neither unseen-task performance nor a Return
Rate gain. No complete v5/r2 return batch had been run at this publication checkpoint.

The validated r2 source was published from patch SHA256
`48c50e0443f88e53736086eb409b532d1d584adbc5a16f02bbf3ef1b7a2eada8`.
Source snapshots and development traces remain in their separate Beacon directories;
this merge does not replace frozen v3/v4 run snapshots or their recorded results.
