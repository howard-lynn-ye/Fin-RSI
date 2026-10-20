# Direct retrieval pilot: October 6, 2026

This report covers one first-date usability probe per model, with paired raw and
library arms. It does not measure Return Rate and does not add completed models
to the requested twenty-model trading comparison.

## Observed outcomes

Inference and verification finished on October 6. Verification completed at
20:56:52 UTC in Slurm job `1912548`. All six records passed source/input,
qualification, decision-receipt and prompt-chain checks. **None of the six
decision opportunities produced an accepted submit or explicit hold.** Passing
record verification does not mean the model completed its investment decision.

| Model | Without library | With library | Failed turns, without / with | Executed library RAG calls |
|---|---|---|---:|---:|
| Qwen3 4B Instruct 2507 | No decision | No decision | 3/8 / 3/8 | 0 |
| Granite 3.3 8B | No decision | No decision | 6/8 / 5/8 | 0 |
| SmolLM3 3B | No decision | No decision | 8/8 / 8/8 | 0 |

All six opportunities consumed eight responses. Qwen3 made two unsupported
argument calls per arm, then kept reading documents through the final response;
the runtime rejected that last research call. Granite also guessed arguments
and unavailable tools: its raw arm attempted `calculate_returns`, while its
library arm attempted a tool named `fin-skills` and produced two unparsed
responses. Both Granite arms requested research during the final decision stage.
SmolLM3 reached the 1,024-token generation limit on all sixteen responses, so
none was executed. These failures are preserved in [snapshot.json](snapshot.json).

The qualification checks exercised direct and Python RAG access successfully in
each library environment and rejected it in each raw environment. However, the
runtime executed no `research_context` or Python calls. SmolLM3 did emit RAG
requests, but duplicated them within a response and reached the output limit;
these were rejected before execution. Therefore this pilot
does not measure the quality or investment benefit of retrieved knowledge.
It exposes unresolved interface use and response-budget problems. The result
does not establish faulty portfolio arithmetic or a causal Return Rate decline.

### Confirmed parser defect and trace diagnosis

Granite's raw responses mixed a valid read with a later submit whose `weights`
value was a Python variable, not valid JSON. The parser counted only successfully
decoded action objects, so it could accept the read while ignoring the malformed
later action. This contradicts the one-action rule and hides useful feedback.
The current source now counts an identifiable malformed tool object when checking
for extra actions. Regression cases cover both action orders, an incomplete tail,
and preservation of the existing single Python-envelope behavior. Completed
inference records retain the original parser and outcomes.

Granite therefore did express portfolio drafts; it did not produce an accepted
executable decision. SmolLM3's reasoning delimiters were already balanced: its
raw responses contained overlong Python code, while its library responses
repeated RAG requests. Turning off extended thinking is not an established fix
for these observed failures. The remaining prompt, argument-discovery and
generation problems require a separately frozen model pilot.

[audit_response_failures.py](audit_response_failures.py) compares the frozen and
candidate parsers on the saved responses without executing any action. It also
distinguishes textual requests from executed calls. Any changed parse count is
a static diagnosis, not a counterfactual trade or a new Return Rate.
The resulting [response_failures.json](response_failures.json) shows that all
eight Granite raw responses contain submit text but no complete submit JSON;
the candidate rejects all eight as multiple actions. All eight SmolLM3 library
responses contain complete RAG request objects, but none is executable as a
single complete response. No trace contains a valid single call discarded solely
because of the generation-length stop. This narrows the diagnosis; it does not
attribute the other five failed decision opportunities to the Granite parser bug.

Slurm job `1913830` verified the public snapshot against all six original records
and all 48 prompt-chain turns, reproduced the parser discrepancy, and passed
64 related model-interface, reasoning-parser, worker and decision-boundary tests.
Index regeneration, package regeneration and repository validation passed in that
order, with generated package/catalog content unchanged. Validation retains the
existing library discovery-budget warning. These checks validate the repair and
record fidelity; they do not demonstrate improved model decisions or returns.

This pilot is not a basis for expanding this exact configuration to a full
twenty-model return campaign. Before doing so, investigate the observed argument
errors, tool-name confusion and generation truncation, then freeze and verify a
new paired pilot. Keep the common objective and equal budgets; do not insert
trades on the model's behalf or change these completed records.

## Protocol and provenance

The frozen interface is v9, from commit
`ad5a9c790aeb6a3f464bbc06d395e67dd722e159`, merged by
[PR #25](https://github.com/howard-lynn-ye/fin-skills/pull/25).
The library arm can call `research_context` for installed financial knowledge,
eligible evidence and exact tool contracts. Both arms receive the same dated
market and evidence inputs and choose their own investments.

The pilot uses Qwen3 4B Instruct 2507, Granite 3.3 8B and SmolLM3 3B, seed 11,
on the first declared decision date, January 2, 2025. Each arm starts in cash and
has at most seven research responses plus one final submit-or-hold response,
with 1,024 output tokens per response. An explicit accepted hold counts as a
decision; exhausting the response budget without one remains a failure.
No fallback allocation or extra model response is supplied.

The runner stops after that first decision date. It never executes the resulting
portfolio through the study deadline, so neither a successful submission nor a
failed decision provides a cumulative Return Rate here. The dated evidence uses
declared retrospective availability assumptions rather than verified historical
archive vintages. This pilot does not extend evidence coverage or test live
collectors inside the confined historical replay.

## Reproduction and verification

[run_pilot.py](run_pilot.py) and [pilot.sbatch](pilot.sbatch) preserve the exact
executed runner and launcher, including the original Beacon paths. The launcher
freezes and qualifies a separate pair for each family before model inference.
Original model responses and evidence remain in the Beacon run directory;
the public report contains aggregate outcomes and receipt fingerprints.

[verify_pilot.py](verify_pilot.py) must import the original frozen source. It
checks source and input hashes, model revisions, qualification bindings, each
decision disposition, the response budget, the complete prompt/feedback hash
chain, and the aggregate counts against all six saved records. It does not run
inference, overwrite records or recompute returns. Execute through Slurm with:

```bash
PYTHONPATH="$PILOT_ROOT/source" "$PYTHON" verify_pilot.py \
  "$PILOT_ROOT" "$ORIGINAL_RUNNER" "$NEW_SUMMARY_PATH"
```

The output must be a new file. The original pilot root ends in
`fin-multisource-models-20261005/rag-v9-pilot-20261006`; inference job: `1910117`.
The preceding v8 probe and the published v7 return snapshot remain separate
experiments. Differences between single-date probes on different GPUs cannot
establish that retrieval caused a change in behavior or returns.
