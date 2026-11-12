# Public external evaluation and typed outputs

The next study has two separate evidence tracks. The six feedback-by-library conditions
test numerical-program repair on the frozen positions tasks. The external track tests
one-shot scalar code delivery with and without the numerical library. Its executable,
finite-output check cannot detect a wrong numerical answer, so it is not evidence that
feedback repairs financial reasoning.

The external source is the public `kensho/bizbench` release at revision
`0a793f2f886156902c72b4a22cab82bb9dceaecf`. The upstream files are
`data/train-00000-of-00001-300a28d2e31daec3.parquet` and
`data/test-00000-of-00001-7b139510152259c8.parquet`. These are pinned source identifiers;
the training file has been downloaded and hash-verified on Beacon. Slurm job 3607628
counted 14,377 training rows: 4,668 CodeFinQA, 2,856 CodeTAT-QA, 7 FinCode and 6,846
SEC-NUM. Subsequent CPU intake 3607718 verified the test file and counted 4,673 rows,
including 795 CodeFinQA and 288 CodeTAT-QA rows. These are actual file counts, not the
dataset card's counts. No formal test inference has run. The data card declares Apache-2.0. Preserve its
notices and underlying source attribution. The public release and
the separately hosted leaderboard must not be treated as identical.

The primary external endpoint uses exact numerical equality, allowing equivalent numeric
representations such as `1` and `1.0`. It does not try multiplying or dividing predictions
by 100, or changing monetary scale, to find a match. Unit instructions must come from the
public question or a source-wide convention established on training material. Section 3.1 of
the official paper uses a 1% reference tolerance; the exact-match endpoint here is a stricter,
explicitly adapted choice. An official public scoring implementation has not been verified.
These new budgets and interfaces define an adapted evaluation, not an
official leaderboard reproduction.

Before any formal E inference, a separate addendum freezes a descriptive 1% relative-tolerance
sensitivity on the same final outputs: absolute error at most 1% of the absolute reference,
with exact zero required for zero references. The primary exact scores remain unchanged.
Exact rational comparisons avoid new rounding or epsilon rules. Both results use the same
complete matrix, paired conditions, source weights and dependency blocks. Reports include
exact-pass, exact-fail-but-relative-pass and both-fail counts for every condition/model, and
explicitly flag a change in effect direction. Numeric proximity does not by itself establish
semantic correctness or identify floating-point/reference-rounding error. This definition is
inspired by the [paper's methods](https://aclanthology.org/2024.acl-long.452.pdf), not a verified
reproduction of its executable scorer. Prior DEV/TRAIN exact outcomes were inspected; no
relative-tolerance scores or threshold sweep preceded this addendum.

`external_packets.py` projects explicit public fields and keeps the answer, scoring policy
and its binding in a separate parent-side record. The public packet excludes the reference
program. `external_study.py` uses the common budget meter, source snapshot, complete plan,
attempt binding and post-selection grading. The source document/template cluster is supplied
by a frozen provenance map; each question or new random seed is not automatically a new
independent unit. The two library conditions each get one call and one program attempt, with
a 2,048 completion-token ceiling. This implementation is ready for development qualification;
it is not a release of a final task/model matrix.

Both positions and scalar code share `ProgramSandbox`, the confined process transport,
bounded result protocol and candidate/infrastructure distinction. The output contracts are
separate: position programs return labelled matrices; scalar programs return the value of
their final expression. Integers retain integer representation across JSON, including values
above the binary64 exact-integer range. No question, reference value or scoring routine is
sent to the scalar worker.

Local synthetic verification does not establish production qualification of the new scalar
interface. The first production CPU qualification timed out and its dependent GPU job never
ran. Its training-source intake then exposed a missing Parquet dependency. Release
`fin-rsi-repair-20261108-v2` preserves those failed receipts, uses an isolated PyArrow 20.0.0
installation, and executes the complete test-node plan with two allocated CPU workers.
It includes the shared scalar interface and two leading CodeFinQA training rows for bounded
development qualification. Production CPU job 3607628 passed all 129 planned tests without
skips. Phi-4 mini, Granite and OLMo completed the bounded positions/scalar development qualification;
the remaining two fixed core models still require their terminal acceptance receipts.
Every scientific run still needs its exact source, model revisions, runtime, task/cluster map,
unit conventions, scope and statistical comparisons frozen before inference. Neither track
claims that public tasks were unseen in model training or privately held by an independent
custodian.

The final design supplement fixes CodeFinQA and CodeTAT-QA as the two external populations,
three generation seeds, and the five core model revisions within the retained 20-model goal.
Enrollment selects at most 200 source groups per population and one question per group by
a frozen hash ordering. Short populations are not padded with extra questions or seeds.
Only exact public question/context mappings establish source provenance; unresolved origins
retain context sampling groups but share one global dependency block. Identical normalized
training inputs are excluded before inference. Both condition prompts plus the full output
allowance must fit every core tokenizer's 32,768-token contract.

Independent review accepted the enrollment implementation. Job 3607809 stopped on upstream
table labels that collapse after normalization. The repair skips only that lossy mapping
representation, preserving the question and cells. Recovery job 3607864 completed the source
mapping, all-row eligibility ledger, joint tokenizer checks and final cell freeze: 200 CodeFinQA
and 167 CodeTAT-QA questions, or 11,010 core-model cells. All selected origins remain unresolved
and share one global dependency block. Results must therefore be descriptive for this fixed
public benchmark, without an independent-document population claim. No formal test inference
has run. The estimator gives each source weight 1/2 and
each selected group equal weight within its source. Shared issuers are resampled together
across sources. Model and seed repetitions do not create independent source documents.

The frozen coordinator and source-balanced analysis passed independent review and production
CPU qualification 3607880 (15 tests, no failures or skips). The added sensitivity analysis and
prospective addendum binding passed separate production qualification 3607924 (30 tests).
This verifies implementation and
process supervision; it does not supply the missing formal model results.

A single TRAIN-only format diagnostic (3607903) found exact normalized question candidates
for all 4,668 CodeFinQA training rows, but none of the finite complete-context reconstructions
matched, including the table markers illustrated in the primary paper. It used no answer or
reference-program field and made no model calls. This supplies no justified mapping repair;
the existing enrollment is retained. Unresolved does not mean that provenance is irrecoverable.

The independent route review is retained locally at
`.progress-audit/external-evaluation-route-20261108.json`. Its broader QuantCode route requires
an order/fill-aware Backtrader adapter and a separately qualified semantic scorer; the current
positions interface cannot supply an official QuantCode score.

Sources: [official dataset card](https://huggingface.co/datasets/kensho/bizbench/blob/0a793f2f886156902c72b4a22cab82bb9dceaecf/README.md),
[BizBench paper](https://aclanthology.org/2024.acl-long.452/),
[publisher demo pipeline](https://github.com/kensho-technologies/benchmarks-pipeline).
