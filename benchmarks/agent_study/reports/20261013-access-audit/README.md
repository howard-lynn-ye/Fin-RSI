# Library access diagnosis and replay policy repair

The original completed Qwen2.5-Coder-7B seed-11 pair emitted submit on its first response
on every date: 44/44 in each arm. There were no research attempts or parsing failures.
Slurm audit 2124935 reconstructed the initial prompts, verified the library entry point
was present, and matched actual pinned-tokenizer counts and input-message hashes.
JSON serialization changes the order of displayed holdings keys; the audit checks those
values separately and then requires all other characters and actual token receipts to match.

An extra unscored GPU probe, job 2124938, demonstrated a successful read_snapshot call
and a successful research_context call from 7B. The subsequent autonomous decision
preflight again had no research calls. This establishes minimal callable access and
observed autonomous non-use; it does not establish the psychological cause of non-use,
result comprehension, complete library coverage, or a causal return benefit.

The first probe incorrectly required HOLD to pass. Granite successfully executed and
received both requested tools, but returned submit and was marked failed. Its failed
report is retained in granite-8b.json. The final gate, real-library-access-v2, measures
successful execution plus delivery; termination is recorded separately. The ordinary
decision preflight still requires valid decisions. No probe order is executed, and no
probe response or history is supplied to a scored investment episode.

The original Granite cycle also had three rejected fetch_market_data requests. The
RAG adapter exposed package call templates without the replay registry's availability.
The repair labels unavailable or unreviewed tools, removes their runnable templates,
and identifies the existing visible-market/evidence readers. Error feedback no longer
suggests a wrapper can enable a prohibited live collector. No alias is auto-executed,
no future data is introduced, and no investment choice is supplied by the harness.

Validation: candidate1 job 2124931 passed 130 tests; candidate2 job 2124941 passed 148;
final candidate3 job 2124942 passed 149, with index/package builds and validate OK.
Executable final source is v9-access-20261013-candidate3/source on Beacon. Its investment
prompt still permits early decisions; model-access probes are a separately frozen gate.
Original structured-cycle outcomes remain under their original source, in PR #32.

Expansion submitted: array 2124959 and job 2124960 process 13 open-weight model IDs,
seeds 11/23/37, with per-model frozen qualification before full cycles. A failed phase
stops that model and remains in the event inventory. Submission is not completion.
The remaining native model track is not yet a completed return experiment. Neither this
report nor the earlier three seed-11 pairs completes the >=20-model requirement.
