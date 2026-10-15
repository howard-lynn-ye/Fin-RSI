# Optional agent playbook improvement

`fin_skills.rsi` implements a bounded reflection loop for tool use, retrieval and experience
notes. It is disabled by default, has no network client, and selects no model or paid provider.
Importing it does not start a loop or change any existing agent. The active source29 32B trading
experiment does not import this module and remains frozen without RSI.

This first implementation evolves a small playbook through successive proposals. It does not
modify model weights, Python source, the optimizer itself, trading rules or guards. It is not
an implementation of the full GEPA search algorithm, and no financial benefit has been measured.

## Agent integration

The host supplies two independent callbacks:

- `generate(prompt) -> str`: an existing permitted model backend, with its own token, timeout
  and spending limits. `JSONProposer` builds the request; it does not open an account or choose
  a provider. The returned JSON can update exactly one entry.
- `evaluate(playbook, split, as_of) -> Evaluation`: trusted execution of development or validation
  tasks. It owns the datasets, checks and raw logs. Every case supplies a score (higher is
  better), execution-error count, guard outcome, availability timestamp and SHA256 of its actual
  evidence. Keep the corresponding raw evidence; a digest is not a replacement for that file.
  Repeated evaluation for stochastic tasks belongs in this callback and its fixed protocol.

```python
from fin_skills.rsi import JSONProposer, Policy, RSILoop, Trace

# existing_generate and evaluate_tasks are the host's configured callbacks, not tools
# exposed to the proposing model. Use a new directory for each model/protocol/seed.
loop = RSILoop(
    "runs/my-agent/rsi",
    model_id="my-local-model-and-version",
    protocol_id="tool-use-eval-v1-data-and-guard-hashes",
    available_at="2026-01-01T00:00:00Z",
    policy=Policy(enabled=True, max_proposals=3, max_evaluations=12),
    proposer=JSONProposer(existing_generate),
    evaluator=evaluate_tasks,
)
loop.observe(Trace(
    trace_id="episode-1-call-2", task_id="development-1",
    available_at="2026-01-02T12:00:00Z", kind="tool_contract",
    summary="Series.cov was called without another series.",
    source_version="agent-and-fin-skills-commit",
    request="returns.cov()", result="TypeError: missing other",
))
decision = loop.improve(as_of="2026-01-03T00:00:00Z")
advice = loop.instructions(as_of="2026-01-03T00:00:00Z")
# The host explicitly supplies advice to its agent as advisory context. Keep system
# policy, executable guards, accounting and permissions outside this editable text.
frozen_version = loop.freeze(as_of="2026-01-03T00:00:00Z")
```

An evaluator returns `Evaluation(playbook.version, split, protocol_id, cases)`, where `cases`
is a tuple of `CaseResult(case_id, score, available_at, evidence_sha256, execution_errors,
guards_passed)`. Validation case identifiers must be disjoint from development cases and all
observed development traces. The proposer never receives validation cases, labels, raw results
or evaluator code through this API. It receives development traces, the current playbook and
prior proposed edits with their decisions. Call `observe` between iterations, not recursively
inside a callback while the archive is locked.

## What the loop enforces

1. Only `tool_usage`, `retrieval` and `experience` entries can change. Stable keys preserve all
   other entries. Path-like keys, unknown fields and over-budget text are rejected; generated
   text is never executed as Python. The archive retains the original proposal response.
2. The host sets fixed call and context budgets. Started calls, including failures, consume the
   persistent budget. Restarting with another model, protocol, seed or policy is rejected for
   the same archive. Callback-internal token/time/spending limits remain the host's responsibility.
3. Future traces are excluded, and future evaluation evidence is rejected. An evolved playbook
   cannot be read before its activation cutoff. A failed future attempt cannot be followed by a
   backdated proposal. Financial loss or an environment outage alone does not trigger a code fix.
4. Candidate and baseline must cover the same cases in each split. The candidate must pass all
   executable guards with zero execution errors, regress on neither split, and improve at least
   one split by more than the fixed `min_gain`. The evaluator must choose meaningful scores;
   this gate does not itself establish statistical significance or trading profitability.
5. A rejected or duplicate candidate never changes the active playbook. Accepted versions,
   rejected candidates, failures and evaluation receipts remain in a hash-chained JSONL journal.
   `rollback(version, reason=..., as_of=...)` selects only a previously accepted version or seed.
6. `freeze` prevents further proposals, observations and rollbacks in that run. Deploy its fixed
   instructions in a separate evaluation; create a new run for further development.

## Persistence and trust boundary

The archive is a single-writer directory with fsynced `events.jsonl` and a temporary
`.writer.lock`. A second writer fails immediately. A crash can leave the lock; only the host
operator should remove it after verifying the original writer has ended. Malformed or incomplete
journal records fail closed rather than silently erasing failed attempts. The hash chain detects
accidental edits; it is not an authenticity signature against an attacker with filesystem access.

Callbacks are trusted host Python code. This package is not an OS sandbox and cannot stop a
malicious callback from using the host's files or network. Give the proposing model only the
JSON context/output channel, not filesystem access to the archive, evaluator, private labels or
credentials. Guard execution and access control must be enforced by the host independently of
the proposed advice. No model-generated score is accepted through the proposal channel.

`audit()` is a host-only copy of the complete journal. It can contain proprietary tool output
or development data; use redacted input traces and a separately reviewed public export when
publishing research. Validation evidence files stay with the evaluator; their hashes are bound
to the evaluated version, split and protocol in the journal.

## Verification and experiments

`python -m pytest -q tests/test_rsi.py` covers promotion/rejection, independent split coverage,
chronology, disabled/frozen behavior, persisted budgets, rollback, resumption and journal integrity.
These use deterministic synthetic evaluators and do not claim a model or Return Rate improvement.
The wheel/sdist smoke check also imports the module and confirms disabled operation is inert.

Finish the unchanged 32B no-library/library cycle first. A subsequent separate study can compare
no-library, library, no-library plus improvement, and library plus improvement. Keep financial
inputs and trading rules matched, disclose optimization calls, freeze candidates before the final
test window, and publish net Return Rate together with every decision and the full capital ledger.
