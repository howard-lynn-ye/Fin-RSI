# P-adopt: does a model use the library when it has a fair chance? (v2, 2026-11-12)

Pre-registration for an exploratory follow-up to the frozen RSI study, approved by the user on
2026-11-12 after review with the independent audit window. It was designed after the frozen
results were seen, so it is exploratory: it can describe what happens, not confirm a hypothesis
the frozen study set out to test. Its numbers are compared only within this study, never pooled
with frozen numbers (the output cap differs). Target: the ACL 2027 version (ARR 2027-01-04).

Code: `fin_skills/rsi/adopt.py` (arms, retriever, manifest), `scripts/run_p_adopt.py` (runner),
`benchmarks/agent_study/adopt_backend.py` (context overflow handling). Tests:
`tests/test_rsi_adopt.py`, `tests/test_p_adopt_runner.py`.

The jobs run from the source snapshot of commit b786934 on Beacon. On master, where the task
suite has grown to 34 tasks and the interface lists `rolling()`, the study code uses
`tasks.frozen_suite()` (the frozen study's 18 tasks and split, checked by hash) and
`library_text(rolling=False)`; the prompt manifests computed there equal the ones the jobs use
(`adopt.manifest` digest 2c1dacd0..., frozen P `study.manifest` digest 14cf788a...).

## Why

The frozen study (tracks P, E, S; 17 of 20 models finished on 2026-11-11) found:

- One `positions()` call reproduces the hidden reference on every held-out P task (6 tasks x 3
  seeds, tolerance 1e-8): the held-out rules are library algorithms and variant a uses their
  default parameters.
- The library arms almost never make that call: 25 programs that called the library ran,
  across 17 models; 6 episodes were valid with a library call.
- The P task text describes the rule in words and never says which library algorithm it is.

So the frozen library contrast measured whether a model *discovers and adopts* the library
from interface text, not what the library adds once used.

## Questions

1. Told which library algorithm implements the rule, how often does a model call it, and get
   the task right? (L2 against L1)
2. Given the catalogue entry a fixed text retriever finds for the rule, with no hand-written
   pointer, does that close the gap? (L3 against L1)
3. Does the interface text by itself change anything against no library? (L1 against L0)
4. Does learning on development tasks (PlaybookLearning) lead a model to the library, and is any
   gain learning in general or learning to use the library? (Playbook-L1 against Playbook-L0)

## Arms

Every arm starts from the frozen study's first prompt (`study.initial_messages`): same system
prompt, including "Implement the rule exactly", and no rule cards. The only differences are in
the table. Feedback after a failed attempt is the frozen `Check` text in every arm: the names of
the failed checks, nothing else. At most 3 attempts.

| Arm | System prompt | Appended to the task prompt |
|---|---|---|
| L0 | no library (`import fin_skills` refused) | nothing |
| L1 | library interface text, as frozen | nothing |
| L2 | as L1 | `The fin_skills library implements this rule as positions("<id>", ...).` |
| L3 | as L1 | `Library entry retrieved for this rule by text search:` and the BM25 top-1 entry (id, name, task, default parameters, caveat) |
| Playbook-L0 | as L0, plus the learned playbook | nothing |
| Playbook-L1 | as L1, plus the learned playbook | nothing |

- L2 is one sentence, no advice about parameters or factor conventions, so it adds only the
  pointer.
- PlaybookLearning is re-implemented on these prompts: no rule cards in the first prompt,
  names-only Check feedback during fitting, and development traces without library hints in
  both conditions (`trace_tools=False`). Fit on the 12 development tasks, fit seed 11, default
  fit budget (48 episodes, 6 proposals, 600 calls, 1.2M tokens). No playbook arm under L2: once
  told which function to call, little is left to learn.

## L3 retrieval, published before any run

BM25 (k1 1.5, b 0.75), query = the task's rule text, documents = the catalogue entries of the 12
algorithms `positions()` accepts (id, name, caveat, tags, objectives, task, skill), stop words
and pure numbers removed, ties broken by id. This replicates the independent window's probe.

| Task | Correct algorithm | Top-3 retrieved | Rank of correct |
|---|---|---|---|
| bollinger_reversion_a, _b | bollinger_reversion | equal_weight, min_variance, donchian_breakout | 4 |
| rsi_reversion_a, _b | rsi_reversion | rsi_reversion, equal_weight, min_variance | 1 |
| donchian_breakout_a, _b | donchian_breakout | donchian_breakout, momentum, min_variance | 1 |

Top-1 is right on 4 of 6 held-out tasks (5 of 12 development tasks). Over all 48 catalogue
algorithms none of the 6 ranks first, which is why the pool is the 12. L3 therefore mixes right
and wrong retrievals by design; results are read by algorithm.

## Models and generation

The protocol's 20 open-weight models at their frozen revisions, BF16, temperature 0.1, top-p 1.
Output cap 4,096 tokens per reply for every model (the frozen 2,048 cut off long reasoning:
deepseek-r1-distill-14b reached it in 170 of 297 frozen P replies). Episode budget: 3 attempts,
6 calls, 12,288 generated tokens. Qwen3 models are the Instruct-2507 releases, which have no
thinking mode; deepseek-r1-distill-14b always thinks and is reported separately; SmolLM3 runs
with its default template.

A reply whose prompt plus 4,096-token reservation would exceed the model's native context is not
generated; the episode ends with status `budget:context` and stays in the denominator.

Seeds: data and generation seeds 101, 211, 307 (as frozen P). Episodes per model: 4 Check arms x
6 tasks x 3 seeds = 72, plus 2 x 18 Playbook test episodes = 108.

## Measures

- Primary: valid (ran, passed every visible check, matches the independent reference).
- Funnel: called the library -> program ran -> passed the visible checks -> valid.
- Error types among library calls: wrong algorithm, wrong factor convention, missing or wrong
  parameters. A call with missing parameters can pass every visible check, so Check feedback
  cannot repair it.
- Also: accepted but wrong; no program in the reply; calls and tokens.

## Analysis (declared now)

- Every contrast reported separately for variant a (library defaults) and variant b (non-default
  parameters and event-ratio factors), and by algorithm.
- Per model: valid counts per arm and the contrasts L2-L1, L3-L1, L1-L0, Playbook-L1 vs
  Playbook-L0, and each Playbook arm against its Check arm.
- After all models finish: the same contrasts paired on model, task and seed. Three algorithm
  clusters cannot support significance claims, so all results are descriptive.
- Every refusal, truncated reply, context stop and failed program stays in the denominator.

## Validation before release

One development-only job (`--dev`: test task macd_a, seed 101, fit on four other development
tasks with a small fit budget) on two small models runs every arm through the real backend. It
must finish every unit, and every declared first prompt of every Check episode must reach the
model unchanged. It checks the pipeline only; no grade from it is read, and nothing is changed
on its results. Main jobs depend on it (`afterok`).

## Execution

- Per model three jobs: `check` (L0-L3), `playbook_L0`, `playbook_L1`. Each unit writes under its
  own job directory; a unit with a complete terminal receipt is skipped on resubmission, so a
  preempted job loses only its unfinished unit. Interrupted directories are kept.
- Scavenger partition (the beacon partition's submission limit is in use by the frozen jobs),
  no automatic requeue.
- GPUs: untyped single GPU for models up to 14B; qwen-coder-32b on two GPUs with its qualified
  dual layout; mistral-small-24b, qwen3-30b-a3b and qwen-general-32b on one 80 GB-class GPU.
- Queue estimates on 2026-11-12 put new GPU jobs at about 2026-11-19.
