# P-adopt: does a model use the library when it has a fair chance? (draft, 2026-11-12)

Draft pre-registration for a follow-up to the frozen RSI study. Not frozen; nothing here has run.
It is meant for the ACL 2027 version (ARR deadline 2027-01-04), not the FinFM short paper.

## Why

The frozen study (tracks P, E, S; 17 of 20 models finished on 2026-11-11) found:

- One `positions()` call reproduces the hidden reference on every held-out P task (6 tasks x 3
  seeds, tolerance 1e-8): the held-out rules are library algorithms and variant a uses their
  default parameters.
- The library arms almost never make that call: 25 programs that called the library ran,
  across 17 models; 6 episodes were valid with a library call.
- The P task text describes the rule in words and never says which library algorithm it is;
  the system prompt asks to "implement the rule exactly".

So the frozen library contrast measured whether a model *discovers and adopts* the library
from interface text, not what the library adds once used. This study separates the two.

## Questions

1. Told which library algorithm implements the rule, how often does a model call it, and get
   the task right? (L2 against L1)
2. Retrieving the matching catalogue entry from the rule text, with no hand-written pointer,
   does that close the gap? (L3 against L1)
3. Does the interface text by itself change anything against no library? (L1 against L0)

## Conditions

All on branch `claude/frozen-study-fixes` with the 2026-11-12 fixes (fence parsing, no-library
traces, output instruction last). Same 6 held-out tasks, 3 seeds and judge as the frozen study;
one feedback setting, `Check`, with at most 3 attempts.

| Arm | System prompt | Task prompt adds |
|---|---|---|
| L0 | no library (`import fin_skills` refused) | nothing |
| L1 | library interface text, as frozen | nothing |
| L2 | as L1 | one sentence naming the algorithm: "The library implements this rule as `positions("<id>", ...)`; check its parameters and factor convention against the rule." |
| L3 | as L1 | the catalogue entry a fixed lexical retriever ranks first for the rule text, with its docstring |

- Variant b tasks need non-default parameters and event-ratio factors, so L2 still tests whether
  the call is made correctly, not only whether it is made.
- L3's retriever is deterministic and frozen before any run; its top-1 accuracy on development
  and held-out tasks is computed and reported in advance (it may well be 100%, in which case L3
  measures the retrieved text rather than retrieval).

## Models and generation

The protocol's 20 open-weight models, same revisions and precision, temperature 0.1. Output cap
4,096 tokens for every model (the frozen 2,048 cut off long reasoning: deepseek-r1-distill-14b
reached it in 170 of 297 P replies). Where a chat template has a thinking switch (Qwen3), it is
set off and recorded.

## Measures

- Primary: valid (ran, passed every visible check, matches the independent reference).
- Secondary: share of programs that call the library; accepted but wrong; no program in the
  reply; calls and tokens.
- Episodes per model: 4 arms x 6 tasks x 3 seeds = 72.

## Analysis (declared now)

- Per model: valid counts per arm and the three contrasts L2-L1, L3-L1, L1-L0, by algorithm.
- After all models finish: the same contrasts paired on model, task and seed, averaged within
  each of the 3 algorithm clusters. Three clusters cannot give p < 0.25, so results are
  descriptive; no significance claims.
- Every refusal, truncated reply and failed program stays in the denominator.

## Validation before release

A development-only job (development tasks, two models) checks that every arm's prompt renders,
that L2's pointer names the right algorithm, that L3's retrieval output matches the frozen
record, and that the judge and grader run end to end. No held-out task is run before release, and
nothing is tuned on held-out results.

## Compute and schedule (estimates)

- Frozen P took 2-6 hours per model for 108 episodes; 72 episodes with up to 3 attempts should
  take roughly 1.5-4 hours per model, about 30-80 GPU-hours for 20 models.
- Models up to 14B fit 48 GB cards and request untyped GPUs; the 24B-32B models need 80 GB-class
  cards.
- Queue estimates on 2026-11-12 put any new GPU job at about 2026-11-19, so results land after
  the FinFM deadline and before the ACL deadline.

## Open for the user

- Whether to add `PlaybookLearning` under L1 (does learning on development tasks discover the
  library?). It adds two fits and 18 episodes per model.
