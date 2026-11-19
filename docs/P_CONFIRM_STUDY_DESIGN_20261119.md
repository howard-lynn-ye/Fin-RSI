# P-confirm: the frozen library on tasks written after the freeze

Status on 2026-11-19: **draft**. Sections 1-4 are in force. Section 5 (the confirmatory run) is not frozen until this file says so and names the library commit and the task hashes. No model may see a confirmatory task before that.

## 1. Why

P-fix (`docs/P_FIX_STUDY_DESIGN_20261118.md`) ended with: valid in 9.7% of episodes without the library, 16.1% with its interface, 29.2% when the library's `route()` first asked the model which catalogued rule the task describes; 23.6% against 64.6% for the eight models of at least 14B. That evidence has limits a confirmatory study must remove:

- six held-out tasks (three rules), 18 episodes per model and arm;
- the library was revised three times after runs on those same tasks, and the routed arm and the 14B split were added after seeing results;
- no confidence intervals, no comparison with text retrieval, no task whose rule the library lacks.

Decided with the user on 2026-11-19: fix what the records still show, freeze the library, write new tasks, add baselines, report intervals. The workshop paper keeps the P-fix v4 results; this study is for the later paper.

## 2. Development material

All 18 tasks of the frozen suite (12 development, 6 earlier held-out) and every P-fix record are development material from here on. Nothing learned from them is reported as a held-out result.

## 3. Pre-freeze fixes (commit 3ba658c5)

Read from the v4 records of the four library arms, in which 368 programs raised inside the library:

- **`window=` for a rule with one lookback.** 44 programs (38 from models of at least 14B) passed the lookback as `window=` and were refused. `positions()` now reads it as `lookback` for a signal rule that has one; other signal rules still refuse it and list their parameters.
- **The hint names the parameters.** `routing.hint(id)` gives the call with its parameter names. In F4, calls that passed no rule parameter were valid in 4 of 18 variant-b episodes; calls that passed one, in 29 of 41.
- **A located error for missing states.** 45 programs (37 from larger models) had a missing state after the first decision; 4 of those episodes recovered. The error now names the first gap (column, date, row).
- **Causes for the property checks.** A property check flagged 415 episodes; 360 had another attempt; 48 ended valid when told only the check's name. `Explain`-style feedback now carries a cause sentence for each of the three checks.

Not changed: unknown ids, wrong input types and `run()` given a table (small models; the errors already name the fix), and missing states themselves (the model's own rule left gaps; refusing is right).

**Correction to P-fix amendment 3.** It says a lexical matcher named the wrong algorithm for 5 of the 12 development tasks, "so no text matcher is used". That was the first matcher tried. BM25 over the catalogue entries with their rule lines names the right algorithm for 16 of the 18 earlier tasks from the task prompt (10 of 12 development, 6 of 6 earlier held-out). Text retrieval is therefore a real baseline here (arm C4), not a strawman.

## 4. Development run (exploratory; its grades are read)

- **Purpose.** See that the fixes work end to end with models, and choose the feedback for the confirmatory run.
- **Design.** The eight models of at least 14B (deepseek-r1-distill-14b, mistral-small-24b, phi4-14b, qwen-coder-14b, qwen-coder-32b, qwen-general-14b, qwen-general-32b, qwen3-30b-a3b), the six earlier held-out tasks and their three seeds, arm C2 (route, then hint) with `check` feedback and with `cause` feedback: 36 episodes per model. Runner: `scripts/run_p_confirm.py --stage dev`. Reference point: P-fix v4 F4 on the same episodes with the earlier hint and library, 93 valid of 144.
- **Rule for the feedback, fixed now.** The confirmatory run uses `cause` if it is valid in at least three more of the 144 episodes than `check`; otherwise `check`, the simpler one.
- **No rule for the fixes.** The comparison with v4 F4 is across runs and is descriptive. If C2 with `check` is valid in fewer than 85 of 144, the fixes are re-examined before the freeze.

## 5. Confirmatory run (to be frozen)

Agreed with the user on 2026-11-19; details are filled in and frozen before any run.

- **Tasks.** 24 new position tasks: 16 whose rule `positions()` has, 8 whose rule it does not have (near misses included, such as an RSI with Wilder smoothing), so that the right answer to `route()` is NONE. New parameter values, new wording, both factor conventions. Each has an independent numpy reference and is qualified like the suite's tasks. Written without any model output on them.
- **Seeds.** Two per task: 48 episodes per model and arm.
- **Models.** The 20 of P-fix. Primary population: the eight of at least 14B, named above.
- **Arms** (`fin_skills/rsi/pconfirm.py`): C0 no library; C1 interface; C2 interface after `route()` with the hint; C3 the hint for the task's own rule when the catalogue has it (upper bound); C4 the hint for the top BM25 entry (text retrieval, cannot abstain). Same prompts, budget and six checks as P-fix; feedback as chosen in section 4.
- **Primary hypothesis.** On the 16 catalogued tasks, in the primary population, C2 is valid more often than C0.
- **Secondary.** C2 against C4 (the model's routing against retrieval) and C3 minus C2 (distance to the upper bound) on catalogued tasks; C2 against C1 on the 8 other tasks (does a routing step do harm when the library lacks the rule?); how often `route()` is right (the id, or NONE); all of these for the 12 smaller models and for all 20.
- **Analysis.** Every episode stays in the denominator. Differences in valid rate with 95% intervals from a bootstrap that resamples tasks and, within the population, models; a sign-flip test over tasks for the primary hypothesis. One run; no arm, task or model is added or dropped after results are seen.
- **Budget.** About 10k SU estimated from P-fix v4 (2.2 SU per episode; 4,800 episodes), of 16.6k left before the development run.
