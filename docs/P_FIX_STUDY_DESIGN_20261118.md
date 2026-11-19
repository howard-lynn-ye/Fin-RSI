# P-fix: the frozen P track on the fixed library interface (exploratory, 2026-11-18)

Written before any P-fix run. Exploratory: it was designed after the frozen study's results were
seen, so it describes what happens on the changed interface; it does not confirm a hypothesis
the frozen study set out to test. Numbers are compared within P-fix only, never pooled with the
frozen study or P-adopt (hardware, software versions and confinement differ; see below).

## Why

In the frozen P study's library arms, 519 programs called the library and 480 of those calls
raised. 167 passed the model's own function as the algorithm, 143 named an id not in the
catalogue, and 126 passed wrong arguments or inputs. Calls that ran often divided an
already-scaled result by the number of assets. For a library built for agents, those are
defects of the interface. PR #77 changed it:
- keyword parameters;
- `run()` on a single input;
- errors that say what to call instead, with did-you-mean ids;
- `signal_positions()` for a model's own rule;
- docstrings stating that the result is final.

P-fix measures whether models now use the library and get the held-out tasks right with it.

## Arms (`fin_skills/rsi/pfix.py`)

| Arm | System prompt | Task prompt |
|---|---|---|
| F0 | frozen system prompt, no library (identical to P-adopt L0) | frozen |
| F1 | frozen system prompt + the current position-task interface text (`library_text(rolling=False, frozen=False)`) | frozen |
| F2 | as F1 | frozen + P-adopt's pointer `The fin_skills library implements this rule as positions("<id>", ...).` |

All arms:
- no rule cards;
- Check feedback: the names of failed checks, and the full error when a program fails to run (as frozen);
- at most 3 attempts, 6 calls and 12,288 generated tokens;
- 4,096-token output cap per reply, temperature 0.1, top-p 1, BF16;
- the 20 protocol models at their frozen revisions;
- 6 held-out tasks x seeds 101, 211, 307: 18 episodes per arm, 54 per model.

## Measures and analysis (declared now)

- **Primary.** Valid episodes per arm and model.
- **Contrasts.** F1 - F0, F2 - F0 and F2 - F1, per model and pooled over models. They are reported by algorithm family and by variant (a: library defaults; b: non-default parameters and ratio factors).
- **Funnel.** Last program calls the library, then it runs, then it passes the visible checks, then it is valid. Library calls that raise are grouped by error, as in jobs 3655794 and 3655802.
- **Statistics.** Three rule families cannot support significance claims, so everything is descriptive.
- **Denominator.** Every refusal, context stop, truncated reply and failed program stays in it.

## Execution and deviations

- **Clusters.** Zaratan (UMD) H100 GPUs, because Beacon could not start new GPU jobs before 2026-11-22. The same jobs are also queued on Beacon. For each model, only the first complete run is kept; the other cluster's job is cancelled before it starts, or its output is not read.
- **Zaratan software.** Python 3.12, torch 2.7.1 + CUDA 12.6 (Zaratan's driver does not run the frozen CUDA 13 build), transformers 4.57.6, accelerate 1.12.0, numpy 1.26.4, pandas 2.2.3, scipy 1.15.2.
- **Runtime files.** Zaratan scratch twice returned "no such file" for installed package files while 13 jobs started (2026-11-18, both jobs failed before loading a model). Jobs submitted after that copy the runtime and the source to node-local disk and check them against every package RECORD hash and the source manifest; the 12 jobs already running read them in place. The analysis lists any program error that names a runtime file under scratch.
- **Confinement.** Zaratan's kernel (4.18) has no Landlock. Model programs keep Beacon's system-call denials (no sockets, exec, fork, signals, namespaces or mounts) and resource limits, but not its filesystem rules (`confine_without_landlock`, accepted only with `FIN_TASKS_DECLARED_DEVIATION=no-landlock` and recorded in each run's protocol). Afterwards, every executed program is scanned for file access outside its inputs. Any such episode is reported.
- **Model weights.** The group's scratch quota (100 GB) cannot hold the 20 models, and Zaratan compute nodes cannot reach the Hugging Face file CDN. Each job therefore receives its model's frozen-revision files from the login node into node-local disk, streamed through its own Slurm allocation. Every file's SHA-256 is compared on both ends, and weight files are also compared with the Hub's recorded SHA-256, before the model loads.
- **Validation first.** One `--dev` job on a small model must finish both units, and every declared first prompt must reach the model. It reads no grade.

## Amendment 1 (2026-11-18, before any v2 run)

The first run (v1, library at commit 01b2ea44) showed two library behaviours that let a wrong call return a plausible result. `positions()` ignored `window=` and `rebalance_every=` for signal algorithms, so a rule kept its default lookback (19 of 98 signal `positions()` calls in the first 12 models passed `window=`). `adjust_prices()` read event ratios as cumulative factors when `factors=` was left at its default (154 variant-b programs that let the library adjust event ratios ran and then failed the split check). The library now refuses both and names the right call, and it points states passed to `positions()` to `signal_positions()` (commits 6eeb9ddc, 321168d4, 001ea472).

- The interface text, prompts, seeds, budgets, arms and contrasts are unchanged: `interface_text()` is the same 5,312 characters, so F1 and F2 send byte-identical messages.
- v2 reruns the `check_fix` unit (F1 and F2) for all 20 models with the library at 001ea472, into a separate output root. F0 never imports the library and is read from v1.
- The analysis reads F0 from v1 and F1/F2 from v2 (`analyze_p_fix.py --fix-root`). v1's F1/F2 records are kept and released, not analysed. v1 jobs still running are stopped once their F0 unit is complete.
- Decided with the user on 2026-11-18, after the v1 results above and before any v2 job ran.


## Amendment 2 (2026-11-19, before any F3 run)

The v2 results were seen before this amendment was written: pooled valid F0 9.4%, F1 16.1%, F2 28.6%. In F1, models mostly wrote the rule themselves and passed the states to `signal_positions()` (547 programs, against 104 calling `positions()`). F2 adds a task-specific pointer naming the algorithm, which the interface alone does not provide. Amendment 2 adds one arm to separate those two things. It does not change F0, F1 or F2, their records, or the contrasts already declared.

- **F3, documented interface.** F1's system prompt, except that each algorithm in the `positions()` listing is followed by one line stating the rule it computes (`fin_skills.tasks.prompts.RULES`), and the listing's header says that if a task's rule is one of these, one `positions()` call with the task's parameters and `factors=` convention computes it. The task prompt is F0's: nothing names the task's algorithm, parameters or variant. Text: `library_text(rolling=False, frozen=False, documented=True)`, 7,281 characters, SHA-256 prefix 9871fe83a0de; F1's text is unchanged (5,312 characters, 120c321f67cd). Every other setting is as for F1.
- **Question.** Does a library that documents what each algorithm computes get used, and get held-out tasks right, without a per-task pointer?
- **Contrasts.** F3 - F1 (rule documentation alone), F3 - F2 (documentation vs a per-task pointer) and F3 - F0, per model and pooled, with the same funnel and error groups. Everything stays descriptive. F3 was designed after seeing v2, so it is labelled exploratory wherever it is reported.
- **Execution.** Library at the v2 commit (001ea472) plus this arm; the library code that programs import is unchanged. Unit `check_doc` (F3 only), all 20 models, on Zaratan, under the same deviations, into a third output root (v3). The analysis reads F0 from v1, F1 and F2 from v2, and F3 from v3. One `--dev` job on a small model runs first.
- **Size split (post hoc).** All arms and contrasts are also reported for models with at least 14B total parameters (8 models; qwen3-30b-a3b counts its 30B total) and for the smaller 12. The 14B cut was chosen after seeing v2, so the split is descriptive only. `analyze_p_fix.py` computes it (`by_size`).
- Decided with the user on 2026-11-19, after the v2 results and before any F3 job ran.
- **Executed 2026-11-19.** Source 536dfa01 on Zaratan; dev job 24141917 (qwen3-4b, one development task, every first prompt reached the model), then jobs 24141918-24141937, one per model, each held until the dev job exited 0. All 21 completed; no program was flagged by the scan. Analysis: `analyze_p_fix.py runs/main out.json --fix-root runs-v2/main --doc-root runs-v3/main`, SHA-256 1b5ac441666a.

## Amendment 3 (2026-11-19, before any v4 run)

Written after all results above (F0 9.4%, F1 16.1%, F2 28.6%, F3 15.3%) and after reading where the recorded programs failed. Three defects of the library were found in those records. They are fixed, and every arm is rerun together (v4). The earlier roots are kept and released; the paper analyses v4 only.

- **Checks accepted wrong programs.** The visible checks were finite output, split representation and execution lag. Job 24142798 (Zaratan standard partition, source 536dfa01, the study's confinement) re-executed the last program of every episode that ran but was not valid. It reproduced the recorded acceptance for all 455. Of the 329 that had passed the three checks, 105 returned zeros on every row, 55 had weights of 1/N^2 (scaled twice) and 230 fail at least one of the checks below. The other 99 compute different states on the right grid (93) or are one row late (6); no property check sees those.
  - New checks (`fin_skills/tasks/properties.py`, on by `Judge(properties=True)`): `takes_positions` (the output is not zero everywhere), `position_scale` (a long/short/flat rule's weights are -1/N, 0 or +1/N; sized rules stay within their cap; portfolio weights are non-negative and sum to at most one) and `warmup_rows_zero` (no position before the task's lookback plus the lag).
  - Each tests a sentence of the task text on the program's own output; none reads the reference. Every reference passes all of them on all 22 position tasks (7 seeds each, `tests/test_task_properties.py`), so a program equal to the reference is never rejected. Valid still means: ran, passed every visible check, equals the reference.
  - Every v4 arm, F0 included, is judged with the six checks. The frozen study and v1-v3 used the first three.
- **`adjust_prices()` refused data it had recognised.** After amendment 1 it raised when the factors looked like the other convention; 105, 109 and 100 programs in F1, F2 and F3 stopped on that error. `factors` now defaults to `"auto"`: the convention is read from the data (event ratios return to 1.0 on the next row, cumulative factors stay changed), recorded in `attrs["factors"]`, and it raises only when a stated convention contradicts the data or the data cannot show it. On every suite task (34 tasks, 15 seeds) it reads the task's convention. The interface text changes with the signature: 5,657 characters (04fbb48cc1a5); documented 7,626 (0a9922028edb).
- **The library could not say that a described rule is one of its own.** The task text describes the rule without naming it. `search_methods()` returns no hit for any of the six held-out rule texts, and a lexical matcher over the rule lines named the wrong algorithm for 5 of the 12 development tasks, so no text matcher is used. `fin_skills.algorithms.routing.route(description, generate)` asks the connected model one question (which catalogue rule, if any, computes this rule?) and returns the id it names, or None.
  - **F4.** One `route()` call on the task prompt (reply capped at 1,024 generated tokens), charged to the episode's budget like any call. If the model names a catalogue id, F2's sentence is added with that id; otherwise the messages are F1's. Nothing in the prompt comes from the task's label. A right route gives exactly F2's prompt. The row records the id named.
  - **Contrasts (exploratory).** F4 - F1, F4 - F2 and F4 - F0, with how often the route was right, wrong or absent, and validity in each case. F1 - F0, F2 - F0 and F2 - F1 stay the declared contrasts; F2 (the id supplied by the harness) is the upper bound for F4.
- **Execution.** Units `check_F0`, `check_fix`, `check_route`, `check_doc` (part `all`), 20 models, Zaratan, same deviations, new root (v4). One `--dev` job first.
- **What this does to the evidence.** This is the third revision of the library made after seeing results on the same six held-out tasks, so they no longer test a library that was fixed in advance; the paper says so. The frozen study's no-library arms (Repeat, Check, Explain and the methods) were judged with the three original checks.
- Decided with the user on 2026-11-19, after the results above and before any v4 job ran.
- **Started 2026-11-19.** Source b8417dcc on Zaratan. Dev job 24142970 (qwen3-4b, one development task, `macd_a`) completed all four units with every declared first prompt forwarded; jobs 24142971-24142990 (one per model, part `all`) were queued behind it and started when it exited 0. Unlike the earlier dev jobs, its five rows were read with their grade (F0, F1 and F3 raised three times; F2 and F4 valid with one `positions()` call, the router having answered `macd`). That is one development task on one model and no held-out result.
- **Completed 2026-11-19.** All 21 jobs completed (longest 2 h 53 min); every model has all four units. Analysis: `analyze_p_fix.py runs-v4/main out.json --doc-root runs-v4/main --route-root runs-v4/main` at 6a782a8d, SHA-256 e2eb603d78cb. The program scan flags nothing. One reply (deepseek-r1-distill-14b, F3) held only the model's reasoning and matched a scan word; text that does not parse cannot run and is now counted (138 of the recorded program texts), not scanned.
- **Result (360 episodes per arm).** Valid: F0 35 (9.7%), F1 58 (16.1%), F2 107 (29.7%), F3 58 (16.1%), F4 105 (29.2%). F4 - F0 +19.4 points (better in 11 models, worse in 0); F4 - F2 -0.6; F4 - F1 +13.1. Models of at least 14B: F0 23.6%, F1 35.4%, F2 63.9%, F3 40.3%, F4 64.6%. The router named the right rule in 274 episodes, none in 61 and a wrong one in 25; the eight models of at least 14B named the right rule in all 144. Final programs that passed the visible checks and were wrong fell from 81 / 89 / 74 / 85 (F0-F3, three checks) to 13 / 33 / 29 / 42 (six checks; F4 28); validity without routing did not rise with them.
