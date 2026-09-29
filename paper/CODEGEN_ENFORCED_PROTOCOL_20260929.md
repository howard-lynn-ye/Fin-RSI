# Enforced library and guard study — protocol (frozen before inference, 2026-09-29)

Question: when a controller enforces the library call and the guards, how many financially
valid programs do open code models deliver, compared with coding from scratch under the same
repair budget? The E4 first-submission study (0/48 valid, 0/24 library uptake) could not
answer this because no model used the library.

Held fixed from E4: the four rules, data seeds 11/23/37, Qwen2.5-Coder 7B and 14B at the E4
revisions, the prompt packet (task, definition, raw split-affected prices, cumulative split
factors), 2,048 tokens per response, temperature 0.1, top-p 1, and the hidden E4 grader.

Three arms, 24 episodes each (72 total), up to three attempts per episode:

| Arm | Prompt | Controller accepts when | Feedback |
| --- | --- | --- | --- |
| `raw_exec` | from scratch, no fin_skills | the program runs | execution error and traceback tail |
| `raw_guards` | from scratch, no fin_skills | runs, lag check and split check pass | the same, plus check findings |
| `library_guards` | E4 library documentation | runs, a `fin_skills.algorithms.run` call is traced, both checks pass | the same, plus the library requirement |

Checks run in the Landlock/seccomp worker on the episode's data. The lag check applies the
library's `assert_causal` guard to the positions shifted up one row at k=40 and k=72, so a
position may not depend on prices at or after its own session. The split check requires
identical output when the same economic prices are given split-adjusted with unit factors.
Feedback never contains reference weights or grades. All arms use the first-complete-fence
parser, chosen after E4.

Primary endpoint: valid deliverable = accepted by the arm's controller and valid under the
unchanged E4 grader (execution; agreement within 1e-8; unchanged positions through session k
under price perturbations at k=32 and 64; split invariance). Secondary: grader validity of the
final and first attempts regardless of acceptance, acceptance, attempts, traced library calls,
tokens, generation and controller seconds, exception types.

Comparisons: `library_guards` vs `raw_exec` is the complete guarded-library workflow against
unassisted coding with the same repair budget. `raw_guards` vs `raw_exec` isolates enforced
checks; `library_guards` vs `raw_guards` adds the library algorithms given the checks.

Qualification before inference (seed 991, no model): sandbox network/private-file denial; a
library program is accepted and valid under the grader; a correct program without a library
call is rejected in `library_guards`; a same-session program fails the lag check; a program
that re-cumulates split factors fails the split check.

Limits: same tasks, seeds, models and grader as E4, and E4 failures informed the checks, so
this is not an independent holdout. Controller checks overlap grader properties; only the
grader checks numerical agreement, with different perturbations. Three seeds on four fixed
rules are not twelve independent problems; two sizes of one model family; synthetic data; no
profitability endpoint. Results are reported for the complete hash-verified batch only; an
interrupted run is recorded, never silently rerun or overwritten.
