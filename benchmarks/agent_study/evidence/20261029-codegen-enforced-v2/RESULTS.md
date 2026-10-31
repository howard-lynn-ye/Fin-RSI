# codegen-enforced-v2 results (job 2906547, fetched and hash-verified 2026-10-31)

Primary endpoint: valid deliverable, meaning accepted by the arm's controller and valid under
the E4 grading equations. Results are reported as v1 -> v2, with 12 episodes per model and arm.

| Group | Valid deliverable | Accepted | Final program runs | Traced library use | Timing failures (grader) |
| --- | --- | --- | --- | --- | --- |
| 7B raw_exec | 1 -> 1 | 11 -> 10 | 11 -> 10 | 0 -> 0 | 3 -> 2 |
| 7B raw_guards | 1 -> 2 | 3 -> 4 | 8 -> 8 | 0 -> 0 | 3 -> 2 |
| 7B library_guards | 0 -> 0 | 0 -> 0 | 1 -> 1 | 1 -> 1 | 1 -> 1 |
| 14B raw_exec | 0 -> 0 | 12 -> 12 | 12 -> 12 | 0 -> 0 | 6 -> 6 |
| 14B raw_guards | 0 -> 0 | 0 -> 0 | 11 -> 11 | 0 -> 0 | 3 -> 3 |
| 14B library_guards | 0 -> 0 | 0 -> 0 | 4 -> 10 | 3 -> 8 | 1 -> 1 |

Strict lag audit of final programs (secondary, post hoc; execution_lag guard with per-asset
factors): 11 / 6 / 2 failures in raw_exec / raw_guards / library_guards, against 8 / 5 / 2
for the grader's timing probes.

## What the library arm did (exploratory diagnosis, decided after seeing the results)

- **14B:** 8 of 12 final programs used the documented call, but applied the split factors
  twice. They called `adjusted = adjust_prices(prices, split_factors)` and then
  `positions(algorithm, adjusted, split_factors, ...)`. Each failed the split check on
  attempts 2 and 3. The feedback named the cause and added the split_once card, whose last
  sentence points to `adjust_prices`, and no program corrected it.
  - Counterfactual, post hoc: passing the raw prices instead of `adjusted` changes nothing
    else. With that change, all 8 programs are accepted by the controller and graded valid.
- **7B:** 22 of 36 attempts called `fin_skills.algorithms...` without importing fin_skills
  (NameError), and the error repeated across attempts. Six final programs reference fin_skills
  without importing it; adding the import makes none of them valid. The rest of each program was hand-written and
  wrong.
- **Other failures:** warm-up NaNs (8 attempts), passing `prices` to a portfolio algorithm,
  and parameters given to the wrong algorithm.

Limits: the counterfactual edits were chosen after seeing the failures and are not an
endpoint. They show where the remaining barrier is, not what a rerun would score. The two
library defects they point to are adjust_prices + positions composing into a double
adjustment, and a rule-card hint that steers toward that composition. Both are library-surface
problems that a repair would have to fix and test before any further model run.
