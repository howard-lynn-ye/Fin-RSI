# Track E: reported endpoint moved from exact matching to BizBench's 1% (2026-11-12)

This records a deviation from the frozen protocol of the RSI study (plan
`fin-rsi-coordinator-cpu-20261109-v4`, frozen 2026-11-08). It changes which of two
pre-declared E grades is reported first. It does not change any run, prompt, seed, model, label
or grade, and the exact-match results stay in the release.

## What the protocol said

- Primary E endpoint: the program runs and returns the public label exactly (scoring policy
  `absolute_tolerance 0, relative_tolerance 0`; `external_study.py` refused any other policy).
- Pre-declared sensitivity analysis: the same final output within 1% of the label
  (`frozen_analysis.tolerance_sensitivity`), which is BizBench's own criterion
  (Koncel-Kedziorski et al. 2024, sec. 3.1: "Answers within 1% of the reference are considered
  to be correct").

## Why it moves

1. **Label defect.** 115 of the 200 CodeFinQA labels in the frozen enrollment are float32
   leftovers: `41.0999984741` for 41.1, `4.1599998474` for 4.16, `95983.5390625` for 95983.54.
   No correct float64 program returns these literals. In the finished runs (17 models, read-only
   Slurm job 3620857 on 2026-11-11) these questions were matched by 0 of 5,865 episodes in each
   library condition. About 29% of E's weight (115/200 x 1/2) could not be earned.
2. **Rounded labels.** Most other labels are rounded (158 integers, 37 with one decimal, 57 with
   two) while the questions rarely ask for rounding: 239.5 / 802.6 x 100 = 29.84 is graded
   wrong against the label `30`. An independent audit that answered 12 questions by hand judged
   10 correct; exact matching passed 3, 1% passed 10.
3. **Source benchmark.** BizBench scores within 1%. The frozen documents already said exact
   matching was this project's own, stricter choice (`docs/EXTERNAL_EVALUATION_20261108.md`).

## What was seen before the decision

This is not a prospective change: per-model E results under both grades were read on
2026-11-11 (job 3620741) before the defect was found the same day (jobs 3620783, 3620857 and
the independent audit in `.progress-audit/independent-audit-20261111/` on the user's PC). The
direction of the library effect differs between the two grades for three models
(qwen-coder-32b, qwen-coder-14b, deepseek-r1-distill-14b). Both grades are therefore reported;
the paper leads with 1% and states that the switch was made after the runs, and why.

## What does not change

- Track E still cannot show that the library helps: its interface text lists 48 trading,
  portfolio and pricing algorithms (25,658 characters, 17 missing a dependency), none of which
  computes these table questions, and no valid answer in any finished model called the library.
  The paper describes E as a measure of what that text costs.
- Tracks P and S grade against independent references with a fixed tolerance and are unaffected.

## For later runs

`external_study.run()` on branch `claude/frozen-study-fixes` now accepts exactly two policies,
exact `(0, 0)` or BizBench's `(0, 0.01)`, one per run, and records which in the manifest.
