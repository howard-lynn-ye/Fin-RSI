# Trading return repair rerun, 2026-10-01

The user requested interface repair followed by a new trading experiment, reporting Return
Rate as a percentage. This rerun uses actual portfolio paths; the earlier numeric-task pilot
cannot supply a trading return and is not used as its result.

## Frozen scope

- Qwen2.5-Coder 7B and 14B, pinned revisions from the existing study.
- Seeds 11, 23, 37; raw/library arms; 44 decisions per path: 528 decisions total.
- Original 12 assets and window: execution 2025-01-03 through 2026-09-25.
- Trade at the next close, hold for ten sessions, long-only, no borrowing, zero cash yield,
  5 bps per side. A missing submission keeps existing holdings.
- Primary reported outcome: cumulative net Return Rate (%) from initial capital before
  the first trade. Report each seed and the three-seed mean, plus library minus raw in
  percentage points. Sharpe, drawdown, exposure and submission rates are secondary.
- Keep the original risk-adjusted trading objective in the agent prompt while repairing
  the interface. Reporting return percentages does not silently change the investment goal.
- Both arms get the repaired execution framework and common interface examples. Library
  access, its documentation and adjusted-data adapters remain the treatment.

The same market history has already been inspected, so this is a development rerun, not
an unseen-market confirmation. The three seeds share that history. No significance claim,
outcome-dependent strategy/prompt tuning, discarded losing seeds, or further model sweep.
The combined interface/permission/accounting changes cannot isolate the effect of each fix.

## Repairs exercised

JSON and Python calls have common signatures; `submit` works inside Python; empty arguments
are accepted on zero-argument tools. One complete Python fence can have surrounding prose;
multiple unlabelled code blocks are not silently concatenated. The common example only
prints CSV column names and does not prescribe a strategy. The library prompt explains the
data-mapping API and keyword-only guards. The HRP study adapter supplies explicit single
linkage; menu order is seeded and shared between paired decisions. Rounded totals up to
1.001 normalize to one. Future source data and the private ledger are unreadable in either
arm. All scoring includes the first trade's fee. Archived v3 decisions/scoring are unchanged.

## Execution and resource ceiling

`trading_study_v4.py` freezes one model/seed pair per directory. It hashes the runner, worker,
runtime, library Python code, library documentation and data. CPU confinement/interface
qualification must pass before GPU inference. Completed decisions are never overwritten;
resume checks their date and pre-decision holdings. Scoring requires the full paired receipt.
An aggregate is emitted only after all six model/seed pairs finish.

The 14B Slurm array (`0-2%1`) runs one seed at a time. The initial 7B array submission was
explicitly rejected by the account's submitted-job limit; its failed receipt is preserved.
The replacement 7B job executes the same three seeds serially, with a four-hour process
timeout per seed and a twelve-hour allocation ceiling. At most two GPUs run concurrently.
Each task uses one L40S: a four-hour hard limit for 7B and eight hours for 14B.
The maximum allocation is 36 GPU-hours across all six tasks; actual use may be lower.
No automatic resubmission or time-limit extension. A timeout is an incomplete experiment,
not a scored failure or permission to omit a seed. The existing 32B study is left running.

Commands are Python module invocations under `benchmarks.agent_study`:

```text
trading_study_v4 freeze PAIR_DIR --family 14b --seed 11
trading_study_v4 qualify PAIR_DIR
trading_study_v4 run PAIR_DIR
trading_study_v4 score PAIR_DIR
trading_study_v4 aggregate BATCH_DIR
```

`beacon_trading_repair_v4.sh` runs inference, then scoring, then attempts whole-batch
aggregation. Each pair writes its completion receipt only after its scores are saved.
Read-only monitoring does not change prompts, infer missing decisions, or report pending
pairs as completed returns.

Submission receipts: 14B array `1812017`, 7B serial job `1812031`. The frozen code commit is
`9a815c2461af630bc8b54afc2851eb500818bc87`; receipt hashes identify the exact staged bytes.
Submission is not proof of completed inference or a new Return Rate result.
