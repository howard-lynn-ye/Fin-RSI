# Handoff: state of the trading study and the paper (2026-10-01, Claude)

Branch: `claude/trading-study-20260930` at `bf0eb87` (pushed). `master` is at `9e5e226`
(PR #11 merged). Uncommitted local changes exist only under `runs/` (gitignored).

## 1. Direction set by the user

The paper's main line is now the TRADING study: the same model with and without fin-skills,
compared on realized returns. The audit/guard material is a minor contribution. The old
code-generation studies (E4, E4b) stay as supporting evidence.

## 2. What is finished and verified

- Trading study v3, Qwen2.5-Coder 7B and 14B, 3 seeds x 44 decisions, window 2025-01-03 to
  2026-09-25, 12 US ETFs/stocks, 5 bps costs, submissions executed at next close and held
  10 sessions. Evidence with hash manifests under
  `benchmarks/agent_study/evidence/20260929-trading-study/results-v3/{7b,14b}/`.
- Results (cumulative return / Sharpe):
  - 7B  raw 48.2% / 1.60   library 31.1% / 1.51
  - 14B raw 46.5% / 1.50   library 32.4% / 1.27
  - Baselines: equal-weight rebalanced 42.2% / 1.52; 60/40 20.9% / 1.11; inverse-vol-252
    rebalanced 31.2% / 1.66 (my recomputation).
- 32B: resume job 1789830 RUNNING, 202/264 decisions at 18:45Z, ~7.5 h left. A scheduled
  check fires at 2026-10-02T03:30Z and will fetch/verify/score. Do NOT resubmit anything.

## 3. Why the library arm earned less (audited twice; no pricing bug)

Checked by me and re-checked by an independent reviewer from a frozen tarball
(sha256 a079b97a...): ledger re-priced to 1e-9, holdings to 1e-15, rebuilt prices to 6e-7,
run_algorithm weights recomputed to 5e-7. The gap is design, not analytics:

- 7B: 88/100 library submissions copy run_algorithm output, 62 of them inverse_volatility;
  time-weighted bond share 0.32-0.34 vs 0.03-0.16 raw; beta vs EW 0.67-0.79 vs 0.90-1.02.
  Pure inverse-vol earns 31.2%, i.e. the library arm == that strategy. Beta-adjusted
  difference -10 +/- 10 bps per period (n.s.); library Sharpe higher in 2 of 3 seeds.
- 14B: library documentation failure. 27% of library turns ended in fin_skills API errors
  (wrong data-mapping format, wrong guard names, tools called from inside Python, hrp
  missing linkage). 98/132 library decisions did not submit; 94 of those ran out of the
  8-turn budget. Submission rate 26% vs 53% raw.
- Prompt bias: the algorithm menu lists risk-based allocators first and the prompt says
  run_algorithm weights are a valid submit argument; small models take that path.
- Statistics: pooled_return_diff bootstraps n=3 (drop it); the macro generator pools
  holding periods across seeds sharing one market path (pseudo-replication). Only 1 of 6
  per-seed intervals excludes zero. Neither arm beats EW-rebalanced with confidence.

## 4. Real defects found (none affected v3 numbers; fix before any rerun)

1. Sandbox: `trading_worker.py` leaves `benchmarks/agent_study/` readable, which contains
   the full data download through 2026-09 (future data). Every traceback prints the
   absolute path. A search of all run_python turns found no out-of-workspace read, but the
   hole is real. Restrict reads; add a qualification test that opening repo data fails.
2. Weight-sum tolerance 1e-4 rejected rounding-only sums (7 library, 14 raw). Allow 1.001.
3. `hrp` has no default `linkage` (8 failed calls).
4. First trade's cost excluded from cumulative return (<= 5 bps).
5. `submit` is not callable inside run_python; models tried ~270 times (NameError).
6. A bare ```python fence should count as run_python.

## 5. Agreed v4 design (user approved direction; GPU scale pending approval)

- Both arms get identical interface documentation: data schema with example rows, one
  strategy-free worked tool-call example, submit callable in Python, one output cap,
  tolerance 1.001. The raw arm does NOT get a split/dividend recipe (that is the
  treatment).
- Library arm additionally: exact signatures with one example per method kind, one API for
  tool and Python paths, hrp default linkage, menu shuffled per decision (order recorded),
  and the "weights are a valid submission" nudge removed.
- Pilot gate outside the test window: pre-registered error/parse-rate ceilings per arm; no
  returns computed; freeze the protocol only after both arms pass.
- Primary endpoint: paired per-period net return difference hedged for market exposure
  (ex-ante beta from trailing 252d). Average over seeds first, then block bootstrap over
  dates, windows as clusters. Sharpe, return, beta, submission rate secondary.
- Scale: 10 seeds x 3 windows (2022 drawdown, 2023-24, 2025-26) x ~44 decisions; each
  decision starts from EW so decisions are independent; continuous path secondary.
- Newer models (Qwen3/GLM class) only with anonymised, date-shifted, price-rescaled data
  plus a canary probe (ask the model to identify the asset; above-chance => contaminated).
  Named-ticker runs stay with Qwen2.5 (pre-2025 cutoff).
- Add non-LLM menu baselines (each ready portfolio method run every decision); optional
  docs-only arm. No arm restricted to library outputs.

## 6. Open items

- 32B fetch/verify/score after job 1789830 completes (scheduled, 03:30Z Oct 2).
- Implement section 4 fixes + new prompts, with tests; then the pilot gate (cheap).
- Full v4 needs explicit user approval for GPU scale (~10x v3).
- Paper: `trading_study.tex` drafted but not yet \input into main.tex; rewrite around the
  trading result once v4 lands. Current honest summary: no clear effect on risk-adjusted
  performance; an exposure effect in a bull window (7B); an interface failure (14B).
- Overleaf sync blocked: browser pane not signed in.

## 7. Do not

- Resubmit or duplicate GPU jobs; never rerun completed decisions.
- Overwrite anything under `benchmarks/agent_study/evidence/` (append new dirs only).
- Type paper numbers by hand; regenerate via scripts/build_*_evidence.py (--check in CI).
- Auto-merge PRs.
