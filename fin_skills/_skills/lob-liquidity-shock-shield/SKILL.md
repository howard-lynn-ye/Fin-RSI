---
name: lob-liquidity-shock-shield
description: >-
  Calculate a daily cross-sectional liquidity instability proxy from observed volatility,
  returns, volume, Amihud illiquidity and margin data. TRIGGER -
  compute_lob_liquidity_shock_shield, audit_lob_liquidity_causality, check_lob_liquidity_gate,
  rolling liquidity dispersion, liquidity proxy warmup or temporal invariance.
  SKIP for passive order fill probabilities (limit-order-book-models), measured execution
  costs (execution-cost-analysis), or parent-order scheduling (execution-algorithms).
license: MIT
compatibility: Python 3.10+ with numpy and pandas; no market connection required.
metadata:
  verified_on: "2026-10-12"
allowed-tools: "Bash(python:*)"
---

# Daily liquidity instability proxy

Adapted from Shwai He's `v2` implementation. This computes a research feature from daily
observations. The historical function name contains LOB; the inputs do not reconstruct
an order book and the score is not a fill probability, execution cost or calibrated alpha.

✅ Code verification, 2026-10-12: run `python scripts/lob_liquidity_shock_shield.py` from
this skill directory. Numerical and temporal regression checks live in
`tests/test_microstructure_lob_liquidity_shock_shield.py` at the repository root.

## Input and time contract

Each row requires `date`, `symbol`, `garman_klass_volatility`, `ret_1d`, `volume`,
`margin_balance_z30` and `amihud_illiquidity`. Dates must parse as timestamps, with unique
symbol/timestamp pairs. Numeric observations must be finite. Volatility, volume and
illiquidity must be nonnegative. Missing features raise instead of receiving invented values.

The script accepts explicit volatility/return aliases listed in `_FEATURES`; these are
different proxies, not numerically equivalent measurements. `volume` must be observed
volume, not a turnover z-score. The margin input must already be standardized causally.

Rolling windows count observations per symbol. Warmup outputs remain NaN; score only rows
with sufficient history. Rankings use the eligible assets at the same timestamp. Constant
cross-sections receive zero. Output preserves the caller's row order and index labels.

🚨 All inputs at a timestamp must already be observable before using its score. A score
using the day's closing data cannot support a trade earlier that afternoon. The function
does not lag scores or verify when a vendor published the data.

## Calculation and audit

`compute_lob_liquidity_shock_shield` combines ranks of rolling return dispersion,
volatility dispersion and mean, log-volume dispersion, illiquidity dispersion, and the
margin input. Fixed research weights and clipping are visible in the script; they are
heuristic choices without a claimed out-of-sample benefit.

`audit_lob_liquidity_causality` perturbs all consumed features after one cutoff and checks
earlier outputs. A supplied `signal_col` must match the built-in formula, including warmup
NaNs. Correlation alone is insufficient. The audit also rejects insufficient history and
degenerate cross-sectional dispersion. Its passing result covers these numerical checks,
not arbitrary feature-producing code, publication timing, or profitability.

For a different signal formula, use `assert_causal` from
`../../../fin-core/skills/research-integrity-guards/SKILL.md` with the actual producer.
For order-book queue models, see `../limit-order-book-models/SKILL.md`.
