---
name: signal-reconciler
description: >-
  [fin-china] Inspect a heuristic conflict resolver for supplied policy, flow, valuation, news, and sentiment scores. Check market scope, evidence dates, and rule assumptions before using its proposed portfolio tilt.
license: MIT
metadata:
  version: "0.1.1"
  verified_on: "2026-10-05"
---

# Signal Reconciler

This module applies fixed rules to scores supplied by the caller. Its channel names and QDII
examples come from China equity and cross-border ETF workflows. It does not collect news,
measure sentiment, verify institutional intent, predict a return, or place an order.

The rules encode an institutional-flow and valuation preference. That is a design assumption,
not evidence that institutions always outperform retail investors. A rule label such as
`CONFLICT_DISTRIBUTION_TRAP` describes the input pattern; it does not establish that a market
trap, bottom, or profitable trade exists. The demonstrations use invented signals and contain
no observed investment returns.

## Establish what each input represents

Before assigning a score, record the source, market, instrument, unit, observation period,
publication time, availability assumption, and any missing information. Use only information
available at the decision time. Do not translate unlike observations into equivalent scores
without a separately validated mapping.

- CFTC futures positions can hedge other exposures. They are not ETF cash flows or a direct
  measure of institutional conviction; do not map them automatically to `SMART_MONEY_FLOW`.
- Survey opinions are not trades. Compare the same sampled population and fieldwork period;
  a retail-client survey and an active-trader survey need not form one time series.
- A filing date is not a transaction date. Annual holdings, household ownership, option
  exercises and new open-market purchases are different facts.
- A policy announcement is not automatically a trading prohibition. Establish an applicable
  constraint before setting a veto flag. The numeric thresholds below are code settings,
  not exchange or regulator rules.
- A QDII premium rule concerns the traded fund relative to its underlying value. Do not apply
  it to an ordinary US stock or ETF merely because an overseas sentiment score is available.

If these mappings cannot be justified, retain the observations as context and decide outside
this heuristic. Missing observations are not evidence of a neutral market view.

## What the implementation does

The current script evaluates the following branches in order. Values in this table are
implementation constants, not calibrated predictive probabilities or recommended limits.
`max_tilt` defaults to 0.015; it expresses a portfolio-weight change, not a target portfolio.

| Branch | Trigger | Output |
|---|---|---|
| Veto | Policy veto flag, policy score at most -0.8, or QDII premium at least 2.5% | Score -1; tilt `-max_tilt` |
| Premium conflict | QDII premium at least 1.5% and social score above 0.3 | Score -0.6; tilt `-0.8 * max_tilt` |
| Flow/social conflict | Flow score at most -0.35 and social score at least 0.45 | Score `max(-1, flow - 0.35 * social)`; tilt `-max_tilt` |
| Anchor/social conflict | `0.55 * flow + 0.45 * valuation` at least 0.25 and social score at most -0.45 | Score `min(1, anchor - 0.25 * social)`; tilt `max_tilt` |
| Otherwise | No earlier branch | Combine scores and damp by dispersion; scores with absolute value below 0.15 produce no tilt |

The final branch combines `0.45 * flow + 0.35 * valuation + 0.10 * news` with a social
contribution: `-0.15 * social` when its absolute value exceeds 0.6, otherwise `0.05 * social`.
The damping multiplier is `clip(exp(-dispersion_damping * disagreement**2), 0.25, 1)`;
`dispersion_damping` defaults to 1.2. These are heuristic settings requiring evaluation on the
intended market and horizon. Do not treat the branch-specific `confidence_multiplier` as an
estimated probability of success.

`disagreement_index` uses the population standard deviation of scores whose absolute value
exceeds 0.05, capped at 1. With fewer than two active scores it returns zero; zero can
therefore indicate insufficient evidence, not consensus. The separate `compute_belief_entropy`
helper normalizes positive input masses and measures concentration. It does not see bullish
versus bearish signs, and it is not used by `reconcile_asset_signals` to select a branch.
Equal masses can have the same entropy whether their underlying opinions agree or disagree.

The current implementation applies `ChannelSignal.confidence` to social scores only;
confidence is not a general credibility weighting of every channel. Inputs must use unique,
recognized channels with finite scores in [-1, 1] and confidence in [0, 1]; invalid values are
rejected. Only `REGULATORY_POLICY` accepts a veto flag. Valid numeric inputs still do not
establish source quality or the meaning of the `evidence` string.

`CHANNEL_BASE_WEIGHTS` is a legacy registry whose values are not used by the branches above.
The `dominant_channel` field is a branch label: both premium branches can name
`REGULATORY_POLICY` even without a policy observation. In the final branch, the nonzero tilt
is `clip(final_score * 1.5 * max_tilt, -max_tilt, max_tilt)`. Scores and calculated tilts are
rounded to four decimal places, except the direct `max_tilt` assignments.

## Executable examples

All observations below are synthetic fixtures. The assertions check branch behavior, not
profitability. Run the script in `scripts/signal_reconciler.py` for its printed demonstrations.

```python
from fin_skills.china.signal_reconciler import ChannelSignal, reconcile_views

signals = [
    ChannelSignal("SMART_MONEY_FLOW", score=-0.75, evidence="synthetic flow score"),
    ChannelSignal("FUNDAMENTAL_VALUATION", score=-0.20, evidence="synthetic valuation score"),
    ChannelSignal("RETAIL_SOCIAL_CN", score=0.85, evidence="synthetic social score"),
]
result = reconcile_views("510300", signals)
assert result.conflict_type == "CONFLICT_DISTRIBUTION_TRAP"
assert result.recommended_tilt == -0.015

signals = [
    ChannelSignal("SMART_MONEY_FLOW", score=0.60, evidence="synthetic flow score"),
    ChannelSignal("FUNDAMENTAL_VALUATION", score=0.85, evidence="synthetic valuation score"),
    ChannelSignal("RETAIL_SOCIAL_CN", score=-0.80, evidence="synthetic social score"),
]
result = reconcile_views("510880", signals)
assert result.conflict_type == "CONFLICT_CONTRARIAN_BOTTOM"
assert result.recommended_tilt == 0.015

result = reconcile_views("513100", signals, qdii_premium_pct=2.85)
assert result.conflict_type == "CONFLICT_VETO_OVERRIDE"
assert result.final_score == -1.0
```

## Before applying a tilt

The caller decides whether to accept, modify or reject the proposal. A positive tilt is not a
complete weight vector, and a negative tilt is not permission to short. Check existing holdings,
cash, concentration, turnover and transaction costs. Validate predictive value separately with
matched information, realistic execution, out-of-sample dates and a no-trade comparison.

This script does not enforce lot settlement, exchange price limits, instrument eligibility,
market hours, data availability, or order execution. Verify current instrument-specific rules
and run the relevant execution and research guards separately. A QDII setting or a branch named
"veto" does not substitute for those checks.
