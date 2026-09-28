---
name: macro-fx-industry-beta-shield
description: >-
  [fin-macro] Convert a 1-D macro liquidity or USD/CNH FX shock into a causal cross-sectional equity shield by multiplying lagged stock-and-industry sensitivity beta by the regime impulse. TRIGGER - macro_fx_industry_beta_shield, compute_macro_fx_beta_shield, estimate_causal_macro_fx_beta, audit_macro_fx_beta_causality, industry beta gated macro FX transmission shield, USD/CNH shock cross-sectional stock sensitivity beta, uniform macro broadcast zero cross-sectional dispersion, full-sample macro beta look-ahead trap, 行业敏感度Beta加权的宏观汇率门控. SKIP for GDP nowcasting (gdp-nowcasting-dynamic-factor), recession indicators (macro-regime-and-recession-indicators), release calendars (macro-release-calendar-and-embargo), macro vintage tables (real-time-macro-backtesting), and FX pip conventions (fx-and-carry-conventions).
license: MIT
compatibility: Python 3.10+, numpy, pandas
metadata:
  verified_on: "2026-09-28"
---

# Industry-Beta-Gated Macro & FX Transmission Shield (`macro-fx-industry-beta-shield`)

## Why Raw 1-D Macro Broadcasts Fail in Cross-Sectional Stock Prediction

In multi-modal equity ranking, a macro liquidity impulse or USD/CNH exchange-rate shock $M_t$ is a **1-D time-series regime variable** (constant across all stocks $i \in \{1, \dots, N\}$ on date $t$). Two failure modes corrupt naive macro features:

1. **The Uniform Broadcast Trap (Zero Cross-Sectional Dispersion)**:
   - Appending $M_t$ directly or adding $+w \cdot M_t$ uniformly to every stock on day $t$ shifts all scores by the same constant, yielding zero cross-sectional Rank IC ($\text{RankIC}_t \equiv 0$) or inverting export-oriented tech/manufacturing ($\beta_{i,\text{FX}} > 0$) against domestic financials ($\beta_{i,\text{FX}} < 0$).
2. **The Full-Sample Beta Look-Ahead Trap**:
   - Estimating stock $i$'s macro sensitivity $\hat{\beta}_i$ via a single regression over the entire backtest $[1, T]$ leaks future regime breaks and post-tariff export re-ratings into historical predictions.

## Strictly Causal (`<= t-1`) Shrinkage Beta $\times$ Regime Impulse

`scripts/macro_fx_industry_beta_shield.py` (`fin_skills.macro.macro_fx_industry_beta_shield`) computes a strictly causal, peer-shrunk sensitivity beta $\hat{\beta}_{i,t-1}$ using only lagged returns $r_{i,\tau}$ and shocks $M_\tau$ ($\tau \le t-1$):

$$\hat{\beta}_{i,t-1}^{\text{raw}} = \frac{\widehat{\text{Cov}}_{W}(r_{i, t-W:t-1},\, M_{t-W:t-1})}{\widehat{\text{Var}}_{W}(M_{t-W:t-1}) + \epsilon}$$

$$\hat{\beta}_{i,t-1}^{\text{shrunk}} = (1 - \lambda)\,\hat{\beta}_{i,t-1}^{\text{raw}} + \lambda\,\frac{1}{|G(i)| - 1}\sum_{j \in G(i),\, j \ne i} \hat{\beta}_{j,t-1}^{\text{raw}}$$

$$S_{i,t}^{\text{MacroFX}} = \text{CS-ZScore}_t\!\left(\hat{\beta}_{i,t-1}^{\text{shrunk}} \cdot \left(M_t + \gamma P_t\right)\right)$$

## Executable API & Guard Verification

```bash
python3 -c "from fin_skills.macro.macro_fx_industry_beta_shield import _demo; raise SystemExit(_demo())"
```

## Cross-References

- `macro-regime-and-recession-indicators` — Macro regime state estimation and look-ahead gates.
- `real-time-macro-backtesting` — Point-in-time macro vintage alignment (`ALFRED` / release dates).
- `cross-board-supply-chain-rsi` — Leave-one-out (`j != i`) A-share peer spillover and board-limit routing.
