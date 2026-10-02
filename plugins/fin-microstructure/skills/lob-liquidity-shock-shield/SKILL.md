---
name: lob-liquidity-shock-shield
description: >-
  [fin-microstructure] Compute a 20-day order-book volatility-of-volatility liquidity absorption shield and estimate L2 queue-position slippage, limit-lockout penalty, and A-share short-borrow fee schedules. TRIGGER - lob_liquidity_shock_shield, compute_lob_liquidity_shock_shield, estimate_intraday_lob_execution_cost, audit_lob_liquidity_causality, check_lob_liquidity_gate, volatility of volatility liquidity absorption shield, 20-day Garman-Klass range vol-of-vol stability, L2 queue position participation slippage and short borrow fee schedule, 限价订单簿波动率之波动率流动性吸收护盾与L2队列冲击融券费率建模. SKIP for Cont-Stoikov-Talreja birth-death queue arrival rates (limit-order-book-models), for Lee-Ready tick classification and Kyle lambda estimation (intraday-microstructure), for Almgren-Chriss parent order scheduling (execution-algorithms), and for A-share board sign bifurcation (cross-board-supply-chain-rsi).
license: MIT
compatibility: Python 3.10+, numpy, pandas
metadata:
  verified_on: "2026-10-02"
---

# 20-Day LOB Volatility-of-Volatility Liquidity Shield & L2 Execution Cost Engine (`lob-liquidity-shock-shield`)

## Why 1-Day Liquidity Snapshots Miss Order-Book Replenishment Fragility

Single-day range volatility ($\text{GK}_{i,t}$) or single-day Amihud illiquidity ($\text{Amihud}_{i,t}$) conflates transient news volume with structural limit-order-book (`LOB`) depletion. Following Chordia, Subrahmanyam & Anshuman (2001) and Lou & Shu (2017), **second-moment liquidity instability (volatility-of-volatility and illiquidity dispersion over a 20-day rolling window)** captures whether passive market makers replenish depth consistently or withdraw during stress:

$$S_{i,t}^{\text{LOB}} = \text{CS-ZScore}_t\!\Bigl(-0.32\,R_t(\sigma_{20\text{d}}(r_i)) - 0.24\,R_t(\sigma_{20\text{d}}(\text{GK}_i)) - 0.20\,R_t(\mu_{20\text{d}}(\text{GK}_i)) - 0.10\,R_t(\sigma_{20\text{d}}(\log V_i)) - 0.06\,R_t(\sigma_{20\text{d}}(\text{Amihud}_i)) - 0.08\,R_t(Z_{30\text{d}}(\text{Margin}_i))\Bigr)$$

## L2 Queue-Position Slippage, Limit-Lockout Penalty & Short-Borrow Schedule

`scripts/lob_liquidity_shock_shield.py` (`fin_skills.microstructure.lob_liquidity_shock_shield`) also provides `estimate_intraday_lob_execution_cost(...)` for live/paper trading (`14:30 CST` execution window):

1. **Passive Queue Fill Probability & Adverse Selection**:
   $$P_{\text{fill}}(q, \pi) = \exp\!\bigl(-1.25\,q\,(1 + 4\pi) - 0.85\,\max(0, \rho_{\text{lim}} - 0.65)\bigr)$$
2. **Square-Root Participation Impact + Price-Limit Lockout Penalty**:
   $$\text{Slippage}_{\text{1-way, bps}} = \text{QueueImpact}_{\text{bps}} + 10^4\,\eta_{\text{board}}\,\sigma_{\text{GK}}\sqrt{\frac{Q}{\text{ADV}}} + 3200\,\max(0, \rho_{\text{lim}} - 0.72)^2$$
3. **A-Share Tiered Short-Borrow (`融券`) Fee Schedule**:
   - `2.50%` p.a. (`Broad_Index_ETF`), `4.50%` p.a. (`10%` Main Board), `6.50%` p.a. (`20%` STAR/ChiNext), plus `+2.00%` p.a. crowding surcharge when $Z_{\text{margin}} > 1.0$.

## Executable API & Guard Verification

```bash
python3 -c "from fin_skills.microstructure.lob_liquidity_shock_shield import _demo; raise SystemExit(_demo())"
```

## Cross-References

- `limit-order-book-models` — Cont-Stoikov-Talreja birth-death queue arrival rates and passive fill probabilities.
- `intraday-microstructure` — Tick-level Lee-Ready classification, Kyle's lambda, and VPIN.
- `cross-board-supply-chain-rsi` — Leave-one-out (`j != i`) A-share peer spillover and `10%` vs `20%` board sign bifurcation.
