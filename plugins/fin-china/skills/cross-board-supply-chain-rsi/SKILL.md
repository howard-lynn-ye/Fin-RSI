---
name: cross-board-supply-chain-rsi
description: >-
  [fin-china] Route cross-board price-limit microstructure sign bifurcations (10% Main Board reversal vs 20% STAR 688 and ChiNext 300 high-threshold continuation) and synthesize leave-one-out supply-chain peer spillover features. SKIP for A-share data fetching (china-ashare-data), backtest order matching (china-trading-stack), KOL Brier weighting (kol-credibility-registry), or entropy conflict resolution (signal-reconciler).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-28"
---

# Cross-Board Microstructure Bifurcation & Supply-Chain Spillover RSI

Across China A-Share (`SH600/601/603`, `SZ000/002`, `SH688 STAR`, `SZ300 ChiNext`) and cross-border (`HK / US`) equity panels, two structural failures degrade single-stock predictive models:

1. **Cross-Board Microstructure Sign Bifurcation (Law Stock-RSI-5)**:
   - On **`±10%` Main Boards** (`SH600/601/603`, `SZ000/002`) with zero account-balance threshold, retail margin-buying surges (`margin_buy_ratio`) and overnight gaps (`overnight_gap_ratio`) represent retail overcrowding and mean-revert over the next 5 days (`+o_1d` Rank IC $= +0.0200$).
   - On **`±20%` Registration Growth Boards** (`SH688 STAR` with a `500,000 RMB` investor suitability threshold and `SZ300 ChiNext` with a `100,000 RMB` threshold), the exact same 1-day margin and overnight-gap signals reflect institutional/high-net-worth information diffusion and **continue** (`+o_1d` Rank IC $= -0.0340$, whereas `-o_1d` continuation achieves Rank IC $= +0.0391$).
   - Applying a single global sign across both boards cancels out alpha on `STAR_ChiNext_20pct`.

2. **Single-Stock Social Sparsity & Retail Noise ($n \in \{1, 2\}$ & Tier-C Tickers)**:
   - Low-coverage tickers ($n \in \{1, 2\}$ social posts) and retail-heavy New Energy / Tech stocks lack reliable single-ticker text signals.
   - However, within each supply-chain and board cluster (`HARD_TECH_SEMI_688`, `GROWTH_EV_BIO_300`, `SZ_MAIN_SZ002`), institutional northbound capital (`northbound_net_buy_shares`) and verified KOL catalysts hit **sector leaders** first and spill over to peer stocks over the subsequent $1\text{d}\text{--}20\text{d}$.

---

## 1. Mathematical Formulation

### 1.1 Leave-One-Out (`j != i`) Supply-Chain Leader Spillover

To prevent self-inclusion look-ahead (`A_ii = 0`), the supply-chain leader spillover for stock $i$ on date $t$ within cluster $c(i)$ is computed strictly over peers $j \in c(i) \setminus \{i\}$:

$$w_{j,t}^{\text{leader}} = \sigma\left(1.8 \, \tilde{r}_{\text{NB}}(j,t) + 1.2 \, \tilde{r}_{\text{KOL}}(j,t)\right), \quad \text{Spillover}_{i,t} = \frac{\sum_{j \in c(i), j \neq i} w_{j,t}^{\text{leader}} \left(0.60 \, \tilde{r}_{\text{ret}, 1\text{d}}(j,t) + 0.40 \, \tilde{r}_{\text{NB}}(j,t)\right)}{\sum_{j \in c(i), j \neq i} w_{j,t}^{\text{leader}}}$$

where $\tilde{r}(\cdot)$ denotes the cross-sectionally centered percentile rank on date $t$.

### 1.2 Law Stock-RSI-5 Price-Limit Sign-Bifurcated Microstructure

Let $s_{\text{rev}}(i) \in \{+1, -1\}$ be the board microstructure sign ($+1$ for `±10%` Main Board and `HK/US`, $-1$ for `±20%` STAR/ChiNext) and $\rho_{\text{lim}}(i,t) = \min\left(1, |r_{i,t}^{1\text{d}}| / L_{\text{board}}(i)\right)$ be the limit-proximity ratio. The bifurcated microstructure signal is:

$$\phi_{\text{bifurc}}(i,t) = s_{\text{rev}}(i) \cdot \left(-0.52 \, \tilde{r}_{\text{margin}}(i,t) - 0.48 \, \tilde{r}_{\text{gap}}(i,t)\right) \cdot \left(1 - 0.25 \, \rho_{\text{lim}}(i,t)\right)$$

---

## 2. Executable Implementation

The reference implementation in `scripts/cross_board_supply_chain_rsi.py` (exported as `fin_skills.china.cross_board_supply_chain_rsi`) provides:

- `classify_board_microstructure_regime(symbol)`: Returns `BoardMicrostructureSpec` with `board_name`, `board_type_id`, `price_limit_pct`, `settlement_rule`, `investor_threshold_rmb`, `micro_reversal_sign`, and `supply_chain_cluster`.
- `compute_cross_board_supply_chain_features(df)`: Appends `supply_chain_leader_spillover`, `smart_vs_retail_divergence`, `cross_board_limit_bifurcation`, `intraday_candle_asymmetry`, and `garp_valuation_quality`.
- `audit_cross_board_spillover_causality(df)`: Executable guard verifying leave-one-out causality and cross-board sign bifurcation.

## Related Skills

- `china-trading-stack` — A-share `T+1`, board lots, and price-limit execution rules.
- `china-ashare-data` — Point-in-Time A-share data sourcing and adjustment traps.
- `kol-credibility-registry` — Dynamic Brier-score KOL credibility weighting.
- `signal-reconciler` — Multi-channel entropy-weighted signal conflict resolution.
