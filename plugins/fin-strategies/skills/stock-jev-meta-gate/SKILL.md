---
name: stock-jev-meta-gate
description: >-
  [fin-decision] Financial JEV (Joint Evaluation Vector) structured meta-decision and regime gating framework. Evaluates macro regimes (choice: Bull Momentum, Chop, High-Vol Shock, Liquidity Contraction), fundamental/tail-risk disqualification (noul), and 5-level ordinal quality scoring (score). TRIGGER - stock_jev, StockJEVDecisionEngine, NeuralStockJEVNet, AnalyticalStockJEV, check_stock_jev_gate, audit_stock_jev_causality, financial JEV structured decision, macro regime choice gating, A-share tail risk noul filter, ordinal alpha score. SKIP for continuous portfolio optimization solvers (portfolio-and-risk) and intraday order routing (execution-algorithms).
license: MIT
compatibility: Python 3.10+, numpy, pandas, torch
metadata:
  verified_on: "2026-10-02"
---

# Financial JEV (Joint Evaluation Vector) Meta-Decision & Risk Gating (`stock-jev-meta-gate`)

## Two-Tier Decoupled Architecture

Continuous portfolio optimization ($w_t \in \Delta^{N-1}$, HRP, Ledoit-Wolf, Barra null-space projection) operates in **$\mathcal{P}$-Space**, whereas high-level discrete market decisions operate in **$\mathcal{Z}$-Space**. 

Following the JEV wire contract (`https://docs.typesafe.ai/api`), `stock-jev-meta-gate` decouples high-level structural evaluation from continuous execution:

```
                    ┌────────────────────────────────────────────────────────┐
                    │      JEV 顶层离散元决策 (Discrete Meta-Decision)         │
                    │  - Choice: 宏观与波动率环境分类 (Bull / Chop / Crisis)    │
                    │  - Noul: 黑天鹅与财报造假排雷 (Disqualification Filter)    │
                    │  - Score: 5 级离散 Alpha 质量评分 (Ordinal Quality)       │
                    └───────────────────────────┬────────────────────────────┘
                                                │ 离散状态与合格股票池
                                                ▼
                    ┌────────────────────────────────────────────────────────┐
                    │     fin-skills 底层连续量化执行器 (Continuous Solver)     │
                    │  - 132 个数学算法 (HRP, Ledoit-Wolf, Barra Null-Space) │
                    │  - 41 个确定性硬守卫 (LOB 冲击、涨跌停、无风险利率、T+1)   │
                    │  - 输出: 严格满足风控与成本曲线的精确持仓权重 w_t          │
                    └────────────────────────────────────────────────────────┘
```

## JEV Decision Primitives in Financial Markets

1. **`choice` (Macro Regime Gating)**:
   - Evaluates point-in-time USD/CNH FX drift, market turnover z-score, 20-day volatility term structure, and LOB depth ratio.
   - Outputs calibrated probability distribution across:
     - `BULL_MOMENTUM`: Target gross exposure 100%, momentum tilt.
     - `CHOP_MEAN_REVERSION`: Target gross exposure 85%, factor-neutral.
     - `HIGH_VOL_SHOCK`: Target gross exposure 45%, minimum variance / risk parity.
     - `LIQUIDITY_CONTRACTION`: Target gross exposure 20%, cash protection.
2. **`noul` (Universe Disqualification Filter)**:
   - Evaluates tail-risk z-score, trading halt risk, and ST warnings.
   - Returns calibrated probability $p(\text{disqualified})$. Tickers with $p > 0.5$ receive $w_i = 0$.
3. **`score` (Ordinal Alpha Calibration)**:
   - 5-level ordinal quality scale ($0 = \text{Severe Downside} \dots 4 = \text{Top Alpha}$).
   - Tilts raw model logits with expected score $\mathbb{E}[S] = \sum_{k=0}^4 k \cdot p_k$.

## Executable API & Guard Verification

```python
from fin_skills.jev import (
    AnalyticalStockJEV,
    NeuralStockJEVAdapter,
    build_macro_regime_question,
    build_disqualification_question,
)
from fin_skills.api.guards.stock_jev_gate import check_stock_jev_gate

engine = AnalyticalStockJEV()
q = build_macro_regime_question()
res = engine.predict({"vol_shock_20d": 1.5, "macro_fx_shock": -0.8}, questions=q)
outcome = check_stock_jev_gate(res)
assert outcome.passed
```
