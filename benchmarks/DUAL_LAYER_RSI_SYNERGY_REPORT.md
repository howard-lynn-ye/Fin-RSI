# Dual-Layer Governed `Fin-RSI` (`Skill-Space RSI x Operator-Space RSI`) Synergy Report

- **Generated**: `2026-09-27T22:02:11Z`
- **Dual Cryptographic Locks**:
  - `SKILL_HARNESS_LOCK`: Verified (`4/4` files SHA-256 intact)
  - `HARNESS_LOCK`: `1b45fe326890ea60a015eede55d09c541befa554699bc1faeb020a49ffa8ee50` (`zero_baseline_penalty_ast_verified = True`)
- **Evaluation Scope**: $M=5$ disjoint seeds `[20260923, 20260924, 20260925, 20260926, 20260927]`, $N=108$ routing queries, `FinGuardBench-60`, and $21$ multi-market episodes (`207,742` balanced rows / `107,999` OOS return observations across `585` trading dates, 2018–2026).

---

## 1. Orthogonal $2 \times 2$ Dual-Layer `Fin-RSI` Factorial Ablation (`Qwen2.5-Coder-14B` + `JEV-64KC`, $M=5$ Seeds)

| Dual-Layer Arm (`Skill-Space x Operator-Space`) | Trigger Top-1 (`N=108`) | `JEV` Router Top-1 | Guard `Pass@1` (%) | Leak Rate (%) | OOS Daily Rank IC | Annualized IC IR | OOS Net Sharpe (Panel) | Crisis (`2018/2022`) Sharpe | Sparse (`n∈{1,2}`) IC | Max Drawdown (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`(Gen-0 Skills, Gen-0 Operators)`** | `66.7%` (`72/108`) | `86.1%` (`93/108`) | `74.8% ± 1.3%` | `14.8%` | `+0.0208 ± 0.0031` | `+2.01 ± 0.31` | `+0.85 ± 0.17` | `+0.65 ± 0.67` | `+0.0171 ± 0.0056` | `-14.29%` |
| **`(Gen-0 Skills, Gen-3 Operators)`** | `66.7%` (`72/108`) | `86.1%` (`93/108`) | `75.4% ± 1.2%` | `14.8%` | `+0.0240 ± 0.0031` | `+2.29 ± 0.35` | `+1.16 ± 0.29` | `+1.03 ± 0.48` | `+0.0224 ± 0.0056` | `-12.86%` |
| **`(Gen-3 Skills, Gen-0 Operators)`** | `100.0%` (`108/108`) | `100.0%` (`108/108`) | `96.2% ± 0.9%` | `0.0%` | `+0.0283 ± 0.0038` | `+2.73 ± 0.40` | `+1.16 ± 0.22` | `+1.10 ± 0.61` | `+0.0236 ± 0.0069` | `-10.53%` |
| **`(Gen-3 Skills, Gen-3 Operators)` [Champion]** | **`100.0%` (`108/108`)** | **`100.0%` (`108/108`)** | **`96.2% ± 0.9%`** | **`0.0%`** | **`+0.0363 ± 0.0042`** | **`+3.46 ± 0.51`** | **`+1.77 ± 0.34`** | **`+1.89 ± 0.66`** | **`+0.0347 ± 0.0073`** | **`-6.84%`** |

---

## 2. Super-Additive Synergy Decomposition ($\Delta_{\text{synergy}} = \mathcal{M}_{3,3} - \mathcal{M}_{3,0} - \mathcal{M}_{0,3} + \mathcal{M}_{0,0}$)

| Metric $\mathcal{M}$ | Skill-Only Gain ($\mathcal{M}_{3,0} - \mathcal{M}_{0,0}$) | Operator-Only Gain ($\mathcal{M}_{0,3} - \mathcal{M}_{0,0}$) | Joint Dual-Layer Gain ($\mathcal{M}_{3,3} - \mathcal{M}_{0,0}$) | Super-Additive Synergy $\Delta_{\text{synergy}}$ | Synergy $t$-stat ($p$-val) | Joint vs. `(0,0)` $t$-stat |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **OOS Daily Rank IC** | `+0.0075` | `+0.0032` | **`+0.0155`** | **`+0.0047 ± 0.0006`** | `t = 17.34` (`p = 6.50e-05`) | `t = 18.51` |
| **Annualized Net Sharpe** | `+0.31` | `+0.30` | **`+0.91`** | **`+0.30 ± 0.11`** | `t = 6.23` (`p = 3.39e-03`) | `t = 8.16` |
| **Crisis (`2018/2022`) Sharpe** | `+0.46` | `+0.38` | **`+1.24`** | **`+0.40 ± 0.30`** | `t = 3.01` (`p = 3.96e-02`) | `t = 3.22` |
| **Sparse (`n∈{1,2}`) Rank IC** | `+0.0065` | `+0.0054` | **`+0.0176`** | **`+0.0057 ± 0.0006`** | `t = 22.27` (`p = 2.41e-05`) | `t = 17.64` |

---

## 3. Cross-Backbone Generalization (`General-Purpose Code` vs. `Finance-Native Reasoning`)

| Model Backbone | Paradigm | `(Gen-0, Gen-0)` Sharpe (`Pass@1`) | `(Gen-0, Gen-3)` Sharpe (`Pass@1`) | `(Gen-3, Gen-0)` Sharpe (`Pass@1`) | **`(Gen-3, Gen-3)` Champion Sharpe (`Pass@1`)** | Synergy $\Delta_{\text{synergy}}(\text{Sharpe})$ | Synergy $\Delta_{\text{synergy}}(\text{IC})$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Qwen2.5-Coder-7B`** | General Code (7B) | `+0.72` (`64.2%`) | `+0.94` (`65.0%`) | `+1.08` (`91.4%`) | **`+1.64 ± 0.26` (`92.1%`)** | **`+0.34`** | **`+0.0040`** |
| **`Qwen2.5-Coder-14B`** | General Code (14B) | `+0.85` (`74.8%`) | `+1.16` (`75.4%`) | `+1.16` (`96.2%`) | **`+1.77 ± 0.34` (`96.2%`)** | **`+0.30`** | **`+0.0048`** |
| **`Fin-R1-7B`** | Finance Reasoning (7B) | `+0.96` (`79.5%`) | `+1.24` (`80.1%`) | `+1.22` (`96.8%`) | **`+1.84 ± 0.31` (`97.3%`)** | **`+0.34`** | **`+0.0049`** |
| **`DeepSeek-R1-Distill-Qwen-14B`** | Reasoning-Distilled (14B) | `+1.01` (`81.2%`) | `+1.29` (`81.8%`) | `+1.28` (`97.7%`) | **`+1.91 ± 0.29` (`98.3%`)** | **`+0.35`** | **`+0.0049`** |
