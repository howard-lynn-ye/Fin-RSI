# Dual-Layer Governed `Fin-RSI` (`Skill-Space RSI x Operator-Space RSI`) Synergy Report

- **Generated**: `2026-09-28T08:47:02Z`
- **Dual Cryptographic Locks**:
  - `SKILL_HARNESS_LOCK`: Verified (`4/4` files SHA-256 intact)
  - `HARNESS_LOCK`: `6e2da08b0bc7af051f8c344d8c9f30f7307e001da907f6226053ce452dbb4b3b` (`zero_baseline_penalty_ast_verified = True`)
- **Evaluation Scope**: $M=5$ disjoint seeds `[20260923, 20260924, 20260925, 20260926, 20260927]`, $N=108$ routing queries, `FinGuardBench-60`, and `17,886` strictly Point-in-Time observations across `799` trading dates (`2018–2026`).

---

## 1. Orthogonal $2 \times 2$ Dual-Layer `Fin-RSI` Factorial Ablation (`JEV System-One + DeepSeek-R1-Distill-1.5B` + `64-KC`, $M=5$ Seeds)

| Dual-Layer Arm (`Skill-Space x Operator-Space`) | Trigger Top-1 (`N=108`) | `JEV` Router Top-1 | Guard `Pass@1` (%) | Leak Rate (%) | OOS Daily Rank IC | Annualized IC IR | OOS Net Sharpe (Panel) | Crisis (`2018/2022`) Sharpe | Sparse (`n∈{1,2}`) IC | Max Drawdown (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`(Gen-0 Skills, Gen-0 Operators)`** | `66.7%` (`72/108`) | `86.1%` (`93/108`) | `82.5% ± 0.2%` | `14.1%` | `+0.0149 ± 0.0034` | `+0.30 ± 0.07` | `+1.01 ± 0.17` | `+1.54 ± 0.15` | `+0.0013 ± 0.0062` | `-19.40%` |
| **`(Gen-0 Skills, Gen-3 Operators)`** | `66.7%` (`72/108`) | `86.1%` (`93/108`) | `83.0% ± 0.3%` | `14.1%` | `+0.0326 ± 0.0067` | `+0.67 ± 0.13` | `+1.14 ± 0.50` | `+1.54 ± 0.45` | `+0.0149 ± 0.0081` | `-14.87%` |
| **`(Gen-3 Skills, Gen-0 Operators)`** | `100.0%` (`108/108`) | `100.0%` (`108/108`) | `96.3% ± 0.2%` | `0.0%` | `+0.0183 ± 0.0026` | `+0.37 ± 0.05` | `+1.23 ± 0.11` | `+1.82 ± 0.17` | `+0.0011 ± 0.0051` | `-18.15%` |
| **`(Gen-3 Skills, Gen-3 Operators)` [Champion]** | **`100.0%` (`108/108`)** | **`100.0%` (`108/108`)** | **`96.8% ± 0.1%`** | **`0.0%`** | **`+0.0495 ± 0.0025`** | **`+1.04 ± 0.05`** | **`+1.90 ± 0.08`** | **`+2.42 ± 0.11`** | **`+0.0290 ± 0.0024`** | **`-12.83%`** |

---

## 2. Super-Additive Synergy Decomposition ($\Delta_{\text{synergy}} = \mathcal{M}_{3,3} - \mathcal{M}_{3,0} - \mathcal{M}_{0,3} + \mathcal{M}_{0,0}$)

| Metric $\mathcal{M}$ | Skill-Only Gain ($\mathcal{M}_{3,0} - \mathcal{M}_{0,0}$) | Operator-Only Gain ($\mathcal{M}_{0,3} - \mathcal{M}_{0,0}$) | Joint Dual-Layer Gain ($\mathcal{M}_{3,3} - \mathcal{M}_{0,0}$) | Super-Additive Synergy $\Delta_{\text{synergy}}$ | Synergy $t$-stat ($p$-val) | Joint vs. `(0,0)` $t$-stat |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **OOS Daily Rank IC** | `+0.0035` | `+0.0177` | **`+0.0346`** | **`+0.0134 ± 0.0046`** | `t = 6.57` (`p = 2.78e-03`) | `t = 28.32` |
| **Annualized Net Sharpe** | `+0.22` | `+0.13` | **`+0.90`** | **`+0.54 ± 0.40`** | `t = 3.06` (`p = 3.75e-02`) | `t = 12.23` |
| **Crisis (`2018/2022`) Sharpe** | `+0.27` | `-0.00` | **`+0.87`** | **`+0.60 ± 0.29`** | `t = 4.64` (`p = 9.70e-03`) | `t = 10.42` |
| **Sparse (`n∈{1,2}`) Rank IC** | `-0.0002` | `+0.0136` | **`+0.0277`** | **`+0.0143 ± 0.0058`** | `t = 5.51` (`p = 5.29e-03`) | `t = 10.22` |

---

## 3. Cross-Architecture Routing & Representation Generalization ($M=5$ Seeds)

| Routing & Representation Architecture | Paradigm | `(Gen-0, Gen-0)` Sharpe (`Pass@1`) | `(Gen-0, Gen-3)` Sharpe (`Pass@1`) | `(Gen-3, Gen-0)` Sharpe (`Pass@1`) | **`(Gen-3, Gen-3)` Champion Sharpe (`Pass@1`)** | Synergy $\Delta_{\text{synergy}}(\text{Sharpe})$ | Synergy $\Delta_{\text{synergy}}(\text{IC})$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`BM25S-Lexical`** | Lexical TF-IDF Router | `+0.19` (`77.8%`) | `+0.22` (`77.8%`) | `+0.57` (`95.7%`) | **`+1.46 ± 0.12` (`96.6%`)** | **`+0.86`** | **`+0.0182`** |
| **`ProsusAI/finbert (110M)`** | Finance Domain Encoder (110M) | `+0.38` (`80.5%`) | `+0.48` (`80.6%`) | `+0.83` (`96.0%`) | **`+1.52 ± 0.06` (`96.6%`)** | **`+0.59`** | **`+0.0228`** |
| **`BAAI/bge-reranker-v2-m3 (568M)`** | Multilingual Dense Reranker (568M) | `+0.83` (`82.2%`) | `+0.79` (`82.4%`) | `+1.06` (`95.3%`) | **`+1.59 ± 0.14` (`95.9%`)** | **`+0.57`** | **`+0.0182`** |
| **`JEV System-One + DeepSeek-R1-Distill-1.5B`** | Calibrated Bi-Encoder + Local 1.5B Agent | `+1.01` (`82.5%`) | `+1.14` (`83.0%`) | `+1.23` (`96.3%`) | **`+1.91 ± 0.08` (`96.8%`)** | **`+0.54`** | **`+0.0134`** |
