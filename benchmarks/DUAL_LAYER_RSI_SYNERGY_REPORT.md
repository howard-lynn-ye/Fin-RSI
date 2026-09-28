# Dual-Layer Governed `Fin-RSI` (`Skill-Space RSI x Operator-Space RSI`) Synergy Report

- **Generated**: `2026-09-28T09:06:36Z`
- **Dual Cryptographic Locks**:
  - `SKILL_HARNESS_LOCK`: Verified (`4/4` files SHA-256 intact)
  - `HARNESS_LOCK`: `6e2da08b0bc7af051f8c344d8c9f30f7307e001da907f6226053ce452dbb4b3b` (`zero_baseline_penalty_ast_verified = True`)
- **Evaluation Scope**: $M=5$ disjoint seeds `[20260923, 20260924, 20260925, 20260926, 20260927]`, $N=108$ routing queries, `FinGuardBench-60`, and `17,886` strictly Point-in-Time observations across `799` trading dates (`2018–2026`).

---

## 1. Orthogonal $2 \times 2$ Dual-Layer `Fin-RSI` Factorial Ablation (`JEV System-One + DeepSeek-R1-Distill-1.5B` + `64-KC`, $M=5$ Seeds)

| Dual-Layer Arm (`Skill-Space x Operator-Space`) | Trigger Top-1 (`N=108`) | `JEV` Router Top-1 | Guard `Pass@1` (%) | Leak Rate (%) | OOS Daily Rank IC | Annualized IC IR | OOS Net Sharpe (Panel) | Crisis (`2018/2022`) Sharpe | Sparse (`n∈{1,2}`) IC | Max Drawdown (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`(Gen-0 Skills, Gen-0 Operators)`** | `66.7%` (`72/108`) | `86.1%` (`93/108`) | `82.7% ± 0.2%` | `13.9%` | `+0.0151 ± 0.0035` | `+0.31 ± 0.07` | `+1.00 ± 0.18` | `+1.54 ± 0.16` | `+0.0012 ± 0.0062` | `-19.37%` |
| **`(Gen-0 Skills, Gen-3 Operators)`** | `66.7%` (`72/108`) | `86.1%` (`93/108`) | `83.2% ± 0.3%` | `13.9%` | `+0.0327 ± 0.0065` | `+0.67 ± 0.13` | `+1.14 ± 0.48` | `+1.54 ± 0.45` | `+0.0152 ± 0.0078` | `-14.89%` |
| **`(Gen-3 Skills, Gen-0 Operators)`** | `100.0%` (`108/108`) | `100.0%` (`108/108`) | `96.3% ± 0.2%` | `0.0%` | `+0.0183 ± 0.0026` | `+0.37 ± 0.05` | `+1.23 ± 0.11` | `+1.82 ± 0.17` | `+0.0011 ± 0.0051` | `-18.15%` |
| **`(Gen-3 Skills, Gen-3 Operators)` [Champion]** | **`100.0%` (`108/108`)** | **`100.0%` (`108/108`)** | **`96.8% ± 0.1%`** | **`0.0%`** | **`+0.0495 ± 0.0025`** | **`+1.04 ± 0.05`** | **`+1.90 ± 0.08`** | **`+2.42 ± 0.11`** | **`+0.0290 ± 0.0024`** | **`-12.83%`** |

---

## 2. Super-Additive Synergy Decomposition ($\Delta_{\text{synergy}} = \mathcal{M}_{3,3} - \mathcal{M}_{3,0} - \mathcal{M}_{0,3} + \mathcal{M}_{0,0}$)

| Metric $\mathcal{M}$ | Skill-Only Gain ($\mathcal{M}_{3,0} - \mathcal{M}_{0,0}$) | Operator-Only Gain ($\mathcal{M}_{0,3} - \mathcal{M}_{0,0}$) | Joint Dual-Layer Gain ($\mathcal{M}_{3,3} - \mathcal{M}_{0,0}$) | Super-Additive Synergy $\Delta_{\text{synergy}}$ | Synergy $t$-stat ($p$-val) | Joint vs. `(0,0)` $t$-stat |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **OOS Daily Rank IC** | `+0.0032` | `+0.0176` | **`+0.0343`** | **`+0.0136 ± 0.0043`** | `t = 6.98` (`p = 2.22e-03`) | `t = 28.55` |
| **Annualized Net Sharpe** | `+0.23` | `+0.14` | **`+0.90`** | **`+0.54 ± 0.38`** | `t = 3.19` (`p = 3.31e-02`) | `t = 12.04` |
| **Crisis (`2018/2022`) Sharpe** | `+0.28` | `+0.01` | **`+0.88`** | **`+0.59 ± 0.28`** | `t = 4.76` (`p = 8.88e-03`) | `t = 10.32` |
| **Sparse (`n∈{1,2}`) Rank IC** | `-0.0002` | `+0.0140` | **`+0.0277`** | **`+0.0139 ± 0.0056`** | `t = 5.58` (`p = 5.06e-03`) | `t = 10.29` |

---

## 3. Cross-Architecture Routing & Representation Generalization ($M=5$ Seeds)

| Routing & Representation Architecture | Paradigm | `(Gen-0, Gen-0)` Sharpe (`Pass@1`) | `(Gen-0, Gen-3)` Sharpe (`Pass@1`) | `(Gen-3, Gen-0)` Sharpe (`Pass@1`) | **`(Gen-3, Gen-3)` Champion Sharpe (`Pass@1`)** | Synergy $\Delta_{\text{synergy}}(\text{Sharpe})$ | Synergy $\Delta_{\text{synergy}}(\text{IC})$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`BM25S-Lexical`** | Lexical TF-IDF Router | `+0.19` (`77.8%`) | `+0.22` (`77.8%`) | `+0.57` (`95.7%`) | **`+1.46 ± 0.12` (`96.6%`)** | **`+0.86`** | **`+0.0182`** |
| **`ProsusAI/finbert (110M)`** | Finance Domain Encoder (110M) | `+0.39` (`79.5%`) | `+0.46` (`79.5%`) | `+0.83` (`96.0%`) | **`+1.52 ± 0.06` (`96.6%`)** | **`+0.62`** | **`+0.0233`** |
| **`BAAI/bge-reranker-v2-m3 (568M)`** | Multilingual Dense Reranker (568M) | `+0.71` (`81.1%`) | `+0.81` (`81.3%`) | `+1.06` (`95.3%`) | **`+1.59 ± 0.14` (`95.9%`)** | **`+0.43`** | **`+0.0193`** |
| **`JEV System-One + DeepSeek-R1-Distill-1.5B`** | Calibrated Bi-Encoder + Local 1.5B Agent | `+1.00` (`82.7%`) | `+1.14` (`83.2%`) | `+1.23` (`96.3%`) | **`+1.91 ± 0.08` (`96.8%`)** | **`+0.54`** | **`+0.0136`** |
