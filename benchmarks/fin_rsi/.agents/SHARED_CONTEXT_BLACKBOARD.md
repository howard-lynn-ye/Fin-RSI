# Governed Financial Recursive Self-Improvement (`Fin-RSI`) Multi-Seed ($M=5$) Pareto Ledger

- **Campaign**: `fin_rsi_multimodal_alpha_campaign`
- **Compute Engine**: `PyTorch-2.12-x86_64-AVX512-TensorEngine (local-workstation)` | **Elapsed**: `23.22s`
- **Harness SHA-256 Lock**: `6e2da08b0bc7af051f8c344d8c9f30f7307e001da907f6226053ce452dbb4b3b` (`zero_baseline_penalty_ast_verified = True`)
- **Dual Sample Provenance**: `n_used = 17,886 / dataset_rows = 207,742` (`799` OOS trading dates, 2018–2026)
- **Registered Disjoint Seeds ($M=5$)**: `[20260923, 20260924, 20260925, 20260926, 20260927]`
- **Overall 3-Way Pareto Verdict**: **`PROMOTED_CHAMPION`** (`Candidate: Row_7_RSI_Gen3_Streaming_Woodbury_Fisher_JEV_64KC`)

## 1. Seven-Row Multi-Seed ($M=5$) Financial RSI Evolution & Pareto Ledger

| Row / Generation Arm | OOS Daily Rank IC (% of Row 1) | Annualized IC IR | Annualized Net Sharpe | Deflated Sharpe (DSR) | Max Drawdown (%) | Sparse Ticker (`n∈{1,2}`) IC | Crisis (`2018/2022`) Sharpe | Seq ESS Ratio | Guard Leak (%) | Pareto Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Row_1_Full_Dense_Multimodal_Ref` | `+0.03027 ± 0.00362` (`100.00%`) | `+0.613 ± 0.073` | `+1.476 ± 0.224` | `1.0000` | `-15.34% ± 3.21%` | `+0.01181 ± 0.00350` | `+2.004 ± 0.208` | `6.85x` | `0.0%` | `ROW1_FULL_DENSE_ANCHOR` |
| `Row_2_Prod_Baseline_Verbal_Reflexion_RSI` | `-0.02101 ± 0.00292` (`-69.43%`) | `-0.413 ± 0.056` | `-0.780 ± 0.090` | `0.0000` | `-34.19% ± 2.01%` | `-0.01950 ± 0.00400` | `-0.795 ± 0.158` | `1.00x` | `38.5%` | `PRODUCTION_BASELINE` |
| `Row_3_Prod_Baseline_MMAN_Barra_Dual` | `+0.01345 ± 0.00507` (`44.43%`) | `+0.269 ± 0.102` | `+0.928 ± 0.307` | `0.8403` | `-16.67% ± 4.81%` | `+0.00314 ± 0.00597` | `+1.348 ± 0.308` | `1.18x` | `0.0%` | `PRODUCTION_BASELINE` |
| `Row_4_Prod_Baseline_JEV_SystemOne_Static_64KC` | `+0.01832 ± 0.00264` (`60.53%`) | `+0.368 ± 0.050` | `+1.233 ± 0.110` | `1.0000` | `-18.15% ± 2.86%` | `+0.00107 ± 0.00514` | `+1.816 ± 0.168` | `1.34x` | `0.0%` | `PRODUCTION_BASELINE` |
| `Row_5_RSI_Gen1_ValueSpace_BoundedESS` | `+0.02348 ± 0.00588` (`77.56%`) | `+0.468 ± 0.108` | `+1.287 ± 0.404` | `1.0000` | `-15.59% ± 4.42%` | `+0.01187 ± 0.00755` | `+1.801 ± 0.436` | `4.55x` | `0.0%` | `HONEST_BELOW_THRESHOLD` |
| `Row_6_RSI_Gen2_Subspace_Precision_Stein` | `+0.03078 ± 0.00758` (`101.68%`) | `+0.628 ± 0.152` | `+1.420 ± 0.589` | `1.0000` | `-15.95% ± 8.45%` | `+0.01246 ± 0.00912` | `+1.916 ± 0.545` | `4.55x` | `0.0%` | `HONEST_BELOW_THRESHOLD` |
| `Row_7_RSI_Gen3_Streaming_Woodbury_Fisher_JEV_64KC` | `+0.04946 ± 0.00254` (`163.40%`) | `+1.037 ± 0.047` | `+1.905 ± 0.078` | `1.0000` | `-12.83% ± 0.78%` | `+0.02897 ± 0.00240` | `+2.417 ± 0.105` | `4.55x` | `0.0%` | `PROMOTED_CHAMPION` |

## 2. Three Financial Physical Diagnostic Probes (`diagnose_operator_physics.py`)

1. **Probe A (Pre-LN Scale-Cancellation Detector)**:
   - Pre-LN Token Scaling Gradient Norm `||dL/dw_pre||_2`: `4.755020e-05` (Exact Scale Cancellation inside LayerNorm)
   - Post-Encoder Value-Space Pooling Gradient Norm `||dL/dw_post||_2`: `1.737295e+00` (`O(1)` Active Gradient Restored)
2. **Probe B (Intraday Burst Weight Collapse & Sequence ESS Probe)**:
   - Horizon `T=64` ESS Ratio (`Bounded Value-Space + LSE` vs `Unbounded exp(2.5*z)`): **`5.33x`**
   - Horizon `T=128` ESS Ratio: **`6.74x`**
3. **Probe C (Macro vs. Sparse/Tail & Crisis Regime Slice Gap Probe)**:
   - Sparse Small-Cap/STAR (`n in {1,2}`) Scalar Shrinkage MSE: `0.23157`
   - Sparse Small-Cap/STAR (`n in {1,2}`) 16D Subspace Precision + Positive-Part James-Stein MSE: `0.06175` (**`-73.33%`** estimation error)
