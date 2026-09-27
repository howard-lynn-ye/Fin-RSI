# Governed Financial Recursive Self-Improvement (`Fin-RSI`) Multi-Seed ($M=5$) Pareto Ledger

- **Campaign**: `fin_rsi_multimodal_alpha_campaign`
- **Compute Engine**: `PyTorch-2.12-x86_64-AVX512-TensorEngine (shwaihe.c.googlers.com)` | **Elapsed**: `23.90s`
- **Harness SHA-256 Lock**: `1b45fe326890ea60a015eede55d09c541befa554699bc1faeb020a49ffa8ee50` (`zero_baseline_penalty_ast_verified = True`)
- **Dual Sample Provenance**: `n_used = 107,999 / dataset_rows = 207,742` (`585` OOS trading dates, 2018–2026)
- **Registered Disjoint Seeds ($M=5$)**: `[20260923, 20260924, 20260925, 20260926, 20260927]`
- **Overall 3-Way Pareto Verdict**: **`PROMOTED_CHAMPION`** (`Candidate: Row_7_RSI_Gen3_Streaming_Woodbury_Fisher_JEV_64KC`)

## 1. Seven-Row Multi-Seed ($M=5$) Financial RSI Evolution & Pareto Ledger

| Row / Generation Arm | OOS Daily Rank IC (% of Row 1) | Annualized IC IR | Annualized Net Sharpe | Deflated Sharpe (DSR) | Max Drawdown (%) | Sparse Ticker (`n∈{1,2}`) IC | Crisis (`2018/2022`) Sharpe | Seq ESS Ratio | Guard Leak (%) | Pareto Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Row_1_Full_Dense_Multimodal_Ref` | `+0.03151 ± 0.00421` (`100.00%`) | `+3.035 ± 0.443` | `+1.409 ± 0.397` | `1.0000` | `-8.85% ± 2.61%` | `+0.02640 ± 0.00807` | `+1.592 ± 0.517` | `3.10x` | `0.0%` | `ROW1_FULL_DENSE_ANCHOR` |
| `Row_2_Prod_Baseline_Verbal_Reflexion_RSI` | `+0.00448 ± 0.00241` (`14.23%`) | `+0.422 ± 0.217` | `+0.182 ± 0.233` | `0.0000` | `-22.47% ± 11.26%` | `+0.00285 ± 0.00440` | `-0.344 ± 0.968` | `1.00x` | `38.5%` | `PRODUCTION_BASELINE` |
| `Row_3_Prod_Baseline_MMAN_Barra_Dual` | `+0.02811 ± 0.00363` (`89.22%`) | `+2.670 ± 0.402` | `+1.256 ± 0.359` | `1.0000` | `-11.81% ± 4.11%` | `+0.02144 ± 0.00568` | `+1.073 ± 0.912` | `1.18x` | `0.0%` | `PRODUCTION_BASELINE` |
| `Row_4_Prod_Baseline_JEV_SystemOne_Static_64KC` | `+0.02831 ± 0.00380` (`89.84%`) | `+2.734 ± 0.402` | `+1.160 ± 0.216` | `1.0000` | `-10.53% ± 2.75%` | `+0.02360 ± 0.00693` | `+1.104 ± 0.609` | `1.34x` | `0.0%` | `PRODUCTION_BASELINE` |
| `Row_5_RSI_Gen1_ValueSpace_BoundedESS` | `+0.02872 ± 0.00411` (`91.14%`) | `+2.750 ± 0.465` | `+1.301 ± 0.130` | `1.0000` | `-8.19% ± 1.87%` | `+0.02107 ± 0.00745` | `+1.322 ± 0.551` | `3.48x` | `0.0%` | `HONEST_BELOW_THRESHOLD` |
| `Row_6_RSI_Gen2_Subspace_Precision_Stein` | `+0.03147 ± 0.00410` (`99.87%`) | `+3.006 ± 0.474` | `+1.508 ± 0.281` | `1.0000` | `-6.79% ± 0.84%` | `+0.02903 ± 0.00746` | `+0.771 ± 0.342` | `3.54x` | `0.0%` | `HONEST_BELOW_THRESHOLD` |
| `Row_7_RSI_Gen3_Streaming_Woodbury_Fisher_JEV_64KC` | `+0.03629 ± 0.00417` (`115.16%`) | `+3.463 ± 0.510` | `+1.766 ± 0.338` | `1.0000` | `-6.84% ± 2.50%` | `+0.03467 ± 0.00731` | `+1.891 ± 0.663` | `3.62x` | `0.0%` | `PROMOTED_CHAMPION` |

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
