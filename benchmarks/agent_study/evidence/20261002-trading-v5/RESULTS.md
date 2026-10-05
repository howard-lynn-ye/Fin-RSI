# Trading interface v5: complete development return benchmark

Three seeds per model on the same previously inspected market window. All planned decisions and failures are retained.
The earlier smoke library-use gate failed. These returns measure the published bundle as actually used, including non-use; they do not establish standalone algorithm benefit.

| Model | Raw mean return (%) | Library mean return (%) | Library - raw (pp) |
|---|---:|---:|---:|
| 7b | 18.582923 | 37.041787 | +18.458863 |
| 14b | 37.368101 | 27.696018 | -9.672083 |

| Model | Seed | Raw return (%) | Library return (%) | Difference (pp) |
|---|---:|---:|---:|---:|
| 7b | 11 | 3.443656 | 38.038808 | +34.595151 |
| 7b | 23 | 31.625206 | 43.976371 | +12.351165 |
| 7b | 37 | 20.679907 | 29.110181 | +8.430274 |
| 14b | 11 | 35.062627 | 23.930393 | -11.132234 |
| 14b | 23 | 39.517371 | 34.211401 | -5.305970 |
| 14b | 37 | 37.524305 | 24.946259 | -12.578046 |

| Model | Seed | Arm | Submitted / decisions | Decisions with successful algorithm call | Successful algorithm calls |
|---|---:|---|---:|---:|---:|
| 7b | 11 | raw | 27/44 | 0 | 0 |
| 7b | 11 | library | 19/44 | 1 | 2 |
| 14b | 11 | raw | 44/44 | 0 | 0 |
| 14b | 11 | library | 36/44 | 1 | 1 |
| 7b | 23 | raw | 29/44 | 0 | 0 |
| 7b | 23 | library | 17/44 | 1 | 1 |
| 14b | 23 | raw | 42/44 | 0 | 0 |
| 14b | 23 | library | 44/44 | 3 | 3 |
| 7b | 37 | raw | 32/44 | 0 | 0 |
| 7b | 37 | library | 24/44 | 2 | 7 |
| 14b | 37 | raw | 42/44 | 0 | 0 |
| 14b | 37 | library | 38/44 | 2 | 2 |

Full risk metrics, failure counts, baselines and token counts: batch/aggregate.json.
Detailed library-use audit and failure messages: analysis.json.
Each pair retains protocol, qualification, inference receipt, decisions, NAV and score hashes.
No claim of significance, independent markets, unseen evaluation or live trading performance.
