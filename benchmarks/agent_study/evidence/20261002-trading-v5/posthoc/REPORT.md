# Completed v5: post hoc diagnostics

These statistics describe existing frozen decisions. They cannot identify which component caused the return differences.

| Model | Arm | Mean return % | Mean Sharpe | Mean max DD % | Mean invested % | Mean turnover | Submitted | Algorithm decisions | Exact adoption |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 7b | raw | 18.583 | 0.743 | -14.781 | 96.110 | 33.797 | 88/132 | 0 | 0 |
| 7b | library | 37.042 | 1.189 | -13.242 | 99.997 | 16.112 | 60/132 | 4 | 0 |
| 14b | raw | 37.368 | 1.120 | -15.257 | 99.230 | 58.042 | 128/132 | 0 | 0 |
| 14b | library | 27.696 | 0.862 | -15.795 | 97.779 | 41.505 | 118/132 | 6 | 6 |

All four groups have zero successful skill-document reads. Algorithm execution and exact adoption are distinct. Missing submissions retain earlier holdings, so a lower submission rate does not mechanically imply lower market exposure. Mean risk metrics average separate seed trajectories; they do not describe a pooled portfolio. Cost sensitivity reprices fixed decisions, not behavior under new costs.

The factorial study is needed to separate document access, numerical-tool access and their interaction under the same adjusted data input. These post hoc observations alone do not establish that library algorithms caused either positive or negative effects.
