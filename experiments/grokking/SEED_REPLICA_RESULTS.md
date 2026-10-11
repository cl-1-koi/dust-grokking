# Initialization replicas: unseen operand pairs

Updated 2026-10-11T17:48:09.203800+00:00.

Primary universe: the fixed 4669 held ordered pairs whose reversals are also held. All Dust runs start from scratch with fresh Adam state and no backprop. BP controls are separate models. Same-split initialization and perturbation factors are crossed; these are pilot replicas, not proof of an always-easy subset.

| Run | Method | Update | Train | All held | Unseen pairs | State |
|---|---|---:|---:|---:|---:|---|
| i0-n0 | dust | 3200 | 100.00% | 100.00% | 4669/4669 (100.00%) | HORIZON_COMPLETE |
| i0-n1 | dust | 3200 | 96.46% | 73.49% | 3106/4669 (66.52%) | HORIZON_COMPLETE |
| i1-n0 | dust | 3200 | 99.89% | 94.44% | 4325/4669 (92.63%) | HORIZON_COMPLETE |
| i1-n1 | dust | 3200 | 42.90% | 26.19% | 987/4669 (21.14%) | HORIZON_COMPLETE |
| i2-n0 | dust | 3200 | 100.00% | 99.73% | 4651/4669 (99.61%) | HORIZON_COMPLETE |
| i2-n1 | dust | 3200 | 83.78% | 45.52% | 1560/4669 (33.41%) | HORIZON_COMPLETE |
| i0-bp | bp | 10000 | 99.82% | 99.01% | 4606/4669 (98.65%) | HORIZON_COMPLETE |
| i1-bp | bp | 10000 | 99.89% | 99.33% | 4633/4669 (99.23%) | HORIZON_COMPLETE |
| i2-bp | bp | 10000 | 100.00% | 100.00% | 4669/4669 (100.00%) | HORIZON_COMPLETE |

The table shows latest durable evaluations, potentially at different updates. Only common-update predictions enter the overlap comparison below. Predeclared primary endpoint is update3177; earlier comparisons are interim.

Execution amendment: original Dust through update 600, then the validated CPU suffix/reduction path. Original model, Adam state and perturbation RNG were resumed without reset. Optimized arithmetic is not bitwise identical. Original absolute runtime bounds remain in force.

## Common update 3177

Correct in all six: **1324**; correct in at least one: **4669**. Counts correct in 0…6 runs: `[0, 1, 30, 284, 1030, 2000, 1324]`.

| Pair of runs | Factor varied | Correct counts | Intersection | Jaccard | Marginal expected | Target-conditioned expected | Margin correlation |
|---|---|---:|---:|---:|---:|---:|---:|
| i0-n0 / i0-n1 | Perturbation, fixed init | 4307 / 3182 | 2966 | 0.656 | 2935.3 | 2945.5 | 0.113 |
| i0-n0 / i1-n0 | Initialization, fixed noise | 4307 / 4132 | 3811 | 0.823 | 3811.6 | 3811.9 | 0.041 |
| i0-n0 / i2-n0 | Initialization, fixed noise | 4307 / 4612 | 4256 | 0.913 | 4254.4 | 4254.6 | 0.035 |
| i0-n1 / i1-n1 | Initialization, fixed noise | 3182 / 4129 | 2825 | 0.630 | 2814.0 | 2826.2 | 0.042 |
| i0-n1 / i2-n1 | Initialization, fixed noise | 3182 / 2615 | 1808 | 0.453 | 1782.2 | 1812.7 | 0.064 |
| i1-n0 / i1-n1 | Perturbation, fixed init | 4132 / 4129 | 3667 | 0.798 | 3654.1 | 3656.2 | 0.105 |
| i1-n0 / i2-n0 | Initialization, fixed noise | 4132 / 4612 | 4080 | 0.875 | 4081.6 | 4084.6 | 0.007 |
| i1-n1 / i2-n1 | Initialization, fixed noise | 4129 / 2615 | 2339 | 0.531 | 2312.6 | 2321.1 | 0.010 |
| i2-n0 / i2-n1 | Perturbation, fixed init | 4612 / 2615 | 2594 | 0.560 | 2583.1 | 2585.7 | 0.063 |

Reversal orientations and pairwise comparisons are dependent. Initial/early predictions do not establish which equations trained models consistently solve. No reported overlap threshold constitutes a pass criterion.

Settings: initialization9278700/1/2 × perturbation9278901/2; P16, eight compute threads per worker, unchanged task/AdamW. Dust cap3200updates/20hours; paired BP cap10000updates/1hour. Every job has an independent runtime guard and5GiB checkpoint disk guard.

Collected artifacts: `runs/seed-fast-collected-20261011`. Latest and milestone model checkpoints remain on each worker.
