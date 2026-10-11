# Frozen probability-ensemble diagnostic

Updated 2026-10-11T17:47:39.820871+00:00.

Pure Dust only; frozen models, zero backward passes or retained updates. Primary endpoint: all six at3177. Early rows do not answer the trained9% question. Uniform rules were fixed before scoring; no held-label fitting.

| Step | Group | Members | Train-selected single on U | Probability mean on U | Logit mean on U | Plurality on U | Soft gains outside top1 union | Union count |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 200 | all_six | 6 | 0.00% | 0.00% | 0.00% | 0.02% | 0 | 17 |
| 200 | noise0 | 3 | 0.00% | 0.00% | 0.00% | 0.02% | 0 | 7 |
| 200 | noise1 | 3 | 0.00% | 0.00% | 0.00% | 0.04% | 0 | 13 |
| 200 | init0 | 2 | 0.00% | 0.00% | 0.00% | 0.00% | 0 | 0 |
| 200 | init1 | 2 | 0.02% | 0.00% | 0.00% | 0.04% | 0 | 4 |
| 200 | init2 | 2 | 0.21% | 0.19% | 0.19% | 0.13% | 2 | 13 |
| 1000 | all_six | 6 | 0.04% | 0.00% | 0.00% | 0.06% | 0 | 10 |
| 1000 | noise0 | 3 | 0.04% | 0.02% | 0.00% | 0.04% | 0 | 6 |
| 1000 | noise1 | 3 | 0.04% | 0.00% | 0.00% | 0.06% | 0 | 4 |
| 1000 | init0 | 2 | 0.04% | 0.00% | 0.00% | 0.06% | 0 | 4 |
| 1000 | init1 | 2 | 0.04% | 0.02% | 0.02% | 0.06% | 0 | 5 |
| 1000 | init2 | 2 | 0.00% | 0.00% | 0.00% | 0.00% | 0 | 1 |
| 2000 | all_six | 6 | 2.33% | 0.32% | 0.47% | 0.39% | 0 | 175 |
| 2000 | noise0 | 3 | 2.33% | 1.01% | 1.33% | 1.05% | 1 | 136 |
| 2000 | noise1 | 3 | 0.13% | 0.36% | 0.26% | 0.21% | 0 | 41 |
| 2000 | init0 | 2 | 2.33% | 1.14% | 1.56% | 1.14% | 2 | 123 |
| 2000 | init1 | 2 | 0.47% | 0.28% | 0.21% | 0.34% | 1 | 34 |
| 2000 | init2 | 2 | 0.13% | 0.19% | 0.15% | 0.21% | 0 | 21 |
| 3000 | all_six | 6 | 100.00% | 100.00% | 100.00% | 99.91% | 0 | 4669 |
| 3000 | noise0 | 3 | 100.00% | 100.00% | 100.00% | 99.83% | 0 | 4669 |
| 3000 | noise1 | 3 | 32.19% | 62.63% | 98.33% | 37.97% | 249 | 3275 |
| 3000 | init0 | 2 | 100.00% | 99.70% | 99.12% | 71.24% | 0 | 4669 |
| 3000 | init1 | 2 | 84.43% | 77.21% | 91.09% | 50.70% | 86 | 4079 |
| 3000 | init2 | 2 | 97.04% | 90.92% | 97.00% | 64.32% | 18 | 4571 |
| 3177 | all_six | 6 | 98.78% | 100.00% | 100.00% | 99.94% | 0 | 4669 |
| 3177 | noise0 | 3 | 98.78% | 100.00% | 100.00% | 99.21% | 1 | 4668 |
| 3177 | noise1 | 3 | 56.01% | 98.16% | 100.00% | 86.55% | 60 | 4573 |
| 3177 | init0 | 2 | 68.15% | 93.55% | 96.34% | 80.12% | 38 | 4523 |
| 3177 | init1 | 2 | 88.43% | 98.65% | 99.96% | 88.58% | 43 | 4594 |
| 3177 | init2 | 2 | 98.78% | 96.96% | 99.46% | 78.26% | 11 | 4633 |
| 3200 | all_six | 6 | 99.61% | 100.00% | 100.00% | 99.98% | 0 | 4669 |
| 3200 | noise0 | 3 | 99.61% | 100.00% | 100.00% | 99.98% | 0 | 4669 |
| 3200 | noise1 | 3 | 66.52% | 75.46% | 97.26% | 49.18% | 185 | 3838 |
| 3200 | init0 | 2 | 100.00% | 99.72% | 99.42% | 82.42% | 0 | 4669 |
| 3200 | init1 | 2 | 92.63% | 86.68% | 94.84% | 53.72% | 76 | 4391 |
| 3200 | init2 | 2 | 99.61% | 97.67% | 98.80% | 65.73% | 4 | 4655 |

## Step200 / all_six

Members: i0-n0, i0-n1, i1-n0, i1-n1, i2-n0, i2-n1. Train-selected member: i0-n0.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i0-n0 | 0 | 5.246667 | 68.16 | 0.006063 | 8.64% | 0.29% |
| i0-n1 | 0 | 5.233267 | 68.09 | 0.006077 | 8.82% | 0.36% |
| i1-n0 | 1 | 5.039652 | 64.19 | 0.007184 | 6.91% | 0.39% |
| i1-n1 | 3 | 5.013918 | 64.30 | 0.007313 | 7.69% | 0.43% |
| i2-n0 | 6 | 4.912028 | 62.45 | 0.007942 | 4.99% | 0.47% |
| i2-n1 | 10 | 4.972790 | 63.42 | 0.007603 | 5.42% | 0.67% |
| probability_mean | 0 | 4.993036 | 72.83 | 0.007030 | 14.31% | 0.24% |
| logit_mean | 0 | 4.994797 | 72.90 | 0.007023 | 15.27% | 0.30% |
| plurality | 1 | — | — | — | 9.74% | 0.23% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-00200/all_six`.

## Step200 / noise0

Members: i0-n0, i1-n0, i2-n0. Train-selected member: i0-n0.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i0-n0 | 0 | 5.246667 | 68.16 | 0.006063 | 8.64% | 0.29% |
| i1-n0 | 1 | 5.039652 | 64.19 | 0.007184 | 6.91% | 0.39% |
| i2-n0 | 6 | 4.912028 | 62.45 | 0.007942 | 4.99% | 0.47% |
| probability_mean | 0 | 4.993224 | 71.63 | 0.007063 | 13.25% | 0.24% |
| logit_mean | 0 | 4.998255 | 71.84 | 0.007035 | 13.67% | 0.29% |
| plurality | 1 | — | — | — | 7.30% | 0.27% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-00200/noise0`.

## Step200 / noise1

Members: i0-n1, i1-n1, i2-n1. Train-selected member: i0-n1.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i0-n1 | 0 | 5.233267 | 68.09 | 0.006077 | 8.82% | 0.36% |
| i1-n1 | 3 | 5.013918 | 64.30 | 0.007313 | 7.69% | 0.43% |
| i2-n1 | 10 | 4.972790 | 63.42 | 0.007603 | 5.42% | 0.67% |
| probability_mean | 0 | 5.003712 | 72.05 | 0.006998 | 13.11% | 0.27% |
| logit_mean | 0 | 5.005479 | 72.24 | 0.006982 | 13.74% | 0.30% |
| plurality | 2 | — | — | — | 7.90% | 0.30% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-00200/noise1`.

## Step200 / init0

Members: i0-n0, i0-n1. Train-selected member: i0-n0.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i0-n0 | 0 | 5.246667 | 68.16 | 0.006063 | 8.64% | 0.29% |
| i0-n1 | 0 | 5.233267 | 68.09 | 0.006077 | 8.82% | 0.36% |
| probability_mean | 0 | 5.222310 | 69.08 | 0.006070 | 9.25% | 0.26% |
| logit_mean | 0 | 5.221302 | 69.05 | 0.006075 | 9.39% | 0.29% |
| plurality | 0 | — | — | — | 8.61% | 0.33% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-00200/init0`.

## Step200 / init1

Members: i1-n0, i1-n1. Train-selected member: i1-n0.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i1-n0 | 1 | 5.039652 | 64.19 | 0.007184 | 6.91% | 0.39% |
| i1-n1 | 3 | 5.013918 | 64.30 | 0.007313 | 7.69% | 0.43% |
| probability_mean | 0 | 5.012765 | 65.17 | 0.007249 | 7.76% | 0.41% |
| logit_mean | 0 | 5.010094 | 65.15 | 0.007269 | 7.86% | 0.43% |
| plurality | 2 | — | — | — | 7.16% | 0.39% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-00200/init1`.

## Step200 / init2

Members: i2-n0, i2-n1. Train-selected member: i2-n1.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i2-n0 | 6 | 4.912028 | 62.45 | 0.007942 | 4.99% | 0.47% |
| i2-n1 | 10 | 4.972790 | 63.42 | 0.007603 | 5.42% | 0.67% |
| probability_mean | 9 | 4.929829 | 63.92 | 0.007772 | 5.53% | 0.39% |
| logit_mean | 9 | 4.927524 | 63.93 | 0.007793 | 5.81% | 0.41% |
| plurality | 6 | — | — | — | 5.31% | 0.52% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-00200/init2`.

## Step1000 / all_six

Members: i0-n0, i0-n1, i1-n0, i1-n1, i2-n0, i2-n1. Train-selected member: i1-n1.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i0-n0 | 2 | 8.016321 | 132.26 | 0.002069 | 83.74% | 7.36% |
| i0-n1 | 2 | 8.238041 | 131.39 | 0.001755 | 84.91% | 5.12% |
| i1-n0 | 3 | 7.522175 | 102.52 | 0.002358 | 71.34% | 5.62% |
| i1-n1 | 2 | 8.128800 | 127.09 | 0.002012 | 88.20% | 10.26% |
| i2-n0 | 1 | 7.858369 | 117.75 | 0.001950 | 62.38% | 3.66% |
| i2-n1 | 0 | 8.208670 | 130.85 | 0.001604 | 84.45% | 6.27% |
| probability_mean | 0 | 6.838366 | 86.50 | 0.001958 | 99.96% | 16.38% |
| logit_mean | 0 | 6.768001 | 121.63 | 0.001710 | 100.00% | 23.05% |
| plurality | 3 | — | — | — | 98.94% | 11.66% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-01000/all_six`.

## Step1000 / noise0

Members: i0-n0, i1-n0, i2-n0. Train-selected member: i0-n0.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i0-n0 | 2 | 8.016321 | 132.26 | 0.002069 | 83.74% | 7.36% |
| i1-n0 | 3 | 7.522175 | 102.52 | 0.002358 | 71.34% | 5.62% |
| i2-n0 | 1 | 7.858369 | 117.75 | 0.001950 | 62.38% | 3.66% |
| probability_mean | 1 | 6.972804 | 91.78 | 0.002125 | 96.81% | 9.82% |
| logit_mean | 0 | 6.835489 | 114.44 | 0.001906 | 99.33% | 15.68% |
| plurality | 2 | — | — | — | 85.26% | 6.48% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-01000/noise0`.

## Step1000 / noise1

Members: i0-n1, i1-n1, i2-n1. Train-selected member: i1-n1.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i0-n1 | 2 | 8.238041 | 131.39 | 0.001755 | 84.91% | 5.12% |
| i1-n1 | 2 | 8.128800 | 127.09 | 0.002012 | 88.20% | 10.26% |
| i2-n1 | 0 | 8.208670 | 130.85 | 0.001604 | 84.45% | 6.27% |
| probability_mean | 0 | 7.265361 | 102.53 | 0.001791 | 99.29% | 13.33% |
| logit_mean | 0 | 7.098390 | 133.25 | 0.001605 | 99.96% | 19.69% |
| plurality | 3 | — | — | — | 95.29% | 8.09% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-01000/noise1`.

## Step1000 / init0

Members: i0-n0, i0-n1. Train-selected member: i0-n1.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i0-n0 | 2 | 8.016321 | 132.26 | 0.002069 | 83.74% | 7.36% |
| i0-n1 | 2 | 8.238041 | 131.39 | 0.001755 | 84.91% | 5.12% |
| probability_mean | 0 | 7.603058 | 119.44 | 0.001912 | 95.50% | 8.03% |
| logit_mean | 0 | 7.548622 | 134.37 | 0.001709 | 97.73% | 9.76% |
| plurality | 3 | — | — | — | 84.41% | 6.44% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-01000/init0`.

## Step1000 / init1

Members: i1-n0, i1-n1. Train-selected member: i1-n1.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i1-n0 | 3 | 7.522175 | 102.52 | 0.002358 | 71.34% | 5.62% |
| i1-n1 | 2 | 8.128800 | 127.09 | 0.002012 | 88.20% | 10.26% |
| probability_mean | 1 | 7.295795 | 99.74 | 0.002185 | 93.27% | 11.07% |
| logit_mean | 1 | 7.272840 | 112.55 | 0.001913 | 95.25% | 13.00% |
| plurality | 3 | — | — | — | 80.87% | 8.24% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-01000/init1`.

## Step1000 / init2

Members: i2-n0, i2-n1. Train-selected member: i2-n1.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i2-n0 | 1 | 7.858369 | 117.75 | 0.001950 | 62.38% | 3.66% |
| i2-n1 | 0 | 8.208670 | 130.85 | 0.001604 | 84.45% | 6.27% |
| probability_mean | 0 | 7.551626 | 112.28 | 0.001777 | 88.91% | 6.30% |
| logit_mean | 0 | 7.514408 | 124.83 | 0.001651 | 90.90% | 7.15% |
| plurality | 0 | — | — | — | 74.11% | 5.06% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-01000/init2`.

## Step2000 / all_six

Members: i0-n0, i0-n1, i1-n0, i1-n1, i2-n0, i2-n1. Train-selected member: i0-n0.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i0-n0 | 109 | 4.873581 | 24.95 | 0.024394 | 88.42% | 15.47% |
| i0-n1 | 14 | 6.326462 | 76.22 | 0.007270 | 63.44% | 4.33% |
| i1-n0 | 12 | 6.125887 | 63.39 | 0.006618 | 57.14% | 4.11% |
| i1-n1 | 22 | 6.313818 | 64.11 | 0.007534 | 74.71% | 7.82% |
| i2-n0 | 15 | 5.886754 | 45.61 | 0.009428 | 77.90% | 6.89% |
| i2-n1 | 6 | 6.997331 | 91.28 | 0.004227 | 83.21% | 6.95% |
| probability_mean | 15 | 5.035110 | 40.02 | 0.009912 | 99.93% | 20.12% |
| logit_mean | 22 | 4.738167 | 31.00 | 0.011926 | 100.00% | 27.18% |
| plurality | 18 | — | — | — | 98.16% | 13.86% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-02000/all_six`.

## Step2000 / noise0

Members: i0-n0, i1-n0, i2-n0. Train-selected member: i0-n0.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i0-n0 | 109 | 4.873581 | 24.95 | 0.024394 | 88.42% | 15.47% |
| i1-n0 | 12 | 6.125887 | 63.39 | 0.006618 | 57.14% | 4.11% |
| i2-n0 | 15 | 5.886754 | 45.61 | 0.009428 | 77.90% | 6.89% |
| probability_mean | 47 | 4.885269 | 32.61 | 0.013480 | 97.66% | 17.37% |
| logit_mean | 62 | 4.462613 | 23.59 | 0.018060 | 99.79% | 24.10% |
| plurality | 49 | — | — | — | 88.31% | 10.64% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-02000/noise0`.

## Step2000 / noise1

Members: i0-n1, i1-n1, i2-n1. Train-selected member: i2-n1.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i0-n1 | 14 | 6.326462 | 76.22 | 0.007270 | 63.44% | 4.33% |
| i1-n1 | 22 | 6.313818 | 64.11 | 0.007534 | 74.71% | 7.82% |
| i2-n1 | 6 | 6.997331 | 91.28 | 0.004227 | 83.21% | 6.95% |
| probability_mean | 17 | 5.755573 | 54.46 | 0.006344 | 96.85% | 11.42% |
| logit_mean | 12 | 5.529926 | 57.10 | 0.006993 | 99.43% | 18.36% |
| plurality | 10 | — | — | — | 86.61% | 6.64% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-02000/noise1`.

## Step2000 / init0

Members: i0-n0, i0-n1. Train-selected member: i0-n0.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i0-n0 | 109 | 4.873581 | 24.95 | 0.024394 | 88.42% | 15.47% |
| i0-n1 | 14 | 6.326462 | 76.22 | 0.007270 | 63.44% | 4.33% |
| probability_mean | 53 | 5.012181 | 33.27 | 0.015832 | 93.69% | 14.71% |
| logit_mean | 73 | 4.832775 | 31.59 | 0.018094 | 96.46% | 16.37% |
| plurality | 53 | — | — | — | 74.07% | 9.31% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-02000/init0`.

## Step2000 / init1

Members: i1-n0, i1-n1. Train-selected member: i1-n1.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i1-n0 | 12 | 6.125887 | 63.39 | 0.006618 | 57.14% | 4.11% |
| i1-n1 | 22 | 6.313818 | 64.11 | 0.007534 | 74.71% | 7.82% |
| probability_mean | 13 | 5.800712 | 53.87 | 0.007076 | 83.56% | 8.31% |
| logit_mean | 10 | 5.684476 | 54.62 | 0.007164 | 89.76% | 10.46% |
| plurality | 16 | — | — | — | 64.43% | 5.50% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-02000/init1`.

## Step2000 / init2

Members: i2-n0, i2-n1. Train-selected member: i2-n1.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i2-n0 | 15 | 5.886754 | 45.61 | 0.009428 | 77.90% | 6.89% |
| i2-n1 | 6 | 6.997331 | 91.28 | 0.004227 | 83.21% | 6.95% |
| probability_mean | 9 | 5.908637 | 51.93 | 0.006827 | 94.16% | 9.60% |
| logit_mean | 7 | 5.798348 | 54.84 | 0.007064 | 97.02% | 13.01% |
| plurality | 10 | — | — | — | 80.16% | 6.83% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-02000/init2`.

## Step3000 / all_six

Members: i0-n0, i0-n1, i1-n0, i1-n1, i2-n0, i2-n1. Train-selected member: i0-n0.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i0-n0 | 4669 | 0.185363 | 1.00 | 0.832939 | 100.00% | 100.00% |
| i0-n1 | 2087 | 1.654052 | 2.49 | 0.273944 | 99.01% | 57.01% |
| i1-n0 | 3942 | 0.719135 | 1.19 | 0.525865 | 99.96% | 88.38% |
| i1-n1 | 892 | 2.689948 | 4.92 | 0.145574 | 48.03% | 24.83% |
| i2-n0 | 4531 | 0.415802 | 1.04 | 0.687835 | 100.00% | 97.84% |
| i2-n1 | 1503 | 1.698362 | 2.39 | 0.266885 | 97.59% | 48.12% |
| probability_mean | 4669 | 0.799757 | 1.00 | 0.455507 | 100.00% | 100.00% |
| logit_mean | 4669 | 0.041507 | 1.00 | 0.959635 | 100.00% | 100.00% |
| plurality | 4665 | — | — | — | 100.00% | 99.94% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-03000/all_six`.

## Step3000 / noise0

Members: i0-n0, i1-n0, i2-n0. Train-selected member: i0-n0.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i0-n0 | 4669 | 0.185363 | 1.00 | 0.832939 | 100.00% | 100.00% |
| i1-n0 | 3942 | 0.719135 | 1.19 | 0.525865 | 99.96% | 88.38% |
| i2-n0 | 4531 | 0.415802 | 1.04 | 0.687835 | 100.00% | 97.84% |
| probability_mean | 4669 | 0.390462 | 1.00 | 0.682213 | 100.00% | 100.00% |
| logit_mean | 4669 | 0.029159 | 1.00 | 0.971383 | 100.00% | 100.00% |
| plurality | 4661 | — | — | — | 100.00% | 99.88% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-03000/noise0`.

## Step3000 / noise1

Members: i0-n1, i1-n1, i2-n1. Train-selected member: i2-n1.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i0-n1 | 2087 | 1.654052 | 2.49 | 0.273944 | 99.01% | 57.01% |
| i1-n1 | 892 | 2.689948 | 4.92 | 0.145574 | 48.03% | 24.83% |
| i2-n1 | 1503 | 1.698362 | 2.39 | 0.266885 | 97.59% | 48.12% |
| probability_mean | 2924 | 1.607543 | 1.86 | 0.228801 | 100.00% | 73.25% |
| logit_mean | 4591 | 0.283874 | 1.02 | 0.774796 | 100.00% | 98.82% |
| plurality | 1773 | — | — | — | 98.80% | 52.64% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-03000/noise1`.

## Step3000 / init0

Members: i0-n0, i0-n1. Train-selected member: i0-n0.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i0-n0 | 4669 | 0.185363 | 1.00 | 0.832939 | 100.00% | 100.00% |
| i0-n1 | 2087 | 1.654052 | 2.49 | 0.273944 | 99.01% | 57.01% |
| probability_mean | 4655 | 0.609383 | 1.00 | 0.553442 | 100.00% | 99.79% |
| logit_mean | 4628 | 0.302789 | 1.01 | 0.750586 | 100.00% | 99.38% |
| plurality | 3326 | — | — | — | 99.47% | 77.59% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-03000/init0`.

## Step3000 / init1

Members: i1-n0, i1-n1. Train-selected member: i1-n0.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i1-n0 | 3942 | 0.719135 | 1.19 | 0.525865 | 99.96% | 88.38% |
| i1-n1 | 892 | 2.689948 | 4.92 | 0.145574 | 48.03% | 24.83% |
| probability_mean | 3605 | 1.169833 | 1.35 | 0.335720 | 98.83% | 82.71% |
| logit_mean | 4253 | 0.592920 | 1.13 | 0.603648 | 99.86% | 93.43% |
| plurality | 2367 | — | — | — | 73.68% | 55.77% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-03000/init1`.

## Step3000 / init2

Members: i2-n0, i2-n1. Train-selected member: i2-n0.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i2-n0 | 4531 | 0.415802 | 1.04 | 0.687835 | 100.00% | 97.84% |
| i2-n1 | 1503 | 1.698362 | 2.39 | 0.266885 | 97.59% | 48.12% |
| probability_mean | 4245 | 0.790552 | 1.11 | 0.477360 | 100.00% | 93.53% |
| logit_mean | 4529 | 0.391973 | 1.04 | 0.701902 | 100.00% | 97.86% |
| plurality | 3003 | — | — | — | 98.58% | 72.55% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-03000/init2`.

## Step3177 / all_six

Members: i0-n0, i0-n1, i1-n0, i1-n1, i2-n0, i2-n1. Train-selected member: i2-n0.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i0-n0 | 4307 | 0.590949 | 1.10 | 0.587245 | 96.49% | 93.40% |
| i0-n1 | 3182 | 1.018841 | 1.50 | 0.427970 | 99.43% | 76.15% |
| i1-n0 | 4132 | 0.625600 | 1.14 | 0.572564 | 99.57% | 91.18% |
| i1-n1 | 4129 | 0.623310 | 1.14 | 0.571960 | 100.00% | 91.74% |
| i2-n0 | 4612 | 0.316174 | 1.01 | 0.748021 | 100.00% | 99.12% |
| i2-n1 | 2615 | 1.147245 | 1.67 | 0.402394 | 100.00% | 68.37% |
| probability_mean | 4669 | 0.607907 | 1.00 | 0.551692 | 100.00% | 100.00% |
| logit_mean | 4669 | 0.017341 | 1.00 | 0.982841 | 100.00% | 100.00% |
| plurality | 4666 | — | — | — | 100.00% | 99.95% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-03177/all_six`.

## Step3177 / noise0

Members: i0-n0, i1-n0, i2-n0. Train-selected member: i2-n0.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i0-n0 | 4307 | 0.590949 | 1.10 | 0.587245 | 96.49% | 93.40% |
| i1-n0 | 4132 | 0.625600 | 1.14 | 0.572564 | 99.57% | 91.18% |
| i2-n0 | 4612 | 0.316174 | 1.01 | 0.748021 | 100.00% | 99.12% |
| probability_mean | 4669 | 0.465999 | 1.00 | 0.635943 | 100.00% | 100.00% |
| logit_mean | 4669 | 0.034556 | 1.00 | 0.966283 | 100.00% | 100.00% |
| plurality | 4632 | — | — | — | 100.00% | 99.41% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-03177/noise0`.

## Step3177 / noise1

Members: i0-n1, i1-n1, i2-n1. Train-selected member: i2-n1.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i0-n1 | 3182 | 1.018841 | 1.50 | 0.427970 | 99.43% | 76.15% |
| i1-n1 | 4129 | 0.623310 | 1.14 | 0.571960 | 100.00% | 91.74% |
| i2-n1 | 2615 | 1.147245 | 1.67 | 0.402394 | 100.00% | 68.37% |
| probability_mean | 4583 | 0.800299 | 1.02 | 0.467441 | 100.00% | 98.69% |
| logit_mean | 4669 | 0.043528 | 1.00 | 0.957957 | 100.00% | 100.00% |
| plurality | 4041 | — | — | — | 100.00% | 90.42% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-03177/noise1`.

## Step3177 / init0

Members: i0-n0, i0-n1. Train-selected member: i0-n1.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i0-n0 | 4307 | 0.590949 | 1.10 | 0.587245 | 96.49% | 93.40% |
| i0-n1 | 3182 | 1.018841 | 1.50 | 0.427970 | 99.43% | 76.15% |
| probability_mean | 4368 | 0.725047 | 1.07 | 0.507607 | 99.93% | 95.37% |
| logit_mean | 4498 | 0.333856 | 1.04 | 0.730543 | 100.00% | 97.36% |
| plurality | 3741 | — | — | — | 97.66% | 84.68% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-03177/init0`.

## Step3177 / init1

Members: i1-n0, i1-n1. Train-selected member: i1-n1.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i1-n0 | 4132 | 0.625600 | 1.14 | 0.572564 | 99.57% | 91.18% |
| i1-n1 | 4129 | 0.623310 | 1.14 | 0.571960 | 100.00% | 91.74% |
| probability_mean | 4606 | 0.590598 | 1.02 | 0.572262 | 100.00% | 99.04% |
| logit_mean | 4667 | 0.130240 | 1.00 | 0.881902 | 100.00% | 99.97% |
| plurality | 4136 | — | — | — | 99.75% | 91.57% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-03177/init1`.

## Step3177 / init2

Members: i2-n0, i2-n1. Train-selected member: i2-n0.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i2-n0 | 4612 | 0.316174 | 1.01 | 0.748021 | 100.00% | 99.12% |
| i2-n1 | 2615 | 1.147245 | 1.67 | 0.402394 | 100.00% | 68.37% |
| probability_mean | 4527 | 0.589398 | 1.03 | 0.575207 | 100.00% | 97.84% |
| logit_mean | 4644 | 0.214047 | 1.01 | 0.818933 | 100.00% | 99.62% |
| plurality | 3654 | — | — | — | 100.00% | 84.48% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-03177/init2`.

## Step3200 / all_six

Members: i0-n0, i0-n1, i1-n0, i1-n1, i2-n0, i2-n1. Train-selected member: i2-n0.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i0-n0 | 4669 | 0.202647 | 1.00 | 0.819107 | 100.00% | 100.00% |
| i0-n1 | 3106 | 1.097008 | 1.63 | 0.411170 | 96.46% | 73.49% |
| i1-n0 | 4325 | 0.537096 | 1.09 | 0.614928 | 99.89% | 94.44% |
| i1-n1 | 987 | 3.406409 | 10.31 | 0.155679 | 42.90% | 26.19% |
| i2-n0 | 4651 | 0.232500 | 1.00 | 0.803335 | 100.00% | 99.73% |
| i2-n1 | 1560 | 1.917254 | 3.11 | 0.240286 | 83.78% | 45.52% |
| probability_mean | 4669 | 0.689009 | 1.00 | 0.507417 | 100.00% | 100.00% |
| logit_mean | 4669 | 0.035114 | 1.00 | 0.965765 | 100.00% | 100.00% |
| plurality | 4668 | — | — | — | 100.00% | 99.98% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-03200/all_six`.

## Step3200 / noise0

Members: i0-n0, i1-n0, i2-n0. Train-selected member: i2-n0.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i0-n0 | 4669 | 0.202647 | 1.00 | 0.819107 | 100.00% | 100.00% |
| i1-n0 | 4325 | 0.537096 | 1.09 | 0.614928 | 99.89% | 94.44% |
| i2-n0 | 4651 | 0.232500 | 1.00 | 0.803335 | 100.00% | 99.73% |
| probability_mean | 4669 | 0.298439 | 1.00 | 0.745790 | 100.00% | 100.00% |
| logit_mean | 4669 | 0.021235 | 1.00 | 0.979052 | 100.00% | 100.00% |
| plurality | 4668 | — | — | — | 100.00% | 99.98% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-03200/noise0`.

## Step3200 / noise1

Members: i0-n1, i1-n1, i2-n1. Train-selected member: i0-n1.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i0-n1 | 3106 | 1.097008 | 1.63 | 0.411170 | 96.46% | 73.49% |
| i1-n1 | 987 | 3.406409 | 10.31 | 0.155679 | 42.90% | 26.19% |
| i2-n1 | 1560 | 1.917254 | 3.11 | 0.240286 | 83.78% | 45.52% |
| probability_mean | 3523 | 1.430689 | 1.53 | 0.269045 | 99.43% | 81.90% |
| logit_mean | 4541 | 0.354773 | 1.03 | 0.747787 | 99.96% | 98.03% |
| plurality | 2296 | — | — | — | 92.14% | 59.69% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-03200/noise1`.

## Step3200 / init0

Members: i0-n0, i0-n1. Train-selected member: i0-n0.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i0-n0 | 4669 | 0.202647 | 1.00 | 0.819107 | 100.00% | 100.00% |
| i0-n1 | 3106 | 1.097008 | 1.63 | 0.411170 | 96.46% | 73.49% |
| probability_mean | 4656 | 0.504493 | 1.00 | 0.615138 | 100.00% | 99.80% |
| logit_mean | 4642 | 0.248914 | 1.01 | 0.787844 | 100.00% | 99.59% |
| plurality | 3848 | — | — | — | 98.19% | 85.97% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-03200/init0`.

## Step3200 / init1

Members: i1-n0, i1-n1. Train-selected member: i1-n0.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i1-n0 | 4325 | 0.537096 | 1.09 | 0.614928 | 99.89% | 94.44% |
| i1-n1 | 987 | 3.406409 | 10.31 | 0.155679 | 42.90% | 26.19% |
| probability_mean | 4047 | 1.013549 | 1.17 | 0.385303 | 99.01% | 89.61% |
| logit_mean | 4428 | 0.522855 | 1.08 | 0.643581 | 99.47% | 95.99% |
| plurality | 2508 | — | — | — | 69.54% | 57.30% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-03200/init1`.

## Step3200 / init2

Members: i2-n0, i2-n1. Train-selected member: i2-n0.

| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |
|---|---:|---:|---:|---:|---:|---:|
| i2-n0 | 4651 | 0.232500 | 1.00 | 0.803335 | 100.00% | 99.73% |
| i2-n1 | 1560 | 1.917254 | 3.11 | 0.240286 | 83.78% | 45.52% |
| probability_mean | 4560 | 0.679703 | 1.02 | 0.521810 | 100.00% | 98.33% |
| logit_mean | 4613 | 0.326316 | 1.01 | 0.741759 | 100.00% | 99.15% |
| plurality | 3069 | — | — | — | 91.89% | 72.00% |

All per-equation distributions, member ranks and transitions: `runs/ensemble-diagnostic-20261011/reports/step-03200/init2`.

U is the fixed4669 ordered equations with neither orientation trained. Reversed pairs are dependent; group counts are saved without independent-example significance claims. The top1 union is not an upper bound on probability ensembling. Ensemble inference evaluates every member; it is not a single improved transformer.

Plurality ties use lowest vocabulary ID and may therefore predict a nonresidue token. No accuracy-dependent stopping, member selection, temperature or weight fitting.
