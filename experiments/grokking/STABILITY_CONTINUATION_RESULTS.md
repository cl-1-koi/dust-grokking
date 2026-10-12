# Frozen-prediction continuation results

Completed 2026-10-11 22:35 UTC; independently audited 2026-10-12 03:22 UTC.

Exploratory continuations from the same pure-Dust i1-n1 update3177 checkpoint.
Every arm retains all2823 original labeled equations. Confidence/stability append256
frozen model predictions; stability ranks replica agreement, then temporal agreement.
Primary evaluation is2331 reserved pairs; both orientations of every candidate
pseudo-pair group are excluded. No backprop or optimizer reset.

| Arm | State | Additional updates | Latest clean accuracy | Minimum recorded | Mean at100–200 |
|---|---|---:|---:|---:|---:|
| baseline | HORIZON_COMPLETE | 200 | 93.65% | 8.07% | 77.80% |
| confidence | HORIZON_COMPLETE | 200 | 92.92% | 26.30% | 84.37% |
| stability | HORIZON_COMPLETE | 200 | 94.25% | 21.71% | 81.63% |

Identical initial model/Adam/RNG across 3 collected arms: True.
Queue exit: 0.

All three arms completed all 200 updates and all 22 planned evaluations.
Minimum includes update 0; late-window means cover all eleven checks from 100 through 200.
A budget stop is not evidence of failure. Compare common completed evaluation
steps if horizons differ. Augmentation changes objective weights and noise draw
trajectories; confidence/stability have matched sizes. Ordinary all-held accuracy
is contaminated after augmentation and is not the primary outcome.

Collected artifacts: `results/stability-continuation`.

The augmentation arms have higher observed late-window means and less severe
sampled minima than the unchanged-data continuation. They still suffer large
regressions, and final accuracies differ little. Confidence selection has the
higher late-window mean despite four incorrect pseudo-labels; agreement selection
has the highest final accuracy. Thus perfect measured pseudo-label precision
does not establish a superior optimization trajectory in this single pilot.
This is one parent checkpoint and one starting noise state per arm; it cannot
establish a reliable treatment effect or justify choosing a winner.

The original data remains intact. Augmentation assigns 8.31% of mean-loss weight
to pseudo-labels and changes subsequent noise draws through batch shape. The
reserved evaluation contains 2,331 pairs disjoint from the pseudo pool in both
orientations. Every arm starts at 87.22% on this reserved set. All optimizer
updates remain zeroth order, with zero backprop calls inherited or performed.

An independent audit verified all 66 saved per-example evaluations and their
retention/recovery counts, equal initial model/Adam/RNG hashes, completed
predeclared summary windows, and unchanged inputs. The baseline at +23 updates
exactly reproduces the original step3200 predictions on the reserved set.
Elapsed times were 32.10, 34.80 and 38.86 minutes, respectively, on eight CPU
threads. See `results/stability-continuation/audit-20261012.json`.
