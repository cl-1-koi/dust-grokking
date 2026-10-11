# Frozen prediction stability diagnostic

Selection endpoint: step 3177. Future retention endpoint: step 3200. This is exploratory analysis of observed pilot replicas, frozen before this diagnostic scored labels or opened future predictions; it is not an independently blinded validation study.

Cross-replica agreement selected high-precision pseudo-labels in these completed runs. Every replica’s top 512 agreement-ranked labels was correct at step 3177. Temporal agreement also selected persistent errors, and correct selected labels did not protect a subsequently unstable individual model from regression. The fixed uniform six-model raw-logit ensemble was correct on all 4,669 unseen pairs at both endpoints.

## Selection and evidence

The original U contains 4,669 ordered input pairs whose reversed orientation is also held out. The frozen follow-up plan assigns 2,338 ordered candidates in 1,184 unordered groups to the pseudo pool, and 2,331 ordered pairs in 1,183 disjoint groups to clean evaluation. Selected sets contain one orientation per group and exclude every clean-evaluation pair and its reverse.

The selectors receive only input coordinates, pool/evaluation IDs, valid residue token IDs, predicted tokens, and label-free confidence/agreement features. Confidence uses the top1–top2 logit gap over all 239 vocabulary entries. Temporal agreement uses snapshots 2700, 2800, 2900, 3000, 3100 and 3177. Replica agreement compares the six current top1 predictions, then breaks ties with temporal agreement and confidence. Final ties use ascending global pair ID. The ensemble uses a fixed uniform raw-logit mean, ranked by current vote agreement then mean-logit gap. Current predicted top1 must be a valid residue token; no arithmetic or correctness filter enters selection.

All 57 sets (six replicas × three rules × three counts, plus three ensemble sets) were committed with their pseudo-label token IDs before evaluation. Counts 64, 256 and 512 cover 5.41%, 21.62% and 43.24% of candidate unordered groups. Each rule’s smaller set is a prefix of its larger set. No requested count was reduced.

Selection manifest SHA256: `3c60bb6d956403a09455f36ec0f9d0a3d35e1f3718295e057c478d83e9028163`. The commit record is [selection_commit.json](results/prediction-stability/selection_commit.json); it records the commitment time and explicitly states that labels and future predictions had not been opened.

## All replicas, top 512

Each entry shows correct pseudo-labels at 3177 → correct predictions by that same replica at 3200 on the fixed selected inputs. The latter evaluates the continuing model, not a change to the frozen pseudo-labels.

| Replica | Confidence | Temporal → confidence | Replica → temporal → confidence |
|---|---:|---:|---:|
| i0-n0 | 508/512 → 512/512 | 512/512 → 512/512 | 512/512 → 512/512 |
| i0-n1 | 422/512 → 399/512 | 368/512 → 376/512 | 512/512 → 399/512 |
| i1-n0 | 509/512 → 509/512 | 495/512 → 497/512 | 512/512 → 498/512 |
| i1-n1 | 504/512 → 156/512 | 467/512 → 152/512 | 512/512 → 130/512 |
| i2-n0 | 512/512 → 512/512 | 512/512 → 512/512 | 512/512 → 512/512 |
| i2-n1 | 348/512 → 241/512 | 242/512 → 197/512 | 512/512 → 282/512 |

All six agreement-ranked top-512 sets had 100% observed precision at 3177; they overlap and must not be pooled as 3,072 independent successes. Temporal-first ranking reduced precision relative to confidence in i0-n1, i1-n0, i1-n1 and i2-n1 at this count. Confidence and temporal agreement are not correctness guarantees.

## Unstable i1-n1, top 256

| Rule | Correct frozen labels at 3177 | Labels retained at 3200 | Correct predictions at 3200 | Stable wrong labels |
|---|---:|---:|---:|---:|
| confidence | 252/256 (98.44%) | 104/256 (40.62%) | 102/256 (39.84%) | 2/4 originally wrong |
| temporal | 242/256 (94.53%) | 89/256 (34.77%) | 86/256 (33.59%) | 4/14 originally wrong |
| replica | 256/256 (100.00%) | 71/256 (27.73%) | 71/256 (27.73%) | none; no original errors |

The confidence and replica-ranked sets share 67/256 selected ordered IDs. Both use i1-n1’s own predicted labels, with peer predictions used only for ranking. Replica-first selection spans 90/97 residue classes; confidence spans 86/97. Full class counts are retained in results.json.

The frozen labels remain unchanged and retain their measured 3177 correctness. The low 3200 accuracy says the model changed its predictions; it does not make the archived labels wrong. This diagnostic contains no pseudo-label training and cannot show that appending them prevents instability.

On the entire U, i1-n1 falls from 4129/4669 (88.43%) to 987/4669 (21.14%). It loses 3,195 previously correct predictions, corrects 53 previously wrong predictions, and keeps 162 wrong labels unchanged out of 540 original errors (30.00%).

## Stable errors

For i1-n1, only 84/4,669 U inputs have identical top1 predictions across all six historical snapshots. Of these, 73/84 are correct and 11/84 are wrong at 3177; 8/11 wrong labels persist through 3200. Thus complete temporal agreement can preserve wrong predictions. By comparison, all-six current replica unanimity covers 1,324/4,669 ordered U inputs and all 1,324 are correct at 3177, although only 290/1,324 remain correct in i1-n1 at 3200.

Examples below were selected by the temporal top-512 rule and have the same wrong label at 3177 and 3200. They are observed stable errors, not intrinsically hard equations.

| Input | True residue | Frozen / future residue | Six-snapshot agreement |
|---|---:|---:|---:|
| 44 + 66 (mod 97) | 13 | 94 / 94 | 1.000 |
| 11 + 29 (mod 97) | 40 | 56 / 56 | 0.833 |
| 55 + 35 (mod 97) | 90 | 74 / 74 | 0.833 |
| 33 + 89 (mod 97) | 25 | 41 / 41 | 0.833 |
| 11 + 79 (mod 97) | 90 | 74 / 74 | 0.833 |
| 79 + 58 (mod 97) | 40 | 24 / 24 | 0.833 |

Temporal top-512 i1-n1 has 45 incorrect frozen labels; 15/45 retain the same wrong label, 3/45 become correct, and 27/45 change to another wrong label at 3200. Confidence top-512 has 8 errors, with 4/8 retained wrong. Replica top-512 has no observed errors at 3177.

## Fixed ensemble and cost

The uniform six-member raw-logit mean scores 4,669/4,669 on U at 3177 and again at 3200. Each of its selected top-64, top-256 and top-512 sets is fully correct and fully retained. Individual replicas at 3177 score at most 4,612/4,669; at 3200 i0-n0 alone is fully correct. These are fixed-endpoint observations and do not establish continued future stability.

The ensemble requires six trained models and six current forward evaluations per new input. Temporal selection requires six historical states of one model; replica-first selection requires six current states plus five earlier states of the named model. The replicas cross three initializations with two matched perturbation seeds and share one split. Their training and inference cost is greater than a single-model teacher. This diagnostic reused cached arrays and took 5.552 seconds under an external 300-second guard with at most two compute threads and a 5-GiB disk guard. It performed zero new model forwards and zero optimizer updates.

## Artifacts and limits

Nine tests pass, including label/future exclusion, ranking and ties, reverse-group exclusion, explicit insufficient-pool handling, commitment integrity and exact transition formulas. Independent recounts verified all 57 selected sets and all 15,808 example CSV rows, including nesting, counts, label correctness, transition partitions and the sealed manifest hash. Every source artifact stayed unchanged.

- [selection_manifest.json](results/prediction-stability/selection_manifest.json): frozen IDs, token labels, label-free selection features and source checksums.
- [results.json](results/prediction-stability/results.json), [summary.csv](results/prediction-stability/summary.csv), [selected_examples.csv](results/prediction-stability/selected_examples.csv): post-selection scores, distributions and every selected example.
- [tests_and_runtime.json](results/prediction-stability/tests_and_runtime.json), [independent_results_checks.json](results/prediction-stability/independent_results_checks.json), [completion.json](results/prediction-stability/completion.json): correctness and guard evidence.
- [prediction_stability.py](src/prediction_stability.py), [test_prediction_stability.py](tests/test_prediction_stability.py), [guard_prediction_stability.sh](scripts/guard_prediction_stability.sh): implementation, tests and bounded runner.

Ordered reversal pairs, overlapping selections, shared training data, matched perturbation seeds and temporal model states create dependence. No independent-binomial significance or general correctness guarantee is inferred. Selection was label-free at the specified endpoints, but the question and diagnostic were chosen after observed pilot behavior. This is neither an MDL measurement nor evidence that pseudo-label training helps.

Re-running the full historical diagnostic requires the original cached per-step top1 archives for all six replicas and the sealed 3177/3200 full-distribution archives listed in the manifest/results checksums, plus the frozen follow-up plan. The public eleven-weight bundle alone is insufficient; this report does not claim those archives are included in that bundle. The sealed selection and scores can be inspected without training or hidden label selection.
