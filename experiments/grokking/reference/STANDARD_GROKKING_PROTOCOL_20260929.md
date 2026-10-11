# Standard-transformer grokking positive control, before code

User redirect 2026-09-29. Replace the previous tied GAB calibration with the
released Power et al. transformer architecture. Four calibration fits, one
fixed seed9278700: modular addition mod97, pair-train fractions30/50/70%, and
S5 composition at50%. The conditional TTT follow-up uses three seeds as
explicitly requested; no seed is selected for successful calibration.

Architecture: two DISTINCT decoder blocks, width128, four attention heads,
head width32, FF width512, ReLU, causal attention, sinusoidal positions,
post-residual LayerNorm epsilon1e-5, no dropout, no biases in attention/FF/
unembedding matrices, learned token embeddings. Use the official Torch
implementation as the numerical reference and source of initialization.
No GAB, tied recurrent blocks, code projection or auxiliary loss. Reference:
https://github.com/openai/grok and https://arxiv.org/pdf/2201.02177 .

AdamW lr.001, decay1 on all parameters, beta(.9,.98), epsilon1e-8, ten-step
linear warmup then constant lr; no clipping (released training defaults).
Full-batch updates are the user's explicit convention: this differs from the
paper's automatic minibatch size. Do not silently substitute minibatches.
Float32 highest matmul precision, no mixed precision unless separately audited
and declared. 100,000 updates; evaluate full train/held sets every500.

Tokenization follows the released equation format: [EOS,a,operator,b,=]
predicts the product, and the teacher-forced product predicts EOS; train CE
on both right-hand-side targets. Report product accuracy as primary and full
RHS accuracy as a secondary check. Include the released shared vocabulary
(arithmetic symbols, residues and permutation IDs) and default initialization.
Use p97 for addition, matching the requested and released modulus.
S5 follows the released composition convention and is audited independently.
Uniform nested shuffled ordered-pair train splits, paired initialization for
the three addition fractions. No held-out gradient, tuning or stopping.

Before training: Torch/JAX forward, loss, gradient and identical-gradient AdamW
audits; causal future-token invariance; data algebra and split audits; a tiny
batch overfit check whose weights are discarded. Compiled full-batch updates
may accumulate exact sub-batch sums before ONE optimizer step if required by
memory; that is still full-batch optimization. Measure complete-step throughput
and checkpoint cost before promising a wall time. A two-hour independent GPU
deadline includes compilation/audits, with no restart budget reset.

Primary readouts: first train>=99%, train100%, held>=99%, held100%; held-out
accuracy at train saturation; delayed jump timing, final/maximum accuracy,
NLL and norm. Retain 500-update resolution and three-check persistence.
Operational modular-addition gate: at least one fraction reaches held99 for
three checks, with a period after train99 during which held accuracy remains
below90%, and at least one500-update evaluation interval between train99 and
held99. This is the sampling-resolution minimum, not a universal grokking timescale.
Report every fraction; simultaneous comprehension is not silently labelled
grokking. If no modular-addition delayed transition occurs, STOP the dependent
TTT child and report the discrepancy, including fitting and numerical audits.
S5 is a separate reference; it does not substitute for the modular gate.

If the gate passes, start a separate budgeted TTT child: same two-layer width128
causal decoder and optimizer (including decay1 and full-batch), fractions60/80%
of all765 orbits using the existing fixed153-orbit held-out partition, seeds
9278700/1/2, 100k updates. Use all eight training views with their existing
multiplicities, no cross-orbit leakage. Input is nine seat-relative square
symbols followed by a readout symbol; replace only the task head with value
CE and nine optimal-membership sigmoid/BCE outputs (mean BCE plus value CE,
as in the TTT ladder). A shared final readout can see every square causally.
Report train/held value, membership, exact-set and joint accuracy, per-seed
delays and curves; no claim of equivariance from multiplication accuracy alone.
Freeze a task-adaptation protocol before writing that child.

Resource rule: C1 has priority because the user said start it now. Avoid GPU
contention that makes either child miss its budget; serialize if measurement
requires it. If full100k calibration cannot fit, declare the measured common
horizon cut before training, mark the result budget-limited, and do not treat
it as evidence that the standard recipe fails. No architectural shrinking or
optimizer changes to manufacture a positive control. Report within4 hours of
each child's launch; CPU sealing under29 minutes, at most half host cores.

Prelaunch clarification 2026-09-29T18:26:33.294028+00:00: the delayed gate requires the below90% held-out interval to occur before the first held99 crossing; the minimum lag is one500-update evaluation interval. No scientific calibration training has started. The CPU-only preflight passed after correcting a shared-buffer audit snapshot, without changing the model or tolerances.

Execution diagnostic, 2026-09-29 18:33 UTC: the initial float32 CUDA pilot
projected 4.19 hours for 100k full-batch updates, selecting a 42,500 common
horizon under the existing two-hour deadline. Before accepting this cut, test
XLA's compensated TF32_TF32_F32_X3 matrix-product algorithm, with float32
parameters, activations, accumulators and optimizer state. It may replace the
matrix-product implementation only if the existing port/causality/padding/
overfit audits pass unchanged and identical-state full-batch one-update
parameters agree with the original within atol=3e-6, rtol=1e-5. Record gradient
and parameter errors and measured throughput. No data, model, objective,
optimizer hyperparameter or RNG change. Pause only this child's trainer during
the discarded benchmark; its original budget deadline continues. If accepted,
resume from the durable checkpoint and recompute the feasible common horizon;
retain original curves and explicitly record the execution switch.

Execution diagnostic outcome 2026-09-29T18:35:18.275226+00:00: compensated TF32 rejected under the predeclared full-batch update tolerance; original highest-precision float32 training resumed unchanged. Requested100k cannot fit at measured throughput; the declared42,500-update common horizon remains budget-limited and may be truncated earlier by the unchanged deadline. No failed100k reproduction will be inferred.

Pending execution recommendation, 2026-09-29T19:00:34.540910+00:00: all four fits have crossed99% held-out. Asked whether to stop after every fit has achieved three consecutive held99 checks and proceed to TTT, instead of retaining the full42,500 budget-reduced tail. The user standing15-minute recommendation rule applies at19:14 UTC absent a reply. This is a post-result resource-allocation change, not a predeclared scientific stop. Per-fit qualifying intervals need not coincide; all later regressions and endpoint accuracies remain reported. Until adopted, training continues unchanged.

Second execution diagnostic before any implementation substitution, 2026-09-29T19:03:03.772916+00:00: benchmark dense one-hot multiplication versus embedding gather, both with highest-precision float32 throughout. These implement the same embedding map but different floating-point reduction orders in its gradient. The full-batch identical-state one-update tolerance remains atol3e-6/rtol1e-5; require unchanged forward/loss/gradient reference checks and a material throughput improvement before considering a switch. Discard benchmark weights; original budget deadline continues during any pause. This diagnostic does not adopt the pending qualification-stop recommendation.

Dense embedding diagnostic outcome 2026-09-29T19:04:38.471701+00:00: full-batch update agreement passed (max3.73e-9), but runtime increased from.1501 to.1562 seconds/update. Reject the change for calibration on throughput grounds; original gather training resumed unchanged.

Execution closure 2026-09-29T19:14:01.249086+00:00: RECOMMENDATION_ADOPTED_AFTER_RESPONSE_WINDOW; stopped at durable update15500 after every fit attained three consecutive held99 checks. This replaces the pending42,500-step tail; requested100k was not completed. Preserved all curves, params/moments, regressions, and the original budget anchor. At most500 unevaluated updates after the durable checkpoint were discarded.
