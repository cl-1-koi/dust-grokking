# CPU Dust modular-addition protocol

Frozen before implementation and scientific training, 2026-10-10 UTC.

## Amendment, 2026-10-11 20:15 UTC — prediction stability diagnostic

Before implementing/scoring this diagnostic, fix the prediction-selection
rules below. This is exploratory analysis of already observed pilot replicas,
not an independently blinded validation study. It does not alter completed
seed training or the six running subset/reconstruction arms.

Question: do prediction agreement across time and across independently trained
replicas identify accurate pseudo-labels better than current-model confidence,
and do these labels survive the observed late update instability?
Selection endpoint is3177. Future3200 predictions are read only AFTER selected
indices and pseudo-labels are frozen, and only for retention evaluation.
Use original fixed unseen-pair U and the pre-existing follow-up plan's pseudo
pool/clean_eval partition, grouped by unordered operand pairs. Never select
from clean_eval, never train an opposite ordering of a clean_eval pair.
Diagnostic historical top1 snapshots are2700,2800,2900,3000,3100,3177;
use only predicted_token, never saved correctness/true-class margin/CE.
Current3177 logits support confidence (top1 minus top2 across all239 tokens).

For each of the six replicas, compare: (a) current confidence alone;
(b) fraction of historical predictions equal to that replica's current top1,
then current confidence; (c) fraction of the six current replica top1s agreeing
with that replica's current top1, then temporal agreement, then confidence.
All rankings are descending, with global pair index ascending as final tie
breaker. Require current top1 to be a valid residue token. Select64,256,512
examples, at most one ordering per unordered group. Return an explicit
insufficient-pool result rather than silently changing a requested count.
Also evaluate the fixed uniform six-member logit-mean labels with current
replica vote agreement, then mean-logit confidence; this is an ensemble teacher
whose inference/training cost must be disclosed. No arithmetic consistency
tests, target labels, held-label tuning, or future snapshots enter selection.

Write a selection-only manifest and its hash before opening evaluation labels
or3200 predictions. Thereafter report pseudo-label correctness, coverage,
class distribution, future prediction retention/correctness, and results by
replica and rule. Correctness is post-selection auditing, never a selection
input or an inferred guarantee of correctness. Pairs are dependent through
reversal and models share a split; avoid independent-binomial significance.
No new optimizer updates occur in this first diagnostic. Coordinator at most
two compute threads, external300s guard and5GiB disk guard for artifacts.
Any later self-training continuation requires a separate pre-run amendment,
an unchanged-data continuation control and a clean evaluation partition.

## Amendment, 2026-10-11 20:22 UTC — bounded pseudo-label continuation

Before looking at the stability diagnostic's correctness results, fix a small
continuation pilot motivated by the observed i1-n1 regression3177→3200. Use
its original full model/Adam/perturbation-RNG checkpoint at3177 in every arm.
This is exploratory checkpoint selection after observing instability, not a
fresh-seed efficacy result. No backprop, optimizer reset, altered decay or
learning-rate adjustment is permitted. Use the validated CpuDust path,
eight CPU threads, original full-batch2823 true training equations.

Three arms: unchanged-data control; append256 current-confidence-selected
pseudo-labels; append256 replica-agreement/temporal-agreement-selected labels.
Selections come from the above3177 manifest for i1-n1, frozen before labels
and3200 predictions are read. Use that model's predicted labels in both
augmentation arms, not the correct answers or a new ensemble teacher.
Temporal-only and logit-ensemble selectors remain diagnostic at this stage.

Augmentation is literal dataset augmentation: concatenate the original ordered
training indices and the256 frozen pseudo indices; normalize once by all
supervised tokens as the existing updater does. Every original true example
remains present. Thus the pseudo part has256/(2823+256)=8.31% of mean-loss
weight; document the consequent dilution of original-data gradients. The
unchanged-data control retains the original objective. Different batch
shapes change subsequent perturbation draws; the starting RNG is identical,
but trajectories do not share identical noise after augmentation. Confidence
and stability arms have identical example/evaluation budgets. Log all counts.
All six inputs' terminal teacher-forced answer tokens AND their answer labels
must use the pseudo-label in augmented examples; EOS targets remain EOS.

Primary evaluation is the pre-existing reserved clean_eval set, excluding both
orientations of all pseudo-pool groups. Report every10updates including0 and
the previously observed23-update regression point; horizon200additional
updates. Report final, minimum across the evaluation trajectory, and the
mean over evaluations from100through200 (all mandatory, no best-checkpoint
selection). Also report original true-training fit, pseudo-target fit,
per-example retention and recovery on clean_eval, answer CE and runtime.
Ordinary all-held accuracy becomes contaminated after augmentation and is
secondary, explicitly marked. This pilot cannot prove selection correctness
or general efficacy from a single unstable teacher.

Run three arms sequentially on worker4 only after its owned pseudo_only
follow-up exits cleanly. Never interrupt an existing process. Waiting bounded
to90minutes from queue launch; each arm external3600seconds/internal3540,
with200-update cap and5GiB disk guard; outer queue16500seconds. Refuse
nonzero/failed predecessor and pre-existing output directories. All checkpoint
and selection files are copied into a new immutable input directory before
launch, with hashes recorded. If an arm stops early, compare only common
completed updates and report the budget stop separately. No automatic retries
or horizon extensions.

Question: can forward-only, zeroth-order Dust learn and subsequently grok
modular addition on the released Power et al. transformer? This is a source
adaptation experiment, not a reproduction of Dust's language-model benchmark.
Official Dust is pinned at b20f7c03eac630b6441ba6f254128c1761dc45ad.
The references are the vendored source and
reference/STANDARD_GROKKING_PROTOCOL_20260929.md. Research arriving from Gemini
will be checked against these sources; implementation need not wait for it.

Motivation clarification, 2026-10-10 23:32 UTC: the user reports that EGGROLL
failed in their experiment, and discussion on Twitter about zeroth-order LLM
pretraining motivated testing Dust next. Accept the reported EGGROLL outcome
without asserting its cause. This CPU modular-addition experiment tests Dust's
update signal and delayed held-out generalization on the stated task; it does
not establish performance on LLM pretraining. No experimental settings change.

## Fixed task and control

Only addition modulo 97, all 9,409 ordered input pairs. Vocabulary is exactly
the reference shared vocabulary: EOS, equals, sorted released operators,
97 residues and 120 permutation IDs. Inputs are `EOS a + b = c` (six tokens);
only logits at positions 4 and 5 receive targets c and EOS. Product accuracy
is primary; exact two-token RHS accuracy and mean RHS CE are secondary.
No reduced three-token surrogate is used for scientific training.
Train split is first round(0.3*9409)=2823 pairs of NumPy RandomState(0)
permutation, held-out is the remaining 6586. Save exact pair indices, tokens,
vocabulary, source hashes and split hashes. Model seed 9278700; perturbation
seed 9278703; initialization and all data are paired across Dust populations
and backprop control. Held-out examples never enter estimates or optimizer.

Two distinct decoder blocks, width128, four heads of width32, FF width512,
ReLU, sinusoidal positions, post-residual LayerNorm epsilon1e-5, no dropout.
Use released Torch initialization. Attention/FF/unembedding matrices have no
bias; each of four LayerNorms has a trainable gain AND bias. No parameters
are frozen or silently omitted. Float32 CPU execution throughout.
AdamW lr0.001, betas(0.9,0.98), epsilon1e-8, weight decay1 on ALL parameters,
10-step warmup starting with step0 lr0, then constant; no clipping.
Full training batch per update; exact sum accumulation across subbatches is
permitted, then exactly ONE optimizer step. Identical optimizer for Dust and
control. AdamW differs from Dust's tuned SGD: it preserves the known grokking
recipe and isolates the gradient estimator in paired comparisons.

## Estimator implementation and explicit adaptations

Implement the released transformer forward with a clean input/output cache,
numerically audit it against the unchanged released architecture, and fuse
head projections only as a CPU execution optimization. Use independent
Gaussian activation perturbations for modules (no weight perturbation for
linear weights), one-sided clean-minus-perturbed loss rewards, exact local
input/error outer products and embedding scatter-add. Linear gradients use
loss INCREASE times noise / sigma (equivalently minus loss drop); if jitter
is epsilon=sigma*a, this is loss increase times epsilon / sigma^2.
Divide by exactly 2*training_pairs supervised tokens, once after accumulation.

Direct sites: embedding, each block's attention writer and FF input/output
linear maps, attention hubs before output projection, and LayerNorm outputs.
Clean caches are never overwritten by perturbation passes. Module noises are
independent; separate streams are used for estimates rather than globally
perturbing all weights. Attention Q/K/V estimates use only their block's
clean cached Q/K/V and direct hub target, scoring head-local attention output
changes. Q gets its own output-position score; K and V sum later causal
scores with gamma=1 for this short task. Unlike the Dust model, released
attention has neither rotary nor RMS-normalized Q/K, value gates/embeddings,
learned residual mixing scalars, squared ReLU, nor logit soft cap. These are
absent by architecture choice, not omitted gradients.

Sparse-supervision adaptation: direct estimates aggregate causal FUTURE
supervised loss increments, with gamma=1, at each perturbed token. The source
uses same-token rewards; that would erase direct operand-position credit.
For the last block's purely tokenwise paths causal aggregation adds noise
but no expected gradient. Compare same-token rewards as a diagnostic only.
Do not set gamma=0 universally. Queries of an earlier block can affect later
supervised tokens through the following block, mediated by the hub targets.

Keep source chunk-wise draw centering and head batch centering, but correct
their finite-sample shrinkage by n/(n-1) or B/(B-1) respectively (no centering
at sample count1). Record this explicit unbiasedness correction, and audit
uncorrected source behavior. Head uses cached CE, one adaptive vocabulary
slab (vocab below256), no soft cap. LayerNorm output errors reconstruct gain
gradient as sum(error * clean normalized input), bias gradient as sum(error).
These affine estimates need no derivatives of normalization; perturbation
forward passes handle its influence on upstream modules. Gain and bias
share one output-error estimate, consistent with local affine reconstruction.

Default noise scales from source: direct writers/hidden/embedding/norm0.2;
hub0.2 in first block and0.4 in second; attention internals and head0.05.
Noise bias is measured at smaller scales before selecting pilot scales.
Direct population labels are declared as per-site draws P, not the original
8-layer total-allocation population: total direct/hub/head/local counts are
reported separately. Start ladder P=4,16,64; diagnostic ladder up to1024/4096
on tiny batches as feasible. Initial equal per-site allocation is an explicit
deviation from Dust's layer-dependent 8-block tuned recipe. No claim that
these settings reproduce the full paper's tuned configuration.

## Required preflight and interpretation

Before pilots: data algebra/split/disjointness checks, released forward/loss
and parameter-gradient agreement, future-token invariance; deterministic
sign and sigma^2 checks on a linear quadratic example; batch/valid-token
normalization; clean-cache preservation; every-parameter coverage including
all attention heads and LayerNorm biases; sparse operand activation credit;
local Q/K/V alignment with exact diagnostic targets; full estimated gradient
cosines/norm ratios and per-family values across populations and sigma.
Autograd is allowed only in tests, gradient diagnostics and BP controls.
Scientific Dust training runs entirely under no_grad with parameters marked
requires_grad=False; assignment of estimated parameter.grad to AdamW is
optimizer plumbing and does not perform backpropagation.

Estimates are noisy, so report multiple draws/seeds and parameter-family
alignment, not just one overall cosine dominated by the unembedding. Separate
oracle-target local diagnostics from the fully estimated two-stage recipe.
A direct finite-difference activation baseline is labelled diagnostic, not
the full source-derived Dust estimator. Discard overfit/benchmark weights.
Measure complete-step CPU throughput, evaluation/checkpoint costs and forward
example counts. Establish a paired full-batch BP control that actually learns.

First bounded pilot: P=4/16/64, matched control, new directories on separate
workers. Up to2 hours externally enforced per first-stage job (including
evaluation), maximum2000 Dust updates /10000 BP updates; evaluate at step0
and every25 Dust or100 BP steps, save every evaluation. Throughput measurements
may motivate a documented next-stage horizon/population before launching it.
Keep coordinator at <=2 CPU threads for diagnostics; workers <=8 threads,
inspect existing processes and preserve them. Each tmux job has GNU timeout
with TERM then KILL grace, plus internal deadline; disk guard refuses each
checkpoint when free disk below5GiB. Atomic latest checkpoint includes model,
optimizer, perturbation RNG and counters. Do not overwrite existing runs.

Metrics: train/held-out product accuracy, RHS accuracy, CE, weight norm,
wall clock, updates, supervised-token presentations, population/draw counts,
full forward batch/example counts and attention/head-only evaluation counts.
Record first train99/100 and held99/100; delayed grokking requires held<90
after train>=99 before the first held99 crossing, at least one evaluation
interval lag, and three consecutive held>=99 checks. Resolution of a pilot
does not establish a universal timescale. BP learning, Dust estimator
alignment, memorization and held-out delayed generalization are separate
claims. A deadline or horizon stop is budget-limited, never a failure result.
Leave bounded longer runs active if useful and report their independent caps.

Protocol amendments are timestamped, justified and made before dependent
runs; pilots and all negative/positive observations are retained.

## Amendment, 2026-10-10 22:29 UTC, before Dust pilots

Official paper sections2.2/2.3, https://qlabs.sh/research/dust, explicitly give
future-loss credit and cached suffix execution. Thus gamma1 is an instantiation
of the paper's general credit rule differing from its tuned language-model
direct gamma0 and the minimal source, rather than an unrelated estimator.
Terminal-block direct sites (writer/hidden/ffout/norm1/norm2/hub) have only
tokenwise downstream computation; use same-token reward there, setting known
zero prefix gradients exactly to zero and removing needless future-loss noise.
Embedding and first-block direct sites retain causal future-loss gamma1;
local K/V in BOTH blocks retain gamma1. No blanket gamma0. Initial all-future
diagnostic results are retained and labelled by saved settings; rerun the
diagnostic with this graph-derived variance reduction before interpretation.
Add early evaluations at1,5,10 to the regular25/100 cadence so bounded CPU
pilots expose initial learning without waiting25 expensive updates.

## Next-stage resource amendment, 2026-10-10 22:39 UTC

The cluster has six16-vCPU workers, plus a16-vCPU coordinator kept at modest
load. Measured single-worker P16 update is18.8s at8threads, compared with
~35s at2threads. Full-batch RHS alignment atP16 is~.95; product-only gradient
alignment is materially lower (~.55 in all-future variant), so report both
instead of treating EOS-dominated alignment as residue-learning evidence.
Ten numerical tests, a two-process full-batch training/checkpoint smoke, and
a two-host Gloo audit pass. Worker aliases are absent on workers: use master
PRIVATE IPv4 and GLOO_SOCKET_IFNAME=ens5; do not alter network configurations.

After first-stage P16/P64 have recorded useful early checkpoints, retain all
curves and optimizer/RNG checkpoints and end those coordinator-owned pilot
jobs for documented resource reallocation (not scientific failure). Finish
the bounded full-batch diagnostic, then pool workers3/4/5/6 for globalP16,
localP4 per rank, eight CPU threads each, identical full batch and independent
noise. Average hub errors across ranks before local attention scores, then
average all parameter estimates before optimizer update, as official Dust.
Fresh paired initialization, same data/architecture/optimizer; new directory.
Maximum3000 updates, external21600s/internal21480s (six hours), eval every25
plus early1/5/10, checkpoint5GiB disk guard. Throughput at actual four-host
setting must be logged before any estimate of reachable horizon is accepted.
Keep original P4 pilot and BP control active within their unchanged caps.
The first-stage population comparison is short and cannot decide grokking;
the shared run increases the scientifically useful update horizon. No claim
of failure from any resource reallocation or bound.

## Noise-scale amendment, 2026-10-10 22:50 UTC, before shared launch

The saved BP1800 checkpoint is fully memorized (train CE.03176), without
held-out selection. Full-batch P16 estimates at source-scale sigma.2 have
cosine about-.008/norm ratio45; sigma.02 gives.013/37. Reducing direct sigma
to.002 improves magnitude substantially (ratio8) but alignment is still.072;
sigma.0002/.00002 plateaus near.070. Thus finite-noise effects and substantial
sampling noise both remain; smaller noise alone does not establish correct
late-stage estimates. Initialization at sigma.002 remains well aligned
(full cosine.969; embedding.713/Q.777/K.538/V.900 in narrowed scale audit).
Use direct/embedding/affine sigma.002 and second-block hub.004 for the shared
P16 pilot, preserving local Q/K/V sigma.05 and head.05. A local.005 diagnostic
did not materially improve the trained-state result, so retain source local
scale. These are explicitly tuned transfer scales, not the LM defaults.
No architecture, data, objective, optimizer or backprop-control change.
Retain all source-scale pilots and compare updates/wall time separately.

## Trained population diagnostic, 2026-10-10 23:02 UTC, before launch

Gemini leads are checked in research/gemini_comparison_20261010.md. Their
settings and arbitrary success/failure thresholds do not change active fits.
To distinguish late-stage sampling limits from finite-noise effects, evaluate
the immutable saved BP1800 state on exactly2823 training pairs, masking EOS
only in this diagnostic. Compare Dust P4/P16/P64 against its exact
addition-answer-only backprop gradient, across perturbation seeds9278703/4/5.
Use graph credit, direct sigma.002/last hub.004, local/head.05, chunk256.
All diagnostic optimizer steps use lr0 and discard weights; no held-out
loss enters selection. Record family cosine/norm ratios and actual work.

Run on worker2 with four threads alongside its existing two-thread P4 fit;
this concurrency can affect wall time and is recorded, without altering
training state or limits. Independent external900s guard plus60s KILL grace,
new output path, partial completed rows retained if bounded. Save checkpoint
SHA256 and source hashes before launch. This is a precision diagnostic, not
a training population comparison or a forecast of grokking time.

## Additional trained population diagnostic, 2026-10-10 23:09 UTC

Initial completed P64 replicates at the same saved BP1800 state improve full
addition-gradient cosine to about.325, but retain substantial family noise,
especially Q/K/V. After the P4/P16/P64 diagnostic exits, run P256 at the same
immutable state with the same three perturbation seeds, direct/local scales,
objective, exact batch, and four-thread worker2 allocation. New output and
launch manifest; independent external1200s TERM guard/60s KILL grace.
Do not infer failure or sufficiency from any one family or perturbation seed.
This diagnostic changes no active training settings and is not a grokking run.

## Source clarification, 2026-10-10 23:19 UTC

The coordinator independently confirmed the Gemini source-audit errors.
These constraints are already implemented and remain unchanged: the head is
zeroth-order cached-logit loss perturbation; upstream noise is scored through
the remaining LayerNorm FORWARD; no analytic LayerNorm derivative is needed.
Dust Adam beta1 in the official appendix is .7; the deliberately matched
grokking comparison retains AdamW betas(.9,.98), lr.001, decay1 on EVERY
parameter and no embedding-specific learning rate. Shared vocabulary239,
released tokenization and trainable LayerNorm biases/gains are preserved.
Gemini cosine thresholds and noise ranges are not adopted. Our later
noise-scale amendment above was based on measured diagnostics.

Execution distinction: pinned minimal dust.py direct_errors invokes the whole
model on repeated perturbed batches. The paper describes an execution-optimized
cached suffix. This repository uses a tested cached-suffix adaptation of the
released transformer; it is neither a verbatim minimal-source implementation
nor a claim of exact reproduction of the paper's tuned LM recipe. Independent
nonzero-suffix/reference-hook tests cover this optimization's residual paths.

## Actual Dust trajectory diagnostic, 2026-10-10 23:23 UTC

The full-batch initialization and memorized-BP diagnostics cover different
states from the ongoing Dust fit. Before the next diagnostic, archive the
latest durable shared-Dust checkpoint selected only by wall time, with a
5GiB disk guard and SHA256. At this fixed state compare the FULL scientific
RHS objective's exact backprop diagnostic gradient against P4/P16/P64,
three perturbation seeds9278703/4/5, graph credit, direct sigma.002/last
hub.004, local/head.05, exact2823-pair batch and chunk256. All diagnostic
optimizer steps use lr0 and discard weights. No held-out loss enters choices.

Run on worker1 with four diagnostic threads alongside its two-thread BP fit;
record this concurrency in timing, without changing training parameters or
original guards. New output/manifest; external900s TERM plus60s KILL grace.
The gradient audit may use autograd; the scientific Dust training continues
entirely forward-only. This measures update signal on the actual Dust
trajectory rather than inferring it solely from a different optimizer's state.

## Adam-discovered subspace audit, 2026-10-10 23:41 UTC, before code/runs

User hypothesis: useful grokking descent may occupy a small subset of update
directions. Test low-dimensional parameter-update spaces first, and report
per-module/coordinate concentration to distinguish them from sparse parameter
masks. No ongoing Dust fit is changed. These are BP-informed diagnostics;
successful restriction would not demonstrate that unassisted Dust discovered
the basis, and the computational cost of its BP teacher must be reported.

First construct separate exploratory bases from the available BP1800,
BP6400, BP6800 and BP10000 snapshots: (a) Adam gradient-driven last-step
updates reconstructed from first/second moments and bias corrections,
excluding decoupled decay; (b) decay-corrected checkpoint displacements.
For constant lr.001/decay1, the latter is theta_b-(.999**(b-a))*theta_a,
a geometrically weighted sum of intervening gradient-driven updates.
Use non-centered SVD; the small snapshot count limits observable rank and
cannot by itself establish a low-dimensional trajectory. Preserve provenance
and snapshot-selection history, including BP6800's known generalization.

Replay the paired BP control from the immutable BP1800 model AND optimizer
state, same train split/full RHS loss/AdamW/warmup schedule, on freed worker1
with four threads. Stop at10000, external3600s TERM+60s KILL grace and internal
3540s. Checkpoint/trace writes require >=5GiB free and new output directories.
Record sums of gradient-driven Adam updates in fixed20-step blocks, separately
from weight decay, and the instantaneous last update. Evaluate every100.
Record source hashes, timings and any numerical divergence from the original
two-thread path; do not assume this replay reproduces its transition exactly.

Fit the PRIMARY dense non-centered basis to fixed1800..4800 update blocks,
then measure captured update energy on4800..6800 and6800..10000. Fit the full
1800..10000 trajectory only as an explicitly retrospective secondary basis.
Ranks1,2,4,8,16,32 are fixed before inspecting spectra; report cumulative
energy/effective rank, captured signal, projection error and module/coordinate
participation, rather than an arbitrary cosine pass threshold. No held-out
labels/losses enter basis construction or rank selection.

At fixed Dust375 and BP1800, audit the FULL training RHS gradient versus
fully estimated Dust P4/P16, seeds9278703/4/5, direct.002/last-hub.004 and
local/head.05. Compare raw gradient, the checkpoint's frozen Adam diagonal
preconditioner, and the exact optimizer's one-step gradient innovation
relative to its zero-gradient/momentum baseline. Compare retained exact
signal energy as well as projected cosine/noise. A signed coordinate
permutation within each parameter tensor provides a same-rank orthonormal
scrambled-basis control with matching per-module basis energy.

Forward central finite differences along basis vectors (L2 step.001/.01/.1)
are a diagnostic of directional derivative/sign/smoothing, not the full
Dust estimator. Report their train-only scores against the diagnostic
autograd oracle, with all weights discarded. Probe actual projected Adam
candidate steps including the same full decoupled decay; separate forward
loss observations from gradient-angle evidence. Worker2 can use four audit
threads beside its existing two-thread P4 fit, externally1200s plus60s grace.
Coordinator stays at <=2 compute threads. Only after measured retained signal
and forward descent justify a restriction will a separately named bounded
Dust-with-BP-subspace training variant be considered; it must retain a matched
projected BP control and must not be labelled the unmodified Dust recipe.

Pre-spectrum clarification, 2026-10-10 23:48 UTC: raw20-step block updates
remain the primary fit. Also report two predeclared secondary early-window
bases: unit-normalized block directions (prevent large update bursts from
dominating their spectrum) and instantaneous last updates. Their fit window,
ranks and time-test windows are identical. Reuse the exact saved Dust gradient
draws when comparing bases, validating checkpoint hashes/layout/settings;
this gives a paired basis comparison without changing sample counts.

## Functional Fourier concentration diagnostic, 2026-10-10 23:50 UTC

A small functional subspace may differ from a fixed whole-parameter update
space, especially across the independently evolved BP and Dust models.
Measure (without training) residue-row embedding/unembedding Fourier energy
and the learned residue-logit function on the ordered97x97 input grid at
BP1800, BP6800, BP10000 and Dust375. For logit L(a,b,c), a function of
a+b-c alone has Fourier support (f,f,-f mod97); report that support's share
of class-centered logit energy and its frequency concentration. This is a
direct diagnostic of our models, not an assumed circuit or a claim from a
research summary. Whole input grid is used only for forward evaluation.

Predeclare ablation ranks1,2,4,8,16,48 of nonzero frequency PAIRS. Rank modes
using the average normalized embedding/head weight energy, independent of
held accuracy. Retain the residue-row mean and all other vocabulary rows,
and forward-evaluate embedding-only, head-only and joint truncations. Include
raw accuracies/losses and restore every original weight. This tests whether
a few weight frequencies suffice for the observed function, without calling
that a low-dimensional whole-model update space. No ablated model is trained
or copied to existing runs. CPU<=2 threads, external360s plus60s grace,
disk>=5GiB for new artifacts, source/checkpoint hashes logged.

## Projection placement diagnostic, 2026-10-10 23:53 UTC

Sparse probes reveal different frozen-preconditioner and actual-Adam
innovation alignment at BP1800: noisy gradients also enter Adam's second
moment, so post-step projection is not equivalent to filtering first.
Before new candidate runs, fix three additional DISCARDED forward probes:
(1) project raw g before Adam; (2) filter in the checkpoint's fixed Adam
metric, g_filtered=P^-1 U U^T P g, before actual Adam; (3) apply that metric
filter and also project the resulting data step onto U. P is the positive
diagonal scaling of the next-step linear gradient innovation with current
second moment held fixed. Actual candidate Adam steps still update their
second moments with the filtered gradient and retain full decoupled decay.
Use the SAME stored exact/Dust gradients and scrambled controls, report
train RHS/answer loss against the original and zero-gradient baselines.
These are estimator/optimizer placement diagnostics, not changed training
or proof of a grokking mechanism. External audit bounds remain1200s.

## Generalization direction sensor, 2026-10-10 23:56 UTC

For the next dense-basis diagnostic only, compute a separate held-out RHS
gradient as an observational sensor. Score the PREDECLARED bases/ranks and
candidate updates by retained held-gradient energy, signed first-order held
loss change and discarded forward held losses at maximum saved rank. This
distinguishes generalization descent from training-fit signal. Held gradients
never construct bases, set ranks/noise/settings, filter training gradients,
or enter optimizer states of scientific fits. Log this evaluation use
explicitly; prior train-only sparse audits remain unchanged. Reuse exactly
the same previously saved training/Dust gradient draws for paired comparison.

## Attribution and fixed late-window follow-up, 2026-10-11 00:07 UTC

Paired dense probes show that projection and imported BP momentum both affect
one-step loss. Before interpreting a filter as stronger Dust signal, subtract
the MATCHED zero-gradient data-step baseline for that filter: full momentum
for prefilters, projected momentum for postfilters. Compare filtered Dust
innovation against filtered exact-gradient innovation, and report gradient-
driven versus decoupled-decay held changes separately. Use saved draws and
maximum predeclared rank32; no new population/rank selection or training.

When the existing bounded teacher reaches its horizon, automatically compute
the complete fixed-time temporal tests and a retrospective full-trajectory
basis. Also fit the fixed4800..6800 late window, independent of the replay's
actual held transition times, and report principal-angle overlap with the
early basis. This observes how update directions change; neither retrospective
basis is an unassisted Dust discovery. A separate follow-up guard bounds
waiting plus analysis to3600s and writes only new outputs. No ongoing Dust
fit is altered. Report teacher updates/forward examples as BP assistance cost.

## BP-assisted restriction pilot, 2026-10-11 00:13 UTC, before code/runs

At BP1800, the predeclared normalized early rank32 basis plus Adam-metric
prefilter/postfilter gives P16 matched-innovation cosine.2882 versus.0253
unfiltered (three fixed draws). Mean train RHS CE after its candidate step
is.0317077 versus its zero-gradient baseline.0317586 and initial.0317638;
exact filtered control gives.0316984. This TRAIN-ONLY diagnostic justifies
a bounded continuation test. Held sensors did not select its basis/rank.

Three new arms start from IDENTICAL immutable BP1800 model AND AdamW state:
P16 Dust unfiltered; P16 Dust with normalized1800..4800 rank32 basis and
g_filtered=P_t^-1 U U^T P_t g before Adam, then projection of its gradient-
driven data step onto U; and exact-BP with the identical filter. Full decay
continues on every parameter, including LayerNorm bias/gain and embeddings.
P_t is derived solely from current optimizer second moments; it is updated
per step and never uses an oracle in Dust. Both Dust arms use fresh common
noise seed9278731, unchanged source-adapted estimator/scales .002/.004/.05,
full2823-pair training RHS and absolute-step learning-rate schedule. No held
gradient/label is used for filtering/training. Initial memorization and BP
assistance (1800 warm start plus3000 basis-teacher steps) are explicitly
inherited; no result can establish that unassisted Dust grokked from random
initialization. Record all costs/provenance and report complete trajectories.

Worker1 unfiltered Dust four threads alongside its existing four-thread
teacher until completion; worker2 filtered Dust four threads plus filtered
BP two alongside original P4 two. Wait for auxiliary builds/audits to finish
before launching so each worker stays <=8 compute threads. Each new arm gets
external1800s TERM+60s KILL and internal1740s, Dust max1000 updates/BP max3000,
eval every10 Dust/100 BP plus1/5/10, new output directories and5GiB checkpoint
guard. Log JSON-array metrics separately from original science fits so that
the main delayed-grokking report cannot confuse assisted continuations with
the from-initialization experiment. No ongoing original job is changed.

## Fixed-metric correction, 2026-10-11 00:26 UTC, before revised runs

Adaptive inverse-metric prefilter is numerically unstable even for the exact
BP control. A replay of its durable200-update checkpoint shows raw gradient
norm~14 while filtered norm doubles through6e5..1e7 and second moment/metric
condition number grows past1e14. This is feedback in the experimental
P_t^-1 projection, not evidence that source Dust cannot learn. Preserve that
run/error, cooperatively stop only its filtered Dust partner, and retain the
valid unfiltered continuation.

For revised filtered BP and Dust arms, freeze P at the ORIGINAL BP1800
optimizer state instead. Actual Adam first/second moments still update
normally using filtered gradients; beta/lr/decay/source estimator/basis/rank/
common noise seed remain identical. New output names include fixed; restart
both at the same BP1800 checkpoint, not the corrupted state. Add fail-fast
checks for filtered-gradient square overflow and nonfinite optimizer/model
state, with failure JSON; preserve latest durable good checkpoint. Dust may
start once the fixed-metric exact-BP partner has100 finite updates. All
external1800s/internal1740s/checkpoint guards remain, and no original from-init
Dust job changes. Long-run stability/grokking remains to be measured.

## Complete-basis signal transfer, 2026-10-11 00:57 UTC, before code/run

Once the full teacher trace is available, score the predeclared early,
transition-window and retrospective bases against the SAME stored exact and
P4/P16 Dust gradients at BP1800/Dust375. Fixed rank32, all three saved noise
seeds, learned versus module-matched scrambled bases. Report raw/frozen/actual
Adam-innovation retention and projected alignment, plus matched innovation
alignment for the existing pre/postfilter choices. This is a tensor-only
diagnostic: no new gradient draws, teacher updates, forward candidate scoring,
held labels, training or optimizer-state mutation. Full-trajectory basis
uses future BP training information and remains explicitly retrospective.
Run on coordinator<=2 CPU threads under180s external TERM/30s KILL, new
output and5GiB disk guard. Original runs and continuation bounds unchanged.

## Gemini variance suggestions audit, 2026-10-11 01:33 UTC, before code/runs

Gemini's short/sparse-loss concern is a hypothesis, not an established cause.
The frozen task has six inputs/two supervised targets, full2823-pair updates,
independent activation/module noise, causal prefix credit and local attention.
AdamW(.9,.98), decay1 and direct sigma.002/hub-last.004 are already in use.
BP1800-to-Dust warm starts exist but their71–73-update budgets do not test
all possible later grokking. No original scientific run/settings change.

Before any training variant, compare discarded full-batch gradient estimates
on the latest durable shared-Dust checkpoint selected by wall time, and the
existing BP1800 checkpoint. Same P16 evaluation budget, same three noise seeds
9278703/4/5, objectives full RHS and answer-only separately on worker1/worker2.
Variants: unchanged source-adapted one-sided Dust; symmetric direct/hub/head
and nonlinear Q/K draws with HALF as many independent direction pairs; and
unchanged one-sided draws with ALL noise scales reduced tenfold (direct.0002,
last-hub.0004, local/head.005). This diagnoses scale sensitivity rather than
adopting an unvalidated loss-dependent schedule. No combined treatment.

Antithetic numerator is L(plus)-L(minus), divided by2sigma; standard noise is
multiplied afterward. Keep corrected draw-chunk/batch centering, supervised
target normalization and all local causal paths. V is linear conditional on
clean attention and remains ordinary one-sided draws to avoid duplicate work.
Advance/discard unused Gaussian directions so module-stream RNG state matches
the baseline at each site; P16 means16 actual evaluations, eight pairs for
nonlinear streams. Record pair count, actual forward/head/attention counts,
wall time and per-family cosine/norm/error against diagnostic-only exact
gradients. Never use an analytic training head or an oracle hub target.

Numerical tests before runs: paired linear/quadratic sign and sigma scaling,
even-order cancellation, finite-sample centering, counts/RNG-budget equality,
sparse causal support and all-parameter coverage. Existing estimator tests
remain applicable because src/dust.py is unchanged. New implementation is a
clearly labeled diagnostic adaptation, not a claim of the exact paper recipe.
Each worker gets900s TERM/60s KILL,<=4 threads, separate output directory and
checkpoint/disk guard5GiB. Gradients/candidate weights are discarded; source
snapshots are immutable and hashed. Held labels are not used for these audits.

Forward AD would be a separate first-order control, not zeroth-order Dust.
Do not launch it or change the task to dense prefix sums for this comparison.
No theoretical probability-of-success labels or conclusion of fundamental
incompatibility follows from a bounded unsuccessful handoff/filter.

## Attention versus MLP isolation, 2026-10-11 03:17 UTC, before code/runs

The user requests an empirical component diagnosis. Preserve the released
two-block transformer, six-token equation, p97/30% split and full RHS objective.
The original shared P16 run completed3000 updates: train90.8962%, held14.3638%.
Archive its final model AND Adam state immutably with SHA256. No original run
is restarted or overwritten. The speculative Fourier-phase explanation and
literature claims supplied by Grok are hypotheses, not established evidence.
Removing attention while retaining tokenwise MLPs would remove operand mixing
at the answer position; it is not an equivalent estimator ablation.

First run a discarded-gradient audit at that Dust3000 checkpoint: exact
full-training-batch RHS and answer-only gradients (diagnostic oracles), then
unchanged source-adapted P16 Dust gradients for three noise seeds9278741/2/3.
Report separate Q/K/V/O, FF input/output, embedding, head and each LayerNorm
gain/bias, grouped attention and MLP, exact gradient energy, raw and actual
Adam-innovation alignment. Score hypothetical one-step full-training losses
for the six interventions below using identical saved Adam state; restore
weights after every candidate. No held labels for these diagnostic gradients
or candidate selection; no oracle hub/head enters a Dust estimate. All
candidates are predeclared and retained in the report rather than selected.
Audit worker1<=4 threads under900s TERM/60s KILL with5GiB disk guard.

Then run six fixed-checkpoint, matched-state diagnostic continuations:
  dust: all parameters receive Dust;
  bp: all parameters receive exact gradients;
  attention: exact Q/K/V and attention output projection in both blocks;
  mlp: exact FF input/output matrices in both blocks;
  mlp_half: exact alternating flattened coordinates of every FF matrix;
  qk: exact Q/K matrices in both blocks, V/O still Dust.
All unselected coordinates, including embeddings, head and ALL LayerNorm
gain/bias, receive Dust in hybrid arms. Attention has131072 parameters, MLP
262144, MLP-half131072, Q/K65536. The half-MLP mask is fixed before observing
new gradients; it matches parameter count, not architecture or gradient energy.
Record masks/group definitions and actual energy fractions. Hybrid gradients
use backpropagation and cannot establish forward-only Dust success. They test
whether supplying directional accuracy to a component helps repair this one
late Dust state; they do not prove which component is necessary/sufficient
from random initialization or which mechanism produces Fourier features.

Use fresh COMMON Dust noise9278741, P16/nrep4, direct.002/hub-last.004/
local-head.05, existing causal credit and complete Dust estimator in every
Dust/hybrid arm. Computing all Dust groups before substituting exact groups
keeps stream RNG/cost matched; redundant estimates are counted, never hidden.
Restore identical model/Adam state in all arms, retain/update moments normally,
absolute-step learning-rate schedule and AdamW(.9,.98), decay1 on ALL params.
No gradient clipping, optimizer reset, altered loss or frozen unselected group.

One arm perworker1..6 in the order above, AFTER original jobs exit. <=8 CPU
threads each; coordinator<=2. Each arm max200 additional updates, eval every10
plus0/1/5, internal3480s and external3600s TERM/60s KILL. Use new directories,
independent tmux guards and refuse checkpoints below5GiB. Save full accuracy/
loss trajectories, wall/update time, actual Dust/BP forward counts and exact
supervised presentations. Compare common update prefixes before different
runtime cutoffs. Separate JSON-array metrics from original unassisted fits.
Require numerical tests for group partition/count/mask, exact-substitution
equivalence with a manual matched Adam update, pure-Dust no-autograd and common
draw/RNG accounting. Check initialization/optimizer hashes equal across arms.
Only held-out generalization with a delayed transition can count as grokking;
hybrid or BP repair of training fit is a component diagnostic alone.

## Attention-path refinement, 2026-10-11 04:10 UTC, before code/run

The original three-seed one-step audit favors exact attention; exact Q/K
alone gives no comparable gain. The first five-update matched continuation
also favors attention, while exact MLP/QK remain near all-Dust. Attention
output matrices hold55.45% of exact RHS gradient energy at Dust3000, all
LayerNorm affine terms41.92%; energy alone is not a causal attribution.

On now-idle worker2, run an explicitly exploratory, training-only one-step
refinement at the SAME immutable Dust3000 model/Adam state and three seeds
9278741/2/3. Retain all six original candidate scores and add exact V only,
attention O only, QKV only, all LayerNorm gain+bias, and attention+MLP matrices
together. All other coordinates remain Dust. Compare actual forward RHS,
answer and EOS loss changes, and first-order changes using each exact
objective's gradient. Include exact RHS versus answer cosine baseline; a
full-RHS estimate compared with answer truth is not answer-only estimator
accuracy. Retain raw gradients for reproducible discarded-candidate analysis.
No training weights retained, no held labels, no model/optimizer mutation to
existing six runs. Worker2<=4 threads, new output,900s external TERM/60s KILL,
5GiB disk guard. This is a follow-up suggested by observed initial results,
not a preregistered independent validation of those results.

## Rainbow alignment diagnostic, 2026-10-11 05:13 UTC, before code/runs

User authorized next steps after the literature review. The interrupted turn
had only inspected files/designed this experiment; no alignment process or
training run was launched. Existing component runs have ended under their
predeclared guards. No training is resumed or modified for this diagnostic.

Use two frozen model/Adam states: Dust3000 and verified grokked BP6800. Reuse
the saved three-seed full-batch Dust3000 exact/Dust gradients; prepare the same
P16 estimates and diagnostic-only exact gradients at BP6800 using seeds
9278741/2/3. No analytic hub/head enters Dust and no gradient is retained as a
training update. Settings remain direct.002/hub-last.004/local-head.05.
Reconstruct actual one-step AdamW candidates: Dust, BP, exact-O, exact-MLP,
exact-QKV, BP data-step rescaled to Dust data-step norm with identical decay,
zero-new-gradient with old momentum+decay, and pure decay. All parameters
and moments remain immutable in the source files; candidates are discarded.

Fit alignments on first1411 training pairs in the established shuffled split;
evaluate transfer on the remaining1412 TRAINING pairs. No held examples or
labels fit/select an alignment. Separate positions1,3,4; omit teacher-forced
answer position5 from feature fitting. Sites: each block's128-D attention hub
(also four32-D heads independently) and512-D post-ReLU MLP hidden features.
Report raw/aligned feature drift, norm and Gram change, singular values and
rank at relative1e-6. Ordinary uncentered Procrustes is descriptive; primary
consumer interventions use identity-regularized orthogonal Procrustes,
min ||Hnew R-Hold||²+lambda||R-I||² with lambda=1e-6*smax(Hnew^T Hold).
This fixed numerical convention limits arbitrary nullspace motion; it is not
a paper-prescribed or performance-tuned hyperparameter. Report unregularized
and regularized fit/transfer errors, and rotation distance from identity.

At each interface, compare old/new producer features crossed with old/new
consumer weights under the SAME old residual/downstream suffix. Also evaluate
transported-old consumer against the identical frozen-old baseline. Then
evaluate old consumer, transported-old consumer and transported-new consumer
inside the full candidate model. For row features Hnew R≈Hold, transported
weight is W R^T. Position4-fitted transforms change the shared weight globally
at every token; report cross-position transfer. Full128-D hub rotations may
mix heads and are functional counterfactuals. Exact attention gauge controls
use only within-head Q/K paired rotations and V/O inverse rotations; ReLU
controls use neuron permutations/positive scales, not arbitrary rotations.

Evaluate full-grid class-centered answer-logit Fourier coefficients before/
after every uncorrected candidate, fixed residue coordinates(f,f,-f), amplitude
and phase together. Rank frequencies by the OLD state's training-only residue
embedding/head spectrum, not held accuracy or post-update outcomes. Report
all modes and reference-energy-weighted complex/phase similarity. Held-input
forwards and labels are evaluation only; no update selection or optimization.
This distinguishes failure to form a Fourier circuit from destroying one.

Worker1 Dust3000 audit, worker2 BP6800 gradient preparation then audit; <=4
threads each, each diagnostic process external1800s TERM/60s KILL (preparation
900s), new directories and5GiB disk guard. Numerical tests first: Procrustes
orientation and rank-deficient identity behavior, independent train transfer,
consumer-transport algebra, exact attention/ReLU gauge invariance, candidate
suffix reproduction at zero intervention and immutable state restoration.
Record every actual clean/suffix/evaluation forward and P16 gradient cost.
Do not interpret rotations or one-step loss recovery as proof of a mechanism.

The user's new correlation question is separate: current component arms share
one initialization, so they cannot establish seed-independent easy held pairs.
Before a multi-initialization fit, define fixed pair-level predictions, common
split/checkpoint budgets, initialization and perturbation seeds varied
separately, and overlap/correlation nulls conditional on each run's accuracy
and target residue. No stronger 'always the same pairs' claim from finite seeds.

## Pair-level difficulty audit, 2026-10-11 05:15 UTC, before code/run

User requests which held-out examples pure Dust solves, and whether the same
pairs would recur across random initializations. First describe available
models without claiming independent-seed evidence. Archive final component
checkpoints immutably, including the pure-Dust3177 state; compare pure-Dust
1875/3000/3177 on the fixed2823/6586 split. These share one initialization
and are temporally related. BP/hybrid final predictions may be comparisons,
with assistance/update-count differences explicitly recorded.

Save predictions, answer CE and true-answer-vs-best-other margin for every
ordered pair, including invalid/nonresidue predicted tokens. Primary score
is answer at position4, with independent arithmetic truth (a+b)%97 and shared
vocabulary mapping. Map errors across a,b,c; stratify wrapped vs unwrapped,
diagonal, zero operands, reversed pair in training, and training frequency of
operands/target residue. Always report group counts, avoid conclusions from
small bins, and distinguish output-class preference from pair difficulty.
Show correctness97x97 grids and aggregate accuracy by residue. Report which
pairs remain correct, become correct or regress across checkpoints; overlap,
Jaccard and excess overlap over independent marginal correctness are only
descriptive null comparisons, not proof of seed-invariant difficulty.

This is held-label evaluation only: no optimizer, gradient, update selection,
hyperparameter selection, fitting or training. New output/5GiB disk guard,
coordinator<=2threads under external300s TERM/30s KILL. Tests independently
verify modular truth/token mapping, held/train partition/group counts,
reversed-pair membership and overlap formulas. No multi-seed training is
launched by this descriptive audit; such runs need a separate paired plan.

## Initialization × perturbation replication, 2026-10-11 05:39 UTC, before code/run

The user specifically asks whether the pure-Dust9.0% on pairs with neither
ordering trained is a fixed subset determined by training coverage, or varies
with initialization. Primary evaluation universe is the4669 established held
ordered pairs whose reversals are also held. This universe, the2823 training
pairs, vocabulary, six-token equation, released architecture, full-batch
objective, optimizer and decay stay fixed. No held label changes a training
choice or stop criterion. Report ordinary6586-pair held accuracy secondarily.

Use a crossed3×2 design: initialization seeds9278700,9278701,9278702 and
Dust perturbation seeds9278901,9278902. Within each initialization both runs
start with identical parameters and fresh Adam states; perturbation generators
are separate from model RNG. Across initializations reuse the same two noise
seed labels as common random numbers, not as independent paired observations.
All six models train from scratch, with no inherited BP update. A single
initialization cannot establish seed-invariant difficulty; three seeds provide
a pilot, not evidence that a subset is mathematically always easy.

Each Dust run uses established small-noise settings: P16/nrep4, direct.002,
last-hub.004, local/head.05, causal credit, zero-order head and LayerNorm
gain/bias. Full2823-pair batch, chunk256, AdamW(.9,.98),lr.001,warmup10,wd1
on every parameter. Use the already-tested src.train.update implementation.
Single-worker execution changes random draw order from the earlier four-worker
trajectory; it does not change the declared estimator settings. The original
3177 checkpoint is historical context, not an exchangeable replication arm.

Predeclare3200update horizon and20-hour external TERM/60s KILL cap per Dust
run, internal71400s with45s shutdown margin; budget stops are not failures.
Measured~19s/full update at8threads projects~17hours for3200updates. Archive
model/Adam checkpoints at1000,2000,3000,3177,3200 and retain latest. Evaluate
at0,1,5,10,every100updates,3177andfinal; preserve all ordered-pair predictions,
answer CE and true-answer margin at each evaluation, allowing matched-update
set comparisons without repeated model scoring. Check disk>=5GiB before
each checkpoint/prediction write; outputs are new and original runs untouched.

Use worker1/2 for initialization9278700, worker3/4 for9278701, worker5/6 for
9278702; first in each pair uses noise9278901, second9278902. Eight compute
threads per worker, coordinator kept modest. Workers1/3/5 first run one paired
BP control each from the identical initialization/fresh Adam state, max10000
updates under3600s external/3540s internal cap. These controls are separate
models and never warm-start a Dust run. Worker2/4/6 can start Dust immediately.
Use sequential wrappers with per-job guards and an outer21-hour5-minute
TERM/120s KILL guard on BP→Dust chains. Each job logs commands, hashes,
runtime, split, initialization, perturbation seed, forward counts and BP counts.

Primary comparison is common3177updates, secondary fixed1000/2000/3000/3200
checkpoints and full trajectories. If some runs stop sooner, report only
available common updates and the limitation; do not select a checkpoint by
held accuracy. Report per-pair number correct across runs, all/any correct,
Jaccard, retained/lost/unique sets, and correctness/margin correlations within
fixed initialization across noise, and across initialization at fixed noise.
Calculate expected overlaps given each arm's observed success count within
the4669-pair universe, then condition on true answer residue/training frequency
where sample sizes permit. Account for (a,b)/(b,a) dependence by treating the
unordered pair as a cluster for any uncertainty/resampling; do not count
pairwise overlaps as independent experiments. Fixed-split replicas distinguish
randomness effects conditional on this dataset; they cannot prove the same
phenomenon across different train splits. No theory-driven numeric cosine or
overlap threshold will be invented as a pass criterion.

Before launch: test same-init equality/different-init inequality with fixed
data and independently reproducible perturbation generators, forward-only
Dust path with backward blocked, saved prediction metrics against existing
evaluation, unseen-unordered-pair mask and checkpoint guards. Smoke-run the
new runner through actual checkpoint/completion/export, retaining smoke files.

## Nested subsets and prediction reconstruction, 2026-10-11 05:52 UTC

User adds two questions while the initialization study runs: whether smaller
training sets solve nested subsets of the same original4669unseen equations,
and whether teacher predictions can reconstruct removed training labels.
Prepare these as separate follow-ups; do not interrupt the initialization
study or repurpose its processes. Fresh-student reconstruction is the default
primary interpretation while the optional fresh/warm preference is pending.
Warm continuation, if requested, must be labeled retention/dynamics, not
independent reconstruction of information already present in teacher weights.

All training subsets are selected by unordered operand group so a removed
training example cannot be revealed by retaining its reversal. Shuffle the
original training groups once with RandomState9278910; keep the first rounded
75% and50% of groups, nested, reporting actual ordered counts. Use model seed
9278700 and perturbation9278902, matching the full-data i0-n1 initialization
arm, fresh Adam, unchanged Dust P16/nrep4/small-noise/AdamW recipe and3200
update horizon. The primary evaluation universe remains the ORIGINAL4669
both-orientations-held examples, independent of the reduced training split.
Compare solved sets at common updates, including intersection, original-only,
reduced-only, containment fraction and marginal/target-conditional nulls.
No assumption that success sets must be nested. Different example counts
mean different presentation/forward budgets; record both, do not equate them.

Frozen teacher: original pure-Dust3177 checkpoint SHA256
35ba0b317bee0976c755cf7a535a623f1267290ef62a1e279832ace7f38432c0.
R=75%-group retained train; H=remaining original train. Teacher has trained
on H: H is held from the STUDENT only. The subset75% arm is the R-only
reconstruction baseline; no redundant identical run is needed.

Split the original-unseen unordered groups into pseudo-candidate/evaluation
halves with RandomState9278911 before selection. Pseudo examples come only
from the candidate half; both orientations of every reserved evaluation group
remain unseen by all students. Choose |H| candidate ordered examples by the
teacher's predicted top1-minus-top2 logit gap, restricting predicted token
to the known97residue output domain; ties by original pair index. This uses
full239-way teacher logits from the saved read-only audit. Neither true-answer
margin, true CE, correctness, nor arithmetic target may enter selection.
Report teacher correctness only after fixing selection, as an audit.

Fresh-student arms: R-only (reuse subset75%), R+P_teacher, P_teacher-only,
R+P_shuffled, R+P_oracle. The shuffled control permutes teacher labels over
the SAME selected P with RandomState9278912, preserving their histogram.
Oracle uses true P labels as an explicitly labeled synthetic-task upper
control, unavailable in the proposed unknown-label real-world setting.
Pseudo training must replace BOTH input answer-token position5 and target
position4; retaining true input answers would contaminate EOS supervision.
All fresh students receive new random parameters and Adam, never teacher
weights/optimizer. Teacher/student masks and data provenance are persisted.

Primary reconstruction measurements: H true-label accuracy/CE/margins versus
R-only at the same update, trained-target fit, agreement with teacher, and
accuracy on reserved unseen groups. Original full held accuracy is secondary
and explicitly includes pseudo-training inputs for these arms; never call it
independent generalization. P-only also measures recovery across original S.
Selection coordinates/labels and retained R carry information; improved
reconstruction alone is not an MDL result. A description-length comparison
must count input selection, labels, model/training specification and residual
prediction cost; teacher information is not free.

Execution plan if fresh-student interpretation stands: queue one follow-up on
each worker AFTER its current Dust replica finishes, without modifying the
current chain: workers1/2 subset75/subset50,3/4 teacher-mix/pseudo-only,5/6
shuffled/oracle-mix. Each first runs a separate matched BP control with same
initialization/data (10000updates,3600s external/3540s internal), then fresh
Dust (3200updates,72000s external/71400s internal),8threads,chunk256.
Queue wait bounded78000s; whole queue capped43hours externally with120s KILL
grace. Refuse failed/incomplete predecessor states and existing output paths.
Disk>=5GiB before all checkpoints/exports. No final accuracy-dependent stop.
If the interpretation changes before launch, record it and revise this plan.

Tests before queue: disjoint unordered R/H and candidate/evaluation groups,
subset nesting, primary-universe invariance, pseudo selection invariance to
changed held ground-truth labels, predicted-confidence (not true margin),
correct pseudo input/target replacement, shuffled histogram preservation,
fresh initialization/optimizer, and meaningful evaluator/training smoke.
All queued executions and caps must be visible in STATUS.md.

## Frozen probability-ensemble diagnostic, 2026-10-11 06:55 UTC

Predeclared before implementation/evaluation. The user requests full prediction
distributions from the existing independent initialization-by-noise replicas.
No training, loss, optimizer, worker allocation or interaction is changed.
Evaluate frozen pure-Dust checkpoints only; backprop controls are excluded.
Primary endpoint remains3177updates for all six replicas. Secondary fixed
milestones:200,1000,2000,3000,3200. At200, the three noise9278902 replicas
already have saved weights; the noise9278901 replicas will reach200 later.
Publish the fixed three-initialization noise9278902 group as an explicitly
early interim result, then all six at exactly200. Never pool different steps.
If200is missed by the collector, mark it missing; do not reconstruct from
different-step weights. Later archived milestones remain eligible.

Capture latest.pt via read-only copies while its embedded step matches200;
otherwise use immutable milestone files. Validate embedded step, split/vocab,
seed/method provenance, source hashes, and prediction agreement with original
saved top1/CE exports. Preserve checkpoint SHA256 and verify before/after
evaluation. Extract position4 logits over all239tokens, forward-only with
requires_gradFalse, no optimizer construction/update and no backward. Confirm
answer-position predictions are invariant to replacing future input token5.
Use CPU coordinator with at most2compute threads; no worker inference added.
All checkpoint/output copies enforce5GiB disk guard. Per-checkpoint extraction
has independent300s TERM/30s KILL guard; report generation600s. Read-only
collector runs in tmux under20h external/71400s internal bounds, polling60s.

Combination rules fixed now: primary uniform arithmetic mean of per-model
softmax probabilities at temperature1 over239classes. Secondary uniform mean
of raw logits followed by softmax; top1 plurality with tied votes broken by
lowest vocabulary token ID. No fitted weights, temperatures, confidence gates
or heldout-based member selection. All-six group primary; fixed noise groups
(three initializations each) and same-initialization noise pairs secondary.
One-member best according to full original TRAIN answer CE is the deployable
selection baseline; observed best held member is descriptive only. Compare
single-model and ensemble inference cost explicitly.

Primary population originalU4669; secondary originaltrain2823, allheld6586,
reversed-trained1917. Save per-pair distributions/predictions and report
accuracy, answerCE, Brier score, true-answer rank, true probability, entropy,
top1 margin, invalid-token prediction rate, vote ties, and corrected/lost
examples relative to train-selected member. Separately count gains where
NONE of the participating models has a correct top1 prediction, and losses
where at least one was correct. Top1-correct union is an upper bound only for
selectors among member top1 outputs, not for probability averaging. Record
individual true-answer ranks/probabilities and competing predictions on these
gains; no manual selection of favorable examples. Summarize reverse-pair
groups as dependent units, without treating9409equations as independent trials.
Early near-chance ensembles cannot answer the trained9%question. No grokking
claim from ensemble performance alone or from accuracy on reversed training
pairs. Endpoint reports update automatically as matched frozen states arrive.

Required tests: constructive all-members-wrong/soft-ensemble-correct case;
all-members-agree-wrong countercase; full-vocabulary normalization including
nonresidue classes; stable softmax under extreme logits; deterministic ties;
per-example logit-offset invariance; single-member identity; step/split/member
mismatch rejection; immutable forward extraction with future-token test and
saved prediction parity; disk refusal. No new training tests are required.

## CPU execution audit, 2026-10-11

User asks whether execution optimizations are being left unused. Audit isolated
fresh models on coordinator, normally two compute threads, under an independent
900s TERM/30s KILL guard and5GiB output guard. No production process changes,
training checkpoint writes, backward passes, optimizer construction or updates.
Use actual P16/nrep4 noise settings and256 training equations. Profile one
gradient estimate with CPU operator timing and nested wall-clock method timers;
report these separately because nested timers overlap and profiler overhead
affects absolute timings. Unprofiled repetitions provide latency comparisons.
Measure fused causal scaled-dot-product attention and computing cross-entropy
only on the two supervised positions, independently and together, with identical
model/noise initialization. These are candidate execution changes, not new Dust
estimators. Validate logits, losses, every estimated parameter gradient, RNG
state, unchanged weights and evaluation counts before interpreting speedups.
Counterexamples/failures are retained. Changes to draw grouping or batch chunks
also change centering/noise assignments and are not treated as exact substitutions.
No live replica is migrated based on this preliminary audit. Any production
candidate needs tests including sparse causal credit and post-checkpoint resume.

Extend this discarded audit to supervised suffix pruning: preserve all earlier
attention keys/values and every original random draw, but after the final
attention compute only the two supervised positions through tokenwise maps.
For direct final-block sites, retain only those positions during suffix
execution; pad unused logits solely to maintain the original loss interface.
Validate same-noise gradients at initialization and the frozen Dust3177 state.
Do not prune prefix positions before their last cross-token dependency.

CPU operator profile exposed104448 small clones from strided population
contractions in one256-equation estimate. Also test replacing the two matching
einsum reductions over draws by elementwise weighting and summation over draws.
Preserve random tensor shapes, order, draw-group centering and normalization;
check floating-point gradient differences, not bitwise identity promises.

After the two-thread audit finishes, permit brief, sequential microbenchmarks
on the otherwise idle coordinator at8threads,8threads pinned to one sibling
per reported core, and16threads. Each process is externally bounded90s, all
three together300s; only256-equation estimates, no training or checkpoint I/O.
The16-thread burst temporarily uses all reported coordinator CPUs to test SMT,
then exits. Use two measured repetitions after a warmup, baseline and the
combined suffix/reduction candidate. This is a preliminary timing screen,
not a production-worker throughput claim or a change to the running replicas.

After the microbenchmark finds a candidate gain at8threads, permit one final
baseline/candidate full2823-equation gradient estimate on coordinator pinned
to CPUs0–7,8threads,120s external TERM/30s KILL. Include complete gradient and
RNG comparison including the short final chunk. No optimizer step; report this
as full-gradient computation latency rather than a measured resumed training
run. Then restore coordinator to its normal low-load monitoring role.

## Seed-study execution migration, 2026-10-11

User asks to replace the expensive seed-study execution with the tested faster
path. Preserve the scientific study and all computed weights; do not restart
from random initialization. Before migration validate production CpuDust against
the audited candidate and original Dust, including resumed AdamW updates from
the frozen Dust3177 checkpoint. Original src/model.py, src/dust.py, src/train.py
and src/seed_train.py remain unchanged. New execution code has explicit source
hashes, same P16/nrep4, noise draws/order, credit, centering, loss normalization,
model/task/split, optimizer and seeds. No backward pass in production or this
resume check. Require exact original-path checkpoint replay and unchanged
perturbation RNG advancement/counts. Record candidate numerical differences;
do not claim bitwise-identical optimized trajectories.

Pause only the six owned waiting follow-up queues before any parent stop.
Gracefully stop each owned Dust process after its current update, preserving
its final checkpoint, optimizer, RNG, counters, predictions and original output.
Do not signal unrelated jobs or alter credentials. Create a new continuation
directory per run under runs/seed-fast-20261011 with immutable parent SHA and
copied historical metrics/predictions. Keep original source configs and stopped
completion metadata in provenance. No previous experiment directory is reused
for new training. Copy original current state exactly; no optimizer reset.

Use one common cutover for all six: the smallest multiple of100 greater than
or equal to the largest gracefully saved parent step. Choose from progress only,
never accuracy. Each continuation uses original Dust up through that boundary
and CpuDust thereafter; archive the boundary weights/optimizer/RNG. Thus the
crossed initialization/noise experiment has one execution schedule across runs.
Record the switch in config, checkpoint provenance, metrics and analysis reports.
Original full-batch2823/chunk256, evaluation schedule, primary3177 and cap3200
remain fixed. Existing BP controls stay separate and unchanged.

Each continuation uses the original absolute start+71400s internal deadline and
start+72000s external deadline (not fresh20h), 5GiB disk guard,8compute threads
pinned to one reported CPU sibling per core after topology verification. Resume
counters and elapsed compute time; separately report recent execution speed.
Restart held follow-up queues against the new predecessor, preserving their
original absolute outer deadline and their existing experiment definitions.
Update collectors/extractors to understand lineage without mixing checkpoints
at different steps. Preserve early200 probability exports and config snapshots.
Retain rollback checkpoints and manifests. Verify live processes, RNG/Adam
restoration, actual transition timing and monitor health before completion.

Deployment selected common600 from gracefully saved parent steps465/530/417/
553/472/537 (progress-only rule). Continuations started08:33UTC. A read-only
cutover watcher runs under10800s external/10700s internal bound, one coordinator
thread, checking each immutable600 checkpoint SHA against switch metadata,
exact Adam/RNG preservation at that boundary and at least five completed CPU
updates. It writes runs/seed-migration-20261011/cutover_status.json. Transition
verification remains pending until the scheduled boundaries are actually reached;
deployment reports must distinguish resumed jobs from an already active CPU path.

## Official-code CPU baselines, 2026-10-11 10:22 UTC (before implementation)

User requests trying the smaller official baselines on CPUs to locate the
execution gap. This is an isolated timing and estimator diagnostic, not a new
grokking or FineWeb training claim. Keep all six seed jobs and follow-up queues
unchanged. The vendored qlabs-eng/dust commit b20f7c03eac630b6441ba6f254128c1761dc45ad
was checked against upstream HEAD today and is still current. Do not edit it.

Import its GPT, make_model, DUST, recipe, optimizer and backprop train_step
directly in a new CPU wrapper, bypassing only the CLI CUDA requirement.
Use CPU float32 and retain the source rotary-buffer behavior; log all dtypes.
Official anchor: 8 layers, width512, 8 heads, vocabulary4096, nominal direct
population256, nrep2, public 1M-token optimizer configurations. Smaller model:
2 layers, width128, 2 heads (head dimension64), otherwise the source model.
Its allocation is an explicit adapter, NOT the unpublished exact small-model
recipe: sum the first/last four entries of each public 8-layer allocation into
two blocks, preserving each stream total; multiply per-block local counts by4,
preserving their total across two blocks (including alternating value gates).
Head1024, embedding10 and 8 antithetic scalar directions remain unchanged.
Thus both configurations have256 total direct-loss draws, not256 per module.

Model seed42, perturbation seed42, fixed synthetic next-token data seed20261011,
8 sequences of2049 uniformly drawn tokens from the source vocabulary. Test
prefix context lengths6,64,256,2048 with config maximum sequence_len2048 held
fixed, so short-prefix tests do not change the source attention windows.
Dense targets supervise every position; terminal-only cells mask all but the
last two positions on EXACTLY the same inputs. These are runtime/gradient
diagnostics, not real language modeling or modular-addition learning. Preserve
source direct gamma0, K/V/gate/value-embedding gamma.98 in both; explicitly
measure rather than conceal the sparse causal-credit limitation. The original
grokking workload is measured separately on its real full2823-equation batch,
with both original and optimized Dust, plus its matched AdamW BP control.

Each cell restores identical starting weights and fresh optimizer state before
each measured update. Record one cold and up to two warm repetitions, including
noise generation, gradient construction and optimizer step; exclude model
construction/data loading/evaluation. Instrument inclusive direct/head/local/
scalar/capture times and actual full-forward/noise counts. Timing on the
coordinator uses2 compute threads pinned to CPUs2,3 for every method; do not
compare these numbers directly to8-thread worker timings. Pair BP and Dust
using identical model/data within each cell; retain each official optimizer
because this diagnoses the released execution path, not a learning contest.

At the short6-token small-model cells, measure per-parameter and whole-model
cosine/relative error against an exact BP gradient at the same initial weights,
including both embeddings, attention internals and residual scalars. Inspect
estimated embedding errors at unsupervised positions in the sparse cell.
Backprop is used ONLY in the separate baseline and gradient diagnostic. Verify
Dust executes with grad disabled, all trainable parameters receive finite
estimates, clean caches persist, and the benchmark wrapper leaves vendor and
production training sources unchanged. No thresholds imply eventual learning.

Run serial cells under tmux with independent per-cell GNU timeout bounds
(180s for short/grokking cells, 450s for longer contexts) and an1800s suite
guard. Internal suite deadline1740s. A timed-out cell reports BUDGET_STOP and
any completed measurements, not algorithm failure; skip later repetitions
when the first update predicts they cannot finish. Refuse writes/checkpoints
below5GiB free. Save no trained model; all updates are disposable. Launch first
short cells before the long-context ladder so initial useful measurements
arrive quickly. Results/configs/commands go under a new official-cpu-baselines
run directory and STATUS.md. EGGROLL runtime remains unmeasured here; the
official EGGROLL inference-throughput figure is not a CPU update-time claim.
