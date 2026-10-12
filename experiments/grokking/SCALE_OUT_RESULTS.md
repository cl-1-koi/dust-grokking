# Scaling the existing follow-ups across six workers

Execution migration on 2026-10-12, requested to use idle workers 2 and 4.
Scientific settings remain unchanged: original Dust implementation, P16 per
site, original 256-example sub-batches, training examples and order, single
perturbation RNG stream, model and Adam state, and the 3200-update horizon.
There is no backprop, population increase, or additional scientific treatment.

Each rank computes alternating complete sub-batches. It generates and discards
only the original noise tensors for sub-batches computed by its peer. Each
original sub-batch estimate is then broadcast and accumulated in the serial
order before one Adam step. Head centering and local attention credit stay
inside their original sub-batches. Logical forward counts remain unchanged;
extra noise-generation work is logged separately. This is data parallelism,
not splitting each module's perturbation population across machines.

## Discarded-update benchmark

Both modes started from the same immutable teacher_mix step1700 checkpoint,
with eight CPU threads per worker. Three full updates were evaluated per mode;
none was retained for training. External limit600s and checkpoint guard5GiB.

| Update | One worker | Two workers, critical path |
|---|---:|---:|
| First | 19.685s | 13.376s |
| Second | 20.276s | 12.886s |
| Third | 20.260s | 13.131s |

The last two updates average20.268s versus13.008s: **1.558x faster**, or
**35.8% less wall time**. Timings include estimator computation, replicated
noise generation, communication and Adam. Post-update audit hashes and file
writes are excluded. Two warm observations are a limited throughput sample.
Twice the workers buys1.56x speed, so this is a wall-time improvement using
idle capacity, not improved aggregate machine-hour efficiency.

All three steps matched the serial model, Adam state, accumulated gradients,
loss, RNG state and logical counts bit for bit. Full final model/optimizer/RNG
and counter tensors were independently compared for both ranks. Six focused
engine/audit tests also pass, including uneven sub-batches, a final one-example
batch, different population/chunk shapes, sparse supervision weights, warmed
Adam and forbidden backward calls.

Raw comparison: `results/scale-out/comparison.json`.

## Live migration

- Teacher mixture: stopped cleanly at1795, resumed on workers3+2.
- Oracle mixture: stopped cleanly at1933, resumed on workers6+4.
- Subset75 stays on worker1; shuffled mixture stays on worker5.

Every continuation restores the original checkpoint and retains the original
absolute internal/external limits, conservatively shortened by10s for original
setup overhead. New outputs are separate from the sealed parents. Per-rank
tmux sessions have independent external guards; collectives have120s timeouts.
Checkpoints refuse free disk below5GiB. No automatic scientific horizon extension.

The first attempt computed matching rank states but failed while writing its
first new prediction file: copying the read-only parent directory preserved
read-only permissions in the new destination. Both failed attempts and their
source snapshots are retained. Only the new prediction directory and appendable
metrics destination were made writable. Eleven continuation tests now pass,
including a real read-only parent-package regression test. The second attempt
restarts from the same original sealed checkpoints, not from the discarded
failed update, and preserves the same deadlines.

Current outputs: `runs/followups-scaled-v2-20261012` on each participating
worker. Migration manifest: `runs/scale-migrations-20261012.json`. The read-only
collector overlays the new rank0 histories onto the old follow-up histories
and updates `FOLLOWUP_RESULTS.md`; full-seed comparisons now use the complete
seed archive rather than its earlier pre-migration collection.

Live v2 checkpoint health confirmed at teacher1800/oracle1938; paired model, Adam
and RNG hashes agree. Observed initial live throughput12.4–13.4s/update.
Original resume/benchmark checkpoints include optimizer and RNG state and remain
in the experiment archive; they are not part of the public evaluation-weight bundle.
