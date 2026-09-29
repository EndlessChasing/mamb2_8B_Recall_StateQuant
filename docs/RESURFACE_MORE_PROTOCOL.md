# More Resurface training under SQ3.25: frozen continuation protocol v3

Frozen 2026-09-28 before continuation fitting. This experiment addresses the
request to try more Resurface by increasing training, starting from the audited
v2 SQ3.25-trained adapter. It does not change adapter capacity or the state
format. Preserve all v2 files and results.

## Parent and unchanged inputs

- Parent v2 protocol SHA256:
  `24466642ce87c75fc2136a42ed14e69733c462b507a0be82c062b4da7f836bcb`.
- Parent completed training report SHA256:
  `e5a77d86cf2fb0e2389247e3cb325f74e89957861a6043e92a891d6d402ae359`.
- Parent final FP32/Adam/scaler checkpoint, 1536 successful updates:
  `bc548dd427d114098048fa1863f8e602c095dc2d9fde56348795628ae8e2c78f`.
- Parent FP16 adapter SHA256:
  `7144b88265a797ef935b1f94845395abec5532d8dc5b2a4a8fa0a8e07ce205f0`.
- Existing no-adapter calibration SHA256:
  `c366cd577967e64635f1dd960237dc1c0024d7413685ce0b4c4e40da869a7023`.

Keep the original pure Mamba2-8B source, FP16 weights, 507 frozen tensors,
original-coordinate SQ3.25 tables, packed per-token forward, masked STE backward,
native convolution/projections, separate S16 original-model prose teacher,
memoryless soft-gated Resurface geometry, and all v2 data identities unchanged.
Calibration and numeric TRAIN manifests remain bound to the v2 protocol.
The new receipt adds `continuation_protocol_sha256`; it must not relabel those
old inputs as freshly calibrated under v3. No Q8 or weight quantization.

## Exact continuation

Restore all 224 FP32 adapter masters, 224 AdamW parameter states and both
parameter groups from the parent final checkpoint. All optimizer steps must
initially be 1536. Verify every master cast to FP16 equals the parent exported
tensor, and compare restored training forward against parent packed inference
bitwise on 128 fixed TRAIN tokens before the first update.

Restore the complete GradScaler state: scale 16, growth factor 2, backoff 0.5,
growth interval 2000 and growth tracker 1536. Verify loaded master, optimizer
and scaler values against the checkpoint, not just their shapes. Explicit
overflow handling preserves the existing trainer's semantics, including its
manual `update(new_scale=scale/2)` behavior. Do not infer the scaler solely
from the total number of overflows.

Perform one discarded resumed smoke update in a separate output directory,
then reload the original parent checkpoint for the formal continuation.
Exactly **3072 additional successful updates**, for **4608 cumulative updates**.
At most eight additional overflow retries; retry the identical example pair.
Keep intermediate recovery checkpoints every 768 successful additional updates.
Only the final 4608-update adapter is the quality candidate. No DEV/CONFIRM
checkpoint selection or adaptive recipe changes within this experiment.

## TRAIN schedule and objective

Reuse the same 1536 numeric TRAIN cases. Concatenate two CPU `torch.randperm(1536)`
orders, generated independently with seeds **2026092804** and **2026092805**.
The successful additional update index `r` ranges from 0 through 3071; retries
do not advance it. Reusing TRAIN is explicit, not a claim of new examples.

Continue the prose schedule using `j = 1536 + r`:
`window = prose_order[j % 448]`,
`start = 512 * ((j // 448) % 4)`.
Each segment is 512 TRAIN tokens with 511 targets. Preserve the pinned TRAIN
file and manifest hashes from v2. DEV and CONFIRM never enter fitting.

Keep the v2 loss:
`MK answer CE + 0.5 prose CE + 0.5 KL(S16 teacher || SQ student) + 3 closure`.
Keep closure budget 0.006 and excess coefficient 10, full 256K vocabulary,
AdamW betas (0.9, 0.999), epsilon 1e-8, zero weight decay, gradient clip 1,
FP32 masters, FP16 forward casts, and block checkpointing.

Use one continuous cosine tail across the additional 3072 updates:
`factor(r) = 0.01 + 0.09 * (1 + cos(pi * r / 3071)) / 2`.
Multiply the original base rates 1e-4 (V/g) and 3e-4 (router). This starts at
the parent's final actual rates 1e-5 / 3e-5 and ends at 1e-6 / 3e-6. Never pass
the global index into the old 1535-denominator schedule, which would reheat it.

Record every attempt's case, prose segment, rate factor, overflow/scale and
successful/cumulative update counts. Bind the report to parent report,
checkpoint, adapter, calibration, data, source and all training code hashes.
Save the complete final checkpoint and actual FP16 export; each final master
cast must equal its exported FP16 value. Require final packed-training versus
deployment forward equality on the fixed 128-token probe and unchanged cache.

## Full evaluation and claims

Run full validation directly; no new pilot is needed for this fixed continuation.
Use the same 130 WikiText-2 validation windows / 264764 target tokens and all
768 CONFIRM prompts (384 normal, 384 target-removed), same reset behavior,
full-vocabulary greedy generation, at most 12 tokens and six-digit scoring.

Execute three fresh full arms in order:

1. Parent v2 adapter with SQ3.25.
2. Continued v3 adapter with the same SQ3.25.
3. Reinstall parent v2 adapter and repeat the entire first arm.

The first arm must exactly match the archived v2 full adapter result for every
window NLL and generated token sequence. The restored final arm must exactly
match that first arm. Verify adapter contents, all frozen base identities,
versions/gradients, and cache allocation before/after each arm. As in v2,
these are not full post-training byte hashes of all original GPU weights.

The primary continuation comparison is **v3 versus v2**, not merely versus
unadapted SQ. Report PPL change, paired MK gains/regressions, and 10000-draw
paired bootstrap 95% intervals using seed 20260928. The improvement gate is
PPL no more than 1% worse, observed MK improvement, and a strictly positive
lower confidence bound. Also report whether PPL itself improved.

Use hash-verified v2 full S16 and unadapted SQ reports as explicitly archived
context, not fresh reruns. Separately show remaining PPL/MK gaps to original
S16. Restoring original PPL still requires no more than a 1% increase.
Report failure without changing the gate or selecting an intermediate checkpoint.

An independent CPU audit reconstructs training schedules/scaler transitions,
checks optimizer/checkpoint/export bindings, decodes all generated IDs,
recomputes NLL/PPL, paired statistics and restoration from recorded evidence.
It does not independently rerun GPU logits. FP32 atomic backward reductions
prevent a promise of bit-identical retraining. These benchmark families have
historical exposure; further training does not establish unseen generalization.

Keep the repository private. No public model release is included.
