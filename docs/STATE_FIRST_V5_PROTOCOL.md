# Optimize unadapted SQ3.25 first, then train fresh Resurface: v5

Frozen before v5 candidate quality measurements, 2026-09-28. The user explicitly
requires optimization without Resurface, followed by Resurface for recall.
Preserve v1-v4 artifacts and implementation files. This experiment must not use
the frozen v2 adapter to select a state representation or initialize training.

## Fixed source, storage and execution

Use the original pure Mamba2-8B checkpoint and tokenizer pinned by v2:
source SHA256 `47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb`;
tokenizer SHA256 `5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09`.
All 507 source tensors / 8,236,999,680 parameters stay frozen FP16. No weight
quantization, Q8, new state codec, optimizer update to the source, or extra cache.

All candidates use the unchanged real packed tier codec: per 128 coordinates,
16 INT8 + 64 INT4 + 48 zero-carry slots and two FP16 scales. This is 52 bytes,
or 3.25 bits/element including scales. One uint8[56,8,128] permutation table
occupies 57,344 bytes. With FP16 convolution cache, batch-one persistent cache
must be exactly 28,499,968 bytes. Quantize carried state after every token,
including prefill; current readout is before that token's carry quantization.

Pin existing RMSNorm forward to 16 warps / three stages / one CTA before any
model load or forward, using the unchanged v3 replay helper. Apply it to
screening, both training models, smoke checks and final evaluation. Record
installed source hashes, precision flags and actual backend configuration.
Native projections, convolution, gating and FP16 residual math remain unchanged.

## Four fixed unadapted candidates

Reuse the completed original-S16, no-adapter TRAIN statistics from v4:
`reports/state_repair_calibration/calibration.pt`, SHA256
`8509bb266d40875608f3e0b3be22407fd6ea1b8aacbc79ce04e958726516b0a4`.
These statistics used only TRAIN rows 0..7, first512 tokens per row (4,096 total),
with independent resets. Their provenance and exact collector check must be
validated. Reuse the unchanged old magnitude permutation, calibration SHA256
`c366cd577967e64635f1dd960237dc1c0024d7413685ce0b4c4e40da869a7023`.

Candidate order, also the final tie break:

1. **magnitude:** exactly the old magnitude permutation.
2. **full_readout:** exactly the v4 stable readout-score permutation, ranking
   `mean((decay_t*h_prev_fp16*C_t)^2)` descending.
3. **preserve_int8:** retain the old first16 coordinates in their original
   order. Rank the other112 coordinates by readout score descending, with
   original-coordinate-index ties; assign first64 to INT4 and last48 to zero.
4. **preserve_retained80:** rank the old first80 retained coordinates by readout
   score descending, with original-coordinate-index ties. Assign first16 to
   INT8 and next64 to INT4. Keep the old final48 zero coordinates in their
   original order.

All tables are derived on CPU and independently reconstructed from the pinned
statistics. Do not fit thresholds, mix scores, choose layers, change scales,
or add candidates after observing v5 results. Dense Q3 variants failed v4 and
are excluded from v5. These score-based candidates are experimental; the
one-step proxy is not exact loss sensitivity or a guarantee of recall.

## TRAIN-only selection, with no Resurface loaded

Use pinned WikiText-2 TRAIN tensor file SHA256
`e54b02e5162e042a9cdd504f4eb1b1652724fb240bbc2c97608967aa26297233` and v2 prose
manifest SHA256 `facb2ca461615a4199781bd21784d642d6674f5b862641b3b9edac3fb499b89d`.
Screen rows8..39, first512 tokens each:32 windows /16,352 prediction targets.
Numeric TRAIN cases use sample indices0,16,...240 in all six N/template cells:
96 normal prompts. Score full-vocabulary greedy generation, max12 tokens with
EOS stop and first standalone six-digit integer, exactly as prior evaluations.
No target-removed prompts are needed for this development screen.

Run magnitude, full_readout, preserve_int8, preserve_retained80, then restored
magnitude. No adapter may be installed or loaded in any screening arm. Require
repeated-reset128-token hidden/cache equality, finite outputs and persistent
state/scales, exact byte budget, frozen-source identity/version/gradient guards,
and exact replay of every magnitude NLL and generated token sequence at the end.

Exclude incomplete, nonfinite, failed-check or over-budget candidates. Baseline
magnitude must complete or the experiment stops. For a new candidate, require
normal MK correct >= max(1,ceil(0.5*baseline normal MK correct)); this is only a
noncollapse guard before recall training. Magnitude is always the fallback.
Among admissible candidates choose lowest TRAIN PPL, then highest normal MK,
then the fixed candidate order. Including magnitude ensures the selected PPL
is no worse than baseline. If magnitude wins, stop v5 after reporting no
selected quantizer improvement; do not duplicate its existing Resurface run.
Equal PPL can still select a new table with higher MK; advancement alone does
not establish PPL improvement. The separate full improvement gate remains decisive.

Only a non-magnitude winner advances. Freeze its table, selection report and
calibration hashes before training. Validation/CONFIRM must not select the table,
training recipe or checkpoint. This is development on historically exposed
benchmark families, not an untouched generalization experiment.

## Fresh Resurface after the table is fixed

Use the selected table in unchanged StateQuantTraining: exact packed forward,
declared live-mask STE backward. Initialize fresh V=0, g=1, router_w=0,
router_b=-4. No v2/v3 adapter, optimizer state, or continued checkpoint is loaded.
Use the unchanged v2 numeric TRAIN corpus, all1536 examples, manifest SHA256
`451b8703c21120667ef0ea272c21a10d4cd4561779a7b9663773ae47981600c0`.
Its numeric data protocol remains v2, SHA256
`24466642ce87c75fc2136a42ed14e69733c462b507a0be82c062b4da7f836bcb`;
this must be recorded separately from the new v5 experiment protocol.

Reuse v2 schedule, pair_for, optimizer_for and attempt math without alteration:
1536 successful updates, schedule seed2026092803, same paired512-token prose
segments, separate frozen unadapted S16 teacher, numeric answer-suffix CE,
prose CE/KL and closure objective, optimizer groups and cosine LR schedule.
The reused v2 trainer source SHA256 is
`547e5b64904a273cd96a7d77624e5225a67edb9fa683950a43efc2d4171392a6`;
record and verify this dependency. Initialize GradScaler with scale1024,
growth_factor2, backoff_factor0.5 and growth_interval2000. Recovery checkpoints
retain all FP32 masters, Adam state and scaler state with success/attempt counters.
Use at most8 total overflow attempts across the entire formal run, retrying
the same schedule entry after each overflow. Record every
attempt. A separate discarded one-successful-update smoke must verify fresh
initialization, source/table freeze, finite gradients and128-token exported
FP16 packed training/inference equality. Formal training restarts fresh.

Only the final1536-update FP16 export is a candidate. Keep recovery checkpoints,
final master-to-FP16 equality, adapter/tensor hashes, training receipts and
post-training packed parity. Backward atomic reductions may prevent identical
checkpoint bytes on a new training run; this does not relax inference replay.
No checkpoint selection using TRAIN loss, DEV, validation or CONFIRM.

## Full confirmation and separate quality gates

After final export, run four arms in one fixed evaluation process:

1. Old magnitude SQ3.25, no adapter; require exact archived full no-adapter SQ
   replay (130 per-window NLLs and768 generated token sequences).
2. Selected SQ3.25, no adapter.
3. Selected SQ3.25 plus the newly trained v5 Resurface adapter.
4. Restored selected SQ3.25, no adapter; require exact arm2 replay.

Every arm uses130 WikiText-2 validation windows /264,764 prediction targets and
384 normal plus384 target-removed CONFIRM prompts. Keep all outputs and report
all arms regardless of candidate quality. The validation token stream remains
`5bbeae08ba8eb34a482f3b6e9d17b182e67229dd14b2853d87f89fc72e5ad027`.
Original S16 and old v2-adapted results are hash-verified archived context;
v4 already exactly replayed the complete v2-adapted control twice.

Report three distinct gates; do not replace a failure with another comparison:

- **Unadapted state optimization:** selected no-adapter PPL at least1% lower
  than old unadapted SQ; same cache. Also report its MK change.
- **Resurface repair on the selected state:** trained PPL <=1.01 times selected
  no-adapter PPL; normal MK increases and its paired95% bootstrap lower bound
  is >0; same cache.
- **Improvement over the prior Q3.25 + v2 endpoint:** trained PPL at least1%
  lower than archived v2 PPL; no observed MK decrease; paired95% bootstrap
  lower bound >=−2 percentage points; same cache.

For all paired MK intervals sort normal cases by ID, use10,000 resamples and
NumPy default_rng seed20260928. Report original-S16 PPL gap and the within1%
restoration check separately. Target-removed zero matches do not imply abstention.
No threshold or training change follows a failed gate in this experiment.

## Evidence and publication

Use new scripts/docs/artifact names; do not alter frozen v1-v4 core files or
receipts. Explicit new formats bind candidate/calibration/selection/protocol,
source, tokenizer, both TRAIN manifests and fresh adapter initialization.
An independent CPU auditor reconstructs tables, selection, training schedule,
export/checkpoint equality, raw scores, bootstrap gates, actual storage and
full replays. It audits recorded evidence, not independently regenerated GPU
logits. Preserve failed probes and negative quality outcomes. Document the
previous adapter-frozen experiment separately. Keep the repository private;
no model publication is included in this request.
