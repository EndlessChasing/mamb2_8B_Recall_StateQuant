# Fresh Resurface on the validated V10 Q3.25 state

This prospective protocol is frozen before any V11 GPU measurement or training.
The user requests Resurface after the unadapted Q3.25 PPL target was achieved.

## Fixed source and state

- Source: pure `nvidia/mamba2-8b-3t-4k`, revision
  `b915550c63ba9359f88f44d1f6a600d85af27302`, original checkpoint SHA256
  `47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb`,
  loaded as FP16. No weight quantization or source fine-tuning.
- V10 selected calibration SHA256
  `a77bbd86ae609d8ef7cf90f10d0904cf95b4df8e059c9ea6cb2fd12886436fb8`;
  receipt `4dbc05cef7f3cd3b5bb43e65c0f25a1befdee2218aef3a476291a43093645b4c`;
  table `b1865e81ff3fbed028027a883872aef9bb91e614e7cf08c72211496a78aeb597`.
- Layout is fixed: 32 INT8, 32 INT4, 64 zero carry, two stored FP16 scales.
  Actual stored-scale RN division and half-away integer rounding are unchanged.
  Carry is quantized after every token including prefill; current readout occurs
  before pruning the carry. No clipping or FP16 shadow state.
- Bind V10 full selected result SHA256
  `598b7d3f8088d58513c4ab99aaf911543b0c355e02d9e79cb59a734c55a3f610`,
  comparison `d0a86c047d992377f8a6cf9c996d8b5525f03309497d378b814dd22d39ddea57`,
  and full audit `31d93219972466f6c62b8e1036c4de6cceee1d4c53d438faffeb02853c0affe5`.
- Unadapted full PPL is 8.186186562837207. V10 MK is unknown at freeze time.

## Implementation and pre-training checks

All new code lives under `scripts/`. Existing core modules and V2–V10 frozen
sources/protocols remain unchanged. Inventory the new protocol and all executed
training, inference, data, objective and backend dependencies in receipts.

The new stateless training controller preserves native projections, convolution,
normalization and Resurface hooks. Its forward calls the deployed V10 packed
scan. Its checkpoint/history recomputation must reproduce that scan's decoded
carry using the stored FP16 scales and exactly 64 retained coordinates.
Backward is a declared live-mask straight-through estimator: retained carry
coordinates use identity derivative, zero carry coordinates block future-state
gradient, and all 128 coordinates contribute to current-token readout gradient.
It is not the mathematical derivative of discrete quantization.

Before training, require controlled CPU/GPU fixtures with every-token and
chunk-boundary carry equality, multiple lengths/chunk sizes, zero and boundary
values, and backward comparison against an independent PyTorch STE reference.
Backward tolerances must be declared in the checker before its first GPU run;
failures remain recorded. Do not claim bitwise equality for floating-point
atomic gradient reductions. Forward and recomputed carries require exact equality.

Then run a separate discarded one-successful-update smoke from fresh parameters.
Check fresh identity and nonzero FP16 export against deployed packed execution
at 128 and 512 TRAIN tokens, finite gradients, source/table immutability, master
cast/export serialization equality, and backend pinning. Formal training restarts
fresh; the smoke adapter is never a candidate or initialization.

## Fixed training recipe

Reuse unchanged `train_quant_first.py` schedule, pair construction, optimizer,
loss and attempt mathematics (SHA256
`547e5b64904a273cd96a7d77624e5225a67edb9fa683950a43efc2d4171392a6`).
Train only 224 Resurface FP32 master tensors / 1,154,104 parameters, fresh
V=0, g=1, router_w=0, router_b=-4; no adapter/checkpoint warmstart.

- 1,536 successful updates, schedule seed 2026092803, final export only.
- Fixed numeric TRAIN manifest SHA256
  `451b8703c21120667ef0ea272c21a10d4cd4561779a7b9663773ae47981600c0`;
  original numeric protocol SHA256
  `24466642ce87c75fc2136a42ed14e69733c462b507a0be82c062b4da7f836bcb`.
- Paired 512-token prose TRAIN segments: token file SHA256
  `e54b02e5162e042a9cdd504f4eb1b1652724fb240bbc2c97608967aa26297233`,
  manifest `facb2ca461615a4199781bd21784d642d6674f5b862641b3b9edac3fb499b89d`.
- Numeric answer-suffix CE weight 1; prose CE 0.5, original frozen S16 teacher
  KL 0.5, router closure 3. Separate unadapted frozen teacher, full vocabulary.
- AdamW mix LR 1e-4, router LR 3e-4; inherited cosine, betas (0.9,0.999),
  eps 1e-8, weight decay 0, clip norm 1.
- GradScaler initial 1024, growth 2, backoff 0.5, interval 2000. At most eight
  overflow attempts in addition to successful updates; retry the same entry.
- Save recovery checkpoints after 384, 768, 1152 and 1536 successful updates.
  The only quality candidate is the final 1536-update FP16 export. No heldout
  checkpoint, hyperparameter, gate strength, or threshold selection.
- Record attempts, finite checks, schedule IDs, losses, scaler/optimizer state,
  immutable source identity/version/gradient checks, table byte hash and peak memory.
  Source checks do not constitute a post-run full byte hash of the loaded model.

## Full deployed evaluation

Pin the historical RMSNorm backend (16 warps, 3 stages, 1 CTA) before first
model forward/load and disable TF32 as in V10. Record exact reset probes.

Evaluate three fixed arms: (1) unadapted V10, (2) V10 with the final new FP16
adapter, (3) the same V10 after adapter removal. Each arm runs:

- WikiText-2 validation, all 130 windows of length 2048, 264,764 targets,
  full 256,000-token vocabulary, lm_head NLL chunks of 64. Token stream SHA256
  `5bbeae08ba8eb34a482f3b6e9d17b182e67229dd14b2853d87f89fc72e5ad027`.
- All 384 normal and 384 target-removed numeric CONFIRM prompts, unchanged
  frozen generation/evaluation code, full-vocabulary greedy, max 12 tokens/EOS,
  first standalone six-digit prediction. Preserve all outputs/generated IDs.

The unadapted arm must exactly replay V10 archived PPL rows, reset hidden/cache
hashes and PPL-end cache receipt. Save PPL-end cache separately from MK cache.
The removed-adapter arm must exactly reproduce the first arm's PPL rows, all
768 generated token sequences/predictions, probes and corresponding cache receipts.
Historical S16 metrics may be shown only as separately hash-bound context.

Report aggregate normal MK, N16/N64, template and query-position strata, and
wrong-present-value misbinding versus unparseable/invented outputs. Target-removed
accidental matches are a diagnostic, not an abstention or recall score.
Use paired 10,000-resample bootstrap, NumPy default_rng seed 20260928, sorted
normal case IDs, for the normal MK difference and its percentile 95% interval.

## Prospective acceptance and memory accounting

Primary success requires all of:

1. Adapted full PPL <= unadapted V10 full PPL 8.186186562837207.
2. Adapted full PPL < 8.25 (report this achieved user threshold separately).
3. Normal MK increases and paired 95% confidence lower bound is positive.
4. Baseline/archive and adapter-removal exact replay pass.
5. Codec, smoke, training, export and independent raw-evidence audits pass.
6. Actual per-request persistent state remains 28,499,968 bytes:
   SSM payload/scales 23,855,104, FP16 convolution 4,587,520, uint8 table 57,344.

The old 1% PPL tolerance is not the primary gate for this experiment. Report
failed gates and metric changes honestly; do not relabel degradation unchanged.
Keep raw failed attempts and do not select a candidate using CONFIRM/PPL.

Report adapter payload separately: expected 2,308,208 FP16 tensor bytes; verify
actual tensors, loaded residency and serialized file size. Cache plus one loaded
adapter is expected 30,808,176 bytes, excluding source weights, temporary tensors,
allocator reserve, training teacher/optimizer/checkpoints and training workspace.
Adapter EMA cache stays disabled. This is a state-memory claim, not a total-GPU
memory or weight-compression claim. No public publication is part of this run.
