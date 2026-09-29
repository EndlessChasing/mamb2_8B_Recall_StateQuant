# Same-budget SQ3.25 repair experiment v4

Frozen before new calibration or candidate quality measurements, 2026-09-28.
Preserve all v1/v2/v3 protocols, source files, checkpoints and reports. This is
a new codec/calibration experiment, not a continuation of v3's quality claim.

## Fixed source, adapter and execution

Use the original pure NVIDIA Mamba2-8B source and tokenizer pinned by v2:
source SHA256 `47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb`;
tokenizer SHA256 `5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09`.
All 507 source tensors remain frozen FP16. The sole adapter in this experiment
is the frozen v2 1536-update FP16 export, SHA256
`7144b88265a797ef935b1f94845395abec5532d8dc5b2a4a8fa0a8e07ce205f0`.
No optimizer updates, adapter selection, Q8, weight quantization or rotation.

Pin RMSNorm forward to the existing 16-warp/3-stage/1-CTA configuration before
any model forward, using the v3 replay policy. Record installed source hashes,
precision flags and actual configuration. All candidate state updates are
causal, quantized after EVERY token, including prompt prefill. Readout uses
the current FP32 recurrence update before quantizing the carry. Keep native
projections, convolution, gating and FP16 residual orchestration unchanged.

## Three candidates, fixed before measurements

1. **readout_tiers:** retain 16 INT8, 64 INT4 and 48 zero-carry coordinates;
   replace only the per-layer/B-C-group permutation. Rank TRAIN coordinates by
   `mean((decay_t * h_prev_fp16 * C_t)^2)`, descending with stable coordinate
   index ties. This estimates one-step output damage from erasing history.
   Ignore coordinate cross terms; do not call this exact loss sensitivity.
2. **dense3:** retain all 128 coordinates with two 64-coordinate scale groups.
   Use the existing v2 magnitude-sorted permutation. Codes are 3-bit two's
   complement, symmetric levels -3..3 (the -4 code is unused), zero exact.
3. **dense3_equalized:** same dense codec, original coordinate order. Replace
   the permutation table with signed INT8 power-of-two exponents. For TRAIN
   mean absolute S16 carry m, set `v=max(m,1e-12)`,
   `k=clamp(round(log2(v)-mean(log2(v),coordinates)),-8,8)` per layer/group.
   Rounding uses ties-to-even for this offline exponent selection. Store
   u=h/2^k; apply B'=B/2^k and C'=C*2^k in FP32 inside the recurrence after
   convolution and SiLU. No projection-weight folding or additional history.

Dense quantization uses an FP32 max-absolute denominator divided by 3, clamped
to at least 1e-8, half-away-from-zero rounding and clipping to [-3,3]. Carry is
decoded using the actual stored FP16 scale. This preserves the existing scale
rounding convention; scale underflow/overflow must be tested explicitly.

Each format uses exactly 48 payload bytes + 4 scale bytes per 128-coordinate
row: **3.25 bits/element including scales**. Dense codes are physically packed
as three bitplanes; no byte-per-code or FP16 shadow carry at inference. Exactly
one 56x8x128 one-byte table is resident per candidate: 57,344 bytes. Including
FP16 convolution cache, batch-one persistent storage must stay **28,499,968 B**.
Offline calibration may store multiple candidate tables, counted separately;
only the selected table is moved to the deployed runtime.

## Calibration and implementation validation

Use the pinned 448x2048 WikiText-2 TRAIN tensor, file SHA256
`e54b02e5162e042a9cdd504f4eb1b1652724fb240bbc2c97608967aa26297233`,
and its v2 manifest. Collect on the unadapted S16 source: first 8 rows, first
512 tokens, reset per row (4096 tokens). Aggregate over time, grouped heads,
channels and batch. Retain the original v2 permutation from calibration SHA256
`c366cd577967e64635f1dd960237dc1c0024d7413685ce0b4c4e40da869a7023`.

Before quality screening require independent reference checks of dense packing,
readout, all scales and persisted state; zero/negative/extreme inputs; batch2,
odd channel counts and grouped B/C; full-sequence versus segmented and one-token
execution. Test both table modes. Require collector S16 outputs and final carry
to equal the unchanged S16 codec. Check actual runtime cache allocation and
128-token full-model repeated-reset determinism. New files must not alter the
frozen existing implementation. Surrogate training gradients are out of scope.

## TRAIN-only screening and fixed selection

Use TRAIN rows 8..15, first512 tokens each: 8 windows,4088 prediction targets.
Generate only the existing numeric TRAIN split and take sample indices
0,32,64,96,128,160,192,224 in each of the six N/template cells:48 normal prompts.
These examples were seen during prior adapter training, so screening is a
development heuristic, not generalization evidence. No CONFIRM/validation
scores are opened for selection.

Evaluate original S16 without adapter, old SQ3.25 without adapter, old SQ3.25
with v2 adapter, and all three candidates both without and with that same v2
adapter. Score PPL and full-vocabulary greedy MK (at most12 generated tokens,
EOS stop, first standalone six-digit integer). Report all nine arms, including
failures. Repeat old SQ+v2 after candidates and require exact score/output replay.

Select ONE candidate using its with-v2-adapter TRAIN scores. Prefer candidates
whose PPL is <=1.01x old-SQ+v2 and MK is no more than2/48 lower. Among eligible
candidates choose lowest PPL, then highest MK, then the candidate order above.
If none is eligible, the lowest-PPL candidate may receive one diagnostic full
validation if its TRAIN PPL is <=1.25x baseline; otherwise stop the family at
screening. Record this as diagnostic advancement, not a screening pass. Do not
change candidates, metadata or selection after observing CONFIRM outcomes.

## Full confirmation and repair gate

Run old SQ+v2, selected candidate+v2, then restored old SQ+v2. Each uses all130
WikiText-2 validation windows/264764 targets and384 normal+384 target-removed
CONFIRM prompts. Require the first parent to exactly replay archived v2 window
NLLs and all generated IDs; require the final parent to exactly repeat it.
Record calibration, codec, table, adapter, source, data and external backend
hashes plus actual storage allocation. Archived original S16 provides explicit
context, not a fresh full arm.

The primary **same-budget repair gate** requires PPL at least1% lower than the
parent, no observed normal-MK decrease, paired-bootstrap MK95% lower bound at
least-2 percentage points (10000 draws,seed20260928), and no extra cache bytes.
Separately report whether PPL is within1% of original S16. Any failed candidate
remains a failed result. CPU evidence audit must independently reconstruct all
score arithmetic, selection, packing/memory receipts and both full replays.
No public release or claim of untouched downstream generalization is included.
