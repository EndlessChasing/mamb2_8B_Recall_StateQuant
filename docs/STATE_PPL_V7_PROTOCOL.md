# v7: unadapted Q3.25 PPL below 8.25

Prospective protocol, 2026-09-29. The user explicitly requests continued
optimization until no-Resurface Q3.25 full PPL is strictly below 8.25. This
replaces the previous experiment's 1% continuation gate. Preserve v1-v6 code
and evidence. No adapter, MK objective, weight update or Q8 experiment.

## Fixed source, data and memory

Original pure nvidia/mamba2-8b-3t-4k, revision
b915550c63ba9359f88f44d1f6a600d85af27302. Source SHA256
47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb;
tokenizer SHA256
5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09.
All source parameters remain frozen FP16. Pin the existing 16-warp RMSNorm
configuration before model loading, TF32 disabled, and retain source identity,
version and gradient guards. These guards are not full post-run content hashes.

Each row keeps 16 INT8, 64 signed 4-bit codes, 48 zero carried coordinates,
and two FP16 scales: exactly 52 bytes per 128 elements, or 3.25 bits including
scales. The static uint8[56,8,128] permutation occupies 57,344 bytes. Total
batch-one persistent cache, including FP16 convolution, is 28,499,968 bytes.
No extra resident lookup, predictor, error feedback or shadow carry is allowed.
Codebook constants compile into arithmetic; their instruction/register use is
not a hidden persistent tensor. Report weights/temporary compute separately.
Read out the current FP32 update, then quantize carry after every token,
including prefill. Do not increase reset frequency.

## Fixed sixteen candidates

Reuse four frozen v5 tables, in order: magnitude, full_readout, preserve_int8,
preserve_retained80. Candidate payload SHA256
cd86a755db5004c716922696cf5532b307c57f5fb7dad4bcca298eb58b55f9a7.
Validate its existing upstream calibration/provenance. For each table test
these four INT4 codebooks in this fixed order:

1. uniform: [0,1,2,3,4,5,6,7]; delegate v6 stored_scale exactly.
2. mild: [0,0.5,1.5,2.5,3.5,4.5,5.5,7].
3. quadratic: [k*k/7 for k in 0..7].
4. fp4like: [0,7/12,7/6,7/4,7/3,7/2,14/3,7].

Each magnitude list is converted to fixed FP32 constants before GPU use;
record their bit patterns. Codes retain the old signed nibble convention
(-7..7, unused -8). Decode sign(code)*level[abs(code)]*stored_scale with
FP32 multiplication, then carry that actual reconstructed FP32 value.
The two scales use absmax/127 and absmax/7, correctly rounded FP32 division,
floor 1e-8, then FP16 storage. No clipping. A zero stored scale implies code
zero. INT8 is unchanged from v6 stored_scale. For nonuniform INT4, choose the
nearest actual FP32 reconstructed magnitude; compare adjacent midpoints in
FP64 and resolve exact ties away from zero. No normalization approximation
may silently change this rule. All codebooks preserve the maximum level seven.

Baseline ID is preserve_int8__uniform, exact v6 selected-no-adapter behavior.
Freeze new source and passing kernel evidence before model measurement. Keep
all new files under scripts/ so old module hash inventories remain unchanged.

## Necessary numerical checks

Verify independent CPU code selection/packing/decode, zero/subnormal scales,
positive/negative midpoint ties and adjacent representable inputs, unused
codes, finite-value failures, actual allocations and constexpr-only levels.
Verify controlled multi-token outputs and carries, random per-kernel
full/segmented/tokenwise equality, and exact uniform delegation to v6.
State clearly any CPU/GPU transcendental/reduction comparison scope; do not
require a different reduction tree to be bitwise identical on arbitrary data.
Retain failed receipts and matching source if implementation repair is needed.
No model-quality measurement may precede passing relevant kernel checks.

## TRAIN-only screen

Pinned TRAIN token file SHA256
e54b02e5162e042a9cdd504f4eb1b1652724fb240bbc2c97608967aa26297233.
Use rows 80..111, all 2048 tokens each: 32 windows / 65,504 targets, disjoint
from v6 selection rows 40..71. Each window independently resets as before.
Execute baseline, other fifteen candidates in table-major/codebook-minor
order, then restored baseline. No adapter or MK measurement anywhere.
Record every raw window NLL and token hash. Repeat the 128-token reset probe,
checking exact hidden/cache hashes, finite values, actual byte budget and
source/backend/table integrity for every arm.

A known nonfinite candidate is recorded and excluded; baseline nonfinite,
compiler/runtime errors and integrity failures stop the run for diagnosis.
Do not swallow arbitrary exceptions. Select minimum aggregate TRAIN PPL;
exact ties prefer baseline, then fixed candidate order. Export the selected
table, codebook and all selection/input/code hashes. The restored baseline
must repeat every NLL and reset/cache value exactly. If baseline wins this
family, retain it and pursue a separately recorded next family.

## Full confirmation and completion criterion

Only the frozen TRAIN-selected nonbaseline candidate advances. Evaluate v6
baseline, selected candidate and restored v6 baseline on all 130 WikiText-2
validation windows / 264,764 targets, the existing 2048-window protocol and
full 256,000-token vocabulary. Validation token SHA256
5bbeae08ba8eb34a482f3b6e9d17b182e67229dd14b2853d87f89fc72e5ad027.
Pin the v6 full selected report and comparison by hashes in the implementation.
Baseline full window NLLs and 128-token hidden/cache hashes must exactly
reproduce v6; restoring baseline must reproduce them again.

Success requires PPL STRICTLY LESS THAN 8.25, no adapter, unchanged memory,
all integrity checks and independent CPU evidence audit. CPU audit recomputes
raw-score arithmetic/provenance; it does not regenerate GPU logits. Report
original S16 PPL 7.334322057221965 and its 122,028,032-byte cache separately.
Do not choose runner-ups or modify this grid using full-validation scores.
If this family misses the target, continue a distinct TRAIN-selected family
under the user's instruction; do not interpret a missed target as impossible.

## Scope and continuation

Historical validation/corpus exposure must be disclosed; the repeated benchmark
is not an untouched generalization set. Do not claim MK recovery from PPL.
Keep the repository private. No Resurface training is part of this request.
A planned independent next route mixes existing tables by layer using TRAIN
calibration, without extra runtime tensors. Its exact search and data split
will be frozen before that route's measurements. Preserve every attempted
candidate and failure rather than replacing reports in place.
