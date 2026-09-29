# v6: prioritize unadapted PPL, then use fresh Resurface for MK

Prospective protocol, 2026-09-29. User instruction: prioritize PPL optimization;
MK is a later Resurface objective. Freeze this document before any v6 model
measurement. Preserve v1-v5 code, protocols and artifacts. All new implementation
files live in scripts/ so prior module inventories remain unchanged.

## Fixed source and accounting

Use the original pure nvidia/mamba2-8b-3t-4k source, not quantized weights:
source SHA256 47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb,
tokenizer SHA256 5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09.
All 507 source tensors / 8,236,999,680 parameters remain frozen FP16. Pin the
existing 16-warp RMSNorm forward backend before loading any model. Record
environment, source/code hashes, precision flags and frozen-source guards.

Every deployable candidate uses the existing physical layout: 16 INT8 + 64
INT4 + 48 zero-carry coordinates per 128 values, plus two FP16 scales. Codes
occupy 48 bytes and scales four bytes per row: exactly 3.25 bits/element.
One uint8[56,8,128] table is 57,344 bytes. Batch-one persistent cache, including
FP16 convolution, must equal 28,499,968 bytes. No additional resident lookup,
residual or predictor tensors are allowed. Global clipping factors are compile-
time scalar policy constants. Count actual allocations; exclude diagnostics
from deployment claims. Current-token readout precedes carry quantization,
which occurs after every token, including prefill.

## Fixed twenty-candidate search

Reuse the four independently validated v5 tables in their fixed order:
magnitude, full_readout, preserve_int8, preserve_retained80. Candidate artifact
SHA256 cd86a755db5004c716922696cf5532b307c57f5fb7dad4bcca298eb58b55f9a7.
Validate the complete v5 selected-calibration provenance as well; its SHA256 is
c525fbf62ef4a72db2d4bb13c13920d9f5d4946485538da0f00a369aab3092e3.

For each table evaluate these five variants, in this order:

1. legacy: call the unchanged old packed codec. Codes are chosen with its FP32
   denominator; the denominator is then stored as FP16.
2. stored_scale: form the denominator, round it to the FP16 value actually
   stored, then choose integer codes using that rounded denominator.
3. clip4_095: stored_scale, but multiply the INT4 block absmax by 0.95 before
   dividing by seven.
4. clip4_090: same, factor 0.90.
5. clip4_080: same, factor 0.80.

INT8 clipping factor is always one. The INT8/INT4 denominator floors remain
1e-8 before FP16 conversion. If that conversion yields zero, stored_scale modes
use zero integer codes and zero reconstructed carry; never divide by zero.
Finite positive scales use half-away-from-zero nearest rounding and clipping
to [-127,127] / [-7,7]. All decoding uses the stored FP16 scale. No claim that
local rounding-error reduction guarantees sequence PPL improvement is assumed.
The fixed factor grid must not expand after observing screen or full results.

Baseline is preserve_int8 + legacy, the completed v5 unadapted model. All
candidates are evaluated without loading or installing any Resurface adapter.

## Diagnostic ablations, not candidates

On the same screen data, measure original S16 and two ablations using the v5
preserve_int8 table:

- prune_only: carry the retained 80 coordinates as FP16; discard the other 48.
- quant_only: retain old legacy INT8/INT4 behavior on those 80 coordinates;
  carry the other 48 in FP16 instead of dropping them.

Diagnostics may use a full FP32 backing tensor to avoid unintended additional
rounding of INT8/INT4 decoded values. Report its actual resident bytes, expected
239,525,888 including convolution and table at batch one. These are neither
3.25-bit candidates nor deployment improvements. An auxiliary legacy_emulation
diagnostic must reproduce the actual packed legacy arithmetic in controlled
kernel tests. Check dtype/rounding semantics independently. Ablation effects
interact through recurrent dynamics; do not sum their PPL changes as independent
error contributions or treat them as theoretical lower bounds.

## TRAIN-only screen and PPL-only selection

Use the pinned TRAIN token file SHA256
e54b02e5162e042a9cdd504f4eb1b1652724fb240bbc2c97608967aa26297233.
Select rows 40..71, all 2048 tokens per row, with independent resets:
32 windows and exactly 65,504 next-token targets. These rows are disjoint from
v5 calibration rows 0..7 and screen rows 8..39, though the TRAIN corpus and
benchmark family have historical exposure. Record each token-window hash.

Execution order: baseline, the three diagnostics, then the other 19 candidates
in table-major/variant-minor order, then restored baseline. Repeat a 128-token
reset/hidden/cache probe for every arm, checking finiteness and exact storage.
Store raw window NLLs and use summed NLL / target count to calculate PPL. No new
MK generation/scoring, adapter loading or held-out validation input is used in
this stage. Historical v5 artifacts may be read solely to verify the fixed
input provenance; their existing MK fields never enter the v6 selection rule.

Deployable candidates require complete finite scores, exact byte budget,
passing reset probe and source/backend/table freeze checks. A known candidate
nonfinite failure is recorded and excluded; baseline failure, arbitrary runtime
errors or failed integrity checks stop the experiment for diagnosis. Never
silently continue after a compiler/runtime error. Diagnostics are ineligible;
any diagnostic nonfinite value or integrity failure stops this run with its
failed evidence preserved, rather than supplying an invalid error attribution.

Select lowest TRAIN PPL, preferring baseline on an exact PPL tie, then the
fixed candidate order. There is no MK guard, MK tie break or recall threshold.
Record exact restored-baseline window NLLs and export the selected table plus
variant and all input/selection hashes. If baseline wins, stop v6 with a
negative optimization result; preserve v5 and do not repeat its training.

## Full PPL confirmation

Only the frozen TRAIN-selected nonbaseline candidate advances. Evaluate:
v5 baseline, selected candidate, restored v5 baseline. Each arm uses the pinned
WikiText-2 validation token stream
5bbeae08ba8eb34a482f3b6e9d17b182e67229dd14b2853d87f89fc72e5ad027,
130 windows / 264,764 targets, the existing 2048-window protocol and full
256,000-token vocabulary. No MK results select or reject this quantizer.

Require baseline and restored-baseline NLLs to exactly match each other and
the archived v5 selected-no-adapter report. Its file hash is obtained from the
pinned v5 full-comparison SHA256
0e5ceb6e91d72a159f46a9a0760a23f3b9301be18bb32590fe01a9196f83d1e9.

Meaningful unadapted improvement gate: candidate PPL <= 0.99 times baseline
PPL, all integrity checks pass, and cache remains exactly 28,499,968 bytes.
Report every measured arm and the S16 gap even when this gate fails. Validation
confirms the one fixed candidate; it must not choose a runner-up or change the
factor grid. If the gate fails, finish this experiment without new Resurface
training and preserve the best previously validated endpoint.

## Subsequent fresh Resurface, conditional on confirmed PPL improvement

Freeze the selected quantizer before training. Reuse the v5 fresh 1536-successful-
update recipe and original numeric/prose TRAIN manifests: no previous adapter,
checkpoint or optimizer initialization. Keep the original source weights and
state table/policy frozen. The separate S16 teacher, paired 512-token prose,
numeric answer CE, prose CE/KL/closure weights, optimizer/scaler recipe and
maximum eight overflow retries remain as documented in STATE_FIRST_V5_PROTOCOL.

For a new scale policy, forward and checkpoint/history recomputation must use
the exact selected deployed codec. Backward remains the explicitly approximate
live-mask STE; it does not differentiate rounding/scales/clipping. A discarded
fresh smoke, actual 128-token FP16 export/packed parity and independent CPU
training audit are mandatory before formal evaluation. Keep all four recovery
checkpoints and use only the final 1536-update export.

Evaluate selected unadapted, selected plus fresh Resurface, and restored selected
on all 130 PPL windows plus 384 normal / 384 target-removed MK prompts. Require
exact unadapted PPL replay against the prior full stage and all generated
sequences to repeat after adapter removal. Resurface repair passes if PPL is
no more than 1% worse than the selected unadapted baseline, normal MK increases,
paired 95% bootstrap lower bound is positive, and cache remains unchanged.
Compare the final endpoint with v5 and original S16 separately, reporting all
PPL/MK tradeoffs rather than assuming recall is recoverable. Paired intervals
use sorted case IDs, 10,000 draws and default_rng seed 20260928.

## Evidence and scope

Freeze protocol and implementation hashes before measurement. Independent
kernel tests cover stored-scale zero/subnormal behavior, code/packing ranges,
segmentation, actual bytes and legacy equivalence. An independent CPU auditor
reconstructs provenance, PPL aggregation, selection, cache and exact replays;
it audits recorded GPU evidence and does not regenerate model logits.
No held-out-driven changes, unmeasured quality forecasts, publication or Q8
experiment are included. Keep the repository private. If numerical integrity
requires a repair, preserve the failed attempt and bind a fresh run to the
repaired source; never overwrite measured evidence.
