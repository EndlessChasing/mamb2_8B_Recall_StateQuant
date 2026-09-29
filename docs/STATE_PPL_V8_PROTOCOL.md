# v8: per-layer table mixing for unadapted Q3.25 PPL below 8.25

Prospective continuation, frozen before measurements. Run only if v7 does not
meet the user's full unadapted PPL target <8.25. This route keeps the verified
v6 stored-scale codec exactly unchanged; it changes only the existing static
uint8[56,8,128] permutation. Do not initialize from an adapter or train weights.

## Fixed inputs and accounting

Use the same original pure Mamba2-8B source, tokenizer, frozen FP16 parameters,
16-warp RMSNorm backend and dataset/token definitions as v7. Source SHA256
47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb;
tokenizer 5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09.
TRAIN file e54b02e5162e042a9cdd504f4eb1b1652724fb240bbc2c97608967aa26297233.

Reuse the four verified v5 tables (payload
cd86a755db5004c716922696cf5532b307c57f5fb7dad4bcca298eb58b55f9a7),
their provenance, and exact v6 stored_scale/clip1 implementation. The baseline
is preserve_int8 in all56 layers, selected v6 artifact
098930d1af5e5821b277640d236f7607c6428b48d36f7117e2ea4aa87e656303.
V6 full baseline report
c3525d74ba0ee35e0b9d83a029e645123d14e90f5ca32b164931d24ec8359253,
comparison c47e381e02edd6e33ba2b0a85a4f9c65d702cc7d3d93dacfe7546d409e6612d4.
Validate existing input receipts/hashes rather than silently inventing new tables.

Physical format stays 16INT8/64INT4/48zero + twoFP16 scales =52B per128
coordinates. One permutation table is57,344B; total persistent batch-one cache
is28,499,968B including FP16 convolution. No new resident tensors/predictors,
extra FP16 state, modified reset frequency, MK scoring or Q8. Current readout
precedes every-token carry quantization, including prefill.

## One-layer TRAIN calibration

Use TRAIN rows72..79, all2048 tokens: eight windows/16,376 prediction targets.
These rows do not overlap v6 selection or v7's80..111 selection. Execute the
baseline, all168 single-layer interventions, then restored baseline (170arms).
For layer0..55, replace only that layer's table with each alternative in order
magnitude, full_readout, preserve_retained80. Every other layer remains baseline.
Each intervention evaluates the actual complete model on all eight windows.
Record each raw NLL/token hash, all128-token reset/hidden/cache probes, actual
bytes and source/backend/table guards. No proxy score selects the interventions.

For each layer choose its lowest aggregate NLL alternative, exact ties fixed
alternative order. Retain it only when NLL is strictly lower than baseline.
Rank retained swaps by (deltaNLL, layer index, alternative order). Export the
baseline and combined top-k tables for k=1,2,4,8,16,all negative layers in that
order; k is capped by the retained count. Deduplicate identical table hashes,
keeping the first label. If no layer improves, export only baseline and record
this negative family result. Do not claim individual NLL effects are additive.

Baseline/nonfinite or arbitrary runtime/integrity failure stops for diagnosis.
A known single-layer nonfinite result may be excluded with failed evidence;
its layer may still choose another complete alternative. Do not exclude finite
poor results. Restored baseline must exactly repeat all window NLLs and probes.
Export full raw-arm hash inventory, swap decisions and reconstructible table
payloads with upstream/source/code/protocol hashes. Preserve failed attempts.

## Disjoint TRAIN screening, then full confirmation

Use TRAIN rows112..143,32 complete2048-token windows/65,504 targets. Execute
baseline, each unique combined top-k candidate, then restored baseline. Each
arm has the same reset/finite/integrity/memory gates as calibration. Select
lowest complete finite aggregate TRAIN PPL; exact ties baseline then candidate
export order. The layer scores only propose combinations; this disjoint full
model screen must establish each combination's performance. Never choose a
combination using validation/MK. Freeze/export one selected table and receipts.

If baseline wins, continue a separately recorded next method under the user's
request. Otherwise evaluate only the frozen winner, v6 baseline and restored
v6 baseline on all130 WikiText-2 validation windows/264,764 targets. Require
exact archived v6 and restored NLLs and128-token hidden/cache hashes. Target
passes only if candidate PPL is STRICTLY BELOW8.25, all integrity checks pass,
and cache is exactly28,499,968B without an adapter. There is no1% continuation
gate. Compare original S16 PPL7.334322057221965 separately.

An independent CPU audit reconstructs score arithmetic, table swaps, exported
tables, selection and replay from raw reports, without rerunning GPU logits.
These benchmark/data families have historical exposure; do not describe them
as untouched generalization evidence. Do not alter this search/grid from full
validation results. If unsuccessful, preserve evidence and continue a distinct
TRAIN-selected family; do not infer impossibility. Keep the repository private.
