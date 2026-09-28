# Quantize first, then train Resurface: protocol v2

Frozen 2026-09-28 before calibration/training of this new candidate. This is a
new experiment. `PROTOCOL.md` and `reports/pilot_v1` remain historical evidence
for attaching an old FP16-trained adapter to quantized state.

## Order and frozen model

1. Load the pure `nvidia/mamba2-8b-3t-4k` checkpoint, cast BF16 to FP16 using
   the pinned native runtime; freeze all 507 tensors / 8,236,999,680 parameters.
2. Calibrate original-coordinate SQ3.25 on the original model with NO adapter.
3. Freeze those permutation/tier tables. Initialize a NEW Resurface adapter:
   V=0, g=1, router_w=0, router_b=-4, soft gates at train and inference.
4. Train only the adapter with the quantized recursive forward path active.
5. Export actual FP16 adapter tensors and evaluate using actual packed cache.

Source revision `b915550c63ba9359f88f44d1f6a600d85af27302`; checkpoint SHA256
`47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb`;
tokenizer SHA256
`5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09`.
No old adapter, weight quantization, Q8, rotation, DCT or z-transform is used.

## New calibration

Use S16 per-token carried state of the original unadapted model and the first
8 rows x first 512 tokens of the pinned 448 WikiText-2 TRAIN windows, 4096 tokens.
TRAIN tensor SHA256
`e54b02e5162e042a9cdd504f4eb1b1652724fb240bbc2c97608967aa26297233`;
manifest SHA256
`facb2ca461615a4199781bd21784d642d6674f5b862641b3b9edac3fb499b89d`.
Reset state per row. Rank coordinates by mean absolute rounded FP16 carried
state, per layer and B/C group, averaging over time, batch, 16 grouped heads
and 64 channels/head; descending stable order breaks ties by coordinate index.
First 16 coordinates INT8, next 64 INT4, last 48 zero carried state.
Save a NEW no-adapter calibration with explicit protocol/source/token binding.

The inference format remains 52 bytes per 128-value row including two FP16
scales: 3.25 bits/element. Current readout uses the FP32 update before carry
quantization; dead coordinates still contribute the current input. Quantization
uses an FP32 max-abs denominator, half-away rounding and FP16 stored scales.

## Training forward and approximate gradient

Training forward must equal the existing packed codec on identical inputs.
Each training example starts from zero convolution and SSM state. Per-layer
checkpoint replay is stateless and cannot reuse a previous invocation's cache.
Keep native projections, convolution, gated RMSNorm and FP16 residual path;
Resurface remains the memoryless post-D/pre-norm correction across 128 heads.

Use a straight-through estimator (STE) for backward only: derivative 1 through
the quantized carry's 80 live coordinates, 0 through the 48 dead coordinates;
ignore derivatives of scale selection, rounding and clipping. Dead coordinates
retain gradients through their CURRENT readout. Gradients must reach x, dt,
B and C (including grouped-head reductions) and earlier adapter layers. This
is a declared surrogate gradient, not a derivative of discrete quantization.
Recompute bounded state checkpoints/chunks in backward; training scratch is
not part of the deployed state-cache budget.

Before fitting require: exact forward versus codec, independent surrogate
gradient tests, finite gradients, identity of new-adapter/no-adapter outputs,
and actual full-model training/inference forward parity. Discard smoke updates.

## Fixed training budget and data

Reuse the original 1536 numeric TRAIN cases, disjoint DEV/CONFIRM key/value
bands and exact generator. Bind the NEW numeric manifest to THIS protocol.
Use seed 2026092803 and one `torch.randperm(1536)` example schedule.
Each update uses full-256K answer-suffix CE plus one 512-token prose TRAIN
segment (511 targets), following the existing 448-window schedule and offsets.
No validation, DEV or CONFIRM losses enter fitting or checkpoint selection.

Load a separate frozen original FP16 model as prose teacher, executing S16
per-token recurrence. No Resurface teacher. Fixed loss:
`MK answer CE + 0.5 prose CE + 0.5 KL(S16 teacher || SQ student) + 3 closure`.
Use temperature 1 and prose-only router closure budget 0.006, excess penalty 10.
This retains the earlier adapter recipe but changes teacher execution to S16
and student execution to the actual quantized recurrence. PPL repair is measured,
not guaranteed by the loss.

Only 1,154,104 adapter master parameters train, in FP32; forward casts to FP16.
AdamW V/g LR 1e-4, router LR 3e-4, betas (.9,.999), eps 1e-8, no weight decay,
clip norm 1, cosine multiplier `0.1+0.9*(1+cos(pi*j/1535))/2`.
Block checkpointing, gradient scale 1024, growth interval 2000; retry identical
examples on overflow, maximum 8 overflow retries / 1544 attempts. Exactly 1536
successful updates, final update is the only candidate; intermediate checkpoints
are for recovery/audit, not selection. Preserve source and permutation identities,
versions and frozen gradients; verify FP16 export and its complete binding.

## Evaluation

Compare THREE current arms with the same serial recurrence and prompt execution:
original S16 without adapter, SQ3.25 without adapter, SQ3.25 with the newly
trained FP16 adapter. The two SQ arms use the same NEW calibration tables.
Evaluate baseline again after adapter removal to test restoration/repeatability.

First run paired DEV screening using the previous fixed four 2048-target windows
[0,32,64,96] and 96 DEV prompts (sample indices [0,9,18,27,36,45,54,63], 48 normal
and 48 removed). This is descriptive checking, not adapter selection; do not
reuse the OLD adapter pilot stop rule to reject the new training experiment.

Evaluate the final adapter and both current baselines on all 130 WikiText-2
validation windows (264764 targets), and all original CONFIRM cases (384 normal
+ 384 removed). Reset each window/prompt. Greedy full-vocabulary generation,
maximum 12 tokens, stop at EOS, first standalone six-digit integer matching.
Record per-window NLL, IDs/digests, generated token IDs/text, cache allocation,
GPU peak memory, frozen-parameter checks and all source/artifact hashes.

Primary repair claim requires MK improvement versus SQ baseline and PPL no
worse than SQ baseline by >1%. Separately report remaining PPL/MK gaps versus
the original S16 model; restoring original PPL requires <=1% relative increase
versus S16. Report paired normal-MK differences with 95% bootstrap intervals
(10000 draws, seed 20260928). No new unseen-generalization claim: these public
evaluation families/corpora have historical exposure. Even a failed candidate
is recorded without redefining the gate after seeing its results.

Deployment cache remains unchanged by the memoryless adapter. Training scratch,
the second teacher model and optimizer memory must never be called deployed
3.25-bit cache. No public release is included in this experiment.
