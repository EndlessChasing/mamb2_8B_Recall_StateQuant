# Frozen StateQuant experiment v1

Frozen before calibration or quality measurement, 2026-09-28.

## Scope and provenance

Pure `nvidia/mamba2-8b-3t-4k`, revision
`b915550c63ba9359f88f44d1f6a600d85af27302`, source checkpoint SHA256
`47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb`.
The native converted runtime casts the original BF16 checkpoint to FP16.
56 layers, 128 heads, 64 channels/head, 128 state coordinates, 8 B/C groups.
All 8,236,999,680 base parameters stay frozen and unquantized.

Use the existing Mamb2_8B_Recall FP16 adapter unchanged:
SHA256 `e8b2b4dfe69f8e85dc9e147c9aeaa3297cff14f4043e558bb1795ad476c1fca0`,
2,374,143 bytes. Its original binding and soft post-D/pre-norm insertion remain
unchanged. No new training. `RESURFACE_PROTOCOL.md` is its historical protocol.

StateQuant reference is the user-supplied archive at commit
`22156f74edf428fd192948d75a0dabf3cf152d48`; archive SHA256
`4743b4d99c1bf2d5d839dba0877e4f5602cba31ec7ca2cfa530a2668511ebaf6`.
The archived implementation is included with its Apache-2.0 license.

## Exactly two state modes

- `s16`: FP16 SSM state, rounded after EVERY token, including prefill.
- `sq3p25`: original-coordinate static tiers: 16 INT8, 64 INT4, 48 dead
  coordinates. NO Q8 experiment, rotations, DCT, or z transform.

For both modes update in FP32, compute the current readout from the updated
FP32 state plus D*x, and then compress the state carried into the next token.
The implementation may fuse a sequence into a serial Triton loop, but must
apply the same carried-state compression between every pair of tokens.
Resurface sees the current readout, with identical frozen gates.

For each live tier the quantizer denominator is FP32
`max(max(abs(h))/qmax, 1e-8)`; round half away from zero and clamp to +/-127
or +/-7. Persist FP16 scales and decode those actual rounded scales next time.
Zeroed coordinates still contribute the current B*dt*x update to readout.
INT8 coordinates use two nibble arrays (8+8 bytes); INT4 uses 32 bytes.
Two FP16 scales add 4 bytes: 52 bytes per 128-element state row, 3.25 bits.
No dense FP16 SSM shadow may persist. Conv history remains FP16.

## TRAIN-only permutation calibration

Use the existing pinned WikiText-2 TRAIN token tensor: first 8 rows, first 512
tokens each (4096 tokens total), with zero state at each row and adapter active.
Tensor file SHA256
`e54b02e5162e042a9cdd504f4eb1b1652724fb240bbc2c97608967aa26297233`;
the copied provenance is `prose_train_manifest.json`.
Collect the sum of abs(FP16-rounded carried state), per layer and original
state coordinate, averaged over tokens, channel dimension, batch, and all
16 heads sharing a B/C group. Stable descending order gives INT8 first,
INT4 next, dead last; tied coordinates retain ascending original index.
Record token digests, statistics, and actual uint8 permutation file hash.
No validation, DEV, or CONFIRM examples may enter calibration.

## Correctness and execution controls

Before quality: packed-byte accounting; independent decoder/update oracle;
whole versus segmented kernel equality; causality; native single-step S16
comparison on identical inputs; whole-model native recurrent S16 probe.
Native prefill convolution uses Conv1d, recurrent convolution uses elementwise
FP16 multiplication and reduction on this host. Whole-sequence projections and
tokenwise GEMMs may differ. Record these execution differences separately.
S16 and SQ3.25 quality runs share identical projections, convolution, readout,
adapter and serial-scan orchestration. Do not reuse historical parallel SSD
PPL 7.0520635 as the sequential S16 baseline.

## Quality selection and stop rules

WikiText-2 raw validation revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, original tokenizer SHA256
`5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09`.
Join documents with two newlines; no automatic BOS/EOS. Disjoint stride-2048
windows, at most 2048 targets/window; reset state at every window. Score all
256000 logits, FP32 cross entropy, chunks of 64 logits tokens.

Pilot PPL uses full-window indices [0,32,64,96], 8192 targets.
Pilot MK uses the immutable 64-sample DEV generator, selecting sample indices
[0,9,18,27,36,45,54,63] in each N=16/64 x three-template cell, including paired
target-removed controls: 48 normal + 48 removed prompts. Greedy <=12 tokens,
stop at EOS; first standalone six-digit integer must equal the target.
Store prompt IDs/digests, output text and generated token IDs.

Stop this fixed candidate if nonfinite, correctness failure, pilot PPL rises
more than 5% against paired S16, or pilot normal MK falls more than 10 percentage
points. These are engineering screening rules, not statistical equivalence.
If stopped, publish an honest negative result without a full-corpus claim.

Only surviving candidates advance to all 130 validation windows (264764 targets)
and the original CONFIRM set (384 normal + 384 removed). Final gate: PPL rise
<=1%, and paired normal-MK noninferiority within 2 percentage points (report
95% paired bootstrap confidence interval with fixed seed 20260928; 10000 draws).
An inconclusive interval is not a pass. CONFIRM has historical exposure and is
not a newly unseen generalization test. Report removed-target counts separately.

## Storage and receipts

Count actual persistent SSM payloads/scales, conv history and uint8 permutations.
At batch 1 the predicted S16 total is 122028032 bytes; SQ3.25 total is 28499968
bytes including 57344 bytes of shared permutation tables. These are format
budgets until verified on actual allocations. Report GPU allocator peaks and
temporary calibration/scan workspace separately. Weight memory stays unchanged.
Record code/protocol/source/adapter/calibration hashes, environment, raw scores,
timing, and parameter identity/version/gradient checks before and after runs.
