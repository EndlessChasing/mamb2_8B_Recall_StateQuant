# Same-budget SQ3.25 repair experiment

Experiment date: 2026-09-28. This follows the frozen
[v4 protocol](STATE_REPAIR_PROTOCOL.md).

**Status: calibration, implementation checks, TRAIN screening and its independent
CPU audit are complete. Full PPL/MK confirmation is running.** Readout-aware
tiers improve TRAIN PPL but lose recall; neither dense Q3 variant improves
quality. No candidate passes the screening gate. Full diagnostic confirmation
of readout-aware tiers follows the frozen advancement rule.

## Scope and fixed controls

This experiment tests three state-cache candidates on the original pure
Mamba2-8B source. All 507 original tensors remain FP16 and frozen. The same
frozen **v2 1,536-update Resurface adapter** is used wherever an adapter is
present. **No adapter training or optimizer updates occur in v4.** There is no
Q8, weight quantization, rotation or selection among different adapters.

The original state codec, controller, training scan and other existing core
modules are preserved. All 11 `mamba2_recall` files recorded in the v2 training
receipt still match their recorded SHA256 values. New calibration and dense
codec implementations live in separate files.

Native projections, convolution, gating and FP16 residual orchestration remain
unchanged. The RMSNorm backend is pinned to the existing 16-warp, three-stage,
one-CTA configuration before model execution, following the
[replay clarification](RESURFACE_MORE_BACKEND_REPLAY.md). Full control replay
is still required before interpreting a new quality result.

| Candidate | Carry allocation | Coordinate metadata |
| --- | --- | --- |
| `readout_tiers` | Existing 16 INT8 + 64 INT4 + 48 zero-carry slots | New TRAIN readout-score permutation |
| `dense3` | All 128 slots use symmetric three-bit codes | Existing v2 magnitude permutation |
| `dense3_equalized` | All 128 slots use symmetric three-bit codes | TRAIN power-of-two exponents in original coordinates |

Dense Q3 retains a storage slot for every coordinate and removes the fixed
48-coordinate zero-carry tier. It also removes the original INT8 protection
for the 16 outlier coordinates. Each coordinate can still round to zero; keeping
all slots does not preserve every value or guarantee better recall/PPL.

## Equal memory budget

Every candidate uses **52 bytes per 128-coordinate row**, including scales:

| Format | Code payload | Scale payload | Total |
| --- | ---: | ---: | ---: |
| Existing/readout tiers | 16 INT8 + 64 INT4 = 48 B | Two FP16 scales = 4 B | 52 B |
| Dense Q3 variants | 128 three-bit codes = 48 B | Two FP16 scales = 4 B | 52 B |

This is `52*8/128 = 3.25 bits/element`. Dense codes are physically packed into
three 16-byte bitplanes. Signed three-bit values use levels -3 through +3;
code -4 is unused, and code zero represents zero exactly. The two scale groups
contain 64 coordinates each.

Quantization uses an FP32 max-absolute denominator divided by three, clamped to
at least 1e-8, half-away-from-zero rounding and clipping to [-3,3]. The next
carry is decoded with the actual FP16 scale stored in the cache. Each token's
current readout uses the updated FP32 state before that carry quantization.

| Batch-one persistent allocation | Bytes |
| --- | ---: |
| SSM codes | 22,020,096 |
| SSM FP16 scales | 1,835,008 |
| FP16 convolution cache | 4,587,520 |
| One 56×8×128 one-byte table | 57,344 |
| **Total** | **28,499,968** |

The total is **27.1796875 MiB**, equal to the existing SQ3.25 cache. Dense
inference does not keep a dense FP16 shadow carry. The equalizer table replaces
the permutation table; both are never resident together for a candidate.
Offline calibration stores alternative tables separately and its workspace is
outside the inference allocation above.

Base weights remain approximately **16.474 GB**. Adapter tensors, temporary
activations, registers and allocator reserves are outside the cache table.
This experiment changes state representation, not base weight size.

## TRAIN calibration

Calibration used the unadapted original model with S16 carried state, on the
first eight rows of the pinned WikiText-2 TRAIN tensor, first 512 tokens per
row: **4,096 tokens**, with independent zero resets. Validation and CONFIRM
samples were excluded. Statistics were aggregated over time, grouped heads,
channels and batch for each of 56 layers and eight B/C groups.

The readout permutation ranks coordinates by
`mean((decay_t * h_prev_fp16 * C_t)^2)`, with stable coordinate-index ties.
This is an estimate of the next readout effect of discarding carried history.
It omits cross-coordinate error terms and subsequent recurrence effects, so
it is not an exact loss-sensitivity measure.

| TRAIN readout-score share assigned to the 48 zero-carry slots | Share |
| --- | ---: |
| Existing magnitude permutation | 0.739246% |
| New readout-score permutation | 0.284000% |

The proxy decreases by **61.58% relative**, or 0.455246 percentage points of
total readout-score mass. This is a calibration statistic optimized on those
TRAIN samples. **It is not a 61.58% PPL improvement**, a measured recall gain,
or a full decomposition of quantization error.

Equalizer exponents are derived from mean absolute S16 carry `m`:
`k = clamp(round(log2(max(m,1e-12)) - mean_coordinate_log2), -8, 8)`.
Offline rounding is ties-to-even. The measured table's actual range is
**[-4,6]**, within the fixed supported bounds [-8,8]. The codec stores
`u=h/2^k`, applies `B'=B/2^k` and `C'=C*2^k` in FP32 inside the recurrence after
convolution and SiLU, and uses original coordinate order. It adds no history
or projection-weight transformation.

## Implementation validation

The calibration collector passed its synthetic grouped/batched tests and the
independent approximate FP32 readout-score oracle. On a **128-token full-model
TRAIN probe**, adding collection preserved the final hidden output, all 56 SSM
carries, all 56 convolution caches and mean-absolute statistics **bitwise**
relative to the unchanged S16 implementation.

The new dense codec passed **36/36 GPU checks** on the RTX PRO 6000 Blackwell
Server Edition:

- Full-sequence, segmented and tokenwise outputs and packed caches match
  bitwise for both permutation and equalizer modes.
- Independent CPU arithmetic and elementwise bit packing match controlled
  one-token fixtures, including negative values, half-integer rounding ties,
  zero inputs, FP16 scale rounding, scale underflow and large finite values.
- Tests cover batch two, odd and single-channel counts, grouped B/C,
  both exponent extremes (-8 and +8), and exact persistent byte counts.
- Boundary validation rejects nonfinite FP16 scales and readouts. Finite
  inputs can overflow a scale after equalizer amplification; such a candidate
  must be rejected during screening.
- Scale underflow to zero is recorded as a representation effect. It can erase
  a small carried value even when inputs are finite; it is not an overflow or
  an implementation failure.

The first dense test attempt found a Triton compile-time loop-variable scope
issue. Initial scale loading was moved before the loop in the new codec; the
failed receipt and source/log snapshots were preserved. The passing receipt
binds the corrected code hashes. These checks validate the specified arithmetic
on bounded probes and do not establish full-model quality.

## Completed TRAIN screening

The fixed TRAIN screen uses rows 8–15, first 512 tokens each: **4,088 prediction
targets**, plus **48 normal numeric TRAIN prompts**. All three candidates are
compared both without and with the same frozen v2 adapter, alongside original
S16, old SQ3.25, and old SQ3.25 plus v2 controls. The old SQ plus v2 control is
repeated after the candidates. These TRAIN examples were exposed during prior
adapter training and are a development screen, not generalization evidence.

| State codec | Adapter | TRAIN PPL | Normal TRAIN MK |
| --- | --- | ---: | ---: |
| Original S16 | None | 10.625768 | 19/48 |
| Old SQ3.25 | None | 11.825468 | 5/48 |
| Old SQ3.25 | Frozen v2 | 10.814218 | 37/48 |
| Readout-aware tiers | None | 11.487385 | 3/48 |
| Readout-aware tiers | Frozen v2 | **10.524672** | **30/48** |
| Dense Q3 | None | 45.377046 | 0/48 |
| Dense Q3 | Frozen v2 | 39.670536 | 0/48 |
| Equalized dense Q3 | None | 44.067662 | 0/48 |
| Equalized dense Q3 | Frozen v2 | 43.611200 | 0/48 |
| Restored old SQ3.25 | Frozen v2 | 10.814218 | 37/48 |

Readout-aware tiers with v2 improve TRAIN PPL by **2.67746%** relative to old
SQ plus v2, but lose **7/48** correct answers. The fixed screen permits at most
two fewer answers, so **no candidate passes screening**. All candidates are
finite and retain the exact byte budget. Under the predeclared fallback,
readout-aware tiers alone advance diagnostically as the lowest-PPL valid
candidate within 1.25 times baseline PPL. This is not a successful repair gate.

Restoring old SQ plus v2 exactly reproduces all eight window NLLs and all 48
generated sequences. The independent CPU audit reconstructs every raw score,
calibration table, cache allocation and selection decision; **14 boundary
fixtures pass**, and CUDA remains uninitialized during the audit.

See [raw screen comparison](../reports/state_repair_screen/screen_comparison.json)
and [independent audit](../reports/state_repair_screen_audit.json).

### What the TRAIN results suggest

An additional [CPU-only descriptive analysis](../reports/state_repair_screen_analysis.json)
reconstructs paired outcomes and table changes without opening validation or
CONFIRM reports. With v2, readout-aware tiers improve seven of eight TRAIN PPL
windows, with four MK gains and eleven regressions. N=16 changes from 23 to
20 correct out of 24; N=64 changes from 14 to 10 out of 24. Without an adapter,
readout-aware tiers also lose two net answers (5 to 3 out of 48). An adapter
distribution shift alone therefore cannot explain every observed difference.

The readout permutation changes 41.18% of coordinate tier assignments. The
matrix below aggregates all 56 layers, eight B/C groups and 128 coordinates;
rows are old tiers and columns are new tiers. These are table entries, not
independent samples or dynamic state magnitudes.

| Old tier → new tier | INT8 | INT4 | Zero carry |
| --- | ---: | ---: | ---: |
| INT8 | 4,368 | 2,448 | 352 |
| INT4 | 2,018 | 17,432 | 9,222 |
| Zero carry | 782 | 8,792 | 11,930 |

Only 60.94% of old INT8 assignments remain INT8; 4.91% are erased. Preserving
the old INT8 protection while optimizing the remaining tiers is a prospective
repair hypothesis. A separate TRAIN-only protocol would be required to test
it or to readapt Resurface to a changed table. Neither is measured here.

The dense failures are consistent with a precision tradeoff: for the same
state row outside the denominator-floor regime, the first dense block uses
`max(abs(first64))/3`, at least `127/3` (42.33) times the old INT8 step
`max(abs(first16))/127`. A value below one sixth of its block maximum can round
to zero, and the error is carried into later token updates. This comparison
explains a mechanism; it does not isolate the causal contribution in the full
model. Both dense variants also fail without an adapter. Independent code and
metadata inspection found no demonstrated mismatch with the specified format.

## Full confirmation — pending

The selected candidate's full confirmation uses 130 WikiText-2 validation
windows / 264,764 prediction targets, 384 normal CONFIRM prompts and 384
target-removed controls per arm. The parent must reproduce archived per-window
NLLs and generated IDs, and reproduce again after restoring it.

The final same-budget repair gate requires at least 1% lower PPL than old
SQ plus v2, no observed normal-MK decrease, a paired MK bootstrap 95% lower bound
of at least -2 percentage points, and no extra cache bytes. Restoration of
original S16 PPL is reported separately. Full confirmation is still pending.

The fresh parent has completed and reproduced all 130 archived window NLLs and
all 768 generated token sequences exactly. The candidate's 130-window PPL is
**8.258201**, versus **8.388906** for old SQ plus v2, an improvement of **1.5581%**.
Candidate MK, final parent restoration and the full independent audit remain
pending, so the joint repair gate has no final outcome yet.

## Artifacts and provenance

### Current calibration and implementation evidence

- [Calibration payload](../reports/state_repair_calibration/calibration.pt) and
  [calibration receipt](../reports/state_repair_calibration/calibration.json),
  including the full-model collector parity check and table identities.
- [Collector checks](../reports/state_repair_collector_checks.json).
- [Passing dense-codec checks](../reports/state_repair_dense3_checks.json) and
  [preserved first-attempt failure](../reports/state_repair_dense3_checks_attempt1.json).
- [Dense codec](../mamba2_recall/state_codec_dense3.py),
  [dense controller](../mamba2_recall/state_quant_dense3.py), and
  [independent codec checker](../scripts/check_state_codec_dense3.py).

### Frozen control artifacts

- [v2 adapter](../reports/quant_first_v2/training/adapter_fp16.pt) and
  [v2 training receipt](../reports/quant_first_v2/training/report.json).
- [Original calibration](../reports/quant_first_v2/calibration/calibration.pt).
- Archived full controls:
  [original S16](../reports/quant_first_v2/evaluation/full_source_s16.json),
  [old SQ3.25](../reports/quant_first_v2/evaluation/full_source_sq3p25.json), and
  [old SQ3.25 plus v2](../reports/quant_first_v2/evaluation/full_resurface_sq3p25.json).
  These are archived context until the required fresh replay finishes.

| Artifact | SHA256 |
| --- | --- |
| Frozen v4 protocol | `f18069d0340449316297077a2d912f7ef42fbf518436d9b311d8a731c96d722c` |
| Calibration payload | `8509bb266d40875608f3e0b3be22407fd6ea1b8aacbc79ce04e958726516b0a4` |
| Calibration receipt | `a960c6518102c09c945b68903b5f556d103c62476d5a13709689035dc52ad4bb` |
| Collector-check receipt | `6c6134a8d08ade38f1aa8ca331c677bc6d156e9bd56cb68930e98f80f77f5b5b` |
| Passing dense-codec receipt | `cebc92d93ef010a308c3816ba608f6ea9df8e9ad252e0951cc22281ed780be03` |
| Dense codec | `b2643076540ec67f4b879ba9dc2c512fdbfdf3c3db7c957902002e1bc5df0a05` |
| Dense controller | `946bdb08d1557d364e8b1f4264412dffcd2481227daee4bfdba6a5238729f9fd` |
| Frozen v2 adapter | `7144b88265a797ef935b1f94845395abec5532d8dc5b2a4a8fa0a8e07ce205f0` |

## Limits

The source is `nvidia/mamba2-8b-3t-4k`, with its pinned BF16 checkpoint cast to
FP16 in the native runtime. Source identity/version/gradient guards are not
full post-execution byte hashes of every GPU weight. The new collector's
128-token model check does not establish equality at arbitrary lengths.

The dense format trades per-coordinate precision for coverage; it may amplify
recirculating error despite preserving all 128 slots. The old adapter was
trained with the old tiered state format and is kept frozen in this experiment.
No dense-codec surrogate training gradient or adaptation result is included.

Both the TRAIN screen and benchmark families have historical project exposure.
The independent CPU result audit, when complete, will reconstruct recorded
evidence rather than rerun GPU logits. No broad downstream, long-context,
throughput, ASIC-energy or unseen-generalization claim follows from the current
implementation checks. No public release is included.
