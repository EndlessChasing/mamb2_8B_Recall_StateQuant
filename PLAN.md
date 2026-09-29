# Checklist

## Historical experiment v1: old Recall adapter, then SQ3.25

- [x] Select original FP16 Mamb2_8B_Recall and unchanged adapter.
- [x] Skip Q8; freeze S16 versus original-coordinate SQ3.25 protocol.
- [x] Preserve base/adapter/archive identities and license scopes.
- [x] Implement real packed state and Resurface-compatible runtime.
- [x] Pass codec oracle, segmented recurrence and native controls.
- [x] Calibrate group-aligned tiers using 4096 TRAIN tokens only.
- [x] Measure actual persistent cache and GPU allocator peaks.
- [x] Run paired pilot PPL and MK, recording raw outputs.
- [x] Apply predeclared stop rule: PPL +13.95%, MK 47/48 to 19/48; stop.
- [x] Independently audit receipts, document results, create private GitHub repository.

Full validation is intentionally not advanced because the pilot failed both
screening thresholds. Q8 remains excluded. See docs/RESULTS.md.

No adaptation training or Hugging Face release is included in this experiment.

## New experiment v2: SQ3.25 first, fresh Resurface second

- [x] Freeze a distinct protocol and preserve v1 evidence.
- [x] Recalibrate on the original unadapted S16 base using TRAIN only.
- [x] Implement exact quantized forward and masked STE backward.
- [x] Validate packed-forward parity, surrogate gradients and stateless replay design.
- [x] Pass full 8B identity and one-step training smoke.
- [x] Train a fresh adapter for 1536 successful updates under SQ3.25 (1542 attempts).
- [x] Export FP16 and verify actual packed inference parity on 128 tokens.
- [x] Evaluate original S16, unadapted SQ3.25, trained SQ3.25 and restored SQ baseline.
- [x] Independently audit and document PPL/MK/cache results.

Full v2 result: SQ3.25 PPL **8.69155 → 8.38891**, normal MK **50/384 → 239/384**.
The repair gate against unadapted SQ passes. Original S16 is **7.33432 / 146/384**;
its PPL is not restored (candidate is 14.38% higher). State cache remains
**27.1797 MiB/sequence**, 76.64% below S16. Entire SQ baseline replay and
independent evidence audit passed. See docs/QUANT_FIRST_RESULTS.md.

See docs/QUANT_FIRST_PROTOCOL.md. No Q8 or old-adapter initialization; no public release.

## Continuation v3: more Resurface training

- [x] Freeze a separate continuation protocol; preserve all v2 evidence.
- [x] Verify the parent FP32 master / Adam / GradScaler checkpoint on CPU.
- [x] Implement and review exact checkpoint restoration and continuation receipts.
- [x] Pass one discarded resumed smoke update and packed-forward parity (1/1, no overflow).
- [x] Complete 3072 additional successful updates (4608 cumulative), final candidate only.
- [x] Export FP16; verify final checkpoint casts and packed inference equality.
- [x] Run full parent / continued / restored-parent PPL and MK comparisons.
- [x] Independently audit training, scores, calibration, memory and full replay.
- [x] Document results and push artifacts to the private repository.

State format, adapter capacity, frozen source, calibration and loss stay fixed.
The parent is the v2 239/384 adapter; this experiment tests additional training
against that parent. See docs/RESURFACE_MORE_PROTOCOL.md. No Q8 or public release.

Full v3 result: PPL **8.388906 → 8.351329** (−0.45%); MK **239 → 237/384**.
There are 34 gains and 36 regressions; paired 95% interval is −4.69 to +3.91
percentage points. The continuation improvement gate **fails**. v3 PPL is
still 13.87% above original S16.
Persistent cache remains **27.1797 MiB/sequence**. Both full parent replays and
the independent CPU audit passed. RMSNorm autotuning drift was diagnosed and
evaluation pinned to the 16-warp configuration that reproduces the archive.
Keep v2 as the recall reference; see docs/RESURFACE_MORE_RESULTS.md.

## Repair v4: improve the codec/calibration within 3.25 bits

- [x] Freeze three candidates, TRAIN-only selection and a separate repair gate.
- [x] Implement readout-aware calibration without changing original S16 forward.
- [x] Collect 4096 TRAIN tokens; verify full-model128-token collector equality.
- [x] Implement real dense3 packing and coordinate equalization at the same budget.
- [x] Pass36 GPU codec/oracle/segmentation/numerical-boundary/storage checks.
- [x] Complete nine TRAIN screening arms and exact parent replay.
- [x] Independently audit screening and select one candidate by the frozen rule.
- [ ] Complete the selected candidate's full PPL/MK and exact baseline replays, if advanced.
- [ ] Audit the full outcome, document limitations and push private artifacts.

No new adapter training in this experiment. Keep original source weights and
the v2 adapter fixed; each candidate retains exactly52B per128-state row plus
one57,344B table. See docs/STATE_REPAIR_PROTOCOL.md.
