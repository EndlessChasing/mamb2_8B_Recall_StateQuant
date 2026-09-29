# Checklist

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
- [ ] Evaluate original S16, unadapted SQ3.25, trained SQ3.25 and restored SQ baseline.
- [ ] Independently audit and document PPL/MK/cache results.

See docs/QUANT_FIRST_PROTOCOL.md. No Q8 or old-adapter initialization; no public release.
