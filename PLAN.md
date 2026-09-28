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
