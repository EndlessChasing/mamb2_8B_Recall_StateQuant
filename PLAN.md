# Checklist

- [x] Select original FP16 Mamb2_8B_Recall and unchanged adapter.
- [x] Skip Q8; freeze S16 versus original-coordinate SQ3.25 protocol.
- [x] Preserve base/adapter/archive identities and license scopes.
- [ ] Implement real packed state and Resurface-compatible runtime.
- [ ] Pass codec oracle, segmented recurrence and native controls.
- [ ] Calibrate group-aligned tiers using 4096 TRAIN tokens only.
- [ ] Measure actual persistent cache and temporary GPU memory.
- [ ] Run paired pilot PPL and MK, recording raw outputs.
- [ ] Apply predeclared stop rule; full validation only if pilot survives.
- [ ] Audit receipts, document results, create GitHub repository.

No adaptation training or Hugging Face release is included in this experiment.
