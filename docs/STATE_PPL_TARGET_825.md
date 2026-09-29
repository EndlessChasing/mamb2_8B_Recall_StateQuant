# Unadapted Q3.25 target: full PPL below 8.25

User objective: continue optimizing original Mamba2-8B state quantization,
without Resurface, until full PPL is strictly below8.25. Keep the actual
28,499,968-byte batch-one persistent cache (Q3.25 state, convolution and one
static table). Source weights remain FP16. MK is not a selection objective.

## Status

No v7/v8 model quality result yet. Prior full unadapted references:

| Configuration | Full PPL | Persistent cache |
| --- | ---: | ---: |
| Original S16 | 7.334322 | 116.3750 MiB |
| v5 Q3.25 | 8.367465 | 27.1797 MiB |
| v6 Q3.25 stored scale | 8.355269 | 27.1797 MiB |

The v6 result remains above target. Its old1% gate does not stop this
user-requested continuation. No Resurface adapter is loaded or trained.

## Route v7: max-preserving nonuniform INT4

- [x] Freeze four tables by four codebooks and TRAIN-only selection.
- [x] Implement actual packed same-budget codebooks with no resident LUT.
- [x] Pass119 GPU checks, including controlled65-token recurrence and exact
  uniform delegation to v6. No numerical codec repair was needed.
- [x] Independently audit kernel/input evidence on CPU.
- [ ] Screen all16 candidates on TRAIN rows80..111.
- [ ] Audit selection and confirm only the winner on all130 validation windows
  with archived/restored baseline checks, if a nonbaseline candidate wins.

Uniform delegates v6 unchanged. Mild, quadratic and E2M1-like codebooks preserve
the maximum level and select the nearest actual FP32 reconstructed magnitude.
INT8 and scale storage stay unchanged. See [v7 protocol](STATE_PPL_V7_PROTOCOL.md).
The first119-check pass was repeated only to add raw boundary evidence for
independent CPU reconstruction; both passing receipts are preserved.

## Route v8: per-layer static table mixing, if needed

- [x] Freeze calibration, disjoint TRAIN selection and full target rule.
- [x] Implement calibration/export; pass11 CPU boundary/provenance checks.
- [ ] Compare168 single-layer replacements on eight full TRAIN windows.
- [ ] Independently reconstruct ranking and proposed combined tables.
- [ ] Screen combined top-k candidates on32 other TRAIN windows.
- [ ] Fully validate the one selected table with exact baseline replays.

This route changes only the existing uint8 permutation table and retains the
v6 codec. It calibrates on rows72..79 and screens on112..143. Individual layer
gains are not assumed additive; combinations must earn their own screen score.
See [v8 protocol](STATE_PPL_V8_PROTOCOL.md).

## Completion and scope

Success requires full130-window/264,764-target PPL<8.25, no adapter, exact cache
accounting, frozen source/backend/table checks, baseline restoration and an
independent CPU evidence audit. CPU audit reconstructs recorded GPU evidence;
it does not rerun model logits. Corpus/validation families have historical
exposure and are not untouched generalization evidence. If a family fails,
continue a separately recorded TRAIN-selected method. Preserve all attempted
results and keep the repository private.
