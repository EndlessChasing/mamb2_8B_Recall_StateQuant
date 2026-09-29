# Unadapted Q3.25 target: full PPL below 8.25

User objective: continue optimizing original Mamba2-8B state quantization,
without Resurface, until full PPL is strictly below 8.25. Keep the actual
28,499,968-byte batch-one persistent cache (Q3.25 state, convolution and one
static table). Source weights remain FP16. MK is not a selection objective.

## Status

V7 screening is complete: the unchanged baseline wins. V8 measurements are
next. Prior full unadapted references:

| Configuration | Full PPL | Persistent cache |
| --- | ---: | ---: |
| Original S16 | 7.334322 | 116.3750 MiB |
| v5 Q3.25 | 8.367465 | 27.1797 MiB |
| v6 Q3.25 stored scale | 8.355269 | 27.1797 MiB |

The v6 result remains above target. Its old 1% gate does not stop this
user-requested continuation. No Resurface adapter is loaded or trained.

## Route v7: max-preserving nonuniform INT4

- [x] Freeze four tables by four codebooks and TRAIN-only selection.
- [x] Implement actual packed same-budget codebooks with no resident LUT.
- [x] Pass 119 GPU checks, including controlled 65-token recurrence and exact
  uniform delegation to v6. No numerical codec repair was needed.
- [x] Independently audit kernel/input evidence on CPU.
- [x] Screen all 16 candidates on TRAIN rows80..111.
- [x] Apply advancement rule: unchanged baseline wins; no redundant full run.

Uniform delegates v6 unchanged. Mild, quadratic and E2M1-like codebooks preserve
the maximum level and select the nearest actual FP32 reconstructed magnitude.
INT8 and scale storage stay unchanged. See [v7 protocol](STATE_PPL_V7_PROTOCOL.md).
The first 119-check pass was repeated only to add raw boundary evidence for
independent CPU reconstruction; both passing receipts are preserved.

### V7 result

All17 arms complete (16 candidates plus restored baseline), each32 windows /
65,504 TRAIN targets. All candidates are valid, and baseline NLLs, hidden/cache
probes and bytes restore exactly. The evaluation loop took506.90 seconds.

| Static table | Uniform | Mild | Quadratic | FP4-like |
| --- | ---: | ---: | ---: | ---: |
| Magnitude | 8.501912 | 8.817339 | 9.297149 | 9.429512 |
| Full readout | 8.292190 | 8.597388 | 8.925671 | 9.228901 |
| Preserve INT8 | **8.179225** | 8.404043 | 8.774023 | 8.895973 |
| Preserve retained80 | 8.485543 | 8.855592 | 9.277441 | 9.540667 |

None of the three new nonuniform codebooks helps on any tested table. This
does not establish that every nonuniform quantizer is unsuitable. TRAIN PPL
8.179225 is not a full-target result: the identical selected baseline retains
its archived full PPL8.355269. Therefore the target remains unmet and v8 proceeds.
See [raw comparison](../reports/state_ppl_v7_screen/screen_comparison.json).
The [independent screen audit](../reports/state_ppl_v7_screen_audit.json) passes;
SHA256 d7cb3071d9ee52b464ac2febad68eb21b5950286acfa6d0ad9a1eee6a9f9be20.
This audited baseline-win outcome satisfies the predeclared v8 start condition.

## Route v8: per-layer static table mixing, if needed

- [x] Freeze calibration, disjoint TRAIN selection and full target rule.
- [x] Implement calibration/export; pass 11 CPU boundary/provenance checks.
- [ ] Compare 168 single-layer replacements on eight full TRAIN windows.
- [ ] Independently reconstruct ranking and proposed combined tables.
- [ ] Screen combined top-k candidates on 32 other TRAIN windows.
- [ ] Fully validate the one selected table with exact baseline replays.

This route changes only the existing uint8 permutation table and retains the
v6 codec. It calibrates on rows72..79 and screens on112..143. Individual layer
gains are not assumed additive; combinations must earn their own screen score.
See [v8 protocol](STATE_PPL_V8_PROTOCOL.md).

## Completion and scope

Success requires full 130-window/264,764-target PPL < 8.25, no adapter, exact cache
accounting, frozen source/backend/table checks, baseline restoration and an
independent CPU evidence audit. CPU audit reconstructs recorded GPU evidence;
it does not rerun model logits. Corpus/validation families have historical
exposure and are not untouched generalization evidence. If a family fails,
continue a separately recorded TRAIN-selected method. Preserve all attempted
results and keep the repository private.
