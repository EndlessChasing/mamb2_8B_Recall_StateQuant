# Unadapted Q3.25 target: full PPL below 8.25

User objective: continue optimizing original Mamba2-8B state quantization,
without Resurface, until full PPL is strictly below 8.25. Keep the actual
28,499,968-byte batch-one persistent cache (Q3.25 state, convolution and one
static table). Source weights remain FP16. MK is not a selection objective.

## Status

V7 screening is complete: the unchanged baseline wins. V8 calibration and
TRAIN screening pass their independent audits; full validation is complete
with PPL 8.283863, still above target. The independent full CPU audit passes. Prior full unadapted references:

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
- [x] Compare 168 single-layer replacements on eight full TRAIN windows.
- [x] Independently reconstruct ranking and proposed combined tables.
- [x] Screen combined top-k candidates on 32 other TRAIN windows.
- [x] Fully validate the one selected table with exact baseline replays.

This route changes only the existing uint8 permutation table and retains the
v6 codec. It calibrates on rows72..79 and screens on112..143. Individual layer
gains are not assumed additive; combinations must earn their own screen score.
See [v8 protocol](STATE_PPL_V8_PROTOCOL.md).

### V8 calibration

All 170 arms (baseline, 168 single-layer replacements, restored baseline)
complete, each eight TRAIN windows / 16,376 targets. Baseline calibration PPL
is 8.277007. Of 56 layers, 46 have at least one strictly better alternative.
This produces seven distinct tables: baseline, top1, top2, top4, top8, top16,
and allnegative (46 replacements). Combined results are still pending.
Baseline per-window NLLs, repeated-reset hidden/cache hashes and allocation
restore exactly. Calibration took 516.21 seconds.

The [independent CPU calibration audit](../reports/state_ppl_v8_calibration_audit.json)
reconstructs all raw scores, ranking, deduplication and actual table entries,
and verifies the audited v7 miss preceding v8. It passes; SHA256
c2606a7768b55da22ccb0510048e029c2ec5277e936321bb234eb968614953ce.
See [raw calibration](../reports/state_ppl_v8_calibration/calibration_comparison.json).


### V8 disjoint TRAIN screen

All seven candidates are valid. The frozen winner is **top8**; the baseline
and restored baseline agree exactly. Each arm uses 32 other TRAIN windows /
65,504 targets. The evaluation loop took 94.05 seconds.

| Combined table | TRAIN PPL |
| --- | ---: |
| Baseline | 8.227528 |
| top1 | 8.174200 |
| top2 | 8.163930 |
| top4 | 8.167044 |
| **top8** | **8.159226** |
| top16 | 8.169884 |
| allnegative (46 layers) | 8.232327 |

The selected eight replacements are layers 0, 3, 6, 1, 4, 2, 7 and 10
(zero-based calibration rank order). The combined screen improves 23/32 windows
and PPL by 0.830168%. Replacing all individually beneficial layers does not
improve the combined result, confirming the need for separate screening.
This is a TRAIN result and does not establish the full target.

The [independent screen audit](../reports/state_ppl_v8_screen_audit.json) passes;
SHA256 90fbed838e0c97f5543b6687b20d6a3740fc4fa0ebade2920d73d08df40526e2.
[Selected artifact](../reports/state_ppl_v8_screen/selected_calibration.pt) SHA256
9b7c04814085abbb67d7a10e0b70d1a6e9b34e97299fc0dc26e4373ea8f8ea39;
its actual table SHA256 is
281f7c9fbfdabd6bfa04964ed44b19761708a08447989436b978fae37dc27298.
Only this frozen TRAIN winner advances to full validation.

### V8 full result

The frozen top8 table achieves **8.28386254265227**, compared with the v6
parent's **8.355268708845868**: a 0.854624% PPL improvement, 100/130 windows
improved and 30 regressed. All three arms complete 130 windows / 264,764 targets.
Both archived-parent and restored-parent NLLs, hidden/cache probes and actual
allocation agree exactly. The evaluation loop took 144.71 seconds.

The target is **not reached**: 8.283863 is greater than 8.25. All source,
backend, table, cache and no-adapter checks pass. Original S16 remains
7.334322; the v8 result is 12.9465% above it. No MK is evaluated.
See [raw full comparison](../reports/state_ppl_v8_full/full_comparison.json).
The [independent full CPU audit](../reports/state_ppl_v8_full_audit.json) passes;
SHA256 051316fe074ea1ba9cfeef87ba7fe4bb9c3a4c18e3c7427c9a2ef256d490e9cf.
This confirms the finite target miss required to start v9.

## Route v9: refine individual groups in the selected eight layers

- [x] Freeze the protocol before viewing the v8 full result.
- [x] Implement and independently audit group interventions and provenance.
- [x] Calibrate up to 192 individual group replacements on TRAIN rows144..151.
- [x] Reconstruct proposals and screen on disjoint TRAIN rows152..183.
- [x] Fully validate the frozen TRAIN winner and independently audit evidence.

The 19 implementation fixtures and 20 independent audit fixtures pass. The
actual input audit confirms 192 retained alternatives, 64 parent-equal labels
and no duplicate alternatives. Sources and evidence were committed before the
first v9 GPU run. [Input audit](../reports/state_ppl_v9_inputs_audit.json) SHA256:
439ecf2531226a2ccb16f87ec9825feb19159af6a918eb94dbf0b6365e4e49a9.
Calibration completes all 194 arms in 589.71 seconds. Among 64 groups,
58 have a strictly better individual alternative, producing eight distinct
combined tables: baseline, top1, top2, top4, top8, top16, top32 and allnegative.
The original and restored baseline repeat exactly. The
[independent calibration audit](../reports/state_ppl_v9_calibration_audit.json)
passes, SHA256 997b60902fcf17fbbcefd2574a6a260974d3fef2a53253dbd9039d101930313b.
All 194 arm tables, raw scores and combined exports are independently rebuilt.
Disjoint TRAIN screening completes all nine arms (eight candidates and restored
baseline), and its independent audit passes. The frozen winner is top2,
PPL 8.176195217465986 versus parent 8.178838569254431: a 0.0323194% improvement,
with 12/32 windows improved and 20 regressed. Other combined candidates do not
beat the parent. This is a small TRAIN gain; full validation is complete.
See [screen comparison](../reports/state_ppl_v9_screen/screen_comparison.json)
and [independent audit](../reports/state_ppl_v9_screen_audit.json), SHA256
4b218de5223ed35cf12be6c341e3dd3d5b75a82656019c8ac12526ce67f60402.
The selected artifact SHA256 is
3467897358f33de22b1b629819cb2035f4cb8912914f0183959aa581fafeab50;
actual table SHA256
b1865e81ff3fbed028027a883872aef9bb91e614e7cf08c72211496a78aeb597.

The v8 TRAIN winner remains the parent. Refine its eight selected layers,
all eight groups per layer, using only existing v5 group tables; the codec
and persistent cache are unchanged. The family grid was fixed while v8 full
was running, before any full outcome was seen. Start only after the v8 full
CPU audit confirms a finite target miss. See [frozen protocol](STATE_PPL_V9_PROTOCOL.md).

### V9 full result

Full PPL is **8.290607333938492**, versus the v8 parent's **8.28386254265227**:
a 0.08142085% regression (53/130 windows improved, 77 regressed). The target
remains unmet. Archived-parent and restored-parent replays, source/backend,
actual cache and no-adapter checks pass; all three arms cover 264,764 targets.
The evaluation loop took 144.86 seconds. This family does not improve the
best full unadapted result, which remains v8 PPL 8.283863.

The [independent full audit](../reports/state_ppl_v9_full_audit.json) passes
for evidence integrity and confirms target failure; SHA256
 dde9b57633c6994380208c606d9c1231e1c74dd39531763de6cef2ca025a3018.
See [raw comparison](../reports/state_ppl_v9_full/full_comparison.json).

## Route v10: change precision versus coverage at the same 3.25-bit budget

- [x] Freeze a distinct four-layout protocol and TRAIN-only selection.
- [ ] Implement actual packed layouts and pass independent codec checks.
- [ ] Independently audit source/input/packing evidence before quality runs.
- [ ] Screen four layouts on 32 full TRAIN windows, rows184..215.
- [ ] Fully validate the frozen TRAIN winner and independently audit evidence.

| Layout | INT8 coordinates | INT4 coordinates | Zero carry | Payload + scales |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 16 | 64 | 48 | 48 + 4 = 52 B |
| More coverage | 8 | 80 | 40 | 48 + 4 = 52 B |
| More INT8 | 24 | 48 | 56 | 48 + 4 = 52 B |
| Most INT8 tested | 32 | 32 | 64 | 48 + 4 = 52 B |

Every row remains exactly 3.25 bits per coordinate including two FP16 scales,
and total persistent cache remains 28,499,968 bytes. This is a new allocation
family; it does not preserve the old 16/64/48 tier counts. The v9 TRAIN-selected
permutation remains fixed irrespective of its full regression. Global layout
selection uses TRAIN PPL only. Existing v6 numerical policy and baseline codec
are retained; nonbaseline layouts require new masked packed kernels.

The family/grid/rows were proposed and approved before the v9 full outcome;
the final protocol document was frozen after the target-failure notification.
No choices were changed using that outcome. See [v10 protocol](STATE_PPL_V10_PROTOCOL.md).

## Completion and scope

Success requires full 130-window/264,764-target PPL < 8.25, no adapter, exact cache
accounting, frozen source/backend/table checks, baseline restoration and an
independent CPU evidence audit. CPU audit reconstructs recorded GPU evidence;
it does not rerun model logits. Corpus/validation families have historical
exposure and are not untouched generalization evidence. If a family fails,
continue a separately recorded TRAIN-selected method. Preserve all attempted
results and keep the repository private.
