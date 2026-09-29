# Unadapted state optimization, then fresh Resurface: v5

Experiment date: 2026-09-28. The [protocol](STATE_FIRST_V5_PROTOCOL.md) was
reviewed and committed before candidate measurements. Its SHA256 is
`ce700910f27230e062fe62e89dd6bddee05bcbf7bb7972c59fdba24cb3db9f5d`.

**Status: all four full evaluation arms and both exact baseline replays are
complete. Unadapted state optimization and subsequent Resurface repair pass;
the stricter comparison against the old v2 endpoint fails its MK interval
condition. The independent full evidence audit passed.** This follows the user's clarified order:

1. Load original pure Mamba2-8B with no Resurface.
2. Compare fixed Q3.25 state tables on TRAIN and freeze one selection.
3. Train a fresh Resurface adapter on that selected quantized recurrence.
4. Evaluate state improvement and recall repair separately.

The previous [v4 experiment](STATE_REPAIR_RESULTS.md) changed the state table
while keeping the old v2 adapter frozen. Its result does not answer this v5
question. No v2/v3 adapter or optimizer state initializes v5 training.

## Fixed storage and candidate preparation

All candidates retain the original real packed codec: 16 INT8 + 64 INT4 +
48 zero-carry coordinates per 128-element row, with two FP16 scales. Storage
is exactly **52 bytes per row / 3.25 bits per element including scales**.
Including the one permutation table and FP16 convolution state, batch-one
persistent cache remains **28,499,968 bytes / 27.1796875 MiB**. Original source
weights remain FP16, approximately 16.474 GB; they are not quantized here.

The four tables use the same pinned unadapted S16 TRAIN statistics as v4:
4,096 tokens, with no validation or CONFIRM data used for calibration.

| Candidate | Fixed rule |
| --- | --- |
| `magnitude` | Original magnitude table; fallback baseline |
| `full_readout` | Rank all coordinates by the v4 one-step readout score |
| `preserve_int8` | Keep old 16 INT8 coordinates; rank the remaining 112 for INT4/zero |
| `preserve_retained80` | Keep old 80 retained coordinates; rank them for INT8/INT4 |

The [candidate payload](../reports/state_first_v5_candidates/candidates.pt)
contains four CPU uint8 tables, with a
[provenance receipt](../reports/state_first_v5_candidates/candidates.json).
Its SHA256 is `cd86a755db5004c716922696cf5532b307c57f5fb7dad4bcca298eb58b55f9a7`
and serialized size is 233,245 bytes. Only one 57,344-byte table is loaded for
a runtime candidate; the multi-candidate offline file is not persistent cache.

The [independent candidate audit](../reports/state_first_v5_candidates_audit.json)
reconstructed all four tables from the pinned statistics and verified set
preservation, stable ties, source/TRAIN provenance and storage. Its 18 independent
selection/storage fixtures passed, and CUDA remained uninitialized. The
[helper CPU checks](../reports/state_first_v5_helper_cpu_checks.json) separately
cover derivation, ties and selection rejection/fallback cases.

## Completed unadapted selection

The expanded TRAIN screen includes rows 8–39, first 512 tokens each:
**32 windows / 16,352 prediction targets**, plus **96 normal numeric prompts**.
No adapter is loaded. Magnitude is repeated after all candidates and must
exactly reproduce all window NLLs and generated sequences.

Selection uses lowest PPL, then highest MK, then fixed candidate order. A new
table must retain at least `max(1, ceil(baseline_MK/2))` correct answers to reject
collapse before recall training. Magnitude is an admissible fallback; if it
wins, v5 stops without duplicating its prior Resurface training. A tie on PPL
may advance a table with higher MK and is not proof of PPL improvement.

| Unadapted state table | TRAIN PPL | Normal TRAIN MK |
| --- | ---: | ---: |
| Original magnitude | 11.129552 | 10/96 |
| Full readout | 10.897633 | 10/96 |
| **Preserve INT8** | **10.845167** | **11/96** |
| Preserve retained80 | 11.161503 | 14/96 |
| Restored magnitude | 11.129552 | 10/96 |

All four candidates are finite, use the exact cache budget and meet the
predeclared MK noncollapse guard of five correct answers. The fixed lowest-PPL
rule selects **preserve_int8**, with **2.5552% lower TRAIN PPL** than magnitude.
The higher-MK retained80 table has worse PPL and does not win this selection.
This is TRAIN evidence and does not establish the full validation gate.

The [screen comparison](../reports/state_first_v5_screen/screen_comparison.json)
and [independent audit](../reports/state_first_v5_screen_audit.json) verify all
32 window NLLs, generated IDs and scoring for each arm, and exact magnitude
restoration. The selected
[calibration payload](../reports/state_first_v5_screen/selected_calibration.pt)
is 60,905 bytes, SHA256
`c525fbf62ef4a72db2d4bb13c13920d9f5d4946485538da0f00a369aab3092e3`.
Its sole table SHA256 is
`6dc1c15ca18d07fb87de8f0517baecbf47adec0846575172fa344f86ac10cde5`.
The selection report SHA256 is
`a7ff703281309d83031d4b3bc3b4c6edf718e8646fa62506ff162dbf1cd62437`.

### Descriptive TRAIN analysis

The [reproducible CPU analysis](../reports/state_first_v5_screen_analysis.json)
confirms that all 7,168 old INT8 table entries are preserved. Across all
56 layers and eight groups, 9,432 old INT4 entries exchange with 9,432 old zero
entries; no INT8 entry is demoted. Lower NLL appears in **29/32** windows versus
magnitude and **23/32** versus full readout.

| Zero-carry coordinates' share of calibration statistic | Magnitude | Full readout | Preserve INT8 |
| --- | ---: | ---: | ---: |
| One-step readout score | 0.739246% | 0.284000% | 0.284141% |
| Mean absolute S16 state | 14.246467% | 19.383264% | 18.803758% |

Preserve INT8 retains nearly all of the readout proxy reduction while preserving
the original high-magnitude INT8 allocation. This is consistent with its TRAIN
PPL improvement; it does not isolate the causal mechanism or predict final MK.
Paired TRAIN MK has four gains and three regressions. Magnitude and preserve
INT8 both score **0/48 on N64** before Resurface, so one more total answer does
not establish reliable recall repair. No new table was fit using this analysis.

## Fresh training — completed and independently audited

The trainer creates all 224 FP32 adapter tensors, Adam and GradScaler from
scratch, with the same v2 training math and fixed 1,536-update recipe. Both
student and separate unadapted S16 teacher use the pinned 16-warp forward
backend. No trained adapter is reused, and a discarded one-update smoke must
pass before formal training starts from scratch again.

The [CPU trainer checks](../reports/state_first_v5_trainer_cpu_checks.json)
passed 11 checks: fresh values, empty optimizer state, scaler initialization,
one synthetic CPU update and complete checkpoint roundtrip. These bookkeeping
checks do not establish GPU packed parity or model quality.

The actual GPU [smoke receipt](../reports/state_first_v5_smoke/report.json)
records **one successful update in two attempts**, with one initial overflow
and the same schedule entry retried after scale reduction 1024→512. All 224
fresh FP32 masters, empty optimizer and initial scaler were verified. Initial
identity and the serialized discarded adapter both passed **128-token bitwise
packed training/inference parity**, with the exact 28,499,968-byte cache.
The [independent smoke audit](../reports/state_first_v5_smoke_audit.json) passed.

The smoke export is explicitly discarded and is not a model candidate. Formal
training reloads the original models and reinitializes every adapter parameter,
optimizer and scaler. It does not continue the smoke update. Only its final
1,536-successful-update export is evaluated.

Formal training completed **1,536 successful updates in 1,543 attempts**, with
seven overflows and same-entry retries. Final loss scale is eight. All four
recovery checkpoints (384, 768, 1152 and 1536 updates) are retained beside the
[training report](../reports/state_first_v5_training/report.json). The final
[FP16 adapter](../reports/state_first_v5_training/adapter_fp16.pt) is **2,375,551
bytes**, containing 1,154,104 parameters / 2,308,208 tensor payload bytes.
Its SHA256 is `48dc63ef46a23711df8e8de6d4df9aae975f2e14c5a003e6bea6c222ff887974`;
the report SHA256 is `055c1d9703f5b43f64b99992e810e5ecca8ebc652a677062af9741795a4e1d3c`.

The [independent training audit](../reports/state_first_v5_training_audit.json)
reconstructed schedules, learning rates, retry/scaler transitions and all four
checkpoint states. It checked all 224 FP32 masters, Adam moment/step states and
parameter mappings. Every final master cast to FP16 equals the actual serialized
export. The 128-token deployed packed-forward parity receipt is bitwise exact,
and the state table bytes and 28,499,968-byte cache remain unchanged. CUDA was
uninitialized during the CPU audit; it audits recorded GPU evidence rather than
re-executing the model.

The source freeze guard checks all 507 tensors' identities, version counters
and gradients; it does not hash their complete post-training contents. A
separate frozen S16 teacher was used. Forward RMSNorm is pinned as documented,
but backward reductions can remain nondeterministic, so a new training run is
not promised to produce identical adapter bytes. These checks establish the
training/export path, not the final PPL or MK quality.

## Full confirmation — completed

Each of the four measured arms covers **130 WikiText-2 validation windows /
264,764 targets** and **384 normal + 384 target-removed CONFIRM prompts**.
Prefill and decoding quantize the carried state after every token; the current
readout precedes carry quantization. MK uses full-vocabulary greedy generation,
at most 12 new tokens or EOS, with the first standalone six-digit answer scored.

| Configuration | Full PPL | Normal MK /384 | N16 /192 | N64 /192 |
| --- | ---: | ---: | ---: | ---: |
| Original S16, no adapter (archived context) | 7.334322 | 146 (38.02%) | 113 | 33 |
| Old magnitude SQ3.25, no adapter | 8.691553 | 50 (13.02%) | 46 | 4 |
| **Selected preserve-INT8 SQ3.25, no adapter** | **8.367465** | **46 (11.98%)** | 43 | 3 |
| **Selected SQ3.25 + fresh v5 Resurface** | **8.093011** | **244 (63.54%)** | **179** | **65** |
| Restored selected SQ3.25, no adapter | 8.367465 | 46 (11.98%) | 43 | 3 |
| Old magnitude SQ3.25 + v2 Resurface (archived context) | 8.388906 | 239 (62.24%) | 169 | 70 |

Every row has **0/384 target-removed matches**. These are matches to the removed
answer, not an abstention score. The original S16 and v2 adapted rows are
hash-verified archived controls, not new v5 GPU runs. Old unadapted SQ was rerun
and exactly matched its archived 130 window NLLs, 768 generated token sequences
and decoded predictions. Removing v5 Resurface then exactly reproduced the
selected no-adapter arm on the same complete population.

### Separate quality decisions

| Comparison | PPL change | MK change | Paired gains / regressions | MK delta 95% interval | Frozen gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Optimize unadapted state | **−3.7288%** | −4 answers / −1.04 pp | 27 / 31 | −4.95 to +2.86 pp | **Pass**: PPL improves at least 1%, same cache |
| Train fresh Resurface on selected state | **−3.2800%** | **+198 answers / +51.56 pp** | **201 / 3** | **+46.35 to +56.77 pp** | **Pass**: MK improvement, positive interval, PPL within 1%, same cache |
| Compare final model against old v2 endpoint | **−3.5272%** | +5 answers / +1.30 pp | 59 / 54 | −4.17 to +6.77 pp | **Fail**: lower bound is below −2 pp |

All intervals use 10,000 paired bootstrap draws, normal cases sorted by ID and
NumPy `default_rng(20260928)`. The raw
[comparison](../reports/state_first_v5_full/full_comparison.json) preserves all
gate checks. State optimization improves NLL in 126/130 windows; adding fresh
Resurface improves all 130/130 windows. Relative to archived v2, 128/130 windows
improve. Window counts are descriptive and not an independent significance test.

The requested order works for **PPL optimization followed by recall repair**.
The final model's observed MK is five answers higher than v2, but that consists
of ten additional N16 answers and five fewer N64 answers. The paired interval
does not establish superiority or the protocol's two-percentage-point
noninferiority condition. Preserve v2 as a reference and v5 as a measured
lower-PPL candidate; do not declare v5 an unconditional replacement.

Final PPL is still **10.3444% above original unadapted S16**, so original PPL is
not restored within 1%. The original S16 comparison has no Resurface adapter;
it does not isolate quantization against a separately trained S16 + Resurface
model. This run also changes the table and trains a fresh adapter, so its
comparison with v2 cannot isolate table effects from training-run variation.

### Storage and evidence boundaries

All four measured SQ arms use **28,499,968 bytes / 27.1796875 MiB** of persistent
batch-one cache: 23,855,104 bytes of packed SSM state including scales,
4,587,520 bytes of FP16 convolution state and a 57,344-byte table. This is
**76.6447% less** than the original 122,028,032-byte S16 cache. v5 improves quality
at the same SQ3.25 storage budget; it does not add another memory reduction.

The original 8,236,999,680 source parameters remain FP16 and occupy
16,473,999,360 weight bytes, separate from cache and temporary runtime memory.
The new adapter adds 2,308,208 resident FP16 parameter bytes, with no additional
recurrent cache. Its serialized file is 2,375,551 bytes. This is state
quantization, not W4 weight quantization or an end-to-end 27 MiB model.

Candidate choice uses TRAIN only. The validation/CONFIRM benchmark families
have historical exposure in this project; these results are not an untouched
generalization claim. Packed forward and approximate masked-STE backward,
one frozen training recipe, the bounded export parity probe and backward
nondeterminism remain relevant limits. No public/Hugging Face release is part
of this experiment.

### Independent audit and retained artifacts

The [full CPU audit](../reports/state_first_v5_full_audit.json) passed. It
independently reconstructs candidate tables and selection, validates fresh
initialization, smoke, all four optimizer checkpoints and the actual FP16
export, then recomputes PPL aggregation, prompt/token identities, generated
answer scoring, all paired bootstrap intervals and the three gate decisions.
Both complete replay comparisons match exactly. All four arms retain the
28,499,968-byte cache, frozen source/table/adapter guards and pinned backend.
The auditor does not rerun GPU logits; CUDA stays uninitialized.

- Full comparison SHA256: `0e5ceb6e91d72a159f46a9a0760a23f3b9301be18bb32590fe01a9196f83d1e9`.
- Full audit SHA256: `9749386e6203c4c45f5ca63396808f1c8579632f331fefb44990db90c741560b`.
- Auditor source SHA256: `9a44d39e1b6ec98d70fb44ee563eed7321e03f9a7a91253c07bc87a418e1863e`.

All four raw reports and replay receipts are retained under
[`reports/state_first_v5_full`](../reports/state_first_v5_full), with the
candidate tables, complete screen, frozen selected calibration, discarded
smoke, formal adapter and all four training checkpoints in adjacent v5 folders.
Copied file hashes were checked against their remote receipts. See
[the reproduction guide](STATE_FIRST_V5_REPRODUCTION.md) for exact commands.
