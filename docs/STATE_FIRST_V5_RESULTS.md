# Unadapted state optimization, then fresh Resurface: v5

Experiment date: 2026-09-28. The [protocol](STATE_FIRST_V5_PROTOCOL.md) was
reviewed and committed before candidate measurements. Its SHA256 is
`ce700910f27230e062fe62e89dd6bddee05bcbf7bb7972c59fdba24cb3db9f5d`.

**Status: candidate preparation and independent CPU audit passed; unadapted
TRAIN screening is running. No v5 trained adapter or full quality result exists
yet.** This follows the user's clarified order:

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

## Unadapted selection — running

The expanded TRAIN screen includes rows 8–39, first 512 tokens each:
**32 windows / 16,352 prediction targets**, plus **96 normal numeric prompts**.
No adapter is loaded. Magnitude is repeated after all candidates and must
exactly reproduce all window NLLs and generated sequences.

Selection uses lowest PPL, then highest MK, then fixed candidate order. A new
table must retain at least `max(1, ceil(baseline_MK/2))` correct answers to reject
collapse before recall training. Magnitude is an admissible fallback; if it
wins, v5 stops without duplicating its prior Resurface training. A tie on PPL
may advance a table with higher MK and is not proof of PPL improvement.

## Fresh training — pending selection

The trainer creates all 224 FP32 adapter tensors, Adam and GradScaler from
scratch, with the same v2 training math and fixed 1,536-update recipe. Both
student and separate unadapted S16 teacher use the pinned 16-warp forward
backend. No trained adapter is reused, and a discarded one-update smoke must
pass before formal training starts from scratch again.

The [CPU trainer checks](../reports/state_first_v5_trainer_cpu_checks.json)
passed 11 checks: fresh values, empty optimizer state, scaler initialization,
one synthetic CPU update and complete checkpoint roundtrip. These bookkeeping
checks do not establish GPU packed parity or model quality. Actual GPU smoke,
formal training, final FP16 export and 128-token packed parity are still pending.

## Full confirmation — pending

The final four arms are old unadapted SQ, selected unadapted SQ, selected SQ plus
fresh Resurface, and restored selected unadapted SQ. Each receives all 130 PPL
windows / 264,764 targets and 384 normal plus 384 target-removed CONFIRM prompts.
Exact archived-old and restored-selected replay are required.

Report separately: at least 1% unadapted PPL improvement; Resurface MK improvement
with positive paired 95% lower bound while PPL stays within 1%; and improvement
over the prior SQ3.25 + v2 endpoint. The exact gates are in the frozen protocol.
Original S16 PPL restoration is a separate check. Historical benchmark exposure
and surrogate-gradient limitations remain; no public model release is included.
