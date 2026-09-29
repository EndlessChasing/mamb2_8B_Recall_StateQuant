# Reproduce v5: optimize unadapted SQ3.25, then train fresh Resurface

Follow the [frozen v5 protocol](STATE_FIRST_V5_PROTOCOL.md), SHA256
`ce700910f27230e062fe62e89dd6bddee05bcbf7bb7972c59fdba24cb3db9f5d`.
The order is fixed: derive four state tables, select using unadapted TRAIN
measurements, freeze the winner, train a fresh adapter, then perform full
validation. If the original magnitude table wins screening, stop before training.

The commands below use input paths verified on the existing CUDA host. Run them
in **Bash from the repository root**, preserving the variables between sections.
Use a new output root for each reproduction. Scripts reject existing output
directories or receipts. Keep source files unchanged between stages because
their hashes bind preparation, selection, training and evaluation.

See [the v2 environment/source guide](QUANT_FIRST_REPRODUCTION.md) for the original
checkpoint, tokenizer and pinned prose tensor. The measured environment uses
Python 3.10.12, PyTorch 2.11.0+cu128, Triton 3.6.0, mamba-ssm 2.3.2.post1,
NumPy 1.26.4, datasets 4.8.5 and sentencepiece 0.2.1 on an RTX PRO 6000 Blackwell
Server Edition. Every v5 GPU entry point pins the existing 16-warp RMSNorm
configuration before model loading; see the [backend replay policy](RESURFACE_MORE_BACKEND_REPLAY.md).

## 1. Paths and frozen inputs

```bash
SQ5_PYTHON=/home/horde/.venvs/lodram/bin/python
SQ5_SOURCE_DIR=/home/horde/Mamba2-8B-E8W5/models/source
SQ5_TRAIN_TOKENS=/home/horde/mamb2_8B_Recall/training_data/prose/training_tokens.pt
SQ5_NUMERIC_DIR=training_data/numeric_v2
SQ5_NUMERIC_SHA=451b8703c21120667ef0ea272c21a10d4cd4561779a7b9663773ae47981600c0
SQ5_OLD_CAL=artifacts/quant_first_v2_calibration/calibration.pt
SQ5_V4_CAL=artifacts/state_repair_calibration/calibration.pt
SQ5_PARENT_REPORT=artifacts/quant_first_v2_training/report.json
SQ5_PARENT_EVAL=artifacts/quant_first_v2_eval
SQ5_DENSE_CHECKS=reports/state_repair_dense3_checks.json
SQ5_RUN_DIR=artifacts/repro_state_first_v5

mkdir -p "$SQ5_RUN_DIR"
```

`SQ5_SOURCE_DIR` holds the original pure Mamba2-8B checkpoint. The parent directory
name is historical. All source weights remain FP16 during v5 inference.
Training loads a separate frozen S16 teacher as well as the student; their
weights alone occupy approximately 32.95 GB, before activations and optimizer
workspace. The 28,499,968-byte cache budget applies to deployed batch-one
persistent state and its permutation table.

The old calibration and v4 statistics require their companion `.json` receipts.
The archived parent report requires its adjacent `adapter_fp16.pt`. The parent
evaluation directory must contain `full_source_s16.json`,
`full_source_sq3p25.json` and `full_resurface_sq3p25.json`.

For a checkout using the copied repository evidence, the corresponding paths are
`reports/quant_first_v2/calibration/calibration.pt`,
`reports/state_repair_calibration/calibration.pt`,
`reports/quant_first_v2/training/report.json` and
`reports/quant_first_v2/evaluation`. Preserve the companion files when changing
these variables. The auditor validates upstream v4 provenance, including hashes
of the archived v2 report/export and the dense-codec test receipt. Candidate
generation and screening use only the unadapted source statistics and tables.

Prepare or verify the existing numeric TRAIN corpus with the unchanged v2
CPU-only preparation path:

```bash
CUDA_VISIBLE_DEVICES= "$SQ5_PYTHON" scripts/prepare_quant_first.py \
  --source-dir "$SQ5_SOURCE_DIR" \
  --train-tokens "$SQ5_TRAIN_TOKENS" \
  --data-root "$SQ5_NUMERIC_DIR" \
  --out "$SQ5_RUN_DIR/numeric_receipt" \
  --prepare-only

CUDA_VISIBLE_DEVICES= "$SQ5_PYTHON" scripts/audit_state_first_v5.py --self-test
```

The trainer requires the exact numeric manifest hash above and the pinned prose
file hash. The numeric data protocol remains v2; the experiment protocol is v5.
Do not replace either identity with a newly chosen manifest or corpus.

## 2. Derive the four tables on CPU and audit them

```bash
CUDA_VISIBLE_DEVICES= "$SQ5_PYTHON" scripts/prepare_state_first_v5.py \
  --original-calibration "$SQ5_OLD_CAL" \
  --v4-calibration "$SQ5_V4_CAL" \
  --out "$SQ5_RUN_DIR/candidates"

SQ5_CANDIDATES="$SQ5_RUN_DIR/candidates/candidates.pt"
SQ5_AUDIT_ARGS=(
  --source-dir "$SQ5_SOURCE_DIR"
  --calibration "$SQ5_OLD_CAL"
  --repair-calibration "$SQ5_V4_CAL"
  --parent-training-report "$SQ5_PARENT_REPORT"
  --prose-tokens "$SQ5_TRAIN_TOKENS"
  --codec-checks "$SQ5_DENSE_CHECKS"
  --candidates "$SQ5_CANDIDATES"
)

CUDA_VISIBLE_DEVICES= "$SQ5_PYTHON" scripts/audit_state_first_v5.py \
  "${SQ5_AUDIT_ARGS[@]}" --stage candidates \
  --output "$SQ5_RUN_DIR/candidates_audit.json"
```

This reuses the pinned original-S16 TRAIN statistics from 4,096 tokens. It derives
`magnitude`, `full_readout`, `preserve_int8` and `preserve_retained80` on CPU.
Each candidate uses the same real packed 16 INT8 / 64 INT4 / 48 zero-carry codec,
including two FP16 scales per 128 coordinates. Each deployed controller retains
one 57,344-byte permutation table. The candidate artifact contains all four
alternatives for offline comparison.

The independent auditor reconstructs every table from the pinned statistics,
checks coordinate ordering and verifies upstream provenance. Require
`complete: true` and `passed: true` before screening.

## 3. Screen without an adapter, audit, then freeze the winner

```bash
SQ5_EVAL_ARGS=(
  --source-dir "$SQ5_SOURCE_DIR"
  --candidates "$SQ5_CANDIDATES"
  --prose-tokens "$SQ5_TRAIN_TOKENS"
)

OMP_NUM_THREADS=8 "$SQ5_PYTHON" scripts/run_state_first_v5.py \
  "${SQ5_EVAL_ARGS[@]}" --stage screen \
  --out "$SQ5_RUN_DIR/screen"

CUDA_VISIBLE_DEVICES= "$SQ5_PYTHON" scripts/audit_state_first_v5.py \
  "${SQ5_AUDIT_ARGS[@]}" --stage screen \
  --eval-dir "$SQ5_RUN_DIR/screen" \
  --output "$SQ5_RUN_DIR/screen_audit.json"
```

The five arms are the four unadapted candidates followed by restored magnitude.
Each uses TRAIN rows 8..39, 512 tokens per row: 32 windows / 16,352 targets,
plus 96 numeric TRAIN prompts. All magnitude window NLLs and generated token
sequences must replay exactly in the last arm. Screening accepts no training
report or adapter argument.

Inspect `screen/screen_comparison.json`. The selector requires a new candidate
to retain at least `max(1, ceil(0.5 * baseline_MK_correct))` TRAIN answers, then
chooses lowest PPL, highest MK and fixed candidate order. Magnitude is the
fallback. **If `selection.stopped` is true, stop here.** A non-magnitude winner
produces `screen/selected_calibration.pt` and its companion `.json` receipt.
Its table hash, candidate artifact hash and completed selection-report hash are
frozen before training. A tied-PPL selection is not a measured PPL improvement;
the later full gate independently requires at least 1% improvement.

Proceed only after a passing screen audit and a non-magnitude selection:

```bash
SQ5_SELECTED="$SQ5_RUN_DIR/screen/selected_calibration.pt"
SQ5_SELECTION_ARGS=(
  --screening-report "$SQ5_RUN_DIR/screen/screen_comparison.json"
  --selected-calibration "$SQ5_SELECTED"
)
SQ5_TRAIN_ARGS=(
  --source-dir "$SQ5_SOURCE_DIR"
  --calibration "$SQ5_SELECTED"
  --candidates "$SQ5_CANDIDATES"
  --data-root "$SQ5_NUMERIC_DIR"
  --train-manifest-sha256 "$SQ5_NUMERIC_SHA"
  --prose-manifest docs/prose_train_manifest.json
  --prose-tokens "$SQ5_TRAIN_TOKENS"
)
```

## 4. Discarded fresh smoke and its CPU audit

```bash
OMP_NUM_THREADS=8 "$SQ5_PYTHON" scripts/train_state_first_v5.py \
  "${SQ5_TRAIN_ARGS[@]}" --smoke \
  --out-dir "$SQ5_RUN_DIR/smoke"

CUDA_VISIBLE_DEVICES= "$SQ5_PYTHON" scripts/audit_state_first_v5.py \
  "${SQ5_AUDIT_ARGS[@]}" "${SQ5_SELECTION_ARGS[@]}" --stage smoke \
  --smoke-report "$SQ5_RUN_DIR/smoke/report.json" \
  --output "$SQ5_RUN_DIR/smoke_audit.json"
```

Smoke starts with fresh `V=0`, `g=1`, `router_w=0`, `router_b=-4`, an empty Adam
optimizer and a new GradScaler. It performs one successful update, then exports
`discarded_smoke_adapter_fp16.pt` only for parity and serialization checks.
Keep that file beside `smoke/report.json` for later audits. It is marked
`discarded: true`, `candidate: false`, and is never a formal initialization.
Require a passing smoke audit before formal training.

## 5. Fresh 1,536-update formal training and CPU audit

```bash
OMP_NUM_THREADS=8 "$SQ5_PYTHON" scripts/train_state_first_v5.py \
  "${SQ5_TRAIN_ARGS[@]}" \
  --smoke-report "$SQ5_RUN_DIR/smoke/report.json" \
  --out-dir "$SQ5_RUN_DIR/training"

CUDA_VISIBLE_DEVICES= "$SQ5_PYTHON" scripts/audit_state_first_v5.py \
  "${SQ5_AUDIT_ARGS[@]}" "${SQ5_SELECTION_ARGS[@]}" --stage training \
  --smoke-report "$SQ5_RUN_DIR/smoke/report.json" \
  --training-report "$SQ5_RUN_DIR/training/report.json" \
  --output "$SQ5_RUN_DIR/training_audit.json"
```

Formal training starts fresh again and verifies the discarded smoke's input,
code and backend identities. It uses the unchanged v2 schedule and training
math, 1,536 successful updates and at most eight total overflow attempts across
the formal run. The selected
state table and source weights remain frozen; only the external Resurface
parameters train. Backward uses the declared live-mask STE while forward uses
the exact deployed packed recurrence.

Keep the complete formal directory: `report.json`, `adapter_fp16.pt`, and
`checkpoint_0384.pt`, `checkpoint_0768.pt`, `checkpoint_1152.pt`,
`checkpoint_1536.pt`. The final export is the sole evaluation candidate.
The CPU audit reconstructs schedules, retry/scaler transitions, optimizer
mapping/steps, fresh initialization, every recovery checkpoint and final
FP32-master-to-FP16-export equality. Require a passing audit before full
evaluation. The CLI has no resume option; preserve an interrupted run and
use a fresh output directory for a new attempt.

## 6. Full four-arm evaluation and independent CPU audit

```bash
OMP_NUM_THREADS=8 "$SQ5_PYTHON" scripts/run_state_first_v5.py \
  "${SQ5_EVAL_ARGS[@]}" --stage full \
  --selected-calibration "$SQ5_SELECTED" \
  --training-report "$SQ5_RUN_DIR/training/report.json" \
  --parent-eval-dir "$SQ5_PARENT_EVAL" \
  --out "$SQ5_RUN_DIR/full"

CUDA_VISIBLE_DEVICES= "$SQ5_PYTHON" scripts/audit_state_first_v5.py \
  "${SQ5_AUDIT_ARGS[@]}" "${SQ5_SELECTION_ARGS[@]}" --stage full \
  --smoke-report "$SQ5_RUN_DIR/smoke/report.json" \
  --training-report "$SQ5_RUN_DIR/training/report.json" \
  --parent-eval-dir "$SQ5_PARENT_EVAL" \
  --eval-dir "$SQ5_RUN_DIR/full" \
  --output "$SQ5_RUN_DIR/full_audit.json"
```

The auditor uses offline dataset access. The full evaluation normally populates
the pinned WikiText validation cache; preserve that cache when moving evidence
to a different host. The source/tokenizer, pinned TRAIN tensor, candidate
artifacts, all screen arms, selected calibration, discarded smoke and complete
formal training directory must remain available for the full audit.

Each arm evaluates all 130 validation windows / 264,764 targets and all 768
CONFIRM prompts, comprising 384 normal and 384 target-removed cases:

| Report suffix | Configuration |
| --- | --- |
| `old_magnitude` | Original magnitude SQ3.25, no adapter; exact archived replay |
| `selected_no_adapter` | Selected SQ3.25 table, no adapter |
| `selected_resurface` | Selected table plus the final fresh v5 adapter |
| `restored_selected` | Remove adapter and exactly replay `selected_no_adapter` |

Preserve all arms and failed quality gates. The comparison reports three
separate gates: unadapted state optimization, Resurface repair on that selected
state, and improvement over the archived v2 endpoint. It also reports the PPL
gap to original S16. The paired MK intervals use sorted case IDs, 10,000 draws
and NumPy `default_rng` seed 20260928. Target-removed zero matches measure lack
of correct answers after removal; they do not establish abstention.

The CPU audit verifies recorded evidence and arithmetic. It does not regenerate
GPU logits or gradients, and reported source identity/version/gradient checks
do not replace post-run hashes of all original weight bytes. Benchmark families
have historical exposure. This procedure does not claim untouched downstream
generalization or authorize model publication.
