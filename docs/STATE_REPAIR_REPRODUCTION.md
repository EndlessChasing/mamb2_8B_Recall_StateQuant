# Reproduce the fixed v4 state repair experiment

Read the [frozen protocol](STATE_REPAIR_PROTOCOL.md) and
[source/environment requirements](QUANT_FIRST_REPRODUCTION.md). This experiment
does not train a new adapter. It uses the archived v2 FP16 adapter and its
original calibration, both included under `reports/quant_first_v2/`.
The large original source checkpoint and pinned TRAIN tensor are external inputs.

The tested environment is Python 3.10.12, PyTorch 2.11.0+cu128, Triton 3.6.0,
mamba-ssm 2.3.2.post1, NumPy 1.26.4, datasets 4.8.5 and sentencepiece 0.2.1,
on an RTX PRO 6000 Blackwell Server Edition. Evaluation pins the existing
16-warp RMSNorm configuration before loading the model; see the
[backend policy](RESURFACE_MORE_BACKEND_REPLAY.md). Package and installed kernel
source hashes are recorded and audited. The pinned WikiText validation cache
is also needed for the offline CPU audit.

## 1. Set paths and run implementation checks

Run the following in Bash from the repository root on the CUDA host. Use a fresh
output directory for each run; completed receipts must not be overwritten.

```bash
SQ_PYTHON=/home/horde/.venvs/lodram/bin/python
SQ_SOURCE_DIR=/home/horde/Mamba2-8B-E8W5/models/source
SQ_TRAIN_TOKENS=/home/horde/mamb2_8B_Recall/training_data/prose/training_tokens.pt
SQ_OLD_CAL=reports/quant_first_v2/calibration/calibration.pt
SQ_PARENT_REPORT=reports/quant_first_v2/training/report.json
SQ_PARENT_EVAL=reports/quant_first_v2/evaluation
SQ_REPAIR_RUN=artifacts/repro_state_repair_v4

mkdir -p "$SQ_REPAIR_RUN"
OMP_NUM_THREADS=8 "$SQ_PYTHON" scripts/check_state_repair_calibration.py \
  --output "$SQ_REPAIR_RUN/collector_checks.json"
OMP_NUM_THREADS=8 "$SQ_PYTHON" scripts/check_state_codec_dense3.py \
  --out "$SQ_REPAIR_RUN/dense3_checks.json"
CUDA_VISIBLE_DEVICES= "$SQ_PYTHON" scripts/audit_state_repair.py --self-test
```

The source path above contains the original checkpoint, despite its parent
directory name. The scripts do not load E8/W5 weights. The TRAIN tensor must
match the pinned serialized hash in the protocol. Follow the v2 guide for its
preparation and manifest; do not substitute validation text.

## 2. Fresh TRAIN calibration

```bash
OMP_NUM_THREADS=8 "$SQ_PYTHON" scripts/prepare_state_repair.py \
  --source-dir "$SQ_SOURCE_DIR" \
  --train-tokens "$SQ_TRAIN_TOKENS" \
  --original-calibration "$SQ_OLD_CAL" \
  --collector-checks "$SQ_REPAIR_RUN/collector_checks.json" \
  --out "$SQ_REPAIR_RUN/calibration"
```

This first verifies collector equality on a 128-token model probe, then collects
4,096 TRAIN tokens on the unadapted source with S16 state. It writes
`calibration.pt` and its companion `calibration.json`. The latter binds the
collector checks and implementation hashes. Do not edit code between stages.

To audit or evaluate the recorded calibration instead, set the repair-calibration
argument below to `reports/state_repair_calibration/calibration.pt` and codec
checks to `reports/state_repair_dense3_checks.json`; keep each companion receipt.

## 3. Screen on TRAIN, then audit

```bash
SQ_REPAIR_ARGS=(
  --source-dir "$SQ_SOURCE_DIR"
  --calibration "$SQ_OLD_CAL"
  --repair-calibration "$SQ_REPAIR_RUN/calibration/calibration.pt"
  --parent-training-report "$SQ_PARENT_REPORT"
  --prose-tokens "$SQ_TRAIN_TOKENS"
  --codec-checks "$SQ_REPAIR_RUN/dense3_checks.json"
)

OMP_NUM_THREADS=8 "$SQ_PYTHON" scripts/run_state_repair.py \
  "${SQ_REPAIR_ARGS[@]}" --stage screen --out "$SQ_REPAIR_RUN/screen"
CUDA_VISIBLE_DEVICES= "$SQ_PYTHON" scripts/audit_state_repair.py \
  "${SQ_REPAIR_ARGS[@]}" --stage screen --eval-dir "$SQ_REPAIR_RUN/screen" \
  --output "$SQ_REPAIR_RUN/screen_audit.json"
```

The nine unique arms include all three candidates without and with the same
v2 adapter. A tenth arm restores old SQ plus v2 and must exactly replay its
window NLLs and generated IDs. The selector uses only TRAIN scores. Inspect
`screen_comparison.json`: if `selection.stopped` is true, stop this experiment.
If the screen audit fails, resolve that failure before advancing. A diagnostic
advancement is explicitly distinct from passing the screening gate.

## 4. Full confirmation and independent audit

```bash
SQ_FULL_ARGS=(
  --screening-report "$SQ_REPAIR_RUN/screen/screen_comparison.json"
  --parent-eval-dir "$SQ_PARENT_EVAL"
)
OMP_NUM_THREADS=8 "$SQ_PYTHON" scripts/run_state_repair.py \
  "${SQ_REPAIR_ARGS[@]}" "${SQ_FULL_ARGS[@]}" --stage full \
  --out "$SQ_REPAIR_RUN/full"
CUDA_VISIBLE_DEVICES= "$SQ_PYTHON" scripts/audit_state_repair.py \
  "${SQ_REPAIR_ARGS[@]}" "${SQ_FULL_ARGS[@]}" --stage full \
  --eval-dir "$SQ_REPAIR_RUN/full" --output "$SQ_REPAIR_RUN/full_audit.json"
```

Full evaluation runs old SQ plus v2, the single selected codec plus v2, then
restored old SQ plus v2. Each arm includes all 130 PPL windows / 264,764 targets
and all 384 normal plus 384 target-removed CONFIRM prompts. The first parent must
exactly reproduce its archived full result before candidate evaluation. The
last parent must exactly replay the first. Candidate choice cannot change after
opening validation/CONFIRM results.

The CPU auditor reconstructs calibration tables, byte accounting, raw score
arithmetic, independent candidate selection, paired bootstrap and repair gates.
It verifies recorded evidence; it does not independently regenerate GPU logits
or claim arbitrary-hardware numerical equality. Preserve both positive and
negative gate results. See [measured results and limitations](STATE_REPAIR_RESULTS.md).
