# Reproduce the fixed v3 Resurface continuation

Read [the frozen protocol](RESURFACE_MORE_PROTOCOL.md) and the environment/source
requirements in [the v2 guide](QUANT_FIRST_REPRODUCTION.md). This continuation
requires the exact v2 FP32/Adam/GradScaler checkpoint, not only its FP16 export.
The four v2 optimizer checkpoints are retained on the experiment host and are
required by the independent audit. They are not included in this Git repository.
Because backward uses atomic reductions, a new v2 training run is not promised
to reproduce the required parent checkpoint bytes.

## Inputs and discarded smoke

Run from this repository root on the CUDA host. The following existing paths
refer to original FP16 source weights and the completed v2 experiment:

```bash
SQ_PYTHON=/home/horde/.venvs/lodram/bin/python
SQ_SOURCE_DIR=/home/horde/Mamba2-8B-E8W5/models/source
SQ_PARENT_DIR=artifacts/quant_first_v2_training
SQ_CALIBRATION=artifacts/quant_first_v2_calibration/calibration.pt
SQ_PARENT_EVAL_DIR=artifacts/quant_first_v2_eval
SQ_RUN_DIR=artifacts/repro_resurface_more_v3

SQ_TRAIN_ARGS=(
  --source-dir "$SQ_SOURCE_DIR"
  --calibration "$SQ_CALIBRATION"
  --data-root training_data/numeric_v2
  --train-manifest-sha256 451b8703c21120667ef0ea272c21a10d4cd4561779a7b9663773ae47981600c0
  --prose-manifest docs/prose_train_manifest.json
  --prose-tokens /home/horde/mamb2_8B_Recall/training_data/prose/training_tokens.pt
  --parent-training-report "$SQ_PARENT_DIR/report.json"
  --parent-checkpoint "$SQ_PARENT_DIR/checkpoint_1536.pt"
)

OMP_NUM_THREADS=8 "$SQ_PYTHON" -u scripts/train_resurface_more.py \
  "${SQ_TRAIN_ARGS[@]}" --out-dir "$SQ_RUN_DIR/smoke" --smoke
```

The script verifies parent hashes, every restored master/Adam/scaler value and
128-token bitwise forward equality with the parent packed FP16 adapter before
the first update. Smoke updates are discarded; no adapter or checkpoint is
exported. The formal run requires the completed smoke receipt and unchanged code.

## Formal continuation

```bash
OMP_NUM_THREADS=8 "$SQ_PYTHON" -u scripts/train_resurface_more.py \
  "${SQ_TRAIN_ARGS[@]}" --out-dir "$SQ_RUN_DIR/training" \
  --smoke-report "$SQ_RUN_DIR/smoke/report.json"
```

This reloads the original parent checkpoint. It does not continue from smoke.
It adds exactly 3072 successful updates with at most eight overflow retries,
and exports only the final 4608-total-update candidate. The two numeric TRAIN
epochs and continuous learning-rate tail are fixed by the protocol. No held-out
loss selects or alters the training recipe.

Keep `training/report.json`, `adapter_fp16.pt` and all four recovery checkpoints
(`checkpoint_0768.pt`, `checkpoint_1536.pt`, `checkpoint_2304.pt`,
`checkpoint_3072.pt`). Checkpoint filenames count **additional** updates; each
payload also records cumulative updates. Input and code hashes are recorded and
verified before fitting, then rechecked during evaluation and audit. Frozen
source/table identities and gradients are checked at every update. New output
directories are required for independent runs.

## Full comparison and independent audit

The evaluator explicitly fixes backbone RMSNorm to 16 warps, the configuration
that reproduces the archived parent. It records and audits this backend choice.
Read the [replay diagnosis and execution clarification](RESURFACE_MORE_BACKEND_REPLAY.md).
The completed training used process-local autotuning; its configuration was not
recorded. A new training run is therefore not promised to reproduce its bytes.

```bash
SQ_EVAL_ARGS=(
  --source-dir "$SQ_SOURCE_DIR"
  --calibration "$SQ_CALIBRATION"
  --parent-training-report "$SQ_PARENT_DIR/report.json"
  --parent-checkpoint "$SQ_PARENT_DIR/checkpoint_1536.pt"
  --training-report "$SQ_RUN_DIR/training/report.json"
  --parent-eval-dir "$SQ_PARENT_EVAL_DIR"
)

OMP_NUM_THREADS=8 "$SQ_PYTHON" -u scripts/evaluate_resurface_more.py \
  "${SQ_EVAL_ARGS[@]}" --out "$SQ_RUN_DIR/evaluation"

CUDA_VISIBLE_DEVICES= "$SQ_PYTHON" scripts/audit_resurface_more.py \
  "${SQ_EVAL_ARGS[@]}" --eval-dir "$SQ_RUN_DIR/evaluation" \
  --output "$SQ_RUN_DIR/full_independent_audit.json"
```

The evaluator runs parent v2, continued v3, then restored parent v2 on all
130 PPL windows and 768 CONFIRM prompts. The fresh parent must exactly replay
the archived parent result, and the final restored parent must exactly replay
the first arm. S16 and unadapted SQ results are explicitly hash-verified archived
context. The primary gate compares continuation against the **trained v2 parent**.

The CPU auditor operates offline and needs the pinned TRAIN/validation dataset
cache as described in the v2 guide. It checks parent and continuation training,
optimizer/scaler schedules and checkpoint/export identities, raw score arithmetic,
paired bootstrap, cache allocation and both complete parent replays. A training-only
audit can run before evaluation by using `--training-only` and omitting `--eval-dir`.
If the smoke receipt was moved, use `--smoke-report` with its actual location.

The audit reconstructs recorded evidence; it does not rerun GPU logits. Original
source checks cover parameter identities, versions and frozen gradients rather
than complete post-training byte hashes. Training workspace, teacher and optimizer
memory are separate from deployed SQ3.25 cache. All v2 evidence is preserved.
