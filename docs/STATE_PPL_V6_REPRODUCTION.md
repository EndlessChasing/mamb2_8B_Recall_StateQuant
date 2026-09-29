# Reproduce v6 PPL-first state optimization

These commands follow the [frozen protocol](STATE_PPL_V6_PROTOCOL.md), SHA256
`86d4e8dc85d79c2c867ce15a846c5938893c86d27e67ad174472975e654e9bff`.
Run from the repository root in Bash on the existing CUDA host. Use a fresh
output root; do not overwrite prior results. The source model, tokenizer,
TRAIN corpus and older artifact provenance are documented in the
[v5 reproduction guide](STATE_FIRST_V5_REPRODUCTION.md).

**Status: codec checks, TRAIN screen and three-arm full PPL confirmation have
completed. The full 1% improvement gate failed, so conditional training was
not executed.** See [the measured results](STATE_PPL_V6_RESULTS.md) and
[numerical validation scope](STATE_PPL_V6_NUMERICS.md).

## Paths and codec checks

```bash
PPL6_PYTHON=/home/horde/.venvs/lodram/bin/python
PPL6_SOURCE=/home/horde/Mamba2-8B-E8W5/models/source
PPL6_TRAIN=/home/horde/mamb2_8B_Recall/training_data/prose/training_tokens.pt
PPL6_CANDIDATES=artifacts/state_first_v5_candidates/candidates.pt
PPL6_V5_SELECTED=artifacts/state_first_v5_screen/selected_calibration.pt
PPL6_OLD_CAL=artifacts/quant_first_v2_calibration/calibration.pt
PPL6_V4_CAL=artifacts/state_repair_calibration/calibration.pt
PPL6_V2_TRAIN=artifacts/quant_first_v2_training/report.json
PPL6_V2_EVAL=artifacts/quant_first_v2_eval
PPL6_V5_EVAL=artifacts/state_first_v5_full
PPL6_DENSE_CHECKS=reports/state_repair_dense3_checks.json
PPL6_RUN=artifacts/repro_state_ppl_v6

mkdir -p "$PPL6_RUN"
```

The source directory contains the original BF16 checkpoint, which the runtime
casts to FP16. Its historical parent-directory name does not select E8/W5.
Preserve companion receipts, all v5 screen reports, and referenced v2/v4
evidence alongside the chosen input paths.

Run the standalone codec checker. Its passing GPU receipt is required for
every model run; a CPU preparation receipt cannot satisfy that requirement:

```bash
PPL6_KERNEL_CHECKS="$PPL6_RUN/codec_checks.json"
CUDA_VISIBLE_DEVICES=0 "$PPL6_PYTHON" scripts/check_state_ppl_codec_v6.py \
  --out "$PPL6_KERNEL_CHECKS"

CUDA_VISIBLE_DEVICES= "$PPL6_PYTHON" scripts/audit_state_ppl_v6.py --self-test
```

## TRAIN screen and independent audit

```bash
PPL6_EVAL_ARGS=(
  --source-dir "$PPL6_SOURCE"
  --candidates "$PPL6_CANDIDATES"
  --v5-selected-calibration "$PPL6_V5_SELECTED"
  --prose-tokens "$PPL6_TRAIN"
  --codec-checks "$PPL6_KERNEL_CHECKS"
)

OMP_NUM_THREADS=8 "$PPL6_PYTHON" scripts/run_state_ppl_v6.py \
  "${PPL6_EVAL_ARGS[@]}" --stage screen --out "$PPL6_RUN/screen"

PPL6_AUDIT_ARGS=(
  --source-dir "$PPL6_SOURCE"
  --calibration "$PPL6_OLD_CAL"
  --repair-calibration "$PPL6_V4_CAL"
  --parent-training-report "$PPL6_V2_TRAIN"
  --prose-tokens "$PPL6_TRAIN"
  --codec-checks "$PPL6_DENSE_CHECKS"
  --candidates "$PPL6_CANDIDATES"
  --v5-calibration "$PPL6_V5_SELECTED"
  --kernel-checks "$PPL6_KERNEL_CHECKS"
)

CUDA_VISIBLE_DEVICES= "$PPL6_PYTHON" scripts/audit_state_ppl_v6.py \
  "${PPL6_AUDIT_ARGS[@]}" --stage screen \
  --eval-dir "$PPL6_RUN/screen" \
  --output "$PPL6_RUN/screen_audit.json"
```

Each arm uses TRAIN rows 40..71, all 2048 tokens: 65,504 prediction targets.
The 20 deployable candidates are selected by PPL alone. Original S16 and the
two larger-memory ablations are diagnostics and cannot win. Legacy restoration
must exactly repeat the baseline. If `selection.stopped` is true in
`screen_comparison.json`, end this experiment and retain v5.

## One frozen candidate: full PPL confirmation

Proceed only after the screen audit passes and selects a nonbaseline candidate.

```bash
PPL6_SELECTED="$PPL6_RUN/screen/selected_calibration.pt"

OMP_NUM_THREADS=8 "$PPL6_PYTHON" scripts/run_state_ppl_v6.py \
  "${PPL6_EVAL_ARGS[@]}" --stage full \
  --selected-calibration "$PPL6_SELECTED" \
  --parent-report "$PPL6_V5_EVAL/full_selected_no_adapter.json" \
  --parent-comparison "$PPL6_V5_EVAL/full_comparison.json" \
  --s16-report "$PPL6_V2_EVAL/full_source_s16.json" \
  --out "$PPL6_RUN/full"

CUDA_VISIBLE_DEVICES= "$PPL6_PYTHON" scripts/audit_state_ppl_v6.py \
  "${PPL6_AUDIT_ARGS[@]}" --stage full \
  --screening-report "$PPL6_RUN/screen/screen_comparison.json" \
  --selected-calibration "$PPL6_SELECTED" \
  --v5-eval-dir "$PPL6_V5_EVAL" \
  --s16-report "$PPL6_V2_EVAL/full_source_s16.json" \
  --eval-dir "$PPL6_RUN/full" \
  --output "$PPL6_RUN/full_audit.json"
```

The baseline, selected candidate and restored baseline each use all 130
validation windows / 264,764 targets. Both baseline runs must match archived
v5 window NLLs and the 128-token hidden/cache probe exactly. Selection cannot
change after validation. The full gate requires at least 1% PPL improvement
with the same 28,499,968-byte persistent cache.

## Conditional recall training

If full PPL improvement is not confirmed, stop and preserve the negative result.
The recorded v6 run takes this branch: 8.367465 → 8.355269 (−0.1458%), below 1%.
If it is confirmed, freeze the quantizer and implement its exact packed forward
and checkpoint/history recomputation for the separately audited fresh Resurface
training path specified in the protocol. Training commands and results are
added only after that prerequisite is met. Do not pass a new scale policy to
the old legacy-only trainer and assume it trains the selected recurrence.
