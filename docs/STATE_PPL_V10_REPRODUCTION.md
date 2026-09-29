# Reproduce v10: same-budget state tier allocations

Status: codec validation passes57 CPU and174 GPU checks. Independent input
audit and model quality measurements are pending; no v10 PPL is available yet. The audited v9 parent has full PPL 8.290607333938492; the best
prior full result is v8 at 8.28386254265227. Neither meets the target.

Follow the [frozen v10 protocol](STATE_PPL_V10_PROTOCOL.md). Run in Bash from
`/home/horde/mamb2_8B_Recall_StateQuant` on the existing CUDA host. Run section 1
of the [earlier reproduction guide](STATE_PPL_TARGET_825_REPRODUCTION.md) first
to define `PPL825_PYTHON`, `PPL825_MODEL_ARGS`, `PPL825_AUDIT_ARGS`,
`PPL825_V6_CHECKS`, and `PPL825_S16`. This uses existing weights and evidence;
it does not require rerunning earlier GPU experiments. Preserve all prior
receipts and use a new output directory for every attempt.

## Fixed inputs and codec checks

```bash
PPL10_RUN=artifacts/repro_state_ppl_v10_01
test ! -e "$PPL10_RUN"
mkdir -p "$PPL10_RUN"

PPL10_PARENT_ARGS=(
  --layer-candidates artifacts/state_ppl_v8_calibration/candidates.pt
  --v8-selected-calibration artifacts/state_ppl_v8_screen/selected_calibration.pt
  --v8-screen-audit reports/state_ppl_v8_screen_audit.json
  --v8-full-dir artifacts/state_ppl_v8_full
  --v8-full-audit reports/state_ppl_v8_full_audit.json
  --group-candidates artifacts/state_ppl_v9_calibration/candidates.pt
  --v9-selected-calibration artifacts/state_ppl_v9_screen/selected_calibration.pt
  --v9-screen-audit reports/state_ppl_v9_screen_audit.json
  --v9-full-dir artifacts/state_ppl_v9_full
  --v9-full-audit reports/state_ppl_v9_full_audit.json
)

CUDA_VISIBLE_DEVICES= "$PPL825_PYTHON" scripts/check_state_ppl_codec_v10.py \
  --cpu-only --out "$PPL10_RUN/codec_cpu.json"

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 "$PPL825_PYTHON" \
  scripts/check_state_ppl_codec_v10.py --out "$PPL10_RUN/codec_gpu.json"

PPL10_MODEL_ARGS=(
  "${PPL825_MODEL_ARGS[@]}"
  --codec-checks "$PPL825_V6_CHECKS"
  "${PPL10_PARENT_ARGS[@]}"
  --kernel-checks "$PPL10_RUN/codec_gpu.json"
)
PPL10_AUDIT_ARGS=(
  "${PPL825_AUDIT_ARGS[@]}"
  --kernel-checks "$PPL825_V6_CHECKS"
  "${PPL10_PARENT_ARGS[@]}"
  --layout-checks "$PPL10_RUN/codec_gpu.json"
  --s16-report "$PPL825_S16"
)

CUDA_VISIBLE_DEVICES= "$PPL825_PYTHON" scripts/audit_state_ppl_v10.py \
  "${PPL10_AUDIT_ARGS[@]}" --stage inputs \
  --output "$PPL10_RUN/inputs_audit.json"
```

Proceed only when the independent input audit passes with CUDA uninitialized.
It verifies the audited v9 miss, fixed TRAIN-selected parent, unchanged source
and tokenizer, codec evidence, and the complete input/source hash chain. Keep
any `.pt` raw evidence emitted beside a codec JSON; the receipt binds it by hash.
CPU audit reconstructs recorded GPU evidence rather than rerunning model logits.

## Four-candidate TRAIN screen

No new coordinate calibration is performed. The exact v9 TRAIN-selected
permutation is fixed, even though its full PPL slightly regressed. Compare
four global `(INT8, INT4, zero)` allocations in this order: `(16,64,48)`,
`(8,80,40)`, `(24,48,56)`, `(32,32,64)`, followed by restored baseline.
All use 48 payload bytes plus two FP16 scales: 52 bytes per 128 coordinates,
with total persistent batch-one cache 28,499,968 bytes. The baseline directly
delegates the frozen v6 codec; all other layouts must pass physical packing,
recurrence and padding checks before this screen.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 "$PPL825_PYTHON" \
  scripts/run_state_ppl_v10.py "${PPL10_MODEL_ARGS[@]}" \
  --stage screen --out "$PPL10_RUN/screen"

CUDA_VISIBLE_DEVICES= "$PPL825_PYTHON" scripts/audit_state_ppl_v10.py \
  "${PPL10_AUDIT_ARGS[@]}" --stage screen \
  --eval-dir "$PPL10_RUN/screen" \
  --output "$PPL10_RUN/screen_audit.json"
```

Every arm uses TRAIN rows184..215, 32 full windows and 65,504 targets. The
minimum complete finite PPL wins; exact ties prefer baseline, then fixed
layout order. The selected payload retains the unchanged permutation and the
winning global layout. If baseline wins, preserve the result and do not run
a redundant full confirmation. Do not choose a runner-up on validation.

## Full confirmation

Only a nonbaseline TRAIN winner advances. The three arms are the v9 parent,
the frozen selected layout and restored v9 parent, each using 130 validation
windows and 264,764 targets.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 "$PPL825_PYTHON" \
  scripts/run_state_ppl_v10.py "${PPL10_MODEL_ARGS[@]}" \
  --stage full --selected-calibration "$PPL10_RUN/screen/selected_calibration.pt" \
  --s16-report "$PPL825_S16" --out "$PPL10_RUN/full"

CUDA_VISIBLE_DEVICES= "$PPL825_PYTHON" scripts/audit_state_ppl_v10.py \
  "${PPL10_AUDIT_ARGS[@]}" --stage full \
  --screening-report "$PPL10_RUN/screen/screen_comparison.json" \
  --selected-calibration "$PPL10_RUN/screen/selected_calibration.pt" \
  --eval-dir "$PPL10_RUN/full" --output "$PPL10_RUN/full_audit.json"
```

Success requires PPL strictly below8.25, a passing independent full audit,
unchanged actual cache bytes, no Resurface, frozen source/backend/table guards,
and exact parent replay/restoration. Baseline replay compares the entire old
PPL, reset-probe and cache dictionaries. New layout storage descriptors are
recorded separately, including each of the56 layers' actual tensor dimensions.

Weights remain FP16;27.1797MiB is persistent state/convolution/table storage,
not total GPU memory. No MK result is produced. These corpus and benchmark
families have historical exposure, so this is not an untouched generalization
benchmark. Keep the repository private and preserve every attempted result.
