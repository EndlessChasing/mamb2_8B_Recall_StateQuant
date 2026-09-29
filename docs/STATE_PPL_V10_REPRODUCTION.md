# Reproduce v10: same-budget state tier allocations

Status: **target achieved**. The TRAIN-selected 32/32/64 layout has full PPL
**8.186186562837207 < 8.25**, no Resurface, and exactly 28,499,968 bytes of
persistent cache. The independent full audit passes, including exact parent
replays. Kernel validation passes 57 CPU and 174 GPU checks; 23 runner fixtures
also pass. See [results and artifact hashes](STATE_PPL_TARGET_825.md).

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

Success requires PPL strictly below 8.25, a passing independent full audit,
unchanged actual cache bytes, no Resurface, frozen source/backend/table guards,
and exact parent replay/restoration. Baseline replay compares the entire old
PPL, reset-probe and cache dictionaries. New layout storage descriptors are
recorded separately, including each of the56 layers' actual tensor dimensions.

Weights remain FP16;27.1797 MiB is persistent state/convolution/table storage,
not total GPU memory. No MK result is produced. These corpus and benchmark
families have historical exposure, so this is not an untouched generalization
benchmark. Keep the repository private and preserve every attempted result.

## Load a successful frozen layout for inference

Run this only after the full audit above passes with `target_pass=true`.
Choose one of the two path setups below. The example checks the saved audit,
raw reports, selected layout/table and source bindings before
loading the original FP16 weights. It is statically checked; documentation
does not add another GPU model test.

For a fresh reproduction, reuse the variables from this guide:

```bash
PPL10_INFER_SELECTED="$PPL10_RUN/screen/selected_calibration.pt"
PPL10_INFER_FULL="$PPL10_RUN/full/full_comparison.json"
PPL10_INFER_AUDIT="$PPL10_RUN/full_audit.json"
PPL10_INFER_ARGS=( "${PPL10_MODEL_ARGS[@]}" )
```

For the existing canonical results, use this setup instead. Define the earlier
guide's section 1 variables and this guide's `PPL10_PARENT_ARGS` array first;
rerunning codec checks or model evaluation is unnecessary. Rebuilding the
argument array below supplies the canonical codec receipt exactly once.

```bash
PPL10_INFER_SELECTED=artifacts/state_ppl_v10_screen/selected_calibration.pt
PPL10_INFER_FULL=artifacts/state_ppl_v10_full/full_comparison.json
PPL10_INFER_AUDIT=reports/state_ppl_v10_full_audit.json
PPL10_INFER_ARGS=(
  "${PPL825_MODEL_ARGS[@]}"
  --codec-checks "$PPL825_V6_CHECKS"
  "${PPL10_PARENT_ARGS[@]}"
  --kernel-checks reports/state_ppl_v10_codec_checks.json
)
```

Then run the same inference command for either setup:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 "$PPL825_PYTHON" - \
  --selected-path "$PPL10_INFER_SELECTED" --full-path "$PPL10_INFER_FULL" \
  --audit-path "$PPL10_INFER_AUDIT" "${PPL10_INFER_ARGS[@]}" <<'PY'
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / 'scripts'))
import torch
from mamba2_recall import runtime, resurface_data as data
from evaluate_resurface_more import pin_replay_backend, check_replay_backend
from state_ppl_codec_v10 import StatePPLQuantV10
import run_state_ppl_v10 as v10

parser = argparse.ArgumentParser()
for name in ('selected-path', 'full-path', 'audit-path',
             *v10.prep.INPUT_NAMES, *v10.NEW_INPUT_NAMES):
    parser.add_argument('--' + name, type=Path, required=True)
args = parser.parse_args()
selected_path, full_path, audit_path = args.selected_path, args.full_path, args.audit_path
audit = json.loads(audit_path.read_text())
full = json.loads(full_path.read_text())
assert audit['format'] == 'MAMBA2_STATE_PPL_V10_INDEPENDENT_AUDIT_V1'
assert audit['complete'] and audit['passed'] and not audit['cuda_initialized']
assert audit['stage'] == 'full' and audit['protocol_sha256'] == v10.PROTOCOL_SHA
assert audit['source_sha256'] == data.sha_file('scripts/audit_state_ppl_v10.py')
for name, digest in audit['auditor_dependency_sha256'].items():
    assert data.sha_file(Path('scripts') / name) == digest
assert full['format'] == v10.COMPARE and full['complete'] and full['stage'] == 'full'
assert audit['full']['comparison_sha256'] == data.sha_file(full_path)
assert audit['full']['target_pass'] is True and full['target_pass'] is True
assert audit['full']['target_checks'] == full['target_checks']
assert all(full['target_checks'].values())
assert audit['full']['report_sha256'] == full['report_sha256']
for arm, digest in full['report_sha256'].items():
    assert data.sha_file(full_path.parent / ('full_' + arm + '.json')) == digest
assert audit['selected_calibration']['sha256'] == data.sha_file(selected_path)
assert full['selected_calibration_sha256'] == data.sha_file(selected_path)

candidates, binding, windows, _ = v10.load_inputs(args)
selected, _, _ = v10.load_selection(selected_path, candidates, binding, windows)
assert audit['inputs']['input_binding'] == full['input_binding'] == binding
assert full['code_hashes'] == v10.code_hashes()
assert full['selected_layout'] == selected['selected_layout']
assert audit['selected_calibration']['table_sha256'] == selected['table_sha256']
assert full['adapter_loaded'] is False and full['mk_used'] is False

torch.set_num_threads(8)
torch.manual_seed(20260929)
torch.cuda.manual_seed_all(20260929)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
backend = pin_replay_backend()
tokenizer = runtime.SentencePieceTokenizer(args.source_dir)
assert tokenizer.sha256 == runtime.TOKENIZER_SHA256
model = runtime.load_source_model(args.source_dir, dtype=torch.float16)
assert v10.v6.no_adapter_hooks(model)

with torch.inference_mode(), StatePPLQuantV10(
    model, selected['permutations'], layout=selected['selected_layout']
) as execution:
    ids = torch.tensor([tokenizer.encode('The capital of France is')],
                       dtype=torch.long, device='cuda')
    hidden = execution.backbone(ids, reset=True)
    assert bool(torch.isfinite(hidden).all())
    next_id = int(model.lm_head(hidden[:, -1:]).argmax(dim=-1).item())
    execution.assert_finite_cache()
    cache = execution.cache_breakdown()
    assert cache['total_bytes'] == 28_499_968
    v10.validate_storage_descriptor(execution.storage_descriptor(), selected['selected_layout'])
    check_replay_backend(backend)
    print(json.dumps(dict(
        selected_layout=selected['selected_layout'], table_sha256=selected['table_sha256'],
        full_audit_sha256=data.sha_file(audit_path), full_target_pass=True,
        next_token_id=next_id, next_token_text=tokenizer.decode([next_id]),
        persistent_cache_bytes=cache['total_bytes'], adapter_loaded=False), indent=2))
PY
```

Continue the same request with `execution.backbone(next_ids, reset=False)`
inside the context; use `reset=True` for a new request. The selected artifact
contains the layout, permutation and provenance, not model weights. This
example uses batch one; native `InferenceParams` and variable-length batching
are unsupported. The 28,499,968-byte figure excludes weights and temporary
computation. No Resurface adapter is installed.
