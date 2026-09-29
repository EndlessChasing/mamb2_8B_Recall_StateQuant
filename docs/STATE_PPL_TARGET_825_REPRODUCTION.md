# Reproduce the unadapted Q3.25 PPL < 8.25 search

These commands implement the frozen [v7 protocol](STATE_PPL_V7_PROTOCOL.md)
(`84ffdce1d9e5d5dbac2b6996072f0db09bd1bf79fd3c55dad8b8f35973cd5d14`)
and conditional [v8 protocol](STATE_PPL_V8_PROTOCOL.md)
(`880839c0b0919d0a9a119647d2d9c1657e5b8eb5389c8c2ee22ee1785917aa01`).
Run from the existing repository on the CUDA host, in **Bash**, one GPU job
at a time. The examples reference existing source weights and inputs in place;
they do not copy weights or change the frozen source.

**Preparation checkpoint:** v7's 119 GPU codec checks and both routes' CPU
input audits have passed. The independently audited v7 screen retained baseline,
so no v7 full run is needed. The guide was prepared before v8 model measurements;
v8 calibration has since completed, with its screen/full confirmation still pending.
The commands below describe the complete flow, including prospective stages;
they do not claim that all stages have run. See [current results/status](STATE_PPL_TARGET_825.md).

Both routes measure PPL without an adapter. There is no MK generation, scoring,
selection or Resurface training. The batch-one persistent cache must remain
**28,499,968 bytes**, including the packed Q3.25 state, FP16 convolution cache
and one 57,344-byte permutation table. This is not total GPU memory. The full
target is **strictly below 8.25**, with exact replay and all integrity checks;
there is no additional 1% improvement gate.

## 1. Existing inputs and fresh output paths

The environment and upstream preparation are described in the
[v5 reproduction guide](STATE_FIRST_V5_REPRODUCTION.md) and
[v6 reproduction guide](STATE_PPL_V6_REPRODUCTION.md). Preserve the companion
JSON receipts and raw reports beside all referenced payloads, including every
v5/v6 screen arm, the original v2 calibration/training provenance, and the
v4 calibration/codec evidence. GPU-kernel receipts also reference their retained
numerical evidence files; copying only a receipt is insufficient.

```bash
set -euo pipefail
cd /home/horde/mamb2_8B_Recall_StateQuant

PPL825_PYTHON=/home/horde/.venvs/lodram/bin/python
PPL825_SOURCE=/home/horde/Mamba2-8B-E8W5/models/source
PPL825_TRAIN=/home/horde/mamb2_8B_Recall/training_data/prose/training_tokens.pt
PPL825_CANDIDATES=artifacts/state_first_v5_candidates/candidates.pt
PPL825_V5_SELECTED=artifacts/state_first_v5_screen/selected_calibration.pt
PPL825_V6_SELECTED=artifacts/state_ppl_v6_screen/selected_calibration.pt
PPL825_V6_EVAL=artifacts/state_ppl_v6_full
PPL825_S16=artifacts/quant_first_v2_eval/full_source_s16.json
PPL825_OLD_CAL=artifacts/quant_first_v2_calibration/calibration.pt
PPL825_V4_CAL=artifacts/state_repair_calibration/calibration.pt
PPL825_V2_TRAIN=artifacts/quant_first_v2_training/report.json
PPL825_DENSE_CHECKS=reports/state_repair_dense3_checks.json
PPL825_V6_CHECKS=reports/state_ppl_v6_codec_checks.json
PPL825_V7_CHECKS=reports/state_ppl_v7_codec_checks.json

PPL825_RUN=artifacts/repro_state_ppl_target_825_01
test ! -e "$PPL825_RUN"
mkdir -p "$PPL825_RUN"

PPL825_MODEL_ARGS=(
  --source-dir "$PPL825_SOURCE"
  --candidates "$PPL825_CANDIDATES"
  --v5-selected-calibration "$PPL825_V5_SELECTED"
  --v6-selected-calibration "$PPL825_V6_SELECTED"
  --prose-tokens "$PPL825_TRAIN"
)
PPL825_AUDIT_ARGS=(
  --source-dir "$PPL825_SOURCE"
  --calibration "$PPL825_OLD_CAL"
  --repair-calibration "$PPL825_V4_CAL"
  --parent-training-report "$PPL825_V2_TRAIN"
  --prose-tokens "$PPL825_TRAIN"
  --codec-checks "$PPL825_DENSE_CHECKS"
  --candidates "$PPL825_CANDIDATES"
  --v5-calibration "$PPL825_V5_SELECTED"
  --v6-selected-calibration "$PPL825_V6_SELECTED"
)
PPL7_AUDIT_ARGS=(
  "${PPL825_AUDIT_ARGS[@]}"
  --v6-kernel-checks "$PPL825_V6_CHECKS"
  --kernel-checks "$PPL825_V7_CHECKS"
)
PPL8_AUDIT_ARGS=(
  "${PPL825_AUDIT_ARGS[@]}"
  --kernel-checks "$PPL825_V6_CHECKS"
)
```

Choose another unused `PPL825_RUN` for a retry; all stage directories and audit
filenames must be fresh. Keep failed attempts. The source path's historical
`Mamba2-8B-E8W5` parent name does not select compressed weights: these scripts
load the original source checkpoint and compute with frozen FP16 weights.

## 2. Codec and input prerequisites

Use the canonical passing receipts already bound to the prior artifacts:

| Evidence | SHA256 | Scope |
| --- | --- | --- |
| v6 codec receipt | `81a65cd0f56e6ad4a7a87fc464abde6509a4315143c38c52c88a67be61e4a54f` | 249 GPU checks; stored-scale codec used by v8 |
| v7 codec receipt | `39e144981be100ce5e5f0003342370623ed2e1c9d349f01a29a8b0b3a32b507a` | 119 GPU checks; uniform delegates v6 exactly |
| v6 selected payload | `098930d1af5e5821b277640d236f7607c6428b48d36f7117e2ea4aa87e656303` | Fixed preserve-INT8 table with stored-scale quantization |

The auditors validate the kernel evidence and source hashes; a CPU-only checker
receipt cannot stand in for the required GPU receipt. In particular, the pinned
v6 selected payload binds the canonical v6 receipt. A newly generated receipt
must not be silently substituted into that existing provenance chain.

Run both independent input audits before model measurements:

```bash
CUDA_VISIBLE_DEVICES= "$PPL825_PYTHON" scripts/audit_state_ppl_v7.py \
  "${PPL7_AUDIT_ARGS[@]}" --stage inputs \
  --output "$PPL825_RUN/v7_inputs_audit.json"

CUDA_VISIBLE_DEVICES= "$PPL825_PYTHON" scripts/audit_state_ppl_v8.py \
  "${PPL8_AUDIT_ARGS[@]}" --stage inputs \
  --output "$PPL825_RUN/v8_inputs_audit.json"
```

Both must finish with `complete=true`, `passed=true`, `cuda_initialized=false`.
This validates v8 inputs without starting its conditional model experiment.
The frozen production/model scripts must continue matching the recorded hashes.

## 3. v7: fixed 16-candidate TRAIN screen

Four existing v5 tables cross four fixed INT4 codebooks: uniform, mild,
quadratic and FP4-like. Each arm uses TRAIN rows 80..111, all 2048 tokens per
row: 32 windows / 65,504 prediction targets. Baseline and restored baseline
use the unchanged v6 stored-scale policy. PPL selects the winner; exact ties
prefer baseline, then the frozen table/codebook order.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 "$PPL825_PYTHON" \
  scripts/run_state_ppl_v7.py "${PPL825_MODEL_ARGS[@]}" \
  --v6-kernel-checks "$PPL825_V6_CHECKS" --kernel-checks "$PPL825_V7_CHECKS" \
  --stage screen --out "$PPL825_RUN/v7_screen"

CUDA_VISIBLE_DEVICES= "$PPL825_PYTHON" scripts/audit_state_ppl_v7.py \
  "${PPL7_AUDIT_ARGS[@]}" --stage screen \
  --eval-dir "$PPL825_RUN/v7_screen" \
  --output "$PPL825_RUN/v7_screen_audit.json"
```

Proceed only after that audit passes. Inspect
`v7_screen/screen_comparison.json → selection.baseline_wins`.

If it is **true**, skip v7 full evaluation and record the audited baseline
outcome for the v8 prerequisite:

```bash
PPL7_OUTCOME="$PPL825_RUN/v7_screen/screen_comparison.json"
PPL7_OUTCOME_AUDIT="$PPL825_RUN/v7_screen_audit.json"
```

The unchanged v6 baseline's pinned full PPL is 8.355268708845868, above target.
If a **nonbaseline** candidate wins, use section 4 instead; its full result
is required before deciding whether v8 may start.

## 4. v7: confirm only the frozen TRAIN winner

Run this section only for a nonbaseline v7 winner. All three arms use the
130 validation windows / 264,764 targets. The first and restored baseline
must reproduce the pinned v6 NLLs and 128-token hidden/cache probes exactly.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 "$PPL825_PYTHON" \
  scripts/run_state_ppl_v7.py "${PPL825_MODEL_ARGS[@]}" \
  --v6-kernel-checks "$PPL825_V6_CHECKS" --kernel-checks "$PPL825_V7_CHECKS" \
  --stage full \
  --selected-calibration "$PPL825_RUN/v7_screen/selected_calibration.pt" \
  --parent-report "$PPL825_V6_EVAL/full_selected.json" \
  --parent-comparison "$PPL825_V6_EVAL/full_comparison.json" \
  --s16-report "$PPL825_S16" --out "$PPL825_RUN/v7_full"

CUDA_VISIBLE_DEVICES= "$PPL825_PYTHON" scripts/audit_state_ppl_v7.py \
  "${PPL7_AUDIT_ARGS[@]}" --stage full \
  --screening-report "$PPL825_RUN/v7_screen/screen_comparison.json" \
  --selected-calibration "$PPL825_RUN/v7_screen/selected_calibration.pt" \
  --v6-eval-dir "$PPL825_V6_EVAL" --s16-report "$PPL825_S16" \
  --eval-dir "$PPL825_RUN/v7_full" --output "$PPL825_RUN/v7_full_audit.json"

PPL7_OUTCOME="$PPL825_RUN/v7_full/full_comparison.json"
PPL7_OUTCOME_AUDIT="$PPL825_RUN/v7_full_audit.json"
```

If the independent audit passes and `target_pass=true`, the user target is
met: **do not run v8**. A failed integrity/runtime check requires diagnosis,
not classification as a poor finite candidate. Only an independently audited
finite v7 miss permits the v8 model stages below.

## 5. Conditional v8: 170-arm single-layer calibration

The v8 baseline stays **v6 preserve-INT8 + stored-scale**, regardless of which
v7 codebook performed best. It uses the unchanged v6 kernel, with no v7 kernel
dependency. First validate the chosen v7 outcome/audit as a CPU prerequisite:

```bash
PPL8_START_ARGS=(
  --v7-outcome "$PPL7_OUTCOME"
  --v7-outcome-audit "$PPL7_OUTCOME_AUDIT"
  --parent-report "$PPL825_V6_EVAL/full_selected.json"
)

CUDA_VISIBLE_DEVICES= "$PPL825_PYTHON" scripts/audit_state_ppl_v8.py \
  "${PPL8_AUDIT_ARGS[@]}" "${PPL8_START_ARGS[@]}" --stage inputs \
  --output "$PPL825_RUN/v8_start_audit.json"
```

This audit rejects a v7 target success or a nonbaseline v7 screen winner
without full confirmation. Launch ordering remains an orchestration obligation;
the audit does not infer when GPU jobs ran from file timestamps.

Calibrate on TRAIN rows 72..79, eight complete 2048-token windows / 16,376
targets. Execute baseline, each of the 56 × 3 single-layer table replacements,
then restored baseline. Per layer, select the smallest raw NLL alternative
(exact ties: magnitude, full-readout, preserve-retained80) and retain only
strict improvement over baseline. Rank retained swaps by delta NLL, layer,
alternative order. Export baseline and combined top-1/2/4/8/16/all-negative
tables, deduplicating identical table contents and keeping the first label.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 "$PPL825_PYTHON" \
  scripts/prepare_state_ppl_v8.py "${PPL825_MODEL_ARGS[@]}" \
  --codec-checks "$PPL825_V6_CHECKS" --out "$PPL825_RUN/v8_calibration"

CUDA_VISIBLE_DEVICES= "$PPL825_PYTHON" scripts/audit_state_ppl_v8.py \
  "${PPL8_AUDIT_ARGS[@]}" "${PPL8_START_ARGS[@]}" --stage calibration \
  --calibration-dir "$PPL825_RUN/v8_calibration" \
  --output "$PPL825_RUN/v8_calibration_audit.json"
```

The audit reconstructs every layer replacement, raw-window NLL arithmetic,
ranking, combined table, deduplication decision and exact restoration. An
individual layer's improvement is a proposal heuristic; gains are not assumed
additive. A known nonfinite single-layer result can be excluded with evidence.
Arbitrary runtime errors and integrity failures are fatal.

If no layer improves, the exported payload contains only baseline and marks
`calibration_stopped=true`. Preserve this negative family result and skip the
redundant screen/full stages. A distinct next family needs its own recorded
TRAIN selection procedure.

## 6. v8: disjoint TRAIN screen of combined tables

After the calibration audit passes and at least one distinct combination
exists, screen the exported tables on TRAIN rows 112..143: 32 complete windows /
65,504 targets. These rows are disjoint from v8 calibration, v7 selection and
v6 selection. Select minimum PPL, with baseline priority and then export order
on exact ties. The baseline is restored after all unique combinations.

```bash
PPL8_LAYER_CANDIDATES="$PPL825_RUN/v8_calibration/candidates.pt"

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 "$PPL825_PYTHON" \
  scripts/run_state_ppl_v8.py "${PPL825_MODEL_ARGS[@]}" \
  --codec-checks "$PPL825_V6_CHECKS" --layer-candidates "$PPL8_LAYER_CANDIDATES" \
  --stage screen --out "$PPL825_RUN/v8_screen"

CUDA_VISIBLE_DEVICES= "$PPL825_PYTHON" scripts/audit_state_ppl_v8.py \
  "${PPL8_AUDIT_ARGS[@]}" "${PPL8_START_ARGS[@]}" --stage screen \
  --calibration-dir "$PPL825_RUN/v8_calibration" \
  --layer-candidates "$PPL8_LAYER_CANDIDATES" --eval-dir "$PPL825_RUN/v8_screen" \
  --output "$PPL825_RUN/v8_screen_audit.json"
```

If `selection.baseline_wins=true`, preserve that result and skip full v8
validation. Otherwise the exported `selected_calibration.pt` fixes one table;
neither validation nor MK may change its swaps or quantizer.

## 7. v8: full confirmation and independent audit

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 "$PPL825_PYTHON" \
  scripts/run_state_ppl_v8.py "${PPL825_MODEL_ARGS[@]}" \
  --codec-checks "$PPL825_V6_CHECKS" --layer-candidates "$PPL8_LAYER_CANDIDATES" \
  --stage full --selected-calibration "$PPL825_RUN/v8_screen/selected_calibration.pt" \
  --parent-report "$PPL825_V6_EVAL/full_selected.json" \
  --parent-comparison "$PPL825_V6_EVAL/full_comparison.json" \
  --s16-report "$PPL825_S16" --out "$PPL825_RUN/v8_full"

CUDA_VISIBLE_DEVICES= "$PPL825_PYTHON" scripts/audit_state_ppl_v8.py \
  "${PPL8_AUDIT_ARGS[@]}" "${PPL8_START_ARGS[@]}" --stage full \
  --calibration-dir "$PPL825_RUN/v8_calibration" \
  --layer-candidates "$PPL8_LAYER_CANDIDATES" \
  --screening-report "$PPL825_RUN/v8_screen/screen_comparison.json" \
  --selected-calibration "$PPL825_RUN/v8_screen/selected_calibration.pt" \
  --parent-comparison "$PPL825_V6_EVAL/full_comparison.json" \
  --s16-report "$PPL825_S16" --eval-dir "$PPL825_RUN/v8_full" \
  --output "$PPL825_RUN/v8_full_audit.json"
```

All three arms—v6 baseline, frozen selected mixture, restored baseline—use
130 windows / 264,764 targets. The v8 runner checks the entire archived v6
PPL, reset-probe and cache dictionaries for exact equality. Success requires
`target_pass=true` and a passing independent full audit, with no adapter and
the exact fixed cache budget. Original S16 PPL 7.334322057221965 is comparison
context, not another candidate.

The CPU auditor reconstructs recorded evidence; it does not regenerate GPU
model logits. These benchmark/data families have historical exposure and are
not untouched generalization evidence. A failed family does not establish
that further compression repair is impossible. Preserve all results, keep the
repository private, and do not expand a frozen grid using validation results.

## 8. Load the frozen v8 table for inference

After the selected table has a completed full audit, this example loads that
exact table and runs one next-token prediction. Run from the repository root
with the same `PPL825_RUN` as above. It checks the saved full evidence, then
reconstructs calibration and TRAIN selection provenance before loading weights.
This is an inference example; it does not replace PPL validation or claim that
the saved candidate met the target. The printed `full_target_pass` comes from
the audited full outcome.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 "$PPL825_PYTHON" - "$PPL825_RUN" <<'PY'
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path.cwd() / 'scripts'))
import torch
from mamba2_recall import runtime, resurface_data as data
from evaluate_resurface_more import pin_replay_backend, check_replay_backend
from state_ppl_codec_v6 import StatePPLQuant
import run_state_ppl_v8 as v8

run = Path(sys.argv[1])
selected_path = run / 'v8_screen/selected_calibration.pt'
full_path = run / 'v8_full/full_comparison.json'
audit = json.loads((run / 'v8_full_audit.json').read_text())
full = json.loads(full_path.read_text())
assert audit['complete'] and audit['passed'] and not audit['cuda_initialized']
assert audit['stage'] == 'full' and full['complete']
assert audit['source_sha256'] == data.sha_file('scripts/audit_state_ppl_v8.py')
assert audit['full']['comparison_sha256'] == data.sha_file(full_path)
assert audit['selected_calibration']['sha256'] == data.sha_file(selected_path)
assert full['selected_calibration_sha256'] == data.sha_file(selected_path)

args = SimpleNamespace(
    candidates=Path('artifacts/state_first_v5_candidates/candidates.pt'),
    v5_selected_calibration=Path('artifacts/state_first_v5_screen/selected_calibration.pt'),
    v6_selected_calibration=Path('artifacts/state_ppl_v6_screen/selected_calibration.pt'),
    prose_tokens=Path('/home/horde/mamb2_8B_Recall/training_data/prose/training_tokens.pt'),
    codec_checks=Path('reports/state_ppl_v6_codec_checks.json'),
    layer_candidates=run / 'v8_calibration/candidates.pt',
)
candidates, binding, train_windows = v8.load_inputs(args)
selected, _, _ = v8.load_selection(selected_path, candidates, binding, train_windows)

torch.set_num_threads(8)
torch.manual_seed(20260929)
torch.cuda.manual_seed_all(20260929)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
backend = pin_replay_backend()  # Must precede the first model forward.
source = Path('/home/horde/Mamba2-8B-E8W5/models/source')
tokenizer = runtime.SentencePieceTokenizer(source)
model = runtime.load_source_model(source, dtype=torch.float16)
assert v8.v6.no_adapter_hooks(model)

with torch.inference_mode(), StatePPLQuant(
    model, selected['permutations'], scale_mode='stored_scale', int4_clip=1.0
) as execution:
    ids = torch.tensor([tokenizer.encode('The capital of France is')],
                       dtype=torch.long, device='cuda')
    hidden = execution.backbone(ids, reset=True)
    next_id = int(model.lm_head(hidden[:, -1:]).argmax(dim=-1).item())
    execution.assert_finite_cache()
    cache = execution.cache_breakdown()
    assert cache['total_bytes'] == 28_499_968
    check_replay_backend(backend)
    print(json.dumps({
        'selected_id': selected['selected_id'],
        'table_sha256': selected['table_sha256'],
        'next_token_id': next_id,
        'next_token_text': tokenizer.decode([next_id]),
        'persistent_cache_bytes': cache['total_bytes'],
        'full_target_pass': full['target_pass'],
        'adapter_loaded': False,
    }, indent=2))
PY
```

The selected artifact contains the static coordinate table and provenance,
not model weights. Weights and linear computation remain FP16; the controller
owns the packed recurrent cache. No Resurface adapter is installed. Continue
the same request with `execution.backbone(next_ids, reset=False)` while the
context is active; use `reset=True` for a new independent request. Leaving the
context restores the original mixer methods and releases its request caches.
Native `InferenceParams` and variable-length batching are not supported by
this controller API.
