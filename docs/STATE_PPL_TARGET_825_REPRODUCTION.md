# Reproduce the unadapted Q3.25 PPL < 8.25 search

These commands implement the frozen [v7 protocol](STATE_PPL_V7_PROTOCOL.md)
(`84ffdce1d9e5d5dbac2b6996072f0db09bd1bf79fd3c55dad8b8f35973cd5d14`)
and conditional [v8 protocol](STATE_PPL_V8_PROTOCOL.md)
(`880839c0b0919d0a9a119647d2d9c1657e5b8eb5389c8c2ee22ee1785917aa01`),
followed by the separately frozen [v9 protocol](STATE_PPL_V9_PROTOCOL.md)
(`9e01c03ee6870a8ecbcd9a0ba9157b1651d2830ea65d2951664ebcb81b4a09b3`).
Run from the existing repository on the CUDA host, in **Bash**, one GPU job
at a time. The examples reference existing source weights and inputs in place;
they do not copy weights or change the frozen source.

**Completed v7/v8 results:** v7's independently audited TRAIN screen retained
baseline, so no v7 full run was needed. V8 calibrated 170 arms, exported seven
distinct combined tables, and selected `top8` on its separate TRAIN screen.
Full validation then measured **8.28386254265227**, versus the unchanged v6
baseline's **8.355268708845868**. The independent full audit passed, including
exact baseline replay/restoration and the unchanged cache budget; the strict
**PPL < 8.25 target remains unmet**. The commands below reproduce those stages.
See [current results/status](STATE_PPL_TARGET_825.md).

The canonical v8 full audit is
[`reports/state_ppl_v8_full_audit.json`](../reports/state_ppl_v8_full_audit.json),
SHA256 `051316fe074ea1ba9cfeef87ba7fe4bb9c3a4c18e3c7427c9a2ef256d490e9cf`.
Its completed finite miss permits the separately frozen v9 procedure described
below; it does not permit changing the v8 winner or revisiting its candidate grid.

All three routes measure PPL without an adapter. There is no MK generation, scoring,
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

## 9. Conditional v9: pinned parent and start audit

V9 refines groups within the v8 TRAIN-selected `top8` table. Its protocol was
frozen while v8 full validation was running, before that result was seen.
The completed finite v8 miss now satisfies its start condition. The parent,
eligible layers, group alternatives, TRAIN rows and candidate grid stay fixed;
v8 validation does not select a new parent or an intervention.

This route pins the **canonical original v8 artifacts**, including their exact
serialized payload and audit hashes. Use the paths below, preserving all their
raw reports and upstream receipts. Outputs freshly reproduced in sections 5–7
can have different receipt hashes even if table bytes agree; they cannot be
substituted for these pinned inputs to the frozen v9 experiment. The existing
`PPL825_MODEL_ARGS`, `PPL825_AUDIT_ARGS`, Python and source variables come from
section 1. Choose a fresh v9 output directory independently of earlier reruns.
To reproduce v9 alone using the retained evidence, run section 1's setup and
then sections 9–13; rerunning the earlier GPU experiments is unnecessary.

```bash
PPL9_RUN=artifacts/repro_state_ppl_v9_01
test ! -e "$PPL9_RUN"
mkdir -p "$PPL9_RUN"

PPL9_PARENT_ARGS=(
  --layer-candidates artifacts/state_ppl_v8_calibration/candidates.pt
  --v8-selected-calibration artifacts/state_ppl_v8_screen/selected_calibration.pt
  --v8-screen-audit reports/state_ppl_v8_screen_audit.json
  --v8-full-dir artifacts/state_ppl_v8_full
  --v8-full-audit reports/state_ppl_v8_full_audit.json
)
PPL9_MODEL_ARGS=(
  "${PPL825_MODEL_ARGS[@]}"
  --codec-checks "$PPL825_V6_CHECKS"
  "${PPL9_PARENT_ARGS[@]}"
)
PPL9_AUDIT_ARGS=(
  "${PPL825_AUDIT_ARGS[@]}"
  --kernel-checks "$PPL825_V6_CHECKS"
  "${PPL9_PARENT_ARGS[@]}"
)

CUDA_VISIBLE_DEVICES= "$PPL825_PYTHON" scripts/audit_state_ppl_v9.py \
  "${PPL9_AUDIT_ARGS[@]}" --stage inputs \
  --output "$PPL9_RUN/inputs_audit.json"
```

Proceed only after the input audit reports `complete=true`, `passed=true`,
`cuda_initialized=false`. It must validate the completed v8 full miss, exact
parent table and the complete upstream provenance. The canonical v8 selected
payload SHA256 is
`9b7c04814085abbb67d7a10e0b70d1a6e9b34e97299fc0dc26e4373ea8f8ea39`;
its full comparison SHA256 is
`4fcafaadf18dbd1b0a5cdbf51b035c16fc66b2cf56a5bfe37dea402293ff493b`.
The full audit hash is recorded at the beginning of this guide. A passing full
audit with PPL below 8.25 would prohibit starting v9 under this protocol.

## 10. v9: single-group calibration

The eligible layers are **0, 3, 6, 1, 4, 2, 7, 10**, in that fixed order from
v8's TRAIN calibration ranking. Each of their eight groups is considered.
For each group, inspect the four existing v5 table alternatives in their
original order, skip bytes identical to the parent, and deduplicate identical
remaining group bytes. The intervention inventory is saved before the first
model forward. The actual pinned inputs yield **192 interventions plus the
parent baseline and its restoration**, 194 arms total.

Every arm uses TRAIN rows **144..151**, eight complete windows / **16,376
targets**. All other groups retain their parent bytes. Within each group, use
the lowest raw NLL alternative with fixed table-order ties, retaining only
strict improvement. Rank retained changes by delta NLL, numeric layer, group,
then table order. The fixed combined proposals are baseline, top1, top2, top4,
top8, top16, top32 and allnegative; cap counts and deduplicate complete tables.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 "$PPL825_PYTHON" \
  scripts/prepare_state_ppl_v9.py "${PPL9_MODEL_ARGS[@]}" \
  --out "$PPL9_RUN/calibration"

CUDA_VISIBLE_DEVICES= "$PPL825_PYTHON" scripts/audit_state_ppl_v9.py \
  "${PPL9_AUDIT_ARGS[@]}" --stage calibration \
  --calibration-dir "$PPL9_RUN/calibration" \
  --output "$PPL9_RUN/calibration_audit.json"
```

The independent CPU audit reconstructs the actual group bytes, intervention
inventory, raw NLL ranking and each exported combination. If
`candidates.json → calibration_stopped=true`, preserve the negative result
and skip screen/full. Single-group gains do not establish additive gains.

## 11. v9: disjoint screen and one frozen winner

After calibration passes its audit, evaluate every distinct combined table
on TRAIN rows **152..183**, 32 complete windows / **65,504 targets**, followed
by restored parent. At most eight candidates plus restoration are allowed.
These rows are disjoint from the preceding v6/v7/v8/v9 selection/calibration
rows. Minimum finite exact-budget PPL wins, with exact ties preferring parent
baseline and then export order. No MK or adapter is used.

```bash
PPL9_GROUP_CANDIDATES="$PPL9_RUN/calibration/candidates.pt"

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 "$PPL825_PYTHON" \
  scripts/run_state_ppl_v9.py "${PPL9_MODEL_ARGS[@]}" \
  --group-candidates "$PPL9_GROUP_CANDIDATES" \
  --stage screen --out "$PPL9_RUN/screen"

CUDA_VISIBLE_DEVICES= "$PPL825_PYTHON" scripts/audit_state_ppl_v9.py \
  "${PPL9_AUDIT_ARGS[@]}" --stage screen \
  --calibration-dir "$PPL9_RUN/calibration" \
  --group-candidates "$PPL9_GROUP_CANDIDATES" --eval-dir "$PPL9_RUN/screen" \
  --output "$PPL9_RUN/screen_audit.json"
```

If `screen_comparison.json → selection.baseline_wins=true`, preserve the
result and skip full confirmation. Otherwise the exported
`screen/selected_calibration.pt` fixes the single table allowed to advance.
A failed integrity/runtime check must be diagnosed; only recognized
nonfinite candidates may be excluded with their evidence retained.

## 12. v9: full confirmation against the exact v8 parent

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 "$PPL825_PYTHON" \
  scripts/run_state_ppl_v9.py "${PPL9_MODEL_ARGS[@]}" \
  --group-candidates "$PPL9_GROUP_CANDIDATES" \
  --stage full --selected-calibration "$PPL9_RUN/screen/selected_calibration.pt" \
  --s16-report "$PPL825_S16" --out "$PPL9_RUN/full"

CUDA_VISIBLE_DEVICES= "$PPL825_PYTHON" scripts/audit_state_ppl_v9.py \
  "${PPL9_AUDIT_ARGS[@]}" --stage full \
  --calibration-dir "$PPL9_RUN/calibration" \
  --group-candidates "$PPL9_GROUP_CANDIDATES" \
  --screening-report "$PPL9_RUN/screen/screen_comparison.json" \
  --selected-calibration "$PPL9_RUN/screen/selected_calibration.pt" \
  --s16-report "$PPL825_S16" --eval-dir "$PPL9_RUN/full" \
  --output "$PPL9_RUN/full_audit.json"
```

All three arms—`v8_baseline`, `selected`, `restored_baseline`—use the same
130 validation windows / **264,764 targets**. Both parent evaluations must
repeat the archived v8 selected result, including complete PPL, reset probe
and actual cache dictionaries. All target checks must pass:
`ppl_strictly_below_8p25`, `cache_same_budget` and
`all_integrity_checks_passed`, followed by the independent full audit.

Group refinement stores one 57,344-byte uint8 permutation table. The existing
v6 codec still uses 52 bytes per state row and **28,499,968 persistent bytes**
at batch one. No new codec, extra resident table, predictor, parameter or
Resurface adapter is introduced. The original FP16 source weights remain
separate from this cache budget. This is a bounded, historically exposed
benchmark search; an improvement does not establish generalization to other
models, tasks or context lengths.

## 13. Load a successful frozen v9 table for inference

Run this example only after section 12 has a **passing full audit and
`target_pass=true`**. It refuses a failed or incomplete target outcome.
The source weights, request-reset semantics and controller API are identical
to section 8; v9 changes only the validated static table passed to that
controller. This documented example has been checked statically, without
an additional GPU run for documentation.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 "$PPL825_PYTHON" - "$PPL9_RUN" <<'PY'
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path.cwd() / 'scripts'))
import torch
from mamba2_recall import runtime, resurface_data as data
from evaluate_resurface_more import pin_replay_backend, check_replay_backend
from state_ppl_codec_v6 import StatePPLQuant
import run_state_ppl_v9 as v9

run = Path(sys.argv[1])
selected_path = run / 'screen/selected_calibration.pt'
full_path = run / 'full/full_comparison.json'
audit = json.loads((run / 'full_audit.json').read_text())
full = json.loads(full_path.read_text())
assert audit['complete'] and audit['passed'] and not audit['cuda_initialized']
assert audit['stage'] == 'full' and full['complete']
assert audit['source_sha256'] == data.sha_file('scripts/audit_state_ppl_v9.py')
assert audit['full']['comparison_sha256'] == data.sha_file(full_path)
assert audit['full']['target_pass'] is True and full['target_pass'] is True
assert all(full['target_checks'].values())
assert audit['selected_calibration']['sha256'] == data.sha_file(selected_path)
assert full['selected_calibration_sha256'] == data.sha_file(selected_path)

args = SimpleNamespace(
    candidates=Path('artifacts/state_first_v5_candidates/candidates.pt'),
    v5_selected_calibration=Path('artifacts/state_first_v5_screen/selected_calibration.pt'),
    v6_selected_calibration=Path('artifacts/state_ppl_v6_screen/selected_calibration.pt'),
    prose_tokens=Path('/home/horde/mamb2_8B_Recall/training_data/prose/training_tokens.pt'),
    codec_checks=Path('reports/state_ppl_v6_codec_checks.json'),
    layer_candidates=Path('artifacts/state_ppl_v8_calibration/candidates.pt'),
    v8_selected_calibration=Path('artifacts/state_ppl_v8_screen/selected_calibration.pt'),
    v8_screen_audit=Path('reports/state_ppl_v8_screen_audit.json'),
    v8_full_dir=Path('artifacts/state_ppl_v8_full'),
    v8_full_audit=Path('reports/state_ppl_v8_full_audit.json'),
    group_candidates=run / 'calibration/candidates.pt',
)
candidates, binding, train_windows, _ = v9.load_inputs(args)
selected, _, _ = v9.load_selection(selected_path, candidates, binding, train_windows)

torch.set_num_threads(8)
torch.manual_seed(20260929)
torch.cuda.manual_seed_all(20260929)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
backend = pin_replay_backend()
source = Path('/home/horde/Mamba2-8B-E8W5/models/source')
tokenizer = runtime.SentencePieceTokenizer(source)
model = runtime.load_source_model(source, dtype=torch.float16)
assert v9.v6.no_adapter_hooks(model)

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

Continue tokens for the same request using `reset=False` inside the active
controller context; use `reset=True` for each new request. The selected payload
is a coordinate-table artifact with provenance, not a model-weight checkpoint.
The demonstrated interface is batch one and does not support native
`InferenceParams` or variable-length batching.
