# Reproduce quantization first, then fresh Resurface training

This guide implements [the frozen v2 protocol](QUANT_FIRST_PROTOCOL.md):

1. Load the original pure Mamba2-8B weights.
2. Calibrate SQ3.25 on that model with no adapter.
3. Train a newly initialized Resurface adapter under quantized recurrence.
4. Evaluate the final serialized adapter, both baselines, and baseline restoration.
5. Independently audit the recorded results on CPU.

The historical `scripts/run_statequant.py` entry point loads the old Recall
adapter before calibration. Use the commands below for the new experiment.
Q8 is excluded. The new guide does not report or assume a PPL/MK outcome.

## 1. Source and environment

Run commands from the repository root on the CUDA host. The recorded environment
was Linux, Python 3.10.12, PyTorch 2.11.0+cu128, Triton 3.6.0,
`mamba-ssm` 2.3.2.post1, NumPy 1.26.4, datasets 4.8.5, and sentencepiece 0.2.1,
on an NVIDIA RTX PRO 6000 Blackwell Server Edition. Install the native Mamba
and causal-convolution dependencies for the CUDA/PyTorch environment separately.
The package's broad dependency ranges do not constitute an exact environment lock.

The base is `nvidia/mamba2-8b-3t-4k`, revision
`b915550c63ba9359f88f44d1f6a600d85af27302`. The source directory must contain:

- `model_optim_rng.pt`, either directly or at `release/mp_rank_00/model_optim_rng.pt`.
- `mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model`.

Pinned SHA256 identities:

| Input | SHA256 |
| --- | --- |
| Original source checkpoint | `47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb` |
| Tokenizer | `5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09` |
| Prose TRAIN tensor file | `e54b02e5162e042a9cdd504f4eb1b1652724fb240bbc2c97608967aa26297233` |
| `docs/prose_train_manifest.json` | `facb2ca461615a4199781bd21784d642d6674f5b862641b3b9edac3fb499b89d` |
| `docs/QUANT_FIRST_PROTOCOL.md` | `24466642ce87c75fc2136a42ed14e69733c462b507a0be82c062b4da7f836bcb` |

The loader validates the source checkpoint and strictly maps all 507 original
tensors. It casts the original BF16 weights to FP16 and freezes all
8,236,999,680 parameters. This is the project's native FP16 reference; it does
not establish numerical equivalence to NVIDIA's original Megatron BF16 runtime.

Set these paths for the existing CUDA host, or replace them for another host:

```bash
SQ_PYTHON=/home/horde/.venvs/lodram/bin/python
SQ_SOURCE_DIR=/home/horde/Mamba2-8B-E8W5/models/source
SQ_RUN_DIR=artifacts/repro_quant_first_v2
SQ_NUMERIC_DIR="$SQ_RUN_DIR/numeric_v2"
SQ_TRAIN_TOKENS="$SQ_RUN_DIR/training_tokens.pt"

"$SQ_PYTHON" -m pip install --no-deps -e .
mkdir -p "$SQ_RUN_DIR"
```

The `models/source` directory above contains the original checkpoint, despite
the parent project's name. These commands do not load E8/W5 or W4 weights.
Choose a new run directory for each independent reproduction. Scripts refuse
to overwrite calibration, training, or completed evaluation outputs.

Training loads a separate original FP16 teacher as well as the student, so
their weights alone occupy approximately 32.95 GB. Optimizer, activation and
scan workspace require additional GPU memory. A minimum training VRAM budget
for other hardware has not been established by this guide. Avoid copying the
large source checkpoint when an existing verified copy is available.

## 2. Prepare fixed TRAIN inputs and run kernel checks

Regenerate the exact 448 TRAIN windows using the pinned text, tokenizer and
window manifest. This step is CPU-only:

```bash
CUDA_VISIBLE_DEVICES= "$SQ_PYTHON" scripts/prepare_prose.py \
  --source-dir "$SQ_SOURCE_DIR" \
  --out "$SQ_TRAIN_TOKENS"
```

Alternatively, set `SQ_TRAIN_TOKENS` to the existing verified file:
`/home/horde/mamb2_8B_Recall/training_data/prose/training_tokens.pt`.
When reusing that file, skip its regeneration command. `prepare_prose.py`
prints both content and serialized-file hashes; the v2 preparation and trainer
require the exact pinned file SHA above. A serialization or dataset-identity
mismatch is an input mismatch, not permission to change the frozen protocol.

Run the inference codec, convolution, and training scan checks:

```bash
"$SQ_PYTHON" scripts/check_state_codec.py \
  --output "$SQ_RUN_DIR/codec_checks.json"

CUDA_VISIBLE_DEVICES= "$SQ_PYTHON" scripts/check_state_runtime_conv.py \
  --output "$SQ_RUN_DIR/conv_checks.json"

"$SQ_PYTHON" scripts/check_state_training_scan.py \
  --output "$SQ_RUN_DIR/training_scan_checks.json" \
  --benchmark-length 256
```

Require the checks to pass before fitting. The training scan checks exact
deployment-forward equality, recomputed carries, independent surrogate-gradient
algebra, dead-coordinate behavior and finite production-geometry gradients.
These checks do not measure language-model quality.

## 3. Calibrate the unadapted original model

```bash
"$SQ_PYTHON" scripts/prepare_quant_first.py \
  --source-dir "$SQ_SOURCE_DIR" \
  --train-tokens "$SQ_TRAIN_TOKENS" \
  --data-root "$SQ_NUMERIC_DIR" \
  --out "$SQ_RUN_DIR/calibration"
```

This first prepares and verifies 1,536 numeric TRAIN examples against the new
protocol. It then freshly loads the original source without any Resurface
adapter and calibrates on the first eight TRAIN rows, 512 tokens per row.
State resets between rows. Statistics rank mean absolute rounded FP16 carried
state separately for each layer and B/C group; ties use coordinate order.

Each group assigns 16 state coordinates to INT8, 64 to INT4, and 48 to zero
carry. The resulting tables remain fixed throughout training and evaluation.
Outputs are:

- `calibration/calibration.pt`: permutations, statistics and complete binding.
- `calibration/calibration.json`: hashes, magnitude accounting and runtime receipt.
- `calibration/numeric_train_receipt.json`: numeric TRAIN manifest identity.
- `numeric_v2/train/manifest.json`, `raw.jsonl`, `tokens.jsonl`: isolated TRAIN data.

An optional CPU-only preparation can be performed first by adding
`--prepare-only` to the same command and prefixing it with
`CUDA_VISIBLE_DEVICES=`. The later normal command re-verifies those numeric
inputs and creates the calibration; CPU preparation itself does not calibrate.

Obtain this run's TRAIN manifest hash rather than borrowing one from another run:

```bash
SQ_TRAIN_MANIFEST_SHA=$("$SQ_PYTHON" - "$SQ_NUMERIC_DIR/train/manifest.json" <<'PY'
import hashlib
from pathlib import Path
import sys
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)
```

## 4. Train a new adapter

First perform one discarded smoke update. This also verifies that the fresh
identity adapter and training forward match the actual packed inference path.

```bash
"$SQ_PYTHON" scripts/train_quant_first.py \
  --source-dir "$SQ_SOURCE_DIR" \
  --calibration "$SQ_RUN_DIR/calibration/calibration.pt" \
  --data-root "$SQ_NUMERIC_DIR" \
  --train-manifest-sha256 "$SQ_TRAIN_MANIFEST_SHA" \
  --prose-manifest docs/prose_train_manifest.json \
  --prose-tokens "$SQ_TRAIN_TOKENS" \
  --out-dir "$SQ_RUN_DIR/smoke" \
  --smoke
```

Then start the formal run from another fresh initialization:

```bash
"$SQ_PYTHON" scripts/train_quant_first.py \
  --source-dir "$SQ_SOURCE_DIR" \
  --calibration "$SQ_RUN_DIR/calibration/calibration.pt" \
  --data-root "$SQ_NUMERIC_DIR" \
  --train-manifest-sha256 "$SQ_TRAIN_MANIFEST_SHA" \
  --prose-manifest docs/prose_train_manifest.json \
  --prose-tokens "$SQ_TRAIN_TOKENS" \
  --out-dir "$SQ_RUN_DIR/training"
```

Only the 1,154,104 external Resurface master parameters train. Initialization is
`V=0, g=1, router_w=0, router_b=-4`; the new run never reads the old adapter.
The fixed budget is 1,536 successful updates, with at most eight overflow
retries. Each update pairs answer-suffix MK CE with prose CE, KL from a frozen
S16 original-model teacher, and the declared prose router-closure loss.

The training forward uses the same SQ3.25 recurrence as packed inference after
every token, including prompt tokens. Backward uses a declared straight-through
estimator: identity through the 80 retained carry coordinates and zero through
the 48 pruned carry coordinates. All 128 coordinates still receive gradients
from their current-token readout. Dynamic scale selection, rounding and
clipping are detached. This is a surrogate gradient for discrete quantization.

The scan recomputes bounded chunks for backward and uses ordinary training
scratch; this scratch is not compressed deployed state. FP32 atomic gradient
reductions do not promise bitwise deterministic retraining. Per-layer replay
starts from zero state, preserving block-checkpoint correctness.

Keep all files in `training/`: its `report.json`, actual `adapter_fp16.pt`, and
four recovery checkpoints at 384/768/1152/1536 successful updates. The final
adapter is the only evaluation candidate. Intermediate checkpoints are not
selected using held-out quality. The current CLI writes recovery checkpoints
but does not expose a resume flag.

The trainer verifies frozen source/table identities and gradients, export
round-trip hashes, and a 128-token equality check between the final training
forward and reloaded FP16 adapter under packed inference.

## 5. Evaluate four runs with three distinct configurations

```bash
"$SQ_PYTHON" scripts/evaluate_quant_first.py \
  --source-dir "$SQ_SOURCE_DIR" \
  --calibration "$SQ_RUN_DIR/calibration/calibration.pt" \
  --training-report "$SQ_RUN_DIR/training/report.json" \
  --out "$SQ_RUN_DIR/evaluation" \
  --split pilot

"$SQ_PYTHON" scripts/evaluate_quant_first.py \
  --source-dir "$SQ_SOURCE_DIR" \
  --calibration "$SQ_RUN_DIR/calibration/calibration.pt" \
  --training-report "$SQ_RUN_DIR/training/report.json" \
  --out "$SQ_RUN_DIR/evaluation" \
  --split full
```

Both commands execute these arms in order:

| Report suffix | Configuration |
| --- | --- |
| `source_s16` | Original model, per-token FP16 carried state, no adapter |
| `source_sq3p25` | Original model, new SQ3.25 tables, no adapter |
| `resurface_sq3p25` | Same SQ3.25 tables plus the final new FP16 adapter |
| `restored_source_sq3p25` | Remove the adapter and repeat the entire SQ baseline |

Pilot scores four windows at indices `[0,32,64,96]` (8,192 targets) and
96 DEV prompts (48 normal, 48 target-removed). Full scores all 130 WikiText-2
validation windows (264,764 targets) and 768 CONFIRM prompts (384 normal,
384 target-removed). The pilot is descriptive; run full regardless of pilot
quality. Full does not load or require a passing pilot report.

Every window/prompt resets state. Both controls and the adapter use per-token
recurrence; historical parallel SSD PPL is not substituted for the new S16
baseline. Greedy generation uses all 256K vocabulary entries, at most 12 tokens,
and EOS stopping. The first standalone six-digit prediction is scored.

The restoration arm repeats the selected full or pilot workload, requiring
every window NLL and generated token sequence to match the original SQ run
exactly. Reports include per-window NLL, generated token IDs, score inputs,
actual cache allocations, timing, parameter checks and hashes. Each split
produces four arm files plus `*_comparison.json` and `*_restoration.json`.

The fixed repair criteria and separate gap to original S16 are in the protocol.
The comparison distinguishes observed MK improvement/PPL preservation from a
positive paired 95% MK bootstrap interval. It does not establish generalization
to unseen corpora, templates or longer distances.

## 6. Audit the evidence on CPU

The auditor operates offline and therefore needs both pinned TRAIN and
validation datasets in the local Hugging Face cache. Preparing prose and
evaluating normally populate them. If the TRAIN tensor was reused from another
host, populate and verify the dataset cache before the offline audit:

```bash
CUDA_VISIBLE_DEVICES= "$SQ_PYTHON" - "$SQ_SOURCE_DIR" <<'PY'
import sys
from mamba2_recall.runtime import SentencePieceTokenizer
from mamba2_recall.calibration import load_wikitext_tokens
tokenizer = SentencePieceTokenizer(sys.argv[1])
for split in ('train', 'validation'):
    _, metadata = load_wikitext_tokens(tokenizer, split)
    print(split, metadata['token_stream_sha256_int64le'])
PY
```

Audit each completed split:

```bash
CUDA_VISIBLE_DEVICES= "$SQ_PYTHON" scripts/audit_quant_first.py \
  --source-dir "$SQ_SOURCE_DIR" \
  --calibration "$SQ_RUN_DIR/calibration/calibration.pt" \
  --training-report "$SQ_RUN_DIR/training/report.json" \
  --eval-dir "$SQ_RUN_DIR/evaluation" \
  --split pilot \
  --output "$SQ_RUN_DIR/pilot_independent_audit.json"

CUDA_VISIBLE_DEVICES= "$SQ_PYTHON" scripts/audit_quant_first.py \
  --source-dir "$SQ_SOURCE_DIR" \
  --calibration "$SQ_RUN_DIR/calibration/calibration.pt" \
  --training-report "$SQ_RUN_DIR/training/report.json" \
  --eval-dir "$SQ_RUN_DIR/evaluation" \
  --split full \
  --output "$SQ_RUN_DIR/full_independent_audit.json"
```

Use the original formal training directory containing all four recovery
checkpoints; a copied `report.json` and adapter alone are insufficient for this
audit. Keep the exact training/evaluation source files unchanged because their
hashes are verified. Audit output files must be new.

This independently reconstructs data selection, calibration ordering, training
schedule, export hashes, token decoding, metric arithmetic, bootstrap flags and
full restoration from recorded evidence. It does not rerun logits or establish
that a recorded generation was the true greedy argmax. Frozen-parameter checks
cover reported identities, versions and gradients, not a post-training hash of
every original weight byte.

## Memory and artifact scope

SQ3.25 encodes each 128-value state row in 52 bytes: 16 INT8 entries, 64 packed
INT4 entries and two FP16 scales; 48 entries have zero carry. The 3.25-bit figure
includes these scales. Current-token readout uses the FP32 update before carry
compression, including the current contribution of the zero-carry coordinates.

At batch 1 the format's persistent budget is 28,499,968 bytes, including FP16
convolution history and the compact layer/group permutation tables, versus
122,028,032 bytes for S16. The evaluator records actual allocations. The
memoryless adapter adds no recurrent state. Original weights remain FP16;
teacher, optimizer, activations, scan scratch and allocator reserves are separate.

The archived v2 evidence uses `reports/quant_first_v2/calibration/` and
`reports/quant_first_v2/training/`. Preserve those artifacts when making a new
run. The older `pretrained/adapter_fp16.pt`, `reports/pilot_v1/` and
`docs/PROTOCOL.md` belong to the historical adapter-first experiment.
