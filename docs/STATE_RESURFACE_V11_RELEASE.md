# Q3.25 state + Resurface: release usage

Release tag: `v0.1.0-q325-resurface`.

This release contains a **3.25-bit recurrent-state quantizer** and a trained
Resurface adapter for the original pure NVIDIA Mamba2-8B. Model weights are
loaded as FP16. The download is an adapter/table bundle, not a standalone 8B
checkpoint. Obtain the original source separately.

## Validated result

| Configuration | WikiText-2 validation PPL | Normal multi-key recall |
| --- | ---: | ---: |
| Original S16, archived reference | 7.334322 | 146/384 |
| Fixed Q3.25 state, no adapter | 8.186187 | 32/384 |
| **Fixed Q3.25 + released Resurface** | **7.855570** | **271/384** |

PPL covers 130 windows of length 2048 and 264,764 predicted tokens. MK uses
384 normal and 384 target-removed prompts; target-removed matches are 0/384
for both Q3.25 arms. Adapter removal exactly reproduces all baseline window
NLLs and all 768 generated sequences. All nine predefined gates and the
independent CPU evidence audit pass. The final 1536-update TRAIN export was
fixed before this evaluation. These benchmark families have historical
exposure; the result does not establish unseen-task or general recall quality.
PPL remains 7.11% higher than the archived S16 reference.

The layout stores 32 INT8, 32 INT4 and 64 zero-carry coordinates per row of
128, plus two FP16 scales: 52 bytes/row, or 3.25 bits/coordinate.

| Persistent memory component | Bytes |
| --- | ---: |
| SSM + convolution cache + one shared table, batch one | 28,499,968 (27.1797 MiB) |
| Separate FP16 adapter payload | 2,308,208 (2.308 MB) |
| Cache plus adapter | 30,808,176 (29.3810 MiB) |
| Original FP16 model weights | 16,473,999,360 (16.474 GB) |

The state/table and adapter figures exclude source weights, temporary
computation and allocator reserve. They are not total GPU memory figures.
The adapter file is 2,375,743 bytes including serialization metadata.

## Download code and release assets

```bash
git clone --branch v0.1.0-q325-resurface --depth 1 \
  https://github.com/EndlessChasing/mamb2_8B_Recall_StateQuant.git
cd mamb2_8B_Recall_StateQuant
mkdir -p downloads
gh release download v0.1.0-q325-resurface \
  --repo EndlessChasing/mamb2_8B_Recall_StateQuant \
  --pattern 'mamba2-8b-q325-resurface-v11.tar.gz' \
  --pattern SHA256SUMS --dir downloads
(cd downloads && shasum -a 256 -c SHA256SUMS)
tar -xzf downloads/mamba2-8b-q325-resurface-v11.tar.gz -C downloads
python3 scripts/release_state_resurface_v11.py \
  --verify downloads/mamba2-8b-q325-resurface-v11
```

The last check uses only Python's standard library. It validates the exact
adapter, calibration, evidence, internal checksums and unchanged measured
runtime files. It does not run a new GPU evaluation. No training-data paths
are required to use this bundle. Full experiment evidence and training
reproduction remain available in the tagged repository.

## Source checkpoint and environment

Obtain the files from
[`nvidia/mamba2-8b-3t-4k` at the fixed revision](https://huggingface.co/nvidia/mamba2-8b-3t-4k/tree/b915550c63ba9359f88f44d1f6a600d85af27302).
Place these files in your chosen `models/source` directory:

- `model_optim_rng.pt`, either directly or under `release/mp_rank_00/`.
- `mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model`.

With the Hugging Face CLI installed:

```bash
hf download nvidia/mamba2-8b-3t-4k \
  release/mp_rank_00/model_optim_rng.pt \
  mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
  --revision b915550c63ba9359f88f44d1f6a600d85af27302 \
  --local-dir models/source
```

The loader enforces the checkpoint SHA256
`47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb`
and tokenizer SHA256
`5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09`.
It casts the original BF16 checkpoint to FP16. Numerical equivalence to the
original NVIDIA Megatron BF16 runtime is not claimed.

The measured environment was Linux, Python 3.10.12, PyTorch 2.11.0+cu128,
Triton 3.6.0, `mamba-ssm` 2.3.2.post1, NumPy 1.26.4, datasets 4.8.5 and
sentencepiece 0.2.1, on an RTX PRO 6000 Blackwell Server Edition. Install the
matching CUDA/PyTorch native Mamba and causal-convolution dependencies first,
then `python -m pip install --no-deps -e .` from the repository root.
The broad ranges in `pyproject.toml` are not an exact environment lock.
The entrypoint verifies recorded backend versions, source hashes, precision
flags and the fixed 16-warp RMSNorm configuration; incompatible environments
fail before loading the model. Other GPU architectures are not validated.

## Generate with the released adapter

```bash
python scripts/infer_state_resurface_v11.py \
  --bundle downloads/mamba2-8b-q325-resurface-v11 \
  --source-dir models/source \
  --prompt 'The capital of France is' \
  --max-new-tokens 32
```

This is greedy batch-one generation using the actual packed carry for every
token, including prompt processing. The original audited codec and adapter
implementation are unchanged. This convenience wrapper is separate from the
full measured evaluation script. The prompt limit is 4096 tokens.
`--verify-only` checks the bundle without importing torch or requiring CUDA.

For integration, keep `native.install_fp16(...)` and `StatePPLQuantV10(...)`
contexts active together. Use `execution.backbone(ids, reset=True)` for a new
request and `reset=False` for continuation. Native `InferenceParams`,
variable-length batching and a serving integration are not provided.
See the repository's
[results](https://github.com/EndlessChasing/mamb2_8B_Recall_StateQuant/blob/v0.1.0-q325-resurface/docs/STATE_RESURFACE_V11_RESULTS.md),
[full reproduction guide](https://github.com/EndlessChasing/mamb2_8B_Recall_StateQuant/blob/v0.1.0-q325-resurface/docs/STATE_RESURFACE_V11_REPRODUCTION.md) and
[frozen protocol](https://github.com/EndlessChasing/mamb2_8B_Recall_StateQuant/blob/v0.1.0-q325-resurface/docs/STATE_RESURFACE_V11_PROTOCOL.md).

## Bundle provenance and licenses

The release includes the exact final adapter, selected table and receipt,
the final training report/audit, all three full raw evaluation arms, the full
comparison/audit, replay receipts, manifest and SHA256 checksums. Training
checkpoints and raw training probes stay in the tagged repository. Base
weights and prose training text are not included in the bundle.

- Final adapter SHA256:
  `339334b3431027504fb8da4ce3167d63c8c2f309920cfb0076bde1f72a115cb7`.
- Selected calibration SHA256:
  `a77bbd86ae609d8ef7cf90f10d0904cf95b4df8e059c9ea6cb2fd12886436fb8`.
- Full independent audit SHA256:
  `ad2b3f14a075b0d60993f3579e99fc98bfbac8aa527f6a0c8d684cc9493e7d72`.

New code and the new adapter are provided under this repository's GPL-3.0
license. The original NVIDIA base and StateQuant reference have their
separate Apache-2.0 licenses. See root `LICENSE` and the repository's
`docs/THIRD_PARTY_NOTICES.md`; copies accompany the bundle. The original
checkpoint is downloaded from NVIDIA separately.

To rebuild the deterministic release asset from retained repository files:

```bash
python3 scripts/release_state_resurface_v11.py --build artifacts/release_v11_new
```

The output directory must not already exist. The builder checks fixed
evidence and source hashes before producing the archive; it does not select
or train an adapter.
