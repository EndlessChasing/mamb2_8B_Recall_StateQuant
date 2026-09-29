---
license: gpl-3.0
library_name: pytorch
pipeline_tag: text-generation
language:
  - en
base_model: nvidia/mamba2-8b-3t-4k
base_model_relation: adapter
inference: false
tags:
  - mamba2
  - state-space-model
  - statequant
  - state-quantization
  - resurface
  - adapter
  - recall
  - custom-runtime
datasets:
  - Salesforce/wikitext
---

# Mamb2_8B_Recall_SQ3.25

**Packed 3.25-bit recurrent state plus a freshly trained Resurface adapter**
for the pure [`nvidia/mamba2-8b-3t-4k`](https://huggingface.co/nvidia/mamba2-8b-3t-4k)
language model. The verified result is **7.855570 WikiText-2 validation PPL**
and **271/384 (70.57%) normal multi-key recall**, using **27.1797 MiB** of
persistent state and table storage per sequence at batch one.

**SQ3.25 describes recurrent state. The 8B model weights remain FP16.** This
repository distributes the adapter, calibration table, custom runtime and
verification evidence. Download the original NVIDIA base separately. It is
not a standalone 8B checkpoint, a 3.25-bit weight model, or a Transformers/PEFT
`from_pretrained` package.

This is the SQ3.25 member of the Mamb2_8B_Recall family. Its adapter was trained
fresh on the fixed quantized-state base; it is distinct from the original
[Mamb2_8B_Recall adapter](https://huggingface.co/EndlessChasing/Mamb2_8B_Recall).

## Verified quality

| Configuration | WikiText-2 validation PPL | Normal MK | N=16 | N=64 |
| --- | ---: | ---: | ---: | ---: |
| Original S16, archived reference | 7.334322057221965 | 146/384 (38.02%) | — | — |
| Fixed Q3.25 state, no adapter | 8.186186562837207 | 32/384 (8.33%) | 30/192 | 2/192 |
| **Q3.25 + released Resurface** | **7.855569605864836** | **271/384 (70.57%)** | **181/192** | **90/192** |
| Q3.25 after adapter removal | 8.186186562837207 | 32/384 (8.33%) | 30/192 | 2/192 |

PPL covers **130 reset windows of up to 2,048 predicted tokens**, with
**264,764 targets** in the pinned WikiText-2 validation split. MK uses
**384 normal prompts and 384 paired target-removed diagnostic prompts**.
Both Q3.25 configurations match the removed target 0/384 times; this is not an
abstention score. The normal-MK denominator excludes those controls.

Against the fixed Q3.25 base, the adapter improves PPL **4.0387%** and normal
MK **62.2396 percentage points**: 242 paired gains and three regressions.
The paired bootstrap 95% interval for the MK change is
**[+57.2917, +67.1875] percentage points**. PPL remains **7.11% higher than
the archived S16 reference**, which was not rerun in this experiment.

All nine predefined quality and integrity gates passed, including the
PPL threshold, positive paired recall improvement, exact control replay and
unchanged persistent cache. Removing the adapter reproduced all 130 baseline
window NLLs and all 768 generated sequences exactly, including reset/cache
receipts. An independent CPU audit of the recorded full evidence also passed;
that audit does not rerun GPU logits. See [full results](docs/STATE_RESURFACE_V11_RESULTS.md),
[raw comparison](release/full_comparison.json) and
[full audit](release/full_audit.json).

## State format and memory

Each 128-coordinate recurrent-state row stores **32 INT8 coordinates,
32 INT4 coordinates and 64 zero-carry coordinates**, plus two FP16 scales.
That is 52 bytes per row, or **3.25 bits per coordinate including scales**.
The custom runtime uses the packed carry for every token, including prompt
processing. The selected coordinate table is fixed before adapter training.

| Persistent component, batch one | Bytes | Size |
| --- | ---: | ---: |
| SSM cache | 23,855,104 | 22.7500 MiB |
| Convolution cache | 4,587,520 | 4.3750 MiB |
| One shared static table | 57,344 | 0.0547 MiB |
| **Cache plus table** | **28,499,968** | **27.1797 MiB** |
| Separate FP16 adapter payload | 2,308,208 | 2.3082 MB |
| **Cache, table and adapter** | **30,808,176** | **29.3810 MiB** |
| Original FP16 model weights, separate | 16,473,999,360 | 16.4740 GB |

The cache/table figure is **76.64% smaller** than the archived S16 cache of
116.3750 MiB. Static table and adapter storage can be shared across requests;
each is counted once here. These are tensor payloads, excluding temporary
computation, allocator reserve and training teacher/optimizer state. They are
not total GPU memory requirements. The serialized adapter file is 2,375,743
bytes; its 1,154,104 FP16 parameters occupy 2,308,208 bytes when loaded.

## Download and verify

With the Hugging Face CLI installed:

```bash
hf download EndlessChasing/Mamb2_8B_Recall_SQ3.25 \
  --local-dir Mamb2_8B_Recall_SQ3.25
cd Mamb2_8B_Recall_SQ3.25
python3 scripts/release_state_resurface_v11.py --verify release
```

The standard-library verifier checks the exact adapter, table, raw evidence,
internal checksums and unchanged measured runtime files. Verification does
not require CUDA or download the base. The original 18-file release bundle
is preserved under `release/`; runtime code, scripts and notices are supplied
alongside it. This verification checks published evidence rather than running
a new quality evaluation.

Download the separate source checkpoint and tokenizer at the pinned revision:

```bash
hf download nvidia/mamba2-8b-3t-4k \
  release/mp_rank_00/model_optim_rng.pt \
  mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
  --revision b915550c63ba9359f88f44d1f6a600d85af27302 \
  --local-dir models/source
```

The loader verifies the checkpoint and tokenizer SHA-256 values, then casts
the official BF16 weights to FP16. Native NVIDIA Megatron BF16 equivalence is
not established by these measurements.

## Environment and inference

The measured environment was **Linux, Python 3.10.12, PyTorch 2.11.0+cu128,
Triton 3.6.0, Mamba-SSM 2.3.2.post1, NumPy 1.26.4, datasets 4.8.5 and
SentencePiece 0.2.1**, on an **RTX PRO 6000 Blackwell Server Edition**.
Install the matching CUDA/PyTorch native Mamba and causal-convolution
dependencies first, then install this package from the downloaded root:

```bash
python -m pip install --no-deps -e .
python scripts/infer_state_resurface_v11.py \
  --bundle release \
  --source-dir models/source \
  --prompt 'The capital of France is' \
  --max-new-tokens 32
```

The entrypoint checks recorded backend versions, external source hashes,
precision flags and the fixed 16-warp RMSNorm configuration. An incompatible
backend fails before model loading. `pyproject.toml` contains broad dependency
ranges, not an exact environment lock; other GPU architectures are not validated.
See the [release guide](docs/STATE_RESURFACE_V11_RELEASE.md).

This wrapper provides greedy batch-one generation with prompts up to 4,096
tokens. Its separate GPU smoke reproduced **one archived 263-token MK prompt
and all 12 continuation token IDs**, with the same cache bytes. That check
validates the wrapper on that case; it does not repeat or extend the full
PPL/MK evaluation. The example prompt above is a usage example, not a reported
quality measurement. Native `InferenceParams`, variable-length batching and
a serving integration are not provided.

To check the package through the inference entrypoint without torch or CUDA:

```bash
python3 scripts/infer_state_resurface_v11.py --bundle release --verify-only
```

## Training and limitations

The original 8,236,999,680 base parameters remain frozen. After fixing the
selected state table, a fresh post-D Resurface-style adapter completed
**1,536 successful TRAIN updates**, with five numerical overflow retries.
Only the final TRAIN-derived FP16 export was evaluated; validation/CONFIRM
did not select a checkpoint. Training/export forward parity was checked at
128 and 512 tokens, with separately audited gradient fixtures.

The validation corpus and CONFIRM prompt families have historical exposure.
The results establish performance on this specified protocol. They do not
establish unseen-template recall, full-4K recall quality, other-language
performance or broad language-model generalization. The wrapper's 4K prompt
limit is an interface constraint, not a demonstrated 4K recall result.
Source identity/version/gradient guards do not constitute a complete post-run
bytewise audit of the base model. Adapter/table transfer to other checkpoints
requires separate validation.

The tagged repository contains the
[frozen protocol](https://github.com/EndlessChasing/mamb2_8B_Recall_StateQuant/blob/be037e1a2635ad7b540e0a1d2e551febf647c346/docs/STATE_RESURFACE_V11_PROTOCOL.md),
[full training/evaluation reproduction guide](https://github.com/EndlessChasing/mamb2_8B_Recall_StateQuant/blob/be037e1a2635ad7b540e0a1d2e551febf647c346/docs/STATE_RESURFACE_V11_REPRODUCTION.md)
and retained evidence. Base weights and training corpora are downloaded
separately.

## Provenance and licenses

This release corresponds to GitHub
[`v0.1.0-q325-resurface`](https://github.com/EndlessChasing/mamb2_8B_Recall_StateQuant/releases/tag/v0.1.0-q325-resurface),
commit **`be037e1a2635ad7b540e0a1d2e551febf647c346`**.

- Adapter SHA-256: `339334b3431027504fb8da4ce3167d63c8c2f309920cfb0076bde1f72a115cb7`.
- Selected calibration SHA-256: `a77bbd86ae609d8ef7cf90f10d0904cf95b4df8e059c9ea6cb2fd12886436fb8`.
- Full independent audit SHA-256: `ad2b3f14a075b0d60993f3579e99fc98bfbac8aa527f6a0c8d684cc9493e7d72`.

New project code and the fresh adapter are distributed under **GPL-3.0**.
The original NVIDIA model, Mamba implementation and StateQuant reference have
their separate **Apache-2.0** licenses. The root GPL license does not relicense
those upstream assets or WikiText text. The base checkpoint is not included.
See [LICENSE](LICENSE) and [third-party notices](docs/THIRD_PARTY_NOTICES.md)
for source attribution, StateQuant provenance and WikiText licensing details.

Method references: [StateQuant](https://github.com/Oso1106/StateQuant) and
[Resurface](https://github.com/Oso1106/Resurface-Multi-Binding-Recall-Is-Latent-in-Mamba-s-State).
The preserved StateQuant reference comes from the user-supplied archive
identified in the notices; current availability of that external repository
is not required to use this packaged runtime.
