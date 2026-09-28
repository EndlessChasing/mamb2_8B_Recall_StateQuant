# Mamb2_8B_Recall + StateQuant

Experimental **pure Mamba2-8B + frozen Resurface + packed recurrent state**.
The base is the original FP16 Recall model, not the W4 weight release.

The only comparison is FP16 carried state versus original-coordinate
StateQuant **3.25-bit** (48 zero + 64 INT4 + 16 INT8 per 128 coordinates,
including two FP16 scales). Q8 is skipped at the user's request.

**Pilot result: this fixed 3.25-bit candidate failed the quality gate.**
The code and packed cache work, but the original Recall adapter does not preserve
recall under this state compression. Full-corpus evaluation was not advanced.

| Mode | Actual cache / sequence | Pilot PPL | Normal MK |
| --- | ---: | ---: | ---: |
| FP16 state + Recall | 116.375 MiB | 6.0405 | 47/48 |
| SQ3.25 state + Recall | 27.1797 MiB | 6.8834 | 19/48 |

PPL increased **13.95%** and MK fell **58.33 percentage points**. Both modes
scored 0/48 on target-removed controls. The pilot contains 8192 PPL targets and
48 paired normal + 48 removed prompts per mode. These are screening results,
not full validation or a claim that all low-bit state methods must fail.
See [results and limitations](docs/RESULTS.md) and [raw comparison](reports/pilot_v1/pilot_comparison.json).

## Storage scope

At batch 1, measured SSM + convolution cache + permutation tables
are 116.375 MiB (FP16) versus 27.1797 MiB (3.25-bit), **76.64% less**.
Base weights remain about 16.47 GB; state-cache savings do not imply
the same reduction in total model memory. Kernel registers and workspace are
temporary computation, not persistent compressed cache.

## Reproduction

See [frozen experiment protocol](docs/PROTOCOL.md) and [checklist](PLAN.md).
Reuse the official `nvidia/mamba2-8b-3t-4k` checkpoint, tokenizer and the included
unchanged Recall adapter. A CUDA GPU and the pinned native Mamba environment
are required. The tested environment used PyTorch 2.11.0+cu128, Triton 3.6.0,
and mamba-ssm 2.3.2.post1 on an RTX PRO 6000 Blackwell.

```bash
pip install -e .
python scripts/check_state_codec.py --output reports/my_codec_checks.json
CUDA_VISIBLE_DEVICES= python scripts/check_state_runtime_conv.py \
  --output reports/my_conv_checks.json

# Set SOURCE_DIR to the official checkpoint/tokenizer directory.
CUDA_VISIBLE_DEVICES= python scripts/prepare_prose.py \
  --source-dir "$SOURCE_DIR" --out training_data/training_tokens.pt
python scripts/run_statequant.py --source-dir "$SOURCE_DIR" \
  --train-tokens training_data/training_tokens.pt --out artifacts/my_pilot
```

Native `mamba-ssm` installation is separate and must match the CUDA environment.
The runner refuses to overwrite prior pilot artifacts. Full evaluation requires
a passing pilot and the same calibration/core implementation; the included
candidate is stopped. The included adapter retains its original training binding.

## Provenance and licenses

- Original base: NVIDIA Mamba2-8B, Apache-2.0; downloaded separately.
- Recall code and unchanged original Recall adapter: inherited GPL-3.0, see
  [LICENSE](LICENSE) and [original project](https://github.com/EndlessChasing/mamb2_8B_Recall).
- Archived StateQuant reference: Apache-2.0, copyright 2026 Kun Yue, see
  [reference license](reference/statequant/LICENSE). It is a user-supplied
  archive; current online availability is not assumed.

This repository does not redistribute the large base checkpoint.
