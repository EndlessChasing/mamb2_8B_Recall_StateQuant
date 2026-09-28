# Mamb2_8B_Recall + StateQuant

Experimental **pure Mamba2-8B + frozen Resurface + packed recurrent state**.
The base is the original FP16 Recall model, not the W4 weight release.

The only comparison is FP16 carried state versus original-coordinate
StateQuant **3.25-bit** (48 zero + 64 INT4 + 16 INT8 per 128 coordinates,
including two FP16 scales). Q8 is skipped at the user's request.

Implementation and quality validation are in progress. No quality-preservation
or deployment claim is made before the paired PPL/MK gates pass.

## Storage scope

At batch 1, the format budget for SSM + convolution cache + permutation tables
is 116.375 MiB (FP16) versus 27.1797 MiB (3.25-bit). Actual allocations will be
measured. Base weights remain about 16.47 GB; state-cache savings do not imply
the same reduction in total model memory. Kernel registers and workspace are
temporary computation, not persistent compressed cache.

## Reproduction

See [frozen experiment protocol](docs/PROTOCOL.md) and [checklist](PLAN.md).
Reuse the official `nvidia/mamba2-8b-3t-4k` checkpoint, tokenizer and the included
unchanged Recall adapter. A CUDA GPU and the pinned native Mamba environment
are required. Commands and measured results will be recorded with receipts.

## Provenance and licenses

- Original base: NVIDIA Mamba2-8B, Apache-2.0; downloaded separately.
- Recall code and unchanged original Recall adapter: inherited GPL-3.0, see
  [LICENSE](LICENSE) and [original project](https://github.com/EndlessChasing/mamb2_8B_Recall).
- Archived StateQuant reference: Apache-2.0, copyright 2026 Kun Yue, see
  [reference license](reference/statequant/LICENSE). It is a user-supplied
  archive; current online availability is not assumed.

This repository does not redistribute the large base checkpoint.
