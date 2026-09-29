# Mamb2_8B_Recall + StateQuant

## In progress: optimize unadapted SQ3.25, then fresh Resurface v5

The user clarified the order: **original source with no adapter → optimize
SQ3.25 → freeze the table → train fresh Resurface for recall**. Four fixed
same-budget tier tables will be compared on an expanded TRAIN screen without
loading any adapter. Only its selected table can advance to fresh training;
no v2/v3 adapter is reused. See the [v5 protocol](docs/STATE_FIRST_V5_PROTOCOL.md)
and [checklist](PLAN.md). No v5 quality result is available yet.

## Completed v4: changing the codec with frozen Resurface trades recall for PPL

This distinct experiment kept the v2 adapter fixed while testing readout-aware
tiers, dense3-bit carry and equalized dense3. Each retained **3.25 bits per
state element including scales** and **27.1797 MiB** total batch-one cache.
Only readout-aware tiers advanced to full diagnostic confirmation; both dense
variants substantially worsened TRAIN quality.

| State table with the same frozen v2 adapter | Full PPL | Normal MK |
| --- | ---: | ---: |
| Original magnitude tiers | 8.388906 | 239/384 (62.24%) |
| Readout-aware tiers | 8.258201 | 204/384 (53.125%) |

PPL improves **1.56%**, but MK falls **9.11 percentage points** (35 gains,
70 regressions; paired 95% interval **−14.32 to −4.17 pp**). The combined repair
gate **fails**. Candidate PPL remains **12.60% above original S16**. Keep the
v2 endpoint as the recall reference. Both exact full parent replays and the
independent CPU audit passed. This did not retrain Resurface for the new table.
See [all results and limits](docs/STATE_REPAIR_RESULTS.md),
[raw comparison](reports/state_repair_full/full_comparison.json),
[independent audit](reports/state_repair_full_audit.json), and
[reproduction commands](docs/STATE_REPAIR_REPRODUCTION.md).

## v3: more training gives lower PPL without a recall improvement

Exact continuation of the v2 FP32 masters, AdamW and GradScaler added
**3072 successful updates (4608 total)**. Source weights, SQ3.25 calibration,
adapter capacity and loss stayed fixed. All three full evaluation arms,
both exact replay checks and the independent full evidence audit are complete.

| SQ3.25 configuration | Full PPL | Normal MK | Cache/sequence |
| --- | ---: | ---: | ---: |
| v2 parent, 1536 updates | 8.388906 | 239/384 (62.24%) | 27.1797 MiB |
| v3 continuation, 4608 total updates | 8.351329 | 237/384 (61.72%) | 27.1797 MiB |

PPL improves **0.45%**, but recall has **34 gains and 36 regressions**.
MK changes by **−0.52 percentage points** (paired 95% interval: −4.69 to +3.91).
**The continuation improvement gate fails.** This does not establish a real
recall decline; it does not demonstrate a positive gain. Keep v2 as the recall
reference. v3 PPL remains **13.87% above original S16**.

RMSNorm autotuning caused an initial baseline replay discrepancy. Pinning the
16-warp configuration that reproduces archived results restored all 130 window
NLLs and 768 generated sequences exactly; reinstalling the parent repeated them
exactly again. The candidate was unchanged. See
[v3 results and limitations](docs/RESURFACE_MORE_RESULTS.md),
[backend diagnosis](docs/RESURFACE_MORE_BACKEND_REPLAY.md),
[frozen continuation protocol](docs/RESURFACE_MORE_PROTOCOL.md), and
[reproduction commands](docs/RESURFACE_MORE_REPRODUCTION.md).
The [full comparison](reports/resurface_more_v3/evaluation/full_comparison.json)
and [audit receipt](reports/resurface_more_v3_full_audit.json) preserve the
negative improvement-gate outcome.

## Completed v2: quantize first, then train a new adapter

The new requested order is **original Mamba2-8B -> SQ3.25 state -> fresh
Resurface training**. Calibration is repeated on the unadapted source model;
the old adapter is not used for initialization. A differentiable training path
uses the exact packed inference forward and a declared masked STE backward.
The fixed 1536-update recipe and three configurations, plus a complete SQ
baseline replay, are specified in [the protocol](docs/QUANT_FIRST_PROTOCOL.md).
Training completed with 1536 successful updates in 1542 attempts; the
2,374,591-byte FP16 adapter passed the 128-token packed training/inference
equality check. Full PPL/MK evaluation and exact SQ baseline restoration are
complete. The new artifact is
[`reports/quant_first_v2/training/adapter_fp16.pt`](reports/quant_first_v2/training/adapter_fp16.pt);
see [current results and evidence](docs/QUANT_FIRST_RESULTS.md) and
[exact reproduction commands](docs/QUANT_FIRST_REPRODUCTION.md).

| v2 configuration | Full PPL | Normal MK | Cache/sequence |
| --- | ---: | ---: | ---: |
| Original S16, no adapter | 7.33432 | 146/384 (38.02%) | 116.3750 MiB |
| SQ3.25, no adapter | 8.69155 | 50/384 (13.02%) | 27.1797 MiB |
| **SQ3.25 → new Resurface** | **8.38891** | **239/384 (62.24%)** | **27.1797 MiB** |

**Repair versus the quantized baseline passes:** PPL improves **3.48%** and
MK improves **49.22 percentage points** (paired 95% interval: +43.75 to +54.69).
PPL remains **14.38% above original S16**; original perplexity is not restored.
This is not a comparison against a separately trained S16 + Resurface model.

Full validation covers 264,764 WikiText-2 targets, 384 normal CONFIRM MK prompts
and 384 target-removed controls (all modes score 0/384 on the latter).
Every window NLL and all 768 generated token sequences reproduce exactly after
removing the adapter. Q8 and weight quantization are excluded.
The independent CPU evidence audit passed; see the
[full comparison](reports/quant_first_v2/evaluation/full_comparison.json) and
[audit receipt](reports/quant_first_v2_full_audit.json).

## Historical experiment: old Recall adapter, then quantize state

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

Use [the v2 reproduction guide](docs/QUANT_FIRST_REPRODUCTION.md) and
[checklist](PLAN.md). It covers original-source loading, no-adapter calibration,
discarded smoke training, fresh formal training, all four evaluation runs and
offline evidence audits. The tested CUDA environment used PyTorch 2.11.0+cu128,
Triton 3.6.0 and mamba-ssm 2.3.2.post1 on an RTX PRO 6000 Blackwell.

The v2 entry points are `prepare_quant_first.py`, `train_quant_first.py`,
`evaluate_quant_first.py` and `audit_quant_first.py` in `scripts/`.
Full v2 validation runs regardless of pilot quality. It uses only the final
1536-update export, with no DEV-based checkpoint selection.

The historical `scripts/run_statequant.py` entry point and
[v1 protocol](docs/PROTOCOL.md) instead load the old Recall adapter before
calibration and use the earlier pilot stop rule. The preserved
`pretrained/adapter_fp16.pt` belongs to that historical experiment.

## Provenance and licenses

- Original base: NVIDIA Mamba2-8B, Apache-2.0; downloaded separately.
- Recall code and unchanged original Recall adapter: inherited GPL-3.0, see
  [LICENSE](LICENSE) and [original project](https://github.com/EndlessChasing/mamb2_8B_Recall).
- New training/runtime code and the new Resurface adapter are provided under
  this repository's GPL-3.0 license.
- Archived StateQuant reference: Apache-2.0, copyright 2026 Kun Yue, see
  [reference license](reference/statequant/LICENSE). It is a user-supplied
  archive; current online availability is not assumed.

This repository does not redistribute the large base checkpoint.
