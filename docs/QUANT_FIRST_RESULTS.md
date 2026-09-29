# SQ3.25 first, then a new Resurface adapter

Experiment date: 2026-09-28. See the pre-training
[frozen protocol](QUANT_FIRST_PROTOCOL.md).

## What was trained

The order is **original pure Mamba2-8B FP16 → SQ3.25 calibration → fresh
Resurface training with SQ3.25 active after every token**. The old Recall
adapter is not loaded. Only the new adapter is trained; base weights and
calibration tables stay fixed. Q8 and weight quantization are excluded.

The candidate completed **1,536 successful updates in 1,542 attempts** with six
overflow retries, taking 32.82 minutes on an RTX PRO 6000 Blackwell Server
Edition. It has 1,154,104 parameters in 224 FP16 tensors; the serialized adapter
is 2,374,591 bytes. Peak allocated training GPU memory was 34,820,548,608 bytes,
including the separate frozen teacher and training workspace.

Training uses the exact deployed packed recurrence in forward. Backward uses
the declared masked straight-through estimator: derivative one for the 80 live
carry coordinates and zero for the 48 discarded carry coordinates. Discarded
coordinates still contribute to the current-token readout. This is a surrogate
gradient; it ignores derivatives of scale selection, rounding and clipping.

The fresh identity adapter matched unadapted SQ3.25 bitwise on the 32-token
initialization probe. The final FP16 export matched training forward bitwise
under the packed inference runtime on a separate 128-token probe. These are
implementation checks, not quality measurements. FP32 atomic reductions in
backward mean another training run need not produce a bit-identical adapter.

## Full validation

The three primary arms and complete SQ baseline replay have finished all
130 WikiText-2 validation windows (264,764 predicted tokens), 384 normal CONFIRM
MK prompts and 384 target-removed controls per arm. The fixed repair gate versus
the quantized baseline passes; original-model PPL is not restored.

| Arm | PPL | Normal MK | Target removed | Cache/sequence |
| --- | ---: | ---: | ---: | ---: |
| Original FP16 weights, S16 state, no adapter | 7.334322 | 146/384 (38.02%) | 0/384 | 116.3750 MiB |
| Original FP16 weights, SQ3.25 state, no adapter | 8.691553 | 50/384 (13.02%) | 0/384 | 27.1797 MiB |
| SQ3.25 → newly trained Resurface | **8.388906** | **239/384 (62.24%)** | 0/384 | 27.1797 MiB |

Relative to unadapted SQ3.25, the new adapter lowers PPL by **3.4821%** and
improves MK by **49.2188 percentage points** (189 additional correct answers).
The paired cases contain 197 improvements and eight regressions. The paired
bootstrap 95% interval is **+43.75 to +54.69 percentage points** (10,000 draws,
seed 20260928). PPL improves on each of the 130 windows.

Relative to the original unadapted S16 model, MK improves by **24.2188 percentage
points** (93 additional correct answers; 135 improvements and 42 regressions;
paired 95% interval **+17.71 to +30.47 percentage points**).
PPL remains **14.3787% higher**, and is worse on every validation window.
This does not restore the original language-model perplexity, nor is it a
comparison against a separately trained S16 + Resurface model.

After removing the adapter, the entire SQ baseline reproduces exactly:
all 130 per-window NLL values and all 768 generated token sequences, including
decoded predictions. All four full runs took 28.32 minutes combined, excluding
model loading; this evaluation timing is not a controlled throughput benchmark.
The independent CPU audit passed all data, score arithmetic, bootstrap, cache,
calibration, training/export binding and restoration checks. CUDA remained
uninitialized during that audit. It validates recorded evidence and does not
independently rerun GPU logits.

| Predeclared criterion | Result |
| --- | --- |
| PPL no more than 1% worse than SQ baseline | Pass: 3.48% better |
| Normal MK improves versus SQ baseline | Pass: +189/384 |
| Paired MK 95% interval strictly above zero | Pass |
| Entire SQ baseline restored exactly | Pass |
| Original S16 PPL restored within 1% | **Fail: 14.38% higher** |

### Recall by record count and template

Here N is the number of key-value records in the prompt, not the number of SSM
state coordinates. Each template/record-count cell has 64 normal CONFIRM cases.

| Record count / template | S16 | SQ3.25 | SQ3.25 + new Resurface |
| --- | ---: | ---: | ---: |
| 16 / T0 | 25/64 | 6/64 | 50/64 |
| 16 / T1 | 29/64 | 12/64 | 61/64 |
| 16 / T2 | 59/64 | 28/64 | 58/64 |
| **16 records subtotal** | **113/192** | **46/192** | **169/192** |
| 64 / T0 | 4/64 | 0/64 | 12/64 |
| 64 / T1 | 3/64 | 0/64 | 32/64 |
| 64 / T2 | 26/64 | 4/64 | 26/64 |
| **64 records subtotal** | **33/192** | **4/192** | **70/192** |

Improvement is not uniform across templates: T2 is 84/128 after training versus
85/128 for original S16, while T0 and T1 improve over the original source.

## Descriptive pilot

Four fixed WikiText-2 validation windows, 8,192 predicted tokens, 48 normal DEV
MK prompts and 48 target-removed controls per arm. This pilot does not select
the adapter or decide whether to run full validation.

| Arm | PPL | Normal MK | Target removed | Cache/sequence |
| --- | ---: | ---: | ---: | ---: |
| Original FP16 weights, S16 state | 6.267285 | 28/48 (58.33%) | 0/48 | 116.3750 MiB |
| Original FP16 weights, SQ3.25 state | 7.216644 | 8/48 (16.67%) | 0/48 | 27.1797 MiB |
| SQ3.25 → newly trained Resurface | 7.014656 | 21/48 (43.75%) | 0/48 | 27.1797 MiB |

Relative to unadapted SQ3.25, the adapter lowers pilot PPL by **2.80%** and
improves MK by **27.08 percentage points**: 13 recovered answers, zero
regressions. The paired bootstrap 95% interval is **+14.58 to +39.58 percentage
points**. It still has **11.92% higher PPL** than the original S16 arm.

The restored SQ baseline exactly reproduces every pilot window NLL and all
96 generated token sequences. The independent CPU audit reconstructs scores
from the saved NLL and generated-token evidence, checks dataset/prompt hashes,
recomputes the paired bootstrap, and verifies training/artifact bindings.
It is an audit of recorded evidence, not an independent GPU rerun of logits.

## Memory accounting

Each row of 128 SSM coordinates contains 16 INT8 values, 64 INT4 values and
48 zero-carry coordinates, plus two FP16 scales:
`(16 × 8 + 64 × 4 + 2 × 16) / 128 = 3.25 bits/element`.
The fresh no-adapter calibration ranks coordinates separately for each of
56 layers and eight B/C groups, using 4,096 TRAIN tokens only.

| Batch-one persistent allocation | S16 | SQ3.25 / SQ3.25 + Resurface |
| --- | ---: | ---: |
| SSM values/codes | 117,440,512 B | 22,020,096 B |
| SSM scales | 0 B | 1,835,008 B |
| Convolution state | 4,587,520 B | 4,587,520 B |
| Compact coordinate permutation tables | 0 B | 57,344 B |
| **Total** | **122,028,032 B** | **28,499,968 B** |

The total persistent cache is **76.6447% smaller**, saving 89.1953 MiB per
batch-one sequence including the fixed table cost in this comparison. The
adapter is memoryless and adds no recurrent cache. Permutation tables can be
shared across batch entries; the batch-one totals are not a general batching
formula.

Base weights remain approximately **16.474 GB**. This experiment does not
reduce whole-model memory by 76.64%. The adapter, GPU kernel registers, temporary
workspace, logits, training teacher and optimizer are outside the cache table.

## Evidence and limits

- [Training report](../reports/quant_first_v2/training/report.json),
  [FP16 adapter](../reports/quant_first_v2/training/adapter_fp16.pt), and
  [independent training audit](../reports/quant_first_v2_training_audit.json).
- [Fresh calibration receipt](../reports/quant_first_v2/calibration/calibration.json).
- [Training scan checks](../reports/state_training_scan_checks.json) and
  [8B smoke receipt](../reports/quant_first_v2/smoke.json).
- [Pilot comparison](../reports/quant_first_v2/evaluation/pilot_comparison.json)
  and [independent pilot audit](../reports/quant_first_v2_pilot_audit.json).
- [Full comparison](../reports/quant_first_v2/evaluation/full_comparison.json),
  [entire baseline restoration](../reports/quant_first_v2/evaluation/full_restoration.json),
  and [independent full audit](../reports/quant_first_v2_full_audit.json).

The source is `nvidia/mamba2-8b-3t-4k`, revision
`b915550c63ba9359f88f44d1f6a600d85af27302`. Its BF16 checkpoint is cast to FP16
in the native runtime. All current arms use the same serial token recurrence;
historical parallel-SSD PPL/MK numbers are not interchangeable with these
baselines. The 507 source tensor identities, versions and frozen gradients
were checked throughout; a full post-training hash of base GPU tensor bytes
was not computed.

Validation corpora and prompt families have historical project exposure.
TRAIN, DEV and CONFIRM key/value bands are separated, but this is not a claim
of untouched benchmark generalization. No broad downstream, 4K/8K context,
ASIC energy, inference throughput or multi-seed training study was performed
for this candidate. The packed kernel implements the documented arithmetic;
it is not claimed bit-identical to the archived upstream StateQuant kernel.

The earlier v1 experiment used an old FP16-trained adapter during calibration
and deployment; see [historical results](RESULTS.md). Its table is not a
controlled comparison of training order alone because calibration also changed.

## Artifact identities

| Artifact | SHA-256 |
| --- | --- |
| Frozen v2 protocol | `24466642ce87c75fc2136a42ed14e69733c462b507a0be82c062b4da7f836bcb` |
| New calibration | `c366cd577967e64635f1dd960237dc1c0024d7413685ce0b4c4e40da869a7023` |
| Training report | `e5a77d86cf2fb0e2389247e3cb325f74e89957861a6043e92a891d6d402ae359` |
| New FP16 adapter | `7144b88265a797ef935b1f94845395abec5532d8dc5b2a4a8fa0a8e07ce205f0` |

Four recovery optimizer checkpoints remain in the experiment environment;
their hashes are included in the training report and checked by the audit.
The small adapter/calibration and measurement receipts are versioned here.
No public model release is part of this experiment.
