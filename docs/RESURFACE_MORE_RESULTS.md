# More Resurface training under SQ3.25

Experiment date: 2026-09-28. See the pre-training
[frozen continuation protocol](RESURFACE_MORE_PROTOCOL.md) and the
[parent v2 results](QUANT_FIRST_RESULTS.md).

**Status: formal continuation training and export checks are complete. Full
PPL/MK evaluation and the independent result audit are pending.** Training
losses and implementation checks do not establish a quality improvement.

## What was trained

The model path remains **original pure Mamba2-8B FP16 → original-coordinate
SQ3.25 state → Resurface**. This experiment continues the v2 adapter from its
final training checkpoint. Adapter capacity, the state format, calibration
and base weights remain fixed. The only quality candidate is the final
4,608-update adapter. There is no intermediate checkpoint selection.

The continuation completed **3,072 additional successful updates in 3,073
attempts**, with **one overflow retry**, bringing the total to **4,608 successful
updates**. Elapsed time from the receipt's start to finish was **64.34 minutes**,
including setup and final export checks. Recorded training attempts account
for 63.57 minutes. Peak allocated GPU memory was **34,818,563,072 bytes**
(32.43 GiB), including the frozen student, separate S16 teacher and training
workspace. These timings are execution receipts, not controlled throughput
benchmarks.

The final adapter contains **1,154,104 parameters in 224 FP16 tensors**. Its
serialized file is **2,375,103 bytes**, including **2,308,208 bytes of tensor
payload**. The adapter is memoryless and adds no recurrent cache.

## Exact parent continuation

Before any continuation update, the trainer verified the pinned parent report,
checkpoint, exported adapter, calibration and all parent training code hashes.
It restored and checked every value of the **224 FP32 master tensors**, all
**224 AdamW states**, both parameter groups and the complete GradScaler state.
Every restored master cast to FP16 matched the parent export. All optimizer
step counters started at 1,536.

The initial GradScaler state was scale 16, growth factor 2, backoff factor 0.5,
growth interval 2,000 and growth tracker 1,536. Resumed training forward matched
the parent FP16 adapter under packed SQ3.25 inference **bitwise on 128 fixed
TRAIN tokens**. A separate one-update smoke run was discarded; formal training
reloaded the immutable parent checkpoint.

The one formal overflow occurred on attempt 2,730, after 2,729 successful
additional updates. It left the update count unchanged and retried the same
numeric case and prose segment. Scale changed from 64 to 32 while growth
tracker 265 was retained, preserving the parent trainer's manual overflow
semantics. Final scaler state was scale 32 and growth tracker 608. The receipt
records and checks the complete scaler state before and after every attempt.

At completion, all final checkpoint masters cast exactly to the actual FP16
export. The saved optimizer and scaler matched the final live training states.
The exported adapter's packed inference matched training forward **bitwise on
the same fixed 128-token probe**, and its allocated persistent cache matched
the parent exactly. These are implementation checks on a fixed probe; full
PPL/MK quality is evaluated separately.

## Fixed training recipe

- Reuse the same 1,536 numeric TRAIN cases in two independent permutations,
  with CPU seeds `2026092804` and `2026092805`. This adds training exposure,
  not new numeric examples.
- Continue the original prose schedule at global index `j = 1536 + r` for
  additional successful update index `r`. Each segment contains 512 TRAIN
  tokens and 511 targets. Overflow retries do not advance either schedule.
- Keep the objective: answer-only MK CE + 0.5 prose CE + 0.5
  KL(S16 teacher || SQ student) + 3 closure. The teacher is the separate frozen
  original model with S16 per-token state.
- Keep the closure budget 0.006, excess coefficient 10, full 256K vocabulary,
  AdamW betas (0.9, 0.999), epsilon 1e-8, zero weight decay, gradient clipping
  at 1, FP32 masters, FP16 forward casts and block checkpointing.
- Apply one continuous cosine tail over all 3,072 additional updates:
  `factor(r) = 0.01 + 0.09 * (1 + cos(pi*r/3071)) / 2`.
  V/g learning rate decreases from 1e-5 to 1e-6; router rate decreases from
  3e-5 to 3e-6. The original learning-rate schedule is not restarted.
- Save recovery checkpoints after 768, 1,536, 2,304 and 3,072 additional
  updates. The final checkpoint is the sole exported quality candidate.

The calibration and numeric TRAIN manifests retain their v2 protocol binding.
The continuation adds its own `continuation_protocol_sha256`; no recalibration
or relabeling of the old inputs occurred. Q8 and weight quantization are excluded.

## Full validation — pending

The frozen comparison evaluates the parent v2 adapter, continued v3 adapter,
and a complete repeat after restoring the parent, all with the same SQ3.25
cache and original FP16 weights. Each arm uses 130 WikiText-2 validation windows
with 264,764 predicted tokens, 384 normal CONFIRM MK prompts and 384
target-removed controls. Generation is full-vocabulary greedy decoding with
at most 12 tokens and the same six-digit scoring rule.

The first parent arm must reproduce the archived v2 result for every per-window
NLL and generated token sequence. The restored parent arm must reproduce that
first arm exactly. The improvement gate compares **v3 against v2**: PPL no more
than 1% worse, observed MK improvement, and a strictly positive lower bound of
the paired MK 95% bootstrap interval (10,000 draws, seed 20260928). Whether PPL
itself improves is reported separately. Restoring original S16 PPL still
requires no more than a 1% increase over the original model.

The hash-verified v2 original S16 and unadapted SQ full results provide archived
context. They are not fresh runs in this continuation experiment. No full
v3 quality result or improvement-gate decision is available in this document
yet.

## Memory accounting

State representation is unchanged: each 128-coordinate row contains 16 INT8
values, 64 INT4 values, 48 zero-carry coordinates and two FP16 scales:
`(16*8 + 64*4 + 2*16) / 128 = 3.25 bits/element`.

| Batch-one persistent allocation | Bytes |
| --- | ---: |
| SSM codes | 22,020,096 |
| SSM scales | 1,835,008 |
| FP16 convolution state | 4,587,520 |
| Compact coordinate permutation tables | 57,344 |
| **Total** | **28,499,968** |

The total is **27.1796875 MiB per batch-one sequence**, identical to v2.
Permutation tables can be shared across batch entries. The original FP16
weights remain approximately **16.474 GB**. Adapter tensors, temporary
activations, GPU registers, allocator reserves, teacher and optimizer memory
are outside the persistent cache total. This experiment does not reduce the
base weight size.

## Artifacts and provenance

- [Completed training receipt](../reports/resurface_more_v3/training/report.json).
- [Actual FP16 adapter](../reports/resurface_more_v3/training/adapter_fp16.pt).
- [Continuation trainer](../scripts/train_resurface_more.py).
- [Frozen continuation protocol](RESURFACE_MORE_PROTOCOL.md).

The local training receipt and adapter file were checked against the hashes
below. Recovery checkpoint identities are recorded by the trainer; independent
checkpoint and full-result auditing are reported separately when complete.

| Artifact | SHA256 |
| --- | --- |
| Continuation protocol | `4c2c47aa00936ded52cf7b337126f1cce9556da7021e0a75c6e9df83f4949330` |
| Parent v2 protocol | `24466642ce87c75fc2136a42ed14e69733c462b507a0be82c062b4da7f836bcb` |
| Parent completed training report | `e5a77d86cf2fb0e2389247e3cb325f74e89957861a6043e92a891d6d402ae359` |
| Parent final training checkpoint | `bc548dd427d114098048fa1863f8e602c095dc2d9fde56348795628ae8e2c78f` |
| Parent FP16 adapter | `7144b88265a797ef935b1f94845395abec5532d8dc5b2a4a8fa0a8e07ce205f0` |
| Unchanged no-adapter calibration | `c366cd577967e64635f1dd960237dc1c0024d7413685ce0b4c4e40da869a7023` |
| Continuation trainer | `54df8c8fb6130fbe7f39e580eec67a620521931d877fc1fdeb4bc93f8404a18e` |
| Continuation completed training report | `ff6325677b8da8b3c4eb094b664558688a06ced311466d606df16c28361bbf4b` |
| Final 4,608-update checkpoint | `e18ce6803d039c73cc1d03217bf944a38f2f899cef3efebe02430a9f42a600c9` |
| Final continued FP16 adapter | `4dbc2ad1e21065a399d9e1189696648105b405c2387da2406f1da8310725bbbd` |

## Scope and limitations

The source remains `nvidia/mamba2-8b-3t-4k`, revision
`b915550c63ba9359f88f44d1f6a600d85af27302`. Its BF16 checkpoint is cast to FP16
in the native runtime. Current comparisons use the same serial token
recurrence; historical parallel-SSD measurements are not interchangeable.

Forward training uses the deployed packed SQ recurrence. Backward uses the
same masked straight-through estimator as v2: derivative one through 80 live
carry coordinates, zero through 48 discarded carry coordinates, with
current-token readout gradients retained. It ignores derivatives of scale
selection, rounding and clipping. FP32 atomic backward reductions prevent a
promise of bit-identical retraining.

All 507 original parameter identities, versions and frozen-gradient states
were checked; a full post-training byte hash of all base GPU weights was not
computed. The calibration guard and frozen teacher checks passed. Fixed-probe
forward equivalence does not demonstrate equivalence at all sequence lengths.

Training reuses the previously prepared TRAIN sets. The evaluation corpora and
prompt families have historical project exposure; even a passing full gate
would not establish untouched generalization. No additional adapter capacity,
new numerical state format, broad downstream benchmark, long-context study,
ASIC energy result or multi-seed training result is claimed. The repository
remains private; this experiment does not include a public model release.
