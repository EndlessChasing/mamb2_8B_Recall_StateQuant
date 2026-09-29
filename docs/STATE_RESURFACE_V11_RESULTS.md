# V11 Resurface on the validated Q3.25 base

Status: fresh training, all three full PPL/MK arms and the independent full
audit completed. All nine prospective quality checks pass: Resurface improves
recall and PPL on the fixed V10 Q3.25 base, with unchanged recurrent cache.

## Checklist

- [x] Freeze the validated V10 source, table, layout and archived PPL 8.186186562837207.
- [x] Freeze the prospective [training and acceptance protocol](STATE_RESURFACE_V11_PROTOCOL.md).
- [x] Verify the 64-live stored-scale training forward, history recomputation and STE backward.
- [x] Complete an independent CPU audit of the recorded GPU kernel evidence.
- [x] Discard a fresh one-update smoke after exact 128/512-token deployment parity checks.
- [x] Train a separately initialized adapter for 1,536 successful updates.
- [x] Audit all updates, source/table freezing, checkpoints and final FP16 export.
- [x] Evaluate unadapted, adapted and adapter-removed arms on full PPL and 768 MK prompts each.
- [x] Audit complete raw results, paired MK confidence interval, memory and exact restoration.
- [x] Save adapter, reproduction commands and evidence in this repository.

The [release usage guide](STATE_RESURFACE_V11_RELEASE.md) covers the packaged
adapter/table, source checkpoint download and inference entrypoint.
The new inference wrapper also passes a [GPU smoke check](../reports/state_resurface_v11_release_wrapper_smoke.json):
all 12 generated tokens match one archived 263-token MK prompt exactly,
with the same 28,499,968-byte state cache. This checks the new wrapper and
does not replace or repeat the full quality evaluation below.

Primary acceptance: adapted PPL <= 8.186186562837207 and < 8.25, positive normal
MK improvement with paired 95% lower bound > 0, exact control replay, and unchanged
28,499,968-byte request cache. The FP16 adapter is accounted for separately.

## Pre-training evidence

The first GPU checker attempt passed all 189 checks over nine raw fixtures;
19 CPU contract/fixture checks and 23 evaluator fixtures also passed. The
independent CPU audit reconstructed the carries, checkpoints and STE gradients.
The discarded model smoke completed one update in three attempts (two dynamic
loss-scale overflows), with exact 128/512-token initialization and exported
FP16 forward parity. Its adapter has 917,456 nonzero V_read values; peak CUDA
allocated memory was 34,219,775,488 bytes. These are implementation/training
checks and do not establish PPL or MK quality.

## Completed training

The fresh adapter completed 1,536 successful updates in 1,541 attempts (five
early dynamic-scale overflows), in 1,984.97 seconds. All four checkpoints,
source/table/teacher guards, exact 128/512-token FP16 export parity and the
independent training audit passed. Peak allocated CUDA memory was
34,786,866,688 bytes. The actual final export is 2,375,743 file bytes and
2,308,208 FP16 tensor bytes, SHA256
`339334b3431027504fb8da4ce3167d63c8c2f309920cfb0076bde1f72a115cb7`.

## Full measured PPL and MK

All 130 WikiText-2 validation windows / 264,764 targets and all 768 CONFIRM
prompts are complete for all three arms. Only the final 1,536-update export is
evaluated. Removing it exactly restores all 130 PPL rows, all 768 generated
sequences/predictions, reset probes, PPL/MK-end cache receipts and storage
descriptors. The first baseline also exactly replays the archived V10 result.
The independent CPU full audit passes without initializing CUDA.

| Configuration | Full PPL | Normal MK /384 |
| --- | ---: | ---: |
| Original S16, separately hash-bound historical reference | 7.334322057 | 146 (38.0208%) |
| V10 Q3.25, no adapter | 8.186186563 | 32 (8.3333%) |
| V10 Q3.25 + fresh V11 Resurface | 7.855569606 | 271 (70.5729%) |
| V10 after removing the new adapter | 8.186186563 | 32 (8.3333%) |

PPL improves 4.0387%. Normal MK improves 62.2396 percentage points: 242
previously wrong cases become correct and 3 previously correct cases regress.
The sorted-case paired 10,000-resample bootstrap 95% interval is
[+57.2917, +67.1875] percentage points (seed 20260928). Both arms get 29 cases
correct and 110 cases wrong. The 384 target-removed prompts are a separate
diagnostic and are not included in this recall denominator.

| Normal MK stratum | No adapter | Fresh Resurface |
| --- | ---: | ---: |
| N16 | 30/192 (15.6250%) | 181/192 (94.2708%) |
| N64 | 2/192 (1.0417%) | 90/192 (46.8750%) |
| Template 0 | 7/128 | 87/128 |
| Template 1 | 5/128 | 92/128 |
| Template 2 | 20/128 | 92/128 |

| Query-position quartile | No adapter | Fresh Resurface |
| --- | ---: | ---: |
| N16, first | 20/51 | 46/51 |
| N16, second | 7/51 | 49/51 |
| N16, third | 1/51 | 49/51 |
| N16, fourth | 2/39 | 37/39 |
| N64, first | 2/48 | 17/48 |
| N64, second | 0/48 | 23/48 |
| N64, third | 0/48 | 28/48 |
| N64, fourth | 0/48 | 22/48 |

Quartile is `floor(4 * zero_based_query_position / N)`. N16 denominators are
uneven under the frozen prompt schedule. Raw reports also retain every exact
position and N/template stratum; N64 exact-position cells contain only three cases.

| Normal prediction category | No adapter | Fresh Resurface |
| --- | ---: | ---: |
| Correct target | 32 | 271 |
| Wrong value present in the prompt | 221 | 33 |
| Value absent from the prompt | 128 | 77 |
| Unparseable | 3 | 3 |

Both arms have 0/384 accidental matches to the removed target. On these removed
prompts, other-present values are 156 → 169, absent values 226 → 210, and
unparseable outputs 2 → 5. This is not an abstention benchmark.

The archived S16 report is separately bound to SHA256
`52f82f83258a2fa3160ea14585f1bd69e1d68d60d8f179d1f0987c636f6546ba`;
it was not rerun in V11. Adapted Q3.25 PPL remains 7.1070% higher than S16.
Normal MK rises by 32.5521 percentage points versus that archived reference
(149 gains, 24 losses; paired 95% interval [+26.8229, +38.2878] pp).
Improvement over V10 does not restore original S16 PPL.

## Storage scope

Actual batch-one persistent cache remains 28,499,968 bytes (27.1796875 MiB):
23,855,104 SSM bytes, 4,587,520 convolution bytes and one 57,344-byte static
shared table. It is 76.6447% below the archived S16 cache of 116.375 MiB.
One loaded FP16 adapter adds 2,308,208 bytes, making cache plus adapter
30,808,176 bytes (29.38097 MiB). Static table and adapter may be shared across
requests; the table is included once in the stated batch-one figure.

These figures exclude the 16,473,999,360 bytes of unchanged FP16 source weights,
intermediate tensors, allocator reserve, training teacher/optimizer and scratch.
This experiment compresses recurrent state; source weights are not quantized.

## Limits

The validation corpus and CONFIRM prompt families have historical exposure;
these results do not establish performance on untouched test data or broad
language/retrieval generalization. Training/export parity is checked exactly
at 128 and 512 tokens; STE gradient fixtures use predeclared tolerances. Source
identity/version/gradient checks are not a post-run full model byte audit.
The CPU audit reconstructs recorded evidence rather than rerunning GPU logits.

The first full-evaluation attempt exhausted disk after 744 baseline MK cases.
Its partial records are preserved separately. All three arms were restarted
from scratch with the same frozen code, inputs and adapter in RAM-backed
scratch; no results from the failed run are pooled. See the
[reproduction and recovery guide](STATE_RESURFACE_V11_REPRODUCTION.md).

## Evidence

- [Frozen protocol](STATE_RESURFACE_V11_PROTOCOL.md).
- [Training receipt and final export](../reports/state_resurface_v11_training/report.json).
- [Independent training audit](../reports/state_resurface_v11_training_audit.json).
- [Complete comparison and all nine gates](../reports/state_resurface_v11_full/full_comparison.json).
- [Independent full audit](../reports/state_resurface_v11_full_audit.json).
- [GPU-to-Mac transfer manifest](../reports/state_resurface_v11_full_ram_copy/transfer_manifest.json).

Full comparison SHA256:
`dc8211177d5eb760ae593f2645a52053794af2c043bf2b04b2deba2dfb0464fc`.
Full independent audit SHA256:
`ad2b3f14a075b0d60993f3579e99fc98bfbac8aa527f6a0c8d684cc9493e7d72`.
