# Mamb2_8B_Recall + StateQuant

## Public release: Q3.25 state + Resurface

Hugging Face family name:
[**Mamb2_8B_Recall_SQ3.25**](https://huggingface.co/EndlessChasing/Mamb2_8B_Recall_SQ3.25).
Its model card includes download and inference instructions for the same
verified adapter, state table and runtime.

Download the adapter/table bundle from
[v0.1.0-q325-resurface](https://github.com/EndlessChasing/mamb2_8B_Recall_StateQuant/releases/tag/v0.1.0-q325-resurface)
and follow the [installation, verification and inference guide](docs/STATE_RESURFACE_V11_RELEASE.md).
**Q3.25 describes recurrent state. The original 8B model weights remain FP16**
and are downloaded separately from NVIDIA. The release includes the exact
audited adapter, calibration table, raw evaluation reports and checksums.

## Full official WikiText-2 test PPL — fixed published checkpoint

| Configuration | WikiText-2 test PPL |
|---|---:|
| Published SQ3.25 state, without Resurface | 8.11328914 |
| Same frozen FP16 base and state + released V11 Resurface | **7.77130287** |

Both complete arms score **300,963 next-token targets in 147 reset windows**,
including the last window of 1,955 targets. The adapter lowers pooled-token PPL
by **4.2151%**. The weights, 32/32/64 state table, final adapter and measured
runtime are unchanged from `v0.1.0-q325-resurface`. No training, recalibration,
checkpoint/candidate selection, or test-dependent stopping was performed.

**Protocol:** pinned `Salesforce/wikitext`, `wikitext-2-raw-v1`, official `test`
split at revision `b08601e04326c79dfdd32d625aee71d232d685c3`; documents joined
by two newlines and NVIDIA SentencePiece without automatic BOS/EOS. Each window
scores at most 2,048 next-token targets with one boundary-token overlap; state
is requantized every token. PPL is `exp(total NLL / 300963)` from FP16 compute
and FP32 logits.

**Data scope:** adapter training used separate official TRAIN data. The test
was scored after publication with the fixed released checkpoint and without
test-driven selection. Earlier project experiments, including 2.7B and E8/W5,
used the official WT2 test text, so this is not an untouched test for the whole
project. Source-model pretraining contamination and cross-split duplication
have not been audited. No S16 test control or new MK evaluation was run here.

The exact [comparison](evaluation/wt2_test_v1/comparison.json),
[adapter-free arm](evaluation/wt2_test_v1/without_resurface.json),
[adapted arm](evaluation/wt2_test_v1/resurface.json), and
[CPU audit](evaluation/wt2_test_v1/cpu_audit_v1.json) contain all per-window
scores/token hashes and the recorded tensor/storage bindings, without raw token
IDs. The CPU audit verifies token coverage, pooled NLL/PPL arithmetic, the
published runtime and 507-weight/224-adapter bindings, and adapter-removal
reset/cache restoration. It does not recompute GPU logits. See the
[reproduction guide](evaluation/wt2_test_v1/README.md),
[frozen test protocol](docs/SQ325_V11_WT2_TEST_V1_PROTOCOL.md), and
[evidence inventory](evaluation/wt2_test_v1/inventory.json).

This is a documentation/evaluation overlay. The original release tag, adapter,
state table, runtime source and sealed `release/` files remain unchanged.
Historical validation and MK CONFIRM results remain separate below; they are
not WT2-test MK or newly collected test recall measurements.

## Historical v11 validation and published MK CONFIRM

The results below are historical validation/CONFIRM measurements; the new test
result is reported above. **All nine predefined quality and integrity gates passed.** Fresh Resurface
training completed 1,536 successful updates on the fixed 32 INT8 / 32 INT4 /
64 zero-carry layout. Original source weights remain frozen FP16.

| Configuration | WikiText-2 validation PPL | Published normal MK /384 |
| --- | ---: | ---: |
| Original S16, archived reference | 7.334322057221965 | 146 (38.02%) |
| V10 Q3.25, no adapter | 8.186186562837207 | 32 (8.33%) |
| **V10 Q3.25 + fresh V11 Resurface** | **7.855569605864836** | **271 (70.57%)** |

Against the fixed V10 base, PPL improves **4.0387%**, remaining below 8.25;
normal MK improves **62.2396 percentage points**, with paired bootstrap 95%
interval **[+57.2917, +67.1875] pp**. There are 242 paired gains and 3 regressions.
N16 recall improves **30 → 181/192** and N64 **2 → 90/192**. Target-removed
accidental matches remain **0/384** in both arms; this is a diagnostic, not
an abstention score.

All three arms complete 130 PPL windows / 264,764 targets and 384 normal +
384 target-removed MK prompts. Removing the adapter reproduces every baseline
window NLL and all 768 generated sequences exactly, including reset/cache
receipts. The archived V10 replay and independent CPU full audit pass.

Persistent state remains **28,499,968 bytes (27.1797 MiB)** per sequence at
batch one. The separate FP16 adapter payload/residency is **2,308,208 bytes**;
cache plus adapter is **30,808,176 bytes (29.3810 MiB)**. These figures exclude
source weights, temporary computation and allocator reserve. Although recall
exceeds the historical S16 result, PPL is still **7.11% higher than S16**.

Only the final TRAIN-derived export was evaluated; validation/CONFIRM did not
select a checkpoint. This benchmark family has historical exposure, so these
results are not a new unseen-benchmark claim. The audited adapter, checkpoints
and reproduction evidence are retained in this public repository.
See [V11 results and checklist](docs/STATE_RESURFACE_V11_RESULTS.md),
[raw full comparison](reports/state_resurface_v11_full/full_comparison.json),
[independent full audit](reports/state_resurface_v11_full_audit.json),
[reproduction](docs/STATE_RESURFACE_V11_REPRODUCTION.md), and
[frozen protocol](docs/STATE_RESURFACE_V11_PROTOCOL.md).

## Completed: unadapted Q3.25 full PPL 8.186187

**The requested no-Resurface PPL < 8.25 target is achieved.** The selected
layout uses 32 INT8 + 32 INT4 + 64 zero-carry coordinates per 128-element state row.
Payload plus two FP16 scales remains 52 bytes: 3.25 bits per coordinate.

| Configuration | Full WikiText-2 PPL | Persistent cache / sequence |
| --- | ---: | ---: |
| Original S16, archived reference | 7.334322 | 116.3750 MiB |
| Previous best unadapted Q3.25 (v8) | 8.283863 | 27.1797 MiB |
| **Selected v10 Q3.25, no Resurface** | **8.186187** | **27.1797 MiB** |

The full 130-window / 264,764-target validation and independent CPU evidence audit
pass, including exact archived/restored parent PPL, hidden/cache probes and
actual allocation checks. Source weights remain frozen FP16. The 27.1797 MiB
figure includes SSM, convolution and one static table; it excludes model weights
and temporary computation. No MK or Resurface training is part of this result.
The benchmark family has historical exposure.

Use the [saved selected layout/table](reports/state_ppl_v10_screen/selected_calibration.pt)
with the [v10 runtime and reproduction guide](docs/STATE_PPL_V10_REPRODUCTION.md).
See [all search results/checklist](docs/STATE_PPL_TARGET_825.md),
[raw full comparison](reports/state_ppl_v10_full/full_comparison.json), and
[independent full audit](reports/state_ppl_v10_full_audit.json).
The target search is complete; V11 above adds fresh Resurface to this fixed base.

## Completed v6: PPL-first scale search finds only a small improvement

The latest user priority is **optimize PPL first; use Resurface for MK after
freezing the quantizer**. This experiment removed the v5 MK selection guard
and compared 20 fixed, same-budget table/scale candidates on 32 full-length
TRAIN windows. The frozen winner keeps the v5 INT8 table and chooses integer
codes against the actually stored FP16 scale.

| Unadapted configuration | Full PPL | Cache/sequence |
| --- | ---: | ---: |
| Original S16, archived reference | 7.334322 | 116.3750 MiB |
| v5 Q3.25 baseline | 8.367465 | 27.1797 MiB |
| v6 Q3.25 stored-scale winner | 8.355269 | 27.1797 MiB |

Full PPL improves **0.1458%**, below the predefined **1%** gate; the experiment
ends without new Resurface training. Every tested clipping factor worsens
TRAIN PPL. Both complete baseline replays pass, including the archived v5
window NLLs and hidden/cache probe. v6 measures no MK and does not establish
recall quality. Separate larger-memory ablations are diagnostic only.
The v5 **8.093011 PPL / 244 of 384 MK** endpoint below was the validated
combined reference at the end of v6; it is not a v6 result. The current V11
endpoint is reported above. Weights remain frozen FP16.

See the [frozen v6 protocol](docs/STATE_PPL_V6_PROTOCOL.md),
[results and evidence](docs/STATE_PPL_V6_RESULTS.md),
[raw full comparison](reports/state_ppl_v6_full/full_comparison.json),
[independent full audit](reports/state_ppl_v6_full_audit.json),
[reproduction](docs/STATE_PPL_V6_REPRODUCTION.md), and [checklist](PLAN.md).

## v5: optimize unadapted SQ3.25, then train fresh Resurface

The requested order is complete: **original source with no adapter → optimize
SQ3.25 → freeze the table → train fresh Resurface for recall**. TRAIN-only
selection preserves the original 16 INT8 coordinates and optimizes INT4/zero
assignments. Fresh training completed 1536 updates, with no old adapter reused.

| Configuration | Full PPL | Normal MK /384 |
| --- | ---: | ---: |
| Old SQ3.25, no adapter | 8.691553 | 50 (13.02%) |
| Optimized SQ3.25, no adapter | 8.367465 | 46 (11.98%) |
| **Optimized SQ3.25 + fresh Resurface** | **8.093011** | **244 (63.54%)** |
| Archived old SQ3.25 + v2 Resurface | 8.388906 | 239 (62.24%) |

**State optimization and subsequent recall repair pass their separate gates.**
Unadapted PPL improves **3.73%**. New Resurface then improves PPL another **3.28%**
and MK by **51.56 percentage points** (paired 95% interval +46.35 to +56.77 pp).
The stricter gate against old v2 **fails**: PPL improves **3.53%**, but the
observed five additional correct answers have an interval of −4.17 to +6.77 pp,
below the required −2 pp lower bound. It does not establish improved MK over v2.

All SQ arms use **27.1797 MiB/sequence** persistent cache, **76.64% below S16**;
source weights stay FP16. Final PPL remains **10.34% above original S16 7.334322**.
All four full arms, exact archived-old replay and exact selected-base restoration
completed, and the independent full evidence audit passed. Validation covers 264,764
PPL targets and 384 normal + 384 target-removed cases per arm.
See [results and limits](docs/STATE_FIRST_V5_RESULTS.md),
[raw comparison](reports/state_first_v5_full/full_comparison.json),
[independent audit](reports/state_first_v5_full_audit.json),
[reproduction commands](docs/STATE_FIRST_V5_REPRODUCTION.md),
[protocol](docs/STATE_FIRST_V5_PROTOCOL.md), and [checklist](PLAN.md).

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
