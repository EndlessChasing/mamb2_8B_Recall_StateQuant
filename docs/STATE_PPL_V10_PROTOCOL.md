# v10: fixed same-byte precision/coverage allocations

The method, parent, four-layout grid and TRAIN rows were proposed and approved
while v9 full validation was running, before its outcome was seen. This final
document is frozen after notification that v9's target flag is false with
cache/integrity flags true; no selected numeric PPL was used and no choices
changed. An independent full audit is still required before quality jobs.
The family uses TRAIN-only history and earlier diagnostic evidence. Do not
use validation to change its layouts, parent, data, order or selection.

## Parent and execution prerequisites

The parent is the frozen **v9 TRAIN-selected top2** coordinate table, regardless
of whether its full result improved or regressed relative to v8. Keep that
exact uint8[56,8,128] permutation unchanged for every layout. Bind its selected
payload, companion receipt, complete screen comparison/raw arms and passing
independent screen audit by SHA256 before measurements. Preserve the full
v5/v6/v8/v9 source, table, group-calibration and selection provenance chains.

The v9 selected payload SHA256 is
`3467897358f33de22b1b629819cb2035f4cb8912914f0183959aa581fafeab50`;
its TRAIN screen comparison is
`4098ad9446cb08d18c0ce23a8ce199c026a3836ae6fe5c5f3d21eadf2222d757`.
The unchanged permutation tensor SHA256 is
`b1865e81ff3fbed028027a883872aef9bb91e614e7cf08c72211496a78aeb597`.

Quality GPU jobs require a completed, independently audited **finite v9 miss**:
selected full PPL at least 8.25, with all source/backend/table/cache checks and
archived/restored replays passing. Bind the v9 full outcome, selected raw report
and full audit. If v9 reaches PPL strictly below 8.25, do not run v10 quality
experiments. Runtime/integrity failures do not count as a finite target miss.
Codec fixtures and CPU preparation may be completed separately before this
condition; they do not select model candidates or measure model quality.

Use the same original pure Mamba2-8B source, tokenizer, frozen FP16 weights,
16-warp RMSNorm backend and dataset/token definitions as v9. Source SHA256:
`47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb`.
Tokenizer SHA256:
`5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09`.
TRAIN file SHA256:
`e54b02e5162e042a9cdd504f4eb1b1652724fb240bbc2c97608967aa26297233`.
V9 protocol SHA256:
`9e01c03ee6870a8ecbcd9a0ba9157b1651d2830ea65d2951664ebcb81b4a09b3`.

Do not load/train Resurface, alter source parameters, score MK, use Q8, add
resident lookup tables/predictors or publish this experiment. This is state
compression; the source weights and linear computation remain FP16.

## Four fixed global layouts

In this exact order, use the identifiers and counts below. Counts refer to
the first INT8 coordinates, following INT4 coordinates and remaining zero
coordinates in the unchanged parent permutation.

| Identifier | INT8 | INT4 | Zero | Payload bytes | FP16 scale bytes | Row bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `16_64_48` (baseline) | 16 | 64 | 48 | 16+32=48 | 4 | 52 |
| `8_80_40` | 8 | 80 | 40 | 8+40=48 | 4 | 52 |
| `24_48_56` | 24 | 48 | 56 | 24+24=48 | 4 | 52 |
| `32_32_64` | 32 | 32 | 64 | 32+16=48 | 4 | 52 |

Choose one layout globally for all layers/heads/channels, not per-layer or
per-group mixtures. The candidate grid has exactly four members. The fixed
permutation is neither recalibrated nor reordered for a different layout.
No clipping, rotation, new codebook, additional scale or equalizer is tested.

Use the verified v6 **stored_scale, int4_clip=1.0** quantization policy.
Baseline `16_64_48` must delegate to the unchanged v6 production path and
reproduce its readouts, persistent bytes and full parent evaluation bitwise.
Its implementation must not silently route through a new equivalent kernel.

The other layouts alter only tier boundaries, the associated packed buffer
lengths and masks/reductions required by those lengths. Preserve the recurrent
update, convolution, current readout before carry quantization, and every-token
carry quantization, including prefill. For each INT8/INT4 tier use its absmax
and the same correctly rounded FP32 division by 127/7, store the scale as FP16,
quantize against that actual stored scale using correctly rounded division
and nearest/half-away integer rounding, and clamp to [-127,127]/[-7,7]. Decode
integer times stored FP16 scale to FP32 exactly as v6; zero scale produces
zero codes. Dead coordinates contribute their current-token update to the
readout before their carried state is discarded. No FP16 carry is retained.

Mask all non-power-of-two tier padding explicitly, including 24/48/80 live
lengths and 40/56 dead lengths; padded lanes must not read/write adjacent rows
or contribute to absmax/readout reductions. New layouts can change floating
reduction association as a consequence of group lengths. Disclose numerical
validation scope; do not claim their readouts equal the baseline layout.

## Actual storage and codec validation

Every state row has exactly 48 payload bytes and two FP16 scales: **52 bytes
per 128 coordinates = 3.25 bits/coordinate including scales**. Actual packed
SSM allocation is 56×128×64×52 = **23,855,104 bytes** at batch one. FP16
convolution allocation is **4,587,520 bytes** and the single original static
permutation is **57,344 bytes**: total **28,499,968 persistent bytes**.
Layout is a global compile-time/runtime scalar configuration, not an extra
resident tensor. No per-layer layout table, padding allocation outside these
counts, additional recurrent state or resident decoding metadata is allowed.
This accounting excludes original weights and temporary computation; it is
not a total GPU-memory claim.

Before any quality run, require a passing actual GPU codec-check receipt with
source hashes and an independent CPU audit. Cover baseline exact delegation;
controlled every-token code/scale/carry oracles for each layout; sign and
rounding boundaries, zero/subnormal scales; partition/chunk equivalence;
non-power-of-two padding and row boundaries; index ranges; and actual tensor
storage bytes for every layout at production geometry. Keep known numerical
errors and failed attempts. Do not weaken acceptance criteria after seeing
model PPL. Bind all new orchestration, codec, checker and auditor sources plus
their imported production sources and this protocol before measurements.

## TRAIN selection

Use TRAIN rows **184..215**, all 2048 tokens per row: 32 independent windows /
**65,504 prediction targets** per arm. These rows are disjoint from v6/v7/v8/v9
search rows. Each window resets the complete state as before.

Execute baseline `16_64_48`, then `8_80_40`, `24_48_56`, `32_32_64`, then
restored baseline. Measure complete model PPL for all four; no calibration
proxy or MK enters selection. Choose minimum complete finite aggregate TRAIN
PPL, exact ties preferring baseline and then the fixed layout order. Freeze
the winning layout with the unchanged parent permutation and provenance.

Baseline and restored baseline must be complete and finite. A named known
nonfinite new-layout result may be excluded with failed evidence. Arbitrary
runtime/compiler errors, invalid buffer geometry, out-of-bounds behavior,
reset mismatch or source/table/backend integrity failure stops the experiment.
Finite poor results remain in the comparison. If baseline wins, record the
negative family and skip redundant full confirmation.

## Full confirmation and completion rule

Evaluate the fixed v9 parent using baseline delegation, the one frozen TRAIN
winner, and restored parent on all **130 validation windows / 264,764 targets**.
Both parent evaluations must exactly replay the pinned v9 selected full report:
all window NLLs, aggregate PPL, 128-token hidden and per-cache-tensor hashes,
and the physical cache allocation. If new receipts append layout-descriptor
fields, compare the common v6 physical-cache fields exactly and separately
verify the descriptor counts; the parent's physical representation is unchanged.

Success requires full candidate PPL **strictly below 8.25**, a passing independent
CPU full audit, all exact replays/integrity guards, no adapter and the unchanged
**28,499,968-byte cache**. There is no additional 1% gate. Original S16 PPL
7.334322057221965 is comparison context, not a candidate. Full validation never
chooses a second layout, changes the tier boundaries or expands the grid.

Each arm records raw token hashes/NLLs/counts/PPL, repeated-reset 128-token
hidden and every persistent cache tensor hash, actual buffer shapes/bytes,
source and CPU/GPU table guards, pinned backend evidence, and no-adapter/no-MK
flags. The CPU auditor independently reconstructs storage, selection, arithmetic,
parent binding, target checks and replay without rerunning model logits or
importing the new selector's scoring/selection implementation.

## Rationale, budget and limits

Earlier TRAIN ablations found a larger PPL penalty for quantization alone than
for pruning alone, but those effects are not additive and did not isolate the
INT4 tier. More INT8 coordinates may reduce quantization damage while fewer
retained coordinates may increase pruning damage; `8_80_40` tests the opposite
tradeoff. The observed ablations do not establish which allocation will win.
This route changes the allocation tradeoff rather than selecting among the
same old 16/64/48 coordinate tables.

Screening is exactly five arms × 32 windows = 160 model windows / 327,520
prediction targets. A nonbaseline winner adds three full arms / 794,292 targets.
Codec validation is a separate prerequisite and is not model-quality evidence.
No quality gain is promised. The reused permutation was built for 16/64/48;
it may be suboptimal at other boundaries. Historical benchmark/data exposure
remains, so this is not untouched generalization evidence. Preserve all results,
keep the repository private and propose a distinct TRAIN-selected route if
this bounded family misses the target.
