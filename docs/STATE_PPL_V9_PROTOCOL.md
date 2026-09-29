# v9: group-level table refinement for unadapted Q3.25 PPL < 8.25

Prospective protocol, frozen before any v8 full-validation outcome was seen.
The v8 full process had already started at protocol freeze; this document does
not claim otherwise. The proposal, rows, candidate family and tie rules below
were chosen using prior TRAIN evidence. No v9 model job is authorized unless
v8's completed independent full audit establishes a finite target miss.

## Fixed source, parent and physical format

Keep the original pure Mamba2-8B source, tokenizer, frozen FP16 parameters,
16-warp RMSNorm backend, reset semantics and token definitions used by v8.
Source SHA256:
`47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb`.
Tokenizer SHA256:
`5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09`.
TRAIN file SHA256:
`e54b02e5162e042a9cdd504f4eb1b1652724fb240bbc2c97608967aa26297233`.

Use exactly the verified v6 `stored_scale`, `int4_clip=1.0` production codec.
Do not add a kernel variant or alter integer rounding, scales, readout order,
every-token carry quantization, convolution or precision. Current readout
precedes carry quantization, including prefill. Do not load/train Resurface,
change source weights, evaluate MK, introduce Q8 or publish this experiment.

The static parent is **the v8 TRAIN-selected table**, irrespective of whether
its full PPL improved or regressed relative to v6. Do not choose the parent
using validation. Bind the completed v8 screen comparison, selected payload,
all raw screen reports and independent screen audit by SHA256 before any v9
measurement. The selected v8 table is the `top8` TRAIN winner; the derivation
and exact table bytes must be verified through its audited v8 evidence chain.
Do not substitute a different top-k mixture after inspecting v8 validation.

v8 protocol SHA256:
`880839c0b0919d0a9a119647d2d9c1657e5b8eb5389c8c2ee22ee1785917aa01`.
The four original v5 table payload SHA256 remains
`cd86a755db5004c716922696cf5532b307c57f5fb7dad4bcca298eb58b55f9a7`.
Their fixed order is:
`magnitude`, `full_readout`, `preserve_int8`, `preserve_retained80`.
Reuse their complete upstream calibration/source/TRAIN provenance.

Runtime retains exactly one uint8[56,8,128] permutation table: 57,344 bytes.
Every state row still contains 16 INT8 coordinates, 64 INT4 coordinates,
48 zero coordinates and two FP16 scales: 52 bytes per 128 coordinates.
Batch-one packed SSM storage is 23,855,104 bytes; FP16 convolution storage is
4,587,520 bytes. Total persistent cache is exactly **28,499,968 bytes**.
Group mixing does not add resident tables, predictors, dense state, residuals
or model parameters. Source weights and linear computation remain FP16; this
cache accounting is not total GPU memory or a compressed-weight claim.

## Start condition and evidence freeze

Before v9 model execution, require the completed v8 full comparison and a
passing independent CPU full audit. The v8 selected full PPL must be finite
and at least 8.25, with its archived/restored replay, source/table/backend and
cache guards all passing. Bind the outcome/audit and exact v8 selected full
report by hashes. If v8 has PPL **strictly below 8.25**, do not run v9.
An integrity/runtime failure is not evidence of a finite target miss.

Hash this protocol, all new orchestration/auditor sources and every imported
production source before measurements. Keep all older measured scripts and
protocols unchanged. CPU preparatory fixtures and source review may happen
before v8's outcome; they may not measure a model or alter this search using
that outcome. Preserve failed artifacts under fresh paths.

## Eligible groups: fixed before validation

Use the first eight layers from the already completed v8 single-layer TRAIN
calibration ranking, in its existing order:

**0, 3, 6, 1, 4, 2, 7, 10**.

The fixed v8 calibration comparison SHA256 is
`e7fa9ce0f245d3843f447be604bc195186645cb956f9d229cf34b1d05a9bdc49`.

Verify these are exactly the first eight entries of the original v8 ranking;
do not recompute that ranking on new rows or use validation errors to choose
layers. All eight groups of each eligible layer are eligible, group indices
0..7. A group shares its coordinate table across the same 16 heads as before.
This search refines only those 64 group tables; all other parent table entries
remain fixed. Parent selection is bound before this group's experiments.

For each eligible layer in the order above, then each group 0..7, inspect the
four existing v5 group permutations in the fixed table order. Skip every
alternative whose actual 128 uint8 bytes equal the parent group's bytes.
Deduplicate other identical alternative group contents, keeping the first
table label. Record every original label, its bytes/hash and disposition
(`parent_equal`, `duplicate_of`, or retained) before GPU measurements.

Normally each group has three alternatives: 64 × 3 = 192 single-group
interventions. Exact content coincidences can reduce this count; they must be
documented by the deterministic byte comparisons above. Never replace skipped
alternatives with new permutations. Baseline + interventions + restored
baseline therefore uses **at most 194 arms**, in the stated order.

## Single-group TRAIN calibration

Use TRAIN rows **144..151**, all 2048 tokens per row: eight independent windows,
16,376 prediction targets per arm. These rows are disjoint from the v6/v7/v8
search rows. Each arm resets each window exactly as earlier PPL evaluations.

Execute the parent baseline, every retained single-group intervention, then
restored parent. Each intervention replaces only its one [layer,group,:]
table with the existing alternative; all remaining groups use the complete
v8 parent table. Evaluate full-model logits and aggregate raw NLL over all
eight windows. Do not use calibration proxy statistics or additive estimates
instead of actual model PPL.

For each (layer,group), choose the complete finite alternative with minimum
**raw aggregate NLL**; exact NLL ties use the original v5 table order. Retain
it only if its NLL is strictly lower than the parent baseline NLL. If all
alternatives are ineligible, retain no change for that group. Rank retained
changes by `(deltaNLL, numeric layer index, group index, v5 table order)`.
Use deltaNLL only for this final ranking, not for per-group raw-NLL selection.

Propose combined tables in this fixed order: parent baseline, top1, top2,
top4, top8, top16, top32, all negative groups. Cap k at the number of retained
changes, apply at most one selected change per group to the parent, then
deduplicate complete table contents keeping the first label. Export explicit
changes, raw NLL deltas, candidate/table hashes, all original proposal labels
and deduplication decisions. At most eight unique tables are exported.
If no group improves, export only baseline, record the negative family result
and skip redundant screen/full evaluation.

## Disjoint TRAIN screen and final selection

Use TRAIN rows **152..183**, all 2048 tokens: 32 independent windows / 65,504
targets per arm. Execute baseline, each unique combined candidate in export
order, then restored baseline: at most nine arms. The single-group calibration
deltas propose combinations; they do not establish additive gains or score
the combined candidates. Every combination must be evaluated with actual
full-model logits on these separate screen rows.

Select the complete finite candidate with lowest aggregate TRAIN PPL, exact
ties preferring baseline and then export order. No MK guard or other tiebreak.
Freeze the chosen table and bind its screen report/receipt/code/protocol
hashes. If baseline wins, preserve this negative result and do not run a
redundant full confirmation. Only one nonbaseline TRAIN winner may advance.

## Three-arm full confirmation

Evaluate the fixed v8 parent, frozen v9 TRAIN winner and restored v8 parent
on all **130 validation windows / 264,764 targets**. The first and restored
parent must exactly repeat the pinned v8 parent report's per-window NLLs,
aggregate PPL and 128-token hidden/cache probe hashes, with identical actual
cache allocations. Bind the completed v8 parent full comparison/report/audit;
the archive is a replay reference, never a selection input. Record original
S16 PPL 7.334322057221965 separately as comparison context.

Success requires selected full PPL **strictly below 8.25**, all integrity
checks and replay passing, no adapter and exactly 28,499,968 persistent bytes.
There is no separate 1% gate. Do not alter the group pool, parent, rows, k-grid,
tie rules or quantizer in response to full results. If this family misses,
retain the evidence; a distinct subsequent route requires its own prospective
TRAIN-only procedure. A failed family does not prove that repair is impossible.

## Integrity, audit and limitations

Each arm records every window's token hash, raw NLL, target count and PPL;
repeated-reset 128-token hidden and every persistent cache tensor hash; actual
allocation counts; unchanged CPU and GPU table hashes; frozen source identity,
version/gradient checks; pinned backend evidence; and no-adapter/no-MK flags.
Baseline and restored baseline must be complete and finite. A named known
nonfinite intervention/candidate may be excluded with its failed evidence;
arbitrary runtime/compiler errors and any reset/cache/source/table/backend
integrity failure stop the experiment. Finite poor results are not excluded.

An independent CPU audit reconstructs the start condition, fixed group pool,
parent/equal/duplicate byte decisions, every intervention table, raw scoring,
ranking, combinations, selection, full target and exact replays without
importing the new selector implementation or regenerating GPU logits.

The upper bound before full confirmation is 194 × 8 + 9 × 32 = 1,840 model
windows / 3,766,480 prediction targets. Based on v8's roughly three seconds
per eight-window arm on the current host, calibration is approximately ten
minutes, screening approximately two minutes, plus loading/auditing; full
confirmation is another three 130-window arms. These are runtime estimates,
not promised quality gains or hard time limits.

The 192 possible interventions can overfit eight calibration rows. A separate
32-row screen limits that risk but does not eliminate it; group interactions
remain nonadditive, and restricting the search to eight layers can miss other
opportunities. Prior TRAIN/benchmark families have historical exposure; do
not describe them as untouched generalization evidence. Keep the repository
private. No public release is authorized by this protocol.
