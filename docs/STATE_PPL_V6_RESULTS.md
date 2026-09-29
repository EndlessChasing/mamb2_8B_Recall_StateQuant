# v6: prioritize unadapted PPL before recall training

**Status: protocol frozen; all 249 GPU kernel checks and the independent CPU
input audit pass; model screen pending.
No v6 quality result is measured yet.** The user explicitly prioritizes PPL
optimization and assigns MK recovery to a later Resurface stage.

Protocol SHA256:
`86d4e8dc85d79c2c867ce15a846c5938893c86d27e67ad174472975e654e9bff`.
See [the frozen protocol](STATE_PPL_V6_PROTOCOL.md).

## What changes from v5

- Compare the four existing static tables with five fixed scale policies:
  legacy, integer rounding against the actually stored FP16 scale, and the
  latter with INT4 clipping factors 0.95, 0.90 or 0.80.
- Use 32 complete 2048-token TRAIN windows / 65,504 targets for selection.
- Select lowest PPL, preferring the v5 baseline on exact ties. No MK selection
  guard, tie break or new MK scoring is used.
- All deployable candidates retain 16 INT8 / 64 INT4 / 48 zero coordinates,
  two FP16 scales, and exactly 28,499,968 bytes of persistent batch-one cache.
- Separately measure pruning-only and quantization-only diagnostics. Their
  larger buffers are explicitly counted; these arms cannot become candidates.

Original source weights remain frozen FP16. No Resurface adapter is loaded
while optimizing the quantizer. Historical v5 artifacts are read only to
validate input provenance; their MK fields never select the v6 winner.

## Decision sequence

1. Pass kernel, format, finite-value and independent numerical checks.
2. Run fixed TRAIN selection and independently audit every raw result.
3. Freeze the one selected nonbaseline candidate and run full unadapted PPL.
4. Require at least 1% lower PPL than v5 unadapted, with the same cache and exact
   baseline replays. A failure ends this experiment and preserves v5.
5. Only after that gate, train a fresh Resurface adapter with the frozen codec
   and validate full PPL/MK plus adapter-removed replay.

The baseline is [v5 preserve-INT8 without an adapter](STATE_FIRST_V5_RESULTS.md):
full PPL **8.367464892540164**, MK **46/384**. Its separately trained v5 endpoint
is **8.093011486666136 / 244/384**. These are previous measured references,
not v6 outcomes or forecasts. Original S16 is **7.334322057221965**.

## Implementation checks before model measurement

The first GPU codec test attempt found a real stored-scale boundary issue.
All one-token controlled fixtures passed, but the 65-token stored-scale oracle
check failed. The first mismatch occurs on the sixth token: previous carries
and stored scale arrays agree, but 240 INT4 codes differ. In an exact-tie
example, `0.1327667236328125 / 0.037933349609375 = 3.5`; approximate GPU division
caused the subsequent rounding to choose three instead of the specified four.
An identical-cache one-token replay reproduced the mismatch. Current-token
readout still agrees at that step; the different carry affects later tokens.

The repair uses correctly rounded FP32 division for stored-scale code selection
while legacy continues to call the unchanged baseline implementation. A separate
CPU reference issue was also identified: `floor(abs(z) + 0.5)` can misround an
FP32 value immediately below a half boundary because the addition itself rounds.
The reference now compares its fractional part with 0.5.
Exact positive/negative half ties and adjacent values are regression cases.

The failed receipt, diagnostic replays and matching source snapshots are
preserved. No model PPL measurement preceded these checks, and no quality
selection criterion or codec-test tolerance is relaxed to accept the mismatch.

After that repair, the nonclipped stored-scale 65-token case passes exactly.
The second test attempt isolates another boundary in the clipped denominator:
at token 21 of the 0.95 fixture, codes, incoming carry and current readout agree,
but the INT4 FP16 scale differs. For absmax 0.1148223876953125, the CPU FP32
clipped value is 0.10908126831054688; division by seven gives exactly
0.015583038330078125, halfway between two FP16 values. CPU tie-to-even selects
0.0155792236328125, while the approximate GPU expression selects
0.01558685302734375. Identical-cache replay confirms this scale-only origin.
Correctly rounded FP32 division is therefore also used when forming new-mode
denominators. The frozen legacy path remains unchanged. The second failure and
its source/diagnostic evidence are retained as well.

The third attempt passes all controlled 65-token cases. Its additional random
dense-versus-packed diagnostic check finds three differing FP16 readout values
out of 5,168 (maximum absolute difference 3.814697265625e-6), while all 17
persisted decoded carries are bitwise identical. Reordering source statements
to match the legacy kernel does not remove this difference. Retained TTGIR/PTX
shows different reduction layouts, summation trees and multiply/add fusion
between packed and dense storage.

This extra random-output bitwise requirement is stronger than the frozen
protocol's controlled-equivalence requirement. Before any model measurement,
an explicit numerical review defined a recurrence-inclusive floating-point
error bound for that diagnostic readout only. All controlled exact tests,
random carry equality, delegated legacy equality and partition equality remain
required. This is a disclosed validation-scope refinement, not a repair of
the observed output difference. The failed attempt remains preserved; passing
random diagnostic output must not be described as bitwise legacy equivalence.
See [numerical evidence and validation scope](STATE_PPL_V6_NUMERICS.md).

Attempt 4 stopped in the added test-only PTX extractor because it hard-coded a
source column that differs for a dedented nested probe. The extractor now uses
the preceding softplus source location; its arithmetic-equivalence criterion
is unchanged. That failure and source are preserved. Attempt 5 passes all 249
checks. All three random fixtures have exact every-token decoded carry and
zero elementwise bound failures. The first retains the disclosed three output
differences (maximum error/bound ratio 0.8881203); the other two are exact.
Actual scalar PTX arithmetic matches the packed, dense and capture kernels in
each fixture. This does not establish arbitrary-input bitwise dense readouts.

Passing receipt SHA256:
`81a65cd0f56e6ad4a7a87fc464abde6509a4315143c38c52c88a67be61e4a54f`.
Production codec SHA256:
`76d76577559c8bb8d86907934f2d65282da27e1d1e8e378392eba5809ae34c54`.
Checker SHA256:
`d5feee53129c54548f8ba3c9a579633c41b153ebab57a5e0550d5dda0316d7b1`.

The independent CPU input audit passes with CUDA uninitialized. It reconstructs
the fixed tables/provenance, decodes every recorded packed carry, checks all
5,370 random-fixture output bounds and verifies the current kernels' scalar
PTX. It audits recorded GPU evidence rather than independently running the
model. Audit SHA256:
`56496a1b4cef996ea8318b030bb36433f8f42e96a03a3028500d5852b6814e13`.
