# v6: prioritize unadapted PPL before recall training

**Status: the fixed TRAIN screen and all three full PPL arms are complete.
PPL improves 0.1458%, below the predefined 1% gate; no new Resurface training.** The user explicitly prioritizes PPL
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

## Fixed TRAIN screen

All 24 arms completed: 20 deployable candidates, three diagnostics, and the
restored baseline. Each uses the same 32 windows / 65,504 targets without an
adapter. All 20 candidates are eligible. The restored baseline repeats every
window NLL, reset hidden/cache hash and byte allocation exactly. The measured
evaluation loop takes 268.14 seconds, excluding input validation/model loading.

| Static table | Legacy | Stored scale | Clip 0.95 | Clip 0.90 | Clip 0.80 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Magnitude | 8.699421 | 8.694646 | 12.450415 | 13.680066 | 15.966522 |
| Full readout | 8.566656 | 8.557345 | 11.175291 | 12.200902 | 13.621451 |
| Preserve INT8 | 8.395356 | **8.369217** | 12.153352 | 13.172917 | 15.143694 |
| Preserve retained 80 | 8.750737 | 8.718169 | 11.228348 | 12.285803 | 13.628988 |

The frozen winner is **preserve_int8__stored_scale**: PPL
8.395355795233426 → 8.36921709114913, a **0.31135% decrease**; 18 windows
improve and 14 regress. It changes the scale used to choose codes, with the
same static table and 28,499,968-byte persistent cache. No MK was measured or
used to choose it. No runner-up may replace this winner after full validation.

| Diagnostic on the same TRAIN data | PPL | Actual persistent bytes |
| --- | ---: | ---: |
| Original S16 | 7.325586 | 122,028,032 |
| Prune only: retained 80 in FP16 | 7.609793 | 239,525,888 |
| Quantize only: other 48 retained in FP16 | 7.833298 | 239,525,888 |
| Deployable v5 Q3.25 baseline | 8.395356 | 28,499,968 |

The two ablations deliberately use a larger FP32 backing buffer and are not
Q3.25 deployment candidates. Their effects interact through the recurrent
trajectory; differences must not be added as independent loss contributions.
Every tested clipping factor worsens TRAIN PPL. Repeatedly attenuating the
carried block maxima is a plausible mechanism, but these aggregate scores do
not establish an event-level causal explanation.

See [raw screening comparison](../reports/state_ppl_v6_screen/screen_comparison.json).
The [independent screen audit](../reports/state_ppl_v6_screen_audit.json) passes,
SHA256 `ed47370ca7e8cf666908efe8f821960be4dab7dc4d916b2e3e33bb787c567bb5`.
Selected payload SHA256:
`098930d1af5e5821b277640d236f7607c6428b48d36f7117e2ea4aa87e656303`.

## Full PPL confirmation and stop decision

Each arm evaluates all 130 WikiText-2 validation windows / 264,764 prediction
targets. The frozen TRAIN-selected candidate is the only new full candidate.
No Resurface adapter is installed and no MK examples are generated or scored.

| Configuration | Full PPL | Persistent bytes |
| --- | ---: | ---: |
| Original S16, archived context | 7.334322057221965 | 122,028,032 |
| v5 baseline, newly evaluated | 8.367464892540164 | 28,499,968 |
| v6 stored-scale selected | 8.355268708845868 | 28,499,968 |
| Restored v5 baseline | 8.367464892540164 | 28,499,968 |

The selected candidate improves PPL by **0.145757%** (absolute decrease
0.012196183694296). Aggregate NLL decreases by 386.1941509246826; 69 windows
improve and 61 regress. This is below the frozen requirement of at least 1%
(PPL at most 8.283790243614762). **The meaningful improvement gate fails.**
No claim of a statistically established gain is made from this small aggregate
change. PPL remains **13.9201% above original S16**, and all 130 windows have
higher NLL than that original-state reference.

The new baseline exactly matches the archived v5 report on all 130 window NLLs,
128-token probe hidden values and persistent-cache hashes. Restoring the
baseline repeats those values exactly again. All source/table/backend/finiteness
checks and the byte budget pass. The full evaluation loop takes 128.75 seconds,
excluding input checks and loading. See [raw full comparison](../reports/state_ppl_v6_full/full_comparison.json).
The [independent CPU full audit](../reports/state_ppl_v6_full_audit.json) passes,
SHA256 `d6b3cb2428a4dd815ccf4ccf33b054639233b58ce29f53a96879f487a68cb146`.
Its integrity audit passes while the experiment's 1% quality gate fails; these
are separate outcomes. As in the earlier audits, it reconstructs recorded GPU
evidence and does not rerun model logits.

Per the prospective gate, this experiment stops without training a fresh
Resurface adapter or evaluating v6 MK. The frozen selected artifact and its
small PPL change are preserved for research, while v5 remains the combined
validated reference: **PPL 8.093011486666136, MK 244/384** with its own fresh
adapter. That old endpoint is not evidence for a v6 adapter or v6 recall quality.

## What this result supports

- Choose quantizer candidates using unadapted PPL; treat later recall repair as
  a separate fresh-Resurface experiment after freezing a sufficiently improved
  quantizer. v6 applied that order without an MK guard.
- Choosing codes against the stored scale makes a small measured improvement
  here. It does not solve the remaining PPL gap.
- Shrinking the INT4 range by the tested fixed factors is a poor direction in
  this configuration. This result does not rule out every scale policy.
- A distinct next hypothesis is multi-step, quantization-aware coordinate-tier
  calibration under the actual packed recurrence. It could reuse the same
  static table and cache size, but has not been implemented or measured.
  A second hypothesis is fixed nonuniform INT4 levels that preserve the block
  range, with arithmetic decoding and no extra resident lookup table. Neither
  hypothesis is a validated improvement or part of the completed 20-candidate grid.

The validation corpus and benchmark family have historical exposure; this is
not a new untouched generalization set. Source freezing uses identity/version
and gradient guards plus pinned input hashes, not a post-run full content hash.
The numerical-test refinement and preserved failures are disclosed above.
No weights were quantized or trained, Q8 was skipped, and no public release
was performed.
