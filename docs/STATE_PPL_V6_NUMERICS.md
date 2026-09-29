# v6 numerical checks and diagnostic readout precision

This note records implementation failures found before any v6 model PPL
measurement. It accompanies the [frozen v6 protocol](STATE_PPL_V6_PROTOCOL.md),
whose SHA256 remains
`86d4e8dc85d79c2c867ce15a846c5938893c86d27e67ad174472975e654e9bff`.
The original protocol and all failed test receipts remain unchanged.

The first two failures changed quantized carries and required implementation
repairs. The third exposed a separate limit: random dense diagnostic readouts
were not bitwise equal to packed legacy readouts, although every recorded
decoded carry was bitwise equal. Kernel checks are distinct from model-quality
evidence; no PPL improvement follows from these checks.

## Preserved attempts

| Attempt | First failing check | Isolated difference | Status of original receipt |
|---|---|---|---|
| 1 | `GPU-65token-independent-exact-stored_scale` | INT4 integer code at an exact half tie | Failed, retained |
| 2 | `GPU-65token-independent-exact-clip4_095` | FP16 INT4 scale at a storage midpoint | Failed, retained |
| 3 | `dense-legacy-emulation-(2, 17, 8, 19, 2)` | Three random FP16 readouts; decoded carries exact | Failed, retained |
| 4 | Scalar PTX extraction assertion | Test parser used a source column invalid for the nested probe | Failed, retained |
| 5 | None | All 249 GPU checks pass, including the disclosed bounded diagnostic readout rule | Passed |

Source snapshots, test sources, logs and receipts are stored together under
[attempt1](../reports/v6_source_snapshots/attempt1/),
[attempt2](../reports/v6_source_snapshots/attempt2/) and
[attempt3](../reports/v6_source_snapshots/attempt3/).
Receipt SHA256 values are:

```text
attempt1 c1f5d28d19d28635f030a6d21b1aed280a11a1cd8bc467bda22ae101a7f73b0a
attempt2 1cae8614f2ea55b8443e2243feed28ff9b970cb55f6059ec840d2a8d2ec3b75e
attempt3 a314b445058dd2df1c1a5222100acc48978b4c79d5d3bb5c54273dca878f7a98
```

### Attempt 1: integer selection against the stored scale

The first mismatch occurs at zero-based token 5, the sixth token, with 240
INT4 code differences. Incoming carries, stored scale arrays and that token's
current readout agree. One representative ratio is exactly:

```text
updated value = 0.1327667236328125
stored scale = 0.037933349609375
updated / stored scale = 3.5
observed GPU code = 3
required half-away-from-zero code = 4
```

The negative counterpart chose -3 instead of -4. An identical-cache one-token
replay reproduced the discrepancy, separating it from accumulated recurrence
drift. Approximate division before integer rounding moved the half tie across
its boundary. Later token outputs then diverged because the carried codes
were different. See the [one-step diagnostic](../reports/state_ppl_v6_stored_scale_attempt1_onestep.json)
and [token trace](../reports/state_ppl_v6_stored_scale_attempt1_diagnostic.json).

The new stored-scale branch now uses `tl.div_rn(value, stored_scale)` before
`libdevice.round`, clipping to the fixed INT8/INT4 ranges. A zero stored scale
produces integer zero without division by zero. The legacy candidate continues
to delegate the unchanged original packed codec.

The investigation also corrected an independent CPU reference issue:
`floor(abs(z) + 0.5)` can move an immediately-below-half FP32 value onto the
boundary during the addition. For example, 0.4999999701976776 incorrectly
became one. The reference now computes `floor(abs(z))` and compares the
remaining fractional part against 0.5, then reapplies the sign. Regression
fixtures include exact positive/negative half ties and adjacent values. This
reference repair did not explain away the GPU failure: the robust reference
still required four for the exact 3.5 example.

### Attempt 2: rounding the scale itself

After the quotient repair, the nonclipped stored-scale 65-token fixture passed.
The 0.95 clipping fixture first differed at zero-based token 20. Integer codes,
incoming carry and current-token output agreed; only the stored `s4` scale
differed. The [identical-cache replay](../reports/state_ppl_v6_clip095_attempt2_diagnostic.json)
records:

```text
INT4 absmax             = 0.1148223876953125
FP32 absmax * 0.95      = 0.10908126831054688
FP32 clipped value / 7  = 0.015583038330078125
observed GPU FP16 scale = 0.01558685302734375
FP16 nearest-even scale = 0.0155792236328125
```

The FP32 denominator is an FP16 midpoint. The approximate denominator
expression selected the other stored scale. New-mode denominator formation
now performs the FP32 clipping multiplication followed by correctly rounded
FP32 division (`tl.div_rn`), then the existing 1e-8 floor and FP16 storage
conversion. INT8 denominator formation uses correctly rounded division by
127 as well. Legacy arithmetic remains unchanged.

Attempt 3 subsequently passed all eight 65-token controlled policy checks:
legacy, stored_scale, all three clipping factors, prune_only, quant_only and
legacy_emulation. This establishes those fixtures' exact output, state and
scale agreement; it does not establish arbitrary-input bitwise equivalence.

## Attempt 3: dense diagnostic versus packed legacy readout

The first random geometry is `(batch, length, heads, channels, groups) =
(2, 17, 8, 19, 2)`. In the [recorded diagnostic](../reports/state_ppl_v6_legacy_emulation_attempt3_diagnostic.json):

- All 17 decoded recurrent carries match bitwise.
- Three of 5,168 FP16 readout values differ.
- Maximum absolute output error is `3.814697265625e-6`.
- Relative output L2 error is `5.491804586199578e-6`.
- The first difference is at zero-based token 2.

The auxiliary dense diagnostic stores decoded carry in FP32 so that it does
not add an FP16 rounding step to `integer * stored_FP16_scale`. It is not a
deployable Q3.25 candidate. Its larger actual allocation remains explicitly
counted. Exact persisted decoded carries do not imply exact unquantized
intermediate updates or identical floating-point readout reductions.

### Evidence about the compiler

A bounded follow-up restored the legacy branch's original source statement
order and removed the redundant multiplication by clipping factor one. It
still reproduced the same aggregate discrepancy. This follow-up source SHA
is `76d76577559c8bb8d86907934f2d65282da27e1d1e8e378392eba5809ae34c54`,
separate from attempt 3's source SHA
`0e4bc27a7c6560140c4acf93fcd1f81813d3557314e8feeeae426486e1b4e722`.

Starting both kernels from identical prior carry and current token reproduced
one different value among 304 outputs, with maximum absolute error
`1.1920928955078125e-7`. The retained [compiler evidence](../reports/v6_source_snapshots/legacy_order_probe/compiler_evidence.json)
and [probe receipt](../reports/v6_source_snapshots/legacy_order_probe/receipt.json)
show different generated layouts:

| Reduction | sizePerThread | threadsPerWarp | warpsPerCTA | order |
|---|---|---|---|---|
| Packed legacy | `[1,1]` | `[16,2]` | `[1,4]` | `[0,1]` |
| Dense live 16 | `[1,2]` | `[4,8]` | `[4,1]` | `[1,0]` |
| Dense live 64 / dead 64 | `[1,4]` | `[2,16]` | `[4,1]` | `[1,0]` |

The retained TTGIR and PTX show shared-memory cross-warp reduction in the old
kernel versus a different within-warp shuffle tree in the dense diagnostic,
with different multiply/add fusion. The readout-only difference is consistent
with those reduction and fusion changes. Exact FP32 elementwise updates were
not independently instrumented, so this is evidence about the likely
mechanism, not a proof that every intermediate is identical.

Raw [old TTGIR](../reports/v6_source_snapshots/legacy_order_probe/old.ttgir),
[dense TTGIR](../reports/v6_source_snapshots/legacy_order_probe/dense.ttgir),
[old PTX](../reports/v6_source_snapshots/legacy_order_probe/old.ptx) and
[dense PTX](../reports/v6_source_snapshots/legacy_order_probe/dense.ptx) are
retained; their hashes appear in the compiler evidence receipt.

## Narrow validation refinement

Before any model PPL measurement, the implementation and independent audit
review agreed on the following limited refinement. The random dense
legacy-emulation readout check changes from bitwise equality to an elementwise
arithmetic-error bound. This is an explicit change to that diagnostic
acceptance rule after examining a failed kernel test. The frozen protocol and
original failed receipt remain available; attempt 3 is not relabeled a pass.

The bound is conditional on common input values, bitwise-identical prior
decoded carry and common actual FP32 `dt` and `decay` scalars. These conditions
are checked independently of the readout difference. The retained
[scalar PTX comparison](../reports/v6_source_snapshots/legacy_order_probe/scalar_ptx_equivalence.json)
contains 48 matching emitted arithmetic instructions after bijective register
renaming, with canonical SHA256
`01069e97f7e554c032a6b5940d23b601616f427492140a9761150bfb62ad71e5`.
Its scope covers the FP16 input conversion, bias addition, softplus branch and
decay expression; memory addressing is checked through the source's common
input indexing. Test-only scalar capture supplies the actual scalar values
used in the bound. This does not presume exact common prequantized updates.

### Bound definition

For each output element, let `a` and `b` be the two finite FP16 readouts;
`s_i` is their common prior decoded carry, and `x`, `B_i`, `C_i`, `D`, `dt`
and `decay` are that element's actual inputs/scalars. Set `s_i=0` in the
48 discarded carry coordinates. The following quantities are evaluated in
FP64:

```text
u32 = 2^-24
gamma(n, u) = n*u / (1 - n*u)

T8 = sum over 16 INT8 coordinates:
     abs(C_i) * (abs(s_i * decay) + abs(B_i * dt * x))
T4 = the same sum over 64 INT4 coordinates
Td = the same sum over 48 zero-carry coordinates

F = 4096 * 2^-126 * (1 + abs(x)) * (1 + sum over all128 abs(C_i))
E = gamma(21,u32)*T8 + gamma(69,u32)*T4 + gamma(69,u32)*Td
    + gamma(3,u32)*abs(x*D) + F

r16(0) = 2^-25
r16(y) = max(2^-25, 2^(floor(log2(abs(y))) - 11)) for y != 0
bound_raw = 2*E + r16(a) + r16(b)
```

The final FP64 computation receives an upward guard using
`1 + gamma(1024, 2^-53)` and `nextafter(..., +infinity)`. Acceptance requires
`abs(a-b) <= bound` for **every** output element, not only an aggregate norm.

The operation counts cover recurrence rounding, multiplication by `C`, each
coordinate reduction and the two outer additions: 21 for the 16-coordinate
group, 69 for the 64-coordinate group and the 48 coordinates reduced in a
zero-padded 64-wide group, and three for the direct `x*D` contribution. The
small `F` term conservatively allows FP32 underflow/flush-to-zero effects in
these finite fixtures. Both arithmetic paths contribute `E`, and both final
FP16 conversions contribute a rounding radius. The upward guard protects
against roundoff while evaluating the bound itself.

These constants come from arithmetic depth and precision, rather than fitting
the observed maximum error. This is a conditional diagnostic fixture check,
not a universal bound on model outputs, recurrent error or PPL.

### Gates retained without a tolerance change

- Every recorded decoded carry in legacy emulation must still agree bitwise.
- Controlled fixture output, packed bytes, scales and decoded carry remain exact.
- Delegation to the original legacy codec remains exact.
- Each kernel's reset and full/segmented/tokenwise execution remain exact.
- The actual persistent allocation and source/table/backend guards remain strict.
- Model baseline window NLLs, probe hidden values and cache hashes must replay exactly.

The deployable candidate grid, TRAIN-only choice, 1% full-PPL gate and cache
accounting are unchanged. Diagnostic effects interact through recurrence and
must not be added as independent error contributions or presented as lower
bounds.

## Attempt 4 parser repair and attempt 5 passing checks

Attempt 4 stopped while checking the test-only scalar capture. The PTX
extractor hard-coded a source column that changed when the nested JIT probe
was dedented (`.loc 0` column 41 versus 45). A CPU-only
[re-extraction of the unchanged PTX](../reports/v6_source_snapshots/attempt4/scalar_extraction_diagnosis.json)
confirmed identical scalar arithmetic. The repair selects locations relative
to the preceding softplus expression. It changes the test parser, with no
production codec change or arithmetic-equivalence relaxation. The failed
[receipt and matching sources](../reports/v6_source_snapshots/attempt4/)
remain preserved; receipt SHA256 is
`b61cbed4e14f0b88029899fb4a3cf06fab1d4777ed5c7ae65914cc1097c4bac3`.

The final runtime extractor selects **47 arithmetic instructions**, beginning
with the bias addition and continuing through softplus and decay. For each
random geometry, packed, dense and scalar-capture kernels have identical
canonical instructions, SHA256
`3c69ef6ef2017c7dd2e89ba2ad8cb00a21aab4da664952295eac1663ccc28e92`.
This is the final three-kernel proof used by the bound; the earlier exploratory
48-instruction two-kernel extraction above has a different extraction scope
and is retained as historical evidence.

[Attempt 5](../reports/state_ppl_v6_codec_checks_attempt5.json) passes all
**249 GPU checks**. Every-token decoded carries remain exact in all three
random geometries. The first geometry retains the disclosed three readout
differences; the other two are bitwise equal. Every element satisfies its
bound:

| Geometry `(batch,length,heads,channels,groups)` | Readout differences | Maximum error / elementwise bound | Bound failures |
|---|---:|---:|---:|
| `(2,17,8,19,2)` | 3 / 5,168 | 0.8881203148736804 | 0 |
| `(1,9,6,3,3)` | 0 / 162 | 0 | 0 |
| `(2,5,4,1,1)` | 0 / 40 | 0 | 0 |

The [independent CPU input audit](../reports/state_ppl_v6_inputs_audit.json)
also passes. It reconstructs all **5,370** elementwise bounds and raw packed
carry decoding, with zero bound failures, and independently extracts the
three-kernel scalar PTX equality. It audits recorded GPU evidence without
regenerating model logits.

```text
Passing GPU receipt:
81a65cd0f56e6ad4a7a87fc464abde6509a4315143c38c52c88a67be61e4a54f
Production codec:
76d76577559c8bb8d86907934f2d65282da27e1d1e8e378392eba5809ae34c54
Final checker:
d5feee53129c54548f8ba3c9a579633c41b153ebab57a5e0550d5dda0316d7b1
Independent CPU input audit receipt:
56496a1b4cef996ea8318b030bb36433f8f42e96a03a3028500d5852b6814e13
```

No model PPL has been measured at this premeasurement documentation checkpoint.
Passing these kernel and input checks does not establish arbitrary-input
bitwise dense readouts or model-quality improvement.
