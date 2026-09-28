# StateQuant pilot results

2026-09-28; original pure Mamba2-8B FP16 base, frozen original Recall adapter.
Q8 was skipped. The experiment compared only S16 and original-coordinate
SQ3.25. Protocol was committed before calibration in `0d13808`; tested
implementation is retained at `3bf088f`. Per-file hashes in the raw receipts
identify the actual remote source used for measurement.

## Result: stop this fixed candidate

| Measurement | S16 + Recall | SQ3.25 + Recall |
| --- | ---: | ---: |
| Pilot validation PPL, 8192 targets | 6.0404637354 | 6.8833874140 |
| Normal multi-key recall | 47/48 (97.92%) | 19/48 (39.58%) |
| Target-removed controls | 0/48 | 0/48 |
| SSM persistent bytes | 117,440,512 | 23,855,104 |
| FP16 convolution bytes | 4,587,520 | 4,587,520 |
| Shared uint8 permutation bytes | 0 | 57,344 |
| Total cache bytes, batch 1 | 122,028,032 | 28,499,968 |
| Cache MiB, batch 1 | 116.375 | 27.1796875 |
| Torch peak allocated bytes | 16,984,986,624 | 16,891,458,560 |
| Torch peak reserved bytes | 17,207,132,160 | 17,230,200,832 |

PPL worsened 13.9546%; recall fell 58.3333 percentage points. Both exceeded
predeclared stopping thresholds (+5% PPL or -10 percentage points MK).
The full 130-window / 768-prompt confirmation evaluation was therefore not run.
This is a negative engineering screen, not a statistical proof against all
state compression or a quality-preserving model release.

The persistent-cache saving is 93,528,064 bytes (89.1953125 MiB), 76.6447%.
The 16.47 GB weight payload remains unchanged. The allocator peak includes
weights and temporary work; allocator reserve is affected by reuse/order and
is not a direct estimate of compressed-cache efficiency. Timing was about
64.24 s for S16 and 67.87 s for SQ3.25 over the complete pilot; these are not
isolated production throughput benchmarks.

## Calibration and correctness

The first 8 x 512 tokens of the pinned TRAIN windows, with adapter active,
determined a static order per layer and each of eight B/C groups. This selected
16 INT8, 64 INT4 and 48 zero-carried coordinates. Those zeroed coordinates
contained 14.2589% of aggregated calibration absolute-state magnitude. That is
an observation about magnitude, not a causal attribution of the quality loss.

Actual SQ storage is 48 code bytes + two FP16 scales per 128-value row. The
runtime owns no dense FP16 SSM shadow. Every token's quantized carry affects
the next token, including during prefill; current readout precedes quantization.
Intermediate decoded state stays in kernel registers. Full and segmented
codec executions matched exactly in the synthetic tests.

- 35/35 codec checks passed, including independent bytes/oracles and exact
  native S16 single-step comparisons at full layer geometry.
- 12/12 CPU convolution controls passed, including real channel count.
- Full-model 32-token native recurrent control: relative RMS 0.0014921,
  maximum absolute difference 0.3125, final-eight argmax matches 8/8. Native
  execution restored bitwise after removing the wrapper. This is a bounded
  numerical control, not full-model bitwise parity.
- All 507 base tensor identities/versions/gradient states were unchanged.
  Full GPU base content was not rehashed after scoring; adapter contents were.
- The adapter file itself was unchanged and remains bound to the original base.

## Numerical and evaluation limits

The SQ kernel is an adaptation of archived StateQuant, not a bitwise clone.
On a 33-token synthetic full-geometry case, its output relative L2 difference
from the archived kernel was 0.00066464; final-state difference 0.00164268;
28/393216 packed-code bytes differed. These bounded differences are disclosed
in the codec receipt. The exact compiler-level cause was not isolated.

S16 versus SQ uses a different FP32 state reduction order (original 128-wide
versus permuted three-part sums). Both quality arms use the same prompt
projections, convolution, adapter and serial recurrent orchestration. Earlier
parallel-SSD Recall PPL 7.0520635 uses a different execution protocol and the
full validation corpus; it must not be compared directly to this 4-window PPL.

No state-aware adapter training, tier sensitivity search, rotation, or Q8 run
was performed. Frozen Recall does not compensate for this compression in the
measured pilot. A different learned/adaptive candidate would require a new
frozen protocol and independent quality evaluation.

## Evidence

- [Protocol](PROTOCOL.md), unchanged SHA256
  `4dde46c89f18a5c68ca29e38a284f47ec9531e1e7f7d3f46f114ecbba2a4af17`.
- [Raw S16](../reports/pilot_v1/pilot_s16.json) and
  [raw SQ3.25](../reports/pilot_v1/pilot_sq3p25.json): window NLL, token hashes,
  all prompt IDs/text, generated tokens, environment and memory receipts.
- [Comparison](../reports/pilot_v1/pilot_comparison.json) and
  [TRAIN calibration receipt](../reports/pilot_v1/calibration.json).
- [Serialized calibration/statistics](../reports/pilot_v1/calibration.pt),
  SHA256 `0029156e5a03d82b8b9945fb29dca1e749780746d7c07beb45ddbf4f76e343c0`.
- [Codec checks](../reports/codec_checks_v4.json) and
  [CPU convolution checks](../reports/state_runtime_conv_cpu_v1.json).
- [Independent receipt audit](../reports/pilot_v1/independent_audit.json): all
  prompt retokenizations and generated-ID decodes matched; PPL was recomputed
  from saved window NLL. There were 28 correct-to-wrong MK pairs and zero
  wrong-to-correct pairs. This audit did not rerun model logits.
- [Stop-guard check](../reports/pilot_full_stop_guard.json): the full evaluation
  entry point rejected this failed pilot before model loading or GPU work.

The GitHub repository contains research code, the original small Recall
adapter, and receipts. No new Hugging Face model has been published.
