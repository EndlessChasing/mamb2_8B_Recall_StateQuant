# StateQuant / pairnib

**Training-free 3.25-bit tiered state-cache quantization for Mamba-2 decoding.**

StateQuant quantizes the Mamba-2 recurrent **state cache** to a `{0,4,8}` magnitude-tiered,
paired-nibble ("pairnib") layout at **3.25 bits/element** — training-free, no reconstruction —
with W8A8 linear projections and a fused Triton decode-scan kernel that runs directly on the
packed state (it never materializes a full-precision state cache).

On Mamba-2 2.7B, StateQuant **matches Q-Mamba's 8-bit state within +0.27 perplexity** and
**beats its 4-bit-with-reconstruction by 2.4 / 3.2 (WT2 / C4)**, while a single-L4 kernel is
**1.24× faster** than a Q-Mamba-INT4 kernel, uses **21% less state memory**, and raises the
decode batch ceiling to **768 vs 256** for INT8 state (**3×**).

## TL;DR

- **What.** A sub-INT8 SSM state-cache format. Per (layer, head-group), the 128 `d_state`
  positions are ranked by decode-mode magnitude: the top 16 are kept at **INT8**, the next 64
  at **INT4**, and the bottom 48 form a **dead** tier (forced to zero, skipped from storage).
  Payload 3.0 b + two fp16 scales 0.25 b = **3.25 b/elem**. The tier ordering is a *static*
  per-(layer, head-group) permutation, so no per-element tier tag is stored.
- **Why it's needed.** The recurrent state cache dominates Mamba decode-time memory and
  bandwidth; its size caps decode batch and tokens/s per GPU. Naive sub-INT8 quantization
  *collapses* because per-step error recirculates through the scan and accumulates.
- **Why it works, training-free.** The per-head decay concentrates state magnitude into a few
  positions, so the low-magnitude tail is droppable; and a *frozen* tier tracks the per-step
  magnitude-optimal tier at **90% agreement**, so no reconstruction fine-tune is needed.

<p align="center">
  <img src="figures/fig_method.png" width="88%" alt="StateQuant {0,4,8} tiered state format">
</p>

## Headline results (Mamba-2 2.7B, faithful per-step recurrent perplexity)

WikiText-2 (full test, 140×2048) / C4 (GPTQ protocol, 256×2048 seed-0 segments). Lower is better.

| Setting | State b/elem | WikiText-2 | C4 |
|---|---:|---:|---:|
| bf16 (baseline) | 16 | 9.06 | 11.95 |
| Q-Mamba W8A8H8 | 8 | 10.36 | 13.69 |
| Q-Mamba W8A8H6 | 6 | 10.84 | 14.46 |
| Q-Mamba W8A8H4 (+ESR) | 4 | 12.99 | 16.90 |
| **Pairnib, 8 head-groups (deployable)** | **3.25** | **10.63** | **13.65** |
| Pairnib, 1 group (conservative) | 3.25 | 10.80 | 14.04 |
| Pairnib, per-(h,p) (accuracy ceiling) | 3.25 | 10.33 | 13.90 |
| Pairnib, 4.0b (16/88/24) | 4.0 | 11.31 | 15.66 |

At 3.25 bits the deployable pairnib lands **between Q-Mamba's 6-bit and 8-bit state and ties
their 8-bit on C4 (13.65 vs 13.69)**, while beating their 4-bit W8A8H4 by 2.4 / 3.2 — with no
reconstruction. Note the counterintuitive 4.0b row: spending the saved bits back on the state
*hurts*, because the dead tier suppresses recirculating quantization noise rather than merely
saving bits.

<p align="center">
  <img src="figures/fig_pareto.png" width="88%" alt="state bits vs perplexity Pareto">
</p>

### Kernel (measured, single NVIDIA L4, Mamba-2 2.7B geometry, batch 128)

Fused Triton decode-scan; throughput is scan-only (batch × layers / s). Q-Mamba's kernel is
not open-source, so its row is an equal-effort reimplementation of its documented
uniform-INT4-cache + dequantize-before-scan design (not their production kernel).

| Decode kernel @ B=128 | scan tok/s | state GB | batch ceiling |
|---|---:|---:|---:|
| Q-Mamba INT4 (H4), reimplementation | 109,461 | 2.77 | 640 |
| INT8 state (H8) | 119,034 | 5.54 | 256 |
| **Pairnib 3.25b, 8 head-groups (deployable)** | **135,927** | **2.18** | **768** |
| Pairnib 3.25b, 1 group | 140,416 | 2.18 | 768 |

Going from 1 → 8 head-group tier orderings costs only ~3% scan throughput at identical state
memory and batch ceiling (the 8 B/C permutations per step are nearly free versus the state
cache), while buying the +0.17 / +0.39 WT2 / C4 accuracy of the finer granularity.

<p align="center">
  <img src="figures/fig_kernel.png" width="88%" alt="decode batch ceiling and scan throughput">
</p>

## Why it works

All ablations below are single-factor, at the deployable 3.25-bit config (16/64/48, 8-group),
on the same protocols as the headline table (bf16 9.06 / 11.95; per-layer 10.63 / 13.65).

**1. The per-head decay concentrates state magnitude** (`N`=5120 heads):

| quantity | value |
|---|---|
| corr( log\|A\| , log per-head magnitude ) | r = −0.59 (r²=0.35) |
| per-head magnitude dynamic range | 6.7 orders of magnitude |
| between-head share of log-magnitude variance | 73% |
| top-32 / 128 readout-contribution share | 67% (uniform = 25%) |

So the effective information in the nominal 128-position state is far smaller than its size —
which is why 3.25 bits recover near-8-bit-quality state.

**2. A frozen tier is within +0.2 ppl of the per-step optimum**, so no reconstruction is needed:

| tier | WT2 (Δ bf16) | C4 (Δ bf16) |
|---|---:|---:|
| dynamic per-step (near-oracle) | 10.42 (+1.36) | 13.46 (+1.51) |
| fixed / deployed | 10.63 (+1.57) | 13.65 (+1.70) |
| **freezing cost** | **+0.21** | **+0.20** |

The frozen tier tracks the per-step tier closely: **90.2% / 90.0%** per-token agreement (WT2 / C4)
versus a 40.6% chance baseline (Cohen's κ = 0.83), with distant dead↔INT8 swaps ≤ 0.13% — a
mis-frozen position lands in a neighboring tier at worst.

**3. Per-layer calibration is a free refinement, not a necessity.** Sharing one tier across all
64 layers costs +1.58 / +1.98 — a graceful degradation, not a collapse — and per-layer
calibration recovers it at zero inference cost (the tier labels are static).

**4. A dead position drops long-range memory, not the current input.** In the kernel a dead
position is scanned as a constant-0 carried state but still accumulates one step of `dB·x` and
contributes to the readout — so zeroing 48/128 low-magnitude positions is the cheap part of the
budget.

## Repository layout

| Path | What |
|---|---|
| `mamba2_pairnib_eval.py` | Main eval: W8A8 body + pairnib `{0,4,8}` state, WT2/C4, `--state_ngroups` (head-group tier granularity), `--k8/--k4` (tier partition). |
| `mamba2_w8a8_eval.py` | W8A8 body + DSQ / chanadapt / uniform-INT state eval (the Q-Mamba-DSQ comparison points). |
| `qmamba_kernel_bench.py` | Fused decode-scan kernel bench: `pairnib` (g1) / `pairnib_g8` / `qmamba_int4` (= Q-Mamba H4 design) / `int8` / `bf16`. |
| `selective_state_update_pairnib.py` | Fused `{0,4,8}` 3.25b paired-nibble Triton scan kernel (the deployable kernel). |
| `selective_state_update_q.py` | INT8-state fused scan kernel. |
| `selective_state_update_nibble.py` | Uniform-INT4 (4.0b) fused scan kernel (= Q-Mamba H4 design). |
| `selective_state_update_mixed.py` | Mixed `{4,8}` kernel + `build_tier_maps`. |
| `mamba_py/mamba2_min.py` | Vendored pure-PyTorch Mamba-2 (`load_mamba2`; per-step and log-space chunked scan; no `mamba_ssm` needed). |
| `mamba_py/chunked_scan_m2.py` | Log-space chunked-recirc SSD scan (recirc=1 == faithful per-step). |
| `quantize/tiered_quantizer.py` | DSQ / ChannelAdaptive / Uniform / readout state quantizers. |

## Running

Requires a CUDA GPU, `torch`, `triton`, `transformers`, `datasets`. The Mamba-2 checkpoint is
`state-spaces/mamba2-2.7b` (a directory with `config.json` + `pytorch_model.bin`); the vendored
loader does not need `mamba_ssm`.

Accuracy (perplexity), pairnib at 8 head-groups, full WT2 and GPTQ-C4:

```bash
python mamba2_pairnib_eval.py /path/to/mamba2-2.7b \
  --dataset wt2 --arms bf16,w8a8_pairnib --state_ngroups 8 \
  --seqlen 2048 --nwin 0 --batch 8 --calib_ns 8
python mamba2_pairnib_eval.py /path/to/mamba2-2.7b \
  --dataset c4  --arms w8a8_pairnib --state_ngroups 8 \
  --seqlen 2048 --nwin 256 --batch 8 --calib_ns 8
```

Kernel bench (run from the repo root so the kernels import):

```bash
python qmamba_kernel_bench.py
```

## Eval protocol and caveats

- **WikiText-2**: full `wikitext-2-raw-v1` test, gpt-neox-20b tokenizer, `"\n\n"`-joined,
  non-overlapping 2048-token windows. Reproduces the published FP16 9.06.
- **C4**: GPTQ-standard protocol — `allenai/c4` `en` validation shard-0, seed 0, 256 random
  2048-token segments. Reproduces Q-Mamba's FP16 11.95.
- **State eval is faithful per-step (recirc=1)**: the state cache is quantized every decode
  step, so quantization error recirculates through the recurrence (the realistic regime).
- **W8A8 is fake-quant** (functional INT8 weight/activation simulation; measures quantization
  quality, not an INT8 fused-linear kernel). The kernel bench is the separately-measured
  packed-state scan.
- **Baseline is bf16** (numerically reproduces the published 9.06 / 11.95).
- The per-(h,p) row is an **accuracy ceiling only** — not realizable in the fused kernel
  (B/C are per-group). The deployable config is the 8-head-group ordering.

## Citation

```bibtex
@misc{yue2026statequant,
  title  = {StateQuant: A Training-Free 3.25-Bit Tiered State-Cache Format for Mamba-2 Decoding},
  author = {Yue, Kun},
  year   = {2026},
  note   = {Preprint. arXiv id to be added.}
}
```

## References

1. T. Chen et al. *Q-Mamba: Towards more efficient Mamba models via post-training quantization.* Findings of ACL 2025.
2. T. Dao, A. Gu. *Transformers are SSMs: generalized models and efficient algorithms through structured state space duality (Mamba-2).* ICML 2024.
3. E. Frantar et al. *GPTQ: accurate post-training quantization for generative pre-trained transformers.* ICLR 2023 (C4 eval protocol).

## License

[Apache-2.0](LICENSE). Copyright 2026 Kun Yue.
