# Full precision Mamba2-8B Resurface control

Declared 2026-09-28 before fitting this repository's adapter. The purpose is to
measure the effect of the same readout adapter and training budget on NVIDIA's
uncompressed pure Mamba2-8B. The E8/W5 experiment is a historical comparator,
not a training dependency. All metrics are reported at the actually measured
precision and protocol, without assuming equality to native Megatron BF16.

## Frozen model and intervention

Load `nvidia/mamba2-8b-3t-4k` revision
`b915550c63ba9359f88f44d1f6a600d85af27302`, checkpoint SHA256
`47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb`
and tokenizer SHA256
`5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09`.
Cast the 507 original BF16 tensors to FP16 in the native state-spaces Mamba2
runtime; freeze all 8,236,999,680 base parameters. Verify strict source tensor
mapping, model geometry and no shared embedding/output storage.

Use the same post-D adapter as the compressed experiment at all 56 native
gated RMSNorm inputs: `y + sigmoid(w dot u+b) * g*(V@y)` across 128 heads.
Its 224 tensors contain 1,154,104 FP32 trainable master parameters, exported
as FP16 for inference. Initialize V=0, g=1, w=0, b=-4; use soft sigmoid at
train and eval, no task switch, EMA, or new recurrent cache. The adapter is
external to the frozen base; check its identities and absence of base grads.

## Training

Reuse the *same exact data generator contract and seeds* as the compressed
adapter: 1,536 TRAIN numeric bindings over three templates and N=16/64,
disjoint six-digit key/value intervals; full 256K CE on every answer suffix
token. Use seed 2026092803 and `torch.randperm(1536)` once for example order.
Each step pairs one MK example with a 512-token segment from the already
prepared 448 WikiText-2 TRAIN windows. Their historical manifest and file
SHA256 are pinned by the train command. Heldout and validation windows are
not included in training. Reusing this fixed TRAIN text gives the control the
same training exposures as the compressed arm.

Use a second, independently loaded **uncompressed FP16** model as the frozen
prose teacher. This is the direct counterpart to the compressed experiment's
teacher, which was its own unadapted compressed base. The fixed per-step loss is
`MK answer CE + 0.5 prose CE + 0.5 KL(own unadapted teacher || student) + 3 C`,
where C is the same prose-only router closure, with budget 0.006 and excess
coefficient 10. Temperature 1, 511 prose targets per step, 64-token staged
head chunks, and one optimizer update after both task gradients.

AdamW FP32 masters: V/g LR 1e-4, w/b LR 3e-4, betas (.9,.999), eps 1e-8,
weight decay 0, clip norm 1. Multiplicative schedule
`0.1+0.9*(1+cos(pi*j/1535))/2` at successful index j. Native FP16 forward,
block checkpointing, gradient scale 1024 with growth interval 2000. Overflow
retries the exact same pair without optimizer/master update, at most 8 retries
and 1,544 total attempts. Run 1,536 **successful** updates; the final step is
the only candidate. Small GPU smokes are discarded before formal training.

## Evaluation and interpretation

After training, restore the actual serialized FP16 adapter onto the same
uncompressed base. Evaluate baseline, adapter enabled and restored baseline
in a single process, with identical tokenizer, MK prompts and full WikiText-2
validation windows. For MK use full 256K greedy generation up to 12 tokens,
first standalone six-digit number, normal and target-removed controls, native
prefill plus recurrent decode with fresh FP16 cache. For PPL use all 130
nonoverlapping reset windows and all 264,764 next-token targets. Record all
raw scores, differences and file hashes; no early selection or task switch.

The earlier compressed arm's independent CONFIRM set has now been observed and
uses the same three template families. Running this source control on it gives
a matched historical comparison, **not a newly untouched holdout**. The
WikiText-2 validation text also informed earlier development. We will not
promote a source+adapter result to a general recall conclusion without new
templates, longer distances and a truly untouched corpus.

Report all four observed arms explicitly: source FP16, source+adapter,
compressed/readapted, compressed/readapted+adapter. The source and compressed
teachers differ by construction. Report PPL and recall together; any cross-arm
claim remains limited by shared test protocols and the historical numerical
replay discrepancy in the compressed project.
