# Complete official WT2 test: fixed SQ3.25 / V11 Resurface

This evidence scores the original published `v0.1.0-q325-resurface` model,
without and with its final V11 adapter. The original FP16 NVIDIA weights,
32/32/64 state table, adapter, and measured runtime remain unchanged. No
training, recalibration, candidate/checkpoint selection, or test-dependent
stopping was performed.

| Arm | Pooled-token PPL | Total NLL | Targets | Windows |
|---|---:|---:|---:|---:|
| Without Resurface | 8.113289138152094 | 630067.0492219925 | 300963 | 147 |
| Released V11 Resurface | 7.771302869888767 | 617105.920583725 | 300963 | 147 |

The relative PPL reduction is 4.215137195778762%. All targets are scored once,
including the last window of 1,955 targets. This update contains no new MK
measurement: recall remains the previously published synthetic CONFIRM result.
No S16 test control was run here; the S16 PPL in the historical table is a
validation result and is not a same-split test baseline.

## Evidence scope

`comparison.json`, `without_resurface.json`, and `resurface.json` are the exact
complete reports, including all per-window scores/token hashes and recorded
507-weight/224-adapter identities. `cpu_audit_v1.json` checks token coverage,
pooled NLL/PPL arithmetic, actual state storage receipts, adapter removal and
the bindings to the original published release and measured source.
`inventory.json` binds these reports and the new reproduction sources.
Raw token IDs are deliberately excluded from the public files.

The CPU audit does not recompute GPU logits. Dataset text hashes/fingerprints
are recorded metadata; the supplied token stream is verified in full. You can
regenerate it from the pinned public dataset below.

Training used separate official TRAIN data. The official test text was used
in earlier project experiments, including 2.7B and E8/W5 runs, so this is not
an untouched test for the whole project. The published SQ3.25/V11 checkpoint
was fixed before these scores and not modified based on them. Source-model
pretraining contamination and cross-split duplication have not been audited.
See the [frozen protocol](../../docs/SQ325_V11_WT2_TEST_V1_PROTOCOL.md).

## Reproduce both complete GPU arms

Use the current GitHub source or the updated HF root, including the new runner
and protocol. The release verifier checks every original measured source file;
new evaluation scripts stay outside the sealed `release/` bundle. The test
runner/protocol hashes are pinned in the comparison and inventory.

For a fresh download from HF:

```bash
hf download EndlessChasing/Mamb2_8B_Recall_SQ3.25 \
  --local-dir Mamb2_8B_Recall_SQ3.25
cd Mamb2_8B_Recall_SQ3.25
SQ325_RELEASE_BUNDLE="$PWD/release"
```

From the GitHub source, obtain the unchanged bundle instead:

```bash
mkdir downloads
gh release download v0.1.0-q325-resurface \
  --repo EndlessChasing/mamb2_8B_Recall_StateQuant \
  --pattern mamba2-8b-q325-resurface-v11.tar.gz \
  --pattern SHA256SUMS --dir downloads
(cd downloads && shasum -a 256 -c SHA256SUMS)
tar -xzf downloads/mamba2-8b-q325-resurface-v11.tar.gz -C downloads
SQ325_RELEASE_BUNDLE="$PWD/downloads/mamba2-8b-q325-resurface-v11"
```

Verify the original sealed artifact and unchanged measured code, then download
the separate source checkpoint and tokenizer:

```bash
python3 scripts/release_state_resurface_v11.py --verify "$SQ325_RELEASE_BUNDLE"
hf download nvidia/mamba2-8b-3t-4k \
  release/mp_rank_00/model_optim_rng.pt \
  mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
  --revision b915550c63ba9359f88f44d1f6a600d85af27302 \
  --local-dir models/source
```

Use the Linux/Python/CUDA/Mamba/Triton stack in the original model card and
[release guide](../../docs/STATE_RESURFACE_V11_RELEASE.md). Its exact backend
policy is checked before inference. Weights remain resident FP16, occupying
16,473,999,360 bytes, in addition to state, adapter, scratch and activations.
Create virtual environments outside the sealed bundle.

The output directory must be fresh and outside that bundle:

```bash
python scripts/run_sq325_v11_wt2_test_v1.py \
  --bundle "$SQ325_RELEASE_BUNDLE" --source-dir models/source \
  --out-dir local_wt2_test_replay
python3 scripts/audit_sq325_v11_wt2_test_v1.py \
  --comparison local_wt2_test_replay/comparison.json \
  --bundle "$SQ325_RELEASE_BUNDLE" \
  --out local_wt2_test_replay/cpu_audit_replay.json
```

Both arms must complete all 147 windows. The quantized-state execution loop
requantizes carry each token and resets at each window; native parallel prefill
is not this measured path. Failed or partial outputs are not a full test result.

## CPU re-audit of the published reports

This route reconstructs tokens and audits the supplied scores; it does not
reproduce logits, need a GPU, or load the NVIDIA weight checkpoint. Obtain the
original bundle and current runtime source as above. Download only the source
tokenizer if the checkpoint is not already present:

```bash
hf download nvidia/mamba2-8b-3t-4k \
  mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
  --revision b915550c63ba9359f88f44d1f6a600d85af27302 \
  --local-dir models/source
mkdir local_wt2_published_audit
cp evaluation/wt2_test_v1/comparison.json \
  evaluation/wt2_test_v1/without_resurface.json \
  evaluation/wt2_test_v1/resurface.json local_wt2_published_audit/
python scripts/reconstruct_wt2_test_tokens_v1.py \
  --tokenizer models/source --protected-dir "$SQ325_RELEASE_BUNDLE" \
  --out local_wt2_published_audit/tokens.int64le
python3 scripts/audit_sq325_v11_wt2_test_v1.py \
  --comparison local_wt2_published_audit/comparison.json \
  --bundle "$SQ325_RELEASE_BUNDLE" \
  --out local_wt2_published_audit/cpu_audit_replay.json
```

Token reconstruction needs `datasets==4.8.5` and `sentencepiece==0.2.1`; the
auditor itself uses only Python's standard library. Do not publish generated
token IDs or put outputs/scripts inside the sealed release directory.
Original validation/CONFIRM reports, release tag, runtime and adapter stay intact.
