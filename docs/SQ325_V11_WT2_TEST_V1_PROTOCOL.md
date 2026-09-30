# Fixed published SQ3.25 / V11 Resurface: full WT2 test

Freeze the public v0.1.0-q325-resurface checkpoint, selected 32/32/64 table,
original FP16 NVIDIA weights and final V11 soft adapter. Verify the sealed
release bundle and all measured runtime source files before loading. Evaluate
both adapter-free and adapted arms completely; no training, calibration,
candidate selection or test-dependent stopping is permitted.

Use Salesforce/wikitext, wikitext-2-raw-v1, official test at revision
b08601e04326c79dfdd32d625aee71d232d685c3, two-newline text joining and the
pinned NVIDIA SentencePiece tokenizer without automatic BOS/EOS. The frozen
stream has 300,964 tokens, SHA256 of little-endian int64 bytes
5b82bd46e833e77fcfc0af62bafeaac62e70e68cfdf214d375f0b7b132d4b608.
Score all 300,963 targets in 147 windows of at most 2,048 targets, resetting
state and overlapping by one context token. Include the final partial window.

Use the unchanged run_state_ppl_v10.evaluate PPL loop, StatePPLQuantV10
32_32_64 codec and published pinned Triton backend. Requantize carry each
token; FP16 compute, FP32 logits, 64-token cross entropy chunks. Compute
PPL as exp(sum NLL / sum targets). Native parallel prefill is not this
quantized-state execution path.

Record per-window NLL/count/token hashes, actual 52-byte state rows and
28,499,968-byte cache, all 507 frozen base hashes and all 224 adapter hashes,
backend and repeated-reset probes. Verify adapter removal restores the exact
unadapted hidden output and cache. Preserve the original sealed bundle and
validation results. A separate CPU audit checks token coverage, hashes and
score arithmetic; it does not recompute GPU logits.

The official test corpus was used in earlier project experiments, including
2.7B and E8/W5 runs. This is a fixed published checkpoint evaluation, not an
untouched test for the entire project. Base pretraining contamination and
cross-split duplication are not audited. Historical MK CONFIRM results remain
separate from this language-model test; no MK retraining or rerun occurs here.
