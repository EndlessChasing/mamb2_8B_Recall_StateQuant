# Third-party notices

## StateQuant

`reference/statequant/selective_state_update_pairnib.py` is preserved from the
user-supplied StateQuant archive, commit
`22156f74edf428fd192948d75a0dabf3cf152d48`, copyright 2026 Kun Yue,
Apache License 2.0. Its license is reproduced in
[`reference/statequant/LICENSE`](../reference/statequant/LICENSE).

`mamba2_recall/state_codec.py` adapts its paired-nibble layout and update rule
to a sequence loop with scalar-head A/dt, group-specific coordinate tables,
explicit owned cache buffers and calibration statistics. It is not a claim
of bitwise equivalence with the general upstream kernel.

## Mamba

The S16 scan arithmetic follows `state-spaces/mamba` selective state update
(Tri Dao and Albert Gu, 2024), Apache License 2.0. Native layers, normalization
and model orchestration are imported from the separately installed `mamba-ssm`
package; the measured environment is recorded in result receipts. The Apache
2.0 license text is available in the reference license linked above.

## Recall and NVIDIA source

The original Recall implementation and the historical adapter used by the v1
experiment were inherited from `EndlessChasing/mamb2_8B_Recall` under GPL-3.0;
see root LICENSE. Later fresh adapters have their own training bindings. In
particular, the V11 release contains the newly trained 1,536-update FP16
adapter, not the historical v1 adapter. New project code and this new adapter
are distributed under GPL-3.0, subject to the separate upstream notices here.

The official [`nvidia/mamba2-8b-3t-4k` checkpoint](https://huggingface.co/nvidia/mamba2-8b-3t-4k/tree/b915550c63ba9359f88f44d1f6a600d85af27302)
remains Apache-2.0 and must be obtained separately. It is not included in this
repository or the adapter/table release bundle. This experiment does not use
Quamba2 weights or its research-only redistribution license.

## WikiText and embedded probe excerpts

Training and PPL evidence uses WikiText by Stephen Merity, Caiming Xiong,
James Bradbury and Richard Socher, described in
[*Pointer Sentinel Mixture Models* (2016)](https://arxiv.org/abs/1609.07843).
The underlying articles were written by Wikipedia contributors. Dataset
provenance is [`Salesforce/wikitext`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3),
configuration `wikitext-2-raw-v1`.

The repository does not contain the full prose corpus. Retained training and
smoke probe evidence does contain small tokenized excerpts, including the
first TRAIN window cropped to 128 and 512 tokens in V11 `probe_evidence.pt`
files. These preserve input IDs for numerical reproduction; tokenization and
cropping are transformations of the dataset text. Raw probe files are in the
tagged repository, not in the adapter/table release bundle.

The [pinned upstream dataset card](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md)
lists [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and
[GFDL](https://www.gnu.org/licenses/fdl-1.3.html) in its metadata, while its
Licensing Information section links to
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). This notice
preserves that upstream attribution and discrepancy; the root GPL-3.0
license does not relicense the embedded dataset text. Consult the pinned
dataset source and its linked Wikipedia provenance for the applicable text
terms and contributor history.
