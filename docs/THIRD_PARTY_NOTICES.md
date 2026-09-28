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

The existing Recall implementation and serialized FP16 adapter are inherited
unchanged from `EndlessChasing/mamb2_8B_Recall` under GPL-3.0; see root LICENSE.
The adapter retains its original training binding. The official NVIDIA
checkpoint remains Apache-2.0 and must be obtained separately. This experiment
does not use Quamba2 weights or its research-only redistribution license.
