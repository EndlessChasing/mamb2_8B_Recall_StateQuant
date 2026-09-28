#!/usr/bin/env python3
"""Synthetic forward/STE-gradient checks; not language-model quality evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mamba2_recall.state_codec import allocate_state, scan, validate_permutation, decode_state
from mamba2_recall.state_training_scan import scan_training, training_workspace_bytes, _carry_checkpoints, TILE


def oracle(inputs, permutation, exact_carries):
    """Independent autograd recurrence replaying deployment carries with masked STE.

    The forward carry values come from one-token deployment codec calls, outside
    this oracle autograd graph. This prevents Torch/Triton exp/FMA differences
    from moving a quantization boundary and changing the evaluation trajectory. Casts to FP16 readouts model output rounding's STE.
    PyTorch and Triton exp/reduction arithmetic can differ, so gradient numerical
    comparisons have declared tolerances. Production forward has a separate
    exact deployment-codec equality gate.
    """
    x, rawdt, A, B, C, D, bias = inputs
    batch, length, heads, dim = x.shape
    groups = B.shape[2]
    group = torch.arange(heads, device=x.device) // (heads // groups)
    order = permutation.long()[group]
    state = torch.zeros((batch, heads, dim, 128), device=x.device, dtype=torch.float32)
    outputs = []
    for t in range(length):
        arg = rawdt[:, t].float() + bias.float()
        dt = torch.where(arg <= 20, torch.log(torch.exp(arg) + 1), arg)
        decay = torch.exp(A * dt)
        bv = B[:, t, group].float().gather(-1, order.expand(batch, -1, -1))
        cv = C[:, t, group].float().gather(-1, order.expand(batch, -1, -1))
        add = (bv * dt[..., None])[:, :, None, :] * x[:, t].float()[..., None]
        # Explicit fused-multiply-add emulation with ordinary autograd gradients.
        h = (state.double() * decay[..., None, None].double() + add.double()).float()
        prod = h * cv[:, :, None, :]
        y = prod[..., :16].sum(-1) + prod[..., 16:80].sum(-1)
        y = y + (prod[..., 80:].sum(-1).double() + x[:, t].double() * D.double()[None, :, None]).float()
        outputs.append(y.half())
        pieces = []
        for a, b, limit in [(0, 16, 127), (16, 80, 7)]:
            part = h[..., a:b]
            den = (part.detach().abs().amax(-1) / limit).clamp_min(1e-8)
            ratio = part.detach() / den[..., None]
            q = (ratio.sign() * (ratio.abs() + .5).floor()).clamp(-limit, limit)
            deq = exact_carries[t][..., a:b]
            pieces.append(part + (deq - part).detach())
        state = torch.cat([*pieces, torch.zeros_like(h[..., 80:])], -1)
    return torch.stack(outputs, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--benchmark-length', type=int, default=256)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    report = {'format': 'STATEQUANT_TRAINING_SCAN_CHECKS_V1', 'passed': False,
              'complete': False, 'checks': [], 'limitations': [
                  'Backward is a masked straight-through approximation, not the derivative of discontinuous quantization.',
                  'A, D, and dt_bias are frozen; base-parameter training is deliberately unsupported.',
                  'Synthetic forward/gradient tests do not establish PPL or MK recovery.',
                  'B/C/group and dt gradients use FP32 atomic accumulation; bitwise backward determinism is not promised.']}
    def record(name, passed, **details):
        row = dict(name=name, passed=bool(passed), **details)
        report['checks'].append(row)
        print(json.dumps(row), flush=True)
        if not passed:
            raise AssertionError(name)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    gen = torch.Generator(device='cuda').manual_seed(2026092817)
    def make(batch, length, heads, dim, groups, scale=.2):
        def rand(shape, factor=scale):
            return (torch.randn(shape, device='cuda', generator=gen) * factor).half()
        tensors = [rand((batch, length, heads, dim)), rand((batch, length, heads)),
                   -torch.exp(torch.randn(heads, device='cuda', generator=gen) * .3),
                   rand((batch, length, groups, 128)), rand((batch, length, groups, 128)),
                   rand((heads,)), torch.full((heads,), -1.5, device='cuda', dtype=torch.float16)]
        permutation = torch.stack([torch.randperm(128, generator=gen, device='cuda')
                                   for _ in range(groups)]).to(torch.uint8)
        validate_permutation(permutation, groups)
        for i in [0, 1, 3, 4]:
            tensors[i].requires_grad_(True)
        return tensors, permutation
    try:
        for batch, length, heads, dim, groups in [(2, 19, 4, 7, 2), (1, 1, 8, 17, 2), (1, 65, 8, 17, 2)]:
            tag = f'B{batch}_L{length}_H{heads}_P{dim}_G{groups}'
            tensors, permutation = make(batch, length, heads, dim, groups)
            state = allocate_state(batch, heads, dim, 128, 'sq3p25', 'cuda')
            deployed, _ = scan(*tensors, state, permutation=permutation)
            cp = torch.empty((batch, (length + 7) // 8, heads, dim, 80), device='cuda')
            _carry_checkpoints[(batch, heads, (dim + TILE - 1) // TILE)](
                tensors[0], tensors[1], tensors[2], tensors[3], tensors[6], permutation, cp,
                length, heads, dim, groups, 8, (length + 7) // 8, TILE, num_warps=4)
            cp_state = allocate_state(batch, heads, dim, 128, 'sq3p25', 'cuda')
            exact_carries = True
            for chunk in range((length + 7) // 8):
                exact_carries &= torch.equal(cp[:, chunk], decode_state(cp_state)[..., :80])
                lo, hi = chunk * 8, min(length, (chunk + 1) * 8)
                pieces = [v[:, lo:hi] if i in [0, 1, 3, 4] else v for i, v in enumerate(tensors)]
                scan(*pieces, cp_state, permutation=permutation)
            record('exact_recomputed_carries_' + tag, exact_carries)
            trained = scan_training(*tensors, permutation, chunk_size=8)
            record('exact_forward_' + tag, torch.equal(trained, deployed))
            gy = torch.randn(trained.shape, device='cuda', generator=gen).half() * .1
            grads = torch.autograd.grad(trained, [tensors[i] for i in [0, 1, 3, 4]], gy)
            reference_tensors = [v.detach().float().clone().requires_grad_(i in [0, 1, 3, 4])
                                 for i, v in enumerate(tensors)]
            replay_state = allocate_state(batch, heads, dim, 128, 'sq3p25', 'cuda')
            exact_carries = []
            for token in range(length):
                replay_inputs = [v[:, token:token+1] if i in [0, 1, 3, 4] else v
                                 for i, v in enumerate(tensors)]
                scan(*replay_inputs, replay_state, permutation=permutation)
                exact_carries.append(decode_state(replay_state))
            reference_y = oracle(reference_tensors, permutation, exact_carries)
            reference_grads = torch.autograd.grad(reference_y,
                [reference_tensors[i] for i in [0, 1, 3, 4]], gy)
            for name, actual, expected in zip(['x', 'dt', 'B', 'C'], grads, reference_grads):
                expected = expected.to(actual.dtype)
                max_abs = (actual.float() - expected.float()).abs().max().item()
                rms = ((actual.float() - expected.float()).square().mean().sqrt() /
                       expected.float().square().mean().sqrt().clamp_min(1e-12)).item()
                close = torch.allclose(actual, expected, atol=2e-5, rtol=.015)
                record('oracle_gradient_' + name + '_' + tag, close,
                       max_absolute_error=max_abs, relative_rms_error=rms, atol=2e-5, rtol=.015)
            grads_alt = torch.autograd.grad(scan_training(*tensors, permutation, chunk_size=13),
                                           [tensors[i] for i in [0, 1, 3, 4]], gy)
            record('chunk_size_invariant_' + tag,
                   all(torch.allclose(a, b, atol=2e-6, rtol=.002) for a, b in zip(grads, grads_alt)))

        # Dead coordinates produce current readout, but never a future carry.
        tensors, permutation = make(1, 5, 2, 3, 1)
        permutation = torch.arange(128, device='cuda', dtype=torch.uint8)[None]
        with torch.no_grad():
            tensors[3][..., :80] = 0
            tensors[4][..., :80] = 0
            tensors[5].zero_()
        y = scan_training(*tensors, permutation, chunk_size=2)
        final_loss = y[:, -1].float().sum()
        dead_grads = torch.autograd.grad(final_loss, [tensors[i] for i in [0, 1, 3, 4]])
        record('dead_coordinates_have_no_future_carry', all(
            torch.count_nonzero(g[:, :-1]).item() == 0 for g in dead_grads))
        record('dead_coordinates_have_current_gradients',
               all(torch.count_nonzero(g[:, -1]).item() > 0 for g in dead_grads))

        # Dequantization scales can underflow; the chosen STE remains explicit.
        tensors, permutation = make(1, 3, 2, 3, 1, scale=0.)
        zero_y = scan_training(*tensors, permutation)
        zero_grads = torch.autograd.grad(zero_y.sum(), [tensors[i] for i in [0, 1, 3, 4]])
        record('zero_inputs_finite', torch.isfinite(zero_y).all() and
               all(torch.isfinite(g).all() for g in zero_grads))

        tensors, permutation = make(1, args.benchmark_length, 128, 64, 8)
        # Warm both forward/backward kernels before measuring.
        warm = scan_training(*tensors, permutation)
        warm.float().square().mean().backward()
        for v in tensors:
            v.grad = None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        start = time.monotonic()
        y = scan_training(*tensors, permutation)
        y.float().square().mean().backward()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        peak_delta = torch.cuda.max_memory_allocated() - before
        record('production_geometry_forward_backward_finite', torch.isfinite(y).all() and
               all(torch.isfinite(tensors[i].grad).all() for i in [0, 1, 3, 4]),
               seconds=elapsed, length=args.benchmark_length, peak_allocated_increment_bytes=peak_delta,
               explicit_workspace=training_workspace_bytes(1, args.benchmark_length))
        report['passed'] = all(row['passed'] for row in report['checks'])
        report['complete'] = True
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        report['source_sha256'] = {str(p.relative_to(Path(__file__).resolve().parents[1])):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),
            Path(__file__).resolve().parents[1] / 'mamba2_recall/state_training_scan.py']}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
