#!/usr/bin/env python3
"""Independent packed-cache, recurrence, causality and native-control checks."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mamba2_recall.state_codec import (allocate_state, scan, decode_state,
                                      encode_state_, validate_permutation, RP)


def numpy_pack(value):
    """Independent scalar NumPy byte encoder; no production helper calls."""
    v = np.asarray(value, dtype=np.float32)
    result = {}
    codes = []
    for suffix, part, limit in [('8', v[..., :16], 127), ('4', v[..., 16:80], 7)]:
        scale = np.maximum(np.max(np.abs(part), axis=-1) / np.float32(limit), np.float32(1e-8))
        divided = part / scale[..., None]
        q = np.clip(np.sign(divided) * np.floor(np.abs(divided) + np.float32(.5)), -limit, limit).astype(np.int32)
        result['s' + suffix] = scale.astype(np.float16)
        codes.append(q)
    q8, q4 = codes
    for name, q in [('lo', q8), ('hi', q8 >> 4), ('q4', q4)]:
        pairs = (q & 15).astype(np.uint8)
        packed = np.empty((*pairs.shape[:-1], pairs.shape[-1] // 2), dtype=np.uint8)
        for n in range(packed.shape[-1]):
            packed[..., n] = pairs[..., 2*n] | (pairs[..., 2*n+1] << 4)
        result[name] = packed
    deq = np.zeros(v.shape, dtype=np.float32)
    deq[..., :16] = q8 * result['s8'][..., None].astype(np.float32)
    deq[..., 16:80] = q4 * result['s4'][..., None].astype(np.float32)
    return result, deq


def numpy_recurrence(inputs, mode, permutation):
    """CPU float32 arithmetic oracle, with explicit fused add emulation.

    exp/log and reduction implementations may differ from CUDA. Comparisons
    therefore disclose numerical tolerances, while pure byte codec checks and
    segmentation/causality/native-step gates require exact equality.
    """
    x, rawdt, A, B, C, D, bias = [t.detach().cpu().float().numpy() for t in inputs]
    batch, length, heads, dim = x.shape
    groups = B.shape[2]
    hgroup = np.arange(heads) // (heads // groups)
    state = np.zeros((batch, heads, dim, 128), dtype=np.float32)
    result = np.empty_like(x)
    stats = np.zeros_like(state)
    pp = permutation.detach().cpu().numpy().astype(np.int64) if permutation is not None else None
    for t in range(length):
        arg = rawdt[:, t] + bias
        dt = np.where(arg <= 20, np.log(np.exp(arg) + np.float32(1)), arg).astype(np.float32)
        decay = np.exp(A * dt).astype(np.float32)
        b = B[:, t, hgroup]
        c = C[:, t, hgroup]
        if mode == 'sq3p25':
            indices = np.broadcast_to(pp[hgroup][None], b.shape)
            b = np.take_along_axis(b, indices, -1)
            c = np.take_along_axis(c, indices, -1)
        add = (b * dt[..., None])[:, :, None, :] * x[:, t, :, :, None]
        # FP32 product plus fused FP32 multiply-add, same expression ordering.
        state = (state.astype(np.float64) * decay[..., None, None].astype(np.float64)
                 + add.astype(np.float64)).astype(np.float32)
        prod = state * c[:, :, None, :]
        if mode == 'sq3p25':
            y = np.sum(prod[..., :16], -1, dtype=np.float32)
            y = y + np.sum(prod[..., 16:80], -1, dtype=np.float32)
            last = np.sum(prod[..., 80:], -1, dtype=np.float32)
            last = (x[:, t].astype(np.float64) * D[None, :, None].astype(np.float64)
                    + last.astype(np.float64)).astype(np.float32)
            y = y + last
        else:
            y = np.sum(prod, -1, dtype=np.float32)
            y = (x[:, t].astype(np.float64) * D[None, :, None].astype(np.float64)
                 + y.astype(np.float64)).astype(np.float32)
        result[:, t] = y.astype(np.float16).astype(np.float32)
        if mode == 'sq3p25':
            _, state = numpy_pack(state)
        else:
            state = state.astype(np.float16).astype(np.float32)
            stats += np.abs(state)
    return result, state, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {'format': 'STATEQUANT_PACKED_CODEC_CHECKS_V1', 'complete': False,
              'passed': False, 'checks': [], 'limitations': [
                  'Synthetic kernel checks, not full-model PPL or MK.',
                  'CPU exp/log/reductions compared with declared tolerance; exact native and segmented gates separate.',
                  'Real model input and full integration checks remain separate.']}
    start = time.monotonic()
    def record(name, passed, **details):
        report['checks'].append(dict(name=name, passed=bool(passed), **details))
        print(json.dumps(report['checks'][-1]), flush=True)
        if not passed:
            raise AssertionError(name)
    def same(a, b):
        return all(torch.equal(a.tensors[k], b.tensors[k]) for k in a.tensors)
    try:
        # All signed representable codes, exact halves, zero, FP16 subnormals.
        v = np.zeros((2, 2, 5, 128), dtype=np.float32)
        for row, arr in enumerate(v.reshape(-1, 128)):
            arr[:16] = np.array([-127, -126, -64, -16, -8, -1, -.5, 0, .5, 1, 7, 8, 15, 63, 126, 127], np.float32)
            arr[16:80] = np.resize(np.array([-7, -6, -3, -1, -.5, 0, .5, 1, 3, 6, 7], np.float32), 64)
            arr *= [1., 0., 2**-16, 2**-26, 4096.][row % 5]
            arr[80:] = 123. # Dead carry always absent.
        for device in ['cpu', 'cuda']:
            state = allocate_state(2, 2, 5, 128, 'sq3p25', device)
            encode_state_(state, torch.from_numpy(v).to(device))
            oracle, decoded = numpy_pack(v)
            record('independent_numpy_bytes_' + device,
                   all(np.array_equal(t.cpu().numpy(), oracle[k]) for k, t in state.tensors.items()))
            record('independent_numpy_decode_' + device,
                   np.array_equal(decode_state(state).cpu().numpy(), decoded))
            record('compact_bytes_' + device, state.nbytes == 2 * 2 * 5 * 52,
                   actual=state.nbytes, expected=2 * 2 * 5 * 52)
            underflow = state.tensors['s8'].reshape(-1)[3].item()
            record('underflow_preserved_' + device, underflow == 0., scale=underflow)
        for bad in [float('nan'), float('inf')]:
            state = allocate_state(1, 1, 1, 128, 'sq3p25', 'cpu')
            try:
                encode_state_(state, torch.full(state.shape, bad))
                rejected = False
            except ValueError:
                rejected = True
            record('reject_nonfinite_' + str(bad), rejected)
        wrong = torch.zeros((2, 128), dtype=torch.uint8)
        try:
            validate_permutation(wrong, 2)
            rejected = False
        except ValueError:
            rejected = True
        record('reject_nonbijective_permutation', rejected)

        from mamba_ssm.ops.triton.selective_state_update import selective_state_update
        upstream_path = Path(__file__).resolve().parents[1] / 'reference/statequant/selective_state_update_pairnib.py'
        upstream_spec = importlib.util.spec_from_file_location('archived_pairnib', upstream_path)
        upstream = importlib.util.module_from_spec(upstream_spec)
        upstream_spec.loader.exec_module(upstream)
        report['upstream_kernel_sha256'] = hashlib.sha256(upstream_path.read_bytes()).hexdigest()
        generator = torch.Generator(device='cuda').manual_seed(2026092807)
        for batch, heads, dim, groups, length in [(2, 8, 17, 2, 19), (1, 128, 64, 8, 33)]:
            tag = f'B{batch}_H{heads}_P{dim}_G{groups}_L{length}'
            def rand(shape, scale=1.):
                return (torch.randn(shape, generator=generator, device='cuda') * scale).half()
            x = rand((batch, length, heads, dim), .3)
            dt = rand((batch, length, heads), .3)
            A = -torch.exp(torch.randn((heads,), generator=generator, device='cuda') * .4)
            B = rand((batch, length, groups, 128), .15)
            C = rand((batch, length, groups, 128), .15)
            D = rand((heads,), .1)
            db = torch.full((heads,), -2., device='cuda', dtype=torch.float16)
            inputs = (x, dt, A, B, C, D, db)
            permutation = torch.stack([torch.randperm(128, generator=generator, device='cuda')
                                       for _ in range(groups)]).to(torch.uint8)
            validate_permutation(permutation, groups)
            for mode in ['s16', 'sq3p25']:
                perm = permutation if mode == 'sq3p25' else None
                state = allocate_state(batch, heads, dim, 128, mode, 'cuda')
                whole, stats = scan(*inputs, state, permutation=perm, collect_stats=mode == 's16')
                segmented = allocate_state(batch, heads, dim, 128, mode, 'cuda')
                pieces = []
                for low, high in [(0, 1), (1, 7), (7, length)]:
                    y, _ = scan(x[:, low:high], dt[:, low:high], A, B[:, low:high], C[:, low:high],
                                D, db, segmented, permutation=perm)
                    pieces.append(y)
                record('segmented_exact_' + mode + '_' + tag,
                       torch.equal(whole, torch.cat(pieces, 1)) and same(state, segmented))
                stepstate = allocate_state(batch, heads, dim, 128, mode, 'cuda')
                rows = []
                for t in range(length):
                    y, _ = scan(x[:, t:t+1], dt[:, t:t+1], A, B[:, t:t+1], C[:, t:t+1],
                                D, db, stepstate, permutation=perm)
                    rows.append(y)
                record('every_step_exact_' + mode + '_' + tag,
                       torch.equal(whole, torch.cat(rows, 1)) and same(state, stepstate))
                changed_x = x.clone(); changed_x[:, 7:] *= 3
                changed_state = allocate_state(batch, heads, dim, 128, mode, 'cuda')
                changed, _ = scan(changed_x, dt, A, B, C, D, db, changed_state, permutation=perm)
                record('causal_prefix_exact_' + mode + '_' + tag, torch.equal(whole[:, :7], changed[:, :7]))
                oy, os, ostats = numpy_recurrence(inputs, mode, perm)
                dy = whole.float().cpu().numpy() - oy
                ds = decode_state(state).cpu().numpy() - os
                relative_y = float(np.linalg.norm(dy.ravel()) / (np.linalg.norm(oy.ravel()) + 1e-20))
                relative_s = float(np.linalg.norm(ds.ravel()) / (np.linalg.norm(os.ravel()) + 1e-20))
                # Long low-bit trajectories can cross a discrete rounding
                # boundary after CPU/CUDA exp/log differences. Exact CUDA
                # upstream packing/recurrence is separately required below.
                tolerance = .005 if mode == 'sq3p25' else .002
                record('independent_numpy_recurrence_' + mode + '_' + tag,
                       relative_y < tolerance and relative_s < tolerance,
                       output_relative_l2=relative_y, state_relative_l2=relative_s,
                       output_max_abs=float(np.max(np.abs(dy))), state_max_abs=float(np.max(np.abs(ds))),
                       tolerance_relative_l2=tolerance)
                if mode == 'sq3p25':
                    original = allocate_state(batch, heads, dim, 128, mode, 'cuda')
                    original_out = []
                    idx = permutation.long()[None].expand(batch, groups, 128)
                    for t in range(length):
                        u = original.tensors
                        y = upstream.pairnib_scan(u['lo'], u['hi'], u['q4'], u['s8'], u['s4'],
                            x[:, t], dt[:, t, :, None].expand(batch, heads, dim),
                            A[:, None, None].expand(heads, dim, 128),
                            B[:, t].gather(-1, idx), C[:, t].gather(-1, idx),
                            D[:, None].expand(heads, dim), db[:, None].expand(heads, dim))
                        original_out.append(y.half())
                    original_out = torch.stack(original_out, 1)
                    mismatches = {k: int((state.tensors[k] != original.tensors[k]).sum()) for k in state.tensors}
                    # Our scalar-head decay/softplus broadcasts differ from the
                    # upstream general A[H,P,N],dt[B,H,P] kernel. Compiler FMA
                    # contraction can differ; do not claim bitwise identity.
                    oy_gpu = original_out.float()
                    os_gpu = decode_state(original)
                    yrel = float(torch.linalg.vector_norm(whole.float()-oy_gpu) / torch.linalg.vector_norm(oy_gpu).clamp_min(1e-20))
                    srel = float(torch.linalg.vector_norm(decode_state(state)-os_gpu) / torch.linalg.vector_norm(os_gpu).clamp_min(1e-20))
                    code_count = sum(state.tensors[k].numel() for k in ('lo', 'hi', 'q4'))
                    code_diff = sum(mismatches[k] for k in ('lo', 'hi', 'q4'))
                    record('upstream_packed_step_bounded_' + tag,
                           yrel < .002 and srel < .005 and code_diff / code_count < .001,
                           output_max_abs=float((whole-original_out).abs().max()),
                           output_relative_l2=yrel, state_relative_l2=srel,
                           packed_mismatch_counts=mismatches, packed_code_mismatch_fraction=code_diff / code_count,
                           bitwise_equal=torch.equal(whole, original_out) and same(state, original),
                           tolerances={'output_relative_l2': .002, 'state_relative_l2': .005,
                                       'packed_code_mismatch_fraction': .001})
                if mode == 's16':
                    native_state = torch.zeros_like(state.tensors['state'])
                    native_out = []; native_stats = torch.zeros_like(native_state, dtype=torch.float32)
                    for t in range(length):
                        y = selective_state_update(native_state, x[:, t], dt[:, t, :, None].expand(batch, heads, dim),
                            A[:, None, None].expand(heads, dim, 128), B[:, t], C[:, t],
                            D[:, None].expand(heads, dim), dt_bias=db[:, None].expand(heads, dim), dt_softplus=True)
                        native_out.append(y)
                        native_stats += native_state.float().abs()
                    native_out = torch.stack(native_out, 1)
                    record('native_step_exact_' + tag,
                           torch.equal(whole, native_out) and torch.equal(state.tensors['state'], native_state),
                           output_max_abs=float((whole-native_out).abs().max()),
                           state_max_abs=float((state.tensors['state']-native_state).abs().max()))
                    wantstats = torch.zeros_like(stats)
                    for tile in range(math.ceil(dim / RP)):
                        wantstats[:, :, tile] = native_stats[:, :, tile*RP:(tile+1)*RP].sum(2)
                    err = float((stats-wantstats).abs().max())
                    record('calibration_native_stats_' + tag, torch.allclose(stats, wantstats, rtol=2e-6, atol=1e-6),
                           max_abs=err, shape=list(stats.shape))
        # Exact algebraic one-step oracle exercises half-away ties IN THE GPU
        # KERNEL, independent of exp/log differences: dt=32, x=1/32, A=-0.
        tie_state = allocate_state(1, 2, 3, 128, 'sq3p25', 'cuda')
        tie_b = torch.zeros((1, 1, 1, 128), device='cuda', dtype=torch.float16)
        tie_b[..., :16] = torch.tensor([-127, -126, -64, -16, -8, -1, -.5, 0, .5, 1, 7, 8, 15, 63, 126, 127], device='cuda')
        tie_b[..., 16:80] = torch.tensor([-7, -.5, .5, 7], device='cuda').repeat(16)
        tie_b[..., 80:] = .25
        pp = torch.arange(128, device='cuda', dtype=torch.uint8)[None]
        tie_y, _ = scan(torch.full((1, 1, 2, 3), 1/32, device='cuda', dtype=torch.float16),
                       torch.zeros((1, 1, 2), device='cuda', dtype=torch.float16),
                       -torch.zeros(2, device='cuda'), tie_b, torch.ones_like(tie_b),
                       torch.zeros(2, device='cuda', dtype=torch.float16),
                       torch.full((2,), 32., device='cuda', dtype=torch.float16), tie_state, permutation=pp)
        expected_dense = tie_b[:, 0, :, None, :].expand(1, 2, 3, 128).cpu().float().numpy()
        expected_codes, expected_decode = numpy_pack(expected_dense)
        record('gpu_half_away_ties_and_dead_readout_exact',
               all(np.array_equal(t.cpu().numpy(), expected_codes[k]) for k, t in tie_state.tensors.items())
               and np.array_equal(decode_state(tie_state).cpu().numpy(), expected_decode)
               and torch.equal(tie_y[:, 0], tie_b.float().sum(-1).half().reshape(1, 1, 1).expand(1, 2, 3)))
        # Exact all-zero update also exercises FP16 underflow scale behavior.
        zstate = allocate_state(1, 8, 3, 128, 'sq3p25', 'cuda')
        zeros = torch.zeros((1, 4, 8, 3), device='cuda', dtype=torch.float16)
        pp = torch.arange(128, device='cuda', dtype=torch.uint8)[None].expand(2, 128).contiguous()
        y, _ = scan(zeros, zeros[..., 0], -torch.ones(8, device='cuda'),
                    torch.zeros((1, 4, 2, 128), device='cuda', dtype=torch.float16),
                    torch.zeros((1, 4, 2, 128), device='cuda', dtype=torch.float16),
                    torch.zeros(8, device='cuda', dtype=torch.float16),
                    torch.zeros(8, device='cuda', dtype=torch.float16), zstate, permutation=pp)
        record('zero_update_exact', not bool(y.any()) and not bool(decode_state(zstate).any())
               and all(not bool(t.any()) for t in zstate.tensors.values()))
        report.update(complete=True, passed=True)
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - start
        report['torch'] = torch.__version__
        import triton
        report['triton'] = triton.__version__
        report['gpu'] = torch.cuda.get_device_name(0)
        report['source_sha256'] = {str(p.relative_to(Path(__file__).resolve().parents[1])):
                                   hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [Path(__file__).resolve(), Path(__file__).resolve().parents[1] / 'mamba2_recall/state_codec.py']}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
