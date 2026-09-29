#!/usr/bin/env python3
"""Validate new dense Q3 codec against independent CPU arithmetic and packing.

GPU execution is explicit when this script is run. No model/data downloads.
Random segmented/full and tokenwise/full checks require bitwise equality.
Exact dyadic one-token fixtures test half-away rounding, signed packing, table
mapping and actual FP16-scale effects against an independent CPU oracle.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba2_recall.state_codec_dense3 import (allocate_state, scan, decode_state,
    validate_table, validate_finite_result)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def packed_cpu(q):
    """Independent element-by-element construction of the documented bitstream."""
    out = torch.zeros((*q.shape[:-1], 3, 16), dtype=torch.uint8)
    u = q.to(torch.int64) & 7
    for coordinate in range(128):
        for plane in range(3):
            bit = ((u[..., coordinate] >> plane) & 1).to(torch.uint8)
            out[..., plane, coordinate//8] |= bit << (coordinate % 8)
    return out


def oracle(inputs, table, kind):
    x, dt, A, B, C, D, bias = [v.detach().cpu().float() for v in inputs]
    table = table.detach().cpu()
    batch, length, heads, dim = x.shape
    groups = B.shape[2]
    state = torch.zeros((batch, heads, dim, 128), dtype=torch.float32)
    group_indices = torch.arange(heads) // (heads//groups)
    table_h = table[group_indices]
    outputs = []
    for t in range(length):
        rawdt = dt[:, t] + bias
        delta = torch.where(rawdt <= 20., torch.log(torch.exp(rawdt)+1.), rawdt)
        decay = torch.exp(A[None, :] * delta)
        b = B[:, t][:, group_indices]
        c = C[:, t][:, group_indices]
        if kind == 'permutation':
            indices = table_h.long()[None].expand(batch, heads, 128)
            b = b.gather(-1, indices)
            c = c.gather(-1, indices)
        else:
            k = table_h.to(torch.int32)[None]
            b = torch.ldexp(b, -k)
            c = torch.ldexp(c, k)
        state = state * decay[:, :, None, None] + (b * delta[:, :, None])[:, :, None, :] * x[:, t, :, :, None]
        y = (state * c[:, :, None, :]).sum(-1) + x[:, t]*D[None, :, None]
        outputs.append(y.half())
        blocks = state.reshape(batch, heads, dim, 2, 64)
        den = (blocks.abs().amax(-1)/3.).clamp_min(1e-8)
        scaled = blocks / den[..., None]
        q = (scaled.sign()*torch.floor(scaled.abs()+.5)).clamp(-3, 3).to(torch.int32)
        scales = den.half()
        state = (q.float()*scales.float()[..., None]).reshape(batch, heads, dim, 128)
    return torch.stack(outputs, 1), packed_cpu(q.reshape_as(state)), scales, state


def inputs_for(batch=2, length=17, heads=8, dim=19, groups=2, *, device='cuda'):
    g = torch.Generator(device='cpu').manual_seed(20260929)
    x = (torch.randn(batch, length, heads, dim, generator=g)*.08).half()
    dt = (torch.randn(batch, length, heads, generator=g)*.2-2).half()
    A = -torch.rand(heads, generator=g).float()-.1
    B = (torch.randn(batch, length, groups, 128, generator=g)*.2).half()
    C = (torch.randn(batch, length, groups, 128, generator=g)*.2).half()
    D = (torch.randn(heads, generator=g)*.1).half()
    bias = torch.zeros(heads, dtype=torch.float16)
    return [v.to(device) for v in (x, dt, A, B, C, D, bias)]


def run_partition(inputs, table, kind, pieces):
    x = inputs[0]; batch, length, heads, dim = x.shape
    if sum(pieces) != length:
        raise ValueError('Partition does not cover sequence')
    state = allocate_state(batch, heads, dim, 128, x.device)
    outputs = []; start = 0
    for count in pieces:
        batch_inputs = [v[:, start:start+count] if i in (0, 1, 3, 4) else v
                        for i, v in enumerate(inputs)]
        output = scan(*batch_inputs, state, table, kind)
        validate_finite_result(state, output)
        outputs.append(output)
        start += count
    return torch.cat(outputs, 1), state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {'format': 'MAMBA2_DENSE3_CODEC_CHECK_V1', 'complete': False,
              'started_unix': time.time(), 'checks': [], 'code_sha256': {
                  rel: digest(ROOT/rel) for rel in ('mamba2_recall/state_codec_dense3.py',
                     'mamba2_recall/state_quant_dense3.py', 'scripts/check_state_codec_dense3.py')}}
    def check(name, condition, **facts):
        if not condition:
            raise AssertionError(name)
        report['checks'].append({'name': name, 'pass': True, **facts})
        print('[PASS] '+name, flush=True)
    try:
        torch.set_num_threads(4)
        torch.manual_seed(20260929)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError('This validation explicitly requires a CUDA device')
        report['gpu'] = torch.cuda.get_device_name()
        # Shape edge cases include odd P, nonmultiples of16, batch2 and grouped heads.
        for geometry in ((2, 17, 8, 19, 2), (1, 9, 6, 3, 3), (2, 5, 4, 1, 1)):
            batch, length, heads, dim, groups = geometry
            values = inputs_for(*geometry)
            generator = torch.Generator(device='cpu').manual_seed(8)
            for kind in ('permutation', 'equalizer'):
                table = (torch.stack([torch.randperm(128, generator=generator) for _ in range(groups)]).to(torch.uint8)
                    if kind == 'permutation' else (torch.arange(groups*128).reshape(groups, 128)%5-2).to(torch.int8))
                table = table.cuda(); validate_table(table, groups, kind)
                full, fs = run_partition(values, table, kind, [length])
                split, ss = run_partition(values, table, kind, [1, 3, length-4])
                token, ts = run_partition(values, table, kind, [1]*length)
                check(f'partition-bitwise-{geometry}-{kind}', torch.equal(full, split) and torch.equal(full, token)
                      and all(torch.equal(fs.tensors[k], ss.tensors[k]) and torch.equal(fs.tensors[k], ts.tensors[k])
                              for k in fs.tensors), tokens=length)
                check(f'payload-52B-{geometry}-{kind}', fs.nbytes == batch*heads*dim*52,
                      payload_bytes=fs.nbytes)
                # Current readout is before quantization, so a one-token random
                # probe has no recirculating boundary amplification in the oracle.
                single = [v[:, :1] if i in (0, 1, 3, 4) else v for i, v in enumerate(values)]
                actual, _ = run_partition(single, table, kind, [1])
                expected, _, _, _ = oracle(single, table, kind)
                delta = actual.cpu().float()-expected.float()
                relative = float(torch.linalg.vector_norm(delta)/(torch.linalg.vector_norm(expected.float())+1e-12))
                check(f'random-readout-oracle-{geometry}-{kind}', relative <= .002,
                      relative_l2=relative, bound=.002)
        # Exact fixtures use A=0, dt=32 and dyadic arithmetic. All seven levels,
        # negative values, and half-integer ties are exercised independently.
        fixture = torch.tensor([-3., -2.5, -2., -1.5, -1., -.5, 0., .5, 1., 1.5, 2., 2.5, 3.])
        for name in ('ties', 'negative', 'zero', 'scale_rounding', 'scale_underflow', 'large_finite'):
            values = inputs_for(2, 1, 4, 17, 2)
            x, dt, A, B, C, D, bias = values
            x.fill_(1); dt.zero_(); A.zero_(); D.zero_(); bias.fill_(32)
            wanted = fixture.repeat(10)[:128]
            if name == 'negative': wanted = -wanted.abs()
            if name == 'zero': wanted.zero_()
            if name == 'scale_rounding': wanted = wanted.sign()*wanted.abs().round(); wanted[0] = 3.001953125
            if name == 'scale_underflow': wanted = wanted.sign()*2**-19
            if name == 'large_finite': wanted = wanted.sign()*131072.
            B.copy_((wanted/32).half().cuda()[None, None, None])
            if name == 'scale_underflow': x.fill_(2**-8)
            # Zero C avoids overflowing the FP16 readout in scale stress tests.
            C.fill_(0 if name in ('large_finite', 'scale_underflow') else 2**-8)
            for kind in ('permutation', 'equalizer'):
                table = (torch.arange(127, -1, -1).expand(2, 128).to(torch.uint8)
                         if kind == 'permutation' else torch.zeros(2, 128, dtype=torch.int8)).cuda()
                expected, codes, scales, decoded = oracle(values, table, kind)
                actual, state = run_partition(values, table, kind, [1])
                check(f'exact-CPU-oracle-{name}-{kind}', torch.equal(actual.cpu(), expected)
                      and torch.equal(state.tensors['codes'].cpu(), codes)
                      and torch.equal(state.tensors['scales'].cpu(), scales)
                      and torch.equal(decode_state(state).cpu(), decoded),
                      zero_scales=int((scales==0).sum()), total_scales=scales.numel())
        # Both extreme allowed exponents are checked on finite representable inputs.
        values = inputs_for(2, 1, 4, 17, 2)
        values[0].fill_(2**-8); values[1].zero_(); values[2].zero_()
        values[3].fill_(2**-8); values[4].fill_(2**-8)
        values[5].zero_(); values[6].fill_(32)
        table = torch.tensor([-8]*64+[8]*64, dtype=torch.int8).expand(2, 128).contiguous().cuda()
        expected, codes, scales, decoded = oracle(values, table, 'equalizer')
        actual, state = run_partition(values, table, 'equalizer', [1])
        check('equalizer-extremes-minus8-plus8', torch.equal(actual.cpu(), expected)
              and torch.equal(state.tensors['codes'].cpu(), codes)
              and torch.equal(state.tensors['scales'].cpu(), scales)
              and torch.equal(decode_state(state).cpu(), decoded), exponent_min=-8, exponent_max=8)
        # Finite inputs can overflow the FP16 scale after write amplification.
        values[0].fill_(1); values[3].fill_(4096); values[4].zero_()
        table.fill_(-8)
        x = values[0]; bad = allocate_state(2, 4, 17, 128, x.device)
        bad_output = scan(*values, bad, table, 'equalizer')
        rejected = False
        try: validate_finite_result(bad, bad_output)
        except FloatingPointError: rejected = True
        check('reject-nonfinite-amplified-FP16-scale', rejected,
              finite_input_tensors=all(bool(torch.isfinite(v).all()) for v in values))
        # Conversely, tiny write amplitude can underflow stored scales to zero.
        check('zero-scale-underflow-is-representable', any(
            c.get('zero_scales', 0)>0 for c in report['checks'] if 'scale_underflow' in c['name']))
        zero = allocate_state(1, 2, 3, 128, 'cuda')
        nonfinite = torch.full((1, 1, 2, 3), float('inf'), device='cuda', dtype=torch.float16)
        rejected = False
        try: validate_finite_result(zero, nonfinite)
        except FloatingPointError: rejected = True
        check('reject-nonfinite-readout', rejected)
        production = allocate_state(1, 128, 64, 128, 'cuda')
        persistent = 56*production.nbytes + 56*10240*4*2 + 56*8*128
        check('production-cache-budget', persistent == 28499968,
              row_bytes=52, ssm_bytes=56*production.nbytes, table_bytes=57344,
              total_bytes=persistent, includes_convolution=True)
        invalid = torch.zeros(2, 128, dtype=torch.int8); invalid[0, 0]=9
        rejected = False
        try: validate_table(invalid, 2, 'equalizer')
        except ValueError: rejected = True
        check('reject-exponent-outside-frozen-range', rejected)
        report['complete'] = True
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        report['finished_unix'] = time.time()
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
