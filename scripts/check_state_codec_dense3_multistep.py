#!/usr/bin/env python3
"""Targeted multi-token dense Q3 oracle diagnostic; no quality measurements.

The CPU oracle owns its own bit packing, decoding, recurrence and quantization.
Controlled dyadic fixtures require exact CPU/GPU output/code/scale equality.
Random recurrence comparisons report every observed discrepancy: they are
numerical diagnostics because CPU exp/log/FMA paths differ from Triton, not
an unconditional exactness gate. No existing codec or test is modified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEED = 2026092807
SOFTPLUS_OUTPUT_REL_L2 = .002
SOFTPLUS_CARRY_REL_L2 = .002


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unpack_cpu(codes, scales):
    shape = (*codes.shape[:-2], 128)
    decoded = torch.empty(shape, dtype=torch.float32)
    integers = torch.empty(shape, dtype=torch.int32)
    for coordinate in range(128):
        unsigned = torch.zeros(codes.shape[:-2], dtype=torch.int32)
        for plane in range(3):
            byte = codes[..., plane, coordinate//8].to(torch.int32)
            unsigned += ((byte >> (coordinate % 8)) & 1) << plane
        signed = torch.where(unsigned < 4, unsigned, unsigned-8)
        integers[..., coordinate] = signed
        decoded[..., coordinate] = signed.float()*scales[..., coordinate//64].float()
    return decoded, integers


def quantize_pack_cpu(updated):
    codes = torch.zeros((*updated.shape[:-1], 3, 16), dtype=torch.uint8)
    scales = torch.empty((*updated.shape[:-1], 2), dtype=torch.float16)
    integers = torch.empty(updated.shape, dtype=torch.int32)
    for block in range(2):
        lo, hi = block*64, (block+1)*64
        values = updated[..., lo:hi]
        denominator = (values.abs().max(-1).values/3.).clamp_min(1e-8)
        z = values/denominator[..., None]
        integers[..., lo:hi] = (z.sign()*torch.floor(z.abs()+.5)).clamp(-3, 3).to(torch.int32)
        scales[..., block] = denominator.to(torch.float16)
    for coordinate in range(128):
        unsigned = integers[..., coordinate] & 7
        for plane in range(3):
            bit = ((unsigned >> plane) & 1).to(torch.uint8)
            codes[..., plane, coordinate//8] |= bit << (coordinate % 8)
    return codes, scales


def cpu_reference(inputs, table, kind):
    x, rawdt, A, B, C, D, bias = [v.detach().cpu().float() for v in inputs]
    table = table.detach().cpu()
    batch, length, heads, dim = x.shape
    groups = B.shape[2]
    group_index = torch.arange(heads)//(heads//groups)
    head_table = table[group_index]
    codes = torch.zeros((batch, heads, dim, 3, 16), dtype=torch.uint8)
    scales = torch.zeros((batch, heads, dim, 2), dtype=torch.float16)
    outputs, code_history, scale_history, delta_history = [], [], [], []
    for t in range(length):
        # Decode bytes anew each token, rather than carrying an unpacked oracle state.
        previous, _ = unpack_cpu(codes, scales)
        rdt = rawdt[:, t] + bias[None]
        delta = torch.where(rdt <= 20., torch.log(torch.exp(rdt)+1.), rdt)
        decay = torch.exp(A[None]*delta)
        b, c = B[:, t][:, group_index], C[:, t][:, group_index]
        if kind == 'permutation':
            indices = head_table.long()[None].expand(batch, heads, 128)
            b, c = b.gather(-1, indices), c.gather(-1, indices)
        elif kind == 'equalizer':
            k = head_table.int()[None]
            b, c = torch.ldexp(b, -k), torch.ldexp(c, k)
        else:
            raise ValueError(kind)
        drive = (b*delta[..., None])[:, :, None, :]*x[:, t, :, :, None]
        updated = previous*decay[:, :, None, None] + drive
        y = (updated*c[:, :, None, :]).sum(-1) + x[:, t]*D[None, :, None]
        outputs.append(y.half())
        codes, scales = quantize_pack_cpu(updated)
        code_history.append(codes.clone()); scale_history.append(scales.clone())
        delta_history.append(delta.clone())
    return {'output': torch.stack(outputs, 1), 'codes': torch.stack(code_history),
            'scales': torch.stack(scale_history), 'delta': torch.stack(delta_history)}


def controlled(kind, length=65, target_softplus_half=False):
    """Four signed drives (+,+,-,-) accumulate then exactly cancel every cycle.

    Strict fixture uses rawdt32 in the linear softplus branch, A0, and B/64
    for an exactly0.5 effective drive. The target-softplus0.5 companion uses
    inverse softplus bias and is approximate in FP32; nearby raw FP32 values
    do not necessarily map to exactly0.5 through log(exp(raw)+1).
    """
    batch, heads, dim, groups = 2, 6, 5, 3
    table = (torch.stack([torch.roll(torch.arange(127, -1, -1), 11*g) for g in range(groups)]).to(torch.uint8)
        if kind == 'permutation' else
        torch.tensor([-8, -4, 0, 4, 8], dtype=torch.int8).repeat(77)[:groups*128].reshape(groups, 128))
    x = torch.empty(batch, length, heads, dim, dtype=torch.float16)
    dt = torch.zeros(batch, length, heads, dtype=torch.float16)
    A = torch.zeros(heads, dtype=torch.float32)
    bias = torch.full((heads,), math.log(math.expm1(.5)) if target_softplus_half else 32., dtype=torch.float32)
    B = torch.empty(batch, length, groups, 128, dtype=torch.float16)
    C = torch.empty_like(B)
    D = torch.tensor([(-1.)**h/8 for h in range(heads)], dtype=torch.float16)
    coordinates = torch.arange(128)
    for t in range(length):
        cycle, within = divmod(t, 4)
        direction = (1., 1., -1., -1.)[within]
        for b in range(batch):
            for h in range(heads):
                for p in range(dim):
                    x[b, t, h, p] = (-1.)**(b+h+p)*2.**((cycle+b+h+p)%3-1)
            for g in range(groups):
                desired_b = ((coordinates+g+cycle+b)%7-3).float()*direction/4
                if not target_softplus_half:
                    desired_b = desired_b/64
                desired_c = ((coordinates*3+g+t+b)%5-2).float()/8
                if kind == 'permutation':
                    B[b, t, g, table[g].long()] = desired_b.half()
                    C[b, t, g, table[g].long()] = desired_c.half()
                else:
                    B[b, t, g] = torch.ldexp(desired_b, table[g].int()).half()
                    C[b, t, g] = torch.ldexp(desired_c, -table[g].int()).half()
    return [x, dt, A, B, C, D, bias], table


def realistic(kind, length=65):
    generator = torch.Generator(device='cpu').manual_seed(SEED)
    batch, heads, dim, groups = 2, 8, 7, 2
    x = (torch.randn(batch, length, heads, dim, generator=generator)*.3).half()
    dt = (torch.randn(batch, length, heads, generator=generator)*.4-2).half()
    A = (-torch.rand(heads, generator=generator)*2-.01).float()
    B = (torch.randn(batch, length, groups, 128, generator=generator)*.2).half()
    C = (torch.randn(batch, length, groups, 128, generator=generator)*.2).half()
    D = (torch.randn(heads, generator=generator)*.1).half()
    bias = (torch.randn(heads, generator=generator)*.3).half()
    table = (torch.stack([torch.randperm(128, generator=generator) for _ in range(groups)]).to(torch.uint8)
        if kind == 'permutation' else torch.randint(-3, 4, (groups, 128), generator=generator).to(torch.int8))
    return [x, dt, A, B, C, D, bias], table


def numbers(actual, expected):
    a, e = actual.float(), expected.float()
    difference = a-e
    return {'exact': bool(torch.equal(actual, expected)),
            'different_elements': int((actual != expected).sum()), 'elements': actual.numel(),
            'max_absolute_error': float(difference.abs().max()),
            'relative_l2_error': float(torch.linalg.vector_norm(difference)/(torch.linalg.vector_norm(e)+1e-30)),
            'finite': bool(torch.isfinite(a).all() and torch.isfinite(e).all())}


def compare_trace(actual, reference):
    per_token = []
    for t in range(reference['codes'].shape[0]):
        decoded, q = unpack_cpu(actual['codes'][t], actual['scales'][t])
        refdecoded, refq = unpack_cpu(reference['codes'][t], reference['scales'][t])
        per_token.append({'token': t, 'output': numbers(actual['output'][:, t], reference['output'][:, t]),
            'packed_bytes': numbers(actual['codes'][t], reference['codes'][t]),
            'scales': numbers(actual['scales'][t], reference['scales'][t]),
            'integer_codes': numbers(q, refq), 'decoded_carry': numbers(decoded, refdecoded)})
    fields = ('output', 'packed_bytes', 'scales', 'integer_codes', 'decoded_carry')
    first = {field: next((r['token'] for r in per_token if not r[field]['exact']), None) for field in fields}
    return {'all_exact': all(v is None for v in first.values()), 'first_difference_token': first,
            'output_all_tokens': numbers(actual['output'], reference['output']),
            'final_decoded_carry': per_token[-1]['decoded_carry'], 'per_token': per_token}


def gpu_runs(inputs, table, kind):
    # Delayed imports keep --cpu-only independent of Triton/CUDA execution.
    from mamba2_recall.state_codec_dense3 import allocate_state, scan, validate_finite_result
    values, gpu_table = [v.cuda() for v in inputs], table.cuda()
    b, length, h, p = inputs[0].shape
    def run(pieces, capture=False):
        state = allocate_state(b, h, p, 128, 'cuda')
        outputs, code_history, scale_history = [], [], []
        start = 0
        for count in pieces:
            part = [v[:, start:start+count] if i in (0, 1, 3, 4) else v for i, v in enumerate(values)]
            y = scan(*part, state, gpu_table, kind)
            validate_finite_result(state, y)
            outputs.append(y.cpu())
            if capture:
                code_history.append(state.tensors['codes'].cpu().clone())
                scale_history.append(state.tensors['scales'].cpu().clone())
            start += count
        result = {'output': torch.cat(outputs, 1),
                  'final_codes': state.tensors['codes'].cpu(), 'final_scales': state.tensors['scales'].cpu()}
        if capture:
            result.update(codes=torch.stack(code_history), scales=torch.stack(scale_history))
        return result
    tokenwise = run([1]*length, capture=True)
    full = run([length]); segmented = run([1, 7, 16, length-24])
    partition = {label: {key: bool(torch.equal(value[key], tokenwise[key]))
                       for key in ('output', 'final_codes', 'final_scales')}
                 for label, value in (('full', full), ('segmented', segmented))}
    return tokenwise, partition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    report = {'format': 'MAMBA2_DENSE3_MULTISTEP_ORACLE_V1', 'complete': False,
        'mode': 'cpu_preparation' if args.cpu_only else 'gpu_diagnostic', 'started_unix': time.time(),
        'seed': SEED, 'code_sha256': {rel: sha(ROOT/rel) for rel in (
            'scripts/check_state_codec_dense3_multistep.py', 'mamba2_recall/state_codec_dense3.py',
            'mamba2_recall/state_quant_dense3.py')}, 'cases': [],
        'scope': 'Strict65-token dyadic effective0.5-drive packed CPU recurrence gate; '
                 'target-softplus0.5 companion uses fixed numerical tolerances; random65-token comparison '
                 'reports all differences and is descriptive, not an exactness or quality gate',
        'criteria_frozen_before_gpu': {'controlled': 'Every output, packed byte and FP16 scale exact every token',
             'partition': 'Full and1/7/16/remainder segmented outputs and final caches exactly match tokenwise',
             'target_softplus_half': {'output_relative_l2_bound': SOFTPLUS_OUTPUT_REL_L2,
                  'every_token_decoded_carry_relative_l2_bound': SOFTPLUS_CARRY_REL_L2,
                  'packed_code_and_scale_disagreements_reported_without_exact_requirement': True},
             'random': 'Report all errors; finite outputs/scales required; no post-result tolerance changes'}}
    phase = 'setup'
    try:
        if not args.cpu_only:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            if not torch.cuda.is_available(): raise RuntimeError('CUDA required for actual kernel diagnostic')
            report['gpu'] = torch.cuda.get_device_name()
        for family, make in (('controlled', controlled),
                             ('target_softplus_half', lambda kind: controlled(kind, target_softplus_half=True)),
                             ('random', realistic)):
            for kind in ('permutation', 'equalizer'):
                phase = 'cpu_fixture_construction'
                inputs, table = make(kind)
                reference = cpu_reference(inputs, table, kind)
                row = {'family': family, 'kind': kind, 'geometry': list(inputs[0].shape),
                    'groups': inputs[3].shape[2],
                    'cpu_delta_min': float(reference['delta'].min()),
                    'cpu_delta_max': float(reference['delta'].max()),
                    'cpu_nonzero_output_elements': int((reference['output']!=0).sum()),
                    'cpu_nonzero_final_packed_bytes': int((reference['codes'][-1]!=0).sum()),
                    'input_tensors_finite': all(bool(torch.isfinite(v).all()) for v in inputs)}
                report['cases'].append(row)
                if family == 'controlled':
                    row['cpu_delta_exact32'] = bool((reference['delta']==32.).all())
                    row['effective_drive_delta'] = .5
                    row['B_divisor'] = 64
                    row['decay_exact1'] = bool((inputs[2]==0).all())
                    row['cpu_every_fourth_carry_zero'] = bool((reference['codes'][3::4]==0).all())
                    if not row['cpu_delta_exact32'] or not row['cpu_every_fourth_carry_zero']:
                        raise AssertionError('Controlled fixture failed its declared CPU construction')
                if family == 'target_softplus_half':
                    row['raw_bias'] = float(inputs[-1][0])
                    row['target_delta'] = .5
                    row['delta_approximation_declared'] = True
                if not args.cpu_only:
                    phase = 'gpu_comparison'
                    actual, partition = gpu_runs(inputs, table, kind)
                    row.update(comparison=compare_trace(actual, reference), partition=partition)
                    if family == 'target_softplus_half':
                        row['fixed_tolerance_gate_passed'] = (
                            row['comparison']['output_all_tokens']['relative_l2_error'] <= SOFTPLUS_OUTPUT_REL_L2
                            and all(v['decoded_carry']['relative_l2_error'] <= SOFTPLUS_CARRY_REL_L2
                                    for v in row['comparison']['per_token']))
                print(json.dumps({'family': family, 'kind': kind,
                    'cpu_delta_min': row['cpu_delta_min'], 'cpu_delta_max': row['cpu_delta_max'],
                    'comparison': None if args.cpu_only else {k:v for k,v in row['comparison'].items() if k!='per_token'}}), flush=True)
        report['cuda_initialized'] = torch.cuda.is_initialized()
        if args.cpu_only:
            if report['cuda_initialized']: raise AssertionError('CPU preparation initialized CUDA')
        else:
            report['controlled_exact_gate_passed'] = all(r['comparison']['all_exact'] for r in report['cases'] if r['family']=='controlled')
            report['partition_exact_gate_passed'] = all(all(all(v.values()) for v in r['partition'].values()) for r in report['cases'])
            report['target_softplus_half_tolerance_gate_passed'] = all(
                r['fixed_tolerance_gate_passed'] for r in report['cases'] if r['family']=='target_softplus_half')
            report['random_all_exact'] = all(r['comparison']['all_exact'] for r in report['cases'] if r['family']=='random')
            if (not report['controlled_exact_gate_passed'] or not report['partition_exact_gate_passed']
                    or not report['target_softplus_half_tolerance_gate_passed']):
                raise AssertionError('Declared controlled or partition equality gate failed; preserve evidence')
        report['complete'] = True
    except BaseException as error:
        report['error'] = repr(error)
        report['failure_stage'] = phase
        raise
    finally:
        report['finished_unix'] = time.time()
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__': main()
