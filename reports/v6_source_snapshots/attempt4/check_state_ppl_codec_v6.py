#!/usr/bin/env python3
"""Independent CPU arithmetic/packing versus actual v6 GPU recurrence.

Strict dyadic fixtures require bitwise output, packed bytes and FP16 scales.
Random tests require exact kernel partition replay and every-token dense carry.
Dense-emulation readout is checked against an analytical FP32/FP16 error bound;
the independent CPU one-token comparison retains its fixed0.002 relative-L2 bound.
CPU-only preparation has a distinct receipt format and cannot pass a GPU gate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import inspect
import re
from pathlib import Path
import sys
import time

import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROTOCOL_SHA = '86d4e8dc85d79c2c867ce15a846c5938893c86d27e67ad174472975e654e9bff'
POLICIES = [('legacy', 'legacy', 1., None), ('stored_scale', 'stored_scale', 1., None),
    ('clip4_095', 'stored_scale', .95, None), ('clip4_090', 'stored_scale', .90, None),
    ('clip4_080', 'stored_scale', .80, None), ('prune_only', 'legacy', 1., 'prune_only'),
    ('quant_only', 'legacy', 1., 'quant_only'), ('legacy_emulation', 'legacy', 1., 'legacy_emulation')]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def half_away_cpu(value):
    # Avoid floor(abs(z)+.5): that addition can round nextafter(.5,0) up to1.
    absolute = value.abs()
    floor = absolute.floor()
    return value.sign()*(floor+((absolute-floor) >= .5).to(value.dtype))


def pack_cpu(q):
    result = torch.zeros((*q.shape[:-1], q.shape[-1]//2), dtype=torch.uint8)
    for coordinate in range(q.shape[-1]):
        nibble = (q[..., coordinate].to(torch.int64) & 15).to(torch.uint8)
        result[..., coordinate//2] |= nibble << (4*(coordinate%2))
    return result


def decode_cpu(buffers):
    shape = (*buffers['s8'].shape, 128)
    state = torch.zeros(shape, dtype=torch.float32)
    for coordinate in range(16):
        shift = 4*(coordinate%2)
        lo = (buffers['lo'][..., coordinate//2].int() >> shift) & 15
        hi = (buffers['hi'][..., coordinate//2].int() >> shift) & 15
        q = (hi << 4) | lo
        q = torch.where(q > 127, q-256, q)
        state[..., coordinate] = q.float()*buffers['s8'].float()
    for coordinate in range(64):
        q = (buffers['q4'][..., coordinate//2].int() >> (4*(coordinate%2))) & 15
        q = torch.where(q > 7, q-16, q)
        state[..., 16+coordinate] = q.float()*buffers['s4'].float()
    return state


def quantize_cpu(updated, mode, clip):
    integers, scales = [], []
    for values, bound, factor in ((updated[..., :16], 127, 1.), (updated[..., 16:80], 7, clip)):
        denominator = (values.abs().amax(-1)*factor/bound).clamp_min(1e-8)
        stored = denominator.half()
        actual_denominator = stored.float() if mode == 'stored_scale' else denominator
        safe = torch.where(actual_denominator > 0, actual_denominator, torch.ones_like(actual_denominator))
        z = values/safe[..., None]
        q = half_away_cpu(z).clamp(-bound, bound).int()
        q = torch.where(actual_denominator[..., None] > 0, q, torch.zeros_like(q))
        integers.append(q); scales.append(stored)
    return {'lo': pack_cpu(integers[0]), 'hi': pack_cpu(integers[0] >> 4),
        'q4': pack_cpu(integers[1]), 's8': scales[0], 's4': scales[1]}


def oracle(inputs, table, policy):
    _, mode, clip, diagnostic = policy
    x, rawdt, A, B, C, D, bias = [v.detach().cpu().float() for v in inputs]
    table = table.cpu().long()
    batch, length, heads, dim = x.shape
    groups = B.shape[2]
    group = torch.arange(heads)//(heads//groups)
    indices = table[group][None].expand(batch, heads, 128)
    previous = torch.zeros((batch, heads, dim, 128), dtype=torch.float32)
    packed = quantize_cpu(previous, mode, clip)
    outputs, caches, histories = [], [], []
    for t in range(length):
        # Packed oracle reconstructs the ACTUALLY STORED bytes/scales each token.
        if diagnostic is None: previous = decode_cpu(packed)
        rdt = rawdt[:, t]+bias
        delta = torch.where(rdt <= 20., torch.log(torch.exp(rdt)+1.), rdt)
        decay = torch.exp(A[None]*delta)
        b = B[:, t][:, group].gather(-1, indices)
        c = C[:, t][:, group].gather(-1, indices)
        updated = previous*decay[:, :, None, None] + (b*delta[..., None])[:, :, None]*x[:, t, :, :, None]
        # Three separate reductions independently represent the codec contract.
        y = (updated[..., :16]*c[:, :, None, :16]).sum(-1)
        y = y + (updated[..., 16:80]*c[:, :, None, 16:80]).sum(-1)
        y = y + ((updated[..., 80:]*c[:, :, None, 80:]).sum(-1)+x[:, t]*D[None, :, None])
        outputs.append(y.half())
        if diagnostic == 'prune_only':
            previous = torch.cat((updated[..., :80].half().float(), torch.zeros_like(updated[..., 80:])), -1)
        else:
            packed = quantize_cpu(updated, mode, clip)
            previous = decode_cpu(packed)
            if diagnostic == 'quant_only': previous[..., 80:] = updated[..., 80:].half().float()
        caches.append({k: v.clone() for k, v in packed.items()} if diagnostic is None else {'dense': previous.clone()})
        histories.append(previous.clone())
    return {'output': torch.stack(outputs, 1), 'caches': caches, 'decoded': torch.stack(histories)}


def controlled(length=33, special='ties'):
    batch, heads, dim, groups = 2, 6, 5, 3
    table = torch.stack([torch.roll(torch.arange(127, -1, -1), 9*g) for g in range(groups)]).byte()
    x = torch.empty((batch, length, heads, dim), dtype=torch.float16)
    dt = torch.zeros((batch, length, heads), dtype=torch.float16)
    A = torch.zeros(heads, dtype=torch.float32)
    bias = torch.full((heads,), 32., dtype=torch.float32)
    D = torch.tensor([(-1.)**h/16 for h in range(heads)], dtype=torch.float16)
    B = torch.empty((batch, length, groups, 128), dtype=torch.float16); C = torch.empty_like(B)
    n = torch.arange(128)
    first = torch.tensor([-127., -126.5, -64., -2., -1.5, -.5, 0., .5, 1.5, 2., 63., 64., 126., 126.5, 127., 0.])
    second = torch.tensor([-7., -6.5, -4., -1.5, -.5, 0., .5, 1.5, 4., 6.5, 7.]).repeat(6)[:64]
    wanted = torch.cat((first, second, ((torch.arange(48)%5)-2).float()))
    if special == 'negative': wanted = -wanted.abs()
    if special == 'zero': wanted.zero_()
    if special == 'rounded_scales': wanted[0] = 127.0625; wanted[16] = 7.00390625
    if special == 'underflow': wanted = wanted.sign()*2**-19
    if special == 'subnormal': wanted = wanted.sign()*2**-11
    if special == 'mixed_scales': wanted[:16] = wanted[:16].sign()*2**-19; wanted[16:] = wanted[16:].sign()*2**-15
    if special == 'large_finite': wanted = wanted.sign()*131072.; wanted[80:] = 0.
    for t in range(length):
        sign = (1., 1., -1., -1.)[t%4]
        for b in range(batch):
            for h in range(heads):
                for p in range(dim):
                    x[b, t, h, p] = (-1.)**(b+h+p)*2.**((t//4+b+h+p)%3-6)
            for g in range(groups):
                B[b, t, g, table[g].long()] = (wanted*sign/32).half()
                C[b, t, g, table[g].long()] = (((n*3+t+b+g)%5-2).float()/256).half()
    if special in ('underflow', 'mixed_scales'): x.fill_(2**-5)
    if special == 'subnormal': x.fill_(2**-5)
    if special == 'large_finite': x.fill_(1.); C.zero_(); D.zero_()
    return [x, dt, A, B, C, D, bias], table


def random_inputs(geometry):
    batch, length, heads, dim, groups = geometry
    generator = torch.Generator().manual_seed(2026092906)
    values = [(torch.randn(batch, length, heads, dim, generator=generator)*.08).half(),
        (torch.randn(batch, length, heads, generator=generator)*.2-2).half(),
        -torch.rand(heads, generator=generator).float()-.1,
        (torch.randn(batch, length, groups, 128, generator=generator)*.2).half(),
        (torch.randn(batch, length, groups, 128, generator=generator)*.2).half(),
        (torch.randn(heads, generator=generator)*.1).half(), torch.zeros(heads, dtype=torch.float16)]
    table = torch.stack([torch.randperm(128, generator=generator) for _ in range(groups)]).byte()
    return values, table


def run(inputs, table, policy, pieces, *, capture=False):
    from state_ppl_codec_v6 import allocate_state, scan, validate_finite_result, decode_state
    _, mode, clip, diagnostic = policy
    batch, length, heads, dim = inputs[0].shape
    if sum(pieces) != length: raise ValueError('Partition differs from sequence')
    state = allocate_state(batch, heads, dim, 128, 'cuda', diagnostic=diagnostic)
    output, caches, decoded = [], [], []; start = 0
    for count in pieces:
        part = [v[:, start:start+count] if i in (0, 1, 3, 4) else v for i, v in enumerate(inputs)]
        y, unused = scan(*part, state, table, scale_mode=mode, int4_clip=clip, diagnostic=diagnostic)
        if unused is not None: raise AssertionError('Unexpected persistent/statistics output')
        validate_finite_result(state, y); output.append(y.cpu())
        if capture:
            caches.append({k: v.cpu().clone() for k, v in state.tensors.items()})
            decoded.append(decode_state(state).cpu())
        start += count
    result = {'output': torch.cat(output, 1), 'final': {k: v.cpu() for k, v in state.tensors.items()}, 'bytes': state.nbytes}
    if capture: result.update(caches=caches, decoded=torch.stack(decoded))
    return result


def same_cache(a, b): return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def error(a, b):
    difference = a.float()-b.float()
    return dict(exact=torch.equal(a, b), different_elements=int((a != b).sum()),
        max_absolute_error=float(difference.abs().max()),
        relative_l2=float(torch.linalg.vector_norm(difference)/(torch.linalg.vector_norm(b.float())+1e-30)))



def scalar_arithmetic(kernel, ptx):
    """Canonical emitted scalar arithmetic, excluding input addressing/layout."""
    lines, start = inspect.getsourcelines(kernel.fn)
    raw = next(start+i for i, line in enumerate(lines) if 'dt = tl.load(DT' in line)
    soft = next(start+i for i, line in enumerate(lines) if 'dt = tl.where(dt <=' in line)
    decay = next(start+i for i, line in enumerate(lines) if 'decay = tl.exp(' in line)
    selected, active, raw_active = [], False, False
    for line in ptx.splitlines():
        match = re.search(r'\.loc\s+1\s+(\d+)\s+(\d+)', line)
        if match:
            number, column = map(int, match.groups())
            active = number in (soft, decay) or (number == 0 and column == 45)
            raw_active = number == raw
        instruction = line.strip()
        if not instruction or instruction.startswith(('.', '//', '$')): continue
        if active or (raw_active and instruction.startswith('add.f32')):
            selected.append(instruction)
    names = {}
    def register(match):
        name = match.group()
        if name not in names: names[name] = '%v'+str(len(names))
        return names[name]
    canonical = re.sub(r'%(?:rs|rd|r|p)\d+', register, '\n'.join(selected))
    canonical = re.sub(r'\s+', ' ', canonical).strip()
    return canonical, selected


def capture_scalar_values(inputs, table, artifact_dir):
    """Capture actual GPU scalars and bind their PTX to both recurrence kernels."""
    from state_ppl_codec_v6 import _variant_scan, allocate_state
    from mamba2_recall import state_codec as old

    @triton.jit
    def scalar_probe(DT, A, DB, Delta, Decay, H: tl.constexpr):
        i = tl.program_id(0)
        h = i % H
        av = tl.load(A+h).to(tl.float32)
        bias = tl.load(DB+h).to(tl.float32)
        dt = tl.load(DT+i).to(tl.float32) + bias
        dt = tl.where(dt <= 20., tl.math.log(tl.math.exp(dt)+1.), dt)
        decay = tl.exp(av*dt)
        tl.store(Delta+i, dt)
        tl.store(Decay+i, decay)

    x, dt, A, B, C, D, bias = [v.cuda().contiguous() for v in inputs]
    permutation = table.cuda().contiguous()
    batch, length, heads, dim = x.shape
    groups = B.shape[2]
    delta, decay = torch.empty_like(dt, dtype=torch.float32), torch.empty_like(dt, dtype=torch.float32)
    probe = scalar_probe[(dt.numel(),)](dt, A, bias, delta, decay, heads, num_warps=4)
    # One-token kernels are the actual path already checked against full/chunk
    # executions. Capture their emitted scalar code without changing their math.
    one = [v[:, :1].contiguous() if i in (0, 1, 3, 4) else v for i, v in enumerate((x, dt, A, B, C, D, bias))]
    packed = old.allocate_state(batch, heads, dim, 128, 'sq3p25', 'cuda')
    dense = allocate_state(batch, heads, dim, 128, 'cuda', diagnostic='legacy_emulation')
    y, z = torch.empty_like(one[0]), torch.empty_like(one[0])
    launch = (batch, heads, triton.cdiv(dim, 16))
    original = old._sq_scan[launch](*one, permutation, *(packed.tensors[k] for k in ('lo', 'hi', 'q4', 's8', 's4')),
        y, 1, heads, dim, groups, 16, num_warps=4)
    backing = dense.tensors['dense']
    emulated = _variant_scan[launch](*one, permutation, *([backing]*5), backing, z,
        1, heads, dim, groups, 16, False, 1., 3, num_warps=4)
    artifacts, canonical = {}, {}
    for name, kernel, compiled in (('packed', old._sq_scan, original), ('dense', _variant_scan, emulated), ('scalar_probe', scalar_probe, probe)):
        ptx = compiled.asm['ptx']
        path = artifact_dir/(name+'.ptx'); path.write_text(ptx)
        text, instructions = scalar_arithmetic(kernel, ptx)
        canonical[name] = text
        artifacts[name] = {'file': str(path), 'sha256': sha(path), 'instructions': len(instructions),
            'scalar_sha256': hashlib.sha256(text.encode()).hexdigest()}
    equal = len(set(canonical.values())) == 1
    if not equal: raise AssertionError('Captured scalar PTX differs from actual packed/dense recurrence')
    if not bool(torch.isfinite(delta).all() and torch.isfinite(decay).all()):
        raise AssertionError('Scalar error-bound scope requires finite delta/decay')
    return delta.cpu(), decay.cpu(), {'exact': equal, 'artifacts': artifacts}


def dense_emulation_bound(inputs, table, packed, dense, artifact_dir):
    """Independent FP64 gamma bound; no tolerance fitted to measured drift.

    Per-path recurrence(3)+C product(1)+sum(N-1)+outeradds(2)
    gives gamma21/69/69. The xD path has at most3 roundings. Underflow
    floor covers fewer than1024 operations, <=3 flush events each, and
    downstream gains from x and C. Both output-rounding cells are included.
    """
    artifact_dir.mkdir(parents=True, exist_ok=False)
    delta, decay, scalar_proof = capture_scalar_values(inputs, table, artifact_dir)
    x, rawdt, A, B, C, D, bias = [v.double() for v in inputs]
    batch, length, heads, dim = x.shape
    group = torch.arange(heads)//(heads//B.shape[2])
    indices = table.long()[group][None].expand(batch, heads, 128)
    u32, u64 = 2.**-24, 2.**-53
    gamma = lambda n: n*u32/(1.-n*u32)
    upward = 1.+1024*u64/(1.-1024*u64)
    bounds = []
    for t in range(length):
        previous = torch.zeros_like(packed['decoded'][0], dtype=torch.float64) if t == 0 else packed['decoded'][t-1].double()
        b = B[:, t][:, group].gather(-1, indices)
        c = C[:, t][:, group].gather(-1, indices)
        old = previous.abs()*decay[:, t, :, None, None].double().abs()
        drive = (b*delta[:, t, :, None].double()).abs()[:, :, None]*x[:, t, :, :, None].abs()
        terms = (old+drive)*c[:, :, None].abs()
        e = gamma(21)*terms[..., :16].sum(-1)+gamma(69)*terms[..., 16:80].sum(-1)+gamma(69)*terms[..., 80:].sum(-1)
        e += gamma(3)*(x[:, t]*D[None, :, None]).abs()
        e += 4096.*2.**-126*(1.+x[:, t].abs())*(1.+c.abs().sum(-1)[:, :, None])
        bounds.append(e)
    e = torch.stack(bounds, 1)
    a, b = packed['output'].double(), dense['output'].double()
    def radius(value):
        _, exponent = torch.frexp(value.abs())
        return torch.pow(2., exponent.double()-12.).clamp_min(2.**-25)
    # frexp(0) returns exponent0, so explicitly set its subnormal cell radius.
    ra = torch.where(a == 0, torch.full_like(a, 2.**-25), radius(a))
    rb = torch.where(b == 0, torch.full_like(b, 2.**-25), radius(b))
    bound = torch.nextafter((2.*e+ra+rb)*upward, torch.full_like(e, float('inf')))
    difference = (a-b).abs()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all() and torch.isfinite(bound).all())
    carry_exact = torch.equal(packed['decoded'], dense['decoded'])
    failures = int((difference > bound).sum())
    path = artifact_dir/'evidence.pt'
    torch.save({'inputs': inputs, 'table': table, 'dt_actual': delta, 'decay_actual': decay,
        'packed_output': packed['output'], 'dense_output': dense['output'],
        'packed_carry': packed['decoded'], 'dense_carry': dense['decoded'],
        'packed_caches': packed['caches'], 'dense_caches': dense['caches'], 'bound': bound}, path)
    facts = {'all_token_carries_exact': carry_exact, 'finite_scope': finite, 'bound_failures': failures,
        'max_error_to_bound_ratio': float((difference/bound).max()),
        'max_bound': float(bound.max()), 'output_error': error(packed['output'], dense['output']),
        'scalar_ptx_equivalence': scalar_proof, 'evidence': {'file': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size},
        'bound_formula': '2*(gamma21*T8+gamma69*T4+gamma69*Td+gamma3*abs(xD)+4096*2^-126*(1+abs(x))*(1+sum(abs(C))))+r16(a)+r16(b)',
        'float64_upward_guard': 'multiply by1+gamma1024(u64), then nextafter(+inf)'}
    return carry_exact and finite and failures == 0 and scalar_proof['exact'], facts

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {'format': 'MAMBA2_STATE_PPL_CODEC_CPU_PREPARATION_V1' if args.cpu_only else 'MAMBA2_STATE_PPL_CODEC_CHECK_V1',
        'complete': False, 'passed': False, 'mode': 'cpu_preparation' if args.cpu_only else 'gpu',
        'protocol_sha256': PROTOCOL_SHA, 'started_unix': time.time(), 'checks': [],
        'code_sha256': {p: sha(ROOT/p) for p in ('scripts/state_ppl_codec_v6.py',
            'scripts/check_state_ppl_codec_v6.py', 'mamba2_recall/state_codec.py', 'mamba2_recall/state_quant.py')},
        'scope': 'Independent controlled arithmetic/packing plus GPU kernel partition and legacy replay; no model PPL/MK measurements',
        'criteria': {'controlled': 'exact every-token output/packed bytes/FP16 scales/decoded carry',
            'random_partition': 'exact full/segmented/tokenwise outputs and final cache',
            'random_one_token_cpu_output_relative_l2_bound': .002,
            'stored_scale_division': 'Correctly rounded FP32 quotient before half-away rounding; zero-scale integer0',
            'stored_scale_denominator': 'FP32 clipping multiplication then correctly rounded FP32 division before FP16 round-to-even storage',
            'diagnostic_legacy_emulation': 'controlled output exact; random every-token carry exact and readout within recurrence-inclusive gamma21/69/69/3 plus FP16-cell and FP32-underflow bound'}}
    def check(name, condition, **facts):
        row = dict(name=name, **{'pass': bool(condition)}, **facts); report['checks'].append(row)
        print(json.dumps(row), flush=True)
        if not condition: raise AssertionError(name)
    try:
        torch.set_num_threads(4)
        check('frozen_protocol', sha(ROOT/'docs/STATE_PPL_V6_PROTOCOL.md') == PROTOCOL_SHA)
        halves = torch.tensor([.5, 3.5], dtype=torch.float32)
        below = torch.nextafter(halves, torch.zeros_like(halves))
        boundary_values = torch.stack((halves, below, -halves, -below), -1).flatten()
        boundary_expected = torch.tensor([1, 0, -1, 0, 4, 3, -4, -3], dtype=torch.int32)
        check('CPU-half-away-near-boundary', torch.equal(half_away_cpu(boundary_values).int(), boundary_expected))
        if not args.cpu_only:
            if not torch.cuda.is_available(): raise RuntimeError('CUDA required for actual codec checks')
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            report['gpu'] = torch.cuda.get_device_name()
            from state_ppl_codec_v6 import quantize_stored_probe
            # Include the exact non-power-of-two FP16 denominator that exposed
            # approximate reciprocal division in preserved failed attempt1.
            scale = torch.tensor(.037933349609375, dtype=torch.float16)
            exact_value = scale.float()*3.5
            below_value = torch.nextafter(exact_value, torch.tensor(0.))
            probe_values = torch.cat((boundary_values, torch.stack((exact_value, below_value, -exact_value, -below_value)), torch.tensor([1., -1., 0.])))
            probe_scales = torch.cat((torch.ones(8, dtype=torch.float16), scale.repeat(4), torch.zeros(3, dtype=torch.float16)))
            probe_expected = torch.cat((boundary_expected, torch.tensor([4, 3, -4, -3, 0, 0, 0], dtype=torch.int32)))
            for limit in (7, 127):
                result = quantize_stored_probe(probe_values.cuda(), probe_scales.cuda(), limit).cpu()
                check('GPU-stored-scale-half-boundaries-'+str(limit), torch.equal(result, probe_expected),
                    actual_codes=result.tolist(), expected_codes=probe_expected.tolist())
        # No implementation codec functions are used to construct these expected results.
        for special in ('ties', 'negative', 'zero', 'rounded_scales', 'underflow', 'subnormal', 'mixed_scales', 'large_finite'):
            inputs, table = controlled(1, special)
            for policy in POLICIES:
                if special == 'large_finite' and policy[3] == 'prune_only': continue
                expected = oracle(inputs, table, policy)
                check('CPU-fixture-finite-'+special+'-'+policy[0], bool(torch.isfinite(expected['output']).all()
                    and torch.isfinite(expected['decoded']).all()))
                if special == 'underflow' and policy[1] == 'stored_scale':
                    buffers = expected['caches'][0]
                    check('CPU-zero-scale-integer-zero-'+policy[0],
                        all(bool((v == 0).all()) for v in buffers.values()))
                if special == 'subnormal' and policy[3] is None:
                    buffers = expected['caches'][0]
                    check('CPU-positive-subnormal-scale-'+policy[0],
                        any(bool(((buffers[k] > 0) & (buffers[k] < torch.finfo(torch.float16).tiny)).any()) for k in ('s8', 's4')))
                if args.cpu_only: continue
                gpu_inputs, gpu_table = [v.cuda() for v in inputs], table.cuda()
                actual = run(gpu_inputs, gpu_table, policy, [1], capture=True)
                check('GPU-independent-exact-'+special+'-'+policy[0],
                    torch.equal(actual['output'], expected['output'])
                    and same_cache(actual['caches'][0], expected['caches'][0])
                    and torch.equal(actual['decoded'], expected['decoded']),
                    output_error=error(actual['output'], expected['output']), carry_error=error(actual['decoded'], expected['decoded']))
        inputs, table = controlled(65)
        controlled_gpu = {}
        for policy in POLICIES:
            expected = oracle(inputs, table, policy)
            check('CPU-multistep-finite-'+policy[0], bool(torch.isfinite(expected['output']).all() and torch.isfinite(expected['decoded']).all()))
            if args.cpu_only: continue
            actual = run([v.cuda() for v in inputs], table.cuda(), policy, [1]*65, capture=True)
            controlled_gpu[policy[0]] = actual
            check('GPU-65token-independent-exact-'+policy[0], torch.equal(actual['output'], expected['output'])
                and all(same_cache(a, b) for a, b in zip(actual['caches'], expected['caches']))
                and torch.equal(actual['decoded'], expected['decoded']),
                output_error=error(actual['output'], expected['output']), carry_error=error(actual['decoded'], expected['decoded']))
        if not args.cpu_only:
            from state_ppl_codec_v6 import allocate_state, scan, validate_options, validate_finite_result
            from mamba2_recall import state_codec as legacy
            for geometry in ((2, 17, 8, 19, 2), (1, 9, 6, 3, 3), (2, 5, 4, 1, 1)):
                inputs, table = random_inputs(geometry); b, length, h, p, g = geometry
                gpu_inputs, gpu_table = [v.cuda() for v in inputs], table.cuda()
                results = {}
                for policy in POLICIES:
                    full = run(gpu_inputs, gpu_table, policy, [length])
                    parts = run(gpu_inputs, gpu_table, policy, [1, 3, length-4])
                    tokens = run(gpu_inputs, gpu_table, policy, [1]*length, capture=True)
                    results[policy[0]] = tokens
                    check('partition-'+str(geometry)+'-'+policy[0], torch.equal(full['output'], parts['output'])
                        and torch.equal(full['output'], tokens['output']) and same_cache(full['final'], parts['final'])
                        and same_cache(full['final'], tokens['final']))
                    row_bytes = 52 if policy[3] is None else 512
                    check('actual-bytes-'+str(geometry)+'-'+policy[0], full['bytes'] == b*h*p*row_bytes,
                        row_bytes=row_bytes, state_bytes=full['bytes'], is_3p25_candidate=policy[3] is None)
                    single = [v[:, :1] if i in (0, 1, 3, 4) else v for i, v in enumerate(inputs)]
                    expected = oracle(single, table, policy)
                    e = error(tokens['output'][:, :1], expected['output'])
                    check('random-one-token-oracle-'+str(geometry)+'-'+policy[0], e['relative_l2'] <= .002, **e)
                bounded, bound_facts = dense_emulation_bound(inputs, table, results['legacy'], results['legacy_emulation'],
                    args.out.parent/(args.out.stem+'_details')/('geometry_'+'_'.join(map(str, geometry))))
                check('dense-legacy-emulation-'+str(geometry), bounded, **bound_facts)
                original_state = legacy.allocate_state(b, h, p, 128, 'sq3p25', 'cuda')
                original_output, _ = legacy.scan(*gpu_inputs, original_state, gpu_table)
                check('delegated-legacy-'+str(geometry), torch.equal(original_output.cpu(), results['legacy']['output'])
                    and same_cache({k: v.cpu() for k, v in original_state.tensors.items()}, results['legacy']['final']))
                first = results['stored_scale']['final']
                check('INT8-unchanged-by-INT4-clipping-'+str(geometry), all(
                    all(torch.equal(first[k], results[name]['final'][k]) for k in ('lo', 'hi', 's8'))
                    for name in ('clip4_095', 'clip4_090', 'clip4_080')))
            for diagnostic in (None, 'prune_only', 'quant_only', 'legacy_emulation'):
                state = allocate_state(1, 128, 64, 128, 'cuda', diagnostic=diagnostic)
                total = 56*state.nbytes+56*10240*4*2+56*8*128
                check('production-budget-'+str(diagnostic), total == (28499968 if diagnostic is None else 239525888),
                    actual_state_bytes=56*state.nbytes, table_bytes=57344, total_bytes=total,
                    is_3p25_candidate=diagnostic is None)
            # Finite inputs can overflow the stored scale/carry; reject them explicitly.
            inputs, table = controlled(1)
            inputs[0].fill_(65504.); inputs[3].fill_(65504.); inputs[4].zero_(); inputs[5].zero_()
            for policy in POLICIES:
                state = allocate_state(2, 6, 5, 128, 'cuda', diagnostic=policy[3])
                output, _ = scan(*[v.cuda() for v in inputs], state, table.cuda(),
                    scale_mode=policy[1], int4_clip=policy[2], diagnostic=policy[3])
                rejected = False
                try: validate_finite_result(state, output)
                except FloatingPointError: rejected = True
                check('reject-nonfinite-carry-or-scale-'+policy[0], rejected)
            rejected = False
            try: validate_options('legacy', .95)
            except ValueError: rejected = True
            check('reject-unfrozen-legacy-clipping-policy', rejected)
        report['cuda_initialized'] = torch.cuda.is_initialized()
        if args.cpu_only: check('CUDA-remained-uninitialized', not report['cuda_initialized'])
        report.update(complete=True, passed=True)
    except BaseException as exc:
        report['error'] = repr(exc)
        raise
    finally:
        report['finished_unix'] = time.time()
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__': main()
