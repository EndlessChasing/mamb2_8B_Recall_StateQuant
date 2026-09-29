#!/usr/bin/env python3
"""Bounded V11 scan/carry/STE checks, with independently auditable raw tensors.

Forward and every recomputed carry must be exact. The CPU PyTorch STE graph
uses actual packed carries as detached trajectory values, not an implementation
of the Triton backward and not finite differences of discrete quantization.
Gradient tolerances below were fixed before the first GPU run. Atomic sums are
not promised bitwise repeatable. No language model or quality data are loaded.
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
PROTOCOL_SHA = 'cebe06473806d74726ac006ff6d5dd316f4fdcfb779eb47f5af271ddb1fa04a6'
FORMAT = 'MAMBA2_STATE_PPL_V11_TRAINING_CHECK_V1'
LAYOUT = '32_32_64'
GRAD_ATOL, GRAD_RTOL = 2e-5, .015
REPLAY_ATOL, REPLAY_RTOL = 2e-6, .002
GRAD_INDICES = (0, 1, 3, 4)
GRAD_NAMES = ('x', 'dt', 'B', 'C')
RANDOM_GEOMETRIES = ((2, 19, 4, 19, 2), (1, 1, 8, 17, 2), (1, 65, 4, 19, 2))
SPECIALS = ('dyadic', 'rounded_scales', 'underflow', 'subnormal', 'wide_safe', 'zero')
SOURCE_FILES = (
    'scripts/state_ppl_training_scan_v11.py', 'scripts/state_ppl_training_v11.py',
    'scripts/check_state_ppl_training_v11.py', 'scripts/state_ppl_codec_v10.py',
    'scripts/state_ppl_codec_v6.py', 'mamba2_recall/state_training.py',
    'mamba2_recall/state_training_scan.py', 'mamba2_recall/state_codec.py',
    'mamba2_recall/state_quant.py', 'docs/STATE_RESURFACE_V11_PROTOCOL.md')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def cpu(values): return [v.detach().cpu().clone() for v in values]
def same_dict(a, b): return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
def subset(values, start, end): return [v[:, start:end] if i in GRAD_INDICES else v for i, v in enumerate(values)]
def fresh(values, device): return [v.detach().to(device).clone().requires_grad_(i in GRAD_INDICES) for i, v in enumerate(values)]


def decode_cpu(buffers):
    """Decode raw persistent nibbles independently of the deployment decoder."""
    state = torch.zeros((*buffers['s8'].shape, 128), dtype=torch.float32)
    for j in range(32):
        low = (buffers['lo'][..., j // 2].long() >> (4 * (j % 2))) & 15
        high = (buffers['hi'][..., j // 2].long() >> (4 * (j % 2))) & 15
        q = (high << 4) | low
        q = torch.where(q >= 128, q - 256, q)
        state[..., j] = q.float() * buffers['s8'].float()
        q4 = (buffers['q4'][..., j // 2].long() >> (4 * (j % 2))) & 15
        q4 = torch.where(q4 >= 8, q4 - 16, q4)
        state[..., 32 + j] = q4.float() * buffers['s4'].float()
    return state


def quantize_cpu(state):
    """Independent IEEE FP32 division, stored FP16 scale, half-away, packing."""
    codes, scales = [], []
    for part, limit in ((state[..., :32], 127), (state[..., 32:64], 7)):
        scale = (part.abs().amax(-1) / limit).clamp_min(1e-8).half()
        safe = torch.where(scale > 0, scale.float(), torch.ones_like(scale).float())
        z = part / safe[..., None]
        whole = z.abs().floor()
        q = z.sign() * (whole + ((z.abs() - whole) >= .5).float())
        codes.append(torch.where(scale[..., None] > 0, q.clamp(-limit, limit), 0).int())
        scales.append(scale)
    def pack(q):
        result = torch.zeros((*q.shape[:-1], 16), dtype=torch.uint8)
        for j in range(32): result[..., j // 2] |= ((q[..., j].long() & 15).byte() << (4 * (j % 2)))
        return result
    return dict(lo=pack(codes[0]), hi=pack(codes[0] >> 4), q4=pack(codes[1]), s8=scales[0], s4=scales[1])


def mapped(tensor, table, heads):
    groups = table.shape[0]
    group = torch.arange(heads, device=tensor.device) // (heads // groups)
    order = table.long()[group][None].expand(tensor.shape[0], heads, 128)
    return tensor[:, group].gather(-1, order)


def ste_oracle(inputs, table, exact_carries):
    """CPU autograd of a 64-live masked STE, all128 current readout gradients.

    The dequantized numerical carry comes from independent decode of actual
    packed bytes. The differentiable identity path follows the unquantized
    updated carry. FP64 FMA emulation models one FP32 rounded multiply-add;
    exp, softplus and reductions may differ by the declared numeric tolerance.
    """
    x, raw, A, B, C, D, bias = inputs
    b, length, heads, dim = x.shape
    previous = torch.zeros((b, heads, dim, 128), dtype=torch.float32, device=x.device)
    outputs = []
    for t in range(length):
        arg = raw[:, t].float() + bias.float()
        dt = torch.where(arg <= 20., torch.log(torch.exp(arg) + 1.), arg)
        decay = torch.exp(A * dt)
        bv, cv = mapped(B[:, t].float(), table, heads), mapped(C[:, t].float(), table, heads)
        add = (bv * dt[..., None])[:, :, None] * x[:, t].float()[..., None]
        updated = (previous.double() * decay[..., None, None].double() + add.double()).float()
        product = updated * cv[:, :, None]
        y = product[..., :32].sum(-1) + product[..., 32:64].sum(-1)
        y = y + (product[..., 64:].sum(-1).double() + x[:, t].double() * D.double()[None, :, None]).float()
        outputs.append(y.half())
        live = updated[..., :64]
        carry = exact_carries[t][..., :64].to(x.device)
        # This arrangement forwards exact carry without subtract/add cancellation.
        live = carry + (live - live.detach())
        previous = torch.cat((live, torch.zeros_like(updated[..., 64:])), -1)
    return torch.stack(outputs, 1)


def controlled_oracle(inputs, table):
    """Independent dyadic recurrence; decode persisted scale before each token."""
    x, raw, A, B, C, D, bias = [v.float() for v in inputs]
    b, length, heads, dim = x.shape
    previous = torch.zeros((b, heads, dim, 128), dtype=torch.float32)
    caches = []; outputs = []
    for t in range(length):
        arg = raw[:, t] + bias
        dt = torch.where(arg <= 20., torch.log(torch.exp(arg) + 1.), arg)
        decay = torch.exp(A * dt)
        bv, cv = mapped(B[:, t], table, heads), mapped(C[:, t], table, heads)
        add = (bv * dt[..., None])[:, :, None] * x[:, t, :, :, None]
        updated = (previous.double() * decay[..., None, None].double() + add.double()).float()
        prod = updated * cv[:, :, None]
        out = prod[..., :32].sum(-1) + prod[..., 32:64].sum(-1)
        out += (prod[..., 64:].sum(-1).double() + x[:, t].double() * D.double()[None, :, None]).float()
        outputs.append(out.half())
        packed = quantize_cpu(updated); caches.append(packed)
        previous = decode_cpu(packed)
    return dict(output=torch.stack(outputs, 1), caches=caches)


def make_random(geometry):
    b, length, heads, dim, groups = geometry
    gen = torch.Generator().manual_seed(2026092911)
    def rand(shape, scale=.15): return (torch.randn(shape, generator=gen) * scale).half()
    values = [rand((b, length, heads, dim)), rand((b, length, heads)),
        -torch.exp(torch.randn(heads, generator=gen) * .2), rand((b, length, groups, 128)),
        rand((b, length, groups, 128)), rand((heads,)), torch.full((heads,), -1.5).half()]
    table = torch.stack([torch.randperm(128, generator=gen) for _ in range(groups)]).byte()
    return values, table


def make_controlled(special):
    b, length, heads, dim, groups = 2, (65 if special == 'dyadic' else 5), 4, 19, 2
    table = torch.stack([torch.roll(torch.arange(127, -1, -1), 13*g) for g in range(groups)]).byte()
    x = torch.empty((b, length, heads, dim), dtype=torch.float16)
    raw = torch.zeros((b, length, heads), dtype=torch.float16)
    A = torch.zeros(heads); bias = torch.full((heads,), 32.)
    B = torch.empty((b, length, groups, 128), dtype=torch.float16); C = torch.zeros_like(B)
    D = torch.zeros(heads, dtype=torch.float16)
    a = torch.tensor([-127., -126.5, -64., -1.5, -.5, 0., .5, 1.5, 64., 126.5, 127.])
    z = torch.tensor([-7., -6.5, -4., -1.5, -.5, 0., .5, 1.5, 4., 6.5, 7.])
    wanted = torch.cat((a.repeat(3)[:32], z.repeat(3)[:32], (torch.arange(64)%5-2).float()))
    if special == 'rounded_scales': wanted[0] = 127.0625; wanted[32] = 7.00390625
    if special == 'underflow': wanted = wanted.sign() * 2**-19
    if special == 'subnormal': wanted = wanted.sign() * 2**-11
    if special == 'wide_safe': wanted = wanted.sign() * 4096.
    if special == 'zero': wanted.zero_()
    for t in range(length):
        for row in range(b):
            for head in range(heads):
                for p in range(dim): x[row,t,head,p] = (-1.)**(row+head+p) * 2**((row+head+p)%3-5)
            for g in range(groups):
                B[row,t,g,table[g].long()] = (wanted * (1.,1.,-1.,-1.)[t%4] / 32).half()
                C[row,t,g,int(table[g,(0,31,32,63,64,127)[t%6]])] = (-1.)**(row+g) * 2**-8
    if special in ('underflow', 'subnormal'): x.fill_(2**-5)
    return [x, raw, A, B, C, D, bias], table


def grad_summary(actual, expected, atol, rtol):
    a, e = actual.float(), expected.to(actual.dtype).float()
    diff = (a-e).abs(); bound = atol + rtol*e.abs()
    return dict(pass_=bool(torch.isfinite(a).all() and torch.isfinite(e).all() and (diff<=bound).all()),
        max_absolute_error=float(diff.max()), max_bound_ratio=float((diff/bound).max()),
        relative_rms_error=float(diff.square().mean().sqrt()/e.square().mean().sqrt().clamp_min(1e-12)),
        atol=atol, rtol=rtol)


def deployed(inputs, table, pieces, capture=False):
    from state_ppl_codec_v10 import allocate_state, scan, validate_finite_result
    b, length, heads, dim = inputs[0].shape
    state = allocate_state(b, heads, dim, 128, inputs[0].device, layout=LAYOUT)
    outputs = []; caches = []; start = 0
    for count in pieces:
        out, stats = scan(*subset(inputs,start,start+count), state, table, layout=LAYOUT)
        assert stats is None
        validate_finite_result(state, out)
        outputs.append(out.detach().cpu());start += count
        if capture: caches.append({k:v.detach().cpu().clone() for k,v in state.tensors.items()})
    assert start == length
    return dict(output=torch.cat(outputs,1),caches=caches,
        final={k:v.detach().cpu().clone() for k,v in state.tensors.items()}, bytes=state.nbytes)


def histories(inputs, table, chunk_size):
    from state_ppl_training_scan_v11 import _carry_checkpoints, _chunk_history, TILE
    b, length, heads, dim = inputs[0].shape; groups = inputs[3].shape[2]
    nc = (length+chunk_size-1)//chunk_size
    cp = torch.full((b,nc,heads,dim,64),float('nan'),device='cuda')
    hist = torch.full((b,chunk_size,heads,dim,64),float('nan'),device='cuda')
    grid = (b,heads,(dim+TILE-1)//TILE)
    _carry_checkpoints[grid](inputs[0],inputs[1],inputs[2],inputs[3],inputs[6],table,cp,
        length,heads,dim,groups,chunk_size,nc,TILE,num_warps=4)
    rows = []
    for chunk in range(nc):
        hist.fill_(float('nan'))
        _chunk_history[grid](inputs[0],inputs[1],inputs[2],inputs[3],inputs[6],table,cp,hist,
            length,heads,dim,groups,chunk_size,nc,chunk,TILE,num_warps=4)
        rows.append(hist[:,:min(chunk_size,length-chunk*chunk_size)].cpu().clone())
    return dict(chunk_size=chunk_size, checkpoints=cp.cpu(), history=torch.cat(rows,1))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--cpu-only',action='store_true')
    args=parser.parse_args();path=Path(args.out)
    if path.exists(): raise FileExistsError(path)
    evidence_path=path.with_name(path.stem+'_evidence.pt')
    if evidence_path.exists(): raise FileExistsError(evidence_path)
    report=dict(format=FORMAT if not args.cpu_only else 'MAMBA2_STATE_PPL_V11_TRAINING_CPU_CHECK_V1',
        protocol_sha256=PROTOCOL_SHA,mode='cpu' if args.cpu_only else 'gpu',complete=False,passed=False,
        layout=LAYOUT,checks=[],criteria=dict(forward='bitwise_equal',carry='every_token_bitwise_equal',
        cpu_controlled='raw_packed_bytes_and_readout_exact',ste_gradient=dict(atol=GRAD_ATOL,rtol=GRAD_RTOL),
        gradient_replay=dict(atol=REPLAY_ATOL,rtol=REPLAY_RTOL),live_coordinates=64,current_readout_coordinates=128),
        limitations=['Declared live-mask STE, not the derivative of hard quantization.',
          'FP32 atomic gradient sums need not be bitwise repeatable.',
          'CPU oracle uses exact deployed carry values with an independent autograd graph.',
          'Synthetic checks do not establish model PPL/MK; model export probes belong to training receipts.'])
    evidence=dict(format='MAMBA2_STATE_PPL_V11_TRAINING_EVIDENCE_V1',protocol_sha256=PROTOCOL_SHA,cases=[])
    def record(name, passed, **details):
        row=dict(name=name,**{'pass':bool(passed)},**details);report['checks'].append(row)
        print(json.dumps(row),flush=True)
        if not passed: raise AssertionError(name)
    torch.set_num_threads(4);started=time.monotonic()
    try:
        record('protocol_hash',sha(ROOT/'docs/STATE_RESURFACE_V11_PROTOCOL.md')==PROTOCOL_SHA)
        from state_ppl_training_scan_v11 import scan_training,training_workspace_bytes
        # CPU contract/fixture checks run in both modes, before any CUDA call.
        record('cuda_uninitialized_before_tests',not torch.cuda.is_initialized())
        for special in SPECIALS:
            values, table=make_controlled(special);ref=controlled_oracle(values,table)
            decoded=[decode_cpu(v) for v in ref['caches']]
            record('cpu_fixture_'+special,torch.isfinite(ref['output']).all() and all(torch.isfinite(v).all() for v in decoded),
                tokens=values[0].shape[1],geometry=list(values[0].shape),dead_carry_zero=all(torch.count_nonzero(v[...,64:])==0 for v in decoded))
            if special=='underflow':record('cpu_scale_underflow_defined_zero',all(torch.count_nonzero(v)==0 for v in decoded))
            if special=='subnormal':record('cpu_scale_subnormal_present',any(((v['s8']>0)&(v['s8']<2**-14)).any() for v in ref['caches']))
        values,table=make_random((1,1,2,3,1))
        for bad in (0,257,True,1.5):
            rejected=False
            try:scan_training(*values,table,chunk_size=bad)
            except ValueError:rejected=True
            record('reject_chunk_'+str(bad),rejected)
        for i,name in ((2,'A'),(5,'D'),(6,'dt_bias')):
            inputs=fresh(values,'cpu');inputs[i].requires_grad_(True);rejected=False
            try:scan_training(*inputs,table)
            except ValueError:rejected=True
            record('reject_trainable_'+name,rejected)
        space=training_workspace_bytes(1,2048)
        record('workspace_64_live_count',space==dict(carry_scratch_bytes=203423744,gradient_scratch_bytes=84934656,total_scratch_bytes=288358400),**space)
        if args.cpu_only:
            record('cpu_run_did_not_initialize_cuda',not torch.cuda.is_initialized())
        else:
            torch.backends.cuda.matmul.allow_tf32=False
            fixtures=[('random_'+str(j),*make_random(g),False) for j,g in enumerate(RANDOM_GEOMETRIES)]
            fixtures += [('controlled_'+s,*make_controlled(s),True) for s in SPECIALS]
            for name,values,table_cpu,controlled in fixtures:
                inputs=fresh(values,'cuda');table=table_cpu.cuda();length=values[0].shape[1]
                full=deployed(inputs,table,[length]);steps=deployed(inputs,table,[1]*length,True)
                carries=torch.stack([decode_cpu(v) for v in steps['caches']])
                expected_prior=torch.cat((torch.zeros_like(carries[:1]),carries[:-1]),0).permute(1,0,2,3,4)[...,:64]
                record(name+'_full_token_partition',torch.equal(full['output'],steps['output']) and same_dict(full['final'],steps['final']))
                record(name+'_physical_52_byte_rows',full['bytes']==values[0].shape[0]*values[0].shape[2]*values[0].shape[3]*52,actual_bytes=full['bytes'])
                if controlled:
                    independent=controlled_oracle(values,table_cpu)
                    record(name+'_independent_packed_oracle',torch.equal(independent['output'],steps['output']) and all(same_dict(a,b) for a,b in zip(independent['caches'],steps['caches'])))
                case=dict(name=name,inputs=cpu(values),table=table_cpu,packed=steps['caches'],deployed_output=full['output'],histories=[])
                evidence['cases'].append(case)
                for k in (1,8,13,32):
                    result=histories(inputs,table,k);case['histories'].append(result)
                    record(name+'_carry_history_K'+str(k),torch.equal(result['history'],expected_prior) and torch.equal(result['checkpoints'],expected_prior[:,::k]))
                y=scan_training(*inputs,table,chunk_size=8)
                record(name+'_training_forward_exact',torch.equal(y.detach().cpu(),full['output']))
                gen=torch.Generator().manual_seed(2026092912)
                gy=(torch.randn(y.shape,generator=gen)*(.00001 if name=='controlled_wide_safe' else .03)).half()
                gradients=torch.autograd.grad(y,[inputs[i] for i in GRAD_INDICES],gy.cuda())
                # CPU FP32 leaves avoid half accumulation; cast expected gradients once.
                reference=[v.float().requires_grad_(i in GRAD_INDICES) for i,v in enumerate(cpu(values))]
                oracle_y=ste_oracle(reference,table_cpu,carries)
                expected=torch.autograd.grad(oracle_y,[reference[i] for i in GRAD_INDICES],gy)
                case.update(gy=gy,gradients=cpu(gradients),reference_gradients=cpu(expected))
                for tag,actual,want in zip(GRAD_NAMES,cpu(gradients),expected):
                    result=grad_summary(actual,want,GRAD_ATOL,GRAD_RTOL)
                    record(name+'_ste_gradient_'+tag,result.pop('pass_'),**result)
                alt=torch.autograd.grad(scan_training(*inputs,table,chunk_size=13),[inputs[i] for i in GRAD_INDICES],gy.cuda())
                case['alternate_chunk_gradients']=cpu(alt)
                for tag,a,b in zip(GRAD_NAMES,cpu(alt),cpu(gradients)):
                    result=grad_summary(a,b,REPLAY_ATOL,REPLAY_RTOL)
                    record(name+'_chunk_gradient_'+tag,result.pop('pass_'),**result)
                # This mirrors activation checkpoint replay used by the outer trainer.
                from torch.utils.checkpoint import checkpoint
                replay=checkpoint(lambda *xs:scan_training(*xs,table,chunk_size=8),*inputs,use_reentrant=False)
                replay_g=torch.autograd.grad(replay,[inputs[i] for i in GRAD_INDICES],gy.cuda())
                case['checkpoint_gradients']=cpu(replay_g)
                record(name+'_checkpoint_forward_exact',torch.equal(replay.detach().cpu(),full['output']))
                record(name+'_checkpoint_gradient',all(grad_summary(a,b,REPLAY_ATOL,REPLAY_RTOL)['pass_'] for a,b in zip(cpu(replay_g),cpu(gradients))),atol=REPLAY_ATOL,rtol=REPLAY_RTOL)
                # Fresh zero state for every training call, including after other examples.
                record(name+'_fresh_replay_exact',torch.equal(scan_training(*inputs,table,chunk_size=32).detach().cpu(),full['output']))
            values,table_cpu=make_random((2,5,4,19,2))
            table_cpu=torch.arange(128).byte().repeat(2,1)
            values[3][...,:64]=0;values[4][...,:64]=0;values[5].zero_()
            inputs=fresh(values,'cuda');y=scan_training(*inputs,table_cpu.cuda(),chunk_size=2)
            gradients=torch.autograd.grad(y[:,-1].float().sum(),[inputs[i] for i in GRAD_INDICES])
            record('dead_future_gradients_exact_zero',all(torch.count_nonzero(g[:,:-1])==0 for g in gradients))
            record('dead_current_gradients_nonzero',all(torch.count_nonzero(g[:,-1])>0 for g in gradients))
            evidence['dead_current_future']=dict(inputs=cpu(values),table=table_cpu,gradients=cpu(gradients),output=y.detach().cpu())
            values,table_cpu=make_random((1,64,128,64,8));inputs=fresh(values,'cuda')
            y=scan_training(*inputs,table_cpu.cuda());grads=torch.autograd.grad(y.float().square().mean(),[inputs[i] for i in GRAD_INDICES])
            record('production_geometry_finite',torch.isfinite(y).all() and all(torch.isfinite(g).all() for g in grads),geometry=list(y.shape),workspace=training_workspace_bytes(1,64))
            evidence_path.parent.mkdir(parents=True,exist_ok=True);torch.save(evidence,evidence_path)
            report['evidence']=dict(file=str(evidence_path),sha256=sha(evidence_path),bytes=evidence_path.stat().st_size,cases=len(evidence['cases']))
        report['passed']=all(row['pass'] for row in report['checks']);report['complete']=True
    except Exception as exc:
        report['error']=repr(exc)
        if evidence['cases'] and not evidence_path.exists():
            evidence_path.parent.mkdir(parents=True,exist_ok=True);torch.save(evidence,evidence_path)
            report['partial_evidence']=dict(file=str(evidence_path),sha256=sha(evidence_path),bytes=evidence_path.stat().st_size)
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started;report['cuda_initialized']=torch.cuda.is_initialized()
        report['code_sha256']={p:sha(ROOT/p) for p in SOURCE_FILES}
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
