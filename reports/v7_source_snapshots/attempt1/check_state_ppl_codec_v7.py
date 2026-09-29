#!/usr/bin/env python3
"""Independent packed v7 codebook arithmetic, causal carry and storage checks.

Nonuniform oracle minimizes FP64 distances to actual decoded FP32 levels,
independent of the kernel's midpoint implementation. Uniform retains v6 math.
The65-token oracle uses sparse readout to isolate recurrence/packing from
reduction-order drift. Random fixtures require bitwise partition replay.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
PROTOCOL_SHA='84ffdce1d9e5d5dbac2b6996072f0db09bd1bf79fd3c55dad8b8f35973cd5d14'
BOOKS=('uniform','mild','quadratic','fp4like')
# Independently declared mathematical grid; never import production levels.
EXPECTED={
    'uniform':list(range(8)),
    'mild':[0,.5,1.5,2.5,3.5,4.5,5.5,7],
    'quadratic':[i*i/7 for i in range(8)],
    'fp4like':[0,7/12,7/6,7/4,7/3,7/2,14/3,7],
}
from check_state_ppl_codec_v6 import controlled,random_inputs,half_away_cpu,pack_cpu,same_cache


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def levels(book):return torch.tensor(EXPECTED[book],dtype=torch.float32)


def nearest_cpu(value,scale,book):
    decoded=(scale.float()[...,None]*levels(book)).float()
    distances=(value.abs().double()[...,None]-decoded.double()).abs()
    # Flip before argmin: exact ties prefer the larger decoded magnitude.
    index=7-distances.flip(-1).argmin(-1)
    index=torch.where(value<0,-index,index).int()
    return torch.where(scale>0,index,torch.zeros_like(index))


def decode_cpu(buffers,book):
    result=torch.zeros((*buffers['s8'].shape,128),dtype=torch.float32)
    for j in range(16):
        shift=4*(j%2)
        low=(buffers['lo'][...,j//2].int()>>shift)&15
        high=(buffers['hi'][...,j//2].int()>>shift)&15
        q=(high<<4)|low;q=torch.where(q>127,q-256,q)
        result[...,j]=q.float()*buffers['s8'].float()
    for j in range(64):
        q=(buffers['q4'][...,j//2].int()>>(4*(j%2)))&15
        q=torch.where(q>7,q-16,q)
        value=levels(book)[q.abs()]*buffers['s4'].float()
        result[...,16+j]=torch.where(q<0,-value,value)
    return result


def quantize_cpu(state,book):
    s8=(state[...,:16].abs().amax(-1)/127).clamp_min(1e-8).half()
    s4=(state[...,16:80].abs().amax(-1)/7).clamp_min(1e-8).half()
    safe8=torch.where(s8>0,s8.float(),torch.ones_like(s8).float())
    q8=half_away_cpu(state[...,:16]/safe8[...,None]).clamp(-127,127).int()
    q8=torch.where(s8[...,None]>0,q8,0)
    if book=='uniform':
        safe4=torch.where(s4>0,s4.float(),torch.ones_like(s4).float())
        q4=half_away_cpu(state[...,16:80]/safe4[...,None]).clamp(-7,7).int()
        q4=torch.where(s4[...,None]>0,q4,0)
    else:
        q4=nearest_cpu(state[...,16:80],s4[...,None],book)
    return dict(lo=pack_cpu(q8),hi=pack_cpu(q8>>4),q4=pack_cpu(q4),s8=s8,s4=s4)


def oracle(inputs,table,book):
    x,rawdt,A,B,C,D,bias=[v.float() for v in inputs]
    batch,length,heads,dim=x.shape
    group=torch.arange(heads)//(heads//B.shape[2])
    indices=table.long()[group][None].expand(batch,heads,128)
    state=torch.zeros((batch,heads,dim,128),dtype=torch.float32)
    cache=quantize_cpu(state,book); outputs=[];caches=[];carries=[]
    for t in range(length):
        state=decode_cpu(cache,book)
        dt=rawdt[:,t]+bias
        dt=torch.where(dt<=20.,torch.log(torch.exp(dt)+1.),dt)
        decay=torch.exp(A[None]*dt)
        b=B[:,t][:,group].gather(-1,indices)
        c=C[:,t][:,group].gather(-1,indices)
        updated=state*decay[:,:,None,None]+(b*dt[...,None])[:,:,None]*x[:,t,:,:,None]
        y=(updated[...,:16]*c[:,:,None,:16]).sum(-1)
        y=y+(updated[...,16:80]*c[:,:,None,16:80]).sum(-1)
        y=y+((updated[...,80:]*c[:,:,None,80:]).sum(-1)+x[:,t]*D[None,:,None])
        outputs.append(y.half())
        cache=quantize_cpu(updated,book)
        caches.append({k:v.clone() for k,v in cache.items()});carries.append(decode_cpu(cache,book))
    return dict(output=torch.stack(outputs,1),caches=caches,decoded=torch.stack(carries))


def run(inputs,table,book,pieces,capture=False):
    from state_ppl_codec_v7 import allocate_state,scan,decode_state,validate_finite_result
    batch,length,heads,dim=inputs[0].shape
    if sum(pieces)!=length:raise ValueError('Invalid partition')
    state=allocate_state(batch,heads,dim,128,'cuda')
    outputs=[];caches=[];carries=[];start=0
    for count in pieces:
        part=[v[:,start:start+count] if i in(0,1,3,4) else v for i,v in enumerate(inputs)]
        y,unused=scan(*part,state,table,codebook=book)
        if unused is not None:raise AssertionError('Unexpected extra state output')
        validate_finite_result(state,y);outputs.append(y.cpu())
        if capture:
            caches.append({k:v.cpu().clone() for k,v in state.tensors.items()})
            carries.append(decode_state(state,codebook=book).cpu())
        start+=count
    result=dict(output=torch.cat(outputs,1),final={k:v.cpu() for k,v in state.tensors.items()},bytes=state.nbytes)
    if capture:result.update(caches=caches,decoded=torch.stack(carries))
    return result


def boundary_fixture(book):
    values=[];scales=[];exact_ties=0
    for s in (0.,2.**-24,2.**-14,.037933349609375,1.,65504.):
        scale=torch.tensor(s,dtype=torch.float16)
        decoded=(levels(book)*scale.float()).float()
        for j in range(7):
            midpoint=(float(decoded[j])+float(decoded[j+1]))*.5
            center=torch.tensor(midpoint,dtype=torch.float32)
            exact_ties+=int(float(center)==midpoint and s>0)
            for v in(torch.nextafter(center,torch.tensor(float('-inf'))),center,torch.nextafter(center,torch.tensor(float('inf')))):
                values.extend((float(v),-float(v)));scales.extend((s,s))
    values.extend((0.,1.,-1.));scales.extend((0.,0.,0.))
    return torch.tensor(values,dtype=torch.float32),torch.tensor(scales,dtype=torch.float16),exact_ties


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cpu-only',action='store_true')
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    paths=('scripts/state_ppl_codec_v7.py','scripts/check_state_ppl_codec_v7.py','scripts/state_ppl_codec_v6.py','scripts/check_state_ppl_codec_v6.py','mamba2_recall/state_codec.py','mamba2_recall/state_quant.py')
    report=dict(format='MAMBA2_STATE_PPL_V7_CODEC_CPU_PREPARATION_V1' if args.cpu_only else 'MAMBA2_STATE_PPL_V7_CODEC_CHECK_V1',
        complete=False,passed=False,mode='cpu_preparation' if args.cpu_only else 'gpu',
        protocol_sha256=PROTOCOL_SHA,code_sha256={p:sha(ROOT/p) for p in paths},checks=[],started_unix=time.time(),
        criteria={'nonuniform':'Nearest actual FP32 decoded level via exact FP64 comparison; ties larger magnitude; zero scale code0',
            'uniform':'Unchanged v6 stored_scale delegation, FP32 quotient then half-away',
            'controlled':'Exact one-step and65-token packed codes/scales/carry/readout; sparse readout in65-token fixture',
            'random':'Exact full/segmented/tokenwise output and final cache; INT8 remains bitwise unchanged across codebooks',
            'storage':'52B/row and28,499,968B production persistent cache; no resident codebook'})
    def check(name,condition,**facts):
        row=dict(name=name,**{'pass':bool(condition)},**facts);report['checks'].append(row)
        print(json.dumps(row),flush=True)
        if not condition:raise AssertionError(name)
    try:
        torch.set_num_threads(4)
        check('frozen-protocol',sha(ROOT/'docs/STATE_PPL_V7_PROTOCOL.md')==PROTOCOL_SHA)
        from state_ppl_codec_v7 import CODEBOOK_BITS,allocate_state,quantize_probe,scan,validate_finite_result
        expected_bits={book:[struct.unpack('<I',struct.pack('<f',float(v)))[0] for v in levels(book)] for book in BOOKS}
        check('fixed-codebook-FP32-bits',CODEBOOK_BITS==expected_bits,codebook_bits=expected_bits)
        if not args.cpu_only:
            if not torch.cuda.is_available():raise RuntimeError('GPU checks require CUDA')
            report['gpu']=torch.cuda.get_device_name()
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        for book in BOOKS[1:]:
            values,scales,ties=boundary_fixture(book)
            expected=nearest_cpu(values,scales,book)
            reconstructed=levels(book)[expected.abs()]*scales.float()
            reconstructed=torch.where(expected<0,-reconstructed,reconstructed)
            check('CPU-signed-midpoint-scope-'+book,bool(torch.isfinite(reconstructed).all()) and ties>0,
                values=values.numel(),exact_positive_midpoints=ties)
            if not args.cpu_only:
                q,decoded=quantize_probe(values.cuda(),scales.cuda(),book)
                check('GPU-signed-midpoints-zero-subnormal-'+book,torch.equal(q.cpu(),expected) and torch.equal(decoded.cpu(),reconstructed),
                    values=values.numel(),codes_exact=torch.equal(q.cpu(),expected),decoded_exact=torch.equal(decoded.cpu(),reconstructed))
        for special in('ties','negative','zero','rounded_scales','underflow','subnormal','large_finite'):
            inputs,table=controlled(1,special)
            for book in BOOKS:
                expected=oracle(inputs,table,book)
                check('CPU-one-step-finite-'+special+'-'+book,bool(torch.isfinite(expected['decoded']).all()))
                if args.cpu_only:continue
                actual=run([v.cuda() for v in inputs],table.cuda(),book,[1],capture=True)
                check('GPU-one-step-exact-'+special+'-'+book,torch.equal(actual['output'],expected['output'])
                    and same_cache(actual['caches'][0],expected['caches'][0]) and torch.equal(actual['decoded'],expected['decoded']))
        inputs,table=controlled(65)
        inputs[4].zero_()
        for t in range(65):
            for g in range(table.shape[0]):inputs[4][:,t,g,int(table[g,(0,15,16,79,80,127)[t%6]])]=1/16
        for book in BOOKS:
            expected=oracle(inputs,table,book)
            check('CPU-65token-finite-'+book,bool(torch.isfinite(expected['decoded']).all()))
            if args.cpu_only:continue
            actual=run([v.cuda() for v in inputs],table.cuda(),book,[1]*65,capture=True)
            check('GPU-65token-packed-oracle-exact-'+book,torch.equal(actual['output'],expected['output'])
                and all(same_cache(a,b) for a,b in zip(actual['caches'],expected['caches']))
                and torch.equal(actual['decoded'],expected['decoded']))
        if not args.cpu_only:
            from state_ppl_codec_v6 import scan as v6_scan
            for geometry in((2,17,8,19,2),(1,9,6,3,3),(2,5,4,1,1)):
                inputs,table=random_inputs(geometry);b,length,h,p,g=geometry
                gpu=[v.cuda() for v in inputs];permutation=table.cuda();results={}
                for book in BOOKS:
                    full=run(gpu,permutation,book,[length])
                    chunk=run(gpu,permutation,book,[1,3,length-4])
                    token=run(gpu,permutation,book,[1]*length,capture=True);results[book]=token
                    check('partition-'+str(geometry)+'-'+book,torch.equal(full['output'],chunk['output']) and torch.equal(full['output'],token['output'])
                        and same_cache(full['final'],chunk['final']) and same_cache(full['final'],token['final']))
                    check('actual-packed-bytes-'+str(geometry)+'-'+book,full['bytes']==b*h*p*52,actual_bytes=full['bytes'],row_bytes=52)
                    valid_codes=all(bool(((cache['q4']&15)!=8).all() and ((cache['q4']>>4)!=8).all()) for cache in token['caches'])
                    check('signed-index-range-'+str(geometry)+'-'+book,valid_codes)
                check('INT8-unchanged-'+str(geometry),all(all(all(torch.equal(a[k],b[k]) for k in('lo','hi','s8'))
                    for a,b in zip(results['uniform']['caches'],results[book]['caches'])) for book in BOOKS[1:]))
                state=allocate_state(b,h,p,128,'cuda')
                output,_=v6_scan(*gpu,state,permutation,scale_mode='stored_scale',int4_clip=1.)
                check('uniform-delegates-v6-'+str(geometry),torch.equal(output.cpu(),results['uniform']['output'])
                    and same_cache({k:v.cpu() for k,v in state.tensors.items()},results['uniform']['final']))
            state=allocate_state(1,128,64,128,'cuda')
            total=56*state.nbytes+56*10240*4*2+56*8*128
            check('actual-production-cache',total==28499968,row_bytes=52,ssm_bytes=56*state.nbytes,
                convolution_bytes=56*10240*4*2,permutation_bytes=57344,resident_codebook_bytes=0,total_bytes=total)
            inputs,table=controlled(1);inputs[0].fill_(65504.);inputs[3].fill_(65504.);inputs[4].zero_();inputs[5].zero_()
            for book in BOOKS:
                state=allocate_state(2,6,5,128,'cuda');y,_=scan(*[v.cuda() for v in inputs],state,table.cuda(),codebook=book)
                rejected=False
                try:validate_finite_result(state,y)
                except FloatingPointError:rejected=True
                check('reject-nonfinite-scale-'+book,rejected)
        report['cuda_initialized']=torch.cuda.is_initialized()
        if args.cpu_only:check('CUDA-remained-uninitialized',not report['cuda_initialized'])
        report.update(complete=True,passed=True)
    except BaseException as exc:
        report['error']=repr(exc);raise
    finally:
        report['finished_unix']=time.time()
        args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
