#!/usr/bin/env python3
"""Independent byte/FP16-scale oracle for same-byte variable state tiers.

Controlled dyadic and sparse-readout cases require exact codes/scales/carry
and FP16 readout every token. Random cases require exact partition replay,
not CPU transcendental/reduction equivalence. Actual packed evidence is saved
for independent CPU reconstruction; no model weights/quality are loaded.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
PROTOCOL_SHA='565707cf401b2ab472b9b78368a7192afeebe73056efd811b7acba610e3d9567'
LAYOUTS=('16_64_48','8_80_40','24_48_56','32_32_64')
# This oracle does not import production counts, quantization, packing or decode.
COUNTS={'16_64_48':(16,64,48),'8_80_40':(8,80,40),'24_48_56':(24,48,56),'32_32_64':(32,32,64)}
SPECIALS=('ties','negative','zero','rounded_scales','underflow','subnormal','mixed_scales','large_finite')
GEOMETRIES=((2,17,8,19,2),(1,9,6,3,3),(2,5,4,1,1))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def same(a,b):return set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a)
def hashes(buffers):return {k:hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k,v in buffers.items()}


def round_cpu(z):
    magnitude=z.abs();whole=magnitude.floor()
    return z.sign()*(whole+((magnitude-whole)>=.5).to(z.dtype))


def pack_cpu(q):
    result=torch.zeros((*q.shape[:-1],q.shape[-1]//2),dtype=torch.uint8)
    for j in range(q.shape[-1]):
        value=(q[...,j].to(torch.int64)&15).byte()
        result[...,j//2] |= value << (4*(j%2))
    return result


def decode_cpu(buffers,layout):
    n8,n4,_=COUNTS[layout]
    output=torch.zeros((*buffers['s8'].shape,128),dtype=torch.float32)
    for j in range(n8):
        low=(buffers['lo'][...,j//2].long()>>(4*(j%2)))&15
        high=(buffers['hi'][...,j//2].long()>>(4*(j%2)))&15
        q=(high<<4)|low;q=torch.where(q>=128,q-256,q)
        output[...,j]=q.float()*buffers['s8'].float()
    for j in range(n4):
        q=(buffers['q4'][...,j//2].long()>>(4*(j%2)))&15
        q=torch.where(q>=8,q-16,q)
        output[...,n8+j]=q.float()*buffers['s4'].float()
    return output


def quantize_cpu(state,layout):
    n8,n4,_=COUNTS[layout];integers=[];scales=[]
    for values,limit in ((state[...,:n8],127),(state[...,n8:n8+n4],7)):
        scale=(values.abs().amax(-1)/limit).clamp_min(1e-8).half()
        safe=torch.where(scale>0,scale.float(),torch.ones_like(scale).float())
        q=round_cpu(values/safe[...,None]).clamp(-limit,limit).int()
        q=torch.where(scale[...,None]>0,q,0)
        integers.append(q);scales.append(scale)
    return dict(lo=pack_cpu(integers[0]),hi=pack_cpu(integers[0]>>4),q4=pack_cpu(integers[1]),s8=scales[0],s4=scales[1])


def oracle(inputs,table,layout,initial=None):
    n8,n4,_=COUNTS[layout]
    x,raw,A,B,C,D,bias=[v.detach().cpu().float() for v in inputs]
    b,L,H,P=x.shape;group=torch.arange(H)//(H//B.shape[2])
    indices=table.cpu().long()[group][None].expand(b,H,128)
    previous=torch.zeros((b,H,P,128),dtype=torch.float32)
    packed=quantize_cpu(previous,layout) if initial is None else {k:v.clone() for k,v in initial.items()}
    outputs=[];caches=[];decoded=[]
    for t in range(L):
        previous=decode_cpu(packed,layout)  # Reload ACTUAL persisted bytes/scales each token.
        dt=raw[:,t]+bias;dt=torch.where(dt<=20.,torch.log(torch.exp(dt)+1.),dt)
        decay=torch.exp(A[None]*dt)
        bv=B[:,t][:,group].gather(-1,indices);cv=C[:,t][:,group].gather(-1,indices)
        update=previous*decay[:,:,None,None]+(bv*dt[...,None])[:,:,None]*x[:,t,:,:,None]
        y=(update[...,:n8]*cv[:,:,None,:n8]).sum(-1)
        y=y+(update[...,n8:n8+n4]*cv[:,:,None,n8:n8+n4]).sum(-1)
        y=y+((update[...,n8+n4:]*cv[:,:,None,n8+n4:]).sum(-1)+x[:,t]*D[None,:,None])
        outputs.append(y.half());packed=quantize_cpu(update,layout)
        caches.append({k:v.clone() for k,v in packed.items()});decoded.append(decode_cpu(packed,layout))
    return dict(output=torch.stack(outputs,1),caches=caches,decoded=torch.stack(decoded))


def controlled(layout,length=1,special='ties',dim=5):
    n8,n4,nzero=COUNTS[layout];b,H,G=2,6,3
    table=torch.stack([torch.roll(torch.arange(127,-1,-1),9*g) for g in range(G)]).byte()
    x=torch.empty((b,length,H,dim),dtype=torch.float16)
    dt=torch.zeros((b,length,H),dtype=torch.float16);A=torch.zeros(H,dtype=torch.float32)
    bias=torch.full((H,),32.,dtype=torch.float32)
    D=torch.tensor([(-1.)**h/16 for h in range(H)],dtype=torch.float16)
    B=torch.empty((b,length,G,128),dtype=torch.float16);C=torch.zeros_like(B)
    first=torch.tensor([-127.,-126.5,-64.,-2.,-1.5,-.5,0.,.5,1.5,2.,63.,64.,126.,126.5,127.,0.])
    second=torch.tensor([-7.,-6.5,-4.,-1.5,-.5,0.,.5,1.5,4.,6.5,7.])
    wanted=torch.cat((first.repeat((n8+15)//16)[:n8],second.repeat((n4+10)//11)[:n4],((torch.arange(nzero)%5)-2).float()))
    if special=='negative':wanted=-wanted.abs()
    if special=='zero':wanted.zero_()
    if special=='rounded_scales':wanted[0]=127.0625;wanted[n8]=7.00390625
    if special=='underflow':wanted=wanted.sign()*2**-19
    if special=='subnormal':wanted=wanted.sign()*2**-11
    if special=='mixed_scales':wanted[:n8]=wanted[:n8].sign()*2**-19;wanted[n8:]=wanted[n8:].sign()*2**-15
    if special=='large_finite':wanted=wanted.sign()*131072.;wanted[n8+n4:]=0.
    boundary=(0,n8-1,n8,n8+n4-1,n8+n4,127,n8+1)
    for t in range(length):
        sign=(1.,1.,-1.,-1.)[t%4]
        for batch in range(b):
            for head in range(H):
                for p in range(dim):x[batch,t,head,p]=(-1.)**(batch+head+p)*2.**((t//4+batch+head+p)%3-6)
            for g in range(G):
                B[batch,t,g,table[g].long()]=(wanted*sign/32).half()
                C[batch,t,g,int(table[g,boundary[t%len(boundary)]])]=(-1.)**(batch+g)/16
    if special in ('underflow','subnormal','mixed_scales'):x.fill_(2**-5)
    if special=='large_finite':x.fill_(1.);C.zero_();D.zero_()
    return [x,dt,A,B,C,D,bias],table


def random_inputs(geometry):
    b,L,H,P,G=geometry;gen=torch.Generator().manual_seed(2026092910)
    values=[(torch.randn(b,L,H,P,generator=gen)*.08).half(),(torch.randn(b,L,H,generator=gen)*.2-2).half(),
        -torch.rand(H,generator=gen).float()-.1,(torch.randn(b,L,G,128,generator=gen)*.2).half(),
        (torch.randn(b,L,G,128,generator=gen)*.2).half(),(torch.randn(H,generator=gen)*.1).half(),torch.zeros(H,dtype=torch.float16)]
    table=torch.stack([torch.randperm(128,generator=gen) for _ in range(G)]).byte()
    return values,table


def poisoned_initial(layout,b=2,H=6,P=19):
    n8,n4,_=COUNTS[layout]
    row=torch.arange(b*H*P).reshape(b,H,P,1)
    q8=((row*31+torch.arange(n8)*17)%255-127).int();q8[...,0]=127
    q4=((row*7+torch.arange(n4)*5)%15-7).int();q4[...,0]=7
    s8=(2.**((row[...,0]%4).float()-4)).half();s4=(2.**((row[...,0]%3).float()-3)).half()
    return dict(lo=pack_cpu(q8),hi=pack_cpu(q8>>4),q4=pack_cpu(q4),s8=s8,s4=s4)


def run(inputs,table,layout,pieces,capture=False,initial=None,state=None):
    from state_ppl_codec_v10 import allocate_state,scan,decode_state,validate_finite_result
    b,L,H,P=inputs[0].shape
    if sum(pieces)!=L:raise ValueError('Partition does not span input')
    state=allocate_state(b,H,P,128,'cuda',layout=layout) if state is None else state
    if initial is not None:
        for key,value in initial.items():state.tensors[key].copy_(value)
    outputs=[];caches=[];decoded=[];start=0
    for count in pieces:
        part=[v[:,start:start+count] if i in (0,1,3,4) else v for i,v in enumerate(inputs)]
        out,unused=scan(*part,state,table,layout=layout)
        if unused is not None:raise AssertionError('Unexpected workspace/statistics output')
        validate_finite_result(state,out);outputs.append(out.cpu())
        if capture:
            caches.append({k:v.cpu().clone() for k,v in state.tensors.items()})
            decoded.append(decode_state(state,layout=layout).cpu())
        start+=count
    result=dict(output=torch.cat(outputs,1),final={k:v.cpu().clone() for k,v in state.tensors.items()},bytes=state.nbytes)
    if capture:result.update(caches=caches,decoded=torch.stack(decoded))
    return result


def scalar_fixture(limit):
    values=[];scales=[]
    for scale in (0.,2.**-24,2.**-14,.037933349609375,1.,65504.):
        for quotient in (-limit-.5,-3.5,-1.5,-.5,.5,1.5,3.5,limit+.5):
            center=torch.tensor(scale*quotient,dtype=torch.float32)
            for value in (torch.nextafter(center,torch.tensor(float('-inf'))),center,torch.nextafter(center,torch.tensor(float('inf')))):
                values.append(float(value));scales.append(scale)
    values.extend((0.,1.,-1.));scales.extend((0.,0.,0.))
    return torch.tensor(values,dtype=torch.float32),torch.tensor(scales,dtype=torch.float16)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--cpu-only',action='store_true')
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    evidence_path=args.out.with_name(args.out.stem+'_evidence.pt')
    if evidence_path.exists():raise FileExistsError(evidence_path)
    paths=('scripts/state_ppl_codec_v10.py','scripts/check_state_ppl_codec_v10.py','scripts/state_ppl_codec_v6.py',
        'mamba2_recall/state_codec.py','mamba2_recall/state_quant.py')
    report=dict(format='MAMBA2_STATE_PPL_V10_CODEC_CPU_PREPARATION_V1' if args.cpu_only else 'MAMBA2_STATE_PPL_V10_CODEC_CHECK_V1',
        complete=False,passed=False,mode='cpu_preparation' if args.cpu_only else 'gpu',protocol_sha256=PROTOCOL_SHA,
        code_sha256={p:sha(ROOT/p) for p in paths},checks=[],started_unix=time.time(),criteria=dict(
            baseline='Direct unchanged v6 stored_scale clip1 delegation; bitwise output/cache and unchanged cache_breakdown dictionary',
            controlled='Exact FP16 readout plus raw packed bytes/FP16 scales/decoded carry every token in one-step,65-token,poisoned-neighbor and dead-impulse fixtures',
            random='Exact full/segmented/tokenwise output and final cache; no CPU transcendental/reduction equality assertion',
            storage='Exact unpadded lo/hi/q4 widths plus two FP16 scales,52B/row;56layer actual allocation28499968B; zero resident layout tensors',
            failure='Nonfinite scales/readout rejected; malformed padded allocations rejected'))
    evidence=dict(format='MAMBA2_STATE_PPL_V10_CODEC_EVIDENCE_V1',protocol_sha256=PROTOCOL_SHA,code_sha256=report['code_sha256'],cases=[])
    def check(name,condition,**facts):
        row=dict(name=name,**{'pass':bool(condition)},**facts);report['checks'].append(row);print(json.dumps(row),flush=True)
        if not condition:raise AssertionError(name)
    def exact_case(name,inputs,table,layout,initial=None):
        expected=oracle(inputs,table,layout,initial)
        check('CPU-finite-'+name,bool(torch.isfinite(expected['output']).all() and torch.isfinite(expected['decoded']).all()))
        if args.cpu_only:return
        actual=run([v.cuda() for v in inputs],table.cuda(),layout,[1]*inputs[0].shape[1],capture=True,initial=initial)
        evidence['cases'].append(dict(name=name,layout=layout,inputs=inputs,table=table,initial=initial,
            output=actual['output'],caches=actual['caches']))
        output_equal=torch.equal(actual['output'],expected['output'])
        cache_equal=all(same(a,b) for a,b in zip(actual['caches'],expected['caches']))
        decoded_equal=torch.equal(actual['decoded'],expected['decoded'])
        check('GPU-oracle-exact-'+name,output_equal and cache_equal and decoded_equal,
            tokens=inputs[0].shape[1],readout_exact=output_equal,packed_every_token_exact=cache_equal,
            decoded_every_token_exact=decoded_equal,final_actual_sha256=hashes(actual['final']),
            max_readout_error=float((actual['output'].float()-expected['output'].float()).abs().max()))
        return actual
    try:
        torch.set_num_threads(4)
        check('frozen-protocol',sha(ROOT/'docs/STATE_PPL_V10_PROTOCOL.md')==PROTOCOL_SHA)
        from state_ppl_codec_v10 import LAYOUTS as actual_layouts,COUNTS as actual_counts,allocate_state,scan,validate_state,validate_finite_result,StatePPLQuantV10
        check('fixed-layout-grid',tuple(actual_layouts)==LAYOUTS and actual_counts==COUNTS,layouts=COUNTS)
        if not args.cpu_only:
            if not torch.cuda.is_available():raise RuntimeError('GPU receipt requires CUDA')
            report['gpu']=torch.cuda.get_device_name();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        for limit in (127,7):
            values,scales=scalar_fixture(limit);safe=torch.where(scales>0,scales.float(),torch.ones_like(scales).float())
            expected=round_cpu(values/safe).clamp(-limit,limit).int();expected=torch.where(scales>0,expected,0)
            check('CPU-signed-half-boundaries-'+str(limit),bool((expected.abs()<=limit).all()),values=len(values))
            if not args.cpu_only:
                from state_ppl_codec_v6 import quantize_stored_probe
                actual=quantize_stored_probe(values.cuda(),scales.cuda(),limit).cpu()
                check('GPU-shared-rounding-boundaries-'+str(limit),torch.equal(actual,expected),actual_codes=actual.tolist(),
                    values_fp32_bits=values.view(torch.int32).long().bitwise_and(0xffffffff).tolist(),
                    scales_fp16_bits=scales.view(torch.int16).int().bitwise_and(0xffff).tolist())
        for layout in LAYOUTS:
            n8,n4,nzero=COUNTS[layout]
            cpu=allocate_state(2,6,19,128,'cpu',layout=layout)
            check('CPU-actual-unpadded-storage-'+layout,validate_state(cpu,'cpu',layout) and cpu.nbytes==2*6*19*52,
                shapes={k:list(v.shape) for k,v in cpu.tensors.items()},storage_bytes={k:v.untyped_storage().nbytes() for k,v in cpu.tensors.items()})
            bad=cpu.clone();bad.tensors['q4']=torch.zeros((2,6,19,n4//2+1),dtype=torch.uint8)
            rejected=False
            try:validate_state(bad,'cpu',layout)
            except ValueError:rejected=True
            check('CPU-reject-padded-payload-'+layout,rejected)
            for special in SPECIALS:
                inputs,table=controlled(layout,1,special)
                exact_case('one-'+special+'-'+layout,inputs,table,layout)
            inputs,table=controlled(layout,65)
            actual=exact_case('65token-'+layout,inputs,table,layout)
            if not args.cpu_only:
                full=run([v.cuda() for v in inputs],table.cuda(),layout,[65]);chunk=run([v.cuda() for v in inputs],table.cuda(),layout,[1,16,48])
                check('controlled65-partitions-'+layout,torch.equal(full['output'],actual['output']) and torch.equal(chunk['output'],actual['output'])
                    and same(full['final'],actual['final']) and same(chunk['final'],actual['final']))
            inputs,table=controlled(layout,7,dim=19);inputs[0].zero_();inputs[3].zero_();inputs[5].zero_()
            exact_case('poisoned-neighbor-boundaries-'+layout,inputs,table,layout,poisoned_initial(layout))
            inputs,table=controlled(layout,2);inputs[0].fill_(1.);inputs[3].zero_();inputs[4].zero_();inputs[5].zero_()
            for group in range(3):
                pos=int(table[group,n8+n4]);inputs[3][:,0,group,pos]=1/32;inputs[4][:,:,group,pos]=1.
            actual=exact_case('dead-current-readout-zero-carry-'+layout,inputs,table,layout)
            if not args.cpu_only:
                check('dead-pulse-current-only-'+layout,bool((actual['output'][:,0]==1).all() and (actual['output'][:,1]==0).all())
                    and all(not bool(v.any()) for cache in actual['caches'] for v in cache.values()))
        if not args.cpu_only:
            from state_ppl_codec_v6 import scan as v6_scan
            for geometry in GEOMETRIES:
                inputs,table=random_inputs(geometry);b,L,H,P,G=geometry;gpu=[v.cuda() for v in inputs];permutation=table.cuda()
                for layout in LAYOUTS:
                    full=run(gpu,permutation,layout,[L]);chunk=run(gpu,permutation,layout,[1,3,L-4]);token=run(gpu,permutation,layout,[1]*L,capture=True)
                    check('random-partitions-'+str(geometry)+'-'+layout,torch.equal(full['output'],chunk['output']) and torch.equal(full['output'],token['output'])
                        and same(full['final'],chunk['final']) and same(full['final'],token['final']))
                    check('random-actual-bytes-'+str(geometry)+'-'+layout,full['bytes']==b*H*P*52,actual_bytes=full['bytes'])
                    valid=all(bool(((c['q4']&15)!=8).all() and ((c['q4']>>4)!=8).all()) for c in token['caches'])
                    check('random-signed-INT4-range-'+str(geometry)+'-'+layout,valid)
                    state=allocate_state(b,H,P,128,'cuda',layout=layout)
                    run(gpu,permutation,layout,[L],state=state);state.zero_();reset=run(gpu,permutation,layout,[L],state=state)
                    check('random-zero-reset-replay-'+str(geometry)+'-'+layout,torch.equal(reset['output'],full['output']) and same(reset['final'],full['final']))
                    if layout==LAYOUTS[0]:
                        state=allocate_state(b,H,P,128,'cuda',layout=layout)
                        y,_=v6_scan(*gpu,state,permutation,scale_mode='stored_scale',int4_clip=1.)
                        check('baseline-delegates-v6-'+str(geometry),torch.equal(y.cpu(),full['output'])
                            and same({k:v.cpu() for k,v in state.tensors.items()},full['final']))
            for layout in LAYOUTS:
                # Exercise actual controller.reset allocation without loading a model.
                controller=object.__new__(StatePPLQuantV10)
                controller.layout=layout;controller.device=torch.device('cuda:0');controller._mixers=[None]*56
                controller._installed=True;controller._cache=[];controller.mode='sq3p25'
                controller.scale_mode='stored_scale';controller.int4_clip=1.;controller.diagnostic=None
                controller.permutations=torch.arange(128,dtype=torch.uint8,device='cuda').expand(56,8,128).clone()
                controller.reset(1);descriptor=controller.storage_descriptor();cache=controller.cache_breakdown()
                expected_widths={'lo':COUNTS[layout][0]//2,'hi':COUNTS[layout][0]//2,'q4':COUNTS[layout][1]//2}
                physical=len(descriptor['layers'])==56 and all(row['state_bytes']==128*64*52 and row['conv_storage_bytes']==10240*4*2
                    and all(row['tensors'][key]['shape']==[1,128,64,width] and row['tensors'][key]['storage_bytes']==128*64*width
                            for key,width in expected_widths.items()) for row in descriptor['layers'])
                check('actual-production-allocation-'+layout,physical and cache['total_bytes']==28499968 and cache['ssm_total_bytes']==23855104
                    and cache['conv_fp16_bytes']==4587520 and cache['permutation_bytes']==57344,
                    storage_descriptor=descriptor,cache=cache)
                if layout==LAYOUTS[0]:
                    from state_ppl_codec_v6 import StatePPLQuant
                    check('baseline-cache-dictionary-delegates-v6',cache==StatePPLQuant.cache_breakdown(controller))
                for row in controller._cache:
                    for value in row.state.tensors.values():value.fill_(1)
                controller.reset(1)
                check('controller-reset-clears-all-state-'+layout,all(not bool(v.any()) for row in controller._cache for v in row.state.tensors.values())
                    and controller.cache_breakdown()==cache)
                controller.clear();del controller
                inputs,table=controlled(layout,1);inputs[0].fill_(65504.);inputs[3].fill_(65504.);inputs[4].zero_();inputs[5].zero_()
                state=allocate_state(2,6,5,128,'cuda',layout=layout);output,_=scan(*[v.cuda() for v in inputs],state,table.cuda(),layout=layout)
                rejected=False
                try:validate_finite_result(state,output)
                except FloatingPointError:rejected=True
                check('reject-nonfinite-FP16-scale-'+layout,rejected)
        report['cuda_initialized']=torch.cuda.is_initialized()
        if args.cpu_only:check('CUDA-remained-uninitialized',not report['cuda_initialized'])
        report.update(complete=True,passed=True)
    except BaseException as error:
        report['error']=repr(error);raise
    finally:
        if evidence['cases']:
            torch.save(evidence,evidence_path)
            report['evidence']=dict(file=str(evidence_path),sha256=sha(evidence_path),bytes=evidence_path.stat().st_size,cases=len(evidence['cases']))
        report['finished_unix']=time.time();args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
