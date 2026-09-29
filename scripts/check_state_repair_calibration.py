#!/usr/bin/env python3
"""Collector equality and independent score checks; synthetic data, no quality claim."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from mamba2_recall import runtime,state_codec as codec
from mamba2_recall.state_repair_calibration import readout_collect_scan,equalizer_exponents


def check_case(b,h,p,g,length):
    x=torch.randn(b,length,h,p,device='cuda',dtype=torch.float16)*.1
    dt=torch.randn(b,length,h,device='cuda',dtype=torch.float16)*.1-2
    A=-torch.rand(h,device='cuda',dtype=torch.float32)*.3-.1
    B=torch.randn(b,length,g,128,device='cuda',dtype=torch.float16)*.1
    C=torch.randn_like(B)*.1
    D=torch.randn(h,device='cuda',dtype=torch.float16)*.1
    bias=torch.randn(h,device='cuda',dtype=torch.float16)*.1
    initial=codec.allocate_state(b,h,p,128,'s16','cuda')
    initial.tensors['state'].copy_(torch.randn_like(initial.tensors['state'])*.1)
    original,new=initial.clone(),initial.clone()
    old_y,old_abs=codec.scan(x,dt,A,B,C,D,bias,original,collect_stats=True)
    new_y,new_abs,score=readout_collect_scan(x,dt,A,B,C,D,bias,new)
    if not torch.equal(old_y,new_y) or not torch.equal(original.tensors['state'],new.tensors['state']):
        raise RuntimeError('Collector changed original S16 output or carried state')
    if not torch.equal(old_abs,new_abs):
        raise RuntimeError('Collector mean-absolute-state arithmetic differs')
    segmented_old,segmented_new=initial.clone(),initial.clone()
    pieces=[];pieces_original=[];seg_score=[]
    starts=sorted(set((0,min(1,length),min(7,length),min(19,length),length)))
    for lo,hi in zip(starts,starts[1:]):
        args=(x[:,lo:hi],dt[:,lo:hi],A,B[:,lo:hi],C[:,lo:hi],D,bias)
        yo,ao=codec.scan(*args,segmented_old,collect_stats=True)
        yn,an,sn=readout_collect_scan(*args,segmented_new)
        if not torch.equal(yo,yn) or not torch.equal(ao,an) or not torch.equal(
            segmented_old.tensors['state'],segmented_new.tensors['state']):
            raise RuntimeError('Segmented collector differs from original S16')
        pieces.append(yn);pieces_original.append(yo);seg_score.append(sn.double())
    if not torch.equal(torch.cat(pieces,1),new_y) or not torch.equal(segmented_new.tensors['state'],new.tensors['state']):
        raise RuntimeError('S16 chunk boundaries changed collector recurrence')
    if not torch.allclose(sum(seg_score),score.double(),rtol=2e-5,atol=2e-8):
        raise RuntimeError('Chunked score accumulation exceeds FP32 summation tolerance')
    # Independent arithmetic oracle using prior states from original per-token
    # S16 scan. Score sums themselves use PyTorch FP64 reduction, not Triton.
    reference=initial.clone();oracle=torch.zeros(b,h,p,128,device='cuda',dtype=torch.float64)
    head_groups=torch.arange(h,device='cuda')//(h//g)
    for t in range(length):
        raw=dt[:,t].float()+bias.float()
        delta=torch.where(raw<=20,torch.log(torch.exp(raw)+1),raw)
        decay=torch.exp(A[None]*delta)
        term=(reference.tensors['state'].float()*decay[:,:,None,None])*C[:,t,head_groups,:].float()[:,:,None,:]
        oracle+=term.double().square()
        codec.scan(x[:,t:t+1],dt[:,t:t+1],A,B[:,t:t+1],C[:,t:t+1],D,bias,reference)
    expected=oracle.sum(2);actual=score.double().sum(2)
    if not torch.allclose(actual,expected,rtol=2e-5,atol=2e-8):
        raise RuntimeError('Readout-damage scores differ from independent one-token S16 oracle')
    return dict(batch=b,heads=h,P=p,groups=g,tokens=length,output_exact=True,carry_exact=True,
        abs_stats_exact=True,segmented_exact=True,score_oracle_close=True,
        score_max_absolute_error=float((actual-expected).abs().max()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(4);torch.manual_seed(2026092901);torch.cuda.manual_seed_all(2026092901)
    means=torch.ones(56,8,128,dtype=torch.float64)
    if bool(equalizer_exponents(means).any()):raise RuntimeError('Uniform equalizer is not identity')
    means[:,:,:32]=2.**-12;means[:,:,96:]=2.**12
    expected=torch.zeros(56,8,128,dtype=torch.int8);expected[:,:,:32]=-8;expected[:,:,96:]=8
    if not torch.equal(equalizer_exponents(means),expected):raise RuntimeError('Equalizer rounding/clamp differs')
    rows=[check_case(*shape) for shape in ((1,4,5,2,1),(2,8,29,2,33),(1,128,64,8,64))]
    receipt=dict(format='MAMBA2_STATE_REPAIR_COLLECTOR_CHECKS_V1',complete=True,passed=True,
        code_sha256={str(path.relative_to(ROOT)):runtime.sha256_file(path) for path in
            (Path(__file__),ROOT/'mamba2_recall/state_repair_calibration.py',ROOT/'mamba2_recall/state_codec.py')},
        equalizer_identity_and_clamp_passed=True,cases=rows,
        scope='Synthetic exact S16 collector output/carry/abs-statistics and independent approximate FP32 readout-score oracle; no model-quality claim')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':main()
