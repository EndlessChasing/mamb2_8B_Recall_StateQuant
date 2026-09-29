#!/usr/bin/env python3
"""Independent CPU audit for v6's PPL-only state-policy experiment.

Reconstructs fixed input tables, raw NLL arithmetic, candidate eligibility,
PPL-only selection, resident storage and exact replay. No GPU execution and no
MK measurements/selection are performed by this auditor.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from audit_quant_first import (sha,need,finite,close,tokhash,read,SOURCE,TOKENIZER,
    PROSE_FILE,PROSE_MANIFEST,VALIDATION_TOKENS)
from audit_state_repair import hash_inventory,frozen,tensor_sha
from audit_state_first_v5 import audit_candidates as audit_v5_candidates
from audit_resurface_more import audit_backend_policy

PROTOCOL='86d4e8dc85d79c2c867ce15a846c5938893c86d27e67ad174472975e654e9bff'
KINDS=('magnitude','full_readout','preserve_int8','preserve_retained80')
VARIANTS=('legacy','stored_scale','clip4_095','clip4_090','clip4_080')
IDS=tuple(kind+'__'+variant for kind in KINDS for variant in VARIANTS)
BASELINE='preserve_int8__legacy'
CACHE=28499968
S16_CACHE=122028032
DIAGNOSTIC_CACHE=239525888
V5_CANDIDATES='cd86a755db5004c716922696cf5532b307c57f5fb7dad4bcca298eb58b55f9a7'
V5_SELECTED='c525fbf62ef4a72db2d4bb13c13920d9f5d4946485538da0f00a369aab3092e3'
V5_SELECTED_RECEIPT='482399518e5b41561d93cf1db2509949d27f522b1a720bea1e6e10a3864c77d6'
V5_SCREEN='a7ff703281309d83031d4b3bc3b4c6edf718e8646fa62506ff162dbf1cd62437'
V5_FULL_COMPARISON='0e5ceb6e91d72a159f46a9a0760a23f3b9301be18bb32590fe01a9196f83d1e9'
V5_BASELINE='0f5412aac14801c7f5691df00308ea0c8ff1734f9d06bced276f38f871b58f69'
S16_ARCHIVE='52f82f83258a2fa3160ea14585f1bd69e1d68d60d8f179d1f0987c636f6546ba'
BOUND_FORMULA='2*(gamma21*T8+gamma69*T4+gamma69*Td+gamma3*abs(xD)+4096*2^-126*(1+abs(x))*(1+sum(abs(C))))+r16(a)+r16(b)'
BOUND_GUARD='multiply by1+gamma1024(u64), then nextafter(+inf)'


def half_radius(value):
    need(math.isfinite(value) and abs(value)<=65504.,'Bound requires finite FP16 readouts')
    return max(2.**-25,math.ldexp(1.,math.frexp(abs(value))[1]-12)) if value else 2.**-25


def diagnostic_bound(t8,t4,td,xd,x,cabs,a,b):
    """Scalar CPU arithmetic, independent of the checker implementation."""
    need(all(math.isfinite(v) and v>=0 for v in (t8,t4,td,xd,x,cabs)),
         'Forward-error magnitudes must be nonnegative and finite')
    u=2.**-24
    gamma=lambda n:n*u/(1.-n*u)
    error=math.fsum((gamma(21)*t8,gamma(69)*t4,gamma(69)*td,gamma(3)*xd,
                    4096.*2.**-126*(1.+x)*(1.+cabs)))
    raw=math.fsum((2.*error,half_radius(a),half_radius(b)))
    upward=1.+1024.*2.**-53/(1.-1024.*2.**-53)
    return math.nextafter(raw*upward,math.inf)


def select_independent(rows):
    """Pure ranking arithmetic. Integrity proofs are separately audited per arm."""
    valid=[];excluded=[]
    for name in IDS:
        row=rows[name]
        if (row.get('complete') is True and 'error' not in row
                and finite(row.get('ppl',{}).get('ppl'))
                and row.get('cache',{}).get('total_bytes')==CACHE):
            valid.append(name)
        else:excluded.append(name)
    need(BASELINE in valid,'Invalid baseline must stop, never choose another candidate')
    winner=min(valid,key=lambda name:(rows[name]['ppl']['ppl'],name!=BASELINE,IDS.index(name)))
    kind,variant=winner.split('__')
    return dict(selected_id=winner,selected_kind=kind,selected_variant=variant,valid=valid,excluded=excluded,
        stopped=winner==BASELINE,adapter_used=False,heldout_used=False,mk_used=False,baseline_id=BASELINE,
        rule='Fixed20 candidates; complete finite exact-budget; lowest TRAIN PPL, baseline priority, fixed table/variant order; diagnostics excluded')


def audit_ppl(result,windows,complete=True):
    need('mk' not in result,'PPL-only result unexpectedly includes an MK measurement')
    raw=result.get('ppl',{});records=raw.get('windows',[])
    need(len(records)<=len(windows) and (not complete or len(records)==len(windows)),
         'PPL window coverage differs')
    total=0.;targets=0
    for record,(start,ids) in zip(records,windows):
        count=len(ids)-1
        need(record['start']==start and record['target_tokens']==count
             and record['token_sha256_int64le']==tokhash(ids)
             and finite(record['nll']) and record['nll']>=0,'PPL window identity/NLL differs')
        close(record['ppl'],math.exp(record['nll']/count),'Window exponent arithmetic differs')
        total+=record['nll'];targets+=count
    if records:
        need(raw['target_tokens']==targets,'Aggregate target count differs')
        close(raw['nll'],total,'Aggregate NLL is not sum of raw windows')
        close(raw['ppl'],math.exp(total/targets),'PPL must exponentiate aggregate per-target NLL')
    return dict(complete=complete,windows=len(records),targets=targets,nll=total,
                ppl=math.exp(total/targets) if targets else None)


def replay_ppl(left,right):
    need(left.get('complete') is True and right.get('complete') is True
         and left['ppl']['windows']==right['ppl']['windows'],'Replay raw windows differ')
    for key in ('nll','ppl','target_tokens'):
        need(left['ppl'][key]==right['ppl'][key],'Replay aggregate differs: '+key)
    need(left['cache']['total_bytes']==right['cache']['total_bytes'],'Replay cache budget differs')
    return dict(complete=True,per_window_nll_exact=True,ppl_windows_repeated=len(left['ppl']['windows']),
                ppl_target_tokens_repeated=left['ppl']['target_tokens'])


def self_test():
    def rows():
        return {name:dict(complete=True,ppl=dict(ppl=10.),cache=dict(total_bytes=CACHE)) for name in IDS}
    checks=[]
    r=rows();need(select_independent(r)['selected_id']==BASELINE,'Baseline must beat earlier candidate on exact tie')
    checks.append('baseline-preferred-over-earlier-table-on-tie')
    r[IDS[0]]['ppl']['ppl']=math.nextafter(10.,0.)
    need(select_independent(r)['selected_id']==IDS[0],'One ULP lower must beat baseline');checks.append('one-ulp-lower-wins')
    r[IDS[1]]['ppl']['ppl']=r[IDS[0]]['ppl']['ppl']
    need(select_independent(r)['selected_id']==IDS[0],'Fixed table/variant order must break nonbaseline tie')
    checks.append('fixed-order-breaks-nonbaseline-tie')
    for value,label in ((float('nan'),'nan'),(float('inf'),'infinity')):
        r=rows();r[IDS[0]]['ppl']['ppl']=value
        need(IDS[0] in select_independent(r)['excluded'],'Nonfinite candidate must exclude');checks.append(label+'-excludes')
    for key,value,label in (('complete',False,'incomplete'),('error','failure','recorded-error')):
        r=rows();r[IDS[0]][key]=value
        need(IDS[0] in select_independent(r)['excluded'],'Failed candidate must exclude');checks.append(label+'-excludes')
    r=rows();r[IDS[0]]['cache']['total_bytes']+=1
    need(IDS[0] in select_independent(r)['excluded'],'One extra byte must exclude');checks.append('one-byte-over-budget')
    r=rows();r['diagnostic_s16']=dict(complete=True,ppl=dict(ppl=1.),cache=dict(total_bytes=S16_CACHE))
    need(select_independent(r)['selected_id']==BASELINE,'Diagnostic must never enter selection');checks.append('diagnostics-ineligible')
    r=rows()
    for name in IDS:
        if name!=BASELINE:r[name]['complete']=False
    need(select_independent(r)['stopped'] is True,'All failed candidates must preserve baseline');checks.append('baseline-only-fallback')
    r=rows();r[BASELINE]['complete']=False
    try:select_independent(r)
    except ValueError:pass
    else:raise ValueError('Failed baseline did not stop')
    checks.append('failed-baseline-fatal')
    need(32*2047==65504 and 16+64//2+4==52
         and 56*128*64*52+56*10240*4*2+57344==CACHE
         and 56*128*64*256+56*10240*4*2==S16_CACHE
         and 56*128*64*512+56*10240*4*2+57344==DIAGNOSTIC_CACHE,
         'Data/physical storage accounting differs');checks.append('target-count-and-three-actual-cache-budgets')
    need(half_radius(0.)==half_radius(2.**-24)==2.**-25
         and half_radius(1.)==2.**-11 and half_radius(-1.)==2.**-11
         and half_radius(65504.)==16.,'FP16 rounding-cell radii differ')
    checks.append('fp16-rounding-cells-zero-subnormal-binade-maxfinite')
    bound=diagnostic_bound(1.,1.,1.,0.,0.,1.,0.,0.)
    need(bound>2.**-24 and abs(0.-0.)<=bound,'Cancellation-safe absolute bound differs')
    need(abs(1.-2.)>diagnostic_bound(0.,0.,0.,0.,0.,0.,1.,2.),
         'Diagnostic bound accepts a macroscopic output mutation')
    checks.append('diagnostic-bound-cancellation-and-mutation-rejection')
    return dict(complete=True,passed=True,count=len(checks),checks=checks,gpu_used=False)


def audit_inputs(args,tokenizer,torch,np):
    need(sha(ROOT/'docs/STATE_PPL_V6_PROTOCOL.md')==PROTOCOL,'Frozen v6 protocol changed')
    need(sha(args.candidates)==V5_CANDIDATES and sha(args.v5_calibration)==V5_SELECTED
         and sha(args.v5_calibration.with_suffix('.json'))==V5_SELECTED_RECEIPT,'Pinned v5 inputs changed')
    train,candidates,receipt,upstream=audit_v5_candidates(args,tokenizer,torch,np)
    old=torch.load(args.v5_calibration,map_location='cpu',weights_only=True)
    old_receipt=read(args.v5_calibration.with_suffix('.json'))
    historical_path=args.v5_calibration.parent/'screen_comparison.json'
    need(sha(historical_path)==V5_SCREEN,'Pinned historical v5 selection receipt differs')
    historical=read(historical_path)
    # Historical artifacts are hashed for provenance only: no MK scoring or selection is recomputed here.
    for name,digest in historical['report_sha256'].items():
        need(sha(historical_path.parent/('screen_'+name+'.json'))==digest,'Historical v5 arm changed')
    expected=dict(format='MAMBA2_STATE_FIRST_CALIBRATION_V1',selected_kind='preserve_int8',
        candidates_sha256=V5_CANDIDATES,selection_report_sha256=V5_SCREEN,
        table_sha256=receipt['table_sha256']['preserve_int8'])
    need(all(old.get(k)==old_receipt.get(k)==v for k,v in expected.items())
         and torch.equal(old['permutations'],candidates['tables']['preserve_int8']),
         'v5 baseline table binding differs')
    need(old_receipt.get('complete') is True and old_receipt['sha256']==V5_SELECTED
         and old_receipt['bytes']==args.v5_calibration.stat().st_size,'v5 baseline receipt differs')
    return train,candidates,receipt,dict(complete=True,candidate_tables_independently_reconstructed=True,
        v5_candidates_sha256=V5_CANDIDATES,v5_baseline_calibration_sha256=V5_SELECTED,
        historical_MK_scope='Only frozen historical file hashes are checked; no historical MK scores affect v6 selection.',
        upstream=upstream)


def evidence_path(record):
    path=Path(record['file'])
    if not path.is_absolute():path=ROOT/path
    need(path.is_file() and sha(path)==record['sha256'],'Codec evidence file missing or changed')
    if 'bytes' in record:need(path.stat().st_size==record['bytes'],'Codec evidence byte count differs')
    return path


def scalar_ptx_independent(ptx_path,source_path,function):
    source=source_path.read_text();lines=source.splitlines()
    nodes=[node for node in ast.walk(ast.parse(source)) if isinstance(node,ast.FunctionDef) and node.name==function]
    need(len(nodes)==1,'Scalar source function identity differs')
    node=nodes[0]
    raw,soft,decay=(next(i+1 for i in range(node.lineno-1,node.end_lineno) if text in lines[i])
        for text in ('dt = tl.load(DT','dt = tl.where(dt <=','decay = tl.exp('))
    selected=[];location=None;scalar_context=False
    for rawline in ptx_path.read_text().splitlines():
        match=re.search(r'\.loc\s+1\s+(\d+)\s+(\d+)',rawline)
        if match:
            location=tuple(map(int,match.groups()))
            scalar_context=location[0] in (soft,decay) or (location[0]==0 and scalar_context)
        text=rawline.strip()
        if not text or text.startswith(('.', '//', '$')) or location is None:continue
        number,_=location
        if scalar_context or (number==raw and text.startswith('add.f32')):
            selected.append(text)
    registers={}
    def rename(match):return registers.setdefault(match.group(),'%v'+str(len(registers)))
    canonical=re.sub(r'%(?:rs|rd|r|p)\d+',rename,'\n'.join(selected))
    canonical=' '.join(canonical.split())
    need(len(selected)==47,'Expected nonempty47 scalar arithmetic instructions')
    return canonical,hashlib.sha256(canonical.encode()).hexdigest()


def audit_diagnostic_evidence(row,geometry):
    import numpy as np
    import torch
    need(row['bound_formula']==BOUND_FORMULA and row['float64_upward_guard']==BOUND_GUARD,
         'Diagnostic readout bound rule changed')
    proof=row['scalar_ptx_equivalence']
    need(proof['exact'] is True and set(proof['artifacts'])=={'packed','dense','scalar_probe'},
         'Common actual-scalar PTX proof missing')
    canonical=[]
    for name,source,function in (
        ('packed','mamba2_recall/state_codec.py','_sq_scan'),
        ('dense','scripts/state_ppl_codec_v6.py','_variant_scan'),
        ('scalar_probe','scripts/check_state_ppl_codec_v6.py','scalar_probe')):
        artifact=proof['artifacts'][name]
        text,digest=scalar_ptx_independent(evidence_path(artifact),ROOT/source,function)
        need(artifact['instructions']==47 and artifact['scalar_sha256']==digest,
             'Scalar PTX canonical hash/count differs')
        canonical.append(text)
    need(len(set(canonical))==1,'Captured scalar math differs from either actual recurrence kernel')
    payload_path=evidence_path(row['evidence'])
    payload=torch.load(payload_path,map_location='cpu',weights_only=True)
    batch,length,heads,dim,groups=geometry
    rng=torch.Generator().manual_seed(2026092906)
    expected=[(torch.randn(batch,length,heads,dim,generator=rng)*.08).half(),
        (torch.randn(batch,length,heads,generator=rng)*.2-2).half(),
        -torch.rand(heads,generator=rng).float()-.1,
        (torch.randn(batch,length,groups,128,generator=rng)*.2).half(),
        (torch.randn(batch,length,groups,128,generator=rng)*.2).half(),
        (torch.randn(heads,generator=rng)*.1).half(),torch.zeros(heads,dtype=torch.float16)]
    table=torch.stack([torch.randperm(128,generator=rng) for _ in range(groups)]).byte()
    need(len(payload['inputs'])==7 and all(a.dtype==b.dtype and torch.equal(a,b)
        for a,b in zip(payload['inputs'],expected)) and torch.equal(payload['table'],table),
        'Fixed random diagnostic fixture inputs/table changed')
    for key in ('dt_actual','decay_actual'):
        value=payload[key]
        need(value.dtype==torch.float32 and tuple(value.shape)==(batch,length,heads)
             and bool(torch.isfinite(value).all()) and bool((value>0).all()),
             'Finite positive actual scalar capture geometry differs')
    shape=(batch,heads,dim,128)
    need(len(payload['packed_caches'])==len(payload['dense_caches'])==length,
         'Every-token packed/dense state evidence required')
    decoded=[]
    for packed,dense in zip(payload['packed_caches'],payload['dense_caches']):
        need(set(packed)=={'lo','hi','q4','s8','s4'} and set(dense)=={'dense'},
             'Random diagnostic actual state representation differs')
        for key,width in (('lo',8),('hi',8),('q4',32)):
            need(packed[key].dtype==torch.uint8 and tuple(packed[key].shape)==shape[:-1]+(width,),
                 'Packed code dtype/shape differs')
        for key in ('s8','s4'):
            need(packed[key].dtype==torch.float16 and tuple(packed[key].shape)==shape[:-1]
                 and bool(torch.isfinite(packed[key]).all()) and bool((packed[key]>=0).all()),
                 'Stored scale dtype/shape/finiteness differs')
        n8=np.arange(16);n4=np.arange(64)
        lo=packed['lo'].numpy().astype(np.int16);hi=packed['hi'].numpy().astype(np.int16)
        q4=packed['q4'].numpy().astype(np.int16)
        code8=((hi[...,n8//2]>>(n8%2*4))&15)*16+((lo[...,n8//2]>>(n8%2*4))&15)
        code8=np.where(code8>127,code8-256,code8).astype(np.float32)
        code4=(q4[...,n4//2]>>(n4%2*4))&15
        code4=np.where(code4>7,code4-16,code4).astype(np.float32)
        value=np.concatenate((code8*packed['s8'].numpy().astype(np.float32)[...,None],
            code4*packed['s4'].numpy().astype(np.float32)[...,None],np.zeros(shape[:-1]+(48,),np.float32)),-1)
        need(dense['dense'].dtype==torch.float32 and tuple(dense['dense'].shape)==shape
             and np.array_equal(value.view(np.uint32),dense['dense'].numpy().view(np.uint32)),
             'Decoded packed bytes/scales and dense carry differ at a token')
        decoded.append(value)
    carry=np.stack(decoded)
    for key in ('packed_carry','dense_carry'):
        value=payload[key]
        need(value.dtype==torch.float32 and tuple(value.shape)==(length,)+shape
             and np.array_equal(value.numpy().view(np.uint32),carry.view(np.uint32)),
             'Recorded carry does not equal independent byte/scale decoding')
    outshape=(batch,length,heads,dim)
    for key in ('packed_output','dense_output'):
        value=payload[key]
        need(value.dtype==torch.float16 and tuple(value.shape)==outshape and bool(torch.isfinite(value).all()),
             'Finite diagnostic FP16 output geometry differs')
    x,_,_,B,C,D,_=[v.numpy().astype(np.float64) for v in expected]
    delta=payload['dt_actual'].numpy().astype(np.float64);decay=payload['decay_actual'].numpy().astype(np.float64)
    a=payload['packed_output'].numpy().astype(np.float64);b=payload['dense_output'].numpy().astype(np.float64)
    bounds=np.empty(outshape,np.float64)
    for ib in range(batch):
        for t in range(length):
            for h in range(heads):
                group=h//(heads//groups);order=table[group].numpy()
                bv=B[ib,t,group,order];cv=C[ib,t,group,order]
                for p in range(dim):
                    previous=carry[t-1,ib,h,p].astype(np.float64) if t else np.zeros(128,np.float64)
                    terms=np.abs(cv)*(np.abs(previous*decay[ib,t,h])+np.abs(bv*delta[ib,t,h]*x[ib,t,h,p]))
                    sums=[math.fsum(terms[part]) for part in (slice(16),slice(16,80),slice(80,None))]
                    bounds[ib,t,h,p]=diagnostic_bound(*sums,abs(x[ib,t,h,p]*D[h]),abs(x[ib,t,h,p]),
                        math.fsum(abs(cv)),a[ib,t,h,p],b[ib,t,h,p])
    recorded=payload['bound']
    need(recorded.dtype==torch.float64 and tuple(recorded.shape)==outshape
         and np.allclose(recorded.numpy(),bounds,rtol=4e-15,atol=0.),
         'Recorded elementwise bound differs from independent FP64 reconstruction')
    difference=np.abs(a-b);failures=int(np.count_nonzero(difference>bounds))
    need(failures==0 and row['bound_failures']==0 and row['all_token_carries_exact'] is True
         and row['finite_scope'] is True,'Random diagnostic recurrence/output bound failed')
    close(row['max_bound'],float(bounds.max()),'Maximum reconstructed readout bound differs',tol=1e-13)
    close(row['max_error_to_bound_ratio'],float((difference/bounds).max()),'Output-to-bound ratio differs',tol=1e-13)
    err=row['output_error'];f32diff=payload['packed_output'].float()-payload['dense_output'].float()
    expected_error=dict(exact=bool(np.array_equal(a,b)),different_elements=int(np.count_nonzero(difference)),
        max_absolute_error=float(difference.max()),relative_l2=float(torch.linalg.vector_norm(f32diff)/
            (torch.linalg.vector_norm(payload['dense_output'].float())+1e-30)))
    need(err==expected_error,'Raw random diagnostic output error differs')
    return dict(geometry=list(geometry),elements=int(a.size),all_token_carries_bitwise_exact=True,
        elementwise_bound_independently_recomputed=True,bound_failures=failures,
        max_error_to_bound_ratio=float((difference/bounds).max()),evidence_sha256=sha(payload_path),
        common_scalar_ptx_independently_verified=True)


def audit_kernel_receipt(path):
    receipt=read(path)
    need(receipt.get('format')=='MAMBA2_STATE_PPL_CODEC_CHECK_V1' and receipt.get('complete') is True
         and receipt.get('passed') is True and receipt.get('mode')=='gpu' and receipt.get('cuda_initialized') is True
         and receipt.get('protocol_sha256')==PROTOCOL and 'error' not in receipt,
         'Actual completed GPU codec checks required; CPU preparation cannot pass')
    hash_inventory(receipt['code_sha256'],{'scripts/state_ppl_codec_v6.py','scripts/check_state_ppl_codec_v6.py',
        'mamba2_recall/state_codec.py','mamba2_recall/state_quant.py'},'GPU codec checks')
    criteria=dict(controlled='exact every-token output/packed bytes/FP16 scales/decoded carry',
        random_partition='exact full/segmented/tokenwise outputs and final cache',
        random_one_token_cpu_output_relative_l2_bound=.002,
        stored_scale_division='Correctly rounded FP32 quotient before half-away rounding; zero-scale integer0',
        stored_scale_denominator='FP32 clipping multiplication then correctly rounded FP32 division before FP16 round-to-even storage',
        diagnostic_legacy_emulation='controlled output exact; random every-token carry exact and readout within recurrence-inclusive gamma21/69/69/3 plus FP16-cell and FP32-underflow bound')
    need(receipt['criteria']==criteria,'Frozen codec validation criteria differ')
    policies=(*VARIANTS,'prune_only','quant_only','legacy_emulation')
    specials=('ties','negative','zero','rounded_scales','underflow','subnormal','mixed_scales','large_finite')
    geometries=((2,17,8,19,2),(1,9,6,3,3),(2,5,4,1,1))
    expected={'frozen_protocol','reject-unfrozen-legacy-clipping-policy','CPU-half-away-near-boundary',
              'GPU-stored-scale-half-boundaries-7','GPU-stored-scale-half-boundaries-127'}
    for special in specials:
        for policy in policies:
            if special=='large_finite' and policy=='prune_only':continue
            expected.add('CPU-fixture-finite-'+special+'-'+policy)
            expected.add('GPU-independent-exact-'+special+'-'+policy)
    expected.update('CPU-zero-scale-integer-zero-'+name for name in VARIANTS if name!='legacy')
    expected.update('CPU-positive-subnormal-scale-'+name for name in VARIANTS)
    for name in policies:
        expected.update(('CPU-multistep-finite-'+name,'GPU-65token-independent-exact-'+name,
                         'reject-nonfinite-carry-or-scale-'+name))
    for geometry in geometries:
        for name in policies:
            expected.update(prefix+str(geometry)+'-'+name for prefix in ('partition-','actual-bytes-','random-one-token-oracle-'))
        expected.update(prefix+str(geometry) for prefix in ('dense-legacy-emulation-','delegated-legacy-','INT8-unchanged-by-INT4-clipping-'))
    expected.update('production-budget-'+str(d) for d in (None,'prune_only','quant_only','legacy_emulation'))
    checks=receipt['checks'];names=[row['name'] for row in checks]
    need(len(names)==len(expected) and set(names)==expected and all(row.get('pass') is True for row in checks),
         'GPU codec checks missing, duplicated, unexpected or failed')
    for row in checks:
        name=row['name']
        if name.startswith(('GPU-independent-exact-','GPU-65token-independent-exact-')):
            for key in ('output_error','carry_error'):
                need(row[key]==dict(exact=True,different_elements=0,max_absolute_error=0.,relative_l2=0.),
                     'Strict independent CPU/GPU oracle error is not zero')
    byname={row['name']:row for row in checks}
    boundary_codes=[1,0,-1,0,4,3,-4,-3,4,3,-4,-3,0,0,0]
    for limit in (7,127):
        boundary=byname['GPU-stored-scale-half-boundaries-'+str(limit)]
        need(boundary['actual_codes']==boundary['expected_codes']==boundary_codes,
             'Stored-scale positive/negative half/nextafter/zero-scale regression differs')
    for geometry in geometries:
        batch,_,heads,dim,_=geometry
        for name in policies:
            row_bytes=52 if name in VARIANTS else 512
            allocated=byname['actual-bytes-'+str(geometry)+'-'+name]
            need(allocated['row_bytes']==row_bytes and allocated['state_bytes']==batch*heads*dim*row_bytes
                 and allocated['is_3p25_candidate']==(name in VARIANTS),'Synthetic actual byte accounting differs')
            e=byname['random-one-token-oracle-'+str(geometry)+'-'+name]
            need(type(e['exact']) is bool and type(e['different_elements']) is int and e['different_elements']>=0
                 and finite(e['max_absolute_error']) and e['max_absolute_error']>=0
                 and finite(e['relative_l2']) and 0<=e['relative_l2']<=.002,'Random one-token output oracle bound differs')
    diagnostic_proofs=[audit_diagnostic_evidence(byname['dense-legacy-emulation-'+str(geometry)],geometry)
                       for geometry in geometries]
    for diagnostic in (None,'prune_only','quant_only','legacy_emulation'):
        row=byname['production-budget-'+str(diagnostic)];state=56*128*64*(52 if diagnostic is None else 512)
        need(row['actual_state_bytes']==state and row['table_bytes']==57344
             and row['total_bytes']==(CACHE if diagnostic is None else DIAGNOSTIC_CACHE)
             and row['is_3p25_candidate']==(diagnostic is None),'Measured production-state allocation differs')
    return dict(complete=True,sha256=sha(path),checks=len(checks),actual_GPU_checks=True,
        criteria=criteria,diagnostic_bound_checks=diagnostic_proofs,
        scope='Independent CPU audit of recorded GPU oracle/partition/bytes receipts and raw per-token diagnostic forward-error bound; no GPU rerun by auditor.')


def run_paths():
    return {str(path.relative_to(ROOT)) for path in list((ROOT/'mamba2_recall').glob('*.py'))+[
        ROOT/'scripts'/name for name in ('run_state_ppl_v6.py','state_ppl_codec_v6.py','check_state_ppl_codec_v6.py',
        'prepare_state_first_v5.py','prepare_quant_first.py','evaluate_quant_first.py','evaluate_resurface_more.py',
        'run_statequant.py')]+[ROOT/'docs/STATE_PPL_V6_PROTOCOL.md',ROOT/'docs/RESURFACE_MORE_BACKEND_REPLAY.md']}


def input_binding(args,receipt):
    return dict(protocol_sha256=PROTOCOL,v5_protocol_sha256='ce700910f27230e062fe62e89dd6bddee05bcbf7bb7972c59fdba24cb3db9f5d',
        source_sha256=SOURCE,tokenizer_sha256=TOKENIZER,train_file_sha256=PROSE_FILE,prose_manifest_sha256=PROSE_MANIFEST,
        candidates_sha256=V5_CANDIDATES,candidates_receipt_sha256=sha(args.candidates.with_suffix('.json')),
        v5_selected_calibration_sha256=V5_SELECTED,v5_selected_receipt_sha256=V5_SELECTED_RECEIPT,
        v5_selection_report_sha256=V5_SCREEN,codec_checks_sha256=sha(args.kernel_checks),
        table_sha256=receipt['table_sha256'],v4_statistics_sha256=sha(args.repair_calibration),
        original_calibration_sha256=sha(args.calibration))


DIAGNOSTICS=('diagnostic_s16','diagnostic_prune_only','diagnostic_quant_only')
SCREEN_ARMS=(BASELINE,*DIAGNOSTICS,*(name for name in IDS if name!=BASELINE),'restored_baseline')
FULL_ARMS=('v5_baseline','selected','restored_baseline')


def spec_for(arm,candidate_id=None):
    if arm in DIAGNOSTICS:
        diagnostic=arm[len('diagnostic_'):]
        return dict(candidate_id=None,candidate_name=None if diagnostic=='s16' else 'preserve_int8',
            variant=None,scale_mode='legacy',int4_clip=1.,diagnostic=diagnostic,deployable=False,
            selection_eligible=False,diagnostic_scope='TRAIN-only decomposition; excluded from candidate selection and Q3.25 claims')
    name=candidate_id if candidate_id is not None else BASELINE if arm in ('restored_baseline','v5_baseline') else arm
    need(name in IDS,'Unknown frozen candidate identity')
    kind,variant=name.split('__')
    clip={'legacy':1.,'stored_scale':1.,'clip4_095':.95,'clip4_090':.90,'clip4_080':.8}[variant]
    return dict(candidate_id=name,candidate_name=kind,variant=variant,
        scale_mode='legacy' if variant=='legacy' else 'stored_scale',int4_clip=clip,diagnostic=None,deployable=True)


def audit_cache(cache,spec,tokens):
    diagnostic=spec['diagnostic'];is_s16=diagnostic=='s16'
    rows=56*128*64;conv=56*10240*4*2
    state=rows*(256 if is_s16 else 512 if diagnostic else 52)
    scales=0 if diagnostic else rows*4;table=0 if is_s16 else 57344
    expected=dict(mode='s16' if is_s16 else 'diagnostic_'+diagnostic if diagnostic else 'sq3p25',
        batch_size=1,allocated_layers=56,conv_fp16_bytes=conv,ssm_payload_bytes=state-scales,
        ssm_scale_bytes=scales,ssm_total_bytes=state,permutation_bytes=table,total_bytes=conv+state+table,
        calibration_workspace_bytes=0,tokens_per_layer=[tokens]*56)
    if not is_s16:
        expected.update(scale_mode=spec['scale_mode'],int4_clip=spec['int4_clip'],diagnostic=diagnostic,
            is_3p25_candidate=diagnostic is None,diagnostic_dense_fp32_bytes=state if diagnostic else 0,
            row_bytes=512 if diagnostic else 52)
    need(all(cache.get(k)==v for k,v in expected.items()),'Actual allocation/dtype policy accounting differs')
    return expected['total_bytes']


def compare_independent(left,right):
    a,b=left['ppl']['windows'],right['ppl']['windows']
    need(len(a)==len(b) and all(all(x[k]==y[k] for k in ('start','target_tokens','token_sha256_int64le'))
         for x,y in zip(a,b)),'PPL comparisons do not use matching windows')
    return dict(control_arm=left['arm'],candidate_arm=right['arm'],control_ppl=left['ppl']['ppl'],
        candidate_ppl=right['ppl']['ppl'],ppl_relative_change=right['ppl']['ppl']/left['ppl']['ppl']-1,
        nll_delta=right['ppl']['nll']-left['ppl']['nll'],
        improved_windows=sum(y['nll']<x['nll'] for x,y in zip(a,b)),
        regressed_windows=sum(y['nll']>x['nll'] for x,y in zip(a,b)),
        unchanged_windows=sum(y['nll']==x['nll'] for x,y in zip(a,b)))


def exact_restoration(left,right,receipt):
    replay_ppl(left,right)
    need(left['repeated_reset_probe']==right['repeated_reset_probe'] and left['cache']==right['cache'],
         'Exact full reset/cache receipt replay differs')
    expected=dict(complete=True,per_window_nll_exact=True,ppl_windows_repeated=len(left['ppl']['windows']),
        target_tokens=left['ppl']['target_tokens'],reset_hidden_exact=True,
        reset_persistent_tensor_hashes_exact=True,cache_allocation_exact=True)
    need(receipt==expected,'Restoration receipt differs')
    return expected


def audit_selected(path,args,candidates,binding,screen,torch):
    payload=torch.load(path,map_location='cpu',weights_only=True);receipt=read(path.with_suffix('.json'))
    choice=screen['selection'];name=choice['selected_kind']
    expected=dict(format='MAMBA2_STATE_PPL_CALIBRATION_V1',input_binding=binding,protocol_sha256=PROTOCOL,
        selected_id=choice['selected_id'],selected_kind=name,selected_variant=choice['selected_variant'],
        table_sha256=binding['table_sha256'][name],selection_report_sha256=screen['comparison_sha256'],
        stopped=choice['stopped'],adapter_loaded=False,adapter_sha256=None,heldout_used=False,mk_used=False)
    need(all(payload.get(k)==receipt.get(k)==v for k,v in expected.items()),'Selected policy/table provenance differs')
    need(receipt.get('complete') is True and receipt['file']==path.name and receipt['sha256']==sha(path)
         and receipt['bytes']==path.stat().st_size,'Selected artifact identity differs')
    need(payload['permutations'].dtype==torch.uint8 and tuple(payload['permutations'].shape)==(56,8,128)
         and torch.equal(payload['permutations'],candidates['tables'][name])
         and tensor_sha(payload['permutations'])==expected['table_sha256'],'Selected actual coordinates differ')
    hash_inventory(receipt['code_hashes'],run_paths(),'Selected policy export')
    return payload,dict(complete=True,sha256=sha(path),receipt_sha256=sha(path.with_suffix('.json')),
        selected_id=choice['selected_id'],table_sha256=expected['table_sha256'],stopped=choice['stopped'])


def audit_run(directory,stage,args,train,candidates,candidate_receipt,np,
              screen=None,selected=None,validation=None,dataset=None):
    comparison_path=directory/(stage+'_comparison.json');comparison=read(comparison_path)
    binding=input_binding(args,candidate_receipt)
    common=dict(format='MAMBA2_STATE_PPL_EVAL_V1',stage=stage,protocol_sha256=PROTOCOL,input_binding=binding,
        adapter_loaded=False,adapter_sha256=None,mk_used=False,heldout_used=stage=='full',heldout_used_for_selection=False,
        selection_scope='Unadapted PPL-only fixed TRAIN screen; full validation confirms a frozen table/variant',
        selected_calibration_sha256=sha(args.selected_calibration) if selected is not None else None,
        parent_report_sha256=V5_BASELINE if stage=='full' else None,
        parent_comparison_sha256=V5_FULL_COMPARISON if stage=='full' else None,
        s16_report_sha256=S16_ARCHIVE if stage=='full' else None)
    need(comparison.get('format')=='MAMBA2_STATE_PPL_COMPARISON_V1' and comparison.get('complete') is True
         and all(comparison.get(k)==v for k,v in common.items() if k in ('stage','protocol_sha256','input_binding',
             'adapter_loaded','adapter_sha256','mk_used','heldout_used','heldout_used_for_selection'))
         and 'mk' not in comparison,'PPL-only comparison provenance differs')
    hash_inventory(comparison['code_hashes'],run_paths(),'PPL evaluation')
    if stage=='screen':
        windows=[(i*2048,train[i,:2048].tolist()) for i in range(40,72)]
        expected_dataset=dict(split='train',file_sha256=PROSE_FILE,rows=list(range(40,72)),tokens_per_row=2048,
            windows=32,target_tokens=65504)
        arms=SCREEN_ARMS
    else:
        need(screen is not None and selected is not None and not screen['selection']['stopped'],'Full requires nonbaseline frozen selection')
        need(tokhash(validation.tolist())==VALIDATION_TOKENS,'Pinned validation token stream differs')
        windows=[(i,validation[i:min(i+2049,len(validation))].tolist()) for i in range(0,len(validation)-1,2048)]
        need(len(windows)==130 and sum(len(ids)-1 for _,ids in windows)==264764,'Full PPL population differs')
        expected_dataset=dataset;arms=FULL_ARMS
    common['dataset']=expected_dataset
    need(tuple(comparison['report_sha256'])==arms,'Fixed execution order/report inventory differs')
    outputs={};metrics={};digests={}
    for arm in arms:
        spec=spec_for(arm,selected['selected_id'] if arm=='selected' else None)
        path=directory/(stage+'_'+arm+'.json');row=read(path);digests[arm]=sha(path)
        table_sha=candidate_receipt['table_sha256'].get(spec['candidate_name'])
        need(digests[arm]==comparison['report_sha256'][arm] and row.get('arm')==arm
             and row.get('candidate_table_sha256')==table_sha
             and all(row.get(k)==v for k,v in {**common,**spec}.items()) and 'mk' not in row,
             'Arm identity, fixed policy, table, dataset or no-MK provenance differs: '+arm)
        need(row['code_hashes']==comparison['code_hashes'] and row.get('candidate_table_unchanged') is True,
             'Code or CPU table changed across arms')
        frozen(row['frozen_source']);audit_backend_policy(row['backend_policy'],row['backend_policy_check'])
        need(row['backend_policy']==comparison['backend_policy'],'Backend differs across arms')
        complete=row.get('complete') is True
        if complete:
            need('error' not in row and not row.get('excluded_from_selection',False) and not row.get('fatal_failure',False)
                 and row.get('runtime_table_unchanged') is True
                 and row.get('persistent_float_finite_checks_passed') is True,'Complete arm failed integrity/finiteness')
            probe=row['repeated_reset_probe'];audit_cache(probe['cache'],spec,128)
            keys=('state','conv') if spec['diagnostic']=='s16' else ('dense','conv') if spec['diagnostic'] else ('lo','hi','q4','s8','s4','conv')
            hashes=probe['cache_tensor_sha256']
            need(probe.get('tokens')==128 and probe.get('hidden_and_cache_exact') is True
                 and probe['token_sha256_int64le']==tokhash(windows[0][1][:128])
                 and re.fullmatch('[0-9a-f]{64}',probe['hidden_sha256']) is not None
                 and set(hashes)=={f'{i}.{k}' for i in range(56) for k in keys}
                 and all(isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) for v in hashes.values()),
                 'Repeated-reset probe token/hash/storage evidence differs')
            audit_cache(row['cache'],spec,len(windows[-1][1])-1)
            need(type(row['zero_scale_observations']) is int and row['zero_scale_observations']>=0
                 and row['zero_scale_scope']=='Final-cache observations of stored s4/s8; includes true zeros and scale underflow, not unique underflow events',
                 'Scale-zero accounting scope differs')
        else:
            allowed=('PPL exponent overflow','Nonfinite PPL','Nonfinite persisted state, FP16 scale, or convolution cache',
                     'Nonfinite first128-token hidden','Nonfinite repeated128-token hidden','Nonfinite PPL hidden','Nonfinite PPL loss')
            need(stage=='screen' and arm in IDS and arm!=BASELINE and row.get('excluded_from_selection') is True
                 and not row.get('fatal_failure',False)
                 and row.get('error_type')=='CandidateInvalid' and row.get('error') in {f'CandidateInvalid({message!r})' for message in allowed},
                 'Incomplete candidate is not a recognized nonfinite exclusion')
        metrics[arm]=audit_ppl(row,windows,complete);outputs[arm]=row
    baseline=BASELINE if stage=='screen' else 'v5_baseline'
    restoration=read(directory/(stage+'_restoration.json'))
    exact_restoration(outputs[baseline],outputs['restored_baseline'],restoration)
    need(comparison['restoration']==restoration,'Comparison restoration differs')
    result=dict(complete=True,arms=metrics,report_sha256=digests,restoration=restoration,
        comparison_sha256=sha(comparison_path),no_MK_measurements=True)
    if stage=='screen':
        chosen=select_independent(outputs)
        need(comparison['selection']==chosen and comparison['selected_table_sha256']==binding['table_sha256'][chosen['selected_kind']]
             and comparison.get('diagnostics_used_for_selection') is False,'Independent PPL-only selection differs')
        diagnostics={arm:compare_independent(outputs['diagnostic_s16'],outputs[arm]) for arm in DIAGNOSTICS[1:]}
        delta=compare_independent(outputs[BASELINE],outputs[chosen['selected_id']])
        baseline_gap=compare_independent(outputs['diagnostic_s16'],outputs[BASELINE])
        need(comparison['diagnostics']==diagnostics and comparison['selected_vs_baseline']==delta
             and comparison['baseline_vs_s16']==baseline_gap,'Independent diagnostic/window comparison arithmetic differs')
        result.update(selection=chosen,diagnostics=diagnostics,selected_vs_baseline=delta,baseline_vs_s16=baseline_gap)
    else:
        parent_path=args.v5_eval_dir/'full_selected_no_adapter.json'
        old_comparison_path=args.v5_eval_dir/'full_comparison.json'
        need(sha(parent_path)==V5_BASELINE and sha(old_comparison_path)==V5_FULL_COMPARISON
             and sha(args.s16_report)==S16_ARCHIVE,'Pinned archived baseline/comparison/S16 changed')
        historical=read(old_comparison_path)
        need(historical['report_sha256']['selected_no_adapter']==V5_BASELINE
             and historical['archived_report_sha256']['source_s16']==S16_ARCHIVE,'Archived context chain differs')
        parent=read(parent_path);s16=read(args.s16_report)
        # Only the PPL section of archived combined PPL/MK reports is evaluated.
        audit_ppl({'ppl':parent['ppl']},windows);audit_ppl({'ppl':s16['ppl']},windows)
        replay_ppl(parent,outputs['v5_baseline'])
        old_probe=parent['persistent_cache_probe'];new_probe=outputs['v5_baseline']['repeated_reset_probe']
        need(old_probe['hidden_sha256']==new_probe['hidden_sha256']
             and old_probe['cache_tensor_sha256']==new_probe['cache_tensor_sha256'],
             'Archived v5 baseline reset hidden/cache values differ')
        replay_record=read(directory/'full_parent_replay.json')
        expected_replay=dict(complete=True,per_window_nll_exact=True,ppl_windows_repeated=130,target_tokens=264764,
            reset_hidden_exact=True,reset_persistent_tensor_hashes_exact=True,cache_bytes_exact=True,mk_evaluated=False)
        need(replay_record==expected_replay and comparison['parent_replay']==replay_record,'Archived replay receipt differs')
        expected=dict(selected_id=selected['selected_id'],selected_kind=selected['selected_kind'],selected_variant=selected['selected_variant'],
            selection=screen['selection'],selected_calibration_sha256=sha(args.selected_calibration),
            selection_report_sha256=screen['comparison_sha256'],parent_report_sha256=V5_BASELINE,
            parent_comparison_sha256=V5_FULL_COMPARISON,s16_report_sha256=S16_ARCHIVE,resurface_trained=False,
            claim_scope='PPL and persistent-cache confirmation only; recall was not evaluated')
        need(all(comparison.get(k)==v for k,v in expected.items()),'Full frozen-candidate/context/no-training binding differs')
        delta=compare_independent(outputs['v5_baseline'],outputs['selected'])
        gap=compare_independent(s16,outputs['selected'])
        checks=dict(ppl_at_least_1pct_better=outputs['selected']['ppl']['ppl']<=.99*outputs['v5_baseline']['ppl']['ppl'],
            cache_same_budget=all(row['cache']['total_bytes']==CACHE for row in outputs.values()))
        need(comparison['comparison']==delta and comparison['original_s16_comparison']==gap
             and comparison['gate_checks']==checks and comparison['gate_pass']==all(checks.values()),
             'Full independent PPL arithmetic/gate differs')
        result.update(selection=screen['selection'],comparison=delta,original_s16_comparison=gap,
            gate_checks=checks,gate_pass=all(checks.values()),parent_replay=replay_record)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--stage',choices=('inputs','screen','full'))
    for name in ('source-dir','calibration','repair-calibration','parent-training-report','prose-tokens',
        'codec-checks','candidates','v5-calibration','kernel-checks','eval-dir','screening-report',
        'selected-calibration','v5-eval-dir','s16-report','output'):
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test(),indent=2));return
    for key in ('stage','source_dir','calibration','repair_calibration','parent_training_report','prose_tokens',
                'codec_checks','candidates','v5_calibration','kernel_checks','output'):
        if getattr(args,key) is None:parser.error('--'+key.replace('_','-')+' is required')
    if args.stage in ('screen','full') and args.eval_dir is None:parser.error('--eval-dir is required')
    if args.stage=='full':
        for key in ('screening_report','selected_calibration','v5_eval_dir','s16_report'):
            if getattr(args,key) is None:parser.error('--'+key.replace('_','-')+' is required')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Set CUDA_VISIBLE_DEVICES= for CPU-only audit')
    need(not args.output.exists(),'Use a fresh output path; preserve all earlier evidence')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1'
    result=dict(format='MAMBA2_STATE_PPL_INDEPENDENT_AUDIT_V1',complete=False,passed=False,
        stage=args.stage,protocol_sha256=PROTOCOL,source_sha256=sha(__file__),
        auditor_dependency_sha256={name:sha(ROOT/'scripts'/name) for name in
            ('audit_quant_first.py','audit_state_repair.py','audit_state_first_v5.py','audit_resurface_more.py')},
        limitations=['Raw GPU NLL evidence and arithmetic are audited; model logits are not regenerated.',
            'Recorded GPU codec tests are hash-bound; this CPU audit does not execute Triton kernels.',
            'Frozen-source identity/version/gradient guards do not replace post-run weight-byte hashes.',
            'Historical benchmark exposure remains; this is not untouched generalization evidence.'])
    try:
        import numpy as np
        import torch
        from mamba2_recall import runtime
        torch.set_num_threads(4)
        need(not torch.cuda.is_initialized(),'CUDA initialized before CPU audit')
        result['self_tests']=self_test()
        tokenizer=runtime.SentencePieceTokenizer(args.source_dir)
        need(tokenizer.sha256==TOKENIZER,'Tokenizer differs')
        train,candidates,receipt,proof=audit_inputs(args,tokenizer,torch,np)
        result['inputs']=proof
        result['kernel_checks']=audit_kernel_receipt(args.kernel_checks)
        if args.stage!='inputs':
            directory=args.eval_dir if args.stage=='screen' else args.screening_report.parent
            if args.stage=='full':need(args.screening_report.name=='screen_comparison.json','Frozen screen filename differs')
            screen=audit_run(directory,'screen',args,train,candidates,receipt,np)
            result['screen']=screen
            selected_path=directory/'selected_calibration.pt'
            if args.selected_calibration is None:args.selected_calibration=selected_path
            need(args.selected_calibration.resolve()==selected_path.resolve(),'Selected artifact must accompany frozen screen')
            selected,selected_proof=audit_selected(selected_path,args,candidates,input_binding(args,receipt),screen,torch)
            result['selected_calibration']=selected_proof
            if args.stage=='full':
                need(not screen['selection']['stopped'],'Baseline fallback cannot advance to full validation')
                from mamba2_recall.calibration import load_wikitext_tokens
                validation,dataset=load_wikitext_tokens(tokenizer,'validation')
                result['full']=audit_run(args.eval_dir,'full',args,train,candidates,receipt,np,screen,selected,validation,dataset)
        need(not torch.cuda.is_initialized(),'CPU audit unexpectedly initialized CUDA')
        result.update(complete=True,passed=True,cuda_initialized=False)
    except BaseException as error:
        result['error']=repr(error);raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
