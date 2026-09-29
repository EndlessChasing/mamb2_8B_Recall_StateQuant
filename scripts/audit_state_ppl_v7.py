#!/usr/bin/env python3
"""Independent CPU evidence audit for the fixed v7 nonuniform-state experiment.

Audits raw measurements and their frozen provenance. Does not execute a GPU,
regenerate logits, measure recall, or choose candidates from full validation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import audit_state_ppl_v6 as v6
from audit_quant_first import sha,need,finite,close,tokhash,read,TOKENIZER,PROSE_FILE,VALIDATION_TOKENS
from audit_state_repair import hash_inventory,frozen,tensor_sha
from audit_resurface_more import audit_backend_policy

PROTOCOL='84ffdce1d9e5d5dbac2b6996072f0db09bd1bf79fd3c55dad8b8f35973cd5d14'
KINDS=('magnitude','full_readout','preserve_int8','preserve_retained80')
BOOKS=('uniform','mild','quadratic','fp4like')
IDS=tuple(kind+'__'+book for kind in KINDS for book in BOOKS)
BASELINE='preserve_int8__uniform'
CACHE=28499968
V6_BASELINE='c3525d74ba0ee35e0b9d83a029e645123d14e90f5ca32b164931d24ec8359253'
V6_COMPARISON='c47e381e02edd6e33ba2b0a85a4f9c65d702cc7d3d93dacfe7546d409e6612d4'
V6_SELECTED='098930d1af5e5821b277640d236f7607c6428b48d36f7117e2ea4aa87e656303'
V6_SCREEN='5ac624862846b61aacc6a5f3b7648db32d5d74bf8419d8d536bbc751cdd117d6'


def fp32(value):return struct.unpack('<f',struct.pack('<f',value))[0]


def book_levels():
    return dict(uniform=[float(i) for i in range(8)],
        mild=[0.,.5,1.5,2.5,3.5,4.5,5.5,7.],
        quadratic=[fp32(i*i/7.) for i in range(8)],
        fp4like=[fp32(x) for x in (0.,7/12,7/6,7/4,7/3,7/2,14/3,7.)])


def book_bits():
    return {name:[struct.unpack('<I',struct.pack('<f',x))[0] for x in levels]
            for name,levels in book_levels().items()}


def decode4_independent(code,scale,book):
    need(type(code) is int and -7<=code<=7,'Reserved INT4 index -8 is not a valid emitted code')
    value=fp32(book_levels()[book][abs(code)]*scale)
    return -value if code<0 else value


def quantize4_independent(value,scale,book):
    need(book in BOOKS and math.isfinite(value) and math.isfinite(scale) and scale>=0,
         'Finite state/scale required by code selection oracle')
    if scale==0:return 0
    if book=='uniform':
        quotient=abs(fp32(value/scale));base=math.floor(quotient)
        index=min(7,base+int(quotient-base>=.5))
    else:
        levels=[fp32(x*scale) for x in book_levels()[book]]
        # Adjacent FP32 decoded levels have bounded exponent separation; their
        # sum and halving are exact in Python's IEEE binary64 arithmetic.
        index=sum(abs(value)>=(a+b)*.5 for a,b in zip(levels,levels[1:]))
    return -index if value<0 else index


def select_independent(rows):
    valid=[];excluded=[]
    for name in IDS:
        row=rows[name]
        if (row.get('complete') is True and 'error' not in row
            and finite(row.get('ppl',{}).get('ppl')) and row.get('cache',{}).get('total_bytes')==CACHE):
            valid.append(name)
        else:excluded.append(name)
    need(BASELINE in valid,'Invalid baseline is fatal')
    winner=min(valid,key=lambda name:(rows[name]['ppl']['ppl'],name!=BASELINE,IDS.index(name)))
    kind,book=winner.split('__')
    return dict(selected_id=winner,selected_kind=kind,selected_codebook=book,valid=valid,excluded=excluded,
        baseline_wins=winner==BASELINE,adapter_used=False,heldout_used=False,mk_used=False,baseline_id=BASELINE,
        rule='Minimum complete finite exact-budget TRAIN PPL; exact ties baseline then fixed candidate order')


def target_gate(ppl,cache_bytes):
    return dict(ppl_strictly_below_8p25=finite(ppl) and ppl<8.25,cache_same_budget=cache_bytes==CACHE)


def self_test():
    checks=[]
    def rows():return {name:dict(complete=True,ppl=dict(ppl=9.),cache=dict(total_bytes=CACHE)) for name in IDS}
    values=rows();need(select_independent(values)['selected_id']==BASELINE,'Baseline tie priority failed')
    checks.append('baseline-exact-tie-priority')
    values[IDS[0]]['ppl']['ppl']=math.nextafter(9.,0.)
    values[IDS[1]]['ppl']['ppl']=values[IDS[0]]['ppl']['ppl']
    need(select_independent(values)['selected_id']==IDS[0],'Lower score or fixed candidate order failed')
    checks.append('one-ulp-lower-and-fixed-nonbaseline-tie-order')
    for bad,label in ((float('nan'),'nan'),(float('inf'),'infinity')):
        values=rows();values[IDS[0]]['ppl']['ppl']=bad
        need(IDS[0] in select_independent(values)['excluded'],'Nonfinite candidate incorrectly eligible')
        checks.append(label+'-excluded')
    values=rows();values[IDS[0]]['cache']['total_bytes']+=1
    need(IDS[0] in select_independent(values)['excluded'],'Extra resident byte accepted')
    checks.append('one-byte-over-budget-excluded')
    values=rows();values[BASELINE]['complete']=False
    try:select_independent(values)
    except ValueError:pass
    else:raise ValueError('Failed baseline did not stop')
    checks.append('failed-baseline-fatal')
    need(not all(target_gate(8.25,CACHE).values())
         and all(target_gate(math.nextafter(8.25,0.),CACHE).values())
         and not all(target_gate(math.nextafter(8.25,math.inf),CACHE).values())
         and not all(target_gate(8.24,CACHE+1).values()),'Strict target or unchanged-cache gate differs')
    checks.append('strict-target-boundary-and-cache-gate')
    levels=book_levels()
    need(tuple(levels)==BOOKS and all(xs[0]==0 and xs[-1]==7 and all(a<b for a,b in zip(xs,xs[1:]))
        for xs in levels.values()),'Fixed level ordering/endpoints differ')
    for name in BOOKS:
        for code in range(-7,8):
            need(quantize4_independent(decode4_independent(code,1.,name),1.,name)==code,
                 'Signed code selection/decode roundtrip differs')
        need(quantize4_independent(1.,0.,name)==0 and quantize4_independent(-1.,0.,name)==0,
             'Zero scale did not force code zero')
    checks.append('fixed-level-bits-signed-code-roundtrip-zero-scale')
    try:decode4_independent(-8,1.,'mild')
    except ValueError:pass
    else:raise ValueError('Reserved -8 code accepted')
    checks.append('reserved-negative-eight-rejected')
    need(32*2047==65504 and 56*128*64*52+56*10240*4*2+57344==CACHE,
         'Token/cache budget arithmetic differs')
    checks.append('targets-and-physical-cache-budget')
    return dict(complete=True,passed=True,count=len(checks),checks=checks,gpu_used=False)


def run_paths():
    return v6.run_paths()|{'scripts/run_state_ppl_v7.py','scripts/state_ppl_codec_v7.py',
        'scripts/check_state_ppl_codec_v7.py','docs/STATE_PPL_V7_PROTOCOL.md'}


def audit_kernel_receipt(path,np):
    r=read(path)
    need(r.get('format')=='MAMBA2_STATE_PPL_V7_CODEC_CHECK_V1' and r.get('complete') is True
         and r.get('passed') is True and r.get('mode')=='gpu' and r.get('cuda_initialized') is True
         and r.get('protocol_sha256')==PROTOCOL and 'error' not in r,'Actual completed v7 GPU receipt required')
    hash_inventory(r['code_sha256'],{'scripts/state_ppl_codec_v7.py','scripts/check_state_ppl_codec_v7.py',
        'scripts/state_ppl_codec_v6.py','scripts/check_state_ppl_codec_v6.py',
        'mamba2_recall/state_codec.py','mamba2_recall/state_quant.py'},'v7 GPU kernel evidence')
    criteria=dict(nonuniform='Nearest actual FP32 decoded level via exact FP64 comparison; ties larger magnitude; zero scale code0',
        uniform='Unchanged v6 stored_scale delegation, FP32 quotient then half-away',
        controlled='Exact one-step and65-token packed codes/scales/carry/readout; sparse readout in65-token fixture',
        random='Exact full/segmented/tokenwise output and final cache; INT8 remains bitwise unchanged across codebooks',
        storage='52B/row and28,499,968B production persistent cache; no resident codebook')
    need(r['criteria']==criteria,'Frozen v7 codec criteria changed')
    expected={'frozen-protocol','fixed-codebook-FP32-bits','actual-production-cache'}
    for book in BOOKS:
        expected.update(prefix+book for prefix in ('CPU-65token-finite-','GPU-65token-packed-oracle-exact-','reject-nonfinite-scale-'))
        for special in ('ties','negative','zero','rounded_scales','underflow','subnormal','large_finite'):
            expected.update(prefix+special+'-'+book for prefix in ('CPU-one-step-finite-','GPU-one-step-exact-'))
    for book in BOOKS[1:]:
        expected.update(prefix+book for prefix in ('CPU-signed-midpoint-scope-','GPU-signed-midpoints-zero-subnormal-'))
    geometries=((2,17,8,19,2),(1,9,6,3,3),(2,5,4,1,1))
    for geometry in geometries:
        for book in BOOKS:
            expected.update(prefix+str(geometry)+'-'+book for prefix in ('partition-','actual-packed-bytes-','signed-index-range-'))
        expected.update(prefix+str(geometry) for prefix in ('INT8-unchanged-','uniform-delegates-v6-'))
    checks=r['checks'];names=[x['name'] for x in checks]
    need(len(expected)==119 and len(names)==119 and set(names)==expected
         and all(x.get('pass') is True for x in checks),'Missing, duplicated, unexpected or failed v7 kernel check')
    rows={x['name']:x for x in checks}
    need(rows['fixed-codebook-FP32-bits']['codebook_bits']==book_bits(),'Codebook FP32 bit patterns differ')
    for book in BOOKS[1:]:
        values=[];scales=[];ties=0
        for scale in (0.,2.**-24,2.**-14,.037933349609375,1.,65504.):
            decoded=[fp32(level*scale) for level in book_levels()[book]]
            for lower,upper in zip(decoded,decoded[1:]):
                midpoint=(lower+upper)*.5;center=fp32(midpoint)
                ties+=int(center==midpoint and scale>0)
                for value in (float(np.nextafter(np.float32(center),np.float32(-math.inf))),center,
                              float(np.nextafter(np.float32(center),np.float32(math.inf)))):
                    values.extend((value,-value));scales.extend((scale,scale))
        values.extend((0.,1.,-1.));scales.extend((0.,0.,0.))
        codes=[quantize4_independent(x,s,book) for x,s in zip(values,scales)]
        bits=[struct.unpack('<I',struct.pack('<f',decode4_independent(q,s,book)))[0] for q,s in zip(codes,scales)]
        cpu=rows['CPU-signed-midpoint-scope-'+book];gpu=rows['GPU-signed-midpoints-zero-subnormal-'+book]
        need(len(values)==255 and cpu['values']==255 and cpu['exact_positive_midpoints']==ties
             and gpu['values']==255 and gpu['codes_exact'] is True and gpu['decoded_exact'] is True
             and gpu['actual_codes']==codes and gpu['actual_decoded_bits']==bits,
             'Independent actual-code/decoded-bit midpoint regression reconstruction differs')
    for geometry in geometries:
        batch,_,heads,dim,_=geometry
        for book in BOOKS:
            row=rows['actual-packed-bytes-'+str(geometry)+'-'+book]
            need(row['actual_bytes']==batch*heads*dim*52 and row['row_bytes']==52,'Actual synthetic packed allocation differs')
    expected_storage=dict(row_bytes=52,ssm_bytes=56*128*64*52,convolution_bytes=56*10240*4*2,
                          permutation_bytes=57344,resident_codebook_bytes=0,total_bytes=CACHE)
    need(all(rows['actual-production-cache'].get(k)==v for k,v in expected_storage.items()),
         'Actual production cache accounting differs')
    return dict(complete=True,sha256=sha(path),checks=119,codebook_bits=book_bits(),
        raw_midpoint_codes_and_decoded_bits_independently_recomputed=3*255,
        criteria=criteria,scope='Recorded GPU kernel evidence checked on CPU; no GPU execution or model logits regenerated.')


def audit_inputs(args,tokenizer,torch,np):
    need(sha(ROOT/'docs/STATE_PPL_V7_PROTOCOL.md')==PROTOCOL,'Frozen v7 protocol changed')
    prior=argparse.Namespace(**vars(args));prior.kernel_checks=args.v6_kernel_checks;prior.selected_calibration=None
    train,candidates,receipt,proof=v6.audit_inputs(prior,tokenizer,torch,np)
    old_kernel=v6.audit_kernel_receipt(args.v6_kernel_checks)
    need(sha(args.v6_selected_calibration)==V6_SELECTED
         and sha(args.v6_selected_calibration.parent/'screen_comparison.json')==V6_SCREEN,
         'Frozen v6 selected policy or comparison differs')
    old_screen=v6.audit_run(args.v6_selected_calibration.parent,'screen',prior,train,candidates,receipt,np)
    old_binding=v6.input_binding(prior,receipt)
    payload,old_selected=v6.audit_selected(args.v6_selected_calibration,prior,candidates,old_binding,old_screen,torch)
    need(payload['selected_id']=='preserve_int8__stored_scale','v7 baseline is not frozen v6 winner')
    binding=dict(v6_input_binding=old_binding,v6_selected_sha256=V6_SELECTED,
        kernel_checks_sha256=sha(args.kernel_checks),protocol_sha256=PROTOCOL,
        source_sha256=v6.SOURCE,tokenizer_sha256=TOKENIZER,train_sha256=PROSE_FILE,
        candidates_sha256=v6.V5_CANDIDATES)
    return train,candidates,receipt,binding,dict(complete=True,upstream=proof,v6_kernel_checks=old_kernel,
        v6_selected_policy=old_selected,v6_screen_comparison_sha256=old_screen['comparison_sha256'])


SCREEN_ARMS=(BASELINE,*(name for name in IDS if name!=BASELINE),'restored_baseline')
FULL_ARMS=('v6_baseline','selected','restored_baseline')


def spec_for(name):
    need(name in IDS,'Unknown frozen candidate identity')
    kind,book=name.split('__')
    return dict(candidate_id=name,candidate_name=kind,variant=book,codebook=book,
        scale_mode='stored_scale',int4_clip=1.,diagnostic=None,deployable=True)


def audit_cache(cache,spec,tokens):
    v6.audit_cache(cache,spec,tokens)
    need(cache.get('codebook')==spec['codebook'] and cache.get('codebook_bits')==book_bits()[spec['codebook']]
         and cache.get('resident_codebook_bytes')==0,'Actual codebook metadata or resident allocation differs')


def audit_selected(path,candidates,binding,screen,torch):
    payload=torch.load(path,map_location='cpu',weights_only=True);receipt=read(path.with_suffix('.json'))
    choice=screen['selection'];name=choice['selected_kind']
    expected=dict(format='MAMBA2_STATE_PPL_V7_CALIBRATION_V1',protocol_sha256=PROTOCOL,input_binding=binding,
        **choice,table_sha256=tensor_sha(candidates['tables'][name]),selection_report_sha256=screen['comparison_sha256'])
    need(all(payload.get(k)==receipt.get(k)==v for k,v in expected.items()),'Frozen selected policy provenance differs')
    need(receipt.get('complete') is True and receipt['file']==path.name and receipt['sha256']==sha(path)
         and receipt['bytes']==path.stat().st_size,'Selected artifact identity differs')
    need(all(receipt.get(k)==v for k,v in payload.items() if k!='permutations'),
         'Selected payload/receipt metadata differs')
    need(payload['permutations'].dtype==torch.uint8 and tuple(payload['permutations'].shape)==(56,8,128)
         and torch.equal(payload['permutations'],candidates['tables'][name])
         and tensor_sha(payload['permutations'])==expected['table_sha256'],'Actual selected coordinates differ')
    hash_inventory(payload['code_hashes'],run_paths(),'v7 selected policy')
    return payload,dict(complete=True,sha256=sha(path),receipt_sha256=sha(path.with_suffix('.json')),
        selected_id=choice['selected_id'],table_sha256=expected['table_sha256'],baseline_wins=choice['baseline_wins'])


def audit_run(directory,stage,args,train,candidates,candidate_receipt,binding,np,
              screen=None,selected=None,validation=None,dataset=None):
    comparison_path=directory/(stage+'_comparison.json');comparison=read(comparison_path)
    common=dict(format='MAMBA2_STATE_PPL_V7_EVAL_V1',stage=stage,protocol_sha256=PROTOCOL,input_binding=binding,
        adapter_loaded=False,adapter_sha256=None,mk_used=False,heldout_used=stage=='full',heldout_used_for_selection=False,
        selected_calibration_sha256=sha(args.selected_calibration) if selected is not None else None,
        parent_report_sha256=V6_BASELINE if stage=='full' else None,
        parent_comparison_sha256=V6_COMPARISON if stage=='full' else None,
        s16_report_sha256=v6.S16_ARCHIVE if stage=='full' else None)
    need(comparison.get('format')=='MAMBA2_STATE_PPL_V7_COMPARISON_V1' and comparison.get('complete') is True
         and all(comparison.get(k)==v for k,v in common.items() if k in ('stage','protocol_sha256','input_binding',
             'adapter_loaded','adapter_sha256','mk_used','heldout_used','heldout_used_for_selection'))
         and 'mk' not in comparison,'v7 comparison provenance differs')
    hash_inventory(comparison['code_hashes'],run_paths(),'v7 evaluation')
    if stage=='screen':
        windows=[(i*2048,train[i,:2048].tolist()) for i in range(80,112)]
        expected_dataset=dict(split='train',file_sha256=PROSE_FILE,rows=list(range(80,112)),tokens_per_row=2048,
            windows=32,target_tokens=65504);arms=SCREEN_ARMS
    else:
        need(screen is not None and selected is not None and not screen['selection']['baseline_wins'],
             'Full requires frozen nonbaseline selection')
        need(tokhash(validation.tolist())==VALIDATION_TOKENS,'Validation token stream differs')
        windows=[(i,validation[i:min(i+2049,len(validation))].tolist()) for i in range(0,len(validation)-1,2048)]
        need(len(windows)==130 and sum(len(ids)-1 for _,ids in windows)==264764,'Full population differs')
        expected_dataset=dataset;arms=FULL_ARMS
    common['dataset']=expected_dataset
    need(tuple(comparison['report_sha256'])==arms,'Fixed execution order or report inventory differs')
    outputs={};metrics={};digests={}
    for arm in arms:
        name=selected['selected_id'] if arm=='selected' else BASELINE if arm in ('v6_baseline','restored_baseline') else arm
        spec=spec_for(name);path=directory/(stage+'_'+arm+'.json');row=read(path);digests[arm]=sha(path)
        need(digests[arm]==comparison['report_sha256'][arm] and row.get('arm')==arm
             and row.get('candidate_table_sha256')==candidate_receipt['table_sha256'][spec['candidate_name']]
             and all(row.get(k)==v for k,v in {**common,**spec}.items()) and 'mk' not in row,
             'Arm identity, fixed policy, table, data or no-MK provenance differs: '+arm)
        need(row['code_hashes']==comparison['code_hashes'] and row.get('candidate_table_unchanged') is True,
             'Code or CPU table changed across arms')
        frozen(row['frozen_source']);audit_backend_policy(row['backend_policy'],row['backend_policy_check'])
        need(row['backend_policy']==comparison['backend_policy'],'Backend differs across arms')
        complete=row.get('complete') is True
        if complete:
            need('error' not in row and not row.get('excluded_from_selection',False) and not row.get('fatal_failure',False)
                 and row.get('runtime_table_unchanged') is True and row.get('persistent_float_finite_checks_passed') is True,
                 'Complete arm failed integrity or finiteness')
            probe=row['repeated_reset_probe'];audit_cache(probe['cache'],spec,128);hashes=probe['cache_tensor_sha256']
            need(probe.get('tokens')==128 and probe.get('hidden_and_cache_exact') is True
                 and probe['token_sha256_int64le']==tokhash(windows[0][1][:128])
                 and re.fullmatch('[0-9a-f]{64}',probe['hidden_sha256']) is not None
                 and set(hashes)=={f'{i}.{k}' for i in range(56) for k in ('lo','hi','q4','s8','s4','conv')}
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
                 and not row.get('fatal_failure',False) and row.get('error_type')=='CandidateInvalid'
                 and row.get('error') in {f'CandidateInvalid({message!r})' for message in allowed},
                 'Incomplete candidate is not a recognized numerical exclusion')
        metrics[arm]=v6.audit_ppl(row,windows,complete);outputs[arm]=row
    baseline=BASELINE if stage=='screen' else 'v6_baseline'
    restoration=read(directory/(stage+'_restoration.json'))
    v6.exact_restoration(outputs[baseline],outputs['restored_baseline'],restoration)
    need(comparison['restoration']==restoration,'Restoration comparison differs')
    result=dict(complete=True,arms=metrics,report_sha256=digests,restoration=restoration,
        comparison_sha256=sha(comparison_path),no_MK_measurements=True,no_adapter=True)
    if stage=='screen':
        choice=select_independent(outputs);delta=v6.compare_independent(outputs[BASELINE],outputs[choice['selected_id']])
        need(comparison['selection']==choice and comparison['selected_vs_baseline']==delta,
             'Independent PPL-only selection or comparison arithmetic differs')
        result.update(selection=choice,selected_vs_baseline=delta)
    else:
        parent_path=args.v6_eval_dir/'full_selected.json';old_comparison_path=args.v6_eval_dir/'full_comparison.json'
        need(sha(parent_path)==V6_BASELINE and sha(old_comparison_path)==V6_COMPARISON
             and sha(args.s16_report)==v6.S16_ARCHIVE,'Pinned v6 baseline/comparison/S16 changed')
        historical=read(old_comparison_path)
        need(historical['report_sha256']['selected']==V6_BASELINE,'Archived context chain differs')
        parent=read(parent_path);s16=read(args.s16_report)
        v6.audit_ppl(parent,windows);v6.audit_ppl({'ppl':s16['ppl']},windows)
        v6.replay_ppl(parent,outputs['v6_baseline'])
        old_probe=parent['repeated_reset_probe'];new_probe=outputs['v6_baseline']['repeated_reset_probe']
        need(all(old_probe[k]==new_probe[k] for k in ('tokens','hidden_sha256','hidden_and_cache_exact',
            'cache_tensor_sha256','token_sha256_int64le')),'Archived v6 baseline reset values differ')
        replay_record=read(directory/'full_parent_replay.json')
        expected_replay=dict(complete=True,per_window_nll_exact=True,ppl_windows_repeated=130,target_tokens=264764,
            reset_hidden_exact=True,reset_cache_exact=True,cache_bytes_exact=True)
        need(replay_record==expected_replay and comparison['parent_replay']==replay_record,'Archived replay receipt differs')
        expected=dict(selected_id=selected['selected_id'],selected_kind=selected['selected_kind'],
            selected_codebook=selected['selected_codebook'],selected_calibration_sha256=sha(args.selected_calibration),
            selection_report_sha256=screen['comparison_sha256'],parent_report_sha256=V6_BASELINE,
            parent_comparison_sha256=V6_COMPARISON,s16_report_sha256=v6.S16_ARCHIVE,resurface_trained=False,mk_evaluated=False)
        need(all(comparison.get(k)==v for k,v in expected.items()),'Full frozen-candidate/no-training binding differs')
        delta=v6.compare_independent(outputs['v6_baseline'],outputs['selected']);gap=v6.compare_independent(s16,outputs['selected'])
        checks=target_gate(outputs['selected']['ppl']['ppl'],CACHE)
        checks['cache_same_budget']=all(row['cache']['total_bytes']==CACHE for row in outputs.values())
        need(comparison['comparison']==delta and comparison['original_s16_comparison']==gap
             and comparison['target_checks']==checks and comparison['target_pass']==all(checks.values()),
             'Independent full target arithmetic differs')
        result.update(selection=screen['selection'],comparison=delta,original_s16_comparison=gap,
            target_checks=checks,target_pass=all(checks.values()),parent_replay=replay_record)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true');parser.add_argument('--stage',choices=('inputs','screen','full'))
    for name in ('source-dir','calibration','repair-calibration','parent-training-report','prose-tokens',
        'codec-checks','candidates','v5-calibration','v6-selected-calibration','v6-kernel-checks','kernel-checks',
        'eval-dir','screening-report','selected-calibration','v6-eval-dir','s16-report','output'):
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test(),indent=2));return
    for key in ('stage','source_dir','calibration','repair_calibration','parent_training_report','prose_tokens',
        'codec_checks','candidates','v5_calibration','v6_selected_calibration','v6_kernel_checks','kernel_checks','output'):
        if getattr(args,key) is None:parser.error('--'+key.replace('_','-')+' is required')
    if args.stage in ('screen','full') and args.eval_dir is None:parser.error('--eval-dir is required')
    if args.stage=='full':
        for key in ('screening_report','selected_calibration','v6_eval_dir','s16_report'):
            if getattr(args,key) is None:parser.error('--'+key.replace('_','-')+' is required')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Set CUDA_VISIBLE_DEVICES= for CPU-only audit')
    need(not args.output.exists(),'Use a fresh output path and preserve earlier receipts')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1'
    result=dict(format='MAMBA2_STATE_PPL_V7_INDEPENDENT_AUDIT_V1',complete=False,passed=False,stage=args.stage,
        protocol_sha256=PROTOCOL,source_sha256=sha(__file__),
        auditor_dependency_sha256={name:sha(ROOT/'scripts'/name) for name in ('audit_state_ppl_v6.py',
            'audit_quant_first.py','audit_state_repair.py','audit_state_first_v5.py','audit_resurface_more.py')},
        limitations=['CPU audit recomputes recorded score arithmetic and provenance, not GPU logits.',
            'Codec receipts contain recorded GPU evidence; this audit launches no kernels.',
            'Identity/version/gradient checks do not replace a post-run weight-byte hash.',
            'Repeated benchmark exposure is not untouched generalization evidence; recall is not measured.'])
    try:
        import numpy as np
        import torch
        from mamba2_recall import runtime
        torch.set_num_threads(4);need(not torch.cuda.is_initialized(),'CUDA initialized before CPU audit')
        result['self_tests']=self_test();tokenizer=runtime.SentencePieceTokenizer(args.source_dir)
        need(tokenizer.sha256==TOKENIZER,'Tokenizer differs')
        train,candidates,receipt,binding,proof=audit_inputs(args,tokenizer,torch,np)
        result['inputs']=proof;result['kernel_checks']=audit_kernel_receipt(args.kernel_checks,np)
        if args.stage!='inputs':
            directory=args.eval_dir if args.stage=='screen' else args.screening_report.parent
            if args.stage=='full':need(args.screening_report.name=='screen_comparison.json','Frozen screen filename differs')
            screen=audit_run(directory,'screen',args,train,candidates,receipt,binding,np);result['screen']=screen
            selected_path=directory/'selected_calibration.pt'
            if args.selected_calibration is None:args.selected_calibration=selected_path
            need(args.selected_calibration.resolve()==selected_path.resolve(),'Selection must accompany frozen screen')
            selected,selected_proof=audit_selected(selected_path,candidates,binding,screen,torch)
            result['selected_calibration']=selected_proof
            if args.stage=='full':
                need(not screen['selection']['baseline_wins'],'Baseline fallback cannot enter full evaluation')
                from mamba2_recall.calibration import load_wikitext_tokens
                validation,dataset=load_wikitext_tokens(tokenizer,'validation')
                result['full']=audit_run(args.eval_dir,'full',args,train,candidates,receipt,binding,np,
                    screen,selected,validation,dataset)
        need(not torch.cuda.is_initialized(),'CPU audit unexpectedly initialized CUDA')
        result.update(complete=True,passed=True,cuda_initialized=False)
    except BaseException as exc:result['error']=repr(exc);raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
