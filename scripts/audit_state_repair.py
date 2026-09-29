#!/usr/bin/env python3
"""CPU-only independent audit of fixed same-budget state repair evidence.

Reconstructs calibration tables, input identities, scores, selection and gates.
Does not rerun GPU logits, calibration traces, packing kernels or model training.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audit_quant_first import (sha, need, finite, close, tokhash, read, recompute_pair,
    SOURCE, TOKENIZER, PROSE_FILE, PROSE_MANIFEST, VALIDATION_TOKENS)
from audit_resurface_more import audit_backend_policy

PROTOCOL = 'f18069d0340449316297077a2d912f7ef42fbf518436d9b311d8a731c96d722c'
OLD_CAL = 'c366cd577967e64635f1dd960237dc1c0024d7413685ce0b4c4e40da869a7023'
PARENT_REPORT = 'e5a77d86cf2fb0e2389247e3cb325f74e89957861a6043e92a891d6d402ae359'
ADAPTER = '7144b88265a797ef935b1f94845395abec5532d8dc5b2a4a8fa0a8e07ce205f0'
ARCHIVE = dict(parent='224a8d1201761e44072bd37211ec0c916cc0872095042bf26cde49d8a349b20b',
               s16='52f82f83258a2fa3160ea14585f1bd69e1d68d60d8f179d1f0987c636f6546ba')
KINDS = ('readout_tiers', 'dense3', 'dense3_equalized')
TABLES = dict(readout_tiers='readout_permutations', dense3='original_permutations',
              dense3_equalized='equalizer_exponents')
CACHE = 28499968
RULE = 'Fixed protocol: eligible <=1.01x PPL and at most2 fewer TRAIN answers; otherwise finite best <=1.25x; PPL/MK/order tiebreak'
SCREEN_ARMS = [('source_s16', 's16', False), ('old_sq_no_adapter', 'old_sq', False),
               ('old_sq_v2', 'old_sq', True)] + [
    (k+s, k, a) for k in KINDS for s,a in (('_no_adapter',False),('_v2',True))
] + [('restored_old_sq_v2', 'old_sq', True)]


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def hash_inventory(actual, required, label):
    need(isinstance(actual,dict) and set(actual)==set(required), label+' source inventory differs')
    for relative,digest in actual.items():
        path=(ROOT/relative).resolve()
        need(path.is_relative_to(ROOT) and sha(path)==digest, label+' source changed: '+relative)


def run_paths():
    return {str(p.relative_to(ROOT)) for p in (list((ROOT/'mamba2_recall').glob('*.py')) + [
        ROOT/'scripts/run_state_repair.py', ROOT/'scripts/prepare_state_repair.py',
        ROOT/'scripts/evaluate_resurface_more.py', ROOT/'scripts/evaluate_quant_first.py',
        ROOT/'docs/STATE_REPAIR_PROTOCOL.md', ROOT/'docs/RESURFACE_MORE_BACKEND_REPLAY.md'])}


def frozen(proof):
    need(proof.get('identity_version_gradients_unchanged') is True and proof.get('tensors')==507
         and proof.get('parameters')==8236999680, 'Frozen base inventory/check differs')


def selection(results):
    """Independent implementation; never calls the evaluation harness selector."""
    baseline=results['old_sq_v2']; p0=baseline['ppl']['ppl']
    m0=baseline['mk']['summary']['normal']['correct']
    valid=[]; eligible=[]
    for candidate in KINDS:
        ok=True
        for suffix in ('_no_adapter','_v2'):
            arm=results[candidate+suffix]
            ok=ok and arm.get('complete') is True
            if ok:
                ok=finite(arm.get('ppl',{}).get('ppl')) and arm.get('cache',{}).get('total_bytes')==CACHE
        if not ok:
            continue
        valid.append(candidate)
        arm=results[candidate+'_v2']
        if arm['ppl']['ppl']<=p0*1.01 and arm['mk']['summary']['normal']['correct']>=m0-2:
            eligible.append(candidate)
    ranked=sorted(eligible if eligible else valid, key=lambda k:(results[k+'_v2']['ppl']['ppl'],
        -results[k+'_v2']['mk']['summary']['normal']['correct'], KINDS.index(k)))
    chosen=ranked[0] if ranked else None
    if not eligible and chosen is not None and results[chosen+'_v2']['ppl']['ppl']>p0*1.25:
        chosen=None
    return dict(selected=chosen,eligible=eligible,valid=valid,screening_pass=bool(eligible),
        diagnostic_advancement=chosen is not None and not eligible,stopped=chosen is None,rule=RULE)


def self_test():
    def fixture():
        base=dict(complete=True,ppl=dict(ppl=10.),mk=dict(summary=dict(normal=dict(correct=40))),
                  cache=dict(total_bytes=CACHE))
        result={'old_sq_v2':copy.deepcopy(base)}
        for k in KINDS:
            for suffix in ('_no_adapter','_v2'):
                result[k+suffix]=dict(complete=False)
        return result,base
    def enable(results,base,k,ppl,mk):
        for suffix in ('_no_adapter','_v2'):
            v=copy.deepcopy(base);v['ppl']['ppl']=ppl;v['mk']['summary']['normal']['correct']=mk
            results[k+suffix]=v
    checks=[]
    r,b=fixture();enable(r,b,KINDS[0],10.1,38)
    need(selection(r)['eligible']==[KINDS[0]],'Inclusive eligibility boundary');checks.append('eligibility-inclusive')
    r[KINDS[0]+'_v2']['ppl']['ppl']=math.nextafter(10.1,math.inf)
    need(selection(r)['eligible']==[] and selection(r)['diagnostic_advancement'],'Eligibility PPL cutoff')
    checks.append('eligibility-one-ulp-above-diagnostic')
    r[KINDS[0]+'_v2']['ppl']['ppl']=10.1
    r[KINDS[0]+'_v2']['mk']['summary']['normal']['correct']=37
    need(selection(r)['diagnostic_advancement'] is True,'MK eligibility boundary');checks.append('mk-minus3-diagnostic')
    r,b=fixture();enable(r,b,KINDS[0],12.5,40)
    need(selection(r)['diagnostic_advancement'] is True,'Inclusive diagnostic boundary');checks.append('diagnostic-inclusive')
    r[KINDS[0]+'_v2']['ppl']['ppl']=math.nextafter(12.5,math.inf)
    need(selection(r)['stopped'] is True,'Diagnostic cutoff');checks.append('diagnostic-above-stop')
    r,b=fixture();enable(r,b,KINDS[0],10.,40);enable(r,b,KINDS[1],10.,41)
    need(selection(r)['selected']==KINDS[1],'MK tiebreak');checks.append('ppl-tie-use-mk')
    r[KINDS[0]+'_v2']['ppl']['ppl']=9.9
    need(selection(r)['selected']==KINDS[0],'PPL must precede MK in selection');checks.append('ppl-before-mk')
    r[KINDS[0]+'_v2']['ppl']['ppl']=10.
    r[KINDS[0]+'_v2']['mk']['summary']['normal']['correct']=41
    need(selection(r)['selected']==KINDS[0],'Candidate order tiebreak');checks.append('full-tie-use-order')
    r[KINDS[0]+'_no_adapter']['complete']=False
    need(selection(r)['valid']==[KINDS[1]],'Failed unadapted arm must exclude codec');checks.append('either-arm-failure-excludes-codec')
    r[KINDS[1]+'_no_adapter']['cache']['total_bytes']+=1
    need(selection(r)['stopped'],'Over-budget candidate');checks.append('one-extra-byte-excludes')
    r,b=fixture();enable(r,b,KINDS[0],float('nan'),40)
    need(selection(r)['stopped'],'Nonfinite candidate');checks.append('nan-excludes')
    r,b=fixture();enable(r,b,KINDS[0],float('inf'),40)
    need(selection(r)['stopped'],'Infinite candidate');checks.append('infinity-excludes')
    r,b=fixture();need(selection(r)['stopped'],'All failed candidates');checks.append('all-failed-stop')
    # Independent bitplane byte accounting and signed interpretation, no Torch/GPU.
    values=[(-3,-2,-1,0,1,2,3)[i%7] for i in range(128)]
    packed=bytearray(48)
    for i,q in enumerate(values):
        for plane in range(3):packed[plane*16+i//8]|=((q&7)>>plane&1)<<(i%8)
    decoded=[]
    for i in range(128):
        u=sum(((packed[p*16+i//8]>>(i%8))&1)<<p for p in range(3))
        decoded.append(u-8 if u>=4 else u)
    need(decoded==values and len(packed)+2*2==52,'Independent packed row accounting')
    checks.append('signed-three-bitplanes-48B-plus-two-FP16-52B')
    return dict(complete=True,passed=True,checks=checks,count=len(checks),gpu_used=False)


def audit_unit_receipts(codec,collector):
    need(codec.get('format')=='MAMBA2_DENSE3_CODEC_CHECK_V1' and codec.get('complete') is True
         and 'error' not in codec,'Dense codec validation incomplete')
    checks=codec['checks'];names=[r['name'] for r in checks]
    required=set()
    geometries=((2,17,8,19,2),(1,9,6,3,3),(2,5,4,1,1))
    for geo in geometries:
        for kind in ('permutation','equalizer'):
            for prefix in ('partition-bitwise','payload-52B','random-readout-oracle'):
                required.add(f'{prefix}-{geo}-{kind}')
    for name in ('ties','negative','zero','scale_rounding','scale_underflow','large_finite'):
        for kind in ('permutation','equalizer'):required.add(f'exact-CPU-oracle-{name}-{kind}')
    required.update(('equalizer-extremes-minus8-plus8','reject-nonfinite-amplified-FP16-scale',
        'zero-scale-underflow-is-representable','reject-nonfinite-readout','production-cache-budget',
        'reject-exponent-outside-frozen-range'))
    need(set(names)==required and len(names)==len(required) and all(r.get('pass') is True for r in checks),
         'Dense codec checks missing, duplicated or failed')
    by_name={r['name']:r for r in checks}
    for geo in geometries:
        b,_,h,p,_=geo
        for kind in ('permutation','equalizer'):
            need(by_name[f'payload-52B-{geo}-{kind}']['payload_bytes']==b*h*p*52,'Synthetic packed bytes differ')
            row=by_name[f'random-readout-oracle-{geo}-{kind}']
            need(finite(row['relative_l2']) and 0<=row['relative_l2']<=row['bound']==.002,'Readout oracle bound differs')
    row=by_name['production-cache-budget']
    need(row['row_bytes']==52 and row['ssm_bytes']==56*128*64*52 and row['table_bytes']==57344
         and row['total_bytes']==CACHE and row['includes_convolution'] is True,'Packed production allocation proof differs')
    hash_inventory(codec['code_sha256'],{'mamba2_recall/state_codec_dense3.py',
        'mamba2_recall/state_quant_dense3.py','scripts/check_state_codec_dense3.py'},'Dense checks')
    need(collector.get('format')=='MAMBA2_STATE_REPAIR_COLLECTOR_CHECKS_V1' and collector.get('complete') is True
         and collector.get('passed') is True and collector.get('equalizer_identity_and_clamp_passed') is True
         and 'error' not in collector,'Collector checks incomplete')
    need(len(collector['cases'])==3,'Collector case count differs')
    for case,geo in zip(collector['cases'],((1,4,5,2,1),(2,8,29,2,33),(1,128,64,8,64))):
        need(tuple(case[k] for k in ('batch','heads','P','groups','tokens'))==geo
             and all(case[k] is True for k in ('output_exact','carry_exact','abs_stats_exact','segmented_exact','score_oracle_close'))
             and finite(case['score_max_absolute_error']) and case['score_max_absolute_error']>=0,
             'Collector geometry/parity/oracle receipt differs')
    hash_inventory(collector['code_sha256'],{'scripts/check_state_repair_calibration.py',
        'mamba2_recall/state_repair_calibration.py','mamba2_recall/state_codec.py'},'Collector checks')
    return dict(dense_checks=len(checks),collector_cases=3,all_bound_passed=True,
                packing_scope='CPU verifies recorded GPU test identities/results and allocation arithmetic, not a GPU kernel rerun')


def audit_inputs(args,tokenizer,torch,np):
    need(sha(ROOT/'docs/STATE_REPAIR_PROTOCOL.md')==PROTOCOL,'Frozen repair protocol changed')
    need(sha(args.calibration)==OLD_CAL and sha(args.parent_training_report)==PARENT_REPORT,
         'Frozen original calibration/parent report changed')
    need(sha(args.prose_tokens)==PROSE_FILE and sha(ROOT/'docs/prose_train_manifest.json')==PROSE_MANIFEST,
         'Pinned prose TRAIN files changed')
    train=torch.load(args.prose_tokens,map_location='cpu',weights_only=True)
    need(train.dtype==torch.int64 and tuple(train.shape)==(448,2048) and bool(((train>=0)&(train<256000)).all()),
         'TRAIN tensor geometry/value range differs')
    manifest=read(ROOT/'docs/prose_train_manifest.json')
    need(tokhash(train.flatten().tolist())==manifest['training_tokens_sha256_int64le']
         and manifest['tokenizer_sha256']==TOKENIZER and manifest['source_checkpoint_sha256']==SOURCE,
         'TRAIN tensor/manifest content differs')
    old=torch.load(args.calibration,map_location='cpu',weights_only=True)
    parent=read(args.parent_training_report)
    adapter=args.parent_training_report.parent/parent['adapter']['file']
    need(parent['complete'] is True and parent['successful_updates']==1536 and sha(adapter)==ADAPTER
         and parent['adapter']['sha256']==ADAPTER,'Frozen v2 adapter identity differs')
    for relative,digest in parent['code_sha256'].items():need(sha(ROOT/relative)==digest,'Frozen original source changed: '+relative)
    frozen(parent['frozen_base_check'])
    payload=torch.load(args.repair_calibration,map_location='cpu',weights_only=True)
    receipt=read(args.repair_calibration.with_suffix('.json'))
    fields=dict(format='MAMBA2_STATE_REPAIR_CALIBRATION_V1',protocol_sha256=PROTOCOL,
        source_sha256=SOURCE,tokenizer_sha256=TOKENIZER,train_file_sha256=PROSE_FILE,
        prose_manifest_sha256=PROSE_MANIFEST,original_calibration_sha256=OLD_CAL,
        adapter=None,adapter_sha256=None,heldout_used=False,calibration_tokens=4096,
        train_token_hashes=[tokhash(train[i,:512].tolist()) for i in range(8)],
        selection='First8 original TRAIN rows, first512tokens; independent zero reset per row')
    need(all(payload.get(k)==v and receipt.get(k)==v for k,v in fields.items()),'Repair calibration source/TRAIN-only binding differs')
    need(receipt['complete'] is True and receipt['fresh_source_no_adapter'] is True
         and receipt['serialization_roundtrip_bitwise'] is True and receipt['sha256']==sha(args.repair_calibration)
         and receipt['bytes']==args.repair_calibration.stat().st_size,'Repair calibration receipt differs')
    required={'scripts/prepare_state_repair.py','scripts/check_state_repair_calibration.py',
        'scripts/prepare_quant_first.py','scripts/evaluate_resurface_more.py','docs/STATE_REPAIR_PROTOCOL.md',
        'mamba2_recall/state_repair_calibration.py','mamba2_recall/state_codec.py',
        'mamba2_recall/state_quant.py','mamba2_recall/runtime.py'}
    hash_inventory(receipt['code_sha256'],required,'Calibration')
    codec=read(args.codec_checks);collector=receipt['collector_checks']
    unit=audit_unit_receipts(codec,collector)
    # The collector receipt was embedded verbatim; verify its original serialization hash.
    serialized=(json.dumps(collector,indent=2)+'\n').encode()
    need(hashlib.sha256(serialized).hexdigest()==receipt['collector_checks_sha256'],'Embedded collector receipt hash differs')
    stats=payload['statistics'];counts=stats['sample_count_per_group']
    need(counts.dtype==torch.int64 and tuple(counts.shape)==(56,) and bool((counts==4096*16*64).all()),
         'Calibration sample count differs')
    for stem in ('abs','readout_score'):
        total,mean=stats['sum_'+stem],stats['mean_'+stem]
        need(total.dtype==mean.dtype==torch.float64 and tuple(total.shape)==tuple(mean.shape)==(56,8,128)
             and bool(torch.isfinite(total).all()) and bool((total>=0).all())
             and np.array_equal(mean.numpy(),total.numpy()/counts.numpy()[:,None,None]),
             'Calibration mean does not derive from finite nonnegative sums: '+stem)
    scores=stats['mean_readout_score'].numpy();means=stats['mean_abs'].numpy()
    ranking=np.argsort(-scores,axis=-1,kind='stable').astype(np.uint8)
    logs=np.log2(np.maximum(means,1e-12));exponents=np.clip(np.rint(logs-logs.mean(-1,keepdims=True)),-8,8).astype(np.int8)
    expected={'readout_permutations':ranking,'equalizer_exponents':exponents,
              'original_permutations':old['permutations'].numpy()}
    for key,values in expected.items():
        table=payload[key]
        dtype=torch.int8 if key=='equalizer_exponents' else torch.uint8
        need(table.dtype==dtype and tuple(table.shape)==(56,8,128) and table.numel()*table.element_size()==57344
             and np.array_equal(table.numpy(),values) and tensor_sha(table)==receipt['table_sha256'][key]
             and receipt['table_shapes'][key]==[56,8,128],'Table hash, shape or independent derivation differs: '+key)
    need(receipt['runtime_table_bytes_per_candidate']==57344 and receipt['readout_tier_counts']==dict(int8=16,int4=64,dead=48)
         and receipt['sample_count_per_group']==counts.tolist(),'Metadata/coordinate accounting differs')
    ordered=np.take_along_axis(scores,ranking.astype(np.int64),axis=-1);total=float(ordered.sum())
    need(total>0,'Empty readout score')
    for k,lo,hi in [('int8',0,16),('int4',16,80),('dead',80,128)]:
        close(receipt['readout_score_tier_mass'][k],float(ordered[...,lo:hi].sum()/total),'Readout tier mass differs')
    previous=np.take_along_axis(scores,old['permutations'].numpy().astype(np.int64),axis=-1)
    close(receipt['original_permutation_dead_readout_score_mass'],float(previous[...,80:].sum()/total),'Old dead readout mass differs')
    need(receipt['equalizer_exponent_counts']=={str(k):int((exponents==k).sum()) for k in range(-8,9)},'Exponent population differs')
    proof=receipt['collection_forward_exactcheck']
    need(all(proof.get(k) is True for k in ('complete','passed','hidden_bitwise_equal','all56_ssm_carries_exact',
        'all56_conv_caches_exact','mean_abs_statistics_exact')) and proof['probe_tokens']==128
        and proof['probe_token_sha256']==tokhash(train[0,:128].tolist())
        and proof['reference_hidden_sha256']==proof['collected_hidden_sha256']
        and len(proof['ssm_state_sha256'])==len(proof['conv_state_sha256'])==56,'Collector full-model parity receipt differs')
    frozen(receipt['frozen_source']);audit_backend_policy(receipt['backend_policy'],receipt['backend_policy_check'])
    return train,parent,payload,receipt,dict(complete=True,table_derivation_independent=True,
        calibration_sha256=sha(args.repair_calibration),calibration_tokens=4096,
        table_bytes_per_candidate=57344,one_packed_row_bytes=52,cache_bytes=CACHE,
        readout_score_tier_mass=receipt['readout_score_tier_mass'],unit_receipts=unit)


def cache_check(value,kind,tokens=None):
    rows=56*128*64;dense=kind.startswith('dense3');s16=kind=='s16'
    scale=0 if s16 else rows*4;state=rows*(256 if s16 else 52)
    table=0 if s16 else 56*8*128;conv=56*10240*4*2
    mode='s16' if s16 else 'dense3_equalizer' if kind=='dense3_equalized' else 'dense3_permutation' if dense else 'sq3p25'
    expected=dict(mode=mode,batch_size=1,allocated_layers=56,conv_fp16_bytes=conv,
        ssm_payload_bytes=state-scale,ssm_scale_bytes=scale,ssm_total_bytes=state,
        total_bytes=state+table+conv,calibration_workspace_bytes=0)
    expected['table_bytes' if dense else 'permutation_bytes']=table
    if dense:expected['table_kind']='equalizer' if kind=='dense3_equalized' else 'permutation'
    if tokens is not None:expected['tokens_per_layer']=[tokens]*56
    need(all(value.get(k)==v for k,v in expected.items()),'Actual packed cache accounting differs: '+kind)
    return expected['total_bytes']


def scores(result,windows,cases,tokenizer,complete=True):
    raw=result.get('ppl',{});reported=raw.get('windows',[])
    need(len(reported)<=len(windows) and (not complete or len(reported)==len(windows)), 'PPL coverage differs')
    nll=0.;count=0
    for row,(start,ids) in zip(reported,windows):
        need(row['start']==start and row['target_tokens']==len(ids)-1
             and row['token_sha256_int64le']==tokhash(ids) and finite(row['nll']) and row['nll']>=0,
             'PPL token identity/count/NLL differs')
        close(row['ppl'],math.exp(row['nll']/(len(ids)-1)),'Window PPL arithmetic differs')
        nll+=row['nll'];count+=len(ids)-1
    if reported:
        close(raw['nll'],nll,'Aggregate NLL differs');need(raw['target_tokens']==count,'PPL target count differs')
        close(raw['ppl'],math.exp(nll/count),'Aggregate PPL arithmetic differs')
    records=result.get('mk',{}).get('rows',[])
    need(len(records)==len({r['id'] for r in records}) and len(records)<=len(cases)
         and (not complete or len(records)==len(cases)),'MK missing/duplicate cases')
    summary={c:dict(correct=0,count=0) for c in ('normal','target_removed')}
    for row,case in zip(records,cases):
        ids=tokenizer.encode(case['prompt']);generated=row['generated_ids']
        need(all(row.get(k)==v for k,v in case.items()) and row['prompt_tokens']==len(ids)
             and row['prompt_token_sha256_int64le']==tokhash(ids),'MK case/token identity differs')
        need(isinstance(generated,list) and 1<=len(generated)<=12
             and all(type(v) is int and 0<=v<256000 for v in generated)
             and tokenizer.eos_token_id not in generated[:-1]
             and (len(generated)==12 or generated[-1]==tokenizer.eos_token_id),'Greedy generation stopping evidence differs')
        text=tokenizer.decode(generated);match=re.search(r'(?<!\d)\d{6}(?!\d)',text)
        prediction=match.group() if match else None;correct=prediction==case['answer']
        need(row['output']==text and row['prediction']==prediction and type(row['correct']) is bool
             and row['correct']==correct,'Decoded six-digit scoring differs')
        entry=summary[case['condition']];entry['count']+=1;entry['correct']+=int(correct)
    for value in summary.values():value['accuracy']=value['correct']/value['count'] if value['count'] else None
    if complete:need(result['mk']['summary']==summary,'MK aggregate arithmetic differs')
    return dict(complete=complete,ppl=math.exp(nll/count) if count else None,
        ppl_windows=len(reported),ppl_targets=count,mk_cases=len(records),mk=summary)


def replay(before,after,receipt):
    need(before['complete'] is True and after['complete'] is True
         and before['ppl']['windows']==after['ppl']['windows'],'Replay per-window NLL differs')
    for k in ('nll','ppl','target_tokens'):need(before['ppl'][k]==after['ppl'][k],'Replay PPL aggregate differs')
    a,b=before['mk']['rows'],after['mk']['rows'];need(len(a)==len(b)>0,'Replay MK coverage differs')
    for x,y in zip(a,b):
        need(all(x[k]==y[k] for k in ('id','prompt_token_sha256_int64le','generated_ids','output','prediction','correct')),
             'Replay generated token sequence/score differs')
    need(before['cache']['total_bytes']==after['cache']['total_bytes'],'Replay memory changed')
    expected=dict(complete=True,per_window_nll_exact=True,all_generated_ids_exact=True,
        all_decoded_predictions_exact=True,ppl_windows_repeated=len(before['ppl']['windows']),
        ppl_target_tokens_repeated=before['ppl']['target_tokens'],mk_cases_repeated=len(a))
    need(all(receipt.get(k)==v for k,v in expected.items()) and isinstance(receipt.get('scope'),str),'Replay receipt differs')
    return {**expected,'scope':receipt['scope']}


def audit_run(directory,stage,args,train,parent,payload,calibration,tokenizer,data,np,validation=None,dataset=None,selected=None):
    comparison=read(directory/f'{stage}_comparison.json')
    need(comparison['format']=='MAMBA2_STATE_REPAIR_COMPARISON_V1' and comparison['complete'] is True
         and comparison['stage']==stage and comparison['protocol_sha256']==PROTOCOL
         and comparison['repair_calibration_sha256']==sha(args.repair_calibration)
         and comparison['parent_adapter_sha256']==ADAPTER and comparison['codec_checks_sha256']==sha(args.codec_checks),
         'Comparison provenance/status differs')
    hash_inventory(comparison['code_hashes'],run_paths(),'Evaluation')
    if stage=='screen':
        windows=[(i*2048,train[i,:512].tolist()) for i in range(8,16)]
        cases=[c for c in data.generate_cases('train') if c['sample'] in range(0,256,32)]
        expected_dataset=dict(split='train',file_sha256=PROSE_FILE,rows=list(range(8,16)),tokens_per_row=512)
        arms=SCREEN_ARMS
        need(len(cases)==48 and sum(len(w)-1 for _,w in windows)==4088,'TRAIN screening population differs')
    else:
        need(selected in KINDS and validation is not None,'Full audit needs independently selected candidate')
        need(tokhash(validation.tolist())==VALIDATION_TOKENS,'Validation token stream differs')
        windows=[(s,validation[s:min(s+2049,len(validation))].tolist()) for s in range(0,len(validation)-1,2048)]
        cases=data.generate_cases('confirm');expected_dataset=dataset
        arms=[('old_sq_v2','old_sq',True),(selected+'_v2',selected,True),('restored_old_sq_v2','old_sq',True)]
        need(len(windows)==130 and sum(len(w)-1 for _,w in windows)==264764 and len(cases)==768,'Full validation population differs')
    common=dict(format='MAMBA2_STATE_REPAIR_EVAL_V1',stage=stage,protocol_sha256=PROTOCOL,
        source_checkpoint_sha256=SOURCE,tokenizer_sha256=TOKENIZER,original_calibration_sha256=OLD_CAL,
        repair_calibration_sha256=sha(args.repair_calibration),
        repair_calibration_receipt_sha256=sha(args.repair_calibration.with_suffix('.json')),
        parent_training_report_sha256=PARENT_REPORT,parent_adapter_sha256=ADAPTER,
        table_sha256={k:tensor_sha(payload[v]) for k,v in TABLES.items()},dataset=expected_dataset,
        codec_checks_sha256=sha(args.codec_checks))
    need(set(comparison['report_sha256'])=={a for a,_,_ in arms},'Arm inventory changed')
    outputs={};metrics={};digests={}
    for arm,kind,adapted in arms:
        path=directory/f'{stage}_{arm}.json';result=read(path);digests[arm]=sha(path)
        need(digests[arm]==comparison['report_sha256'][arm] and result['arm']==arm and result['kind']==kind
             and result.get('adapter_sha256')==(ADAPTER if adapted else None)
             and all(result.get(k)==v for k,v in common.items()),'Arm provenance/file identity differs: '+arm)
        need(result['code_hashes']==comparison['code_hashes'],'Code changed across arms')
        frozen(result['frozen_source']);need(result['adapter_content_unchanged'] is True,'Adapter mutation')
        audit_backend_policy(result['backend_policy'],result['backend_policy_check'])
        need(result['backend_policy']==comparison['backend_policy'],'Backend policy changed across arms')
        done=result.get('complete') is True
        if not done:
            need(stage=='screen' and kind in KINDS and result.get('excluded_from_selection') is True
                 and isinstance(result.get('error'),str) and result['error'],'Unexpected failed arm')
        else:
            need('error' not in result and not result.get('excluded_from_selection',False)
                 and result['persistent_float_finite_checks_passed'] is True,'Completed arm has failure/nonfinite flags')
            proof=result['repeated_reset_probe'];need(proof['tokens']==128 and proof['hidden_and_cache_exact'] is True
                and proof['token_sha256']==tokhash(windows[0][1][:128])
                and re.fullmatch('[0-9a-f]{64}',proof['hidden_sha256']) is not None
                and proof['cache_bytes']==(122028032 if kind=='s16' else CACHE),'128-token reset/cache proof differs')
            last=result['mk']['rows'][-1]
            token_count=last['prompt_tokens']+len(last['generated_ids'])-1
            cache_check(result['cache'],kind,token_count)
            need(type(result['zero_scale_observations']) is int and result['zero_scale_observations']>=0
                 and result['zero_scale_scope']=='Repeated final-cache observations; includes true zero states and FP16 scale underflow; not unique underflow events',
                 'Zero-scale observation scope/accounting differs')
        metrics[arm]=scores(result,windows,cases,tokenizer,done);outputs[arm]=result
    restoration=read(directory/f'{stage}_restoration.json')
    replay(outputs['old_sq_v2'],outputs['restored_old_sq_v2'],restoration)
    need(comparison['restoration']==restoration,'Comparison restoration differs')
    result=dict(complete=True,arms=metrics,report_sha256=digests,restoration=restoration,
                comparison_sha256=sha(directory/f'{stage}_comparison.json'))
    if stage=='screen':
        chosen=selection(outputs);need(comparison['selection']==chosen,'Independent screening selection differs')
        result['selection']=chosen
    else:
        need(args.parent_eval_dir is not None,'Full audit requires archived parent directory')
        parent_path=args.parent_eval_dir/'full_resurface_sq3p25.json';s16_path=args.parent_eval_dir/'full_source_s16.json'
        need(sha(parent_path)==ARCHIVE['parent'] and sha(s16_path)==ARCHIVE['s16'],'Archived context changed')
        archived_parent=read(parent_path);s16=read(s16_path)
        scores(archived_parent,windows,cases,tokenizer);scores(s16,windows,cases,tokenizer)
        parent_replay=read(directory/'full_parent_replay.json');replay(archived_parent,outputs['old_sq_v2'],parent_replay)
        need(comparison['parent_replay']==parent_replay and comparison['selected']==selected
             and comparison['screening_report_sha256']==sha(args.screening_report),'Full selection/archive replay binding differs')
        screen_comparison=read(args.screening_report)
        need(comparison['selection']==screen_comparison['selection'],'Full candidate differs from frozen screen selection')
        change=recompute_pair(outputs['old_sq_v2'],outputs[selected+'_v2'],np)
        gap=recompute_pair(s16,outputs[selected+'_v2'],np)
        need(comparison['comparison']==change and comparison['remaining_gap_vs_archived_s16']==gap,
             'Independent paired bootstrap/arithmetic differs')
        checks=dict(ppl_at_least_1pct_better=change['ppl_relative_change']<=-.01,
            mk_no_observed_decrease=change['normal_mk_correct_delta']>=0,
            mk_95ci_lower_at_least_minus2pp=change['normal_mk_paired_bootstrap_95ci'][0]>=-.02,
            cache_same_budget=outputs[selected+'_v2']['cache']['total_bytes']==CACHE)
        need(comparison['gate_checks']==checks and type(comparison['repair_gate_pass']) is bool
             and comparison['repair_gate_pass']==all(checks.values())
             and comparison['original_ppl_restored_within_1pct']==(gap['ppl_relative_change']<=.01),
             'Repair/original-PPL gate differs')
        result.update(selection=comparison['selection'],paired=change,remaining_gap=gap,gate_checks=checks,
            repair_gate_pass=all(checks.values()),parent_replay=parent_replay)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true',help='Only independent CPU selection/packing fixtures; no external files')
    parser.add_argument('--stage',choices=('screen','full'))
    for name in ('source-dir','calibration','repair-calibration','parent-training-report','prose-tokens',
                 'codec-checks','eval-dir','screening-report','parent-eval-dir','output'):
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(),indent=2));return
    for key in ('stage','source_dir','calibration','repair_calibration','parent_training_report',
                'prose_tokens','codec_checks','eval_dir','output'):
        if getattr(args,key) is None:parser.error('--'+key.replace('_','-')+' is required')
    if args.stage=='full' and (args.screening_report is None or args.parent_eval_dir is None):
        parser.error('Full audit requires --screening-report and --parent-eval-dir')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Set CUDA_VISIBLE_DEVICES= for CPU-only audit')
    need(not args.output.exists(),'Preserve audit receipts: choose a fresh output')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1'
    result=dict(format='MAMBA2_STATE_REPAIR_INDEPENDENT_AUDIT_V1',complete=False,passed=False,stage=args.stage,
        protocol_sha256=PROTOCOL,source_sha256=sha(__file__),auditor_dependency_sha256={
            p:sha(ROOT/'scripts'/p) for p in ('audit_quant_first.py','audit_resurface_more.py')},limitations=[
            'Recorded NLL and generated IDs are audited; GPU logits/greedy execution are not independently rerun.',
            'Calibration table derivations are reconstructed from recorded sums; original GPU calibration traces are not rerun.',
            'Packing tests and actual allocation receipts are bound and checked; this CPU audit does not launch packing kernels.',
            'Base identity/version/gradient reports do not replace post-inference model-weight byte hashes.',
            'Zero-scale observations combine true zeros and underflow; they are not unique underflow event counts.',
            'Historically exposed datasets and prompt families do not establish untouched generalization.'])
    try:
        import numpy as np
        import torch
        from mamba2_recall import runtime,resurface_data as data
        from mamba2_recall.calibration import load_wikitext_tokens
        torch.set_num_threads(4)
        need(not torch.cuda.is_initialized(),'CUDA unexpectedly initialized before audit')
        result['self_tests']=self_test()
        tokenizer=runtime.SentencePieceTokenizer(args.source_dir)
        need(tokenizer.sha256==TOKENIZER,'Tokenizer changed')
        train,parent,payload,calibration,proof=audit_inputs(args,tokenizer,torch,np)
        result['inputs']=proof
        screen_dir=args.eval_dir if args.stage=='screen' else args.screening_report.parent
        if args.stage=='full':need(args.screening_report.name=='screen_comparison.json','Unexpected screening receipt filename')
        screen=audit_run(screen_dir,'screen',args,train,parent,payload,calibration,tokenizer,data,np)
        result['screen']=screen
        if args.stage=='full':
            selected=screen['selection']['selected'];need(selected is not None,'Stopped screen cannot advance')
            validation,dataset=load_wikitext_tokens(tokenizer,'validation')
            result['full']=audit_run(args.eval_dir,'full',args,train,parent,payload,calibration,tokenizer,data,np,
                                    validation,dataset,selected)
        need(not torch.cuda.is_initialized(),'CPU audit unexpectedly initialized CUDA')
        result.update(complete=True,passed=True,cuda_initialized=False)
    except BaseException as error:
        result['error']=repr(error);raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
