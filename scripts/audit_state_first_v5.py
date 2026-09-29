#!/usr/bin/env python3
"""Independent CPU audit of state-first v5 provenance, selection and results.

Reconstructs tables and selection without importing the v5 selector. Audits
recorded scores/checkpoints; does not regenerate GPU logits or gradients.
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
from audit_quant_first import (sha, need, finite, close, tokhash, read,
    recompute_pair, SOURCE, TOKENIZER, PROSE_FILE, PROSE_MANIFEST, VALIDATION_TOKENS)
from audit_state_repair import (audit_inputs as audit_v4_inputs, tensor_sha,
    hash_inventory, frozen, scores, replay, cache_check)
from audit_resurface_more import audit_backend_policy

PROTOCOL = 'ce700910f27230e062fe62e89dd6bddee05bcbf7bb7972c59fdba24cb3db9f5d'
NUMERIC_PROTOCOL = '24466642ce87c75fc2136a42ed14e69733c462b507a0be82c062b4da7f836bcb'
V4_STATISTICS = '8509bb266d40875608f3e0b3be22407fd6ea1b8aacbc79ce04e958726516b0a4'
OLD_CAL = 'c366cd577967e64635f1dd960237dc1c0024d7413685ce0b4c4e40da869a7023'
TRAIN_MANIFEST = '451b8703c21120667ef0ea272c21a10d4cd4561779a7b9663773ae47981600c0'
KINDS = ('magnitude', 'full_readout', 'preserve_int8', 'preserve_retained80')
CACHE = 28499968
RULE = 'Fixed v5: finite complete exact-budget; new MK>=max(1,ceil(.5*baseline)); lowest PPL, highest MK, candidate order; magnitude fallback stops'
ARCHIVES = dict(s16='52f82f83258a2fa3160ea14585f1bd69e1d68d60d8f179d1f0987c636f6546ba',
    raw='d8ec7b1ca239b385e444cfbec6c34f36728a459c6973c7f92690951e0447bc0d',
    v2='224a8d1201761e44072bd37211ec0c916cc0872095042bf26cde49d8a349b20b')


def binding():
    return dict(protocol_sha256=PROTOCOL, numeric_protocol_sha256=NUMERIC_PROTOCOL,
        source_sha256=SOURCE, tokenizer_sha256=TOKENIZER, train_file_sha256=PROSE_FILE,
        prose_manifest_sha256=PROSE_MANIFEST, train_manifest_sha256=TRAIN_MANIFEST,
        v4_statistics_sha256=V4_STATISTICS, original_calibration_sha256=OLD_CAL,
        adapter=None, adapter_sha256=None, heldout_used=False, calibration_tokens=4096)


def selection(results):
    """Separate arithmetic implementation, including a mandatory valid fallback."""
    valid=[]
    for name in KINDS:
        row=results[name]
        if (row.get('complete') is True and 'error' not in row and finite(row.get('ppl',{}).get('ppl'))
                and row.get('cache',{}).get('total_bytes')==CACHE
                and row.get('persistent_float_finite_checks_passed') is True
                and row.get('repeated_reset_probe',{}).get('hidden_and_cache_exact') is True
                and row.get('frozen_source',{}).get('identity_version_gradients_unchanged') is True
                and row.get('backend_policy_check',{}).get('singleton_config_unchanged') is True
                and row.get('adapter_loaded') is False and row.get('adapter_sha256') is None):
            valid.append(name)
    need('magnitude' in valid, 'Invalid magnitude baseline must stop the experiment')
    baseline_mk=results['magnitude']['mk']['summary']['normal']['correct']
    minimum=max(1,(baseline_mk+1)//2)
    admissible=[k for k in valid if k=='magnitude'
        or results[k]['mk']['summary']['normal']['correct']>=minimum]
    ranked=sorted(admissible,key=lambda k:(results[k]['ppl']['ppl'],
        -results[k]['mk']['summary']['normal']['correct'],KINDS.index(k)))
    chosen=ranked[0]
    return dict(selected_kind=chosen,valid=valid,admissible=admissible,minimum_mk_correct=minimum,
        baseline_mk_correct=baseline_mk,stopped=chosen=='magnitude',adapter_used=False,
        heldout_used=False,rule=RULE)


def derive_tables(old,scores_array,np):
    """Independent per-row Python set filtering of NumPy stable score ranks."""
    expected=dict(magnitude=old.copy(), full_readout=np.argsort(-scores_array,axis=-1,kind='stable').astype(np.uint8))
    keep8=np.empty_like(old);keep80=np.empty_like(old)
    for layer in range(56):
        for group in range(8):
            previous=[int(x) for x in old[layer,group]]
            ranked=[int(x) for x in expected['full_readout'][layer,group]]
            s8=set(previous[:16]);s80=set(previous[:80])
            keep8[layer,group]=previous[:16]+[x for x in ranked if x not in s8]
            keep80[layer,group]=[x for x in ranked if x in s80]+previous[80:]
    expected.update(preserve_int8=keep8,preserve_retained80=keep80)
    return expected


def self_test():
    def fixture(mk=11):
        base=dict(complete=True,ppl=dict(ppl=10.),cache=dict(total_bytes=CACHE),
                  mk=dict(summary=dict(normal=dict(correct=mk))),persistent_float_finite_checks_passed=True,
                  repeated_reset_probe=dict(hidden_and_cache_exact=True),
                  frozen_source=dict(identity_version_gradients_unchanged=True),
                  backend_policy_check=dict(singleton_config_unchanged=True),adapter_loaded=False,adapter_sha256=None)
        return {k:copy.deepcopy(base) for k in KINDS}
    checks=[]
    r=fixture();need(selection(r)['selected_kind']=='magnitude','Full tie must favor magnitude');checks.append('full-tie-baseline-fallback')
    r['full_readout']['ppl']['ppl']=9.;r['full_readout']['mk']['summary']['normal']['correct']=6
    need(selection(r)['selected_kind']=='full_readout' and selection(r)['minimum_mk_correct']==6,'Inclusive odd MK guard');checks.append('odd-ceil-guard-inclusive')
    r['full_readout']['mk']['summary']['normal']['correct']=5
    need(selection(r)['selected_kind']=='magnitude','Below guard excluded');checks.append('one-below-guard-excluded')
    r=fixture(10);need(selection(r)['minimum_mk_correct']==5,'Even MK guard');checks.append('even-half-guard')
    r=fixture(0);r['full_readout']['ppl']['ppl']=9.
    need(selection(r)['selected_kind']=='magnitude' and selection(r)['minimum_mk_correct']==1,'Zero baseline requires one');checks.append('zero-baseline-minimum-one')
    r['full_readout']['mk']['summary']['normal']['correct']=1
    need(selection(r)['selected_kind']=='full_readout','One at zero-baseline guard');checks.append('zero-baseline-one-admissible')
    r=fixture();r['full_readout']['mk']['summary']['normal']['correct']=12
    need(selection(r)['selected_kind']=='full_readout','Equal PPL higher MK advances');checks.append('equal-ppl-higher-mk-advances')
    r['preserve_int8']['mk']['summary']['normal']['correct']=12
    need(selection(r)['selected_kind']=='full_readout','Fixed candidate order');checks.append('new-candidate-tie-order')
    r['preserve_int8']['ppl']['ppl']=9.99
    need(selection(r)['selected_kind']=='preserve_int8','PPL ranks before MK');checks.append('ppl-primary')
    for bad,label in ((float('nan'),'nan'),(float('inf'),'infinite')):
        r=fixture();r['full_readout']['ppl']['ppl']=bad
        need('full_readout' not in selection(r)['valid'],'Nonfinite must exclude');checks.append(label+'-excluded')
    r=fixture();r['full_readout']['cache']['total_bytes']+=1
    need('full_readout' not in selection(r)['valid'],'Budget differs');checks.append('one-byte-over-budget')
    r=fixture();r['full_readout']['complete']=False
    need('full_readout' not in selection(r)['valid'],'Incomplete must exclude');checks.append('incomplete-excluded')
    for key,value in (('error','failed'),('persistent_float_finite_checks_passed',False),('adapter_loaded',True)):
        r=fixture();r['full_readout'][key]=value
        need('full_readout' not in selection(r)['valid'],'Failed evidence must exclude');checks.append(key+'-excludes')
    r=fixture();r['magnitude']['complete']=False
    try:selection(r)
    except ValueError:pass
    else:raise ValueError('Invalid magnitude was not fatal')
    checks.append('invalid-baseline-fatal')
    need(16+64//2+2*2==52 and 52*8/128==3.25 and 56*128*64*52+56*10240*4*2+57344==CACHE,
         'Tier payload/cache arithmetic differs');checks.append('52-byte-row-includes-scales-and-full-cache')
    start=dict(INITIAL_SCALER,_growth_tracker=17)
    after=scaler_next(start,True)
    need(after['scale']==512. and after['_growth_tracker']==17 and start['scale']==1024.,
         'Manual overflow must preserve successful growth tracker');checks.append('overflow-halves-scale-preserves-tracker')
    after=scaler_next(after,False)
    need(after['scale']==512. and after['_growth_tracker']==18,'Successful scaler advance');checks.append('success-increments-tracker')
    after=scaler_next(dict(INITIAL_SCALER,_growth_tracker=1999),False)
    need(after['scale']==2048. and after['_growth_tracker']==0,'Growth interval transition');checks.append('growth-at-2000-resets-tracker')
    need(factor(0)==1. and factor(1535)==.1 and factor(767)>factor(768),
         'Independent v2 cosine endpoints/monotonicity');checks.append('fresh1536-cosine-endpoints')
    return dict(complete=True,passed=True,count=len(checks),checks=checks,gpu_used=False)


def audit_candidates(args,tokenizer,torch,np):
    need(sha(ROOT/'docs/STATE_FIRST_V5_PROTOCOL.md')==PROTOCOL,'Frozen v5 protocol changed')
    need(sha(args.repair_calibration)==V4_STATISTICS,'Pinned original-S16 statistics changed')
    train,parent,v4,v4receipt,v4proof=audit_v4_inputs(args,tokenizer,torch,np)
    old=torch.load(args.calibration,map_location='cpu',weights_only=True)
    means=old['statistics']['mean_abs'].numpy()
    need(np.array_equal(old['permutations'].numpy(),np.argsort(-means,axis=-1,kind='stable').astype(np.uint8)),
         'Original magnitude permutation is not the stable magnitude ranking')
    expected=derive_tables(old['permutations'].numpy(),v4['statistics']['mean_readout_score'].numpy(),np)
    payload=torch.load(args.candidates,map_location='cpu',weights_only=True)
    receipt=read(args.candidates.with_suffix('.json'))
    fields=dict(format='MAMBA2_STATE_FIRST_CANDIDATES_V1',**binding(),
                train_token_hashes=[tokhash(train[i,:512].tolist()) for i in range(8)])
    need(all(payload.get(k)==receipt.get(k)==v for k,v in fields.items()),'Candidate provenance differs')
    need(receipt.get('complete') is True and receipt.get('fresh_source_no_adapter') is True
         and receipt['sha256']==sha(args.candidates) and receipt['bytes']==args.candidates.stat().st_size
         and receipt['runtime_table_bytes']==57344,'Candidate identity/storage receipt differs')
    need(tuple(payload['tables'])==tuple(receipt['table_sha256'])==KINDS,'Candidate inventory/order differs')
    for name,values in expected.items():
        table=payload['tables'][name]
        need(table.dtype==torch.uint8 and tuple(table.shape)==(56,8,128)
             and np.array_equal(table.numpy(),values)
             and np.array_equal(np.sort(values,axis=-1),np.broadcast_to(np.arange(128,dtype=np.uint8),values.shape))
             and tensor_sha(table)==receipt['table_sha256'][name], 'Independent table reconstruction differs: '+name)
    need(receipt['v4_calibration_receipt_sha256']==sha(args.repair_calibration.with_suffix('.json'))
         and receipt['collector_forward_exactcheck']==v4receipt['collection_forward_exactcheck'],
         'Candidate collector provenance differs')
    required={'scripts/prepare_state_first_v5.py','scripts/prepare_state_repair.py','scripts/prepare_quant_first.py',
        'scripts/run_state_repair.py','mamba2_recall/runtime.py','mamba2_recall/resurface_data.py',
        'docs/STATE_FIRST_V5_PROTOCOL.md','docs/STATE_REPAIR_PROTOCOL.md'}
    hash_inventory(receipt['code_sha256'],required,'CPU candidate preparation')
    return train,payload,receipt,dict(complete=True,independent_table_derivation=True,
        candidates_sha256=sha(args.candidates),table_sha256=receipt['table_sha256'],table_bytes=57344,
        packed_row_bytes_including_scales=52,total_cache_bytes=CACHE,upstream_calibration_audit=v4proof)


def run_paths():
    return {str(p.relative_to(ROOT)) for p in list((ROOT/'mamba2_recall').glob('*.py'))+[
        ROOT/'scripts'/name for name in ('run_state_first_v5.py','prepare_state_first_v5.py',
        'run_state_repair.py','prepare_quant_first.py','evaluate_quant_first.py',
        'evaluate_resurface_more.py','run_statequant.py')]+[
        ROOT/'docs/STATE_FIRST_V5_PROTOCOL.md',ROOT/'docs/RESURFACE_MORE_BACKEND_REPLAY.md']}


def audit_selected(path,args,candidates,receipt,screen,torch):
    chosen=screen['selection']['selected_kind']
    need(chosen!='magnitude' and not screen['selection']['stopped'],'Magnitude fallback cannot train')
    payload=torch.load(path,map_location='cpu',weights_only=True);record=read(path.with_suffix('.json'))
    expected=dict(format='MAMBA2_STATE_FIRST_CALIBRATION_V1',**binding(),selected_kind=chosen,
        candidates_sha256=sha(args.candidates),selection_report_sha256=screen['comparison_sha256'],
        table_sha256=receipt['table_sha256'][chosen],train_token_hashes=candidates['train_token_hashes'])
    need(all(payload.get(k)==record.get(k)==v for k,v in expected.items()),'Frozen selected table provenance differs')
    need(record.get('complete') is True and record.get('fresh_source_no_adapter') is True
         and record['sha256']==sha(path) and record['bytes']==path.stat().st_size
         and record['file']==path.name,'Selected payload receipt differs')
    need(payload['permutations'].dtype==torch.uint8 and tuple(payload['permutations'].shape)==(56,8,128)
         and torch.equal(payload['permutations'],candidates['tables'][chosen])
         and tensor_sha(payload['permutations'])==record['table_sha256'],'Frozen selected coordinates differ')
    hash_inventory(record['code_sha256'],run_paths(),'Selection export')
    return payload,record


def audit_run(directory,stage,args,train,candidates,receipt,tokenizer,data,np,
              screen=None,selected=None,training=None,validation=None,dataset=None):
    comparison=read(directory/(stage+'_comparison.json'))
    need(comparison.get('format')=='MAMBA2_STATE_FIRST_COMPARISON_V1' and comparison.get('complete') is True
         and comparison.get('stage')==stage and comparison.get('protocol_sha256')==PROTOCOL
         and comparison.get('candidates_sha256')==sha(args.candidates),'Evaluation comparison binding differs')
    hash_inventory(comparison['code_hashes'],run_paths(),'Evaluation')
    if stage=='screen':
        windows=[(i*2048,train[i,:512].tolist()) for i in range(8,40)]
        cases=[c for c in data.generate_cases('train') if c['sample'] in range(0,256,16)]
        expected_dataset=dict(split='train',file_sha256=PROSE_FILE,rows=list(range(8,40)),tokens_per_row=512)
        arms=[(k,k,False) for k in KINDS]+[('restored_magnitude','magnitude',False)]
        need(len(cases)==96 and sum(len(w)-1 for _,w in windows)==16352,'TRAIN screen geometry differs')
        need(comparison.get('adapter_sha256') is None,'Screen must be unadapted')
    else:
        need(screen is not None and selected is not None and training is not None,'Full inputs absent')
        need(tokhash(validation.tolist())==VALIDATION_TOKENS,'Validation token stream differs')
        windows=[(s,validation[s:min(s+2049,len(validation))].tolist()) for s in range(0,len(validation)-1,2048)]
        cases=data.generate_cases('confirm');expected_dataset=dataset
        chosen=selected['selected_kind']
        arms=[('old_magnitude','magnitude',False),('selected_no_adapter',chosen,False),
              ('selected_resurface',chosen,True),('restored_selected',chosen,False)]
        need(len(windows)==130 and sum(len(w)-1 for _,w in windows)==264764 and len(cases)==768,'Full population differs')
    common=dict(format='MAMBA2_STATE_FIRST_EVAL_V1',stage=stage,**binding(),source_checkpoint_sha256=SOURCE,
        candidates_sha256=sha(args.candidates),candidates_receipt_sha256=sha(args.candidates.with_suffix('.json')),
        dataset=expected_dataset,engine_kind='old_sq',
        engine_scope='Unchanged packed16INT8/64INT4/48zero codec; candidate_name/table hash identifies the actual coordinates',
        selected_calibration_sha256=sha(args.selected_calibration) if selected else None,
        training_report_sha256=sha(args.training_report) if training else None,
        training_binding=training['binding'] if training else None,
        selection_scope='Unadapted TRAIN only; table frozen before fresh training and validation/CONFIRM')
    # The per-arm adapter SHA intentionally overrides the common no-adapter preparation binding.
    common.pop('adapter_sha256')
    common.pop('adapter')
    common.update(heldout_used=stage=='full',heldout_used_for_selection=False,calibration_heldout_used=False)
    need(set(comparison['report_sha256'])=={arm for arm,_,_ in arms},'Evaluation arm inventory differs')
    outputs={};metrics={};digests={}
    for arm,name,adapted in arms:
        path=directory/(stage+'_'+arm+'.json');row=read(path);digests[arm]=sha(path)
        need(digests[arm]==comparison['report_sha256'][arm] and row.get('arm')==arm
             and row.get('candidate_name')==name and row.get('candidate_table_sha256')==receipt['table_sha256'][name]
             and row.get('adapter_loaded') is adapted
             and row.get('adapter_sha256')==(training['adapter']['sha256'] if adapted else None)
             and all(row.get(k)==v for k,v in common.items()),'Arm identity/binding differs: '+arm)
        need(row['code_hashes']==comparison['code_hashes'] and row.get('adapter_content_unchanged') is True
             and row.get('candidate_table_unchanged') is True,'Code/adapter/table changed across arms')
        frozen(row['frozen_source']);audit_backend_policy(row['backend_policy'],row['backend_policy_check'])
        need(row['backend_policy']==comparison['backend_policy'],'Evaluation backend changed across arms')
        done=row.get('complete') is True
        if not done:
            need(stage=='screen' and name!='magnitude' and row.get('excluded_from_selection') is True
                 and isinstance(row.get('error'),str) and row['error'],'Unexpected failed arm')
        else:
            need('error' not in row and not row.get('excluded_from_selection',False)
                 and row.get('persistent_float_finite_checks_passed') is True and row.get('kind')=='old_sq',
                 'Completed evaluation lacks finite packed result')
            probe=row['persistent_cache_probe'];cache_check(probe['cache'],'old_sq',128)
            hashes=probe['cache_tensor_sha256']
            expected_keys={f'{i}.{k}' for i in range(56) for k in ('lo','hi','q4','s8','s4','conv')}
            need(probe.get('tokens')==128 and probe.get('both_repeats_exact') is True
                 and re.fullmatch('[0-9a-f]{64}',probe['hidden_sha256']) is not None
                 and set(hashes)==expected_keys
                 and all(isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) for v in hashes.values()),
                 'Repeated packed/conv cache hash proof differs')
            internal=row['repeated_reset_probe']
            need(internal.get('tokens')==128 and internal.get('hidden_and_cache_exact') is True
                 and internal['token_sha256']==tokhash(windows[0][1][:128])
                 and internal['hidden_sha256']==probe['hidden_sha256'] and internal['cache_bytes']==CACHE,
                 'Independent preflight/engine repeated-reset evidence differs')
            last=row['mk']['rows'][-1];cache_check(row['cache'],'old_sq',last['prompt_tokens']+len(last['generated_ids'])-1)
            need(type(row['zero_scale_observations']) is int and row['zero_scale_observations']>=0
                 and row['zero_scale_scope']=='Repeated final-cache observations; includes true zero states and FP16 scale underflow; not unique underflow events',
                 'Zero-scale observation scope differs')
        metrics[arm]=scores(row,windows,cases,tokenizer,done);outputs[arm]=row
    before,after=('magnitude','restored_magnitude') if stage=='screen' else ('selected_no_adapter','restored_selected')
    restoration=read(directory/(stage+'_restoration.json'));replay(outputs[before],outputs[after],restoration)
    need(comparison['restoration']==restoration and restoration['scope']=='Exact no-adapter replay after candidate/adapter removal',
         'Restoration receipt/scope differs')
    result=dict(complete=True,arms=metrics,report_sha256=digests,restoration=restoration,
                comparison_sha256=sha(directory/(stage+'_comparison.json')))
    if stage=='screen':
        chosen=selection(outputs)
        need(comparison['selection']==chosen and comparison['selected_table_sha256']==receipt['table_sha256'][chosen['selected_kind']],
             'Independent unadapted TRAIN selection differs')
        result['selection']=chosen
        if chosen['stopped']:
            need(not (directory/'selected_calibration.pt').exists() and not (directory/'selected_calibration.json').exists(),
                 'Stopped magnitude screen unexpectedly exported a training table')
    else:
        need(args.parent_eval_dir is not None,'Full audit requires archived context')
        files={'s16':'full_source_s16.json','raw':'full_source_sq3p25.json','v2':'full_resurface_sq3p25.json'}
        archives={}
        for name,file in files.items():
            path=args.parent_eval_dir/file;need(sha(path)==ARCHIVES[name],'Archived context differs: '+name)
            archives[name]=read(path);scores(archives[name],windows,cases,tokenizer)
        need(archives['raw']['adapter_sha256'] is None and archives['s16']['adapter_sha256'] is None
             and archives['v2']['adapter_sha256']=='7144b88265a797ef935b1f94845395abec5532d8dc5b2a4a8fa0a8e07ce205f0',
             'Archived adapter/control identity differs')
        parent_replay=read(directory/'full_parent_replay.json');replay(archives['raw'],outputs['old_magnitude'],parent_replay)
        need(parent_replay['scope']=='Exact archived unadapted magnitude SQ3.25 replay'
             and comparison['parent_replay']==parent_replay,'Archived unadapted replay differs')
        expected_archives={files[k][5:-5]:v for k,v in ARCHIVES.items()}
        need(comparison['selected_kind']==selected['selected_kind'] and comparison['selection']==screen['selection']
             and comparison['selected_calibration_sha256']==sha(args.selected_calibration)
             and comparison['selection_report_sha256']==screen['comparison_sha256']
             and comparison['training_report_sha256']==sha(args.training_report)
             and comparison['adapter_sha256']==training['adapter']['sha256']
             and comparison['archived_report_sha256']==expected_archives,'Full selection/training/context binding differs')
        changes=dict(unadapted_state_optimization=recompute_pair(outputs['old_magnitude'],outputs['selected_no_adapter'],np),
            resurface_repair=recompute_pair(outputs['selected_no_adapter'],outputs['selected_resurface'],np),
            prior_endpoint_improvement=recompute_pair(archives['v2'],outputs['selected_resurface'],np),
            original_s16=recompute_pair(archives['s16'],outputs['selected_resurface'],np))
        q,r,p,o=(changes[k] for k in ('unadapted_state_optimization','resurface_repair','prior_endpoint_improvement','original_s16'))
        same=all(v['cache']['total_bytes']==CACHE for v in outputs.values())
        gates=dict(unadapted_state_optimization=dict(ppl_at_least_1pct_better=q['ppl_relative_change']<=-.01,cache_same_budget=same),
            resurface_repair=dict(ppl_no_more_than_1pct_worse=r['ppl_relative_change']<=.01,
                normal_mk_increases=r['normal_mk_correct_delta']>0,mk_bootstrap_lower_positive=r['normal_mk_paired_bootstrap_95ci'][0]>0,
                cache_same_budget=same),
            prior_endpoint_improvement=dict(ppl_at_least_1pct_better=p['ppl_relative_change']<=-.01,
                mk_no_observed_decrease=p['normal_mk_correct_delta']>=0,
                mk_95ci_lower_at_least_minus2pp=p['normal_mk_paired_bootstrap_95ci'][0]>=-.02,cache_same_budget=same))
        need(comparison['comparisons']==changes and comparison['gate_checks']==gates
             and comparison['gates_pass']=={k:all(v.values()) for k,v in gates.items()}
             and comparison['original_ppl_restored_within_1pct']==(o['ppl_relative_change']<=.01),
             'Independent full paired bootstrap/three gates differ')
        result.update(selection=screen['selection'],comparisons=changes,gate_checks=gates,
            gates_pass={k:all(v.values()) for k,v in gates.items()},parent_replay=parent_replay)
    return result


V2_TRAINER = '547e5b64904a273cd96a7d77624e5225a67edb9fa683950a43efc2d4171392a6'
INITIAL_SCALER = dict(scale=1024.,growth_factor=2.,backoff_factor=.5,growth_interval=2000,_growth_tracker=0)
INITIAL_ADAPTER = 'fresh V=0,g=1,w=0,b=-4; no pretrained adapter or checkpoint'


def training_paths():
    return {str(p.relative_to(ROOT)) for p in list((ROOT/'mamba2_recall').glob('*.py'))+[
        ROOT/'scripts'/name for name in ('train_state_first_v5.py','train_quant_first.py','train_resurface_more.py',
        'evaluate_quant_first.py','evaluate_resurface_more.py','run_statequant.py','prepare_state_first_v5.py')]+[
        ROOT/'docs'/name for name in ('STATE_FIRST_V5_PROTOCOL.md','QUANT_FIRST_PROTOCOL.md','RESURFACE_MORE_BACKEND_REPLAY.md')]}


def shapes():
    return {f'layer{i}.{name}':shape for i in range(56) for name,shape in (
        ('V_read',(128,128)),('g_read',(128,)),('router_w',(4096,)),('router_b',()))}


def scaler_next(before,overflow):
    expected=copy.deepcopy(before)
    if overflow:expected['scale']*=.5
    else:
        expected['_growth_tracker']+=1
        if expected['_growth_tracker']==expected['growth_interval']:
            expected['_growth_tracker']=0;expected['scale']*=2.
    return expected


def factor(index):
    return .1+.9*(1+math.cos(math.pi*index/1535))/2


def check_optimizer_independent(checkpoint,step,torch):
    masters=checkpoint['masters'];expected_shapes=shapes()
    need(set(masters)==set(expected_shapes),'Master tensor inventory differs')
    for name,value in masters.items():
        need(value.dtype==torch.float32 and tuple(value.shape)==expected_shapes[name]
             and bool(torch.isfinite(value).all()) and bool(torch.isfinite(value.half()).all()),'Invalid FP32 master: '+name)
    names=[[f'layer{i}.{field}' for i in range(56) for field in fields]
           for fields in (('V_read','g_read'),('router_w','router_b'))]
    need(checkpoint['optimizer_parameter_names']==names,'Optimizer-to-master group names differ')
    optimizer=checkpoint['optimizer'];groups=optimizer['param_groups'];states=optimizer['state']
    need(len(groups)==2 and len(states)==224,'Adam state inventory differs')
    seen=[]
    for group,group_names,base_lr in zip(groups,names,(1e-4,3e-4)):
        need(len(group['params'])==112 and group['base_lr']==base_lr
             and group['betas']==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==0.
             and group['amsgrad'] is False and group['maximize'] is False and group['differentiable'] is False,
             'Adam recipe differs')
        close(group['lr'],base_lr*factor(step-1),'Adam checkpoint LR differs',tol=1e-14)
        for index,name in zip(group['params'],group_names):
            seen.append(index);value=states[index]
            need(set(value)=={'step','exp_avg','exp_avg_sq'} and isinstance(value['step'],torch.Tensor)
                 and value['step'].numel()==1 and value['step'].dtype==torch.float32
                 and value['step'].item()==step,'Adam step/status differs')
            for key in ('exp_avg','exp_avg_sq'):
                moment=value[key]
                need(moment.dtype==torch.float32 and tuple(moment.shape)==expected_shapes[name]
                     and bool(torch.isfinite(moment).all()),'Invalid Adam moment: '+name)
            need(bool((value['exp_avg_sq']>=0).all()),'Adam second moment is negative')
    need(len(set(seen))==224 and set(seen)==set(states),'Adam missing or duplicated parameter state')
    return names


def audit_training_report(path,mode,args,selected,train,tokenizer,torch,data,smoke=None):
    report=read(path);limit=1 if mode=='smoke' else 1536
    need(report.get('format')=='MAMBA2_STATE_FIRST_TRAIN_V1' and report.get('complete') is True
         and report.get('mode')==mode and report.get('successful_updates')==limit
         and report.get('planned_successful_updates')==1536 and limit<=report.get('attempts',-1)<=limit+8
         and 'error' not in report,'Incomplete fresh training/smoke report')
    need(sha(ROOT/'scripts/train_quant_first.py')==V2_TRAINER,'Reused v2 training math changed')
    hash_inventory(report['code_sha256'],training_paths(),'Fresh training')
    need(report['source_sha256']==sha(ROOT/'scripts/train_state_first_v5.py'),'Trainer source identity differs')
    manifest=read(ROOT/'docs/prose_train_manifest.json')
    expected_binding=dict(calibration_sha256=sha(args.selected_calibration),
        calibration_receipt_sha256=sha(args.selected_calibration.with_suffix('.json')),
        calibration_format='MAMBA2_STATE_FIRST_CALIBRATION_V1',selected_kind=selected['selected_kind'],
        table_sha256=selected['table_sha256'],candidates_sha256=sha(args.candidates),
        selection_report_sha256=selected['selection_report_sha256'],v4_statistics_sha256=V4_STATISTICS,
        original_calibration_sha256=OLD_CAL,initial_adapter=INITIAL_ADAPTER,fresh_initialization=True,
        prior_adapter_loaded=False,checkpoint_loaded=False,v2_trainer_sha256=V2_TRAINER,
        state_mode='sq3p25; exact packed forward; live-mask STE backward',
        teacher='separate unadapted source, S16 per-token carry',source_checkpoint_sha256=SOURCE,
        tokenizer_sha256=TOKENIZER,protocol_sha256=PROTOCOL,numeric_protocol_sha256=NUMERIC_PROTOCOL,
        train_manifest_sha256=TRAIN_MANIFEST,prose_manifest_sha256=PROSE_MANIFEST,
        prose_tokens_sha256=PROSE_FILE,prose_tokens_int64le_sha256=manifest['training_tokens_sha256_int64le'],
        adapter='MAMBA2_POST_D_RESURFACE_FP16_V1',successful_updates=1536)
    need(report['binding']==expected_binding,'Fresh training selected table/data/recipe binding differs')
    fresh=report['fresh_initialization'];expected_shapes=shapes()
    need(fresh.get('all_224_masters_exact_fresh_values') is True and fresh.get('masters_dtype')=='float32'
         and fresh.get('master_tensors')==224 and fresh.get('parameters')==1154104
         and fresh.get('optimizer_state_empty') is True and fresh.get('optimizer_steps')==0
         and fresh.get('scaler_exact_initial') is True and fresh.get('scaler')==INITIAL_SCALER
         and fresh.get('prior_adapter_loaded') is False and fresh.get('checkpoint_loaded') is False,
         'Fresh master/optimizer/scaler initialization receipt differs')
    expected_initial={}
    for name,shape in expected_shapes.items():
        value={'V_read':0.,'g_read':1.,'router_w':0.,'router_b':-4.}[name.rsplit('.',1)[-1]]
        expected_initial[name]=tensor_sha(torch.full(shape,value,dtype=torch.float32))
    need(fresh['initial_tensor_sha256']==expected_initial,'Fresh constant master tensor hashes differ')
    names=[[f'layer{i}.{field}' for i in range(56) for field in fields]
           for fields in (('V_read','g_read'),('router_w','router_b'))]
    need(report['optimizer_parameter_names']==names,'Fresh optimizer parameter mapping differs')
    probe=tokhash(train[0,:128].tolist())
    init=report['initialization_check'];deployed=report['deployed_export_check']
    need(init.get('fresh_identity_matches_packed_bitwise') is True and init.get('probe_tokens')==128
         and init.get('probe_token_sha256')==probe and init.get('packed_probe_finite') is True,
         'Initial packed identity parity receipt differs')
    cache_check(init['cache'],'old_sq',128);cache_check(deployed['cache'],'old_sq',128)
    need(deployed.get('packed_training_forward_bitwise_equal') is True and deployed.get('probe_tokens')==128
         and deployed.get('probe_token_sha256')==probe and deployed.get('cache_unchanged_from_selected_unadapted') is True
         and deployed.get('packed_probe_finite') is True and init['cache']==deployed['cache'],
         'Actual exported FP16 packed parity evidence differs')
    frozen(report['frozen_base_check'])
    need(report.get('frozen_state_calibration_check') is True and report.get('selected_table_bytes_unchanged') is True
         and report.get('teacher_base_parameters_frozen') is True
         and report.get('export_master_check')==dict(all_master_casts_equal_export=True,master_tensors=224),
         'Frozen source/table/teacher or master-cast receipt differs')
    audit_backend_policy(report['backend_policy'],report['backend_final_check'])
    ordered=torch.randperm(1536,generator=torch.Generator().manual_seed(2026092803)).tolist()
    prose_order=torch.randperm(448,generator=torch.Generator().manual_seed(20260928)).tolist()
    need(prose_order==manifest['schedule'],'Prose schedule differs')
    need(report['schedule']==dict(seed=2026092803,successful_updates=1536,
        numeric_order_sha256_int64le=tokhash(ordered),prose_global_start=0,
        unchanged_v2_schedule_pair_optimizer_attempt=True),'Numeric schedule identity differs')
    cases=data.generate_cases('train');need(len(cases)==1536,'Numeric TRAIN inventory differs')
    success=overflows=0;scale=copy.deepcopy(INITIAL_SCALER);checkpoint_rows={}
    need(len(report['history'])==report['attempts'],'Attempt history missing')
    for attempt,row in enumerate(report['history'],1):
        need(success<limit and type(row['overflow']) is bool,'Unexpected training update/overflow flag')
        index=success;case=cases[ordered[index]]
        prompt=tokenizer.encode(case['prompt']);full=tokenizer.encode(case['prompt']+' '+case['answer'])
        need(full[:len(prompt)]==prompt,'Answer token prefix mismatch')
        need(row['attempt']==attempt and row['update_index']==index and row['schedule_entry']==ordered[index]
             and row['case_id']==case['id'] and row['prose_window']==prose_order[index%448]
             and row['prose_start']==512*((index//448)%4) and row['answer_targets']==len(full)-len(prompt),
             'Successful-update schedule/pairing differs')
        need(row['scaler_before']==scale and row['loss_scale_before']==scale['scale'],'Scaler before-attempt differs')
        scale=scaler_next(scale,row['overflow'])
        need(row['scaler_after']==scale and row['loss_scale']==scale['scale'],'Scaler transition differs')
        close(row['lr_factor'],factor(index),'Cosine schedule differs',tol=1e-14)
        need(len(row['learning_rates'])==2,'Learning-rate group count differs')
        for value,base_lr in zip(row['learning_rates'],(1e-4,3e-4)):
            close(value,base_lr*factor(index),'Learning-rate group value differs',tol=1e-14)
        need(all(finite(row[k]) for k in ('mk_ce','prose_ce','prose_kl','prose_closure','seconds'))
             and row['seconds']>=0 and row.get('successful_attempt_requires_v2_preclip_and_hidden_gradients_finite') is True
             and type(row['all_adapter_gradients_finite_after_attempt']) is bool,'Training loss/gradient evidence differs')
        if row['overflow']:
            overflows+=1;need(row['gradient_norm_before_clip'] is None,'Overflow unexpectedly updated parameters')
        else:
            success+=1
            need(row['all_adapter_gradients_finite_after_attempt'] is True
                 and finite(row['gradient_norm_before_clip']) and row['gradient_norm_before_clip']>=0,
                 'Successful update lacks finite gradients')
            if success%384==0:checkpoint_rows[success]=row
        need(row['successful_updates']==success,'Successful update count differs')
    need(success==limit and overflows<=8 and report['overflows']==overflows and report['attempts']==success+overflows
         and report['final_scaler']==scale,'Final step/total overflow/scaler differs')
    export=report['smoke_adapter'] if mode=='smoke' else report['adapter']
    need(Path(export['file']).name==export['file'],'Unsafe adapter filename')
    adapter_path=path.parent/export['file']
    need(sha(adapter_path)==export['sha256'] and adapter_path.stat().st_size==export['bytes']
         and export.get('roundtrip_bitwise_equal') is True and export.get('gate_mode')=='soft',
         'Serialized adapter identity differs')
    adapter=torch.load(adapter_path,map_location='cpu',weights_only=True)
    need(adapter['format']=='MAMBA2_POST_D_RESURFACE_FP16_V1' and adapter['gate_mode']=='soft'
         and adapter['variant']=='post-D native norm-prehook; memoryless cross-head mixing'
         and adapter['binding']==expected_binding
         and adapter['geometry']==[dict(width=4096,heads=128,head_dim=64)]*56,'Export adapter header differs')
    tensors=adapter['tensors'];need(set(tensors)==set(expected_shapes),'Adapter tensor inventory differs')
    for name,value in tensors.items():
        need(value.dtype==torch.float16 and tuple(value.shape)==expected_shapes[name] and bool(torch.isfinite(value).all())
             and tensor_sha(value)==export['tensor_sha256'][name],'Actual FP16 export tensor differs: '+name)
    need(sum(v.numel() for v in tensors.values())==export['parameters']==1154104
         and sum(v.numel()*v.element_size() for v in tensors.values())==export['payload_bytes']==2308208,
         'FP16 adapter parameter/byte accounting differs')
    if mode=='smoke':
        need(report['checkpoints']==[] and 'adapter' not in report and 'final_checkpoint' not in report
             and export.get('discarded') is True and export.get('candidate') is False
             and export['file']=='discarded_smoke_adapter_fp16.pt','Smoke was not explicitly discarded')
    else:
        need(smoke is not None and args.smoke_report.parent.resolve()!=path.parent.resolve(),
             'Formal training requires separate discarded smoke evidence')
        smoke_report,smoke_proof=smoke
        expected_smoke=dict(path=str(args.smoke_report),sha256=sha(args.smoke_report),discarded=True,
            successful_updates=1,formal_reinitializes_masters_optimizer_scaler=True,
            smoke_export_sha256=smoke_report['smoke_adapter']['sha256'],backend_policy_exact=True)
        # Stored path may be remote while auditing a copied evidence directory.
        actual_smoke=report['discarded_smoke_check'];need(Path(actual_smoke['path']).name==args.smoke_report.name,'Smoke path identity differs')
        need({k:v for k,v in actual_smoke.items() if k!='path'}=={k:v for k,v in expected_smoke.items() if k!='path'}
             and smoke_report['binding']==report['binding'] and smoke_report['code_sha256']==report['code_sha256']
             and smoke_report['backend_policy']==report['backend_policy'],'Formal/smoke identity or fresh restart binding differs')
        need(len(report['checkpoints'])==4,'Fixed recovery checkpoint count differs')
        final_checkpoint=None
        for step,item in zip((384,768,1152,1536),report['checkpoints']):
            checkpoint_path=path.parent/Path(item['path']).name
            need(checkpoint_path.name==item['file']==f'checkpoint_{step:04d}.pt'
                 and checkpoint_path.stat().st_size==item['bytes'] and sha(checkpoint_path)==item['sha256']
                 and item['successful_updates']==step and item['attempts']==checkpoint_rows[step]['attempt'],
                 'Recovery checkpoint receipt/count differs')
            checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
            need(checkpoint['format']=='MAMBA2_STATE_FIRST_CHECKPOINT_V1' and checkpoint['binding']==expected_binding
                 and checkpoint['successful_updates']==step and checkpoint['attempts']==item['attempts']
                 and checkpoint['scaler']==checkpoint_rows[step]['scaler_after'],'Checkpoint binding/scaler differs')
            check_optimizer_independent(checkpoint,step,torch)
            final_checkpoint=checkpoint
        need(report['final_checkpoint']==report['checkpoints'][-1]
             and report['final_checkpoint_export_check']==dict(all_master_casts_equal_export=True,master_tensors=224,
                 optimizer_exact=True,scaler_exact=True,final_checkpoint_sha256=report['final_checkpoint']['sha256']),
             'Final checkpoint/export receipt differs')
        need(all(torch.equal(value.half(),tensors[name]) for name,value in final_checkpoint['masters'].items()),
             'Final checkpoint FP32 master casts do not equal actual exported FP16 tensors')
    proof=dict(complete=True,mode=mode,report_sha256=sha(path),adapter_sha256=export['sha256'],
        successful_updates=success,attempts=report['attempts'],overflows=overflows,
        independent_initial_master_hashes=True,independent_schedule_and_scaler=True,
        recovery_checkpoints_verified=0 if mode=='smoke' else 4,
        actual_final_checkpoint_cast_equals_export=mode=='formal',actual_serialized_FP16_parity_recorded=True,
        optimizer_scope=('Smoke records fresh empty optimizer and retry/scaler evidence; no stored checkpoint moments or masters are available for independent comparison.'
            if mode=='smoke' else 'Checks complete finite moments, mapping, recipe and step counts; does not recompute GPU gradients or Adam updates.'))
    return report,proof


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--stage',choices=('candidates','screen','smoke','training','full'))
    for name in ('source-dir','calibration','repair-calibration','parent-training-report','prose-tokens','codec-checks',
                 'candidates','eval-dir','screening-report','selected-calibration','training-report',
                 'smoke-report','parent-eval-dir','output'):
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test(),indent=2));return
    for key in ('stage','source_dir','calibration','repair_calibration','parent_training_report','prose_tokens',
                'codec_checks','candidates','output'):
        if getattr(args,key) is None:parser.error('--'+key.replace('_','-')+' is required')
    if args.stage in ('screen','full') and args.eval_dir is None:parser.error('--eval-dir is required')
    if args.stage in ('smoke','training','full'):
        for key in ('screening_report','selected_calibration','smoke_report'):
            if getattr(args,key) is None:parser.error('--'+key.replace('_','-')+' is required')
    if args.stage in ('training','full') and args.training_report is None:parser.error('--training-report is required')
    if args.stage=='full' and args.parent_eval_dir is None:parser.error('--parent-eval-dir is required')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Set CUDA_VISIBLE_DEVICES= for CPU-only audit')
    need(not args.output.exists(),'Choose a fresh audit output; preserve previous receipts')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1'
    result=dict(format='MAMBA2_STATE_FIRST_INDEPENDENT_AUDIT_V1',complete=False,passed=False,stage=args.stage,
        protocol_sha256=PROTOCOL,source_sha256=sha(__file__),auditor_dependency_sha256={
            p:sha(ROOT/'scripts'/p) for p in ('audit_quant_first.py','audit_state_repair.py','audit_resurface_more.py')},
        limitations=['CPU audit verifies recorded evidence; it does not independently regenerate GPU logits or training gradients.',
            'Calibration tables are reconstructed from recorded sums, not fresh original GPU traces.',
            'Source identity/version/gradient receipts do not replace post-inference weight byte hashes.',
            'Historically exposed benchmark families do not establish untouched generalization.'])
    try:
        import numpy as np
        import torch
        from mamba2_recall import runtime,resurface_data as data
        torch.set_num_threads(4)
        need(not torch.cuda.is_initialized(),'CUDA initialized before CPU audit')
        result['self_tests']=self_test()
        tokenizer=runtime.SentencePieceTokenizer(args.source_dir)
        need(tokenizer.sha256==TOKENIZER,'Tokenizer changed')
        train,payload,receipt,proof=audit_candidates(args,tokenizer,torch,np)
        result['candidates']=proof
        if args.stage!='candidates':
            screen_dir=args.eval_dir if args.stage=='screen' else args.screening_report.parent
            if args.stage!='screen':need(args.screening_report.name=='screen_comparison.json','Screening receipt name differs')
            screen=audit_run(screen_dir,'screen',args,train,payload,receipt,tokenizer,data,np)
            result['screen']=screen
            if not screen['selection']['stopped']:
                selected_path=screen_dir/'selected_calibration.pt'
                if args.selected_calibration is None:args.selected_calibration=selected_path
                need(args.selected_calibration.resolve()==selected_path.resolve(),'Selected table must accompany frozen screen')
                selected,selected_receipt=audit_selected(selected_path,args,payload,receipt,screen,torch)
                result['selected_calibration']=dict(complete=True,sha256=sha(selected_path),
                    receipt_sha256=sha(selected_path.with_suffix('.json')),selected_kind=selected['selected_kind'],
                    table_sha256=selected['table_sha256'])
            elif args.stage in ('smoke','training','full'):
                raise ValueError('Stopped magnitude screen cannot advance')
            if args.stage in ('smoke','training','full'):
                smoke=audit_training_report(args.smoke_report,'smoke',args,selected,train,tokenizer,torch,data)
                result['discarded_smoke']=smoke[1]
            if args.stage in ('training','full'):
                training,training_proof=audit_training_report(args.training_report,'formal',args,selected,train,tokenizer,torch,data,smoke)
                result['training']=training_proof
            if args.stage=='full':
                from mamba2_recall.calibration import load_wikitext_tokens
                validation,dataset=load_wikitext_tokens(tokenizer,'validation')
                result['full']=audit_run(args.eval_dir,'full',args,train,payload,receipt,tokenizer,data,np,
                    screen,selected,training,validation,dataset)
        need(not torch.cuda.is_initialized(),'CPU audit unexpectedly initialized CUDA')
        result.update(complete=True,passed=True,cuda_initialized=False)
    except BaseException as error:
        result['error']=repr(error);raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
