#!/usr/bin/env python3
"""Independent CPU audit of v8 layer interventions, mixtures and PPL evidence.

Never imports the v8 calibration/selection implementation. Rebuilds the fixed
170-arm inventory, raw NLL ranking, layer swaps and actual-table deduplication.
No GPU execution, adapter training, MK scoring or regenerated model logits.
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import audit_state_ppl_v6 as v6
from audit_quant_first import sha,need,finite,tokhash,read,TOKENIZER,PROSE_FILE,VALIDATION_TOKENS
from audit_state_repair import hash_inventory,frozen,tensor_sha
from audit_resurface_more import audit_backend_policy

PROTOCOL='880839c0b0919d0a9a119647d2d9c1657e5b8eb5389c8c2ee22ee1785917aa01'
V6_SELECTED='098930d1af5e5821b277640d236f7607c6428b48d36f7117e2ea4aa87e656303'
V6_SCREEN='5ac624862846b61aacc6a5f3b7648db32d5d74bf8419d8d536bbc751cdd117d6'
V6_BASELINE='c3525d74ba0ee35e0b9d83a029e645123d14e90f5ca32b164931d24ec8359253'
V6_COMPARISON='c47e381e02edd6e33ba2b0a85a4f9c65d702cc7d3d93dacfe7546d409e6612d4'
BASELINE_KIND='preserve_int8'
BASELINE_POLICY='preserve_int8__stored_scale'
ALTERNATIVES=('magnitude','full_readout','preserve_retained80')
CALIBRATION_ROWS=tuple(range(72,80))
SCREEN_ROWS=tuple(range(112,144))
ARMS=('baseline',)+tuple(f'layer{layer:02d}_{kind}' for layer in range(56) for kind in ALTERNATIVES)+('restored_baseline',)
REQUESTS=(('baseline',0),('top1',1),('top2',2),('top4',4),('top8',8),('top16',16),('allnegative',None))
CACHE=28499968
CAL_FORMAT='MAMBA2_STATE_PPL_LAYER_EVAL_V1'
CAL_COMPARISON='MAMBA2_STATE_PPL_LAYER_CALIBRATION_V1'
CANDIDATES_FORMAT='MAMBA2_STATE_PPL_LAYER_CANDIDATES_V1'
ALLOWED_ERRORS={f'CandidateInvalid({message!r})' for message in (
    'PPL exponent overflow','Nonfinite PPL','Nonfinite persisted state, FP16 scale, or convolution cache',
    'Nonfinite first128-token hidden','Nonfinite repeated128-token hidden','Nonfinite PPL hidden','Nonfinite PPL loss')}


def calibration_paths():
    return v6.run_paths()|{'scripts/prepare_state_ppl_v8.py','docs/STATE_PPL_V8_PROTOCOL.md'}


def run_paths():
    return calibration_paths()|{'scripts/run_state_ppl_v8.py'}


def spec_for(name):
    return dict(candidate_id=name,candidate_name='static_layer_mix',variant='stored_scale',
                scale_mode='stored_scale',int4_clip=1.,diagnostic=None,deployable=True)


def eligible(row):
    # Integrity gates are audited before calling the ranking functions.
    return (row.get('complete') is True and 'error' not in row and 'mk' not in row
        and finite(row.get('ppl',{}).get('nll')) and finite(row.get('ppl',{}).get('ppl'))
        and row.get('cache',{}).get('total_bytes')==CACHE)


def rank_independent(rows):
    need(set(rows)==set(ARMS),'Missing or additional single-layer arm')
    need(eligible(rows['baseline']),'Calibration baseline failure must stop')
    baseline=rows['baseline']['ppl'];layer_records=[];improving=[]
    for layer in range(56):
        options=[]
        for kind in ALTERNATIVES:
            name=f'layer{layer:02d}_{kind}';valid=eligible(rows[name])
            score=rows[name]['ppl'] if valid else None
            options.append(dict(arm=name,kind=kind,valid=valid,nll=score['nll'] if valid else None,
                delta_nll=score['nll']-baseline['nll'] if valid else None,ppl=score['ppl'] if valid else None))
        # Stable sort retains protocol alternative order on an exact tie.
        finite_options=sorted((r for r in options if r['valid']),key=lambda r:r['nll'])
        winner=finite_options[0] if finite_options else None
        keep=winner is not None and winner['nll']<baseline['nll']
        layer_records.append(dict(layer=layer,alternatives=options,best_alternative=winner,improving=keep))
        if keep:improving.append(dict(layer=layer,**winner))
    improving.sort(key=lambda r:(r['delta_nll'],r['layer'],ALTERNATIVES.index(r['kind'])))
    return dict(baseline_nll=baseline['nll'],baseline_ppl=baseline['ppl'],layers=layer_records,
        ranked_improving_swaps=improving,improving_layers=len(improving),stopped=not improving,
        calibration_rows=list(CALIBRATION_ROWS),future_screen_rows=list(SCREEN_ROWS),
        rule='Per layer minimum finite complete alternative NLL; fixed alternative order ties; strictly negative delta only; rank(deltaNLL,layer,alternative order)',
        interpretation='Single-layer deltas propose combinations; they do not predict additive combined gains',
        adapter_used=False,heldout_used=False,mk_used=False)


def reconstruct_candidates(tables,ranked):
    """Independent actual-byte construction; identical tables keep first label."""
    unique={};specs={};proposals=[];first_by_bytes={}
    for label,requested in REQUESTS:
        count=len(ranked) if requested is None else min(requested,len(ranked))
        table=tables[BASELINE_KIND].clone().contiguous()
        swaps=[]
        for change in ranked[:count]:
            layer,kind=change['layer'],change['kind']
            table[layer]=tables[kind][layer]
            swaps.append(dict(layer=layer,kind=kind,calibration_delta_nll=change['delta_nll'],calibration_arm=change['arm']))
        raw=table.numpy().tobytes();digest=hashlib.sha256(raw).hexdigest()
        first=first_by_bytes.get(raw)
        if first is None:
            first=label;first_by_bytes[raw]=label;unique[label]=table
            specs[label]=dict(table_sha256=digest,swap_count=count,swaps=swaps,scale_mode='stored_scale',int4_clip=1.,cache_bytes=CACHE)
        proposals.append(dict(requested_name=label,requested_top_k='all' if requested is None else requested,
            effective_swap_count=count,retained_name=first,duplicate_of=None if first==label else first,table_sha256=digest))
    need(len({record['table_sha256'] for record in specs.values()})==len(specs),'Hash collision across unequal table bytes')
    return unique,specs,proposals


def select_independent(rows,order):
    need(order and order[0]=='baseline' and len(set(order))==len(order),'Combined candidate order differs')
    valid=[name for name in order if eligible(rows[name])]
    need('baseline' in valid,'Failed screen baseline must stop')
    winner=min(valid,key=lambda name:(rows[name]['ppl']['ppl'],name!='baseline',order.index(name)))
    return dict(selected_id=winner,selected_variant='stored_scale',baseline_id='baseline',valid=valid,
        excluded=[name for name in order if name not in valid],baseline_wins=winner=='baseline',
        adapter_used=False,heldout_used=False,mk_used=False,
        rule='Minimum complete finite exact-budget disjoint TRAIN PPL; exact ties baseline then frozen candidate export order')


def target_pass(ppl,cache):return finite(ppl) and ppl<8.25 and cache==CACHE


def self_test():
    import torch
    checks=[]
    def rows():return {arm:dict(complete=True,ppl=dict(nll=100.,ppl=math.exp(100./16376)),cache=dict(total_bytes=CACHE)) for arm in ARMS}
    values=rows();r=rank_independent(values)
    need(r['stopped'] and r['improving_layers']==0,'Equal baseline NLL must not count as improvement');checks.append('strict-negative-only')
    values['layer00_magnitude']['ppl']['nll']=math.nextafter(100.,0.)
    values['layer00_full_readout']['ppl']['nll']=math.nextafter(100.,0.)
    values['layer01_full_readout']['ppl']['nll']=math.nextafter(100.,0.)
    ranked=rank_independent(values)['ranked_improving_swaps']
    need([(x['layer'],x['kind']) for x in ranked]==[(0,'magnitude'),(1,'full_readout')],'Alternative or layer exact-tie order differs')
    checks.append('one-ulp-improvement-alternative-and-layer-ties')
    cancellation=rows();cancellation['layer00_magnitude']['ppl']['nll']=2e-15;cancellation['layer00_full_readout']['ppl']['nll']=1e-15
    need(cancellation['layer00_magnitude']['ppl']['nll']-100.==cancellation['layer00_full_readout']['ppl']['nll']-100.
         and rank_independent(cancellation)['ranked_improving_swaps'][0]['kind']=='full_readout',
         'Per-layer NLL ranking must not use rounded delta ties')
    checks.append('raw-NLL-order-survives-subtraction-cancellation')
    values['layer00_magnitude']['complete']=False
    need(rank_independent(values)['ranked_improving_swaps'][0]['kind']=='full_readout','Known excluded alternative should not remove its whole layer')
    checks.append('excluded-alternative-keeps-valid-layer-option')
    for kind in ALTERNATIVES:values[f'layer01_{kind}']['complete']=False
    need(rank_independent(values)['improving_layers']==1,'All excluded layer must contribute no swap');checks.append('all-excluded-layer')
    values['baseline']['complete']=False
    try:rank_independent(values)
    except ValueError:pass
    else:raise ValueError('Failed baseline accepted')
    checks.append('baseline-failure-fatal')
    base=torch.arange(128,dtype=torch.uint8).expand(56,8,128).clone()
    tables={name:base.clone() for name in (BASELINE_KIND,*ALTERNATIVES)}
    tables['full_readout'][1]=torch.roll(tables['full_readout'][1],1,-1)
    # First proposed swap is an actual-table no-op: dedup must inspect bytes.
    unique,specs,proposals=reconstruct_candidates(tables,ranked)
    need(tuple(unique)==('baseline','top2') and proposals[1]['duplicate_of']=='baseline'
         and all(p['retained_name']=='top2' for p in proposals[2:]),'Content dedup/capped top-k/first retained label differs')
    need(specs['top2']['swap_count']==2 and len(specs['top2']['swaps'])==2,'Ranking evidence must retain no-op swaps when larger table is kept')
    checks.append('actual-content-dedup-capped-k-noop-swap-first-label')
    unique,_,proposals=reconstruct_candidates(tables,[])
    need(tuple(unique)==('baseline',) and all(p['retained_name']=='baseline' for p in proposals),'No-improvement family must export only baseline')
    checks.append('zero-improvement-baseline-only-export')
    candidates={name:dict(complete=True,ppl=dict(nll=100.,ppl=9.),cache=dict(total_bytes=CACHE)) for name in ('baseline','top1','top2')}
    need(select_independent(candidates,list(candidates))['selected_id']=='baseline','Screen tie must prefer baseline')
    candidates['top1']['ppl']['ppl']=math.nextafter(9.,0.);candidates['top2']['ppl']['ppl']=math.nextafter(9.,0.)
    need(select_independent(candidates,list(candidates))['selected_id']=='top1','Screen nonbaseline tie must prefer export order')
    checks.append('screen-baseline-and-export-order-ties')
    candidates['top1']['cache']['total_bytes']+=1
    need(select_independent(candidates,list(candidates))['selected_id']=='top2','Extra resident byte cannot win');checks.append('screen-budget-gate')
    need(not target_pass(8.25,CACHE) and target_pass(math.nextafter(8.25,0.),CACHE)
         and not target_pass(8.24,CACHE+1),'Strict8.25/budget boundary differs');checks.append('strict-full-target-boundary')
    need(len(ARMS)==170 and len(set(ARMS))==170 and 8*2047==16376 and 32*2047==65504,'Fixed inventory/target counts differ')
    checks.append('170arms-and-token-populations')
    return dict(complete=True,passed=True,count=len(checks),checks=checks,gpu_used=False)


def audit_inputs(args,tokenizer,torch,np):
    need(sha(ROOT/'docs/STATE_PPL_V8_PROTOCOL.md')==PROTOCOL,'Frozen v8 protocol changed')
    prior=argparse.Namespace(**vars(args));prior.selected_calibration=None
    train,original,receipt,proof=v6.audit_inputs(prior,tokenizer,torch,np)
    kernel=v6.audit_kernel_receipt(args.kernel_checks)
    need(sha(args.v6_selected_calibration)==V6_SELECTED and sha(args.v6_selected_calibration.parent/'screen_comparison.json')==V6_SCREEN,
         'Frozen v6 selection changed')
    screen=v6.audit_run(args.v6_selected_calibration.parent,'screen',prior,train,original,receipt,np)
    old_binding=v6.input_binding(prior,receipt)
    selected,selected_proof=v6.audit_selected(args.v6_selected_calibration,prior,original,old_binding,screen,torch)
    need(selected['selected_id']==BASELINE_POLICY,'v8 must use unchanged v6 winner')
    binding=dict(protocol_sha256=PROTOCOL,v6_input_binding=old_binding,v6_selected_calibration_sha256=V6_SELECTED,
        v6_selected_receipt_sha256=sha(args.v6_selected_calibration.with_suffix('.json')),
        v6_selection_report_sha256=selected['selection_report_sha256'],baseline_policy=BASELINE_POLICY,
        baseline_table_sha256=tensor_sha(original['tables'][BASELINE_KIND]),scale_mode='stored_scale',int4_clip=1.,
        adapter_loaded=False,adapter_sha256=None,heldout_used=False,mk_used=False)
    return train,original,receipt,binding,dict(complete=True,upstream=proof,kernel=kernel,v6_selection=selected_proof)


def audit_arm(row,windows,common,spec,table_digest,comparison,allow_exclusion):
    need(all(row.get(k)==v for k,v in {**common,**spec}.items()) and 'mk' not in row
         and row.get('candidate_table_sha256')==table_digest,'Arm data/policy/table/provenance differs: '+str(row.get('arm')))
    need(row.get('code_hashes')==comparison['code_hashes'] and row.get('candidate_table_unchanged') is True,
         'Source inventory or CPU table changed')
    frozen(row['frozen_source']);audit_backend_policy(row['backend_policy'],row['backend_policy_check'])
    need(row['backend_policy']==comparison['backend_policy'],'Backend differs across arms')
    complete=row.get('complete') is True
    if complete:
        need('error' not in row and not row.get('fatal_failure',False) and not row.get('excluded_from_selection',False)
             and row.get('runtime_table_unchanged') is True and row.get('persistent_float_finite_checks_passed') is True,
             'Complete arm integrity/finiteness failure')
        probe=row['repeated_reset_probe'];v6.audit_cache(probe['cache'],spec,128)
        hashes=probe['cache_tensor_sha256']
        need(probe.get('tokens')==128 and probe.get('hidden_and_cache_exact') is True
             and probe['token_sha256_int64le']==tokhash(windows[0][1][:128])
             and re.fullmatch('[0-9a-f]{64}',probe['hidden_sha256']) is not None
             and set(hashes)=={f'{layer}.{name}' for layer in range(56) for name in ('lo','hi','q4','s8','s4','conv')}
             and all(isinstance(x,str) and re.fullmatch('[0-9a-f]{64}',x) for x in hashes.values()),'Reset/hidden/cache probe differs')
        v6.audit_cache(row['cache'],spec,len(windows[-1][1])-1)
        need(type(row['zero_scale_observations']) is int and row['zero_scale_observations']>=0
             and row['zero_scale_scope']=='Final-cache observations of stored s4/s8; includes true zeros and scale underflow, not unique underflow events',
             'Zero-scale accounting differs')
    else:
        need(allow_exclusion and row.get('excluded_from_selection') is True and not row.get('fatal_failure',False)
             and row.get('error_type')=='CandidateInvalid' and row.get('error') in ALLOWED_ERRORS,
             'Unrecognized/fatal failure cannot be excluded')
    metrics=v6.audit_ppl(row,windows,complete)
    if row.get('ppl',{}).get('windows'):
        need(row['ppl']['nll']==metrics['nll'] and row['ppl']['ppl']==metrics['ppl'],
             'Selection scores must exactly equal sequential raw-window arithmetic')
    return metrics


def audit_calibration(directory,args,train,original,binding,torch,np):
    path=directory/'calibration_comparison.json';comparison=read(path)
    need(comparison.get('format')==CAL_COMPARISON and comparison.get('complete') is True
         and comparison.get('stage')=='calibration' and comparison.get('protocol_sha256')==PROTOCOL
         and comparison.get('input_binding')==binding and comparison.get('adapter_loaded') is False
         and comparison.get('adapter_sha256') is None and comparison.get('heldout_used') is False
         and comparison.get('mk_used') is False and 'mk' not in comparison,'Calibration comparison provenance differs')
    hash_inventory(comparison['code_hashes'],calibration_paths(),'v8 layer calibration')
    need(tuple(comparison['report_sha256'])==ARMS,'Expected fixed170-arm execution inventory')
    windows=[(i*2048,train[i,:2048].tolist()) for i in CALIBRATION_ROWS]
    dataset=dict(split='train',file_sha256=PROSE_FILE,rows=list(CALIBRATION_ROWS),tokens_per_row=2048,windows=8,target_tokens=16376)
    common=dict(format=CAL_FORMAT,stage='calibration',protocol_sha256=PROTOCOL,input_binding=binding,dataset=dataset,
        adapter_loaded=False,adapter_sha256=None,heldout_used=False,heldout_used_for_selection=False,mk_used=False,
        selection_scope='Single-layer full-model TRAIN PPL proposes static combinations; later disjoint TRAIN screen chooses one')
    rows={};metrics={}
    for arm in ARMS:
        layer=None if arm in ('baseline','restored_baseline') else int(arm[5:7])
        kind=None if layer is None else arm[8:]
        table=original['tables'][BASELINE_KIND].clone()
        if layer is not None:table[layer]=original['tables'][kind][layer]
        arm_path=directory/('calibration_'+arm+'.json');row=read(arm_path)
        need(sha(arm_path)==comparison['report_sha256'][arm],'Raw calibration report changed: '+arm)
        identity=dict(common,arm=arm,layer_replacement=dict(layer=layer,kind=kind))
        metrics[arm]=audit_arm(row,windows,identity,spec_for(arm),tensor_sha(table),comparison,layer is not None)
        rows[arm]=row
    ranking=rank_independent(rows)
    need(comparison['ranking']==ranking,'Independent raw-NLL layer ranking differs')
    restoration=read(directory/'calibration_restoration.json')
    v6.exact_restoration(rows['baseline'],rows['restored_baseline'],restoration)
    need(comparison['restoration']==restoration,'Calibration restoration receipt differs')
    candidates,specs,proposals=reconstruct_candidates(original['tables'],ranking['ranked_improving_swaps'])
    candidate_path=directory/'candidates.pt';payload=torch.load(candidate_path,map_location='cpu',weights_only=True)
    receipt=read(candidate_path.with_suffix('.json'))
    metadata=dict(protocol_sha256=PROTOCOL,input_binding=binding,calibration_report_sha256=sha(path),
        candidate_order=list(candidates),candidate_specs=specs,proposals=proposals,calibration_stopped=ranking['stopped'],
        baseline_name='baseline',scale_mode='stored_scale',int4_clip=1.,runtime_table_bytes=57344,cache_bytes=CACHE,
        adapter_loaded=False,adapter_sha256=None,heldout_used=False,mk_used=False,
        candidate_scope='Unmeasured combinations proposed by TRAIN single-layer deltas; separate TRAIN screen required')
    need(payload.get('format')==receipt.get('format')==CANDIDATES_FORMAT
         and all(payload.get(k)==receipt.get(k)==v for k,v in metadata.items()),'Combined proposal provenance differs')
    need(receipt.get('complete') is True and receipt['file']==candidate_path.name and receipt['sha256']==sha(candidate_path)
         and receipt['bytes']==candidate_path.stat().st_size,'Combined candidate artifact identity differs')
    hash_inventory(receipt['code_hashes'],calibration_paths(),'v8 combined candidates')
    need(tuple(payload['tables'])==tuple(candidates),'Combined table export inventory/order differs')
    for name,table in candidates.items():
        actual=payload['tables'][name]
        need(actual.dtype==torch.uint8 and tuple(actual.shape)==(56,8,128) and actual.is_contiguous()
             and torch.equal(actual,table) and tensor_sha(actual)==specs[name]['table_sha256']
             and np.array_equal(np.sort(actual.numpy(),axis=-1),np.broadcast_to(np.arange(128,dtype=np.uint8),actual.shape)),
             'Exported mixed table differs from actual layer reconstruction: '+name)
    proof=dict(complete=True,arms=metrics,raw_arm_hashes=comparison['report_sha256'],comparison_sha256=sha(path),
        ranking=ranking,restoration=restoration,candidate_order=list(candidates),candidate_specs=specs,proposals=proposals,
        candidates_sha256=sha(candidate_path),candidate_receipt_sha256=sha(candidate_path.with_suffix('.json')),
        all170_arm_tables_independently_reconstructed=True,actual_content_deduplication_reconstructed=True)
    return payload,receipt,proof


def audit_v7_start_condition(args):
    """Audit predecessor outcome, without inferring execution order from mtimes."""
    outcome=read(args.v7_outcome);audit=read(args.v7_outcome_audit)
    stage=outcome.get('stage');need(stage in ('screen','full'),'Expected completed v7 screen/full outcome')
    need(outcome.get('format')=='MAMBA2_STATE_PPL_V7_COMPARISON_V1' and outcome.get('complete') is True
         and outcome.get('protocol_sha256')=='84ffdce1d9e5d5dbac2b6996072f0db09bd1bf79fd3c55dad8b8f35973cd5d14'
         and audit.get('format')=='MAMBA2_STATE_PPL_V7_INDEPENDENT_AUDIT_V1'
         and audit.get('complete') is True and audit.get('passed') is True and audit.get('cuda_initialized') is False
         and audit.get('stage')==stage and audit.get('source_sha256')==sha(ROOT/'scripts/audit_state_ppl_v7.py'),
         'Completed independent v7 outcome/audit required')
    dependencies={'audit_state_ppl_v6.py','audit_quant_first.py','audit_state_repair.py','audit_state_first_v5.py','audit_resurface_more.py'}
    need(set(audit['auditor_dependency_sha256'])==dependencies and all(
        sha(ROOT/'scripts'/name)==digest for name,digest in audit['auditor_dependency_sha256'].items()),'v7 audit dependency changed')
    proof=audit[stage]
    need(proof.get('complete') is True and proof['comparison_sha256']==sha(args.v7_outcome)
         and proof['report_sha256']==outcome['report_sha256'],'v7 outcome is not the audited artifact')
    import audit_state_ppl_v7 as prior
    hash_inventory(outcome['code_hashes'],prior.run_paths(),'v7 predecessor outcome')
    for arm,digest in outcome['report_sha256'].items():
        path=args.v7_outcome.parent/(stage+'_'+arm+'.json')
        need(sha(path)==digest,'Audited v7 raw report changed')
    if stage=='full':
        need(outcome['target_pass'] is False and proof['target_pass'] is False
             and outcome['target_checks']['ppl_strictly_below_8p25'] is False,
             'v7 already meets unadapted target; v8 measurements are not authorized by this protocol')
        criterion='Completed independently audited v7 full outcome did not reach PPL<8.25'
        predecessor_ppl=read(args.v7_outcome.parent/'full_selected.json')['ppl']['ppl']
        need(finite(predecessor_ppl) and predecessor_ppl>=8.25,'v7 miss does not match raw selected PPL')
    else:
        need(outcome['selection']['baseline_wins'] is True and proof['selection']['baseline_wins'] is True,
             'A nonbaseline v7 winner requires its full outcome before v8')
        need(args.parent_report is not None and sha(args.parent_report)==V6_BASELINE,'Screen fallback requires pinned v6 full baseline')
        predecessor_ppl=read(args.parent_report)['ppl']['ppl']
        need(finite(predecessor_ppl) and predecessor_ppl>=8.25,'v7 baseline already reaches target')
        criterion='Completed independently audited v7 screen kept unchanged v6 baseline, whose pinned full PPL is>=8.25'
    return dict(complete=True,outcome_sha256=sha(args.v7_outcome),audit_sha256=sha(args.v7_outcome_audit),
        stage=stage,criterion=criterion,predecessor_full_ppl=predecessor_ppl,
        chronology_scope='Outcome evidence is audited; calibration launch ordering is root orchestration, not inferred from timestamps.')


def screen_binding(args,candidates,receipt,calibration):
    return dict(protocol_sha256=PROTOCOL,calibration_input_binding=candidates['input_binding'],
        layer_candidates_sha256=sha(args.layer_candidates),layer_candidates_receipt_sha256=sha(args.layer_candidates.with_suffix('.json')),
        layer_calibration_report_sha256=calibration['comparison_sha256'],baseline_policy=BASELINE_POLICY,
        source_sha256=v6.SOURCE,tokenizer_sha256=TOKENIZER,train_file_sha256=PROSE_FILE)


def audit_selected(path,candidates,binding,screen,torch):
    payload=torch.load(path,map_location='cpu',weights_only=True);receipt=read(path.with_suffix('.json'))
    chosen=screen['selection'];name=chosen['selected_id']
    expected=dict(format='MAMBA2_STATE_PPL_V8_CALIBRATION_V1',protocol_sha256=PROTOCOL,input_binding=binding,
        **chosen,selected_candidate_spec=candidates['candidate_specs'][name],table_sha256=tensor_sha(candidates['tables'][name]),
        scale_mode='stored_scale',int4_clip=1.,runtime_table_bytes=57344,cache_bytes=CACHE,
        selection_report_sha256=screen['comparison_sha256'])
    need(all(payload.get(k)==receipt.get(k)==value for k,value in expected.items()),'Selected mixed-table provenance differs')
    need(receipt.get('complete') is True and receipt['file']==path.name and receipt['sha256']==sha(path)
         and receipt['bytes']==path.stat().st_size and all(receipt.get(k)==v for k,v in payload.items() if k!='permutations'),
         'Selected payload/receipt identity differs')
    hash_inventory(payload['code_hashes'],run_paths(),'v8 selected table')
    table=payload['permutations']
    need(table.dtype==torch.uint8 and tuple(table.shape)==(56,8,128) and table.is_contiguous()
         and torch.equal(table,candidates['tables'][name]) and tensor_sha(table)==expected['table_sha256'],
         'Actual selected coordinates differ from independently reconstructed candidate')
    return payload,dict(complete=True,sha256=sha(path),receipt_sha256=sha(path.with_suffix('.json')),
        selected_id=name,table_sha256=expected['table_sha256'],baseline_wins=chosen['baseline_wins'])


def audit_run(directory,stage,args,train,candidates,binding,screen=None,selected=None,validation=None,dataset=None):
    comparison_path=directory/(stage+'_comparison.json');comparison=read(comparison_path)
    common=dict(format='MAMBA2_STATE_PPL_V8_EVAL_V1',stage=stage,protocol_sha256=PROTOCOL,input_binding=binding,
        adapter_loaded=False,adapter_sha256=None,mk_used=False,heldout_used=stage=='full',heldout_used_for_selection=False,
        selected_calibration_sha256=sha(args.selected_calibration) if selected is not None else None,
        parent_report_sha256=V6_BASELINE if stage=='full' else None,
        parent_comparison_sha256=V6_COMPARISON if stage=='full' else None,s16_report_sha256=v6.S16_ARCHIVE if stage=='full' else None)
    need(comparison.get('format')=='MAMBA2_STATE_PPL_V8_COMPARISON_V1' and comparison.get('complete') is True
         and all(comparison.get(k)==value for k,value in common.items() if k in ('stage','protocol_sha256','input_binding',
             'adapter_loaded','adapter_sha256','mk_used','heldout_used','heldout_used_for_selection')) and 'mk' not in comparison,
         'v8 comparison identity/provenance differs')
    hash_inventory(comparison['code_hashes'],run_paths(),'v8 PPL evaluation')
    if stage=='screen':
        windows=[(i*2048,train[i,:2048].tolist()) for i in SCREEN_ROWS]
        expected_dataset=dict(split='train',file_sha256=PROSE_FILE,rows=list(SCREEN_ROWS),tokens_per_row=2048,windows=32,target_tokens=65504)
        arms=tuple(candidates['candidate_order'])+('restored_baseline',)
    else:
        need(screen is not None and selected is not None and not screen['selection']['baseline_wins'],'Full requires frozen nonbaseline TRAIN selection')
        need(tokhash(validation.tolist())==VALIDATION_TOKENS,'Pinned validation tokens differ')
        windows=[(i,validation[i:min(i+2049,len(validation))].tolist()) for i in range(0,len(validation)-1,2048)]
        need(len(windows)==130 and sum(len(ids)-1 for _,ids in windows)==264764,'Full population differs')
        expected_dataset=dataset;arms=('v6_baseline','selected','restored_baseline')
    common['dataset']=expected_dataset
    need(tuple(comparison['report_sha256'])==arms,'Combined screen/full arm inventory or execution order differs')
    outputs={};metrics={};digests={}
    for arm in arms:
        name=selected['selected_id'] if arm=='selected' else 'baseline' if arm in ('v6_baseline','restored_baseline') else arm
        identity=dict(common,arm=arm,layer_mix_spec=candidates['candidate_specs'][name])
        path=directory/(stage+'_'+arm+'.json');row=read(path);digests[arm]=sha(path)
        need(digests[arm]==comparison['report_sha256'][arm],'Raw combined-candidate report changed')
        metrics[arm]=audit_arm(row,windows,identity,spec_for(name),tensor_sha(candidates['tables'][name]),comparison,
            stage=='screen' and name!='baseline' and arm!='restored_baseline')
        outputs[arm]=row
    baseline='baseline' if stage=='screen' else 'v6_baseline'
    restoration=read(directory/(stage+'_restoration.json'))
    v6.exact_restoration(outputs[baseline],outputs['restored_baseline'],restoration)
    need(comparison['restoration']==restoration,'Combined baseline restoration receipt differs')
    result=dict(complete=True,arms=metrics,report_sha256=digests,restoration=restoration,comparison_sha256=sha(comparison_path),no_MK_measurements=True)
    if stage=='screen':
        selection=select_independent(outputs,candidates['candidate_order'])
        need(comparison['selection']==selection,'Independent disjoint TRAIN candidate selection differs')
        delta=v6.compare_independent(outputs['baseline'],outputs[selection['selected_id']])
        need(comparison['selected_vs_baseline']==delta,'Screen window comparison arithmetic differs')
        result.update(selection=selection,selected_vs_baseline=delta)
    else:
        need(sha(args.parent_report)==V6_BASELINE and sha(args.parent_comparison)==V6_COMPARISON
             and sha(args.s16_report)==v6.S16_ARCHIVE,'Pinned v6/S16 archived context differs')
        historical=read(args.parent_comparison);parent=read(args.parent_report);s16=read(args.s16_report)
        need(historical['report_sha256']['selected']==V6_BASELINE,'Archived v6 comparison chain differs')
        v6.audit_ppl(parent,windows);v6.audit_ppl({'ppl':s16['ppl']},windows)
        current=outputs['v6_baseline']
        need(parent['ppl']==current['ppl'] and parent['repeated_reset_probe']==current['repeated_reset_probe']
             and parent['cache']==current['cache'] and current['cache']['total_bytes']==CACHE,
             'Archived v6 complete PPL/reset/cache replay differs')
        expected_replay=dict(complete=True,per_window_nll_exact=True,ppl_windows_repeated=130,target_tokens=264764,
            reset_hidden_exact=True,reset_cache_exact=True,cache_bytes_exact=True)
        replay=read(directory/'full_parent_replay.json')
        need(replay==expected_replay and comparison['parent_replay']==replay,'Archived replay receipt differs')
        expected=dict(selected_id=selected['selected_id'],selected_variant='stored_scale',
            selected_candidate_spec=selected['selected_candidate_spec'],selected_calibration_sha256=sha(args.selected_calibration),
            selection_report_sha256=screen['comparison_sha256'],parent_report_sha256=V6_BASELINE,
            parent_comparison_sha256=V6_COMPARISON,s16_report_sha256=v6.S16_ARCHIVE,resurface_trained=False,mk_evaluated=False)
        need(all(comparison.get(k)==value for k,value in expected.items()),'Full frozen-candidate/context fields differ')
        delta=v6.compare_independent(current,outputs['selected']);gap=v6.compare_independent(s16,outputs['selected'])
        checks=dict(ppl_strictly_below_8p25=outputs['selected']['ppl']['ppl']<8.25,
            cache_same_budget=all(row['cache']['total_bytes']==CACHE for row in outputs.values()),all_integrity_checks_passed=True)
        need(comparison['comparison']==delta and comparison['original_s16_comparison']==gap
             and comparison['target_checks']==checks and comparison['target_pass']==all(checks.values()),'Full target arithmetic differs')
        result.update(comparison=delta,original_s16_comparison=gap,target_checks=checks,target_pass=all(checks.values()),parent_replay=replay)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true');parser.add_argument('--stage',choices=('inputs','calibration','screen','full'))
    for name in ('source-dir','calibration','repair-calibration','parent-training-report','prose-tokens',
        'codec-checks','candidates','v5-calibration','v6-selected-calibration','kernel-checks',
        'calibration-dir','layer-candidates','eval-dir','screening-report','selected-calibration',
        'parent-report','parent-comparison','s16-report','v7-outcome','v7-outcome-audit','output'):
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test(),indent=2));return
    for key in ('stage','source_dir','calibration','repair_calibration','parent_training_report','prose_tokens',
                'codec_checks','candidates','v5_calibration','v6_selected_calibration','kernel_checks','output'):
        if getattr(args,key) is None:parser.error('--'+key.replace('_','-')+' is required')
    if args.stage!='inputs':
        for key in ('calibration_dir','v7_outcome','v7_outcome_audit'):
            if getattr(args,key) is None:parser.error('--'+key.replace('_','-')+' is required')
    if args.stage in ('screen','full'):
        for key in ('layer_candidates','eval_dir'):
            if getattr(args,key) is None:parser.error('--'+key.replace('_','-')+' is required')
    if args.stage=='full':
        for key in ('screening_report','selected_calibration','parent_report','parent_comparison','s16_report'):
            if getattr(args,key) is None:parser.error('--'+key.replace('_','-')+' is required')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Set CUDA_VISIBLE_DEVICES= for CPU-only audit')
    need(not args.output.exists(),'Use a fresh output path; preserve earlier evidence')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1'
    dependencies=('audit_state_ppl_v6.py','audit_state_ppl_v7.py','audit_quant_first.py','audit_state_repair.py','audit_state_first_v5.py','audit_resurface_more.py')
    result=dict(format='MAMBA2_STATE_PPL_V8_INDEPENDENT_AUDIT_V1',complete=False,passed=False,stage=args.stage,
        protocol_sha256=PROTOCOL,source_sha256=sha(__file__),auditor_dependency_sha256={name:sha(ROOT/'scripts'/name) for name in dependencies},
        limitations=['CPU reconstructs raw measurements and arithmetic, not GPU model logits.',
            'Frozen identity/version/gradient guards do not replace post-run source-weight byte hashes.',
            'Historical benchmark exposure remains; MK and Resurface are absent.',
            'The v7 outcome start condition is checked; execution chronology remains orchestration evidence.'])
    try:
        import numpy as np
        import torch
        from mamba2_recall import runtime
        torch.set_num_threads(4);need(not torch.cuda.is_initialized(),'CUDA initialized before CPU-only audit')
        result['self_tests']=self_test();tokenizer=runtime.SentencePieceTokenizer(args.source_dir)
        need(tokenizer.sha256==TOKENIZER,'Tokenizer differs')
        train,original,receipt,binding,proof=audit_inputs(args,tokenizer,torch,np);result['inputs']=proof
        if args.v7_outcome is not None:
            need(args.v7_outcome_audit is not None,'v7 outcome must have independent audit')
            result['v7_start_condition']=audit_v7_start_condition(args)
        if args.stage!='inputs':
            mixed,mixed_receipt,calibration=audit_calibration(args.calibration_dir,args,train,original,binding,torch,np)
            result['calibration']=calibration
            if args.stage in ('screen','full'):
                need(args.layer_candidates.resolve()==(args.calibration_dir/'candidates.pt').resolve(),'Layer candidates must accompany audited calibration')
                mixed_binding=screen_binding(args,mixed,mixed_receipt,calibration)
                directory=args.eval_dir if args.stage=='screen' else args.screening_report.parent
                if args.stage=='full':need(args.screening_report.name=='screen_comparison.json','Frozen screen filename differs')
                screen=audit_run(directory,'screen',args,train,mixed,mixed_binding);result['screen']=screen
                path=directory/'selected_calibration.pt'
                if args.selected_calibration is None:args.selected_calibration=path
                need(args.selected_calibration.resolve()==path.resolve(),'Selection must accompany audited screen')
                selected,selected_proof=audit_selected(path,mixed,mixed_binding,screen,torch);result['selected_calibration']=selected_proof
                if args.stage=='full':
                    from mamba2_recall.calibration import load_wikitext_tokens
                    validation,dataset=load_wikitext_tokens(tokenizer,'validation')
                    result['full']=audit_run(args.eval_dir,'full',args,train,mixed,mixed_binding,screen,selected,validation,dataset)
        need(not torch.cuda.is_initialized(),'CPU-only audit initialized CUDA')
        result.update(complete=True,passed=True,cuda_initialized=False)
    except BaseException as error:result['error']=repr(error);raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
