#!/usr/bin/env python3
"""Independent CPU audit of v9 group interventions and frozen PPL confirmation.

Does not import either v9 preparation or evaluation implementation. Rebuilds
actual-byte intervention dispositions, raw-NLL ranking, combinations, selection
and strict target gates. Reuses frozen independent auditors for old evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import audit_state_ppl_v6 as v6
import audit_state_ppl_v8 as v8
from audit_quant_first import sha,need,finite,tokhash,read,TOKENIZER,PROSE_FILE,VALIDATION_TOKENS
from audit_state_repair import hash_inventory,tensor_sha

PROTOCOL='9e01c03ee6870a8ecbcd9a0ba9157b1651d2830ea65d2951664ebcb81b4a09b3'
V8_CALIBRATION='e7fa9ce0f245d3843f447be604bc195186645cb956f9d229cf34b1d05a9bdc49'
V8_SELECTED='9b7c04814085abbb67d7a10e0b70d1a6e9b34e97299fc0dc26e4373ea8f8ea39'
V8_SELECTED_RECEIPT='9f9b6764856274222e227683adf27d155a2b4727faa82b3c0b43b558f7fca596'
V8_SCREEN='ac3604395393d0dccda8e299021ce686653a02babbeaba33bfe58d6d4ba7b16c'
V8_SCREEN_AUDIT='90fbed838e0c97f5543b6687b20d6a3740fc4fa0ebade2920d73d08df40526e2'
V8_TABLE='281f7c9fbfdabd6bfa04964ed44b19761708a08447989436b978fae37dc27298'
V8_AUDITOR='7970c74c31713129b95f285ed68cd28dfeff50b4c5a339b462a3f993bd7cc63b'
KINDS=('magnitude','full_readout','preserve_int8','preserve_retained80')
LAYERS=(0,3,6,1,4,2,7,10)
CALIBRATION_ROWS=tuple(range(144,152))
SCREEN_ROWS=tuple(range(152,184))
REQUESTS=(('baseline',0),('top1',1),('top2',2),('top4',4),('top8',8),('top16',16),('top32',32),('allnegative',None))
CACHE=28499968
CAL_FORMAT='MAMBA2_STATE_PPL_GROUP_EVAL_V1'
CAL_COMPARISON='MAMBA2_STATE_PPL_GROUP_CALIBRATION_V1'
CANDIDATES_FORMAT='MAMBA2_STATE_PPL_GROUP_CANDIDATES_V1'
V8_DEPENDENCIES={'audit_state_ppl_v6.py','audit_state_ppl_v7.py','audit_quant_first.py',
    'audit_state_repair.py','audit_state_first_v5.py','audit_resurface_more.py'}


def calibration_paths():
    return v8.run_paths()|{'scripts/prepare_state_ppl_v9.py','scripts/audit_state_ppl_v9.py','docs/STATE_PPL_V9_PROTOCOL.md'}


def run_paths():return calibration_paths()|{'scripts/run_state_ppl_v9.py'}


def spec_for(name):
    return dict(candidate_id=name,candidate_name='static_group_mix',variant='stored_scale',
        scale_mode='stored_scale',int4_clip=1.,diagnostic=None,deployable=True)


def group_inventory_independent(tables,parent):
    """Compare complete group bytes; dedup is local to one intervention site."""
    records=[]
    for layer in LAYERS:
        for group in range(8):
            parent_bytes=parent[layer,group].numpy().tobytes();prior=[];alternatives=[]
            for kind in KINDS:
                raw=tables[kind][layer,group].numpy().tobytes()
                same=next((label for label,content in prior if raw==content),None)
                if raw==parent_bytes:disposition,duplicate,arm='parent_equal',None,None
                elif same is not None:disposition,duplicate,arm='duplicate_of',same,None
                else:
                    disposition,duplicate,arm='retained',None,f'layer{layer:02d}_group{group}_{kind}'
                    prior.append((kind,raw))
                alternatives.append(dict(kind=kind,group_sha256=hashlib.sha256(raw).hexdigest(),
                    disposition=disposition,duplicate_of=duplicate,arm=arm))
            need(any(row['disposition']=='parent_equal' for row in alternatives),'Parent group absent from original tables')
            records.append(dict(layer=layer,group=group,parent_group_sha256=hashlib.sha256(parent_bytes).hexdigest(),alternatives=alternatives))
    return records


def interventions(inventory):
    return [('baseline',None,None,None)]+[(alt['arm'],entry['layer'],entry['group'],alt['kind'])
        for entry in inventory for alt in entry['alternatives'] if alt['disposition']=='retained']+[('restored_baseline',None,None,None)]


def eligible(row):return v8.eligible(row)


def rank_independent(rows,inventory):
    need(set(rows)=={entry[0] for entry in interventions(inventory)},'Group arm inventory differs')
    need(eligible(rows['baseline']),'Failed calibration baseline must stop')
    nll0=rows['baseline']['ppl']['nll'];groups=[];improving=[]
    for entry in inventory:
        choices=[]
        for alternative in entry['alternatives']:
            if alternative['disposition']!='retained':continue
            arm=alternative['arm'];valid=eligible(rows[arm]);score=rows[arm]['ppl'] if valid else None
            choices.append(dict(arm=arm,kind=alternative['kind'],valid=valid,nll=score['nll'] if valid else None,
                delta_nll=score['nll']-nll0 if valid else None,ppl=score['ppl'] if valid else None))
        # Sort raw NLL, because subtraction can collapse distinct scores to a tie.
        valid=sorted((r for r in choices if r['valid']),key=lambda r:(r['nll'],KINDS.index(r['kind'])))
        best=valid[0] if valid else None;keep=best is not None and best['nll']<nll0
        groups.append(dict(layer=entry['layer'],group=entry['group'],alternatives=choices,best_alternative=best,improving=keep))
        if keep:improving.append(dict(layer=entry['layer'],group=entry['group'],**best))
    improving.sort(key=lambda row:(row['delta_nll'],row['layer'],row['group'],KINDS.index(row['kind'])))
    return dict(baseline_nll=nll0,baseline_ppl=rows['baseline']['ppl']['ppl'],groups=groups,
        ranked_improving_swaps=improving,improving_groups=len(improving),stopped=not improving,
        calibration_rows=list(CALIBRATION_ROWS),future_screen_rows=list(SCREEN_ROWS),
        rule='Per-group minimum raw NLL; fixed v5 table order ties; strict improvement; global(deltaNLL,layer,group,table order)',
        interpretation='Individual group deltas propose combinations; effects are not assumed additive',
        adapter_used=False,heldout_used=False,mk_used=False)


def reconstruct_candidates(tables,parent,ranked):
    need(len({(r['layer'],r['group']) for r in ranked})==len(ranked),'More than one selected change per group')
    unique={};specs={};proposals=[];seen={}
    for name,requested in REQUESTS:
        count=len(ranked) if requested is None else min(requested,len(ranked))
        table=parent.clone().contiguous();swaps=[]
        for change in ranked[:count]:
            layer,group,kind=change['layer'],change['group'],change['kind']
            need(layer in LAYERS and 0<=group<8 and kind in KINDS,'Change outside frozen group pool')
            table[layer,group]=tables[kind][layer,group]
            swaps.append(dict(layer=layer,group=group,kind=kind,calibration_delta_nll=change['delta_nll'],calibration_arm=change['arm']))
        raw=table.numpy().tobytes();digest=hashlib.sha256(raw).hexdigest();first=seen.get(raw)
        if first is None:
            first=name;seen[raw]=name;unique[name]=table
            specs[name]=dict(table_sha256=digest,swap_count=count,swaps=swaps,scale_mode='stored_scale',int4_clip=1.,cache_bytes=CACHE)
        proposals.append(dict(requested_name=name,requested_top_k='all' if requested is None else requested,
            effective_swap_count=count,retained_name=first,duplicate_of=None if first==name else first,table_sha256=digest))
    need(len({s['table_sha256'] for s in specs.values()})==len(specs),'Table hash collision')
    return unique,specs,proposals


def select_independent(rows,order):
    need(order and order[0]=='baseline' and len(set(order))==len(order),'Candidate order differs')
    valid=[name for name in order if eligible(rows[name])];need('baseline' in valid,'Failed screen baseline must stop')
    winner=min(valid,key=lambda name:(rows[name]['ppl']['ppl'],name!='baseline',order.index(name)))
    return dict(selected_id=winner,selected_variant='stored_scale',baseline_id='baseline',valid=valid,
        excluded=[name for name in order if name not in valid],baseline_wins=winner=='baseline',
        adapter_used=False,heldout_used=False,mk_used=False,
        rule='Minimum complete finite exact-budget disjoint TRAIN PPL; exact ties baseline then frozen candidate export order')


def target_pass(ppl,cache):return finite(ppl) and ppl<8.25 and cache==CACHE


def assert_finite_miss(ppl,checks,target):
    need(finite(ppl) and ppl>=8.25 and target is False and checks==dict(ppl_strictly_below_8p25=False,
        cache_same_budget=True,all_integrity_checks_passed=True),'Only a finite audited v8 target miss permits v9')


def self_test():
    import torch
    checks=[]
    def rejects(call,label):
        try:call()
        except ValueError:checks.append(label)
        else:raise ValueError(label+' was accepted')
    parent=torch.arange(128,dtype=torch.uint8).expand(56,8,128).clone()
    tables={kind:torch.roll(parent,k,-1) for k,kind in enumerate(KINDS)}
    inv=group_inventory_independent(tables,parent)
    need(len(inv)==64 and len(interventions(inv))==194,'Maximum inventory differs');checks.append('64groups-192interventions-194arms')
    original_inv=inv
    tables['preserve_int8'][0,0]=tables['full_readout'][0,0]
    tables['preserve_retained80'][0,0]=parent[0,0]
    inv=group_inventory_independent(tables,parent)
    row=inv[0]['alternatives']
    need([r['disposition'] for r in row]==['parent_equal','retained','duplicate_of','parent_equal']
         and row[2]['duplicate_of']=='full_readout' and row[2]['arm'] is None,'Per-group equality/first-label dedup differs')
    need(inv[1]['alternatives'][1]['disposition']=='retained','Identical bytes at another group must remain independent')
    checks.extend(['parent-equal-and-first-alternative-byte-dedup','no-dedup-across-distinct-groups'])
    tables={kind:torch.roll(parent,k,-1) for k,kind in enumerate(KINDS)};inv=original_inv
    def rows():return {a:dict(complete=True,ppl=dict(nll=100.,ppl=9.),cache=dict(total_bytes=CACHE)) for a,_,_,_ in interventions(inv)}
    values=rows();need(rank_independent(values,inv)['stopped'],'Equal NLL counted as improvement');checks.append('strict-negative-only')
    for arm in ('layer03_group0_full_readout','layer00_group1_full_readout','layer00_group0_full_readout','layer00_group0_preserve_int8'):
        values[arm]['ppl']['nll']=math.nextafter(100.,0.)
    ranking=rank_independent(values,inv);ranked=ranking['ranked_improving_swaps']
    need([(r['layer'],r['group'],r['kind']) for r in ranked]==[(0,0,'full_readout'),(0,1,'full_readout'),(3,0,'full_readout')],
         'Raw score/kind/numeric layer/group exact ties differ');checks.append('raw-score-and-layer-group-kind-ties')
    cancellation=rows();cancellation['layer00_group0_full_readout']['ppl']['nll']=2e-15
    cancellation['layer00_group0_preserve_int8']['ppl']['nll']=1e-15
    need(2e-15-100.==1e-15-100. and rank_independent(cancellation,inv)['ranked_improving_swaps'][0]['kind']=='preserve_int8',
         'Per-group raw NLL must precede delta subtraction');checks.append('raw-NLL-order-through-subtraction-cancellation')
    values['layer00_group0_full_readout']['complete']=False
    need(rank_independent(values,inv)['groups'][0]['best_alternative']['kind']=='preserve_int8','Failed alternative removed whole group')
    for kind in KINDS[1:]:values['layer00_group1_'+kind]['complete']=False
    need(rank_independent(values,inv)['improving_groups']==2,'All-failed group retained');checks.append('known-excluded-alternatives-and-all-failed-group')
    values['baseline']['complete']=False;rejects(lambda:rank_independent(values,inv),'calibration-baseline-failure-fatal')
    unique,specs,proposals=reconstruct_candidates(tables,parent,ranked)
    need(tuple(unique)==('baseline','top1','top2','top4') and all(p['retained_name']=='top4' for p in proposals[3:])
         and torch.equal(unique['top4'][4],parent[4]) and specs['top4']['swap_count']==3,'Capping/dedup/single-group construction differs')
    checks.append('topk-cap-actual-table-dedup-untouched-parent-groups')
    unique,_,proposals=reconstruct_candidates(tables,parent,[])
    need(tuple(unique)==('baseline',) and all(r['retained_name']=='baseline' for r in proposals),'Empty ranking export differs');checks.append('baseline-only-stop')
    rejects(lambda:reconstruct_candidates(tables,parent,[ranked[0],ranked[0]]),'duplicate-selected-group-rejected')
    candidates={name:dict(complete=True,ppl=dict(nll=100.,ppl=9.),cache=dict(total_bytes=CACHE)) for name in ('baseline','top1','top2')}
    need(select_independent(candidates,list(candidates))['selected_id']=='baseline','Baseline tie failed')
    for name in ('top1','top2'):candidates[name]['ppl']['ppl']=math.nextafter(9.,0.)
    need(select_independent(candidates,list(candidates))['selected_id']=='top1','Export order tie failed');checks.append('screen-baseline-and-export-order-ties')
    candidates['top1']['cache']['total_bytes']+=1
    need(select_independent(candidates,list(candidates))['selected_id']=='top2','Extra cache byte selected');checks.append('screen-actual-budget-gate')
    candidates['baseline']['complete']=False;rejects(lambda:select_independent(candidates,list(candidates)),'screen-baseline-failure-fatal')
    need(not target_pass(8.25,CACHE) and target_pass(math.nextafter(8.25,0.),CACHE)
         and not target_pass(8.24,CACHE+1),'Strict target differs');checks.append('strict-8p25-and-same-cache')
    good=dict(ppl_strictly_below_8p25=False,cache_same_budget=True,all_integrity_checks_passed=True)
    assert_finite_miss(8.25,good,False)
    rejects(lambda:assert_finite_miss(8.24,good,False),'v8-target-success-prohibits-v9')
    rejects(lambda:assert_finite_miss(float('nan'),good,False),'v8-nonfinite-is-not-target-miss')
    rejects(lambda:assert_finite_miss(8.3,{**good,'all_integrity_checks_passed':False},False),'v8-integrity-failure-is-not-target-miss')
    rejects(lambda:assert_finite_miss(8.3,{**good,'cache_same_budget':False},False),'v8-cache-failure-is-not-target-miss')
    need(not set(CALIBRATION_ROWS)&set(SCREEN_ROWS) and 8*2047==16376 and 32*2047==65504,'Token populations differ')
    checks.append('disjoint-token-populations')
    return dict(complete=True,passed=True,count=len(checks),checks=checks,gpu_used=False)


def audit_v8_receipt(path,stage):
    receipt=read(path)
    need(sha(ROOT/'scripts/audit_state_ppl_v8.py')==V8_AUDITOR
         and receipt.get('format')=='MAMBA2_STATE_PPL_V8_INDEPENDENT_AUDIT_V1'
         and receipt.get('complete') is True and receipt.get('passed') is True and receipt.get('cuda_initialized') is False
         and receipt.get('stage')==stage and receipt.get('protocol_sha256')==v8.PROTOCOL
         and receipt.get('source_sha256')==V8_AUDITOR,'Completed frozen independent v8 audit required')
    need(set(receipt['auditor_dependency_sha256'])==V8_DEPENDENCIES and all(
        sha(ROOT/'scripts'/name)==digest for name,digest in receipt['auditor_dependency_sha256'].items()),'v8 audit dependency changed')
    return receipt


def audit_v8_miss(args,parent,candidates,binding,screen,selected_proof,tokenizer):
    """Recompute current raw full arithmetic/guards; bind earlier independent replay proof."""
    from mamba2_recall.calibration import load_wikitext_tokens
    validation,dataset=load_wikitext_tokens(tokenizer,'validation')
    need(tokhash(validation.tolist())==VALIDATION_TOKENS,'v8 archived validation stream differs')
    windows=[(i,validation[i:min(i+2049,len(validation))].tolist()) for i in range(0,len(validation)-1,2048)]
    need(len(windows)==130 and sum(len(ids)-1 for _,ids in windows)==264764,'v8 full token population differs')
    directory=args.v8_full_dir;path=directory/'full_comparison.json';comparison=read(path)
    audited=audit_v8_receipt(args.v8_full_audit,'full');proof=audited['full']
    need(audited['screen']==screen and audited['selected_calibration']==selected_proof
         and proof.get('complete') is True and proof['comparison_sha256']==sha(path)
         and proof['report_sha256']==comparison['report_sha256'],'v8 full audit does not bind exact TRAIN parent/outcome')
    expected=dict(format='MAMBA2_STATE_PPL_V8_COMPARISON_V1',complete=True,stage='full',protocol_sha256=v8.PROTOCOL,
        input_binding=binding,adapter_loaded=False,adapter_sha256=None,mk_used=False,heldout_used=True,heldout_used_for_selection=False,
        selected_id='top8',selected_variant='stored_scale',selected_candidate_spec=parent['selected_candidate_spec'],
        selected_calibration_sha256=V8_SELECTED,selection_report_sha256=V8_SCREEN,
        parent_report_sha256=v8.V6_BASELINE,parent_comparison_sha256=v8.V6_COMPARISON,
        s16_report_sha256=v6.S16_ARCHIVE,resurface_trained=False,mk_evaluated=False)
    need(all(comparison.get(k)==value for k,value in expected.items()) and 'mk' not in comparison,'v8 full provenance differs')
    hash_inventory(comparison['code_hashes'],v8.run_paths(),'v8 completed full')
    arms=('v6_baseline','selected','restored_baseline');need(tuple(comparison['report_sha256'])==arms,'v8 full arm order differs')
    rows={};metrics={}
    common=dict(format='MAMBA2_STATE_PPL_V8_EVAL_V1',stage='full',protocol_sha256=v8.PROTOCOL,input_binding=binding,
        adapter_loaded=False,adapter_sha256=None,mk_used=False,heldout_used=True,heldout_used_for_selection=False,
        selected_calibration_sha256=V8_SELECTED,parent_report_sha256=v8.V6_BASELINE,
        parent_comparison_sha256=v8.V6_COMPARISON,s16_report_sha256=v6.S16_ARCHIVE,dataset=dataset)
    for arm in arms:
        raw_path=directory/('full_'+arm+'.json');row=read(raw_path);name='top8' if arm=='selected' else 'baseline'
        need(sha(raw_path)==comparison['report_sha256'][arm],'Audited v8 full raw arm changed')
        metrics[arm]=v8.audit_arm(row,windows,dict(common,arm=arm,layer_mix_spec=candidates['candidate_specs'][name]),
            v8.spec_for(name),tensor_sha(candidates['tables'][name]),comparison,False);rows[arm]=row
    need(metrics==proof['arms'],'Independent v8 raw arithmetic differs from audited full evidence')
    restoration=read(directory/'full_restoration.json')
    v6.exact_restoration(rows['v6_baseline'],rows['restored_baseline'],restoration)
    need(restoration==comparison['restoration']==proof['restoration'],'v8 restored baseline differs')
    replay=dict(complete=True,per_window_nll_exact=True,ppl_windows_repeated=130,target_tokens=264764,
        reset_hidden_exact=True,reset_cache_exact=True,cache_bytes_exact=True)
    need(read(directory/'full_parent_replay.json')==comparison['parent_replay']==proof['parent_replay']==replay,
         'Previously independently verified v6 archive replay receipt differs')
    delta=v6.compare_independent(rows['v6_baseline'],rows['selected'])
    need(comparison['comparison']==proof['comparison']==delta,'v8 full window comparison arithmetic differs')
    checks=dict(ppl_strictly_below_8p25=rows['selected']['ppl']['ppl']<8.25,
        cache_same_budget=all(row['cache']['total_bytes']==CACHE for row in rows.values()),all_integrity_checks_passed=True)
    need(comparison['target_checks']==proof['target_checks']==checks
         and comparison['target_pass']==proof['target_pass']==all(checks.values()),'v8 full target differs')
    assert_finite_miss(rows['selected']['ppl']['ppl'],checks,comparison['target_pass'])
    start=dict(complete=True,comparison_sha256=sha(path),audit_sha256=sha(args.v8_full_audit),
        parent_report_sha256=sha(directory/'full_selected.json'),selected_calibration_sha256=V8_SELECTED,
        parent_ppl=rows['selected']['ppl']['ppl'],parent_selection='Frozen v8 TRAIN top8, irrespective of validation comparison',
        criterion='Independent full audit passed; finite v8 PPL >=8.25; all integrity guards passed')
    return rows['selected'],start,dict(complete=True,raw_metrics=metrics,target_checks=checks,target_pass=False,
        start_condition=start,restoration=restoration,parent_replay=replay,
        scope='Current v8 raw token/NLL/cache/guards independently recomputed; earlier archive comparison bound to unchanged independent full audit')


def audit_inputs(args,tokenizer,torch,np):
    need(sha(ROOT/'docs/STATE_PPL_V9_PROTOCOL.md')==PROTOCOL,'Frozen v9 protocol changed')
    prior=argparse.Namespace(**vars(args));prior.selected_calibration=None
    train,original,original_receipt,calibration_binding,upstream=v8.audit_inputs(prior,tokenizer,torch,np)
    need(sha(args.layer_candidates.parent/'calibration_comparison.json')==V8_CALIBRATION,'Pinned v8 calibration differs')
    mixed,mixed_receipt,calibration=v8.audit_calibration(args.layer_candidates.parent,prior,train,original,calibration_binding,torch,np)
    need(args.layer_candidates.name=='candidates.pt' and sha(args.layer_candidates)==calibration['candidates_sha256'],'v8 layer payload differs')
    need(tuple(row['layer'] for row in calibration['ranking']['ranked_improving_swaps'][:8])==LAYERS,'Frozen eligible layer order differs')
    old_binding=v8.screen_binding(prior,mixed,mixed_receipt,calibration)
    need(sha(args.v8_selected_calibration)==V8_SELECTED and sha(args.v8_selected_calibration.with_suffix('.json'))==V8_SELECTED_RECEIPT
         and sha(args.v8_selected_calibration.parent/'screen_comparison.json')==V8_SCREEN
         and sha(args.v8_screen_audit)==V8_SCREEN_AUDIT,'Pinned v8 TRAIN selected artifacts differ')
    screen=v8.audit_run(args.v8_selected_calibration.parent,'screen',prior,train,mixed,old_binding)
    parent,parent_proof=v8.audit_selected(args.v8_selected_calibration,mixed,old_binding,screen,torch)
    need(parent['selected_id']=='top8' and not parent['baseline_wins'] and tensor_sha(parent['permutations'])==V8_TABLE,
         'Parent must remain exact v8 TRAIN top8 regardless validation')
    screen_audit=audit_v8_receipt(args.v8_screen_audit,'screen')
    need(screen_audit['screen']==screen and screen_audit['calibration']==calibration
         and screen_audit['selected_calibration']==parent_proof,'v8 screen audit differs from independently reconstructed selection')
    archive,start,start_proof=audit_v8_miss(args,parent,mixed,old_binding,screen,parent_proof,tokenizer)
    inventory=group_inventory_independent(original['tables'],parent['permutations']);arms=interventions(inventory)
    need(len(inventory)==64 and 2<=len(arms)<=194,'Group inventory exceeds protocol')
    binding=dict(protocol_sha256=PROTOCOL,v8_input_binding=old_binding,v8_selected_calibration_sha256=V8_SELECTED,
        v8_selected_receipt_sha256=V8_SELECTED_RECEIPT,v8_screen_audit_sha256=V8_SCREEN_AUDIT,
        v8_selection_report_sha256=V8_SCREEN,v8_calibration_report_sha256=V8_CALIBRATION,parent_table_sha256=V8_TABLE,
        parent_selected_id='top8',start_condition=start,source_sha256=v6.SOURCE,tokenizer_sha256=TOKENIZER,train_file_sha256=PROSE_FILE)
    source_hashes={name:sha(ROOT/name) for name in sorted(run_paths())}
    hash_inventory(source_hashes,run_paths(),'v9 prospective sources')
    proof=dict(complete=True,upstream=upstream,v8_calibration=calibration,v8_screen=screen,v8_selected_calibration=parent_proof,
        v8_screen_audit_sha256=sha(args.v8_screen_audit),v8_start_condition=start_proof,input_binding=binding,
        intervention_inventory=inventory,arms=[row[0] for row in arms],intervention_count=len(arms)-2,
        disposition_counts={status:sum(alt['disposition']==status for item in inventory for alt in item['alternatives'])
            for status in ('parent_equal','duplicate_of','retained')},code_hashes=source_hashes,
        actual_group_bytes_independently_compared=True,runtime_table_bytes=57344,cache_bytes=CACHE,
        no_model_execution=True,no_MK_measurements=True)
    return train,original,parent,binding,inventory,archive,proof


def audit_calibration(directory,args,train,original,parent,binding,inventory,torch,np):
    path=directory/'calibration_comparison.json';comparison=read(path);arms=interventions(inventory)
    expected=dict(format=CAL_COMPARISON,complete=True,stage='calibration',protocol_sha256=PROTOCOL,input_binding=binding,
        intervention_inventory=inventory,adapter_loaded=False,adapter_sha256=None,heldout_used=False,mk_used=False)
    need(all(comparison.get(k)==value for k,value in expected.items()) and 'mk' not in comparison,'v9 calibration provenance differs')
    hash_inventory(comparison['code_hashes'],calibration_paths(),'v9 group calibration')
    need(tuple(comparison['report_sha256'])==tuple(row[0] for row in arms),'Exact group intervention population/order differs')
    inventory_path=directory/'intervention_inventory.json'
    inventory_receipt=dict(complete=True,input_binding=binding,protocol_sha256=PROTOCOL,intervention_inventory=inventory,
        arms=[row[0] for row in arms],code_hashes=comparison['code_hashes'])
    need(read(inventory_path)==inventory_receipt and comparison['intervention_inventory_sha256']==sha(inventory_path),
         'Premodel actual-byte disposition receipt differs')
    windows=[(i*2048,train[i,:2048].tolist()) for i in CALIBRATION_ROWS]
    dataset=dict(split='train',file_sha256=PROSE_FILE,rows=list(CALIBRATION_ROWS),tokens_per_row=2048,windows=8,target_tokens=16376)
    common=dict(format=CAL_FORMAT,stage='calibration',protocol_sha256=PROTOCOL,input_binding=binding,dataset=dataset,
        adapter_loaded=False,adapter_sha256=None,heldout_used=False,heldout_used_for_selection=False,mk_used=False)
    rows={};metrics={}
    for arm,layer,group,kind in arms:
        table=parent['permutations'].clone().contiguous()
        if layer is not None:table[layer,group]=original['tables'][kind][layer,group]
        raw_path=directory/('calibration_'+arm+'.json');row=read(raw_path)
        need(sha(raw_path)==comparison['report_sha256'][arm],'Raw group report changed: '+arm)
        identity=dict(common,arm=arm,group_replacement=dict(layer=layer,group=group,kind=kind))
        metrics[arm]=v8.audit_arm(row,windows,identity,spec_for(arm),tensor_sha(table),comparison,layer is not None)
        rows[arm]=row
    ranking=rank_independent(rows,inventory);need(comparison['ranking']==ranking,'Independent group raw-NLL ranking differs')
    restoration=read(directory/'calibration_restoration.json')
    v6.exact_restoration(rows['baseline'],rows['restored_baseline'],restoration)
    need(comparison['restoration']==restoration,'Group calibration restoration receipt differs')
    candidates,specs,proposals=reconstruct_candidates(original['tables'],parent['permutations'],ranking['ranked_improving_swaps'])
    candidate_path=directory/'candidates.pt';payload=torch.load(candidate_path,map_location='cpu',weights_only=True)
    receipt=read(candidate_path.with_suffix('.json'))
    metadata=dict(protocol_sha256=PROTOCOL,input_binding=binding,intervention_inventory=inventory,calibration_report_sha256=sha(path),
        candidate_order=list(candidates),candidate_specs=specs,proposals=proposals,calibration_stopped=ranking['stopped'],
        baseline_name='baseline',scale_mode='stored_scale',int4_clip=1.,runtime_table_bytes=57344,cache_bytes=CACHE,
        adapter_loaded=False,adapter_sha256=None,heldout_used=False,mk_used=False)
    need(payload.get('format')==receipt.get('format')==CANDIDATES_FORMAT
         and all(payload.get(k)==receipt.get(k)==value for k,value in metadata.items()),'Group proposal provenance differs')
    need(receipt.get('complete') is True and receipt['file']==candidate_path.name and receipt['sha256']==sha(candidate_path)
         and receipt['bytes']==candidate_path.stat().st_size,'Group candidate artifact identity differs')
    hash_inventory(receipt['code_hashes'],calibration_paths(),'v9 combined candidates')
    need(tuple(payload['tables'])==tuple(candidates),'Candidate actual-table inventory/order differs')
    for name,table in candidates.items():
        actual=payload['tables'][name]
        need(actual.dtype==torch.uint8 and tuple(actual.shape)==(56,8,128) and actual.is_contiguous()
             and torch.equal(actual,table) and tensor_sha(actual)==specs[name]['table_sha256']
             and np.array_equal(np.sort(actual.numpy(),axis=-1),np.broadcast_to(np.arange(128,dtype=np.uint8),actual.shape)),
             'Exported group mixture differs from reconstructed 57344 bytes: '+name)
    proof=dict(complete=True,arms=metrics,raw_arm_hashes=comparison['report_sha256'],comparison_sha256=sha(path),
        intervention_inventory_sha256=sha(inventory_path),intervention_inventory=inventory,ranking=ranking,restoration=restoration,
        candidate_order=list(candidates),candidate_specs=specs,proposals=proposals,candidates_sha256=sha(candidate_path),
        candidate_receipt_sha256=sha(candidate_path.with_suffix('.json')),all_arm_tables_independently_reconstructed=True,
        actual_content_deduplication_reconstructed=True)
    return payload,receipt,proof


def screen_binding(args,candidates,receipt,calibration):
    return dict(protocol_sha256=PROTOCOL,calibration_input_binding=candidates['input_binding'],
        group_candidates_sha256=sha(args.group_candidates),group_candidates_receipt_sha256=sha(args.group_candidates.with_suffix('.json')),
        group_calibration_report_sha256=calibration['comparison_sha256'],parent_policy='Frozen v8 TRAIN top8 with v6 stored_scale',
        source_sha256=v6.SOURCE,tokenizer_sha256=TOKENIZER,train_file_sha256=PROSE_FILE)


def audit_selected(path,candidates,binding,screen,torch):
    payload=torch.load(path,map_location='cpu',weights_only=True);receipt=read(path.with_suffix('.json'))
    chosen=screen['selection'];name=chosen['selected_id']
    expected=dict(format='MAMBA2_STATE_PPL_V9_CALIBRATION_V1',protocol_sha256=PROTOCOL,input_binding=binding,
        **chosen,selected_candidate_spec=candidates['candidate_specs'][name],table_sha256=tensor_sha(candidates['tables'][name]),
        scale_mode='stored_scale',int4_clip=1.,runtime_table_bytes=57344,cache_bytes=CACHE,
        selection_report_sha256=screen['comparison_sha256'])
    need(all(payload.get(k)==receipt.get(k)==value for k,value in expected.items()),'Selected mixed-table provenance differs')
    need(receipt.get('complete') is True and receipt['file']==path.name and receipt['sha256']==sha(path)
         and receipt['bytes']==path.stat().st_size and all(receipt.get(k)==v for k,v in payload.items() if k!='permutations'),
         'Selected payload/receipt identity differs')
    hash_inventory(payload['code_hashes'],run_paths(),'v9 selected table')
    table=payload['permutations']
    need(table.dtype==torch.uint8 and tuple(table.shape)==(56,8,128) and table.is_contiguous()
         and torch.equal(table,candidates['tables'][name]) and tensor_sha(table)==expected['table_sha256'],
         'Actual selected coordinates differ from independently reconstructed candidate')
    return payload,dict(complete=True,sha256=sha(path),receipt_sha256=sha(path.with_suffix('.json')),
        selected_id=name,table_sha256=expected['table_sha256'],baseline_wins=chosen['baseline_wins'])


def audit_run(directory,stage,args,train,candidates,binding,archive,screen=None,selected=None,validation=None,dataset=None):
    comparison_path=directory/(stage+'_comparison.json');comparison=read(comparison_path)
    start=candidates['input_binding']['start_condition']
    common=dict(format='MAMBA2_STATE_PPL_V9_EVAL_V1',stage=stage,protocol_sha256=PROTOCOL,input_binding=binding,
        adapter_loaded=False,adapter_sha256=None,mk_used=False,heldout_used=stage=='full',heldout_used_for_selection=False,
        selected_calibration_sha256=sha(args.selected_calibration) if selected is not None else None,
        parent_report_sha256=start['parent_report_sha256'] if stage=='full' else None,
        parent_comparison_sha256=start['comparison_sha256'] if stage=='full' else None,
        s16_report_sha256=v6.S16_ARCHIVE if stage=='full' else None)
    need(comparison.get('format')=='MAMBA2_STATE_PPL_V9_COMPARISON_V1' and comparison.get('complete') is True
         and all(comparison.get(k)==value for k,value in common.items() if k in ('stage','protocol_sha256','input_binding',
             'adapter_loaded','adapter_sha256','mk_used','heldout_used','heldout_used_for_selection')) and 'mk' not in comparison,
         'v9 comparison identity/provenance differs')
    hash_inventory(comparison['code_hashes'],run_paths(),'v9 PPL evaluation')
    need(not candidates['calibration_stopped'] and len(candidates['candidate_order'])>1,
         'No improving groups: protocol requires skipping redundant screen/full')
    if stage=='screen':
        windows=[(i*2048,train[i,:2048].tolist()) for i in SCREEN_ROWS]
        expected_dataset=dict(split='train',file_sha256=PROSE_FILE,rows=list(SCREEN_ROWS),tokens_per_row=2048,windows=32,target_tokens=65504)
        arms=tuple(candidates['candidate_order'])+('restored_baseline',)
    else:
        need(screen is not None and selected is not None and not screen['selection']['baseline_wins'],'Full requires frozen nonbaseline TRAIN winner')
        need(tokhash(validation.tolist())==VALIDATION_TOKENS,'Pinned validation tokens differ')
        windows=[(i,validation[i:min(i+2049,len(validation))].tolist()) for i in range(0,len(validation)-1,2048)]
        need(len(windows)==130 and sum(len(ids)-1 for _,ids in windows)==264764,'Full population differs')
        expected_dataset=dataset;arms=('v8_baseline','selected','restored_baseline')
    common['dataset']=expected_dataset;need(tuple(comparison['report_sha256'])==arms,'Group screen/full arm inventory/order differs')
    outputs={};metrics={};digests={}
    for arm in arms:
        name=selected['selected_id'] if arm=='selected' else 'baseline' if arm in ('v8_baseline','restored_baseline') else arm
        identity=dict(common,arm=arm,group_mix_spec=candidates['candidate_specs'][name])
        path=directory/(stage+'_'+arm+'.json');row=read(path);digests[arm]=sha(path)
        need(digests[arm]==comparison['report_sha256'][arm],'Raw group-combination report changed')
        metrics[arm]=v8.audit_arm(row,windows,identity,spec_for(name),tensor_sha(candidates['tables'][name]),comparison,
            stage=='screen' and name!='baseline' and arm!='restored_baseline')
        outputs[arm]=row
    baseline='baseline' if stage=='screen' else 'v8_baseline'
    restoration=read(directory/(stage+'_restoration.json'))
    v6.exact_restoration(outputs[baseline],outputs['restored_baseline'],restoration)
    need(comparison['restoration']==restoration,'Group parent restoration receipt differs')
    result=dict(complete=True,arms=metrics,report_sha256=digests,restoration=restoration,comparison_sha256=sha(comparison_path),no_MK_measurements=True)
    if stage=='screen':
        selection=select_independent(outputs,candidates['candidate_order'])
        need(comparison['selection']==selection,'Independent disjoint TRAIN selection differs')
        delta=v6.compare_independent(outputs['baseline'],outputs[selection['selected_id']])
        need(comparison['selected_vs_baseline']==delta,'Screen comparison arithmetic differs')
        result.update(selection=selection,selected_vs_baseline=delta)
    else:
        need(sha(args.v8_full_dir/'full_selected.json')==start['parent_report_sha256']
             and sha(args.v8_full_dir/'full_comparison.json')==start['comparison_sha256']
             and sha(args.v8_full_audit)==start['audit_sha256'] and sha(args.s16_report)==v6.S16_ARCHIVE,'Pinned parent/S16 context differs')
        s16=read(args.s16_report);v6.audit_ppl({'ppl':s16['ppl']},windows)
        current=outputs['v8_baseline']
        need(archive['ppl']==current['ppl'] and archive['repeated_reset_probe']==current['repeated_reset_probe']
             and archive['cache']==current['cache'] and current['cache']['total_bytes']==CACHE,
             'Archived v8 complete PPL/reset/cache replay differs')
        replay=dict(complete=True,per_window_nll_exact=True,ppl_windows_repeated=130,target_tokens=264764,
            reset_hidden_exact=True,reset_cache_exact=True,cache_bytes_exact=True)
        need(read(directory/'full_parent_replay.json')==comparison['parent_replay']==replay,'Archived parent replay receipt differs')
        expected=dict(selected_id=selected['selected_id'],selected_variant='stored_scale',selected_candidate_spec=selected['selected_candidate_spec'],
            selected_calibration_sha256=sha(args.selected_calibration),selection_report_sha256=screen['comparison_sha256'],
            parent_report_sha256=start['parent_report_sha256'],parent_comparison_sha256=start['comparison_sha256'],
            s16_report_sha256=v6.S16_ARCHIVE,resurface_trained=False,mk_evaluated=False)
        need(all(comparison.get(k)==value for k,value in expected.items()),'Full frozen candidate/context fields differ')
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
        'codec-checks','candidates','v5-calibration','v6-selected-calibration','kernel-checks','layer-candidates',
        'v8-selected-calibration','v8-screen-audit','v8-full-dir','v8-full-audit',
        'calibration-dir','group-candidates','eval-dir','screening-report','selected-calibration','s16-report','output'):
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test(),indent=2));return
    for key in ('stage','source_dir','calibration','repair_calibration','parent_training_report','prose_tokens',
        'codec_checks','candidates','v5_calibration','v6_selected_calibration','kernel_checks','layer_candidates',
        'v8_selected_calibration','v8_screen_audit','v8_full_dir','v8_full_audit','output'):
        if getattr(args,key) is None:parser.error('--'+key.replace('_','-')+' is required')
    if args.stage!='inputs' and args.calibration_dir is None:parser.error('--calibration-dir is required')
    if args.stage in ('screen','full'):
        for key in ('group_candidates','eval_dir'):
            if getattr(args,key) is None:parser.error('--'+key.replace('_','-')+' is required')
    if args.stage=='full':
        for key in ('screening_report','selected_calibration','s16_report'):
            if getattr(args,key) is None:parser.error('--'+key.replace('_','-')+' is required')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Set CUDA_VISIBLE_DEVICES= for CPU-only audit')
    need(not args.output.exists(),'Use a fresh output path; preserve earlier evidence')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1'
    dependencies=tuple(sorted(V8_DEPENDENCIES|{'audit_state_ppl_v8.py'}))
    result=dict(format='MAMBA2_STATE_PPL_V9_INDEPENDENT_AUDIT_V1',complete=False,passed=False,stage=args.stage,
        protocol_sha256=PROTOCOL,source_sha256=sha(__file__),auditor_dependency_sha256={name:sha(ROOT/'scripts'/name) for name in dependencies},
        limitations=['CPU reconstructs raw measurements and arithmetic, not GPU model logits.',
            'Frozen identity/version/gradient guards do not replace post-run source-weight byte hashes.',
            'Historical TRAIN/benchmark exposure remains; MK and Resurface are absent.',
            'Start-condition and inventory content are audited; execution chronology remains root orchestration evidence.',
            'The old v6 archive replay is bound to the unchanged independent v8 full audit; current v8 raw guards/arithmetic are recomputed.'])
    try:
        import numpy as np
        import torch
        from mamba2_recall import runtime
        torch.set_num_threads(4);need(not torch.cuda.is_initialized(),'CUDA initialized before CPU-only audit')
        result['self_tests']=self_test();tokenizer=runtime.SentencePieceTokenizer(args.source_dir)
        need(tokenizer.sha256==TOKENIZER,'Tokenizer differs')
        train,original,parent,binding,inventory,archive,proof=audit_inputs(args,tokenizer,torch,np);result['inputs']=proof
        if args.stage!='inputs':
            mixed,mixed_receipt,calibration=audit_calibration(args.calibration_dir,args,train,original,parent,binding,inventory,torch,np)
            result['calibration']=calibration
            if args.stage in ('screen','full'):
                need(args.group_candidates.resolve()==(args.calibration_dir/'candidates.pt').resolve(),'Group candidates must accompany audited calibration')
                mixed_binding=screen_binding(args,mixed,mixed_receipt,calibration)
                directory=args.eval_dir if args.stage=='screen' else args.screening_report.parent
                if args.stage=='full':need(args.screening_report.name=='screen_comparison.json','Frozen screen filename differs')
                screen=audit_run(directory,'screen',args,train,mixed,mixed_binding,archive);result['screen']=screen
                path=directory/'selected_calibration.pt'
                if args.selected_calibration is None:args.selected_calibration=path
                need(args.selected_calibration.resolve()==path.resolve(),'Selection must accompany audited screen')
                selected,selected_proof=audit_selected(path,mixed,mixed_binding,screen,torch);result['selected_calibration']=selected_proof
                if args.stage=='full':
                    from mamba2_recall.calibration import load_wikitext_tokens
                    validation,dataset=load_wikitext_tokens(tokenizer,'validation')
                    result['full']=audit_run(args.eval_dir,'full',args,train,mixed,mixed_binding,archive,screen,selected,validation,dataset)
        need(not torch.cuda.is_initialized(),'CPU-only audit initialized CUDA')
        result.update(complete=True,passed=True,cuda_initialized=False)
    except BaseException as error:result['error']=repr(error);raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
