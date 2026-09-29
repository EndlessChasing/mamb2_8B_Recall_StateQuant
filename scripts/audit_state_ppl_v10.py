#!/usr/bin/env python3
"""Independent CPU audit of fixed 52-byte v10 state-tier allocations.

Reuses frozen independent auditors for predecessor evidence; never imports the
v10 scorer/selector or regenerates model logits. No GPU execution.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import audit_state_ppl_v6 as v6
import audit_state_ppl_v8 as v8
import audit_state_ppl_v9 as v9
from audit_quant_first import sha,need,finite,tokhash,read,TOKENIZER,PROSE_FILE,VALIDATION_TOKENS
from audit_state_repair import hash_inventory,tensor_sha,frozen
from audit_resurface_more import audit_backend_policy

PROTOCOL='565707cf401b2ab472b9b78368a7192afeebe73056efd811b7acba610e3d9567'
V9_SELECTED='3467897358f33de22b1b629819cb2035f4cb8912914f0183959aa581fafeab50'
V9_SELECTED_RECEIPT='4e4bb614ff1ba0dfbbd1ba6f3116474cd959555e8fb22406590190e65e9d9a05'
V9_SCREEN='4098ad9446cb08d18c0ce23a8ce199c026a3836ae6fe5c5f3d21eadf2222d757'
V9_SCREEN_AUDIT='4b218de5223ed35cf12be6c341e3dd3d5b75a82656019c8ac12526ce67f60402'
V9_TABLE='b1865e81ff3fbed028027a883872aef9bb91e614e7cf08c72211496a78aeb597'
V9_AUDITOR='fbb30790aa47431713806fdf8673a4d30db29cddd427cdc4374919458f1b2cb3'
LAYOUTS={'16_64_48':(16,64,48),'8_80_40':(8,80,40),'24_48_56':(24,48,56),'32_32_64':(32,32,64)}
BASELINE='16_64_48'
SCREEN_ROWS=tuple(range(184,216))
CACHE=28499968


def run_paths():
    return v9.run_paths()|{'scripts/run_state_ppl_v10.py','scripts/state_ppl_codec_v10.py',
        'scripts/check_state_ppl_codec_v10.py','scripts/audit_state_ppl_v10.py','docs/STATE_PPL_V10_PROTOCOL.md'}


def independent_geometry(layout,batch=1,heads=128,dim=64,layers=56):
    n8,n4,nzero=LAYOUTS[layout];rows=batch*heads*dim;prefix=[batch,heads,dim]
    need(n8+n4+nzero==128 and n8+n4//2+4==52,'Layout does not retain52-byte row')
    tensors={name:dict(shape=prefix+[count],dtype='torch.uint8',bytes=rows*count)
        for name,count in (('lo',n8//2),('hi',n8//2),('q4',n4//2))}
    tensors.update({name:dict(shape=prefix,dtype='torch.float16',bytes=rows*2) for name in ('s8','s4')})
    return dict(tensors=tensors,per_layer_bytes=rows*52,all_layer_bytes=layers*rows*52,
        payload_bytes=layers*rows*48,scale_bytes=layers*rows*4,n8=n8,n4=n4,nzero=nzero,row_bytes=52)


def eligible(row):
    return (row.get('complete') is True and 'error' not in row and 'mk' not in row
        and finite(row.get('ppl',{}).get('nll')) and finite(row.get('ppl',{}).get('ppl'))
        and row.get('cache',{}).get('total_bytes')==CACHE)


def select_independent(rows):
    need(set(rows)==set(LAYOUTS)|{'restored_baseline'},'Fixed4-layout screen inventory differs')
    valid=[name for name in LAYOUTS if eligible(rows[name])];need(BASELINE in valid,'Failed baseline must stop')
    return min(valid,key=lambda name:(rows[name]['ppl']['ppl'],name!=BASELINE,list(LAYOUTS).index(name))),valid


def target_pass(ppl,cache):return finite(ppl) and ppl<8.25 and cache==CACHE


def self_test():
    checks=[]
    for name,(n8,n4,nzero) in LAYOUTS.items():
        g=independent_geometry(name)
        need(g['all_layer_bytes']==23855104 and g['payload_bytes']==22020096 and g['scale_bytes']==1835008
             and sum(x['bytes'] for x in g['tensors'].values())==g['per_layer_bytes'],'Physical52-byte accounting differs')
        need(g['tensors']['lo']['shape'][-1]==n8//2 and g['tensors']['q4']['shape'][-1]==n4//2,'Tier padded in physical allocation')
    checks.append('all4-actual52-byte-geometries-no-padding')
    for name in LAYOUTS:
        g=independent_geometry(name,batch=2,heads=6,dim=19,layers=1)
        need(g['per_layer_bytes']==2*6*19*52 and sum(x['bytes'] for x in g['tensors'].values())==g['per_layer_bytes'],'Odd geometry accounting differs')
    checks.append('odd-P-multibatch-actual-storage')
    rows={name:dict(complete=True,ppl=dict(nll=100.,ppl=9.),cache=dict(total_bytes=CACHE)) for name in (*LAYOUTS,'restored_baseline')}
    need(select_independent(rows)[0]==BASELINE,'Exact tie must retain baseline');checks.append('baseline-exact-tie')
    rows['8_80_40']['ppl']['ppl']=rows['24_48_56']['ppl']['ppl']=math.nextafter(9.,0.)
    need(select_independent(rows)[0]=='8_80_40','Layout order tiebreak differs');checks.append('nonbaseline-exact-tie-order')
    rows['8_80_40']['complete']=False
    need(select_independent(rows)[0]=='24_48_56','Known failed alternative removes valid candidate');checks.append('known-nonfinite-exclusion')
    rows['24_48_56']['cache']['total_bytes']+=1
    need(select_independent(rows)[0]==BASELINE,'One extra byte accepted');checks.append('actual-byte-budget')
    rows[BASELINE]['complete']=False
    try:select_independent(rows)
    except ValueError:pass
    else:raise ValueError('Failed baseline accepted')
    checks.append('failed-baseline-fatal')
    need(not target_pass(8.25,CACHE) and target_pass(math.nextafter(8.25,0.),CACHE)
         and not target_pass(8.24,CACHE+1),'Strict target/budget boundary differs');checks.append('strict8p25-and-same-budget')
    need(not set(SCREEN_ROWS)&set(v9.CALIBRATION_ROWS+v9.SCREEN_ROWS) and 32*2047==65504,'Disjoint token population differs')
    checks.append('fixed-disjoint32-TRAIN-windows')
    return dict(complete=True,passed=True,count=len(checks),checks=checks,gpu_used=False)


def audit_v9_receipt(path,stage):
    row=read(path);dependencies=v9.V8_DEPENDENCIES|{'audit_state_ppl_v8.py'}
    need(sha(ROOT/'scripts/audit_state_ppl_v9.py')==V9_AUDITOR
         and row.get('format')=='MAMBA2_STATE_PPL_V9_INDEPENDENT_AUDIT_V1'
         and row.get('complete') is True and row.get('passed') is True and row.get('cuda_initialized') is False
         and row.get('stage')==stage and row.get('protocol_sha256')==v9.PROTOCOL and row.get('source_sha256')==V9_AUDITOR,
         'Completed unchanged independent v9 audit required')
    need(set(row['auditor_dependency_sha256'])==dependencies and all(
        sha(ROOT/'scripts'/name)==digest for name,digest in row['auditor_dependency_sha256'].items()),'v9 auditor dependencies changed')
    return row


def audit_predecessor(args,tokenizer,torch,np):
    """Reconstruct full v9 chain using its unchanged independent auditor."""
    need(sha(ROOT/'docs/STATE_PPL_V10_PROTOCOL.md')==PROTOCOL,'Frozen v10 protocol changed')
    prior=argparse.Namespace(**vars(args));prior.selected_calibration=None
    train,original,v8parent,v9binding,inventory,v8archive,upstream=v9.audit_inputs(prior,tokenizer,torch,np)
    candidates,receipt,calibration=v9.audit_calibration(args.group_candidates.parent,prior,train,original,v8parent,v9binding,inventory,torch,np)
    need(args.group_candidates.name=='candidates.pt' and sha(args.group_candidates)==calibration['candidates_sha256'],'v9 group candidates differ')
    oldbinding=v9.screen_binding(prior,candidates,receipt,calibration)
    need(sha(args.v9_selected_calibration)==V9_SELECTED and sha(args.v9_selected_calibration.with_suffix('.json'))==V9_SELECTED_RECEIPT
         and sha(args.v9_selected_calibration.parent/'screen_comparison.json')==V9_SCREEN
         and sha(args.v9_screen_audit)==V9_SCREEN_AUDIT,'Frozen v9 TRAIN selection differs')
    screen=v9.audit_run(args.v9_selected_calibration.parent,'screen',prior,train,candidates,oldbinding,v8archive)
    parent,selectedproof=v9.audit_selected(args.v9_selected_calibration,candidates,oldbinding,screen,torch)
    need(parent['selected_id']=='top2' and not parent['baseline_wins'] and tensor_sha(parent['permutations'])==V9_TABLE,
         'Parent must remain v9 TRAIN top2 regardless full result')
    screen_audit=audit_v9_receipt(args.v9_screen_audit,'screen')
    need(screen_audit['screen']==screen and screen_audit['selected_calibration']==selectedproof
         and screen_audit['calibration']==calibration,'v9 screen audit no longer matches reconstructed evidence')
    from mamba2_recall.calibration import load_wikitext_tokens
    validation,dataset=load_wikitext_tokens(tokenizer,'validation')
    prior.selected_calibration=args.v9_selected_calibration
    full=v9.audit_run(args.v9_full_dir,'full',prior,train,candidates,oldbinding,v8archive,screen,parent,validation,dataset)
    full_audit=audit_v9_receipt(args.v9_full_audit,'full')
    need(full_audit['full']==full and full_audit['screen']==screen and full_audit['selected_calibration']==selectedproof,
         'v9 full audit no longer matches reconstructed evidence')
    archive=read(args.v9_full_dir/'full_selected.json')
    v9.assert_finite_miss(archive['ppl']['ppl'],full['target_checks'],full['target_pass'])
    return train,parent,oldbinding,archive,dict(complete=True,upstream=upstream,v9_calibration=calibration,
        v9_screen=screen,v9_selected_calibration=selectedproof,v9_full=full,
        v9_screen_audit_sha256=sha(args.v9_screen_audit),v9_full_audit_sha256=sha(args.v9_full_audit)),validation,dataset


def descriptor_independent(layout):
    n8,n4,nzero=LAYOUTS[layout]
    return dict(layout=layout,n8=n8,n4=n4,nzero=nzero,payload_bytes=48,scale_bytes=4,row_bytes=52,
        resident_layout_metadata_bytes=0,tensor_widths=dict(lo=n8//2,hi=n8//2,q4=n4//2))


def audit_storage(actual,layout):
    descriptor=descriptor_independent(layout);geometry=independent_geometry(layout)
    need(all(actual.get(key)==value for key,value in descriptor.items()) and len(actual.get('layers',[]))==56,
         'Physical layout descriptor/layer inventory differs')
    expected_tensors={name:dict(shape=entry['shape'],dtype=entry['dtype'],storage_bytes=entry['bytes'])
        for name,entry in geometry['tensors'].items()}
    state_bytes=conv_bytes=0
    for layer in actual['layers']:
        need(layer==dict(state_shape=[1,128,64,128],state_bytes=geometry['per_layer_bytes'],
            tensors=expected_tensors,conv_shape=[1,10240,4],conv_storage_bytes=81920),
            'Actual packed tensor geometry/storage has padding or changed allocation')
        state_bytes+=layer['state_bytes'];conv_bytes+=layer['conv_storage_bytes']
    need(state_bytes==23855104 and conv_bytes==4587520 and state_bytes+conv_bytes+57344==CACHE,
         'Physical56-layer allocation accounting differs')
    return dict(complete=True,layout=layout,ssm_bytes=state_bytes,conv_bytes=conv_bytes,table_bytes=57344,total_bytes=CACHE,
        tensors_verified=56*6,row_bytes=52)


def spec_for(layout):
    return dict(candidate_id=layout,candidate_name='fixed_parent_allocation',variant='stored_scale',layout=layout,
        scale_mode='stored_scale',int4_clip=1.,diagnostic=None,deployable=True)


def selection_record(rows):
    selected,valid=select_independent(rows)
    return dict(selected_id=selected,selected_variant='stored_scale',selected_layout=selected,baseline_id=BASELINE,
        valid=valid,excluded=[name for name in LAYOUTS if name not in valid],baseline_wins=selected==BASELINE,
        adapter_used=False,heldout_used=False,mk_used=False,
        rule='Minimum complete finite exact-budget disjoint TRAIN PPL; exact ties baseline then frozen candidate export order')


def audit_inputs(args,tokenizer,torch,np):
    train,parent,oldbinding,archive,predecessor,validation,dataset=audit_predecessor(args,tokenizer,torch,np)
    kernel=audit_kernel_receipt(args.layout_checks,torch,np)
    start=dict(complete=True,comparison_sha256=sha(args.v9_full_dir/'full_comparison.json'),audit_sha256=sha(args.v9_full_audit),
        parent_report_sha256=sha(args.v9_full_dir/'full_selected.json'),selected_calibration_sha256=V9_SELECTED,
        parent_ppl=archive['ppl']['ppl'],parent_selection='Frozen v9 TRAIN top2 irrespective of full comparison',
        criterion='Passing independent full audit; finite v9 PPL >=8.25; all integrity guards pass')
    binding=dict(protocol_sha256=PROTOCOL,v9_input_binding=oldbinding,v9_selected_calibration_sha256=V9_SELECTED,
        v9_selected_receipt_sha256=V9_SELECTED_RECEIPT,v9_screen_audit_sha256=V9_SCREEN_AUDIT,
        v9_selection_report_sha256=V9_SCREEN,parent_table_sha256=V9_TABLE,start_condition=start,
        kernel_checks_sha256=sha(args.layout_checks),source_sha256=v6.SOURCE,tokenizer_sha256=TOKENIZER,train_file_sha256=PROSE_FILE)
    candidates=dict(candidate_order=list(LAYOUTS),tables={name:parent['permutations'] for name in LAYOUTS},
        candidate_specs={name:descriptor_independent(name) for name in LAYOUTS})
    source_hashes={name:sha(ROOT/name) for name in sorted(run_paths())}
    hash_inventory(source_hashes,run_paths(),'v10 prospective sources')
    return train,candidates,binding,archive,validation,dataset,dict(complete=True,predecessor=predecessor,
        kernel=kernel,input_binding=binding,code_hashes=source_hashes,
        independently_reconstructed_layouts={name:independent_geometry(name) for name in LAYOUTS},
        parent_table_sha256=V9_TABLE,parent_full_ppl=archive['ppl']['ppl'],no_model_execution=True,no_MK_measurements=True)


def audit_selected(path,candidates,binding,screen,torch):
    payload=torch.load(path,map_location='cpu',weights_only=True);receipt=read(path.with_suffix('.json'))
    chosen=screen['selection'];name=chosen['selected_id']
    expected=dict(format='MAMBA2_STATE_PPL_V10_CALIBRATION_V1',protocol_sha256=PROTOCOL,input_binding=binding,
        **chosen,selected_candidate_spec=descriptor_independent(name),table_sha256=V9_TABLE,
        scale_mode='stored_scale',int4_clip=1.,runtime_table_bytes=57344,cache_bytes=CACHE,
        selection_report_sha256=screen['comparison_sha256'])
    need(all(payload.get(k)==receipt.get(k)==value for k,value in expected.items()),'Selected layout provenance differs')
    need(receipt.get('complete') is True and receipt['file']==path.name and receipt['sha256']==sha(path)
         and receipt['bytes']==path.stat().st_size and all(receipt.get(k)==v for k,v in payload.items() if k!='permutations'),
         'Selected layout payload/receipt identity differs')
    hash_inventory(payload['code_hashes'],run_paths(),'v10 selected layout')
    table=payload['permutations']
    need(table.dtype==torch.uint8 and tuple(table.shape)==(56,8,128) and table.is_contiguous()
         and torch.equal(table,candidates['tables'][name]) and tensor_sha(table)==V9_TABLE,
         'Coordinate table changed during allocation search')
    return payload,dict(complete=True,sha256=sha(path),receipt_sha256=sha(path.with_suffix('.json')),
        selected_id=name,selected_layout=name,table_sha256=V9_TABLE,baseline_wins=chosen['baseline_wins'])


def audit_run(directory,stage,args,train,candidates,binding,archive,screen=None,selected=None,validation=None,dataset=None):
    path=directory/(stage+'_comparison.json');comparison=read(path);start=binding['start_condition']
    common=dict(format='MAMBA2_STATE_PPL_V10_EVAL_V1',stage=stage,protocol_sha256=PROTOCOL,input_binding=binding,
        adapter_loaded=False,adapter_sha256=None,mk_used=False,heldout_used=stage=='full',heldout_used_for_selection=False,
        selected_calibration_sha256=sha(args.selected_calibration) if selected is not None else None,
        parent_report_sha256=start['parent_report_sha256'] if stage=='full' else None,
        parent_comparison_sha256=start['comparison_sha256'] if stage=='full' else None,
        s16_report_sha256=v6.S16_ARCHIVE if stage=='full' else None)
    need(comparison.get('format')=='MAMBA2_STATE_PPL_V10_COMPARISON_V1' and comparison.get('complete') is True
         and all(comparison.get(k)==value for k,value in common.items() if k in ('stage','protocol_sha256','input_binding',
             'adapter_loaded','adapter_sha256','mk_used','heldout_used','heldout_used_for_selection')) and 'mk' not in comparison,
         'Allocation comparison identity/provenance differs')
    hash_inventory(comparison['code_hashes'],run_paths(),'v10 PPL evaluation')
    if stage=='screen':
        windows=[(i*2048,train[i,:2048].tolist()) for i in SCREEN_ROWS]
        expected_dataset=dict(split='train',file_sha256=PROSE_FILE,rows=list(SCREEN_ROWS),tokens_per_row=2048,windows=32,target_tokens=65504)
        arms=tuple(LAYOUTS)+('restored_baseline',)
    else:
        need(screen is not None and selected is not None and not screen['selection']['baseline_wins'],'Full requires frozen nonbaseline TRAIN winner')
        need(tokhash(validation.tolist())==VALIDATION_TOKENS,'Pinned validation stream differs')
        windows=[(i,validation[i:min(i+2049,len(validation))].tolist()) for i in range(0,len(validation)-1,2048)]
        need(len(windows)==130 and sum(len(ids)-1 for _,ids in windows)==264764,'Full population differs')
        expected_dataset=dataset;arms=('v9_baseline','selected','restored_baseline')
    common['dataset']=expected_dataset;need(tuple(comparison['report_sha256'])==arms,'Fixed allocation arm inventory/order differs')
    outputs={};metrics={};digests={};storage={}
    for arm in arms:
        name=selected['selected_id'] if arm=='selected' else BASELINE if arm in ('v9_baseline','restored_baseline') else arm
        identity=dict(common,arm=arm,allocation_spec=descriptor_independent(name))
        raw_path=directory/(stage+'_'+arm+'.json');row=read(raw_path);digests[arm]=sha(raw_path)
        need(digests[arm]==comparison['report_sha256'][arm],'Raw allocation report changed')
        metrics[arm]=v8.audit_arm(row,windows,identity,spec_for(name),V9_TABLE,comparison,
            stage=='screen' and name!=BASELINE and arm!='restored_baseline')
        if row.get('complete') is True:
            need(row.get('allocation_storage_validated') is True,'Actual physical storage validation absent')
            probe=audit_storage(row['storage_descriptor_probe'],name);final=audit_storage(row['storage_descriptor'],name)
            need(row['storage_descriptor_probe']==row['storage_descriptor'],'Physical storage changed between probe and final window')
            storage[arm]=dict(probe=probe,final=final)
        outputs[arm]=row
    baseline=BASELINE if stage=='screen' else 'v9_baseline'
    restoration=read(directory/(stage+'_restoration.json'))
    v6.exact_restoration(outputs[baseline],outputs['restored_baseline'],restoration)
    need(comparison['restoration']==restoration,'Allocation baseline restoration differs')
    result=dict(complete=True,arms=metrics,report_sha256=digests,storage=storage,restoration=restoration,
        comparison_sha256=sha(path),no_MK_measurements=True)
    if stage=='screen':
        selection=selection_record(outputs);need(comparison['selection']==selection,'Independent layout selection differs')
        delta=v6.compare_independent(outputs[BASELINE],outputs[selection['selected_id']])
        need(comparison['selected_vs_baseline']==delta,'Screen raw-window comparison differs')
        result.update(selection=selection,selected_vs_baseline=delta)
    else:
        need(sha(args.v9_full_dir/'full_selected.json')==start['parent_report_sha256']
             and sha(args.v9_full_dir/'full_comparison.json')==start['comparison_sha256']
             and sha(args.v9_full_audit)==start['audit_sha256'] and sha(args.s16_report)==v6.S16_ARCHIVE,'Pinned predecessor/S16 context differs')
        s16=read(args.s16_report);v6.audit_ppl({'ppl':s16['ppl']},windows);current=outputs['v9_baseline']
        # Entire legacy dictionaries, including every original field, remain exact.
        need(archive['ppl']==current['ppl'] and archive['repeated_reset_probe']==current['repeated_reset_probe']
             and archive['cache']==current['cache'],'Archived v9 complete PPL/probe/cache dictionary replay differs')
        replay=dict(complete=True,per_window_nll_exact=True,ppl_windows_repeated=130,target_tokens=264764,
            reset_hidden_exact=True,reset_cache_exact=True,cache_bytes_exact=True)
        need(read(directory/'full_parent_replay.json')==comparison['parent_replay']==replay,'Archived parent replay receipt differs')
        expected=dict(selected_id=selected['selected_id'],selected_variant='stored_scale',selected_layout=selected['selected_layout'],
            selected_candidate_spec=descriptor_independent(selected['selected_layout']),selected_calibration_sha256=sha(args.selected_calibration),
            selection_report_sha256=screen['comparison_sha256'],parent_report_sha256=start['parent_report_sha256'],
            parent_comparison_sha256=start['comparison_sha256'],s16_report_sha256=v6.S16_ARCHIVE,resurface_trained=False,mk_evaluated=False)
        need(all(comparison.get(k)==value for k,value in expected.items()),'Full frozen layout/context fields differ')
        delta=v6.compare_independent(current,outputs['selected']);gap=v6.compare_independent(s16,outputs['selected'])
        checks=dict(ppl_strictly_below_8p25=outputs['selected']['ppl']['ppl']<8.25,
            cache_same_budget=all(row['cache']['total_bytes']==CACHE for row in outputs.values()),all_integrity_checks_passed=True)
        need(comparison['comparison']==delta and comparison['original_s16_comparison']==gap
             and comparison['target_checks']==checks and comparison['target_pass']==all(checks.values()),'Full target arithmetic differs')
        result.update(comparison=delta,original_s16_comparison=gap,target_checks=checks,target_pass=all(checks.values()),parent_replay=replay)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true');parser.add_argument('--stage',choices=('inputs','screen','full'))
    for name in ('source-dir','calibration','repair-calibration','parent-training-report','prose-tokens',
        'codec-checks','candidates','v5-calibration','v6-selected-calibration','kernel-checks','layer-candidates',
        'v8-selected-calibration','v8-screen-audit','v8-full-dir','v8-full-audit','group-candidates',
        'v9-selected-calibration','v9-screen-audit','v9-full-dir','v9-full-audit','layout-checks','s16-report',
        'eval-dir','screening-report','selected-calibration','output'):
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test(),indent=2));return
    for key in ('stage','source_dir','calibration','repair_calibration','parent_training_report','prose_tokens',
        'codec_checks','candidates','v5_calibration','v6_selected_calibration','kernel_checks','layer_candidates',
        'v8_selected_calibration','v8_screen_audit','v8_full_dir','v8_full_audit','group_candidates',
        'v9_selected_calibration','v9_screen_audit','v9_full_dir','v9_full_audit','layout_checks','s16_report','output'):
        if getattr(args,key) is None:parser.error('--'+key.replace('_','-')+' is required')
    if args.stage!='inputs' and args.eval_dir is None:parser.error('--eval-dir is required')
    if args.stage=='full':
        for key in ('screening_report','selected_calibration'):
            if getattr(args,key) is None:parser.error('--'+key.replace('_','-')+' is required')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Set CUDA_VISIBLE_DEVICES= for CPU-only audit')
    need(not args.output.exists(),'Use a fresh output path; preserve earlier evidence')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1'
    dependencies=tuple(sorted(v9.V8_DEPENDENCIES|{'audit_state_ppl_v8.py','audit_state_ppl_v9.py'}))
    result=dict(format='MAMBA2_STATE_PPL_V10_INDEPENDENT_AUDIT_V1',complete=False,passed=False,stage=args.stage,
        protocol_sha256=PROTOCOL,source_sha256=sha(__file__),auditor_dependency_sha256={name:sha(ROOT/'scripts'/name) for name in dependencies},
        limitations=['CPU verifies serialized test evidence and model-score arithmetic, not GPU logits.',
            'Identity/version/gradient guards do not replace post-run source-weight byte hashes.',
            'Physical persistent cache excludes model weights, temporary activations/registers and allocator reserve.',
            'Prior benchmark exposure remains; no MK or Resurface evidence is generated.',
            'Execution chronology is root orchestration evidence, not inferred from timestamps.'])
    try:
        import numpy as np
        import torch
        from mamba2_recall import runtime
        torch.set_num_threads(4);need(not torch.cuda.is_initialized(),'CUDA initialized before CPU audit')
        result['self_tests']=self_test();tokenizer=runtime.SentencePieceTokenizer(args.source_dir)
        need(tokenizer.sha256==TOKENIZER,'Tokenizer differs')
        train,candidates,binding,archive,validation,dataset,proof=audit_inputs(args,tokenizer,torch,np);result['inputs']=proof
        if args.stage!='inputs':
            directory=args.eval_dir if args.stage=='screen' else args.screening_report.parent
            if args.stage=='full':need(args.screening_report.name=='screen_comparison.json','Frozen screen filename differs')
            screen=audit_run(directory,'screen',args,train,candidates,binding,archive);result['screen']=screen
            path=directory/'selected_calibration.pt'
            if args.selected_calibration is None:args.selected_calibration=path
            need(args.selected_calibration.resolve()==path.resolve(),'Selected layout must accompany audited screen')
            selected,selectedproof=audit_selected(path,candidates,binding,screen,torch);result['selected_calibration']=selectedproof
            if args.stage=='full':result['full']=audit_run(args.eval_dir,'full',args,train,candidates,binding,archive,screen,selected,validation,dataset)
        need(not torch.cuda.is_initialized(),'CPU audit initialized CUDA');result.update(complete=True,passed=True,cuda_initialized=False)
    except BaseException as error:result['error']=repr(error);raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)


# Kernel-receipt verifier is defined below before main executes.


SPECIALS=('ties','negative','zero','rounded_scales','underflow','subnormal','mixed_scales','large_finite')
GEOMETRIES=((2,17,8,19,2),(1,9,6,3,3),(2,5,4,1,1))


def expected_check_names():
    names=['frozen-protocol','fixed-layout-grid']
    for limit in (127,7):names += ['CPU-signed-half-boundaries-'+str(limit),'GPU-shared-rounding-boundaries-'+str(limit)]
    for layout in LAYOUTS:
        names += ['CPU-actual-unpadded-storage-'+layout,'CPU-reject-padded-payload-'+layout]
        cases=['one-'+special+'-'+layout for special in SPECIALS]+['65token-'+layout]
        for name in cases:names += ['CPU-finite-'+name,'GPU-oracle-exact-'+name]
        names += ['controlled65-partitions-'+layout]
        for name in ('poisoned-neighbor-boundaries-'+layout,'dead-current-readout-zero-carry-'+layout):
            names += ['CPU-finite-'+name,'GPU-oracle-exact-'+name]
        names += ['dead-pulse-current-only-'+layout]
    for geometry in GEOMETRIES:
        for layout in LAYOUTS:
            names += [prefix+str(geometry)+'-'+layout for prefix in (
                'random-partitions-','random-actual-bytes-','random-signed-INT4-range-','random-zero-reset-replay-')]
            if layout==BASELINE:names += ['baseline-delegates-v6-'+str(geometry)]
    for layout in LAYOUTS:
        names += ['actual-production-allocation-'+layout]
        if layout==BASELINE:names += ['baseline-cache-dictionary-delegates-v6']
        names += ['controller-reset-clears-all-state-'+layout,'reject-nonfinite-FP16-scale-'+layout]
    return names


def signed_round_np(values,np):
    magnitude=np.abs(values);whole=np.floor(magnitude)
    return np.sign(values)*(whole+(magnitude-whole>=np.float32(.5)).astype(np.float32))


def pack_np(values,np):
    nibbles=(values.astype(np.int64)&15).astype(np.uint8)
    return nibbles[...,::2] | (nibbles[...,1::2]<<np.uint8(4))


def quantize_np(state,layout,np):
    n8,n4,_=LAYOUTS[layout];codes=[];scales=[]
    for values,limit in ((state[...,:n8],127),(state[...,n8:n8+n4],7)):
        denominator=np.maximum(np.max(np.abs(values),axis=-1)/np.float32(limit),np.float32(1e-8))
        scale=denominator.astype(np.float16);safe=np.where(scale>0,scale.astype(np.float32),np.float32(1))
        rounded=signed_round_np(values/safe[...,None],np)
        code=np.where(scale[...,None]>0,np.clip(rounded,-limit,limit),0).astype(np.int32)
        codes.append(code);scales.append(scale)
    return dict(lo=pack_np(codes[0],np),hi=pack_np(codes[0]>>4,np),q4=pack_np(codes[1],np),s8=scales[0],s4=scales[1])


def decode_np(buffers,layout,np):
    def unpack(array):
        out=np.empty((*array.shape[:-1],array.shape[-1]*2),dtype=np.int32)
        out[...,::2]=array&15;out[...,1::2]=array>>4
        return out
    n8,n4,_=LAYOUTS[layout]
    q8=(unpack(buffers['hi'])<<4)|unpack(buffers['lo']);q8=np.where(q8>=128,q8-256,q8)
    q4=unpack(buffers['q4']);q4=np.where(q4>=8,q4-16,q4)
    output=np.zeros((*buffers['s8'].shape,128),dtype=np.float32)
    output[...,:n8]=q8.astype(np.float32)*buffers['s8'].astype(np.float32)[...,None]
    output[...,n8:n8+n4]=q4.astype(np.float32)*buffers['s4'].astype(np.float32)[...,None]
    return output


def controlled_np(layout,length,special,dim,np):
    n8,n4,nzero=LAYOUTS[layout];batch,heads,groups=2,6,3
    table=np.stack([np.roll(np.arange(127,-1,-1),9*g) for g in range(groups)]).astype(np.uint8)
    x=np.empty((batch,length,heads,dim),np.float16)
    raw=np.zeros((batch,length,heads),np.float16);A=np.zeros(heads,np.float32);bias=np.full(heads,32.,np.float32)
    D=np.array([(-1.)**h/16 for h in range(heads)],np.float16)
    B=np.empty((batch,length,groups,128),np.float16);C=np.zeros_like(B)
    eight=np.array([-127.,-126.5,-64.,-2.,-1.5,-.5,0.,.5,1.5,2.,63.,64.,126.,126.5,127.,0.],np.float32)
    four=np.array([-7.,-6.5,-4.,-1.5,-.5,0.,.5,1.5,4.,6.5,7.],np.float32)
    wanted=np.concatenate((np.tile(eight,(n8+15)//16)[:n8],np.tile(four,(n4+10)//11)[:n4],(np.arange(nzero)%5-2).astype(np.float32)))
    if special=='negative':wanted=-np.abs(wanted)
    elif special=='zero':wanted.fill(0)
    elif special=='rounded_scales':wanted[0]=127.0625;wanted[n8]=7.00390625
    elif special=='underflow':wanted=np.sign(wanted)*np.float32(2**-19)
    elif special=='subnormal':wanted=np.sign(wanted)*np.float32(2**-11)
    elif special=='mixed_scales':wanted[:n8]=np.sign(wanted[:n8])*np.float32(2**-19);wanted[n8:]=np.sign(wanted[n8:])*np.float32(2**-15)
    elif special=='large_finite':wanted=np.sign(wanted)*np.float32(131072.);wanted[n8+n4:]=0.
    boundaries=(0,n8-1,n8,n8+n4-1,n8+n4,127,n8+1)
    for token in range(length):
        sign=(1.,1.,-1.,-1.)[token%4]
        for b in range(batch):
            for h in range(heads):
                for p in range(dim):x[b,token,h,p]=(-1.)**(b+h+p)*2.**((token//4+b+h+p)%3-6)
            for g in range(groups):
                B[b,token,g,table[g]]=(wanted*np.float32(sign)/np.float32(32)).astype(np.float16)
                C[b,token,g,table[g,boundaries[token%len(boundaries)]]]=(-1.)**(b+g)/16
    if special in ('underflow','subnormal','mixed_scales'):x.fill(2**-5)
    if special=='large_finite':x.fill(1);C.fill(0);D.fill(0)
    return [x,raw,A,B,C,D,bias],table


def poisoned_np(layout,np):
    n8,n4,_=LAYOUTS[layout];row=np.arange(2*6*19).reshape(2,6,19,1)
    q8=((row*31+np.arange(n8)*17)%255-127).astype(np.int32);q8[...,0]=127
    q4=((row*7+np.arange(n4)*5)%15-7).astype(np.int32);q4[...,0]=7
    s8=np.exp2((row[...,0]%4).astype(np.float32)-4).astype(np.float16)
    s4=np.exp2((row[...,0]%3).astype(np.float32)-3).astype(np.float16)
    return dict(lo=pack_np(q8,np),hi=pack_np(q8>>4,np),q4=pack_np(q4,np),s8=s8,s4=s4)


def case_definition(name,layout,np):
    initial=None
    if name.startswith('one-'):
        special=name[len('one-'):-len('-'+layout)];need(special in SPECIALS,'Unexpected controlled special')
        inputs,table=controlled_np(layout,1,special,5,np)
    elif name=='65token-'+layout:inputs,table=controlled_np(layout,65,'ties',5,np)
    elif name=='poisoned-neighbor-boundaries-'+layout:
        inputs,table=controlled_np(layout,7,'ties',19,np)
        for index in (0,3,5):inputs[index].fill(0)
        initial=poisoned_np(layout,np)
    elif name=='dead-current-readout-zero-carry-'+layout:
        inputs,table=controlled_np(layout,2,'ties',5,np);inputs[0].fill(1)
        for index in (3,4,5):inputs[index].fill(0)
        n8,n4,_=LAYOUTS[layout]
        for group in range(3):
            pos=table[group,n8+n4];inputs[3][:,0,group,pos]=1/32;inputs[4][:,:,group,pos]=1
    else:raise ValueError('Unexpected controlled case: '+name)
    return inputs,table,initial


def audit_controlled_case(case,row,torch,np):
    layout=case['layout'];name=case['name'];expected_inputs,table,initial=case_definition(name,layout,np)
    need(set(case)=={'name','layout','inputs','table','initial','output','caches'},'Controlled evidence fields differ')
    need(len(case['inputs'])==7 and all(t.device.type=='cpu' and t.numpy().dtype==e.dtype and t.numpy().shape==e.shape
         and t.contiguous().numpy().tobytes()==e.tobytes() for t,e in zip(case['inputs'],expected_inputs)),
         'Controlled fixture input bits differ: '+name)
    need(case['table'].dtype==torch.uint8 and case['table'].device.type=='cpu'
         and np.array_equal(case['table'].numpy(),table),'Controlled permutation differs')
    if initial is None:need(case['initial'] is None,'Unexpected initial carry')
    else:
        need(set(case['initial'])==set(initial) and all(case['initial'][k].numpy().tobytes()==v.tobytes()
             and case['initial'][k].numpy().dtype==v.dtype and tuple(case['initial'][k].shape)==v.shape for k,v in initial.items()),
             'Poisoned adjacent-row bytes differ')
    inputs=[v.astype(np.float32) for v in expected_inputs]
    x,raw,A,B,C,D,bias=inputs;b,length,heads,dim=x.shape;n8,n4,_=LAYOUTS[layout]
    need(np.all(raw==0) and np.all(A==0) and np.all(bias==32),'Oracle exact dt32/decay1 scope differs')
    group=np.arange(heads)//(heads//B.shape[2]);indices=table[group][None,:,:].repeat(b,axis=0)
    packed=quantize_np(np.zeros((b,heads,dim,128),np.float32),layout,np) if initial is None else initial
    need(len(case['caches'])==length and case['output'].dtype==torch.float16
         and tuple(case['output'].shape)==(b,length,heads,dim),'Controlled token/readout geometry differs')
    actual_output=case['output'].numpy();buffers_verified=0
    for token in range(length):
        previous=decode_np(packed,layout,np)
        bv=np.take_along_axis(B[:,token][:,group],indices,axis=-1)
        cv=np.take_along_axis(C[:,token][:,group],indices,axis=-1)
        # dt32 and decay1 make these controlled operations exact before sum.
        updated=previous+(bv*np.float32(32))[:,:,None,:]*x[:,token,:,:,None]
        out=np.sum(updated[...,:n8]*cv[:,:,None,:n8],axis=-1,dtype=np.float32)
        out=out+np.sum(updated[...,n8:n8+n4]*cv[:,:,None,n8:n8+n4],axis=-1,dtype=np.float32)
        out=out+(np.sum(updated[...,n8+n4:]*cv[:,:,None,n8+n4:],axis=-1,dtype=np.float32)+x[:,token]*D[None,:,None])
        need(np.array_equal(out.astype(np.float16),actual_output[:,token]),'Independent sparse readout differs: '+name+'/'+str(token))
        packed=quantize_np(updated,layout,np);actual=case['caches'][token]
        need(set(actual)==set(packed),'Controlled packed field inventory differs')
        for key,expected in packed.items():
            value=actual[key]
            need(value.device.type=='cpu' and value.numpy().dtype==expected.dtype and tuple(value.shape)==expected.shape
                 and value.contiguous().numpy().tobytes()==expected.tobytes(),'Independent packed codes/scales differ: '+name+'/'+str(token)+'/'+key)
            buffers_verified+=1
    expected_hashes={key:__import__('hashlib').sha256(value.tobytes()).hexdigest() for key,value in packed.items()}
    need(row.get('tokens')==length and row.get('readout_exact') is True and row.get('packed_every_token_exact') is True
         and row.get('decoded_every_token_exact') is True and row.get('max_readout_error')==0.
         and row.get('final_actual_sha256')==expected_hashes,'Controlled exactness receipt differs')
    return dict(name=name,layout=layout,tokens=length,packed_buffers_recomputed=buffers_verified,
        independently_exact_raw_codes_scales_and_readout=True)


def audit_kernel_receipt(path,torch,np):
    receipt=read(path)
    need(receipt.get('format')=='MAMBA2_STATE_PPL_V10_CODEC_CHECK_V1' and receipt.get('complete') is True
         and receipt.get('passed') is True and receipt.get('mode')=='gpu' and receipt.get('cuda_initialized') is True
         and receipt.get('protocol_sha256')==PROTOCOL and 'error' not in receipt,'Actual completed GPU layout receipt required')
    inventory={'scripts/state_ppl_codec_v10.py','scripts/check_state_ppl_codec_v10.py','scripts/state_ppl_codec_v6.py',
        'mamba2_recall/state_codec.py','mamba2_recall/state_quant.py'}
    hash_inventory(receipt['code_sha256'],inventory,'v10 kernel fixture sources')
    expected_names=expected_check_names();need(len(expected_names)==174 and len(set(expected_names))==174,'Auditor check inventory inconsistent')
    need([r['name'] for r in receipt['checks']]==expected_names and all(r.get('pass') is True for r in receipt['checks']),
         'Actual GPU fixture inventory/order/pass differs')
    rows={r['name']:r for r in receipt['checks']}
    need(rows['fixed-layout-grid']['layouts']=={key:list(value) for key,value in LAYOUTS.items()},'Fixture layout grid differs')
    criteria=dict(baseline='Direct unchanged v6 stored_scale clip1 delegation; bitwise output/cache and unchanged cache_breakdown dictionary',
        controlled='Exact FP16 readout plus raw packed bytes/FP16 scales/decoded carry every token in one-step,65-token,poisoned-neighbor and dead-impulse fixtures',
        random='Exact full/segmented/tokenwise output and final cache; no CPU transcendental/reduction equality assertion',
        storage='Exact unpadded lo/hi/q4 widths plus two FP16 scales,52B/row;56layer actual allocation28499968B; zero resident layout tensors',
        failure='Nonfinite scales/readout rejected; malformed padded allocations rejected')
    need(receipt['criteria']==criteria,'Frozen codec acceptance criteria differ')
    for limit in (127,7):
        values=[];scales=[]
        for scale in (0.,2.**-24,2.**-14,.037933349609375,1.,65504.):
            for quotient in (-limit-.5,-3.5,-1.5,-.5,.5,1.5,3.5,limit+.5):
                center=np.float32(scale*quotient)
                for value in (np.nextafter(center,np.float32(-np.inf)),center,np.nextafter(center,np.float32(np.inf))):
                    values.append(value);scales.append(scale)
        values.extend((0.,1.,-1.));scales.extend((0.,0.,0.))
        values=np.asarray(values,dtype=np.float32);scales=np.asarray(scales,dtype=np.float16)
        q=np.where(scales>0,np.clip(signed_round_np(values/np.where(scales>0,scales.astype(np.float32),np.float32(1)),np),-limit,limit),0).astype(np.int32)
        actual=rows['GPU-shared-rounding-boundaries-'+str(limit)]
        need(rows['CPU-signed-half-boundaries-'+str(limit)]['values']==len(values)==147
             and actual['actual_codes']==q.tolist() and actual['values_fp32_bits']==values.view(np.uint32).tolist()
             and actual['scales_fp16_bits']==scales.view(np.uint16).tolist(),'Independent signed half-boundary evidence differs')
    storage={}
    for layout in LAYOUTS:
        cpu=rows['CPU-actual-unpadded-storage-'+layout];g=independent_geometry(layout,batch=2,heads=6,dim=19,layers=1)
        need(cpu['shapes']=={k:v['shape'] for k,v in g['tensors'].items()}
             and cpu['storage_bytes']=={k:v['bytes'] for k,v in g['tensors'].items()},'CPU unpadded buffer evidence differs')
        prod=rows['actual-production-allocation-'+layout]
        storage[layout]=audit_storage(prod['storage_descriptor'],layout);v6.audit_cache(prod['cache'],spec_for(layout),0)
        for geometry in GEOMETRIES:
            b,L,H,P,G=geometry
            need(rows['random-actual-bytes-'+str(geometry)+'-'+layout]['actual_bytes']==b*H*P*52,'Random physical storage differs')
    evidence_info=receipt['evidence'];evidence_path=Path(evidence_info['file'])
    if not evidence_path.is_absolute():evidence_path=ROOT/evidence_path
    need(sha(evidence_path)==evidence_info['sha256'] and evidence_path.stat().st_size==evidence_info['bytes'],'Controlled raw evidence identity differs')
    evidence=torch.load(evidence_path,map_location='cpu',weights_only=True)
    need(evidence.get('format')=='MAMBA2_STATE_PPL_V10_CODEC_EVIDENCE_V1' and evidence.get('protocol_sha256')==PROTOCOL
         and evidence.get('code_sha256')==receipt['code_sha256'],'Controlled raw evidence provenance differs')
    expected_cases=[name[len('GPU-oracle-exact-'):] for name in expected_names if name.startswith('GPU-oracle-exact-')]
    need(len(evidence['cases'])==evidence_info['cases']==len(expected_cases)==44
         and [c['name'] for c in evidence['cases']]==expected_cases,'Controlled44-case population differs')
    controlled=[audit_controlled_case(case,rows['GPU-oracle-exact-'+case['name']],torch,np) for case in evidence['cases']]
    return dict(complete=True,sha256=sha(path),checks=174,evidence_sha256=evidence_info['sha256'],controlled=controlled,
        controlled_cases_independently_recomputed=44,production_storage=storage,
        scalar_boundary_values_recomputed=294,scope='Independent NumPy recomputation of controlled fixture inputs/readouts/raw codes/scales; random GPU partition results are source-bound receipts')


if __name__=='__main__':main()
