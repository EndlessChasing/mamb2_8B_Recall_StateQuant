#!/usr/bin/env python3
"""TRAIN-selected max-preserving INT4 codebooks; unadapted full PPL target<8.25."""
from __future__ import annotations
import argparse
import contextlib
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from mamba2_recall import runtime,resurface_data as data,resurface_native as native
from mamba2_recall.calibration import load_wikitext_tokens
from mamba2_recall.evaluation import ppl_windows
from prepare_quant_first import load_train_tokens,TRAIN_SHA
from prepare_state_first_v5 import KINDS,CACHE_BYTES,need,read_json,tensor_sha,check_table,write_payload
from evaluate_quant_first import FrozenBase,VALIDATION_TOKENS_SHA
from evaluate_resurface_more import pin_replay_backend,check_replay_backend
from run_statequant import save_json
import run_state_ppl_v6 as v6
from run_state_ppl_v6 import (CandidateInvalid,no_adapter_hooks,finite_exp,cache_identity,
    finite_cache,require_cache,window_identity,check_ppl,valid_candidate,check_restoration,compare_ppl)
PROTOCOL_SHA='84ffdce1d9e5d5dbac2b6996072f0db09bd1bf79fd3c55dad8b8f35973cd5d14'
BOOKS=('uniform','mild','quadratic','fp4like')
IDS=tuple(k+'__'+b for k in KINDS for b in BOOKS)
BASELINE='preserve_int8__uniform'
SCREEN_ARMS=(BASELINE,*(n for n in IDS if n!=BASELINE),'restored_baseline')
V6_SELECTED='098930d1af5e5821b277640d236f7607c6428b48d36f7117e2ea4aa87e656303'
V6_REPORT='c3525d74ba0ee35e0b9d83a029e645123d14e90f5ca32b164931d24ec8359253'
V6_COMPARISON='c47e381e02edd6e33ba2b0a85a4f9c65d702cc7d3d93dacfe7546d409e6612d4'
FORMAT='MAMBA2_STATE_PPL_V7_EVAL_V1'
COMPARE='MAMBA2_STATE_PPL_V7_COMPARISON_V1'
CALIBRATION='MAMBA2_STATE_PPL_V7_CALIBRATION_V1'

def code_hashes():
    result=v6.code_hashes()
    for rel in ('scripts/run_state_ppl_v7.py','scripts/state_ppl_codec_v7.py',
                'scripts/check_state_ppl_codec_v7.py','scripts/run_state_ppl_v6.py','docs/STATE_PPL_V7_PROTOCOL.md'):
        result[rel]=data.sha_file(ROOT/rel)
    return dict(sorted(result.items()))

def candidate_spec(name):
    need(name in IDS,'Unknown fixed candidate')
    kind,book=name.split('__')
    return dict(candidate_id=name,candidate_name=kind,variant=book,codebook=book,
                scale_mode='stored_scale',int4_clip=1.,diagnostic=None,deployable=True)

def make_execution(model,table,spec):
    from state_ppl_codec_v7 import StatePPLQuantV7
    return StatePPLQuantV7(model,table,codebook=spec['codebook'])
@contextlib.contextmanager
def guarded_execution(model, table, spec):
    with make_execution(model, table, spec) as execution:
        digest = tensor_sha(table) if table is not None else None
        try:
            yield execution
        finally:
            need((execution.permutations is None if digest is None else
                  native.tensor_hash(execution.permutations) == digest),
                 'Actual GPU permutation table mutated during execution')

@torch.inference_mode()
def evaluate(model, table, spec, windows, path, common):
    result = dict(common, **spec, complete=False, ppl=dict(windows=[]))
    started = time.time()
    save_json(path, result)
    torch.cuda.reset_peak_memory_stats()
    with guarded_execution(model, table, spec) as execution:
        probe = windows[0][1][:128].cuda()[None]
        first = execution.backbone(probe, reset=True)
        if not bool(torch.isfinite(first).all()):
            raise CandidateInvalid('Nonfinite first128-token hidden')
        zero_scales = finite_cache(execution)
        cache0 = cache_identity(execution)
        hidden_sha = native.tensor_hash(first)
        second = execution.backbone(probe, reset=True)
        if not bool(torch.isfinite(second).all()):
            raise CandidateInvalid('Nonfinite repeated128-token hidden')
        finite_cache(execution)
        if not torch.equal(first, second) or cache0 != cache_identity(execution):
            raise RuntimeError('Repeated-reset128-token hidden/cache differed')
        result['repeated_reset_probe'] = dict(tokens=128, hidden_sha256=hidden_sha,
            hidden_and_cache_exact=True, cache_tensor_sha256=cache0,
            token_sha256_int64le=runtime.token_digest(probe.cpu().numpy()),
            cache=require_cache(execution, spec))
        del first, second, probe, cache0
        total = 0.
        count = 0
        for index, (start, window) in enumerate(windows):
            tokens = window.cuda()
            hidden = execution.backbone(tokens[:-1][None], reset=True)
            if not bool(torch.isfinite(hidden).all()):
                raise CandidateInvalid('Nonfinite PPL hidden')
            zero_scales += finite_cache(execution)
            loss_sum = 0.
            for pos in range(0, hidden.shape[1], 64):
                end = min(pos+64, hidden.shape[1])
                logits = model.lm_head(hidden[:,pos:end]).float()
                loss = F.cross_entropy(logits.reshape(-1,256000), tokens[pos+1:end+1], reduction='sum')
                loss_sum += float(loss)
                del logits, loss
            if not math.isfinite(loss_sum):
                raise CandidateInvalid('Nonfinite PPL loss')
            targets = len(window)-1
            total += loss_sum
            count += targets
            result['ppl']['windows'].append(dict(start=start, target_tokens=targets,
                token_sha256_int64le=runtime.token_digest(window.numpy()), nll=loss_sum,
                ppl=finite_exp(loss_sum/targets)))
            result['ppl'].update(nll=total, target_tokens=count, ppl=finite_exp(total/count))
            result['cache'] = require_cache(execution, spec)
            save_json(path, result)
            if index == 0 or (index+1) % 8 == 0 or index+1 == len(windows):
                print(f'[{common["arm"]} PPL] {index+1}/{len(windows)} ppl={result["ppl"]["ppl"]:.6f}', flush=True)
            del hidden, tokens
        result['persistent_float_finite_checks_passed'] = True
        result['zero_scale_observations'] = zero_scales
        result['zero_scale_scope'] = 'Final-cache observations of stored s4/s8; includes true zeros and scale underflow, not unique underflow events'
        result['gpu_memory'] = runtime.gpu_memory_receipt()
    result.update(complete=True, runtime_table_unchanged=True, elapsed_seconds=time.time()-started)
    return result


def validate_kernel(path):
    r=read_json(path)
    need(r.get('format')=='MAMBA2_STATE_PPL_V7_CODEC_CHECK_V1' and r.get('complete') is True
        and r.get('passed') is True and r.get('mode')=='gpu' and r.get('cuda_initialized') is True
        and 'error' not in r,'Passing actual GPU kernel receipt required')
    need(r.get('protocol_sha256')==PROTOCOL_SHA and r.get('checks')
        and all(c.get('pass') is True for c in r['checks']),'Protocol or kernel checks differ')
    need({'scripts/state_ppl_codec_v7.py','scripts/check_state_ppl_codec_v7.py'}<=set(r['code_sha256']),
         'Incomplete kernel source inventory')
    for name,digest in r['code_sha256'].items():
        need(data.sha_file(ROOT/name)==digest,'Kernel evidence source changed: '+name)
    return r

def load_inputs(args):
    need(data.sha_file(ROOT/'docs/STATE_PPL_V7_PROTOCOL.md')==PROTOCOL_SHA,'Frozen protocol differs')
    validate_kernel(args.kernel_checks)
    candidates,binding,old_windows,_=v6.load_inputs(args.candidates,args.v5_selected_calibration,
                                                 args.v6_kernel_checks,args.prose_tokens)
    need(data.sha_file(args.v6_selected_calibration)==V6_SELECTED,'Pinned v6 selection differs')
    selected,_,_=v6.load_selected_calibration(args.v6_selected_calibration,candidates,binding,old_windows)
    need(selected['selected_id']=='preserve_int8__stored_scale','v6 baseline policy differs')
    train=load_train_tokens(args.prose_tokens)
    windows=[(i*2048,train[i].clone()) for i in range(80,112)]
    need(all(len(w)==2048 for _,w in windows),'Expected complete TRAIN windows')
    binding=dict(v6_input_binding=binding,v6_selected_sha256=V6_SELECTED,
        kernel_checks_sha256=data.sha_file(args.kernel_checks),protocol_sha256=PROTOCOL_SHA,
        source_sha256=runtime.SOURCE_CHECKPOINT_SHA256,tokenizer_sha256=runtime.TOKENIZER_SHA256,
        train_sha256=TRAIN_SHA,candidates_sha256=data.sha_file(args.candidates))
    return candidates,binding,windows

def choose_candidate(rows):
    need(valid_candidate(rows[BASELINE]),'Baseline must remain valid')
    valid=[name for name in IDS if valid_candidate(rows[name])]
    winner=min(valid,key=lambda n:(rows[n]['ppl']['ppl'],n!=BASELINE,IDS.index(n)))
    kind,book=winner.split('__')
    return dict(selected_id=winner,selected_kind=kind,selected_codebook=book,
        baseline_id=BASELINE,valid=valid,excluded=[n for n in IDS if n not in valid],
        baseline_wins=winner==BASELINE,adapter_used=False,heldout_used=False,mk_used=False,
        rule='Minimum complete finite exact-budget TRAIN PPL; exact ties baseline then fixed candidate order')

def export_selection(out,candidates,binding,comparison):
    s=comparison['selection'];table=candidates['tables'][s['selected_kind']].clone()
    payload=dict(format=CALIBRATION,protocol_sha256=PROTOCOL_SHA,input_binding=binding,
        **s,permutations=table,table_sha256=tensor_sha(table),code_hashes=code_hashes(),
        selection_report_sha256=data.sha_file(out/'screen_comparison.json'))
    path=out/'selected_calibration.pt';write_payload(path,payload)
    receipt={k:v for k,v in payload.items() if k!='permutations'}
    receipt.update(complete=True,file=path.name,sha256=data.sha_file(path),bytes=path.stat().st_size)
    save_json(path.with_suffix('.json'),receipt)
    return receipt

def load_selection(path,candidates,binding,windows):
    payload=torch.load(path,map_location='cpu',weights_only=True);receipt=read_json(path.with_suffix('.json'))
    need(receipt.get('complete') is True and receipt['sha256']==data.sha_file(path)
         and receipt['bytes']==path.stat().st_size,'Selected artifact receipt differs')
    need(payload.get('format')==receipt.get('format')==CALIBRATION,'Selection format differs')
    for key,value in payload.items():
        if key!='permutations':need(receipt.get(key)==value,'Selected payload/receipt differs: '+key)
    need(payload['protocol_sha256']==PROTOCOL_SHA and payload['input_binding']==binding
         and payload['code_hashes']==code_hashes(),'Selection source/input binding differs')
    comp_path=path.parent/'screen_comparison.json'
    need(data.sha_file(comp_path)==payload['selection_report_sha256'],'Selection comparison changed')
    comp=read_json(comp_path)
    need(comp.get('complete') is True and comp.get('format')==COMPARE and comp.get('stage')=='screen'
         and comp.get('input_binding')==binding and comp.get('code_hashes')==code_hashes()
         and comp.get('adapter_loaded') is False and comp.get('mk_used') is False
         and comp.get('heldout_used') is False,'Screen provenance differs')
    need(tuple(comp['report_sha256'])==SCREEN_ARMS,'Screen arm population/order differs')
    identities=window_identity(windows);rows={}
    for arm in SCREEN_ARMS:
        p=path.parent/('screen_'+arm+'.json');need(data.sha_file(p)==comp['report_sha256'][arm],'Screen report hash differs')
        r=read_json(p);spec=candidate_spec(BASELINE if arm=='restored_baseline' else arm)
        need(r.get('format')==FORMAT and r.get('stage')=='screen' and r.get('arm')==arm
             and r.get('input_binding')==binding and r.get('code_hashes')==code_hashes()
             and r.get('adapter_loaded') is False and r.get('heldout_used') is False
             and r.get('mk_used') is False and all(r.get(k)==v for k,v in spec.items()),'Screen row binding differs')
        table=candidates['tables'][spec['candidate_name']]
        need(r['candidate_table_sha256']==tensor_sha(table),'Screen table hash differs')
        if r.get('complete'):
            check_ppl(r,identities);need(valid_candidate(r),'Screen integrity guard failed')
        else:need(arm in IDS and arm!=BASELINE and r.get('error_type')=='CandidateInvalid'
                  and r.get('excluded_from_selection') is True,'Unrecognized incomplete arm')
        rows[arm]=r
    selection=choose_candidate(rows)
    need(comp['selection']==selection and all(payload.get(k)==v for k,v in selection.items()),'Selection rule differs')
    need(comp['restoration']==check_restoration(rows[BASELINE],rows['restored_baseline']),'Screen restoration differs')
    check_table(payload['permutations'])
    need(torch.equal(payload['permutations'],candidates['tables'][selection['selected_kind']])
         and payload['table_sha256']==tensor_sha(payload['permutations']),'Selected table differs')
    return payload,receipt,comp

def archive_replay(archived,current):
    need(archived.get('complete') is True and current.get('complete') is True
         and archived['ppl']==current['ppl'],'Archived v6 raw NLL/PPL differs')
    a,b=archived['repeated_reset_probe'],current['repeated_reset_probe']
    for key in ('tokens','hidden_sha256','hidden_and_cache_exact','cache_tensor_sha256','token_sha256_int64le'):
        need(a[key]==b[key],'Archived v6 reset probe differs: '+key)
    need(archived['cache']['total_bytes']==current['cache']['total_bytes']==CACHE_BYTES,'Archived cache differs')
    return dict(complete=True,per_window_nll_exact=True,ppl_windows_repeated=130,target_tokens=264764,
                reset_hidden_exact=True,reset_cache_exact=True,cache_bytes_exact=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('screen','full'),required=True)
    for name in ('source-dir','candidates','v5-selected-calibration','v6-selected-calibration',
                 'prose-tokens','v6-kernel-checks','kernel-checks','out'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('selected-calibration','parent-report','parent-comparison','s16-report'):
        p.add_argument('--'+name,type=Path)
    args=p.parse_args();need(not args.out.exists(),'Fresh output directory required')
    candidates,binding,train_windows=load_inputs(args)
    selected=archive=s16=None
    if args.stage=='screen':
        need(all(getattr(args,n) is None for n in ('selected_calibration','parent_report','parent_comparison','s16_report')),
             'Screen accepts no heldout/selected inputs')
        windows=train_windows;dataset=dict(split='train',file_sha256=TRAIN_SHA,rows=list(range(80,112)),
                                          tokens_per_row=2048,windows=32,target_tokens=65504)
        arms=[(a,candidate_spec(BASELINE if a=='restored_baseline' else a)) for a in SCREEN_ARMS]
    else:
        need(all(getattr(args,n) is not None for n in ('selected_calibration','parent_report','parent_comparison','s16_report')),
             'Full requires frozen selection and archive references')
        selected,_,_=load_selection(args.selected_calibration,candidates,binding,train_windows)
        need(not selected['baseline_wins'],'Baseline winner does not advance to redundant full evaluation')
        need(data.sha_file(args.parent_report)==V6_REPORT and data.sha_file(args.parent_comparison)==V6_COMPARISON,
             'Pinned v6 archive differs')
        historical=read_json(args.parent_comparison)
        need(historical['report_sha256']['selected']==V6_REPORT,'Archive comparison binding differs')
        need(data.sha_file(args.s16_report)==v6.S16_REPORT_SHA,'Original S16 reference differs')
        archive=read_json(args.parent_report);s16=read_json(args.s16_report)
        arms=[('v6_baseline',candidate_spec(BASELINE)),('selected',candidate_spec(selected['selected_id'])),
              ('restored_baseline',candidate_spec(BASELINE))]
    tokenizer=runtime.SentencePieceTokenizer(args.source_dir)
    need(tokenizer.sha256==runtime.TOKENIZER_SHA256,'Tokenizer differs')
    if args.stage=='full':
        ids,dataset=load_wikitext_tokens(tokenizer,'validation');windows=ppl_windows(ids,2048)
        need(len(windows)==130 and sum(len(w)-1 for _,w in windows)==264764
             and dataset['token_stream_sha256_int64le']==VALIDATION_TOKENS_SHA,'Validation population differs')
    torch.set_num_threads(8);torch.manual_seed(20260929);torch.cuda.manual_seed_all(20260929)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');policy=pin_replay_backend()
    args.out.mkdir(parents=True);model=runtime.load_source_model(args.source_dir);frozen=FrozenBase(model)
    need(no_adapter_hooks(model),'Unexpected source adapter hooks')
    common=dict(format=FORMAT,stage=args.stage,protocol_sha256=PROTOCOL_SHA,input_binding=binding,
        dataset=dataset,environment=runtime.environment_receipt(),code_hashes=code_hashes(),backend_policy=policy,
        adapter_loaded=False,adapter_sha256=None,mk_used=False,heldout_used=args.stage=='full',heldout_used_for_selection=False,
        selected_calibration_sha256=data.sha_file(args.selected_calibration) if selected else None,
        parent_report_sha256=V6_REPORT if archive else None,parent_comparison_sha256=V6_COMPARISON if archive else None,
        s16_report_sha256=v6.S16_REPORT_SHA if s16 else None)
    rows={};started=time.time();identities=window_identity(windows);parent_replay=None
    for arm,spec in arms:
        print('[state-ppl-v7 arm] '+arm,flush=True);need(no_adapter_hooks(model),'Adapter leaked');frozen.check()
        table=candidates['tables'][spec['candidate_name']];digest=tensor_sha(table)
        path=args.out/(args.stage+'_'+arm+'.json');ac=dict(common,arm=arm,candidate_table_sha256=digest)
        try:result=evaluate(model,table,spec,windows,path,ac)
        except CandidateInvalid as exc:
            result=read_json(path);result.update(complete=False,error=repr(exc),error_type='CandidateInvalid')
            if args.stage!='screen' or arm not in IDS or arm==BASELINE:
                result['fatal_failure']=True;save_json(path,result);raise
            result['excluded_from_selection']=True
        except Exception as exc:
            result=read_json(path) if path.exists() else dict(ac,**spec)
            result.update(complete=False,error=repr(exc),error_type=type(exc).__name__,fatal_failure=True)
            save_json(path,result);raise
        result['frozen_source']=frozen.check();result['backend_policy_check']=check_replay_backend(policy)
        need(tensor_sha(table)==digest and no_adapter_hooks(model),'CPU table or hooks changed')
        result['candidate_table_unchanged']=True
        if result['complete']:check_ppl(result,identities)
        save_json(path,result);rows[arm]=result
        if archive and arm=='v6_baseline':
            parent_replay=archive_replay(archive,result);save_json(args.out/'full_parent_replay.json',parent_replay)
    baseline=BASELINE if args.stage=='screen' else 'v6_baseline'
    restoration=check_restoration(rows[baseline],rows['restored_baseline'])
    save_json(args.out/(args.stage+'_restoration.json'),restoration)
    result=dict(format=COMPARE,complete=True,stage=args.stage,protocol_sha256=PROTOCOL_SHA,input_binding=binding,
        code_hashes=code_hashes(),backend_policy=policy,restoration=restoration,adapter_loaded=False,adapter_sha256=None,
        mk_used=False,heldout_used=args.stage=='full',heldout_used_for_selection=False,
        report_sha256={a:data.sha_file(args.out/(args.stage+'_'+a+'.json')) for a in rows},elapsed_seconds=time.time()-started)
    if args.stage=='screen':
        selection=choose_candidate(rows);result.update(selection=selection,
            selected_vs_baseline=compare_ppl(rows[BASELINE],rows[selection['selected_id']]))
    else:
        checks=dict(ppl_strictly_below_8p25=rows['selected']['ppl']['ppl']<8.25,
                    cache_same_budget=all(r['cache']['total_bytes']==CACHE_BYTES for r in rows.values()))
        result.update(selected_id=selected['selected_id'],selected_kind=selected['selected_kind'],
            selected_codebook=selected['selected_codebook'],selected_calibration_sha256=data.sha_file(args.selected_calibration),
            selection_report_sha256=selected['selection_report_sha256'],parent_report_sha256=V6_REPORT,
            parent_comparison_sha256=V6_COMPARISON,s16_report_sha256=v6.S16_REPORT_SHA,parent_replay=parent_replay,
            comparison=compare_ppl(rows['v6_baseline'],rows['selected']),original_s16_comparison=compare_ppl(s16,rows['selected']),
            target_checks=checks,target_pass=all(checks.values()),resurface_trained=False,mk_evaluated=False)
    save_json(args.out/(args.stage+'_comparison.json'),result)
    if args.stage=='screen':
        export_selection(args.out,candidates,binding,result)
        load_selection(args.out/'selected_calibration.pt',candidates,binding,train_windows)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
