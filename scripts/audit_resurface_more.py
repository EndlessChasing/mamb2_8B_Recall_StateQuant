#!/usr/bin/env python3
"""CPU-only independent continuation provenance, scoring and arithmetic audit.

Reconstructs TRAIN schedules, scaler/optimizer-step transitions and data hashes;
decodes recorded greedy IDs and recomputes metrics. Does not re-run GPU logits,
recompute optimizer moments from gradients, or byte-hash post-fit base weights.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audit_quant_first import (sha, need, finite, close, tokhash, read, check_cache,
    audit_training as audit_parent_training, audit_evaluation as audit_parent_evaluation,
    recompute_pair, PROTOCOL as PARENT_PROTOCOL, SOURCE, TOKENIZER, VALIDATION_TOKENS, ADAPTER_FORMAT)

PROTOCOL = '4c2c47aa00936ded52cf7b337126f1cce9556da7021e0a75c6e9df83f4949330'
PARENT_REPORT = 'e5a77d86cf2fb0e2389247e3cb325f74e89957861a6043e92a891d6d402ae359'
PARENT_CHECKPOINT = 'bc548dd427d114098048fa1863f8e602c095dc2d9fde56348795628ae8e2c78f'
PARENT_ADAPTER = '7144b88265a797ef935b1f94845395abec5532d8dc5b2a4a8fa0a8e07ce205f0'
CALIBRATION = 'c366cd577967e64635f1dd960237dc1c0024d7413685ce0b4c4e40da869a7023'
ARCHIVE = {
    'source_s16': '52f82f83258a2fa3160ea14585f1bd69e1d68d60d8f179d1f0987c636f6546ba',
    'source_sq3p25': 'd8ec7b1ca239b385e444cfbec6c34f36728a459c6973c7f92690951e0447bc0d',
    'resurface_sq3p25': '224a8d1201761e44072bd37211ec0c916cc0872095042bf26cde49d8a349b20b',
}
ARMS = ('parent_resurface_sq3p25','continued_resurface_sq3p25','restored_parent_resurface_sq3p25')
INITIAL_SCALER = dict(scale=16.,growth_factor=2.,backoff_factor=.5,growth_interval=2000,_growth_tracker=1536)


def factor(r):
    return .01+.09*(1+math.cos(math.pi*r/3071))/2


def shapes():
    return {f'layer{i}.{name}':shape for i in range(56) for name,shape in
        [('V_read',(128,128)),('g_read',(128,)),('router_w',(4096,)),('router_b',())]}


def expected_names():
    return [[n for n in shapes() if n.endswith(('.V_read','.g_read'))],
            [n for n in shapes() if n.endswith(('.router_w','.router_b'))]]


def audit_masters(values, torch):
    need(set(values) == set(shapes()), 'FP32 master inventory differs')
    for n, shape in shapes().items():
        v = values[n]
        need(isinstance(v,torch.Tensor) and v.dtype == torch.float32 and tuple(v.shape) == shape
             and bool(torch.isfinite(v).all()) and bool(torch.isfinite(v.half()).all()),
             'FP32 master invalid: '+n)


def audit_optimizer(value, masters, updates, lr_factor, torch):
    groups, states = value['param_groups'],value['state']
    need(len(groups) == 2 and len(states) == 224, 'Adam group/state count differs')
    seen = []
    for g,names,base in zip(groups,expected_names(),(1e-4,3e-4)):
        need(len(g['params']) == 112 and g['base_lr'] == base and tuple(g['betas']) == (.9,.999)
             and g['eps'] == 1e-8 and g['weight_decay'] == 0. and g['amsgrad'] is False
             and g['maximize'] is False and g['differentiable'] is False, 'Adam recipe changed')
        close(g['lr'],base*lr_factor,'Adam rate differs',tol=1e-14)
        for index,name in zip(g['params'],names):
            seen.append(index); state = states[index]
            need(set(state) == {'step','exp_avg','exp_avg_sq'}
                 and state['step'].numel() == 1 and state['step'].item() == updates,
                 'Adam per-parameter successful update count differs')
            for key in ('exp_avg','exp_avg_sq'):
                t = state[key]
                need(t.dtype == torch.float32 and t.shape == masters[name].shape and bool(torch.isfinite(t).all()),
                     'Adam moment dtype/shape/finiteness differs: '+name)
            need(bool((state['exp_avg_sq'] >= 0).all()),'Adam variance negative')
    need(len(set(seen)) == 224 and set(seen) == set(states), 'Missing/duplicate Adam parameter identity')


def audit_resume(report, windows):
    proof = report['parent_resume_check']
    need(all(proof.get(k) is True for k in ('masters_exact','optimizer_exact','scaler_exact',
        'master_fp16_cast_equals_parent_export','packed_training_forward_bitwise_equal'))
        and proof['optimizer_steps'] == 1536 and proof['optimizer_states'] == proof['master_tensors'] == 224
        and proof['probe_tokens'] == 128 and proof['probe_token_sha256'] == tokhash(windows[0][:128])
        and proof['scaler'] == INITIAL_SCALER, 'Exact parent state/forward restoration proof differs')
    check_cache(proof['cache'],'sq3p25',128)


def audit_history(report, windows, train_cases, tokenizer, torch, limit):
    order = []
    for seed in (2026092804,2026092805):
        order += torch.randperm(1536,generator=torch.Generator().manual_seed(seed)).tolist()
    prose_order = torch.randperm(448,generator=torch.Generator().manual_seed(20260928)).tolist()
    need(report['schedule'] == dict(numeric_seeds=[2026092804,2026092805],
        numeric_order_sha256_int64le=tokhash(order),prose_global_start=1536,additional_updates=3072),
        'Continuation TRAIN schedule identity differs')
    need(len(report['history']) == report['additional_attempts'] == report['attempts'], 'Attempt history incomplete')
    successful=overflows=0; state=dict(INITIAL_SCALER); checkpoints={}
    for attempt,row in enumerate(report['history'],1):
        need(successful < limit and type(row['overflow']) is bool, 'Invalid continuation overflow/status')
        r=successful; j=1536+r; case=train_cases[order[r]]
        prompt=tokenizer.encode(case['prompt']); full=tokenizer.encode(case['prompt']+' '+case['answer'])
        need(full[:len(prompt)] == prompt,'Unstable TRAIN answer prefix')
        need(row['attempt'] == attempt and row['additional_update_index'] == r
             and row['schedule_entry'] == order[r] and row['case_id'] == case['id']
             and row['global_prose_index'] == j and row['prose_window'] == prose_order[j%448]
             and row['prose_start'] == 512*((j//448)%4)
             and row['answer_targets'] == len(full)-len(prompt), 'TRAIN schedule/answer count differs')
        need(row['scaler_before'] == state and row['loss_scale_before'] == state['scale'],
             'GradScaler continuity before attempt differs')
        close(row['lr_factor'],factor(r),'Cosine tail factor differs',tol=1e-14)
        need(len(row['learning_rates']) == 2,'Wrong optimizer rate groups')
        for actual,base in zip(row['learning_rates'],(1e-4,3e-4)):
            close(actual,base*factor(r),'Attempt learning rate differs',tol=1e-14)
        need(all(finite(row[k]) for k in ('mk_ce','prose_ce','prose_kl','prose_closure','seconds')),
             'Nonfinite reported losses/time')
        fields=('all_adapter_gradients_finite','mk_scaled_hidden_gradient_finite','prose_scaled_hidden_gradient_finite')
        need(all(type(row[k]) is bool for k in fields),'Gradient finiteness evidence missing')
        need(row['overflow'] == (not all(row[k] for k in fields)),'Overflow disagrees with finiteness checks')
        state=dict(state)
        if row['overflow']:
            overflows+=1; state['scale']*=.5
            # GradScaler.update(new_scale=...) intentionally leaves growth tracker unchanged.
            need(row['gradient_norm_before_clip'] is None,'Overflow unexpectedly updated optimizer')
        else:
            successful+=1; state['_growth_tracker']+=1
            if state['_growth_tracker'] == state['growth_interval']:
                state['scale']*=state['growth_factor']; state['_growth_tracker']=0
            need(finite(row['gradient_norm_before_clip']) and row['gradient_norm_before_clip']>=0,
                 'Invalid successful-update gradient norm')
        need(row['scaler_after'] == state and row['loss_scale'] == state['scale']
             and row['additional_successful_updates'] == successful
             and row['cumulative_successful_updates'] == row['successful_updates'] == 1536+successful,
             'GradScaler/success accounting differs after attempt')
        if successful and successful%768 == 0 and not row['overflow']:
            checkpoints[successful]=(attempt,dict(state))
    need(successful == limit and overflows <= 8 and report['additional_successful_updates'] == limit
         and report['cumulative_successful_updates'] == report['successful_updates'] == 1536+limit
         and report['additional_overflows'] == overflows and report['final_scaler'] == state,
         'Final continuation step/scaler accounting differs')
    return dict(additional_updates=successful,attempts=len(report['history']),overflows=overflows,
                final_scaler=state),checkpoints


def audit_continuation(args,parent,calibration,windows,tokenizer,torch,data):
    need(sha(ROOT/'docs/RESURFACE_MORE_PROTOCOL.md') == PROTOCOL
         and sha(args.parent_training_report) == PARENT_REPORT
         and sha(args.parent_checkpoint) == PARENT_CHECKPOINT and sha(args.calibration) == CALIBRATION,
         'Frozen parent/continuation inputs changed')
    parent_checkpoint=torch.load(args.parent_checkpoint,map_location='cpu',weights_only=True)
    need(parent_checkpoint['format']=='MAMBA2_SQ_FIRST_RESURFACE_CHECKPOINT_V1'
         and parent_checkpoint['binding']==parent['binding'] and parent_checkpoint['successful_updates']==1536
         and parent_checkpoint['attempts']==parent['attempts'] and parent_checkpoint['scaler']==INITIAL_SCALER
         and parent['checkpoints'][-1]['sha256']==PARENT_CHECKPOINT,'Parent checkpoint binding differs')
    audit_masters(parent_checkpoint['masters'],torch)
    audit_optimizer(parent_checkpoint['optimizer'],parent_checkpoint['masters'],1536,.1,torch)
    parent_export=torch.load(args.parent_training_report.parent/parent['adapter']['file'],map_location='cpu',weights_only=True)
    need(all(torch.equal(v.half(),parent_export['tensors'][n]) for n,v in parent_checkpoint['masters'].items()),
         'Parent FP32 master cast differs from actual parent FP16 export')
    report=read(args.training_report)
    need(report['format']=='MAMBA2_SQ_MORE_RESURFACE_TRAIN_V1' and report['complete'] is True
         and report['mode']=='formal' and 'error' not in report,'Incomplete formal continuation')
    binding={**parent['binding'],'continuation_protocol_sha256':PROTOCOL,
        'parent_training_report_sha256':PARENT_REPORT,'parent_checkpoint_sha256':PARENT_CHECKPOINT,
        'parent_adapter_sha256':PARENT_ADAPTER,'parent_successful_updates':1536,
        'additional_successful_updates':3072,'successful_updates':4608,
        'initial_adapter':'exact parent FP32 masters/Adam/GradScaler checkpoint continuation'}
    need(report['binding']==binding,'Continuation retained-input/export bindings differ')
    need(report['optimizer_parameter_names']==expected_names(),'Optimizer parameter name order differs')
    parent_info=report['parent']
    need(parent_info['training_report_sha256']==PARENT_REPORT and parent_info['checkpoint_sha256']==PARENT_CHECKPOINT
         and parent_info['adapter_sha256']==PARENT_ADAPTER and parent_info['attempts']==parent['attempts'],
         'Parent receipt differs')
    for relative,digest in report['code_sha256'].items():
        need(sha(ROOT/relative)==digest,'Continuation training source changed: '+relative)
    need(report['source_sha256']==sha(ROOT/'scripts/train_resurface_more.py'),'Trainer source receipt differs')
    audit_resume(report,windows)
    cases=data.generate_cases('train')
    proof,expected_checkpoints=audit_history(report,windows,cases,tokenizer,torch,3072)
    smoke_receipt=report['discarded_smoke']
    smoke_path=args.smoke_report or Path(smoke_receipt['path'])
    need(sha(smoke_path)==smoke_receipt['sha256'] and smoke_receipt['discarded'] is True
         and smoke_receipt['parent_reloaded_for_formal'] is True and smoke_receipt['additional_successful_updates']==1
         and smoke_path.parent.resolve()!=args.training_report.parent.resolve(),'Discarded smoke binding differs')
    smoke=read(smoke_path)
    need(smoke['format']==report['format'] and smoke['complete'] is True and smoke['mode']=='smoke'
         and smoke['binding']==binding and smoke['code_sha256']==report['code_sha256']
         and smoke['checkpoints']==[] and 'adapter' not in smoke and 'error' not in smoke,
         'Smoke was not separate, discarded and identically bound')
    audit_resume(smoke,windows)
    smoke_proof,_=audit_history(smoke,windows,cases,tokenizer,torch,1)
    for item in (report,smoke):
        need(item['frozen_base_check']['identity_version_gradients_unchanged'] is True
             and item['frozen_base_check']['tensors']==507 and item['frozen_base_check']['parameters']==8236999680
             and item['frozen_state_calibration_check'] is True and item['teacher_base_parameters_frozen'] is True,
             'Frozen source/calibration/teacher check failed')
    need(len(report['checkpoints'])==4,'Expected four fixed continuation recovery checkpoints')
    final=None
    for updates,item in zip((768,1536,2304,3072),report['checkpoints']):
        path=args.training_report.parent/Path(item['path']).name
        need(item['file']==path.name==f'checkpoint_{updates:04d}.pt' and sha(path)==item['sha256']
             and path.stat().st_size==item['bytes'] and item['additional_successful_updates']==updates
             and item['cumulative_successful_updates']==1536+updates,'Checkpoint receipt differs')
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        attempts,scaler=expected_checkpoints[updates]
        need(checkpoint['format']=='MAMBA2_SQ_MORE_RESURFACE_CHECKPOINT_V1' and checkpoint['binding']==binding
             and checkpoint['additional_successful_updates']==updates
             and checkpoint['cumulative_successful_updates']==checkpoint['successful_updates']==1536+updates
             and checkpoint['additional_attempts']==checkpoint['attempts']==attempts
             and checkpoint['scaler']==scaler and checkpoint['optimizer_parameter_names']==expected_names(),
             'Checkpoint schedule/step/scaler/binding differs')
        audit_masters(checkpoint['masters'],torch)
        audit_optimizer(checkpoint['optimizer'],checkpoint['masters'],1536+updates,factor(updates-1),torch)
        final=checkpoint
    exported=report['adapter']; need(Path(exported['file']).name==exported['file'],'Unsafe export name')
    path=args.training_report.parent/exported['file']
    need(sha(path)==exported['sha256'] and path.stat().st_size==exported['bytes']
         and exported['roundtrip_bitwise_equal'] is True and exported['gate_mode']=='soft','Adapter receipt differs')
    adapter=torch.load(path,map_location='cpu',weights_only=True)
    need(adapter['format']==ADAPTER_FORMAT and adapter['binding']==binding and adapter['gate_mode']=='soft'
         and adapter['variant']=='post-D native norm-prehook; memoryless cross-head mixing'
         and adapter['geometry']==[dict(width=4096,heads=128,head_dim=64)]*56
         and set(adapter['tensors'])==set(shapes()),'Adapter inventory/header/binding differs')
    for name,value in adapter['tensors'].items():
        need(value.dtype==torch.float16 and tuple(value.shape)==shapes()[name]
             and bool(torch.isfinite(value).all()) and torch.equal(value,final['masters'][name].half())
             and hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()==exported['tensor_sha256'][name],
             'Final master cast/export tensor differs: '+name)
    need(sum(v.numel() for v in adapter['tensors'].values())==exported['parameters']==1154104
         and exported['payload_bytes']==2308208,'FP16 export accounting differs')
    check=report['final_checkpoint_export_check']
    need(all(check[k] is True for k in ('all_master_casts_equal_export','optimizer_exact','scaler_exact'))
         and check['master_tensors']==224 and check['final_checkpoint_sha256']==report['checkpoints'][-1]['sha256'],
         'Final checkpoint/export exactness check differs')
    deploy=report['deployed_export_check']
    need(deploy['packed_training_forward_bitwise_equal'] is True and deploy['probe_tokens']==128
         and deploy['probe_token_sha256']==tokhash(windows[0][:128]) and deploy['cache_unchanged_from_parent'] is True
         and deploy['cache']==report['parent_resume_check']['cache'],'Final deployed parity/cache proof differs')
    check_cache(deploy['cache'],'sq3p25',128)
    proof.update(cumulative_updates=4608,parent_masters_cast_to_export_exact=True,
        all_four_recovery_checkpoint_hashes_verified=True,all_adam_steps_and_scaler_transitions_verified=True,
        final_master_casts_equal_actual_export=True,adapter_sha256=exported['sha256'],
        adapter_parameters=1154104,adapter_tensors=224,discarded_smoke=smoke_proof)
    return report,proof


def audit_arm(result,windows,cases,tokenizer,dataset):
    need(result['complete'] is True and result['stage']=='full' and result['dataset']==dataset,
         'Incomplete/incorrect evaluation dataset')
    need(result['frozen_source']['identity_version_gradients_unchanged'] is True
         and result['frozen_source']['tensors']==507 and result['frozen_source']['parameters']==8236999680
         and result['adapter_content_unchanged'] is True,'Frozen base/adapter check differs')
    need(len(result['ppl']['windows'])==130,'Wrong PPL window count')
    nll=0.; count=0
    for row,(start,ids) in zip(result['ppl']['windows'],windows):
        need(row['start']==start and row['target_tokens']==len(ids)-1
             and row['token_sha256_int64le']==tokhash(ids) and finite(row['nll']) and row['nll']>=0,
             'PPL input/NLL evidence differs')
        close(row['ppl'],math.exp(row['nll']/(len(ids)-1)),'Window PPL arithmetic differs')
        nll+=row['nll'];count+=len(ids)-1
    need(count==result['ppl']['target_tokens']==264764,'Wrong total PPL targets')
    close(result['ppl']['nll'],nll,'Total NLL differs')
    close(result['ppl']['ppl'],math.exp(nll/count),'Total PPL differs')
    rows=result['mk']['rows']
    need(len(rows)==len({r['id'] for r in rows})==768,'MK missing/duplicate cases')
    summary={c:dict(correct=0,count=0) for c in ('normal','target_removed')}
    for row,case in zip(rows,cases):
        ids=tokenizer.encode(case['prompt'])
        need(all(row.get(k)==v for k,v in case.items()) and row['prompt_tokens']==len(ids)
             and row['prompt_token_sha256_int64le']==tokhash(ids),'MK prompt identity differs')
        generated=row['generated_ids']
        need(isinstance(generated,list) and 1<=len(generated)<=12
             and all(type(t) is int and 0<=t<256000 for t in generated)
             and tokenizer.eos_token_id not in generated[:-1]
             and (len(generated)==12 or generated[-1]==tokenizer.eos_token_id),'Invalid greedy sequence/stop')
        text=tokenizer.decode(generated);match=re.search(r'(?<!\d)\d{6}(?!\d)',text)
        predicted=match.group() if match else None;correct=predicted==case['answer']
        need(row['output']==text and row['prediction']==predicted and type(row['correct']) is bool
             and row['correct']==correct,'MK decoded score differs')
        summary[case['condition']]['count']+=1;summary[case['condition']]['correct']+=int(correct)
    for condition,value in summary.items():
        value['accuracy']=value['correct']/value['count']
        need(value['count']==384 and result['mk']['summary'][condition]==value,'MK aggregate differs')
    cache=check_cache(result['cache'],'sq3p25',len(windows[-1][1])-1)
    return dict(ppl=math.exp(nll/count),nll=nll,ppl_target_tokens=count,normal=summary['normal'],
                target_removed=summary['target_removed'],cache_bytes=cache)


def replay(before,after,scope):
    need(before['ppl']['windows']==after['ppl']['windows']
         and before['mk']['rows']==after['mk']['rows'],'Complete parent per-window NLL/generated-ID replay differs')
    for k in ('nll','target_tokens','ppl'):
        need(before['ppl'][k]==after['ppl'][k],'Replay PPL aggregate differs')
    need(before['cache']['total_bytes']==after['cache']['total_bytes'],'Replay cache changed')
    return dict(complete=True,per_window_nll_exact=True,all_generated_ids_exact=True,
        all_decoded_predictions_exact=True,ppl_windows_repeated=130,ppl_target_tokens_repeated=264764,
        mk_cases_repeated=768,scope=scope)



def audit_backend_policy(policy, check):
    """Independently validate the recorded external-kernel execution policy."""
    import inspect
    import importlib.metadata
    from mamba_ssm.ops.triton import layer_norm
    from mamba_ssm.utils import determinism
    from triton.runtime import autotuner
    expected_configs=[dict(kwargs={},num_warps=w,num_stages=3,num_ctas=1,
                           maxnreg=None,pre_hook_is_none=True) for w in (1,2,4,8,16,32)]
    selected=expected_configs[4]
    need(policy['format']=='MAMBA2_ARCHIVED_RMSNORM_REPLAY_POLICY_V1'
         and policy['kernel']=='mamba_ssm.ops.triton.layer_norm._layer_norm_fwd_1pass_kernel'
         and policy['original_configs']==expected_configs and policy['selected_config']==selected
         and policy['singleton_config_bypasses_autotuning'] is True
         and policy['applied_before_first_model_load'] is True
         and policy['candidate_used_for_selection'] is False,
         'RMSNorm replay backend policy differs')
    need(check==dict(singleton_config_unchanged=True,selected_config=selected,best_config=selected),
         'Actual selected singleton RMSNorm configuration differs')
    hashes={module.__name__:sha(inspect.getfile(module)) for module in (layer_norm,determinism,autotuner)}
    need(policy['external_source_sha256']==hashes,'Installed external RMSNorm/helper/autotuner source changed')
    versions={name:importlib.metadata.version(name) for name in ('torch','mamba-ssm','triton')}
    need(policy['package_versions']==versions,'Recorded backend package versions differ')
    import torch
    need(policy['cudnn_version']==torch.backends.cudnn.version(),'Recorded cuDNN version differs')
    expected_flags=dict(tf32_matmul=False,tf32_cudnn=False,fp16_reduced_precision_reduction=True,
        bf16_reduced_precision_reduction=True,cudnn_benchmark=False,cudnn_deterministic=False,
        deterministic_algorithms=False,float32_matmul_precision='highest')
    need(policy['precision_flags']==expected_flags,'Evaluation precision flags differ from replay configuration')
    environment={key:os.environ.get(key) for key in ('MAMBA_DETERMINISTIC','TRITON_CACHE_AUTOTUNING',
        'TRITON_AUTOTUNE_BLOCK_SIZE_M','TRITON_AUTOTUNE_BLOCK_SIZE_N',
        'TRITON_AUTOTUNE_BLOCK_SIZE_K','TRITON_AUTOTUNE_BLOCK_SIZE_DSTATE')}
    need(policy['determinism_environment']==environment,'Relevant backend environment changed since evaluation')
    need(policy['clarification_sha256']==sha(ROOT/'docs/RESURFACE_MORE_BACKEND_REPLAY.md'),
         'Backend execution clarification changed')
    return dict(singleton_rmsnorm_warps=16,external_sources_verified=True,
                policy_receipt_verified=True,clarification_sha256=policy['clarification_sha256'])

def audit_evaluation(args,training,calibration,validation,dataset,tokenizer,data,np):
    windows=[(s,validation[s:min(s+2049,len(validation))].tolist()) for s in range(0,len(validation)-1,2048)]
    need(len(windows)==130 and sum(len(w)-1 for _,w in windows)==264764
         and tokhash(validation.tolist())==VALIDATION_TOKENS,'Frozen validation input differs')
    cases=data.generate_cases('confirm');need(len(cases)==768,'Frozen CONFIRM population differs')
    archived={}
    for arm,digest in ARCHIVE.items():
        path=args.parent_eval_dir/f'full_{arm}.json'
        need(sha(path)==digest,'Archived context hash differs')
        archived[arm]=read(path)
    common=dict(format='MAMBA2_MORE_RESURFACE_EVAL_V1',stage='full',protocol_sha256=PARENT_PROTOCOL,
        continuation_protocol_sha256=PROTOCOL,source_checkpoint_sha256=SOURCE,tokenizer_sha256=TOKENIZER,
        calibration=calibration,calibration_receipt_sha256=sha(args.calibration.with_suffix('.json')),
        parent_training_report_sha256=PARENT_REPORT,parent_checkpoint_sha256=PARENT_CHECKPOINT,
        parent_adapter_sha256=PARENT_ADAPTER,training_report_sha256=sha(args.training_report),
        training_binding=training['binding'],candidate_adapter_sha256=training['adapter']['sha256'],
        archived_report_sha256=ARCHIVE,dataset=dataset)
    outputs={};metrics={};hashes={};backend=None
    for arm in ARMS:
        path=args.eval_dir/f'full_{arm}.json';result=read(path)
        need(all(result.get(k)==v for k,v in common.items()) and result['arm']==arm and result['mode']=='sq3p25'
             and result['adapter_sha256']==(training['adapter']['sha256'] if arm==ARMS[1] else PARENT_ADAPTER),
             'Evaluation provenance/arm differs')
        for relative,digest in result['code_hashes'].items():
            need(sha(ROOT/relative)==digest,'Evaluation code changed: '+relative)
        check_cache(result['cache_allocation_before'],'sq3p25',0)
        check_cache(result['cache_allocation_after'],'sq3p25',0)
        need(result['cache_allocation_before']==result['cache_allocation_after'],
             'Actual pre/post-arm cache allocation differs')
        audit_backend_policy(result['backend_policy'],result['backend_policy_check'])
        if backend is None:
            backend=result['backend_policy']
        need(result['backend_policy']==backend,'Backend replay policy changed across arms')
        metrics[arm]=audit_arm(result,windows,cases,tokenizer,dataset)
        outputs[arm]=result;hashes[arm]=sha(path)
    parent_replay=replay(archived['resurface_sq3p25'],outputs[ARMS[0]],
        'Fresh parent exactly replays archived v2 full Resurface SQ3.25 NLL and generated IDs')
    restoration=replay(outputs[ARMS[0]],outputs[ARMS[2]],
        'Full parent repeated after removing the continued adapter and reinstalling the parent')
    need(read(args.eval_dir/'full_parent_replay.json')==parent_replay
         and read(args.eval_dir/'full_restoration.json')==restoration,'Replay receipts differ')
    comparison=read(args.eval_dir/'full_comparison.json')
    backend_proof=audit_backend_policy(comparison['backend_policy'],comparison['backend_policy_check'])
    need(comparison['backend_policy']==backend,'Comparison backend policy differs from its arms')
    need(comparison['format']=='MAMBA2_MORE_RESURFACE_COMPARISON_V1' and comparison['complete'] is True
         and comparison['stage']=='full' and comparison['protocol_sha256']==PARENT_PROTOCOL
         and comparison['continuation_protocol_sha256']==PROTOCOL
         and comparison['training_report_sha256']==sha(args.training_report)
         and comparison['parent_training_report_sha256']==PARENT_REPORT
         and comparison['parent_checkpoint_sha256']==PARENT_CHECKPOINT
         and comparison['parent_adapter_sha256']==PARENT_ADAPTER and comparison['calibration_sha256']==CALIBRATION
         and comparison['adapter_sha256']==training['adapter']['sha256']
         and comparison['report_sha256']==hashes and comparison['archived_report_sha256']==ARCHIVE
         and comparison['restoration']==restoration and comparison['parent_replay']==parent_replay,
         'Comparison provenance/hash/replay differs')
    pairs={}
    for label,left,right in [('continuation_vs_parent',outputs[ARMS[0]],outputs[ARMS[1]]),
        ('repair_vs_archived_sq_baseline',archived['source_sq3p25'],outputs[ARMS[1]]),
        ('remaining_gap_vs_archived_original_s16',archived['source_s16'],outputs[ARMS[1]])]:
        value=recompute_pair(left,right,np)
        need(comparison[label]==value,'Independent paired/bootstrap calculation differs: '+label)
        pairs[label]=value
    change=pairs['continuation_vs_parent'];gap=pairs['remaining_gap_vs_archived_original_s16']
    point=change['ppl_relative_change']<=.01 and change['normal_mk_accuracy_delta']>0
    ci=change['normal_mk_paired_bootstrap_95ci'][0]>0
    flags=dict(ppl_no_worse_than_parent_1pct=change['ppl_relative_change']<=.01,
        ppl_improved_vs_parent=change['ppl_relative_change']<0,
        mk_observed_improvement_vs_parent=change['normal_mk_accuracy_delta']>0,
        mk_improvement_95ci_above_zero=ci,observed_joint_gate_pass=point,
        continuation_gate_pass=point and ci,final_claim_ready=point and ci,
        original_ppl_restored_within_1pct=gap['ppl_relative_change']<=.01,full_validation_complete=True)
    need(all(type(comparison[k]) is bool and comparison[k]==v for k,v in flags.items()),'Continuation gate flags differ')
    need(comparison['cache_bytes']=={arm:m['cache_bytes'] for arm,m in metrics.items()},'Cache comparison differs')
    close(comparison['cache_reduction_vs_archived_s16'],1-28499968/122028032,'Cache reduction arithmetic differs')
    need(comparison['frozen_source_final']['identity_version_gradients_unchanged'] is True,'Final frozen source check failed')
    for relative,digest in comparison['code_hashes'].items():
        need(sha(ROOT/relative)==digest,'Comparison code provenance differs')
    hashes.update(comparison=sha(args.eval_dir/'full_comparison.json'),
        restoration=sha(args.eval_dir/'full_restoration.json'),parent_replay=sha(args.eval_dir/'full_parent_replay.json'))
    return dict(arms=metrics,paired=pairs,quality_flags=flags,restoration=restoration,
                parent_replay=parent_replay,report_sha256=hashes,backend_policy=backend_proof)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-dir','calibration','parent-training-report','parent-checkpoint','training-report','parent-eval-dir','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--smoke-report',type=Path,help='Optional relocated copy of the bound discarded smoke report')
    parser.add_argument('--eval-dir',type=Path)
    parser.add_argument('--training-only',action='store_true')
    args=parser.parse_args()
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Set CUDA_VISIBLE_DEVICES= for CPU-only audit')
    need(args.training_only or args.eval_dir is not None,'--eval-dir is required unless --training-only')
    need(not args.output.exists(),'Preserve existing audit; choose a fresh output')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1'
    import numpy as np
    import torch
    from mamba2_recall import runtime,resurface_data as data
    from mamba2_recall.calibration import load_wikitext_tokens
    torch.set_num_threads(4)
    result=dict(format='MAMBA2_MORE_RESURFACE_INDEPENDENT_AUDIT_V1',complete=False,passed=False,
        training_only=args.training_only,source_sha256=sha(__file__),limitations=[
            'CPU reconstruction audits recorded NLL/generated IDs; it does not rerun GPU logits or generation.',
            'Adam moment values are checked for binding, shapes, finiteness and step counts; gradients are not archived to recompute every update.',
            'Frozen base checks cover reported identity/version/gradients, not full post-training weight byte hashes.',
            'Pinned tokenizer/case generator are reused; scoring and aggregate arithmetic are independent of the new evaluator.',
            'Historically exposed validation and prompt families do not establish unseen generalization.'])
    try:
        tokenizer=runtime.SentencePieceTokenizer(args.source_dir)
        train,meta=load_wikitext_tokens(tokenizer,'train')
        manifest=read(ROOT/'docs/prose_train_manifest.json')
        need(meta==manifest['dataset'],'Pinned TRAIN stream provenance differs')
        parent_args=SimpleNamespace(calibration=args.calibration,training_report=args.parent_training_report,
                                    eval_dir=args.parent_eval_dir,split='full')
        parent,calibration,proof=audit_parent_training(parent_args,tokenizer,torch,data,train)
        result['parent_training']=proof
        windows=[train[s:s+2048].tolist() for s in manifest['training_starts']]
        training,proof=audit_continuation(args,parent,calibration,windows,tokenizer,torch,data)
        result['continuation_training']=proof
        if not args.training_only:
            validation,dataset=load_wikitext_tokens(tokenizer,'validation')
            result['archived_parent_evaluation']=audit_parent_evaluation(parent_args,tokenizer,parent,calibration,
                                                                        validation,dataset,np,data)
            result['evaluation']=audit_evaluation(args,training,calibration,validation,dataset,tokenizer,data,np)
        need(not torch.cuda.is_initialized(),'CPU audit unexpectedly initialized CUDA')
        result.update(complete=True,passed=True,cuda_initialized=False,continuation_protocol_sha256=PROTOCOL,
                      training_report_sha256=sha(args.training_report))
    except BaseException as exc:
        result['error']=repr(exc);raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as f:
            json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
