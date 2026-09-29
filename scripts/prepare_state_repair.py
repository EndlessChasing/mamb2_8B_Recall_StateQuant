#!/usr/bin/env python3
"""Calibrate fixed state-repair candidates using original unadapted TRAIN S16 traces."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from mamba2_recall import runtime,resurface_data as data,resurface_native as native
from mamba2_recall.state_quant import StateQuant
from mamba2_recall.state_repair_calibration import ReadoutCalibration,equalizer_exponents
from prepare_quant_first import load_train_tokens,write_new_json,TRAIN_SHA,PROSE_MANIFEST_SHA
from evaluate_quant_first import FrozenBase
from evaluate_resurface_more import pin_replay_backend,check_replay_backend

PROTOCOL_SHA='f18069d0340449316297077a2d912f7ef42fbf518436d9b311d8a731c96d722c'
ORIGINAL_CALIBRATION_SHA='c366cd577967e64635f1dd960237dc1c0024d7413685ce0b4c4e40da869a7023'
FORMAT='MAMBA2_STATE_REPAIR_CALIBRATION_V1'


def tensor_sha(value):
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def hashes():
    paths=[Path(__file__),ROOT/'scripts/check_state_repair_calibration.py',ROOT/'scripts/prepare_quant_first.py',
        ROOT/'scripts/evaluate_resurface_more.py',ROOT/'docs/STATE_REPAIR_PROTOCOL.md',
        ROOT/'mamba2_recall/state_repair_calibration.py',ROOT/'mamba2_recall/state_codec.py',
        ROOT/'mamba2_recall/state_quant.py',ROOT/'mamba2_recall/runtime.py']
    return {str(p.relative_to(ROOT)):data.sha_file(p) for p in paths}


@torch.inference_mode()
def collection_forward_check(model,ids):
    """Compare both actual controllers on one fixed128-token TRAIN prefix."""
    with StateQuant(model,'s16',collect_stats=True) as old:
        reference=old.backbone(ids,reset=True)
        reference_hash=native.tensor_hash(reference)
        old_states=[native.tensor_hash(c.state.tensors['state']) for c in old._cache]
        old_conv=[native.tensor_hash(c.conv) for c in old._cache]
        old_stats=old.statistics()
    with ReadoutCalibration(model) as new:
        actual=new.backbone(ids,reset=True)
        state_hashes=[native.tensor_hash(c.state.tensors['state']) for c in new._cache]
        conv_hashes=[native.tensor_hash(c.conv) for c in new._cache]
        new_stats=new.statistics()
    if (not torch.equal(reference,actual) or old_states!=state_hashes or old_conv!=conv_hashes
            or not torch.equal(old_stats['sum_abs'],new_stats['sum_abs'])
            or not torch.equal(old_stats['mean_abs'],new_stats['mean_abs'])
            or not torch.equal(old_stats['sample_count_per_group'],new_stats['sample_count_per_group'])):
        raise RuntimeError('Readout collector changed original full-model S16 forward/carry/conv/abs statistics')
    return dict(complete=True,passed=True,probe_tokens=128,probe_token_sha256=runtime.token_digest(ids.cpu().numpy()),
        hidden_bitwise_equal=True,all56_ssm_carries_exact=True,all56_conv_caches_exact=True,
        mean_abs_statistics_exact=True,reference_hidden_sha256=reference_hash,
        collected_hidden_sha256=native.tensor_hash(actual),ssm_state_sha256=old_states,conv_state_sha256=old_conv,
        scope='Single128-token first TRAIN prefix, original unadapted source, pinned16-warp RMSNorm; standalone synthetic tests cover chunks')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--train-tokens',type=Path,required=True)
    p.add_argument('--original-calibration',type=Path,required=True)
    p.add_argument('--collector-checks',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists() or args.out.is_symlink():raise FileExistsError('Fresh state-repair calibration directory required')
    if data.sha_file(ROOT/'docs/STATE_REPAIR_PROTOCOL.md')!=PROTOCOL_SHA:
        raise ValueError('Frozen state-repair protocol changed')
    if data.sha_file(args.original_calibration)!=ORIGINAL_CALIBRATION_SHA:
        raise ValueError('Pinned original v2 calibration changed')
    checks=json.loads(args.collector_checks.read_text())
    if (checks.get('complete') is not True or checks.get('passed') is not True
            or checks.get('format')!='MAMBA2_STATE_REPAIR_COLLECTOR_CHECKS_V1'):
        raise ValueError('Completed synthetic collector equivalence receipt required')
    for name,digest in checks['code_sha256'].items():
        if data.sha_file(ROOT/name)!=digest:raise ValueError('Collector code changed after equivalence tests: '+name)
    torch.set_num_threads(8)
    torch.manual_seed(2026092901);torch.cuda.manual_seed_all(2026092901)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    train=load_train_tokens(args.train_tokens)
    tokenizer=runtime.SentencePieceTokenizer(args.source_dir)
    original=torch.load(args.original_calibration,map_location='cpu',weights_only=True)
    if (original['format']!='MAMBA2_QUANT_FIRST_CALIBRATION_V1' or original['adapter'] is not None
            or original['source_sha256']!=runtime.SOURCE_CHECKPOINT_SHA256
            or original['tokenizer_sha256']!=tokenizer.sha256):
        raise ValueError('Original calibration source/adapter provenance differs')
    original_permutations=original['permutations'].clone().contiguous()
    if (original_permutations.dtype!=torch.uint8 or tuple(original_permutations.shape)!=(56,8,128)
            or not torch.equal(original_permutations.sort(-1).values,
                               torch.arange(128,dtype=torch.uint8).expand(56,8,128))):
        raise ValueError('Original permutation geometry/content differs')
    args.out.mkdir(parents=True)
    policy=pin_replay_backend()
    started=time.monotonic()
    model=runtime.load_source_model(args.source_dir)
    frozen=FrozenBase(model)
    if any(mx._forward_pre_hooks or mx._forward_hooks or mx.norm._forward_pre_hooks
           for mx in (layer.mixer for layer in model.backbone.layers)):
        raise RuntimeError('Calibration source unexpectedly has adapter hooks')
    torch.cuda.reset_peak_memory_stats()
    proof=collection_forward_check(model,train[0,:128].cuda()[None])
    frozen.check()
    print('[collector] original and instrumented128-token full-model forward/carry/conv/abs statistics exact',flush=True)
    with torch.inference_mode(),ReadoutCalibration(model) as collector:
        for row in range(8):
            hidden=collector.backbone(train[row,:512].cuda()[None],reset=True)
            if not bool(torch.isfinite(hidden).all()):raise FloatingPointError('Nonfinite calibration output')
            del hidden
            print(f'[TRAIN readout calibration] {row+1}/8',flush=True)
        stats=collector.statistics()
        cache=collector.cache_breakdown()
    if (not bool((stats['sample_count_per_group']==4096*16*64).all())
            or any(tuple(stats[k].shape)!=(56,8,128) or not bool(torch.isfinite(stats[k]).all())
                   or bool((stats[k]<0).any()) for k in ('mean_abs','mean_readout_score','sum_abs','sum_readout_score'))):
        raise ValueError('Calibration statistic geometry/finiteness/count differs')
    readout_permutations=torch.argsort(stats['mean_readout_score'],dim=-1,descending=True,stable=True).to(torch.uint8).contiguous()
    exponents=equalizer_exponents(stats['mean_abs'])
    token_hashes=[runtime.token_digest(train[i,:512].numpy()) for i in range(8)]
    binding=dict(protocol_sha256=PROTOCOL_SHA,source_sha256=runtime.SOURCE_CHECKPOINT_SHA256,
        tokenizer_sha256=tokenizer.sha256,train_file_sha256=TRAIN_SHA,prose_manifest_sha256=PROSE_MANIFEST_SHA,
        original_calibration_sha256=ORIGINAL_CALIBRATION_SHA,adapter=None,adapter_sha256=None,
        heldout_used=False,calibration_tokens=4096,train_token_hashes=token_hashes,
        selection='First8 original TRAIN rows, first512tokens; independent zero reset per row')
    payload=dict(format=FORMAT,**binding,readout_permutations=readout_permutations,
        equalizer_exponents=exponents,original_permutations=original_permutations,statistics=stats)
    output=args.out/'calibration.pt';pending=output.with_suffix('.pt.pending')
    with pending.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    roundtrip=torch.load(pending,map_location='cpu',weights_only=True)
    if set(roundtrip)!=set(payload) or roundtrip['format']!=FORMAT or any(roundtrip[k]!=v for k,v in binding.items()):
        raise RuntimeError('Calibration header/provenance serialization changed')
    for key in ('readout_permutations','equalizer_exponents','original_permutations'):
        if not torch.equal(roundtrip[key],payload[key]):raise RuntimeError('Table roundtrip failed: '+key)
    for key,value in stats.items():
        if not (torch.equal(roundtrip['statistics'][key],value) if isinstance(value,torch.Tensor)
                else roundtrip['statistics'][key]==value):raise RuntimeError('Statistics roundtrip failed: '+key)
    pending.replace(output)
    ranked=stats['mean_readout_score'].gather(-1,readout_permutations.long())
    total=ranked.sum()
    if not bool(total>0):raise FloatingPointError('Readout calibration score is empty')
    receipt=dict(format=FORMAT,complete=True,**binding,file=output.name,sha256=data.sha_file(output),bytes=output.stat().st_size,
        fresh_source_no_adapter=True,collection_forward_exactcheck=proof,
        collector_checks_sha256=data.sha_file(args.collector_checks),collector_checks=checks,
        backend_policy=policy,backend_policy_check=check_replay_backend(policy),
        table_sha256={key:tensor_sha(payload[key]) for key in ('readout_permutations','equalizer_exponents','original_permutations')},
        table_shapes={key:list(payload[key].shape) for key in ('readout_permutations','equalizer_exponents','original_permutations')},
        runtime_table_bytes_per_candidate=57344,runtime_table_policy='Eachcandidate retains exactly ONE table; calibration artifact holds alternatives offline',
        readout_tier_counts=dict(int8=16,int4=64,dead=48),
        readout_score_tier_mass={name:float(ranked[...,lo:hi].sum()/total)
            for name,lo,hi in [('int8',0,16),('int4',16,80),('dead',80,128)]},
        original_permutation_dead_readout_score_mass=float(stats['mean_readout_score'].gather(-1,original_permutations.long())[...,80:].sum()/total),
        equalizer_exponent_counts={str(k):int((exponents==k).sum()) for k in range(-8,9)},
        exponent_rounding='FP64 log2(max(meanabs,1e-12)) minus coordinate meanlog2; ties-to-even round; clamp[-8,8]',
        sample_count_per_group=stats['sample_count_per_group'].tolist(),cache_and_workspace=cache,
        frozen_source=frozen.check(),serialization_roundtrip_bitwise=True,code_sha256=hashes(),
        environment=runtime.environment_receipt(),gpu_memory=runtime.gpu_memory_receipt(),elapsed_seconds=time.monotonic()-started)
    write_new_json(args.out/'calibration.json',receipt)
    print(json.dumps({k:receipt[k] for k in ('complete','sha256','bytes','calibration_tokens','readout_score_tier_mass',
        'original_permutation_dead_readout_score_mass','equalizer_exponent_counts')},indent=2),flush=True)

if __name__=='__main__':main()
