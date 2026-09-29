#!/usr/bin/env python3
"""CPU-only independent arithmetic/provenance audit of quantize-first results.

No model loading or generation; this reconstructs data identities, checks frozen
artifacts and recomputes scores from raw reported NLL/generated-token evidence.
It cannot independently prove reported logits or replace a GPU repeat run.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROTOCOL = '24466642ce87c75fc2136a42ed14e69733c462b507a0be82c062b4da7f836bcb'
SOURCE = '47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb'
TOKENIZER = '5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09'
PROSE_MANIFEST = 'facb2ca461615a4199781bd21784d642d6674f5b862641b3b9edac3fb499b89d'
PROSE_FILE = 'e54b02e5162e042a9cdd504f4eb1b1652724fb240bbc2c97608967aa26297233'
VALIDATION_TOKENS = '5bbeae08ba8eb34a482f3b6e9d17b182e67229dd14b2853d87f89fc72e5ad027'
CAL_FORMAT = 'MAMBA2_QUANT_FIRST_CALIBRATION_V1'
ADAPTER_FORMAT = 'MAMBA2_POST_D_RESURFACE_FP16_V1'
ARMS = ('source_s16', 'source_sq3p25', 'resurface_sq3p25', 'restored_source_sq3p25')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def close(a, b, message, tol=1e-12):
    need(finite(a) and finite(b) and math.isclose(a, b, rel_tol=tol, abs_tol=tol), message)


def tokhash(ids):
    return hashlib.sha256(b''.join(int(v).to_bytes(8, 'little', signed=True) for v in ids)).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def expected_cache(mode, tokens=None):
    rows = 56 * 128 * 64
    state = rows * (128 * 2 if mode == 's16' else 52)
    scales = 0 if mode == 's16' else rows * 4
    tables = 0 if mode == 's16' else 56 * 8 * 128
    conv = 56 * (8192 + 2 * 8 * 128) * 4 * 2
    result = dict(mode=mode, batch_size=1, allocated_layers=56,
                  conv_fp16_bytes=conv, ssm_payload_bytes=state-scales,
                  ssm_scale_bytes=scales, ssm_total_bytes=state,
                  permutation_bytes=tables, total_bytes=conv+state+tables,
                  calibration_workspace_bytes=0)
    if tokens is not None:
        result['tokens_per_layer'] = [tokens] * 56
    return result


def check_cache(actual, mode, tokens=None):
    expected = expected_cache(mode, tokens)
    need(all(actual.get(k) == v for k, v in expected.items()), 'Persistent cache allocation differs')
    return expected['total_bytes']


def audit_training(args, tokenizer, torch, data, train_stream):
    calibration = torch.load(args.calibration, map_location='cpu', weights_only=True)
    receipt = read(args.calibration.with_suffix('.json'))
    cal_sha = sha(args.calibration)
    prose_manifest = read(ROOT / 'docs/prose_train_manifest.json')
    need(sha(ROOT / 'docs/QUANT_FIRST_PROTOCOL.md') == PROTOCOL, 'Frozen protocol changed')
    need(sha(ROOT / 'docs/prose_train_manifest.json') == PROSE_MANIFEST, 'Pinned prose manifest changed')
    windows = [train_stream[s:s+2048].tolist() for s in prose_manifest['training_starts']]
    need(len(windows) == 448 and all(len(w) == 2048 for w in windows), 'Wrong prose TRAIN windows')
    need(tokhash([x for w in windows for x in w]) == prose_manifest['training_tokens_sha256_int64le'],
         'Reconstructed TRAIN window content differs')
    for actual, w in zip(prose_manifest['training_windows'], windows):
        need(actual['token_sha256_int64le'] == tokhash(w), 'TRAIN individual window digest differs')
    selection = [tokhash(w[:512]) for w in windows[:8]]
    bound = dict(format=CAL_FORMAT, protocol_sha256=PROTOCOL, source_sha256=SOURCE,
                 tokenizer_sha256=TOKENIZER, train_file_sha256=PROSE_FILE,
                 prose_manifest_sha256=PROSE_MANIFEST, adapter=None, adapter_sha256=None)
    need(all(calibration.get(k) == v and receipt.get(k) == v for k, v in bound.items()),
         'Calibration was not bound to the original unadapted source')
    need(receipt['complete'] is True and receipt['fresh_source_no_adapter'] is True
         and receipt['heldout_used'] is False and receipt['calibration_tokens'] == 4096
         and receipt['sha256'] == cal_sha and receipt['bytes'] == args.calibration.stat().st_size,
         'Calibration receipt identity/status differs')
    need(calibration['train_token_hashes'] == receipt['train_token_hashes'] == receipt['token_hashes'] == selection,
         'Calibration selection does not match original TRAIN tokens')
    need(calibration['train_manifest_sha256'] == receipt['train_manifest_sha256'], 'TRAIN manifest binding differs')
    stats = calibration['statistics']
    sums, means, counts = (stats[k] for k in ('sum_abs', 'mean_abs', 'sample_count_per_group'))
    perms = calibration['permutations']
    need(sums.dtype == means.dtype == torch.float64 and counts.dtype == torch.int64
         and tuple(sums.shape) == tuple(means.shape) == (56, 8, 128)
         and tuple(counts.shape) == (56,) and bool((counts == 4096*16*64).all())
         and bool(torch.isfinite(sums).all()) and bool((sums >= 0).all())
         and torch.equal(means, sums / counts[:, None, None]), 'Calibration means/counts do not derive from sums')
    ranked = torch.argsort(means, dim=-1, descending=True, stable=True).to(torch.uint8)
    need(perms.dtype == torch.uint8 and tuple(perms.shape) == (56, 8, 128)
         and torch.equal(perms, ranked) and receipt['sample_count_per_group'] == counts.tolist()
         and hashlib.sha256(perms.numpy().tobytes()).hexdigest() == receipt['permutations_sha256_uint8'],
         'Frozen tier table does not derive from stable magnitude ordering')
    ordered = means.gather(-1, perms.long())
    total = ordered.sum()
    need(float(total) > 0, 'Empty calibration magnitude')
    for label, low, high in [('int8', 0, 16), ('int4', 16, 80), ('dead', 80, 128)]:
        close(receipt['coordinate_abs_mass_fractions'][label], float(ordered[..., low:high].sum()/total),
              'Calibration tier mass fraction differs')
    group_total = ordered.sum(-1)
    dead_fraction = torch.where(group_total > 0, ordered[..., 80:].sum(-1)/group_total,
                                torch.zeros_like(group_total))
    need(dead_fraction.tolist() == receipt['dead_coordinate_abs_mass_fraction_per_layer_group'],
         'Per-group discarded magnitude differs')
    close(receipt['dead_coordinate_abs_mass_fraction'], float(ordered[..., 80:].sum()/total),
          'Global discarded magnitude differs')
    need(receipt['tier_coordinate_counts'] == dict(int8=16, int4=64, dead=48)
         and receipt['permutation_payload_bytes'] == 57344
         and receipt['frozen_source_identity_version_gradients_unchanged'] is True
         and receipt['serialization_roundtrip_bitwise'] is True, 'Calibration scope/status differs')

    training = read(args.training_report)
    need(training['format'] == 'MAMBA2_SQ_FIRST_RESURFACE_TRAIN_V1' and training['complete'] is True
         and training['mode'] == 'formal' and training['successful_updates'] == 1536
         and 1536 <= training['attempts'] <= 1544 and 'error' not in training,
         'A completed formal 1536-success report is required')
    binding = training['binding']
    expected_binding = dict(calibration_sha256=cal_sha, calibration_format=CAL_FORMAT,
        source_checkpoint_sha256=SOURCE, tokenizer_sha256=TOKENIZER, protocol_sha256=PROTOCOL,
        train_manifest_sha256=calibration['train_manifest_sha256'], prose_manifest_sha256=PROSE_MANIFEST,
        prose_tokens_sha256=PROSE_FILE, prose_tokens_int64le_sha256=prose_manifest['training_tokens_sha256_int64le'],
        prose_file_exact_historical=True, adapter=ADAPTER_FORMAT, successful_updates=1536,
        initial_adapter='fresh V=0,g=1,w=0,b=-4; no pretrained adapter',
        state_mode='sq3p25; exact packed forward; live-mask STE backward',
        teacher='separate unadapted source, S16 per-token carry')
    need(binding == expected_binding, 'Training/export binding or no-old-adapter recipe differs')
    need(training['initialization_check']['fresh_identity_matches_packed_bitwise'] is True
         and training['initialization_check']['probe_tokens'] == 32
         and training['initialization_check']['probe_token_sha256'] == tokhash(windows[0][:32]),
         'Fresh identity adapter probe differs')
    need(training['frozen_base_check']['identity_version_gradients_unchanged'] is True
         and training['frozen_base_check']['parameters'] == 8236999680
         and training['frozen_base_check']['tensors'] == 507
         and training['frozen_state_calibration_check'] is True
         and training['teacher_base_parameters_frozen'] is True
         and training['deployed_export_check']['packed_training_forward_bitwise_equal'] is True,
         'Frozen parameters/calibration or deployment parity failed')
    check_cache(training['deployed_export_check']['cache'], 'sq3p25', 128)
    schedule = torch.randperm(1536, generator=torch.Generator().manual_seed(2026092803)).tolist()
    prose_schedule = torch.randperm(448, generator=torch.Generator().manual_seed(20260928)).tolist()
    need(prose_schedule == prose_manifest['schedule'], 'Pinned prose schedule differs')
    train_cases = data.generate_cases('train')
    need(len(training['history']) == training['attempts'], 'Training history incomplete')
    successful = overflows = 0
    for attempt, row in enumerate(training['history'], 1):
        need(successful < 1536 and type(row['overflow']) is bool, 'Invalid overflow/status field')
        entry = schedule[successful]
        raw = train_cases[entry]
        prompt_ids = tokenizer.encode(raw['prompt'])
        full_ids = tokenizer.encode(raw['prompt'] + ' ' + raw['answer'])
        need(full_ids[:len(prompt_ids)] == prompt_ids, 'TRAIN answer prefix instability')
        need(row['attempt'] == attempt and row['schedule_entry'] == entry and row['case_id'] == raw['id']
             and row['prose_window'] == prose_schedule[successful % 448]
             and row['prose_start'] == 512*((successful//448) % 4)
             and row['answer_targets'] == len(full_ids)-len(prompt_ids), 'Training data schedule/targets differ')
        need(all(finite(row[k]) for k in ('mk_ce', 'prose_ce', 'prose_kl', 'prose_closure', 'seconds')),
             'Nonfinite training losses/timing')
        if row['overflow']:
            overflows += 1
            need(row['gradient_norm_before_clip'] is None, 'Overflow unexpectedly took a gradient update')
        else:
            successful += 1
            need(finite(row['gradient_norm_before_clip']) and row['gradient_norm_before_clip'] >= 0,
                 'Successful update has invalid gradient norm')
        need(row['successful_updates'] == successful and row['loss_scale'] == 1024/(2**overflows),
             'Successful-update or gradient-scale accounting differs')
    need(successful == 1536 and overflows <= 8, 'Frozen update/overflow budget differs')
    need(len(training['checkpoints']) == 4, 'Expected four fixed recovery checkpoints')
    for step, item in zip((384, 768, 1152, 1536), training['checkpoints']):
        p = args.training_report.parent / Path(item['path']).name
        need(p.name == f'checkpoint_{step:04d}.pt' and p.is_file()
             and p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], 'Recovery checkpoint identity differs')
    for relative, digest in training['code_sha256'].items():
        need(sha(ROOT / relative) == digest, 'Training source changed: ' + relative)

    exported = training['adapter']
    need(Path(exported['file']).name == exported['file'], 'Unsafe adapter filename')
    adapter_path = args.training_report.parent / exported['file']
    need(sha(adapter_path) == exported['sha256'] and adapter_path.stat().st_size == exported['bytes']
         and exported['roundtrip_bitwise_equal'] is True and exported['gate_mode'] == 'soft',
         'Final adapter file identity differs')
    adapter = torch.load(adapter_path, map_location='cpu', weights_only=True)
    need(adapter['format'] == ADAPTER_FORMAT and adapter['gate_mode'] == 'soft'
         and adapter['variant'] == 'post-D native norm-prehook; memoryless cross-head mixing'
         and adapter['binding'] == binding
         and adapter['geometry'] == [dict(width=4096, heads=128, head_dim=64)]*56,
         'Exported adapter header/binding differs')
    expected_shapes = {f'layer{i}.{name}': shape for i in range(56)
                       for name, shape in [('V_read', (128,128)), ('g_read',(128,)),
                                           ('router_w',(4096,)), ('router_b',())]}
    values = adapter['tensors']
    need(set(values) == set(expected_shapes) and len(values) == 224, 'Adapter tensor inventory differs')
    for name, value in values.items():
        need(isinstance(value, torch.Tensor) and value.dtype == torch.float16
             and tuple(value.shape) == expected_shapes[name] and bool(torch.isfinite(value).all())
             and hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest() == exported['tensor_sha256'][name],
             'Invalid FP16 adapter tensor: ' + name)
    need(sum(v.numel() for v in values.values()) == exported['parameters'] == 1154104
         and sum(v.numel()*2 for v in values.values()) == exported['payload_bytes'] == 2308208,
         'Adapter parameter/payload accounting differs')
    summary = dict(successful_updates=successful, attempts=training['attempts'], overflow_attempts=overflows,
                   new_identity_initialization=True, original_base_and_calibration_frozen=True,
                   adapter_tensors=224, adapter_parameters=1154104, adapter_sha256=exported['sha256'],
                   calibration_sha256=cal_sha, calibration_discarded_abs_mass=float(ordered[...,80:].sum()/total),
                   four_recovery_checkpoint_hashes_verified=True)
    return training, receipt, summary


def recompute_pair(left, right, np):
    lrows = {r['id']: r for r in left['mk']['rows'] if r['condition'] == 'normal'}
    rrows = {r['id']: r for r in right['mk']['rows'] if r['condition'] == 'normal'}
    need(lrows.keys() == rrows.keys() and lrows, 'Normal MK pairing differs')
    differences = np.array([int(rrows[k]['correct'])-int(lrows[k]['correct']) for k in sorted(lrows)])
    rng = np.random.default_rng(20260928)
    sampled = differences[rng.integers(0, len(differences), size=(10000, len(differences)))].mean(1)
    ci = [float(x) for x in np.quantile(sampled, [.025, .975])]
    return dict(control_arm=left['arm'], candidate_arm=right['arm'], control_ppl=left['ppl']['ppl'],
        candidate_ppl=right['ppl']['ppl'], ppl_relative_change=right['ppl']['ppl']/left['ppl']['ppl']-1,
        control_normal_mk=left['mk']['summary']['normal'], candidate_normal_mk=right['mk']['summary']['normal'],
        normal_mk_accuracy_delta=float(differences.mean()), normal_mk_correct_delta=int(differences.sum()),
        paired_improvements=int((differences>0).sum()), paired_regressions=int((differences<0).sum()),
        paired_unchanged=int((differences==0).sum()), normal_mk_paired_bootstrap_95ci=ci,
        bootstrap_draws=10000, bootstrap_seed=20260928,
        control_target_removed_mk=left['mk']['summary']['target_removed'],
        candidate_target_removed_mk=right['mk']['summary']['target_removed'])


def audit_evaluation(args, tokenizer, training, calibration, validation, dataset, np, data):
    windows = [(s, validation[s:min(s+2049,len(validation))].tolist())
               for s in range(0,len(validation)-1,2048)]
    need(len(windows) == 130 and sum(len(w)-1 for _,w in windows) == 264764
         and tokhash(validation.tolist()) == VALIDATION_TOKENS, 'Pinned validation token stream differs')
    if args.split == 'pilot':
        windows = [windows[i] for i in (0,32,64,96)]
        samples = {0,9,18,27,36,45,54,63}
        cases = [r for r in data.frozen_development_cases() if r['sample'] in samples]
        expected_case_count, expected_targets = 96, 8192
    else:
        cases = data.generate_cases('confirm')
        expected_case_count, expected_targets = 768, 264764
    need(len(cases) == expected_case_count, 'Wrong frozen MK case inventory')
    prompt_info = [(tokenizer.encode(case['prompt']), case) for case in cases]
    outputs, metrics, hashes = {}, {}, {}
    common_binding = dict(format='MAMBA2_QUANT_FIRST_EVAL_V1', stage=args.split,
        protocol_sha256=PROTOCOL, source_checkpoint_sha256=SOURCE, tokenizer_sha256=TOKENIZER,
        calibration=calibration, calibration_receipt_sha256=sha(args.calibration.with_suffix('.json')),
        training_report_sha256=sha(args.training_report), training_binding=training['binding'],
        candidate_adapter_sha256=training['adapter']['sha256'], dataset=dataset)
    for arm in ARMS:
        path = args.eval_dir / f'{args.split}_{arm}.json'
        result = read(path)
        need(result['complete'] is True and result['arm'] == arm and
             all(result.get(k) == v for k,v in common_binding.items()), 'Evaluation provenance/incomplete arm: ' + arm)
        mode = 's16' if arm == 'source_s16' else 'sq3p25'
        need(result['mode'] == mode and result['adapter_sha256'] == (
            training['adapter']['sha256'] if arm == 'resurface_sq3p25' else None), 'Wrong adapter/mode in arm')
        need(result['frozen_source']['identity_version_gradients_unchanged'] is True
             and result['frozen_source']['parameters'] == 8236999680
             and result['frozen_source']['tensors'] == 507, 'Frozen source inventory differs')
        if arm == 'resurface_sq3p25':
            need(result['adapter_content_unchanged'] is True, 'Adapter inference content changed')
        for relative,digest in result['code_hashes'].items():
            need(sha(ROOT/relative) == digest, 'Evaluation source changed: '+relative)
        recorded = result['ppl']['windows']
        need(len(recorded) == len(windows), 'Omitted/extra PPL windows')
        nll, ntokens = 0., 0
        for row, (start, ids) in zip(recorded, windows):
            need(row['start'] == start and row['target_tokens'] == len(ids)-1
                 and row['token_sha256_int64le'] == tokhash(ids)
                 and finite(row['nll']) and row['nll'] >= 0, 'PPL window identity/NLL differs')
            close(row['ppl'], math.exp(row['nll']/(len(ids)-1)), 'Per-window PPL arithmetic differs')
            nll += row['nll']; ntokens += len(ids)-1
        need(ntokens == expected_targets == result['ppl']['target_tokens'], 'PPL target count differs')
        close(result['ppl']['nll'], nll, 'Aggregate NLL differs')
        close(result['ppl']['ppl'], math.exp(nll/ntokens), 'Aggregate PPL differs')
        rows = result['mk']['rows']
        need(len(rows) == expected_case_count and len({r['id'] for r in rows}) == expected_case_count,
             'MK case omission or duplicate')
        summary = {condition: dict(correct=0,count=0) for condition in ('normal','target_removed')}
        for row, (ids, case) in zip(rows, prompt_info):
            need(all(row.get(k) == v for k,v in case.items()), 'MK case not exact frozen reconstruction')
            need(row['prompt_tokens'] == len(ids) and row['prompt_token_sha256_int64le'] == tokhash(ids),
                 'MK prompt tokenization differs')
            generated = row['generated_ids']
            need(isinstance(generated,list) and 1 <= len(generated) <= 12
                 and all(type(t) is int and 0 <= t < 256000 for t in generated)
                 and tokenizer.eos_token_id not in generated[:-1]
                 and (len(generated) == 12 or generated[-1] == tokenizer.eos_token_id), 'Invalid greedy stop/token sequence')
            text = tokenizer.decode(generated)
            match = re.search(r'(?<!\d)\d{6}(?!\d)', text)
            prediction = match.group(0) if match else None
            correct = prediction == case['answer']
            need(row['output'] == text and row['prediction'] == prediction
                 and type(row['correct']) is bool and row['correct'] == correct,
                 'Generated ID decode/regex correctness differs')
            summary[case['condition']]['correct'] += int(correct)
            summary[case['condition']]['count'] += 1
        for condition, counts in summary.items():
            counts['accuracy'] = counts['correct']/counts['count']
            need(counts['count'] == expected_case_count//2 and result['mk']['summary'][condition] == counts,
                 'Recomputed MK aggregate differs')
        cache = check_cache(result['cache'],mode,len(windows[-1][1])-1)
        metrics[arm] = dict(ppl=math.exp(nll/ntokens), nll=nll, ppl_target_tokens=ntokens,
                            normal=summary['normal'], target_removed=summary['target_removed'],cache_bytes=cache)
        outputs[arm] = result; hashes[arm] = sha(path)
    before, after = outputs['source_sq3p25'], outputs['restored_source_sq3p25']
    need(before['ppl']['windows'] == after['ppl']['windows'] and before['mk']['rows'] == after['mk']['rows'],
         'Entire SQ baseline did not replay after adapter removal')
    restoration = dict(complete=True,per_window_nll_exact=True,all_generated_ids_exact=True,
        all_decoded_predictions_exact=True,ppl_windows_repeated=len(windows),ppl_target_tokens_repeated=expected_targets,
        mk_cases_repeated=expected_case_count,
        scope='Entire selected pilot/full SQ baseline repeated after removing the new adapter')
    need(read(args.eval_dir/f'{args.split}_restoration.json') == restoration, 'Restoration summary differs')
    comparison = read(args.eval_dir/f'{args.split}_comparison.json')
    need(comparison['complete'] is True and comparison['stage'] == args.split
         and comparison['protocol_sha256'] == PROTOCOL
         and comparison['training_report_sha256'] == sha(args.training_report)
         and comparison['calibration_sha256'] == sha(args.calibration)
         and comparison['adapter_sha256'] == training['adapter']['sha256']
         and comparison['report_sha256'] == hashes and comparison['restoration'] == restoration,
         'Comparison hash/binding/restoration differs')
    recomputed = {}
    for key, left, right in [('repair_vs_sq_baseline','source_sq3p25','resurface_sq3p25'),
                             ('remaining_gap_vs_original_s16','source_s16','resurface_sq3p25'),
                             ('sq_quantization_effect','source_s16','source_sq3p25')]:
        values = recompute_pair(outputs[left],outputs[right],np)
        need(comparison[key] == values, 'Paired comparison/bootstrap differs: '+key)
        recomputed[key] = values
    repair = recomputed['repair_vs_sq_baseline']
    gap = recomputed['remaining_gap_vs_original_s16']
    point = repair['ppl_relative_change'] <= .01 and repair['normal_mk_accuracy_delta'] > 0
    positive_ci = repair['normal_mk_paired_bootstrap_95ci'][0] > 0
    flags = dict(ppl_no_worse_than_sq_baseline_1pct=repair['ppl_relative_change'] <= .01,
        mk_observed_improvement_vs_sq=repair['normal_mk_accuracy_delta'] > 0,
        mk_improvement_95ci_above_zero=positive_ci, observed_joint_repair_gate_pass=point,
        original_ppl_restored_within_1pct=gap['ppl_relative_change'] <= .01,
        full_validation_complete=args.split=='full',final_claim_ready=args.split=='full' and point and positive_ci)
    need(all(comparison[k] == v for k,v in flags.items()), 'Quality claim flags differ from fixed protocol')
    caches = {arm:m['cache_bytes'] for arm,m in metrics.items()}
    need(comparison['cache_bytes'] == caches, 'Comparison cache accounting differs')
    close(comparison['cache_reduction_fraction'],1-caches['source_sq3p25']/caches['source_s16'],
          'Cache reduction arithmetic differs')
    hashes.update(comparison=sha(args.eval_dir/f'{args.split}_comparison.json'),
                  restoration=sha(args.eval_dir/f'{args.split}_restoration.json'))
    return dict(arms=metrics,paired=recomputed,quality_flags=flags,restoration=restoration,report_sha256=hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,required=True,help='Pinned tokenizer directory/file; weights are never loaded')
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--training-report',type=Path,required=True)
    parser.add_argument('--eval-dir',type=Path,required=True)
    parser.add_argument('--split',choices=('pilot','full'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Set CUDA_VISIBLE_DEVICES= for CPU-only audit')
    need(not args.output.exists(), 'Preserve existing audit; choose a fresh output file')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    import numpy as np
    import torch
    from mamba2_recall import runtime, resurface_data as data
    from mamba2_recall.calibration import load_wikitext_tokens
    torch.set_num_threads(4)
    result = dict(format='MAMBA2_QUANT_FIRST_INDEPENDENT_AUDIT_V1',complete=False,passed=False,
                  stage=args.split,source_sha256=sha(__file__),limitations=[
        'CPU re-audit of recorded NLL/token evidence; does not re-run model logits or greedy generation.',
        'Frozen source checks certify reported identity/version/gradients, not a post-training full-weight byte hash.',
        'Uses the pinned package tokenizer and frozen case generator; independently recomputes metric arithmetic and scoring.',
        'Historically exposed validation and prompt families do not establish unseen generalization.'])
    try:
        tokenizer = runtime.SentencePieceTokenizer(args.source_dir)
        train_stream, train_meta = load_wikitext_tokens(tokenizer,'train')
        manifest = read(ROOT/'docs/prose_train_manifest.json')
        need(train_meta == manifest['dataset'], 'Cached pinned TRAIN text/token identity differs')
        training, calibration, proof = audit_training(args,tokenizer,torch,data,train_stream)
        result['training'] = proof
        validation, dataset = load_wikitext_tokens(tokenizer,'validation')
        result['evaluation'] = audit_evaluation(args,tokenizer,training,calibration,validation,dataset,np,data)
        need(not torch.cuda.is_initialized(), 'Audit unexpectedly initialized CUDA')
        result.update(complete=True,passed=True,cuda_initialized=False,
                      protocol_sha256=PROTOCOL,training_report_sha256=sha(args.training_report))
    except BaseException as exc:
        result['error'] = repr(exc)
        raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as f:
            json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
