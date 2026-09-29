#!/usr/bin/env python3
"""Select unadapted SQ3.25 on TRAIN, then verify its separately trained fresh adapter."""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from mamba2_recall import runtime, resurface_data as data, resurface_native as native
from mamba2_recall.state_quant import StateQuant
from mamba2_recall.calibration import load_wikitext_tokens
from mamba2_recall.evaluation import ppl_windows
from prepare_quant_first import load_train_tokens, write_new_json
from prepare_state_first_v5 import (PROTOCOL_SHA, NUMERIC_PROTOCOL_SHA, KINDS, CACHE_BYTES,
    CALIBRATION_FORMAT, base_binding, check_protocol, choose_candidate, load_candidates,
    load_selected_calibration, need, read_json, tensor_sha, write_payload)
from evaluate_quant_first import FrozenBase, compare_pair, check_restoration, VALIDATION_TOKENS_SHA
from evaluate_resurface_more import pin_replay_backend, check_replay_backend, ARCHIVE_HASHES
from run_state_repair import evaluate, finite_cache, CandidateInvalid
from run_statequant import save_json


def code_hashes():
    paths = list((ROOT/'mamba2_recall').glob('*.py'))
    paths += [ROOT/'scripts'/name for name in ('run_state_first_v5.py', 'prepare_state_first_v5.py',
        'run_state_repair.py', 'prepare_quant_first.py', 'evaluate_quant_first.py',
        'evaluate_resurface_more.py', 'run_statequant.py')]
    paths += [ROOT/'docs/STATE_FIRST_V5_PROTOCOL.md', ROOT/'docs/RESURFACE_MORE_BACKEND_REPLAY.md']
    return {str(p.relative_to(ROOT)):data.sha_file(p) for p in sorted(paths)}


def no_adapter_hooks(model):
    return not any(mx._forward_pre_hooks or mx._forward_hooks or mx.norm._forward_pre_hooks
                   for mx in (layer.mixer for layer in model.backbone.layers))


@torch.inference_mode()
def cache_probe(model, table, tokens):
    """Record both repeats' actual persistent tensor hashes; engine repeats its own guard."""
    with StateQuant(model, 'sq3p25', table) as execution:
        def run():
            hidden = execution.backbone(tokens[:128].cuda()[None], reset=True)
            if not bool(torch.isfinite(hidden).all()):
                raise CandidateInvalid('Nonfinite preflight hidden')
            finite_cache(execution)
            return native.tensor_hash(hidden), {f'{i}.{key}':native.tensor_hash(value)
                for i, row in enumerate(execution._cache)
                for key, value in {**row.state.tensors, 'conv':row.conv}.items()}
        first, second = run(), run()
        if first != second:
            raise CandidateInvalid('Preflight persistent cache/hidden failed exact repeat')
        cache = execution.cache_breakdown()
        if cache['total_bytes'] != CACHE_BYTES:
            raise CandidateInvalid('Preflight cache differs from fixed budget')
    return dict(tokens=128, hidden_sha256=first[0], cache_tensor_sha256=first[1],
                both_repeats_exact=True, cache=cache)


def load_fresh_training(path, selected, selected_path):
    """Fail closed on source/table/export/final-checkpoint identity before inference."""
    report = read_json(path)
    need(report.get('format') == 'MAMBA2_STATE_FIRST_TRAIN_V1' and report.get('complete') is True
         and report.get('mode') == 'formal' and report.get('successful_updates') == 1536
         and 1536 <= report.get('attempts', -1) <= 1544 and 'error' not in report,
         'Completed formal fresh1536-update training required')
    need(report.get('frozen_base_check', {}).get('identity_version_gradients_unchanged') is True
         and report.get('frozen_state_calibration_check') is True
         and report.get('teacher_base_parameters_frozen') is True
         and report.get('initialization_check', {}).get('fresh_identity_matches_packed_bitwise') is True
         and report.get('deployed_export_check', {}).get('packed_training_forward_bitwise_equal') is True
         and report['deployed_export_check'].get('probe_tokens') == 128,
         'Fresh initialization/frozen source/deployed parity proof missing')
    check = report.get('final_checkpoint_export_check', {})
    need(all(check.get(k) is True for k in ('all_master_casts_equal_export', 'optimizer_exact', 'scaler_exact'))
         and check.get('master_tensors') == 224, 'Final master/optimizer/scaler export proof missing')
    fresh = report.get('fresh_initialization', {})
    need(all(fresh.get(k) is True for k in ('all_224_masters_exact_fresh_values', 'optimizer_state_empty', 'scaler_exact_initial'))
         and fresh.get('prior_adapter_loaded') is False and fresh.get('checkpoint_loaded') is False
         and fresh.get('masters_dtype') == 'float32' and fresh.get('master_tensors') == 224
         and fresh.get('parameters') == 1154104 and fresh.get('optimizer_steps') == 0
         and fresh.get('scaler') == dict(scale=1024., growth_factor=2., backoff_factor=.5, growth_interval=2000, _growth_tracker=0),
         'Exact fresh adapter/optimizer/scaler initialization proof missing')
    binding = report['binding']
    required = dict(calibration_sha256=data.sha_file(selected_path), calibration_format=CALIBRATION_FORMAT,
        selected_kind=selected['selected_kind'], table_sha256=selected['table_sha256'],
        candidates_sha256=selected['candidates_sha256'], selection_report_sha256=selected['selection_report_sha256'],
        v4_statistics_sha256=selected['v4_statistics_sha256'], protocol_sha256=PROTOCOL_SHA,
        numeric_protocol_sha256=NUMERIC_PROTOCOL_SHA, source_checkpoint_sha256=runtime.SOURCE_CHECKPOINT_SHA256,
        tokenizer_sha256=runtime.TOKENIZER_SHA256, train_manifest_sha256=selected['train_manifest_sha256'],
        prose_manifest_sha256=selected['prose_manifest_sha256'], prose_tokens_sha256=selected['train_file_sha256'],
        initial_adapter='fresh V=0,g=1,w=0,b=-4; no pretrained adapter or checkpoint',
        fresh_initialization=True, prior_adapter_loaded=False, checkpoint_loaded=False,
        state_mode='sq3p25; exact packed forward; live-mask STE backward',
        teacher='separate unadapted source, S16 per-token carry', adapter=native.FORMAT, successful_updates=1536)
    need(all(binding.get(k) == v for k, v in required.items()), 'Fresh training binding differs')
    for relative, digest in report['code_sha256'].items():
        need(data.sha_file(ROOT/relative) == digest, 'Training code changed: '+relative)
    schedule = torch.randperm(1536, generator=torch.Generator().manual_seed(2026092803)).tolist()
    successful = 0
    need(len(report['history']) == report['attempts'], 'Incomplete fresh training history')
    for attempt, row in enumerate(report['history'], 1):
        need(successful < 1536 and row['attempt'] == attempt and row['schedule_entry'] == schedule[successful]
             and type(row['overflow']) is bool, 'Training successful-update schedule differs')
        successful += int(not row['overflow'])
        need(row['successful_updates'] == successful, 'Training count differs')
    need(successful == 1536, 'Final candidate update count differs')
    export = report['adapter']
    need(Path(export['file']).name == export['file'], 'Unsafe adapter relative path')
    adapter_path = path.parent/export['file']
    need(data.sha_file(adapter_path) == export['sha256'] and adapter_path.stat().st_size == export['bytes']
         and export.get('roundtrip_bitwise_equal') is True and export.get('parameters') == 1154104
         and export.get('gate_mode') == 'soft', 'Final adapter file differs')
    tensors = native.read_fp16(adapter_path, expected_binding=binding)['tensors']
    need(len(tensors) == 224 and {k:native.tensor_hash(v) for k,v in tensors.items()} == export['tensor_sha256'],
         'Final adapter tensors differ')
    final = report['final_checkpoint']
    need(Path(final['file']).name == final['file'], 'Unsafe final checkpoint path')
    checkpoint_path = path.parent/final['file']
    need(data.sha_file(checkpoint_path) == final['sha256'] and checkpoint_path.stat().st_size == final['bytes'],
         'Final checkpoint receipt differs')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    need(checkpoint['format'] == 'MAMBA2_STATE_FIRST_CHECKPOINT_V1' and checkpoint['binding'] == binding
         and checkpoint['successful_updates'] == 1536 and checkpoint['attempts'] == report['attempts'],
         'Final checkpoint identity/count differs')
    masters = checkpoint['masters']
    need(set(masters) == set(tensors) and all(v.dtype == torch.float32 and bool(torch.isfinite(v).all())
         and torch.equal(v.half(), tensors[k]) for k,v in masters.items()), 'Final FP32 master casts differ from export')
    optimizer = checkpoint['optimizer']
    need(len(optimizer['state']) == 224 and all(int(state['step']) == 1536 for state in optimizer['state'].values()),
         'Final Adam state steps differ')
    return report, adapter_path


def export_selection(out, candidates, candidate_receipt, candidates_path, comparison):
    name = comparison['selection']['selected_kind']
    need(name != 'magnitude', 'No redundant old-table training export')
    common = dict(base_binding(), selected_kind=name, candidates_sha256=data.sha_file(candidates_path),
        selection_report_sha256=data.sha_file(out/'screen_comparison.json'),
        table_sha256=candidate_receipt['table_sha256'][name], train_token_hashes=candidates['train_token_hashes'])
    payload = dict(format=CALIBRATION_FORMAT, **common, permutations=candidates['tables'][name].clone())
    path = out/'selected_calibration.pt'
    write_payload(path, payload)
    receipt = dict(format=CALIBRATION_FORMAT, **common, complete=True, fresh_source_no_adapter=True,
        file=path.name, sha256=data.sha_file(path), bytes=path.stat().st_size, code_sha256=code_hashes())
    write_new_json(path.with_suffix('.json'), receipt)
    load_selected_calibration(path, candidates_path)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('screen', 'full'), required=True)
    for name in ('source-dir', 'candidates', 'prose-tokens', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('selected-calibration', 'training-report', 'parent-eval-dir'):
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args()
    need(not args.out.exists(), 'Fresh output directory required')
    check_protocol()
    if args.stage == 'screen':
        need(all(getattr(args, name) is None for name in ('selected_calibration', 'training_report', 'parent_eval_dir')),
             'Screening cannot receive an adapter/training/heldout input')
    else:
        need(all(getattr(args, name) is not None for name in ('selected_calibration', 'training_report', 'parent_eval_dir')),
             'Full evaluation needs frozen selection, final fresh training, and archived controls')
    torch.set_num_threads(8); torch.manual_seed(20260928); torch.cuda.manual_seed_all(20260928)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    candidates, candidate_receipt = load_candidates(args.candidates)
    tokenizer = runtime.SentencePieceTokenizer(args.source_dir)
    need(tokenizer.sha256 == runtime.TOKENIZER_SHA256, 'Tokenizer differs')
    training = adapter_path = selected = None
    archived = {}
    if args.stage == 'screen':
        train = load_train_tokens(args.prose_tokens)
        windows = [(i*2048, train[i, :512]) for i in range(8, 40)]
        cases = [row for row in data.generate_cases('train') if row['sample'] in range(0, 256, 16)]
        need(len(cases) == 96 and sum(len(w)-1 for _,w in windows) == 16352, 'TRAIN screen population differs')
        dataset = dict(split='train', file_sha256=data.sha_file(args.prose_tokens), rows=list(range(8, 40)), tokens_per_row=512)
        arms = [(name, name, False) for name in KINDS] + [('restored_magnitude', 'magnitude', False)]
    else:
        selected, _, selection = load_selected_calibration(args.selected_calibration, args.candidates)
        training, adapter_path = load_fresh_training(args.training_report, selected, args.selected_calibration)
        for name, digest in ARCHIVE_HASHES.items():
            path = args.parent_eval_dir/('full_'+name+'.json')
            need(data.sha_file(path) == digest, 'Archived context changed: '+name)
            archived[name] = read_json(path)
        ids, dataset = load_wikitext_tokens(tokenizer, 'validation')
        windows = ppl_windows(ids, 2048); cases = data.generate_cases('confirm')
        need(len(windows) == 130 and sum(len(w)-1 for _,w in windows) == 264764
             and dataset['token_stream_sha256_int64le'] == VALIDATION_TOKENS_SHA and len(cases) == 768,
             'Full population differs')
        name = selected['selected_kind']
        arms = [('old_magnitude', 'magnitude', False), ('selected_no_adapter', name, False),
                ('selected_resurface', name, True), ('restored_selected', name, False)]
    policy = pin_replay_backend()
    args.out.mkdir(parents=True)
    model = runtime.load_source_model(args.source_dir); frozen = FrozenBase(model)
    need(no_adapter_hooks(model), 'Source unexpectedly has adapter hooks')
    common = dict(format='MAMBA2_STATE_FIRST_EVAL_V1', stage=args.stage, **base_binding(),
        source_checkpoint_sha256=runtime.SOURCE_CHECKPOINT_SHA256,
        candidates_sha256=data.sha_file(args.candidates), candidates_receipt_sha256=data.sha_file(args.candidates.with_suffix('.json')),
        dataset=dataset, environment=runtime.environment_receipt(), code_hashes=code_hashes(), backend_policy=policy,
        engine_kind='old_sq', engine_scope='Unchanged packed16INT8/64INT4/48zero codec; candidate_name/table hash identifies the actual coordinates',
        selected_calibration_sha256=data.sha_file(args.selected_calibration) if selected else None,
        training_report_sha256=data.sha_file(args.training_report) if training else None,
        training_binding=training['binding'] if training else None,
        selection_scope='Unadapted TRAIN only; table frozen before fresh training and validation/CONFIRM')
    common.pop('adapter')
    common.update(heldout_used=args.stage == 'full', heldout_used_for_selection=False,
                  calibration_heldout_used=False)
    results = {}; started = time.time(); parent_replay = None
    for arm, name, adapted in arms:
        print('[state-first-v5 arm] '+arm, flush=True)
        need(no_adapter_hooks(model), 'Adapter hooks leaked between arms')
        path = args.out/(args.stage+'_'+arm+'.json')
        table = candidates['tables'][name]
        table_digest = tensor_sha(table)
        with native.install_fp16(model, adapter_path, expected_binding=training['binding']) if adapted else contextlib.nullcontext() as bank:
            frozen.check()
            if bank:
                need({k:native.tensor_hash(v) for k,v in bank.masters.items()} == training['adapter']['tensor_sha256'],
                     'Installed adapter differs')
            arm_common = dict(common, arm=arm, candidate_name=name, candidate_table_sha256=table_digest,
                adapter_loaded=adapted, adapter_sha256=training['adapter']['sha256'] if adapted else None)
            try:
                probe = cache_probe(model, table, windows[0][1])
                result = evaluate(model, tokenizer, 'old_sq', table, {}, windows, cases, path,
                                  dict(arm_common, persistent_cache_probe=probe))
            except FloatingPointError as error:
                if args.stage != 'screen' or name == 'magnitude':
                    raise
                result = read_json(path) if path.exists() else arm_common
                result.update(complete=False, error=repr(error), excluded_from_selection=True)
            result['frozen_source'] = frozen.check()
            result['backend_policy_check'] = check_replay_backend(policy)
            result['adapter_content_unchanged'] = bank is None or {
                k:native.tensor_hash(v) for k,v in bank.masters.items()} == training['adapter']['tensor_sha256']
            need(result['adapter_content_unchanged'] and tensor_sha(table) == table_digest, 'Adapter or table mutated')
            result['candidate_table_unchanged'] = True
            save_json(path, result); results[arm] = result
        frozen.check()
        need(no_adapter_hooks(model), 'Adapter removal left hooks')
        if args.stage == 'full' and arm == 'old_magnitude':
            parent_replay = check_restoration(archived['source_sq3p25'], result)
            parent_replay['scope'] = 'Exact archived unadapted magnitude SQ3.25 replay'
            save_json(args.out/'full_parent_replay.json', parent_replay)
    before, after = ('magnitude', 'restored_magnitude') if args.stage == 'screen' else ('selected_no_adapter', 'restored_selected')
    restoration = check_restoration(results[before], results[after])
    restoration['scope'] = 'Exact no-adapter replay after candidate/adapter removal'
    save_json(args.out/(args.stage+'_restoration.json'), restoration)
    outcome = dict(format='MAMBA2_STATE_FIRST_COMPARISON_V1', complete=True, stage=args.stage,
        protocol_sha256=PROTOCOL_SHA, candidates_sha256=data.sha_file(args.candidates), adapter_sha256=None,
        code_hashes=code_hashes(), backend_policy=policy, restoration=restoration,
        report_sha256={arm:data.sha_file(args.out/(args.stage+'_'+arm+'.json')) for arm in results},
        elapsed_seconds=time.time()-started)
    if args.stage == 'screen':
        selection = choose_candidate(results)
        outcome.update(selection=selection, selected_table_sha256=candidate_receipt['table_sha256'][selection['selected_kind']])
    else:
        quantizer = compare_pair(results['old_magnitude'], results['selected_no_adapter'])
        repair = compare_pair(results['selected_no_adapter'], results['selected_resurface'])
        prior = compare_pair(archived['resurface_sq3p25'], results['selected_resurface'])
        original = compare_pair(archived['source_s16'], results['selected_resurface'])
        same = all(row['cache']['total_bytes'] == CACHE_BYTES for row in results.values())
        gates = dict(
            unadapted_state_optimization=dict(ppl_at_least_1pct_better=quantizer['ppl_relative_change'] <= -.01, cache_same_budget=same),
            resurface_repair=dict(ppl_no_more_than_1pct_worse=repair['ppl_relative_change'] <= .01,
                normal_mk_increases=repair['normal_mk_correct_delta'] > 0,
                mk_bootstrap_lower_positive=repair['normal_mk_paired_bootstrap_95ci'][0] > 0, cache_same_budget=same),
            prior_endpoint_improvement=dict(ppl_at_least_1pct_better=prior['ppl_relative_change'] <= -.01,
                mk_no_observed_decrease=prior['normal_mk_correct_delta'] >= 0,
                mk_95ci_lower_at_least_minus2pp=prior['normal_mk_paired_bootstrap_95ci'][0] >= -.02, cache_same_budget=same))
        outcome.update(selected_kind=selected['selected_kind'], selection=selection['selection'],
            selected_calibration_sha256=data.sha_file(args.selected_calibration), selection_report_sha256=selected['selection_report_sha256'],
            training_report_sha256=data.sha_file(args.training_report), adapter_sha256=training['adapter']['sha256'],
            archived_report_sha256=ARCHIVE_HASHES, parent_replay=parent_replay,
            comparisons=dict(unadapted_state_optimization=quantizer, resurface_repair=repair,
                             prior_endpoint_improvement=prior, original_s16=original),
            gate_checks=gates, gates_pass={name:all(checks.values()) for name,checks in gates.items()},
            original_ppl_restored_within_1pct=original['ppl_relative_change'] <= .01)
    save_json(args.out/(args.stage+'_comparison.json'), outcome)
    if args.stage == 'screen' and not outcome['selection']['stopped']:
        receipt = export_selection(args.out, candidates, candidate_receipt, args.candidates, outcome)
        print('[frozen selection] '+json.dumps({k:receipt[k] for k in ('selected_kind', 'sha256', 'table_sha256')}), flush=True)
    print(json.dumps(outcome, indent=2), flush=True)


if __name__ == '__main__':
    main()
