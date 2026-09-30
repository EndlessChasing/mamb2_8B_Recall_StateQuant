#!/usr/bin/env python3
"""Evaluate the fixed public SQ3.25/V11 adapter with its original tokenwise codec."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
PROTOCOL = ROOT / 'docs/SQ325_V11_WT2_TEST_V1_PROTOCOL.md'
FORMAT = 'SQ325_V11_WT2_TEST_V1'
ARMS = ('without_resurface', 'resurface')
STREAM = '5b82bd46e833e77fcfc0af62bafeaac62e70e68cfdf214d375f0b7b132d4b608'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def put(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def weight_hashes(model, native, expected=None):
    values = model.state_dict()
    need(len(values) == 507 and sum(v.numel() * v.element_size() for v in values.values()) == 16473999360,
         'Frozen FP16 base geometry differs')
    actual = {}
    for name, v in values.items():
        need(str(v.dtype) == 'torch.float16' and not v.requires_grad and not v.is_meta,
             'Base dtype/frozen flag differs: ' + name)
        actual[name] = native.tensor_hash(v)
    if expected is not None:
        need(actual == expected, 'Actual base bytes changed')
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    need(not args.out_dir.exists() and not args.out_dir.resolve().is_relative_to(args.bundle.resolve()),
         'Fresh output outside sealed bundle required')
    from release_state_resurface_v11 import verify_bundle, PINNED
    manifest = verify_bundle(args.bundle)
    from mamba2_recall import runtime, resurface_native as native
    from mamba2_recall.calibration import load_wikitext_tokens
    from mamba2_recall.evaluation import ppl_windows
    from evaluate_quant_first import FrozenBase
    from evaluate_resurface_more import pin_replay_backend, check_replay_backend
    from state_ppl_codec_v10 import StatePPLQuantV10
    import run_state_ppl_v10 as v10
    from run_state_ppl_v6 import cache_identity
    from run_state_resurface_v11 import adapter_storage
    import torch
    need(torch.cuda.is_available(), 'Recorded CUDA stack required')
    args.out_dir.mkdir(parents=True)
    path = args.out_dir / 'comparison.json'
    started = time.time()
    comp = dict(format=FORMAT, complete=False, release_tag=manifest['release_tag'],
                github_commit='be037e1a2635ad7b540e0a1d2e551febf647c346',
                hf_commit='ebd2ca7cc644ba0e11e4b1595ba03afb8d27e6e4', bundle_manifest_sha256=sha(args.bundle / 'manifest.json'),
                runner_sha256=sha(__file__), protocol_sha256=sha(PROTOCOL),
                checkpoint_frozen_before_test=True, test_used_for_training_or_selection=False,
                project_test_text_seen_in_earlier_experiments=True, reports={})
    put(path, comp)
    try:
        torch.set_num_threads(8)
        torch.manual_seed(20260929)
        torch.cuda.manual_seed_all(20260929)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        policy = pin_replay_backend()
        for k in ('package_versions', 'external_source_sha256', 'selected_config', 'precision_flags', 'determinism_environment'):
            need(policy[k] == manifest['backend_policy'][k], 'Released backend differs: ' + k)
        comp['backend_policy'] = policy
        tokenizer = runtime.SentencePieceTokenizer(args.source_dir)
        ids, dataset = load_wikitext_tokens(tokenizer, 'test')
        need(len(ids) == 300964 and dataset['split'] == 'test' and
             dataset['token_stream_sha256_int64le'] == STREAM, 'Frozen test population differs')
        windows = ppl_windows(ids, 2048)
        need(len(windows) == 147 and sum(len(w)-1 for _, w in windows) == 300963, 'Target coverage differs')
        tokens_path = args.out_dir / 'tokens.int64le'
        tokens_path.write_bytes(ids.numpy().astype('<i8', copy=False).tobytes())
        selected = torch.load(args.bundle / 'selected_calibration.pt', map_location='cpu', weights_only=True)
        table = selected['permutations']
        need(selected['selected_layout'] == '32_32_64' and table.dtype == torch.uint8 and
             tuple(table.shape) == (56, 8, 128) and
             native.tensor_hash(table) == manifest['adapter_binding']['table_sha256'], 'Published table differs')
        train = json.loads((args.bundle / 'training_report.json').read_text())
        comp.update(dataset=dataset, expected_adapter=train['adapter'],
                    table_sha256=native.tensor_hash(table), state_layout='32_32_64',
                    tokens_file=dict(file=tokens_path.name, bytes=tokens_path.stat().st_size, sha256=sha(tokens_path)),
                    windowing=dict(targets_per_full_window=2048, logits_chunk_tokens=64,
                                   state_reset_each_window=True, state_requantized_each_token=True,
                                   final_partial_included=True, window_count=147, target_tokens=300963),
                    measured_source_sha256=manifest['measured_code_sha256'],
                    released_ppl_loop_sha256=sha(ROOT / 'scripts/run_state_ppl_v10.py'))
        put(path, comp)
        model = runtime.load_source_model(args.source_dir, dtype=torch.float16)
        expected = weight_hashes(model, native)
        comp['expected_weight_hashes'] = expected
        comp['source_receipt'] = model._package_receipt
        frozen = FrozenBase(model)
        snapshot = [(m, m.forward, dict(m._forward_pre_hooks), dict(m._forward_hooks))
                    for layer in model.backbone.layers for m in (layer.mixer, layer.mixer.norm)]
        spec = v10.candidate_spec('32_32_64', dict(candidate_order=list(v10.LAYOUTS)))
        rows = {}
        for arm in ARMS:
            adapted = arm == 'resurface'
            manager = native.install_fp16(model, args.bundle / 'adapter_fp16.pt',
                       expected_binding=manifest['adapter_binding']) if adapted else contextlib.nullcontext()
            common = dict(format=FORMAT, arm=arm, adapter_loaded=adapted, dataset=dataset,
                          adapter_sha256=PINNED['adapter_fp16.pt'] if adapted else None)
            arm_path = args.out_dir / (arm + '.json')
            with manager as bank:
                row = v10.evaluate(model, table, spec, windows, arm_path, common)
                row['adapter_storage'] = adapter_storage(bank, train['adapter'])
                row['frozen_base_check'] = frozen.check()
                row['actual_weight_sha256'] = weight_hashes(model, native, expected)
                row['backend_check'] = check_replay_backend(policy)
            need(all(m.forward == f and dict(m._forward_pre_hooks) == pre and dict(m._forward_hooks) == post
                     for m, f, pre, post in snapshot), 'Native forwards/hooks not restored')
            row['native_restoration'] = True
            put(arm_path, row)
            rows[arm] = row
            comp['reports'][arm] = dict(file=arm_path.name, sha256=sha(arm_path), ppl=row['ppl']['ppl'])
            put(path, comp)
        with torch.inference_mode(), StatePPLQuantV10(model, table, layout='32_32_64') as execution:
            hidden = execution.backbone(windows[0][1][:128].cuda()[None], reset=True)
            probe = rows['without_resurface']['repeated_reset_probe']
            need(native.tensor_hash(hidden) == probe['hidden_sha256'] and
                 cache_identity(execution) == probe['cache_tensor_sha256'], 'Adapter removal changed output/cache')
        comp.update(complete=True, adapter_removal_reset_and_cache_exact=True,
                    final_weight_sha256=weight_hashes(model, native, expected),
                    backend_final_check=check_replay_backend(policy),
                    ppl={a: rows[a]['ppl']['ppl'] for a in ARMS},
                    ppl_relative_change=rows['resurface']['ppl']['ppl']/rows['without_resurface']['ppl']['ppl']-1,
                    memory=manifest['state'], elapsed_seconds=time.time()-started)
        put(path, comp)
        print(json.dumps({k: comp[k] for k in ('complete', 'ppl', 'ppl_relative_change', 'elapsed_seconds')}, indent=2), flush=True)
    except Exception as e:
        comp.update(complete=False, error=str(e), traceback=traceback.format_exc(), elapsed_seconds=time.time()-started)
        put(path, comp)
        raise


if __name__ == '__main__':
    main()
