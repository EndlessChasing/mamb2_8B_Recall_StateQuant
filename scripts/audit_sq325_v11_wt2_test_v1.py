#!/usr/bin/env python3
"""Stdlib-only audit of SQ3.25/V11 saved full test coverage and score bindings."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from release_state_resurface_v11 import verify_bundle, PINNED
ARMS = ('without_resurface', 'resurface')
TOKEN_SHA = '5b82bd46e833e77fcfc0af62bafeaac62e70e68cfdf214d375f0b7b132d4b608'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def pairs(values):
        result = {}
        for k, v in values:
            need(k not in result, 'Duplicate key: ' + k)
            result[k] = v
        return result
    def bad(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=bad)


def audit(args):
    manifest = verify_bundle(args.bundle)
    comp = read(args.comparison)
    need(comp['format'] == 'SQ325_V11_WT2_TEST_V1' and comp['complete'] is True and 'error' not in comp,
         'Completed fixed test run required')
    need(comp['checkpoint_frozen_before_test'] is True and comp['test_used_for_training_or_selection'] is False and
         comp['adapter_removal_reset_and_cache_exact'] is True and
         comp['bundle_manifest_sha256'] == sha(args.bundle / 'manifest.json') and
         comp['release_tag'] == manifest['release_tag'] and
         comp['github_commit'] == 'be037e1a2635ad7b540e0a1d2e551febf647c346' and
         comp['hf_commit'] == 'ebd2ca7cc644ba0e11e4b1595ba03afb8d27e6e4', 'Fixed publication bindings differ')
    need(comp['runner_sha256'] == sha(ROOT / 'scripts/run_sq325_v11_wt2_test_v1.py') and
         comp['protocol_sha256'] == sha(ROOT / 'docs/SQ325_V11_WT2_TEST_V1_PROTOCOL.md') and
         comp['released_ppl_loop_sha256'] == sha(ROOT / 'scripts/run_state_ppl_v10.py') and
         comp['measured_source_sha256'] == manifest['measured_code_sha256'], 'Measured source binding differs')
    train = read(args.bundle / 'training_report.json')
    need(comp['expected_adapter'] == train['adapter'] and train['adapter']['sha256'] == PINNED['adapter_fp16.pt'] and
         comp['table_sha256'] == manifest['adapter_binding']['table_sha256'] and
         comp['state_layout'] == '32_32_64' and comp['memory'] == manifest['state'], 'State/adapter binding differs')
    raw = (args.comparison.parent / 'tokens.int64le').read_bytes()
    need(len(raw) == 300964 * 8 and hashlib.sha256(raw).hexdigest() == TOKEN_SHA and
         comp['tokens_file'] == dict(file='tokens.int64le', bytes=len(raw), sha256=TOKEN_SHA) and
         all(0 <= v[0] < 256000 for v in struct.iter_unpack('<q', raw)), 'Full test tokens differ')
    dataset = comp['dataset']
    expected_dataset = dict(dataset='Salesforce/wikitext', configuration='wikitext-2-raw-v1', split='test',
        revision_argument='b08601e04326c79dfdd32d625aee71d232d685c3',
        tokenizer_sha256=manifest['source']['tokenizer_sha256'], automatic_special_tokens=False,
        document_join='two newline characters', token_stream_sha256_int64le=TOKEN_SHA, total_tokens=300964,
        text_sha256='696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83')
    need(all(dataset.get(k) == v for k, v in expected_dataset.items()), 'Pinned test dataset identity differs')
    identities = [dict(start=s, target_tokens=min(2048, 300963-s),
        token_sha256_int64le=hashlib.sha256(raw[s*8:(s+min(2048,300963-s)+1)*8]).hexdigest())
        for s in range(0, 300963, 2048)]
    need(comp['windowing'] == dict(targets_per_full_window=2048, logits_chunk_tokens=64,
        state_reset_each_window=True, state_requantized_each_token=True, final_partial_included=True,
        window_count=147, target_tokens=300963), 'Coverage protocol differs')
    expected = comp['expected_weight_hashes']
    need(len(expected) == 507 and comp['final_weight_sha256'] == expected and
         comp['source_receipt']['source_checkpoint_sha256'] == manifest['source']['checkpoint_sha256'],
         'Source/frozen507 binding differs')
    for k in ('package_versions', 'external_source_sha256', 'selected_config', 'precision_flags', 'determinism_environment'):
        need(comp['backend_policy'][k] == manifest['backend_policy'][k], 'Released backend differs: ' + k)
    need(set(comp['reports']) == set(ARMS) and set(comp['ppl']) == set(ARMS), 'Two complete fixed arms required')
    for arm in ARMS:
        info = comp['reports'][arm]
        need(info['file'] == arm + '.json', 'Unsafe arm filename')
        p = args.comparison.parent / info['file']
        need(sha(p) == info['sha256'], 'Arm hash differs')
        row = read(p)
        adapted = arm == 'resurface'
        need(row['complete'] is True and row['format'] == comp['format'] and row['arm'] == arm and
             row['dataset'] == dataset and row['adapter_loaded'] is adapted and
             row['actual_weight_sha256'] == expected and row['native_restoration'] is True and
             row['frozen_base_check']['identity_version_gradients_unchanged'] is True and
             row['runtime_table_unchanged'] is True and row['allocation_storage_validated'] is True,
             'Complete frozen arm differs: ' + arm)
        score = row['ppl']
        need(len(score['windows']) == 147, 'Missing test windows')
        total = 0.
        for identity, w in zip(identities, score['windows']):
            need({k: w[k] for k in identity} == identity and math.isfinite(w['nll']) and w['nll'] >= 0 and
                 w['ppl'] == math.exp(w['nll']/w['target_tokens']), 'Window coverage/arithmetic differs')
            total += w['nll']
        need(score['target_tokens'] == 300963 and score['nll'] == total and
             score['ppl'] == math.exp(total/300963) == info['ppl'] == comp['ppl'][arm], 'Pooled PPL differs')
        need(row['cache']['total_bytes'] == 28499968 and row['cache']['row_bytes'] == 52 and
             row['cache']['tokens_per_layer'] == [1955]*56, 'Packed final state accounting differs')
        desc = row['storage_descriptor']
        need(desc['layout'] == '32_32_64' and desc['row_bytes'] == 52 and len(desc['layers']) == 56,
             'Actual storage geometry differs')
        for layer in desc['layers']:
            need(layer['state_bytes'] == 128*64*52 and layer['conv_storage_bytes'] == 81920 and
                 set(layer['tensors']) == {'lo','hi','q4','s8','s4'}, 'Packed layer storage differs')
        probe = row['repeated_reset_probe']
        need(probe['hidden_and_cache_exact'] is True and probe['tokens'] == 128 and
             probe['token_sha256_int64le'] == hashlib.sha256(raw[:128*8]).hexdigest(), 'Repeated reset differs')
        storage = row['adapter_storage']
        need(storage['loaded'] is adapted and storage['resident_storage_bytes'] == (2308208 if adapted else 0) and
             storage['cache_plus_adapter_bytes'] == (30808176 if adapted else 28499968) and
             row['adapter_sha256'] == (train['adapter']['sha256'] if adapted else None), 'Adapter residency differs')
        if adapted:
            need(storage['tensor_sha256'] == train['adapter']['tensor_sha256'], '224 adapter hashes differ')
        need(row['backend_check']['singleton_config_unchanged'] is True and
             row['backend_check']['selected_config'] == comp['backend_policy']['selected_config'], 'Arm backend changed')
    need(comp['ppl_relative_change'] == comp['ppl']['resurface']/comp['ppl']['without_resurface']-1,
         'Paired PPL comparison differs')
    return dict(complete=True, passed=True, format='SQ325_V11_WT2_TEST_CPU_AUDIT_V1',
        uses_stdlib_only=True, cuda_initialized=False, input_report_sha256=sha(args.comparison),
        auditor_sha256=sha(__file__), runner_sha256=comp['runner_sha256'], protocol_sha256=comp['protocol_sha256'],
        bundle_manifest_sha256=comp['bundle_manifest_sha256'], ppl=comp['ppl'],
        target_tokens=300963, window_count=147, last_window_targets=1955,
        all_target_tokens_scored_once=True, all_window_token_hashes_verified=True,
        pooled_nll_ppl_arithmetic_verified=True, recorded_507_weight_and_224_adapter_bindings_verified=True,
        actual_state_storage_receipts_verified=True, adapter_removal_reset_and_cache_exact=True,
        scope='CPU arithmetic/token coverage and published bindings of recorded GPU identities. '
              'GPU logits, source pretraining contamination and cross-split duplicates not independently recomputed.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('comparison', 'bundle', 'out'):
        p.add_argument('--'+n, type=Path, required=True)
    args = p.parse_args()
    need(not args.out.exists(), 'Fresh audit output required')
    receipt = audit(args)
    args.out.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
