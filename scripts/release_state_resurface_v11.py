#!/usr/bin/env python3
"""Build or verify the fixed V11 public inference bundle; standard library only.

This checks already audited evidence. It does not rerun the independent audit
or the GPU quality evaluation. The original NVIDIA checkpoint is not bundled.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
TAG = 'v0.1.0-q325-resurface'
NAME = 'mamba2-8b-q325-resurface-v11'
FORMAT = 'MAMBA2_STATE_RESURFACE_V11_RELEASE_V1'
SOURCE = {
    'repository': 'nvidia/mamba2-8b-3t-4k',
    'revision': 'b915550c63ba9359f88f44d1f6a600d85af27302',
    'checkpoint_sha256': '47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb',
    'tokenizer_sha256': '5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09',
    'runtime_weight_dtype': 'float16',
    'weight_parameters': 8236999680,
    'weight_payload_bytes': 16473999360,
    'included': False,
}
FILES = {
    'adapter_fp16.pt': 'reports/state_resurface_v11_training/adapter_fp16.pt',
    'selected_calibration.pt': 'reports/state_ppl_v10_screen/selected_calibration.pt',
    'selected_calibration.json': 'reports/state_ppl_v10_screen/selected_calibration.json',
    'training_report.json': 'reports/state_resurface_v11_training/report.json',
    'training_audit.json': 'reports/state_resurface_v11_training_audit.json',
    'full_audit.json': 'reports/state_resurface_v11_full_audit.json',
    'full_comparison.json': 'reports/state_resurface_v11_full/full_comparison.json',
    'full_v10_no_adapter.json': 'reports/state_resurface_v11_full/full_v10_no_adapter.json',
    'full_v10_resurface.json': 'reports/state_resurface_v11_full/full_v10_resurface.json',
    'full_restored_v10_no_adapter.json': 'reports/state_resurface_v11_full/full_restored_v10_no_adapter.json',
    'full_parent_replay.json': 'reports/state_resurface_v11_full/full_parent_replay.json',
    'full_restoration.json': 'reports/state_resurface_v11_full/full_restoration.json',
    'LICENSE': 'LICENSE',
    'reference/statequant/LICENSE': 'reference/statequant/LICENSE',
    'docs/THIRD_PARTY_NOTICES.md': 'docs/THIRD_PARTY_NOTICES.md',
    'RELEASE.md': 'docs/STATE_RESURFACE_V11_RELEASE.md',
}
PINNED = {
    'adapter_fp16.pt': '339334b3431027504fb8da4ce3167d63c8c2f309920cfb0076bde1f72a115cb7',
    'selected_calibration.pt': 'a77bbd86ae609d8ef7cf90f10d0904cf95b4df8e059c9ea6cb2fd12886436fb8',
    'selected_calibration.json': '4dbc05cef7f3cd3b5bb43e65c0f25a1befdee2218aef3a476291a43093645b4c',
    'training_report.json': '090c02ee03658b5558afb65664f671a09edb98a698aec8a05b0723932631d02c',
    'training_audit.json': '70d98f85982f01029d45323ec4a5822712297ef66f91f8491b51b6840616fc14',
    'full_audit.json': 'ad2b3f14a075b0d60993f3579e99fc98bfbac8aa527f6a0c8d684cc9493e7d72',
    'full_comparison.json': 'dc8211177d5eb760ae593f2645a52053794af2c043bf2b04b2deba2dfb0464fc',
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def evidence(directory, code_root=ROOT):
    """Verify immutable artifact/report hashes and their cross-file bindings."""
    directory, code_root = Path(directory), Path(code_root)
    for name in FILES:
        path = directory / name
        need(path.is_file() and not path.is_symlink(), 'Missing regular bundle file: ' + name)
    for name, digest in PINNED.items():
        need(sha256(directory / name) == digest, 'Pinned artifact differs: ' + name)
    training = read_json(directory / 'training_report.json')
    audit = read_json(directory / 'full_audit.json')
    comparison = read_json(directory / 'full_comparison.json')
    calibration = read_json(directory / 'selected_calibration.json')
    need(audit['complete'] is True and audit['passed'] is True and audit['stage'] == 'full'
         and audit['cuda_initialized'] is False, 'A passing complete independent audit is required')
    need(comparison['complete'] is True and comparison['quality_gate_pass'] is True
         and all(comparison['quality_checks'].values()), 'Predeclared quality gates did not pass')
    need(audit['full']['comparison_sha256'] == PINNED['full_comparison.json']
         and audit['full']['quality_checks'] == comparison['quality_checks']
         and audit['training_report_sha256'] == PINNED['training_report.json'],
         'Audit/evaluation/training binding differs')
    need(comparison['candidate_adapter_sha256'] == training['adapter']['sha256'] == PINNED['adapter_fp16.pt']
         and comparison['training_report_sha256'] == PINNED['training_report.json']
         and comparison['training_audit_sha256'] == PINNED['training_audit.json'], 'Training export differs')
    need(training['complete'] is True and training['successful_updates'] == 1536
         and training['binding']['calibration_sha256'] == PINNED['selected_calibration.pt']
         and training['binding']['calibration_receipt_sha256'] == PINNED['selected_calibration.json']
         and training['binding']['source_checkpoint_sha256'] == SOURCE['checkpoint_sha256']
         and training['binding']['tokenizer_sha256'] == SOURCE['tokenizer_sha256'], 'Source/layout binding differs')
    need(calibration['selected_layout'] == '32_32_64'
         and calibration['table_sha256'] == training['binding']['table_sha256']
         and calibration['sha256'] == PINNED['selected_calibration.pt'], 'Selected calibration differs')
    for arm, digest in comparison['report_sha256'].items():
        need(sha256(directory / ('full_' + arm + '.json')) == digest, 'Raw arm differs: ' + arm)
    for filename, key in [('full_parent_replay.json', 'parent_replay'), ('full_restoration.json', 'restoration')]:
        need(read_json(directory / filename) == comparison[key], 'Replay receipt differs: ' + filename)
    for name, digest in comparison['code_hashes'].items():
        need(sha256(code_root / name) == digest, 'Measured source file differs: ' + name)
    return training, comparison


def make_manifest(directory, code_root=ROOT):
    training, comparison = evidence(directory, code_root)
    return {
        'format': FORMAT,
        'release_tag': TAG,
        'source': SOURCE,
        'state': {'layout': '32_32_64', 'bits_per_coordinate_including_scales': 3.25,
                  'int8': 32, 'int4': 32, 'zero_carry': 64, 'row_bytes': 52,
                  'batch_one_cache_bytes': 28499968, 'adapter_payload_bytes': 2308208,
                  'cache_plus_adapter_bytes': 30808176,
                  'scope': 'SSM and convolution caches plus one shared static table; source weights, temporaries and allocator reserve excluded'},
        'adapter_binding': training['binding'],
        'comparison': comparison['comparison'],
        'quality_scope': comparison['quality_scope'],
        'selection_scope': comparison['selection_scope'],
        'backend_policy': comparison['backend_policy'],
        'training_environment': training['environment'],
        'measured_code_sha256': comparison['code_hashes'],
        'files': {name: {'sha256': sha256(Path(directory) / name), 'bytes': (Path(directory) / name).stat().st_size}
                  for name in sorted(FILES)},
        'verification_scope': 'Checks the fixed published evidence and exact runtime files; does not rerun GPU quality or independent audit'}


def verify_bundle(directory, code_root=ROOT):
    directory = Path(directory)
    manifest = read_json(directory / 'manifest.json')
    expected = make_manifest(directory, code_root)
    need(manifest == expected, 'Manifest differs from the pinned evidence and actual files')
    entries = sorted([*FILES, 'manifest.json'])
    expected_sums = ''.join(f'{sha256(directory / name)}  {name}\n' for name in entries)
    need((directory / 'SHA256SUMS').read_text() == expected_sums, 'Bundle checksum list differs')
    need(not any(p.is_symlink() for p in directory.rglob('*')), 'Bundle contains a symlink')
    need({str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
         == set(entries) | {'SHA256SUMS'}, 'Unexpected bundle entries')
    return manifest


def build(output_dir, code_root=ROOT):
    output_dir, code_root = Path(output_dir), Path(code_root)
    output_dir.mkdir(parents=True, exist_ok=False)
    bundle = output_dir / NAME
    bundle.mkdir()
    for name, relative in FILES.items():
        source = code_root / relative
        need(source.is_file() and not source.is_symlink(), 'Missing source artifact: ' + relative)
        (bundle / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, bundle / name)
    write_json(bundle / 'manifest.json', make_manifest(bundle, code_root))
    names = sorted([*FILES, 'manifest.json'])
    (bundle / 'SHA256SUMS').write_text(''.join(f'{sha256(bundle / n)}  {n}\n' for n in names))
    verify_bundle(bundle, code_root)
    archive = output_dir / (NAME + '.tar.gz')
    with archive.open('xb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as tar:
            for path in sorted(p for p in bundle.rglob('*') if p.is_file()):
                info = tar.gettarinfo(str(path), arcname=NAME + '/' + str(path.relative_to(bundle)))
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ''
                info.mode = 0o644
                with path.open('rb') as stream:
                    tar.addfile(info, stream)
    receipt = {'format': FORMAT, 'archive': archive.name, 'sha256': sha256(archive),
               'bytes': archive.stat().st_size, 'manifest_sha256': sha256(bundle / 'manifest.json'),
               'bundle_files': sum(p.is_file() for p in bundle.rglob('*')), 'verified': True}
    write_json(output_dir / 'build_receipt.json', receipt)
    (output_dir / 'SHA256SUMS').write_text(f"{receipt['sha256']}  {archive.name}\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--build', type=Path, help='Create a NEW output directory with bundle and deterministic tar.gz')
    mode.add_argument('--verify', type=Path, help='Check an extracted bundle without torch/CUDA')
    args = parser.parse_args()
    if args.build:
        result = build(args.build)
    else:
        manifest = verify_bundle(args.verify)
        result = {'verified': True, 'scope': manifest['verification_scope'],
                  'adapter_sha256': PINNED['adapter_fp16.pt'],
                  'ppl': manifest['comparison']['candidate_ppl'],
                  'normal_mk': manifest['comparison']['candidate_normal_mk']}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
