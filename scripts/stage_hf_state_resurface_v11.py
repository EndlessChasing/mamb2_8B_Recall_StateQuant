#!/usr/bin/env python3
"""Deterministically stage the published GitHub Q3.25 release for Hugging Face.

Local filesystem only: no Git writes, Hub operations, training or GPU work.
The README is supplied separately after model-card review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = 'EndlessChasing/Mamb2_8B_Recall_SQ3.25'
GITHUB_REPO = 'https://github.com/EndlessChasing/mamb2_8B_Recall_StateQuant'
GITHUB_COMMIT = 'be037e1a2635ad7b540e0a1d2e551febf647c346'
GITHUB_TAG = 'v0.1.0-q325-resurface'
ARCHIVE_SHA = '6f5c219f7c92cbfa9541af610970fd015d7005e9257fbb9c7a9ab772e5d4ef80'
ARCHIVE_NAME = 'mamba2-8b-q325-resurface-v11.tar.gz'
ARCHIVE_TOP = 'mamba2-8b-q325-resurface-v11'
ADAPTER_SHA = '339334b3431027504fb8da4ce3167d63c8c2f309920cfb0076bde1f72a115cb7'
FORMAT = 'MAMBA2_STATE_RESURFACE_V11_HF_PAYLOAD_V1'
GENERATED_EXCLUSIONS = ['publish_manifest.json', 'SHA256SUMS']
EXTRA_FILES = [
    'scripts/release_state_resurface_v11.py', 'scripts/infer_state_resurface_v11.py',
    'pyproject.toml', 'LICENSE', 'reference/statequant/LICENSE',
    'docs/THIRD_PARTY_NOTICES.md', 'docs/STATE_RESURFACE_V11_RELEASE.md',
    'docs/STATE_RESURFACE_V11_RESULTS.md', 'docs/STATE_RESURFACE_V11_REPRODUCTION.md',
]
LINK_REWRITE_DOCS = {
    'docs/THIRD_PARTY_NOTICES.md', 'docs/STATE_RESURFACE_V11_RELEASE.md',
    'docs/STATE_RESURFACE_V11_RESULTS.md', 'docs/STATE_RESURFACE_V11_REPRODUCTION.md',
}
GITATTRIBUTES = b'*.pt filter=lfs diff=lfs merge=lfs -text\n'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def git_blob(path):
    return subprocess.check_output(['git', 'show', f'{GITHUB_COMMIT}:{path}'], cwd=ROOT)


def linked_document(path, raw):
    """Only unmeasured presentation docs get absolute upstream links."""
    def replace(match):
        label, target = match.groups()
        if target.startswith(('https://', 'http://', '#', 'mailto:')):
            return match.group(0)
        need(not target.startswith('/'), 'Unexpected absolute local documentation link')
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(path), target))
        need(not resolved.startswith('../'), 'Documentation link escapes repository')
        return f'[{label}]({GITHUB_REPO}/blob/{GITHUB_COMMIT}/{resolved})'
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replace, raw.decode()).encode()


def expected_sources(stage):
    comparison = read_json(stage / 'release/full_comparison.json')
    measured = comparison['code_hashes']
    need(len(measured) == 46, 'Unexpected measured source inventory')
    sources = {}
    for path in sorted(set(measured) | set(EXTRA_FILES)):
        raw = git_blob(path)
        original_sha = sha_bytes(raw)
        if path in measured:
            need(original_sha == measured[path], 'Published code differs from evaluation: ' + path)
        rewrite = path in LINK_REWRITE_DOCS
        need(not (rewrite and path in measured), 'Never rewrite a measured source file')
        payload = linked_document(path, raw) if rewrite else raw
        sources[path] = (payload, {
            'source': 'github_commit', 'source_path': path, 'source_commit': GITHUB_COMMIT,
            'source_sha256': original_sha,
            'transformation': 'relative Markdown links rewritten to exact GitHub commit' if rewrite else 'byte-identical',
        })
    return sources


def stage_release(stage, archive):
    need(sha_file(archive) == ARCHIVE_SHA, 'GitHub release archive SHA256 differs')
    stage.mkdir(parents=True, exist_ok=False)
    (stage / 'release').mkdir()
    names = set()
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar.getmembers():
            name = PurePosixPath(member.name)
            need(member.isfile() and not name.is_absolute() and '..' not in name.parts
                 and len(name.parts) >= 2 and name.parts[0] == ARCHIVE_TOP,
                 'Unsafe release archive entry')
            relative = str(PurePosixPath(*name.parts[1:]))
            need(relative not in names, 'Duplicate archive member')
            names.add(relative)
            output = stage / 'release' / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(tar.extractfile(member).read())
    need(len(names) == 18, 'Expected exactly eighteen unmodified release files')
    for path, (raw, _) in expected_sources(stage).items():
        output = stage / path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(raw)
    (stage / '.gitattributes').write_bytes(GITATTRIBUTES)


def verify_release_source(stage, archive):
    """Recheck every extracted byte against the pinned public archive."""
    need(sha_file(archive) == ARCHIVE_SHA, 'GitHub release archive SHA256 differs')
    expected = set()
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar.getmembers():
            name = PurePosixPath(member.name)
            need(member.isfile() and not name.is_absolute() and '..' not in name.parts
                 and len(name.parts) >= 2 and name.parts[0] == ARCHIVE_TOP,
                 'Unsafe release archive entry')
            relative = str(PurePosixPath(*name.parts[1:]))
            need(relative not in expected, 'Duplicate release archive member')
            expected.add(relative)
            path = stage / 'release' / relative
            need(path.is_file() and not path.is_symlink()
                 and path.read_bytes() == tar.extractfile(member).read(),
                 'Extracted public release file differs: ' + relative)
    actual = {str(p.relative_to(stage / 'release')) for p in (stage / 'release').rglob('*') if p.is_file()}
    need(len(expected) == 18 and actual == expected, 'Public release member inventory differs')


def validate_and_manifest(stage, card=None):
    stage = Path(stage)
    if card is not None:
        raw = Path(card).read_bytes()
        need(raw.startswith(b'---\n') and REPO_ID.encode() in raw, 'Expected reviewed HF card for exact target ID')
        (stage / 'README.md').write_bytes(raw)
    sources = expected_sources(stage)
    for path, (raw, _) in sources.items():
        need((stage / path).read_bytes() == raw, 'Staged source differs: ' + path)
    need((stage / '.gitattributes').read_bytes() == GITATTRIBUTES, 'LFS attributes differ')
    need(sha_file(stage / 'release/adapter_fp16.pt') == ADAPTER_SHA, 'Adapter payload differs')
    result = subprocess.run([
        sys.executable, '-B', 'scripts/infer_state_resurface_v11.py',
        '--bundle', 'release', '--verify-only'], cwd=stage, text=True, capture_output=True, check=True)
    verification = json.loads(result.stdout)
    need(verification['verified'] is True, 'Staged inference verification failed')
    release_manifest = read_json(stage / 'release/manifest.json')
    release_files = set(release_manifest['files']) | {'manifest.json', 'SHA256SUMS'}
    allowed = set(sources) | {'release/' + p for p in release_files} | {'.gitattributes'}
    if (stage / 'README.md').exists():
        allowed.add('README.md')
    found = set()
    for path in stage.rglob('*'):
        need(not path.is_symlink(), 'Symlink in publication payload')
        if path.is_file():
            found.add(str(path.relative_to(stage)))
    need(found - set(GENERATED_EXCLUSIONS) == allowed, 'Unexpected payload files; reject caches or unrelated data')
    files = {}
    for path in sorted(allowed):
        actual = stage / path
        entry = {'sha256': sha_file(actual), 'bytes': actual.stat().st_size}
        if path in sources:
            entry.update(sources[path][1])
        elif path.startswith('release/'):
            entry.update(source='github_release_asset', archive_sha256=ARCHIVE_SHA,
                         archive_member=ARCHIVE_TOP + '/' + path.removeprefix('release/'))
        else:
            entry.update(source='reviewed_hf_model_card' if path == 'README.md' else 'generated_lfs_attributes')
        files[path] = entry
    manifest = {
        'format': FORMAT, 'target_repo_id': REPO_ID,
        'github_repository': GITHUB_REPO, 'github_tag': GITHUB_TAG,
        'github_commit': GITHUB_COMMIT, 'github_archive_name': ARCHIVE_NAME,
        'github_archive_sha256': ARCHIVE_SHA,
        'adapter_sha256': ADAPTER_SHA,
        'source_model': release_manifest['source'],
        'readme_present': 'README.md' in allowed,
        'ready_for_upload': 'README.md' in allowed,
        'file_count_excluding_manifest_and_root_checksums': len(files),
        'bytes_excluding_manifest_and_root_checksums': sum(v['bytes'] for v in files.values()),
        'files': files,
        'file_inventory_exclusions': GENERATED_EXCLUSIONS,
        'exclusion_reason': 'Self-containing hashes are circular; root SHA256SUMS covers publish_manifest.json and every listed file, and excludes only itself.',
        'checks': {'exact_github_release_members': 18, 'exact_measured_source_files': 46,
                   'staged_verify_only_passed': True,
                   'no_source_weights_or_training_data_required_for_verification': True,
                   'new_gpu_evaluation_performed': False},
        'scope': 'Published GitHub runtime, final state table and Resurface adapter plus reviewed Hub presentation. Original NVIDIA source checkpoint is obtained separately.',
    }
    write_json(stage / 'publish_manifest.json', manifest)
    all_names = sorted([*files, 'publish_manifest.json'])
    (stage / 'SHA256SUMS').write_text(''.join(f'{sha_file(stage / p)}  {p}\n' for p in all_names))
    receipt = {
        'format': FORMAT, 'target_repo_id': REPO_ID, 'ready_for_upload': manifest['ready_for_upload'],
        'file_count': len(files) + 2,
        'total_file_bytes': sum((stage / p).stat().st_size for p in [*files, *GENERATED_EXCLUSIONS]),
        'publish_manifest_sha256': sha_file(stage / 'publish_manifest.json'),
        'root_sha256sums_sha256': sha_file(stage / 'SHA256SUMS'),
        'github_archive_sha256': ARCHIVE_SHA,
        'checks': manifest['checks'], 'verify_only_output': verification,
    }
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--build', type=Path, help='Create a new staging directory')
    mode.add_argument('--finalize', type=Path, help='Verify existing staged payload and refresh its manifest/checksums')
    parser.add_argument('--archive', type=Path, default=ROOT / 'artifacts/release_v11_public_final' / ARCHIVE_NAME)
    parser.add_argument('--card', type=Path, help='Reviewed card to copy to payload README.md')
    parser.add_argument('--receipt', type=Path, help='Optional receipt OUTSIDE the publication directory')
    args = parser.parse_args()
    stage = (args.build or args.finalize).resolve()
    if args.receipt:
        need(not args.receipt.resolve().is_relative_to(stage), 'Keep staging audit receipts outside publication payload')
    if args.build:
        stage_release(stage, args.archive)
    verify_release_source(stage, args.archive)
    receipt = validate_and_manifest(stage, args.card)
    if args.receipt:
        write_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
