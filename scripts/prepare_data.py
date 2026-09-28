#!/usr/bin/env python3
"""Prepare one provenance-bound numeric split without initializing CUDA."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba2_recall import resurface_data as data
from mamba2_recall.runtime import SentencePieceTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('train', 'dev', 'confirm'), required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT/'training_data'/'numeric_v1')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES= for CPU data preparation')
    protocol = ROOT/'docs'/'PROTOCOL.md'
    tokenizer = SentencePieceTokenizer(args.source_dir)
    manifest = data.prepare_split(args.data_root, args.split, tokenizer,
        protocol_path=protocol, protocol_sha256=data.sha_file(protocol),
        tokenizer_sha256=data.TOKENIZER_SHA)
    path = args.data_root/args.split/'manifest.json'
    loaded, _, _ = (data.load_training(args.data_root, data.sha_file(path), tokenizer)
                    if args.split == 'train' else
                    data.load_evaluation(args.data_root, args.split, data.sha_file(path), tokenizer))
    if loaded != manifest:
        raise RuntimeError('Prepared split failed exact readback')
    print(json.dumps({'split': args.split, 'rows': manifest['row_count'],
                      'manifest_sha256': data.sha_file(path),
                      'protocol_sha256': data.sha_file(protocol),
                      'cuda_initialized': False}), flush=True)


if __name__ == '__main__':
    main()
