#!/usr/bin/env python3
"""Regenerate the exact historical 448 WikiText TRAIN token windows by content."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mamba2_recall import calibration, runtime

MANIFEST_SHA='facb2ca461615a4199781bd21784d642d6674f5b862641b3b9edac3fb499b89d'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES= to keep this CPU-only')
    if args.out.exists():
        raise FileExistsError(args.out)
    manifest_path=ROOT/'docs'/'prose_train_manifest.json'
    if runtime.sha256_file(manifest_path)!=MANIFEST_SHA:
        raise ValueError('Historical prose selection manifest changed')
    manifest=json.loads(manifest_path.read_text())
    tokenizer=runtime.SentencePieceTokenizer(args.source_dir)
    ids,dataset=calibration.load_wikitext_tokens(tokenizer,'train')
    starts=manifest['training_starts']
    if (len(starts)!=448 or len(ids)!=2533678 or dataset!=manifest['dataset']
            or len(set(starts))!=448 or any(type(s) is not int or s<0 or s+2048>len(ids) for s in starts)):
        raise ValueError('Historical corpus/windows identity differs')
    tokens=torch.stack([ids[start:start+2048] for start in starts])
    content_sha=runtime.token_digest(tokens.flatten().numpy())
    if content_sha!=manifest['training_tokens_sha256_int64le']:
        raise ValueError('Recreated TRAIN token content differs')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('xb') as stream:
        torch.save(tokens,stream)
    roundtrip=torch.load(args.out,map_location='cpu',weights_only=True)
    if not torch.equal(roundtrip,tokens):
        raise RuntimeError('Serialized TRAIN windows differ')
    print(json.dumps({'windows':448,'tokens':int(tokens.numel()),
        'content_sha256':content_sha,'file_sha256':runtime.sha256_file(args.out),
        'historical_file_sha256':manifest['training_tokens_file_sha256'],
        'identical_file':runtime.sha256_file(args.out)==manifest['training_tokens_file_sha256']}))


if __name__=='__main__':main()
