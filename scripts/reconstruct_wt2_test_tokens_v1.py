#!/usr/bin/env python3
"""Recreate pinned NVIDIA-tokenized WT2 test audit input without model/GPU use.

Works with the source tokenizer or a redistributed identical tokenizer. Raw
token IDs are local audit input and are not part of the public evidence overlay.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

sys.dont_write_bytecode = True
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
TOKENIZER_SHA = "5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09"
TEXT_SHA = "696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83"
TOKEN_SHA = "5b82bd46e833e77fcfc0af62bafeaac62e70e68cfdf214d375f0b7b132d4b608"
TOKENIZER_NAME = "mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True,
                        help="Pinned tokenizer file or its containing directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protected-dir", type=Path, action="append", default=[],
                        help="Reject output inside this directory (repeatable)")
    args = parser.parse_args()
    need(args.out.name == "tokens.int64le" and not args.out.exists() and
         not args.out.is_symlink() and all(not args.out.resolve().is_relative_to(p.resolve())
                                          for p in args.protected_dir),
         "Fresh tokens.int64le outside protected directories required")
    tokenizer = args.tokenizer
    if tokenizer.is_dir():
        tokenizer = tokenizer / TOKENIZER_NAME
    need(hashlib.sha256(tokenizer.read_bytes()).hexdigest() == TOKENIZER_SHA,
         "Pinned source tokenizer hash differs")
    from datasets import load_dataset
    import sentencepiece as spm
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                           split="test", revision=REVISION)
    text = "\n\n".join(dataset["text"])
    need(hashlib.sha256(text.encode()).hexdigest() == TEXT_SHA,
         "Pinned official test text differs")
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer))
    ids = processor.encode_as_ids(text)
    need(len(ids) == 300964 and all(0 <= value < 256000 for value in ids),
         "Pinned complete token population differs")
    raw = b"".join(struct.pack("<q", value) for value in ids)
    need(hashlib.sha256(raw).hexdigest() == TOKEN_SHA, "Pinned test token hash differs")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as stream:
        stream.write(raw)
    print(json.dumps(dict(complete=True, tokens=len(ids), target_tokens=len(ids)-1,
                          bytes=len(raw), sha256=TOKEN_SHA, cuda_initialized=False,
                          model_logits_recomputed=False), indent=2))


if __name__ == "__main__":
    main()
