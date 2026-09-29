#!/usr/bin/env python3
"""Greedy batch-one inference with the fixed released V11 Q3.25 state adapter.

No training corpora or historical checkpoint directories are required.
Uses the unchanged, audited V10 packed runtime and native FP16 adapter hooks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from release_state_resurface_v11 import verify_bundle, need, PINNED


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path)
    parser.add_argument('--prompt')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--verify-only', action='store_true', help='Verify release bytes and measured code without CUDA')
    args = parser.parse_args()
    manifest = verify_bundle(args.bundle)
    if args.verify_only:
        print(json.dumps({'verified': True, 'scope': manifest['verification_scope'],
                          'adapter_sha256': PINNED['adapter_fp16.pt']}, indent=2))
        return
    if args.source_dir is None or not args.prompt or args.max_new_tokens < 1:
        parser.error('Inference requires --source-dir, a nonempty --prompt and positive --max-new-tokens')

    import torch
    from mamba2_recall import runtime, resurface_native as native
    from evaluate_resurface_more import pin_replay_backend, check_replay_backend
    from state_ppl_codec_v10 import StatePPLQuantV10
    need(torch.cuda.is_available(), 'CUDA with the recorded Mamba/Triton stack is required')
    torch.set_num_threads(8)
    torch.manual_seed(20260929)
    torch.cuda.manual_seed_all(20260929)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    backend = pin_replay_backend()
    for key in ('package_versions', 'external_source_sha256', 'selected_config', 'precision_flags', 'determinism_environment'):
        need(backend[key] == manifest['backend_policy'][key],
             'Runtime differs from the measured backend policy: ' + key)
    tokenizer = runtime.SentencePieceTokenizer(args.source_dir)
    prompt_ids = tokenizer.encode(args.prompt)
    need(0 < len(prompt_ids) <= 4096, 'The prompt must contain 1 to 4096 tokens')
    selected = torch.load(args.bundle / 'selected_calibration.pt', map_location='cpu', weights_only=True)
    table = selected['permutations']
    need(selected['selected_layout'] == '32_32_64' and table.dtype == torch.uint8
         and tuple(table.shape) == (56, 8, 128)
         and native.tensor_hash(table) == manifest['adapter_binding']['table_sha256'], 'Packed state table differs')
    model = runtime.load_source_model(args.source_dir, dtype=torch.float16)
    generated = []
    with torch.inference_mode(), native.install_fp16(
        model, args.bundle / 'adapter_fp16.pt', expected_binding=manifest['adapter_binding']
    ) as bank:
        with StatePPLQuantV10(model, table, layout='32_32_64') as execution:
            hidden = execution.backbone(torch.tensor([prompt_ids], device='cuda'), reset=True)[:, -1:]
            for step in range(args.max_new_tokens):
                logits = model.lm_head(hidden)
                need(bool(torch.isfinite(logits).all()), 'Nonfinite generation logits')
                token = int(logits.argmax(-1).item())
                generated.append(token)
                if token == tokenizer.eos_token_id or step + 1 == args.max_new_tokens:
                    break
                hidden = execution.backbone(torch.tensor([[token]], device='cuda'), reset=False)
            execution.assert_finite_cache()
            cache = execution.cache_breakdown()
            need(cache['total_bytes'] == 28499968, 'Packed cache accounting differs')
            resident = sum(v.untyped_storage().nbytes() for v in bank.masters.values())
            need(resident == 2308208, 'Adapter parameter storage differs')
            check_replay_backend(backend)
    print(json.dumps({'prompt': args.prompt, 'completion': tokenizer.decode(generated),
                      'generated_token_ids': generated, 'prompt_tokens': len(prompt_ids),
                      'adapter_sha256': PINNED['adapter_fp16.pt'],
                      'batch_one_cache_bytes': cache['total_bytes'], 'adapter_storage_bytes': resident,
                      'weight_payload_bytes': manifest['source']['weight_payload_bytes'],
                      'memory_scope': 'Persistent tensor payload only; excludes temporaries and allocator reserve'},
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
