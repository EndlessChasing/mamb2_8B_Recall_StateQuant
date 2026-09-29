#!/usr/bin/env python3
"""Bounded first-window diagnostic; never selects or changes an adapter."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from mamba2_recall import runtime, resurface_native as native
from mamba2_recall.calibration import load_wikitext_tokens
from mamba2_recall.evaluation import ppl_windows
from mamba2_recall.state_quant import StateQuant

torch.set_num_threads(8)
torch.manual_seed(20260928)
torch.cuda.manual_seed_all(20260928)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
source = Path('/home/horde/Mamba2-8B-E8W5/models/source')
tokenizer = runtime.SentencePieceTokenizer(source)
ids, _ = load_wikitext_tokens(tokenizer, 'validation')
window = ppl_windows(ids, 2048)[0][1]
perms = torch.load(ROOT/'artifacts/quant_first_v2_calibration/calibration.pt', map_location='cpu', weights_only=True)['permutations']
parent = ROOT/'artifacts/quant_first_v2_training/adapter_fp16.pt'
model = runtime.load_source_model(source)
archive = {name:json.loads((ROOT/f'artifacts/quant_first_v2_eval/full_{name}.json').read_text())['ppl']['windows'][0]['nll']
           for name in ('source_s16', 'source_sq3p25', 'resurface_sq3p25')}
results = []

@torch.inference_mode()
def probe(label, mode, reference):
    with StateQuant(model, mode, perms if mode == 'sq3p25' else None) as execution:
        tokens = window.cuda()
        hidden = execution.backbone(tokens[:-1][None], reset=True)
        nll = 0.
        for pos in range(0, hidden.shape[1], 64):
            end = min(pos+64, hidden.shape[1])
            logits = model.lm_head(hidden[:, pos:end]).float()
            loss = F.cross_entropy(logits.reshape(-1, 256000), tokens[pos+1:end+1], reduction='sum')
            nll += float(loss)
            del logits, loss
        row = dict(label=label, mode=mode, nll=nll, archive_nll=archive[reference],
                   exact_archive=nll==archive[reference], hidden_sha256=native.tensor_hash(hidden))
        results.append(row)
        print(json.dumps(row), flush=True)

with native.install_fp16(model, parent):
    probe('parent_cold_no_preflight', 'sq3p25', 'resurface_sq3p25')
    probe('parent_repeat', 'sq3p25', 'resurface_sq3p25')
    with StateQuant(model, 'sq3p25', perms) as preflight:
        preflight.reset(1)
    probe('parent_after_preflight', 'sq3p25', 'resurface_sq3p25')
probe('source_s16', 's16', 'source_s16')
probe('source_sq', 'sq3p25', 'source_sq3p25')
with native.install_fp16(model, parent):
    probe('parent_after_baselines', 'sq3p25', 'resurface_sq3p25')
    probe('parent_after_baselines_repeat', 'sq3p25', 'resurface_sq3p25')
path = ROOT/'reports/resurface_more_v3_replay_diagnostic.json'
path.write_text(json.dumps(dict(complete=True, scope='First PPL window only; diagnostic, not quality selection',
    environment=runtime.environment_receipt(), rows=results), indent=2)+'\n')
