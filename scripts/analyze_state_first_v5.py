#!/usr/bin/env python3
"""CPU-only descriptive analysis of the frozen v5 TRAIN selection and state tiers."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re

PROTOCOL = 'ce700910f27230e062fe62e89dd6bddee05bcbf7bb7972c59fdba24cb3db9f5d'
V4_STATS = '8509bb266d40875608f3e0b3be22407fd6ea1b8aacbc79ce04e958726516b0a4'
KINDS = ('magnitude', 'full_readout', 'preserve_int8', 'preserve_retained80')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def need(condition, message):
    if not condition:
        raise ValueError(message)


def metrics(report):
    need(report['stage'] == 'screen' and report['complete'] is True
         and report['dataset']['split'] == 'train' and report['adapter_loaded'] is False
         and report['adapter_sha256'] is None and report['heldout_used'] is False,
         'Only completed unadapted TRAIN screen arms are allowed')
    rows = report['mk']['rows']
    need(len(rows) == 96 and len({r['id'] for r in rows}) == 96, 'Unexpected MK inventory')
    for row in rows:
        need(row['split'] == 'train' and row['condition'] == 'normal'
             and row['sample'] in range(0, 256, 16), 'MK data is not the fixed TRAIN screen')
        match = re.search(r'(?<!\d)\d{6}(?!\d)', row['output'])
        prediction = match.group() if match else None
        need(prediction == row['prediction'] and (prediction == row['answer']) == row['correct'],
             'Recorded MK score differs from output text')
    windows = report['ppl']['windows']
    need([w['start'] for w in windows] == [i * 2048 for i in range(8, 40)], 'TRAIN window selection differs')
    targets = sum(w['target_tokens'] for w in windows)
    nll = sum(w['nll'] for w in windows)
    ppl = math.exp(nll / targets)
    need(targets == 16352 and math.isclose(ppl, report['ppl']['ppl'], rel_tol=1e-13), 'PPL arithmetic differs')
    correct = sum(r['correct'] for r in rows)
    need(correct == report['mk']['summary']['normal']['correct'], 'MK aggregate differs')
    return dict(ppl=ppl, nll=nll, target_tokens=targets, mk_correct=correct, mk_count=96,
                persistent_cache_bytes=report['cache']['total_bytes'])


def paired(old, new):
    a = {r['id']:r for r in old['mk']['rows']}
    b = {r['id']:r for r in new['mk']['rows']}
    need(set(a) == set(b), 'Paired MK identities differ')
    rows = []
    for key in sorted(a):
        before, after = a[key], b[key]
        for name in ('N', 'template', 'query_position', 'answer', 'row_sha256', 'prompt_token_sha256_int64le'):
            need(before[name] == after[name], 'Paired MK input differs: ' + name)
        delta = int(after['correct']) - int(before['correct'])
        rows.append(dict(id=key, N=before['N'], template=before['template'], query_position=before['query_position'],
            query_position_quartile=min(3, before['query_position'] * 4 // before['N']) + 1,
            records_after_target=before['N'] - 1 - before['query_position'],
            old_correct=before['correct'], new_correct=after['correct'], delta=delta,
            old_prediction=before['prediction'], new_prediction=after['prediction'], answer=before['answer']))

    def grouped(keys):
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[k] for k in keys)].append(row)
        return [dict(zip(keys, key), count=len(values), old_correct=sum(r['old_correct'] for r in values),
                     new_correct=sum(r['new_correct'] for r in values),
                     gains=sum(r['delta'] == 1 for r in values), regressions=sum(r['delta'] == -1 for r in values),
                     unchanged=sum(r['delta'] == 0 for r in values))
                for key, values in sorted(groups.items())]

    windows = []
    for before, after in zip(old['ppl']['windows'], new['ppl']['windows']):
        for name in ('start', 'target_tokens', 'token_sha256_int64le'):
            need(before[name] == after[name], 'Paired PPL input differs')
        windows.append(dict(start=before['start'], old_ppl=before['ppl'], new_ppl=after['ppl'],
            relative_ppl_change=after['ppl'] / before['ppl'] - 1, nll_delta=after['nll'] - before['nll']))
    return dict(old_arm=old['arm'], new_arm=new['arm'], relative_ppl_change=new['ppl']['ppl'] / old['ppl']['ppl'] - 1,
        nll_delta=sum(w['nll_delta'] for w in windows), ppl_windows_improved=sum(w['nll_delta'] < 0 for w in windows),
        ppl_windows_regressed=sum(w['nll_delta'] > 0 for w in windows), ppl_windows=windows,
        mk_overall=grouped([])[0], mk_by_N=grouped(['N']), mk_by_template=grouped(['template']),
        mk_by_N_template=grouped(['N', 'template']), mk_by_query_position_quartile=grouped(['query_position_quartile']),
        mk_changed_cases=[r for r in rows if r['delta']])


def state_statistics(tables, statistics, torch):
    tiers = {}
    for name, table in tables.items():
        need(table.dtype == torch.uint8 and tuple(table.shape) == (56, 8, 128)
             and torch.equal(table.sort(-1).values, torch.arange(128, dtype=torch.uint8).expand_as(table)),
             'Invalid candidate permutation: ' + name)
        ranks = torch.cat([torch.zeros(16, dtype=torch.long), torch.ones(64, dtype=torch.long),
                           torch.full((48,), 2, dtype=torch.long)]).expand_as(table)
        tiers[name] = torch.empty_like(ranks).scatter_(-1, table.long(), ranks)

    def transition(old_name, new_name):
        old, new = tiers[old_name], tiers[new_name]
        matrices = torch.stack([torch.bincount((old[i] * 3 + new[i]).flatten(), minlength=9).reshape(3, 3)
                                for i in range(56)])
        total = matrices.sum(0)
        return dict(old=old_name, new=new_name, tier_order=['int8', 'int4', 'zero_carry'],
            matrix_rows='old tier', matrix_columns='new tier', matrix=total.tolist(),
            per_layer=[dict(layer=i, matrix=m.tolist()) for i,m in enumerate(matrices)],
            old_int8_preserved=int(total[0,0]), old_int8_total=int(total[0].sum()),
            old_int8_preserved_fraction=float(total[0,0] / total[0].sum()),
            old_zero_became_retained=int(total[2,:2].sum()), old_retained_became_zero=int(total[:2,2].sum()),
            unchanged_tier=int(total.diag().sum()), entries=int(total.sum()))

    masses = {}
    for name, tier in tiers.items():
        values = {}
        for field in ('mean_readout_score', 'mean_abs'):
            stat = statistics[field].double()
            need(tuple(stat.shape) == (56,8,128) and bool(torch.isfinite(stat).all()) and bool((stat >= 0).all())
                 and stat.sum() > 0, 'Invalid recorded calibration statistic')
            sums = [float(stat[tier == i].sum()) for i in range(3)]
            total = float(stat.sum())
            values[field] = dict(total=total, tier_sums=sums, tier_fractions=[v / total for v in sums],
                zero_fraction=sums[2] / total,
                per_layer_zero_fraction=[float(stat[i][tier[i] == 2].sum() / stat[i].sum()) for i in range(56)])
        masses[name] = values
    transitions = {a+'_to_'+b:transition(a,b) for a,b in (
        ('magnitude','preserve_int8'), ('full_readout','preserve_int8'), ('magnitude','full_readout'),
        ('magnitude','preserve_retained80'))}
    need(transitions['magnitude_to_preserve_int8']['old_int8_preserved_fraction'] == 1.,
         'Selected table failed exact original INT8 preservation')
    return dict(coordinate_scope='56 layers x 8 groups x 128 state-coordinate table entries; not model parameters',
        tier_order=['int8', 'int4', 'zero_carry'], transitions=transitions, calibration_mass=masses,
        mass_scope='Global sums of recorded per-coordinate TRAIN S16 means; fractions of proxy statistics, not output error or recall information')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen-dir', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--v4-calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Set CUDA_VISIBLE_DEVICES= for CPU-only analysis')
    need(not args.output.exists(), 'Preserve earlier analysis: choose a fresh output')
    import torch
    torch.set_num_threads(2)
    need(not torch.cuda.is_initialized(), 'CUDA unexpectedly initialized')
    comparison_path = args.screen_dir/'screen_comparison.json'
    comparison = read(comparison_path)
    need(comparison['complete'] is True and comparison['stage'] == 'screen' and comparison['protocol_sha256'] == PROTOCOL
         and comparison['adapter_sha256'] is None and comparison['selection']['selected_kind'] == 'preserve_int8'
         and comparison['selection']['adapter_used'] is False and comparison['selection']['heldout_used'] is False,
         'Expected the already-frozen unadapted preserve_int8 TRAIN selection')
    need(sha(args.v4_calibration) == V4_STATS and sha(args.candidates) == comparison['candidates_sha256'],
         'Frozen candidate/statistics hashes differ')
    candidates = torch.load(args.candidates, map_location='cpu', weights_only=True)
    receipt = read(args.candidates.with_suffix('.json'))
    calibration = torch.load(args.v4_calibration, map_location='cpu', weights_only=True)
    need(candidates['protocol_sha256'] == PROTOCOL and candidates['v4_statistics_sha256'] == V4_STATS
         and candidates['adapter'] is None and candidates['heldout_used'] is False
         and receipt['sha256'] == sha(args.candidates), 'Candidate provenance differs')
    need(calibration['adapter'] is None and calibration['heldout_used'] is False
         and calibration['calibration_tokens'] == 4096, 'Statistics must come from unadapted TRAIN S16')
    need(tuple(candidates['tables']) == KINDS, 'Candidate inventory/order differs')
    for name, table in candidates['tables'].items():
        need(hashlib.sha256(table.contiguous().numpy().tobytes()).hexdigest() == receipt['table_sha256'][name],
             'Table digest differs: '+name)
    selected_path = args.screen_dir/'selected_calibration.pt'
    selected = torch.load(selected_path, map_location='cpu', weights_only=True)
    selected_receipt = read(selected_path.with_suffix('.json'))
    need(selected['selection_report_sha256'] == sha(comparison_path) and selected['candidates_sha256'] == sha(args.candidates)
         and selected['selected_kind'] == 'preserve_int8'
         and selected_receipt['sha256'] == sha(selected_path)
         and torch.equal(selected['permutations'], candidates['tables']['preserve_int8']), 'Selected artifact binding differs')
    hashes = {comparison_path.name:sha(comparison_path), selected_path.name:sha(selected_path),
              selected_path.with_suffix('.json').name:sha(selected_path.with_suffix('.json'))}
    reports = {}
    for arm, digest in comparison['report_sha256'].items():
        path = args.screen_dir/('screen_'+arm+'.json')
        need(sha(path) == digest, 'Raw screen report hash differs: '+arm)
        reports[arm] = read(path); hashes[path.name] = digest
    need(set(reports) == {*KINDS, 'restored_magnitude'}, 'Screen arm inventory differs')
    summary = {name:metrics(report) for name,report in reports.items()}
    original, restored = reports['magnitude'], reports['restored_magnitude']
    need(original['ppl']['windows'] == restored['ppl']['windows'], 'Magnitude PPL restoration differs')
    for a,b in zip(original['mk']['rows'], restored['mk']['rows']):
        need(all(a[k] == b[k] for k in ('id','generated_ids','output','prediction','correct')), 'Magnitude MK restoration differs')
    restoration_path = args.screen_dir/'screen_restoration.json'
    need(read(restoration_path) == comparison['restoration'], 'Restoration receipt differs')
    hashes[restoration_path.name] = sha(restoration_path)
    comparisons = {a+'_to_'+b:paired(reports[a],reports[b]) for a,b in (
        ('magnitude','preserve_int8'), ('full_readout','preserve_int8'),
        ('magnitude','full_readout'), ('magnitude','preserve_retained80'))}
    output = dict(format='MAMBA2_STATE_FIRST_V5_TRAIN_SCREEN_ANALYSIS_V1', complete=True,
        scope='Descriptive analysis of frozen unadapted TRAIN selection; no new candidate, fitting, training change or CONFIRM access',
        heldout_used=False, cuda_initialized=False, protocol_sha256=PROTOCOL,
        analysis_script_sha256=sha(__file__), input_screen_sha256=hashes,
        candidates_sha256=sha(args.candidates), candidates_receipt_sha256=sha(args.candidates.with_suffix('.json')),
        v4_statistics_sha256=sha(args.v4_calibration), selected_calibration_sha256=sha(selected_path),
        selection=comparison['selection'], summary=summary, paired_train=comparisons,
        state_statistics=state_statistics(candidates['tables'],calibration['statistics'],torch),
        limitations=[
            'PPL uses32 TRAIN windows; MK uses96 TRAIN cases. These are selection/development observations, not heldout confirmation.',
            'Tier mass uses original unadapted S16 prose traces. SQ recurrence can change the actual state trajectory.',
            'Readout score is a one-step diagonal carry proxy and omits future C, cross-coordinate covariance and downstream sensitivity.',
            'Preserving INT8 also changes other tier membership/scales and reduction order; aggregate differences do not isolate a single causal mechanism.',
            'A one-answer MK difference does not establish a reliable recall improvement.',
            'This analysis rechecks recorded output-text scoring and NLL arithmetic; it does not independently regenerate GPU logits or decode token IDs.'])
    need(not torch.cuda.is_initialized(), 'CPU analysis initialized CUDA')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(complete=True, output=str(args.output), sha256=sha(args.output),
        summary=summary, selected_pair=comparisons['magnitude_to_preserve_int8']['mk_overall'],
        selected_tier_matrix=output['state_statistics']['transitions']['magnitude_to_preserve_int8']['matrix']),indent=2))


if __name__ == '__main__':
    main()
