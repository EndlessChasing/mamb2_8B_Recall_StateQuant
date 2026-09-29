#!/usr/bin/env python3
"""CPU-only descriptive analysis of the fixed TRAIN screen; never reads CONFIRM."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def metrics(report):
    need(report['stage'] == 'screen' and report['complete'] is True,
         'Only completed TRAIN screen arms may be analyzed')
    need(report['dataset']['split'] == 'train', 'Non-TRAIN PPL input')
    rows = report['mk']['rows']
    need(len(rows) == 48 and len({r['id'] for r in rows}) == 48, 'MK inventory differs')
    for row in rows:
        need(row['split'] == 'train' and row['condition'] == 'normal', 'Non-TRAIN MK input')
        match = re.search(r'(?<!\d)\d{6}(?!\d)', row['output'])
        prediction = match.group(0) if match else None
        need(prediction == row['prediction'] and (prediction == row['answer']) == row['correct'],
             'Recorded MK score differs from decoded output')
    windows = report['ppl']['windows']
    need(len(windows) == 8 and [w['start'] for w in windows] == [i * 2048 for i in range(8, 16)],
         'Unexpected TRAIN PPL windows')
    targets = sum(w['target_tokens'] for w in windows)
    nll = sum(w['nll'] for w in windows)
    ppl = math.exp(nll / targets)
    need(targets == 4088 and abs(ppl - report['ppl']['ppl']) < 1e-12, 'PPL arithmetic differs')
    return dict(ppl=ppl, nll=nll, target_tokens=targets, mk_correct=sum(r['correct'] for r in rows),
                mk_count=len(rows))


def paired(old, new):
    a = old['mk']['rows']
    b = new['mk']['rows']
    need([r['id'] for r in a] == [r['id'] for r in b], 'Paired MK order differs')
    details = []
    for before, after in zip(a, b):
        for key in ('N', 'template', 'query_position', 'answer', 'row_sha256',
                    'prompt_token_sha256_int64le'):
            need(before[key] == after[key], 'Paired MK data differs: ' + key)
        delta = int(after['correct']) - int(before['correct'])
        details.append(dict(id=before['id'], N=before['N'], template=before['template'],
            query_position=before['query_position'],
            query_position_quartile=min(3, before['query_position'] * 4 // before['N']) + 1,
            records_after_target=before['N'] - 1 - before['query_position'],
            old_correct=before['correct'], new_correct=after['correct'], delta=delta,
            answer=before['answer'], old_prediction=before['prediction'], new_prediction=after['prediction'],
            new_prediction_is_record_value=after['prediction'] in {str(v) for _, v in after['records']}))

    def group(keys):
        grouped = defaultdict(list)
        for row in details:
            grouped[tuple(row[k] for k in keys)].append(row)
        return [dict(zip(keys, key), count=len(rows), old_correct=sum(r['old_correct'] for r in rows),
                     new_correct=sum(r['new_correct'] for r in rows),
                     gains=sum(r['delta'] == 1 for r in rows),
                     regressions=sum(r['delta'] == -1 for r in rows),
                     unchanged=sum(r['delta'] == 0 for r in rows))
                for key, rows in sorted(grouped.items())]

    windows = []
    for before, after in zip(old['ppl']['windows'], new['ppl']['windows']):
        for key in ('start', 'target_tokens', 'token_sha256_int64le'):
            need(before[key] == after[key], 'Paired PPL data differs')
        windows.append(dict(start=before['start'], old_ppl=before['ppl'], new_ppl=after['ppl'],
                            relative_ppl_change=after['ppl'] / before['ppl'] - 1,
                            nll_delta=after['nll'] - before['nll']))
    return dict(overall=group([])[0], by_N=group(['N']), by_template=group(['template']),
        by_N_template=group(['N', 'template']), by_query_position_quartile=group(['query_position_quartile']),
        by_N_query_position_quartile=group(['N', 'query_position_quartile']),
        changed_cases=[r for r in details if r['delta']], ppl_windows=windows,
        ppl_windows_improved=sum(w['nll_delta'] < 0 for w in windows),
        relative_ppl_change=new['ppl']['ppl'] / old['ppl']['ppl'] - 1)


def tier_churn(calibration, torch):
    a = calibration['original_permutations'].long()
    b = calibration['readout_permutations'].long()
    expected = torch.arange(128).expand(56, 8, 128)
    need(tuple(a.shape) == (56, 8, 128) and torch.equal(a.sort(-1).values, expected)
         and torch.equal(b.sort(-1).values, expected), 'Calibration permutations invalid')
    tier = torch.cat([torch.zeros(16, dtype=torch.long), torch.ones(64, dtype=torch.long),
                      torch.full((48,), 2, dtype=torch.long)]).expand_as(a)
    old = torch.empty_like(a).scatter_(-1, a, tier)
    new = torch.empty_like(b).scatter_(-1, b, tier)
    matrices = torch.stack([torch.bincount((old[i] * 3 + new[i]).flatten(), minlength=9).reshape(3, 3)
                            for i in range(56)])
    total = matrices.sum(0)

    def describe(matrix):
        return dict(matrix=matrix.tolist(), int8_kept=int(matrix[0, 0]),
            int8_demoted_to_int4=int(matrix[0, 1]), int8_demoted_to_zero=int(matrix[0, 2]),
            new_int8_from_int4=int(matrix[1, 0]), new_int8_from_zero=int(matrix[2, 0]),
            zero_to_retained=int(matrix[2, :2].sum()), retained_to_zero=int(matrix[:2, 2].sum()),
            unchanged_tier=int(matrix.diag().sum()), entries=int(matrix.sum()))

    mass = {}
    for key in ('mean_readout_score', 'mean_abs'):
        value = calibration['statistics'][key].double()
        need(bool(torch.isfinite(value).all()) and bool((value >= 0).all()) and value.sum() > 0,
             'Invalid calibration statistic')
        matrix = [[float(value[(old == i) & (new == j)].sum() / value.sum())
                   for j in range(3)] for i in range(3)]
        mass[key] = dict(transition_fraction=matrix, old_zero_fraction=sum(matrix[2]),
                         new_zero_fraction=sum(row[2] for row in matrix))
    return dict(coordinate_scope='56 layers x 8 groups x 128 original state coordinates; table entries, not parameters',
        tier_order=['int8', 'int4', 'zero_carry'], matrix_rows='old tiers', matrix_columns='new tiers',
        aggregate=describe(total), per_layer=[dict(layer=i, **describe(m)) for i, m in enumerate(matrices)],
        calibration_statistic_mass=mass,
        interpretation='Tier churn and TRAIN output differences coexist; this analysis does not establish causality.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen-dir', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Set CUDA_VISIBLE_DEVICES= for CPU-only analysis')
    need(not args.output.exists(), 'Choose a fresh output; preserve earlier analysis')
    import torch
    torch.set_num_threads(2)
    need(not torch.cuda.is_initialized(), 'CUDA unexpectedly initialized')
    comparison_path = args.screen_dir / 'screen_comparison.json'
    comparison = read(comparison_path)
    need(comparison['stage'] == 'screen' and comparison['complete'] is True, 'Screen incomplete')
    need(sha(args.calibration) == comparison['repair_calibration_sha256'], 'Calibration hash differs')
    calibration = torch.load(args.calibration, map_location='cpu', weights_only=True)
    need(calibration['heldout_used'] is False and calibration['calibration_tokens'] == 4096,
         'Calibration not the fixed TRAIN trace')
    need(calibration['protocol_sha256'] == comparison['protocol_sha256'], 'Protocol binding differs')
    reports, input_hashes = {}, {comparison_path.name: sha(comparison_path)}
    for arm, digest in comparison['report_sha256'].items():
        path = args.screen_dir / ('screen_' + arm + '.json')
        need(sha(path) == digest, 'Screen arm hash differs: ' + arm)
        reports[arm] = read(path)
        need(reports[arm]['repair_calibration_sha256'] == sha(args.calibration), 'Arm calibration differs')
        input_hashes[path.name] = digest
    summary = {arm: metrics(report) for arm, report in reports.items()}
    restoration = args.screen_dir / 'screen_restoration.json'
    need(read(restoration) == comparison['restoration'], 'Restoration receipt differs')
    input_hashes[restoration.name] = sha(restoration)
    result = dict(format='MAMBA2_STATE_REPAIR_TRAIN_SCREEN_ANALYSIS_V1', complete=True,
        scope='Descriptive TRAIN screen only; no validation or CONFIRM reports read; no fitted or selected new candidate',
        heldout_used=False, cuda_initialized=False, protocol_sha256=comparison['protocol_sha256'],
        analysis_script_sha256=sha(__file__), input_sha256=input_hashes,
        calibration_sha256=sha(args.calibration), summary=summary, selection=comparison['selection'],
        readout_vs_old_with_v2=paired(reports['old_sq_v2'], reports['readout_tiers_v2']),
        readout_vs_old_without_adapter=paired(reports['old_sq_no_adapter'], reports['readout_tiers_no_adapter']),
        tier_churn=tier_churn(calibration, torch), limitations=[
            '48 TRAIN cases, eight per N/template cell, were seen during prior adapter training; not a generalization estimate.',
            'Query position is the zero-based record index, not the position of the final query token.',
            'Readout score measures one-step diagonal carry damage under unadapted S16 prose traces, not future query sensitivity.',
            'The score ignores cross-coordinate covariance, downstream sensitivity and actual INT8/INT4 shared-scale errors.',
            'Runtime recurrent quantization and the frozen adapter can change trajectories relative to calibration.',
            'Subgroup and tier-churn associations do not prove why individual predictions changed.'],
        prospective_hypotheses=[
            'In a new preregistered TRAIN experiment, preserve all original INT8 coordinates and use readout scores only to select INT4 versus zero among the remaining112; this isolates damage from demoting high-magnitude carries at unchanged storage.',
            'Readapt Resurface to a separately frozen readout-tier recurrence with a fixed TRAIN numeric/prose teacher objective and budget; require PPL preservation and MK recovery.',
            'The no-adapter MK decrease means adapter mismatch alone is insufficient as an explanation; recovery is not assured.',
            'Do not choose per-template/per-position changes from this small screen or tune on current CONFIRM outcomes.'])
    need(not torch.cuda.is_initialized(), 'Analysis initialized CUDA')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(complete=True, output=str(args.output), sha256=sha(args.output),
                         paired=result['readout_vs_old_with_v2']['overall'],
                         tier_matrix=result['tier_churn']['aggregate']['matrix']), indent=2))


if __name__ == '__main__':
    main()
