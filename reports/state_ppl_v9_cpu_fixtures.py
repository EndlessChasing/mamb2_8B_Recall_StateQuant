#!/usr/bin/env python3
"""Reproducible CPU fixture artifact; no model construction or GPU kernels."""
import copy
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import torch
import prepare_state_ppl_v9 as p
import run_state_ppl_v9 as r
from mamba2_recall import resurface_data as data
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
torch.set_num_threads(8)
checks=[]
def check(name,ok):
    assert ok,name
    checks.append(dict(name=name,passed=True))
def good(nll=100.,ppl=2.):
    return dict(complete=True,deployable=True,diagnostic=None,ppl=dict(nll=nll,ppl=ppl),cache=dict(total_bytes=p.v6.CACHE_BYTES),
        persistent_float_finite_checks_passed=True,repeated_reset_probe=dict(hidden_and_cache_exact=True),
        frozen_source=dict(identity_version_gradients_unchanged=True),backend_policy_check=dict(singleton_config_unchanged=True),
        candidate_table_unchanged=True,runtime_table_unchanged=True,adapter_loaded=False,adapter_sha256=None)
base=torch.arange(128,dtype=torch.uint8).expand(56,8,128).clone()
tables=dict(magnitude=base,full_readout=torch.roll(base,1,-1),preserve_int8=torch.roll(base,1,-1),preserve_retained80=torch.roll(base,2,-1))
inv=p.group_inventory(tables,base)
check('parent bytes skipped and alternative bytes deduplicated',all(
    [a['disposition'] for a in g['alternatives']]==['parent_equal','retained','duplicate_of','retained']
    and g['alternatives'][2]['duplicate_of']=='full_readout' for g in inv))
check('actual unique intervention order',len(p.arm_specs(inv))==130 and p.arm_specs(inv)[1][0]=='layer00_group0_full_readout')
rows={a:good() for a,*_ in p.arm_specs(inv)};rank=p.rank_groups(rows,inv);t,s,props=p.derive_candidates(tables,base,rank)
check('strict equality yields baseline only',rank['stopped'] and list(t)==['baseline'] and len(props)==8)
rows['layer00_group0_full_readout']=good(2e-15);rows['layer00_group0_preserve_retained80']=good(1e-15)
check('raw NLL before collapsed delta',2e-15-100.==1e-15-100. and p.rank_groups(rows,inv)['ranked_improving_swaps'][0]['kind']=='preserve_retained80')
rows['layer00_group0_full_readout']=good(1e-15)
check('exact raw NLL table-order tie',p.rank_groups(rows,inv)['ranked_improving_swaps'][0]['kind']=='full_readout')
rows={a:good() for a,*_ in p.arm_specs(inv)}
for arm in ('layer03_group0_full_readout','layer00_group1_full_readout','layer00_group0_full_readout'):rows[arm]=good(99.)
rank=p.rank_groups(rows,inv)
check('global delta numeric layer/group ordering',[(x['layer'],x['group']) for x in rank['ranked_improving_swaps']]==[(0,0),(0,1),(3,0)])
t,s,props=p.derive_candidates(tables,base,rank)
check('fixed topK cap and whole-table dedup',list(t)==['baseline','top1','top2','top4'] and props[-1]['retained_name']=='top4' and s['top4']['swap_count']==3)
check('group mixing changes only selected coordinates',torch.equal(t['top1'][0,0],tables['full_readout'][0,0])
    and torch.equal(t['top1'][0,1:],base[0,1:]) and torch.equal(t['top1'][1:],base[1:]))
rows['layer00_group0_full_readout']=dict(complete=False,error='nonfinite')
check('excluded alternative does not win',p.rank_groups(rows,inv)['improving_groups']==2)
c=dict(candidate_order=list(t),tables=t,candidate_specs=s);selectionrows={n:good() for n in c['candidate_order']}
check('screen ties prefer baseline',r.choose_candidate(selectionrows,c)['selected_id']=='baseline')
selectionrows['top1']=good(ppl=1.9);selectionrows['top2']=good(ppl=1.9)
check('screen candidate tie uses export order',r.choose_candidate(selectionrows,c)['selected_id']=='top1')
check('fixed storage and unchanged v6 codec',all(x.numel()*x.element_size()==57344 for x in t.values())
    and all(r.candidate_spec(n,c)['scale_mode']=='stored_scale' and r.candidate_spec(n,c)['int4_clip']==1 for n in c['candidate_order']))
for stopped,order,label in ((True,['baseline','top1'],'stopped calibration rejected before model'),
                           (False,['baseline'],'baseline-only export rejected before model')):
    fake=dict(calibration_stopped=stopped,candidate_order=order)
    with patch.object(r.prep,'load_group_candidates',return_value=(fake,None,None,None)):
        try:r.load_inputs(SimpleNamespace(group_candidates=Path('cpu-fixture.pt')))
        except ValueError:check(label,True)
        else:raise AssertionError('accepted redundant screen')
with tempfile.TemporaryDirectory(prefix='v9cpu_') as tmp:
    out=Path(tmp);binding=dict(synthetic_cpu_fixture=True);windows=[(152*2048,torch.tensor([1,2,3],dtype=torch.long))];raw={}
    for arm in r.screen_arms(c):
        name='baseline' if arm=='restored_baseline' else arm;nll=dict(baseline=2.,top1=1.8,top2=1.9,top4=1.95)[name]
        win=dict(p.v6.window_identity(windows)[0],nll=nll,ppl=math.exp(nll/2));row=good(nll,math.exp(nll/2));row.update(r.candidate_spec(name,c))
        row.update(format=r.FORMAT,stage='screen',arm=arm,protocol_sha256=r.PROTOCOL_SHA,input_binding=binding,dataset=r.screen_dataset(),
            code_hashes=r.code_hashes(),candidate_table_sha256=r.tensor_sha(c['tables'][name]),group_mix_spec=c['candidate_specs'][name],
            heldout_used=False,heldout_used_for_selection=False,mk_used=False,ppl=dict(windows=[win],nll=nll,target_tokens=2,ppl=math.exp(nll/2)),
            selected_calibration_sha256=None,parent_report_sha256=None,parent_comparison_sha256=None,s16_report_sha256=None)
        r.save_json(out/('screen_'+arm+'.json'),row);raw[arm]=row
    restoration=p.v6.check_restoration(raw['baseline'],raw['restored_baseline']);r.save_json(out/'screen_restoration.json',restoration)
    comp=dict(format=r.COMPARE,complete=True,stage='screen',protocol_sha256=r.PROTOCOL_SHA,input_binding=binding,code_hashes=r.code_hashes(),
        adapter_loaded=False,adapter_sha256=None,heldout_used=False,heldout_used_for_selection=False,mk_used=False,
        report_sha256={n:data.sha_file(out/('screen_'+n+'.json')) for n in r.screen_arms(c)},selection=r.choose_candidate(raw,c),
        restoration=restoration,selected_vs_baseline=p.v6.compare_ppl(raw['baseline'],raw['top1']))
    r.save_json(out/'screen_comparison.json',comp);r.export_selection(out,c,binding,comp)
    loaded,_,_=r.load_selection(out/'selected_calibration.pt',c,binding,windows)
    check('selected export and raw reconstruction',loaded['selected_id']=='top1' and torch.equal(loaded['permutations'],t['top1']))
    raw['top2']['ppl']['nll']+=1;r.save_json(out/'screen_top2.json',raw['top2'])
    try:r.load_selection(out/'selected_calibration.pt',c,binding,windows)
    except ValueError:check('selected raw-report tamper rejected',True)
    else:raise AssertionError('accepted tamper')
archive=json.loads((ROOT/'artifacts/state_ppl_v8_full/full_selected.json').read_text())
check('exact actual v8 archive replay',r.archive_replay(archive,copy.deepcopy(archive))['complete'])
bad=copy.deepcopy(archive);bad['cache']['total_bytes']+=1
try:r.archive_replay(archive,bad)
except ValueError:check('archive cache mutation rejected',True)
else:raise AssertionError('accepted cache drift')
check('CPU only',not torch.cuda.is_initialized())
print(json.dumps(dict(format='MAMBA2_STATE_PPL_V9_CPU_FIXTURES_V1',complete=True,passed=True,checks=checks,
    command='CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 /home/horde/.venvs/lodram/bin/python reports/state_ppl_v9_cpu_fixtures.py',
    fixture_sha256=data.sha_file(__file__),preparer_sha256=data.sha_file(p.__file__),runner_sha256=data.sha_file(r.__file__),
    protocol_sha256=p.PROTOCOL_SHA,cuda_initialized=False,quality_measured=False),indent=2))
