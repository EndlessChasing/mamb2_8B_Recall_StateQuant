#!/usr/bin/env python3
"""Synthetic CPU-only orchestration fixtures, never model quality evidence."""
import copy
import json
import math
import os
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import torch
import run_state_ppl_v10 as r
from mamba2_recall import resurface_data as data
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
torch.set_num_threads(8)
checks=[]
def check(name,ok):
    assert ok,name
    checks.append(dict(name=name,passed=True))
def reject(name,fn):
    try:fn()
    except (ValueError,RuntimeError):check(name,True)
    else:raise AssertionError('Expected rejection: '+name)
def good(ppl=2.):
    return dict(complete=True,deployable=True,diagnostic=None,ppl=dict(nll=100.,ppl=ppl),cache=dict(total_bytes=r.v6.CACHE_BYTES),
        persistent_float_finite_checks_passed=True,repeated_reset_probe=dict(hidden_and_cache_exact=True),
        frozen_source=dict(identity_version_gradients_unchanged=True),backend_policy_check=dict(singleton_config_unchanged=True),
        candidate_table_unchanged=True,runtime_table_unchanged=True,adapter_loaded=False,adapter_sha256=None,
        allocation_storage_validated=True)
def physical(name):
    desc=r.layout_spec(name)
    tensors={k:dict(shape=[1,128,64,w],dtype='torch.uint8',storage_bytes=128*64*w) for k,w in desc['tensor_widths'].items()}
    tensors.update({k:dict(shape=[1,128,64],dtype='torch.float16',storage_bytes=128*64*2) for k in ('s8','s4')})
    layer=dict(state_shape=[1,128,64,128],state_bytes=128*64*52,tensors=tensors,
        conv_shape=[1,10240,4],conv_storage_bytes=10240*4*2)
    return dict(**desc,layers=[copy.deepcopy(layer) for _ in range(56)])
base=torch.arange(128,dtype=torch.uint8).expand(56,8,128).clone()
c=dict(candidate_order=list(r.LAYOUTS),tables={name:base for name in r.LAYOUTS},
    candidate_specs={name:r.layout_spec(name) for name in r.LAYOUTS})
check('fixed layout order and unchanged table',r.screen_arms(c)==(*r.LAYOUTS,'restored_baseline')
    and all(torch.equal(x,base) and x.numel()==57344 for x in c['tables'].values()))
for name,(n8,n4,nzero) in r.LAYOUTS.items():
    check('actual52-byte tensor geometry '+name,r.validate_storage_descriptor(physical(name),name)
        and n8+n4+nzero==128 and n8+n4//2+4==52)
    bad=physical(name);bad['layers'][0]['tensors']['q4']['storage_bytes']+=1
    reject('one byte padding rejected '+name,lambda:r.validate_storage_descriptor(bad,name))
bad=physical(r.BASELINE);bad['layers'][0]['tensors']['extra']=dict(shape=[1],dtype='torch.uint8',storage_bytes=1)
reject('extra resident tensor rejected',lambda:r.validate_storage_descriptor(bad,r.BASELINE))
rows={name:good() for name in r.LAYOUTS}
check('exact PPL ties prefer baseline',r.choose_candidate(rows,c)['selected_id']==r.BASELINE)
rows['8_80_40']=good(1.9);rows['24_48_56']=good(1.9)
check('nonbaseline tie uses layout order',r.choose_candidate(rows,c)['selected_id']=='8_80_40')
rows['8_80_40']=dict(complete=False,error_type='CandidateInvalid',excluded_from_selection=True)
check('known nonfinite excluded',r.choose_candidate(rows,c)['selected_id']=='24_48_56')
rows[r.BASELINE]=dict(complete=False)
reject('failed baseline fatal',lambda:r.choose_candidate(rows,c))
check('disjoint fixed TRAIN population',list(r.SCREEN_ROWS)==list(range(184,216)) and r.screen_dataset()['target_tokens']==65504)
with tempfile.TemporaryDirectory(prefix='v10cpu_') as tmp:
    out=Path(tmp);binding=dict(synthetic_cpu_fixture=True)
    windows=[(184*2048,torch.tensor([1,2,3],dtype=torch.long))];raw={}
    for arm in r.screen_arms(c):
        name=r.BASELINE if arm=='restored_baseline' else arm
        nll={'16_64_48':2.,'8_80_40':1.8,'24_48_56':1.9,'32_32_64':1.95}[name]
        win=dict(r.v6.window_identity(windows)[0],nll=nll,ppl=math.exp(nll/2));row=good(math.exp(nll/2))
        row.update(r.candidate_spec(name,c))
        row.update(format=r.FORMAT,stage='screen',arm=arm,protocol_sha256=r.PROTOCOL_SHA,input_binding=binding,dataset=r.screen_dataset(),
            code_hashes=r.code_hashes(),candidate_table_sha256=r.tensor_sha(base),allocation_spec=c['candidate_specs'][name],
            storage_descriptor_probe=physical(name),storage_descriptor=physical(name),heldout_used=False,heldout_used_for_selection=False,
            mk_used=False,ppl=dict(windows=[win],nll=nll,target_tokens=2,ppl=math.exp(nll/2)),selected_calibration_sha256=None,
            parent_report_sha256=None,parent_comparison_sha256=None,s16_report_sha256=None)
        r.save_json(out/('screen_'+arm+'.json'),row);raw[arm]=row
    restoration=r.v6.check_restoration(raw[r.BASELINE],raw['restored_baseline']);r.save_json(out/'screen_restoration.json',restoration)
    comp=dict(format=r.COMPARE,complete=True,stage='screen',protocol_sha256=r.PROTOCOL_SHA,input_binding=binding,code_hashes=r.code_hashes(),
        adapter_loaded=False,adapter_sha256=None,heldout_used=False,heldout_used_for_selection=False,mk_used=False,
        report_sha256={n:data.sha_file(out/('screen_'+n+'.json')) for n in r.screen_arms(c)},selection=r.choose_candidate(raw,c),
        restoration=restoration,selected_vs_baseline=r.v6.compare_ppl(raw[r.BASELINE],raw['8_80_40']))
    r.save_json(out/'screen_comparison.json',comp);r.export_selection(out,c,binding,comp)
    path=out/'selected_calibration.pt';loaded,_,_=r.load_selection(path,c,binding,windows)
    check('selected export and raw screen reconstruction',loaded['selected_id']=='8_80_40' and loaded['selected_layout']=='8_80_40'
        and torch.equal(loaded['permutations'],base))
    original=path.read_bytes();original_receipt=path.with_suffix('.json').read_bytes()
    tampered=copy.deepcopy(loaded);tampered['selected_layout']='24_48_56';torch.save(tampered,path)
    receipt={k:v for k,v in tampered.items() if k!='permutations'}
    receipt.update(complete=True,file=path.name,sha256=data.sha_file(path),bytes=path.stat().st_size)
    r.save_json(path.with_suffix('.json'),receipt)
    reject('self-consistent selected layout tamper rejected',lambda:r.load_selection(path,c,binding,windows))
    path.write_bytes(original);path.with_suffix('.json').write_bytes(original_receipt)
    raw['24_48_56']['ppl']['nll']+=1;r.save_json(out/'screen_24_48_56.json',raw['24_48_56'])
    reject('raw arm hash tamper rejected',lambda:r.load_selection(path,c,binding,windows))
archive=json.loads((ROOT/'artifacts/state_ppl_v9_full/full_selected.json').read_text())
check('actual whole v9 archive replay',r.archive_replay(archive,copy.deepcopy(archive))['complete'])
for field,label in (('cache','entire cache'),('repeated_reset_probe','entire probe')):
    bad=copy.deepcopy(archive);bad[field]['unexpected']=1
    reject('archive '+label+' addition rejected',lambda:r.archive_replay(archive,bad))
bad=copy.deepcopy(archive);bad['ppl']['windows'][0]['nll']=math.nextafter(bad['ppl']['windows'][0]['nll'],math.inf)
reject('archive one ULP NLL mutation rejected',lambda:r.archive_replay(archive,bad))
check('CPU only',not torch.cuda.is_initialized())
print(json.dumps(dict(format='MAMBA2_STATE_PPL_V10_CPU_FIXTURES_V1',complete=True,passed=True,checks=checks,
    command='CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 /home/horde/.venvs/lodram/bin/python reports/state_ppl_v10_cpu_fixtures.py',
    fixture_sha256=data.sha_file(__file__),runner_sha256=data.sha_file(r.__file__),code_hashes=r.code_hashes(),
    protocol_sha256=r.PROTOCOL_SHA,cuda_initialized=False,quality_measured=False),indent=2))
