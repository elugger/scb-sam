"""One bounded zero-update compatibility repair; preserves attempt01, never launches."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT=Path('__RECORDED_PROJECT_ROOT__')
HERE=ROOT/'expes/semi05_gate/public_sup_20261018'
BASE=ROOT/'expes/work_dir/semi05_gate/PUBLIC_BASIC_SUP_FOLD2_20261018'
STORE=ROOT/'expes/artifacts/FCD_A_MATRIX_20261006/PUBLIC_BASIC_SUP_FOLD2_20261018'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_replace(p,value):
    tmp=Path(str(p)+'.r2tmp')
    with tmp.open('x') as f:
        json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)


def main():
    old_sha='3df297b497a5bd9528b02056b8cc8c9b7f9415e2684ef7d3cd0cc7d415bdb38e'
    assert sha(HERE/'preparation_ready.json')==old_sha, 'Already re-prepared or unexpected original'
    preserved=BASE/'attempt01_preserved'
    assert sha(preserved/'preservation.json')=='58b9a74858aa23ca7d8fb469313d7363f3624f9c2b7c09a9c081c3ea33a9a387'
    previous=json.loads((HERE/'preparation_ready.json').read_text())
    fail=json.loads((preserved/'output_sup05/failure.json').read_text())
    assert fail['successful_updates']==0 and fail['completed_epochs']==0 and not fail['partial_test_present']
    assert not Path('/proc/12388').exists()
    for arm in ['sup05','sup10','sup100']:
        assert not (BASE/arm).exists() and not (STORE/arm).exists() and not list((BASE/'control').glob(arm+'.*'))
    import train_public_basic_sup_20261018 as runner
    import public_basic_sup_guard_20261018 as guard
    import torch
    # Tiny CPU API qualification, no dataset, model, forward, GPU or test inference.
    memory=io.BytesIO();torch.save({'tensor':torch.tensor([1.,2.])},memory);memory.seek(0)
    loaded=runner.load_trusted_checkpoint(memory,map_location='cpu')
    assert torch.equal(loaded['tensor'],torch.tensor([1.,2.]))
    parameter=torch.nn.Parameter(torch.tensor([1.]));optimizer=torch.optim.AdamW([parameter],lr=1e-4)
    counter={'count':0};original_step=runner.install_optimizer_step_counter(optimizer,counter)
    assert counter['count']==0
    parameter.grad=torch.tensor([1.]);optimizer.step()
    assert counter['count']==1 and parameter.item()!=1.0
    optimizer.step=original_step
    tests=subprocess.run([sys.executable,str(HERE/'test_public_basic_sup_20261018.py')],cwd=HERE,capture_output=True,text=True)
    assert tests.returncode==0, tests.stdout+tests.stderr
    sources={name:sha(ROOT/name) for name in previous['source_sha256']}
    changed=[n for n in sources if sources[n]!=previous['source_sha256'][n]]
    assert set(changed)=={str((HERE/n).relative_to(ROOT)) for n in ['train_public_basic_sup_20261018.py','test_public_basic_sup_20261018.py']},changed
    sources[str(Path(__file__).resolve().relative_to(ROOT))]=sha(Path(__file__))
    configs={}
    for arm in ['sup05','sup10','sup100']:
        path=HERE/(arm+'.json')
        assert sha(path)==previous['config_sha256'][path.name]
        cfg=json.loads(path.read_text());cfg['source_sha256']=sources;cfg['engineering_attempt']=2
        runner.validate_config(cfg);runner.verify_manifests(cfg)
        write_replace(path,cfg);configs[path.name]=sha(path)
    with guard.original_lock():
        resource=guard.capacity_snapshot(14*guard.GIB+guard.MARGIN)
    receipt={**previous,'created_at':datetime.now(timezone.utc).isoformat(),'source_sha256':sources,
             'config_sha256':configs,'cpu_test_output':tests.stdout+tests.stderr,'resource_snapshot':resource,
             'engineering_attempt':2,'previous_preparation_sha256':old_sha,'preservation_sha256':sha(preserved/'preservation.json'),
             'compatibility_qualification':{'torch_version':torch.__version__,'torch_file':torch.__file__,
                'tiny_cpu_serialization_passed':True,'tiny_cpu_adamw_step_counter_passed':True,
                'training_data_read':False,'model_forward_calls':0,'test_evaluations':0,'formal_updates':0},
             'scientific_recipe_changed':False}
    write_replace(HERE/'preparation_ready.json',receipt)
    print(json.dumps({'status':receipt['status'],'engineering_attempt':2,'sha256':sha(HERE/'preparation_ready.json'),
                      'test_summary':(tests.stdout+tests.stderr)[-150:],'changed_sources':changed}))


if __name__=='__main__':main()
