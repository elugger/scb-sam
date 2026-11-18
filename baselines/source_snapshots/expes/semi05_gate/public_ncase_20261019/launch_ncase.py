"""Own GPU0 five-case launch only: verify -> explicit prepare -> single dispatch.

No network, uploading, automatic queue, resume, retry or GPU1 scheduling.
Run this only after authorized deployment and actual bundle verification.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from datetime import datetime, timezone

from ncase_protocol import ROOT, SRC, BASE, artifact_dir, ids, sha, validate_splits, make_config

HERE = Path(__file__).resolve().parent
PYTHON = '__RECORDED_PYTHON__'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def verify_bundle(expected):
    require(HERE == Path(SRC), 'Unexpected deployment location')
    require(sha(HERE/'deployment_manifest.json') == expected, 'Deployment manifest differs')
    rows = json.loads((HERE/'deployment_manifest.json').read_text())['files']
    require(rows and len({r['name'] for r in rows}) == len(rows), 'Empty/duplicate manifest')
    for row in rows:
        relative = Path(row['name'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Invalid bundle path')
        p = HERE/relative
        require(not any(x.is_symlink() for x in (p,*p.parents)), 'Symlink in code bundle')
        require(p.is_file() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], 'Bundle member differs')
    members = {r['name'] for r in rows}
    require({p.name for p in HERE.glob('*.py')} <= members, 'Unbound executable in directory')
    return rows


def checked_config():
    import train_ncase as runner
    import ncase_inputs
    cfg = make_config(HERE, 5)
    runner.validate_config(cfg)
    runner.verify_manifests(cfg)
    splits = validate_splits(HERE/'splits')
    oldroot = Path(ROOT)/'expes/splits/public/public_semi05_cv5/fold2'
    for old, new in (('train_lab.txt','original_lab17.txt'), ('train_whole.txt','train_whole.txt'), ('test.txt','test.txt')):
        require(ids(oldroot/old) == ids(HERE/'splits'/new), 'Local snapshot IDs differ from actual remote fold')
    cfg['case_file_metadata'] = ncase_inputs.bind(cfg, splits[5]['lab'], splits[5]['test'])
    # Actual scientific imports are checked in the remote Torch environment, without CUDA work.
    for name in ('segment_anything.auto_sam_seg','segment_anything.build_sam3D_cascade',
                 'segment_anything.modeling.sam3D_cascade','utils.auto_sam_loss',
                 'utils.cascade_train_utils','expes.common.semi_data','expes.common.seg_metrics'):
        module = importlib.import_module(name)
        p = Path(module.__file__).resolve()
        rel = str(p.relative_to(ROOT))
        require(rel in cfg['source_sha256'] and sha(p) == cfg['source_sha256'][rel], 'Imported scientific source differs')
    import ncase_resources as resources
    with resources.original_lock():
        capacity = resources.capacity_snapshot(config=cfg)
    return cfg, capacity


def assert_fresh(cfg):
    control = Path(BASE)/'control'
    require(not any(control.glob('sup_n005.*')), 'Existing intent/dispatch/boot/error: inspect, never retry')
    for target in (Path(cfg['output_dir']), Path(cfg['artifact_dir'])):
        require(not any(p.is_symlink() for p in (target,*target.parents)), 'Symlink run path')
        require(not os.path.lexists(target), 'Existing output/checkpoint/partial directory')


def idle_gpu0():
    rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total',
                                    '--format=csv,noheader,nounits'], text=True)
    selected = next([v.strip() for v in row.split(',')] for row in rows.splitlines() if row.split(',')[0].strip()=='0')
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name',
                                         '--format=csv,noheader'], text=True)
    require(int(selected[2]) < 256 and int(selected[3]) >= 23000 and selected[1] not in processes,
            'GPU0 not idle/large enough')
    return selected[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('check','prepare','launch'), required=True)
    parser.add_argument('--bundle-sha', required=True)
    parser.add_argument('--preparation-sha')
    args = parser.parse_args()
    require(Path(sys.executable).resolve() == Path(PYTHON).resolve(), 'Wrong actual Python')
    sys.path.insert(0, str(ROOT))
    verify_bundle(args.bundle_sha)
    cfg, capacity = checked_config()
    control = Path(BASE)/'control'
    prepfile = control/'preparation.json'
    cfgfile = control/'config_n005.json'
    if args.action in ('check','prepare'):
        check = subprocess.run([PYTHON,'-m','unittest','-v','test_ncase'], cwd=HERE,
                               capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        require(check.returncode == 0, check.stdout+check.stderr)
        report = dict(status='REMOTE_CPU_CHECKED_NOT_LAUNCHED', time=datetime.now(timezone.utc).isoformat(),
                      bundle_sha256=args.bundle_sha, capacity=capacity, test_output=check.stdout+check.stderr,
                      labeled_cases=5, successful_updates=0, test_evaluations=0)
        if args.action == 'prepare':
            require(not Path(BASE).exists(), 'Existing preparation: preserve, do not repeat')
            assert_fresh(cfg)
            control.mkdir(parents=True, exist_ok=False)
            write_new(cfgfile, cfg)
            report['config_sha256'] = sha(cfgfile)
            write_new(prepfile, report)
            report['preparation_sha256'] = sha(prepfile)
        print(json.dumps(report, ensure_ascii=False))
        return
    require(args.preparation_sha and sha(prepfile)==args.preparation_sha, 'Own preparation SHA required')
    prep = json.loads(prepfile.read_text())
    require(prep['bundle_sha256']==args.bundle_sha and prep['status']=='REMOTE_CPU_CHECKED_NOT_LAUNCHED', 'Wrong own preparation')
    require(sha(cfgfile)==prep['config_sha256'] and json.loads(cfgfile.read_text())==cfg, 'Actual inputs/config changed since prepare')
    import fcntl
    require(not any(p.is_symlink() for p in (control,*control.parents)), 'Symlink control')
    fd = os.open(control/'gpu0.lock', os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW, 0o600)
    try:
        claim = os.fstat(fd)
        require(stat.S_ISREG(claim.st_mode) and claim.st_nlink==1, 'Bad GPU claim')
        fcntl.flock(fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert_fresh(cfg)
        uuid = idle_gpu0()
        import ncase_resources as resources
        capacity = resources.enroll(cfg)  # Original save.lock, live shared registry, atomic write.
        command = [PYTHON,'-u',str(HERE/'train_ncase.py'),'--config',str(cfgfile),
                   '--config-sha256',sha(cfgfile),'--gpu','0','--gpu-claim-fd',str(fd),
                   '--gpu-claim-device',str(claim.st_dev),'--gpu-claim-inode',str(claim.st_ino)]
        intent = dict(time=datetime.now(timezone.utc).isoformat(), gpu=0, gpu_uuid=uuid, command=command,
                      config_sha256=sha(cfgfile), preparation_sha256=args.preparation_sha,
                      capacity=capacity, automatic_retry=False)
        write_new(control/'sup_n005.intent.json', intent)
        try:
            with (control/'sup_n005.boot.log').open('xb') as log:
                child = subprocess.Popen(command, cwd=str(ROOT), stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(fd,),
                           env=dict(os.environ, PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES='0',
                                    PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1'))
            intent.update(pid=child.pid, status='DISPATCHED_NOT_STEP_VERIFIED')
            write_new(control/'sup_n005.dispatch.json', intent)
            print(json.dumps(intent))
        except BaseException as error:
            write_new(control/'sup_n005.dispatch_error.json', dict(error=str(error), automatic_retry=False))
            raise
    finally:
        os.close(fd)


if __name__=='__main__':
    main()
