import argparse
import fcntl
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import uuid
from by_protocol import *
from prepare_by5 import dump_new, stamp

def verify_preparation(bundle_sha, preparation_sha):
    verify_bundle(bundle_sha)
    p = Path(CONTROL / 'preparation.json')
    require(sha(p) == preparation_sha, 'Preparation SHA changed')
    report = json.loads(p.read_text())
    require(report['status'] == 'CPU_PREPARED_NOT_DISPATCHED' and report['source_manifest_sha256'] == bundle_sha, 'Wrong preparation')
    config_path = Path(CONTROL / 'supervised_config.json')
    require(sha(config_path) == report['config_sha256'], 'Prepared configuration changed')
    cfg = json.loads(config_path.read_text())
    validate_supervised_config(cfg)
    from train_by5 import verify_manifests
    verify_manifests(cfg)
    from by_inputs import check_case
    for role, cases in cfg['case_file_metadata'].items():
        for case, files in cases.items():
            check_case(cfg, case, role, files['imagesTr']['path'], files['labelsTr']['path'])
    return (cfg, report)

def fresh():
    control = Path(CONTROL)
    require(not any(control.glob('supervised.*')), 'Existing supervised intent/dispatch/exit; no repeat')
    require(not Path(BASE / 'supervised').exists() and (not Path(STORE / 'supervised').exists()), 'Existing supervised output')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle-sha', required=True)
    parser.add_argument('--preparation-sha', required=True)
    parser.add_argument('--wait-for-gpu', action='store_true')
    args = parser.parse_args()
    require(__debug__ and Path(__file__).resolve() == Path(SRC / 'launch_by5.py'), 'Wrong launcher location')
    os.chdir(str(ROOT))
    cfg, prep = verify_preparation(args.bundle_sha, args.preparation_sha)
    from by_resources import original_lock, capacity_snapshot, enroll_locked, infra
    from by_runtime import snapshot, is_idle
    fresh()
    claim_path = Path(CONTROL / 'gpu0.lock')
    infra().no_symlink(claim_path)
    fd = os.open(claim_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 384)
    try:
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'Invalid GPU claim')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fresh()
        with original_lock():
            capacity = capacity_snapshot(config=cfg)
            gpu = snapshot()
            require(args.wait_for_gpu or is_idle(gpu), 'GPU0 busy; prelaunch waiting was not requested')
            enrollment = enroll_locked()
            nonce = uuid.uuid4().hex
            cmd = [sys.executable, '-u', '-B', str(SRC / 'supervise_by5.py'), '--bundle-sha', args.bundle_sha, '--preparation-sha', args.preparation_sha, '--nonce', nonce, '--claim-fd', str(fd), '--claim-device', str(info.st_dev), '--claim-inode', str(info.st_ino)]
            intent = dict(at=stamp(), nonce=nonce, command=cmd, gpu=0, uuid=GPU_UUID, config_sha256=prep['config_sha256'], source_manifest_sha256=args.bundle_sha, preparation_sha256=args.preparation_sha, initial_gpu0=gpu, capacity=capacity, enrollment=enrollment, automatic_training_retry=False, wait_for_gpu=args.wait_for_gpu, maximum_gpu_wait_seconds=86400, stage='SHARED_BY5_SUPERVISION_ONLY')
            dump_new(Path(CONTROL / 'supervised.intent.json'), intent)
            with Path(CONTROL / 'supervised.boot.log').open('xb') as log:
                child = subprocess.Popen(cmd, cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(fd,), env=dict(os.environ, PYTHONPATH=':'.join(map(str, (SRC, INFRA, ROOT))), CUDA_VISIBLE_DEVICES=GPU_UUID, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1'))
            dispatch = dict(at=stamp(), pid=child.pid, nonce=nonce, stage='ADMISSION_WORKER_DISPATCHED', training_started=False, intent_sha256=sha(CONTROL / 'supervised.intent.json'))
            dump_new(Path(CONTROL / 'supervised.dispatch.json'), dispatch)
        print(json.dumps(dispatch))
    finally:
        os.close(fd)
if __name__ == '__main__':
    main()
