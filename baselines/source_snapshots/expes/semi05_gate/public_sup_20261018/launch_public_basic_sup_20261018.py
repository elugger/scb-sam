"""One explicit fresh arm launch. No queue, resume, retry or scientific edits."""
import argparse
import fcntl
import hashlib
import json
import os
import stat
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path('__RECORDED_PROJECT_ROOT__')
HERE = ROOT/'expes/semi05_gate/public_sup_20261018'
BASE = ROOT/'expes/work_dir/semi05_gate/PUBLIC_BASIC_SUP_FOLD2_20261018'
STORE = ROOT/'expes/artifacts/FCD_A_MATRIX_20261006/PUBLIC_BASIC_SUP_FOLD2_20261018'
PYTHON = '__RECORDED_PYTHON__'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, choices=['sup05', 'sup10', 'sup100'])
    parser.add_argument('--gpu', required=True, type=int, choices=[0, 1])
    parser.add_argument('--preparation-sha', required=True)
    args = parser.parse_args()
    require(Path(sys.executable).resolve() == Path(PYTHON).resolve(), 'Wrong Python environment')
    prep_path = HERE/'preparation_ready.json'
    require(sha(prep_path) == args.preparation_sha, 'Preparation identity mismatch')
    prep = json.loads(prep_path.read_text())
    require(prep['status'] == 'CPU_READY_NOT_LAUNCHED', 'Preparation not ready')
    require(isinstance(prep['source_sha256'], dict) and prep['source_sha256'], 'Empty sources')
    require(isinstance(prep['config_sha256'], dict) and prep['config_sha256'], 'Empty configurations')
    for name in ('launch_public_basic_sup_20261018.py', 'train_public_basic_sup_20261018.py',
                 'public_basic_sup_guard_20261018.py'):
        require(str((HERE/name).relative_to(ROOT)) in prep['source_sha256'], 'Unbound required source: '+name)
    require(args.arm+'.json' in prep['config_sha256'], 'Selected config not bound')
    for name, digest in prep['source_sha256'].items():
        require(sha(ROOT/name) == digest, 'Source changed: '+name)
    for name, digest in prep['config_sha256'].items():
        require(sha(HERE/name) == digest, 'Configuration changed: '+name)
    config_path = HERE/(args.arm+'.json')
    config = json.loads(config_path.read_text())
    require(config['artifact_dir'] == str(STORE/args.arm) and config['output_dir'] == str(BASE/args.arm),
            'Noncanonical arm directories')
    import train_public_basic_sup_20261018 as runner
    import public_basic_sup_guard_20261018 as guard
    runner.validate_config(config)
    runner.verify_manifests(config)
    control = BASE/'control'
    guard.no_symlink(control)
    require(control.is_dir() and not control.is_symlink(), 'Control directory not prepared')
    claim = os.open(control/('gpu'+str(args.gpu)+'.lock'), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    claim_stat = os.fstat(claim)
    require(stat.S_ISREG(claim_stat.st_mode) and claim_stat.st_nlink == 1, 'Invalid GPU claim file')
    fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not any(control.glob(args.arm+'.*')), 'Existing dispatch/boot/intent: inspect, do not relaunch')
    for path in (BASE/args.arm, STORE/args.arm):
        guard.no_symlink(path)
        require(not os.path.lexists(path), 'Existing arm output/artifact/partial')
    # A completed 5% arm is the explicit prerequisite for reusing GPU0 for 10%.
    if args.arm == 'sup10':
        result_path = BASE/'sup05/final_metrics.json'
        require(result_path.is_file(), '5% must finish and be inspected before 10%')
        result = json.loads(result_path.read_text())
        require(result['status'] == 'COMPLETE' and result['successful_updates'] == 1020,
                '5% completion is not valid')
        for key, expected in [('completed_epochs',60), ('fold',2), ('label_percent',5), ('seed',2027),
                              ('test_evaluations',1), ('selection_rule','fixed_epoch_60_student')]:
            require(result[key] == expected, '5% completion field mismatch: '+key)
        test_ids = runner.read_case_ids(runner.resolve_path(config, config['test_list']))
        require(result['test_actual_case_ids'] == test_ids and len(test_ids) == 87, '5% test coverage mismatch')
        require(sha(STORE/'sup05/final.pth') == result['final_sha256'], '5% final identity mismatch')
        require(json.loads((BASE/'sup05/progress.json').read_text()) == result, '5% progress mismatch')
        require(not list((STORE/'sup05').glob('*.partial')), '5% partial save remains')
        acceptance = json.loads((control/'sup05.acceptance.json').read_text())
        require(acceptance['status'] == 'VERIFIED_COMPLETE' and
                acceptance['final_metrics_sha256'] == sha(result_path) and
                acceptance['final_sha256'] == result['final_sha256'], 'Independent 5% acceptance missing/mismatched')
        require(not (BASE/'sup05/failure.json').exists(), '5% failure must be resolved first')
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total',
                                     '--format=csv,noheader,nounits'], text=True)
    gpu_rows = [row.split(', ') for row in query.strip().splitlines()]
    selected = next(row for row in gpu_rows if int(row[0]) == args.gpu)
    require(int(selected[2]) < 256 and int(selected[3]) >= 23000, 'Selected GPU is not idle/large enough')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                                         '--format=csv,noheader'], text=True)
    require(selected[1] not in processes, 'Selected GPU already has a compute process')
    with guard.original_lock():
        capacity = guard.capacity_snapshot(required_ram=14*guard.GIB+guard.MARGIN)
    command = [PYTHON, '-u', str(HERE/'train_public_basic_sup_20261018.py'), '--config', str(config_path),
               '--gpu', str(args.gpu), '--config-sha256', sha(config_path),
               '--gpu-claim-fd', str(claim), '--gpu-claim-device', str(claim_stat.st_dev),
               '--gpu-claim-inode', str(claim_stat.st_ino)]
    intent = {'time': datetime.now(timezone.utc).isoformat(), 'arm': args.arm, 'gpu': args.gpu,
              'gpu_uuid': selected[1], 'command': command, 'preparation_sha256': args.preparation_sha,
              'config_sha256': sha(config_path), 'capacity': capacity, 'automatic_retry': False}
    intent['gpu_claim'] = {'device':claim_stat.st_dev, 'inode':claim_stat.st_ino, 'nlink':claim_stat.st_nlink}
    write_new(control/(args.arm+'.intent.json'), intent)
    try:
        env = os.environ.copy()
        env.update(PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED='1')
        with (control/(args.arm+'.boot.log')).open('xb') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(claim,))
        intent['pid'] = child.pid
        intent['state'] = 'DISPATCHED_NOT_YET_STEP_VERIFIED'
        write_new(control/(args.arm+'.dispatch.json'), intent)
        print(json.dumps({'arm': args.arm, 'pid': child.pid, 'gpu': args.gpu,
                          'state': intent['state'], 'config_sha256': intent['config_sha256']}))
    except BaseException as error:
        write_new(control/(args.arm+'.dispatch_error.json'), {'error': str(error), 'no_automatic_retry': True})
        raise
    finally:
        os.close(claim)  # Child holds the same GPU claim open until it exits.


if __name__ == '__main__':
    main()
