import argparse, importlib, json, os, stat, subprocess, sys, time, traceback, uuid
from pathlib import Path
from four_contract import *

def prepare(args):
    cfg, ancestor, accepted, initial = verify_sources(args.bundle_sha)
    require(not CONTROL.exists(), 'Existing preparation; investigate before another action')
    import torch
    require(not torch.cuda.is_initialized(), 'Preparation may not initialize CUDA')
    for name in ('expes.semi05_gate.paired_fcd_matrix_20261006', 'expes.semi05_gate.fcd_deterministic_losses_r2_20261005', 'expes.common.seg_metrics', 'expes.common.semi_data', 'utils.cascade_train_utils'):
        importlib.import_module(name)
    sources = {}
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        p = Path(filename).resolve()
        if ROOT in p.parents and p.is_file():
            sources[str(p)] = sha(p)
    require(len(sources) > 30, 'Scientific dependency collection incomplete')
    from train_by5 import read_case_ids
    from by_inputs import bind, identity
    split = ROOT / 'expes/splits/by_semi05_cv5/fold2'
    ids = {k: read_case_ids(split / (k + '.txt')) for k in ('train_lab', 'train_unlab', 'test')}
    require([len(ids[k]) for k in ids] == [5, 95, 25] and len(set(sum(ids.values(), []))) == 125, 'Case split differs')
    metadata = bind(cfg, ids['train_lab'] + ids['train_unlab'], ids['test'])
    inputs = {}
    for cases in metadata.values():
        for pair in cases.values():
            for row in pair.values():
                p = row['path']
                inputs[p] = dict(identity=row, sha256=sha(p))
                require(identity(p, ROOT / 'data') == row, 'Input changed while hashing')
    from by_runtime import snapshot
    with original_lock():
        capacity = capacity_snapshot(config=cfg)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(SRC) + ':' + str(ROOT))
    test = subprocess.run([sys.executable, '-B', str(SRC / 'test_four.py')], cwd=ROOT, env=env, capture_output=True, text=True)
    require(test.returncode == 0, 'CPU contract checks failed: ' + test.stdout + test.stderr)
    CONTROL.mkdir()
    result = dict(status='PREPARED_NO_TRAINING', at=stamp(), bundle_sha256=args.bundle_sha, source_hashes=sources, initializer_sha256=INIT_SHA, initialization_check_sha256=sha(BASE / 'four_arm_initialization_check/result.json'), case_file_metadata=metadata, inputs=inputs, case_ids=ids, torch_version=str(torch.__version__), python_version=sys.version, capacity=capacity, gpu0=snapshot(), cpu_tests=test.stdout + test.stderr, input_files=len(inputs), automatic_retry=False)
    dump_new(CONTROL / 'preparation.json', result)
    print(json.dumps(dict(status=result['status'], at=result['at'], preparation_sha256=sha(CONTROL / 'preparation.json'), source_count=len(sources), input_files=len(inputs), cpu_tests=result['cpu_tests'])), flush=True)

def launch(args):
    cfg, *_ = verify_prepared(args.bundle_sha, args.prepare_sha)
    phase = args.phase
    out = BASE / phase
    require(not out.exists() and (not any(CONTROL.glob(phase + '.*'))), 'Existing phase intent/output; no repeated dispatch')
    bridge = validate_bridge(args.bundle_sha, args.prepare_sha) if phase == 'formal' else None
    if phase == 'formal':
        require(args.bridge_sha == bridge['sha256'], 'Expected bridge receipt differs')
        require(all((not (STORE / g).exists() for g in GROUPS)), 'Nonempty formal store')
    import fcntl
    claim_path = BASE / 'control/gpu0.lock'
    infra().no_symlink(claim_path)
    fd = os.open(claim_path, os.O_RDWR | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        s = os.fstat(fd)
        require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, 'Invalid original GPU claim')
        from by_runtime import ensure_gpu0_idle
        gpu = ensure_gpu0_idle()
        with original_lock():
            capacity = capacity_snapshot(config=cfg)
        nonce = uuid.uuid4().hex
        command = [sys.executable, '-u', '-B', str(SRC / 'four_control.py'), '--action', 'supervise', '--phase', phase, '--bundle-sha', args.bundle_sha, '--prepare-sha', args.prepare_sha, '--nonce', nonce, '--fd', str(fd), '--dev', str(s.st_dev), '--ino', str(s.st_ino)]
        if bridge:
            command += ['--bridge-sha', bridge['sha256']]
        dump_new(CONTROL / (phase + '.intent.json'), dict(at=stamp(), nonce=nonce, command=command, bundle_sha256=args.bundle_sha, preparation_sha256=args.prepare_sha, bridge=bridge, phase=phase, gpu0=gpu, capacity=capacity, automatic_retry=False, automatic_resume=False))
        with (CONTROL / (phase + '.boot.log')).open('xb') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, pass_fds=(fd,), start_new_session=True, env=dict(os.environ, PYTHONPATH=str(SRC) + ':' + str(ROOT), CUDA_VISIBLE_DEVICES=GPU, CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
        result = dict(at=stamp(), phase=phase, pid=child.pid, nonce=nonce, intent_sha256=sha(CONTROL / (phase + '.intent.json')), stage='DISPATCHED_NOT_YET_VERIFIED_IN_UPDATE', bundle_sha256=args.bundle_sha, preparation_sha256=args.prepare_sha)
        dump_new(CONTROL / (phase + '.dispatch.json'), result)
        print(json.dumps(result), flush=True)
    finally:
        os.close(fd)

def supervise(args):
    cfg, *_ = verify_prepared(args.bundle_sha, args.prepare_sha)
    from train_by5 import validate_gpu_claim
    validate_gpu_claim(cfg, 0, args.fd, args.dev, args.ino)
    intent = json.loads((CONTROL / (args.phase + '.intent.json')).read_text())
    require(intent['nonce'] == args.nonce and intent['bundle_sha256'] == args.bundle_sha and (intent['preparation_sha256'] == args.prepare_sha), 'Control binding differs')
    if args.phase == 'formal':
        require(validate_bridge(args.bundle_sha, args.prepare_sha)['sha256'] == args.bridge_sha, 'Bridge differs')
    out = BASE / args.phase
    out.mkdir(exist_ok=False)
    dump_new(out / 'started.json', dict(at=stamp(), phase=args.phase, pid=os.getpid(), nonce=args.nonce, intent_sha256=sha(CONTROL / (args.phase + '.intent.json')), groups=list(GROUPS)))
    status = 'FAILED'
    error = None
    try:
        for group in GROUPS:
            command = [sys.executable, '-u', '-B', str(SRC / 'four_control.py'), '--action', 'run', '--phase', args.phase, '--group', group, '--bundle-sha', args.bundle_sha, '--prepare-sha', args.prepare_sha, '--fd', str(args.fd), '--dev', str(args.dev), '--ino', str(args.ino)]
            if args.bridge_sha:
                command += ['--bridge-sha', args.bridge_sha]
            dump_new(out / (group + '.intent.json'), dict(at=stamp(), command=command, group=group, phase=args.phase, automatic_retry=False))
            with (out / (group + '.log')).open('xb') as log:
                p = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, pass_fds=(args.fd,))
                dump_new(out / (group + '.dispatch.json'), dict(at=stamp(), pid=p.pid, group=group, intent_sha256=sha(out / (group + '.intent.json'))))
                code = p.wait()
            dump_new(out / (group + '.exited.json'), dict(at=stamp(), returncode=code, group=group))
            require(code == 0, 'Arm failed; stop entire batch: ' + group)
        rows = {g: json.loads((out / g / 'result.json').read_text()) for g in GROUPS}
        status = 'PASS_FOUR_ARM_FINITE_BRIDGE' if args.phase == 'bridge' else 'FOUR_ARMS_COMPLETE_PENDING_INDEPENDENT_ACCEPTANCE'
        result = dict(at=stamp(), status=status, bundle_sha256=args.bundle_sha, preparation_sha256=args.prepare_sha, child_result_sha256={g: sha(out / g / 'result.json') for g in GROUPS})
        if args.phase == 'formal':
            dice = {g: 100 * rows[g]['metrics']['dice'] for g in GROUPS}
            differences = {'HP-H': dice['HP'] - dice['H'], 'HQ-H': dice['HQ'] - dice['H'], 'HPQ-HQ': dice['HPQ'] - dice['HQ'], 'HPQ-HP': dice['HPQ'] - dice['HP']}
            result.update(dice_percent=dice, differences_pp=differences, target_met_before_independent_acceptance=all((v >= 1 for v in differences.values())), independent_test=False, selection='same_batch_fixed_epoch35_student')
        dump_new(out / 'result.json', result)
    except BaseException:
        error = traceback.format_exc()
        raise
    finally:
        dump_new(out / 'exited.json', dict(at=stamp(), status=status, error=error, automatic_retry=False))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--action', choices=('prepare', 'launch', 'supervise', 'run'), required=True)
    p.add_argument('--bundle-sha', required=True)
    p.add_argument('--prepare-sha')
    p.add_argument('--phase', choices=('bridge', 'formal'))
    p.add_argument('--bridge-sha')
    p.add_argument('--group', choices=GROUPS)
    p.add_argument('--nonce')
    for n in ('fd', 'dev', 'ino'):
        p.add_argument('--' + n, type=int)
    args = p.parse_args()
    os.chdir(ROOT)
    require(Path(__file__).resolve() == SRC / 'four_control.py', 'Noncanonical control')
    if args.action == 'prepare':
        prepare(args)
    elif args.action == 'launch':
        launch(args)
    elif args.action == 'supervise':
        supervise(args)
    else:
        from four_training import run
        run(args)
if __name__ == '__main__':
    main()
