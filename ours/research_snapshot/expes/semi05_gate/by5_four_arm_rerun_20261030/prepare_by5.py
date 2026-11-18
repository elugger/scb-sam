import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
from by_protocol import *

def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()

def dump_new(path, value):
    from by_resources import infra
    infra().no_symlink(path)
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    infra().fsync_directory(Path(path).parent)

def make_bound_config(bundle_sha):
    cfg = base_config()
    cfg['source_sha256'] = verify_bundle(bundle_sha)
    for name in ('dependency_pins.json', 'infrastructure_pins.json'):
        for p, h in json.loads((Path(SRC) / name).read_text()).items():
            full = str(ROOT / p) if not Path(p).is_absolute() else p
            require(full not in cfg['source_sha256'] or cfg['source_sha256'][full] == h, 'Conflicting source pin')
            require(sha(full) == h, 'Dependency changed: ' + full)
            cfg['source_sha256'][full] = h
    from train_by5 import read_case_ids, validate_case_sets, verify_manifests
    cases = {}
    for name, h in SPLIT_SHA.items():
        require(sha(SPLIT / name) == h, 'BY5 split changed')
        cases[name] = read_case_ids(SPLIT / name)
    require(tuple((len(cases[n]) for n in SPLIT_SHA)) == (5, 95, 25), 'BY5 counts differ')
    validate_case_sets(cases['train_lab.txt'], cases['test.txt'], 5, 25)
    require(not set(cases['train_lab.txt']) & set(cases['train_unlab.txt']), 'L/U overlap')
    require(not set(cases['train_unlab.txt']) & set(cases['test.txt']), 'U/development overlap')
    from by_inputs import bind
    cfg['case_file_metadata'] = bind(cfg, cases['train_lab.txt'], cases['test.txt'])
    for role in cfg['case_file_metadata'].values():
        for case in role.values():
            for identity in case.values():
                p = identity['path']
                cfg['input_sha256'][p] = sha(p)
    cfg['source_manifest_sha256'] = bundle_sha
    validate_supervised_config(cfg)
    verify_manifests(cfg)
    require(bind(cfg, cases['train_lab.txt'], cases['test.txt']) == cfg['case_file_metadata'], 'Case identity changed during binding')
    return cfg

def prepare(bundle_sha):
    require(__debug__ and Path(__file__).resolve() == Path(SRC / 'prepare_by5.py'), 'Wrong deployment or optimized Python')
    os.chdir(str(ROOT))
    cfg = make_bound_config(bundle_sha)
    from by_resources import original_lock, capacity_snapshot, infra
    from by_runtime import snapshot
    tests = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'test_by5'], cwd=str(SRC), capture_output=True, text=True, timeout=120)
    require(tests.returncode == 0, 'CPU tests failed: ' + tests.stdout + tests.stderr)
    control = Path(CONTROL)
    infra().no_symlink(control)
    require(not control.exists(), 'Existing preparation/control; inspect instead of repeating')
    for p, _ in targets():
        infra().no_symlink(Path(p))
        require(not os.path.lexists(p) and (not os.path.lexists(p + '.partial')), 'Existing target')
    require(not Path(BASE / 'supervised').exists() and (not Path(STORE).exists()), 'Existing output/store')
    with original_lock():
        capacity = capacity_snapshot(config=cfg)
        gpu = snapshot()
        control.mkdir(parents=True, exist_ok=False)
        dump_new(control / 'supervised_config.json', cfg)
        report = dict(at=stamp(), status='CPU_PREPARED_NOT_DISPATCHED', source_manifest_sha256=bundle_sha, config_sha256=sha(control / 'supervised_config.json'), train_count=5, unlab_count=95, development_count=25, source_count=len(cfg['source_sha256']), supervised_updates=300, supervised_epochs=60, selected_endpoint='fixed_epoch60_student', source_derivation='same supervised training computation as pinned train_ncase; BY5 contract and admission only', cpu_test_stdout=tests.stdout, cpu_test_stderr=tests.stderr, case_input_files=60, case_input_bytes_sha_bound=True, capacity=capacity, gpu0=gpu, training_started=False)
        dump_new(control / 'preparation.json', report)
    print(json.dumps(dict(status=report['status'], preparation_sha256=sha(control / 'preparation.json'), source_manifest_sha256=bundle_sha, config_sha256=report['config_sha256'], source_count=report['source_count'], supervised_updates=300, cpu_tests=tests.stderr, gpu0=gpu, filesystems=[{k: v for k, v in r.items() if k != 'targets'} for r in capacity['filesystems']], memory=capacity['memory']), ensure_ascii=False))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle-sha', required=True)
    prepare(parser.parse_args().bundle_sha)
