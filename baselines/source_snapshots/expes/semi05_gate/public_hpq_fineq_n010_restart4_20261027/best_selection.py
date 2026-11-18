"""Separate best artifact, verified before replacement; never select a rolling slot."""
import json
import os
from pathlib import Path

from hpq_contract import STORE, INIT_SHA, require, sha_file
from hpq_inputs import plain_file
from hpq_resources import backend, RAM_REQUIRED, no_symlink, fsync_directory
from hpq_safety_state import cpu_tree, write_verified_partial
from selection_protocol import policy_sha
from capacity_wait import admitted_capacity

FORMAT = 'PUBLIC_N010_FINEQ_SELECTED_SIX_ROOTS_V1'


def validate_metadata(value, epoch, updates, manifest_sha, metric_sha, case_sha):
    require(value['format'] == FORMAT and value['epoch'] == epoch and value['updates'] == updates,
            'Selected checkpoint epoch/update mismatch')
    require(value['initializer_sha256'] == INIT_SHA and value['manifest_sha256'] == manifest_sha
            and value['metrics_sha256'] == metric_sha and value['cases_sha256'] == case_sha
            and value['selection_policy_sha256'] == policy_sha(), 'Selected evidence binding mismatch')
    require(set(value['roots']) == {'student', 'head', 'adapter', 'teacher', 'teacher_head', 'teacher_adapter'},
            'Missing selected roots')
    require(value['automatic_resume'] is False, 'Best is not a continuation checkpoint')


def validate_previous(path, previous):
    no_symlink(path)
    if previous is None:
        require(not os.path.lexists(path), 'Unowned selected checkpoint exists')
    else:
        require(previous['path'] == str(path), 'Foreign previous best')
        s = plain_file(path)
        require([s.st_dev, s.st_ino, s.st_size] == previous['identity']
                and sha_file(path) == previous['sha256'], 'Previous best identity changed')


def save_best(engine, epoch, previous=None, probe=False):
    from hpq_engine import write_json
    require((probe and engine.spec['phase'] == 'bridge' and epoch == -1) or
            (not probe and engine.spec['phase'] == 'formal' and 0 <= epoch < 36
             and engine.t.matrix_updates == (epoch + 1) * 68), 'Wrong best-save boundary')
    target = Path(STORE) / 'best/final.pth'
    receipt_path = engine.output / ('best_probe.json' if probe else 'best_epoch_%02d.json' % epoch)
    require(not receipt_path.exists(), 'Best receipt already exists')
    before, rng = engine.state(), engine.rng()
    guard = backend()
    with admitted_capacity(guard, RAM_REQUIRED, engine.output) as capacity:
        validate_previous(target, previous)
        require(not os.path.lexists(str(target) + '.partial'), 'Preserve existing best partial')
        manifest_sha = sha_file(engine.output / 'manifest.json')
        metric_sha = None if probe else sha_file(engine.output / ('epoch_%02d_test_metrics.json' % epoch))
        case_sha = None if probe else sha_file(engine.output / ('epoch_%02d_test_cases.jsonl' % epoch))
        value = dict(format=FORMAT, epoch=epoch, updates=engine.t.matrix_updates,
                     initializer_sha256=INIT_SHA, cfg=engine.cfg, manifest_sha256=manifest_sha,
                     metrics_sha256=metric_sha, cases_sha256=case_sha, selection_policy_sha256=policy_sha(),
                     roots=cpu_tree(engine.torch, {k: m.state_dict() for k, m in engine.t.roots().items()}),
                     automatic_resume=False, development_selection_used=True, probe=probe)
        validate_metadata(value, epoch, engine.t.matrix_updates, manifest_sha, metric_sha, case_sha)
        created = [p for p in (target.parent, Path(STORE)) if not p.exists()]
        target.parent.mkdir(parents=True, exist_ok=True)
        partial, row = write_verified_partial(engine, guard, value, target, capacity)
        validate_previous(target, previous)
        require(engine.state() == before and engine.rng() == rng, 'Best save changed state/RNG')
        info = plain_file(partial)
        require([info.st_dev, info.st_ino, info.st_size] == row['identity']
                and sha_file(partial) == row['sha256'], 'Verified partial changed')
        if probe:
            # Only remove this successful newly owned probe; any prior error
            # preserves it. No old data is removed.
            partial.unlink()
            fsync_directory(target.parent)
            for p in sorted(created, key=lambda p: len(p.parts), reverse=True):
                require(not any(p.iterdir()), 'Unexpected probe directory residual')
                p.rmdir()
        else:
            os.replace(partial, target)
            fsync_directory(target.parent)
            require(sha_file(target) == row['sha256'], 'Published best differs')
        receipt = dict(row, status='BEST_PROBE_PASS' if probe else 'BEST_SAVED_READBACK_VERIFIED',
                       epoch=epoch, updates=engine.t.matrix_updates, metrics_sha256=metric_sha,
                       cases_sha256=case_sha, manifest_sha256=manifest_sha,
                       previous_sha256=None if previous is None else previous['sha256'],
                       selection_policy_sha256=policy_sha(), strict_cpu_readback_verified=True,
                       previous_preserved_until_verified_replace=True, automatic_resume=False)
        write_json(receipt_path, receipt)
        fsync_directory(engine.output)
    return receipt


def verify_endpoint(engine, selected):
    """Read selected bytes and bind their metrics; no second model evaluation."""
    from hpq_engine import load_torch
    target = Path(STORE) / 'best/final.pth'
    guard = backend()
    with admitted_capacity(guard, RAM_REQUIRED, engine.output) as capacity:
        validate_previous(target, selected)
        value = load_torch(engine.torch, str(target))
        epoch = selected['epoch']
        metrics = engine.output / ('epoch_%02d_test_metrics.json' % epoch)
        cases = engine.output / ('epoch_%02d_test_cases.jsonl' % epoch)
        validate_metadata(value, epoch, (epoch + 1) * 68, sha_file(engine.output / 'manifest.json'),
                          sha_file(metrics), sha_file(cases))
        require(engine.signature(value) == selected['content_sha256'], 'Endpoint selected contents differ')
        validate_previous(target, selected)
        del value
    return dict(status='SELECTED_MODEL_ENDPOINT_RELOAD_VERIFIED', selected=selected,
                metrics=json.loads(metrics.read_text()), capacity=capacity, new_evaluations=0)
