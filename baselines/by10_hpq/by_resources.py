"""New BY5 reservations layered over the unchanged latest historical union."""
import json
import os
from pathlib import Path
import sys
from by_protocol import ROOT, INFRA, STORE, targets, require, sha, GIB

RAM_REQUIRED = 12 * GIB + 16 * 1024**2
REGISTRY = Path(ROOT / 'expes/work_dir/semi05_gate/MAIN_TABLE_FIXED_FINAL_20261019/active_targets.json')


def infra():
    if str(INFRA) not in sys.path:
        sys.path.insert(0, str(INFRA))
    import hpq_resources
    return hpq_resources


def merge(current, extra):
    result = {str(p): b for p, b in current}
    require(len(result) == len(current), 'Duplicate current reservations')
    for p, b in extra:
        require(p not in result or result[p] == b, 'Own reservation bound changed')
        result[p] = b
    return [(Path(p), b) for p, b in result.items()]


def backend():
    g = infra().backend()
    previous = g.targets
    g.targets = lambda: merge(previous(), targets())
    return g


def original_lock():
    return backend().original_lock()


def capacity_snapshot(*, config=None):
    if config is not None:
        require(config['artifact_dir'] == str(STORE / 'supervised'), 'Wrong BY5 supervisor target')
    return backend().capacity_snapshot(RAM_REQUIRED)


def enroll_locked():
    """Caller owns original_lock; no historical row is removed or changed."""
    h = infra()
    h.no_symlink(REGISTRY)
    rows = json.loads(REGISTRY.read_text())
    require(len({r['path'] for r in rows}) == len(rows), 'Duplicate registry rows')
    old_rows = list(rows)
    for p, b in targets():
        found = [r for r in rows if r['path'] == p]
        require(not found or found == [dict(path=p, bound=b)], 'Registered bound differs')
        if not found:
            rows.append(dict(path=p, bound=b))
    if rows != old_rows:
        temporary = Path(str(REGISTRY) + '.by5_20261030.partial')
        h.no_symlink(temporary)
        with temporary.open('x') as f:
            json.dump(rows, f, indent=2)
            f.flush(); os.fsync(f.fileno())
        os.replace(temporary, REGISTRY)
        h.fsync_directory(REGISTRY.parent)
    return dict(before_rows=len(old_rows), after_rows=len(rows), old_rows_preserved=True,
                own_targets=targets())


def guarded_torch_save(*, payload, target, config, save_kind):
    target = Path(target)
    require(target == Path(STORE / 'supervised/final.pth') and save_kind == 'final', 'Wrong save target')
    require(payload.get('completed_epochs') == 60 and payload.get('epoch') == 59
            and payload.get('successful_updates') == 600
            and payload.get('test_evaluations_before_save') == 0, 'Incomplete supervisor budget')
    require(not any(k in payload for k in ('optimizer', 'scaler', 'rng')), 'Unexpected resume payload')
    h = infra()
    with original_lock():
        capacity = capacity_snapshot(config=config)
        temporary = Path(str(target) + '.partial')
        h.no_symlink(target); h.no_symlink(temporary)
        require(not os.path.lexists(target) and not os.path.lexists(temporary), 'Existing save/partial')
        import torch
        with temporary.open('xb') as f:
            torch.save(payload, h.BoundedWriter(f, GIB))
            f.flush(); os.fsync(f.fileno())
        digest, size = sha(temporary), temporary.stat().st_size
        require(0 < size <= GIB, 'Checkpoint exceeded bound')
        h.publish_no_replace(temporary, target)
        h.fsync_directory(target.parent)
        return dict(path=str(target), sha256=digest, bytes=size, capacity_before=capacity)


def check_load_capacity(*, target, config):
    require(Path(target) == Path(STORE / 'supervised/final.pth') and Path(target).is_file(), 'Wrong reload target')
    with original_lock():
        return capacity_snapshot(config=config)
