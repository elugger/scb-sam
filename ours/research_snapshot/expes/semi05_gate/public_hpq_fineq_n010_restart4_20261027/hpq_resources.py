import contextlib
import os
import json
from pathlib import Path
import types
from hpq_contract import ROOT, INIT, STORE, require, sha_file
from hpq_inputs import plain_file, bound_small_bytes
BACKEND = ROOT / 'expes/semi05_gate/public_sup_20261018/public_recovery_guard_20261019.py'
BACKEND_SHA = '6032a10c6768f7bbefde76a0e87146d17ae79953a3bd8e3772625baf9d15a76d'
GIB = 1024 ** 3
FINAL_BOUND = 2 * GIB
RAM_REQUIRED = 12 * GIB + 16 * 1024 ** 2
WEIGHT_FORMAT = 'public05_hpq_fixed_final_six_roots_v1'
ROOT_NAMES = {'student', 'head', 'adapter', 'teacher', 'teacher_head', 'teacher_adapter'}
EVIDENCE_GROWTH_BOUND = 512 * 1024 ** 2
RETIRED = ROOT / 'expes/artifacts/FCD_A_MATRIX_20261006/PUBLIC05_HPQ_FROM_SUP_20261019/fold2/A1B/formal_r1'
STOP = ROOT / 'expes/work_dir/semi05_gate/PUBLIC05_HPQ_FROM_SUP_20261019/control/formal.user_stop_verified_20261019.json'
STOP_SHA = 'b47a658a9d56d1e1163423961f9a66cc722f60ad07db39789370ecc37177722b'

def verify_retired():
    receipt = json.loads(bound_small_bytes(STOP, STOP_SHA))
    require(receipt['status'] == 'USER_STOP_VERIFIED_PROTOCOL_CHANGE' and receipt['pid_absent'] is True and (receipt['artifact_files'] == []) and (receipt['new_test_evaluations'] == 0) and (receipt['ledger']['successful_updates'] == 7), 'Retirement receipt differs')
    no_symlink(RETIRED)
    require(Path(RETIRED).is_dir() and (not list(Path(RETIRED).iterdir())), 'Retired target no longer empty')

def owned_targets():
    from hpq_safety_state import targets as safety_targets
    return [(Path(STORE) / 'best/final.pth', FINAL_BOUND), (Path(STORE) / 'evidence_growth/final.pth', EVIDENCE_GROWTH_BOUND)] + safety_targets()

def backend():
    raw = bound_small_bytes(BACKEND, BACKEND_SHA)
    module = types.ModuleType('_hpq_pinned_joint_resources')
    module.__file__ = str(BACKEND)
    exec(compile(raw, str(BACKEND), 'exec'), module.__dict__)
    require(str(module.HPQ) == str(RETIRED / 'final.pth'), 'Backend retired HPQ target differs')
    verify_retired()
    module.HPQ = Path(ROOT / 'expes/artifacts/FCD_A_MATRIX_20261006/PUBLIC05_HPQ_EPOCHDIAG_20261019/fold2/A1B/formal_r1/final.pth')
    original_targets = module.targets

    def targets():
        verify_retired()
        registry = Path(ROOT / 'expes/work_dir/semi05_gate/MAIN_TABLE_FIXED_FINAL_20261019/active_targets.json')
        root = Path(ROOT / 'expes/artifacts/FCD_A_MATRIX_20261006/MAIN_TABLE_FIXED_FINAL_20261019')
        rows = json.loads(registry.read_text()) if registry.exists() else []
        entries = []
        for row in rows:
            p = Path(row['path'])
            require(root in p.parents and p.name == 'final.pth' and (type(row['bound']) is int) and (0 < row['bound'] <= FINAL_BOUND), 'Invalid shared target registration')
            no_symlink(p)
            entries.append((p, row['bound']))
        require(len({str(p) for p, b in entries}) == len(entries), 'Duplicate shared target')
        for own, bound in owned_targets():
            matches = [b for p, b in entries if p == own]
            require(not matches or matches == [bound], 'Own target bound changed')
            if not matches:
                entries.append((own, bound))
        from reservation_history import preserve_history
        from capacity_retirement import apply_retirement
        registered = {r['path'] for r in rows}
        historical = preserve_history(original_targets() + [(Path(RETIRED) / 'final.pth', 0)] + entries, registered)
        return apply_retirement(historical, registered)
    module.targets = targets
    require(module.MARGIN == GIB + 16 * 1024 ** 2, 'Backend reserve differs')
    return module

def enroll_own_target():
    guard = backend()
    registry = Path(ROOT / 'expes/work_dir/semi05_gate/MAIN_TABLE_FIXED_FINAL_20261019/active_targets.json')
    with guard.original_lock():
        capacity = guard.capacity_snapshot(RAM_REQUIRED)
        no_symlink(registry)
        rows = json.loads(registry.read_text()) if registry.exists() else []
        changed = False
        for own, bound in owned_targets():
            matches = [r['bound'] for r in rows if r['path'] == str(own)]
            require(not matches or matches == [bound], 'Registered bound differs')
            if not matches:
                rows.append(dict(path=str(own), bound=bound))
                changed = True
        if changed:
            partial = Path(str(registry) + '.partial')
            no_symlink(partial)
            with partial.open('x') as stream:
                json.dump(rows, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(partial, registry)
            fsync_directory(registry.parent)
        return capacity

def no_symlink(path):
    require(not any((p.is_symlink() for p in (Path(path), *Path(path).parents))), 'Symlink path rejected')

@contextlib.contextmanager
def initializer_load_guard():
    guard = backend()
    with guard.original_lock():
        before = guard.capacity_snapshot(RAM_REQUIRED)
        plain_file(INIT)
        yield before

def validate_final_payload(payload):
    require(payload.get('format') == WEIGHT_FORMAT, 'Wrong final format')
    require(payload.get('candidate_schema') == 'PUBLIC_N005_FINE_Q_V1' and payload.get('development_selection_used') is True, 'Wrong candidate/disclosure')
    for key, expected in (('epoch', 35), ('completed_epochs', 36), ('successful_updates', 2448), ('microbatches', 12240), ('test_evaluations_before_save', 35), ('fold', 2), ('seed', 2027)):
        require(type(payload.get(key)) is int and payload[key] == expected, 'Final budget mismatch: ' + key)
    require(payload.get('selection_rule') == 'fixed_epoch35_native_student', 'Wrong selection rule')
    require(payload.get('exact_resume_supported') is False, 'Research archive is not exact resume')
    require(payload.get('branch_routing_measured') is False, 'Formal routing was not measured')
    weights = payload.get('weights')
    require(isinstance(weights, dict) and set(weights) == ROOT_NAMES, 'Missing six roots')
    require(all((isinstance(s, dict) and s for s in weights.values())), 'Empty root state')
    require(not any((key in payload for key in ('optimizer', 'scaler', 'rng'))), 'Unexpected resume payload')

class BoundedWriter:

    def __init__(self, stream, bound):
        self.stream, self.bound = (stream, bound)

    def write(self, data):
        require(self.stream.tell() + len(data) <= self.bound, 'Serialized checkpoint exceeds reservation')
        return self.stream.write(data)

    def tell(self):
        return self.stream.tell()

    def flush(self):
        return self.stream.flush()

def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def publish_no_replace(temporary, target):
    before = plain_file(temporary)
    os.link(temporary, target)
    after = Path(target).stat()
    require((after.st_dev, after.st_ino, after.st_nlink) == (before.st_dev, before.st_ino, 2), 'Unexpected atomic publication identity; preserve both paths')
    Path(temporary).unlink()
    plain_file(target)

def save_final(payload):
    if not callable(payload):
        validate_final_payload(payload)
    target = Path(STORE) / 'final.pth'
    temporary = Path(str(target) + '.partial')
    no_symlink(target)
    no_symlink(temporary)
    guard = backend()
    with guard.original_lock():
        capacity = guard.capacity_snapshot(RAM_REQUIRED)
        if callable(payload):
            payload = payload()
        validate_final_payload(payload)
        require(target.parent.is_dir(), 'Prepared artifact directory missing')
        require(not os.path.lexists(target) and (not os.path.lexists(temporary)), 'Existing final/partial; preserve and investigate')
        import torch
        with temporary.open('xb') as stream:
            torch.save(payload, BoundedWriter(stream, FINAL_BOUND))
            stream.flush()
            os.fsync(stream.fileno())
        info = plain_file(temporary)
        require(0 < info.st_size <= FINAL_BOUND, 'Invalid final size')
        digest = sha_file(temporary)
        require(not os.path.lexists(target), 'Final appeared while saving')
        publish_no_replace(temporary, target)
        fsync_directory(target.parent)
        return dict(path=str(target), sha256=digest, bytes=info.st_size, capacity_before=capacity, strict_reload_verified=False, test_evaluations=35)

@contextlib.contextmanager
def final_reload_guard(expected_sha):
    guard = backend()
    with guard.original_lock():
        capacity = guard.capacity_snapshot(RAM_REQUIRED)
        target = Path(STORE) / 'final.pth'
        plain_file(target)
        require(sha_file(target) == expected_sha, 'Final changed before strict reload')
        yield (target, capacity)

def probe_read_capacity(guard, initial, path, identity, owned_target=None):
    current = plain_file(path)
    require((current.st_dev, current.st_ino, current.st_size) == identity, 'Own probe identity changed')
    credit = min(current.st_size, current.st_blocks * 512)
    rows = []
    targets = {str(p): b for p, b in guard.targets()}
    require(set(targets) == {t['path'] for g in initial['filesystems'] for t in g['targets']}, 'Joint target set changed')
    if owned_target is not None:
        owned_target = Path(owned_target)
        require(str(owned_target) in targets, 'Unregistered owned safety target')
        require(Path(path) == Path(str(owned_target) + '.partial'), 'Owned safety partial path mismatch')
        require(0 < current.st_size <= targets[str(owned_target)], 'Owned safety partial exceeds registered bound')
    for group in initial['filesystems']:
        for target in group['targets']:
            p = Path(target['path'])
            no_symlink(p)
            if owned_target is not None and p == owned_target:
                own = plain_file(Path(str(p) + '.partial'))
                require((own.st_dev, own.st_ino, own.st_size) == identity, 'Owned safety partial identity changed')
            else:
                require(not os.path.lexists(str(p) + '.partial'), 'Concurrent permanent partial')
            actual = plain_file(p).st_size if p.exists() else 0
            require(actual == target['actual_bytes'] and targets[str(p)] == target['bound_bytes'], 'Joint target changed during probe')
        fs = os.statvfs(group['probe'])
        allocated = credit if group['filesystem_device'] == current.st_dev else 0
        require(allocated <= group['atomic_peak'], 'Probe exceeds shared atomic reservation')
        required = group['remaining_permanent'] + group['atomic_peak'] - allocated + guard.MARGIN
        free = fs.f_bavail * fs.f_frsize
        require(free >= required, 'Insufficient filesystem capacity during probe reload')
        rows.append(dict(filesystem_device=group['filesystem_device'], free_bytes=free, required_bytes=required, own_atomic_occupied_bytes=allocated))
    memory = guard.memory_snapshot()
    require(memory['effective_available'] >= RAM_REQUIRED, 'Insufficient actual RAM during probe reload')
    return dict(filesystems=rows, memory=memory, owned_target=str(owned_target) if owned_target is not None else None)

@contextlib.contextmanager
def temporary_disk_probe(payload_factory):
    guard = backend()
    path = Path(STORE) / 'update0_probe.pth.partial'
    no_symlink(path)
    with guard.original_lock():
        capacity = guard.capacity_snapshot(RAM_REQUIRED)
        require(path.parent.is_dir() and (not os.path.lexists(path)), 'Disk probe residual/missing directory')
        require(not os.path.lexists(Path(STORE) / 'final.pth'), 'Final already exists before update zero')
        import torch
        with path.open('xb') as stream:
            torch.save(payload_factory(), BoundedWriter(stream, FINAL_BOUND))
            stream.flush()
            os.fsync(stream.fileno())
        fsync_directory(path.parent)
        info = plain_file(path)
        require(0 < info.st_size <= FINAL_BOUND, 'Invalid disk-probe size')
        identity = (info.st_dev, info.st_ino, info.st_size)
        digest = sha_file(path)
        read_capacity = probe_read_capacity(guard, capacity, path, identity)
        receipt = dict(path=str(path), bytes=info.st_size, sha256=digest, capacity_before=capacity, capacity_before_reload=read_capacity)
        yield (path, receipt)
        after = plain_file(path)
        require((after.st_dev, after.st_ino, after.st_size) == identity and sha_file(path) == digest, 'Probe changed after roundtrip; preserve it')
        path.unlink()
        fsync_directory(path.parent)
