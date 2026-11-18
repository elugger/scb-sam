import os
import sys
from pathlib import Path
ASNET_ROOT = Path(os.environ.get('ASNET_ROOT', '.')).resolve()
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT = Path(str(ASNET_ROOT))
FINAL_BOUND = 32 * 1024 ** 2
RAM_REQUIRED = 7 * 1024 ** 3 + 16 * 1024 ** 2
REGISTRY = ROOT / 'expes/work_dir/semi05_gate/MAIN_TABLE_FIXED_FINAL_20261019/active_targets.json'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8 * 1024 ** 2), b''):
            h.update(data)
    return h.hexdigest()

def resource_backend(target, new_bound=FINAL_BOUND):
    target = Path(target)
    base = ROOT / 'expes/artifacts/FCD_A_MATRIX_20261006/MAIN_TABLE_FIXED_FINAL_20261019'
    if base not in target.parents or target.name != 'final.pth':
        raise ValueError('Noncanonical own checkpoint')
    sys.path.insert(0, str(ROOT / 'expes/semi05_gate/public_hpq_epochdiag_20261019'))
    from hpq_resources import backend
    guard = backend()
    original = guard.targets

    def registered():
        rows = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else []
        entries = []
        for row in rows:
            p = Path(row['path'])
            if base not in p.parents or p.name != 'final.pth' or (not 0 < row['bound'] <= 2 * 1024 ** 3):
                raise ValueError('Invalid shared main-table target registry')
            entries.append((p, row['bound']))
        if len({str(p) for p, _ in entries}) != len(entries):
            raise ValueError('Duplicate main-table target')
        if target not in [p for p, _ in entries]:
            entries.append((target, new_bound))
        return entries
    guard.targets = lambda: original() + registered()
    return guard

def enroll_target(target, bound=FINAL_BOUND):
    if type(bound) is not int or not 0 < bound <= 2 * 1024 ** 3:
        raise ValueError('Invalid measured checkpoint reservation')
    guard = resource_backend(target, bound)
    with guard.original_lock():
        capacity = guard.capacity_snapshot(RAM_REQUIRED)
        rows = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else []
        if str(target) not in [r['path'] for r in rows]:
            rows.append(dict(path=str(target), bound=bound))
            REGISTRY.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(str(REGISTRY) + '.partial')
            with temporary.open('x') as stream:
                json.dump(rows, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, REGISTRY)
        return capacity

class BoundedWriter:

    def __init__(self, stream, bound):
        self.stream, self.bound = (stream, bound)

    def write(self, value):
        if self.stream.tell() + len(value) > self.bound:
            raise ValueError('Own checkpoint exceeds reservation')
        return self.stream.write(value)

    def tell(self):
        return self.stream.tell()

    def flush(self):
        return self.stream.flush()

def save_terminal(target, payload):
    import torch
    target, partial = (Path(target), Path(str(target) + '.partial'))
    guard = resource_backend(target)
    with guard.original_lock():
        capacity = guard.capacity_snapshot(RAM_REQUIRED)
        entries = json.loads(REGISTRY.read_text())
        matches = [row['bound'] for row in entries if row['path'] == str(target)]
        if len(matches) != 1:
            raise ValueError('Exact checkpoint target not enrolled')
        own_bound = matches[0]
        if any((p.is_symlink() for p in (target, partial, *target.parents))):
            raise ValueError('Symlink checkpoint path')
        if target.exists() or partial.exists():
            raise ValueError('Existing final/partial, preserve and investigate')
        target.parent.mkdir(parents=True, exist_ok=True)
        with partial.open('xb') as stream:
            torch.save(payload, BoundedWriter(stream, own_bound))
            stream.flush()
            os.fsync(stream.fileno())
        digest, size = (sha(partial), partial.stat().st_size)
        if not 0 < size <= own_bound:
            raise ValueError('Invalid own terminal size')
        os.link(partial, target)
        if partial.stat().st_ino != target.stat().st_ino or target.stat().st_nlink != 2:
            raise ValueError('Atomic publication identity differs')
        partial.unlink()
        fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return dict(path=str(target), sha256=digest, bytes=size, resource_lock_verified=True, capacity_before=capacity)
