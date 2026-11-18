"""Bounded DualTeacher two-generation storage, with an injected original shared gate.

No registry mutation, dispatch, pickle import, model import or automatic recovery.
Production must supply the original save.lock guard, full target reservations,
torch serializer and native state/ledger validator. CPU mocks prove IO only.
"""
import hashlib
import json
import os
from pathlib import Path
import stat

from dualteacher_budget import require


def syncdir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def safe_file(path, root):
    path, root = Path(path).absolute(), Path(root).absolute()
    require('..' not in path.parts and '..' not in root.parts, 'Parent traversal in storage path')
    require(root in path.parents, 'Storage path outside bound root')
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'Symlink in storage path')
    if path.exists():
        s = path.stat()
        require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, 'Nonregular or shared storage file')
    return path


def read_pin(path, root, include_raw=False):
    path = safe_file(path, root)
    if not path.exists():
        return dict(bytes=0, sha256=hashlib.sha256(b'').hexdigest(), inode=None, device=None,
                    **({'raw': b''} if include_raw else {}))
    digest, chunks = hashlib.sha256(), []
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        n = 0
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            n += len(chunk)
            digest.update(chunk)
            if include_raw:
                chunks.append(chunk)
        after = os.fstat(stream.fileno())
        current = path.stat()
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_nlink)
        require(identity(before) == identity(after) == identity(current) and n == before.st_size,
                'Storage file changed during same-fd read')
    return dict(bytes=n, sha256=digest.hexdigest(), inode=before.st_ino, device=before.st_dev,
                **({'raw': b''.join(chunks)} if include_raw else {}))


class BoundedSink:
    def __init__(self, stream, bound):
        self.stream, self.bound, self.high_water = stream, bound, 0

    def write(self, value):
        end = self.stream.tell() + len(value)
        require(end <= self.bound, 'Serialization exceeds reserved bound')
        n = self.stream.write(value)
        require(n == len(value), 'Short serialization write')
        self.high_water = max(self.high_water, end)
        return n

    def tell(self):
        return self.stream.tell()

    def seek(self, offset, whence=0):
        position = self.stream.seek(offset, whence)
        require(0 <= position <= self.bound, 'Serialization seek outside bound')
        return position

    def flush(self):
        self.stream.flush()


class RecoveryStore:
    def __init__(self, job, artifact_root, slots, bound, log_bound, identity,
                 guard, validate_state, dump, load, sync_directory=syncdir):
        self.job, self.root = Path(job).absolute(), Path(artifact_root).absolute()
        require(len(slots) == 2 and len(set(map(str, slots))) == 2, 'Exactly two distinct recovery slots required')
        self.slots = [safe_file(p, self.root) for p in slots]
        require(type(bound) is int and bound > 0 and type(log_bound) is int and log_bound > 0,
                'Invalid declared storage bounds')
        require(len(identity) == 64 and all(x in '0123456789abcdef' for x in identity), 'Invalid contract SHA')
        self.bound, self.log_bound, self.identity = bound, log_bound, identity
        self.guard, self.validate_state = guard, validate_state
        self.dump, self.load, self.sync_directory = dump, load, sync_directory

    def _gate(self):
        # The production guard must include EVERY future artifact, both recovery
        # slots, qualification files, all logs and CPAC r5 auxiliary reservations.
        return self.guard.capacity_snapshot()

    def _record_locked(self, path, value):
        path = safe_file(path, self.job)
        raw = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
        require(len(raw) <= 1024**2, 'Evidence record exceeds per-record bound')
        require(not path.exists(), 'Evidence record already exists')
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive hard-link commit never replaces a previous evidence record.
        temporary = safe_file(Path(str(path) + '.partial'), self.job)
        with temporary.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        self.sync_directory(path.parent)

    def record(self, name, value, consistent=False):
        path = safe_file(self.job / name, self.job)
        with self.guard.original_lock():
            self._gate()
            if consistent and path.exists():
                pin = read_pin(path, self.job, True)
                require(json.loads(pin['raw']) == json.loads(json.dumps(value, allow_nan=False)),
                        'Existing evidence has different identity')
                return
            self._record_locked(path, value)

    def ledger_pin(self):
        pin = read_pin(self.job / 'formal' / 'steps.jsonl', self.job, True)
        require(not pin['raw'] or pin['raw'].endswith(b'\n'), 'Partial ledger requires independent review')
        pin.pop('raw')
        return pin

    def append_step(self, value):
        path = safe_file(self.job / 'formal' / 'steps.jsonl', self.job)
        raw = (json.dumps(value, allow_nan=False) + '\n').encode()
        require(len(raw) <= 8192, 'Ledger row exceeds reserved bound')
        with self.guard.original_lock():
            self._gate()
            previous = read_pin(path, self.job, True)
            require(not previous['raw'] or previous['raw'].endswith(b'\n'), 'Partial ledger cannot be appended')
            require(previous['bytes'] + len(raw) <= self.log_bound, 'Ledger exceeds reserved log bound')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('ab') as stream:
                opened = os.fstat(stream.fileno())
                if previous['inode'] is not None:
                    require((opened.st_dev, opened.st_ino, opened.st_size) ==
                            (previous['device'], previous['inode'], previous['bytes']), 'Ledger identity changed')
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            self.sync_directory(path.parent)

    def _commits(self):
        directory = self.job / 'recovery_commits'
        require(not directory.is_symlink(), 'Symlink commit directory')
        result = []
        for path in sorted(directory.glob('commit_*.json')):
            raw = read_pin(path, self.job, True)['raw']
            row = json.loads(raw)
            expected = len(result)
            require(path.name == 'commit_%06d.json' % expected and row['generation'] == expected,
                    'Missing or reordered recovery commit')
            require(row['contract_sha256'] == self.identity and row['execution_role'] == 'formal'
                    and row['path'] == str(self.slots[expected % 2]), 'Unbound recovery commit')
            result.append(row)
        return result

    def checkpoint(self, state):
        with self.guard.original_lock():
            capacity = self._gate()
            self.validate_state(state)
            require(state['contract_sha256'] == self.identity and state['execution_role'] == 'formal',
                    'Only bound formal states may enter formal recovery slots')
            require(state['ledger'] == self.ledger_pin(), 'Checkpoint does not bind current full ledger')
            commits = self._commits()
            generation = len(commits)
            for slot in self.slots:
                earlier = [r for r in commits if r['path'] == str(slot)]
                if earlier:
                    actual = read_pin(slot, self.root)
                    require(all(actual[k] == earlier[-1][k] for k in ('bytes', 'sha256')),
                            'Current recovery slot differs from latest committed identity')
                else:
                    require(not slot.exists(), 'Uncommitted preexisting recovery slot')
            target = self.slots[generation % 2]
            temporary = safe_file(Path(str(target) + '.partial'), self.root)
            require(not temporary.exists(), 'Unreviewed partial checkpoint')
            # No direct rewriting after an interrupted replacement, unknown file
            # or missing committed generation. External review must resolve it.
            previous_slot = [r for r in commits if r['path'] == str(target)]
            if previous_slot:
                actual = read_pin(target, self.root)
                require(all(actual[k] == previous_slot[-1][k] for k in ('bytes', 'sha256')),
                        'Previous slot differs from last commit')
            else:
                require(not target.exists(), 'Uncommitted preexisting recovery slot')
            target.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open('xb') as stream:
                self.dump(state, BoundedSink(stream, self.bound))
                stream.flush()
                os.fsync(stream.fileno())
            pin = read_pin(temporary, self.root)
            os.replace(temporary, target)
            self.sync_directory(target.parent)
            row = dict(path=str(target), bytes=pin['bytes'], sha256=pin['sha256'],
                       contract_sha256=self.identity, execution_role='formal', generation=generation,
                       counts=state['counts'], ledger=state['ledger'])
            self._record_locked(self.job / 'recovery_commits' / ('commit_%06d.json' % generation), row)
            return dict(row, resource_lock_verified=True, capacity_before=capacity)

    def available(self):
        # Hash each current slot ONCE, then compare immutable historical commits.
        # This avoids rehashing the same large slot for every old generation.
        with self.guard.original_lock():
            pins = {str(p): read_pin(p, self.root) for p in self.slots}
            latest = {}
            for row in self._commits():
                latest[row['path']] = row
            return sorted([r for r in latest.values()
                           if all(pins[r['path']][k] == r[k] for k in ('bytes', 'sha256'))],
                          key=lambda r: r['generation'], reverse=True)

    def load_reviewed(self, commit, review):
        """Read only. Review consumption/reconciliation belong to the operator."""
        with self.guard.original_lock():
            self._gate()
            require(review.get('contract_sha256') == self.identity
                    and review.get('checkpoint_sha256') == commit['sha256']
                    and review.get('generation') == commit['generation']
                    and review.get('independent_resume_review_passed') is True
                    and review.get('consumed_by_pid') == os.getpid(), 'Unbound or unconsumed resume review')
            require(commit in self._commits(), 'Unknown recovery commit')
            require(commit['ledger'] == self.ledger_pin(), 'Reviewed ledger not reconciled exactly')
            path = safe_file(commit['path'], self.root)
            # Validate bytes and deserialize from that SAME descriptor; a path
            # replacement cannot redirect deserialization after SHA verification.
            with path.open('rb') as stream:
                s = os.fstat(stream.fileno())
                digest = hashlib.sha256()
                for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
                    digest.update(chunk)
                require(s.st_size == commit['bytes'] and digest.hexdigest() == commit['sha256'], 'Checkpoint SHA differs')
                stream.seek(0)
                state = self.load(stream)
                now = os.fstat(stream.fileno())
                require((s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns) ==
                        (now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns), 'Checkpoint changed during load')
                current = path.stat()
                require((current.st_dev, current.st_ino) == (s.st_dev, s.st_ino), 'Checkpoint path replaced during load')
            self.validate_state(state)
            require(state['execution_role'] == 'formal' and state['contract_sha256'] == self.identity
                    and state['counts'] == commit['counts'] and state['ledger'] == commit['ledger'],
                    'Loaded payload and reviewed commit differ')
            return state
