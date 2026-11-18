#!/usr/bin/env python3
"""Two-phase, fail-closed retirement of three explicitly authorized Public roots.

This file does nothing when imported. Inventory creates an exclusive evidence
directory outside the targets. Execute requires that inventory's SHA-256 and
never follows a symlink, deletes a shared lock, or recursively invokes a shell.
Large weights/images are NOT backed up and cannot be recovered by this tool.
"""
import argparse
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import tarfile


ROOTS = (
    '__RECORDED_PROJECT_ROOT__/expes/work_dir/public',
    '__RECORDED_PROJECT_ROOT__/expes/work_dir/semi05_gate/FCD_PUBLIC05_A1B_20261015',
    '__RECORDED_PROJECT_ROOT__/expes/artifacts/FCD_A_MATRIX_20261006/PUBLIC05_A1B_20261015',
)
RETIREMENT_PARENT = Path('__RECORDED_PROJECT_ROOT__/expes/work_dir/semi05_gate')
DEFAULT_RETIREMENT = RETIREMENT_PARENT / 'PUBLIC_RETIREMENT_20261018'
LOCK = Path('__RECORDED_PROJECT_ROOT__/expes/artifacts/FCD_A_MATRIX_20261006/save.lock')
LOCK_ID = (2049, 168299140, 1)
SCHEMA = 'PUBLIC_RETIREMENT_EXACT_V1'
SMALL_FILE_LIMIT = 16 * 1024 ** 2
ARCHIVE_LIMIT = 128 * 1024 ** 2
TEXT_SUFFIXES = {'.txt', '.md', '.json', '.jsonl', '.yaml', '.yml', '.csv', '.tsv', '.log'}
PROTECTED_PARTS = {'save.lock', 'data', 'splits', 'ckpt', 'code', 'BY', 'by'}
NOFOLLOW = os.O_NOFOLLOW
DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | NOFOLLOW


def require(value, message):
    if not value:
        raise RuntimeError(message)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + '\n').encode('utf-8')


def write_exclusive(path, value):
    payload = json_bytes(value)
    with open(path, 'xb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return digest_bytes(payload)


def sha_file(path):
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | NOFOLLOW)
    with os.fdopen(descriptor, 'rb') as handle:
        require(stat.S_ISREG(os.fstat(handle.fileno()).st_mode), 'Not a regular file: ' + str(path))
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def beneath(path, root):
    return path == root or path.startswith(root + '/')


def target_path(path):
    return any(beneath(path, root) for root in ROOTS)


def canonical_ancestors(path):
    path = Path(path)
    require(path.is_absolute() and str(path) == os.path.normpath(str(path)), 'Noncanonical absolute path')
    for ancestor in reversed((path, *path.parents)):
        if os.path.lexists(ancestor):
            require(not stat.S_ISLNK(ancestor.lstat().st_mode), 'Symlink ancestor: ' + str(ancestor))
    require(str(path.resolve(strict=False)) == str(path), 'Resolved path differs: ' + str(path))


def validate_retirement(path):
    path = Path(path)
    require(path.parent == RETIREMENT_PARENT and path.name.startswith('PUBLIC_RETIREMENT_20261018'),
            'Evidence directory must be a dedicated PUBLIC_RETIREMENT_20261018* sibling')
    canonical_ancestors(path)
    require(not target_path(str(path)), 'Evidence overlaps deletion targets')
    return path


def metadata(info):
    mode = info.st_mode
    kind = 'regular' if stat.S_ISREG(mode) else 'symlink' if stat.S_ISLNK(mode) else 'directory' if stat.S_ISDIR(mode) else 'special'
    return dict(kind=kind, size=info.st_size, device=info.st_dev, inode=info.st_ino,
                mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns, mode=mode,
                nlink=info.st_nlink, uid=info.st_uid, gid=info.st_gid,
                allocated_bytes=info.st_blocks * 512)


def inventory_snapshot():
    entries, blockers, missing = [], [], []
    for root in ROOTS:
        canonical_ancestors(root)
        if not os.path.lexists(root):
            missing.append(root)
            continue
        root_stat = os.lstat(root)
        require(stat.S_ISDIR(root_stat.st_mode), 'Target root is not an actual directory: ' + root)
        stack = [Path(root)]
        while stack:
            path = stack.pop()
            info = path.lstat()
            row = dict(path=str(path), root=root, relative=str(path.relative_to(root)), **metadata(info))
            if path != Path(root) and any(part in PROTECTED_PARTS for part in path.relative_to(root).parts):
                blockers.append('Protected path component: ' + str(path))
            if row['kind'] == 'symlink':
                row['link_text'] = os.readlink(path)
                try:
                    row['resolved_link_target'] = str(path.resolve(strict=False))
                    row['target_within_authorized_roots'] = target_path(row['resolved_link_target'])
                except (OSError, RuntimeError) as error:
                    row['resolved_link_target'] = None
                    row['target_within_authorized_roots'] = False
                    row['resolution_error'] = type(error).__name__
                row['deletion_policy'] = 'unlink_link_only_never_follow_target'
            else:
                require(str(path.resolve(strict=True)) == str(path), 'Content escaped through symlink: ' + str(path))
                require(beneath(str(path), root), 'Content escaped target root')
                if info.st_dev != root_stat.st_dev:
                    blockers.append('Nested mount/device: ' + str(path))
                if row['kind'] == 'directory':
                    with os.scandir(path) as scan:
                        children = sorted((Path(item.path) for item in scan), key=str, reverse=True)
                    stack.extend(children)
                elif row['kind'] == 'regular':
                    if info.st_nlink != 1:
                        blockers.append('Hard-linked regular file: ' + str(path))
                    row['archive_eligible'] = path.suffix.lower() in TEXT_SUFFIXES and info.st_size <= SMALL_FILE_LIMIT
                    if row['archive_eligible']:
                        row['sha256'] = sha_file(path)
                        require(metadata(path.lstat()) == metadata(info), 'File changed during inventory hashing: ' + str(path))
                else:
                    blockers.append('Special file: ' + str(path))
            entries.append(row)
    entries.sort(key=lambda row: row['path'])
    return entries, sorted(blockers), missing


def summarize(entries):
    regular = [row for row in entries if row['kind'] == 'regular']
    unique = {(row['device'], row['inode']): row for row in regular}
    return dict(entries=len(entries), regular_files=len(regular),
                symlinks=sum(row['kind'] == 'symlink' for row in entries),
                directories=sum(row['kind'] == 'directory' for row in entries),
                regular_logical_bytes=sum(row['size'] for row in regular),
                unique_regular_allocated_bytes=sum(row['allocated_bytes'] for row in unique.values()),
                archive_eligible_files=sum(row.get('archive_eligible', False) for row in regular),
                archive_eligible_logical_bytes=sum(row['size'] for row in regular if row.get('archive_eligible')),
                note='Allocated-byte totals are not guaranteed free-space gain; hard links block execution.')


def validate_lock(fd=None):
    canonical_ancestors(LOCK)
    info = LOCK.lstat()
    require(stat.S_ISREG(info.st_mode) and (info.st_dev, info.st_ino, info.st_nlink) == LOCK_ID,
            'Original shared save.lock identity mismatch')
    if fd is not None:
        opened = os.fstat(fd)
        require(stat.S_ISREG(opened.st_mode) and (opened.st_dev, opened.st_ino, opened.st_nlink) == LOCK_ID,
                'Opened original shared save.lock identity mismatch')


@contextlib.contextmanager
def original_lock():
    validate_lock()
    descriptor = os.open(LOCK, os.O_RDWR | NOFOLLOW)
    try:
        validate_lock(descriptor)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate_lock(descriptor)
        yield descriptor
        validate_lock(descriptor)
    finally:
        os.close(descriptor)


def process_references(entries):
    """Fail on references or uninspectable human-user processes; never kill any."""
    identities = {(row['device'], row['inode']) for row in entries}
    references, inaccessible, system_skipped = [], [], []
    own_pid = os.getpid()
    for process in sorted(Path('/proc').iterdir(), key=lambda item: item.name):
        if not process.name.isdigit() or int(process.name) == own_pid:
            continue
        pid = int(process.name)
        uid = None
        try:
            uid = process.stat().st_uid
            command = (process / 'cmdline').read_bytes()
            for root in ROOTS:
                if root.encode() in command or root.removeprefix('__RECORDED_PROJECT_ROOT__/').encode() in command:
                    references.append(dict(pid=pid, uid=uid, source='cmdline', target=root))
            for name in ('cwd', 'root', 'exe'):
                link = process / name
                try:
                    value = os.readlink(link).removesuffix(' (deleted)')
                    if target_path(value):
                        references.append(dict(pid=pid, uid=uid, source=name, target=value))
                except FileNotFoundError:
                    pass  # Kernel threads/zombies have no exe/cwd; fd access is still checked.
            for descriptor in (process / 'fd').iterdir():
                try:
                    value = os.readlink(descriptor).removesuffix(' (deleted)')
                    opened = descriptor.stat()
                    if target_path(value) or (opened.st_dev, opened.st_ino) in identities:
                        references.append(dict(pid=pid, uid=uid, source='open_fd', target=value))
                except FileNotFoundError:
                    continue
        except FileNotFoundError:
            continue  # Process exited while scanning.
        except PermissionError as error:
            row = dict(pid=pid, uid=uid, error=str(error))
            if uid is None or uid >= 1000 or uid == os.getuid():
                inaccessible.append(row)
            else:
                system_skipped.append(row)
        except OSError as error:
            inaccessible.append(dict(pid=pid, error=str(error)))
    result = dict(checked_at=utc_now(), references=references, uninspectable_user_processes=inaccessible,
                  uninspectable_system_processes=system_skipped,
                  qualification='All readable processes checked; unreadable non-system user processes block deletion.')
    require(not references, 'Live process references authorized targets: ' + json.dumps(references))
    require(not inaccessible, 'Uninspectable user processes: ' + json.dumps(inaccessible))
    return result


def archive_small(evidence, entries):
    members = [row for row in entries if row.get('archive_eligible')]
    # PAX headers can add several blocks for long paths/mtime; budget conservatively.
    upper_bound = 10240 + sum(((row['size'] + 511) // 512) * 512 + 8192 for row in members)
    require(upper_bound <= ARCHIVE_LIMIT, 'All eligible small evidence exceeds 128 MiB cap; no deletion is authorized by this inventory.')
    destination = evidence / 'small_evidence.tar'
    with open(destination, 'xb') as output:
        with tarfile.open(fileobj=output, mode='w', format=tarfile.PAX_FORMAT) as archive:
            for row in members:
                path = Path(row['path'])
                require(metadata(path.lstat()) == {key: row[key] for key in metadata(path.lstat())},
                        'Archive source changed: ' + str(path))
                descriptor = os.open(path, os.O_RDONLY | NOFOLLOW)
                with os.fdopen(descriptor, 'rb') as source:
                    require(metadata(os.fstat(source.fileno())) == {key: row[key] for key in metadata(os.fstat(source.fileno()))},
                            'Archive descriptor changed: ' + str(path))
                    digest = hashlib.sha256()
                    for chunk in iter(lambda: source.read(1024 * 1024), b''):
                        digest.update(chunk)
                    require(digest.hexdigest() == row['sha256'], 'Archive source SHA mismatch: ' + str(path))
                    source.seek(0)
                    member = tarfile.TarInfo(path.relative_to('__RECORDED_PROJECT_ROOT__').as_posix())
                    member.size, member.mode, member.mtime = row['size'], stat.S_IMODE(row['mode']), row['mtime_ns'] // 10 ** 9
                    archive.addfile(member, source)
                    require(metadata(os.fstat(source.fileno())) == {key: row[key] for key in metadata(os.fstat(source.fileno()))},
                            'Archive source changed while copying: ' + str(path))
        output.flush()
        os.fsync(output.fileno())
    require(destination.stat().st_size <= ARCHIVE_LIMIT, 'Archive exceeded hard cap')
    # Verify each saved payload, not only the tar container hash.
    with tarfile.open(destination, 'r') as archive:
        actual = archive.getmembers()
        require(len(actual) == len(members), 'Archive member count mismatch')
        for saved, row in zip(actual, members):
            require(saved.isfile() and saved.size == row['size'], 'Unexpected archive entry')
            source = archive.extractfile(saved)
            digest = hashlib.sha256()
            with source:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(chunk)
            require(digest.hexdigest() == row['sha256'], 'Archived payload SHA mismatch')
    return dict(path=str(destination), bytes=destination.stat().st_size, sha256=sha_file(destination),
                members=len(members), per_file_limit=SMALL_FILE_LIMIT, total_limit=ARCHIVE_LIMIT,
                exclusion='No checkpoints, images, extensionless or oversized files; symlinks are inventoried, not dereferenced.')


def run_inventory(evidence):
    require(not os.path.lexists(evidence), 'Evidence directory already exists; refusing overwrite')
    entries, blockers, missing = inventory_snapshot()
    require(entries, 'No target contents exist')
    validate_lock()
    evidence.mkdir(mode=0o700, parents=False, exist_ok=False)
    archive = archive_small(evidence, entries)
    payload = dict(schema=SCHEMA, script_sha256=sha_file(Path(__file__).absolute()),
                   created_at=utc_now(), roots=list(ROOTS), missing_roots=missing,
                   lock=dict(path=str(LOCK), device=LOCK_ID[0], inode=LOCK_ID[1], nlink=LOCK_ID[2]),
                   entries=entries, blockers=blockers, summary=summarize(entries), archive=archive,
                   notice='Execute is irreversible for weights/images/other unarchived files. No source paths were modified.')
    receipt_sha = write_exclusive(evidence / 'inventory.json', payload)
    write_exclusive(evidence / 'inventory_summary.json', dict(inventory_sha256=receipt_sha,
                    summary=payload['summary'], blockers=blockers, archive=archive, missing_roots=missing))
    print(json.dumps(dict(status='INVENTORIED_NOT_DELETED', inventory=str(evidence / 'inventory.json'),
                          inventory_sha256=receipt_sha, summary=payload['summary'], blockers=blockers, archive=archive)))


@contextlib.contextmanager
def anchored_parent(path, rows):
    """Walk from / using O_NOFOLLOW descriptors; validate every inventoried directory."""
    require(target_path(path), 'Deletion target outside exact roots')
    components = Path(path).parts[1:]
    descriptor = os.open('/', DIR_FLAGS)
    current = ''
    try:
        for component in components[:-1]:
            following = os.open(component, DIR_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = following
            current += '/' + component
            if current in rows:
                info, row = os.fstat(descriptor), rows[current]
                require(stat.S_ISDIR(info.st_mode) and (info.st_dev, info.st_ino) == (row['device'], row['inode']),
                        'Anchored directory identity changed: ' + current)
        yield descriptor, components[-1]
    finally:
        os.close(descriptor)


def unlink_exact(row, rows):
    with anchored_parent(row['path'], rows) as (descriptor, name):
        info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        current = metadata(info)
        if row['kind'] == 'directory':
            require(current['kind'] == 'directory' and (info.st_dev, info.st_ino) == (row['device'], row['inode']),
                    'Directory changed before rmdir: ' + row['path'])
            os.rmdir(name, dir_fd=descriptor)  # Nonempty/new contents stop deletion.
        else:
            require(current == {key: row[key] for key in current}, 'File identity changed before unlink: ' + row['path'])
            if row['kind'] == 'symlink':
                require(os.readlink(name, dir_fd=descriptor) == row['link_text'], 'Symlink changed before unlink')
            else:
                require(row['kind'] == 'regular' and info.st_nlink == 1, 'Unlink of special/shared file refused')
            os.unlink(name, dir_fd=descriptor)


def run_execute(evidence, approved_sha):
    require(approved_sha and len(approved_sha) == 64 and all(char in '0123456789abcdef' for char in approved_sha),
            'Execute requires an explicit lowercase reviewed --inventory-sha256')
    require(not (evidence / 'execution_started.json').exists(), 'Execution already attempted; inspect its receipt/journal, never blindly rerun')
    path = evidence / 'inventory.json'
    require(sha_file(path) == approved_sha, 'Reviewed inventory SHA mismatch')
    payload = json.loads(path.read_text('utf-8'))
    require(payload['schema'] == SCHEMA and payload['roots'] == list(ROOTS), 'Inventory scope/schema mismatch')
    require(payload['script_sha256'] == sha_file(Path(__file__).absolute()), 'Script changed since reviewed inventory')
    require(not payload['blockers'], 'Inventory contains blockers')
    require(payload['lock'] == dict(path=str(LOCK), device=LOCK_ID[0], inode=LOCK_ID[1], nlink=LOCK_ID[2]), 'Inventory lock mismatch')
    archive = payload['archive']
    require(archive['path'] == str(evidence / 'small_evidence.tar') and sha_file(archive['path']) == archive['sha256'], 'Small archive identity mismatch')
    require(os.lstat(archive['path']).st_size == archive['bytes'] <= ARCHIVE_LIMIT, 'Small archive size mismatch')
    rows = {row['path']: row for row in payload['entries']}
    require(len(rows) == len(payload['entries']), 'Duplicate inventory paths')
    deleted, result = [], None
    with original_lock():
        entries, blockers, missing = inventory_snapshot()
        require(entries == payload['entries'] and blockers == payload['blockers'] and missing == payload['missing_roots'],
                'Target contents/metadata/SHA changed after approved inventory; no deletion performed')
        process_check = process_references(entries)
        before = {root: os.statvfs(root).f_bavail * os.statvfs(root).f_frsize for root in ROOTS if os.path.exists(root)}
        write_exclusive(evidence / 'execution_started.json', dict(started_at=utc_now(), inventory_sha256=approved_sha,
                        process_check=process_check, free_bytes_by_target_fs_snapshot=before))
        try:
            ordered = sorted(entries, key=lambda row: (row['kind'] == 'directory', -len(Path(row['path']).parts), row['path']))
            with open(evidence / 'deletion_journal.jsonl', 'xb', buffering=0) as journal:
                for index, row in enumerate(ordered):
                    if index and index % 128 == 0:
                        process_references([entry for entry in entries if entry['path'] not in set(deleted)])
                    require(Path(row['path']).name != 'save.lock', 'A shared lock must never be removed')
                    unlink_exact(row, rows)
                    deleted.append(row['path'])
                    journal.write(json.dumps(dict(deleted_at=utc_now(), path=row['path'], kind=row['kind'],
                                              size=row['size'], device=row['device'], inode=row['inode']), sort_keys=True).encode() + b'\n')
                    os.fsync(journal.fileno())
            require(all(not os.path.lexists(root) for root in ROOTS), 'An authorized old target still exists')
            result = dict(status='COMPLETE', completed_at=utc_now())
        except BaseException as error:
            result = dict(status='PARTIAL_STOPPED_NO_AUTOMATIC_RETRY', stopped_at=utc_now(),
                          error_type=type(error).__name__, error=str(error))
            raise
        finally:
            if result is not None:
                deleted_rows = [rows[name] for name in deleted]
                free_after = {root: os.statvfs(Path(root).parent).f_bavail * os.statvfs(Path(root).parent).f_frsize for root in ROOTS}
                result.update(inventory_sha256=approved_sha, script_sha256=payload['script_sha256'],
                              roots=list(ROOTS), deleted_paths=deleted,
                              deleted_summary=summarize(deleted_rows), archive=archive,
                              free_bytes_by_target_fs_before=before, free_bytes_by_target_fs_after=free_after,
                              recoverability='Only listed small archive members can be recovered; deleted large checkpoints/images and other excluded files have NO backup.',
                              shared_save_lock_untouched=dict(path=str(LOCK), device=LOCK_ID[0], inode=LOCK_ID[1], nlink=LOCK_ID[2]))
                write_exclusive(evidence / 'execution_result.json', result)
    print(json.dumps(dict(status=result['status'], receipt=str(evidence / 'execution_result.json'),
                          deleted_summary=result['deleted_summary'], recoverability=result['recoverability'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--inventory', action='store_true')
    actions.add_argument('--execute', action='store_true')
    parser.add_argument('--retirement-dir', default=str(DEFAULT_RETIREMENT))
    parser.add_argument('--inventory-sha256')
    args = parser.parse_args()
    evidence = validate_retirement(args.retirement_dir)
    if args.inventory:
        require(args.inventory_sha256 is None, 'Inventory must not receive an old approval SHA')
        run_inventory(evidence)
    else:
        run_execute(evidence, args.inventory_sha256)


if __name__ == '__main__':
    main()
