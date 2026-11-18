"""Capacity and atomic-save gate for three fixed Public basic-supervised arms."""
import contextlib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat

ROOT = Path('__RECORDED_PROJECT_ROOT__')
STORE = ROOT/'expes/artifacts/FCD_A_MATRIX_20261006/PUBLIC_BASIC_SUP_FOLD2_20261018'
LOCK = ROOT/'expes/artifacts/FCD_A_MATRIX_20261006/save.lock'
GIB = 1024**3
MARGIN = GIB + 16*1024**2
BOUNDS = {'latest.pth': 3*GIB, 'final.pth': GIB}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def no_symlink(path):
    p = Path(path)
    for item in (p, *p.parents):
        require(not item.is_symlink(), 'Symlink disallowed: '+str(item))


def targets():
    return [(STORE/arm/name, bound) for arm in ('sup05', 'sup10', 'sup100')
            for name, bound in BOUNDS.items()]


def check_lock(fd=None):
    no_symlink(LOCK)
    s = LOCK.stat()
    require(stat.S_ISREG(s.st_mode) and (s.st_dev, s.st_ino, s.st_nlink) == (2049, 168299140, 1),
            'Original save.lock identity changed')
    if fd is not None:
        f = os.fstat(fd)
        require((f.st_dev, f.st_ino, f.st_nlink) == (s.st_dev, s.st_ino, 1), 'Lock fd changed')


@contextlib.contextmanager
def original_lock():
    import fcntl
    check_lock()
    fd = os.open(LOCK, os.O_RDWR | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        check_lock(fd)
        yield fd
        check_lock(fd)
    finally:
        os.close(fd)


def memory_snapshot(proc=Path('/proc'), cgroups=Path('/sys/fs/cgroup')):
    info = dict(line.split(':', 1) for line in (proc/'meminfo').read_text().splitlines())
    available = int(info['MemAvailable'].split()[0])*1024
    require(available >= 0, 'Invalid host memory availability')
    limits, skipped, memberships, mounts = [], [], [], []
    mountinfo = proc/'self/mountinfo'

    def unescape_mount(value):
        # mountinfo escapes whitespace and backslash using octal sequences.
        for encoded, literal in ((r'\040', ' '), (r'\011', '\t'), (r'\012', '\n'), (r'\134', '\\')):
            value = value.replace(encoded, literal)
        return value

    if mountinfo.exists():
        for line in mountinfo.read_text().splitlines():
            before, separator, after = line.partition(' - ')
            require(separator, 'Malformed mountinfo')
            left, right = before.split(), after.split()
            require(len(left) >= 6 and len(right) >= 3, 'Malformed mountinfo fields')
            if right[0] not in ('cgroup', 'cgroup2'):
                continue
            mount_root = unescape_mount(left[3])
            mountpoint = Path(unescape_mount(left[4]))
            require(mount_root.startswith('/') and '..' not in PurePosixPath(mount_root).parts,
                    'Invalid cgroup mount root')
            require(mountpoint.is_absolute(), 'Invalid cgroup mountpoint')
            mounts.append({'type': right[0], 'root': mount_root.rstrip('/') or '/',
                           'mountpoint': mountpoint, 'options': set(right[2].split(',')) | set(left[5].split(','))})
    for line in (proc/'self/cgroup').read_text().splitlines():
        _, controllers, membership = line.split(':', 2)
        if controllers != '' and 'memory' not in controllers.split(','):
            continue
        require(membership.startswith('/') and '..' not in PurePosixPath(membership).parts,
                'Invalid cgroup membership')
        memberships.append((controllers, membership))
    # Validate the actual v1 memory hierarchy first, so a hybrid v2 hierarchy
    # can be declared non-memory only with positive evidence of a v1 owner.
    memberships.sort(key=lambda entry: entry[0] == '')
    v1_memory_checked = False
    for controllers, membership in memberships:
        is_v2 = controllers == ''
        if mountinfo.exists():
            candidates = [mount for mount in mounts if
                          (mount['type'] == 'cgroup2' if is_v2 else
                           mount['type'] == 'cgroup' and 'memory' in mount['options'])]
            candidates = [mount for mount in candidates if mount['root'] == '/' or
                          membership == mount['root'] or membership.startswith(mount['root'] + '/')]
            require(candidates, 'No actual cgroup mount maps memory membership: '+membership)
            # Prefer the broadest visible hierarchy, retaining every accessible
            # ancestor limit rather than a deeper bind mount that hides them.
            broadest = min(len(mount['root']) for mount in candidates)
            candidates = [mount for mount in candidates if len(mount['root']) == broadest]
            require(len(candidates) == 1, 'Ambiguous actual cgroup memory mount mapping')
            selected = candidates[0]
            base, mount_root = selected['mountpoint'], selected['root']
            relative = membership.lstrip('/') if mount_root == '/' else membership[len(mount_root):].lstrip('/')
        else:
            # Explicit legacy mapping for hosts/CPU fixtures without mountinfo.
            # Hybrid hosts may mount cgroup2 under the unified subdirectory.
            base = (cgroups/'unified' if is_v2 and (cgroups/'unified/cgroup.controllers').exists()
                    else cgroups if is_v2 else cgroups/'memory')
            mount_root, relative = '/', membership.lstrip('/')
        current = base/relative
        require(current == base or base in current.parents, 'Invalid cgroup membership')
        limit_name, usage_name = ('memory.max', 'memory.current') if is_v2 else ('memory.limit_in_bytes', 'memory.usage_in_bytes')
        if is_v2:
            controller_file = base/'cgroup.controllers'
            if controller_file.exists():
                enabled_controllers = controller_file.read_text().split()
                if ('memory' not in enabled_controllers and not (base/limit_name).exists()
                        and not (current/limit_name).exists()):
                    require(v1_memory_checked, 'V2 has no memory controller and no verified v1 memory hierarchy')
                    skipped.append({'path': str(current), 'mountpoint': str(base), 'mount_root': mount_root,
                                    'controllers': enabled_controllers,
                                    'reason': 'v2_root_explicitly_has_no_memory_controller_verified_v1_memory_used'})
                    continue
        memory_files_checked = 0
        while True:
            limit_file, usage_file = current/limit_name, current/usage_name
            if limit_file.exists():
                memory_files_checked += 1
                value = limit_file.read_text().strip()
                usage = int(usage_file.read_text().strip())
                require(usage >= 0, 'Invalid cgroup memory values')
                if value != 'max':
                    limit = int(value)
                    require(limit > 0 and usage >= 0, 'Invalid cgroup memory values')
                    remaining = max(0, limit-usage)
                    available = min(available, remaining)
                    limits.append({'path': str(current), 'limit': limit, 'usage': usage,
                                   'remaining': remaining})
                else:
                    limits.append({'path': str(current), 'limit': 'max', 'usage': usage})
            else:
                require(current == base, 'Missing memory controller at '+str(current))
            if current == base:
                break
            current = current.parent
        if not is_v2:
            require(memory_files_checked > 0, 'V1 memory controller has no readable memory limits')
            v1_memory_checked = True
    return {'host_mem_available': int(info['MemAvailable'].split()[0])*1024,
            'effective_available': available, 'cgroup_ancestors': limits,
            'non_memory_hierarchies': skipped, 'mountinfo_used': mountinfo.exists()}


def existing_parent(path):
    p = Path(path)
    while not p.exists():
        p = p.parent
    return p


def capacity_snapshot(required_ram=7*GIB+16*1024**2):
    groups = {}
    for path, bound in targets():
        no_symlink(path)
        require(not os.path.lexists(str(path)+'.partial'), 'Existing atomic partial: '+str(path))
        parent = existing_parent(path.parent)
        dev = parent.stat().st_dev
        actual = path.stat().st_size if path.exists() else 0
        if path.exists():
            require(path.is_file() and path.stat().st_nlink == 1, 'Invalid checkpoint target')
        require(actual <= bound, 'Checkpoint bound exceeded')
        group = groups.setdefault(dev, {'filesystem_device': dev, 'probe': str(parent),
                    'remaining_permanent': 0, 'atomic_peak': 0, 'targets': []})
        group['remaining_permanent'] += max(0, bound-actual)
        group['atomic_peak'] = max(group['atomic_peak'], bound)
        group['targets'].append({'path': str(path), 'actual_bytes': actual, 'bound_bytes': bound})
    for group in groups.values():
        fs = os.statvfs(group['probe'])
        group['free_bytes'] = fs.f_bavail*fs.f_frsize
        group['required_bytes'] = group['remaining_permanent'] + group['atomic_peak'] + MARGIN
        group['margin_bytes'] = group['free_bytes']-group['required_bytes']
        require(group['margin_bytes'] >= 0, 'Insufficient filesystem capacity: '+json.dumps(group))
    memory = memory_snapshot()
    memory['required_bytes'] = required_ram
    require(memory['effective_available'] >= required_ram, 'Insufficient RAM including cgroup ancestors')
    return {'filesystems': list(groups.values()), 'memory': memory,
            'save_lock': {'path': str(LOCK), 'device': 2049, 'inode': 168299140, 'nlink': 1}}


def validate_target(target, config):
    target = Path(target)
    expected_arm = {5: 'sup05', 10: 'sup10', 100: 'sup100'}[config['label_percent']]
    require(target.parent == STORE/expected_arm and target.name in BOUNDS, 'Noncanonical save target')
    require(Path(config['artifact_dir']) == target.parent, 'Configuration artifact path mismatch')
    no_symlink(target)
    return target


def guarded_torch_save(*, payload, target, config, save_kind):
    import torch
    target = validate_target(target, config)
    require(save_kind in ('latest', 'final') and target.name == save_kind+'.pth', 'Save kind mismatch')
    bound = BOUNDS[target.name]
    with original_lock():
        before = capacity_snapshot(2*bound+MARGIN)
        require(target.parent.is_dir(), 'Artifact directory absent')
        require(save_kind == 'latest' or not os.path.lexists(target), 'Final checkpoint already exists')
        temp = Path(str(target)+'.partial')
        with temp.open('xb') as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        require(temp.stat().st_size <= bound, 'Serialized checkpoint exceeded bound; partial preserved')
        digest, size = sha(temp), temp.stat().st_size
        os.replace(temp, target)
        fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return {'path': str(target), 'sha256': digest, 'bytes': size, 'capacity_before': before}


def check_load_capacity(*, target, config):
    target = validate_target(target, config)
    require(target.name == 'final.pth' and target.is_file(), 'Expected saved final checkpoint')
    with original_lock():
        return capacity_snapshot(2*target.stat().st_size+MARGIN)
