"""Read-only, exact-target capacity planning for a NEW matrix run pair.

No glob-based discounts, directory creation, deletion or checkpoint loading.
Existing old tags never reduce a new pair's reservation. The caller must hold
the common filesystem save/launch lock while checking and committing a launch.
This is infrastructure preparation, not model qualification.
"""
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import stat


GIB = 1024 ** 3
MIB = 1024 ** 2


@dataclass(frozen=True)
class Target:
    path: str
    maximum_bytes: int


def capacity_bytes(maximum_bytes, present_bytes, *, safety_bytes=GIB,
                   evidence_bytes=16*MIB):
    """Pure byte accounting; present_bytes=None means a not-yet-saved target.

    Existing complete atomic destinations consume current disk already. If
    their next version might be larger, reserve the positive growth too. One
    full largest atomic replacement is reserved even when all targets exist.
    Partial files are intentionally NOT targets: their usage is reflected in
    measured free space, so they never earn a reservation discount.
    """
    sizes, present = list(maximum_bytes), list(present_bytes)
    if not sizes or len(sizes) != len(present):
        raise ValueError('Nonempty, aligned exact targets required')
    if any(type(n) is not int or n <= 0 for n in sizes):
        raise ValueError('Positive integer checkpoint size bounds required')
    if type(safety_bytes) is not int or safety_bytes < GIB:
        raise ValueError('At least one GiB of safety is required')
    if type(evidence_bytes) is not int or evidence_bytes < 0:
        raise ValueError('Evidence reservation must be nonnegative integer')
    for n, old in zip(sizes, present):
        if old is not None and (type(old) is not int or old <= 0 or old > n):
            raise ValueError('Existing checkpoint is empty or exceeds measured bound')
    permanent = sum(n if old is None else n-old for n, old in zip(sizes, present))
    atomic = max(sizes)
    return dict(remaining_permanent_bytes=permanent, atomic_bytes=atomic,
                safety_bytes=safety_bytes, evidence_bytes=evidence_bytes,
                required_bytes=permanent+atomic+safety_bytes+evidence_bytes)


def inspect_capacity(root, targets, *, disk_usage=shutil.disk_usage,
                     safety_bytes=GIB, evidence_bytes=16*MIB):
    """Inspect a declared pair in one existing, canonical, non-link root.

    Sizes are bounds from an actual same-layout checkpoint, not a way to omit
    optimizer state. The returned report is JSON serializable. Passing this
    check neither validates a source/qualification nor authorizes a launch.
    """
    raw_root = Path(root)
    if not raw_root.is_absolute() or not raw_root.is_dir():
        raise ValueError('Existing absolute storage root required')
    canonical = raw_root.resolve(strict=True)
    if canonical != raw_root or raw_root.is_symlink():
        raise ValueError('Canonical storage root required; no root symlink')
    device = canonical.stat().st_dev
    declarations, seen = list(targets), set()
    sizes, present, rows = [], [], []
    for declaration in declarations:
        path = Path(declaration.path)
        if not path.is_absolute() or '..' in path.parts:
            raise ValueError('Explicit absolute target paths without traversal required')
        try:
            relative = path.relative_to(canonical)
        except ValueError as exc:
            raise ValueError('Target escapes declared storage root') from exc
        if len(relative.parts) < 2 or path.name not in ('best.pth','final.pth','qualification.pth'):
            raise ValueError('Target must be a named checkpoint within a run directory')
        if str(path) in seen:
            raise ValueError('Duplicate target')
        seen.add(str(path))
        cursor = canonical
        for part in relative.parts:
            cursor = cursor/part
            if os.path.lexists(cursor):
                info = cursor.lstat()
                if stat.S_ISLNK(info.st_mode) or info.st_dev != device:
                    raise ValueError('Symlink or cross-filesystem target ancestor')
        old = None
        if os.path.lexists(path):
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError('Checkpoint target must be a regular unlinked file')
            old = info.st_size
        sizes.append(declaration.maximum_bytes)
        present.append(old)
        rows.append(dict(path=str(path), maximum_bytes=declaration.maximum_bytes,
                         already_on_disk_bytes=old))
    budget = capacity_bytes(sizes, present, safety_bytes=safety_bytes,
                            evidence_bytes=evidence_bytes)
    free = disk_usage(canonical).free
    return dict(storage_root=str(canonical), device=device, targets=rows,
                free_bytes=free, **budget, passes=free >= budget['required_bytes'],
                margin_bytes=free-budget['required_bytes'],
                requirement='Recheck under common filesystem lock immediately before launch/save; '
                            'one concurrent atomic writer on this filesystem')
