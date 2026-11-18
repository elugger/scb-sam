"""Read-only binding to actual fixed-fold split bytes and accepted sup05 receipt."""
import hashlib
import json
import stat
from pathlib import Path

from hpq_contract import ROOT, INIT, INIT_SHA, require, validate_splits, validate_initialization

SPLIT_ROOT = ROOT / 'expes/semi05_gate/public_hpq_fineq_r3_20261020/splits/fold2'
SPLIT_SHA = {
    "train_lab": "1ac1f18841b6d46134c71e75ddcfe83202b94c175a7f1c24a9aef8087e3eb808",
    "train_unlab": "027156909bd08ed658480129eee5e6903faa86e32ee872f69e7c2211eef92a25",
    "train_whole": "9fdf84b9a66f1e73a8c2b791b7d9c81d72a43bab5f42a59613dfce5a0b3be639",
    "test": "f50e39909a88adf23550d165d4538f832d9b7e012c12af957acf3393c11000a6"
}
ACCEPTANCE = ROOT / 'expes/semi05_gate/public_hpq_fineq_r3_20261020/supervised_acceptance.json'
ACCEPTANCE_SHA = '69113b239e3b0cd56b7f763bc262d2fa6018891c7d71191296c9ce393dbb1404'


def plain_file(path):
    path = Path(path)
    require(path.is_absolute(), 'Expected absolute file path')
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'Symlink/ancestor rejected')
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'Expected single-link regular file')
    return info


def bound_small_bytes(path, expected_sha, maximum=1024 * 1024):
    path = Path(path)
    before = plain_file(path)
    require(before.st_size <= maximum, 'Oversized metadata file')
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    after = plain_file(path)
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    require(signature(before) == signature(after), 'Metadata changed during read')
    require(len(raw) <= maximum and hashlib.sha256(raw).hexdigest() == expected_sha, 'Metadata SHA mismatch')
    return raw


def read_bound_splits():
    splits = {role: bound_small_bytes(Path(SPLIT_ROOT) / (role + '.txt'), digest).decode('utf-8').split()
              for role, digest in SPLIT_SHA.items()}
    validate_splits(splits)
    return splits


def read_accepted_n005():
    acceptance = json.loads(bound_small_bytes(ACCEPTANCE, ACCEPTANCE_SHA))
    # This binds acceptance only; the actual weight bytes are verified separately
    # inside the original locked load session before deserialization.
    validate_initialization(INIT, INIT_SHA, acceptance)
    return acceptance
