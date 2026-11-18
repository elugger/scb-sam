import hashlib
import json
import stat
from pathlib import Path
from hpq_contract import ROOT, INIT, INIT_SHA, require, validate_splits, validate_initialization
SPLIT_ROOT = ROOT / 'expes/splits/public/public_semi05_cv5/fold2'
SPLIT_SHA = {'train_lab': '34f4f47574cccb6fb6b09d44e284bb75a2a53b3779a10628dd6e4f331d9fe002', 'train_unlab': '00a4dacf2706c577e1c28a7e52f0b8d18b379c793ae610e37690effa486a3e85', 'train_whole': '9fdf84b9a66f1e73a8c2b791b7d9c81d72a43bab5f42a59613dfce5a0b3be639', 'test': 'f50e39909a88adf23550d165d4538f832d9b7e012c12af957acf3393c11000a6'}
ACCEPTANCE = ROOT / 'expes/work_dir/semi05_gate/PUBLIC_BASIC_SUP_FOLD2_20261018/control/sup05.acceptance.json'
ACCEPTANCE_SHA = 'b5cb2d91d35e30aed473651bc0f428cc1530a0b752c7719769a8de3c405cfb19'

def plain_file(path):
    path = Path(path)
    require(path.is_absolute(), 'Expected absolute file path')
    require(not any((p.is_symlink() for p in (path, *path.parents))), 'Symlink/ancestor rejected')
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
    splits = {role: bound_small_bytes(Path(SPLIT_ROOT) / (role + '.txt'), digest).decode('utf-8').split() for role, digest in SPLIT_SHA.items()}
    validate_splits(splits)
    return splits

def read_accepted_sup05():
    acceptance = json.loads(bound_small_bytes(ACCEPTANCE, ACCEPTANCE_SHA))
    validate_initialization(INIT, INIT_SHA, acceptance)
    return acceptance
