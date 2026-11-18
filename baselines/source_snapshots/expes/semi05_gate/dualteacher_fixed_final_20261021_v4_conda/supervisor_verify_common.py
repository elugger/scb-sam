import ast,hashlib,json,math,random,re,subprocess,sys
from pathlib import Path
ROOT=Path('__RECORDED_PROJECT_ROOT__')
PROC=Path('/proc')
PYTHON='__RECORDED_PYTHON__'
PROTOCOL='PRIMARY_3D_SSL_EXPOSURE_V1_20261019'
LOCK_ID=dict(device=2049,inode=168299140,nlink=1)
class VerificationError(ValueError):
    pass

def need(condition, message):
    if not condition:
        raise VerificationError(message)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(chunk)
    return digest.hexdigest()

def digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None

def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result

def read_json(path):
    need(path.is_file() and not path.is_symlink(), 'Missing/symlink JSON: ' + str(path))
    return decode(path.read_text(encoding='utf-8'))

def read_lines(path):
    need(path.is_file() and not path.is_symlink(), 'Missing/symlink ledger: ' + str(path))
    raw = path.read_bytes()
    need(bool(raw) and raw.endswith(b'\n'), 'Empty or incomplete terminal ledger: ' + str(path))
    return [decode(line) for line in raw.decode('utf-8').splitlines()]

def number(value, minimum=0, maximum=None):
    return (type(value) in (int, float) and math.isfinite(value) and value >= minimum
            and (maximum is None or value <= maximum))

def integer(value, expected):
    return type(value) is int and value == expected

def same_path(a, b):
    return Path(a).absolute() == Path(b).absolute()

def bound_data_functions(source_dir, root):
    """Use only the pinned source's metadata/schedule functions, never BoundCases."""
    path = source_dir / 'runtime_data.py'
    selected = ('bound_splits', 'case_file_identity', 'bind_case_files', 'schedule')
    tree = ast.parse(path.read_bytes(), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in selected]
    need({n.name for n in nodes} == set(selected) and len(nodes) == len(selected),
         'Bound data source lacks unambiguous metadata-only functions')
    need(not any(isinstance(n, (ast.Import, ast.ImportFrom)) for node in nodes for n in ast.walk(node)),
         'Selected bound metadata function unexpectedly imports a runtime')
    scope = dict(Path=Path, ROOT=root, HERE=source_dir, json=json, hashlib=hashlib, random=random)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope

def verify_runtime_quiescent(c, pid):
    need(PROC.is_dir(), 'Cannot verify actual process state: /proc unavailable')
    need(not (PROC / str(pid)).exists(), 'Dispatched PID still exists; final acceptance deferred')
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                                       text=True).splitlines()
    uuids = [line.split(',')[1].strip() for line in gpu_rows if int(line.split(',')[0]) == c['gpu']]
    need(len(uuids) == 1, 'Designated GPU identity unavailable/ambiguous')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                         '--format=csv,noheader'], text=True)
    need(not any(line.split(',')[0].strip() == uuids[0] for line in processes.splitlines() if line.strip()),
         'Designated GPU has a compute process')
    return dict(dispatched_pid_absent=True, gpu=c['gpu'], gpu_uuid=uuids[0], designated_gpu_idle=True)


def decode(raw):
    return json.loads(raw, object_pairs_hook=unique_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          VerificationError('Nonfinite JSON constant: ' + value)))
