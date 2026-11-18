import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.parse import urlsplit
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]

def verify_file(path, row):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != row['original_bytes']:
        raise ValueError('CHECKPOINT_SIZE_MISMATCH')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != row['expected_original_sha256']:
        raise ValueError('CHECKPOINT_SHA256_MISMATCH')
    return {'status': 'VERIFIED', 'name': row['name'], 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}

def download(row, directory, opener=urlopen):
    directory = Path(directory).resolve()
    if directory.is_relative_to(ROOT):
        raise ValueError('WEIGHTS_MUST_BE_OUTSIDE_CODE_REPOSITORY')
    name = row['name']
    if Path(name).name != name or '/' in name or '\\' in name:
        raise ValueError('INVALID_CHECKPOINT_NAME')
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    if target.exists() or target.is_symlink():
        if target.is_symlink():
            raise ValueError('CHECKPOINT_TARGET_IS_SYMLINK')
        return verify_file(target, row)
    url = row.get('download_url')
    if not url:
        raise ValueError('NO_VERIFIED_DOWNLOAD_URL_USE_VERIFY_WITH_EXISTING_ASSET')
    parsed = urlsplit(url)
    if parsed.scheme != 'https' or parsed.hostname not in ('huggingface.co', 'dl.fbaipublicfiles.com') or parsed.username or parsed.password:
        raise ValueError('UNAPPROVED_ASSET_ORIGIN')
    if shutil.disk_usage(directory).free < row['original_bytes'] + 1024 ** 3:
        raise ValueError('INSUFFICIENT_DISK_SPACE')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.' + name + '.', suffix='.partial', dir=directory, delete=False) as output:
            temporary = Path(output.name)
            with opener(url, timeout=60) as response:
                if urlsplit(response.geturl()).scheme != 'https':
                    raise ValueError('INSECURE_ASSET_REDIRECT')
                size = 0
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > row['original_bytes']:
                        raise ValueError('CHECKPOINT_SIZE_EXCEEDED')
                    output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        result = verify_file(temporary, row)
        if os.name == 'nt':
            os.rename(temporary, target)
        else:
            os.link(temporary, target)
        return result
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['list', 'verify', 'download'])
    parser.add_argument('--name')
    parser.add_argument('--path', type=Path)
    parser.add_argument('--directory', type=Path)
    args = parser.parse_args()
    rows = json.loads((ROOT / 'checkpoints/manifest.json').read_text(encoding='utf-8'))['rows']
    if args.action == 'list':
        print(json.dumps([{'name': r['name'], 'category': r['category'], 'download_available': bool(r.get('download_url')), 'bytes': r['original_bytes']} for r in rows], indent=2))
        return 0
    selected = [r for r in rows if r['name'] == args.name]
    if len(selected) != 1:
        parser.error('--name must match an entry in checkpoints/manifest.json')
    if args.action == 'verify' and args.path is None:
        parser.error('--path is required for verify')
    if args.action == 'download' and args.directory is None:
        parser.error('--directory is required for download')
    try:
        result = verify_file(args.path, selected[0]) if args.action == 'verify' else download(selected[0], args.directory)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError) as error:
        message = str(error) if isinstance(error, ValueError) else type(error).__name__
        print(json.dumps({'status': 'FAILED', 'reason': message, 'model_deserialized': False}))
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
