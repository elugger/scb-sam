import argparse
import hashlib
import json
import io
import os
from pathlib import Path, PurePosixPath
import tempfile
import subprocess
from urllib.parse import quote, urlparse
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def curl_open(url, timeout=30):
    result = subprocess.run(['curl', '--proto', '=https', '--proto-redir', '=https', '--location', '--fail', '--silent', '--show-error', '--retry', '2', '--retry-all-errors', '--max-time', str(timeout), '--max-filesize', '10485760', url], capture_output=True, check=True)
    return io.BytesIO(result.stdout)


def github_open(url, timeout=30):
    parts = urlparse(url).path.strip('/').split('/', 3)
    owner, repository, revision, path = parts
    endpoint = 'repos/' + owner + '/' + repository + '/contents/' + path + '?ref=' + revision
    result = subprocess.run(['gh', 'api', endpoint, '--header', 'Accept: application/vnd.github.raw+json'], capture_output=True, check=True, timeout=timeout)
    return io.BytesIO(result.stdout)


def entries(component=None):
    rows = json.loads((ROOT / 'scripts/upstream_sources.json').read_text())['files']
    if component:
        rows = [row for row in rows if row['component'] == component]
        if not rows:
            raise ValueError('Unknown component')
    return rows


def destination(root, row):
    relative = PurePosixPath(row['local_path'])
    if relative.is_absolute() or '..' in relative.parts or '\\' in str(relative):
        raise ValueError('Invalid source path')
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError('Source path escapes destination')
    return path


def matches(path, row):
    return path.is_file() and path.stat().st_size == row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']


def fetch_one(root, row, opener=urlopen):
    path = destination(root, row)
    if path.exists():
        if not matches(path, row):
            raise ValueError('Existing source differs; preserved: ' + row['local_path'])
        return 'already_verified'
    if len(row['revision']) != 40 or any(c not in '0123456789abcdef' for c in row['revision']):
        raise ValueError('A fixed upstream revision is required')
    url = 'https://raw.githubusercontent.com/' + row['repository'] + '/' + row['revision'] + '/' + quote(row['upstream_path'], safe='/')
    upstream_bytes = row.get('upstream_bytes', row['bytes'])
    with opener(url, timeout=30) as stream:
        raw = stream.read(upstream_bytes + 1)
    if len(raw) != upstream_bytes or hashlib.sha256(raw).hexdigest() != row.get('upstream_sha256', row['sha256']):
        raise ValueError('Original upstream source hash mismatch: ' + row['local_path'])
    for change in row.get('transforms', []):
        if change['operation'] in ('lf', 'crlf'):
            raw = raw.replace(b'\r\n', b'\n')
            if change['operation'] == 'crlf':
                raw = raw.replace(b'\n', b'\r\n')
        elif change['operation'] == 'replace':
            old, new = change['old'].encode(), change['new'].encode()
            if raw.count(old) != change['count']:
                raise ValueError('Source transformation count mismatch')
            raw = raw.replace(old, new)
        else:
            raise ValueError('Unknown source transformation')
    if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
        raise ValueError('Upstream source hash mismatch: ' + row['local_path'])
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.parent.resolve().is_relative_to(root):
        raise ValueError('Source parent escapes destination')
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.source-')
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return 'downloaded_verified'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['list', 'verify', 'fetch'])
    parser.add_argument('--component', choices=['semisam', 'sdcl', 'sfr', 'relif', 'dualteacher'])
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--transport', choices=['urllib', 'curl', 'github-cli'], default='urllib')
    args = parser.parse_args()
    rows = entries(args.component)
    if args.action == 'list':
        print(json.dumps(rows, indent=2))
        return 0
    if args.destination is None:
        parser.error('--destination is required')
    root = args.destination.resolve()
    if args.action == 'fetch' and (root == ROOT or root.is_relative_to(ROOT)):
        parser.error('Acquire upstream sources outside the publication repository')
    results = []
    for row in rows:
        if args.action == 'fetch':
            status = fetch_one(root, row, {'curl': curl_open, 'urllib': urlopen, 'github-cli': github_open}[args.transport])
        else:
            status = 'verified' if matches(destination(root, row), row) else 'missing_or_different'
        results.append(dict(path=row['local_path'], status=status))
    print(json.dumps(dict(files=results, source_execution_performed=False, training_qualified=False), indent=2))
    return int(any(r['status'] == 'missing_or_different' for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
