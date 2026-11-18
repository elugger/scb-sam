import hashlib
import importlib.util
import io
import json
import marshal
from pathlib import Path
import sys
import tokenize
import types

ROOT = Path(__file__).resolve().parents[1]

def normalize(code):
    return code.replace(co_filename='<source>', co_consts=tuple(normalize(x) if isinstance(x, types.CodeType) else x for x in code.co_consts))

def main():
    if sys.version_info[:2] != (3, 10) or importlib.util.MAGIC_NUMBER.hex() != '6f0d0d0a':
        print(json.dumps({'status': 'REQUIRES_CPYTHON_310_FOR_BYTECODE_COMPARISON', 'python': sys.version, 'training_started': False}))
        return 2
    manifest = json.loads((ROOT / 'docs/CASCADE_SOURCE_VERIFICATION.json').read_text(encoding='utf-8'))
    rows = []
    for item in manifest['modules']:
        for relative in item['release_paths']:
            path = ROOT / relative
            raw = path.read_bytes()
            compiled = compile(raw, '<source>', 'exec', dont_inherit=True, optimize=0)
            source_sha = hashlib.sha256(raw).hexdigest()
            code_sha = hashlib.sha256(marshal.dumps(normalize(compiled), manifest['marshal_serialization_version'])).hexdigest()
            comments = sum(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(raw.decode('utf-8')).readline))
            rows.append({'file': relative, 'source_sha_matches': source_sha == item['release_sha256'], 'complete_code_sha_matches_training_reference': code_sha == item['filename_normalized_complete_code_sha256'], 'comments': comments})
    passed = all(r['source_sha_matches'] and r['complete_code_sha_matches_training_reference'] and r['comments'] == 0 for r in rows)
    print(json.dumps({'status': 'PASS' if passed else 'FAIL', 'normalization': manifest['normalization'], 'files': rows, 'source_execution_performed': False, 'training_started': False}, indent=2))
    return 0 if passed else 1

if __name__ == '__main__':
    raise SystemExit(main())
