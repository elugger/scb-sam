import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def validate_split(value):
    dataset = value['dataset']
    if dataset not in ('BY', 'Public433') or value.get('fold') != 2:
        raise ValueError('Unknown dataset/fold')
    n = value['labeled_cases']
    if type(n) is not int or n not in (5, 10):
        raise ValueError('Expected five or ten labeled cases')
    train_count, test_count = (100, 25) if dataset == 'BY' else (346, 87)
    counts = {'train': train_count, 'labeled': n, 'test': test_count}
    for role, count in counts.items():
        ids = value[role]
        if not isinstance(ids, list) or len(ids) != count:
            raise ValueError('Wrong split count: ' + role)
        if any(not isinstance(x, str) or not re.fullmatch(dataset.lower() + r'_case_[0-9]{4}', x) for x in ids):
            raise ValueError('Invalid release alias')
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate split alias')
    if not set(value['labeled']) <= set(value['train']):
        raise ValueError('Labeled cases outside training split')
    if set(value['train']) & set(value['test']):
        raise ValueError('Train/evaluation overlap')
    return {'dataset': dataset, 'labeled': n, 'unlabeled': train_count - n, 'evaluation': test_count}


def verify(root=ROOT):
    folder = root / 'splits'
    provenance = json.loads((folder / 'provenance.json').read_text(encoding='utf-8'))
    expected = {f'{dataset}/fold2_n{n:03d}.json' for dataset in ('by', 'public433') for n in (5, 10)}
    files = provenance['files']
    if len(files) != 4 or {r['path'] for r in files} != expected:
        raise ValueError('Incomplete split manifest')
    data = {}
    summaries = []
    for row in files:
        raw = (folder / row['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('Split bytes changed: ' + row['path'])
        value = json.loads(raw)
        summaries.append(validate_split(value))
        data[value['dataset'], value['labeled_cases']] = value
    if set(data) != {(d, n) for d in ('BY', 'Public433') for n in (5, 10)}:
        raise ValueError('Split identity differs from catalog')
    for dataset in ('BY', 'Public433'):
        five, ten = data[dataset, 5], data[dataset, 10]
        if five['train'] != ten['train'] or five['test'] != ten['test']:
            raise ValueError('Five/ten fold order differs')
        if not set(five['labeled']) <= set(ten['labeled']):
            raise ValueError('Five-label subset is not nested in ten-label subset')
    return {'status': 'PASS_RELEASE_ALIAS_SPLITS', 'splits': summaries, 'native_data_binding_verified': False, 'training_started': False}


if __name__ == '__main__':
    try:
        print(json.dumps(verify(), indent=2))
    except (OSError, KeyError, ValueError) as error:
        print(json.dumps({'status': 'SPLIT_VERIFICATION_FAILED', 'reason': str(error)}))
        raise SystemExit(2)
