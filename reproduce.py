import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8-sig'))

def catalog():
    rows = []
    for item in read('records/main_table.json')['rows']:
        rows.append(dict(item, table='main'))
    for item in read('ours/accepted_summary.json')['groups']:
        rows.append(dict(id='ablation_' + item['group'], table='ablation', group=item['group'], metrics={'dice': item['dice_percent'] / 100}, source_result_sha256=item['source_result_sha256'], config='records/configurations/' + item['group'] + '.json', checkpoint_available=item['checkpoint_available']))
    return rows

def selected(identifier):
    matches = [row for row in catalog() if row['id'] == identifier]
    if len(matches) != 1:
        raise ValueError('Unknown experiment: ' + identifier)
    return matches[0]

def routes():
    path = ROOT / 'scripts/experiment_routes.json'
    if not path.exists():
        return []
    value = read('scripts/experiment_routes.json')
    if isinstance(value, list):
        return value
    return value.get('routes', value.get('experiments', value.get('rows', [])))

def verify():
    rows = catalog()
    errors = []
    try:
        from scripts.verify_splits import verify as verify_splits
        verify_splits()
    except (OSError, KeyError, ValueError) as error:
        errors.append('Split validation: ' + str(error))
    if len(rows) != 32 or len({row['id'] for row in rows}) != 32:
        errors.append('Expected 28 main-table and 4 ablation entries')
    for row in rows:
        path = ROOT / row['config']
        if not path.is_file() or not path.resolve().is_relative_to(ROOT):
            errors.append('Missing or invalid config: ' + row['id'])
    for name in ['README.md', 'LICENSE', 'docs/READINESS.json', 'scripts/experiment_routes.json']:
        if not (ROOT / name).is_file():
            errors.append('Missing project file: ' + name)
    mapping = routes()
    if len(mapping) != 32 or {row.get('id') for row in mapping} != {row['id'] for row in rows}:
        errors.append('Route catalog differs from the 32 recorded experiments')
    for route in mapping:
        path = (ROOT / route['entrypoint']).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            errors.append('Missing or invalid entrypoint: ' + route['id'])
        elif hashlib.sha256(path.read_bytes()).hexdigest() != route['entrypoint_sha256']:
            errors.append('Entrypoint checksum mismatch: ' + route['id'])
    for path in [ROOT / 'reproduce.py', ROOT / 'ours/hp_hq_final.py']:
        try:
            ast.parse(path.read_text(encoding='utf-8'))
        except SyntaxError as error:
            errors.append(str(error))
    binaries = [str(p.relative_to(ROOT)) for suffix in ['*.pth', '*.pt', '*.ckpt', '*.safetensors'] for p in ROOT.rglob(suffix)]
    if binaries:
        errors.append('Weight binaries must remain outside the code repository')
    state = read('docs/READINESS.json') if (ROOT / 'docs/READINESS.json').exists() else {'ready_to_release': False, 'blockers': ['readiness_file_missing']}
    return dict(status='STRUCTURE_ERRORS' if errors else ('READY' if state['ready_to_release'] else 'NOT_READY_TO_RELEASE'), structural_checks_passed=not errors, experiment_count=len(rows), weight_binaries_in_repository=len(binaries), training_started=False, errors=errors, readiness=state)

def plan(identifier):
    row = selected(identifier)
    matching = [entry for entry in routes() if entry.get('id', entry.get('experiment_id')) == identifier]
    return dict(experiment=row, recorded_route=matching[0] if matching else None, readiness=read('docs/READINESS.json'), training_started=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['list', 'report', 'plan', 'verify', 'train'])
    parser.add_argument('--experiment')
    args = parser.parse_args()
    if args.action in ['plan', 'train'] and not args.experiment:
        parser.error('--experiment is required')
    if args.action == 'list':
        value = [dict(id=row['id'], table=row['table']) for row in catalog()]
    elif args.action == 'report':
        value = selected(args.experiment) if args.experiment else catalog()
    elif args.action == 'plan':
        value = plan(args.experiment)
    elif args.action == 'verify':
        value = verify()
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0 if value['status'] == 'READY' else 2
    else:
        selected(args.experiment)
        state = read('docs/READINESS.json')
        print(json.dumps(dict(status='TRAINING_NOT_DISPATCHED', reason='The clean-machine reproduction runtime has not been qualified. Recorded controllers and result reports do not establish runnable reproduction.', blockers=state['blockers'], training_started=False), ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, KeyError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
