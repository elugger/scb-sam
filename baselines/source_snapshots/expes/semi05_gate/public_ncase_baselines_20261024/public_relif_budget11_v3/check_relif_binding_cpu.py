"""New binding-interface checks only; no prior metadata/IO qualification rerun."""
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from relif_binding import ReLiFBinding, stable


def main():
    ap = argparse.ArgumentParser()
    for name in ('native-dir', 'by-contract', 'budget-file', 'split-dir', 'prior-metadata', 'out'):
        ap.add_argument('--'+name, required=True)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    require = lambda v: None if v else (_ for _ in ()).throw(AssertionError('Check failed'))
    require(not out.exists())
    prior_bytes = Path(args.prior_metadata).read_bytes()
    require(hashlib.sha256(prior_bytes).hexdigest() ==
            '17cd6da0f0754992274f5f20269d770d71e9d88dec4f9db575b53f816766b97a')
    prior = json.loads(prior_bytes)
    inputs = dict(native_dir=args.native_dir, by_contract=args.by_contract,
                  budget_file=args.budget_file, split_dir=args.split_dir)
    checks, evidence = [], []

    def ok(name, value):
        require(value)
        checks.append(name)

    def reject(name, operation):
        try:
            operation()
        except (ValueError, FileNotFoundError):
            checks.append(name)
            return
        raise AssertionError('Did not reject: '+name)

    for n in (5, 10):
        binding = ReLiFBinding(n, **inputs)
        r = binding.recipe
        old = next(row for row in prior['records'] if row['labeled_cases'] == n)
        tag = str(n)+'_'
        ok(tag+'bound_full_clock_and_cutoff', binding.evidence()['original_pair_count'] ==
           old['original_clock_iterations'] and r['ssl_iterations'] == old['ssl_iterations'])
        ok(tag+'new_binding_matches_preserved_index_proof',
           stable(binding.executed_prefix(r['ssl_iterations'])) == old['pair_prefix_sha256'])
        ok(tag+'new_binding_matches_preserved_last_clock',
           binding.original_clock(r['ssl_iterations']-1) == old['last_clock'])
        reject(tag+'compressed_core_total', lambda: binding.require_core_total(r['ssl_iterations']))
        reject(tag+'after_formal_cutoff', lambda: binding.pair(r['ssl_iterations']))
        reject(tag+'warmup_window_cannot_extend', lambda: binding.pair(4, 'engineering_warmup'))
        reject(tag+'larc_window_cannot_start_early', lambda: binding.pair(
            r['original_larc_start_index']-1, 'engineering_larc'))
        start = r['original_larc_start_index']
        ok(tag+'engineering_history_excludes_unexecuted_warmup',
           binding.executed_prefix(start, 'engineering_larc') == [] and
           len(binding.executed_prefix(start+2, 'engineering_larc')) == 2)
        reject(tag+'engineering_resume_cannot_claim_zero_cursor',
               lambda: binding.executed_prefix(0, 'engineering_larc'))
        changed = binding.recipe
        changed['original_clock_iterations'] = 1
        changed = binding.splits
        changed['train_lab'].clear()
        changed = binding.pair(0)
        changed.clear()
        ok(tag+'caller_cannot_mutate_bound_metadata', binding.recipe['original_clock_iterations'] ==
           old['original_clock_iterations'] and len(binding.splits['train_lab']) == n and bool(binding.pair(0)))
        evidence.append(binding.evidence())

    for bad in (True, 17, 35):
        reject('wrong_label_setting_'+str(bad), lambda bad=bad: ReLiFBinding(bad, **inputs))
    reject('boolean_clock_rejected', lambda: binding.original_clock(True))
    reject('unknown_execution_role', lambda: binding.pair(0, 'unknown'))

    # Keep these small adversarial fixtures with the receipt. No protected file
    # is edited and no recursive deletion or model checkpoint is involved.
    fixtures = out.parent / (out.stem+'_fixtures')
    fixtures.mkdir(exist_ok=False)
    for field, path in (('by_contract', args.by_contract), ('budget_file', args.budget_file)):
        tampered = fixtures / (field+'.json')
        tampered.write_bytes(Path(path).read_bytes()+b'\n')
        changed = dict(inputs, **{field: str(tampered)})
        reject('changed_'+field+'_fails_before_native_execution',
               lambda changed=changed: ReLiFBinding(5, **changed))
    native_dir = fixtures / 'native'
    native_dir.mkdir()
    (native_dir/'budget_protocol.py').write_bytes(b'raise RuntimeError("MUST_NOT_EXECUTE")\n')
    reject('changed_native_definition_not_executed',
           lambda: ReLiFBinding(5, **dict(inputs, native_dir=str(native_dir))))
    split_dir = fixtures / 'splits'
    split_dir.mkdir()
    for name in json.loads(Path(args.budget_file).read_bytes())['split_pins']:
        data = (Path(args.split_dir)/name).read_bytes()
        if name == 'n005_lab.txt':
            data += b'\n'
        (split_dir/name).write_bytes(data)
    reject('modified_split_bytes_rejected',
           lambda: ReLiFBinding(5, **dict(inputs, split_dir=str(split_dir))))
    ok('no_model_dependency_imported', not any(n == 'torch' or n.startswith('torch.')
       or n in ('numpy', 'SimpleITK', 'torchio') for n in sys.modules))
    record = dict(status='LOCAL_STDLIB_BINDING_INTERFACE_PASS_NOT_REMOTE_QUALIFICATION',
        at=datetime.now(timezone.utc).isoformat(), python=sys.version,
        platform=platform.platform(), check_count=len(checks), checks=checks,
        bound_evidence=evidence, remote_execution=False, remote_gpu_state_verified=False,
        prior_98_metadata_checks_reused_not_rerun=True, prior_31_io_checks_reused_not_rerun=True,
        model_loads=0, cuda_loads=0, patient_loads=0, optimizer_updates=0, test_evaluations=0,
        new_formal_jobs=0, new_reservations=0, gpu_qualified=False,
        sources={name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                 for name in ('relif_binding.py', 'check_relif_binding_cpu.py', 'relif_state.py')})
    with out.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(status=record['status'], check_count=len(checks),
                         receipt_sha256=hashlib.sha256(out.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
