"""Local new ledger integration checks; synthetic receipts, no model or GPU."""
import argparse
import copy
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from relif_binding import ReLiFBinding
from relif_ledger import project, project_file, scan
from relif_state import counts, require
from relif_storage import read_pin


def main():
    ap = argparse.ArgumentParser()
    for name in ('native-dir', 'by-contract', 'budget-file', 'split-dir', 'out'):
        ap.add_argument('--'+name, required=True)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    require(not out.exists(), 'Receipt already exists')
    binding = ReLiFBinding(5, native_dir=a.native_dir, by_contract=a.by_contract,
                           budget_file=a.budget_file, split_dir=a.split_dir)
    recipe = binding.recipe
    checks = []

    def ok(name, condition):
        require(condition, name)
        checks.append(name)

    def reject(name, action):
        try:
            action()
        except (ValueError, KeyError, TypeError):
            checks.append(name)
            return
        raise AssertionError('Did not reject: '+name)

    def dump(value):
        return (json.dumps(value, allow_nan=False)+'\n').encode()

    def pin_bytes(raw):
        return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), device=11, inode=22)

    rows = []
    for i in range(348):
        pair, clock = binding.pair(i), binding.original_clock(i)
        native = dict(step=i+1, clock=clock, primary_labeled=1, primary_unlabeled=1,
            optimizer_updates=1, student_case_forwards=4, foundation_case_forwards=2,
            sobf_views=2, teacher_case_forwards=0, extra_donor_inputs=0, test_evaluations=0,
            unlab_gt_in_loss=False, case_ids=dict(labeled=pair['labeled'], unlabeled=pair['unlabeled']))
        step = dict(actual_pair=pair, completed_primary_pairs=i+1, labeled_inputs=1, unlabeled_inputs=1,
            successful_optimizer_updates=1, finite_loss=True, finite_nonzero_gradients=True,
            actual_case_ids_verified=True, teacher_no_grad=True, skip_count=0, unlabeled_gt_in_loss=False,
            native_receipt=native, losses={k:1. for k in
                ('loss','supervised_ce','supervised_dice','distillation','cgcr','larc')}, gradient_norm=1.)
        rows.append(dict(step=step, cumulative=counts(i+1)))
    lines = [dump(row) for row in rows]
    prefix, tail, partial = b''.join(lines[:300]), b''.join(lines[300:]), b'{"step":'
    raw = prefix+tail+partial
    commit = dict(contract_sha256='a'*64, execution_role='formal', generation=3, sha256='b'*64,
                  counts=counts(300), ledger=pin_bytes(prefix))
    result = project(raw, pin_bytes(raw), commit, binding, 'a'*64)
    ok('committed_counts_remain_300_observed_348', result['committed_counts'] == counts(300)
       and result['observed_complete_counts'] == counts(348))
    ok('tail_uses_relif_one_update_one_input_per_stream', all(result['complete_tail'][k] == 48
       for k in ('iterations','successful_optimizer_updates','labeled_inputs','unlabeled_inputs')))
    ok('opaque_partial_tail_kept_byte_exact', result['partial_tail'] ==
       dict(bytes=len(partial),sha256=hashlib.sha256(partial).hexdigest()))
    ok('projection_never_approves_resume_or_loads_checkpoint',
       result['independent_resume_review_passed'] is False and result['checkpoint_deserialized'] is False
       and result['ledger_modified'] is False)
    ok('failed_inflight_cost_not_assumed_zero', result['failed_or_inflight_compute_fully_measured'] is False
       and result['failed_or_inflight_compute_must_not_be_assumed_zero'] is True)
    ok('native_cycle_coverage_matches_actual_rows', result['complete_reference_cycles'] == 1
       and result['actual_train_cases'] == 346)
    empty = dict(contract_sha256='a'*64, execution_role='formal', generation=0, sha256='c'*64,
                 counts=counts(0), ledger=dict(pin_bytes(b''), inode=None, device=None))
    ok('zero_commit_before_ledger_creation', project(raw,pin_bytes(raw),empty,binding,'a'*64)
       ['complete_tail']['iterations'] == 348)

    def changed_commit(name, field, value):
        modified = copy.deepcopy(commit)
        modified[field] = value
        reject(name, lambda: project(raw,pin_bytes(raw),modified,binding,'a'*64))
    changed_commit('wrong_contract', 'contract_sha256', 'd'*64)
    changed_commit('engineering_payload_in_formal_ledger', 'execution_role', 'engineering_larc')
    changed_commit('invalid_checkpoint_sha', 'sha256', 'not-a-sha')
    changed_commit('boolean_generation', 'generation', True)
    changed_commit('checkpoint_past_observed_ledger', 'counts', counts(349))
    modified = copy.deepcopy(commit)
    modified['ledger']['inode'] += 1
    reject('unreviewed_inode_replacement',lambda:project(raw,pin_bytes(raw),modified,binding,'a'*64))
    modified = copy.deepcopy(commit)
    modified['ledger']['bytes'] += 1
    reject('commit_not_at_original_complete_row_boundary',lambda:project(raw,pin_bytes(raw),modified,binding,'a'*64))
    edited_prefix = b' '+raw
    reject('even_semantically_equal_committed_bytes_cannot_change',
           lambda:project(edited_prefix,pin_bytes(edited_prefix),commit,binding,'a'*64))
    bad_pin = dict(pin_bytes(raw),sha256='0'*64)
    reject('raw_bytes_do_not_match_same_fd_pin',lambda:project(raw,bad_pin,commit,binding,'a'*64))
    reject('unknown_complete_tail_not_silently_discarded',lambda:scan(prefix+b'{}\n',binding))
    reject('unparseable_complete_tail_preserved_for_review',lambda:scan(prefix+b'{broken}\n',binding))
    reject('duplicate_json_key_not_silently_overwritten',lambda:scan(
        b''.join(lines[:300])+lines[300].replace(b'{"step":', b'{"step":{},"step":',1),binding))

    def changed_tail(name, modify):
        row = copy.deepcopy(rows[300])
        modify(row)
        reject(name,lambda:scan(prefix+dump(row),binding))
    changed_tail('repeated_native_index',lambda row:row['step'].update(actual_pair=binding.pair(299)))
    changed_tail('wrong_cumulative_count',lambda row:row.update(cumulative=counts(302)))
    changed_tail('compressed_native_clock',lambda row:row['step']['native_receipt']['clock'].update(
        total=recipe['ssl_iterations']))
    changed_tail('two_updates_cannot_be_hidden_as_relif_one',lambda row:row['step'].update(
        successful_optimizer_updates=2))
    changed_tail('teacher_compute_cannot_be_fabricated',lambda row:row['step']['native_receipt'].update(
        teacher_case_forwards=1))
    fixture = out.parent/(out.stem+'_fixtures')
    fixture.mkdir(exist_ok=False)
    ledger = fixture/'steps.jsonl'
    with ledger.open('xb') as f:
        f.write(raw)
    before = read_pin(ledger,fixture)
    commit['ledger'].update(device=before['device'],inode=before['inode'])
    on_disk = project_file(ledger,fixture,commit,binding,'a'*64)
    ok('storage_same_fd_reader_and_binding_integrate_without_writes',
       read_pin(ledger,fixture) == before and on_disk['complete_tail']['iterations'] == 48)
    ok('no_model_patient_or_tensor_dependency_imported', not any(n == 'torch' or n.startswith('torch.')
       or n in ('numpy','torchio','SimpleITK') for n in sys.modules))
    here = Path(__file__).parent
    record = dict(status='LOCAL_LEDGER_BINDING_STORAGE_INTEGRATION_PASS_NOT_RESUME_APPROVAL',
        at=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),
        checks=checks,check_count=len(checks),synthetic_receipts=True,actual_training_updates=0,
        model_loads=0,cuda_loads=0,patient_loads=0,actual_test_evaluations=0,
        remote_execution=False,new_job_created=False,new_reservations=False,
        checkpoint_tensor_roundtrip=False,independent_resume_review_passed=False,
        preserved_prior_checks_not_rerun=[98,31,30],
        sources={name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in
                 ('relif_ledger.py','check_relif_ledger_cpu.py','relif_binding.py','relif_state.py','relif_storage.py')},
        representative_projection=on_disk)
    with out.open('x',encoding='utf-8') as f:
        json.dump(record,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status=record['status'],check_count=len(checks),
                         receipt_sha256=hashlib.sha256(out.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
