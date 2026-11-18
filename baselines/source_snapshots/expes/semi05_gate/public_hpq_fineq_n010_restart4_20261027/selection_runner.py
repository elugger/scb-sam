"""One bounded run; fixed science, prospective early stop and durable best."""
import json
import time
from hpq_contract import INIT_SHA, require, sha_file
from hpq_diagnostics import validate_prior
from hpq_safety_state import save_epoch
from best_selection import save_best, verify_endpoint
from selection_protocol import decision, evaluation_record, budget, POLICY, policy_sha


def run(engine):
    from hpq_engine import write_json
    from hpq_inspect import read_rows, validate_training_ledgers
    before, rng = engine.state(), engine.rng()
    with engine.t.isolated_context(0, 0):
        iterator = engine.raw_groups(0)
        first = next(iterator)
        image = first[0]['image_l']
        del iterator, first
    require(engine.state() == before and engine.rng() == rng, 'Startup input changed state')
    write_json(engine.output / 'native_initial_identity.json', engine.native_initial_identity(image))
    write_json(engine.output / 'startup_disk_gate.json', engine.disk_roundtrip(image))
    del image
    history, selected = [], None
    for epoch in range(36):
        started, count = time.monotonic(), 0
        engine.t.eval_epoch = epoch
        for items in engine.raw_groups(epoch, record=True):
            row = engine.checked_step(items)
            count += 1
            require(engine.t.matrix_updates == epoch * 68 + count, 'Nonconsecutive update')
            engine.append('matrix_steps.jsonl', dict(epoch=epoch, successful_update=engine.t.matrix_updates, **row))
            print('HPQ_UPDATE', engine.t.matrix_updates, flush=True)
        require(count == 68, 'Incomplete cycle')
        engine.append('train_epochs.jsonl', dict(epoch=epoch, completed_epochs=epoch + 1,
                      updates=engine.t.matrix_updates, microbatches=(epoch + 1) * 340,
                      seconds=time.monotonic() - started, test_evaluations=epoch))
        metrics = engine.diagnostic_test(epoch)
        cases = read_rows(engine.output / ('epoch_%02d_test_cases.jsonl' % epoch))
        entry = evaluation_record(epoch + 1, engine.t.matrix_updates, cases, engine.input_splits['test'])
        history.append(entry)
        choice = decision(history)
        safety = save_epoch(engine, epoch)
        if choice['current_is_strict_best']:
            selected = save_best(engine, epoch, selected)
        require(selected is not None and selected['epoch'] + 1 == choice['best_completed_epoch'],
                'Best model and selected metrics differ')
        engine.append('selection_history.jsonl', dict(**entry, decision=choice,
                      metrics_sha256=sha_file(engine.output / ('epoch_%02d_test_metrics.json' % epoch)),
                      best_sha256=selected['sha256'], safety_epoch=safety['epoch']))
        print('FINEQ_SELECTION', json.dumps(dict(**entry, decision=choice)), flush=True)
        if choice['stop']:
            break
    require(choice['stop'], 'Run did not end on frozen rule')
    n = len(history)
    ledger = validate_training_ledgers(engine.output, engine.input_splits, complete=False)
    require((ledger['successful_updates'], ledger['recorded_microbatches'], ledger['completed_epochs']) ==
            (n * 68, n * 340, n), 'Endpoint ledger incomplete')
    require(len(read_rows(engine.output / 'epoch_case_coverage.jsonl')) == n
            and len(read_rows(engine.output / 'selection_history.jsonl')) == n,
            'Endpoint coverage or selection history incomplete')
    validate_prior(engine.output, n)
    endpoint = verify_endpoint(engine, selected)
    write_json(engine.output / 'selected_endpoint_verified.json', endpoint)
    result = dict(status='SELECTION_COMPLETE', completed_epochs=n, successful_updates=n * 68,
                  microbatches=n * 340, development_evaluations=n, policy=POLICY, policy_sha256=policy_sha(),
                  selection=choice, selected_endpoint=endpoint, budget=budget(n), ledger=ledger,
                  initializer_sha256=INIT_SHA, automatic_resume=False, clean_confirmation=False,
                  gt_roi_used=True, patient_alias_boundary_verified=False,
                  actual_total_compute_includes_all_completed_cycles=True,
                  wall_seconds=time.monotonic() - engine.begin)
    write_json(engine.output / 'selection_metrics.json', result)
    return result
