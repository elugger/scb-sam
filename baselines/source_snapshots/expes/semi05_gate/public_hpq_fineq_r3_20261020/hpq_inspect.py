"""Read-only persisted ledger validation. No training or evaluation calls."""
import json
from pathlib import Path
from hpq_contract import require, validate_epoch_trace, validate_micro_trace


def read_rows(path, live=False):
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        require(live, 'Incomplete persisted JSONL: ' + path.name)
        raw = raw[:raw.rfind(b'\n') + 1]
    return [json.loads(line) for line in raw.splitlines() if line]


def validate_training_ledgers(output, splits, complete=False, live=False):
    from hpq_engine import Execution, validate_style
    from expes.semi05_gate.paired_rng_data_20260908 import labeled_tokens
    output = Path(output)
    # Read completed steps first: trailing microtrace may be in-flight.
    steps = read_rows(output / 'matrix_steps.jsonl', live)
    trace = read_rows(output / 'pair_trace.jsonl', live)
    epochs = read_rows(output / 'train_epochs.jsonl', live)
    coverage = read_rows(output / 'epoch_case_coverage.jsonl', live)
    helper = object.__new__(Execution)
    helper.group, helper.spec = 'A1B', {'phase': 'formal'}
    require(len(steps) <= 2484 and len(trace) <= 12420 and len(epochs) <= 36, 'Budget exceeded')
    for number, row in enumerate(steps, 1):
        epoch, within = divmod(number - 1, 69)
        require(row['epoch'] == epoch and row['successful_update'] == row['update'] == number,
                'Noncontinuous successful update ledger')
        helper.check_step(row)
        validate_style(row, splits['train_unlab'])
        require([m['step'] for m in row['microbatches']] == list(range(within * 5, within * 5 + 5)),
                'Update/microbatch correspondence differs')
    require(len(trace) >= len(steps) * 5, 'Successful update missing actual input trace')
    orders = {epoch: [t[2] for t in labeled_tokens(5, 345, epoch, 2027, False)] for epoch in range(36)}
    for index, row in enumerate(trace):
        epoch, micro = divmod(index, 345)
        validate_micro_trace(row, splits, epoch, micro, orders[epoch])
    require(len(coverage) >= len(epochs), 'Missing full-epoch coverage receipt')
    for epoch, row in enumerate(epochs):
        require(row['epoch'] == epoch and row['completed_epochs'] == epoch + 1
                and row['updates'] == (epoch + 1) * 69 and row['microbatches'] == (epoch + 1) * 345
                and row['test_evaluations'] == min(epoch, 35), 'Invalid epoch ledger')
        # During live reads the epoch file may advance after steps were read.
        if live and (epoch + 1) * 69 > len(steps):
            require(epoch == len(epochs) - 1, 'More than one in-flight epoch')
            continue
        require((epoch + 1) * 69 <= len(steps), 'Completed epoch lacks updates')
        receipt = validate_epoch_trace(trace[epoch * 345:(epoch + 1) * 345], splits, epoch, orders[epoch])
        require(receipt == coverage[epoch], 'Coverage receipt differs from actual cases')
    if complete:
        require((len(steps), len(trace), len(epochs), len(coverage)) == (2484, 12420, 36, 36),
                'Full training ledger incomplete')
    return dict(successful_updates=len(steps), recorded_microbatches=len(trace), completed_epochs=len(epochs),
                percent=len(steps) / 2484 * 100, all_persisted_steps_checked=True,
                actual_case_trace_checked=True, formal_branch_routing_measured=False)
