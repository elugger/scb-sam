"""Exclusive epoch diagnostics; development test use, never peak selection."""
import contextlib
import json
import os
from pathlib import Path
from hpq_contract import BASE, require, sha_file, is_sha


def marker_name(epoch):
    require(type(epoch) is int and 0 <= epoch < 36, 'Diagnostic epoch outside 0..35')
    return 'epoch_{:02d}_test_started.json'.format(epoch)


def validate_prior(output, count):
    """Require completed, hash-bound predecessors, not merely test intents."""
    for epoch in range(count):
        marker = Path(output) / marker_name(epoch)
        result = Path(output) / 'epoch_{:02d}_test_metrics.json'.format(epoch)
        require(marker.is_file() and result.is_file(), 'Missing prior completed diagnostic')
        m, r = json.loads(marker.read_text()), json.loads(result.read_text())
        require(m['epoch'] == r['epoch'] == epoch and r['test_cases'] == 87
                and r['updates'] == (epoch + 1) * 68 and r['isolation_verified'] is True
                and r['intent_sha256'] == sha_file(marker)
                and r['manifest_sha256'] == sha_file(Path(output) / 'manifest.json'),
                'Prior diagnostic identity/count differs')
        cases = Path(output) / 'epoch_{:02d}_test_cases.jsonl'.format(epoch)
        require(cases.is_file() and r['cases_sha256'] == sha_file(cases), 'Diagnostic cases changed')
        rows = [json.loads(line) for line in cases.read_text().splitlines() if line]
        require(len(rows) == 87 and [row['case_id'] for row in rows] == m['case_ids']
                and [row['actual_index'] for row in rows] == list(range(87))
                and all(row['epoch'] == epoch and row['updates'] == (epoch + 1) * 68 for row in rows),
                'Diagnostic persisted case coverage differs')


def claim_diagnostic(output, epoch, updates, phase, case_ids):
    output = Path(output)
    name = marker_name(epoch)
    require(phase == 'formal' and output.as_posix() == str(BASE / 'formal_r1'), 'Only own formal diagnostics')
    require(type(updates) is int and updates == (epoch + 1) * 68, 'Incomplete epoch cannot evaluate')
    require(len(case_ids) == len(set(case_ids)) == 87, 'Wrong test set')
    require(not any(p.is_symlink() for p in (output, *output.parents)), 'Symlink output')
    validate_prior(output, epoch)
    require(not (output / 'test_evaluation_started.json').exists(), 'Final evaluation already started')
    marker = output / name
    with marker.open('x', encoding='utf-8') as stream:
        json.dump(dict(epoch=epoch, updates=updates, phase=phase, test_cases=87,
                       case_ids=list(case_ids), manifest_sha256=sha_file(output / 'manifest.json'),
                       selection_rule='fixed_epoch35_native_student', development_test_use=True,
                       automatic_retry=False), stream)
        stream.flush(); os.fsync(stream.fileno())
    return marker


def validate_permit(permit, case_ids):
    p = Path(permit) if permit is not None else None
    require(p is not None and p.parent.as_posix() == str(BASE / 'formal_r1')
            and p.is_file() and not any(x.is_symlink() for x in (p, *p.parents)),
            'No unique final test intent or epoch diagnostic intent')
    m = json.loads(p.read_text())
    require(len(case_ids) == len(set(case_ids)) == m['test_cases'] == 87, 'Wrong permitted test count')
    if p.name == 'test_evaluation_started.json':
        require(is_sha(m['final_sha256']) and m['selection_rule'] == 'fixed_epoch35_native_student',
                'Wrong final permit')
    else:
        require(p.name == marker_name(m['epoch']) and m['phase'] == 'formal'
                and m['updates'] == (m['epoch'] + 1) * 68 and m['case_ids'] == list(case_ids)
                and m['manifest_sha256'] == sha_file(p.parent / 'manifest.json')
                and m['development_test_use'] is True, 'Wrong diagnostic permit')


@contextlib.contextmanager
def isolated_evaluation(engine):
    """Frozen cache/buffer/mode/RNG restoration plus independent full state checks."""
    before, rng, updates = engine.state(), engine.rng(), engine.t.matrix_updates
    roots = engine.t.roots()
    modules = [(m, m.training) for root in roots.values() for m in root.modules()]
    parameters = [p for root in roots.values() for p in root.parameters()]
    gradients = engine.signature([p.grad for p in parameters])
    try:
        with engine.t.isolated_context(0, 0), engine.torch.no_grad():
            for root in roots.values(): root.eval()
            yield
    finally:
        require(engine.state() == before and engine.rng() == rng
                and engine.t.matrix_updates == updates, 'Evaluation changed roots/optimizer/scaler/RNG/update')
        require(all(m.training == mode for m, mode in modules), 'Evaluation changed module modes')
        require(engine.signature([p.grad for p in parameters]) == gradients, 'Evaluation changed gradients')
