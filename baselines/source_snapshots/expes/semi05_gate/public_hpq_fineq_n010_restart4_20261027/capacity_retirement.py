"""Exact approved10GiB retirement, preserving every unrelated historical target."""
import json
import os
from pathlib import Path
from hpq_contract import ROOT, require
from hpq_inputs import bound_small_bytes

RECEIPT = ROOT / 'expes/work_dir/semi05_gate/MAIN_TABLE_FIXED_FINAL_20261019/gpu0_approved_10g_empty_retirement_20261027/receipt.json'
RECEIPT_SHA = '5d8bfd2622ee7a98bb44a93bb7474e37705b6d2eb3014dcd0ce3cff9b6d92334'

def expected_paths():
    a = ROOT / 'expes/artifacts/FCD_A_MATRIX_20261006/MAIN_TABLE_FIXED_FINAL_20261019/hpq_ncase/Public433'
    result = {str(a / 'lab005/fold2/fine_q_r2/r1/safety_slot0' / p / 'final.pth') for p in ('weights', 'optimizer')}
    result |= {str(a / 'lab010/fold2/fine_q_best/r1/safety_slot1' / p / 'final.pth') for p in ('weights', 'optimizer')}
    result.add(str(ROOT / 'expes/artifacts/FCD_A_MATRIX_20261006/PUBLIC05_HPQ_EPOCHDIAG_20261019/fold2/A1B/formal_r1/final.pth'))
    return result

def apply_retirement(current, registered):
    proof = json.loads(bound_small_bytes(RECEIPT, RECEIPT_SHA))
    expected = expected_paths()
    require(proof['status'] == 'AUTHORIZED_EMPTY_10G_RESERVATIONS_RETIRED'
            and proof['files_deleted'] == 0 and proof['other_rows_exactly_preserved'] is True
            and proof['retired_reserved_bytes'] == 10 * 1024**3
            and {r['path'] for r in proof['removed']} == expected
            and all(r['bound'] == 2 * 1024**3 for r in proof['removed']), 'Wrong approved10GiB receipt')
    for p, v in proof['terminal_evidence'].items():
        bound_small_bytes(Path(p), v['sha256'])
    merged = {str(p): b for p, b in current}
    for row in proof['before_guard_targets']:
        merged[row['path']] = max(merged.get(row['path'], 0), row['bound'])
    require(not expected.intersection(registered), 'Approved retired target re-registered')
    for n in expected:
        for p in (Path(n), Path(n + '.partial')):
            require(not any(q.is_symlink() for q in (p, *p.parents)) and not os.path.lexists(p),
                    'Approved retired target occupied/symlinked')
        merged.pop(n, None)
    return [(Path(p), b) for p, b in merged.items()]
