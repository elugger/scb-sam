"""Conservative historical union, minus only five explicitly retired empty targets."""
import json
import os
from pathlib import Path
from hpq_contract import ROOT, require
from hpq_inputs import bound_small_bytes

HISTORY = ROOT / 'expes/work_dir/semi05_gate/PUBLIC_N005_NATIVEQ_FAST_20261024/control/formal.intent.json'
HISTORY_SHA = '4394ea102b73fc50b75d3c387e824813b5861d0dfb7612a82d5e9cb0ff837211'
RECEIPT = ROOT / 'expes/work_dir/semi05_gate/MAIN_TABLE_FIXED_FINAL_20261019/gpu0_empty_retirement_20261024/receipt.json'
RECEIPT_SHA = '7e2dc6fd612af5b9e71e59d8dd69bc704450f337d777e16c09906f310a4f04d4'
ART = ROOT / 'expes/artifacts/FCD_A_MATRIX_20261006/MAIN_TABLE_FIXED_FINAL_20261019/hpq_ncase/Public433/lab005/fold2'
RETIRED_PATHS = {str(ART / 'native_q_fast/r1' / suffix) for suffix in
                 ('final.pth', 'safety_slot1/weights/final.pth', 'safety_slot1/optimizer/final.pth')}
RETIRED_PATHS |= {str(ART / 'fine_q_r2/r1' / ('safety_slot1/' + part + '/final.pth'))
                  for part in ('weights', 'optimizer')}

def merge_reservations(current, historical, retired, occupied, registered):
    require(set(retired) == RETIRED_PATHS, 'Unapproved retirement set')
    require(not set(retired).intersection(occupied), 'Retired target or partial occupied')
    require(not set(retired).intersection(registered), 'Retired target re-registered; investigate')
    merged = dict(current)
    for p, b in historical.items():
        merged[p] = max(merged.get(p, 0), b)
    for p in retired:
        require(p in merged, 'Retired reservation missing from history')
        merged.pop(p)
    return merged

def preserve_history(current, registered):
    receipt = json.loads(bound_small_bytes(RECEIPT, RECEIPT_SHA))
    require(receipt['status'] == 'AUTHORIZED_EMPTY_RESERVATIONS_RETIRED_FILES_UNTOUCHED'
            and receipt['files_deleted'] == 0 and receipt['gpu1_managed'] is False
            and receipt['other_rows_exactly_preserved'] is True
            and receipt['retired_reserved_bytes'] == 10 * 1024**3, 'Wrong retirement proof')
    for p, digest in receipt['terminal_evidence_sha256'].items():
        bound_small_bytes(Path(p), digest)
    history = json.loads(bound_small_bytes(HISTORY, HISTORY_SHA))
    historical = {r['path']: r['bound_bytes'] for fs in history['capacity']['filesystems'] for r in fs['targets']}
    retired = {r['path'] for r in receipt['removed']}
    occupied = set()
    for value in retired:
        for p in (Path(value), Path(value + '.partial')):
            require(not any(v.is_symlink() for v in (p, *p.parents)), 'Retired target symlink')
            if os.path.lexists(p):
                occupied.add(value)
    merged = merge_reservations({str(p): b for p,b in current}, historical, retired, occupied, registered)
    # Additional exact, hash-bound GPU0 retirements. Preserve their historical
    # target unions, then subtract only the independently proven empty sets.
    extras = (
        ('gpu0_native_softq_empty_retirement_20261025',
         'cbf8862c636c8ba3e596e6a9b598e4ddccad693e1d08c1c7319ba1dfe1633777',
         {str(ART / 'native_soft_q/r1' / s) for s in ('final.pth', 'safety_slot1/weights/final.pth', 'safety_slot1/optimizer/final.pth')}),
        ('gpu0_fineq_r1_empty_retirement_20261025',
         '69f75b3ec4262b827b7f4874a55f9ad98c23e533310816b237eae72c66981556',
         {str(ART / 'fine_q/r1' / s) for s in ('final.pth', 'safety_slot0/weights/final.pth', 'safety_slot0/optimizer/final.pth', 'safety_slot1/weights/final.pth', 'safety_slot1/optimizer/final.pth')}),
    )
    all_retired = set(retired)
    for directory, digest, expected in extras:
        path = ROOT / 'expes/work_dir/semi05_gate/MAIN_TABLE_FIXED_FINAL_20261019' / directory / 'receipt.json'
        proof = json.loads(bound_small_bytes(path, digest))
        require(proof['status'] == 'AUTHORIZED_EMPTY_RESERVATIONS_RETIRED_FILES_UNTOUCHED'
                and proof['files_deleted'] == 0 and proof['other_rows_exactly_preserved'] is True
                and {r['path'] for r in proof['removed']} == expected,
                'Additional retirement proof differs')
        for p, h in proof['terminal_evidence_sha256'].items():
            bound_small_bytes(Path(p), h)
        for row in proof['before_guard_targets']:
            merged[row['path']] = max(merged.get(row['path'], 0), row['bound'])
        all_retired |= expected
    # The separate stopped35 arm has a different, explicitly bound receipt.
    # Preserve its entire observed history before excluding only its six empty
    # targets. This is not permission to retire any new/current target.
    stopped = ROOT / 'expes/work_dir/semi05_gate/MAIN_TABLE_FIXED_FINAL_20261019/gpu0_stopped_p10_empty_retirement_20261026/receipt.json'
    proof = json.loads(bound_small_bytes(stopped, '1a9dfe34dde88582e067bf9440be797eaa7610ba180c1b7cbfcffaaa6b7ef66a'))
    prefix = ROOT / 'expes/artifacts/FCD_A_MATRIX_20261006/MAIN_TABLE_FIXED_FINAL_20261019/hpq_percent/Public433/pct010/fold2/fine_q_best/r1'
    expected = {str(prefix / suffix) for suffix in ('best/final.pth', 'evidence_growth/final.pth',
                'safety_slot0/weights/final.pth', 'safety_slot0/optimizer/final.pth',
                'safety_slot1/weights/final.pth', 'safety_slot1/optimizer/final.pth')}
    require(proof['status'] == 'AUTHORIZED_STOPPED_P10_EMPTY_RESERVATIONS_RETIRED'
            and proof['files_deleted'] == 0 and proof['other_rows_exactly_preserved'] is True
            and {row['path'] for row in proof['removed']} == expected, 'Stopped35 retirement proof differs')
    for p, h in proof['terminal_evidence_sha256'].items():
        bound_small_bytes(Path(p), h)
    for row in proof['before_guard_targets']:
        merged[row['path']] = max(merged.get(row['path'], 0), row['bound'])
    all_retired |= expected
    # Latest human rows4/5 authorization: retire only the ended, empty R2 final.
    # No weight removal or broad historical subtraction is permitted.
    proof = json.loads(bound_small_bytes(
        ROOT / 'expes/work_dir/semi05_gate/MAIN_TABLE_FIXED_FINAL_20261019/gpu0_r2_final_empty_retirement_20261026/receipt.json',
        'a4fbf0d7d505942ae979dba779f04544cba11cbf3a371fe2711c6e91c0390318'))
    expected = {str(ART / 'fine_q_r2/r1/final.pth')}
    require(proof['status'] == 'AUTHORIZED_EMPTY_R2_FINAL_RESERVATION_RETIRED'
            and proof['files_deleted'] == 0 and proof['other_rows_exactly_preserved'] is True
            and proof['retired_reserved_bytes'] == 2 * 1024**3
            and {row['path'] for row in proof['removed']} == expected,
            'R2 final empty retirement proof differs')
    for p, h in proof['terminal_evidence_sha256'].items():
        bound_small_bytes(Path(p), h)
    for row in proof['before_guard_targets']:
        merged[row['path']] = max(merged.get(row['path'], 0), row['bound'])
    all_retired |= expected
    for value in all_retired:
        require(value not in registered, 'Retired target registered again')
        for p in (Path(value), Path(value + '.partial')):
            require(not any(v.is_symlink() for v in (p, *p.parents)) and not os.path.lexists(p),
                    'Retired path occupied or symlinked')
        merged.pop(value, None)
    # Successor-only retirement. Old run/source/bridge evidence remain frozen.
    p = ROOT / 'expes/work_dir/semi05_gate/MAIN_TABLE_FIXED_FINAL_20261019/gpu0_deleted_safety_retirement_20261027/receipt.json'
    proof = json.loads(bound_small_bytes(p, 'c75eb02646b9dd3abbc33ad8c78a0bd303ef30b3e80d442269a26caf4c1c8af6'))
    expected = {str(ART / c / 'r1' / slot / part / 'final.pth')
                for c, slots in [('fine_q_r3', ['safety_slot0', 'safety_slot1']),
                                 ('native_q_fast', ['safety_slot0']), ('native_soft_q', ['safety_slot0'])]
                for slot in slots for part in ('weights', 'optimizer')}
    deletion = ROOT / 'expes/work_dir/semi05_gate/MAIN_TABLE_FIXED_FINAL_20261019/gpu0_three_category_cleanup_20261026/receipt.json'
    deleted = json.loads(bound_small_bytes(deletion, 'd406c40ba0f5c3a510dfe19da13aff55473ad62472d26fadc7fe5ba0dba7fa2e'))
    return apply_deleted_retirement(merged, registered, all_retired, expected, proof, deleted)


def apply_deleted_retirement(merged, registered, all_retired, expected, proof, deleted):
    """Preserve the complete historical union, excluding only bound retirements."""
    require(deleted['status'] == 'AUTHORIZED_THREE_CATEGORIES_DELETED_VERIFIED'
            and {r['path'] for r in deleted['deleted'] if r['group'] == 'weights'} == expected,
            'Wrong deletion evidence')
    require(proof['status'] == 'AUTHORIZED_DELETED_GPU0_SAFETY_RESERVATIONS_RETIRED'
            and proof['files_deleted'] == 0 and proof['other_rows_exactly_preserved'] is True
            and proof['retired_reserved_bytes'] == 16 * 1024**3
            and {r['path'] for r in proof['removed']} == expected,
            'Wrong successor retirement evidence')
    for row in proof['before_guard_targets']:
        merged[row['path']] = max(merged.get(row['path'], 0), row['bound'])
    # Earlier retired targets may reappear in earlier historical snapshots.
    for value in all_retired | expected:
        require(value not in registered, 'Retired target registered again')
        for candidate in (Path(value), Path(value + '.partial')):
            require(not any(v.is_symlink() for v in (candidate, *candidate.parents))
                    and not os.path.lexists(candidate), 'Retired target occupied')
        merged.pop(value, None)
    return [(Path(p), b) for p,b in merged.items()]
