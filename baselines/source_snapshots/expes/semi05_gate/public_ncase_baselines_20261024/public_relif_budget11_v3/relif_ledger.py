"""Read-only ReLiF ledger binding and uncommitted-tail accounting.

Connects the existing cutoff receipt schema, original-clock binding and storage
reader. This is NOT a resume approval or reconciler: it never loads a checkpoint,
truncates a ledger, consumes a review, or starts a process. An independent review
must also bind source/input/foundation, test-history absence, dead old PID, idle
GPU1, lease, both current recovery generations and the complete resource gate.
"""
import hashlib
import json

from relif_state import accept, counts, require, validate
from relif_storage import read_pin


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key in ledger')
        result[key] = value
    return result


def scan(raw, binding):
    """Validate complete rows; keep a trailing partial row opaque and separate."""
    require(type(raw) is bytes, 'Ledger must be the same-fd byte snapshot')
    recipe = binding.recipe
    complete_end = raw.rfind(b'\n') + 1
    complete, partial = raw[:complete_end], raw[complete_end:]
    progress, ends = counts(0), [0]
    actual, cycles = [], []
    for line in complete.splitlines(keepends=True):
        require(line.endswith(b'\n') and line.strip(), 'Empty/malformed complete ledger line')
        try:
            row = json.loads(line, object_pairs_hook=_unique_object)
        except (ValueError, UnicodeError) as error:
            raise ValueError('Unknown complete ledger data; preserve without reconciliation') from error
        require(type(row) is dict and set(row) == {'step', 'cumulative'}, 'Unexpected ledger row schema')
        index = progress['iterations']
        pair = binding.pair(index)
        proposed = accept(progress, row['step'], recipe, 'formal', pair, binding.original_clock(index))
        require(validate(row['cumulative'], recipe) == proposed, 'Cumulative ledger counters differ')
        progress = proposed
        actual.append(pair)
        ends.append(ends[-1]+len(line))
        per_cycle = recipe['primary_pairs_per_cycle']
        if progress['iterations'] % per_cycle == 0:
            rows = actual[-per_cycle:]
            split = binding.splits
            require(set(p['labeled'] for p in rows) == set(split['train_lab'])
                    and set(p['unlabeled'] for p in rows) == set(split['train_unlab'])
                    and sum(p['unlabeled_replay'] for p in rows) == per_cycle-recipe['unlabeled_cases'],
                    'Actual completed-cycle coverage differs')
            cycles.append(progress['iterations']//per_cycle-1)
    require(ends[-1] == complete_end, 'Ledger row boundaries differ')
    return dict(counts=progress, line_end_offsets=ends, complete_bytes=complete_end,
                complete_sha256=hashlib.sha256(complete).hexdigest(),
                partial_bytes=len(partial), partial_sha256=hashlib.sha256(partial).hexdigest(),
                actual_train_cases=len({p[k] for p in actual for k in ('labeled', 'unlabeled')}),
                complete_reference_cycles=len(cycles))


def project(raw, current_pin, commit, binding, contract_sha256):
    """Bind an immutable commit to the exact original prefix and cost of its tail.

    The caller must separately prove that commit is one of the current valid
    serialized generations. Merely supplying a dict is not recovery authority.
    A valid return is evidence for a later review, never permission to truncate.
    """
    require(isinstance(contract_sha256, str) and len(contract_sha256) == 64
            and all(x in '0123456789abcdef' for x in contract_sha256), 'Invalid contract identity')
    require(commit.get('contract_sha256') == contract_sha256 and commit.get('execution_role') == 'formal',
            'Checkpoint contract/role differs')
    require(type(commit.get('generation')) is int and commit['generation'] >= 0, 'Invalid generation')
    digest = commit.get('sha256')
    require(isinstance(digest, str) and len(digest) == 64
            and all(x in '0123456789abcdef' for x in digest), 'Missing checkpoint SHA')
    require(current_pin['bytes'] == len(raw) and current_pin['sha256'] == hashlib.sha256(raw).hexdigest(),
            'Ledger bytes and same-fd pin differ')
    result = scan(raw, binding)
    committed = validate(commit['counts'], binding.recipe)
    n = committed['iterations']
    require(n <= result['counts']['iterations'], 'Ledger lost checkpoint-committed rows')
    prefix_bytes = result['line_end_offsets'][n]
    prefix = raw[:prefix_bytes]
    old = commit['ledger']
    require(old['bytes'] == prefix_bytes and old['sha256'] == hashlib.sha256(prefix).hexdigest(),
            'Original committed byte prefix differs')
    if old['inode'] is not None:
        require((old['device'], old['inode']) == (current_pin['device'], current_pin['inode']),
                'Ledger device/inode changed outside reviewed history')
    else:
        require(n == 0 and old['bytes'] == 0 and old['device'] is None,
                'Missing original ledger identity')
    complete_tail = raw[prefix_bytes:result['complete_bytes']]
    full_tail = raw[prefix_bytes:]
    tail_updates = result['counts']['iterations'] - n
    return dict(status='PREFIX_AND_TAIL_PROJECTED_NOT_RESUME_APPROVAL',
        contract_sha256=contract_sha256, generation=commit['generation'], checkpoint_sha256=digest,
        committed_counts=committed, observed_complete_counts=result['counts'],
        original_ledger_pin={k: current_pin[k] for k in ('bytes', 'sha256', 'device', 'inode')},
        original_committed_prefix=dict(old), complete_reference_cycles=result['complete_reference_cycles'],
        actual_train_cases=result['actual_train_cases'],
        complete_tail=dict(bytes=len(complete_tail), sha256=hashlib.sha256(complete_tail).hexdigest(),
                           iterations=tail_updates, successful_optimizer_updates=tail_updates,
                           labeled_inputs=tail_updates, unlabeled_inputs=tail_updates),
        entire_uncommitted_tail=dict(bytes=len(full_tail), sha256=hashlib.sha256(full_tail).hexdigest()),
        partial_tail=dict(bytes=result['partial_bytes'], sha256=result['partial_sha256']),
        failed_or_inflight_compute_fully_measured=False,
        failed_or_inflight_compute_must_not_be_assumed_zero=True,
        ledger_modified=False, checkpoint_deserialized=False, independent_resume_review_passed=False,
        required_before_resume=['verify serialized complete checkpoint and two current generations',
            'verify complete source/input/foundation identity and all preserved failure/stdout/history',
            'verify no final/native test intent anywhere in current or preserved lineage',
            'verify old PID absent, GPU1 idle, lease and full resource gate',
            'preserve complete original ledger and opaque tail before reviewed reconciliation',
            'consume one bound independent review in the new process'])


def project_file(path, job, commit, binding, contract_sha256):
    pin = read_pin(path, job, include_raw=True)
    raw = pin.pop('raw')
    return project(raw, pin, commit, binding, contract_sha256)
