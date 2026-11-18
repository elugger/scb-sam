"""Read-only source/split/clock binding for the unqualified ReLiF adapter.

This does not create a contract, reserve space, import a model, or authorize a
run. The eventual native backend must also bind all patient files, foundation
tensors, resource reservations, lease, recovery qualification and admission.
"""
import ast
import copy
import dataclasses
import hashlib
import json
import math
import os
import random
import stat
from pathlib import Path

from relif_state import plan, require, role_window

BY_CONTRACT_SHA = 'f99f62d687ce1be41b95e116d5b2aad5dfc533a3e446b11ee742137f5f0579a9'
BUDGET_SHA = '9c79216dafe33283c4e49bd64c40e4dbe9c9ff92cff8f0a3b3af02c6c844677b'
OLD_REMOTE = '__RECORDED_PROJECT_ROOT__/expes/semi05_gate/main_table_20261019_r4/'
NAMES = ('budget_protocol.py', 'runtime_data.py', 'relif_runtime.py',
         'relif_components.py', 'relif_native.py', 'relif_backend.py',
         'relif_official_pin.json')


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def pinned_bytes(path, expected, size=None):
    """Hash and parse the same bytes; never import a later reread of the file."""
    path = Path(path)
    require(not path.is_symlink(), 'Symlink source/input rejected')
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size <= 8 * 1024**2,
                'Expected a small regular source/input file')
        data = stream.read()
        after = os.fstat(stream.fileno())
        named = path.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    # Windows path stat and descriptor stat can expose different legacy ctime
    # (creation-time) values. Always compare descriptor ctime before/after;
    # additionally compare path ctime on POSIX, where it is change time.
    require(identity(before) == identity(after) == identity(named)
            and before.st_ctime_ns == after.st_ctime_ns
            and (os.name == 'nt' or after.st_ctime_ns == named.st_ctime_ns)
            and not path.is_symlink(), 'File changed during source/input binding')
    require(len(data) == before.st_size and (size is None or size == len(data)), 'Pinned size differs')
    require(hashlib.sha256(data).hexdigest() == expected, 'Pinned SHA differs: '+path.name)
    return data


def _selected(definitions):
    """Execute only named, SHA-bound stdlib definitions, never native imports."""
    env = dict(__name__=__name__, dataclass=dataclasses.dataclass,
               random=random, math=math, NATIVE_ITERATIONS=30000)
    for name, data, kind, wanted in definitions:
        tree = ast.parse(data, filename=name)
        nodes = [node for node in tree.body if isinstance(node, kind) and node.name == wanted]
        require(len(nodes) == 1, 'Missing/duplicate pinned definition: '+wanted)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), name, 'exec'), env)
    return env


def _split_ids(data):
    rows = data.decode('utf-8').splitlines()
    require(rows and all(x and x == x.strip() and x.startswith('PUBLIC_NATIVE_PREFIX_')
                        and x[13:].isdigit() for x in rows), 'Unsafe/empty case ID')
    require(len(rows) == len(set(rows)), 'Duplicate case ID')
    return rows


class ReLiFBinding:
    """Validated metadata only; return copies so a caller cannot alter the clock."""
    def __init__(self, labels, *, native_dir, by_contract, budget_file, split_dir):
        self._recipe = plan(labels)  # Reject bool, percentage labels and old 17/35.
        contract = json.loads(pinned_bytes(by_contract, BY_CONTRACT_SHA))
        budget = json.loads(pinned_bytes(budget_file, BUDGET_SHA))
        require((contract['method'], contract['dataset'], contract['fold'], contract['seed'])
                == ('relif3d', 'BY', 2, 2027), 'Original scientific contract differs')
        require((budget['dataset'], budget['fold'], budget['seed'], budget['reference_cycles'])
                == ('Public433', 2, 2027, 11), 'Budget identity differs')
        require(budget['labels_are_case_counts_not_percent'] is True
                and budget['performance_early_stop'] is False, 'Budget selection differs')
        mapping = [m for row in budget['rows'] if row['labeled_cases'] == labels
                   for m in row['method_mapping'] if m['method'] == 'relif3d']
        require(len(mapping) == 1, 'Missing/duplicate method budget')
        m = mapping[0]
        require(m['inputs_per_stream_per_iteration'] == 1
                and m['native_ssl_iterations'] == m['actual_primary_inputs_per_stream']
                == m['expected_optimizer_updates'] == self._recipe['ssl_iterations']
                and m['overshoot_per_stream'] == 0, 'Method cutoff differs')

        native, source_pins = {}, {}
        for name in NAMES:
            pins = [r for r in contract['sources'] if r['path'] == OLD_REMOTE+name]
            require(len(pins) == 1, 'Native source binding missing/ambiguous: '+name)
            pin = pins[0]
            native[name] = pinned_bytes(Path(native_dir)/name, pin['sha256'], pin['bytes'])
            source_pins[name] = dict(sha256=pin['sha256'], bytes=pin['bytes'])
        self._native = _selected([
            ('budget_protocol.py', native['budget_protocol.py'], ast.ClassDef, 'Budget'),
            ('runtime_data.py', native['runtime_data.py'], ast.FunctionDef, 'schedule'),
            ('relif_runtime.py', native['relif_runtime.py'], ast.FunctionDef, 'clock')])
        files = {name: _split_ids(pinned_bytes(Path(split_dir)/name, digest))
                 for name, digest in budget['split_pins'].items()}
        whole, test, original = (files[n] for n in ('train_whole.txt', 'test.txt', 'original_lab17.txt'))
        require((len(whole), len(test), len(original)) == (346, 87, 17)
                and not set(whole) & set(test) and set(original) < set(whole), 'Fold boundary differs')
        ranked = sorted(original, key=lambda c: hashlib.sha256(
            ('Public433/fold2/ncase/2027/'+c).encode()).hexdigest())
        for n in (5, 10):
            lab, unlab = files['n%03d_lab.txt'%n], files['n%03d_unlab.txt'%n]
            require(lab == ranked[:n] and unlab == [c for c in whole if c not in lab],
                    'Nested selection/order differs')
        self._split = dict(train_lab=files['n%03d_lab.txt'%labels],
                           train_unlab=files['n%03d_unlab.txt'%labels], test=test)
        # The ratio slot is native API compatibility only; labels are CASE counts.
        self._clock_budget = self._native['Budget']('Public433', labels, labels, 346-labels, cycles=36)
        self._pairs = self._native['schedule'](self._clock_budget, self._split)
        require(len(self._pairs) == self._recipe['original_clock_iterations'], 'Original clock shortened')
        cutoff = self._recipe['ssl_iterations']
        self._evidence = dict(status='SOURCE_SPLIT_CLOCK_BOUND_NOT_GPU_QUALIFIED',
            by_contract_sha256=BY_CONTRACT_SHA, budget_sha256=BUDGET_SHA,
            native_sources=source_pins, split_sha256=budget['split_pins'],
            recipe=self._recipe, original_pair_count=len(self._pairs),
            cutoff_pair_count=cutoff, cutoff_pairs_sha256=stable(self._pairs[:cutoff]),
            frozen_foundation_declared=contract['relif_recipe']['original_sam']['sha256'],
            foundation_tensor_signature_checked=False, patient_files_bound=False,
            resource_admitted=False, gpu_qualified=False)

    @property
    def recipe(self):
        return copy.deepcopy(self._recipe)

    @property
    def splits(self):
        return copy.deepcopy(self._split)

    def evidence(self):
        return copy.deepcopy(self._evidence)

    def require_core_total(self, total):
        require(type(total) is int and total == self._recipe['original_clock_iterations'],
                'ReLiF/SOBF clock must retain the original 36-cycle denominator')
        return total

    def pair(self, index, role='formal'):
        start, stop = role_window(self._recipe, role)
        require(type(index) is int and start <= index < stop, 'Index outside declared role/cutoff')
        return copy.deepcopy(self._pairs[index])

    def original_clock(self, index, role='formal'):
        self.pair(index, role)
        return self._native['clock'](index, self._clock_budget.primary_pairs)

    def executed_prefix(self, position, role='formal'):
        start, stop = role_window(self._recipe, role)
        require(type(position) is int and start <= position <= stop, 'Invalid resume position/role')
        return copy.deepcopy(self._pairs[start:position])
