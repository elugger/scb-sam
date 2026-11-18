"""Accepted-supervised initializer and a restricted frozen-trainer adapter.

This is a component, NOT a launcher or a qualification receipt. The caller
must bind all source hashes; actual split bytes and the joint resource gate
are bound by this adapter's own fixed inputs and resource modules.
No model/data dependency is imported until the class factory is called.
"""
import copy
import hashlib
import inspect
import json
from pathlib import Path

from hpq_contract import (BASE, INIT, INIT_SHA, ROOT, STORE, require, sha_file,
                          validate_checkpoint_metadata, validate_initialization,
                          validate_scientific_config, validate_splits)

ROOT_NAMES = {'student', 'head', 'adapter', 'teacher', 'teacher_head', 'teacher_adapter'}
STUDENT_ROOTS = ('student', 'head', 'adapter')
PAIRS = (('student', 'teacher'), ('head', 'teacher_head'), ('adapter', 'teacher_adapter'))
BLOCKED_METHODS = ('_evaluate', '_save', '_save_ucp', '_loaders', '_labeled_error_audit',
                   '_refresh_pseudo_loader', 'preflight', 'pretrain', 'semi_train',
                   'run', 'run_ucp', 'run_paired')


def tensor_manifest(state):
    """Hash EVERY state entry, including buffers and registered aliases."""
    require(bool(state), 'Empty tensor state')
    result = {}
    for key, value in sorted(state.items()):
        require(isinstance(key, str), 'Invalid state key')
        tensor = value.detach().cpu().contiguous()
        metadata = dict(dtype=str(tensor.dtype), shape=list(tensor.shape))
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
        digest.update(tensor.numpy().tobytes())
        result[key] = dict(**metadata, sha256=digest.hexdigest())
    return result


def require_state(module, expected, name):
    actual = tensor_manifest(module.state_dict())
    require(actual == expected, 'Full tensor identity mismatch: ' + name)
    return len(actual)


def strict_load(module, state, name):
    expected = tensor_manifest(state)
    # Validate keys before loading; then verify values, not just return codes.
    require(set(module.state_dict()) == set(state), 'Strict keys differ: ' + name)
    module.load_state_dict(state, strict=True)
    require_state(module, expected, name)
    return expected


def verify_cold_zero_outputs(roots):
    """Necessary zero-residual condition, NOT an output-equivalence proof."""
    require(set(roots) == ROOT_NAMES, 'Wrong six roots')
    needed = {'head': ('out.weight', 'out.bias', 'gate.weight', 'gate.bias'),
              'adapter': ('up.weight', 'up.bias', 'cross_projection.weight', 'cross_projection.bias')}
    for root, keys in needed.items():
        state = roots[root].state_dict()
        for key in keys:
            require(key in state and bool((state[key] == 0).all()), 'Cold output is nonzero: ' + root + '.' + key)


def verify_initial_state(trainer, expected):
    roots = trainer.roots()
    require(set(roots) == ROOT_NAMES and set(expected) == set(STUDENT_ROOTS), 'Invalid identity roots')
    counts = {}
    for student, teacher in PAIRS:
        counts[student] = require_state(roots[student], expected[student], student)
        counts[teacher] = require_state(roots[teacher], expected[student], teacher)
        require(all(p.grad is None for p in roots[teacher].parameters()), 'Teacher already has gradients')
    verify_cold_zero_outputs(roots)
    require(trainer.matrix_updates == 0, 'Initialization updated training')
    require(not trainer.optimizer.state, 'Optimizer must start without moments')
    groups = trainer.optimizer.param_groups
    require([g['lr'] for g in groups] == [5e-8, 2e-7, 2e-7, 2e-7, 1e-4, 1e-4], 'Optimizer learning rules changed')
    require(all(tuple(g['betas']) == (.9, .999) and g['weight_decay'] == 0 for g in groups),
            'Optimizer betas/decay changed')
    ids = [id(p) for g in groups for p in g['params']]
    wanted = {id(p) for name in STUDENT_ROOTS for p in roots[name].parameters()}
    require(len(ids) == len(set(ids)) and set(ids) == wanted, 'Missing/duplicate/foreign optimizer parameters')
    require(not set(ids) & {id(p) for _, name in PAIRS for p in roots[name].parameters()}, 'Teacher in optimizer')
    scaler = trainer.scaler.state_dict()
    require(scaler == {'scale': 65536., 'growth_factor': 2., 'backoff_factor': .5,
                       'growth_interval': 2000, '_growth_tracker': 0}, 'Not original fresh AMP scaler')
    return dict(status='INITIAL_TENSORS_VERIFIED', tensor_counts=counts,
                root_manifests={name: tensor_manifest(root.state_dict()) for name, root in roots.items()},
                successful_updates=0, test_evaluations=0, native_output_identity_verified=False,
                branch_routing_measured=False, gpu_bridge_verified=False)


def _blocked(self, *args, **kwargs):
    raise RuntimeError('Legacy evaluation/training/save entry disabled; use the new fixed-final engine')


class AcceptedSupervisedMixin:
    """Transport/data/load changes only; scientific methods inherited verbatim."""
    def __init__(self, cfg, *, splits, ancestor, phase):
        from hpq_inputs import read_bound_splits, read_accepted_sup05
        from hpq_resources import initializer_load_guard
        validate_scientific_config(cfg, ancestor)
        validate_splits(splits)
        require(splits == read_bound_splits(), 'Supplied IDs/order differ from pinned actual split files')
        read_accepted_sup05()
        require(phase in ('bridge', 'formal'), 'Invalid phase')
        tag = phase + '_r1'
        require(cfg['experiment']['output_dir'] == str(BASE / tag), 'Foreign output path')
        expected_store = STORE if phase == 'formal' else STORE.parent / tag
        require(cfg['semi']['paired_checkpoint_dir'] == str(expected_store), 'Foreign checkpoint path')
        self.input_splits = copy.deepcopy(splits)
        self._hpq_load_guard = initializer_load_guard
        self._hpq_phase = phase
        # No test data/loader is constructed by the inherited constructor:
        # dynamic dispatch goes to our train-only _build_dataloaders below.
        super().__init__(copy.deepcopy(cfg))
        require(tuple(self._style_donors) == tuple(self.input_splits['train_unlab']), 'Donor list differs')
        require(not hasattr(self, 'test_loader') and not hasattr(self, 'test_set'), 'Legacy test dataset leaked')
        self._hpq_cold = {name: {k: v.detach().cpu().clone() for k, v in self.roots()[name].state_dict().items()}
                          for name in ('head', 'adapter')}
        verify_cold_zero_outputs(self.roots())

    def _build_dataloaders(self):
        from strict_hpq_data import make_dataset
        directory = str(ROOT / 'data/Public_433_Folds/fold_2')
        self.lab_set = make_dataset(directory, self.input_splits['train_lab'], 'train_lab')
        self.unlab_set = make_dataset(directory, self.input_splits['train_unlab'], 'train_unlab')

    def _load(self, path):
        """Called by unchanged initialize_matrix, never by old formal/preflight."""
        require(str(path) == str(INIT), 'Only accepted sup05 final can initialize')
        import torch
        # The production guard holds the original lock and accounts joint disk
        # and all RAM ancestors; it cannot be supplied as a no-op by the caller.
        with self._hpq_load_guard():
            p = Path(path)
            require(p.is_file() and not p.is_symlink() and all(not x.is_symlink() for x in p.parents),
                    'Initializer path is not a plain file')
            before = p.stat()
            require(before.st_nlink == 1, 'Unexpected initializer hardlink')
            from hpq_inputs import read_accepted_sup05
            acceptance = read_accepted_sup05()
            validate_initialization(path, sha_file(p), acceptance)
            kwargs = dict(map_location='cpu')
            if 'weights_only' in inspect.signature(torch.load).parameters:
                kwargs['weights_only'] = False
            payload = torch.load(str(p), **kwargs)
            after = p.stat()
            require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                    (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'Initializer changed while loading')
            validate_checkpoint_metadata(payload)
            roots = self.roots()
            expected = {'student': strict_load(roots['student'], payload['student'], 'student')}
            for name in ('head', 'adapter'):
                expected[name] = strict_load(roots[name], self._hpq_cold[name], name)
            # Full state includes buffers, not just a zip of parameters.
            for student, teacher in PAIRS:
                strict_load(roots[teacher], roots[student].state_dict(), teacher)
            self._hpq_expected = expected
            self._fclr_loaded_head = False
            self._matrix_initial_head_state = self._hpq_cold['head']
            del payload

    def initialize_matrix(self, path):
        super().initialize_matrix(path)
        self.hpq_initialization_receipt = verify_initial_state(self, self._hpq_expected)
        self.hpq_initialization_receipt['initializer_sha256'] = INIT_SHA
        return self.hpq_initialization_receipt


for _name in BLOCKED_METHODS:
    setattr(AcceptedSupervisedMixin, _name, _blocked)


def make_trainer_class():
    from expes.semi05_gate.fcd_public05_trainer_adapter_20261015 import PublicQualifiedStyleBalanced

    class AcceptedSupervisedHPQ(AcceptedSupervisedMixin, PublicQualifiedStyleBalanced):
        pass

    # Fail if adapter changes any of the actual frozen scientific calculations.
    for name in ('step_matrix', '_losses', 'contextual_gradients', 'isolated_context',
                 '_forward_main_logits', '_forward_with_fclr', '_build_ucp_optimizer', '_update_teacher'):
        require(getattr(AcceptedSupervisedHPQ, name) is getattr(PublicQualifiedStyleBalanced, name),
                'Frozen method replaced: ' + name)
    return AcceptedSupervisedHPQ
