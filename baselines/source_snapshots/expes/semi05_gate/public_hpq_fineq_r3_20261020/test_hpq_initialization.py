"""Synthetic CPU tests. No torch, real weights, images or GPU proof."""
import copy
import struct
import sys
import types
import unittest
from unittest.mock import patch

import hpq_initialization as init


class Tensor:
    def __init__(self, value=0, dtype='torch.float32', shape=(1,)):
        self.value, self.dtype, self.shape, self.grad = value, dtype, shape, None
    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def numpy(self): return self
    def tobytes(self): return struct.pack('d', self.value)
    def __eq__(self, other): return types.SimpleNamespace(all=lambda: self.value == other)


class Module:
    def __init__(self, keys):
        self.state = {key: Tensor() for key in keys}
    def state_dict(self): return self.state
    def parameters(self): return list(self.state.values())
    def load_state_dict(self, state, strict):
        assert strict
        self.state = copy.deepcopy(state)


def fixture():
    keys = {'student': ['encoder', 'prompt', 'mask', 'coarse'],
            'head': ['out.weight', 'out.bias', 'gate.weight', 'gate.bias'],
            'adapter': ['up.weight', 'up.bias', 'cross_projection.weight', 'cross_projection.bias']}
    roots = {name: Module(value) for name, value in keys.items()}
    for student, teacher in init.PAIRS:
        roots[teacher] = copy.deepcopy(roots[student])
    params = roots['student'].parameters()
    groups = [dict(params=[p], lr=lr, betas=(.9, .999), weight_decay=0)
              for p, lr in zip(params, (5e-8, 2e-7, 2e-7, 2e-7))]
    groups += [dict(params=roots[name].parameters(), lr=1e-4, betas=(.9, .999), weight_decay=0)
               for name in ('head', 'adapter')]
    scaler_state = dict(scale=65536., growth_factor=2., backoff_factor=.5, growth_interval=2000, _growth_tracker=0)
    trainer = types.SimpleNamespace(roots=lambda: roots, matrix_updates=0,
        optimizer=types.SimpleNamespace(state={}, param_groups=groups),
        scaler=types.SimpleNamespace(state_dict=lambda: scaler_state))
    expected = {name: init.tensor_manifest(roots[name].state_dict()) for name in init.STUDENT_ROOTS}
    return trainer, expected, roots, scaler_state


class InitializationTests(unittest.TestCase):
    def test_manifest_tracks_shape_dtype_value_and_alias_names(self):
        a = Tensor(2)
        original = init.tensor_manifest({'a': a, 'alias': a})
        self.assertEqual(set(original), {'a', 'alias'})
        for changed in (Tensor(3), Tensor(2, 'torch.float64'), Tensor(2, shape=(1, 1))):
            self.assertNotEqual(original, init.tensor_manifest({'a': changed, 'alias': a}))

    def test_empty_state_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Empty'):
            init.tensor_manifest({})

    def test_strict_load_checks_all_values(self):
        model = Module(['x'])
        expected = init.strict_load(model, {'x': Tensor(4)}, 'student')
        self.assertEqual(init.require_state(model, expected, 'student'), 1)

    def test_strict_load_rejects_missing_and_extra_keys(self):
        for state in ({'y': Tensor()}, {'x': Tensor(), 'y': Tensor()}):
            with self.assertRaisesRegex(ValueError, 'Strict keys'):
                init.strict_load(Module(['x']), state, 'student')

    def test_lying_load_return_does_not_pass(self):
        model = Module(['x'])
        model.load_state_dict = lambda *args, **kwargs: None
        with self.assertRaisesRegex(ValueError, 'Full tensor identity'):
            init.strict_load(model, {'x': Tensor(4)}, 'student')

    def test_initial_success_is_not_bridge_or_output_proof(self):
        trainer, expected, _, _ = fixture()
        result = init.verify_initial_state(trainer, expected)
        self.assertFalse(result['gpu_bridge_verified'])
        self.assertFalse(result['native_output_identity_verified'])
        self.assertEqual(result['test_evaluations'], 0)
        self.assertEqual(set(result['tensor_counts']), init.ROOT_NAMES)

    def test_every_root_mismatch_rejected(self):
        for name in init.ROOT_NAMES:
            trainer, expected, roots, _ = fixture()
            next(iter(roots[name].state.values())).value = 1
            with self.assertRaises(ValueError, msg=name):
                init.verify_initial_state(trainer, expected)

    def test_missing_teacher_buffer_rejected(self):
        trainer, expected, roots, _ = fixture()
        roots['teacher'].state.pop('coarse')
        with self.assertRaisesRegex(ValueError, 'teacher'):
            init.verify_initial_state(trainer, expected)

    def test_nonzero_cold_head_even_if_manifests_agree_rejected(self):
        trainer, expected, roots, _ = fixture()
        for name in ('head', 'teacher_head'):
            roots[name].state['out.weight'].value = 1
        expected['head'] = init.tensor_manifest(roots['head'].state)
        with self.assertRaisesRegex(ValueError, 'Cold output'):
            init.verify_initial_state(trainer, expected)

    def test_teacher_gradient_rejected(self):
        trainer, expected, roots, _ = fixture()
        roots['teacher'].parameters()[0].grad = Tensor()
        with self.assertRaisesRegex(ValueError, 'Teacher already'):
            init.verify_initial_state(trainer, expected)

    def test_duplicate_parameter_rejected(self):
        trainer, expected, _, _ = fixture()
        groups = trainer.optimizer.param_groups
        groups[0]['params'].append(groups[1]['params'][0])
        with self.assertRaisesRegex(ValueError, 'optimizer parameters'):
            init.verify_initial_state(trainer, expected)

    def test_teacher_optimizer_parameter_rejected(self):
        trainer, expected, roots, _ = fixture()
        trainer.optimizer.param_groups[0]['params'] = [roots['teacher'].parameters()[0]]
        with self.assertRaises(ValueError):
            init.verify_initial_state(trainer, expected)

    def test_optimizer_moments_and_updates_rejected(self):
        for field in ('moments', 'updates'):
            trainer, expected, _, _ = fixture()
            if field == 'moments': trainer.optimizer.state['x'] = 1
            else: trainer.matrix_updates = 1
            with self.assertRaises(ValueError):
                init.verify_initial_state(trainer, expected)

    def test_changed_learning_rule_rejected(self):
        for key, value in (('lr', 1e-4), ('weight_decay', .1), ('betas', (.8, .999))):
            trainer, expected, _, _ = fixture()
            trainer.optimizer.param_groups[0][key] = value
            with self.assertRaises(ValueError):
                init.verify_initial_state(trainer, expected)

    def test_recovery_scaler_not_inherited(self):
        trainer, expected, _, scaler = fixture()
        scaler['scale'] = 1024
        with self.assertRaisesRegex(ValueError, 'AMP'):
            init.verify_initial_state(trainer, expected)

    def test_all_legacy_paths_fail_closed(self):
        adapter = object.__new__(init.AcceptedSupervisedMixin)
        for name in init.BLOCKED_METHODS:
            with self.assertRaisesRegex(RuntimeError, 'Legacy', msg=name):
                getattr(adapter, name)()

    def test_wrong_initializer_rejected_before_import_or_io(self):
        adapter = object.__new__(init.AcceptedSupervisedMixin)
        for path in ('sup100/final.pth', 'old/test_best.pth', 'sup10/final.pth'):
            with self.assertRaisesRegex(ValueError, 'Only accepted'):
                adapter._load(path)

    def test_constructor_loader_only_requests_training_roles(self):
        adapter = object.__new__(init.AcceptedSupervisedMixin)
        adapter.input_splits = {'train_lab': ['lab'], 'train_unlab': ['unlab'], 'test': ['test']}
        with patch('strict_hpq_data.make_dataset', return_value='mock_dataset') as factory:
            adapter._build_dataloaders()
        self.assertEqual([call.args[2] for call in factory.call_args_list], ['train_lab', 'train_unlab'])
        self.assertFalse(hasattr(adapter, 'test_set'))
        self.assertFalse(hasattr(adapter, 'test_loader'))

    def test_labeled_transform_keeps_original_flip(self):
        from strict_hpq_data import role_transform
        from unittest.mock import Mock
        train, evaluation = Mock(return_value='train'), Mock(return_value='eval')
        self.assertEqual(role_transform('train_lab', train, evaluation), 'train')
        train.assert_called_once_with(128, random_flip=True)
        evaluation.assert_not_called()

    def test_unlabeled_and_test_transforms_never_add_flip(self):
        from strict_hpq_data import role_transform
        from unittest.mock import Mock
        for role in ('train_unlab', 'test'):
            train, evaluation = Mock(return_value='train'), Mock(return_value='eval')
            self.assertEqual(role_transform(role, train, evaluation), 'eval')
            evaluation.assert_called_once_with(128)
            train.assert_not_called()

    def test_unknown_transform_role_rejected(self):
        from strict_hpq_data import role_transform
        with self.assertRaises(ValueError):
            role_transform('other', None, None)

    def test_factory_inherits_scientific_methods_without_overrides(self):
        methods = ('step_matrix', '_losses', 'contextual_gradients', 'isolated_context',
                   '_forward_main_logits', '_forward_with_fclr', '_build_ucp_optimizer', '_update_teacher')
        frozen = type('FakeFrozen', (), {name: lambda self: None for name in methods})
        name = 'hpq_trainer_adapter'
        module = types.ModuleType(name)
        module.PublicQualifiedStyleBalanced = frozen
        with patch.dict(sys.modules, {name: module}):
            adapter = init.make_trainer_class()
        for name in set(methods) - {'contextual_gradients'}:
            self.assertIs(getattr(adapter, name), getattr(frozen, name))
        from fine_q import FineQMixin
        self.assertIs(adapter.contextual_gradients, FineQMixin.contextual_gradients)


if __name__ == '__main__':
    unittest.main()
