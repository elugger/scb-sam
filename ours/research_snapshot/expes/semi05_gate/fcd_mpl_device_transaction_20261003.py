import enum
import random
from collections import defaultdict
import numpy as np
import torch

class DeviceTree:

    def __init__(self, parameters):
        self.parameters = {id(p) for p in parameters}
        self.memo, self.active = ({}, set())
        self.bytes = 0

    def capture(self, value):
        if isinstance(value, torch.nn.Parameter):
            if id(value) not in self.parameters:
                raise ValueError('unregistered state parameter')
            return ('ref', value)
        if value is None or isinstance(value, (bool, int, float, str, enum.Enum, torch.dtype, torch.device)):
            return ('ref', value)
        identity = id(value)
        if identity in self.active:
            raise ValueError('cyclic replay state')
        if identity in self.memo:
            return self.memo[identity]
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided or value.device.type not in ('cpu', 'cuda'):
                raise ValueError('unsupported replay tensor')
            saved = value.detach().cpu().clone()
            self.bytes += saved.numel() * saved.element_size()
            node = ('tensor', value, saved, value.device, value.requires_grad)
        elif type(value) in (dict, defaultdict, list, tuple):
            self.active.add(identity)
            try:
                if isinstance(value, dict):
                    pairs = []
                    for key, child in value.items():
                        if not isinstance(key, (str, int, torch.device, torch.nn.Parameter)):
                            raise TypeError('unsupported state key')
                        if isinstance(key, torch.nn.Parameter) and id(key) not in self.parameters:
                            raise ValueError('unregistered optimizer key')
                        pairs.append((key, self.capture(child)))
                    node = ('dict', value, pairs, value.default_factory if isinstance(value, defaultdict) else None)
                else:
                    node = ('seq', value, [self.capture(v) for v in value])
            finally:
                self.active.remove(identity)
        else:
            raise TypeError('undeclared replay object: ' + type(value).__name__)
        self.memo[identity] = node
        return node

    def restore(self, node, restored):
        kind, original = node[:2]
        if kind == 'ref' or id(original) in restored:
            return original
        restored.add(id(original))
        if kind == 'tensor':
            saved, device, requires_grad = node[2:]
            if original.device != device or original.shape != saved.shape or original.dtype != saved.dtype:
                raise RuntimeError('replay tensor metadata changed')
            with torch.no_grad():
                original.copy_(saved)
                original.requires_grad_(requires_grad)
        elif kind == 'dict':
            pairs = [(k, self.restore(child, restored)) for k, child in node[2]]
            original.clear()
            original.update(pairs)
            if isinstance(original, defaultdict):
                original.default_factory = node[3]
        else:
            children = [self.restore(child, restored) for child in node[2]]
            if isinstance(original, list):
                original[:] = children
        return original

    def assert_restored(self, node, current, checked):
        kind, original = node[:2]
        if kind == 'ref':
            if isinstance(original, torch.nn.Parameter):
                assert current is original, 'parameter state identity changed'
            else:
                assert current == original, 'scalar replay state changed'
            return
        assert current is original, 'replay container/tensor identity changed'
        if id(original) in checked:
            return
        checked.add(id(original))
        if kind == 'tensor':
            saved, device, requires_grad = node[2:]
            assert original.device == device and original.requires_grad == requires_grad
            assert_same_bits(original, saved)
        elif kind == 'dict':
            assert list(original.keys()) == [k for k, _ in node[2]], 'replay dictionary bindings changed'
            if isinstance(original, defaultdict):
                assert original.default_factory is node[3]
            for key, child in node[2]:
                self.assert_restored(child, original[key], checked)
        else:
            assert len(original) == len(node[2])
            for child, value in zip(node[2], original):
                self.assert_restored(child, value, checked)

def assert_same_bits(actual, saved_cpu):
    assert actual.shape == saved_cpu.shape and actual.dtype == saved_cpu.dtype
    a = actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    b = saved_cpu.contiguous().reshape(-1).view(torch.uint8)
    assert torch.equal(a, b), 'replay tensor value changed'

class DeviceSnapshot:

    def __init__(self, registry, optimizers=(), scalers=(), caches=()):
        registry.assert_alias_integrity()
        self.registry = registry
        parameters = [r.parameter for r in registry.parameter_records]
        self.tree = DeviceTree(parameters)
        self.parameters = []
        for p in parameters:
            value = p.detach().cpu().clone()
            self.tree.bytes += value.numel() * value.element_size()
            grad = None if p.grad is None else self.tree.capture(p.grad)
            self.parameters.append((p, value, p.requires_grad, grad))
        self.buffers = [self.tree.capture(r.buffer) for r in registry.buffer_records]
        self.modes = [(r.module, r.module.training) for r in registry.module_records]
        self.hooks = [(r.module, attr, dict(getattr(r.module, attr))) for r in registry.module_records for attr in ('_forward_hooks', '_forward_pre_hooks', '_backward_hooks')]
        self.parameter_hooks = [(p, dict(p._backward_hooks or {})) for p in parameters]
        self.targets = []
        self.absent = []
        for optimizer in optimizers:
            for attr in ('state', 'param_groups'):
                self.targets.append((optimizer, attr, self.tree.capture(getattr(optimizer, attr))))
        for scaler in scalers:
            self.targets.append((scaler, None, self.tree.capture(vars(scaler))))
        for owner, attr in caches:
            if not hasattr(owner, attr):
                self.absent.append((owner, attr))
                continue
            self.targets.append((owner, attr, self.tree.capture(getattr(owner, attr))))
        self.python_rng = random.getstate()
        self.numpy_rng = np.random.get_state()
        self.cpu_rng = torch.get_rng_state().clone()
        self.cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []

    def restore(self):
        restored = set()
        with torch.no_grad():
            for p, value, requires_grad, grad in self.parameters:
                p.copy_(value)
                p.requires_grad_(requires_grad)
                p.grad = None if grad is None else self.tree.restore(grad, restored)
            for buffer in self.buffers:
                self.tree.restore(buffer, restored)
        for module, training in self.modes:
            module.training = training
        for owner, attr, node in self.targets:
            value = self.tree.restore(node, restored)
            if attr is not None:
                setattr(owner, attr, value)
        for owner, attr in self.absent:
            if hasattr(owner, attr):
                delattr(owner, attr)
        random.setstate(self.python_rng)
        np.random.set_state(self.numpy_rng)
        torch.set_rng_state(self.cpu_rng)
        if self.cuda_rng:
            torch.cuda.set_rng_state_all(self.cuda_rng)
        self.registry.assert_alias_integrity()

    def assert_restored(self):
        self.registry.assert_alias_integrity()
        checked = set()
        for p, value, requires_grad, grad in self.parameters:
            assert p.requires_grad == requires_grad
            assert_same_bits(p, value)
            if grad is None:
                assert p.grad is None
            else:
                self.tree.assert_restored(grad, p.grad, checked)
        for buffer in self.buffers:
            self.tree.assert_restored(buffer, buffer[1], checked)
        for module, training in self.modes:
            assert module.training == training
        for owner, attr, node in self.targets:
            current = vars(owner) if attr is None else getattr(owner, attr)
            self.tree.assert_restored(node, current, checked)
        assert all((not hasattr(owner, attr) for owner, attr in self.absent))
        for module, attr, saved in self.hooks:
            actual = getattr(module, attr)
            assert list(actual) == list(saved) and all((actual[k] is v for k, v in saved.items()))
        for p, saved in self.parameter_hooks:
            actual = p._backward_hooks or {}
            assert list(actual) == list(saved) and all((actual[k] is v for k, v in saved.items()))
        assert random.getstate() == self.python_rng
        now = np.random.get_state()
        assert now[0] == self.numpy_rng[0] and np.array_equal(now[1], self.numpy_rng[1]) and (now[2:] == self.numpy_rng[2:])
        assert torch.equal(torch.get_rng_state(), self.cpu_rng)
        if self.cuda_rng:
            assert all((torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), self.cuda_rng)))
