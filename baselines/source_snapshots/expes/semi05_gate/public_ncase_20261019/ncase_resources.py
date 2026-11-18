"""Own final-only checkpoint, sharing the live original lock and target registry.

Registry storage is shared infrastructure; it does not change GPU1 queue or jobs.
No old target/reservation is deleted or repurposed.
"""
import importlib.util
import json
from pathlib import Path
import sys

from ncase_protocol import ROOT, artifact_dir, sha

FINAL_BOUND = 1024**3
RAM_REQUIRED = 7*1024**3 + 16*1024**2
RESOURCE_FILE = ROOT/'expes/semi05_gate/main_table_20261019_r3/runtime_resources.py'
RESOURCE_SHA = 'ecb24e61b93a567eea3d9fb44550f050a7a39f2558d572e3a418f75dc825c34c'


def shared():
    if sha(RESOURCE_FILE) != RESOURCE_SHA:
        raise ValueError('Frozen shared resource implementation differs')
    spec = importlib.util.spec_from_file_location('_ncase_shared_resources', str(RESOURCE_FILE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def target_for(config):
    expected = Path(artifact_dir(config['labeled_cases']))/'final.pth'
    if Path(config['artifact_dir']) != expected.parent:
        raise ValueError('Own final target mismatch')
    return expected


def backend(config):
    return shared().resource_backend(target_for(config), FINAL_BOUND)


def original_lock():
    # The same mutex regardless of own count; no new reservation is enrolled here.
    return shared().resource_backend(Path(artifact_dir(5))/'final.pth', FINAL_BOUND).original_lock()


def capacity_snapshot(*, config):
    return backend(config).capacity_snapshot(RAM_REQUIRED)


def enroll(config):
    return shared().enroll_target(target_for(config), FINAL_BOUND)


def guarded_torch_save(*, payload, target, config, save_kind):
    if Path(target) != target_for(config) or save_kind != 'final':
        raise ValueError('Only own final student save is supported')
    if (payload.get('completed_epochs') != 60 or payload.get('epoch') != 59
            or payload.get('successful_updates') != config['labeled_cases']*60
            or payload.get('test_evaluations_before_save') != 0
            or any(k in payload for k in ('optimizer', 'scaler', 'rng'))):
        raise ValueError('Final budget/format differs')
    return shared().save_terminal(Path(target), payload)


def check_load_capacity(*, target, config):
    if Path(target) != target_for(config) or not Path(target).is_file():
        raise ValueError('Own final absent or wrong path')
    guard = backend(config)
    with guard.original_lock():
        return guard.capacity_snapshot(RAM_REQUIRED)
