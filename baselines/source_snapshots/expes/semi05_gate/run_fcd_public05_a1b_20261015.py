"""Independent Public-433 5% five-fold / seed2027 original A1B development execution.

Only this new entry changes the ratio, matching historical init, budget and runtime paths. Frozen helpers are
imported directly. Bridge checks create no permanent qualification checkpoint;
formal update zero must pass an actual best.pth disk roundtrip before training.
Import and --help are stdlib-only. No SSH, queue, resume or automatic retry.
"""
import argparse
import contextlib
import copy
import hashlib
import importlib
import inspect
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import time
import traceback

ROOT = PurePosixPath('__RECORDED_PROJECT_ROOT__')
BASE = ROOT/'expes/work_dir/semi05_gate/FCD_PUBLIC05_A1B_20261015'
OLD_BASE = ROOT/'expes/work_dir/semi05_gate/FCD_A_MATRIX_20261006'
STORE_ROOT = ROOT/'expes/artifacts/FCD_A_MATRIX_20261006'
GROUPS = ('A1B',)
TAGS = {'bridge': 'bridge05_s2027_r1', 'formal': 'formal05_s2027_r1'}
SUCCESS_COUNTS = {'A00': 7, 'A0B': 9, 'A1B': 9}
CHECKPOINT_BOUND = 1624671895
ROOT_NAMES = {'student', 'head', 'adapter', 'teacher', 'teacher_head', 'teacher_adapter'}
ROUNDTRIP_RAM_REQUIRED = 3*CHECKPOINT_BOUND+1024**3
PROTOCOL = 'PUBLIC05_A1B_HISTORICAL_DEVELOPMENT_V1'
PURPOSE = 'historical_test_selected_Public05_original_A1B_development'
MANIFEST_SCHEMA = 'FCD_PUBLIC05_A1B_MANIFEST_V1'
BRIDGE_SCHEMA = 'FCD_PUBLIC05_A1B_BRIDGE_V1'
STARTUP_SCHEMA = 'FCD_PUBLIC05_A1B_STARTUP_DISK_GATE_V1'
BRIDGE_STATUS = 'BRIDGE_PASS_FORMAL_DISK_GATE_PENDING'
STARTUP_STATUS = 'PASS_UPDATE0_DISK_ROUNDTRIP'
INITIAL_BEST_LIFETIME = 'startup event only; normal best saves may replace this file; not a permanent proof dependency'
OLD_INIT_FROM = 'expes/artifacts/selftrain-insam/ckpt/Semi-05/fold2.pth'
OLD_INIT_SHA = 'f7945881792237780dbdf8c3cb6e223755e0c7d360e54560d398b3a144c3bd1f'
OLD_FLOOR = .6871769710500747
OLD_SPLIT_HASHES = {
    'train_lab.txt': '829c22473199cff9a938a831357d7217ba3c3cf1979cb642358f2536828b5ab1',
    'train_unlab.txt': '915cd2a00e6bbecafb1af4fe4a30e983759558f940a43fd8a5129ded8b98a45d',
    'test.txt': '60ed81fb7520f795f52d62c9e3c83235969671453c810b8f9b22defb02550eda',
}

from expes.semi05_gate import fcd_public05_contract_20261015 as contract
WEIGHT_SCHEMA = 'FCD_PUBLIC05_A1B_WEIGHT_V1'
# Every fold binds its own real init/splits through the reviewed input registry.
HELPERS = {
    'A00': ('paired_fcd_matrix_20261006', 'MatrixHalf'),
    'A0B': ('paired_fcd_a0b_20261007', 'QualifiedBalancedHalf'),
    'A1B': ('fcd_public05_trainer_adapter_20261015', 'PublicQualifiedStyleBalanced'),
}
FROZEN = {
    'paired_fcd_matrix_20261006.py': '0154d1af46d05bbb85d42ed82e736b30f56af513a23f5864619eb5f79fee31c0',
    'paired_fcd_a0b_20261007.py': '01367d02f4864d81aebc7d6083a7fe6d3aff2955b3601375fbd105a2d3fa11a8',
    'paired_fcd_a1b_20261009.py': '877648a845acb362e9880c921ffaaabf18ea33db539f030906a9955b53f186b5',
    'fcd_matrix_storage_gate_20261006.py': 'd3815bb17d1d29c4b38bc1eed658fee67cf1918aa7b29e4a12c26a8f9d98837d',
    'fcd_hard_spatial_relations_20261004.py': '07f79d1858dd75e46043a79ac735c0c8a717e9c34782d16f2626c726db622076',
}
ANCESTORS = {
    'A00': dict(tag='qualification05_r4', proof_sha256='dc9cbe232abea74623a2382c8d0f37e63e0e0908f9c35ae4d751ac38899fcea8',
                config_sha256='c477e254f56af2a4d77f409fb1325f6d56dc3d174ea11556574b3c2825fd8fab',
                checkpoint_sha256='526c61e9261c05ba86b901e34cd78728062a8851879ba06c7b9ebbd807b852a6', checkpoint_bytes=1623623063),
    'A0B': dict(tag='qualification05_r1', proof_sha256='a4ab7e162aebc76ef7c2a53349af5ca9d79d5837e7e9a12a6e7e66a0e6f7096d',
                config_sha256='e23476ff387673d952dc067dbadcd830496b23bbe54ea2276569a4aafe5b80e1',
                checkpoint_sha256='f7d31a4441dbfce5008aa907fa577cdacb2d3100c48f067e087805f30a390014', checkpoint_bytes=1623623063),
    'A1B': dict(tag='qualification05_r1', proof_sha256='82cd537067503b3e4279b5c831a980fefaed9a7ff6c86822cff4a53f822e8d27',
                config_sha256='aad473f4e7e8d5377ec66e13bd7078e473ee9e505ba17444d556c984c372c8e0',
                checkpoint_sha256='fe908b36970dab091c7a47c951d5d3b51b6049b506c48cc1be821bdad76aded6', checkpoint_bytes=1623623127),
}
BRIDGE_GATES = ('cold_warm_legacy_exact', 'repeat_exact', 'allowed_train_lab_probe_state_unchanged',
                'allowed_train_lab_probe_then_step_exact', 'overflow_skip', 'six_root_roundtrip_exact',
                'teacher_no_grad', 'six_groups_positive_finite', 'clip_ema_exact')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_sources(sources):
    require(isinstance(sources, dict) and sources and all(isinstance(k, str) and _is_sha(v) for k, v in sources.items()), 'Invalid source hashes')
    for name, digest in FROZEN.items():
        require(sources.get(str(ROOT/'expes/semi05_gate'/name)) == digest, 'Frozen source missing/changed: '+name)
    for name in contract.ENTRY_SOURCES:
        require(_is_sha(sources.get(str(ROOT/'expes/semi05_gate'/name))), 'New registered source missing: '+name)
    return True


def validate_scope(group='A1B', phase='formal', gpu='0', ratio='05', fold=0, seed=2027, tag=None):
    value = contract.validate_scope(group, ratio, fold, seed, gpu, phase)
    require(tag is None or tag == value['tag'], 'No retag or retry')
    return value


def build_spec(group='A1B', gpu='0', phase='formal', fold=0, root=ROOT):
    spec = validate_scope(group, phase, gpu, fold=fold)
    root = PurePosixPath(str(root).replace('\\', '/'))
    base = root/'expes/work_dir/semi05_gate/FCD_PUBLIC05_A1B_20261015'/('fold'+str(fold))
    store_root = root/'expes/artifacts/FCD_A_MATRIX_20261006'
    output = base/group/spec['tag']
    store = (store_root/'PUBLIC05_A1B_20261015'/('fold'+str(fold))/group/spec['tag']
             if phase == 'formal' else output/'runtime_checkpoints')
    spec.update(root=str(root), output=str(output), store=str(store), log=str(output)+'.log',
                storage_root=str(store_root), save_lock=str(store_root/'save.lock'),
                bridge_report=str(base/group/TAGS['bridge']/'bridge_report.json'),
                startup_gate=str(base/group/TAGS['formal']/'startup_disk_gate.json'))
    return spec

def capacity_targets(root=STORE_ROOT):
    return contract.capacity_targets(root)


def capacity_bytes(present_bytes):
    return contract.capacity_bytes(list(present_bytes))


def assert_fresh(spec):
    """Read-only. Empty/failed outputs, links and partials are not retryable."""
    for key in ('output', 'store', 'log'):
        path = Path(spec[key])
        require(not os.path.lexists(path), 'Refuse existing '+key+': '+str(path))
        for ancestor in path.parents:
            require(not ancestor.is_symlink(), 'Symlink ancestor: '+str(ancestor))
    for target in capacity_targets(spec['storage_root']):
        path = Path(target['path']+'.partial')
        require(not os.path.lexists(path), 'Existing atomic partial: '+str(path))
    return True


def semantic_config(cfg):
    value = copy.deepcopy(cfg)
    for key in ('gpu', 'output_dir'):
        value['experiment'].pop(key, None)
    value['semi'].pop('paired_checkpoint_dir', None)
    return value


def validate_config_bridge(cfg, ancestor_cfg):
    return contract.validate_config(cfg, ancestor_cfg)


def validate_input_identity(manifest, spec):
    validate_manifest(manifest, spec)
    inputs = contract.input_spec(spec['fold'], contract.load_registry(manifest['input_registry']['sha256']))
    path = _regular(ROOT/inputs['init']['path'])
    require(path.stat().st_size == inputs['init']['bytes'] and sha(path) == inputs['init']['sha256'], 'Init weight changed')
    split = Path(ROOT/contract.SPLIT_ROOT/('fold'+str(spec['fold'])))
    require({k: sha(_regular(split/k)) for k in inputs['split_hashes']} == inputs['split_hashes'], 'Split files changed')
    lists = {k: _regular(split/(k+'.txt')).read_text().splitlines() for k in ('train_lab', 'train_unlab', 'train_whole', 'test')}
    contract.validate_lists(lists, spec['fold'])
    return inputs


def validate_manifest(manifest, spec=None):
    require(manifest.get('schema') == MANIFEST_SCHEMA, 'Not a new low-label manifest')
    actual = validate_scope(manifest['group'], manifest['phase'], manifest['gpu'], manifest['ratio'],
                            manifest['fold'], manifest['seed'], manifest['tag'])
    if spec is not None:
        require(all(manifest[k] == spec[k] for k in actual), 'Manifest scope mismatch')
    registered = build_spec(manifest['group'], manifest['gpu'], manifest['phase'], manifest['fold'])
    require(manifest.get('paths') == registered, 'Noncanonical manifest paths')
    require(manifest.get('purpose') == PURPOSE and manifest.get('clean_confirmation') is False, 'Must disclose development-only init')
    inputs = contract.input_spec(manifest['fold'])
    require(manifest.get('init_from') == inputs['init']['path'] and manifest.get('init_sha256') == inputs['init']['sha256'], 'Wrong initialization')
    require(manifest.get('split_hashes') == inputs['split_hashes'], 'Wrong split')
    require(manifest.get('tail_policy') == contract.TAIL_POLICY and manifest.get('budget') == contract.budget(manifest['fold']), 'Data schedule changed')
    ref = manifest.get('input_registry', {})
    require(ref.get('path') == str(BASE/contract.REGISTRY_NAME) and _is_sha(ref.get('sha256')), 'Missing input registry identity')
    contract.load_registry(ref['sha256'])
    require(manifest.get('protocol') == PROTOCOL and manifest.get('primary_metric') == 'native_student_epoch35_final', 'Protocol changed')
    require(manifest.get('epochs') == 36 and manifest.get('updates') == contract.UPDATES and manifest.get('microbatches') == contract.MICROBATCHES, 'Budget changed')
    require(manifest.get('checkpoint_bound') == CHECKPOINT_BOUND and manifest.get('permanent_qualification_checkpoint') is False, 'Storage contract changed')
    for key in ('config_sha256', 'semantic_config_sha256'):
        require(_is_sha(manifest.get(key)), 'Missing config hash')
    validate_sources(manifest.get('source_hashes'))
    identity = manifest.get('dispatch_identity', {})
    require(set(identity) == {'dispatch_sha256', 'preparation_sha256', 'started_sha256'}
            and all(_is_sha(v) for v in identity.values()), 'Missing protected dispatch identity')
    ready_path = _regular(BASE/contract.PREPARATION_NAME)
    require(sha(ready_path) == identity['preparation_sha256'], 'Reviewed preparation changed')
    ready = json.loads(ready_path.read_text())
    require(manifest.get('save_lock') == ready['save_lock'], 'Manifest original lock identity changed')
    contract.validate_save_lock(manifest['save_lock'])
    ancestor = manifest.get('ancestor', {})
    expected = ANCESTORS[manifest['group']]
    require(ancestor.get('role') == 'engineering_ancestry_only' and ancestor.get('proof_sha256') == expected['proof_sha256'], 'Wrong ancestor proof')
    require(ancestor.get('proof_path') == str(OLD_BASE/manifest['group']/expected['tag']/'qualification.json'), 'Wrong ancestor proof path')
    require(ancestor.get('config_path') == str(OLD_BASE/manifest['group']/expected['tag']/'config.yaml')
            and ancestor.get('config_sha256') == expected['config_sha256'], 'Wrong ancestor config')
    ck = ancestor.get('checkpoint', {})
    require(ck.get('sha256') == expected['checkpoint_sha256'] and ck.get('bytes') == expected['checkpoint_bytes']
            and ck.get('path') == str(STORE_ROOT/manifest['group']/expected['tag']/'qualification.pth'), 'Wrong ancestor weight')
    if manifest['phase'] == 'bridge':
        require(manifest.get('bridge') is None, 'Bridge phase cannot borrow another bridge')
    else:
        bridge = manifest.get('bridge') or {}
        require(bridge.get('path') == registered['bridge_report'] and _is_sha(bridge.get('sha256'))
                and _is_sha(bridge.get('manifest_sha256')), 'Formal needs its own bridge reference')
    return True


def validate_bridge_report(proof, manifest=None):
    require(proof.get('schema') == BRIDGE_SCHEMA and proof.get('status') == BRIDGE_STATUS, 'Old PASS is not a seed2027 bridge')
    validate_scope(proof['group'], 'bridge', proof.get('gpu', '0'), proof['ratio'], proof['fold'], proof['seed'], proof['tag'])
    require(proof.get('purpose') == PURPOSE and proof.get('clean_confirmation') is False, 'Bridge must remain development-only')
    inputs = contract.input_spec(proof['fold'])
    require(proof.get('init_from') == inputs['init']['path'] and proof.get('init_sha256') == inputs['init']['sha256'] and proof.get('split_hashes') == inputs['split_hashes'], 'Wrong bridge inputs')
    require(_is_sha(proof.get('manifest_sha256')) and _is_sha(proof.get('semantic_config_sha256')), 'Bridge hashes missing')
    validate_sources(proof.get('source_hashes'))
    require(proof.get('successful_updates') == SUCCESS_COUNTS[proof['group']] and proof.get('artificial_overflows') == 1, 'Wrong bridge update budget')
    require(proof.get('new_test_forwards') == 0 and proof.get('permanent_qualification_checkpoint') is False and 'checkpoint' not in proof, 'Bridge must not evaluate test or save qualification weight')
    require(all(proof.get(key) is True for key in BRIDGE_GATES), 'Bridge engineering gate incomplete')
    roundtrip = proof.get('roundtrip', {})
    require(roundtrip.get('storage') == 'BytesIO' and roundtrip.get('six_root_roundtrip_exact') is True
            and roundtrip.get('native_student_teacher_output_exact') is True and roundtrip.get('declared_state_restored') is True
            and roundtrip.get('rng_unchanged') is True and roundtrip.get('optimizer_scaler_unchanged') is True,
            'Missing real memory serialization roundtrip')
    require(_is_sha(roundtrip.get('sha256')) and type(roundtrip.get('bytes')) is int
            and 0 < roundtrip['bytes'] <= CHECKPOINT_BOUND, 'Missing serialized memory identity')
    root_hashes = roundtrip.get('root_sha256', {})
    require(isinstance(root_hashes, dict) and set(root_hashes) == ROOT_NAMES
            and all(_is_sha(value) for value in root_hashes.values()), 'Missing six serialized root identities')
    memory = roundtrip.get('memory', {})
    require(memory.get('passes') is True and memory.get('required_bytes') == ROUNDTRIP_RAM_REQUIRED
            and type(memory.get('available_bytes')) is int and memory['available_bytes'] >= ROUNDTRIP_RAM_REQUIRED,
            'Missing locked RAM gate')
    require(proof.get('formal_disk_gate_pending') is True, 'Bridge cannot stand in for disk gate')
    if proof['group'] != 'A00':
        require(proof.get('new_loss_changes_update') is True, 'Active B must change update')
    if proof['group'] == 'A0B':
        require(proof.get('restore_component_check', {}).get('status') == 'COMPONENTS_EXACT_ACCUMULATION_NOT_CLAIMED', 'Missing restore components')
    if proof['group'] == 'A1B':
        require(proof.get('true_two_branch_routing') is True and proof.get('context_vb_component_exact') is True, 'Missing A1B routes/components')
    if manifest is not None:
        validate_manifest(manifest)
        for key in ('group', 'ratio', 'fold', 'seed', 'source_hashes', 'init_from', 'init_sha256', 'split_hashes', 'semantic_config_sha256', 'ancestor'):
            if key == 'source_hashes' and manifest.get('phase') == 'formal':
                from expes.semi05_gate.launch_fcd_public05_a1b_20261015 import validate_bridge_sources
                validate_bridge_sources(proof[key], manifest[key])
            else:
                require(proof.get(key) == manifest.get(key), 'Bridge identity mismatch: '+key)
    return True


def validate_startup_gate(report, manifest=None):
    require(report.get('schema') == STARTUP_SCHEMA and report.get('status') == STARTUP_STATUS, 'Missing formal disk gate')
    validate_scope(report['group'], 'formal', report.get('gpu', '0'), report['ratio'], report['fold'], report['seed'])
    require(report.get('updates_before') == report.get('updates_after') == 0 and report.get('saved_epoch') == -1, 'Disk gate not before update1')
    require(report.get('new_test_forwards') == 0 and report.get('floor_evaluations') == 1, 'Extra/missing floor')
    require(all(report.get(k) is True for k in ('six_root_roundtrip_exact', 'native_student_teacher_output_exact',
                                               'declared_state_restored', 'rng_unchanged', 'optimizer_scaler_unchanged', 'atomic_save')), 'Disk gate incomplete')
    ck = report.get('checkpoint', {})
    require(ck.get('path') == build_spec(report['group'], report.get('gpu', '0'), fold=report['fold'])['store']+'/best.pth'
            and type(ck.get('bytes')) is int and 0 < ck['bytes'] <= CHECKPOINT_BOUND
            and _is_sha(ck.get('sha256')), 'Missing initial best identity')
    require(_is_sha(report.get('manifest_sha256')) and _is_sha(report.get('bridge_sha256')), 'Missing startup source chain')
    require(report.get('initial_best_lifetime') == INITIAL_BEST_LIFETIME, 'Initial best is event evidence, not a retained proof weight')
    if manifest is not None:
        validate_manifest(manifest)
        for key in ('group', 'ratio', 'fold', 'seed'):
            require(report.get(key) == manifest.get(key), 'Disk gate scope mismatch')
        require(manifest['phase'] == 'formal' and report['bridge_sha256'] == manifest['bridge']['sha256'], 'Startup bridge mismatch')
    return True


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())


def _regular(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Required regular file: '+str(path))
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size > 0, 'Invalid evidence file: '+str(path))
    return path


def verify_ancestor(group):
    expected = ANCESTORS[group]
    path = Path(OLD_BASE/group/expected['tag']/'qualification.json')
    require(sha(_regular(path)) == expected['proof_sha256'], 'Ancestor proof changed')
    proof = json.loads(path.read_text())
    require(proof['status'] == 'PASS' and proof['group'] == group and proof['ratio'] == '05', 'Wrong ancestor')
    require(proof['successful_updates'] == SUCCESS_COUNTS[group] and proof['artificial_overflows'] == 1, 'Incomplete ancestor')
    require(proof['init_from'] == OLD_INIT_FROM and proof['init_sha256'] == OLD_INIT_SHA and proof['split_hashes'] == OLD_SPLIT_HASHES, 'Ancestor identity changed')
    checkpoint = Path(STORE_ROOT/group/expected['tag']/'qualification.pth')
    ck = dict(path=str(checkpoint), bytes=expected['checkpoint_bytes'], sha256=expected['checkpoint_sha256'])
    require(proof['checkpoint'] == ck and _regular(checkpoint).stat().st_size == ck['bytes'] and sha(checkpoint) == ck['sha256'], 'Ancestor checkpoint changed')
    for name, digest in proof['source_hashes'].items():
        require(sha(_regular(name)) == digest, 'Frozen ancestor source changed: '+name)
    config_path = path.parent/'config.yaml'
    require(sha(_regular(config_path)) == expected['config_sha256'], 'Ancestor scientific config changed')
    return dict(role='engineering_ancestry_only', proof_path=str(path), proof_sha256=expected['proof_sha256'],
                config_path=str(config_path), config_sha256=expected['config_sha256'], checkpoint=ck), proof


def _gpu_empty(gpu):
    def query(args):
        return subprocess.run(['nvidia-smi', *args, '--format=csv,noheader,nounits'], check=True,
                              capture_output=True, text=True, timeout=20).stdout.splitlines()
    mapping = dict(tuple(x.strip() for x in line.split(',', 1)) for line in query(['--query-gpu=index,uuid']))
    require(gpu in mapping, 'Selected GPU missing')
    occupied = [line for line in query(['--query-compute-apps=gpu_uuid,pid']) if line.split(',', 1)[0].strip() == mapping[gpu]]
    require(not occupied, 'Selected GPU is occupied; no launch')


def _memory_gate(proc_root=Path('/proc'), cgroup_root=Path('/sys/fs/cgroup')):
    info = dict(line.split(':', 1) for line in (proc_root/'meminfo').read_text().splitlines())
    available = int(info['MemAvailable'].strip().split()[0])*1024
    limits = [available]
    measured = []
    # Read the current process' actual controller and every enclosing limit.
    # Hybrid Ubuntu hosts can have a v2 session path but memory on v1.
    for line in (proc_root/'self/cgroup').read_text().splitlines():
        hierarchy, controllers, relative = line.split(':', 2)
        require(relative.startswith('/') and '..' not in PurePosixPath(relative).parts, 'Invalid cgroup path')
        if 'memory' in controllers.split(','):
            base, names, version = cgroup_root/'memory', ('memory.limit_in_bytes', 'memory.usage_in_bytes'), 'v1'
        elif hierarchy == '0' and controllers == '':
            base, names, version = cgroup_root, ('memory.max', 'memory.current'), 'v2'
            if not (base/relative.lstrip('/')/names[0]).is_file():
                continue  # Hybrid hierarchy without the memory controller.
        else:
            continue
        group = base/relative.lstrip('/')
        while True:
            limit_path, used_path = group/names[0], group/names[1]
            require(limit_path.is_file() and used_path.is_file(), 'Unreadable cgroup memory controller: '+str(group))
            raw_limit = limit_path.read_text().strip()
            used = int(used_path.read_text().strip())
            require(used >= 0, 'Invalid cgroup memory usage')
            limit = None if raw_limit == 'max' and version == 'v2' else int(raw_limit)
            if limit is not None:
                require(limit > 0, 'Invalid cgroup memory limit')
                limits.append(limit-used)
            measured.append(dict(version=version, path=str(group), limit_bytes=limit, usage_bytes=used))
            if group == base:
                break
            group = group.parent
    required = ROUNDTRIP_RAM_REQUIRED
    require(min(limits) >= required, 'Insufficient RAM for strict serialization/tamper/load gate')
    return dict(available_bytes=min(limits), host_available_bytes=available,
                controllers=measured, required_bytes=required, passes=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=GROUPS, required=True)
    parser.add_argument('--phase', choices=tuple(TAGS), required=True)
    parser.add_argument('--gpu', choices=('0', '1'), required=True)
    parser.add_argument('--fold', type=int, choices=range(5), required=True)
    args = parser.parse_args()
    require(__debug__, 'Do not execute with Python -O; frozen helper assertions are required')
    _execute(build_spec(args.group, args.gpu, args.phase, args.fold))


def _execute(spec):
    # This draft cannot execute until the separately reviewed preparation receipt exists.
    ready = json.loads(_regular(Path(BASE)/contract.PREPARATION_NAME).read_text())
    require(contract.readiness(ready.get('checks', {}))['status'] == 'READY_FOR_OWN_BRIDGE', 'Public05 preparation incomplete')
    require(ready.get('runner_sha256') == sha(__file__), 'Preparation does not bind this runner')
    registry = contract.load_registry(ready['input_registry_sha256'])
    inputs = contract.input_spec(spec['fold'], registry)
    from expes.semi05_gate.launch_fcd_public05_a1b_20261015 import require_claim
    dispatch_identity = require_claim(spec)
    # Every scientific dependency is imported only after entering execution.
    os.chdir(str(ROOT))
    os.environ['CUDA_VISIBLE_DEVICES'] = spec['gpu']
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['OMP_NUM_THREADS'] = '2'
    _gpu_empty(spec['gpu'])
    import fcntl
    import torch
    import yaml
    from utils.cascade_train_utils import normalize_batch
    from expes.semi05_gate.fcd_matrix_storage_gate_20261006 import Target, inspect_capacity
    from expes.semi05_gate.fcd_deterministic_losses_r2_20261005 import deterministic_training
    from expes.semi05_gate.paired_rng_data_20260908 import stream_seed, tensor_digest

    group, phase = spec['group'], spec['phase']
    ancestor, old_proof = verify_ancestor(group)
    module, cls = HELPERS[group]
    trainer_class = getattr(importlib.import_module('expes.semi05_gate.'+module), cls)
    old = Path(ROOT/'expes/work_dir/semi05_gate/PAIRED_ZEA_CROSS_SCALE_HALF_R2_fold2/by_semi05_cv5/fold2')
    parent = json.loads((old/'manifest.json').read_text())
    require(parent['init_from'] == OLD_INIT_FROM and parent['init_sha256'] == OLD_INIT_SHA and parent['split_hashes'] == OLD_SPLIT_HASHES, '05 engineering parent input changed')
    require(sha(_regular(ROOT/inputs['init']['path'])) == inputs['init']['sha256'] and _regular(ROOT/inputs['init']['path']).stat().st_size == inputs['init']['bytes'], 'Actual init changed')
    require(json.loads((old/'final_metrics.json').read_text())['floor_dice'] == OLD_FLOOR, '05 engineering parent floor changed')
    cfg = yaml.safe_load((old/'config.yaml').read_text())
    require(cfg['dataset']['fold'] == 2 and cfg['dataset']['split_root'] == 'expes/splits/by_semi05_cv5', 'Wrong dataset scope')
    require(cfg['semi']['ucp_epochs'] == 36 and cfg['semi']['ucp_accumulation'] == 5, 'Wrong frozen budget')
    cfg['dataset'].update(split_root=contract.SPLIT_ROOT, fold=spec['fold'], fold_cv_dir_template=contract.DATA_TEMPLATE)
    split = Path(ROOT/contract.SPLIT_ROOT/('fold'+str(spec['fold'])))
    contract.validate_lists({k: (split/(k+'.txt')).read_text().splitlines() for k in ('train_lab', 'train_unlab', 'train_whole', 'test')}, spec['fold'])
    require({key: sha(_regular(split/key)) for key in inputs['split_hashes']} == inputs['split_hashes'], 'Actual split changed')
    sources = dict(old_proof['source_hashes'])
    for name, digest in parent['source_hashes'].items():
        require(sha(_regular(name)) == digest, 'Parent frozen source changed: '+name)
        sources[name] = digest
    for name, digest in FROZEN.items():
        path = Path(ROOT/'expes/semi05_gate'/name)
        require(sha(_regular(path)) == digest, 'Frozen helper changed: '+name)
        sources[str(path)] = digest
    for name in contract.ENTRY_SOURCES:
        path = Path(ROOT/'expes/semi05_gate'/name)
        sources[str(path)] = sha(_regular(path))
    require(Path(__file__).resolve() == Path(ROOT/'expes/semi05_gate/run_fcd_public05_a1b_20261015.py'), 'Execute the registered deployed source only')
    for base in trainer_class.__mro__:
        if base.__module__.startswith('expes.'):
            path = str(Path(inspect.getfile(base)).resolve())
            digest = sha(_regular(path))
            pinned = sources.get(path, sources.get(str(Path(path).relative_to(str(ROOT)))))
            require(pinned == digest, 'Unregistered/changed MRO source: '+path)
            sources[path] = digest
    output, store = Path(spec['output']), Path(spec['store'])
    cfg['experiment'].update(gpu=spec['gpu'], seed=2027, output_dir=str(output))
    cfg['semi'].update(paired_checkpoint_dir=str(store), paired_workers=0, xsa_cross_gain=.5,
                       matrix_group=group, matrix_c='0' if group == 'A00' else 'B')
    if group == 'A1B':
        cfg['semi']['cross_style_complement'] = 'two-donor-full-decoder'
    validate_config_bridge(cfg, yaml.safe_load(Path(ancestor['config_path']).read_text()))
    config_text = yaml.safe_dump(cfg, sort_keys=False)
    semantic_hash = json_sha(semantic_config(cfg))
    bridge_ref = None
    if phase == 'formal':
        bridge_path = _regular(spec['bridge_report'])
        bridge_manifest_path = _regular(bridge_path.parent/'manifest.json')
        bridge_manifest = json.loads(bridge_manifest_path.read_text())
        validate_manifest(bridge_manifest, build_spec(group, bridge_manifest['gpu'], 'bridge', spec['fold']))
        bridge = json.loads(bridge_path.read_text())
        validate_bridge_report(bridge, bridge_manifest)
        require(bridge['manifest_sha256'] == sha(bridge_manifest_path), 'Bridge manifest changed')
        from expes.semi05_gate.launch_fcd_public05_a1b_20261015 import validate_bridge_sources
        validate_bridge_sources(bridge['source_hashes'], sources)
        require(bridge['semantic_config_sha256'] == semantic_hash and bridge['ancestor'] == ancestor, 'Bridge no longer matches execution')
        require(not (bridge_path.parent/'failure.json').exists(), 'Bridge failure evidence exists')
        bridge_ref = dict(path=str(bridge_path), sha256=sha(bridge_path), manifest_sha256=sha(bridge_manifest_path))
    targets = [Target(**row) for row in capacity_targets()]
    def resource_gate():
        for row in capacity_targets():
            require(not os.path.lexists(row['path']+'.partial'), 'Unresolved atomic partial')
        gate = inspect_capacity(str(STORE_ROOT), targets)
        require(gate['passes'], 'Capacity insufficient: '+json.dumps(gate))
        return gate
    require(Path(STORE_ROOT).is_dir(), 'Original shared storage root must already exist')
    with contract.locked_save_lock(ready['save_lock']) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert_fresh(spec); capacity = resource_gate(); _memory_gate(); _gpu_empty(spec['gpu'])
        output.mkdir(parents=True, exist_ok=False)
        store.mkdir(parents=True, exist_ok=False)
        log = Path(spec['log']).open('x', encoding='utf-8', buffering=1)
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        started = time.monotonic(); actual_successes = 0; trainer = None
        try:
            with (output/'config.yaml').open('x', encoding='utf-8', newline='\n') as handle:
                handle.write(config_text)
            manifest = dict(schema=MANIFEST_SCHEMA, **{k: spec[k] for k in ('group', 'phase', 'gpu', 'ratio', 'fold', 'seed', 'tag')},
                            purpose=PURPOSE, clean_confirmation=False, source_hashes=sources, dispatch_identity=dispatch_identity,
                            init_from=inputs['init']['path'], init_sha256=inputs['init']['sha256'], split_hashes=inputs['split_hashes'],
                            input_registry=dict(path=str(BASE/contract.REGISTRY_NAME), sha256=ready['input_registry_sha256']),
                            save_lock=ready['save_lock'],
                            budget=contract.budget(spec['fold']), tail_policy=contract.TAIL_POLICY,
                            config_sha256=sha(output/'config.yaml'), semantic_config_sha256=semantic_hash,
                            ancestor=ancestor, bridge=bridge_ref, protocol=PROTOCOL, primary_metric='native_student_epoch35_final',
                            epochs=36, updates=contract.UPDATES, microbatches=contract.MICROBATCHES, checkpoint_bound=CHECKPOINT_BOUND,
                            permanent_qualification_checkpoint=False, capacity=capacity, paths=spec,
                            storage_layout='HOME_TEN_FORMAL_TARGETS_SHARED_ORIGINAL_SAVE_LOCK_ONE_ATOMIC',
                            best_role='diagnostic side archive; never changes budget or final selection',
                            data_boundary='No hidden/donor/test GT in new losses; inherited GT ROI, old test-selected Semi and seen test curves remain')
            validate_manifest(manifest, spec); dump(output/'manifest.json', manifest)
            def append(name, row):
                with (output/name).open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row, allow_nan=False)+'\n')
            torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
            with deterministic_training():
                trainer = trainer_class(cfg)
                if group == 'A0B': trainer.qualification_component_checks = phase == 'bridge'
                if group == 'A1B': trainer.qualification_routing = phase == 'bridge'
                trainer.initialize_matrix(inputs['init']['path'])
                require(trainer.pair_seed == 2027 and trainer.matrix_updates == 0, 'Trainer ignored seed/cold init')
                if phase == 'formal': append('initialization_memory.jsonl', trainer.initialization_memory)
                torch.cuda.reset_peak_memory_stats()
                engine = _Engine(torch, trainer, spec, cfg, manifest, output, store, append, resource_gate,
                                 normalize_batch, stream_seed, tensor_digest)
                if phase == 'bridge':
                    engine.bridge()
                else:
                    engine.formal()
        except Exception:
            if 'engine' in locals(): actual_successes = engine.actual_successes
            dump(output/'failure.json', dict(status='FAIL_ENGINEERING', group=group, phase=phase,
                                            successful_qualification_updates=actual_successes,
                                            current_model_updates=getattr(trainer, 'matrix_updates', None),
                                            wall_seconds=time.monotonic()-started, traceback=traceback.format_exc()))
            raise
        finally:
            log.flush()


class _Engine:
    """Execution-only adapter, never a replacement training helper."""
    def __init__(self, torch, trainer, spec, cfg, manifest, output, store, append,
                 resource_gate, normalize_batch, stream_seed, tensor_digest):
        self.torch, self.t, self.spec, self.cfg = torch, trainer, spec, cfg
        self.manifest, self.output, self.store = manifest, output, store
        self.append, self.resource_gate = append, resource_gate
        self.normalize_batch, self.stream_seed, self.tensor_digest = normalize_batch, stream_seed, tensor_digest
        self.group = spec['group']; self.actual_successes = 0; self.begin = time.monotonic()

    def signature(self, value):
        h = hashlib.sha256()
        def walk(x):
            if self.torch.is_tensor(x):
                h.update(str((x.dtype, tuple(x.shape))).encode()); h.update(x.detach().cpu().contiguous().numpy().tobytes())
            elif isinstance(x, dict):
                for key in sorted(x, key=str): h.update(str(key).encode()); walk(x[key])
            elif isinstance(x, (tuple, list)):
                for part in x: walk(part)
            else: h.update(repr(x).encode())
        walk(value); return h.hexdigest()

    def state(self, scaler=True):
        values = [{k: m.state_dict() for k, m in self.t.roots().items()}, self.t.optimizer.state_dict()]
        if scaler: values.append(self.t.scaler.state_dict())
        return self.signature(values)

    def rng(self):
        import random
        import numpy as np
        state = np.random.get_state()
        value = [state[0], state[1].dtype.str, state[1].shape, state[1].tobytes(), *state[2:]]
        return self.signature([random.getstate(), value, self.torch.get_rng_state(), self.torch.cuda.get_rng_state_all()])

    def raw_groups(self, epoch, record=False):
        # Preserve inherited label cycle/seed semantics, but complete exactly one
        # explicit final group on 329-case folds. No dropped or hidden test case.
        from itertools import cycle
        from expes.semi05_gate.paired_rng_data_20260908 import labeled_tokens, paired_loader
        fold = self.spec['fold']; expected = contract.budget(fold)
        require(len(self.t.lab_set) == 17 and len(self.t.unlab_set) == expected['unlab_count'], 'Loaded dataset length changed')
        tokens = contract.epoch_tokens(fold, epoch)
        labs = labeled_tokens(17, 330, epoch, self.t.pair_seed, self.t.refresh)
        loader = paired_loader(self.t.lab_set, labs, self.t.pair_seed, 'lab', self.t.workers)
        lab = iter(loader) if self.t.refresh else cycle(loader)
        unlab = paired_loader(self.t.unlab_set, tokens, self.t.pair_seed, 'unlab', self.t.workers)
        case_ids = [Path(x).name.removesuffix('.nii.gz') for x in self.t.unlab_set.image_paths]
        batch = []
        for step, bu in enumerate(unlab):
            bl = next(lab); xl, yl = self.t._prepare_batch(bl)
            xu = self.normalize_batch(bu['image'], self.t.norm_transform).to(self.t.device)
            token = tokens[step]
            require(int(bu['requested_index']) == token[2], 'Loader ignored requested token')
            require(int(bu['augmentation_seed']) == self.stream_seed(2027, 'unlab', *token), 'Loader ignored replay seed')
            batch.append(dict(epoch=epoch, step=step, image_l=xl, label_l=yl, image_u=xu))
            if record:
                row = dict(epoch=epoch, step=step, labeled_index=int(bl['requested_index']), label_seed=int(bl['augmentation_seed']),
                           unlabeled_index=int(bu['requested_index']), unlabeled_seed=int(bu['augmentation_seed']),
                           student_seed=self.stream_seed(2027, 'student-step', epoch, step),
                           unlabeled_replay=token[1], padding=token[1] == 1,
                           padding_case_id=case_ids[token[2]] if token[1] == 1 else None)
                if epoch == 0: row.update(label_image_sha=self.tensor_digest(bl['image']), unlab_image_sha=self.tensor_digest(bu['image']))
                self.append('pair_trace.jsonl', row)
            if len(batch) == 5: yield batch; batch = []
        require(not batch, 'Incomplete accumulated group')

    def step(self, items, **kwargs):
        # Measure all groups, including helpers whose historical rows omit EMA.
        calls = []; original = self.t._update_teacher; previous = self.t.matrix_updates
        had_instance_ema = '_update_teacher' in self.t.__dict__
        instance_ema = self.t.__dict__.get('_update_teacher')
        def observed(momentum):
            calls.append(momentum); return original(momentum)
        self.t._update_teacher = observed
        try:
            row = self.t.step_matrix(items, **kwargs)
        finally:
            if had_instance_ema: self.t._update_teacher = instance_ema
            else: self.t.__dict__.pop('_update_teacher', None)
            self.actual_successes += max(0, self.t.matrix_updates-previous)
        require(calls == ([] if row['skipped'] else [.999]), 'EMA count/momentum changed')
        if 'ema_calls' in row: require(row['ema_calls'] == len(calls), 'Nested EMA measurement differs')
        row['ema_calls'] = len(calls)
        return row

    def check_step(self, row, context=True, reduction=None):
        reduction = reduction if reduction is not None else ('0' if self.group == 'A00' else 'B')
        require(not row['skipped'] and row['teacher_no_grad'] and row['clip_calls'] == row['ema_calls'] == 1, 'Step/teacher/clip/EMA failure')
        require(len(row['gradient_groups']) == 6 and all(v is not None and math.isfinite(v) and v > 0 for v in row['gradient_groups'].values()), 'Six gradients not positive finite')
        require(row['context_computed'] == context and len(row['microbatches']) == 5, 'Wrong accumulated context')
        if self.group == 'A1B':
            measured = self.spec['phase'] == 'bridge'
            require(row['branch_routing_measured'] is measured, 'Branch probe execution changed')
        for micro in row['microbatches']:
            if self.group == 'A1B' and self.spec['phase'] == 'bridge': require(len(micro['branch_routes']) == 2, 'Missing measured branches')
            if context:
                c = micro['context']
                require(c['reduction'] == reduction and c['coefficient'] == 1. and c['applied'] == (reduction != '0'), 'C rule changed')
                require(c['declared_state_restored'] and c['no_gt_argument'] and not c['relation_computed'], 'C boundary failed')
                require(c['foreground_voxels']+c['background_voxels'] == 64*c['reliable_blocks'], 'C count mismatch')
                if c['reliable_blocks']: require(len(c['raw_gradient_norms']) == 6 and all(math.isfinite(v) and v > 0 for v in c['raw_gradient_norms'].values()), 'C gradient failed')

    def native_probe(self, image):
        before, rng, updates = self.state(), self.rng(), self.t.matrix_updates
        with self.t.isolated_context(0, 0), self.torch.no_grad(), self.torch.cuda.amp.autocast():
            for root in self.t.roots().values(): root.eval()
            values = [self.t._forward_main_logits(m, image).cpu() for m in (self.t.student, self.t.teacher)]
        require(self.state() == before and self.rng() == rng and self.t.matrix_updates == updates, 'Allowed output probe changed state')
        return values

    def payload(self, epoch):
        return dict(roots={k: self.t._compact_state(m) for k, m in self.t.roots().items()}, cfg=self.cfg,
                    saved_epoch=epoch, updates=self.t.matrix_updates, group=self.group, ratio='05', fold=self.spec['fold'], seed=2027,
                    exact_resume_supported=False, schema='FCD_PUBLIC05_A1B_WEIGHT_V1')

    def save(self, name, epoch):
        import fcntl
        require(name in ('best.pth', 'final.pth'), 'No permanent qualification checkpoint')
        with contract.locked_save_lock(self.manifest['save_lock']) as lock:
            fcntl.flock(lock, fcntl.LOCK_EX); self.resource_gate(); _memory_gate()
            path = self.store/name; partial = self.store/(name+'.partial')
            with partial.open('xb') as handle:
                self.torch.save(self.payload(epoch), handle); handle.flush(); os.fsync(handle.fileno())
            require(partial.stat().st_size <= CHECKPOINT_BOUND, 'Checkpoint exceeds bound; keep partial and fail')
            os.replace(partial, path)
            fd = os.open(str(self.store), os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)
            return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))

    def roundtrip(self, image, checkpoint=None):
        # Serialize heavy CPU snapshots one at a time across both GPUs; the
        # actual memory measurement is inside the same original storage lock.
        import fcntl
        with contract.locked_save_lock(self.manifest['save_lock']) as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._roundtrip_locked(image, checkpoint)

    def _roundtrip_locked(self, image, checkpoint=None):
        memory = _memory_gate(); before, rng, updates = self.state(), self.rng(), self.t.matrix_updates
        native = self.native_probe(image); roots_before = {k: self.signature(m.state_dict()) for k, m in self.t.roots().items()}
        require(len(roots_before) == 6, 'Exactly six roots required')
        buffer = None
        if checkpoint is None:
            buffer = io.BytesIO(); self.torch.save(self.payload(-1), buffer)
            size = buffer.getbuffer().nbytes
            require(size <= CHECKPOINT_BOUND, 'Memory archive exceeds bound')
            digest = hashlib.sha256(buffer.getbuffer()).hexdigest(); buffer.seek(0); source = buffer
        else:
            source = checkpoint['path']; size = Path(source).stat().st_size; digest = sha(source)
            require(size == checkpoint['bytes'] and digest == checkpoint['sha256'], 'Disk archive identity changed')
        with self.t.isolated_context(0, 0):
            with self.torch.no_grad():
                for root in self.t.roots().values(): next(root.parameters()).add_(.1)
            require(all(self.signature(m.state_dict()) != roots_before[k] for k, m in self.t.roots().items()), 'Not all six roots were tampered')
            ck = self.torch.load(source, map_location='cpu')
            require(ck['group'] == self.group and ck['seed'] == 2027 and ck['updates'] == updates and ck['saved_epoch'] == -1
                    and ck['cfg'] == self.cfg and set(ck['roots']) == set(self.t.roots()), 'Serialized metadata mismatch')
            for key, root in self.t.roots().items(): root.load_state_dict(ck['roots'][key], strict=True)
            del ck
            require(all(self.signature(m.state_dict()) == roots_before[k] for k, m in self.t.roots().items()), 'Six-root strict load mismatch')
        after = self.native_probe(image)
        require(all(self.torch.equal(x, y) for x, y in zip(native, after)), 'Native student/teacher outputs changed')
        require(self.state() == before and self.rng() == rng and self.t.matrix_updates == updates, 'Roundtrip changed declared state')
        if buffer is not None: buffer.close()
        return dict(storage='BytesIO' if checkpoint is None else 'atomic_disk_best', bytes=size, sha256=digest,
                    six_root_roundtrip_exact=True, native_student_teacher_output_exact=True, declared_state_restored=True,
                    rng_unchanged=True, optimizer_scaler_unchanged=True, root_sha256=roots_before, memory=memory)

    def record(self, row):
        keys = ('step', 'supervised', 'pseudo', 'wsv', 'baseline_total', 'views')
        if self.group == 'A1B': keys += ('native_reliable_ratio', 'style')
        full = copy.deepcopy({key: value for key, value in row.items() if key != 'seconds'})
        # The frozen A0B helper records this diagnostic flag only on its first
        # two active component probes. All scientific fields, gradients and
        # complete model/optimizer/scaler state still compare exactly.
        if self.group == 'A0B':
            for micro in full['microbatches']:
                micro.get('context', {}).pop('old_restore_ce_component_exact', None)
        return dict(state=self.state(), original=[{key: micro[key] for key in keys} for micro in row['microbatches']], full=full)

    def bridge(self):
        iterator = self.raw_groups(0); raw = [next(iterator), next(iterator)]; del iterator
        results = {}; cold_reference = None
        if self.group != 'A1B':
            self.t.initialize_matrix(self.manifest['init_from'])
            cold = self.step(raw[0], compute_context=False); self.check_step(cold, False, '0')
            self.append('cold_cuda_storage_reference.jsonl', cold); cold_reference = self.state()
        branches = [('native', False, '0'), ('off', True, '0')]
        branches += [('off_repeat', True, '0')] if self.group == 'A00' else [('active_B', True, 'B'), ('active_B_repeat', True, 'B')]
        for branch, context, reduction in branches:
            self.t.initialize_matrix(self.manifest['init_from']); self.t.matrix_c = reduction
            if branch.endswith('repeat') and self.group != 'A1B': self.native_probe(raw[0][0]['image_l'])
            records = []; self.append('initialization_memory.jsonl', self.t.initialization_memory)
            for index, items in enumerate(raw):
                if self.group == 'A1B' and branch == 'native':
                    before, rng = self.state(), self.rng()
                    component = self.t.probe_context_reductions(items[0]['image_u'], 0, items[0]['step'])
                    require(self.state() == before and self.rng() == rng, 'Zero-update component probe changed state')
                    require(all(p.grad is None for root in self.t.roots().values() for p in root.parameters()), 'Probe materialized grads')
                    self.append('context_vb_components.jsonl', dict(phase=('cold', 'warm')[index], state_unchanged=True, rng_unchanged=True, **component))
                row = self.step(items, compute_context=context); self.check_step(row, context, reduction)
                self.append('qualification_steps.jsonl', dict(branch=branch, phase=('cold', 'warm')[index], **row))
                record = self.record(row); records.append(record)
                if index == 0 and reduction == '0' and cold_reference is not None:
                    require(record['state'] == cold_reference, 'Cold native storage replay mismatch')
            results[branch] = records
        for index in range(2):
            native, off = results['native'][index], results['off'][index]
            require(native['state'] == off['state'] and native['original'] == off['original'], 'C-off changed frozen loss/update')
            if self.group == 'A00':
                require(off == results['off_repeat'][index], 'C-off repeat or inserted output changed update')
            else:
                active, repeat = results['active_B'][index], results['active_B_repeat'][index]
                require(active == repeat and active['state'] != native['state'], 'B repeat/change gate failed')
        if self.group == 'A1B':
            self.t.initialize_matrix(self.manifest['init_from']); self.t.matrix_c = 'B'; self.native_probe(raw[0][0]['image_l'])
            row = self.step(raw[0]); self.check_step(row)
            self.append('qualification_steps.jsonl', dict(branch='allowed_train_lab_probe_then_step', phase='cold', **row))
            require(self.record(row) == results['active_B'][0], 'A1B probe-then-step changed active cold state')
        require(self.actual_successes == SUCCESS_COUNTS[self.group], 'Wrong successful bridge budget')
        roundtrip = self.roundtrip(raw[0][0]['image_l'])
        before, updates = self.state(scaler=False), self.t.matrix_updates
        overflow = self.step(raw[0], inject_overflow=True)
        require(overflow['skipped'] and overflow['clip_calls'] == overflow['ema_calls'] == 0
                and overflow['scale_after'] == overflow['scale_before']/2 and self.t.matrix_updates == updates
                and self.state(scaler=False) == before, 'Overflow changed optimizer/roots/update')
        self.append('overflow.jsonl', overflow)
        identity = {key: self.manifest[key] for key in ('group', 'gpu', 'ratio', 'fold', 'seed', 'tag', 'purpose', 'clean_confirmation',
                    'source_hashes', 'init_from', 'init_sha256', 'split_hashes', 'semantic_config_sha256', 'ancestor')}
        report = dict(schema=BRIDGE_SCHEMA, status=BRIDGE_STATUS, **identity,
                      manifest_sha256=sha(self.output/'manifest.json'), successful_updates=self.actual_successes, artificial_overflows=1,
                      new_test_forwards=0, permanent_qualification_checkpoint=False, formal_disk_gate_pending=True,
                      roundtrip=roundtrip, **{key: True for key in BRIDGE_GATES},
                      new_loss_changes_update=self.group != 'A00', true_two_branch_routing=self.group == 'A1B',
                      context_vb_component_exact=self.group == 'A1B',
                      wall_seconds=time.monotonic()-self.begin, peak_bytes=self.torch.cuda.max_memory_allocated(),
                      limitation='Finite own-fold Public05 engineering bridge plus old proof ancestry. Not disk gate, 36-epoch replay, exact resume or method effectiveness.')
        if self.group == 'A0B': report['restore_component_check'] = self.t.component_check_report()
        validate_bridge_report(report, self.manifest)
        dump(self.output/'bridge_report.json', report)
        print('BRIDGE_COMPLETE '+json.dumps(report), flush=True)

    def formal(self):
        floor = self.t._evaluate()
        contract.validate_floor(float(floor['dice']), contract.input_spec(self.spec['fold']))
        actual_floor = float(floor['dice'])
        self.t.best_dice, self.t.best_epoch = actual_floor, -1
        self.append('metrics.jsonl', dict(epoch=-1, **floor))
        weights = {'best.pth': self.save('best.pth', -1)}
        # Load one allowed train_lab input inside declared RNG/cache isolation;
        # do not emit duplicate epoch0 data records or alter the real loader.
        before, rng = self.state(), self.rng()
        with self.t.isolated_context(0, 0):
            probe_iterator = self.raw_groups(0, record=False)
            first = next(probe_iterator); image = first[0]['image_l']
            del probe_iterator, first
        require(self.state() == before and self.rng() == rng, 'Disk-probe input acquisition changed state')
        roundtrip = self.roundtrip(image, weights['best.pth']); del image
        report = dict(schema=STARTUP_SCHEMA, status=STARTUP_STATUS,
                      **{key: self.spec[key] for key in ('group', 'gpu', 'ratio', 'fold', 'seed')},
                      manifest_sha256=sha(self.output/'manifest.json'), bridge_sha256=self.manifest['bridge']['sha256'],
                      checkpoint=weights['best.pth'], saved_epoch=-1, updates_before=0, updates_after=self.t.matrix_updates,
                      floor_evaluations=1, floor_dice=actual_floor,
                      historical_floor=contract.input_spec(self.spec['fold']).get('historical_floor'),
                      floor_role='Own actual full-precision initialization; no effect claim', new_test_forwards=0, atomic_save=True,
                      **{key: roundtrip[key] for key in ('six_root_roundtrip_exact', 'native_student_teacher_output_exact',
                             'declared_state_restored', 'rng_unchanged', 'optimizer_scaler_unchanged')},
                      initial_best_lifetime=INITIAL_BEST_LIFETIME)
        validate_startup_gate(report, self.manifest); dump(self.output/'startup_disk_gate.json', report)
        for epoch in range(36):
            started = time.monotonic(); self.t.eval_epoch = epoch; count = 0
            for items in self.raw_groups(epoch, record=True):
                row = self.step(items); self.check_step(row); row['epoch'] = epoch
                self.append('matrix_steps.jsonl', row); count += 1
                print('PUBLIC05_A1B_UPDATE '+json.dumps(dict(group=self.group, epoch=epoch, update=self.t.matrix_updates)), flush=True)
            require(count == contract.UPDATES_PER_EPOCH, 'Wrong per-epoch budget')
            metrics = self.t._evaluate(); self.append('metrics.jsonl', dict(epoch=epoch, **metrics))
            if metrics['dice'] > self.t.best_dice:
                self.t.best_dice, self.t.best_epoch = float(metrics['dice']), epoch
                weights['best.pth'] = self.save('best.pth', epoch)
            row = dict(epoch=epoch, dice=float(metrics['dice']), best=self.t.best_dice,
                       updates=self.t.matrix_updates, seconds=time.monotonic()-started)
            self.append('train_epochs.jsonl', row); print('PUBLIC05_A1B_EPOCH '+json.dumps(row), flush=True)
        require(self.t.matrix_updates == self.actual_successes == contract.UPDATES, 'Wrong formal total')
        weights['final.pth'] = self.save('final.pth', 35)
        final = dict(group=self.group, ratio='05', fold=self.spec['fold'], gpu=self.spec['gpu'], seed=2027, completed_epochs=36, stop_reason='max_epochs',
                     best_dice=self.t.best_dice, best_epoch=self.t.best_epoch, final_dice=float(metrics['dice']),
                     updates=contract.UPDATES, microbatches=contract.MICROBATCHES, floor_dice=actual_floor, tail_policy=contract.TAIL_POLICY, weights=weights,
                     wall_seconds=time.monotonic()-self.begin, peak_bytes=self.torch.cuda.max_memory_allocated(),
                     exact_resume_supported=False, protocol=PROTOCOL, primary_metric='native_student_epoch35_final',
                     startup_gate_sha256=sha(self.output/'startup_disk_gate.json'), purpose=PURPOSE, clean_confirmation=False,
                     best_role='diagnostic only; never reloaded after update0 or used to change budget')
        dump(self.output/'final_metrics.json', final); print('PUBLIC05_A1B_COMPLETE '+json.dumps(final), flush=True)


if __name__ == '__main__':
    main()


