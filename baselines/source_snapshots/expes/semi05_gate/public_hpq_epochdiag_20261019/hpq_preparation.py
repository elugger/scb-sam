"""Own source/input preparation; no model construction, data forwards or GPU job."""
import copy
import inspect
import json
from pathlib import Path
import sys
import unittest

from hpq_contract import ROOT, BASE, STORE, INIT_SHA, require, sha_file, validate_scientific_config
from hpq_inputs import bound_small_bytes, plain_file, read_bound_splits, read_accepted_sup05
from hpq_resources import backend, RAM_REQUIRED
from hpq_engine import write_json

SRC = ROOT / 'expes/semi05_gate/public_hpq_epochdiag_20261019'
ANCESTOR_DIR = ROOT / 'expes/work_dir/semi05_gate/FCD_PUBLIC05_A1B_20261015/fold0/A1B/bridge05_s2027_r1'
CONFIG_SHA = 'b1029c1f8375de32df91e9f7ba186fc8948872593effecd0e7cacae131c19bcd'
ANCESTOR_SHA = '287878a8f8e8749aea96d46853e1b7e761473b67f7dfc0943270d6c602b2e70a'
READY = BASE.parent.parent / 'preparation_ready.json'
TEST_MODULES = ('test_hpq_contract', 'test_hpq_pairing', 'test_hpq_initialization',
                'test_hpq_resources_inputs', 'test_hpq_execution', 'test_hpq_diagnostics')


def science_anchor():
    import yaml
    cfg = yaml.safe_load(bound_small_bytes(ANCESTOR_DIR / 'config.yaml', CONFIG_SHA))
    manifest = json.loads(bound_small_bytes(ANCESTOR_DIR / 'manifest.json', ANCESTOR_SHA))
    sources = {}
    for name, digest in manifest['source_hashes'].items():
        path = Path(name)
        if not path.is_absolute():
            path = Path(ROOT) / path
        path = str(path)
        require(path not in sources or sources[path] == digest, 'Conflicting source ancestry')
        sources[path] = digest
    return cfg, sources


def phase_config(phase, gpu, ancestor):
    require(phase in ('bridge', 'formal') and gpu in (0, 1), 'Invalid phase/GPU')
    cfg = copy.deepcopy(ancestor)
    cfg['dataset']['fold'] = 2
    cfg['experiment'].update(gpu=str(gpu), output_dir=str(BASE / (phase + '_r1')))
    cfg['semi']['paired_checkpoint_dir'] = str(STORE if phase == 'formal' else STORE.parent / 'bridge_r1')
    validate_scientific_config(cfg, ancestor)
    return cfg


def verify_sources(sources):
    for name, digest in sources.items():
        path = Path(name)
        require(path.is_absolute() and Path(ROOT) in path.parents and path.suffix == '.py', 'Invalid source path')
        plain_file(path)
        require(sha_file(path) == digest, 'Source SHA changed: ' + name)


def imported_project_sources():
    sources = {}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if not name:
            continue
        path = Path(name).resolve()
        if Path(ROOT) in path.parents and path.suffix == '.py':
            plain_file(path)
            sources[str(path)] = sha_file(path)
    return sources


def validate_preparation(expected_sha):
    ready = json.loads(bound_small_bytes(READY, expected_sha))
    require(ready['status'] == 'CPU_PREPARED_GPU_BRIDGE_REQUIRED' and ready['initializer_sha256'] == INIT_SHA,
            'Wrong preparation identity')
    verify_sources(ready['source_hashes'])
    require(ready['splits'] == read_bound_splits(), 'Pinned input order changed')
    read_accepted_sup05()
    ancestor, old = science_anchor()
    require(ready['ancestor_config_sha256'] == CONFIG_SHA and ready['ancestor_manifest_sha256'] == ANCESTOR_SHA,
            'Science ancestry changed')
    require(all(ready['source_hashes'].get(k) == v for k, v in old.items()), 'Missing scientific source')
    return ready, ancestor


def prepare(bundle_sha):
    require(Path(__file__).resolve() == Path(SRC) / 'hpq_preparation.py', 'Only registered remote source')
    bundle = json.loads(bound_small_bytes(SRC / 'source_bundle.json', bundle_sha))
    require(bundle['schema'] == 'PUBLIC5_HPQ_SOURCE_BUNDLE_V1', 'Wrong source bundle')
    own = {str(SRC / name): digest for name, digest in bundle['sources'].items()}
    require(all(Path(name).name == name and name.endswith('.py') for name in bundle['sources']), 'Unsafe bundle names')
    verify_sources(own)
    require(set(p.name for p in Path(SRC).glob('*.py')) == set(bundle['sources']), 'Unexpected/missing deployed source')
    require(not Path(READY).exists(), 'Preparation already exists; do not repeat')
    for phase in ('bridge', 'formal'):
        require(not (Path(BASE) / (phase + '_r1')).exists(), 'Existing phase evidence')
    ancestor, sources = science_anchor()
    verify_sources(sources)
    from hpq_initialization import make_trainer_class
    trainer_class = make_trainer_class()
    # Import actual helpers used later, without constructing datasets/models.
    from expes.common import seg_metrics
    from expes.semi05_gate.fcd_deterministic_losses_r2_20261005 import deterministic_training
    for cls in trainer_class.__mro__:
        if cls.__module__.startswith('expes.'):
            path = str(Path(inspect.getfile(cls)).resolve())
            require(sources.get(path) == sha_file(path), 'Unbound scientific MRO: ' + path)
    for path, digest in imported_project_sources().items():
        require(path not in sources or sources[path] == digest, 'Imported science source changed')
        sources[path] = digest
    sources.update(own)
    tests = unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES)
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    require(result.wasSuccessful() and result.testsRun >= 90, 'Own CPU tests not fully passed')
    splits = read_bound_splits()
    read_accepted_sup05()
    guard = backend()
    with guard.original_lock():
        capacity = guard.capacity_snapshot(RAM_REQUIRED)
        Path(READY).parent.mkdir(parents=True, exist_ok=True)
        value = dict(status='CPU_PREPARED_GPU_BRIDGE_REQUIRED', source_hashes=sources,
                     source_bundle_sha256=bundle_sha, splits=splits, initializer_sha256=INIT_SHA,
                     ancestor_config_sha256=CONFIG_SHA, ancestor_manifest_sha256=ANCESTOR_SHA,
                     cpu_tests=result.testsRun, capacity=capacity, new_test_forwards=0,
                     gpu_bridge_verified=False, formal_started=False)
        write_json(READY, value)
    return dict(path=str(READY), sha256=sha_file(READY), source_count=len(sources), cpu_tests=result.testsRun)
