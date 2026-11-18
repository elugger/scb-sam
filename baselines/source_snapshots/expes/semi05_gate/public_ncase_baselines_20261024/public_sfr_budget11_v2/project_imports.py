"""Process-local project import prerequisite binding; never imports a model."""
import hashlib
import importlib.util
from pathlib import Path
import sys

EXPECTED = {
    'expes/common/__init__.py': '63f047e35fc8a2b8f1bda566b02615342db86ba8ec5873a208c25bd8e956faee',
    'expes/common/seg_metrics.py': '8f6289463c2ae10d6743d7e74b60b18df77b373e1b91ec972edb4bb97ee567c9',
}


def configure_project_imports(root):
    root = Path(root)
    assert root.is_absolute() and root.is_dir() and not root.is_symlink()
    value = str(root)
    while value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)


def validate_project_imports(root, source_pins):
    root = Path(root)
    assert sys.path[0] == str(root), 'Project root must precede ambient module paths'
    pins = {r['path']: r['sha256'] for r in source_pins}
    for relative, digest in EXPECTED.items():
        path = root/relative
        assert not any(p.is_symlink() for p in [path, *path.parents])
        assert pins.get(str(path)) == digest
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    target = root/'expes/common/seg_metrics.py'
    spec = importlib.util.find_spec('expes.common.seg_metrics')
    assert spec is not None and spec.origin is not None
    assert Path(spec.origin).resolve() == target.resolve(), 'Unexpected metrics module origin'
    for name in ('numpy', 'torch', 'surface_distance'):
        assert importlib.util.find_spec(name) is not None, 'Missing metrics prerequisite: '+name
    return dict(project_root=str(root), metrics_module=str(target), metrics_sha256=EXPECTED['expes/common/seg_metrics.py'],
                package_init_sha256=EXPECTED['expes/common/__init__.py'], module_origin_verified=True,
                dependencies_resolvable=True, new_model_loads=0, new_optimization_updates=0, new_test_evaluations=0)
