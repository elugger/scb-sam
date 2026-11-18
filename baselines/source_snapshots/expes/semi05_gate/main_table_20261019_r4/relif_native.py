"""Read-only, hash-pinned ReLiF native components without its training CLI.

Only import paths are renamed for isolation. SOBF, ProjectionHead and DiceLoss
are compiled from original AST definitions, so importing never parses CLI args,
opens datasets, evaluates test cases or initializes CUDA.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
COMMIT = 'bbda984aa337520c009091d4a02d2e104bb176d6'
DEFINITIONS = ('ProjectionHead', '_gauss_blur3d', '_per_sample_q',
               'sobf_augment', '_lin_sched')


def pinned_sources(code_root):
    root = Path(code_root).resolve(strict=True)
    pin = json.loads((HERE / 'relif_official_pin.json').read_text(encoding='utf-8'))
    if pin['commit'] != COMMIT:
        raise ValueError('ReLiF official commit pin changed')
    for relative, identity in pin['files'].items():
        path = root / relative
        raw = path.read_bytes()
        if len(raw) != identity['bytes'] or hashlib.sha256(raw).hexdigest() != identity['sha256']:
            raise ValueError('ReLiF official source changed: ' + relative)
    return root, pin


def selected_tree(source, names):
    tree = ast.parse(source)
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in nodes} != set(names) or len(nodes) != len(names):
        raise ValueError('Missing or duplicate pinned native definitions')
    return ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))


def load_native(code_root):
    root, pin = pinned_sources(code_root)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    # A private namespace avoids replacing networks/segment_anything in the
    # host process, and skips unused 2D architectures in package __init__.py.
    prefix = '_relif_native_' + hashlib.sha256(str(root).encode()).hexdigest()[:12]

    def package(name, directory):
        existing = sys.modules.get(name)
        if existing is not None:
            if existing.__path__ != [str(directory)]:
                raise ValueError('Native package namespace collision')
            return existing
        module = types.ModuleType(name)
        module.__path__ = [str(directory)]
        module.__package__ = name
        module.__spec__ = importlib.util.spec_from_loader(name, loader=None, is_package=True)
        sys.modules[name] = module
        return module

    package(prefix, root)
    package(prefix + '.networks', root / 'networks')
    package(prefix + '.segment_anything', root / 'segment_anything')
    modeling = package(prefix + '.segment_anything.modeling', root / 'segment_anything/modeling')

    def module(relative):
        name = prefix + '.' + relative[:-3].replace('/', '.')
        if name in sys.modules:
            return sys.modules[name]
        path = root / relative
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and (
                    node.module == 'networks' or node.module.startswith('networks.')):
                node.module = prefix + '.' + node.module
        loaded = types.ModuleType(name)
        loaded.__file__ = str(path)
        loaded.__package__ = name.rsplit('.', 1)[0]
        sys.modules[name] = loaded
        try:
            exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), loaded.__dict__)
        except BaseException:
            del sys.modules[name]
            raise
        return loaded

    module('networks/networks_other.py')
    module('networks/utils.py')
    backbone = module('networks/unet_3D.py').unet_3D
    for filename, attributes in (
        ('image_encoder3D.py', ('ImageEncoderViT3D',)),
        ('prompt_encoder3D.py', ('PromptEncoder3D',)),
        ('mask_decoder3D.py', ('MaskDecoder3D', 'TwoWayTransformer3D')),
        ('sam3D.py', ('Sam3D',)),
    ):
        loaded = module('segment_anything/modeling/' + filename)
        for attribute in attributes:
            setattr(modeling, attribute, getattr(loaded, attribute))
    builder = module('segment_anything/build_sam3D.py').sam_model_registry3D['vit_b_ori']
    native = {'torch': torch, 'nn': nn, 'F': F}
    for relative, names in [('ReLiF_3D_train_la.py', DEFINITIONS),
                            ('utils/losses.py', ('DiceLoss',))]:
        source = (root / relative).read_text(encoding='utf-8')
        exec(compile(selected_tree(source, names), str(root / relative), 'exec'), native)
    native.update(backbone=backbone, sam_builder=builder, source_pin=pin)
    return native
