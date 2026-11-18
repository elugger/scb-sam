"""Isolated import-only compatibility; original official checkouts stay intact."""
import ast
import hashlib
import importlib.util
import importlib.abc
from pathlib import Path
import types


def sfr_lora_tree(source):
    tree=ast.parse(source)
    changed=0
    for node in tree.body:
        if isinstance(node,ast.ImportFrom) and (node.module=='segment_anything'
                                               or node.module.startswith('segment_anything.')):
            node.module=node.module.replace('segment_anything','segment_anything_lora',1)
            changed+=1
    if changed!=3:
        raise ValueError('Unexpected official LoRA import layout; re-audit')
    return ast.fix_missing_locations(tree)


def load_sfr_lora(code_root):
    """Call only inside the dedicated SFR worker, never inside another method."""
    import sys
    root=Path(code_root).resolve(strict=True)
    sys.path.insert(0,str(root))
    # The vendored package also contains absolute imports to its old name.
    # Repair only that package in the isolated worker, never alias the project's
    # 3D segment_anything globally or edit the original checkout in place.
    class ImportOnlyLoader(importlib.abc.Loader):
        def __init__(self,path): self.path=path
        def create_module(self,spec): return None
        def exec_module(self,module):
            tree=ast.parse(self.path.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom) and node.module and (
                        node.module=='segment_anything' or node.module.startswith('segment_anything.')):
                    node.module=node.module.replace('segment_anything','segment_anything_lora',1)
            exec(compile(ast.fix_missing_locations(tree),str(self.path),'exec'),module.__dict__)
    class IsolatedPackageFinder(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname!='segment_anything_lora' and not fullname.startswith('segment_anything_lora.'):
                return None
            stem=root.joinpath(*fullname.split('.'))
            filename=stem/'__init__.py' if stem.is_dir() else stem.with_suffix('.py')
            if not filename.is_file(): return None
            spec=importlib.util.spec_from_loader(fullname,ImportOnlyLoader(filename),
                                               is_package=stem.is_dir())
            if stem.is_dir(): spec.submodule_search_locations=[str(stem)]
            return spec
    if any(n=='segment_anything_lora' or n.startswith('segment_anything_lora.') for n in sys.modules):
        raise ValueError('SFR package already loaded; use a fresh isolated worker')
    sys.meta_path.insert(0,IsolatedPackageFinder())
    source=(root/'sam_lora_image_encoder.py').read_text()
    module=types.ModuleType('_main_sfr_lora_import_compat')
    exec(compile(sfr_lora_tree(source),str(root/'sam_lora_image_encoder.py'),'exec'),module.__dict__)
    return module,dict(top_level_import_changes=3,vendored_absolute_imports_scoped=True,scientific_body_changed=False,
                       source_sha256=hashlib.sha256(source.encode()).hexdigest())


def load_sfr_architecture(code_root,kind='vnet'):
    # Import only the selected official architecture. VNet does not need the
    # absent UNet-only networks.utils / networks.networks_other modules.
    if kind!='vnet':
        raise ValueError('Only the pinned official default VNet is qualified for adaptation')
    p=Path(code_root)/'networks/hierarchical_vnet.py'
    spec=importlib.util.spec_from_file_location('_main_sfr_native_vnet',p)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.VNet


def sfr_native_patch_plan(shape=(112,112,80),canvas=1024):
    # Preserve the official LA patch instead of silently truncating a 128-deep
    # ROI to the 64 slices that fit an 8x8 board. Real training still needs a
    # documented ROI->native crop and complete-volume inference adapter.
    h,w,d=shape
    rows,cols=canvas//h,canvas//w
    if rows*cols<d:
        raise ValueError('Stitch board would discard real slices')
    return dict(shape=list(shape),rows=rows,cols=cols,real_slices=d,
                padding_tiles=rows*cols-d,all_slices_covered=True)
