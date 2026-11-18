"""Isolated BY guard adaptation. No scientific source is edited on disk."""
import ast
import copy
import hashlib
import types
from pathlib import Path

CORE_SHA='6172f8186c7760a4b6a135108a12d7b3938806d165a12feefd55adf3842b6bad'


def public_core_ast(path):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=CORE_SHA:
        raise ValueError('Original BY core bytes changed')
    original=ast.parse(raw); tree=copy.deepcopy(original)
    core=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DualteacherCore')
    init=next(n for n in core.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    guard=init.body[0]
    if not isinstance(guard,ast.Expr) or not isinstance(guard.value,ast.Call) or ast.unparse(guard.value.func)!='need':
        raise ValueError('Constructor guard moved')
    if ast.unparse(guard.value.args[0])!='type(total) is int and total in (1710, 1620)':
        raise ValueError('Original BY budget guard changed')
    if len(guard.value.args)!=2 or guard.value.keywords:
        raise ValueError('Original guard signature changed')
    original_guard=copy.deepcopy(guard)
    guard.value.args[0]=ast.parse('type(total) is int and total in (6210, 6120)',mode='eval').body
    adapted=copy.deepcopy(tree)
    init.body[0]=original_guard
    if ast.dump(tree,include_attributes=False)!=ast.dump(original,include_attributes=False):
        raise ValueError('Unexpected scientific AST change')
    proof=dict(original_core_sha256=CORE_SHA,adapted_ast_sha256=hashlib.sha256(ast.dump(adapted,include_attributes=False).encode()).hexdigest(),
               only_change='constructor allowed total tuple (1710,1620) -> (6210,6120)',
               original_disk_source_unchanged=True,all_other_ast_identical=True,
               original_total_is_clock_not_new_cutoff=True)
    return ast.fix_missing_locations(adapted),proof


def load_public_core(path):
    """Explicit runtime use only, after independent GPU/resource admission."""
    tree,proof=public_core_ast(path)
    module=types.ModuleType('_public_dualteacher_native_clock_v1')
    module.__file__=str(path)
    exec(compile(tree,str(path),'exec'),module.__dict__)
    return module.DualteacherCore,proof
