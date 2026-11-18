"""Extract pinned scientific update bodies without historical selection loops.

No source import or torch execution occurs during extraction. Loss, backward,
optimizer and EMA statements are copied unchanged, allowing an AST audit of
the science while a separate controller governs the endpoint and evaluation.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_pinned(relative_path):
    inventory = json.loads((HERE / "source_inventory.json").read_text(encoding="utf-8"))
    matches = [r for r in inventory if r["remote_path"] == relative_path]
    if len(matches) != 1:
        raise ValueError("Source absent or duplicated in inventory: " + relative_path)
    path = HERE / matches[0]["local_path"]
    if not path.exists() and Path('__RECORDED_PROJECT_ROOT__').is_dir():
        path = Path('__RECORDED_PROJECT_ROOT__') / relative_path
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != matches[0]["remote_sha256"]:
        raise ValueError("Pinned historical source changed: " + relative_path)
    return ast.parse(data, filename=str(path)), path


MT_SOURCE = "expes/artifacts/mt-semisam-plus/code/expes/mt_semisam_plus/trainer_mt_semisam.py"
UA_SOURCE = "expes/artifacts/uamt-semisam-plus/code/expes/uamt_semisam_plus/trainer_uamt_semisam.py"
SDCL_SOURCE = "expes/sdcl/trainer_sdcl.py"

SPECS = {
    "mt_semisam": (MT_SOURCE, "MTSemiSAMTrainer", "semi_train", "mt"),
    "uamt_semisam": (MT_SOURCE, "MTSemiSAMTrainer", "semi_train", "mt"),
    "sdcl_pretrain": (SDCL_SOURCE, "SDCLTrainer", "pretrain", "sdcl"),
    "sdcl": (SDCL_SOURCE, "SDCLTrainer", "self_train", "sdcl"),
}


def class_method(tree, class_name, method_name):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)


def update_body(method):
    loops = [n for n in ast.walk(method) if isinstance(n, ast.For)
             and ast.dump(n.iter, include_attributes=False) == ast.dump(
                 ast.parse("range(steps)", mode="eval").body, include_attributes=False)]
    if len(loops) != 1:
        raise ValueError("Expected exactly one audited range(steps) update loop")
    out = []
    for statement in loops[0].body:
        out.append(copy.deepcopy(statement))
        if isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Attribute):
            if isinstance(statement.target.value, ast.Name) and statement.target.value.id == "self" and statement.target.attr == "global_step":
                break
    else:
        raise ValueError("Audited update boundary global_step += 1 missing")
    return out


def extract_kernel(method_name):
    source, cls, method, kind = SPECS[method_name]
    tree, path = load_pinned(source)
    body = update_body(class_method(tree, cls, method))
    if kind == "mt":
        args = ["self", "lab_iter", "unlab_iter", "max_iterations", "scfg", "ema_decay", "labeled_bs"]
        result = "dict(loss=float(loss.item()), supervised=float(supervised_loss.item()), consistency=float(consistency_loss.item()), sam=float(sam_con_loss.item()))"
    else:
        args = (["self", "it_a", "it_b"] if method_name == "sdcl_pretrain" else ["self", "it"])
        result = "dict(loss_v=float(loss_v.item()), loss_r=float(loss_r.item()))"
    fn = ast.FunctionDef(
        name="fixed_update_" + method_name,
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=a) for a in args],
                           kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=body + [ast.Return(ast.parse(result, mode="eval").body)],
        decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    calls = [n.func.attr for n in ast.walk(module)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    forbidden = {"_evaluate", "_save", "_load", "_apply_resume", "test_all_case"}
    if forbidden.intersection(calls):
        raise ValueError("Selection/evaluation present inside scientific kernel")
    if any(isinstance(n, (ast.Break, ast.Continue)) for n in ast.walk(module)):
        raise ValueError("Historical loop termination leaked into extracted kernel")
    return module, {"source": str(path), "source_method": method,
                    "body_ast_sha256": hashlib.sha256(ast.dump(
                        ast.Module(body=body, type_ignores=[]), include_attributes=False).encode()).hexdigest(),
                    "retained_statements": len(body), "native_optimizer_calls": calls.count("step"),
                    "runtime_verified": False}


def bind_kernel(method_name, namespace):
    tree, audit = extract_kernel(method_name)
    scope = dict(namespace)
    exec(compile(tree, audit["source"], "exec"), scope)
    return scope["fixed_update_" + method_name], audit


def method_definitions(source, cls, names):
    """Compile selected original helpers; never execute the original module."""
    tree, path = load_pinned(source)
    methods = [copy.deepcopy(class_method(tree, cls, name)) for name in names]
    return ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), str(path)
