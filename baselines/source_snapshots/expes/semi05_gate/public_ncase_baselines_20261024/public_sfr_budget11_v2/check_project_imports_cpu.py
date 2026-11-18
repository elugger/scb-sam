"""Bounded import-repair checks; no models, patients, optimizer or test entry."""
import ast, hashlib, importlib.util, json, sys
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from project_imports import configure_project_imports,validate_project_imports,EXPECTED
ROOT=Path('__RECORDED_PROJECT_ROOT__')
OLD=ROOT/'expes/semi05_gate/public_ncase_baselines_20261024/public_sfr_budget11_v1'
checks=[]
def ok(name,value=True):
 assert value,name
 checks.append(name)
def reject(name,fn):
 try:fn()
 except (AssertionError,ModuleNotFoundError,ValueError):ok(name);return
 raise AssertionError('Accepted invalid prerequisite: '+name)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tree(s):return ast.dump(ast.parse(s),include_attributes=False)
pins=[dict(path=str(ROOT/n),sha256=h) for n,h in EXPECTED.items()]
for n,h in EXPECTED.items():ok('actual pinned '+n,sha(ROOT/n)==h)
init=ast.parse((ROOT/'expes/common/__init__.py').read_text())
ok('package init has only literal docstring',len(init.body)==1 and isinstance(init.body[0],ast.Expr) and isinstance(init.body[0].value,ast.Constant) and isinstance(init.body[0].value.value,str))
configure_project_imports(ROOT);configure_project_imports(ROOT)
ok('idempotent root first',sys.path[0]==str(ROOT) and sys.path.count(str(ROOT))==1)
v=validate_project_imports(ROOT,pins)
ok('actual module resolution',v['module_origin_verified'] and v['dependencies_resolvable'])
ok('no heavy dependencies imported',not any(n in sys.modules for n in ('numpy','torch','surface_distance','expes.common.seg_metrics')))
for i in range(2):
 bad=[dict(r) for r in pins];bad[i]['sha256']='0'*64
 reject('wrong contract pin '+str(i),lambda:validate_project_imports(ROOT,bad))
 reject('missing contract pin '+str(i),lambda:validate_project_imports(ROOT,pins[:i]+pins[i+1:]))
sys.path.insert(0,'/untrusted/ambient')
reject('root precedence',lambda:validate_project_imports(ROOT,pins))
configure_project_imports(ROOT)
orig=importlib.util.find_spec
class WrongSpec:origin='/untrusted/seg_metrics.py'
with patch('project_imports.importlib.util.find_spec',side_effect=lambda n:WrongSpec() if n=='expes.common.seg_metrics' else orig(n)):
 reject('wrong metrics origin',lambda:validate_project_imports(ROOT,pins))
for dep in ('expes.common.seg_metrics','numpy','torch','surface_distance'):
 with patch('project_imports.importlib.util.find_spec',side_effect=lambda n,d=dep:None if n==d else orig(n)):
  reject('missing module '+dep,lambda:validate_project_imports(ROOT,pins))
reject('missing root',lambda:configure_project_imports(ROOT/'missing_sfr_import_root'))
protocol=(HERE/'sfr_protocol.py').read_text()
protocol=protocol.replace('from project_imports import configure_project_imports,validate_project_imports\n','').replace('sys.path.insert(0,str(OLD))\nconfigure_project_imports(ROOT)','sys.path.insert(0,str(OLD))').replace(" configure_project_imports(ROOT);validate_project_imports(ROOT,c['sources'])\n",'')
ok('protocol AST unchanged except declared import guards',tree(protocol)==tree((OLD/'sfr_protocol.py').read_text()))
operator=(HERE/'public_operator.py').read_text().replace(" package_init=ROOT/'expes/common/__init__.py'\n if str(package_init) not in {r['path'] for r in src}:src.append(dict(path=str(package_init),sha256=sha(package_init)))\n",'')
ok('operator AST unchanged except package binding',tree(operator)==tree((OLD/'public_operator.py').read_text()))
for p in sorted(OLD.iterdir()):
 if p.suffix in ('.py','.json') and p.name not in ('sfr_protocol.py','public_operator.py','template.json'):
  ok('unchanged frozen component '+p.name,sha(p)==sha(HERE/p.name))
t=json.loads((HERE/'template.json').read_text())
for r in t['sources']:
 if Path(r['path']).name in ('sfr_backend.py','sfr_runtime_components.py','sfr_stage_protocol.py','sfr_training_core.py'):
  ok('unchanged native scientific core '+Path(r['path']).name,sha(r['path'])==r['sha256'])
ok('native metric import retained',"from expes.common.seg_metrics import" in Path(next(r['path'] for r in t['sources'] if Path(r['path']).name=='sfr_backend.py')).read_text())
ok('no model or metrics import from checks',not any(n in sys.modules for n in ('numpy','torch','surface_distance','sfr_backend','expes.common.seg_metrics')))
result=dict(passed=True,checks=len(checks),check_names=checks,actual_import_binding=v,source={p.name:sha(p) for p in sorted(HERE.glob('*.py'))},
 new_model_loads=0,new_cuda_loads=0,new_patient_loads=0,new_optimization_updates=0,new_test_evaluations=0,new_jobs=0,new_reservations=0,
 scope='Real small source/module resolution and bounded negative guards; native science AST preserved. No GPU/tensor/qualification/admission proof.')
print(json.dumps(result,sort_keys=True))
