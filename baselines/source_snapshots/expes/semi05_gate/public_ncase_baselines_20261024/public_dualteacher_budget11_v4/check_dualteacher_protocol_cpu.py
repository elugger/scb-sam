"""Protocol wiring and actual read-only shared resource projection, no enrollment.
No model/CUDA/patient load, tensor serialization, optimization or evaluation.
"""
import copy,hashlib,json,pathlib,sys,time
import public_dualteacher_protocol as p
checks=[]
def yes(n,v):
 if not v:raise AssertionError(n)
 checks.append(n)
def reject(n,fn):
 try:fn()
 except (ValueError,KeyError,TypeError):checks.append(n);return
 raise AssertionError('Accepted invalid protocol: '+n)
def main(out):
 template=p.read(p.HERE/'template.json');origin=p.read(template['origin_contract']['path']);records=[]
 for labels in (5,10):
  job,target=p.paths(labels,'r1');c=dict(labeled_cases=labels,ratio=labels,tag='r1',output=str(job/'formal'),target=str(target),foundation=origin['foundation'],official_code_root=origin['official_code_root'],native_validate_sha256=origin['native_validate_sha256'])
  yes(f'{labels}_new_job_absent',not job.exists() and not target.exists());b=p.binding(c);c['recipe']=b.recipe;c['splits']=b.splits
  yes(f'{labels}_own_original_clock',p.clock_budget(labels).primary_pairs==2*b.recipe['original_clock_iterations'])
  recipe=p.public_recipe(c);native=recipe.pop('source_native_BY_recipe');own=copy.deepcopy(recipe)
  yes(f'{labels}_original_metadata_preserved',native['dataset']=='BY' and own['dataset']=='Public433' and own['fixed_endpoint']['test_case_count']==87 and native['fixed_endpoint']['test_case_count']==25)
  yes(f'{labels}_cutoff_and_original_clock_separate',own['native_iterations']==b.recipe['ssl_iterations'] and own['original_clock_iterations']==b.recipe['original_clock_iterations'])
  yes(f'{labels}_two_effective_updates',own['optimizer']['all_native_effective_optimizer_updates_expected']==2*b.recipe['ssl_iterations'] and own['optimizer']['op_step_calls']==2*b.recipe['ssl_iterations'])
  for key in ('native_algorithm_config','student','prototype','consistency','sam','preprocessing','process_environment'):yes(f'{labels}_native_'+key,own[key]==native[key])
  targets=p.rows(c);yes(f'{labels}_six_own_growth_targets',len(targets)==6 and len({r['path'] for r in targets})==6)
  yes(f'{labels}_full_state_bounds',targets[0]['bound']==256*1024**2 and [r['bound'] for r in targets[1:5]]==[768*1024**2]*4)
  p.output_check(c,c['output'],'formal');p.output_check(c,job/'qualification_engineering_sam','engineering_sam');yes(f'{labels}_role_output_bindings',True)
  reject(f'{labels}_cross_role_output',lambda:p.output_check(c,c['output'],'engineering_sam'))
  reject(f'{labels}_unknown_tag',lambda:p.paths(labels,'../r1'))
  c['sources']=list({row['path']:row for row in origin['sources']+origin['environment']['files']}.values())
  diagnostic=dict(stdout='creating model SAM-Med3D\n',stderr='',failed=False,warnings=[]);p.check_initialization_diagnostics(c,diagnostic);yes(f'{labels}_pinned_init_empty_warning',True)
  warning=dict(message='Importing from timm.models.layers is deprecated, please import via timm.layers',category='FutureWarning',filename='__RECORDED_HOME__/anaconda3/envs/sam3d/lib/python3.10/site-packages/timm/models/layers/__init__.py',lineno=48)
  diagnostic['warnings']=[warning];p.check_initialization_diagnostics(c,diagnostic);yes(f'{labels}_exact_pinned_warning_AST',True)
  for name,key,value in [('unknown_output','stdout','unknown'),('unknown_stderr','stderr','unknown'),('partial_warning','warnings',[dict(warning,message='partial')]),('duplicate_warning','warnings',[warning,warning]),('failed_init','failed',True)]:
   bad=copy.deepcopy(diagnostic);bad[key]=value;reject(f'{labels}_'+name,lambda:p.check_initialization_diagnostics(c,bad))
  p.guard_test_lineage(c);yes(f'{labels}_unopened_lineage',True)
  guard=p.resource(c);projected=guard.targets();yes(f'{labels}_own_full_projection',all((pathlib.Path(r['path']),r['bound']) in projected for r in targets))
  aux=p.read(p.SHARED/'cpac_auxiliary_targets_20261022.json');yes(f'{labels}_all_CPAC_aux_preserved',all((pathlib.Path(r['path']),r['bound']) in projected for r in aux))
  yes(f'{labels}_projection_unique',len(projected)==len({str(x[0]) for x in projected}))
  # Actual original save.lock capacity/RAM check over projected future growth.
  # This does not enroll a target or admit an incompletely integrated method.
  registry_before=pathlib.Path(p.REGISTRY).read_bytes();capacity=p.capacity(c,start=True)
  yes(f'{labels}_real_readonly_full_gate',all(r['margin_bytes']>=2*1024**3 for r in capacity['filesystems']))
  yes(f'{labels}_registry_unchanged',pathlib.Path(p.REGISTRY).read_bytes()==registry_before and not job.exists() and not target.exists())
  records.append(dict(labels=labels,original_clock=b.recipe['original_clock_iterations'],cutoff=b.recipe['ssl_iterations'],capacity=capacity,projection_targets=len(projected),cpac_aux_count=len(aux)))
 p.gpu_idle();yes('actual_GPU1_uuid_pci_idle',True)
 fd=p.lease_open();import fcntl,os
 try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(fd,fcntl.LOCK_UN);yes('read_only_lease_available',True)
 finally:os.close(fd)
 p.check_logs(b'');reject('unknown_worker_stdout',lambda:p.check_logs(b'unknown\n'));yes('no_heavy_modules',all(n not in sys.modules for n in ('torch','numpy','SimpleITK')))
 result=dict(status='PASS_PROTOCOL_CPU_AND_READONLY_ORIGINAL_RESOURCE_PROJECTION',checks=len(checks),check_names=checks,at=time.time(),records=records,source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.HERE.glob('*.py')},model_loads=0,cuda_loads=0,patient_loads=0,actual_optimization_updates=0,actual_test_evaluations=0,new_jobs=0,new_reservations=0,registry_mutated=False,independent_admission=False,gpu_qualification=False,scope='Real small source/config/warning AST and actual original-lock projected capacity/RAM/CPAC aux check; not enrolled resources, full operator admission, native tensors or GPU qualification.')
 pathlib.Path(out).open('x').write(json.dumps(result,indent=2));print(json.dumps(dict(status=result['status'],checks=len(checks))))
if __name__=='__main__':main(sys.argv[1])
