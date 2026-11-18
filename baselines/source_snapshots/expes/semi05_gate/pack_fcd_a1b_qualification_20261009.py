"""CPU-only small A1B qualification evidence; no model forward or weight backup."""
import argparse, json, hashlib, tarfile
from pathlib import Path
ROOT=Path('__RECORDED_PROJECT_ROOT__')
BASE=ROOT/'expes/work_dir/semi05_gate/FCD_A_MATRIX_20261006/A1B'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--tag05',default='qualification05_r1');p.add_argument('--tag20',default='qualification20_r1');a=p.parse_args()
 assert a.tag05.replace('_','').isalnum() and a.tag20.replace('_','').isalnum()
 destination=BASE/'qualification_evidence_r1';assert not destination.exists()
 reports={}
 for ratio,tag in [('05',a.tag05),('20',a.tag20)]:
  directory=BASE/tag;report=json.loads((directory/'qualification.json').read_text())
  assert report['status']=='PASS' and report['group']=='A1B' and report['successful_updates']==9 and report['artificial_overflows']==1
  assert report['cold_warm_legacy_exact'] and report['active_repeat_exact'] and report['true_two_branch_routing'] and report['context_vb_component_exact'] and report['new_loss_changes_update'] and report['allowed_train_lab_probe_then_step_exact']
  assert report['checkpoint']['bytes']==Path(report['checkpoint']['path']).stat().st_size
  assert sha(report['checkpoint']['path'])==report['checkpoint']['sha256']
  assert report['ratio']==ratio and report['protocol']=='FIXED_FINAL_DEVELOPMENT_V3_1'
  assert report['overflow_skip'] and report['six_root_roundtrip_exact'] and report['allowed_train_lab_probe_state_unchanged']
  assert sha(report['init_from'])==report['init_sha256']
  for path,digest in report['source_hashes'].items():assert sha(path)==digest,path
  steps=[json.loads(x) for x in (directory/'qualification_steps.jsonl').read_text().splitlines()]
  assert len(steps)==9 and all(not x['skipped'] and x['ema_calls']==x['clip_calls']==1 for x in steps)
  assert [x['branch'] for x in steps]==['native_LB']*2+['off']*2+['active_B']*2+['active_B_repeat']*2+['allowed_train_lab_probe_then_step']
  components=[json.loads(x) for x in (directory/'context_vb_components.jsonl').read_text().splitlines()]
  assert [x['phase'] for x in components]==['cold','warm']
  assert all(x['state_unchanged'] and x['rng_unchanged'] and x['same_target_mask'] and x['frozen_formula_gradient_exact'] and x['optimizer_updates']==0 for x in components)
  reports[ratio]=report
 assert reports['05']['source_hashes']==reports['20']['source_hashes']
 summary=dict(status='PASS',group='A1B',protocol='FIXED_FINAL_DEVELOPMENT_V3_1',ratios=reports,test_forwards_added_by_pack=0,model_updates_added_by_pack=0)
 destination.mkdir()
 reportpath=destination/'report.json';reportpath.write_text(json.dumps(summary,indent=2,allow_nan=False))
 bundle=destination/'evidence.tar.gz'
 with tarfile.open(bundle,'w:gz') as tar:
  tar.add(reportpath,arcname='report.json')
  for ratio,tag in [('05',a.tag05),('20',a.tag20)]:
   for item in (BASE/tag).rglob('*'):
    if item.is_file() and item.suffix in ('.json','.jsonl','.yaml','.log'):
     tar.add(item,arcname=str(Path(ratio)/item.relative_to(BASE/tag)))
  for path in reports['05']['source_hashes']:
   source=Path(path)
   if source.suffix=='.py':tar.add(source,arcname='sources/'+source.name)
 receipt=dict(bundle=str(bundle),bundle_sha256=sha(bundle),report_sha256=sha(reportpath),bytes=bundle.stat().st_size)
 (destination/'transport.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
