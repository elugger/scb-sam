"""Read-only own ledgers and fixed-final acceptance, with zero model loading/testing."""
import argparse
import json
import math
from pathlib import Path

from ncase_protocol import BASE, ids, sha
from train_ncase import validate_config, validate_case_sets, resolve_path, verify_manifests


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []


def verify_ledgers(cfg, steps, epochs):
    budget = validate_config(cfg)
    lab, test = [ids(resolve_path(cfg,cfg[name])) for name in ('train_list','test_list')]
    validate_case_sets(lab, test, cfg['labeled_cases'], 87)
    n = len(lab)
    if len(steps)>budget['total_updates'] or len(epochs)>60:
        raise ValueError('Budget exceeded')
    for i,row in enumerate(steps):
        if (row['successful_update'] != i+1 or row['epoch'] != i//n or row['epoch_position'] != i%n
                or row['case_id'] not in lab or row['requested_index'] != lab.index(row['case_id'])
                or row['optimizer_steps'] != 1 or row['skipped'] is not False or row['gradient_finite'] is not True
                or row['amp_scale'] != 1024 or not math.isfinite(row['loss'])):
            raise ValueError('Step identity/numerics/count differs: '+str(i+1))
    for begin in range(0,len(steps),n):
        batch = steps[begin:begin+n]
        cases = [r['case_id'] for r in batch]
        if len(set(cases)) != len(cases) or (len(batch)==n and set(cases)!=set(lab)):
            raise ValueError('Actual case coverage differs')
    if len(epochs)>len(steps)//n or len(steps)//n-len(epochs)>1:
        raise ValueError('Epoch ledger inconsistent with successful steps')
    for i,row in enumerate(epochs):
        actual = [r['case_id'] for r in steps[i*n:(i+1)*n]]
        if (row['epoch'] != i or row['epoch_number'] != i+1 or row['successful_updates'] != (i+1)*n
                or row['actual_case_ids'] != actual or row['test_evaluations'] != 0):
            raise ValueError('Complete-epoch ledger differs')
    return dict(successful_updates=len(steps), total_updates=budget['total_updates'],
                completed_epochs=len(epochs), percentage=100*len(steps)/budget['total_updates'])


def inspect(folder):
    folder = Path(folder)
    cfg = json.loads((folder/'config.json').read_text())
    verify_manifests(cfg)
    status = verify_ledgers(cfg,rows(folder/'training_steps.jsonl'),rows(folder/'training_epochs.jsonl'))
    status.update(state='INCOMPLETE_READ_ONLY_SNAPSHOT', failure=None, test_evaluations=0)
    if (folder/'failure.json').exists():
        status.update(state='FAILED_PRESERVED',failure=json.loads((folder/'failure.json').read_text()))
        return status
    target = Path(cfg['artifact_dir'])/'final.pth'
    metrics_path=folder/'final_metrics.json'
    if not metrics_path.exists():
        status['final_test_started']=(folder/'test_evaluation_started.json').exists()
        return status
    metrics=json.loads(metrics_path.read_text())
    cases=rows(folder/'final_test_cases.jsonl')
    test=ids(resolve_path(cfg,cfg['test_list']))
    reload=json.loads((folder/'final_reload_verified.json').read_text())
    marker=json.loads((folder/'test_evaluation_started.json').read_text())
    if status['successful_updates']!=cfg['labeled_cases']*60 or status['completed_epochs']!=60:
        raise ValueError('Premature final')
    digest=sha(target)
    if (metrics['status']!='COMPLETE' or metrics['test_evaluations']!=1 or metrics['completed_epochs']!=60
            or metrics['successful_updates']!=cfg['labeled_cases']*60 or metrics['labeled_cases']!=cfg['labeled_cases']
            or metrics['final_sha256']!=digest or reload['sha256']!=digest or reload['strict_load'] is not True
            or marker['final_sha256']!=digest or marker['test_count']!=87
            or metrics!=json.loads((folder/'progress.json').read_text())):
        raise ValueError('Final/reload/progress identity differs')
    if [c['actual_case_id'] for c in cases]!=test or metrics['test_actual_case_ids']!=test:
        raise ValueError('Final test case accounting')
    for row in cases:
        if row['final_sha256']!=digest or row['case_replaced'] is not False or row['requested_case_id']!=row['actual_case_id']:
            raise ValueError('Final per-case identity')
    for key,value in metrics['metrics'].items():
        vals=[float(c['metrics'][key]) for c in cases]
        if not all(math.isfinite(v) for v in vals) or not math.isfinite(value) or abs(sum(vals)/87-value)>1e-10:
            raise ValueError('Final aggregation differs')
    if list(folder.rglob('*.partial')) or list(target.parent.rglob('*.partial')):
        raise ValueError('Partial artifact remains')
    receipt=rows(folder/'save_receipts.jsonl')
    if len(receipt)!=1 or receipt[0]['kind']!='final' or receipt[0]['receipt']['sha256']!=digest:
        raise ValueError('Not exactly one terminal save')
    status.update(state='LEDGERS_FINAL_VERIFIED_PROCESS_EXIT_AND_GPU_IDLE_STILL_REQUIRED',
                  final_sha256=digest, final_metrics_sha256=sha(metrics_path), metrics=metrics['metrics'],test_evaluations=1)
    return status


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(BASE)/'sup_n005')
    args=parser.parse_args()
    print(json.dumps(inspect(args.output),ensure_ascii=False))
