import math
from by_protocol import require
GROUPS={"HPQ":("A1B","B")}
def check_step(row,group,overflow=False):
    require(row['skipped'] is overflow and row['teacher_no_grad'],'Unexpected overflow or teacher gradient')
    require(row['clip_calls']==row['ema_calls']==(0 if overflow else 1),'Clip/EMA count differs')
    require(row['context_computed'] and len(row['microbatches'])==5,'Wrong accumulation')
    if not overflow:
        require(len(row['gradient_groups'])==6 and all(v is not None and __import__('math').isfinite(v) and v>0
            for v in row['gradient_groups'].values()),'Nonfinite/zero component gradient')
    for m in row['microbatches']:
        c=m['context']
        require(c['reduction']==GROUPS[group][1] and c['coefficient']==1 and c['applied']==(group in ('HQ','HPQ')),'CCBS switch differs')
        require(c['no_gt_argument'] and c['declared_state_restored'] and not c['relation_computed'],'CCBS boundary differs')
        require(c['foreground_voxels']+c['background_voxels']==64*c['reliable_blocks'],'CCBS counts differ')
        require(all(__import__('math').isfinite(float(m[k])) for k in ('supervised','pseudo','wsv','baseline_total')),'Nonfinite loss')
