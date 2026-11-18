import os
import sys
from pathlib import Path
ASNET_ROOT = Path(os.environ.get('ASNET_ROOT', '.')).resolve()
import argparse
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
ROOT = Path(str(ASNET_ROOT))
PREFIX = ROOT / 'expes/semi05_gate/by5_four_arm_rerun_20261030'
BASE = ROOT / 'expes/work_dir/semi05_gate/BY5_FOUR_ARM_RERUN_20261030'
STORE = ROOT / 'expes/artifacts/FCD_A_MATRIX_20261006/MAIN_TABLE_FIXED_FINAL_20261019/by5_four_arm_rerun_20261030'
DEST = ROOT / 'expes/semi05_gate/by5_four_arm_initialization_check_20261030'
GPU = os.environ['ASNET_GPU_UUID']
INIT_SHA = 'b113d1731a8e535ff482bebe7c2ca0d821760d44e5ea02e59b33cae58557c4be'
GROUPS = {'H': ('A00', '0'), 'HP': ('A10', '0'), 'HQ': ('A0B', 'B'), 'HPQ': ('A1B', 'B')}
OUT = BASE / 'four_arm_initialization_check'
EXPECTED_MANIFEST_SHA = None
sys.path.insert(0, str(PREFIX))
from by_protocol import require, sha
from prepare_by5 import dump_new, stamp
from launch_by5 import verify_preparation
from by_resources import original_lock, capacity_snapshot
from by_runtime import ensure_gpu0_idle, own_gpu0_fds

def verify():
    cfg, prep = verify_preparation('13ea706ebe7106421222df9df95db385e4eeb77809e9edb53f8d5bd47a7c3e98', 'a03c0c05f433dbf33e35e69162d8c921981118c73737a3cffa7649d35c61e4c2')
    require(EXPECTED_MANIFEST_SHA and sha(DEST / 'manifest.json') == EXPECTED_MANIFEST_SHA, 'Qualification manifest differs')
    manifest = json.loads((DEST / 'manifest.json').read_text())
    require(set(manifest['files']) == {'qualify_four.py', 'ancestry.json', 'acceptance.json'}, 'Qualification file set differs')
    for name, h in manifest['files'].items():
        require(not (DEST / name).is_symlink() and sha(DEST / name) == h, 'Qualification source changed: ' + name)
    ancestry = json.loads((DEST / 'ancestry.json').read_text())
    require(hashlib.sha256(ancestry['config_text'].encode()).hexdigest() == ancestry['config_sha256'], 'Ancestor config text differs')
    require(sha(ancestry['config_path']) == ancestry['config_sha256'], 'Ancestor config file changed')
    for name, h in ancestry['source_hashes'].items():
        require(sha(name) == h, 'Scientific source changed: ' + name)
    accepted = json.loads((DEST / 'acceptance.json').read_text())
    require(accepted['status'] == 'PASS_SHARED_INITIALIZER_INDEPENDENT_ACCEPTANCE' and accepted['initializer']['sha256'] == INIT_SHA and (accepted['successful_updates'] == 300), 'Initializer acceptance differs')
    require(sha(STORE / 'supervised/final.pth') == INIT_SHA, 'Initializer changed')
    return (cfg, prep, ancestry, accepted)

def frozen_config(ancestor, group):
    cfg = copy.deepcopy(ancestor)
    cfg['experiment'].update(output_dir=str(OUT / group), gpu=GPU, seed=2027)
    cfg['semi'].update(paired_checkpoint_dir=str(OUT / group / 'unused_checkpoint_dir'), paired_workers=0, matrix_group=GROUPS[group][0], matrix_c=GROUPS[group][1])
    return cfg

def child(group, fd, dev, ino):
    os.chdir(ROOT)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == GPU, 'GPU environment differs')
    prefix_cfg, prep, ancestry, accepted = verify()
    from train_by5 import validate_gpu_claim, load_trusted_checkpoint, state_dict_sha256
    claim = validate_gpu_claim(prefix_cfg, 0, fd, dev, ino)
    ensure_gpu0_idle()
    with original_lock():
        capacity = capacity_snapshot(config=prefix_cfg)
    import yaml
    cfg = frozen_config(yaml.safe_load(ancestry['config_text']), group)
    require(not (OUT / group).exists(), 'Existing qualification arm')
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    from expes.semi05_gate.paired_fcd_matrix_20261006 import MatrixHalf, MatrixStyle, MatrixMixin
    from expes.semi05_gate.paired_zea_cross_scale_half_20260911 import PairedCrossScaleHalf
    from expes.semi05_gate.paired_l_cross_style_complement_20261001 import PairedCrossStyleComplement
    from expes.semi05_gate.paired_rng_data_20260908 import SeededCases
    from train_by5 import make_strict_dataset, read_case_ids
    Base = MatrixStyle if group in ('HP', 'HPQ') else MatrixHalf

    class FreshBY(Base):

        def _build_dataloaders(self):
            self.lab_set = make_strict_dataset(prefix_cfg, read_case_ids(prefix_cfg['train_list']), 'train')

        def _load(self, path):
            require(Path(path) == STORE / 'supervised/final.pth', 'Wrong initializer')
            with original_lock():
                capacity_snapshot(config=prefix_cfg)
                require(sha(path) == INIT_SHA, 'Initializer changed before read')
                payload = load_trusted_checkpoint(path, map_location='cpu')
                require(payload['successful_updates'] == 300 and payload['completed_epochs'] == 60, 'Initializer budget changed')
                self.student.load_state_dict(payload['student'], strict=True)
                require(state_dict_sha256(self.student.state_dict()) == accepted['initializer']['state_sha256'], 'Student does not reproduce accepted tensor values')
                self.student_fclr_head.load_state_dict(self.cold['head'], strict=True)
                self.student_zea_adapter.load_state_dict(self.cold['adapter'], strict=True)
                for s, t in (('student', 'teacher'), ('head', 'teacher_head'), ('adapter', 'teacher_adapter')):
                    self.roots()[t].load_state_dict(self.roots()[s].state_dict(), strict=True)
                self._fclr_loaded_head = False
                self._matrix_initial_head_state = self.cold['head']
                del payload

        def _save_ucp(self, *args, **kwargs):
            raise RuntimeError('Qualification cannot save weights')

        def _evaluate(self, *args, **kwargs):
            raise RuntimeError('Qualification cannot evaluate development data')
    require(FreshBY.step_matrix is MatrixMixin.step_matrix, 'Update calculation differs')
    require(FreshBY.contextual_gradients is MatrixMixin.contextual_gradients, 'CCBS differs between arms')
    require(FreshBY._losses is (PairedCrossStyleComplement._losses if group in ('HP', 'HPQ') else PairedCrossScaleHalf._losses), 'Wrong CCL factor')
    t = FreshBY(cfg)
    t.cold = {name: {k: v.detach().cpu().clone() for k, v in t.roots()[name].state_dict().items()} for name in ('head', 'adapter')}
    t.initialize_matrix(str(STORE / 'supervised/final.pth'))
    roots = t.roots()
    expected_names = {'student', 'teacher', 'head', 'teacher_head', 'adapter', 'teacher_adapter'}
    require(set(roots) == expected_names, 'Wrong model roots')
    hashes = {name: state_dict_sha256(model.state_dict()) for name, model in roots.items()}
    for s, te in (('student', 'teacher'), ('head', 'teacher_head'), ('adapter', 'teacher_adapter')):
        require(hashes[s] == hashes[te], 'Teacher initialization differs')
    require(hashes['student'] == accepted['initializer']['state_sha256'], 'Wrong base tensors')
    require(not t.optimizer.state and t.matrix_updates == 0, 'Initialization performed updates')
    lrs = [g['lr'] for g in t.optimizer.param_groups]
    require(lrs == [5e-08, 2e-07, 2e-07, 2e-07, 0.0001, 0.0001], 'Wrong common learning rates')
    require(all((tuple(g['betas']) == (0.9, 0.999) and g['weight_decay'] == 0 for g in t.optimizer.param_groups)), 'Wrong common optimizer')
    ids = [id(p) for g in t.optimizer.param_groups for p in g['params']]
    wanted = {id(p) for name in ('student', 'head', 'adapter') for p in roots[name].parameters()}
    require(len(ids) == len(set(ids)) and set(ids) == wanted, 'Optimizer parameter set differs')
    require(not set(ids) & {id(p) for n in ('teacher', 'teacher_head', 'teacher_adapter') for p in roots[n].parameters()}, 'Teacher in optimizer')
    for n, keys in {'head': ('out.weight', 'out.bias', 'gate.weight', 'gate.bias'), 'adapter': ('up.weight', 'up.bias', 'cross_projection.weight', 'cross_projection.bias')}.items():
        require(all((bool((roots[n].state_dict()[k] == 0).all()) for k in keys)), 'Nonzero cold residual')
    before = hashes
    item = SeededCases(t.lab_set, 2027, 'four-arm-initial-identity')[0, 0, 0]
    batch = {k: v.unsqueeze(0) for k, v in item.items() if torch.is_tensor(v)}
    image, label = t._prepare_batch(batch)
    with t.isolated_context(0, 0), torch.no_grad(), torch.cuda.amp.autocast():
        for model in roots.values():
            model.eval()
        native = t._forward_main_logits(t.student, image).cpu()
        teacher = t._forward_main_logits(t.teacher, image).cpu()
    require(torch.equal(native, teacher), 'Student/teacher native prediction differs at initialization')
    after = {name: state_dict_sha256(model.state_dict()) for name, model in roots.items()}
    require(after == before, 'Identity probe changed model state')
    result = dict(status='PASS_ZERO_UPDATE_INITIALIZATION_ONLY', at=stamp(), group=group, initializer_sha256=INIT_SHA, root_sha256=hashes, optimizer_lrs=lrs, scaler=t.scaler.state_dict(), native_output_sha256=hashlib.sha256(native.contiguous().numpy().tobytes()).hexdigest(), actual_training_case=item['case_id'], input_sha256=hashlib.sha256(image.cpu().contiguous().numpy().tobytes()).hexdigest(), successful_updates=0, development_evaluations=0, own_gpu_fds=own_gpu0_fds(), claim=claim, no_unlab_or_test_dataset_built=True, formal_training_authorized_by_this_receipt=False)
    dump_new(OUT / group / 'result.json', result)
    print(json.dumps(result), flush=True)

def main():
    global EXPECTED_MANIFEST_SHA
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', choices=GROUPS)
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--fd', type=int)
    parser.add_argument('--dev', type=int)
    parser.add_argument('--ino', type=int)
    a = parser.parse_args()
    EXPECTED_MANIFEST_SHA = a.manifest_sha
    if a.child:
        child(a.child, a.fd, a.dev, a.ino)
        return
    import fcntl
    os.chdir(ROOT)
    cfg, prep, ancestry, accepted = verify()
    require(Path(__file__).resolve() == DEST / 'qualify_four.py', 'Noncanonical qualification source')
    require(not OUT.exists(), 'Existing qualification; never repeat automatically')
    claim_path = BASE / 'control/gpu0.lock'
    fd = os.open(claim_path, os.O_RDWR | os.O_NOFOLLOW)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    s = os.fstat(fd)
    require(s.st_nlink == 1, 'Invalid claim identity')
    try:
        ensure_gpu0_idle()
        with original_lock():
            capacity_snapshot(config=cfg)
        OUT.mkdir()
        dump_new(OUT / 'intent.json', dict(at=stamp(), pid=os.getpid(), stage='ZERO_UPDATE_FOUR_ARM_INITIALIZATION', source_manifest_sha256=sha(DEST / 'manifest.json'), initializer_sha256=INIT_SHA, groups=list(GROUPS), successful_updates=0, development_evaluations=0, automatic_retry=False))
        status = 'FAILED'
        error = None
        try:
            for group in GROUPS:
                with (OUT / (group + '.log')).open('xb') as log:
                    process = subprocess.run([sys.executable, '-B', str(DEST / 'qualify_four.py'), '--child', group, '--manifest-sha', EXPECTED_MANIFEST_SHA, '--fd', str(fd), '--dev', str(s.st_dev), '--ino', str(s.st_ino)], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, pass_fds=(fd,), env=dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
                require(process.returncode == 0, 'Arm qualification failed: ' + group)
            results = {g: json.loads((OUT / g / 'result.json').read_text()) for g in GROUPS}
            reference = results['H']
            for group, row in results.items():
                for key in ('root_sha256', 'optimizer_lrs', 'scaler', 'native_output_sha256', 'input_sha256', 'actual_training_case'):
                    require(row[key] == reference[key], 'Four-arm common initialization differs: ' + group + ' ' + key)
            status = 'PASS_FOUR_ARM_COMMON_INITIALIZATION'
            dump_new(OUT / 'result.json', dict(at=stamp(), status=status, results=results, temporary_optimizer_updates=0, development_evaluations=0, finite_update_bridge_pending=True, formal_training_started=False))
        except BaseException:
            error = traceback.format_exc()
            raise
        finally:
            dump_new(OUT / 'exited.json', dict(at=stamp(), status=status, error=error, temporary_optimizer_updates=0, formal_training_started=False, automatic_retry=False))
    finally:
        os.close(fd)
if __name__ == '__main__':
    main()
