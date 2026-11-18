import argparse
import json
import os
from pathlib import Path
import time
import traceback
from by_protocol import *
from prepare_by5 import dump_new, stamp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle-sha', required=True)
    parser.add_argument('--preparation-sha', required=True)
    parser.add_argument('--nonce', required=True)
    parser.add_argument('--claim-fd', type=int, required=True)
    parser.add_argument('--claim-device', type=int, required=True)
    parser.add_argument('--claim-inode', type=int, required=True)
    a = parser.parse_args()
    require(__debug__ and Path(__file__).resolve() == Path(SRC / 'supervise_by5.py'), 'Wrong worker')
    os.chdir(str(ROOT))
    control = Path(CONTROL)
    status, error = ('FAILED_BEFORE_TRAINING', None)
    began = time.monotonic()
    try:
        from launch_by5 import verify_preparation
        cfg, prep = verify_preparation(a.bundle_sha, a.preparation_sha)
        intent = json.loads((control / 'supervised.intent.json').read_text())
        require(intent['nonce'] == a.nonce and intent['source_manifest_sha256'] == a.bundle_sha and (intent['preparation_sha256'] == a.preparation_sha), 'Intent does not bind this worker')
        from train_by5 import validate_gpu_claim, append_jsonl, run
        claim = validate_gpu_claim(cfg, 0, a.claim_fd, a.claim_device, a.claim_inode)
        dump_new(control / 'supervised.started.json', dict(at=stamp(), pid=os.getpid(), nonce=a.nonce, proc_start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19], stage='ADMISSION_NOT_TRAINING', claim=claim, config_sha256=prep['config_sha256']))
        from by_runtime import snapshot, is_idle
        from by_resources import original_lock, capacity_snapshot
        announced_wait = False
        while True:
            verify_bundle(a.bundle_sha)
            require(not Path(cfg['output_dir']).exists() and (not Path(cfg['artifact_dir']).exists()), 'Output appeared during wait')
            with original_lock():
                capacity = capacity_snapshot(config=cfg)
                gpu = snapshot()
                ready = is_idle(gpu)
                if ready:
                    append_jsonl(control / 'supervised.admission.jsonl', dict(at=stamp(), state='READY', waiting_seconds=time.monotonic() - began, gpu0=gpu, capacity=capacity))
                    break
            if not announced_wait:
                append_jsonl(control / 'supervised.admission.jsonl', dict(at=stamp(), state='WAITING_GPU0', successful_updates=0, gpu0=gpu, releases_original_lock_while_waiting=True))
                announced_wait = True
            require(intent['wait_for_gpu'], 'GPU0 became busy without wait authorization')
            require(time.monotonic() - began < intent['maximum_gpu_wait_seconds'], 'GPU admission wait timeout')
            time.sleep(60)
        cfg, prep = verify_preparation(a.bundle_sha, a.preparation_sha)
        dump_new(control / 'supervised.training_entered.json', dict(at=stamp(), pid=os.getpid(), nonce=a.nonce, config_sha256=prep['config_sha256'], training_function_calls=1))
        status = 'TRAINING_FAILED'
        run(cfg, prep['config_sha256'], 0, gpu_claim=claim)
        result = json.loads((Path(cfg['output_dir']) / 'final_metrics.json').read_text())
        require(result['status'] == 'COMPLETE' and result['successful_updates'] == 300 and (result['completed_epochs'] == 60), 'Supervisor did not complete fixed budget')
        status = 'SUPERVISION_COMPLETE_PENDING_FOUR_ARM_QUALIFICATION'
    except BaseException:
        error = traceback.format_exc()
        raise
    finally:
        dump_new(control / 'supervised.exited.json', dict(at=stamp(), pid=os.getpid(), nonce=a.nonce, status=status, error=error, elapsed_seconds=time.monotonic() - began, automatic_restart=False, downstream_training_launched=False))
        os.close(a.claim_fd)
if __name__ == '__main__':
    main()
