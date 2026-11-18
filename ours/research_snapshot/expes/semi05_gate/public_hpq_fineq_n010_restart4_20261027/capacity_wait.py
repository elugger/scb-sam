import contextlib
import datetime
import json
import os
from pathlib import Path
import time
PREFIX = 'Insufficient filesystem capacity: '

def event(output, status, attempts, detail=None):
    row = dict(status=status, attempts=attempts, at=datetime.datetime.now().astimezone().isoformat(), detail=detail)
    print('CAPACITY_ADMISSION ' + json.dumps(row, sort_keys=True), flush=True)
    if output is not None:
        path = Path(output) / 'capacity_wait.jsonl'
        if any((p.is_symlink() for p in (path, *path.parents))):
            raise ValueError('Capacity wait event symlink rejected')
        with path.open('a') as f:
            f.write(json.dumps(row, sort_keys=True) + '\n')
            f.flush()
            os.fsync(f.fileno())

@contextlib.contextmanager
def admitted_capacity(guard, required_ram, output=None, interval=60, sleep=None):
    sleeper = time.sleep if sleep is None else sleep
    attempts = 0
    while True:
        rejection = None
        with contextlib.ExitStack() as stack:
            stack.enter_context(guard.original_lock())
            try:
                capacity = guard.capacity_snapshot(required_ram)
            except RuntimeError as exc:
                if not str(exc).startswith(PREFIX):
                    raise
                rejection = json.loads(str(exc)[len(PREFIX):])
                if not isinstance(rejection, dict) or rejection.get('margin_bytes', 0) >= 0:
                    raise
                rejection.pop('targets', None)
            if rejection is None:
                if attempts:
                    event(output, 'CAPACITY_READY_CONTINUE_SAME_LIVE_STATE', attempts)
                yield capacity
                return
        attempts += 1
        event(output, 'WAITING_FOR_FILESYSTEM_CAPACITY_NO_TRAINING', attempts, rejection)
        sleeper(interval)
