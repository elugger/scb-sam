import os
from pathlib import Path
import re
import sys
from by_protocol import INFRA, GPU_UUID, require

def snapshot():
    sys.path.insert(0, str(INFRA)) if str(INFRA) not in sys.path else None
    from gpu0_readonly import read_gpu0
    row = read_gpu0()
    require(row['uuid'] == GPU_UUID and row['index'] == 0, 'Wrong physical GPU')
    return row

def is_idle(row):
    displays = row.get('graphics', [])
    return row['uuid'] == GPU_UUID and row['index'] == 0 and (row['process_counts']['compute'] == 0) and (len(displays) == row['process_counts']['graphics']) and all((p['name'] == 'Xorg' and 0 <= p['memory_bytes'] <= 32 * 1024 ** 2 for p in displays)) and (row['utilization_percent'] == 0) and (row['memory_bytes'] < 512 * 1024 ** 2)

def ensure_gpu0_idle():
    row = snapshot()
    require(is_idle(row), 'GPU0_BUSY: ' + str(row))
    return row

def own_gpu0_fds():
    paths = []
    for path in Path('/proc/self/fd').iterdir():
        try:
            target = os.readlink(path)
        except FileNotFoundError:
            continue
        if re.fullmatch('/dev/nvidia[0-9]+', target):
            paths.append(target)
    require(set(paths) == {'/dev/nvidia0'}, 'Own CUDA FDs do not exclusively identify GPU0')
    return dict(pid=os.getpid(), physical_device_fds=sorted(set(paths)), gpu_uuid=GPU_UUID)
