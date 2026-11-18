import os
import sys
from pathlib import Path
ASNET_ROOT = Path(os.environ.get('ASNET_ROOT', '.')).resolve()
import ctypes as C
import subprocess
from hpq_contract import require
UUID = os.environ['ASNET_GPU_UUID']

class Util(C.Structure):
    _fields_ = [('gpu', C.c_uint), ('memory', C.c_uint)]

class Memory(C.Structure):
    _fields_ = [('total', C.c_ulonglong), ('free', C.c_ulonglong), ('used', C.c_ulonglong)]

class ProcessV2(C.Structure):
    _fields_ = [('pid', C.c_uint), ('usedGpuMemory', C.c_ulonglong), ('gpuInstanceId', C.c_uint), ('computeInstanceId', C.c_uint)]

def read_gpu0():
    lib = C.CDLL('libnvidia-ml.so.1')
    require(lib.nvmlInit_v2() == 0, 'NVML initialization failed')
    try:
        handle, util, memory, index, temperature = (C.c_void_p(), Util(), Memory(), C.c_uint(), C.c_uint())
        require(lib.nvmlDeviceGetHandleByUUID(UUID.encode(), C.byref(handle)) == 0, 'GPU0 UUID unavailable')
        require(lib.nvmlDeviceGetIndex(handle, C.byref(index)) == 0 and index.value == 0, 'Physical GPU0 UUID/index changed')
        require(lib.nvmlDeviceGetUtilizationRates(handle, C.byref(util)) == 0 and lib.nvmlDeviceGetMemoryInfo(handle, C.byref(memory)) == 0 and (lib.nvmlDeviceGetTemperature(handle, 0, C.byref(temperature)) == 0), 'GPU0 telemetry unavailable')
        counts = {}
        for kind in ('Compute', 'Graphics'):
            count = C.c_uint()
            code = getattr(lib, 'nvmlDeviceGet' + kind + 'RunningProcesses_v2')(handle, C.byref(count), None)
            require(code in (0, 7), 'GPU0 process query unavailable')
            counts[kind.lower()] = int(count.value)
        graphics = []
        if counts['graphics']:
            require(counts['graphics'] <= 8, 'Too many GPU0 display processes')
            count = C.c_uint(counts['graphics'])
            entries = (ProcessV2 * count.value)()
            require(lib.nvmlDeviceGetGraphicsRunningProcesses_v2(handle, C.byref(count), entries) == 0, 'GPU0 display process list changed')
            for entry in entries[:count.value]:
                require(entry.pid > 0, 'Unknown display process')
                name = subprocess.check_output(['ps', '-p', str(entry.pid), '-o', 'comm='], text=True, timeout=5).strip()
                graphics.append(dict(pid=int(entry.pid), name=name, memory_bytes=int(entry.usedGpuMemory)))
        return dict(uuid=UUID, index=int(index.value), utilization_percent=int(util.gpu), memory_bytes=int(memory.used), temperature_c=int(temperature.value), process_counts=counts, graphics=graphics)
    finally:
        lib.nvmlShutdown()
