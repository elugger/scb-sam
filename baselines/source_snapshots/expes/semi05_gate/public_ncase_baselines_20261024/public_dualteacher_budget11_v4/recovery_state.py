"""Complete value cloning/signatures; torch is imported only by explicit calls."""
import copy
import hashlib
import random


def cpu(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu(v) for v in value)
    return copy.deepcopy(value)


def tree_hash(value):
    import numpy as np
    import torch
    h = hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            h.update(b'tensor'+str(v.dtype).encode()+str(tuple(v.shape)).encode()+v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v, np.ndarray):
            h.update(b'array'+str(v.dtype).encode()+str(v.shape).encode()+v.tobytes())
        elif isinstance(v, dict):
            h.update(b'dict')
            for k in sorted(v, key=repr):
                visit(k); visit(v[k])
        elif isinstance(v, (tuple, list)):
            h.update(type(v).__name__.encode())
            for x in v:
                visit(x)
        else:
            h.update(type(v).__name__.encode()+repr(v).encode()+b'\0')
    visit(value)
    return h.hexdigest()


def rng():
    import numpy as np
    import torch
    return cpu((random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()))


def restore_rng(value):
    import numpy as np
    import torch
    random.setstate(value[0]); np.random.set_state(value[1])
    torch.set_rng_state(value[2]); torch.cuda.set_rng_state_all(value[3])
