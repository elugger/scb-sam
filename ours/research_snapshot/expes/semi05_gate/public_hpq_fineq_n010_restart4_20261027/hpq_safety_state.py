import copy
import io
import json
import os
from pathlib import Path
import random
import numpy as np
from hpq_contract import STORE, INIT_SHA, require, sha_file
from hpq_resources import backend, RAM_REQUIRED, FINAL_BOUND, BoundedWriter, no_symlink, fsync_directory, probe_read_capacity
FORMAT = 'PUBLIC_N005_FINEQ_FULL_STATE_PRESERVATION_V1'
from capacity_wait import admitted_capacity

def targets():
    return [(Path(STORE) / ('safety_slot%d' % slot) / part / 'final.pth', FINAL_BOUND) for slot in (0, 1) for part in ('weights', 'optimizer')]

def cpu_tree(torch, value, tensor_aliases=None):
    if tensor_aliases is None:
        tensor_aliases = {}
    if torch.is_tensor(value):
        require(value.layout == torch.strided, 'Unsupported safety tensor layout')
        key = (str(value.device), str(value.dtype), value.storage().data_ptr(), value.storage_offset(), tuple(value.shape), tuple(value.stride()))
        if key not in tensor_aliases:
            tensor_aliases[key] = value.detach().cpu().clone()
        return tensor_aliases[key]
    if isinstance(value, dict):
        return {k: cpu_tree(torch, v, tensor_aliases) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)((cpu_tree(torch, v, tensor_aliases) for v in value))
    return copy.deepcopy(value)

def pieces(engine, epoch):
    torch, trainer = (engine.torch, engine.t)
    numpy_state = np.random.get_state()
    common = dict(format=FORMAT, epoch=epoch, updates=trainer.matrix_updates, initializer_sha256=INIT_SHA, cfg=engine.cfg, preparation_sha256=engine.manifest.get('preparation_sha256'), continuation_replay_verified=False, automatic_resume=False)
    weights = dict(common, part='weights', roots=cpu_tree(torch, {k: m.state_dict() for k, m in trainer.roots().items()}), scaler=copy.deepcopy(trainer.scaler.state_dict()), rng=dict(python=random.getstate(), numpy=[numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]], torch=torch.get_rng_state().clone(), cuda=[v.clone() for v in torch.cuda.get_rng_state_all()]), modes={k: [m.training for m in root.modules()] for k, root in trainer.roots().items()})
    optimizer = dict(common, part='optimizer', optimizer=cpu_tree(torch, trainer.optimizer.state_dict()))
    return {'weights': weights, 'optimizer': optimizer}

def write_verified_partial(engine, guard, value, path, capacity):
    from hpq_engine import load_torch
    partial = Path(str(path) + '.partial')
    no_symlink(partial)
    require(not os.path.lexists(partial), 'Existing safety partial must be preserved')
    with partial.open('xb') as stream:
        engine.torch.save(value, BoundedWriter(stream, FINAL_BOUND))
        stream.flush()
        os.fsync(stream.fileno())
    info = partial.stat()
    identity = (info.st_dev, info.st_ino, info.st_size)
    read_capacity = probe_read_capacity(guard, capacity, partial, identity, owned_target=path)
    loaded = load_torch(engine.torch, str(partial))
    require(engine.signature(loaded) == engine.signature(value), 'Saved full-state part differs')
    row = dict(path=str(path), bytes=info.st_size, sha256=sha_file(partial), identity=list(identity), content_sha256=engine.signature(value), capacity_before=capacity, capacity_before_reload=read_capacity)
    del loaded
    return (partial, row)

def memory_probe(engine):
    from hpq_engine import load_torch
    guard = backend()
    before, rng = (engine.state(), engine.rng())
    result, disk, created_directories = ({}, {}, set())
    with admitted_capacity(guard, RAM_REQUIRED, engine.output) as capacity:
        payloads = pieces(engine, -1)
        for name, value in payloads.items():
            with io.BytesIO() as stream:
                engine.torch.save(value, BoundedWriter(stream, FINAL_BOUND))
                size = stream.tell()
                stream.seek(0)
                loaded = load_torch(engine.torch, stream)
                require(engine.signature(loaded) == engine.signature(value), 'Full safety state roundtrip differs')
                result[name] = dict(bytes=size, content_sha256=engine.signature(value))
                del loaded
        for name, value in payloads.items():
            path = Path(STORE) / 'safety_slot0' / name / 'final.pth'
            no_symlink(path)
            require(not os.path.lexists(path), 'Existing safety target before bridge')
            for directory in (path.parent, *path.parent.parents):
                if (directory == Path(STORE) or Path(STORE) in directory.parents) and (not directory.exists()):
                    created_directories.add(directory)
            path.parent.mkdir(parents=True, exist_ok=True)
            part_capacity = guard.capacity_snapshot(RAM_REQUIRED)
            partial, row = write_verified_partial(engine, guard, value, path, part_capacity)
            info = partial.stat()
            require([info.st_dev, info.st_ino, info.st_size] == row['identity'] and sha_file(partial) == row['sha256'], 'Bridge safety probe changed')
            partial.unlink()
            fsync_directory(path.parent)
            disk[name] = dict(row, own_verified_probe_removed=True)
        for directory in sorted(created_directories, key=lambda p: len(p.parts), reverse=True):
            no_symlink(directory)
            require(not any(directory.iterdir()), 'Unexpected bridge probe residue')
            directory.rmdir()
    require(engine.state() == before and engine.rng() == rng, 'Safety probe altered training')
    return dict(status='FULL_STATE_SERIALIZATION_VERIFIED', parts=result, capacity=capacity, disk_roundtrip=dict(status='REGISTERED_OWN_PARTIAL_DISK_ROUNDTRIP_VERIFIED', parts=disk), continuation_replay_verified=False, optimizer_updates=0, new_test_forwards=0)

def save_epoch(engine, epoch):
    from hpq_engine import write_json, load_torch
    require(engine.spec['phase'] == 'formal' and 0 <= epoch < 36, 'Invalid safety save phase')
    require(engine.t.matrix_updates == (epoch + 1) * 68, 'Safety save requires complete epoch')
    slot = epoch % 2
    receipt_path = engine.output / ('safety_epoch_%02d.json' % epoch)
    require(not receipt_path.exists(), 'Safety generation already recorded')
    before, rng = (engine.state(), engine.rng())
    guard = backend()
    with admitted_capacity(guard, RAM_REQUIRED, engine.output) as capacity:
        paths = {name: Path(STORE) / ('safety_slot%d' % slot) / name / 'final.pth' for name in ('weights', 'optimizer')}
        previous = None
        if epoch >= 2:
            previous = json.loads((engine.output / ('safety_epoch_%02d.json' % (epoch - 2))).read_text())
            require(previous['epoch'] == epoch - 2 and previous['slot'] == slot, 'Wrong previous safety slot')
        for name, path in paths.items():
            no_symlink(path)
            require(not os.path.lexists(str(path) + '.partial'), 'Safety partial requires investigation')
            if previous:
                require(path.is_file() and sha_file(path) == previous['parts'][name]['sha256'], 'Prior safety generation changed')
            else:
                require(not os.path.lexists(path), 'Unexpected safety target')
            path.parent.mkdir(parents=True, exist_ok=True)
        payloads = pieces(engine, epoch)
        rows = {}
        for name, value in payloads.items():
            part_capacity = guard.capacity_snapshot(RAM_REQUIRED)
            path = paths[name]
            partial, rows[name] = write_verified_partial(engine, guard, value, path, part_capacity)
            os.replace(partial, path)
            fsync_directory(path.parent)
        require(engine.state() == before and engine.rng() == rng, 'Safety serialization altered state')
        receipt = dict(status='FULL_STATE_SAVED_VERIFIED_NOT_REPLAY_QUALIFIED', epoch=epoch, slot=slot, updates=engine.t.matrix_updates, parts=rows, state_signature=before, rng_signature=rng, capacity_before=capacity, continuation_replay_verified=False, automatic_resume=False, preparation_sha256=engine.manifest['preparation_sha256'])
        write_json(receipt_path, receipt)
        fsync_directory(engine.output)
    return receipt
