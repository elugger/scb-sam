"""Stateless augmentation streams for paired WSV controls; preprocessing unchanged."""
from contextlib import contextmanager
import hashlib
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def stream_seed(base, stream, *keys):
    value = repr((int(base), stream, keys)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little")


@contextmanager
def scoped_rng(seed, cuda=False):
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = [torch.cuda.current_device()] if cuda else []
    with torch.random.fork_rng(devices=devices):
        random.seed(seed)
        np.random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        if cuda:
            torch.cuda.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)


class SeededCases(Dataset):
    def __init__(self, original, base_seed, stream):
        self.original, self.base_seed, self.stream = original, base_seed, stream

    def __len__(self):
        return len(self.original)

    def __getitem__(self, token):
        epoch, replay, index = token
        seed = stream_seed(self.base_seed, self.stream, epoch, replay, index)
        with scoped_rng(seed):
            batch = self.original[index]
        # Requested index is recorded; legacy dataset may internally resample
        # low-volume cases. We do not change that inherited behavior here.
        return dict(batch, requested_index=index, augmentation_seed=seed)


def labeled_tokens(count, steps, epoch, seed, refresh):
    order = np.random.RandomState(stream_seed(seed, "lab-order", epoch)).permutation(count).tolist()
    total = steps if refresh else count
    return [(epoch, step // count if refresh else 0, order[step % count])
            for step in range(total)]


def paired_loader(dataset, tokens, seed, stream, workers):
    generator = torch.Generator().manual_seed(stream_seed(seed, "worker", stream, tokens[0][0]))
    return DataLoader(SeededCases(dataset, seed, stream), sampler=tokens,
                      batch_size=1, num_workers=workers, pin_memory=True,
                      generator=generator, persistent_workers=False)


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
