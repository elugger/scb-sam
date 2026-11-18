"""Bound fixed-fold cases; no random replacement and no unlab GT output."""
import hashlib
import json
from pathlib import Path
import random

from budget_protocol import BUDGETS

ROOT = Path('__RECORDED_PROJECT_ROOT__')
HERE = Path(__file__).resolve().parent


def bound_splits(dataset, ratio):
    pins = json.loads((HERE / 'dataset_pins.json').read_text())
    prefix = 'expes/splits/by_semi' if dataset == 'BY' else 'expes/splits/public/public_semi'
    directory = ROOT / (prefix + str(ratio).zfill(2) + '_cv5/fold2')
    result = {}
    for role, pin in pins[dataset + '_' + str(ratio)].items():
        p = directory / (role + '.txt')
        if p.is_symlink():
            raise ValueError('Split symlink')
        raw = p.read_bytes()
        if hashlib.sha256(raw).hexdigest() != pin['sha256']:
            raise ValueError('Actual split changed: ' + role)
        ids = raw.decode().split()
        if len(ids) != pin['count'] or len(set(ids)) != len(ids):
            raise ValueError('Split count/identity differs')
        result[role] = ids
    if (set(result['train_lab']) & set(result['train_unlab']) or
            set(result['test']) & (set(result['train_lab']) | set(result['train_unlab']))):
        raise ValueError('Split leakage')
    return result


def schedule(budget, splits):
    """Cycle-complete unlabeled coverage and explicit legal padding."""
    pairs = []
    for epoch in range(budget.cycles):
        rng = random.Random(2027 + epoch * 104729)
        u = list(splits['train_unlab'])
        l = list(splits['train_lab'])
        rng.shuffle(u)
        rng.shuffle(l)
        extra = [u[(epoch + i) % len(u)] for i in range(budget.replay_per_cycle)]
        for i, case in enumerate(u + extra):
            pairs.append(dict(epoch=epoch, within=i, labeled=l[i % len(l)], unlabeled=case,
                              unlabeled_replay=i >= len(u)))
    if len(pairs) != budget.primary_pairs:
        raise ValueError('Exposure schedule differs')
    return pairs


def case_file_identity(path, data_root):
    """Bind the dataset's legitimate case symlink without permitting substitution."""
    path, data_root = Path(path), Path(data_root).resolve(strict=True)
    resolved = path.resolve(strict=True)
    if (not resolved.is_file() or resolved.name != path.name
            or data_root not in resolved.parents):
        raise ValueError('Case target outside named dataset identity: ' + str(path))
    link, target = path.lstat(), resolved.stat()
    return dict(path=str(path), resolved=str(resolved), symlink=path.is_symlink(),
                link=[link.st_dev, link.st_ino, link.st_size, link.st_mtime_ns],
                target=[target.st_dev, target.st_ino, target.st_size, target.st_mtime_ns])


def bind_case_files(dataset, splits):
    """Metadata only, including test paths; no test image read or evaluation."""
    folder = 'BY_125_Folds' if dataset == 'BY' else 'Public_433_Folds'
    base = ROOT / ('data/' + folder + '/fold_2')
    return {role: {case: {kind: case_file_identity(
        base / ('test' if role == 'test' else 'train') / kind / (case + '.nii.gz'),
        ROOT / 'data') for kind in ('imagesTr', 'labelsTr')} for case in ids}
        for role, ids in splits.items()}


class BoundCases:
    def __init__(self, dataset, splits, test_permit=None, training_flip=True, file_bindings=None):
        self.dataset, self.splits = dataset, splits
        self.test_permit, self.training_flip = test_permit, training_flip
        folder = 'BY_125_Folds' if dataset == 'BY' else 'Public_433_Folds'
        self.folder = ROOT / ('data/' + folder + '/fold_2')
        self.file_bindings = file_bindings if file_bindings is not None else bind_case_files(dataset, splits)

    def load(self, role, case_id):
        if role not in self.splits or case_id not in self.splits[role]:
            raise ValueError('Unknown actual case')
        if role == 'test':
            if self.test_permit is None:
                raise ValueError('Test unopened until terminal permit')
            permit = json.loads(Path(self.test_permit).read_text())
            if permit['test_case_ids'] != self.splits['test'] or permit['test_evaluation'] != 1:
                raise ValueError('Test permit differs')
        import SimpleITK as sitk
        import torch
        import torchio as tio
        sub = 'test' if role == 'test' else 'train'
        image_path = self.folder / sub / 'imagesTr' / (case_id + '.nii.gz')
        label_path = self.folder / sub / 'labelsTr' / (case_id + '.nii.gz')
        for kind, path in (('imagesTr', image_path), ('labelsTr', label_path)):
            if case_file_identity(path, ROOT / 'data') != self.file_bindings[role][case_id][kind]:
                raise ValueError('Actual named case mapping/content metadata changed: ' + str(path))
        image, label = sitk.ReadImage(str(image_path)), sitk.ReadImage(str(label_path))
        if image.GetOrigin() != label.GetOrigin():
            image.SetOrigin(label.GetOrigin())
        if image.GetDirection() != label.GetDirection():
            image.SetDirection(label.GetDirection())
        # Disclosed shared historical GT ROI; no label-dependent case replacement.
        transforms = [tio.ToCanonical(), tio.CropOrPad(mask_name='label', target_shape=(128,128,128))]
        if role != 'test' and self.training_flip:
            transforms.append(tio.RandomFlip(axes=(0,1,2)))
        subject = tio.Compose(transforms)(tio.Subject(image=tio.ScalarImage.from_sitk(image),
                                                     label=tio.LabelMap.from_sitk(label)))
        target = subject.label.data[0].long().clone() if role != 'train_unlab' else None
        if target is not None and not bool(((target == 0) | (target == 1)).all()):
            raise ValueError('Nonbinary actual labeled case')
        return dict(image=subject.image.data.clone(), label=target, case_id=case_id,
                    actual_index=self.splits[role].index(case_id), role=role, gt_roi_used=True,
                    case_replaced=False, unlab_gt_returned=False)
