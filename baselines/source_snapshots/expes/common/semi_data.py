"""Semi-supervised datasets driven by shared case-list splits."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torchio as tio
from torch.utils.data import DataLoader, Dataset

from utils.data_loader import Union_Dataloader


def read_case_list(path: str | Path) -> List[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def resolve_case_paths(fold_cv_dir: str | Path, case_stem: str) -> tuple[str, str]:
    fold_cv_dir = Path(fold_cv_dir)
    image_path = fold_cv_dir / "train" / "imagesTr" / f"{case_stem}.nii.gz"
    label_path = fold_cv_dir / "train" / "labelsTr" / f"{case_stem}.nii.gz"
    if not image_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(f"Missing pair for {case_stem} under {fold_cv_dir}")
    return str(image_path), str(label_path)


def resolve_test_paths(fold_cv_dir: str | Path, case_stem: str) -> tuple[str, str]:
    fold_cv_dir = Path(fold_cv_dir)
    image_path = fold_cv_dir / "test" / "imagesTr" / f"{case_stem}.nii.gz"
    label_path = fold_cv_dir / "test" / "labelsTr" / f"{case_stem}.nii.gz"
    if not image_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(f"Missing test pair for {case_stem} under {fold_cv_dir}")
    return str(image_path), str(label_path)


class CaseListDataset(Dataset):
    """Load NIfTI pairs filtered by a case-name list."""

    def __init__(
        self,
        fold_cv_dir: str | Path,
        case_names: Sequence[str],
        transform: Optional[tio.Compose] = None,
        threshold: int = 1000,
        split: str = "train",
    ):
        self.fold_cv_dir = Path(fold_cv_dir)
        self.transform = transform
        self.threshold = threshold
        self.split = split
        self.image_paths: List[str] = []
        self.label_paths: List[str] = []

        resolver = resolve_test_paths if split == "test" else resolve_case_paths
        for case in case_names:
            img, lab = resolver(self.fold_cv_dir, case)
            self.image_paths.append(img)
            self.label_paths.append(lab)

        if not self.image_paths:
            raise ValueError(f"No cases loaded for split={split}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        import SimpleITK as sitk
        import numpy as np

        sitk_image = sitk.ReadImage(self.image_paths[index])
        sitk_label = sitk.ReadImage(self.label_paths[index])

        if sitk_image.GetOrigin() != sitk_label.GetOrigin():
            sitk_image.SetOrigin(sitk_label.GetOrigin())
        if sitk_image.GetDirection() != sitk_label.GetDirection():
            sitk_image.SetDirection(sitk_label.GetDirection())

        subject = tio.Subject(
            image=tio.ScalarImage.from_sitk(sitk_image),
            label=tio.LabelMap.from_sitk(sitk_label),
        )

        if self.transform:
            subject = self.transform(subject)

        if subject.label.data.sum() <= self.threshold:
            return self.__getitem__(np.random.randint(self.__len__()))

        label = subject.label.data.clone().detach().long()
        if label.dim() == 4 and label.shape[0] == 1:
            label = label.squeeze(0)

        return {
            "image": subject.image.data.clone().detach(),
            "label": label,
        }


def build_train_transform(img_size: int, random_flip: bool = True) -> tio.Compose:
    transforms = [tio.ToCanonical(), tio.CropOrPad(mask_name="label", target_shape=(img_size, img_size, img_size))]
    if random_flip:
        transforms.append(tio.RandomFlip(axes=(0, 1, 2)))
    return tio.Compose(transforms)


def build_eval_transform(img_size: int) -> tio.Compose:
    return tio.Compose([
        tio.ToCanonical(),
        tio.CropOrPad(mask_name="label", target_shape=(img_size, img_size, img_size)),
    ])


def make_case_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
    drop_last: bool = False,
) -> Union_Dataloader:
    return Union_Dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


class PseudoLabelDataset(Dataset):
    """In-memory pseudo-label set. repeat=2 matches semi-V L0; repeat=1 is faster."""

    def __init__(self, images, pseudo_labels, masks, repeat: int = 2):
        self.images = images
        self.pseudo_labels = pseudo_labels
        self.masks = masks
        self.repeat = max(1, int(repeat))

    def __len__(self) -> int:
        return len(self.images) * self.repeat

    def __getitem__(self, idx: int):
        i = idx % len(self.images)
        return (
            self.images[i].squeeze(0),
            self.pseudo_labels[i].squeeze(0).long(),
            self.masks[i].squeeze(0).float(),
        )


def make_pseudo_dataloader(
    images,
    pseudo_labels,
    masks,
    batch_size: int,
    num_workers: int = 0,
    repeat: int = 2,
) -> DataLoader:
    ds = PseudoLabelDataset(images, pseudo_labels, masks, repeat=repeat)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )


def split_list_path(split_root: str | Path, fold_idx: int, name: str) -> Path:
    return Path(split_root) / f"fold{fold_idx}" / f"{name}.txt"
