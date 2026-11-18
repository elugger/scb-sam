"""Strict case loading, with disclosed legacy GT ROI but no unlab GT output.

Not yet a GPU-qualified entry point. Caller must bind source/input hashes.
"""
from hpq_contract import require, validate_loaded_case


def role_transform(role, train_factory, eval_factory):
    require(role in ('train_lab', 'train_unlab', 'test'), 'Unknown transform role')
    # Frozen SelfTrainTrainer._build_dataloaders applies flips ONLY to lab.
    # P's own unlabeled strong views are computed later by the frozen loss.
    return (train_factory(128, random_flip=True) if role == 'train_lab'
            else eval_factory(128))


def sample_boundary(image, label, case_id, actual_index, role,
                    origin_adjusted=False, direction_adjusted=False):
    require(role in ('train_lab', 'train_unlab', 'test'), 'Unknown data role')
    sample = dict(image=image, case_id=case_id, actual_index=actual_index,
                  gt_roi_used=True, origin_adjusted=origin_adjusted,
                  direction_adjusted=direction_adjusted)
    if role != 'train_unlab':
        sample['label'] = label
    validate_loaded_case(sample, case_id, actual_index, role != 'train_unlab')
    return sample


def make_dataset(fold_dir, case_ids, role, final_test_permit=None):
    # Test dataset construction is also gated; creating the loader is not proof
    # of permission. Permit must be the exclusive intent created by the runner.
    require(role in ('train_lab', 'train_unlab', 'test'), 'Unknown role')
    from hpq_contract import ROOT
    require(str(fold_dir) == str(ROOT / 'data/Public_433_Folds/fold_2'), 'Only fixed fold2 data allowed')
    if role == 'test':
        from hpq_diagnostics import validate_permit
        validate_permit(final_test_permit, case_ids)
    import SimpleITK as sitk
    import torchio as tio
    from expes.common.semi_data import CaseListDataset, build_train_transform, build_eval_transform

    class StrictDataset(CaseListDataset):
        def __init__(self):
            transform = role_transform(role, build_train_transform, build_eval_transform)
            super().__init__(fold_dir, case_ids, transform=transform, threshold=1000,
                             split='test' if role == 'test' else 'train')
            self.actual_case_ids = tuple(case_ids)

        def __getitem__(self, index):
            require(type(index) is int and 0 <= index < len(self), 'Invalid actual index')
            image = sitk.ReadImage(self.image_paths[index])
            label = sitk.ReadImage(self.label_paths[index])
            origin_adjusted = image.GetOrigin() != label.GetOrigin()
            direction_adjusted = image.GetDirection() != label.GetDirection()
            if origin_adjusted:
                image.SetOrigin(label.GetOrigin())
            if direction_adjusted:
                image.SetDirection(label.GetDirection())
            subject = self.transform(tio.Subject(image=tio.ScalarImage.from_sitk(image),
                                                label=tio.LabelMap.from_sitk(label)))
            # No threshold rejection, recursion or replacement; requested == actual.
            target = None
            if role != 'train_unlab':
                target = subject.label.data.clone().detach().long()
                if target.dim() == 4 and target.shape[0] == 1:
                    target = target.squeeze(0)
                require(bool(((target == 0) | (target == 1)).all()), 'Nonbinary labeled/test target')
            return sample_boundary(subject.image.data.clone().detach(), target,
                                   self.actual_case_ids[index], index, role,
                                   origin_adjusted, direction_adjusted)

    return StrictDataset()
