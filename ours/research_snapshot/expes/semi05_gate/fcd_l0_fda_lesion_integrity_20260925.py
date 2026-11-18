import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import torch
import torchio as tio
from expes.common.semi_data import CaseListDataset, build_eval_transform, read_case_list
from utils.cascade_train_utils import normalize_batch

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()

def center_crop(image):
    size = 128
    pads = [(max(0, size - d) // 2, max(0, size - d + 1) // 2) for d in image.shape]
    if any((left or right for left, right in pads)):
        image = np.pad(image, pads, mode='constant')
    starts = [(d - size) // 2 for d in image.shape]
    return image[starts[0]:starts[0] + size, starts[1]:starts[1] + size, starts[2]:starts[2] + size]

def read_donor(path):
    raw = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
    valid = raw > 0
    if valid.sum() < 1000:
        raise RuntimeError(f'No valid signal in donor: {path}')
    mean, std = (raw[valid].mean(), raw[valid].std())
    if std <= 0:
        raise RuntimeError(f'Zero intensity std: {path}')
    return ((center_crop(raw) - mean) / std).astype(np.float32)

def fda(source, donor, source_valid):
    a = np.fft.fftshift(np.fft.fftn(source))
    b = np.fft.fftshift(np.fft.fftn(donor))
    index = np.s_[58:71, 58:71, 58:71]
    source_amp = np.abs(a[index])
    donor_amp = np.abs(b[index])
    a[index] = (0.5 * source_amp + 0.5 * donor_amp) * np.exp(1j * np.angle(a[index]))
    transformed = np.fft.ifftn(np.fft.ifftshift(a)).real.astype(np.float32)
    transformed[~source_valid] = source[~source_valid]
    return transformed

def contrast(image, lesion, ring):
    return float(np.median(image[lesion]) - np.median(image[ring]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', choices=('05', '20'), required=True)
    args = parser.parse_args()
    split = Path('expes/splits') / f'by_semi{args.ratio}_cv5/fold2'
    labeled_names = read_case_list(split / 'train_lab.txt')
    unlabeled_names = read_case_list(split / 'train_unlab.txt')
    assert len(labeled_names) == int(args.ratio)
    assert len(labeled_names) + len(unlabeled_names) == 100
    assert not set(labeled_names) & set(unlabeled_names)
    root = Path('data/BY_125_Folds/fold_2/train/imagesTr')
    dataset = CaseListDataset(root.parent.parent, labeled_names, build_eval_transform(128), threshold=-1, split='train')
    norm = tio.ZNormalization(masking_method=lambda image: image > 0)
    rng = np.random.default_rng(2026)
    donor_cache = {}
    output = Path('expes/work_dir/semi05_gate/FCD_L_UNLABELED_STYLE_20260925') / f'lesion_integrity_{args.ratio}'
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for index, source_name in enumerate(labeled_names):
        item = dataset[index]
        source = normalize_batch(item['image'].unsqueeze(0), norm)[0, 0].numpy()
        lesion = item['label'].numpy() > 0
        if lesion.shape != source.shape or lesion.sum() == 0:
            raise RuntimeError(f'Invalid labeled ROI: {source_name}')
        valid = source > source.min() + 0.1
        ring = ndimage.binary_dilation(lesion, iterations=3) & ~lesion & valid
        if ring.sum() < 100:
            raise RuntimeError(f'Insufficient lesion surround: {source_name}')
        before = contrast(source, lesion, ring)
        donor_indices = rng.choice(len(unlabeled_names), size=4, replace=False)
        records = []
        for donor_index in donor_indices:
            donor_name = unlabeled_names[int(donor_index)]
            if donor_name not in donor_cache:
                donor_cache[donor_name] = read_donor(root / f'{donor_name}.nii.gz')
            transformed = fda(source, donor_cache[donor_name], valid)
            after = contrast(transformed, lesion, ring)
            ratio = abs(after) / max(abs(before), 1e-06)
            records.append(dict(donor=donor_name, contrast_after=after, sign_preserved=bool(before * after > 0), contrast_ratio=float(ratio), finite=bool(np.isfinite(transformed).all()), median_abs_delta_valid=float(np.median(np.abs(transformed[valid] - source[valid]))), background_max_change=float(np.max(np.abs(transformed[~valid] - source[~valid]))), transformed_quantiles=[float(v) for v in np.quantile(transformed[valid], [0.1, 0.5, 0.9])]))
        cases.append(dict(case=source_name, original_contrast=before, original_quantiles=[float(v) for v in np.quantile(source[valid], [0.1, 0.5, 0.9])], source_valid_fraction=float(valid.mean()), lesion_voxels=int(lesion.sum()), ring_voxels=int(ring.sum()), donor_records=records, sign_gate=sum((r['sign_preserved'] for r in records)) >= 3, median_contrast_ratio=float(np.median([r['contrast_ratio'] for r in records]))))
        print(f'CASE {args.ratio} {index + 1}/{len(labeled_names)} {source_name}', flush=True)
    sign_count = sum((c['sign_gate'] for c in cases))
    contrast_count = sum((c['median_contrast_ratio'] >= 0.75 for c in cases))
    both_count = sum((c['sign_gate'] and c['median_contrast_ratio'] >= 0.75 for c in cases))
    report = dict(status='COMPLETE', ratio=args.ratio, labeled=len(cases), donors_per_case=4, unlabeled_image_donors=len(set((r['donor'] for c in cases for r in c['donor_records']))), unlabeled_label_files_read=0, test_files_read=0, split_lab_sha256=digest(split / 'train_lab.txt'), split_unlab_sha256=digest(split / 'train_unlab.txt'), script_sha256=digest(__file__), sign_gate_cases=sign_count, contrast_gate_cases=contrast_count, both_gate_cases=both_count, needed_both_gate_cases=4 if args.ratio == '05' else 16, median_contrast_ratio=float(np.median([c['median_contrast_ratio'] for c in cases])), median_valid_intensity_delta=float(np.median([r['median_abs_delta_valid'] for c in cases for r in c['donor_records']])), note='Training-label visual integrity proxy only; no method training or new test Dice.')
    (output / 'cases.json').write_text(json.dumps(cases, indent=2))
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
if __name__ == '__main__':
    main()
