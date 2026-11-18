from pathlib import Path

def identity(path, data_root):
    path = Path(path)
    root = Path(data_root).resolve(strict=True)
    actual = path.resolve(strict=True)
    if not actual.is_file() or actual.name != path.name or root not in actual.parents:
        raise ValueError('Data case identity/path differs: ' + str(path))
    link, target = (path.lstat(), actual.stat())
    return dict(path=str(path), resolved=str(actual), link=[link.st_dev, link.st_ino, link.st_size, link.st_mtime_ns], target=[target.st_dev, target.st_ino, target.st_size, target.st_mtime_ns])

def bind(config, train, test):
    base = Path(config['fold_cv_dir'])
    return {role: {case: {kind: identity(base / role / kind / (case + '.nii.gz'), Path(config['project_root']) / 'data') for kind in ('imagesTr', 'labelsTr')} for case in cases} for role, cases in (('train', train), ('test', test))}

def check_case(config, case, role, image_path, label_path):
    bound = config['case_file_metadata'][role][case]
    for kind, path in (('imagesTr', image_path), ('labelsTr', label_path)):
        if identity(path, Path(config['project_root']) / 'data') != bound[kind]:
            raise ValueError('Bound case mapping/metadata changed: ' + case)
