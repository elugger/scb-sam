"""ReLiF outer adapter for an independently bound fixed-final controller.

Not registered in the frozen r2/r3 launcher. No CLI, dispatch, download or
fallback initializer. A new launcher must bind this module, the original SAM,
the prospective mapping and all dependencies before its own GPU1 qualification.
"""
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import sys

from budget_protocol import BUDGETS
from fixed_endpoint import exclusive_json
from runtime_data import ROOT, BoundCases, bind_case_files, bound_splits, schedule
from runtime_resources import RAM_REQUIRED, REGISTRY, BoundedWriter, resource_backend, save_terminal, sha

HERE = Path(__file__).resolve().parent
FINAL_BOUND = 32 * 1024 ** 2
MAPPING_POLICY = 'BY_PUBLIC_SHARED_ROI128_POSITIVE_ZSCORE_NATIVE_GEOMETRY_V1_20261019'
IMPLEMENTATION = 'paper-aligned ReLiF-3D repair; not unchanged official execution'
GEOMETRY_NAMES = ('RandomRotFlip', 'RandomCrop', 'ToTensor')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_binding(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), 'Missing/symlink dependency: ' + str(path))
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def prospective_mapping(code_root, normalization_source=None):
    """Metadata-only prospective contract builder, explicitly authorized 2026-10-19.

    Native H5 preparation is not published. The caller cannot silently claim
    value equivalence to those H5 files or drop native geometric augmentation.
    """
    code = Path(code_root).absolute()
    norm = Path(normalization_source) if normalization_source else ROOT / 'utils/cascade_train_utils.py'
    return dict(policy=MAPPING_POLICY, decision='prospective_cross_dataset_mapping_20261019',
        native_h5_numeric_equivalence='unknown_not_claimed', historical_gt_roi_disclosed=True,
        patient_alias_identity_unresolved_disclosed=True, independent_confirmation_claimed=False,
        train_order=['shared_ToCanonical_GT_CropOrPad_128', 'shared_positive_mask_ZNormalization',
                     'official_RandomRotFlip_once', 'official_RandomCrop_128_once'],
        test_order=['shared_ToCanonical_GT_CropOrPad_128', 'shared_positive_mask_ZNormalization'],
        native_crop_padding_retained=True, native_patch=[128, 128, 128],
        native_test_stride=[64, 64, 64], shared_roi_test_windows=1,
        normalization_order_reason='native loaders receive preprocessed H5 before native geometric transforms',
        source_evidence=dict(shared_roi=file_binding(HERE / 'runtime_data.py'),
                             shared_normalization=file_binding(norm),
                             native_la_loader=file_binding(code / 'dataloaders/la.py'),
                             native_brats_loader=file_binding(code / 'dataloaders/brats2019.py'),
                             native_train=file_binding(code / 'ReLiF_3D_train_la.py'),
                             native_test=file_binding(code / 'val_3D_la.py')))


def recipe(code_root, original_sam_binding, normalization_source=None):
    """New-contract fragment only; never mutates/prepares an existing job."""
    binding = copy.deepcopy(original_sam_binding)
    require(binding.get('model_family') == 'SAM-Med3D_original' and binding.get('variant') == 'vit_b_ori',
            'Original SAM-Med3D binding required; no Turbo substitution')
    require(isinstance(binding.get('provenance'), str) and binding['provenance'], 'Original provenance missing')
    require(type(binding.get('bytes')) is int and binding['bytes'] > 0
            and isinstance(binding.get('sha256'), str) and re.fullmatch('[0-9a-f]{64}', binding['sha256']),
            'Original weight size/SHA missing')
    return dict(implementation=IMPLEMENTATION, official_code_root=str(Path(code_root).absolute()),
                original_sam=binding, preprocessing=prospective_mapping(code_root, normalization_source),
                native_labeled_inputs=1, native_unlabeled_inputs=1, native_optimizer_updates=1,
                original_reference_iterations=30000, reference_clock='fractional_primary_input_progress',
                initialization='seed2027_random_official_unet_plus_frozen_original_SAM_Med3D',
                qualification_probe_policy='two_engineering_updates_at_native_warmup_and_LARC_start_then_full_restore',
                prefix_training_updates=0, final_contains=['student', 'projection'],
                checkpoint_bound=FINAL_BOUND, exact_resume_supported=False)


def signature(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def same_state(left, right, torch):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(same_state(left[k], right[k], torch) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(same_state(a, b, torch) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_geometry(mapping, torch, numpy):
    row = mapping['source_evidence']['native_la_loader']
    require(file_binding(row['path']) == row, 'Native geometry source identity changed')
    tree = ast.parse(Path(row['path']).read_text(encoding='utf-8'))
    selected = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in GEOMETRY_NAMES]
    require(len(selected) == len(GEOMETRY_NAMES), 'Native geometry class set changed')
    namespace = dict(torch=torch, np=numpy)
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), row['path'], 'exec'), namespace)
    return [namespace['RandomRotFlip'](), namespace['RandomCrop']((128, 128, 128)), namespace['ToTensor']()]


class ReLiFBackend:
    def __init__(self, contract, output, verify_contract):
        self.contract, self.output, self.verify_contract = contract, Path(output), verify_contract
        self.target = Path(contract['target'])
        self.position, self.actual_rows, self.cycle_rows = 0, [], []
        self.core = None
        self.reloaded_sha, self.test_started = None, False

    def validate(self, endpoint):
        c = self.contract
        require(c.get('method') == 'relif3d' and c.get('gpu') == 1 and c.get('seed') == 2027 and c.get('fold') == 2,
                'Wrong fixed ReLiF method/GPU/seed/fold')
        require(c.get('selection') == 'terminal_after_fixed_exposure' and c.get('test_evaluations') == 1
                and c.get('early_stop') is False and c.get('exact_resume_supported') is False,
                'Wrong ReLiF selection policy')
        require(isinstance(c.get('tag'), str) and re.fullmatch('r[1-9][0-9]*', c['tag']), 'Invalid independent job tag')
        self.budget = next((b for b in BUDGETS if b.dataset == c['dataset'] and b.ratio == c['ratio']), None)
        require(self.budget is not None and c['primary_pairs'] == self.budget.primary_pairs, 'Wrong ReLiF input budget')
        rel = Path('relif3d') / c['dataset'] / ('semi%02d' % c['ratio']) / 'fold2' / c['tag']
        expected = ROOT / 'expes/artifacts/FCD_A_MATRIX_20261006/MAIN_TABLE_FIXED_FINAL_20261019' / rel / 'final.pth'
        require(self.target == expected and c.get('checkpoint_bound') == FINAL_BOUND, 'Wrong ReLiF target/reservation')
        job = ROOT / 'expes/work_dir/semi05_gate/MAIN_TABLE_FIXED_FINAL_20261019' / rel
        require(Path(c['output']) == job / 'formal' and self.output in (job / 'formal', job / 'qualification_backend'),
                'Wrong independent ReLiF output directory')
        require(REGISTRY.is_file() and not REGISTRY.is_symlink(), 'Main-table target registry unavailable')
        registered = json.loads(REGISTRY.read_text(encoding='utf-8'))
        require([row['bound'] for row in registered if row['path'] == str(self.target)] == [FINAL_BOUND],
                'Exact ReLiF terminal target must be enrolled once before qualification')
        require(endpoint.method == 'relif3d' and endpoint.labeled_inputs == endpoint.unlabeled_inputs == self.budget.primary_pairs
                and endpoint.optimizer_updates_per_iteration == 1, 'Endpoint differs from native ReLiF budget')
        r = c['relif_recipe']
        from relif_native import pinned_sources
        pinned_sources(r['official_code_root'])  # All official pins before any CUDA initialization.
        norm = r['preprocessing']['source_evidence']['shared_normalization']['path']
        require(Path(norm).absolute() == (ROOT / 'utils/cascade_train_utils.py').absolute(),
                'Wrong shared normalization implementation')
        require(r == recipe(r['official_code_root'], r['original_sam'], norm), 'Prospective ReLiF recipe/mapping changed')
        # The caller must bind the exact new module set; old r2/r3 contracts
        # cannot satisfy this merely by changing their method name.
        source_rows = c['sources']
        paths = {str(Path(row['path']).absolute()): row for row in source_rows}
        require(len(paths) == len(source_rows), 'Duplicate ReLiF source binding')
        required = [HERE / name for name in ('relif_backend.py', 'relif_runtime.py', 'relif_native.py',
                    'relif_components.py', 'relif_official_pin.json', 'runtime_data.py', 'runtime_resources.py',
                    'fixed_endpoint.py', 'budget_protocol.py', 'dataset_pins.json')]
        required += [Path(row['path']) for row in r['preprocessing']['source_evidence'].values()]
        required += [ROOT / 'expes/common/seg_metrics.py']
        for path in required:
            bound = paths.get(str(path.absolute()))
            require(bound is not None and sha(path) == bound['sha256'], 'Unbound/changed ReLiF dependency: ' + str(path))
        require(not self.target.exists() and not Path(str(self.target) + '.partial').exists(),
                'Existing final/partial must not be reused')
        return r

    def prepare(self, endpoint):
        require(self.core is None, 'Prepare is single-use per backend')
        self.verify_contract(self.contract)
        r = self.validate(endpoint)
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '1', 'New launcher must exclusively bind physical GPU1')
        self.splits = bound_splits(self.contract['dataset'], self.contract['ratio'])
        require(self.splits == self.contract['splits'] and tuple(self.splits['test']) == endpoint.test_case_ids,
                'ReLiF actual split/test identities changed')
        require(bind_case_files(self.contract['dataset'], self.splits) == self.contract['case_files'],
                'ReLiF actual case file identities changed')
        self.pairs = schedule(self.budget, self.splits)
        self.data = BoundCases(self.contract['dataset'], self.splits, training_flip=False,
                               file_bindings=self.contract['case_files'])
        guard = resource_backend(self.target, FINAL_BOUND)
        with guard.original_lock():
            capacity = guard.capacity_snapshot(RAM_REQUIRED)
            import torch
            import numpy as np
            import torchio as tio
            from relif_runtime import build_core
            self.torch, self.np = torch, np
            torch.set_num_threads(2)
            random.seed(2027)
            np.random.seed(2027)
            torch.manual_seed(2027)
            torch.cuda.manual_seed_all(2027)
            self.device = torch.device('cuda:0')
            self.core = build_core(r['official_code_root'], r['original_sam'], self.budget.primary_pairs, self.device)
            self.initial = dict(student=signature(self.core.model), projection=signature(self.core.projection))
            tensor_bytes = sum(t.numel() * t.element_size() for model in (self.core.model, self.core.projection)
                               for t in model.state_dict().values())
            require(tensor_bytes < FINAL_BOUND - 1024 ** 2, 'Native terminal exceeds frozen reservation margin')
        sys.path.insert(0, str(ROOT))
        norm_path = r['preprocessing']['source_evidence']['shared_normalization']['path']
        self.normalize_batch = load_file('_relif_shared_normalization', norm_path).normalize_batch
        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
        self.geometry = load_geometry(r['preprocessing'], torch, np)
        self.metrics = load_file('_relif_shared_metrics', ROOT / 'expes/common/seg_metrics.py')
        return dict(source_pins_verified=True, input_pins_verified=True, initialization_verified=True,
                    resource_gate_passed=True, test_unopened=True, capacity=capacity,
                    initial_state_signatures=self.initial, native_terminal_tensor_bytes=tensor_bytes,
                    original_sam_checkpoint=r['original_sam'], preprocessing=r['preprocessing'],
                    implementation=IMPLEMENTATION, initialization=r['initialization'], prefix_training_updates=0,
                    clock_mapping=r['reference_clock'], native_inputs_per_stream=1, native_optimizer_updates=1)

    def transform(self, sample, training):
        torch = self.torch
        require(sample['role'] in ('train_lab', 'train_unlab', 'test'), 'Unknown case role')
        require(sample['image'].shape == (1, 128, 128, 128), 'Shared ROI shape changed')
        require(sample.get('case_replaced') is False and sample.get('unlab_gt_returned') is False,
                'Replaced case or returned unlabeled target')
        if sample['role'] == 'train_unlab':
            require(sample['label'] is None, 'Unlabeled GT crossed adapter boundary')
        else:
            require(sample['label'] is not None, 'Missing labeled/test target')
        image = self.normalize_batch(sample['image'].unsqueeze(0), self.norm_transform)[0]
        require(bool(torch.isfinite(image).all()), 'Nonfinite normalized ReLiF input')
        target = sample['label']
        if training:
            # Geometry is label-value independent in the pinned official code.
            # An all-zero placeholder supplies only shape for the unlabeled case.
            label = target if target is not None else torch.zeros((128, 128, 128), dtype=torch.long)
            value = dict(image=image[0].numpy(), label=label.numpy())
            for transform in self.geometry:
                value = transform(value)
            image = value['image']
            target = value['label'] if target is not None else None
        require(image.shape == (1, 128, 128, 128) and bool(torch.isfinite(image).all()), 'Native crop shape/values changed')
        return image, target

    def consume_pair(self, index):
        pair = self.pairs[index]
        left = self.data.load('train_lab', pair['labeled'])
        right = self.data.load('train_unlab', pair['unlabeled'])
        for sample, role, case in ((left, 'train_lab', pair['labeled']), (right, 'train_unlab', pair['unlabeled'])):
            require(sample['role'] == role and sample['case_id'] == case
                    and sample['actual_index'] == self.splits[role].index(case), 'Actual ReLiF case substituted')
        a, target = self.transform(left, True)
        b, unlabeled = self.transform(right, True)
        require(unlabeled is None, 'Unlabeled GT entered ReLiF loss')
        require(self.core.step_count == index, 'Core and primary input clocks differ')
        native = self.core.update(self.torch.stack((a, b)).to(self.device), target.unsqueeze(0).to(self.device),
                                  dict(labeled=pair['labeled'], unlabeled=pair['unlabeled']))
        self.torch.cuda.synchronize()
        require(native['optimizer_updates'] == 1 and native['primary_labeled'] == native['primary_unlabeled'] == 1,
                'Native ReLiF accounting differs')
        losses = {key: native[key] for key in ('loss', 'supervised_ce', 'supervised_dice', 'distillation', 'cgcr', 'larc')}
        require(all(math.isfinite(value) for value in losses.values()), 'Nonfinite ReLiF receipt')
        gradient_norm = math.sqrt(sum(float(p.grad.detach().double().square().sum())
            for model in (self.core.model, self.core.projection) for p in model.parameters() if p.grad is not None))
        require(math.isfinite(gradient_norm) and gradient_norm > 0, 'Invalid joint ReLiF gradient norm')
        return dict(labeled_inputs=1, unlabeled_inputs=1, successful_optimizer_updates=1,
                    finite_loss=True, finite_nonzero_gradients=native['student_gradient_l1'] > 0,
                    teacher_no_grad=not any(p.grad is not None for p in self.core.prior.model.parameters()),
                    skip_count=0, actual_case_ids_verified=True, unlabeled_gt_in_loss=False,
                    actual_pair=pair, completed_primary_pairs=index + 1, losses=losses,
                    gradient_norm=gradient_norm,
                    gradient_l1=native['student_gradient_l1'], projection_gradient_l1=native['projection_gradient_l1'],
                    native_receipt=native)

    def train_iteration(self, remaining):
        require(remaining == dict(labeled_inputs=self.budget.primary_pairs - self.position,
                                 unlabeled_inputs=self.budget.primary_pairs - self.position)
                and self.position < self.budget.primary_pairs, 'Remaining ReLiF budget/cursor differs')
        receipt = self.consume_pair(self.position)
        self.actual_rows.append(receipt['actual_pair'])
        self.position += 1
        if self.position % self.budget.primary_pairs_per_cycle == 0:
            epoch = self.position // self.budget.primary_pairs_per_cycle - 1
            rows = self.actual_rows[-self.budget.primary_pairs_per_cycle:]
            require(set(p['labeled'] for p in rows) == set(self.splits['train_lab'])
                    and set(p['unlabeled'] for p in rows) == set(self.splits['train_unlab'])
                    and sum(p['unlabeled_replay'] for p in rows) == self.budget.replay_per_cycle,
                    'Actual ReLiF cycle coverage differs')
            self.cycle_rows.append(dict(epoch=epoch, actual_case_coverage_verified=True,
                                        unlabeled_replay=self.budget.replay_per_cycle))
        return receipt

    def verify_training_coverage(self):
        require(self.position == self.core.step_count == self.budget.primary_pairs
                and self.actual_rows == self.pairs and len(self.cycle_rows) == 36,
                'Incomplete actual ReLiF input/optimizer coverage')
        return dict(actual_case_coverage_verified=True, cycles=self.cycle_rows,
                    primary_labeled_inputs=self.position, primary_unlabeled_inputs=self.position,
                    native_optimizer_updates=self.position, student_case_forwards=4 * self.position,
                    foundation_case_forwards=2 * self.position, sobf_views=2 * self.position,
                    prefix_training_updates=0, extra_donor_inputs=0)

    def save_terminal(self, counts):
        self.verify_training_coverage()
        require(counts == dict(labeled_inputs=self.position, unlabeled_inputs=self.position,
                              iterations=self.position, successful_optimizer_updates=self.position), 'Terminal counts differ')
        self.final_signatures = dict(student=signature(self.core.model), projection=signature(self.core.projection))
        require(self.final_signatures['student'] != self.initial['student'], 'Native terminal student unchanged')
        return save_terminal(self.target, dict(format='relif_fixed_terminal_v1', method='relif3d',
            counts=counts, selection='fixed_exposure_terminal', signatures=self.final_signatures,
            student=self.core.model.state_dict(), projection=self.core.projection.state_dict(),
            original_sam=self.contract['relif_recipe']['original_sam'],
            preprocessing=self.contract['relif_recipe']['preprocessing'], implementation=IMPLEMENTATION))

    def strict_reload(self, artifact):
        require(self.reloaded_sha is None, 'Strict terminal reload is single-use')
        require(Path(artifact['path']) == self.target, 'Wrong terminal artifact path')
        guard = resource_backend(self.target, FINAL_BOUND)
        with guard.original_lock():
            guard.capacity_snapshot(RAM_REQUIRED)
            require(sha(self.target) == artifact['sha256'], 'Terminal changed before reload')
            value = self.torch.load(str(self.target), map_location=self.device)
            require(value['method'] == 'relif3d' and value['implementation'] == IMPLEMENTATION
                    and value['signatures'] == self.final_signatures
                    and value['original_sam'] == self.contract['relif_recipe']['original_sam']
                    and value['preprocessing'] == self.contract['relif_recipe']['preprocessing'], 'Reload metadata differs')
            self.core.model.load_state_dict(value['student'], strict=True)
            self.core.projection.load_state_dict(value['projection'], strict=True)
            require(signature(self.core.model) == self.final_signatures['student']
                    and signature(self.core.projection) == self.final_signatures['projection'], 'Reload native tensor identity differs')
        receipt = dict(strict=True, sha256=artifact['sha256'], native_student_identity_verified=True,
                       native_projection_identity_verified=True, signatures=self.final_signatures)
        exclusive_json(self.output / 'relif_reload_verified.json', receipt)
        self.reloaded_sha = artifact['sha256']
        return receipt

    def test(self, artifact, permit):
        require(not self.test_started and self.reloaded_sha == artifact['sha256'], 'Missing reload or repeated test')
        require(Path(permit) == self.output / 'final_test_intent.json' and sha(self.target) == artifact['sha256'],
                'Wrong terminal test permit/artifact')
        with Path(permit).open(encoding='utf-8') as stream:
            authorization = json.load(stream)
        require(authorization == dict(selection='terminal_after_fixed_exposure', final_sha256=artifact['sha256'],
                    test_case_ids=self.splits['test'], test_evaluation=1), 'Unique test permit differs')
        # Mark before opening/reading cases. Any exception preserves an unusable
        # partial attempt; a new process also rejects the exclusive case ledger.
        self.test_started = True
        data = BoundCases(self.contract['dataset'], self.splits, test_permit=permit, training_flip=False,
                          file_bindings=self.contract['case_files'])
        self.core.hook.remove()
        self.core.features.clear()
        self.core.model.eval()
        rows = []
        with (self.output / 'final_test_cases.jsonl').open('x', encoding='utf-8') as stream:
            for index, case in enumerate(self.splits['test']):
                sample = data.load('test', case)
                require(sample['case_id'] == case and sample['actual_index'] == index, 'Actual test identity changed')
                image, target = self.transform(sample, False)
                with self.torch.no_grad():
                    pred = self.metrics.logits_to_binary(self.core.model(image.unsqueeze(0).to(self.device)))[0]
                row = dict(case_id=case, actual_index=index,
                           **self.metrics.binary_case_metrics(pred, target, spacing=(1., 1., 1.)))
                require(all(math.isfinite(row[k]) and row[k] >= 0 for k in ('dice', 'jaccard', 'hd95', 'asd'))
                        and row['dice'] <= 1 and row['jaccard'] <= 1, 'Invalid terminal case metric')
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                rows.append(row)
        require(sha(self.target) == artifact['sha256'], 'Terminal changed during unique test')
        return dict(case_ids=[r['case_id'] for r in rows], case_dice=[r['dice'] for r in rows],
                    **{key: sum(r[key] for r in rows) / len(rows) for key in ('dice', 'jaccard', 'hd95', 'asd')},
                    final_sha256=artifact['sha256'], test_evaluations=1, distance_unit='voxel')

    def qualify(self, endpoint):
        preparation = self.prepare(endpoint)
        torch, np = self.torch, self.np
        state = copy.deepcopy(self.core.state_dict())
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state())
        active = math.ceil(1000 * self.budget.primary_pairs / 30000)
        receipts = []
        for index, phase in ((0, 'native_warmup'), (active, 'LARC_start_engineering_probe')):
            # Explicitly recorded engineering clock probes, never formal resume
            # or consumed formal budget. Fresh formal worker reinitializes fully.
            self.core.step_count = index
            row = self.consume_pair(index)
            row.update(qualification_phase=phase, qualification_schedule_index=index, formal_budget_inputs=0)
            receipts.append(row)
        require(receipts[1]['native_receipt']['clock']['larc_enabled'] is True
                and receipts[1]['native_receipt']['larc_valid_cases'] > 0
                and receipts[1]['projection_gradient_l1'] > 0,
                'Actual active LARC probe had no valid prior/projection gradient; preserve and investigate')
        self.core.load_state_dict(state)
        for model in (self.core.model, self.core.projection):
            for parameter in model.parameters():
                parameter.grad = None
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        torch.cuda.set_rng_state(rng[3])
        require(same_state(self.core.state_dict(), state, torch), 'Own qualification state restore differs')
        current_np = np.random.get_state()
        require(random.getstate() == rng[0] and current_np[0] == rng[1][0]
                and np.array_equal(current_np[1], rng[1][1]) and current_np[2:] == rng[1][2:]
                and torch.equal(torch.get_rng_state(), rng[2]) and torch.equal(torch.cuda.get_rng_state(), rng[3]),
                'Own qualification RNG restore differs')
        probe = self.target.parent / 'own_relif_update0_probe.pth.partial'
        guard = resource_backend(self.target, FINAL_BOUND)
        with guard.original_lock():
            guard.capacity_snapshot(RAM_REQUIRED)
            require(not probe.exists() and not any(p.is_symlink() for p in (probe, *probe.parents)), 'Existing/symlink own probe')
            probe.parent.mkdir(parents=True, exist_ok=True)
            with probe.open('xb') as stream:
                torch.save(state, BoundedWriter(stream, 96 * 1024 ** 2))
                stream.flush()
                os.fsync(stream.fileno())
            identity = (probe.stat().st_dev, probe.stat().st_ino, probe.stat().st_size)
            value = torch.load(str(probe), map_location=self.device)
            self.core.load_state_dict(value)
            require(same_state(self.core.state_dict(), state, torch), 'Own native disk roundtrip differs')
            require(identity == (probe.stat().st_dev, probe.stat().st_ino, probe.stat().st_size), 'Own probe identity changed')
            probe.unlink()  # Only this new, strictly verified own probe, inside original lock.
        return dict(status='QUALIFIED', preparation=preparation, actual_extra_engineering_updates=2,
                    actual_extra_engineering_primary_inputs_per_stream=2, formal_training_updates=0,
                    qualification_clock_probe_indices=[0, active], new_test_evaluations=0,
                    restored_model_optimizer_rng=True, actual_disk_roundtrip=True, receipts=receipts,
                    native_state_signatures=self.initial, implementation=IMPLEMENTATION)
