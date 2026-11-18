"""Fixed-final HPQ execution; no old formal loop or evaluator is callable.

Only seven measured, SHA-bound execution helpers are reused. Scientific loss,
optimizer, context and EMA calculations remain in the frozen trainer.
"""
import ast
import copy
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import time

from hpq_contract import (ROOT, INIT, INIT_SHA, BASE, STORE, require, sha_file,
                          claim_final_test)
from hpq_inputs import bound_small_bytes
from hpq_resources import (backend, RAM_REQUIRED, FINAL_BOUND, BoundedWriter,
                           save_final, final_reload_guard, validate_final_payload)
from hpq_pairing import raw_groups
from hpq_diagnostics import isolated_evaluation, claim_diagnostic, validate_prior

REFERENCE = ROOT / 'expes/semi05_gate/run_fcd_public05_a1b_20261015.py'
REFERENCE_SHA = 'c50f9a49dd35e761a10c055b6d21453a568564731a9b4b57f00dfb009f1a12ae'
METHODS = ('signature', 'state', 'rng', 'step', 'check_step', 'native_probe', 'record')


def frozen_execution_helpers():
    tree = ast.parse(bound_small_bytes(REFERENCE, REFERENCE_SHA))
    cls, = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == '_Engine']
    selected = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
    require({n.name for n in selected} == set(METHODS), 'Missing frozen execution helper')
    # Do not execute the old module, __init__, bridge, formal, save or imports.
    module = ast.Module(body=[ast.ClassDef(name='MeasuredHelpers', bases=[], keywords=[],
                                          body=selected, decorator_list=[])], type_ignores=[])
    env = dict(copy=copy, hashlib=hashlib, math=math, require=require)
    exec(compile(ast.fix_missing_locations(module), str(REFERENCE), 'exec'), env)
    return env['MeasuredHelpers']


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def load_torch(torch, source):
    kwargs = dict(map_location='cpu')
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    return torch.load(source, **kwargs)


def validate_style(row, allowed):
    for micro in row['microbatches']:
        style = micro['style']
        # Exact field names follow the frozen QualifiedStyleBalanced ledger.
        donors = [style['donor1'], style['donor2']]
        require(len(donors) == 2 and donors[0] != donors[1] and set(donors) <= set(allowed),
                'Illegal style donor pair')
        require(style['complementary_sum'] == 2. and not style['donor_labels_used']
                and style['native_labeled_supervision'] and style['native_teacher']
                and not style['old_fvc_active'], 'Style/mask/GT boundary changed')


class Execution:
    def __init__(self, torch, trainer, phase, gpu, cfg, manifest, output):
        self.torch, self.t, self.cfg = torch, trainer, cfg
        self.spec = dict(group='A1B', phase=phase, gpu=str(gpu), fold=2)
        self.group, self.manifest, self.output = 'A1B', manifest, Path(output)
        self.actual_successes, self.begin = 0, time.monotonic()
        self.input_splits = trainer.input_splits
        from utils.cascade_train_utils import normalize_batch
        self.normalize_batch = normalize_batch

    def append(self, name, row):
        require(Path(name).name == name and name.endswith('.jsonl'), 'Invalid ledger name')
        with (self.output / name).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def raw_groups(self, epoch, record=False):
        yield from raw_groups(self, epoch, record)

    def checked_step(self, items, context=True, reduction='B'):
        row = self.step(items, compute_context=context)
        self.check_step(row, context, reduction)
        validate_style(row, self.input_splits['train_unlab'])
        return row

    def native_initial_identity(self, image):
        require(self.t.matrix_updates == 0, 'Identity check is initialization only')
        before, rng = self.state(), self.rng()
        with self.t.isolated_context(0, 0), self.torch.no_grad(), self.torch.cuda.amp.autocast():
            for root in self.t.roots().values():
                root.eval()
            base = self.t.student(image)
            native = self.t._forward_main_logits(self.t.student, image)
            difference = float((base.float() - native.float()).abs().max())
            require(difference <= 1e-5 and self.torch.equal(base.argmax(1), native.argmax(1)),
                    'Cold H/adapter changed supervised native output')
        require(self.state() == before and self.rng() == rng, 'Initial identity probe changed state')
        return dict(max_abs_difference=difference, tolerance=1e-5, argmax_exact=True,
                    role='train_lab', new_test_forwards=0, successful_updates=0)

    def qualification_payload(self):
        return dict(format='public05_hpq_own_roundtrip_v1', cfg=self.cfg,
                    initializer_sha256=INIT_SHA, updates=self.t.matrix_updates,
                    weights={k: self.t._compact_state(m) for k, m in self.t.roots().items()})

    def roundtrip_locked(self, image, source, digest, size):
        before, rng, updates = self.state(), self.rng(), self.t.matrix_updates
        native = self.native_probe(image)
        roots_before = {k: self.signature(m.state_dict()) for k, m in self.t.roots().items()}
        with self.t.isolated_context(0, 0):
            with self.torch.no_grad():
                for root in self.t.roots().values():
                    next(root.parameters()).add_(.1)
            require(all(self.signature(m.state_dict()) != roots_before[k]
                        for k, m in self.t.roots().items()), 'Not all six roots were perturbed')
            ck = load_torch(self.torch, source)
            require(ck['format'] == 'public05_hpq_own_roundtrip_v1' and ck['cfg'] == self.cfg
                    and ck['initializer_sha256'] == INIT_SHA and ck['updates'] == updates
                    and set(ck['weights']) == set(roots_before), 'Roundtrip metadata mismatch')
            for name, root in self.t.roots().items():
                root.load_state_dict(ck['weights'][name], strict=True)
                require(self.signature(root.state_dict()) == roots_before[name], 'Strict six-root mismatch')
            del ck
        after = self.native_probe(image)
        require(all(self.torch.equal(x, y) for x, y in zip(native, after)), 'Roundtrip native outputs changed')
        require(self.state() == before and self.rng() == rng and self.t.matrix_updates == updates,
                'Roundtrip changed state/RNG/update count')
        return dict(bytes=size, sha256=digest, six_root_roundtrip_exact=True,
                    native_student_teacher_output_exact=True, declared_state_restored=True,
                    rng_unchanged=True, optimizer_scaler_unchanged=True, root_sha256=roots_before)

    def memory_roundtrip(self, image):
        guard = backend()
        with guard.original_lock():
            capacity = guard.capacity_snapshot(RAM_REQUIRED)
            with io.BytesIO() as stream:
                self.torch.save(self.qualification_payload(), BoundedWriter(stream, FINAL_BOUND))
                size = stream.tell()
                digest = hashlib.sha256(stream.getbuffer()).hexdigest()
                stream.seek(0)
                result = self.roundtrip_locked(image, stream, digest, size)
            return dict(storage='BytesIO', capacity=capacity, **result)

    def disk_roundtrip(self, image):
        from hpq_resources import temporary_disk_probe
        require(self.actual_successes == self.t.matrix_updates == 0, 'Disk gate must precede training')
        with temporary_disk_probe(self.qualification_payload) as (path, receipt):
            result = self.roundtrip_locked(image, str(path), receipt['sha256'], receipt['bytes'])
        return dict(status='PASS_UPDATE0_DISK_ROUNDTRIP', updates_before=0, updates_after=0,
                    new_test_forwards=0, temporary_removed=True, permanent_qualification_weight=False,
                    manifest_sha256=sha_file(self.output / 'manifest.json'), disk=receipt, **result)

    def bridge(self):
        iterator = self.raw_groups(0)
        raw = [next(iterator), next(iterator)]
        del iterator
        identity = self.native_initial_identity(raw[0][0]['image_l'])
        write_json(self.output / 'native_initial_identity.json', identity)
        results = {}
        for branch, context, reduction in [('native', False, '0'), ('off', True, '0'),
                                            ('active_B', True, 'B'), ('active_B_repeat', True, 'B')]:
            self.t.initialize_matrix(str(INIT))
            self.t.matrix_c = reduction
            records = []
            self.append('initializations.jsonl', self.t.hpq_initialization_receipt)
            for index, items in enumerate(raw):
                if branch == 'native':
                    before, rng = self.state(), self.rng()
                    component = self.t.probe_context_reductions(items[0]['image_u'], 0, items[0]['step'])
                    require(self.state() == before and self.rng() == rng, 'Component probe changed state')
                    require(all(p.grad is None for root in self.t.roots().values() for p in root.parameters()),
                            'Component probe materialized gradients')
                    self.append('context_vb_components.jsonl', dict(phase=index, **component))
                row = self.checked_step(items, context, reduction)
                self.append('qualification_steps.jsonl', dict(branch=branch, phase=index, **row))
                records.append(self.record(row))
            results[branch] = records
        for index in range(2):
            native, off = results['native'][index], results['off'][index]
            require(native['state'] == off['state'] and native['original'] == off['original'],
                    'Context-off changed original loss/update')
            active = results['active_B'][index]
            require(active == results['active_B_repeat'][index] and active['state'] != native['state'],
                    'Context B exact replay/change failed')
        self.t.initialize_matrix(str(INIT))
        self.t.matrix_c = 'B'
        self.native_probe(raw[0][0]['image_l'])
        # Actual metric/evaluation path on training cases only, no test access.
        evaluation_probe = self.evaluate_cases('train_lab', None, 'bridge_eval_cases.jsonl')
        require(evaluation_probe['test_cases'] == 17, 'Incomplete evaluation isolation qualification')
        row = self.checked_step(raw[0])
        self.append('qualification_steps.jsonl', dict(branch='allowed_train_lab_probe_then_step', phase=0, **row))
        require(self.record(row) == results['active_B'][0], 'Allowed probe changed next update')
        require(self.actual_successes == 9, 'Wrong bridge success budget')
        roundtrip = self.memory_roundtrip(raw[0][0]['image_l'])
        before, updates = self.state(scaler=False), self.t.matrix_updates
        overflow = self.step(raw[0], inject_overflow=True)
        require(overflow['skipped'] and overflow['clip_calls'] == overflow['ema_calls'] == 0
                and overflow['scale_after'] == overflow['scale_before'] / 2
                and self.t.matrix_updates == updates and self.state(scaler=False) == before,
                'Artificial overflow did not preserve weights/optimizer')
        self.append('overflow.jsonl', overflow)
        report = dict(status='OWN_BRIDGE_PASS_DISK_GATE_PENDING', successful_updates=9,
                      artificial_overflows=1, new_test_forwards=0, permanent_qualification_weight=False,
                      native_initial_identity=identity, roundtrip=roundtrip,
                      native_context_off_exact=True, active_replay_exact=True,
                      allowed_probe_then_step_exact=True, six_groups_positive_finite=True,
                      evaluation_then_step_exact=True, evaluation_probe=evaluation_probe,
                      initializer_sha256=INIT_SHA, preparation_sha256=self.manifest['preparation_sha256'],
                      manifest_sha256=sha_file(self.output / 'manifest.json'),
                      wall_seconds=time.monotonic() - self.begin)
        write_json(self.output / 'bridge_report.json', report)
        return report

    def final_payload(self):
        require(self.actual_successes == self.t.matrix_updates == 2376, 'Incomplete formal budget')
        return dict(format='public05_hpq_fixed_final_six_roots_v1', epoch=35, completed_epochs=36,
                    successful_updates=2376, microbatches=11880, test_evaluations_before_save=35,
                    fold=2, seed=2027, selection_rule='fixed_epoch35_native_student',
                    exact_resume_supported=False, branch_routing_measured=False,
                    initializer_sha256=INIT_SHA, cfg=self.cfg,
                    weights={k: self.t._compact_state(m) for k, m in self.t.roots().items()})

    def reload_final(self, saved):
        before = {k: self.signature(m.state_dict()) for k, m in self.t.roots().items()}
        with final_reload_guard(saved['sha256']) as (path, capacity):
            ck = load_torch(self.torch, str(path))
            validate_final_payload(ck)
            require(ck['cfg'] == self.cfg and ck['initializer_sha256'] == INIT_SHA, 'Final metadata differs')
            for name, root in self.t.roots().items():
                root.load_state_dict(ck['weights'][name], strict=True)
                require(self.signature(root.state_dict()) == before[name], 'Final root identity differs')
            del ck
        return dict(strict_reload_verified=True, reloaded_sha256=saved['sha256'],
                    root_sha256=before, capacity=capacity)

    def evaluate_cases(self, role, permit, ledger_name, metadata=None):
        from strict_hpq_data import make_dataset
        from hpq_contract import validate_loaded_case
        from expes.common.seg_metrics import binary_case_metrics, aggregate_metrics, logits_to_binary
        require(role == 'test' or (role == 'train_lab' and self.spec['phase'] == 'bridge'), 'Invalid evaluation role')
        ids = self.input_splits[role]
        rows = []
        # Include dataset construction, transform RNG and all forwards inside restoration.
        with isolated_evaluation(self):
            dataset = make_dataset(str(ROOT / 'data/Public_433_Folds/fold_2'), ids,
                                   role, final_test_permit=permit)
            for index, case_id in enumerate(ids):
                sample = dataset[index]
                validate_loaded_case(sample, case_id, index, True)
                image = self.normalize_batch(sample['image'].unsqueeze(0), self.t.norm_transform).to(self.t.device)
                with self.torch.cuda.amp.autocast():
                    logits = self.t._forward_main_logits(self.t.student, image)
                metrics = binary_case_metrics(logits_to_binary(logits)[0], sample['label'], spacing=(1., 1., 1.))
                row = dict(case_id=case_id, actual_index=index, **(metadata or {}), **metrics)
                self.append(ledger_name, row)
                rows.append(row)
        require(len(rows) == len(ids), 'Incomplete evaluation')
        return dict(**aggregate_metrics(rows), test_cases=len(ids), distance_units='voxel',
                    evaluation_role=role, isolation_verified=True,
                    cases_sha256=sha_file(self.output / ledger_name))

    def diagnostic_test(self, epoch):
        marker = claim_diagnostic(self.output, epoch, self.t.matrix_updates, self.spec['phase'],
                                  self.input_splits['test'])
        started = time.monotonic()
        metrics = self.evaluate_cases('test', marker, 'epoch_{:02d}_test_cases.jsonl'.format(epoch),
                                      dict(epoch=epoch, updates=self.t.matrix_updates))
        result = dict(**metrics, epoch=epoch, updates=self.t.matrix_updates,
                      intent_sha256=sha_file(marker), manifest_sha256=sha_file(self.output / 'manifest.json'),
                      development_test_use=True, seconds=time.monotonic() - started)
        write_json(self.output / 'epoch_{:02d}_test_metrics.json'.format(epoch), result)
        print('HPQ_EPOCH_DIAGNOSTIC', epoch, json.dumps(result), flush=True)
        return result

    def final_test(self, saved, reloaded):
        validate_prior(self.output, 35)
        state = dict(phase='formal', completed_epochs=36, updates=2376, microbatches=11880,
                     saved_epoch=35, test_evaluations_started=35, final_path=saved['path'],
                     final_sha256=saved['sha256'], **reloaded)
        marker = claim_final_test(self.output, state)
        metrics = self.evaluate_cases('test', marker, 'final_test_cases.jsonl', dict(final_sha256=saved['sha256']))
        return dict(**metrics, test_evaluations=36, final_test_evaluations=1, diagnostic_test_evaluations=35)

    def formal(self):
        before, rng = self.state(), self.rng()
        with self.t.isolated_context(0, 0):
            iterator = self.raw_groups(0)
            first = next(iterator)
            image = first[0]['image_l']
            del iterator, first
        require(self.state() == before and self.rng() == rng, 'Startup input changed state')
        identity = self.native_initial_identity(image)
        write_json(self.output / 'native_initial_identity.json', identity)
        write_json(self.output / 'startup_disk_gate.json', self.disk_roundtrip(image))
        del image
        for epoch in range(36):
            started, count = time.monotonic(), 0
            self.t.eval_epoch = epoch
            for items in self.raw_groups(epoch, record=True):
                row = self.checked_step(items)
                count += 1
                require(self.t.matrix_updates == epoch * 66 + count, 'Nonconsecutive update')
                self.append('matrix_steps.jsonl', dict(epoch=epoch, successful_update=self.t.matrix_updates, **row))
                print('HPQ_UPDATE', self.t.matrix_updates, flush=True)
            require(count == 66, 'Wrong epoch budget')
            self.append('train_epochs.jsonl', dict(epoch=epoch, completed_epochs=epoch + 1,
                        updates=self.t.matrix_updates, microbatches=(epoch + 1) * 330,
                        seconds=time.monotonic() - started, test_evaluations=min(epoch, 35)))
            if epoch < 35:
                self.diagnostic_test(epoch)
        # Independent persisted-ledger validation precedes final publication/test.
        from hpq_inspect import validate_training_ledgers
        ledger = validate_training_ledgers(self.output, self.input_splits, complete=True)
        saved = save_final(self.final_payload)
        write_json(self.output / 'final_save.json', saved)
        reloaded = self.reload_final(saved)
        write_json(self.output / 'final_reload.json', reloaded)
        metrics = self.final_test(saved, reloaded)
        result = dict(status='COMPLETE', fold=2, seed=2027, completed_epochs=36, updates=2376,
                      microbatches=11880, metrics=metrics, final=saved, strict_reload_verified=True,
                      initializer_sha256=INIT_SHA, extra_training_budget_vs_supervised=True,
                      branch_routing_measured=False, exact_resume_supported=False, clean_confirmation=False,
                      patient_alias_boundary_verified=False, unique_patient_count=None, gt_roi_used=True,
                      development_test_use=True, diagnostic_test_evaluations=35,
                      ledger=ledger, wall_seconds=time.monotonic() - self.begin)
        write_json(self.output / 'final_metrics.json', result)
        return result


def make_engine(*args, **kwargs):
    helpers = frozen_execution_helpers()
    # Only these methods exist on helpers; neither old bridge nor formal can run.
    cls = type('FixedFinalHPQEngine', (Execution, helpers), {})
    return cls(*args, **kwargs)
