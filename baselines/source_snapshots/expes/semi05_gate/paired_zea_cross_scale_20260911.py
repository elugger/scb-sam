"""XSA R1: cross-scale coarse-feature conditioning of the successful spatial adapter.

No auxiliary pseudo-label teacher or extra forward. Coarse features are detached
only along the new conditioning edge; original coarse/SAM training remains.
"""
import copy
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from expes.semi05_gate.paired_zea_spatial_20260910 import PairedZEASpatial, SpatialEmbeddingAdapter
from expes.semi05_gate.paired_ghr_fvc_20260909 import PairedGHRFVC
from expes.semi05_gate.paired_rng_data_20260908 import SeededCases, scoped_rng, stream_seed
from expes.semi05_gate.zea_teacher_quality_audit_20260910 import alias_preserving_cpu_state


class CrossScaleEmbeddingAdapter(SpatialEmbeddingAdapter):
    def __init__(self):
        super().__init__(384,16)
        self.cross_projection = nn.Conv3d(64,16,1)
        nn.init.zeros_(self.cross_projection.weight)
        nn.init.zeros_(self.cross_projection.bias)

    def forward(self, embedding, coarse_context):
        hidden = self.down(embedding) + self.cross_projection(coarse_context.detach())
        return embedding + self.up(self.spatial(self.activation(hidden)))


class PairedCrossScale(PairedZEASpatial):
    def __init__(self, cfg):
        super().__init__(cfg)
        old = self.student_zea_adapter
        with scoped_rng(stream_seed(self.pair_seed, 'zea-cross-scale-init'), cuda=True):
            adapter = CrossScaleEmbeddingAdapter().to(self.device)
        missing, unexpected = adapter.load_state_dict(old.state_dict(), strict=False)
        assert set(missing) == {'cross_projection.weight', 'cross_projection.bias'} and not unexpected
        assert all(torch.equal(v, adapter.state_dict()[k]) for k, v in old.state_dict().items())
        self.student_zea_adapter = adapter
        self.teacher_zea_adapter = copy.deepcopy(adapter).requires_grad_(False).eval()
        self._zea_initial_state = self._compact_state(adapter)
        self._zea_gradient_stats = {name: [] for name, _ in adapter.named_parameters()}
        for name, parameter in adapter.named_parameters():
            def record(gradient, name=name):
                values = self._zea_gradient_stats[name]
                if len(values) < 16:
                    scale = max(1., float(self.scaler.get_scale())) if hasattr(self, 'scaler') else 1.
                    values.append(float(gradient.detach().float().norm()) / scale)
                return gradient
            parameter.register_hook(record)
        assert sum(p.numel() for p in adapter.parameters()) == 14176
        self._xsa_context = {}
        self._zea_student_hook.remove(); self._zea_teacher_hook.remove()
        self._xsa_capture_handles = []
        for tag, model in [('student', self.student), ('teacher', self.teacher)]:
            def capture(_module, _inputs, feature, tag=tag):
                assert feature.shape[1:] == (64, 64, 64, 64)
                self._xsa_context[tag] = F.adaptive_avg_pool3d(feature.detach(), (8,8,8))
            stage = model.cascade.coarse_head.network.decoder.stages[self._COARSE_DECODER_STAGE]
            self._xsa_capture_handles.append(stage.register_forward_hook(capture))
            def inject(_module, _inputs, embedding, tag=tag):
                context = self._xsa_context.pop(tag, None)
                if context is None: raise RuntimeError('Missing fresh same-forward coarse feature')
                assert context.shape[0] == embedding.shape[0] and context.shape[-3:] == embedding.shape[-3:]
                if not self._zea_active: return embedding
                head = self.student_zea_adapter if tag == 'student' else self.teacher_zea_adapter
                return head(embedding, context)
            handle = model.cascade.image_encoder.register_forward_hook(inject)
            if tag == 'student': self._zea_student_hook = handle
            else: self._zea_teacher_hook = handle

    @staticmethod
    def _compact_state(module):
        return alias_preserving_cpu_state(module)

    def _save_ucp(self, name, epoch):
        assert name in {'best.pth', 'final.pth', 'storage_preflight.pth'}
        base = Path('/var/tmp/zc_asnet_zea_spatial_20260910').resolve()
        directory = self.checkpoint_dir.resolve()
        assert directory.is_relative_to(base) and directory != base
        target = directory / name
        temporary = directory / (name + '.partial')
        link = self.output_dir / name
        if link.is_symlink():
            assert link.resolve() == target
        elif link.exists():
            raise RuntimeError('Refusing unrelated output checkpoint overwrite')
        with (base / 'save.lock').open('a+') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if temporary.exists():
                raise RuntimeError('Existing interrupted checkpoint; preserve for investigation')
            if shutil.disk_usage(directory).free < 1.15 * 1024**3:
                raise RuntimeError('Insufficient persistent space for atomic checkpoint and reserve')
            payload = dict(student=self._compact_state(self.student),
                student_fclr_head=self._compact_state(self.student_fclr_head),
                teacher_fclr_head=self._compact_state(self.teacher_fclr_head),
                student_zea_adapter=self._compact_state(self.student_zea_adapter),
                teacher_zea_adapter=self._compact_state(self.teacher_zea_adapter),
                zea_format_version=2, architecture='ZEA-CROSS-SCALE-R1',
                best_dice=getattr(self, 'best_dice', None), best_epoch=getattr(self, 'best_epoch', -1),
                saved_epoch=epoch, cfg=self.cfg, exact_resume_supported=False,
                serialization='lossless exact registered tensor aliases')
            with temporary.open('wb') as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            restored = torch.load(temporary, map_location='cpu')
            for key in ('student', 'student_fclr_head', 'teacher_fclr_head',
                        'student_zea_adapter', 'teacher_zea_adapter'):
                assert payload[key].keys() == restored[key].keys()
                assert all(torch.equal(v, restored[key][k]) for k, v in payload[key].items())
            del restored, payload
            digest = hashlib.sha256()
            with temporary.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024**2), b''):
                    digest.update(block)
            size = temporary.stat().st_size
            os.replace(temporary, target)
            fd = os.open(str(directory), os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
            if not link.is_symlink(): link.symlink_to(target)
            self._append('checkpoint_integrity.jsonl', dict(name=name, epoch=epoch,
                path=str(target), bytes=size, sha256=digest.hexdigest(),
                all_tensor_keys_values_bitwise_equal=True, fsync=True,
                persistent_filesystem=True, exact_resume_supported=False))
        return target

    def preflight(self, init_from, expected_floor):
        self._load(init_from)
        data = SeededCases(self.lab_set, self.pair_seed, 'zea-identity')
        batch = data[(0, 0, 0)]
        image, _ = self._prepare_batch({k: v.unsqueeze(0) for k, v in batch.items() if torch.is_tensor(v)})
        self.student.eval(); self.student_fclr_head.eval(); self.student_zea_adapter.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(): active = self._forward_main_logits(self.student, image)
        self._zea_active = False
        try:
            with torch.no_grad(), torch.cuda.amp.autocast(): disabled = self._forward_main_logits(self.student, image)
        finally: self._zea_active = True
        assert torch.equal(active, disabled)
        # Avoid parent's hard-coded non-spatial adapter restore constructor.
        result = PairedGHRFVC.preflight(self, init_from, expected_floor)
        norms = {k: max(v) if v else 0. for k, v in self._zea_gradient_stats.items()}
        assert all(v > 0 for v in norms.values()), norms
        restored = CrossScaleEmbeddingAdapter().to(self.device)
        restored.load_state_dict(self.student_zea_adapter.state_dict())
        assert all(torch.equal(a, b) for a, b in zip(restored.state_dict().values(), self.student_zea_adapter.state_dict().values()))
        del restored
        disk_roundtrip = None
        if len(self.lab_set) == 5:
            self.student.eval(); self.student_fclr_head.eval(); self.student_zea_adapter.eval()
            with torch.no_grad(), torch.cuda.amp.autocast(): before = self._forward_main_logits(self.student, image)
            target = self._save_ucp('storage_preflight.pth', -1)
            self._load(target)
            with torch.no_grad(), torch.cuda.amp.autocast(): after = self._forward_main_logits(self.student, image)
            assert torch.equal(before, after)
            disk_roundtrip = dict(path=str(target), logits_bitwise_equal=True, bytes=target.stat().st_size)
        result.update(control='ZEA-CROSS-SCALE-R1', paired_base='ZEA-SPATIAL-R1', zea_status='PASS',
            adapter_parameters=14176, added_parameters=1040, adapter_lr=self.zea_lr,
            initial_logits_bitwise_equal=True, adapter_gradient_norms=norms,
            adapter_roundtrip_bitwise_equal=True, persistent_disk_roundtrip=disk_roundtrip,
            only_algorithm_change='zero-init projection of detached same-case 64-grid coarse features into spatial adapter bottleneck',
            coarse_feature_resolution=64, pooled_feature_resolution=8,
            cross_projection_gradient_positive=norms['cross_projection.weight'] > 0,
            context_is_detached=True, no_extra_coarse_forward=True,
            unlabeled_gt_extra=False)
        (self.output_dir / 'preflight.json').write_text(json.dumps(result, indent=2))
        return result
