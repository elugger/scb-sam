"""Unscheduled ReLiF-3D training core with explicit paper-aligned repairs.

This module contains no launch, case loader, checkpoint writer or test runner.
The main-table supervisor must bind source/weight/cases and perform independent
real GPU qualification before exposing it through a new frozen runtime version.
"""
import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from relif_components import (case_local_prompt, foundation_probabilities,
                              gated_co_regularization, larc_loss,
                              register_encoder_hook, verify_projection_optimizer)
from relif_native import load_native

NATIVE_ITERATIONS = 30000
IMPLEMENTATION = 'paper-aligned ReLiF-3D repair; not unchanged official execution'


def clock(step, total):
    """Freeze native 30k clocks onto primary input progress, before outcomes."""
    if isinstance(step, bool) or isinstance(total, bool) or not isinstance(step, int) or not isinstance(total, int):
        raise ValueError('Integer primary-pair clock required')
    if total <= 0 or step < 0 or step >= total:
        raise ValueError('Training clock outside fixed endpoint')
    progress = step / total
    native = NATIVE_ITERATIONS * progress
    consistency_epoch = min(200, math.floor(native / 150))
    return dict(step=step, total=total, native_iteration=native, progress=progress,
                overlap_weight=1 - progress, pixel_weight=progress,
                confidence_threshold=.9 * (1 - progress) + .7 * progress,
                consistency_weight=.1 * math.exp(-5 * (1 - consistency_epoch / 200) ** 2),
                larc_enabled=native >= 1000,
                larc_weight=.08 * min(1., native / 2000),
                next_learning_rate=.01 * (1 - progress) ** .9)


def binary_temperature_distribution(foreground_logits, temperature=2.):
    """Softmax([0,z]/tau) preserves the original binary log-odds of SAM."""
    if temperature != 2. or foreground_logits.ndim != 5 or foreground_logits.shape[1] != 1:
        raise ValueError('Pinned continuous SAM logit/temperature contract differs')
    if not bool(torch.isfinite(foreground_logits).all()):
        raise ValueError('Nonfinite foundation logits')
    return torch.cat((torch.zeros_like(foreground_logits), foreground_logits.detach()), 1).div(temperature).softmax(1)


class FrozenMaskPrior:
    """Official mask-prompt forward, returning continuous decoder logits.

    Per-case forwards retain the original batch behavior and cannot broadcast
    one patient's prompt across another patient. No SAM preprocessing is added:
    the official semisam mask branch passes its supplied volume to the encoder.
    """
    def __init__(self, model, identity, expected_shape=(128, 128, 128)):
        self.model, self.identity = model, dict(identity)
        self.expected_shape = tuple(expected_shape)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, images, prompts):
        if (images.ndim != 5 or images.shape[1] != 1 or images.shape != prompts.shape
                or tuple(images.shape[2:]) != self.expected_shape):
            raise ValueError('SAM mask/image shape or case batch mismatch')
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError('Foundation model is not frozen/eval')
        if not bool(torch.isfinite(images).all() and torch.isfinite(prompts).all()):
            raise ValueError('Nonfinite SAM input')
        if not bool(((prompts >= 0) & (prompts <= 1)).all()):
            raise ValueError('Mask prompt outside continuous probability range')
        results = []
        for index in range(images.shape[0]):
            image = images[index:index+1]
            prompt = prompts[index:index+1]
            low_prompt = F.interpolate(prompt.float(), size=tuple(s // 4 for s in self.expected_shape), mode='nearest')
            embedding = self.model.image_encoder(image)
            sparse, dense = self.model.prompt_encoder(points=None, boxes=None, masks=low_prompt)
            logits, _ = self.model.mask_decoder(image_embeddings=embedding,
                image_pe=self.model.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense, multimask_output=False)
            logits = F.interpolate(logits, size=image.shape[2:], mode='trilinear', align_corners=False)
            if logits.shape != image.shape or not bool(torch.isfinite(logits).all()):
                raise ValueError('Continuous SAM decoder contract differs')
            results.append(logits)
        return torch.cat(results, 0)


def bind_original_prior(native, binding, device):
    """No download/fallback. The supervisor supplies verified original identity."""
    if binding.get('model_family') != 'SAM-Med3D_original' or binding.get('variant') != 'vit_b_ori':
        raise ValueError('An explicitly bound original SAM-Med3D checkpoint is required')
    if not binding.get('provenance'):
        raise ValueError('Checkpoint provenance is not bound')
    path = Path(binding['path']).resolve(strict=True)
    if not path.is_file() or path.stat().st_size != binding['bytes']:
        raise ValueError('Original checkpoint size changed')
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != binding['sha256']:
        raise ValueError('Original checkpoint SHA changed')
    checkpoint = torch.load(path, map_location='cpu')
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        raise ValueError('Original checkpoint wrapper differs')
    model = native['sam_builder'](checkpoint=None)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    del checkpoint
    return FrozenMaskPrior(model.to(device), binding)


class ReLiFCore:
    """One native optimizer update consumes exactly lab1 + unlab1.

    Construct fresh for a formal run after qualification; qualification updates
    never count toward the formal primary-input budget.
    """
    def __init__(self, native, model, prior, total, device='cpu', expected_shape=(128, 128, 128)):
        clock(0, total)
        self.native, self.model, self.prior = native, model.to(device), prior
        self.total, self.step_count = total, 0
        self.expected_shape = tuple(expected_shape)
        self.projection = native['ProjectionHead'](out_dim=128).to(device)
        # Original center has 256 channels (default feature_scale=4). Realize
        # LazyLinear before optimizer/checkpoint construction, without cases.
        channels = [m.out_channels for m in self.model.center.modules() if isinstance(m, torch.nn.Conv3d)]
        if not channels:
            raise ValueError('Native encoder feature channels absent')
        with torch.no_grad():
            self.projection(torch.zeros(1, channels[-1], device=device))
        self.optimizer = torch.optim.SGD(list(self.model.parameters()) + list(self.projection.parameters()),
                                         lr=.01, momentum=.9, weight_decay=.0001)
        verify_projection_optimizer(self.optimizer, self.projection)
        self.dice = native['DiceLoss'](2)
        self.features = []
        self.hook = register_encoder_hook(self.model, self.features)
        self.model.train()
        self.projection.train()

    def close(self):
        self.hook.remove()
        self.features.clear()

    def update(self, images, labeled_target, case_ids):
        c = clock(self.step_count, self.total)
        if (images.shape != (2, 1, *self.expected_shape)
                or labeled_target.shape != (1, *self.expected_shape)
                or labeled_target.dtype != torch.long
                or not bool(((labeled_target == 0) | (labeled_target == 1)).all())):
            raise ValueError('Require native lab1/unlab1 with only the labeled target')
        if (set(case_ids) != {'labeled', 'unlabeled'}
                or not all(isinstance(v, str) and v for v in case_ids.values())
                or case_ids['labeled'] == case_ids['unlabeled']):
            raise ValueError('Distinct named labeled/unlabeled identities required')
        if not bool(torch.isfinite(images).all()):
            raise ValueError('Nonfinite training input')
        if not self.model.training or not self.projection.training:
            raise ValueError('Student unexpectedly left training mode')
        verify_projection_optimizer(self.optimizer, self.projection)
        self.features.clear()
        augmented = self.native['sobf_augment'](images, self.step_count, self.total,
            k_mul=41, k_add=21, a_mul_max=.25, a_add_max=.12)
        augmented_logits = self.model(augmented)
        clean_logits = self.model(images)
        if len(self.features) != 2:
            raise ValueError('Expected exactly two live encoder features')
        augmented_features, clean_features = self.features
        self.features.clear()
        probability = augmented_logits.softmax(1)
        prompt = case_local_prompt(labeled_target, probability, 1)
        noisy_images = images + (torch.randn_like(images) * .1).clamp(-.2, .2)
        with torch.no_grad():
            foreground_logits = self.prior(noisy_images, prompt)
            prior_probability = foundation_probabilities(foreground_logits)
            prior_temperature = binary_temperature_distribution(foreground_logits)
        supervised_ce = F.cross_entropy(augmented_logits[:1], labeled_target)
        supervised_dice = self.dice(probability[:1], labeled_target.unsqueeze(1))
        supervised = c['overlap_weight'] * supervised_dice + c['pixel_weight'] * supervised_ce
        # Keep official KL reduction='mean' and tau=2, with stable log_softmax.
        distillation = F.kl_div(F.log_softmax(augmented_logits / 2., dim=1),
                               prior_temperature, reduction='mean')
        cgcr, gate = gated_co_regularization(clean_logits[1:], augmented_logits[1:], c['confidence_threshold'])
        if c['larc_enabled']:
            larc, valid = larc_loss(augmented_features[1:], clean_features[1:], prior_probability[1:], self.projection)
        else:
            # The native warmup sets LARC to a Python zero. Do not attach a
            # zero-valued projection graph: SGD weight decay would then update
            # its parameters before the native 1000-iteration start.
            valid = prior_probability[1:, 1:2].pow(1.5).sum((2, 3, 4)).squeeze(1) > 50
            larc = augmented_logits.sum() * 0.
        loss = supervised + c['consistency_weight'] * distillation + cgcr + c['larc_weight'] * larc
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite ReLiF loss before optimizer')
        self.optimizer.zero_grad()
        loss.backward()
        parameters = list(self.model.parameters()) + list(self.projection.parameters())
        if not all(bool(torch.isfinite(p.grad).all()) for p in parameters if p.grad is not None):
            raise ValueError('Nonfinite ReLiF gradient before optimizer')
        student_gradient = sum(float(p.grad.detach().abs().sum()) for p in self.model.parameters() if p.grad is not None)
        projection_gradient = sum(float(p.grad.detach().abs().sum()) for p in self.projection.parameters() if p.grad is not None)
        if student_gradient <= 0:
            raise ValueError('Student has no training gradient')
        if any(p.grad is not None for p in self.prior.model.parameters()):
            raise ValueError('Frozen foundation received gradients')
        self.optimizer.step()
        for group in self.optimizer.param_groups:
            group['lr'] = c['next_learning_rate']
        self.step_count += 1
        return dict(step=self.step_count, case_ids=dict(case_ids), primary_labeled=1,
                    primary_unlabeled=1, optimizer_updates=1, student_case_forwards=4,
                    foundation_case_forwards=2, sobf_views=2, extra_donor_inputs=0,
                    teacher_case_forwards=0, unlab_gt_in_loss=False, test_evaluations=0,
                    loss=float(loss.detach()), supervised_ce=float(supervised_ce.detach()),
                    supervised_dice=float(supervised_dice.detach()), distillation=float(distillation.detach()),
                    cgcr=float(cgcr.detach()), larc=float(larc.detach()),
                    gate_voxels=int(gate.sum()), larc_valid_cases=int(valid.sum()),
                    student_gradient_l1=student_gradient, projection_gradient_l1=projection_gradient,
                    clock=c, implementation=IMPLEMENTATION)

    def state_dict(self):
        return dict(student=self.model.state_dict(), projection=self.projection.state_dict(),
                    optimizer=self.optimizer.state_dict(), step=self.step_count, total=self.total,
                    shape=self.expected_shape, implementation=IMPLEMENTATION)

    def load_state_dict(self, state):
        if (state['total'] != self.total or tuple(state['shape']) != self.expected_shape
                or state['implementation'] != IMPLEMENTATION
                or not isinstance(state['step'], int) or not 0 <= state['step'] <= self.total):
            raise ValueError('ReLiF state clock/identity differs')
        self.model.load_state_dict(state['student'], strict=True)
        self.projection.load_state_dict(state['projection'], strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.step_count = state['step']
        verify_projection_optimizer(self.optimizer, self.projection)


def build_core(code_root, original_checkpoint_binding, total, device):
    native = load_native(code_root)
    model = native['backbone'](n_classes=2, in_channels=1)
    prior = bind_original_prior(native, original_checkpoint_binding, device)
    return ReLiFCore(native, model, prior, total, device)
