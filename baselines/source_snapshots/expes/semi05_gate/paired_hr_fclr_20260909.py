"""HR-FCLR: native 32^3 SAM path plus zero-initialized 64^3 residual.

The base upsampling path remains exact; only the residual is upsampled from 64.
"""
import torch
import torch.nn.functional as F
from torch.cuda import amp
from expes.semi05_gate.paired_fclr_control_20260909 import PairedFCLRControl
from expes.semi05_gate.final_logit_contextual_residual_wsv_ucp import FinalLogitContextualResidualHead
from expes.semi05_gate.paired_rng_data_20260908 import scoped_rng, stream_seed


class HighResolutionResidualHead(FinalLogitContextualResidualHead):
    def forward(self, coarse_feature, sam_embedding, low_res_logit):
        assert tuple(coarse_feature.shape[-3:]) == (64,64,64)
        assert tuple(low_res_logit.shape[-3:]) == (32,32,32)
        conditioning = F.interpolate(low_res_logit.detach().float(), (64,64,64),
                                     mode='trilinear', align_corners=False)
        _, delta = super().forward(coarse_feature, sam_embedding, conditioning)
        return delta


class PairedHRFCLR(PairedFCLRControl):
    _COARSE_DECODER_STAGE = 3

    def __init__(self, cfg):
        super().__init__(cfg)
        with scoped_rng(stream_seed(self.pair_seed,'hr-fclr-head-init'),cuda=True):
            kwargs=dict(coarse_channels=64,sam_channels=int(self.student.cascade.prompt_encoder.embed_dim),
                        projection_channels=8,hidden_channels=16,delta_max=.5)
            self.student_fclr_head=HighResolutionResidualHead(**kwargs).to(self.device)
            self.teacher_fclr_head=HighResolutionResidualHead(**kwargs).to(self.device)
        self.teacher_fclr_head.requires_grad_(False)
        self.teacher_fclr_head.eval()
        self._update_teacher_fclr(0.)
        self._register_fclr_gradient_hooks()

    def _forward_with_fclr(self, model, head, image):
        assert not bool(getattr(model,'gate_mask_prompt',False))
        head.train(model.training and model is self.student)
        prompt, coarse_logits, coarse_feature=self._coarse_forward_with_feature(model,image)
        cascade=model.cascade
        embedding=cascade.image_encoder(image)
        sparse,dense=cascade.prompt_encoder(points=None,boxes=None,masks=prompt)
        low_res,_=cascade.mask_decoder(image_embeddings=embedding,
            image_pe=cascade.prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,multimask_output=False)
        delta=head(coarse_feature,embedding,low_res)
        base_full=F.interpolate(low_res,size=image.shape[-3:],mode='trilinear',align_corners=False)
        # Do not upsample the base 32->64->128: that would change the initial
        # prediction even with a zero head. Add only the 64-grid residual.
        delta_full=F.interpolate(delta.float(),size=image.shape[-3:],mode='trilinear',align_corners=False)
        refined_full=(base_full.float()+delta_full).to(base_full.dtype)
        base_logits=model._fg_to_logits(base_full)
        logits=model._fg_to_logits(refined_full)
        if model is self.student and model.training:
            self._fclr_last_student_delta=delta
            self._fclr_stats.append(dict(delta_mean=float(delta.detach().abs().mean()),
                delta_max=float(delta.detach().abs().max()),
                delta_active=float((delta.detach().abs()>=1e-3).float().mean()),
                base_logit_mean=float(low_res.detach().float().abs().mean())))
        return logits,coarse_logits,base_logits,delta

    def preflight(self, init_from, expected_floor):
        import json
        result=super().preflight(init_from,expected_floor)
        result.update(control='HR-FCLR-R1',paired_base='C0',residual_resolution=64,
                      base_resolution=32,coarse_stage=3,
                      head_parameters=sum(p.numel() for p in self.student_fclr_head.parameters()))
        (self.output_dir/'preflight.json').write_text(json.dumps(result,indent=2))
        return result
