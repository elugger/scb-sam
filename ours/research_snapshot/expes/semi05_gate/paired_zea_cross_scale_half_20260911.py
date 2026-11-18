import json
import torch
import torch.nn.functional as F
from expes.semi05_gate.paired_zea_cross_scale_20260911 import PairedCrossScale

class PairedCrossScaleHalf(PairedCrossScale):

    def __init__(self, cfg):
        assert cfg['semi']['xsa_cross_gain'] == 0.5
        super().__init__(cfg)
        self._xsa_half_handles = []
        for adapter in (self.student_zea_adapter, self.teacher_zea_adapter):
            self._xsa_half_handles.append(adapter.cross_projection.register_forward_hook(lambda _module, _inputs, value: value * 0.5))

    def preflight(self, init_from, expected_floor):
        result = super().preflight(init_from, expected_floor)
        checks = {}
        with torch.no_grad():
            for name, adapter in [('student', self.student_zea_adapter), ('teacher', self.teacher_zea_adapter)]:
                layer = adapter.cross_projection
                value = torch.ones((1, 64, 8, 8, 8), device=self.device)
                actual = layer(value)
                expected = F.conv3d(value, layer.weight, layer.bias) * 0.5
                assert torch.equal(actual, expected)
                checks[name] = True
        result.update(control='ZEA-CROSS-SCALE-HALF-R2', paired_base='ZEA-CROSS-SCALE-R1', xsa_cross_gain=0.5, gain_applies_to_student_and_teacher=checks, only_algorithm_change='fixed cross projection gain 0.5 during training and native-view inference', checkpoint_reconstruction='Use PairedCrossScaleHalf with saved cfg; base tensor schema remains XSA R1')
        (self.output_dir / 'preflight.json').write_text(json.dumps(result, indent=2))
        return result
