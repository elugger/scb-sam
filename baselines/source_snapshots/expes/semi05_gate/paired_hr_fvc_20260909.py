"""One preregistered combination of independently positive HR and FVC modules."""
import json
from expes.semi05_gate.paired_fvc_control_20260909 import PairedFVCControl
from expes.semi05_gate.paired_hr_fclr_20260909 import PairedHRFCLR


class PairedHRFVC(PairedFVCControl, PairedHRFCLR):
    # FVC wraps only _forward_main_logits for the strong student. HR's
    # supervised step calls its own forward directly and remains unperturbed.
    # Teacher/test also bypass FVC; both retain the HR residual architecture.
    def preflight(self, init_from, expected_floor):
        result = super().preflight(init_from, expected_floor)
        assert result['mechanism_status'] == 'PASS'
        assert result['residual_resolution'] == 64
        result.update(control='HR-FVC-R1', paired_base='C0',
                      architecture_control='HR-FCLR-R1', training_control='FVC-R1',
                      audit_note='labeled perturbation audit follows two disposable HR optimizer steps')
        (self.output_dir/'preflight.json').write_text(json.dumps(result, indent=2))
        return result
