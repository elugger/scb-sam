# Environment evidence

`observed_runtime.json` is an earlier distribution-metadata inventory, not an installable lock. It contains mutually inconsistent torch/torchvision entries and must not be copied directly into a training environment.

`import_audit_20261112.json` records successful CPU-only imports under the original server interpreter. The actual import resolver selects several packages from the user site before the Conda environment. It loaded PyTorch 1.12.1+cu113, torchvision 0.13.1+cu113 and NumPy 1.26.4. CUDA was not initialized. The original BY5 preparation receipt separately records PyTorch 1.12.1+cu113, matching this import; that receipt does not establish all other package versions.

Neither inventory is a clean-environment validation. Import success is weaker than numerical agreement, finite-gradient bridge validation and full checkpoint readback. A clean runtime must use an isolated environment with user-site imports disabled, then explicitly install its chosen compatible dependencies and qualify the original scientific computation. Silently adding `PYTHONNOUSERSITE=1` to the old server command would change which libraries are imported; it is not an already validated fix.

No package was installed, uninstalled or upgraded on the training server during this audit. No historical or cancelled training job was restarted.
