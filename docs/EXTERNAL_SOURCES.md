# External source acquisition

This source preview excludes 173 upstream dependency files from Git publication. Local review copies are retained. The manifest records official repositories, immutable revisions, original hashes, any explicit conversion and the final expected hash. It contains 146 unchanged files and 27 transformed files: 20 line-ending conversions, four path redactions/configuration substitutions, and three previously recorded DualTeacher adaptations. No model or loss change is introduced by this packaging step.

```bash
python scripts/upstream_sources.py fetch --destination /absolute/external/source-cache --transport github-cli
python scripts/upstream_sources.py verify --destination /absolute/external/source-cache
```

The GitHub CLI transport requires a locally authenticated `gh`. Alternatives are `--transport urllib` or `--transport curl`; all retain TLS verification. Components are semisam, sdcl, sfr, relif and dualteacher. Sources are never executed by acquisition. A different existing file is preserved and rejected. Conversion verifies both original and final hashes and checks replacement counts.

The cache retains repository-relative paths. Automatic runtime assembly from this cache is not yet implemented; downloading dependencies does not make `reproduce.py train` executable. Training remains disabled pending portable runtime qualification.

For SemiSAM and SDCL, the audited commit establishes byte identity of the listed files, not historical identity of the entire checkout. SFR, ReLiF and DualTeacher use the experimental commit pins. DualTeacher's original repository is https://github.com/Aguang094/Dualteacher-with-SAM at 3710e3216db8bdb444296f0460eab9532b253848.

Two initial SDCL differences were local files: an empty package marker remains in the preview; the binary segmentation losses adapter remains excluded pending provenance review. The standalone src/scb_seg/models/vnet.py review copy is also excluded pending its provenance review. Neither is silently substituted. These are disclosed source gaps in this preview; the original local copies remain intact.

SAM architecture copies carry their upstream Apache-2.0 license texts in third_party_licenses, and historical copyright headers removed from the comment-free export are retained in ours/SNAPSHOT_NOTICES.md. The project license applies to original work, not a blanket relicense of dependencies. Upstream terms still govern separately acquired sources.

Validation covers corruption rejection, replacement-count and two-stage hash checking, path escape, idempotency and preservation of modified files. The live acquisition receipts are summarized in RELEASE_VALIDATION.json. Source verification is not clean-environment training validation.
