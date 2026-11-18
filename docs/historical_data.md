# Data access and identifiers

No patient images, masks, identifying metadata or raw internal case IDs are stored in Git.
BY is a private dataset; publish only the permitted access statement and preprocessing protocol.
Public433 is the study subset of IDEAS (The Imaging Database for Epilepsy And Surgery) containing 433 preoperative T1-weighted scans with resection masks. The user confirmed this source on 2026-11-12. The masks were derived from postoperative imaging and manually checked/corrected. The source study reports 442 patients with preoperative MRI, of whom 433 have resection masks; the project must not describe all 442 as the 433-case segmentation cohort.

Official acquisition starts at the [IDEAS author website](https://www.cnnp-lab.com/ideas-data). The [OpenNeuro ds005602 release](https://openneuro.org/datasets/ds005602) contains raw T1w/FLAIR images, **not the resection masks**; its [official README](https://github.com/OpenNeuroDatasets/ds005602/blob/1.0.0/README) directs users to separate processed-data downloads. Downloading OpenNeuro alone is insufficient for these segmentation experiments.

| Component | Official source | Version/space qualification |
| --- | --- | --- |
| Raw T1w/FLAIR | [OpenNeuro ds005602 v1.0.0](https://openneuro.org/datasets/ds005602/versions/1.0.0), DOI `10.18112/openneuro.ds005602.v1.0.0` | The pinned [dataset description](https://raw.githubusercontent.com/OpenNeuroDatasets/ds005602/1.0.0/dataset_description.json) declares CC0. This identifies an available upstream version, not yet the historical training download. |
| Original IDEAS masks | [Author-linked mask collection](https://figshare.com/s/31ab43d1829b12ac13e8) | The OpenNeuro v1.0.0 README describes both native and FreeSurfer `orig.mgz` spaces. Select the space paired with the actual image; do not infer a match from array shape. |
| Current mask link | [Author website's current native-T1w masks](https://figshare.com/s/476b37fd883c14f50324) | The author website now combines IDEAS I/II resources. Its current link must not silently replace the original experiment input. |

The two Figshare landing pages returned HTTP 403 to the release audit on 2026-11-12. Their links and descriptions were verified from author-controlled pages, but archive contents, versions, checksums and the mask-specific license have not been verified. OpenNeuro's CC0 declaration is not automatically applied to a separately hosted mask collection.

Cite [The Imaging Database for Epilepsy And Surgery, DOI 10.1111/epi.18192](https://doi.org/10.1111/epi.18192). The author website additionally requests citation of [Automated generation of epilepsy surgery resection masks: The RAMPS pipeline, DOI 10.1162/IMAG.a.147](https://doi.org/10.1162/IMAG.a.147) for the masks. The manuscript reference number [14] is not a persistent identifier. These references describe the source; they do not establish which RAMPS version generated the historical masks. Keep data terms and citations separate from this project's Apache-2.0 code license.

The exact historical dataset snapshot and alias-to-official-subject mapping remain unresolved. On 2026-11-12 the user declined further pursuit of that mapping. The fixed original split lists and their ordering have been verified separately (see `splits/provenance.json`), but this is not a verified correspondence to a fresh official download. Do not replace the original split by a newly shuffled download or claim that identifying the upstream dataset completes the data-binding checks. No dataset images or masks have been downloaded or redistributed during release preparation.

splits/ contains stable release aliases preserving fold2 membership and nested labeled subsets.
The alias-to-native-ID mapping is kept outside Git. These alias lists alone cannot locate original scans:
a lawful data holder must supply a local mapping or an approved native-ID split export.
Record transformations and data versions; never silently reassign test cases.

GTROI128 uses ground-truth localization and must be disclosed. Existing reports are not evidence of
an automatic lesion-localization pipeline.
