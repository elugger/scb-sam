# Latest main-table record and source mapping

The current manuscript has **28 rows**: BY and Public433, 5 and 10 labels, and seven complete-system recipes. The older 40-row archive also contained 12 supervised/VNet references; those are not silently treated as current main-table rows. Four BY configurations are included as a separate configuration table. The extra shared-H with-SAM experiments are outside the current delivery scope.

`records/main_table.json` is the aggregate record. Dice and Jaccard are fractions there; the table below expresses percentages. HD95 and ASD use the recorded voxel convention. No model training or new evaluation was performed while assembling these files.

| Dataset | Labels | Method | Dice % | Jaccard % | HD95 | ASD | Precision |
|---|---:|---|---:|---:|---:|---:|---|
| BY | 5 | SemiSAM+ / MT | 42.81 | 29.28 | 17.58 | 5.28 | recorded full precision |
| BY | 5 | SemiSAM+ / UA-MT | 45.22 | 31.06 | 20.65 | 5.20 | recorded full precision |
| BY | 5 | SDCL | 56.69 | 41.88 | 13.25 | 3.96 | recorded full precision |
| BY | 5 | SFR | 23.44 | 15.09 | 41.47 | 24.06 | recorded full precision |
| BY | 5 | ReLiF-3D | 59.95 | 44.94 | 14.54 | 3.56 | recorded full precision |
| BY | 5 | DualTeacher+SAM | 32.20 | 21.98 | 32.43 | 11.52 | recorded full precision |
| BY | 5 | SCB-SAM | 67.79 | 52.93 | 10.24 | 2.63 | recorded full precision |
| BY | 10 | SemiSAM+ / MT | 63.23 | 48.76 | 11.74 | 3.08 | recorded full precision |
| BY | 10 | SemiSAM+ / UA-MT | 62.17 | 48.23 | 12.53 | 3.16 | recorded full precision |
| BY | 10 | SDCL | 65.92 | 50.78 | 11.41 | 2.83 | recorded full precision |
| BY | 10 | SFR | 50.95 | 36.84 | 18.30 | 4.45 | recorded full precision |
| BY | 10 | ReLiF-3D | 60.56 | 46.43 | 13.54 | 3.35 | recorded full precision |
| BY | 10 | DualTeacher+SAM | 45.10 | 33.90 | 29.55 | 8.64 | recorded full precision |
| BY | 10 | SCB-SAM | 69.60 | 55.47 | 9.57 | 2.55 | recorded full precision |
| Public433 | 5 | SemiSAM+ / MT | 55.08 | 42.86 | 15.48 | 7.33 | recorded full precision |
| Public433 | 5 | SemiSAM+ / UA-MT | 56.45 | 44.63 | 19.00 | 11.35 | recorded full precision |
| Public433 | 5 | SDCL | 67.83 | 54.31 | 8.00 | 2.07 | recorded full precision |
| Public433 | 5 | SFR | 56.27 | 46.46 | 46.54 | 39.86 | recorded full precision |
| Public433 | 5 | ReLiF-3D | 62.52 | 50.89 | 16.18 | 7.64 | recorded full precision |
| Public433 | 5 | DualTeacher+SAM | 55.20 | 45.26 | 32.26 | 17.59 | recorded full precision |
| Public433 | 5 | SCB-SAM | 70.63 | 57.51 | 6.58 | 1.86 | recorded full precision |
| Public433 | 10 | SemiSAM+ / MT | 65.19 | 54.34 | 8.93 | 2.56 | recorded full precision |
| Public433 | 10 | SemiSAM+ / UA-MT | 65.22 | 54.46 | 16.10 | 10.74 | recorded full precision |
| Public433 | 10 | SDCL | 70.36 | 57.90 | 7.24 | 1.85 | recorded full precision |
| Public433 | 10 | SFR | 59.15 | 49.98 | 59.52 | 54.54 | recorded full precision |
| Public433 | 10 | ReLiF-3D | 66.00 | 54.95 | 8.47 | 2.44 | recorded full precision |
| Public433 | 10 | DualTeacher+SAM | 59.18 | 49.66 | 20.39 | 10.99 | recorded full precision |
| Public433 | 10 | SCB-SAM | 72.76 | 60.08 | 6.03 | 1.64 | recorded full precision |

## Endpoints and unequal recipe budgets

BY five-label SCB-SAM uses the new shared supervised initializer, 36 epochs, 684 successful updates and 3420 main input pairs. BY ten-label SCB-SAM uses its ten-label supervised initializer, 38 epochs, 684 updates and 3420 pairs; its archived comparators used 3240 primary inputs per stream. Supervised initialization is separate (60 epochs; 300 versus 600 updates).

Public433 SCB-SAM five-label uses the fixed epoch35 endpoint of a 36-epoch run. The ten-label value uses development-selected epoch14 from a 16-epoch run. Comparators retain their fixed 11-cycle primary exposure recipes (five-label 3795 reference inputs, with 3796 for some native batches; ten-label 3740). Mandatory SDCL/SFR prefixes and auxiliary computation are retained in each record. These are complete-recipe comparisons, not equal total computation or uniform endpoint selection.

The old release stored Public SCB-SAM epoch11 values (71.1171% / 72.7160%), which do not produce the current table (70.63% / 72.76%). The old values are retained under `superseded_20261030_manifest_record`. Matching original Public endpoint receipts were retrieved read-only on November 10 and added with their SHA256 values. Public5 completed 36 epochs, 2484 updates and 12420 pairs. Public10 completed 16 epochs, 1088 updates and 5440 pairs; its selected15th-epoch checkpoint records1020 updates (5100 pairs by the fixed340-pair cycle). Exact metrics now come from these receipts, not the manuscript rounding. No additional model evaluation was performed.

Public ten-label ReLiF HD95 has a minor manuscript discrepancy: archived accepted full precision is 8.474122725399148, which rounds to 8.47, whereas current paper text shows 8.48. The exact accepted record is preserved here rather than changing it to match a rounded manuscript transcription.

The four BY configuration records are under `records/configurations/`. HQ used development-tuned CCBS=0.25; its accepted historical run did not save a weight because the then-active save condition was not met. Do not promise a downloadable checkpoint for that HQ result or interpret it as a matched CCL deletion.

## Source families

| Recipe | Recorded source family |
|---|---|
| SemiSAM+ MT / UA-MT, SDCL BY | main_table_20261019_r3 and native SemiSAM/SDCL model sources |
| SFR / ReLiF BY | main_table_20261019_r4, pinned official sources and native adapters |
| DualTeacher+SAM BY | dualteacher_fixed_final_20261021_v4_conda and native files |
| Public comparators | public_ncase_baselines_20261024 (method-specific contracts/controllers) |
| Public SCB-SAM 5 labels | public_hpq_fineq_r3_20261020 |
| Public SCB-SAM 10 labels | public_hpq_fineq_n010_restart4_20261027 |
| BY SCB-SAM 5 labels | by5_four_arm_rerun_20261030 and by5_four_arm_training_20261030 |
| BY SCB-SAM 10 labels | by10_hpq/src with run_by10.py |

The most complete earlier source corpus is the October 30 release source map; private original bytes remain separate. Recorded controller code still has AST-loaded dependencies, frozen hashes and environment assumptions. Source availability does not by itself establish a portable training entry. Three cascade components formerly recorded only as `.pyc` now have readable source whose complete compiled code matches the training bytecode after filename-only normalization. The builder was reconstructed from bytecode, rather than recovered as an original historical file; see `CASCADE_SOURCE_RECOVERY.md`. Historical source/data contracts still require explicit new-host preparation.

## Interpretation and distribution

Evaluation partitions participated in development and use reference-localized ROIs. Single-run file variation is not between-seed uncertainty. Historical BY comparator weights were retired and are not advertised as downloadable. New source files, a repository license or preserved hashes do not grant redistribution rights for third-party pretrained weights or patient data. The optional with-SAM legacy results and untrained v4 revision are excluded from this delivery.
