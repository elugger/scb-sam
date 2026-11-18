# Historical comparison protocol and current endpoints

The common comparison unit is a main 3D training input, separately counted for labeled and unlabeled streams.
It is not an equal-FLOP, equal-voxel, equal-update, equal-initialization or convergence guarantee.

The following budget is the comparator reference, not the current SCB-SAM budget for every row. The current table is defined in [MAIN_TABLE.md](MAIN_TABLE.md) and `records/main_table.json`.

| Comparator dataset / labels | Reference cycles | L stream | U stream |
| --- | ---: | ---: | ---: |
| BY / 5 | 36 | 3420 | 3420 |
| BY / 10 | 36 | 3240 | 3240 |
| Public / 5 | 11 | 3795 | 3795 |
| Public / 10 | 11 | 3740 | 3740 |

The reference-cycle calculation is cycles × 5 × ceil(U / 5). BY U=95/90; Public U=341/336.
Original batch sizes are retained. Public SDCL/SFR/DualTeacher five-case runs use 3796 per stream, one input above target (about 0.026%).
Original learning-rate/ramp clocks and mandatory prefixes are preserved; Public's terminal cutoff does not compress a 36-cycle schedule.

Count native prefixes separately, including original SAM fine-tuning and dual-student training.
SFR five-case prefix has its recorded native labeled-input count (10000), not an inferred 6000×2;
ten-case prefix is 12000. Read the exact per-run budget record.
Public SDCL and DualTeacher count two effective optimizer updates per native round where specified;
BY's older optimizer ledger convention must be separately audited before cross-dataset compute totals are claimed.

The earlier Public epoch11 reporting points for SCB-SAM were superseded; they are not the current main-table endpoints. Current SCB-SAM endpoints are:

| Dataset / labels | Completed epochs / updates | Reported endpoint | Main L/U pairs |
| --- | --- | --- | ---: |
| BY / 5 | 36 / 684 | fixed epoch35 | 3420 |
| BY / 10 | 38 / 684 | fixed epoch37 | 3420 |
| Public433 / 5 | 36 / 2484 | fixed epoch35 | 12420 |
| Public433 / 10 | 16 / 1088 | development-selected epoch14, update1020 | 5440 total; 5100 at selection |

Epoch indices above are zero-based. The separate supervised initialization and native comparator prefixes are additional work. These are unequal complete-recipe budgets. The Public10 selection and historical development exposure must remain disclosed; the evaluation partition was not untouched.

The older 40-row corpus included supervised/VNet references. Those twelve rows and their execution schedules are outside this 28-row main-table package. No schedule in an archival document authorizes new training.

Metrics are case-wise Dice/Jaccard/HD95/ASD means. Distances use the recorded voxel convention.
GT ROI localization and method-specific crop/sliding-window behavior are limitations, not clinical automatic localization.
BY supervised historical references use seed2026 and log-rounded metrics with comparability pending.


H/HP/HQ/HPQ are a separate BY5 configuration table. The accepted HQ uses CCBS=0.25 after development tuning, so it is not a matched single-factor deletion of CCL from Full. Historical HQ weights were not saved; preserving the configuration and measured result does not create a recoverable checkpoint.
