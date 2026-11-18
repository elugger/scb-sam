# Recorded BY ten-label HPQ sources

This directory contains the fourteen source/configuration files associated with the November 7 BY ten-label run. Private machine roots and GPU UUIDs were replaced. `release_source_provenance.json` preserves both original and distributed SHA256 values. The original source manifest remains historical evidence; it is not a valid digest manifest for the transformed files.

The observed experiment completed a 60-epoch, 600-update supervised prefix, then 38 SSL epochs, 684 successful updates and 3420 main input pairs. Its final student Dice was 69.59686464448257%.

`run_by10.py` has `prepare`, `launch`, `supervise` and `worker` actions; `--bundle` is required. The public command router must refuse execution until the project root, dataset bindings, initializer, GPU/resource guards, source contracts and fresh qualification are validated. Merely replacing the placeholder root or historical hash values is not port validation. No training was run while copying these files.
