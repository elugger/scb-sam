# Fixed release-alias splits

The four files cover BY and Public433 at five and ten labeled cases, fold 2. They were copied unchanged from the earlier release and checked against original scientific-contract SHA pins and the private alias map on 2026-11-12. Labeled, whole-training, unlabeled and evaluation membership and order agree. BY uses 100 training / 25 evaluation cases; Public433 uses 346 / 87. Five-label sets are nested in the corresponding ten-label sets.

Run `python scripts/verify_splits.py` to verify the packaged file digests, aliases, counts, disjointness and nesting. `provenance.json` records source evidence without raw internal identifiers. BY5/BY10/Public5 were compared with live read-only server lists; Public10 was checked against the local original bytes and its frozen training-contract hashes.

Release aliases are not OpenNeuro subject IDs or image filenames. The private internal-ID mapping is not bundled. Mapping these aliases to a fresh official IDEAS download remains unverified. The user declined further pursuit of this mapping on 2026-11-12; no mapping is inferred from numeric suffixes and no replacement split is generated. A lawful data holder must provide their own verified file correspondence. This limitation prevents claiming turnkey reproduction from the public download alone.

The five/ten-label configuration table and main table share these partitions; references to evaluation/test in historical code do not make these partitions untouched holdouts. Their development use and GT-localized ROI preprocessing are disclosed in the protocol documentation.
