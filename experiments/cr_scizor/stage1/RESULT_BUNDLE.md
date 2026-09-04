# Stage 1A result bundle

This lightweight bundle contains the run configurations and logs, the dataset
inventory, the frozen seed-0 checkpoint SHA-256 record, and both exported parquet
result tables. It intentionally contains no checkpoints or other large binaries.

It excludes all checkpoints, the two scored HDF5 working copies (about 4.6 GB),
and the 2,000--8,000 step intermediate checkpoints. The HDF5 copies can be
regenerated from the original downloaded datasets plus the retained frozen model
and scorer.

Verification summary:

- Can: 23,207 scored transitions across 200 demos.
- Square: 30,154 scored transitions across 200 demos.
- `original_transition_scores.parquet`: 53,361 rows.
- `chunk_evidence_interface_smoke.parquet`: 486 rows with `V_c` in `[0, 1]`.
