# SCIZOR Stage 3A A–D audit

This report covers only 3A-A (freeze), 3A-B (proposals), 3A-C (action library), and 3A-D (simulator oracle). Stages 3A-E onward were not run.

## Outcome

- Structural A–D checks: **PASS**
- Stage 2 decision frozen: `NO_GO_SWITCH_DIRECTION`
- Train oracle: 27301 raw rows from 27301 planned rows; 25985 target-valid rows.
- Validation oracle: 9463 raw rows from 9463 planned rows; 8839 target-valid rows.
- Simulator state alignment smoke: **FAIL**; median next-state L2 = `3.50334445387125`.

## Important caveats

The requested `scizor-robomimic` conda environment was absent. The original error was recorded verbatim in the audit, and the verified `mimicgen` environment was used with `MUJOCO_GL=egl`. Optional robosuite task-zoo imports emitted warnings but did not prevent Can/Square replay.

The alignment smoke did not meet the required `<1e-4` threshold (the two measured errors were recorded in `oracle/state_alignment.json`). Therefore these results are simulator outputs under the verified environment, not a claim of benchmark-state alignment.

The action library now contains real FAISS `IndexFlatL2` files. The library contents, thresholds, medoids, and plan replacement identities were preserved when converting the prior NumPy payload files to real FAISS indices.
