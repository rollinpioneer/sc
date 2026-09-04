# Stage 1: DataMIL baseline infrastructure

This directory records the reproducible baseline contract for the
Symmetry-Calibrated DataMIL project. The checked-out repository is SCIZOR, not
the original DataMIL implementation. Consequently the configuration and
wrappers are deliberately **blocked** until both a DataMIL entry point and the
Robomimic HDF5 dataset are supplied; they never substitute SCIZOR suboptimal
scores for DataMIL influence scores.

Run the non-mutating preflight with:

```bash
bash experiments/stage1/scripts/run_stage1_smoke.sh
```

See `reports/mvp_config_freeze.md` for the exact missing inputs.
