# Repository map and Stage-1 entry audit

Project root: `/home/xushijie/work/cr_scizor/SCIZOR` (commit `75051b3d45676a033533896da16a6a7abf8ac42e`).

| Required Stage-1 function | Actual repository entry | Status |
| --- | --- | --- |
| Data preparation | `stage1/src/inspect_hdf5.py` validates a supplied Robomimic HDF5; SCIZOR curation expects an existing HDF5 | Dataset absent |
| Candidate clusters | `curation/semdedup/{semdedup.py,sort_clusters.py}` | SCIZOR semantic-dedup clusters, not original DataMIL clusters |
| Target demonstrations | No target-demo CLI or configuration found | Missing |
| DataMIL influence/datamodel | No `DataMIL`, datamodel, influence, or metagradient code found | Missing / blocking |
| Top-k selection | SCIZOR uses percentile curation in `robomimic/.../bc_curation.json`, not DataMIL Top-k cluster selection | Missing / blocking |
| Policy training | `robomimic/robomimic/scripts/train.py --config robomimic/robomimic/exps/curation_exps/robomimic/can/bc_curation.json` | Present; system Python lacks `psutil` |
| Rollout evaluation | Built into `robomimic/robomimic/scripts/train.py`; standalone `robomimic/robomimic/scripts/run_trained_agent.py` | Present contingent on training env/data |
| Default outputs | Robomimic `train.output_dir`; Stage-1 standardized root is `outputs/stage1/<experiment_id>/` | Defined |

The closest existing task is Robomimic `can` (pick-and-place). It supports simulator states in HDF5 and is a suitable future SE(3) diagnostic task, but no HDF5 file is available under the configured data root.

No SCIZOR score is interchangeable with a DataMIL influence score. The Stage-1 wrapper therefore stops at preflight rather than silently running a different method.
