# Stage 1.1 configuration freeze

## Status

**Blocked, not frozen.** This is an evidence-based stop condition from the Stage-1 instruction: the checkout has neither an original DataMIL influence/selection implementation nor the required demonstration dataset. A valid fixed pool, five true target-demo IDs, and a DataMIL budget cannot be invented.

## Provisional MVP choice

- Benchmark/task: Robomimic `can` multi-human image task.
- Intended data: `robomimic_can_mh_image_hdf5_required` beneath `/home/xushijie/work/cr_scizor/data/robomimic`.
- Rationale: SCIZOR already carries a BC-GMM configuration for `can`; Robomimic HDF5 contains simulator state and supports grasp / pick-and-place behavior.
- Policy baseline once unblocked: existing Robomimic BC-GMM config (`batch_size=16`, Adam `1e-4`, 600 epochs, in-distribution rollout 80 episodes).

## Entry commands actually verified

```bash
PYTHONPATH="$PWD/robomimic" python robomimic/robomimic/scripts/train.py --help
bash experiments/stage1/scripts/run_stage1_smoke.sh
```

The first reaches the repository CLI but stops on a missing `psutil` dependency in the active Python. The second correctly writes a failed, traceable manifest and explains the configuration inputs that are missing.

## Freeze status by field

| Field | Status |
| --- | --- |
| Candidate pool | Not available; `candidate_pool_ids.txt` intentionally has no fabricated IDs |
| Five target demos | Not available; must be read from the supplied HDF5 |
| Cluster method / DataMIL influence | Original implementation absent |
| Selection budget | Cannot derive before original DataMIL cluster count exists |
| Backbone / policy train defaults | Provisionally frozen to existing Robomimic `can` BC-GMM configuration |
| Smoke evaluation | 5 episodes after data/pipeline are available |
| Formal evaluation | 80 episodes, matching existing repository config |
| Compute | GPU 0 (A100 40GB); exact cost pending first successful run |

## Artifacts and next executable entrance

- Base config: `experiments/stage1/configs/mvp_base.yaml`
- Smoke config: `experiments/stage1/configs/mvp_smoke.yaml`
- Preflight: `bash experiments/stage1/scripts/run_stage1_smoke.sh`

To unblock, provide (1) the original DataMIL repository/module and its command that produces cluster influence scores, and (2) the immutable Robomimic `can` HDF5 candidate-pool path. The first unblocking action is to inspect that HDF5, deterministically select five IDs and a candidate pool, set the original DataMIL command and budget in `mvp_base.yaml`, then run smoke.

Known limitation: the local `scizor-curation` environment currently lacks PyTorch, so it must be repaired from the repository requirements before policy training.
