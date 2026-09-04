# Stage 3A Replay Rescue Report

## Decision

`STOP_STAGE3A_REPLAY_RESCUE`

This is an engineering stop. It does not change the frozen Stage 2 No-Go decision or claim that the counterfactual method is invalid.

## Evidence

1. **HDF5 lineage.** Clean benchmark actions are identical to the corresponding source actions (`median_action_max_abs = 0.0`) for both Can and Square. Benchmark states are not identical to source states at the same index. The nearest tested relation is source `t+1`, but its median row L2 remains `2.0574` for Can and `0.04128` for Square, far above the `1e-8` identity threshold.
2. **Stage 1B provenance.** The historical generator commit and original environment were not recoverable from the repository history or saved logs. The current diagnostic runtime is `mimicgen`, Python 3.9.25, robosuite 1.4.1, MuJoCo 2.3.2, with the SCIZOR robomimic fork.
3. **Legacy runtime.** No installed environment containing the required `mujoco-py` / MuJoCo 2.0 stack was found. The available environments use modern `mujoco` or have an incompatible robosuite installation.
4. **Action replay.** Under the available runtime and Stage 1B reset semantics, neither source demonstrations nor v0.1 clean rollouts pass the required all-step `flat max_abs <= 1e-10` criterion. Errors are primarily in `qpos` and `qvel`; simulation time matches, and no exact legacy `act` comparison is available in the failing path.
5. **Determinism pilot.** Two independently created current-runtime environments replayed Can `demo_0` for 118 steps and Square `demo_0` for 127 steps with zero state, reward, staged-reward, and success disagreement. This validates determinism of the candidate runtime only; it does not validate v0.1 lineage.
6. **v0.2 and oracle.** No replay-locked v0.2 HDF5 benchmark was generated, and no paired-clean or teacher-forced oracle ceiling was run. Therefore the v0.2 continuation gates cannot pass.

## Answers to the rescue questions

| Question | Result |
|---|---|
| v0.1 clean state relation to source | Actions match; states do not match same-index or adjacent-index within threshold |
| Original Stage 1B environment | Unknown |
| Legacy robosuite deterministic test | Not executed: required legacy environment unavailable |
| Source exact replay | No |
| v0.1 exact replay | No |
| Dominant error component | qpos and qvel; time matches |
| v0.2 generated | No; only deterministic runtime probes exist |
| Twin-prefix determinism | Current-runtime clean twin pilot: yes; v0.2 oracle: not run |
| Oracle ceiling | Not run |
| Continue 3A-E | No, stop replay rescue |

## Frozen-scope statement

Stage 1 benchmark, labels, split, evaluation results, Stage 2 checkpoints/proposals/No-Go conclusion, and the prior Stage 3A alignment-repair artifacts were not overwritten. Rescue artifacts are isolated below `stage3a/replay_rescue`.

## Artifacts

- `config/frozen_inputs.sha256`
- `lineage/hdf5_lineage.json`
- `provenance/stage1b_provenance_summary.json`
- `legacy_env/existing_replay_envs.txt`
- `replay_checks/can_stage1b_replay.json`
- `replay_checks/square_stage1b_replay.json`
- `v02/can_demo_0_determinism_probe.json`
- `v02/square_demo_0_determinism_probe.json`
- `metrics/replay_rescue_decision.json`
