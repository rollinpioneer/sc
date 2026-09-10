# HB4 Square 基础策略恢复经验吸收报告

## Material Passport

- Execution status: `DEVELOPMENT_COMPLETE_FORMAL_NOT_RUN`
- Source commit: `febb2cdde6acea51e06d9f1c9623ef36f4b465bd`
- Analysis code commit: `704821402b5c58df2b02117b8af1d28207cefa19`
- Training protocol SHA256: `6119993ba58d10f8e6c421f1010032c282cbd48895d942fe9de42befc5771abc`
- Development frozen protocol SHA256: `fba64772d3e8fe7f32f943fe989ce633d3a00e1f3c8c83edabc54716f1313455`
- Base checkpoint SHA256: `e4d5c7d0e873198d21321f29f3aa38492b115d6a3d31a8c9449453af58a5a0d6`
- Frozen teacher SHA256: `7b2e5f22e0bc3722c9c70f4d6bcb3ab79696026500fc121572d50bf530349d62`
- Scope: no-help Square evaluation after matched-budget offline finetuning
- Run root: `/home/__compress_data/xushijie/work/cr_scizor_hb4_runs/hb4_square_absorb_v1`

## 历史状态

保留 HB3 `FORMAL_NONINFERIORITY_PASS` 的边界；本报告不将其改写为学习型退出的效用优越性、视觉增益或一般动态控制成立。

## 数据来源与配平

- Historical source cohort: `400` roots; eligible short-handoff roots `53`; exclusion counts: `{"ELIGIBLE": 53, "L60_NOT_GENUINE_HANDOFF": 320, "NONE_SUCCESS": 13, "NONE_SUCCESS;L60_NOT_GENUINE_HANDOFF": 14}`.
- Standard teacher source: `200/200` requested trajectories processed, `120` valid for matching, engineering failures `0`.
- Observation reconstruction: `53` roots, `106` branch records, failed records `0`; training images are `RECONSTRUCTED_FOR_HB4`.
- Matched budget: `K=53`, `N=2120` unique added labels per added arm; four non-overlapping 10-step blocks per root; HANDOFF and FIXED_L80 use the same recovery roots.
- Roles: historical `HB4_source_development`, standard `HB4_standard_source`, development `HB4_development`, test `HB4_test`.

## 训练与开发

- Training matrix: `12` jobs, four arms x three seeds, 4000 optimizer updates, batch 32 with 16 D0 + 16 added sequences, sequence length 10; all jobs passed: `True`.
- Added-slot exposure: `640000` action positions per job; total training wall time: `12288.23` seconds; per-arm seconds: `{"FIXED_L80_RECOVERY": 3333.776447534561, "HANDOFF_RECOVERY": 3415.088853120804, "MATCHED_STANDARD_DATA": 3441.308582305908, "REPLAY_ONLY": 2098.057786464691}`.
- Standard collection wall time: `1742.9303617477417` seconds; pilot wall time: `101.6416666507721` seconds. Final step 4000 only was used; no intermediate checkpoint selection.
- Development coverage: `1040` records across `80` roots; engineering failures `0`.

## 开发门槛

- Development coverage complete: `True`
- Development decision: `HB4_DEVELOPMENT_NO_GO`
- Gate details: `{"development_go": false, "engineering_valid": true, "handoff_mean_not_below_trained_controls": true, "handoff_point_gain_vs_base_at_least_0_05": false, "handoff_positive_seeds_vs_base": 1, "handoff_positive_seeds_vs_base_at_least_2": false, "seed_mean_gains_vs_controls": {"FIXED_L80_RECOVERY": {"0": 0.0625, "1": 0.0, "2": -0.025}, "MATCHED_STANDARD_DATA": {"0": 0.075, "1": 0.05, "2": 0.05}}}`
- Layered status: `{"baseline_preservation_evidence": "LIMITED_OR_INCONCLUSIVE", "engineering_valid": true, "extra_training_control_gain": true, "handoff_positive_seeds_vs_base": 1, "no_help_autonomy_gain": false, "ordinary_data_increment": true, "short_handoff_data_increment": false, "status": "HB4_NO_CONFIRMED_AUTONOMY_GAIN", "training_seed_consistency": false}`
- The development result was used only as the pre-registered unlock decision; no method, seed, hyperparameter, threshold, or checkpoint was selected from the development outcomes.

## 正式测试

- Formal test: `NOT_RUN_DEVELOPMENT_NO_GO`; the development gate did not unlock formal evaluation. No formal 400-root records, formal confidence intervals, or formal PASS are inferred from missing records.

## 预设配对比较

- `HANDOFF_RECOVERY-BASE_FROZEN`: estimate `0.008333`; paired roots `80`; bootstrap 95% CI `[-0.058333, 0.079167]`; lower bound > 0: `False`.
- `HANDOFF_RECOVERY-REPLAY_ONLY`: estimate `0.062500`; paired roots `80`; bootstrap 95% CI `[0.020833, 0.112500]`; lower bound > 0: `True`.
- `HANDOFF_RECOVERY-MATCHED_STANDARD_DATA`: estimate `0.058333`; paired roots `80`; bootstrap 95% CI `[0.016667, 0.104167]`; lower bound > 0: `True`.
- `HANDOFF_RECOVERY-FIXED_L80_RECOVERY`: estimate `0.012500`; paired roots `80`; bootstrap 95% CI `[-0.037500, 0.062500]`; lower bound > 0: `False`.
- `BASE_FROZEN` seed `None`: `6/80` = `7.500%`.
- `FIXED_L80_RECOVERY` seed `0`: `3/80` = `3.750%`.
- `FIXED_L80_RECOVERY` seed `1`: `6/80` = `7.500%`.
- `FIXED_L80_RECOVERY` seed `2`: `8/80` = `10.000%`.
- `HANDOFF_RECOVERY` seed `0`: `8/80` = `10.000%`.
- `HANDOFF_RECOVERY` seed `1`: `6/80` = `7.500%`.
- `HANDOFF_RECOVERY` seed `2`: `6/80` = `7.500%`.
- `MATCHED_STANDARD_DATA` seed `0`: `2/80` = `2.500%`.
- `MATCHED_STANDARD_DATA` seed `1`: `2/80` = `2.500%`.
- `MATCHED_STANDARD_DATA` seed `2`: `2/80` = `2.500%`.
- `REPLAY_ONLY` seed `0`: `2/80` = `2.500%`.
- `REPLAY_ONLY` seed `1`: `3/80` = `3.750%`.
- `REPLAY_ONLY` seed `2`: `0/80` = `0.000%`.

## 能力保留与分层结论

- Layered decision: `HB4_NO_CONFIRMED_AUTONOMY_GAIN`.
- No-help autonomy gain: `False`; extra-training control gain: `True`; ordinary-data increment: `True`; short-handoff increment: `False`.
- HANDOFF seed consistency: `1/3` positive versus BASE; formal test was not unlocked.
- Baseline preservation evidence: `LIMITED_OR_INCONCLUSIVE`; the development four-cell counts are recorded below and in `metrics/development/baseline_preservation.csv`.
- Four-cell `REPLAY_ONLY` seed `0`: common success `0`, new success `2`, lost baseline success `6`, common failure `72`; baseline-success denominator `6`/`80`.
- Four-cell `REPLAY_ONLY` seed `1`: common success `1`, new success `2`, lost baseline success `5`, common failure `72`; baseline-success denominator `6`/`80`.
- Four-cell `REPLAY_ONLY` seed `2`: common success `0`, new success `0`, lost baseline success `6`, common failure `74`; baseline-success denominator `6`/`80`.
- Four-cell `MATCHED_STANDARD_DATA` seed `0`: common success `0`, new success `2`, lost baseline success `6`, common failure `72`; baseline-success denominator `6`/`80`.
- Four-cell `MATCHED_STANDARD_DATA` seed `1`: common success `1`, new success `1`, lost baseline success `5`, common failure `73`; baseline-success denominator `6`/`80`.
- Four-cell `MATCHED_STANDARD_DATA` seed `2`: common success `1`, new success `1`, lost baseline success `5`, common failure `73`; baseline-success denominator `6`/`80`.
- Four-cell `FIXED_L80_RECOVERY` seed `0`: common success `0`, new success `3`, lost baseline success `6`, common failure `71`; baseline-success denominator `6`/`80`.
- Four-cell `FIXED_L80_RECOVERY` seed `1`: common success `0`, new success `6`, lost baseline success `6`, common failure `68`; baseline-success denominator `6`/`80`.
- Four-cell `FIXED_L80_RECOVERY` seed `2`: common success `2`, new success `6`, lost baseline success `4`, common failure `68`; baseline-success denominator `6`/`80`.
- Four-cell `HANDOFF_RECOVERY` seed `0`: common success `0`, new success `8`, lost baseline success `6`, common failure `66`; baseline-success denominator `6`/`80`.
- Four-cell `HANDOFF_RECOVERY` seed `1`: common success `1`, new success `5`, lost baseline success `5`, common failure `69`; baseline-success denominator `6`/`80`.
- Four-cell `HANDOFF_RECOVERY` seed `2`: common success `2`, new success `4`, lost baseline success `4`, common failure `70`; baseline-success denominator `6`/`80`.
- Approximate fixed-n=400 MDE: `0.04391239200486931`; planning only, conditional on frozen checkpoints and not a power guarantee.

## 局限与下一阶段

单任务 Square、同分布新根、固定基础 checkpoint/教师、条件化恢复源选择、三个训练种子；本轮没有在线帮助需求结论。由于 `HB4_DEVELOPMENT_NO_GO`，不建议在同一协议内扩大数据、挑选种子或追加正式测试；应另立协议后再研究数据配方或泛化。

## 本地大资产与交接

完整 checkpoint、HDF5、图像缓存、原始开发轨迹和运行日志保留在 `RUN_ROOT`，不进入轻量仓库包；路径与哈希见 `report/LOCAL_ONLY_ARTIFACTS.md`。正式矩阵的锁止日志记录为 `formal test not unlocked: HB4_DEVELOPMENT_NO_GO`；当前没有仍在运行的本轮进程。
