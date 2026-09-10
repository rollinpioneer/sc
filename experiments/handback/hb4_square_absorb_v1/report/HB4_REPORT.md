# HB4 Square 基础策略恢复经验吸收报告

## Material Passport

- Execution status: `DEVELOPMENT_COMPLETE_FORMAL_NOT_RUN`
- Source commit: febb2cdde6acea51e06d9f1c9623ef36f4b465bd
- Scope: no-help Square evaluation after matched-budget offline finetuning
- Run root: `/home/__compress_data/xushijie/work/cr_scizor_hb4_runs/hb4_square_absorb_v1`

## 历史状态

保留 HB3 `FORMAL_NONINFERIORITY_PASS` 的边界；本报告不将其改写为学习型退出的效用优越性。

## 数据与训练

- HB4-C matching audit: see `data/matching_audit.json`.
- Training matrix: four arms x three seeds, 4000 optimizer updates, final step only used for evaluation.
- Student input: two RGB cameras plus 9-D proprioception; helper policy and selector are excluded from the no-help evaluator.

## 开发门槛

- Development coverage complete: `True`
- Development decision: `HB4_DEVELOPMENT_NO_GO`
- Gate details: `{"development_go": false, "engineering_valid": true, "handoff_mean_not_below_trained_controls": true, "handoff_point_gain_vs_base_at_least_0_05": false, "handoff_positive_seeds_vs_base": 1, "handoff_positive_seeds_vs_base_at_least_2": false, "seed_mean_gains_vs_controls": {"FIXED_L80_RECOVERY": {"0": 0.0625, "1": 0.0, "2": -0.025}, "MATCHED_STANDARD_DATA": {"0": 0.075, "1": 0.05, "2": 0.05}}}`

## 正式测试

- Formal test: `NOT_RUN_DEVELOPMENT_NO_GO`; the development gate did not unlock formal evaluation, and no PASS is inferred from missing records.

## 功效规划与能力保留

- Approximate fixed-n=400 MDE: `0.04391239200486931`; this is planning only.
- Methods summary: `[{"method_id": "BASE_FROZEN", "training_seed": null, "records": 80, "engineering_failures": 0, "roots": 80, "successes": 6, "success_rate": "7.500%"}, {"method_id": "FIXED_L80_RECOVERY", "training_seed": 0, "records": 80, "engineering_failures": 0, "roots": 80, "successes": 3, "success_rate": "3.750%"}, {"method_id": "FIXED_L80_RECOVERY", "training_seed": 1, "records": 80, "engineering_failures": 0, "roots": 80, "successes": 6, "success_rate": "7.500%"}, {"method_id": "FIXED_L80_RECOVERY", "training_seed": 2, "records": 80, "engineering_failures": 0, "roots": 80, "successes": 8, "success_rate": "10.000%"}, {"method_id": "HANDOFF_RECOVERY", "training_seed": 0, "records": 80, "engineering_failures": 0, "roots": 80, "successes": 8, "success_rate": "10.000%"}, {"method_id": "HANDOFF_RECOVERY", "training_seed": 1, "records": 80, "engineering_failures": 0, "roots": 80, "successes": 6, "success_rate": "7.500%"}, {"method_id": "HANDOFF_RECOVERY", "training_seed": 2, "records": 80, "engineering_failures": 0, "roots": 80, "successes": 6, "success_rate": "7.500%"}, {"method_id": "MATCHED_STANDARD_DATA", "training_seed": 0, "records": 80, "engineering_failures": 0, "roots": 80, "successes": 2, "success_rate": "2.500%"}, {"method_id": "MATCHED_STANDARD_DATA", "training_seed": 1, "records": 80, "engineering_failures": 0, "roots": 80, "successes": 2, "success_rate": "2.500%"}, {"method_id": "MATCHED_STANDARD_DATA", "training_seed": 2, "records": 80, "engineering_failures": 0, "roots": 80, "successes": 2, "success_rate": "2.500%"}, {"method_id": "REPLAY_ONLY", "training_seed": 0, "records": 80, "engineering_failures": 0, "roots": 80, "successes": 2, "success_rate": "2.500%"}, {"method_id": "REPLAY_ONLY", "training_seed": 1, "records": 80, "engineering_failures": 0, "roots": 80, "successes": 3, "success_rate": "3.750%"}, {"method_id": "REPLAY_ONLY", "training_seed": 2, "records": 80, "engineering_failures": 0, "roots": 80, "successes": 0, "success_rate": "0.000%"}]`
- Baseline preservation rows: `12`

## 局限

单任务 Square、同分布新根、固定基础 checkpoint/教师、条件化恢复源选择、三个训练种子；本轮没有在线帮助需求结论。

## 本地大资产

完整 checkpoint、HDF5、图像缓存和运行日志保留在 `RUN_ROOT`，不进入轻量仓库包；其路径与哈希见 `report/LOCAL_ONLY_ARTIFACTS.md`。
