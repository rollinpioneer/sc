# Local-Only Artifacts

The following runtime assets remain on the experiment machine and are excluded from the lightweight package.

| Path | Files | Bytes | Reason |
|---|---:|---:|---|
| `/home/__compress_data/xushijie/work/cr_scizor/experiments/handback/hb3p_v1/roots/hb3p_test` | 592 | 252568590 | canonical payloads, images, baseline trajectories, and Parquet |
| `/home/__compress_data/xushijie/work/cr_scizor/experiments/handback/hb3p_v1/episodes/test` | 1134 | 42705021 | complete online NPZ trajectories and decision traces |
| `/home/__compress_data/xushijie/work/cr_scizor/experiments/handback/hb3p_v1/ipc` | 3 | 789 | ephemeral local inference queue |
| `/home/__compress_data/xushijie/work/cr_scizor/experiments/handback/hb3p_v1/logs` | 14 | 42461 | complete runtime logs |

Checkpoint identities and required hashes are recorded in `assets/resolved_inputs.json` and `config/frozen_protocol.json`; no weights are included.
