# Local-Only Artifacts

The following runtime assets remain on the experiment machine and are excluded from the lightweight package.

| Path | Files | Bytes | Reason |
|---|---:|---:|---|
| `/home/__compress_data/xushijie/work/cr_scizor/experiments/handback/hb3p_exit_probe_v1/roots/hb3p_exit_probe` | 300 | 118981239 | canonical payloads, images, baseline trajectories, and Parquet |
| `/home/__compress_data/xushijie/work/cr_scizor/experiments/handback/hb3p_exit_probe_v1/episodes` | 602 | 22130711 | complete online NPZ trajectories and decision traces |
| `/home/__compress_data/xushijie/work/cr_scizor/experiments/handback/hb3p_exit_probe_v1/ipc` | 0 | 0 | ephemeral local inference queue |
| `/home/__compress_data/xushijie/work/cr_scizor/experiments/handback/hb3p_exit_probe_v1/logs` | 8 | 27988 | complete runtime logs |

Checkpoint identities and required hashes are recorded in the assets and frozen protocol files; no weights are included.
