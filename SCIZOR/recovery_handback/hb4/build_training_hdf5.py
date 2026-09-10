"""Materialize HB4 training blocks while keeping D0 as external links."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from recovery_handback.common import atomic_json_dump, sha256_file, write_table


BASE_MANIFEST = Path("/home/__compress_data/xushijie/work/cr_scizor/experiments/handback/hb1_v1/base/square/data/bc_source_manifest.json")
BASE_CHECKPOINT = Path("/tmp/hb1_runtime/hb1_repair_v1/checkpoints/square_base_epoch_200.pth")
RUN_ROOT = Path("/home/__compress_data/xushijie/work/cr_scizor_hb4_runs/hb4_square_absorb_v1")
EXPORT_ROOT = Path("/home/__compress_data/xushijie/work/cr_scizor_github_sc_worktrees/hb4-square-absorb-v1/experiments/handback/hb4_square_absorb_v1")
OBS_KEYS = ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(group, name: str, values) -> None:
    array = np.asarray(values)
    kwargs = {"compression": "gzip", "compression_opts": 1} if array.size > 1024 else {}
    group.create_dataset(name, data=array, **kwargs)


def _list_value(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def _copy_block(destination, row: dict, cache: dict[str, np.ndarray], actions: np.ndarray) -> None:
    event_ids = _list_value(row["event_ids"])
    demo = destination.create_group(event_ids[0].replace(":", "__"))
    start = int(row["block_start"])
    end = start + int(row["block_length"])
    obs_group = demo.create_group("obs")
    cache_start = int(row["obs_index"])
    for key in OBS_KEYS:
        _write(obs_group, key, cache[key][cache_start:cache_start + 10])
    _write(demo, "actions", actions[start:end])
    _write(demo, "dones", np.asarray([0] * 9 + [1], dtype=np.int64))
    demo.attrs["num_samples"] = 10
    demo.attrs["hb4_arm"] = row["arm"]
    demo.attrs["canonical_group_id"] = row["canonical_group_id"]
    demo.attrs["root_seed"] = int(row["root_seed"])
    demo.attrs["block_start"] = start
    demo.attrs["label_mask_all_true"] = True


def _load_recovery(row: dict) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with np.load(row["obs_artifact"], allow_pickle=False) as handle:
        cache = {key: np.asarray(handle[key]) for key in OBS_KEYS}
    with np.load(row["trajectory_path"], allow_pickle=False) as handle:
        actions = np.asarray(handle["actions"], dtype=np.float32)
    return cache, actions


def _load_standard(row: dict, hdf5_path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with h5py.File(hdf5_path, "r") as handle:
        demo = handle[f"data/{row['source_demo_id']}"]
        cache = {key: np.asarray(demo[f"obs/{key}"]) for key in OBS_KEYS}
        actions = np.asarray(demo["actions"], dtype=np.float32)
    return cache, actions


def build(export_root: Path, run_root: Path) -> dict:
    manifest_path = export_root / "data/training_blocks_manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    rows = _rows(manifest_path)
    with BASE_MANIFEST.open(encoding="utf-8") as handle:
        base_manifest = json.load(handle)
    source_train = Path(base_manifest["training_view"]).resolve()
    standard_hdf5 = run_root / "data/standard_teacher/ordinary_teacher.hdf5"
    output_dir = run_root / "data/training_hdf5"
    output_dir.mkdir(parents=True, exist_ok=True)
    arm_paths = {}
    arm_counts = {}
    for arm in ("REPLAY_ONLY", "MATCHED_STANDARD_DATA", "FIXED_L80_RECOVERY", "HANDOFF_RECOVERY"):
        path = output_dir / f"{arm}.hdf5"
        if path.exists():
            path.unlink()
        with h5py.File(path, "w") as out:
            data = out.create_group("data")
            mask = out.create_group("mask")
            for demo_id in base_manifest["train_demo_ids"]:
                data[demo_id] = h5py.ExternalLink(str(source_train), f"/data/{demo_id}")
            d0_ids = list(base_manifest["train_demo_ids"])
            mask.create_dataset("d0", data=np.asarray(d0_ids, dtype="S"))
            if arm != "REPLAY_ONLY":
                selected = [row for row in rows if row["arm"] == arm]
                new_ids = []
                for row in selected:
                    if row["source_branch"] == "STANDARD":
                        cache, actions = _load_standard(row, standard_hdf5)
                    else:
                        cache, actions = _load_recovery(row)
                    _copy_block(data, row, cache, actions)
                    new_ids.append(_list_value(row["event_ids"])[0].replace(":", "__"))
                mask.create_dataset("new", data=np.asarray(new_ids, dtype="S"))
            else:
                mask.create_dataset("new", data=np.asarray([], dtype="S"))
            out.attrs["hb4_arm"] = arm
            out.attrs["hb4_base_source_manifest"] = str(BASE_MANIFEST.resolve())
            out.attrs["hb4_base_checkpoint_sha256"] = sha256_file(BASE_CHECKPOINT)
            out.attrs["hb4_new_blocks"] = 0 if arm == "REPLAY_ONLY" else sum(1 for row in rows if row["arm"] == arm)
            out.flush()
        arm_paths[arm] = str(path.resolve())
        arm_counts[arm] = 0 if arm == "REPLAY_ONLY" else sum(int(row["block_length"]) for row in rows if row["arm"] == arm)
    manifest = {
        "schema_version": "hb4_training_hdf5_manifest_v1", "base_source_manifest": str(BASE_MANIFEST.resolve()),
        "base_training_view": str(source_train), "base_demo_count": len(base_manifest["train_demo_ids"]),
        "arms": arm_paths, "unique_new_labels": arm_counts,
        "new_data_is_external_source_only": False, "d0_is_external_linked": True,
        "student_observation_keys": list(OBS_KEYS), "sequence_length": 10,
    }
    atomic_json_dump(manifest, export_root / "data/training_hdf5_manifest.json")
    return {"arms": arm_paths, "unique_new_labels": arm_counts, "status": "PASS_HB4_C_DATA_MATERIALIZED"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-root", type=Path, default=EXPORT_ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    args = parser.parse_args()
    print(json.dumps(build(args.export_root, args.run_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
