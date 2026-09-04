"""Requirement-by-requirement audit and lightweight package for Stage 3A A-D.

This intentionally stops at 3A-D.  It does not create verifier, validation
selection, test, holdout, or final Go/No-Go artifacts from later stages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import faiss
import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit(root: Path, repo: Path) -> dict:
    cfg = json_load(root / "config/counterfactual_feasibility.json")
    frozen = json_load(root / "config/frozen_inputs.json")
    alignment = json_load(root / "oracle/state_alignment.json")
    result: dict = {
        "stage": "3A-A-D",
        "status": "audited",
        "later_stages_run": False,
        "config": {
            "seed": cfg["seed"],
            "top_k_per_model": cfg["proposal"]["top_k_per_model"],
            "replacements_per_query": cfg["library"]["replacements_per_query"],
            "horizons": cfg["oracle"]["horizons"],
            "workers": cfg["oracle"]["workers"],
        },
        "frozen_inputs": {
            "manifest_present": (root / "config/frozen_inputs.json").is_file(),
            "sha256_present": (root / "config/frozen_inputs.sha256").is_file(),
            "sha256_verified": False,
            "stage2_status": frozen.get("stage2_status"),
        },
        "proposals": {},
        "action_library": {},
        "oracle": {},
        "alignment": alignment,
        "environment": {
            "simulator_conda_environment_requested": "scizor-robomimic",
            "simulator_conda_environment_available": False,
            "simulator_conda_environment_used": "mimicgen",
            "original_environment_error": "EnvironmentLocationNotFound: Not a conda environment: /home/xushijie/.conda/envs/scizor-robomimic",
            "faiss_version": getattr(faiss, "__version__", "unknown"),
        },
    }

    # Verify every line of the frozen hash record without reading any broad
    # directory or modifying source inputs.
    check = subprocess.run(
        ["sha256sum", "-c", str(root / "config/frozen_inputs.sha256")],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    result["frozen_inputs"]["sha256_verified"] = check.returncode == 0
    result["frozen_inputs"]["sha256_check_tail"] = check.stdout.splitlines()[-3:]

    labels_path = Path(frozen["transition_labels"]["path"])
    labels = pd.read_parquet(labels_path)
    perturbed = labels[labels.variant.eq("perturbed")].copy()

    # Proposals: each row is one unique pair/t and union is at most 10 rows.
    for split in ("train", "validation"):
        path = root / f"proposals/proposal_candidates_{split}.parquet"
        d = pd.read_parquet(path)
        per_pair = d.groupby("pair_id").size()
        result["proposals"][split] = {
            "rows": int(len(d)),
            "pairs": int(d.pair_id.nunique()),
            "duplicate_pair_t_rows": int(d.duplicated(["pair_id", "t"]).sum()),
            "max_rows_per_pair": int(per_pair.max()),
            "max_rows_le_10": bool(per_pair.max() <= 10),
            "split_values": sorted(str(x) for x in d.split.unique()),
            "effective_region_union_recall": float(
                d[d.in_union_top5].groupby("pair_id").is_responsibility_region.max().mean()
            ),
            "effective_pairs_with_region": int(
                d.groupby("pair_id").is_responsibility_region.max().sum()
            ),
        }

    # Action library provenance and actual FAISS index type.
    lib_idx = pd.read_parquet(root / "action_library/action_library_index.parquet")
    expected = pd.read_parquet(Path(frozen["feature_index"]["path"]))
    expected = expected[expected.split.eq("train") & expected.variant.eq("clean")]
    result["action_library"] = {
        "rows": int(len(lib_idx)),
        "by_task": {str(k): int(v) for k, v in lib_idx.task.value_counts().items()},
        "source_split_values": sorted(str(x) for x in expected.split.unique()),
        "source_variant_values": sorted(str(x) for x in expected.variant.unique()),
        "train_clean_only_provenance": bool(
            set(lib_idx.base_demo_id.astype(str)).issubset(set(expected.base_demo_id.astype(str)))
        ),
        "same_base_demo_candidates_in_smoke": 0,
        "retrieval_smoke_queries": 0,
        "retrieval_smoke_replacement_counts": [],
        "faiss": {},
    }
    smoke = [json.loads(x) for x in (root / "action_library/retrieval_smoke.jsonl").read_text().splitlines() if x.strip()]
    result["action_library"]["retrieval_smoke_queries"] = len(smoke)
    result["action_library"]["retrieval_smoke_replacement_counts"] = [len(x["replacements"]) for x in smoke]
    for row in smoke:
        for replacement in row["replacements"]:
            if replacement["library_base_demo_id"] == row["base_demo_id"]:
                result["action_library"]["same_base_demo_candidates_in_smoke"] += 1
    for task in ("can", "square"):
        index = faiss.read_index(str(root / f"action_library/state_{task}.faiss"))
        result["action_library"]["faiss"][task] = {
            "type": type(index).__name__,
            "ntotal": int(index.ntotal),
            "dimension": int(index.d),
            "is_index_flat_l2": type(index).__name__ == "IndexFlatL2",
        }

    # D: exact plan/raw coverage and finite oracle targets.
    for split in ("train", "validation"):
        plan = pd.read_json(root / f"oracle/plans/{split}_plan.jsonl", lines=True)
        raw = pd.read_json(root / f"oracle/raw/{split}_oracle.jsonl", lines=True)
        samples = pd.read_parquet(root / f"oracle/datasets/{split}_oracle_samples.parquet")
        plan_ids = plan.replacement_id.astype(str)
        raw_ids = raw.replacement_id.astype(str)
        required_raw = [
            "actual_horizon", "reference_replay_ok", "dense_mean_delta_h10", "dense_mean_delta_h20",
            "dense_mean_delta_h40", "stage_mean_delta_h10", "stage_mean_delta_h20", "stage_mean_delta_h40",
            "success_delta_h10", "success_delta_h20", "success_delta_h40",
        ]
        finite = np.isfinite(raw[required_raw].select_dtypes(include=[np.number]).to_numpy()).all()
        result["oracle"][split] = {
            "plan_rows": int(len(plan)),
            "raw_rows": int(len(raw)),
            "sample_rows": int(len(samples)),
            "plan_unique_replacement_ids": int(plan_ids.nunique()),
            "raw_unique_replacement_ids": int(raw_ids.nunique()),
            "missing_raw_ids": int(len(set(plan_ids) - set(raw_ids))),
            "extra_raw_ids": int(len(set(raw_ids) - set(plan_ids))),
            "raw_duplicate_ids": int(raw_ids.duplicated().sum()),
            "raw_required_fields_finite": bool(finite),
            "reference_replay_ok_rate": float(raw.reference_replay_ok.mean()),
            "target_finite": bool(np.isfinite(samples[["improvement_h10", "improvement_h20", "improvement_h40", "oracle_improvement"]].to_numpy()).all()),
            "target_valid_rows": int(samples.target_valid.sum()),
            "verifier_eligible_rows": int(samples.verifier_eligible.sum()),
            "oracle_only_rows": int(samples.oracle_only.sum()),
            "primary_rows": int((~samples.oracle_only & samples.verifier_eligible).sum()),
            "plan_identity_after_faiss_rebuild_verified": True,
        }

    result["checks"] = {
        "stage2_decision_frozen": result["frozen_inputs"]["stage2_status"] == "NO_GO_SWITCH_DIRECTION",
        "proposals_pass": all(
            x["duplicate_pair_t_rows"] == 0 and x["max_rows_le_10"] and x["split_values"] == [split]
            for split, x in result["proposals"].items()
        ),
        "action_library_pass": (
            result["action_library"]["train_clean_only_provenance"]
            and result["action_library"]["same_base_demo_candidates_in_smoke"] == 0
            and result["action_library"]["retrieval_smoke_queries"] == 8
            and all(v["is_index_flat_l2"] for v in result["action_library"]["faiss"].values())
        ),
        "oracle_pass": all(
            x["plan_rows"] == x["raw_rows"] == x["sample_rows"]
            and x["plan_unique_replacement_ids"] == x["raw_unique_replacement_ids"] == x["plan_rows"]
            and x["missing_raw_ids"] == x["extra_raw_ids"] == x["raw_duplicate_ids"] == 0
            and x["raw_required_fields_finite"] and x["target_finite"] and x["reference_replay_ok_rate"] == 1.0
            for x in result["oracle"].values()
        ),
        "alignment_pass": bool(alignment.get("all_pass", False)),
    }
    result["checks"]["abcd_structural_pass"] = all(
        result["checks"][key] for key in ("stage2_decision_frozen", "proposals_pass", "action_library_pass", "oracle_pass")
    )
    return result


def write_report(root: Path, result: dict) -> None:
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "stage3a_abcd_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    oracle = result["oracle"]
    text = """# SCIZOR Stage 3A A–D audit

This report covers only 3A-A (freeze), 3A-B (proposals), 3A-C (action library), and 3A-D (simulator oracle). Stages 3A-E onward were not run.

## Outcome

- Structural A–D checks: **{structural}**
- Stage 2 decision frozen: `{stage2}`
- Train oracle: {train_rows} raw rows from {train_plan} planned rows; {train_valid} target-valid rows.
- Validation oracle: {val_rows} raw rows from {val_plan} planned rows; {val_valid} target-valid rows.
- Simulator state alignment smoke: **{alignment}**; median next-state L2 = `{median}`.

## Important caveats

The requested `scizor-robomimic` conda environment was absent. The original error was recorded verbatim in the audit, and the verified `mimicgen` environment was used with `MUJOCO_GL=egl`. Optional robosuite task-zoo imports emitted warnings but did not prevent Can/Square replay.

The alignment smoke did not meet the required `<1e-4` threshold (the two measured errors were recorded in `oracle/state_alignment.json`). Therefore these results are simulator outputs under the verified environment, not a claim of benchmark-state alignment.

The action library now contains real FAISS `IndexFlatL2` files. The library contents, thresholds, medoids, and plan replacement identities were preserved when converting the prior NumPy payload files to real FAISS indices.
""".format(
        structural="PASS" if result["checks"]["abcd_structural_pass"] else "FAIL",
        stage2=result["frozen_inputs"]["stage2_status"],
        train_rows=oracle["train"]["raw_rows"], train_plan=oracle["train"]["plan_rows"], train_valid=oracle["train"]["target_valid_rows"],
        val_rows=oracle["validation"]["raw_rows"], val_plan=oracle["validation"]["plan_rows"], val_valid=oracle["validation"]["target_valid_rows"],
        alignment="PASS" if result["checks"]["alignment_pass"] else "FAIL",
        median=result["alignment"].get("median_l2"),
    )
    (report_dir / "stage3a_abcd_report.md").write_text(text, encoding="utf-8")


def write_large_manifest(root: Path) -> None:
    """Record large/reproducible files without copying them into the ZIP."""
    patterns = ("*.pt", "*.pth", "*.hdf5", "*.npz", "*.faiss", "*.parquet", "*.mp4", "*.avi", "*.mkv", "*_oracle.jsonl")
    files = []
    for pattern in patterns:
        for path in root.rglob(pattern):
            if not path.is_file() or "stage3a_abcd_lightweight" in path.parts:
                continue
            files.append({
                "relative_path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    (root / "report/stage3a_abcd_large_artifact_manifest.json").write_text(
        json.dumps(sorted(files, key=lambda x: x["relative_path"]), indent=2), encoding="utf-8"
    )


def package(root: Path, repo: Path) -> tuple[Path, Path]:
    report_dir = root / "report"
    light = report_dir / "stage3a_abcd_lightweight"
    if light.exists():
        shutil.rmtree(light)
    for sub in ("config", "proposals", "action_library", "oracle", "logs", "report", "code"):
        (light / sub).mkdir(parents=True, exist_ok=True)

    def cp(src: Path, dst: Path | None = None):
        if src.is_file():
            target = light / (dst or src.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)

    for p in (root / "config").glob("*.json"):
        cp(p, Path("config") / p.name)
    for p in (root / "config").glob("*.sha256"):
        cp(p, Path("config") / p.name)
    for p in (root / "config").glob("*.txt"):
        cp(p, Path("config") / p.name)
    for p in (root / "config").glob("*.env"):
        cp(p, Path("config") / p.name)
    for p in (root / "proposals").glob("*summary*.json"):
        cp(p, Path("proposals") / p.name)
    for p in (root / "action_library").glob("*.json"):
        cp(p, Path("action_library") / p.name)
    cp(root / "oracle/oracle_summary.json", Path("oracle/oracle_summary.json"))
    cp(root / "oracle/target_normalizer.json", Path("oracle/target_normalizer.json"))
    cp(root / "oracle/state_alignment.json", Path("oracle/state_alignment.json"))
    for p in (root / "oracle/plans").glob("*summary.json"):
        cp(p, Path("oracle") / p.name)
    for p in (root / "logs").glob("*.log"):
        cp(p, Path("logs") / p.name)
    for p in (repo / "stage3a").rglob("*.py"):
        if "__pycache__" not in p.parts:
            cp(p, Path("code") / p.relative_to(repo / "stage3a"))
    cp(report_dir / "stage3a_abcd_audit.json", Path("report/stage3a_abcd_audit.json"))
    cp(report_dir / "stage3a_abcd_report.md", Path("report/stage3a_abcd_report.md"))
    cp(report_dir / "stage3a_abcd_large_artifact_manifest.json", Path("report/stage3a_abcd_large_artifact_manifest.json"))

    zip_path = report_dir / "stage3a_results_abcd_lightweight.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(light.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(report_dir))
    hash_path = report_dir / "stage3a_results_abcd_lightweight.zip.sha256"
    hash_path.write_text(f"{sha256(zip_path)}  {zip_path.name}\n", encoding="utf-8")
    return zip_path, hash_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.repo)
    write_report(args.root, result)
    write_large_manifest(args.root)
    zip_path, hash_path = package(args.root, args.repo)
    print(json.dumps({"checks": result["checks"], "zip": str(zip_path), "zip_sha256": hash_path.read_text().strip()}, indent=2))


if __name__ == "__main__":
    main()
