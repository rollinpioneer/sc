from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def info(name):
    try:
        module = importlib.import_module(name)
        return {"version": getattr(module, "__version__", None), "file": getattr(module, "__file__", None)}
    except Exception as exc:
        return {"error": repr(exc)}


def file_hash(path):
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(args, cwd):
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main():
    root = Path(os.environ["SCIZOR_ROOT"])
    payload = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "modules": {name: info(name) for name in ("robomimic", "robosuite", "mujoco", "numpy", "h5py", "mimicgen")},
        "scizor_git_head": git(["rev-parse", "HEAD"], str(root)),
        "scizor_git_status": git(["status", "--short"], str(root)),
        "important_file_hashes": {rel: file_hash(root / rel) for rel in (
            "stage1/benchmark/simulator_replay.py", "stage1/benchmark/perturbations.py",
            "stage1/benchmark/intervention_times.py", "stage1/benchmark/outcome_labels.py",
            "stage3a/rescue_v02/common.py")},
    }
    out = Path(os.environ["V02_ROOT"]) / "config" / "runtime_fingerprint.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(out), "scizor_git_head": payload["scizor_git_head"]}, indent=2))


if __name__ == "__main__":
    main()
