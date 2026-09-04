"""Emit a JSON runtime fingerprint for a candidate replay environment."""
from __future__ import annotations

import importlib
import contextlib
import io
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def package(name: str) -> dict:
    try:
        # Several legacy libraries print import warnings to stdout. The
        # fingerprint is intended to be machine-readable JSON.
        with contextlib.redirect_stdout(io.StringIO()):
            module = importlib.import_module(name)
        return {
            "version": getattr(module, "__version__", None),
            "file": getattr(module, "__file__", None),
        }
    except Exception as exc:
        return {"error": repr(exc)}


def git_head_for(module_path: str | None) -> dict | None:
    if not module_path:
        return None
    path = Path(module_path).resolve()
    for parent in (path, *path.parents):
        if (parent / ".git").exists():
            try:
                return {
                    "root": str(parent),
                    "head": subprocess.check_output(["git", "-C", str(parent), "rev-parse", "HEAD"], text=True).strip(),
                    "status": subprocess.check_output(["git", "-C", str(parent), "status", "--short"], text=True).strip(),
                }
            except Exception:
                return {"root": str(parent)}
    return None


def main() -> None:
    result = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "env": {key: os.environ.get(key) for key in ("PYTHONPATH", "LD_LIBRARY_PATH", "MUJOCO_PY_MUJOCO_PATH", "MUJOCO_GL", "CUDA_VISIBLE_DEVICES")},
        "packages": {},
    }
    for name in ("numpy", "h5py", "robomimic", "robosuite", "mujoco", "mujoco_py", "mimicgen"):
        result["packages"][name] = package(name)
    for name in ("robomimic", "robosuite", "mimicgen"):
        result["packages"][name]["git"] = git_head_for(result["packages"][name].get("file"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
