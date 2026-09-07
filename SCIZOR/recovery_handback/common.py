"""Small shared helpers for the HB1 execution protocol."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def stable_seed(*parts: object) -> int:
    text = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") % (2**31 - 1)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _table_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return value


def write_table(rows: Iterable[dict], path: Path) -> None:
    """Atomically write records as real Parquet or JSONL based on suffix."""
    path = Path(path)
    records = [{str(key): _table_value(value) for key, value in row.items()} for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        if path.suffix == ".parquet":
            import pandas as pd

            frame = pd.DataFrame.from_records(records)
            for column in frame.columns:
                if frame[column].dtype != object:
                    continue
                values = frame[column].dropna().tolist()
                scalar_types = {type(value) for value in values}
                if len(scalar_types) > 1:
                    frame[column] = frame[column].map(lambda value: None if value is None else str(value))
            frame.to_parquet(tmp_path, index=False, engine="pyarrow")
        else:
            tmp_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in records)
                + ("\n" if records else ""),
                encoding="utf-8",
            )
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def read_table(path: Path) -> list[dict]:
    path = Path(path)
    if path.is_dir():
        parquet = path / "roots.parquet"
        jsonl = path / "roots.jsonl"
        path = parquet if parquet.is_file() else jsonl
    if path.suffix == ".parquet":
        import pandas as pd

        frame = pd.read_parquet(path, engine="pyarrow")
        return frame.where(frame.notna(), None).to_dict(orient="records")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def flatten_numeric(value: Any) -> np.ndarray:
    parts: list[np.ndarray] = []

    def visit(item: Any) -> None:
        try:
            import torch
        except ImportError:  # pragma: no cover - torch is part of the HB1 runtime
            torch = None
        if torch is not None and torch.is_tensor(item):
            parts.append(item.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1))
        elif isinstance(item, np.ndarray):
            parts.append(item.astype(np.float64, copy=False).reshape(-1))
        elif isinstance(item, dict):
            for key in sorted(item):
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif isinstance(item, (int, float, bool, np.number)):
            parts.append(np.asarray([item], dtype=np.float64))

    visit(value)
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
