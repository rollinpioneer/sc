"""Synchronous local file-queue client for frozen M4 inference."""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from recovery_handback.common import atomic_json_dump, sha256_file
from recovery_handback.hb2.metrics import choose_length
from recovery_handback.hb3p.predictors import arrays_sha256, history_arrays


class FileQueueClient:
    def __init__(self, queue_dir: Path, protocol_path: Path, *, timeout_seconds: float = 300.0):
        self.queue_dir = Path(queue_dir)
        self.protocol_path = Path(protocol_path)
        self.protocol_hash = sha256_file(self.protocol_path)
        self.protocol = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        self.timeout_seconds = float(timeout_seconds)
        for name in ("requests", "responses", "payloads"):
            (self.queue_dir / name).mkdir(parents=True, exist_ok=True)

    def predict(self, history: list[dict], absolute_t: int, method_id: str, root_id: str) -> dict:
        arrays = history_arrays(history)
        request_id = f"{self.protocol_hash}:{method_id}:{root_id}:{int(absolute_t)}"
        file_id = __import__("hashlib").sha256(request_id.encode("utf-8")).hexdigest()
        payload_path = self.queue_dir / "payloads" / f"{file_id}.npz"
        fd, tmp_name = tempfile.mkstemp(prefix=f".{file_id}.", suffix=".npz", dir=str(payload_path.parent))
        os.close(fd)
        try:
            np.savez_compressed(tmp_name, **arrays)
            os.replace(tmp_name, payload_path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        request = {
            "request_id": request_id,
            "protocol_hash": self.protocol_hash,
            "model_id": "M4_paired",
            "absolute_t": int(absolute_t),
            "payload_npz": str(payload_path.resolve()),
            "payload_sha256": sha256_file(payload_path),
            "history_sha256": arrays_sha256(arrays),
        }
        request_path = self.queue_dir / "requests" / f"{file_id}.ready.json"
        response_path = self.queue_dir / "responses" / f"{file_id}.json"
        if request_path.exists():
            raise RuntimeError(f"duplicate M4 request already in flight: {request_id}")
        response_path.unlink(missing_ok=True)
        atomic_json_dump(request, request_path)
        started = time.perf_counter()
        while time.perf_counter() - started < self.timeout_seconds:
            if response_path.is_file():
                response = json.loads(response_path.read_text(encoding="utf-8"))
                if response.get("request_id") != request_id or response.get("protocol_hash") != self.protocol_hash:
                    raise RuntimeError("M4 response identity mismatch")
                if response.get("model_id") != "M4_paired":
                    raise RuntimeError("M4 response model mismatch")
                if int(response.get("absolute_t", -1)) != int(absolute_t):
                    raise RuntimeError("M4 response absolute-time mismatch")
                if response.get("payload_sha256") != request["payload_sha256"]:
                    raise RuntimeError("M4 response payload mismatch")
                if response.get("history_sha256") != request["history_sha256"]:
                    raise RuntimeError("M4 response history mismatch")
                if not response.get("engineering_ok", False):
                    raise RuntimeError(response.get("exception_reason", "M4 worker failed"))
                decision = self.protocol["decision"]
                selected = choose_length(
                    response,
                    float(decision["primary_lambda"]),
                    denominator=float(decision["cost_denominator"]),
                    tolerance=float(decision["tie_tolerance"]),
                )
                if int(response.get("selected_length", -1)) != selected:
                    raise RuntimeError("M4 worker selected length disagrees with frozen client formula")
                response["client_selected_length"] = selected
                response["ipc_roundtrip_seconds"] = time.perf_counter() - started
                response_path.unlink(missing_ok=True)
                payload_path.unlink(missing_ok=True)
                return response
            time.sleep(0.02)
        raise TimeoutError(f"M4 inference timed out: {request_id}")
