"""Persistent DINOv2/M4 inference worker for HB3-P."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from recovery_handback.common import atomic_json_dump, sha256_file
from recovery_handback.hb2.runtime import RuntimePredictor
from recovery_handback.hb3p.predictors import arrays_sha256


def serve(config_path: Path, protocol_path: Path, queue_dir: Path, device: str, stop_file: Path | None) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_hash = sha256_file(protocol_path)
    hb2_protocol_path = Path(protocol["hb2_f_protocol"])
    if sha256_file(hb2_protocol_path) != protocol["hb2_f_protocol_sha256"]:
        raise ValueError("frozen HB2 F protocol hash mismatch")
    hb2_protocol = json.loads(hb2_protocol_path.read_text(encoding="utf-8"))
    if hb2_protocol["selected_visual_model"] != "M4_paired":
        raise ValueError("HB3-P M4 worker requires frozen M4_paired selection")
    hb2_m4 = hb2_protocol["models"]["M4_paired"]["seed0"]
    if hb2_m4["checkpoint_sha256"] != protocol["models"]["M4_paired"]["checkpoint_sha256"]:
        raise ValueError("HB3-P and HB2 M4 checkpoint identities differ")
    predictor = RuntimePredictor(
        config_path,
        hb2_protocol_path,
        device=device,
    )
    requests = queue_dir / "requests"
    responses = queue_dir / "responses"
    requests.mkdir(parents=True, exist_ok=True)
    responses.mkdir(parents=True, exist_ok=True)
    ready = {
        "ready": True,
        "pid": os.getpid(),
        "protocol_hash": protocol_hash,
        "model_id": "M4_paired",
        "checkpoint_sha256": hb2_m4["checkpoint_sha256"],
        "device": str(predictor.device),
    }
    atomic_json_dump(ready, queue_dir / "worker_ready.json")
    print(json.dumps(ready), flush=True)
    while not (stop_file and stop_file.exists()):
        paths = sorted(requests.glob("*.ready.json"))
        if not paths:
            time.sleep(0.05)
            continue
        for request_path in paths:
            file_id = request_path.name.removesuffix(".ready.json")
            response_path = responses / f"{file_id}.json"
            if response_path.is_file():
                request_path.unlink(missing_ok=True)
                continue
            request = json.loads(request_path.read_text(encoding="utf-8"))
            response = {
                "request_id": request.get("request_id"),
                "protocol_hash": protocol_hash,
                "model_id": request.get("model_id"),
                "absolute_t": request.get("absolute_t"),
                "payload_sha256": request.get("payload_sha256"),
                "history_sha256": request.get("history_sha256"),
                "engineering_ok": False,
                "exception_reason": None,
            }
            try:
                if request.get("protocol_hash") != protocol_hash:
                    raise ValueError("request protocol hash mismatch")
                if request.get("model_id") != "M4_paired":
                    raise ValueError("request model mismatch")
                payload_path = Path(request["payload_npz"])
                if sha256_file(payload_path) != request["payload_sha256"]:
                    raise ValueError("request payload hash mismatch")
                with np.load(payload_path, allow_pickle=False) as handle:
                    if set(handle.files) != {"rgb", "proprio", "base_actions", "absolute_times"}:
                        raise ValueError(f"invalid M4 payload keys: {sorted(handle.files)}")
                    rgb = np.asarray(handle["rgb"], dtype=np.uint8)
                    proprio = np.asarray(handle["proprio"], dtype=np.float32)
                    actions = np.asarray(handle["base_actions"], dtype=np.float32)
                    times = np.asarray(handle["absolute_times"], dtype=np.int64)
                count = len(rgb)
                if not 1 <= count <= 4:
                    raise ValueError(f"invalid M4 history length: {count}")
                if (
                    rgb.ndim != 5 or rgb.shape[1] != 2 or rgb.shape[-1] != 3
                    or proprio.shape != (count, 9)
                    or actions.shape != (count, 7)
                    or times.shape != (count,)
                ):
                    raise ValueError(
                        f"invalid M4 payload shapes: rgb={rgb.shape}, proprio={proprio.shape}, "
                        f"actions={actions.shape}, times={times.shape}"
                    )
                if not np.isfinite(proprio).all() or not np.isfinite(actions).all():
                    raise ValueError("non-finite M4 low-dimensional input")
                arrays = {
                    "rgb": rgb, "proprio": proprio, "base_actions": actions,
                    "absolute_times": times,
                }
                if arrays_sha256(arrays) != request.get("history_sha256"):
                    raise ValueError("request history hash mismatch")
                if int(times[-1]) != int(request["absolute_t"]):
                    raise ValueError("request absolute time disagrees with payload")
                if np.any(np.diff(times) < 0) or np.any(np.diff(times) > 1):
                    raise ValueError("M4 history times are not a padded contiguous prefix")
                history_rgb = [
                    {"agentview_image": rgb[index, 0], "robot0_eye_in_hand_image": rgb[index, 1]}
                    for index in range(len(rgb))
                ]
                result = predictor.predict_before_takeover(
                    history_rgb,
                    proprio,
                    actions,
                    int(times[-1]),
                )
                response.update(result)
                response["engineering_ok"] = True
                response["absolute_t"] = int(times[-1])
            except Exception as exc:
                response["exception_reason"] = f"{type(exc).__name__}: {exc}"
            atomic_json_dump(response, response_path)
            request_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stop-file", type=Path)
    args = parser.parse_args()
    serve(args.config, args.protocol, args.queue_dir, args.device, args.stop_file)


if __name__ == "__main__":
    main()
