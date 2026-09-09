"""Persistent DINOv2/M4 inference worker for HB3-P."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from recovery_handback.common import atomic_json_dump, sha256_file
from recovery_handback.hb2.runtime import RuntimePredictor


def serve(config_path: Path, protocol_path: Path, queue_dir: Path, device: str, stop_file: Path | None) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_hash = sha256_file(protocol_path)
    predictor = RuntimePredictor(
        config_path,
        protocol["hb2_f_protocol"],
        device=device,
    )
    requests = queue_dir / "requests"
    responses = queue_dir / "responses"
    requests.mkdir(parents=True, exist_ok=True)
    responses.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"ready": True, "protocol_hash": protocol_hash, "device": str(predictor.device)}), flush=True)
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
                "engineering_ok": False,
                "exception_reason": None,
            }
            try:
                if request.get("protocol_hash") != protocol_hash:
                    raise ValueError("request protocol hash mismatch")
                payload_path = Path(request["payload_npz"])
                if sha256_file(payload_path) != request["payload_sha256"]:
                    raise ValueError("request payload hash mismatch")
                with np.load(payload_path, allow_pickle=False) as handle:
                    rgb = np.asarray(handle["rgb"], dtype=np.uint8)
                    proprio = np.asarray(handle["proprio"], dtype=np.float32)
                    actions = np.asarray(handle["base_actions"], dtype=np.float32)
                    times = np.asarray(handle["absolute_times"], dtype=np.int64)
                if rgb.ndim != 5 or rgb.shape[1] != 2 or len(rgb) != len(proprio):
                    raise ValueError(f"invalid M4 payload shapes: {rgb.shape}, {proprio.shape}")
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
