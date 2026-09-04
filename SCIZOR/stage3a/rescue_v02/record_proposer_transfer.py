"""Record the frozen proposer-transfer gate without using forbidden legacy data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ceiling", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    ceiling = json.loads(args.ceiling.read_text(encoding="utf-8"))
    # Proposer transfer is a downstream check.  The protocol explicitly gates
    # it on a passing oracle ceiling; no v0.2 recall is inferred from v0.1.
    passed = bool(ceiling.get("oracle_ceiling_pass"))
    result = {
        "evaluated": False,
        "full_top5_recall": None,
        "action_only_top5_recall": None,
        "protocol_gate": "oracle_ceiling_pass",
        "protocol_gate_value": passed,
        "reason": "skipped because v0.2 oracle ceiling did not pass; v0.1 proposer scores are not v0.2 transfer evidence",
        "frozen_checkpoints": {
            "full": "../stage2/runs/full_seed_0/best.pt",
            "action_only": "../stage2/runs/action_only_seed_0/best.pt",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
