"""Audit the frozen HB4 seed ranges before creating new evaluation roots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recovery_handback.common import atomic_json_dump


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(export_root: Path) -> dict:
    roles_path = export_root / "assets/data_roles.json"
    registry_path = export_root / "assets/seed_registry.json"
    roles = _load(roles_path)
    ranges = {
        name: (int(value["seed_start"]), int(value["seed_start"]) + int(value.get("count", value.get("seed_stop_exclusive", 0) - value["seed_start"])))
        for name, value in roles.items()
        if isinstance(value, dict) and "seed_start" in value and name != "historical_cohort"
    }
    collisions = []
    names = sorted(ranges)
    for index, left_name in enumerate(names):
        left_start, left_stop = ranges[left_name]
        for right_name in names[index + 1 :]:
            right_start, right_stop = ranges[right_name]
            overlap = max(left_start, right_start) < min(left_stop, right_stop)
            if overlap:
                collisions.append({"left": left_name, "right": right_name})

    historical = set(range(800000, 800400))
    reserved = {seed for start, stop in ranges.values() for seed in range(start, stop)}
    historical_collisions = sorted(historical & reserved)
    result = {
        "schema_version": "hb4_seed_collision_audit_v1",
        "ranges": {name: {"start": start, "stop_exclusive": stop, "count": stop - start} for name, (start, stop) in ranges.items()},
        "historical_range": {"start": 800000, "stop_exclusive": 800400, "count": 400},
        "range_collisions": collisions,
        "historical_collisions": historical_collisions,
        "passed": not collisions and not historical_collisions,
    }
    atomic_json_dump(result, export_root / "assets/seed_collision_audit.json")
    registry = _load(registry_path)
    registry["new_ranges_collision_checked"] = bool(result["passed"])
    registry["collision_audit_path"] = str((export_root / "assets/seed_collision_audit.json").resolve())
    roles["collision_audit"] = "PASS_HB4_SEED_COLLISION_AUDIT" if result["passed"] else "HOLD_HB4_SEED_COLLISION"
    registry["ranges"] = roles
    atomic_json_dump(roles, roles_path)
    atomic_json_dump(registry, registry_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.export_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
