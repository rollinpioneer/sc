from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(); p.add_argument("--inputs", nargs="+", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--id-key", default="pair_id"); p.add_argument("--allow-duplicate-id", action="store_true"); a = p.parse_args()
    rows = []
    for path in a.inputs:
        rows.extend(json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip())
    if not a.allow_duplicate_id:
        ids = [str(r[a.id_key]) for r in rows]
        if len(ids) != len(set(ids)): raise ValueError(f"duplicate {a.id_key}")
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(a.output)}, indent=2))


if __name__ == "__main__": main()
