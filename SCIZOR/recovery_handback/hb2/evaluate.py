"""Evaluate frozen predictions and observed finite branches on new roots."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from recovery_handback.common import atomic_json_dump, read_table, write_table
from recovery_handback.hb2.metrics import choose_length, root_mean


def _load_predictions(path: Path) -> dict[tuple[str, str], dict]:
    result = {}
    for file in sorted(path.glob("*.json")):
        if file.name == "summary.json": continue
        payload = json.loads(file.read_text(encoding="utf-8"))
        for row in payload.get("records", []): result[(payload["model"], payload["seed"], row["example_id"])] = row
    return result


def _method_rows(rows: list[dict], predictions: list[dict], method: str, lambda_value: float,
                 risk_threshold: float = 0.5, risk_length: int = 20) -> list[dict]:
    by_id = {row["example_id"]: row for row in rows}
    output = []
    for prediction in predictions:
        row = by_id[prediction["example_id"]]
        if method == "fixed_none": length = 0
        elif method.startswith("fixed_l"): length = int(method[7:])
        elif method == "risk_fixed": length = risk_length if 1 - float(prediction["p0"]) >= risk_threshold else 0
        elif method == "oracle":
            scores = [(int(row["y0"]), 0)] + [(int(row[f"y_genuine_l{l}"]) - lambda_value * int(row[f"helper_steps_l{l}"]) / 400.0, l) for l in (5,20,80)]
            length = max(scores, key=lambda item: (item[0], -item[1]))[1]
        else:
            length = choose_length(prediction, lambda_value)
        if length == 0:
            y = int(row["y0"]); sys_y = y; cost = 0
        else:
            y = int(row[f"y_genuine_l{length}"]); sys_y = int(row[f"category_l{length}"] != 0); cost = int(row[f"helper_steps_l{length}"])
        output.append({"example_id": row["example_id"], "stat_group_id": row["stat_group_id"], "root_id": row["root_id"],
                       "selected_length": length, "y_autonomous": y, "y_system": sys_y, "y0": int(row["y0"]),
                       "helper_steps_actual": cost, "utility": y - lambda_value * cost / 400.0,
                       "interference": int(bool(row["y0"]) and not bool(sys_y)) if length else 0,
                       "rescue": int(not bool(row["y0"]) and bool(y))})
    return output


def _aggregate(values: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in values: grouped[row["stat_group_id"]].append(row)
    root_values = [{"stat_group_id": root, "utility": float(np.mean([x["utility"] for x in items])),
                    "y_autonomous": float(np.mean([x["y_autonomous"] for x in items])),
                    "y_system": float(np.mean([x["y_system"] for x in items])),
                    "helper_steps_actual": float(np.mean([x["helper_steps_actual"] for x in items])),
                    "interference": sum(x["interference"] for x in items), "interference_denominator": sum(x["y0"] for x in items),
                    "rescue": sum(x["rescue"] for x in items)} for root, items in grouped.items()]
    return {"valid_roots": len(root_values), "valid_anchors": len(values),
            "primary_utility_U_lambda_0p25": float(np.mean([x["utility"] for x in root_values])) if root_values else None,
            "autonomous_completion_rate": float(np.mean([x["y_autonomous"] for x in root_values])) if root_values else None,
            "root_mean_system_success": float(np.mean([x["y_system"] for x in root_values])) if root_values else None,
            "mean_helper_steps_actual": float(np.mean([x["helper_steps_actual"] for x in root_values])) if root_values else None,
            "helped_fraction": float(np.mean([x["helper_steps_actual"] > 0 for x in values])) if values else None,
            "interference_count": int(sum(x["interference"] for x in values)),
            "interference_denominator": int(sum(x["y0"] and x["selected_length"] > 0 for x in values)),
            "genuine_rescue_anchors": int(sum(x["rescue"] for x in values)),
            "genuine_rescue_unique_roots": int(len({x["stat_group_id"] for x in values if x["rescue"]})),
            "root_values": root_values}


def _bootstrap(method_values: dict[str, list[dict]], reference: str, repeats: int, seed: int) -> dict:
    roots = sorted({row["stat_group_id"] for values in method_values.values() for row in values})
    rng = np.random.default_rng(seed); diffs = []
    for _ in range(repeats):
        sampled = rng.choice(roots, size=len(roots), replace=True)
        means = {}
        for method, values in method_values.items():
            grouped = defaultdict(list)
            for row in values: grouped[row["stat_group_id"]].append(row["utility"])
            means[method] = float(np.mean([np.mean(grouped[root]) for root in sampled]))
        for method in method_values:
            if method != reference: diffs.append({"method": method, "reference": reference, "difference": means[method] - means[reference]})
    result = {}
    for method in method_values:
        if method == reference: continue
        values = np.asarray([x["difference"] for x in diffs if x["method"] == method])
        result[f"{method}_vs_{reference}"] = {"mean_bootstrap": float(values.mean()), "ci95_percentile": [float(np.quantile(values, .025)), float(np.quantile(values, .975))]}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--handoff-protocol", type=Path); parser.add_argument("--dataset", type=Path, required=True); parser.add_argument("--predictions", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); config=json.loads(args.config.read_text(encoding="utf-8")); protocol=json.loads(args.protocol.read_text(encoding="utf-8"))
    rows=[row for row in read_table(args.dataset/'anchor_examples.parquet') if row.get('split')=='test' and row.get('complete_pair')]
    pred_index=_load_predictions(args.predictions); selected=protocol['selected_visual_model']; canonical=next((x for x in pred_index.values() if x.get('model')==selected and x.get('seed')=='seed0'), None)
    if canonical is None: raise SystemExit('canonical selected prediction missing')
    visual=[x for x in pred_index.values() if x.get('model')==selected and x.get('seed')=='seed0']
    methods={}
    for name in ('fixed_none','fixed_l5','fixed_l20','fixed_l80','oracle','risk_fixed'):
        methods[name]=_method_rows(rows, visual, name, .25)
    for model_id in ('M0_time','M1_proprio',selected):
        values=[x for x in pred_index.values() if x.get('model')==model_id and x.get('seed')=='seed0']
        if values: methods[f'model_{model_id}']=_method_rows(rows, values, f'model_{model_id}', .25)
    aggregates={name:_aggregate(values) for name, values in methods.items()}
    reference=protocol.get('selection',{}).get('validation_best_comparator','fixed_none')
    if reference not in methods: reference='fixed_none'
    bootstrap=_bootstrap(methods, reference, int(config['hb2']['statistics']['bootstrap_repeats']), int(config['hb2']['statistics']['seed']))
    test_roots=len({row['stat_group_id'] for row in rows}); rescuable=len({row['stat_group_id'] for row in rows if any(not row['y0'] and row[f'y_genuine_l{l}'] for l in (5,20,80))})
    payload={'schema_version':'hb2_test_metrics_v1','selected_visual_model':selected,'canonical_seed':'seed0','valid_roots':test_roots,
             'valid_anchors':len(rows),'full_branch_coverage':all(row.get('complete_pair') for row in rows),'oracle_rescuable_roots':rescuable,
             'methods':{name:{key:value for key,value in metric.items() if key!='root_values'} for name,metric in aggregates.items()},
             'bootstrap':bootstrap,'bootstrap_reference':reference,'method_rows':methods}
    args.output_dir.mkdir(parents=True,exist_ok=True); atomic_json_dump(payload,args.output_dir/'summary.json')
    flat=[]
    for name,metric in aggregates.items(): flat.append({'method':name,**{key:value for key,value in metric.items() if key!='root_values'}})
    write_table(flat,args.output_dir/'methods.csv'); print(json.dumps({'valid_roots':test_roots,'selected':selected,'reference':reference},indent=2))


if __name__=='__main__': main()

