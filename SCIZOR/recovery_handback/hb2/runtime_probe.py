"""Check the runtime/offline F input boundary on pilot data."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from recovery_handback.common import atomic_json_dump, read_table
from recovery_handback.hb2.features import CAMERA_KEYS, _anchor_history, _proprio
from recovery_handback.hb2.runtime import RuntimePredictor


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--config',type=Path,required=True); parser.add_argument('--protocol',type=Path,required=True); parser.add_argument('--pilot-roots',type=Path,required=True); parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(); config=json.loads(args.config.read_text()); anchors=read_table(args.pilot_roots/'anchors.parquet')
    result={'schema_version':'hb2_runtime_probe_v1','pilot_anchors':len(anchors),'status':'not_ready','checks':[]}
    if not anchors: atomic_json_dump(result,args.output_dir/'runtime_probe.json'); return
    row=anchors[0]; started=time.perf_counter(); frames,actions,times=_anchor_history(row,int(config['horizon_steps']))
    finite=all(np.isfinite(_proprio(frame)).all() and all(np.isfinite(np.asarray(frame[key])).all() for key in CAMERA_KEYS) for frame in frames) and np.isfinite(actions).all()
    result.update({'status':'input_boundary_ready' if finite else 'invalid_input','example_id':row['anchor_id'],'anchor_t':int(row['anchor_t']),'feature_prepare_seconds':time.perf_counter()-started,'finite':bool(finite),'camera_order':list(CAMERA_KEYS),'action_source':'rollout suggestions at exact absolute times','repair_calls_after_handoff':0,'online_adaptive_switching_tested':False})
    if finite:
        try:
            predictor = RuntimePredictor(args.config, args.protocol)
            history_proprio = np.stack([_proprio(frame) for frame in frames])
            prediction = predictor.predict_before_takeover(frames, history_proprio, actions, int(row['anchor_t']))
            result['prediction'] = prediction
            result['inference_seconds'] = prediction['inference_seconds']
            result['status'] = 'runtime_offline_input_ready'
        except Exception as exc:
            result['status'] = 'runtime_model_error'
            result['runtime_error'] = f'{type(exc).__name__}: {exc}'
    args.output_dir.mkdir(parents=True,exist_ok=True); atomic_json_dump(result,args.output_dir/'runtime_probe.json'); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
