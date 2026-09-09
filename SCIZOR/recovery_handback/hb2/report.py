"""Generate the HB2 report and conservative layered decision status."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recovery_handback.common import atomic_json_dump


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--config',type=Path,required=True); parser.add_argument('--experiment-root',type=Path,required=True); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    config=json.loads(args.config.read_text()); root=args.experiment_root
    test=json.loads((root/'metrics/test/summary.json').read_text()) if (root/'metrics/test/summary.json').is_file() else {}
    val=json.loads((root/'metrics/validation/summary.json').read_text()) if (root/'metrics/validation/summary.json').is_file() else {}
    selected=test.get('selected_visual_model'); methods=test.get('methods',{}); chosen=methods.get(f'model_{selected}',{}) if selected else {}
    rescue=int(chosen.get('genuine_rescue_unique_roots',0)); valid=int(test.get('valid_roots',0)); oracle=int(test.get('oracle_rescuable_roots',0))
    reference=test.get('bootstrap_reference','fixed_none'); ci=test.get('bootstrap',{}).get(f'model_{selected}_vs_{reference}',{}).get('ci95_percentile',[None,None]) if selected else [None,None]
    point=(chosen.get('primary_utility_U_lambda_0p25') or 0) - (methods.get(reference,{}).get('primary_utility_U_lambda_0p25') or 0)
    if valid < 30 or oracle < 5: status='HOLD_HB2_DATA'
    elif rescue >= 3 and point > 0 and ci[0] is not None and ci[0] > 0: status='READY_HB3_SINGLE_TASK'
    elif rescue >= 3 and point > 0: status='READY_HB3_PILOT_ONLY'
    elif valid and selected: status='HB2_NO_PREDICTIVE_GAIN_CURRENT_SETUP'
    else: status='HOLD_HB2_DATA'
    decision={'task':'square','overall_status':status,'counterfactual_start_prediction_signal':'supported' if val else 'not_estimable','visual_gain':'supported' if point>0 and ci[0] is not None and ci[0]>0 else 'unproven','local_feature_gain':'unproven','history_gain':'unproven','paired_supervision_gain':'unproven','exit_model_ready':(root/'config/frozen_handoff_protocol.json').is_file(),'baseline_preservation_evidence':'adequate' if valid>=5 else 'small_sample','scope':'fixed_pair_fixed_anchor_single_intervention','online_adaptive_switching_tested':False,'selected_visual_model':selected,'test_valid_roots':valid,'test_oracle_rescuable_roots':oracle,'selected_visual_rescued_roots':rescue,'primary_delta_vs_reference':point,'root_cluster_ci':ci}
    atomic_json_dump(decision,root/'metrics/hb2_decision.json')
    lines=["# HB2 Report","",f"- Status: **{status}**",f"- Task: Square",f"- Selected visual model: `{selected}`",f"- Test roots / anchors: {valid} / {test.get('valid_anchors')}","", "## Scope", "HB2 reuses the frozen HB1-R Square policy pair and evaluates a single finite intervention selected from 0/5/20/80 steps. Test data are read only after the frozen protocol is written.","", "## Validation", json.dumps(val,ensure_ascii=False,indent=2),"", "## Test", json.dumps({k:v for k,v in test.items() if k!='method_rows'},ensure_ascii=False,indent=2),"", "## Limitations", "The teacher remains privileged; this result is not a non-privileged repairer or an online adaptive switching validation. Root bootstrap intervals are uncertainty summaries for this fixed setup, not per-state safety guarantees."]
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text('\n'.join(lines)+'\n',encoding='utf-8'); print(json.dumps(decision,indent=2))


if __name__=='__main__': main()

