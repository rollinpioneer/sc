"""Produce compact, auditable proposal coverage summaries."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--inputs", nargs="+", type=Path, required=True); p.add_argument("--output", type=Path, required=True); a=p.parse_args()
    rows=[]
    for path in a.inputs:
        d=pd.read_parquet(path)
        per=d.groupby("pair_id",as_index=False).agg(rows=("t","size"),has_responsible=("is_responsibility_region","max"))
        eff=d.groupby("pair_id",as_index=False).agg(is_effective=("is_effective_intervention","max"),responsible_covered=("is_responsibility_region","max"))
        q=per.merge(eff,on="pair_id")
        rows.append({"path":str(path),"split":str(d.split.iloc[0]),"rows":int(len(d)),"pairs":int(d.pair_id.nunique()),"max_rows_per_pair":int(per.rows.max()),"effective_pairs":int(q.is_effective.sum()),"effective_pairs_with_region":int((q.is_effective&q.responsible_covered).sum()),"effective_coverage":float((q.is_effective&q.responsible_covered).sum()/max(1,q.is_effective.sum())),"no_effect_pairs":int((~q.is_effective).sum())})
    out={"files":rows}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2),encoding="utf-8"); print(json.dumps(out,indent=2))
if __name__=="__main__": main()
