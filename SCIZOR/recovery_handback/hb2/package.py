"""Package only lightweight HB2 summaries and source code."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--experiment-root',type=Path,required=True); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args(); root=args.experiment_root
    worktree=Path(__file__).resolve().parents[3]; files=[]
    for path in sorted((root/'config').glob('*.json'))+sorted((root/'metrics').glob('**/*.json'))+sorted((root/'metrics').glob('**/*.csv'))+sorted((root/'report').glob('*.md')):
        if path.is_file(): files.append((path,path.relative_to(root)))
    for path in sorted((worktree/'SCIZOR/recovery_handback/hb2').glob('*.py')):
        files.append((path,Path('code')/path.name))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path,arcname in files: archive.write(path,arcname.as_posix())
    digest=hashlib.sha256(args.output.read_bytes()).hexdigest(); args.output.with_suffix('.zip.sha256').write_text(digest+'  '+args.output.name+'\n',encoding='utf-8')
    print(json.dumps({'output':str(args.output),'files':len(files),'sha256':digest},indent=2))


if __name__=='__main__': main()

