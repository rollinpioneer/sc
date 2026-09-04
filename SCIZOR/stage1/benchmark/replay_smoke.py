import argparse, json
from pathlib import Path
import h5py, imageio, numpy as np
from .simulator_replay import create_shaped_env, replay_episode

def first_success_demo(env, dataset):
    with h5py.File(dataset, 'r') as f:
        for key in sorted(f['data']):
            demo=f['data'][key]
            if 'states' not in demo or 'actions' not in demo: continue
            state={'states': demo['states'][0]}
            stored=demo['obs/agentview_image'][0] if 'obs/agentview_image' in demo else None
            result=replay_episode(env,state,demo['actions'][:])
            if bool(result['success'][-1]):
                return key, state, demo['actions'][:], stored, result
    raise RuntimeError('no successful replayable demonstration found')

def run(task, dataset, output_dir):
    env, _=create_shaped_env(dataset)
    key, state, actions, stored, result=first_success_demo(env,dataset)
    env.close()
    raw=float(np.mean(np.abs(result['images'][0].astype(float)-stored.astype(float)))) if stored is not None else None
    flip=float(np.mean(np.abs(result['images'][0][::-1].astype(float)-stored.astype(float)))) if stored is not None else None
    flip_applied=bool(flip is not None and flip < raw)
    frames=result['images'][:, ::-1] if flip_applied else result['images']
    video=Path(output_dir)/(task+'_open_loop.mp4'); imageio.mimsave(video,frames[::2],fps=10)
    return {'task':task,'dataset':dataset,'base_demo_id':key,'episode_length':int(len(actions)),'final_success':bool(result['success'][-1]),'video_path':str(video),'render_flip_applied':flip_applied,'render_err_raw':raw,'render_err_flip':flip}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--can',required=True); p.add_argument('--square',required=True); p.add_argument('--output-dir',required=True); args=p.parse_args()
    Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    records=[run('can',args.can,args.output_dir),run('square',args.square,args.output_dir)]
    Path(args.output_dir,'smoke_summary.json').write_text(json.dumps(records,indent=2))
    if not all(x['final_success'] for x in records): raise SystemExit('clean replay did not end in task success')

if __name__=='__main__': main()
