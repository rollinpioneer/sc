"""HDF5-only Stage 1A trainer; avoids unused DALI/OXE imports."""
import argparse, json, os, random
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
from pathlib import Path
import numpy as np
import torch
from curation.suboptimal_classifier.config.config import get_config
from curation.suboptimal_classifier.dataset.hdf5_dataset import HDF5Dataset
from curation.suboptimal_classifier.discriminator.discriminator import Discriminator

def main():
 p=argparse.ArgumentParser(); p.add_argument('--data-dir',required=True); p.add_argument('--save-root',required=True); p.add_argument('--seed',type=int,default=0); p.add_argument('--steps',type=int,default=10000); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--save-interval',type=int,default=2000); a=p.parse_args()
 random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
 c=get_config(); c.seed=a.seed; c.num_steps=a.steps; c.save_interval=a.save_interval; c.hdf5_dataset_kwargs.data_dir=a.data_dir; c.hdf5_dataset_kwargs.batch_size=a.batch_size; c.hdf5_dataset_kwargs.num_workers=4; c.future_image=True; c.window_size=1; c.action_horizon=1; c.future_action=0
 d=c.discriminator; c.discriminator_dataset_kwargs.image_key='agentview_image'; d.action_query_length=1; d.num_blocks=6; d.head_token='cls'; d.no_action_input=True; d.no_text_input=True; d.frozen_encoder=True; d.encoder_type='dinov2'; d.d_model=768; d.fusion_blocks_type='self-attn'; d.head_type='rank'; d.loss_fn_type='cross_entropy'; d.window_size=1; d.action_horizon=1; d.future_action=0; d.future_image=True
 c.optimizer.lr=1e-4
 out=Path(a.save_root); out.mkdir(parents=True,exist_ok=True); (out/'config.json').write_text(json.dumps(c.to_dict(),indent=2))
 dataset=HDF5Dataset(c); model=Discriminator(**d).cuda(); optim=torch.optim.AdamW(model.parameters(),**c.optimizer); sched=torch.optim.lr_scheduler.OneCycleLR(optim,c.optimizer.lr,total_steps=a.steps,pct_start=c.scheduler.pct_start)
 model.train()
 for step in range(a.steps):
  images,_,scores,_,_,_,_=next(dataset); images,scores=images.cuda(non_blocking=True),scores.cuda(non_blocking=True)
  result=model(images,None,None,score=scores,training=True); loss=result['loss']; loss.backward(); optim.step(); optim.zero_grad(); sched.step()
  print(f'step={step+1} loss={loss.item():.6f}',flush=True)
  if (step+1)%a.save_interval==0: torch.save(model.state_dict(),out/f'model_{step+1}.pth')
if __name__=='__main__': main()
