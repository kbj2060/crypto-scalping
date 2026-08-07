#!/usr/bin/env python3
"""Stage-1 corrected-vol Zigzag pretraining then Stage-2 triple-barrier finetuning."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from ensemble.deep_features.btc_deepfeat_dataset_20260806 import build_dataset

ZIG=ROOT/'data/splits/year_oos/btc_5m_zigzag_correctedvol_labels_20260806.parquet'
TRIPLE=ROOT/'data/splits/year_oos/btc_5m_tripbarrier_tradeoutcome_labels_flatsmooth_20260806.parquet'
OUT=ROOT/'tmp/btc_curriculum_transformer_r20'

def stage1_adapter() -> Path:
 OUT.mkdir(parents=True,exist_ok=True); out=OUT/'correctedvol_zigzag_stage1_soft_labels.parquet'
 raw=pd.read_parquet(ZIG)
 action=raw['zigzag_correctedvol_action'].to_numpy()
 if not set(np.unique(action).tolist()) <= {0,1,2}: raise ValueError('unexpected corrected-vol Zigzag action contract')
 pd.DataFrame({'timestamp':raw['timestamp'],'zigzag_correctedvol_action':action,'zigzag_soft_cash':(action==0).astype(float),'zigzag_soft_long':(action==1).astype(float),'zigzag_soft_short':(action==2).astype(float)}).to_parquet(out,index=False)
 return out

class Model(nn.Module):
 def __init__(self,width):
  super().__init__(); self.inp=nn.Linear(width,96); layer=nn.TransformerEncoderLayer(96,4,192,.1,batch_first=True,norm_first=True); self.enc=nn.TransformerEncoder(layer,3); self.head=nn.Linear(96,3)
 def forward(self,x):
  n=x.size(1); mask=torch.triu(torch.ones(n,n,device=x.device,dtype=torch.bool),1); return self.head(self.enc(self.inp(x),mask=mask)[:,-1])

class SequenceDataset(Dataset):
 def __init__(self,values,end_index,labels,window=96): self.values=values; self.end_index=np.asarray(end_index); self.labels=labels; self.window=window
 def __len__(self): return len(self.end_index)
 def __getitem__(self,i):
  end=int(self.end_index[i]); start=end-self.window+1
  if start<0: raise ValueError('sequence reaches before available causal history')
  return torch.from_numpy(self.values[start:end+1]),torch.tensor(int(self.labels[end]),dtype=torch.long)

def fit(model,values,end_index,labels,epochs,freeze_prefix):
 for i,block in enumerate(model.enc.layers):
  for p in block.parameters(): p.requires_grad=i>=freeze_prefix
 opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=3e-4,weight_decay=1e-4); loader=DataLoader(SequenceDataset(values,end_index,labels),batch_size=256,shuffle=True)
 for _ in range(epochs):
  for a,b in loader:
   z=model(a); loss=nn.functional.cross_entropy(z,b); opt.zero_grad();loss.backward();opt.step()

def evaluate(model,values,end_index,labels):
 loader=DataLoader(SequenceDataset(values,end_index,labels),batch_size=256); prob=[]; truth=[]
 with torch.no_grad():
  for x,y in loader: prob.append(torch.softmax(model(x),1).numpy()); truth.append(y.numpy())
 p=np.concatenate(prob); y=np.concatenate(truth); pred=p.argmax(1)
 return {'n':int(len(y)),'accuracy':float((pred==y).mean()),'log_loss':float(-np.log(np.clip(p[np.arange(len(y)),y],1e-8,1)).mean()),'cash_pred_fraction':float((pred==0).mean())}

def main():
 zig=build_dataset(window=96,train_stride=4,label_path=stage1_adapter(),hard_col='zigzag_correctedvol_action',soft_cols=['zigzag_soft_cash','zigzag_soft_long','zigzag_soft_short'])
 x=np.asarray(zig.feat_std,dtype=np.float32)
 if x.ndim!=2: raise ValueError(f'expected causal feature frame, got {x.shape}')
 model=Model(x.shape[-1]); fit(model,x,zig.end_idx['train'],zig.y_hard_all.astype(np.int64),3,0)
 stage1=evaluate(model,x,zig.end_idx['val'],zig.y_hard_all.astype(np.int64))
 triple=build_dataset(window=96,train_stride=4,label_path=TRIPLE,hard_col='trade_outcome_action',soft_cols=['trade_outcome_soft_cash','trade_outcome_soft_long','trade_outcome_soft_short'])
 if not np.array_equal(zig.end_idx['train'],triple.end_idx['train']): raise ValueError('stage datasets do not share the same causal rows')
 tx=np.asarray(triple.feat_std,dtype=np.float32); fit(model,tx,triple.end_idx['train'],triple.y_hard_all.astype(np.int64),3,2)
 stage2=evaluate(model,tx,triple.end_idx['val'],triple.y_hard_all.astype(np.int64))
 OUT.mkdir(parents=True,exist_ok=True); torch.save({'state_dict':model.state_dict(),'window':96},OUT/'stage2_model.pt'); (OUT/'report.json').write_text(json.dumps({'architecture':'3-block causal Transformer; no MoE, diffusion, or MAML','stage1_correctedvol_zigzag_val':stage1,'stage2_tripbarrier_val':stage2,'oos_opened':False,'promotion_eligible':False},indent=2)+'\n');print(json.dumps({'stage1':stage1,'stage2':stage2},indent=2))
if __name__=='__main__': main()
