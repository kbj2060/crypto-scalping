#!/usr/bin/env python3
from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
import json
import torch
import sys
from typing import Dict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from ensemble.train_rl_dsac_agent import DSACRouter, GaussianActor, DSAC_STATE_DIM

BASE_ENV = {
    "DSAC_SPECIALISTS_ONLY": "1",
}

def load_actor(path: str):
    ckpt = torch.load(path, map_location="cpu")
    actor = GaussianActor(state_dim=int(ckpt.get("state_dim", DSAC_STATE_DIM)))
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor

def set_env(overrides: Dict[str,str]):
    backup = {k: os.environ.get(k) for k in overrides}
    for k,v in overrides.items():
        os.environ[k]=v
    return backup

def restore_env(backup:Dict[str,str]):
    for k,v in backup.items():
        if v is None:
            os.environ.pop(k,None)
        else:
            os.environ[k]=v

def simulate(df, router):
    fee=0.0005
    slip=0.0002
    balance=1.0
    trades=0
    wins=0
    pos=None
    entry=0.0
    lev=0.0
    eq=[1.0]
    features=df.drop(columns=["timestamp"]).to_dict(orient="records")
    for i in range(len(features)-1):
        stats={k:float(v or 0.0) for k,v in features[i].items()}
        action_raw,lev_raw,_=router.decide(stats,{"type":pos,"entry_price":entry,"unrealized":0.0,"mdd":0.0,"hold_norm":0.0})
        action=int(np.clip(round(action_raw),0,2))
        kelly=float(np.clip(lev_raw,0,1))
        price=float(df.iloc[i+1]["open"])
        if pos is None:
            if action==1 and kelly>0:
                pos="LONG";entry=price*(1+slip);lev=kelly;balance-=balance*fee*lev
            elif action==2 and kelly>0:
                pos="SHORT";entry=price*(1-slip);lev=kelly;balance-=balance*fee*lev
        else:
            close=True if action in (0,) or (pos=="LONG" and action==2) or (pos=="SHORT" and action==1) else False
            if close:
                trades+=1
                if pos=="LONG": pnl=((price*(1-slip)-entry)/entry)*lev
                else: pnl=((entry-price*(1+slip))/entry)*lev
                balance*=1+pnl
                balance-=balance*fee*lev
                if pnl>0:wins+=1
                pos=None;entry=0;lev=0
            else:
                pass
        eq.append(balance)
    arr=np.array(eq)
    run_max=np.maximum.accumulate(arr)
    dd=arr/np.maximum(run_max,1e-12)-1.0
    rets=np.diff(arr)/np.maximum(arr[:-1],1e-12)
    sharpe=float(np.mean(rets)/np.std(rets)*np.sqrt(365*24*12)) if len(rets)>2 and np.std(rets)>0 else 0.0
    return {
        "pnl_pct":(arr[-1]-1.0)*100,
        "trades":trades,
        "wr":(wins/trades*100 if trades else 0.0),
        "mdd_pct":float(np.min(dd))*100,
        "sharpe":sharpe
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--rl-csv",default="data/splits/year_oos/rl_base_2025.csv")
    parser.add_argument("--ckpt",default="data/ensemble/ckpt/best_dsac_agents.pth")
    parser.add_argument("--limit",type=int,default=1500)
    args=parser.parse_args()
    df=pd.read_csv(args.rl_csv)
    df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce")
    split=df[(df["timestamp"]>="2025-01-01")&(df["timestamp"]<="2025-06-30")]
    subset=split.head(args.limit).reset_index(drop=True)
    actor=load_actor(args.ckpt)
    router=DSACRouter(actor,device="cpu")
    results={}
    for label,val in {"primary_on":0,"primary_off":1}.items():
        backup=set_env({"DSAC_SPECIALISTS_ONLY":str(val)})
        metrics=simulate(subset,router)
        results[label]=metrics
        restore_env(backup)
    print(json.dumps(results,indent=2))
if __name__=="__main__":
    main()
