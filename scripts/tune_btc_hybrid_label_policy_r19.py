#!/usr/bin/env python3
"""Validation-only soft-fusion tuning for the r18 hybrid labels."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.train_eval_btc_hybrid_label_policy_r18 import (
    DEV_END, TRAIN_DATA, VAL_DATA, VAL_END, VAL_START, auxiliary_labels, feature_sets,
    fit_auxiliary, fit_primary, labels_for, market, read_window, simulate,
)

OUT=ROOT/'tmp/btc_hybrid_label_policy_r19_soft_fusion'

def policy(pdir,pqual,bdir,bqual,struct,base,df,qf,entry,large,barrier_weight,structure_weight):
    states=np.array([-.3,-.15,0,.15,.3],dtype=np.float32); dv=base[df].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32); qv=base[qf].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)
    bd=bdir.predict(dv); bq=np.clip(bqual.predict(qv),0,1); zs=struct.predict(dv); current=0.; out=[]
    for i in range(len(base)):
        state=states[np.argmin(abs(states-current))]
        direction=float(pdir.predict(np.append(dv[i],state)[None,:])[0])+barrier_weight*float(bd[i])+structure_weight*float(zs[i])
        quality=float(np.clip(pqual.predict(np.append(qv[i],state)[None,:])[0]+barrier_weight*(bq[i]-.5),0,1))
        current=float(np.sign(direction)*(.30 if quality>=large else .15 if quality>=entry else 0.)); out.append(current)
    return np.asarray(out)

def main():
    df,qf=feature_sets(); allf=list(dict.fromkeys([*df,*qf])); b24=read_window(TRAIN_DATA,allf,'2024-01-01','2024-12-31 23:59:59+00:00'); b25=read_window(VAL_DATA,allf,'2025-01-01',DEV_END); base=pd.concat([b24,b25],ignore_index=True); teacher=pd.concat([labels_for(b24),labels_for(b25)],ignore_index=True); state=base.merge(teacher,left_on='timestamp',right_on='decision_timestamp',how='inner')
    pdir,pqual=fit_primary(state,df,qf); bdir,bqual,struct=fit_auxiliary(base,auxiliary_labels('2024-01-01',DEV_END),df,qf); val,ret=market(read_window(VAL_DATA,allf,VAL_START,VAL_END)); rows=[]
    for entry,large,bw,sw in [(e,l,b,0.) for e in (.50,.60,.65) for l in (.75,.80) if l>e for b in (0.,.25)]:
        a=policy(pdir,pqual,bdir,bqual,struct,val,df,qf,entry,large,bw,sw); m=simulate(a,ret); rows.append({'entry':e,'large':l,'barrier_weight':bw,'structure_weight':sw,**m,'eligible':bool(m['action_events']>=15 and m['pnl_pct']>0)})
    grid=pd.DataFrame(rows); c=grid[grid.eligible]; selected=None if c.empty else c.sort_values(['action_events','pnl_pct'],ascending=[True,False]).iloc[0].to_dict(); OUT.mkdir(parents=True,exist_ok=True); grid.to_csv(OUT/'validation_grid.csv',index=False); (OUT/'report.json').write_text(json.dumps({'diagnostic_only':True,'selection':'validation only; soft fusion with no hard oracle gate','selected':selected,'oos_opened':False,'promotion_eligible':False},indent=2)+'\n'); print(json.dumps({'selected':selected,'positive':int((grid.pnl_pct>0).sum())},indent=2))
if __name__=='__main__': main()
