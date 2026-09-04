"""HOLDOUT 구간의 레짐/측면 진단 -- 이미 노출된 구간에 대한 **기술 통계**(재최적화 아님)."""
import importlib.util, sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/llewyn/crypto-scalping")
for p in (ROOT, ROOT/"scripts"):
    sys.path.insert(0, str(p))
def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT/r); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m
_pf = _load("pf_d", "scripts/research_eth_v_rebound_ensemble_portfolio_sim_20260902.py")
_s1, _bt = _pf._s1, _pf._bt
TIER0, FB = _pf.TIER0, _pf.FORWARD_BARS
sim_exit = _pf.sim_exit
CUT = 0.8158
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC"); HS = pd.Timestamp("2026-04-01", tz="UTC")
_s1.VAL_END = pd.Timestamp("2099-01-01", tz="UTC")
sig, feat, eth = _s1.build_sig()
sb = _s1.label_param(sig, True, ambig="drop", anchor="wick", atr_mult=1.5, t_sustain=0.20, full_bars=12)
st = _s1.label_param(sig, False, ambig="drop", anchor="wick", atr_mult=1.5, t_sustain=0.20, full_bars=12)
long = _s1.long_frame_for(sig, feat, sb, st)
kl = eth[["timestamp","open","high","low","close"]].copy(); kl["timestamp"]=kl["timestamp"].dt.tz_localize(None)
pos_of = {t:i for i,t in enumerate(kl["timestamp"].to_numpy())}
o,h,l,c = (kl[x].to_numpy() for x in ("open","high","low","close")); nk=len(kl)
long["pos"] = [pos_of.get(np.datetime64(t.tz_localize(None)), -1) for t in long["timestamp"]]
long = long.loc[(long["pos"]>=0)&(long["pos"]+FB+1<nk)].reset_index(drop=True)
print("=== 월별 시장 드리프트 (ETH 단순보유) ===")
e2 = eth.loc[eth["timestamp"] >= HS]
for k, v in e2.groupby(e2["timestamp"].dt.to_period("M")):
    px = v["close"].to_numpy()
    print(f"  {k}: {(px[-1]/px[0]-1)*100:+7.2f}%")
tr_set = long.loc[long["timestamp"] < TRAIN_END].copy()
ii = tr_set["pos"].to_numpy().astype(int); sg = np.where(tr_set["is_downside"].to_numpy()==1,1.0,-1.0)
at = tr_set["atr"].to_numpy(float); net = np.full(len(tr_set), np.nan)
for s_ in range(0, len(tr_set), 40000):
    e_=min(s_+40000,len(tr_set)); j=ii[s_:e_]
    H=np.stack([h[x+1:x+1+FB] for x in j]); L=np.stack([l[x+1:x+1+FB] for x in j]); C=np.stack([c[x+1:x+1+FB] for x in j])
    pn,_=sim_exit(o[j+1], at[s_:e_], sg[s_:e_], H, L, C, 5.0,1.5,0.1); net[s_:e_]=pn*1e4-10.0
tr_set["y"]=(net>0).astype(float)
from tabpfn import TabPFNClassifier
hd = long.loc[long["timestamp"]>=HS].copy()
P=[]
for sd in _pf.SEEDS:
    rng=np.random.default_rng(sd)
    ctx=tr_set.iloc[np.sort(rng.choice(len(tr_set), size=min(18000,len(tr_set)), replace=False))]
    m=TabPFNClassifier(device="cuda", random_state=sd, ignore_pretraining_limits=True)
    m.fit(ctx[TIER0], ctx["y"].to_numpy())
    P.append(np.concatenate([m.predict_proba(hd[TIER0].iloc[k:k+20000])[:,1] for k in range(0,len(hd),20000)]))
hd["p"]=np.vstack(P).mean(axis=0)
sel = hd.loc[hd["p"]>=CUT]
print("\n=== HOLDOUT 호출 측면 구성 (월별) ===")
for k, v in sel.groupby(sel["timestamp"].dt.to_period("M")):
    nl=int((v["is_downside"]==1).sum())
    print(f"  {k}: n={len(v):>5,}  롱 {nl:>5,} ({nl/len(v)*100:5.1f}%)  숏 {len(v)-nl:>5,} ({(1-nl/len(v))*100:5.1f}%)")
nl=int((sel["is_downside"]==1).sum())
print(f"  전체: n={len(sel):,}  롱 {nl:,} ({nl/len(sel)*100:.1f}%)  숏 {len(sel)-nl:,} ({(1-nl/len(sel))*100:.1f}%)")
