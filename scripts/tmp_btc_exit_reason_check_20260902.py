"""SL 확대가 '손실 회피'인지 '손실 지연'인지 -- 청산 사유 분포로 판정."""
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/llewyn/crypto-scalping")
for p in (ROOT, ROOT/"scripts"): sys.path.insert(0, str(p))
def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT/r); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m
_pf = _load("pf", "scripts/research_eth_v_rebound_ensemble_portfolio_sim_20260902.py")
_sc = _load("sc", "scripts/research_btc_v_rebound_econ_label_screen_20260902.py")
sim_exit, FB, TIER0 = _pf.sim_exit, _pf.FORWARD_BARS, _sc.TIER0
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC"); COST=10.0
long, meta = _sc.build_long(); df = meta.pop("df")
o,h,l,c = (df[x].to_numpy(float) for x in ("open","high","low","close")); nb=len(df)
long = long.dropna(subset=TIER0)
long = long.loc[long["bar_idx"]+FB+1 < nb]
tr = long.loc[long["timestamp"] < TRAIN_END].reset_index(drop=True)
ii = tr["bar_idx"].to_numpy().astype(int)
sg = np.where(tr["is_downside"].to_numpy()==1, 1.0, -1.0); at = tr["atr"].to_numpy(float)
print(f"{'셀':>18s} {'라벨률':>7s} {'평균':>8s} {'손익비':>7s} {'스톱청산':>8s} {'만기청산':>8s} {'보유중앙':>8s}")
for cell in [(5.0,1.5,0.1),(6.0,1.5,0.1),(8.0,1.5,0.1),(12.0,1.5,0.1)]:
    nets, exs, dones = [], [], []
    for s_ in range(0, len(tr), 40000):
        e_=min(s_+40000,len(tr)); j=ii[s_:e_]
        H=np.stack([h[x+1:x+1+FB] for x in j]); L=np.stack([l[x+1:x+1+FB] for x in j]); C=np.stack([c[x+1:x+1+FB] for x in j])
        pn, ex = sim_exit(o[j+1], at[s_:e_], sg[s_:e_], H, L, C, *cell)
        nets.append(pn*1e4-COST); exs.append(ex); dones.append(ex < FB-1)
    v=np.concatenate(nets); ex=np.concatenate(exs); dn=np.concatenate(dones)
    w=v>0
    print(f"{str(cell):>18s} {w.mean():7.4f} {v.mean():+7.2f}bp "
          f"{v[w].mean()/-v[~w].mean():7.3f} {dn.mean()*100:7.1f}% {(1-dn.mean())*100:7.1f}% {np.median(ex+1):8.0f}봉")
