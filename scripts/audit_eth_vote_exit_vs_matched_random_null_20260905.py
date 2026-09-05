#!/usr/bin/env python3
"""표결식 청산 vs **트리거율 매칭 무작위 청산** — 타이밍 실력이 있는가 (2026-09-05).

`research_eth_vote_based_exit_20260905.py`가 144팔 전부 R에 졌는데, 그것만으로는
"표결이 무작위 조기청산보다 나은가"를 못 가른다(어떤 조기청산이든 R에 지므로).
여기서 각 팔의 **트리거율과 보유봉 분포를 매칭한 무작위 청산**(B=200)을 만들어 백분위를 낸다.
백분위 ≈ 50이면 타이밍 실력 없음 = 표결이 하는 일은 "무작위로 일찍 나가기"와 같다."""
import importlib.util, sys, json, itertools
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path('/home/kbj20/crypto-scalping'); sys.path.insert(0, str(ROOT/'scripts'))
def _load(n,r):
    s=importlib.util.spec_from_file_location(n,ROOT/r); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
VB = _load('vb','scripts/research_eth_vote_based_exit_20260905.py')
C1 = _load('c1n','scripts/research_eth_composite_direction_trend_pullback_20260905.py')
XA = _load('xan','scripts/research_crossasset_fire_continuation_replication_20260905.py')
rng = np.random.default_rng(20260905)
B = C1.build()
pos,sd,split,ts,bidx = B['pos'],B['sd'],B['split'],B['ts'],B['bidx']
cont_bp,cont_ex,atr,entry,cs = B['cont_bp'],B['cont_ex'],B['atr'],B['entry'],B['cont_sign']
o,h,l,c = B['o'],B['h'],B['l'],B['c']; n=len(c); FWD=C1.FWD
kl=pd.read_csv(VB.KL,usecols=['timestamp','open','high','low','close','volume','trades','taker_buy_base'],parse_dates=['timestamp']).drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
btc=pd.read_csv(VB.KL_BTC,usecols=['timestamp','open','high','low','close','volume','trades','taker_buy_base'],parse_dates=['timestamp']).drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
sig=XA.DASH.compute_signals(kl.copy(),btc_df=btc,funding_df=None)
k0=int(np.searchsorted(kl['timestamp'].to_numpy(),np.datetime64(B['bar']['timestamp'].iloc[0])))
seg=sig.iloc[k0:k0+n].reset_index(drop=True)
def votes(kind,S):
    bot=np.zeros(n,np.int8); top=np.zeros(n,np.int8)
    for s_ in XA.SIGNALS:
        for side,acc in (('bottom','bot'),('top','top')):
            col=f'{side}_{s_}'
            if col not in seg.columns: continue
            v=seg[col].fillna(False).to_numpy(bool)
            if kind=='first': v=XA.first_fire_mask(v,XA.GAP)
            if S>1: v=pd.Series(v.astype(np.int8)).rolling(S,min_periods=1).max().to_numpy()>0
            (bot if acc=='bot' else top)[v]+=1
    return bot,top
ix=(bidx+1)[:,None]+np.arange(FWD); O_,H_,L_,C_=o[ix],h[ix],l[ix],c[ix]
unreal=cs[:,None]*(C_-entry[:,None])/entry[:,None]; arm=(C1.CELL[1]*atr/entry)[:,None]
prof={'any':np.ones_like(unreal,bool),'profit':unreal>0,'armed':unreal>=arm}
base={w:C1.pf(C1.cand_of(ts[split==w],pos[split==w]+1,pos[split==w]+1+cont_ex[split==w],cont_bp[split==w])) for w in C1.WINDOWS}
def run(cond):
    r,ex=VB.sim_exit_with_vote(entry,atr,cs,O_,H_,L_,C_,*C1.CELL,vote_ok=cond)
    return r*1e4-C1.COST, ex
ARMS=[('first',6,2,'profit'),('raw',6,3,'profit'),('first',12,2,'profit'),('first',6,1,'profit')]
print(f"{'팔':24s} {'트리거율':>7} " + " ".join(f"{w+' 관측Δ / 귀무평균 / 백분위':>30}" for w in C1.WINDOWS))
for kind,S,th,pc in ARMS:
    bot,top=votes(kind,S)
    pt=(cs<0)  # 지속 숏 -> 야당 = 천장
    opp=np.where(pt[:,None],top[ix],bot[ix]).astype(np.int8)
    sup=np.where(pt[:,None],bot[ix],top[ix]).astype(np.int8)
    cond=((opp-sup)>=th)&prof[pc]
    p,ex=run(cond)
    trig=ex<cont_ex; tr_rate=float(trig.mean())
    obs={}; nul={w:[] for w in C1.WINDOWS}
    for w in C1.WINDOWS:
        m=split==w
        r=C1.pf(C1.cand_of(ts[m],pos[m]+1,pos[m]+1+ex[m],p[m]))
        obs[w]=C1.day_paired(r['pnl'],r['ts'],base[w]['pnl'],base[w]['ts'],B=1)['diff_bp_day']
    hold_pool=(ex[trig]).astype(int)          # 트리거된 거래의 실제 보유 봉수 분포
    for _ in range(200):
        fire=rng.random(len(bidx))<tr_rate
        draw=rng.choice(hold_pool,len(bidx),replace=True)
        ex2=np.where(fire&(draw<cont_ex),draw,cont_ex)
        # 무작위 청산 시점의 종가로 청산(같은 규약: 다음 봉 시가)
        eb=np.minimum(bidx+1+ex2+1,n-1)
        p2=np.where(ex2<cont_ex, cs*(o[eb]-entry)/entry*1e4-C1.COST, cont_bp)
        for w in C1.WINDOWS:
            m=split==w
            r2=C1.pf(C1.cand_of(ts[m],pos[m]+1,pos[m]+1+ex2[m],p2[m]))
            if r2 is not None: nul[w].append(C1.day_paired(r2['pnl'],r2['ts'],base[w]['pnl'],base[w]['ts'],B=1)['diff_bp_day'])
    line=f"{kind}_S{S}_th{th}_{pc:6s} {tr_rate:>7.3f} "
    for w in C1.WINDOWS:
        v=np.asarray(nul[w]); line+=f"{obs[w]:>8.2f} / {v.mean():>8.2f} / {float((v<obs[w]).mean()*100):>5.1f}%     "
    print(line)
