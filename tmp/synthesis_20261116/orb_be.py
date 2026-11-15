# 손익분기 점검: ORB 2라운드 결판 비율 p·결판 몫 × 뉴욕 개장 범위폭(bp) → 사건당 총 기대 bp vs 표준 수수료
import pandas as pd, numpy as np
R='/home/kbj20/crypto-scalping/data'; E='/home/kbj20/crypto-scalping/.claude/worktrees/fujimoto-126-trading-analysis-724dd5/tmp/orb_r2_ny_open_20261115'
def widths(coin):
    d=pd.concat([pd.read_csv(f'{R}/{coin.lower()}_5m_{p}.csv',usecols=['ts','high','low','close']) for p in ('2022_2023','2024_2026')])
    t=pd.to_datetime(d.ts,unit='ms',utc=True).dt.tz_convert('America/New_York'); d=d.assign(t=t).set_index('t').sort_index()
    out={}
    for m in (30,60):
        x=d.between_time('09:30',f'{9+(30+m-5)//60:02d}:{(30+m-5)%60:02d}')
        x=x[x.index.dayofweek<5]; g=x.groupby(x.index.date)
        out[m]=((g.high.max()-g.low.min())/g.close.last()).median()*1e4
    return out
for grp,coins in (('new',['ADA','AVAX','BNB','DOGE','LINK']),('orig',['BTC','ETH','SOL','XRP'])):
    ev=pd.read_parquet(f'{E}/events_{grp}.parquet'); ev=ev[(ev.cell=='NY')&(ev.draw<0)]
    for coin in coins:
        w=widths(coin)
        for tf,m in (('or30',30),('or60',60)):
            x=ev[(ev.coin==coin)&(ev.tf==tf)]; dec=x[x.y>=0]; sh=len(dec)/len(x); p=dec.y.mean()
            g=sh*(2*p-1)*w[m]
            print(grp,coin,tf,f'n={len(x)} 결판={sh:.2f} p={p:.3f} 범위폭중앙={w[m]:.0f}bp 총기대={g:+.1f}bp  테이커왕복10bp뒤={g-10:+.1f}  메이커진입+테이커청산7bp뒤={g-7:+.1f}')
