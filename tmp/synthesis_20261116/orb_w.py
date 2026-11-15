# 응용 점검: 뉴욕 개장 범위폭(진입 때 이미 아는 값)으로 사건을 나누면 수수료 뒤 남는 구간이 있나 (새 코인 5종만)
import pandas as pd, numpy as np
R='/home/kbj20/crypto-scalping/data'; E='/home/kbj20/crypto-scalping/.claude/worktrees/fujimoto-126-trading-analysis-724dd5/tmp/orb_r2_ny_open_20261115'
ev=pd.read_parquet(f'{E}/events_new.parquet'); ev=ev[(ev.cell=='NY')&(ev.draw<0)].copy()
ev['date']=pd.to_datetime(ev.tk,unit='ms',utc=True).dt.tz_convert('America/New_York').dt.date
rows=[]
for coin in ev.coin.unique():
    d=pd.concat([pd.read_csv(f'{R}/{coin.lower()}_5m_{p}.csv',usecols=['ts','high','low','close']) for p in ('2022_2023','2024_2026')])
    d.index=pd.DatetimeIndex(pd.to_datetime(d.ts,unit='ms',utc=True)).tz_convert('America/New_York'); d=d.sort_index()
    for tf,end in (('or30','09:55'),('or60','10:25')):
        x=d.between_time('09:30',end); g=x.groupby(x.index.date)
        w=((g.high.max()-g.low.min())/g.close.last()).rename('w').reset_index().rename(columns={'index':'date'})
        w['coin']=coin; w['tf']=tf; rows.append(w)
W=pd.concat(rows); ev=ev.merge(W,on=['coin','tf','date'],how='left'); print('w 결측',ev.w.isna().mean().round(4))
ev['pnl']=np.where(ev.y<0,0.0,(2*ev.y-1)*ev.w)*1e4   # 미결 = 0bp 가정
ev['yr']=pd.to_datetime(ev.tk,unit='ms').dt.year
for tf,x in ev.groupby('tf'):
    x=x.copy(); x['wq']=pd.qcut(x.w,3,labels=['좁음','중간','넓음'])
    print(tf); print(x.groupby('wq',observed=True).agg(n=('pnl','size'),w_bp=('w',lambda s:s.median()*1e4),결판=('y',lambda s:(s>=0).mean()),p=('y',lambda s:s[s>=0].mean()),총bp=('pnl','mean')).assign(테이커뒤=lambda t:t.총bp-10,메이커진입뒤=lambda t:t.총bp-7).round(3).to_string())
    top=x[x.wq=='넓음']; print('넓음 연도별 총bp', top.groupby('yr').pnl.mean().round(1).to_dict())
    # 일 블록 부트스트랩 하한(넓음, 테이커 뒤)
    dd=top.groupby('date').pnl.sum(); nper=len(top)/len(dd); rng=np.random.default_rng(0)
    bs=[(dd.sample(len(dd),replace=True,random_state=int(rng.integers(1e9))).sum()/ (len(top)))-10 for _ in range(1000)]
    print('넓음 테이커뒤 단측90% 하한', np.percentile(bs,10).round(2))
