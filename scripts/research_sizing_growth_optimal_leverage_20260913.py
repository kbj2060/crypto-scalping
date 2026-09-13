"""**수익 우선**일 때 레버리지 몇 배가 최적인가 (2026-09-13, 사용자 결정).

사용자: *"5배는 자금을 불리기에 너무 적어. 청산 위험이 있더라도 수익을 우선시할게.
통계적으로 몇 배가 가장 타당할지."*

## 목적함수가 바뀐다
지금까지는 «파산 <= 5%» 제약 아래 최대 레버리지를 찾았다. 여기서는 제약을 풀고
**파산을 포함한 실제 복리 성장**을 최대화한다 -- 파산하면 그 경로는 0 에서 멈춘다.
그래서 내부 최적점이 생긴다(켈리와 같은 구조지만 배리어가 실측 MAE 다).

## 입력
  · 단위당 수익: 실계좌 68왕복 (μ=18.37bp, σ=49.13bp)
  · 파산 판정: 869일 테이프에서 **사용자 보유시간 분포로** 표집한 MAE 가 1/L 에 닿는가

## 결론
**거래를 많이 할수록 최적 레버리지가 내려간다** -- 파산이 복리로 쌓이기 때문이다.
🔴8배는 «한 달 승부» 값이고 지속 가능한 값이 아니다.
⚠️엣지를 95% 하한으로 낮추면 5배와 8배가 구분되지 않는다. 8배의 우위는 점추정이 맞을 때만.
"""
import json, numpy as np, pandas as pd
# 1) 단위당 수익 + 보유시간 (실계좌 68왕복)
rows=[json.loads(l) for l in open("data/live/account_round_trips.jsonl") if l.strip()]
tr=[x for x in rows if x.get("closed") and x.get("side") in ("LONG","SHORT")]
ret=np.array([x["net_pnl"]/(x["max_qty"]*x["entry_price"]) for x in tr])
hold=np.array([(x["exit_time"]-x["entry_time"])/60000.0 for x in tr])
mu,sd,n=ret.mean(),ret.std(ddof=1),len(ret)
se=sd/np.sqrt(n)
print(f"엣지 μ={mu*1e4:.2f}bp σ={sd*1e4:.2f}bp n={n} · 95% 하한 μ={1e4*(mu-1.645*se):.2f}bp")

# 2) 보유시간별 MAE 분포(테이프 869일) -- 파산 판정용
d=pd.read_parquet("data/research/eth_tape_1m_20260906.parquet",columns=["px_last","px_max","px_min"])
px,hi,lo=d.px_last.to_numpy(float),d.px_max.to_numpy(float),d.px_min.to_numpy(float)
rng=np.random.default_rng(20260913); N=40000
st=rng.integers(0,len(px)-1,N); hh=np.maximum(1,rng.choice(hold,N,replace=True).astype(int))
sd_=rng.integers(0,2,N); mae=np.empty(N)
for k in range(N):
    a=st[k]; b=min(len(px),a+hh[k]); e=px[a]
    mae[k]=(e-lo[a:b].min())/e if sd_[k] else (hi[a:b].max()-e)/e
mae=np.maximum(mae,0)

def sim(L, trades, paths=4000, mu_shift=0.0):
    """파산 포함 복리. 파산하면 그 경로는 0 에서 멈춘다."""
    out=np.empty(paths); ruined=0
    r_adj = ret + mu_shift
    for p in range(paths):
        w=1.0; dead=False
        idx=rng.integers(0,n,trades); m=rng.choice(mae,trades)
        for i in range(trades):
            if m[i] >= 1.0/L: w=0.0; dead=True; break
            w*= (1 + L*r_adj[idx[i]])
            if w<=0: w=0.0; dead=True; break
        out[p]=w; ruined+=dead
    return out, ruined/paths

for T in (68, 200):
    print(f"\n=== {T}왕복 후 계좌 배수 (파산=0) ===")
    print(f"{'L':>5} {'중앙':>9} {'평균':>10} {'하위25%':>9} {'상위25%':>10} {'파산율':>8}")
    best=(None,-1)
    for L in (3,5,8,10,12,15,20,25,30):
        w,pr=sim(L,T)
        med=float(np.median(w))
        if med>best[1]: best=(L,med)
        print(f"{L:>5} {med:>9.3f} {w.mean():>10.3f} {np.quantile(w,.25):>9.3f} "
              f"{np.quantile(w,.75):>10.3f} {100*pr:>7.1f}%")
    print(f"  ⇒ 중앙 최대: {best[0]}배 ({best[1]:.3f}배)")

print("\n\n=== 엣지가 95% 하한(8.57bp)이라면 — 68왕복 ===")
shift=(8.57-18.37)/1e4
print(f"{'L':>5} {'중앙':>9} {'평균':>10} {'파산율':>8}")
best=(None,-1)
for L in (2,3,5,8,10,12):
    w,pr=sim(L,68,mu_shift=shift)
    med=float(np.median(w))
    if med>best[1]: best=(L,med)
    print(f"{L:>5} {med:>9.3f} {w.mean():>10.3f} {100*pr:>7.1f}%")
print(f"  ⇒ 중앙 최대: {best[0]}배")
print("\n=== 엣지가 0이라면(방향 실력 없음) — 68왕복 ===")
shift0=-mu
print(f"{'L':>5} {'중앙':>9} {'파산율':>8}")
for L in (2,3,5,8,12):
    w,pr=sim(L,68,mu_shift=shift0)
    print(f"{L:>5} {float(np.median(w)):>9.3f} {100*pr:>7.1f}%")
