"""견고성: 일별 부호 일치 · 부분상관(5분 평균회귀 통제) · 스프레드 틱 단위 · 청산 버스트 에피소드 수."""
import sys; sys.argv=[sys.argv[0]]
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
exec(open(ROOT / "scripts/research_rt5_1s_panel_analyze_20260920.py").read().split("# ── 1. 원천별 단독")[0])  # 피쳐만 재사용
import numpy as np, pandas as pd
day = pd.to_datetime(P.index, unit="s").date
P["day"] = day
def ic_by_day(x, y):
    out = {}
    for d, g in P.groupby("day"):
        dd = g[[x, y]].dropna()
        if len(dd) > 3600: out[str(d)] = round(dd[x].corr(dd[y], method="spearman"), 3)
    return out
print("## 일별 IC (부호 일치 확인)")
for x, y in [("tr_imb300","fwd300"),("rt_net300","fwd300"),("wh_net300","fwd300"),("dd_imb50","fwd300"),("dd_wall_asym","fwd300"),
             ("dd_imb25","fwd300"),("lq_prev_net","fwd300"),("bt_qi_last","fwd1"),("dd_imb5","fwd1"),("dd_ofi","fwd1"),("dmid300","fwd300"),("tr_vol60","frng300")]:
    print(f"{x:13}->{y:7}", ic_by_day(x, y))
print("\n## 스프레드: bp 는 가격수준 대리변수 -- 틱 단위로 다시")
P["spread_tick"] = (P.bt_spread_bp * P.bt_mid / 1e4 / 0.01)
print("spread_tick 분포", P.spread_tick.quantile([.5,.9,.95,.99]).round(2).to_dict(), " >1틱 비율", (P.spread_tick>1.5).mean().round(4))
print("IC spread_tick->fwd300", ic("spread_tick","fwd300"), " IC bt_mid(가격수준)->fwd300", ic("bt_mid","fwd300"))
print("\n## 부분상관: dd_imb50/wall_asym -> fwd300, dmid300·tr_imb300·가격수준 통제 (순위 회귀 잔차)")
from numpy.linalg import lstsq
def partial(x, y, ctrls):
    d = P[[x, y] + ctrls].dropna()
    R = d.rank()
    X = np.column_stack([np.ones(len(R))] + [R[c].values for c in ctrls])
    rx = R[x].values - X @ lstsq(X, R[x].values, rcond=None)[0]
    ry = R[y].values - X @ lstsq(X, R[y].values, rcond=None)[0]
    return round(np.corrcoef(rx, ry)[0,1], 3), len(d)
for x in ["dd_imb50","dd_wall_asym","dd_imb25","lq_prev_net","rt_net300","wh_net300"]:
    print(x, "raw", round(P[[x,"fwd300"]].dropna().corr(method="spearman").iloc[0,1],3),
          "| dmid300", partial(x,"fwd300",["dmid300"]), "| dmid300+tr_imb300+bt_mid", partial(x,"fwd300",["dmid300","tr_imb300","bt_mid"]))
print("\n## 5분 평균회귀 자체: dmid300 -> fwd300", ic("dmid300","fwd300"), " dmid60->fwd60", ic("dmid60","fwd60"), " dmid10->fwd10?", ic("dmid10","fwd15"))
print("\n## 청산 버스트 에피소드(분 단위 고유 개수)")
lt = P.lq_prev_tot; thr = lt[lt>0].quantile(0.95); print("p95 임계 $", round(thr))
for nm, m in [("long", P.lq_prev_long>=thr), ("short", P.lq_prev_short>=thr)]:
    mins = (P.index[m]//60).unique(); print(nm, "분 수", len(mins), " 평균 fwd300 by minute:", P.loc[m].groupby(P.index[m]//60).fwd300.mean().describe()[["mean","std","count"]].round(2).to_dict())
print("\n## OI 갱신 지연 확인: tr_net -> oi_d CCF 1~8초")
print({l: round(P.tr_net.corr(P.oi_d.shift(-l)),3) for l in range(0,9)})
print("## OI 절대변화 vs 거래량(60초): 스피어만", ic("tr_vol60","oi_d60"), " |oi_d60|:", P.oi_d60.abs().corr(P.tr_vol60, method="spearman").round(3))
