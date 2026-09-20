"""후보 13칸의 fwd300 이 «직전 300초 이동 분위» 안에서도 남는가 (잔차 = fwd300 − 같은 dmid300 분위 평균)."""
import json, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; import sys; sys.path.insert(0, str(ROOT))
src = open(ROOT / "scripts/research_rt5_1s_panel_sr_20260920.py").read()
exec(src.split("# ── 상태 프레임")[0])
J = json.load(open(ROOT / "tmp/rt_probe_20260920/combos.json"))
hr = pd.to_datetime(P.index, unit="s").hour
def tri(s, lo, mid_, hi, by_hour=False):
    r = s.groupby(hr).rank(pct=True) if by_hour else s.rank(pct=True)
    o = pd.Series(np.where(r <= .2, lo, np.where(r >= .8, hi, mid_)), index=s.index, dtype=object); o[s.isna()] = None; return o
S = pd.DataFrame(index=P.index)
S["가격60"] = tri(P.dmid60, "하락", "횡보", "상승"); S["흡수60"] = tri(P.fp_absorb60, "효율", "보통", "흡수"); S["QI10"] = tri(P.bt_qi10, "QI매도", "QI중립", "QI매수")
S["depth10"] = tri(P.dd_imb10, "매도벽", "벽중립", "매수벽"); S["depth50"] = tri(P.dd_imb50, "깊은매도벽", "깊은중립", "깊은매수벽"); S["OFI10"] = tri(P.dd_ofi10, "OFI매도", "OFI중립", "OFI매수")
S["활동60"] = tri(P.tr_vol60, "조용", "보통", "활발", by_hour=True)
lp = P.lq_prev_net
S["청산(직전분)"] = pd.Series(np.select([P.lq_prev_tot.isna(), P.lq_prev_tot == 0, lp > 0, lp < 0, lp == 0], [None, "청산없음", "롱청산우세", "숏청산우세", "청산균형"], None), index=P.index, dtype=object)
dec = pd.qcut(P.dmid300.rank(method="first"), 20, labels=False)
resid = P.fwd300 - P.fwd300.groupby(dec).transform("mean")
dec2 = pd.qcut(P.range_pos.rank(method="first"), 10, labels=False) if "range_pos" in P else None
hi24 = P.bt_mid.ffill(limit=5).rolling(86400, min_periods=3600).max(); lo24 = P.bt_mid.ffill(limit=5).rolling(86400, min_periods=3600).min()
rp = ((P.bt_mid.ffill(limit=5) - lo24) / (hi24 - lo24)); dec2 = pd.qcut(rp.rank(method="first"), 10, labels=False)
resid2 = resid - resid.groupby(dec2).transform("mean")
out = []
for c in J["candidates"]:
    m = (S[c["a"]] == c["va"]) & (S[c["b"]] == c["vb"])
    r1 = cond(m, "fwd300"); 
    P["_r"] = resid; r2 = cond(m, "_r"); P["_r2"] = resid2; r3 = cond(m, "_r2")
    out.append(dict(cell=f'{c["va"]} × {c["vb"]}', n=c["n"], raw=f"{r1[0]:+.2f}±{r1[1]:.2f}", resid_dmid300=f"{r2[0]:+.2f}±{r2[1]:.2f}", resid_dmid_range=f"{r3[0]:+.2f}±{r3[1]:.2f}", dmid300_med=round(float(P.dmid300[m].median()), 1)))
    print(out[-1])
json.dump(out, open(ROOT / "tmp/rt_probe_20260920/cells_residual.json", "w"), ensure_ascii=False, indent=1)
