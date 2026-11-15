# O4: 이번 met 원천의 스탬프 규약을 시험 구간(2026-06-01~10-27) 일별로 -- |ΔOI(T−5→T)| 와 5분 가격 RV 순위 상관, 창 끝 T vs T+5
import sys; sys.path.insert(0, "scripts")
import numpy as np, pandas as pd, research_radar_oi_20261116 as RO
RR = RO.RR
for c in RR.COINS:
    cl, hi, lo, m0 = RR.load(c); r = np.diff(np.log(cl), prepend=np.nan)
    d = RO.load_oi(c); T = d["T"].values; lo_ = np.log(d.oi.values)
    ok = np.r_[False, np.diff(T) == 300]; dOI = np.abs(np.r_[np.nan, np.diff(lo_)]); mi = T // 60 - m0
    def rv(sh):
        idx = (mi + sh)[:, None] + np.arange(-5, 0)[None, :]; g = (idx.min(1) >= 0) & (idx.max(1) < len(r))
        out = np.full(len(T), np.nan); out[g] = np.nansum(r[np.clip(idx[g], 0, len(r) - 1)] ** 2, 1); return out
    A, B = rv(0), rv(5)
    day = pd.to_datetime(T, unit="s").normalize()
    m = ok & (day >= "2026-06-01") & (day < "2026-10-28") & (A > 0) & (B > 0)
    df = pd.DataFrame({"day": day[m], "o": dOI[m], "a": np.log(A[m]), "b": np.log(B[m])})
    s = df.groupby("day").apply(lambda g: pd.Series({"A": g.o.corr(g.a, method="spearman"), "B": g.o.corr(g.b, method="spearman")}))
    print(c, "일수", len(s), "B>A 일수", int((s.B > s.A).sum()), "A>B 일", [str(x.date()) for x in s.index[s.A >= s.B]][:6], "중앙 A", round(s.A.median(), 3), "B", round(s.B.median(), 3))
