# B2 거래량 하한(vclip) 재내보내기 점검: 시험 2026-06-01~10-27 QLIKE·띠 포함률 · 거래량 0 분 σ̂ 비(A10 대비)
import sys, json; sys.path.insert(0, "scripts")
import numpy as np, pandas as pd, research_realtime_risk_radar_20261115 as R
SP = sys.argv[1]
PA = json.load(open("dashboard/risk_radar_params_20261115.json")); PO = json.load(open(f"{SP}/params_b2_old.json")); PN = json.load(open("dashboard/risk_radar_params_20261116.json"))
def pred(F, P, c, h):
    X = R.design(F, P["cols"]); p = P["coins"][c]; return p[f"c{h}"] * np.exp(2 * X @ np.array(p[f"beta{h}"]))
for c in R.COINS:
    cl, hi, lo, m0 = R.load(c); qv = R.load_qv(c)
    F = R.features(cl, m0); M, day = R.masks(F["mins"]); te = M["test"]
    FO = R.vol_features(dict(F), qv); FN = R.vol_features(dict(F), qv, PN["coins"][c]["vclip"])
    hmax = np.r_[pd.Series(hi[::-1]).rolling(15, min_periods=15).max().to_numpy()[::-1][1:], np.nan]
    lmin = np.r_[pd.Series(lo[::-1]).rolling(15, min_periods=15).min().to_numpy()[::-1][1:], np.nan]
    out = {}
    for h in (15, 60):
        y = R.target(F["r2"], h); fa, fo, fn = pred(F, PA, c, h), pred(FO, PO, c, h), pred(FN, PN, c, h)
        ok = te & np.isfinite(y) & np.isfinite(fa) & np.isfinite(fo) & np.isfinite(fn)
        q = {k: R.qlike(y[ok] ** 2, f[ok]).mean() for k, f in (("A10", fa), ("B2old", fo), ("B2new", fn))}
        out[h] = f"A10 {q['A10']:.4f} · B2구 {q['B2old']:.4f} ({q['B2old']/q['A10']-1:+.1%}) · B2새 {q['B2new']:.4f} ({q['B2new']/q['A10']-1:+.1%})"
    f15 = pred(FN, PN, c, 15); exc = np.maximum(hmax / cl - 1, 1 - lmin / cl) / np.sqrt(f15); okx = te & np.isfinite(exc)
    p = PN["coins"][c]; cov = ((exc[okx] <= p["q80"]).mean(), (exc[okx] <= p["q95"]).mean())
    zero = np.flatnonzero((qv == 0) & np.isfinite(pred(F, PA, c, 15)))
    fa15, fo15 = pred(F, PA, c, 15), pred(FO, PO, c, 15)
    rz = lambda f: np.nanmedian(np.sqrt(f[zero] / fa15[zero])) if len(zero) else np.nan
    print(c, "15분", out[15], "| 1시간", out[60], f"| 띠 포함 80 {cov[0]:.1%} 95 {cov[1]:.1%} | 거래량0분 {len(zero)}개 σ̂/A10 구 {rz(fo15):.2f} 새 {rz(f15):.2f}")
