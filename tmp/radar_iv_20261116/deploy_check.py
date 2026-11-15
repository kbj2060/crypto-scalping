# 배포 계수(risk_radar_params_20261116.json, IV 1시간 판) 시험 2026-06-01~10-27 점검: 1시간 QLIKE IV판 vs B2 · 앞 1시간 조용/격함 정확도(워커 compute 그대로)
import sys, json; sys.path.insert(0, "scripts")
import numpy as np, research_realtime_risk_radar_20261115 as R, live_risk_radar_20261115 as L
P = json.load(open("dashboard/risk_radar_params_20261116.json"))
for c in ("ETH", "BTC"):
    cl, hi, lo, m0 = R.load(c); qv = R.load_qv(c); p = P["coins"][c]
    _, b = L.compute(cl, hi, lo, m0, p, P, qv); _, a = L.compute(cl, hi, lo, m0, p, P, qv, R.load_iv(c))
    F = R.features(cl, m0); M, _ = R.masks(F["mins"]); te = M["test"]; y = R.target(F["r2"], 60)
    ok = te & np.isfinite(y) & np.isfinite(a["f60"]) & np.isfinite(b["f60"])
    qa, qb = R.qlike(y[ok] ** 2, a["f60"][ok]).mean(), R.qlike(y[ok] ** 2, b["f60"][ok]).mean()
    us = b["usual"]; okr = ok & np.isfinite(us); big = y > us
    acc = {}
    for nm, f in (("IV", a["f60"]), ("B2", b["f60"])):
        rt = np.sqrt(f) / us; qs, lb = okr & (rt <= P["quiet"]), okr & (rt >= P["loud"])
        acc[nm] = f"조용 {(~big[qs]).mean():.1%}({qs.sum() / okr.sum():.0%}) 격함 {big[lb].mean():.1%}({lb.sum() / okr.sum():.0%})"
    print(c, f"1시간 QLIKE IV {qa:.4f} vs B2 {qb:.4f} ({qa / qb - 1:+.2%}) | IV판 {acc['IV']} | B2 {acc['B2']}")
