"""실시간 위험 레이더 -- 사전등록 docs/prereg/realtime_risk_radar_20261115.md (33cf9d64).

  python scripts/research_realtime_risk_radar_20261115.py selftest
  python scripts/research_realtime_risk_radar_20261115.py run
결정 τ = 1분봉 t 마감. 피쳐 = 분 ≤ t. 표적 = 분 t+1..t+H.
"""
import glob, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VIS = Path("/home/kbj20/crypto-scalping/data/binance_vision")
OUT = ROOT / "tmp/realtime_risk_radar_20261115"
COINS = ["ETH", "BTC", "SOL", "XRP"]
SPLIT = {"train": ("2024-01-08", "2026-01-01"), "val": ("2026-01-01", "2026-06-01"), "test": ("2026-06-01", "2026-10-28")}
WIN = {"lrv1": 1, "lrv5": 5, "lrv15": 15, "lrv60": 60, "lrv240": 240, "lrv1d": 1440, "lrv7d": 10080}
SEED, B = 20261115, 2000


def load(c):
    fs = sorted(glob.glob(str(VIS / f"klines1m/{c}USDT-1m-2023-12.parquet"))) + sorted(glob.glob(str(VIS / f"klines1m/{c}USDT-1m-202[4-6]-*.parquet"))) \
        + sorted(glob.glob(str(VIS / f"klines1m_daily/{c}USDT-1m-2026-*.parquet")))
    k = pd.concat([pd.read_parquet(f) for f in fs]).drop_duplicates("t").sort_values("t")
    m0, m1 = int(k.t.iloc[0]) // 60000, int(k.t.iloc[-1]) // 60000
    g = k.assign(m=k.t // 60000).set_index("m").reindex(range(m0, m1 + 1))         # 빈 분 = NaN
    return g.c.to_numpy(), g.h.to_numpy(), g.l.to_numpy(), m0


def wsum(x, w):
    """x[t-w+1..t] 합과 그 안의 NaN 개수(자기 포함, 과거만)."""
    v = np.nan_to_num(x); n = np.isnan(x).astype(float)
    cs, cn = np.r_[0, np.cumsum(v)], np.r_[0, np.cumsum(n)]
    s = np.full(len(x), np.nan); k = np.full(len(x), np.nan)
    s[w - 1:] = cs[w:] - cs[:-w]; k[w - 1:] = cn[w:] - cn[:-w]
    return s, k


def fwd(x, H):
    """x[t+1..t+H] 합과 NaN 개수(미래)."""
    v = np.nan_to_num(x); n = np.isnan(x).astype(float)
    cs, cn = np.r_[0, np.cumsum(v)], np.r_[0, np.cumsum(n)]
    s = np.full(len(x), np.nan); k = np.full(len(x), np.nan)
    s[:len(x) - H] = cs[H + 1:] - cs[1:len(x) - H + 1]; k[:len(x) - H] = cn[H + 1:] - cn[1:len(x) - H + 1]
    return s, k


def features(c, m0):
    """분 t 마감 시점 피쳐(과거만) + 시계 더미. 반환 dict of arrays."""
    r = np.diff(np.log(c), prepend=np.nan); r2 = r * r
    F = {}
    for nm, w in WIN.items():
        s, k = wsum(r2, w)
        ok = (k == 0) if w <= 240 else (k <= 0.05 * w)
        F[nm] = np.where(ok, 0.5 * np.log(np.maximum(s * w / np.maximum(w - k, 1), 1e-10 * w)), np.nan)
    ts = pd.to_datetime((np.arange(len(c)) + m0 + 1) * 60, unit="s")              # 표적 창 첫 분(시계로 정해짐)
    F["cell"] = (ts.hour.to_numpy() + 24 * (ts.dayofweek.to_numpy() >= 5)).astype(int)
    F["r"] = r; F["r2"] = r2
    # B0: 직전 완결 168개 1시간봉 로그수익 표준편차
    mins = np.arange(len(c)) + m0
    hc = np.where((mins + 1) % 60 == 0, np.log(c), np.nan)                        # 시간봉 종가(그 시의 59분)
    hidx = np.flatnonzero(~np.isnan(hc) | ((mins + 1) % 60 == 0))
    hr = pd.Series(hc[hidx]).diff()
    sd = hr.rolling(168, min_periods=150).std().to_numpy()
    last = np.searchsorted(hidx, np.arange(len(c)), "right") - 1                  # t 이하의 마지막 시간봉 마감
    F["b0sd"] = np.where(last >= 0, sd[np.maximum(last, 0)], np.nan)
    F["mins"] = mins
    return F


def target(r2, H):
    s, k = fwd(r2, H)
    return np.where(k == 0, np.sqrt(np.maximum(s, (1e-4) ** 2 * H)), np.nan)


def design(F, cols):
    D = np.zeros((len(F["cell"]), 48)); D[np.arange(len(F["cell"])), F["cell"]] = 1
    return np.column_stack([np.ones(len(F["cell"])), *[F[c] for c in cols], D[:, 1:]])


def qlike(rv2, f2):
    x = rv2 / f2; return x - np.log(x) - 1


def boot_days(day, vals, rng, stat=np.mean):
    """일 단위로 다시 뽑기(값 배열 여러 개를 같은 날 인덱스로)."""
    u = np.unique(day); idx = {d: np.flatnonzero(day == d) for d in u}
    out = []
    for _ in range(B):
        pick = np.concatenate([idx[d] for d in rng.choice(u, len(u))])
        out.append(stat(*[v[pick] for v in vals]) if isinstance(vals, tuple) else stat(vals[pick]))
    return np.array(out)


def masks(mins):
    t = pd.to_datetime(mins * 60, unit="s")
    return {k: ((t >= a) & (t < b)) for k, (a, b) in SPLIT.items()}, (mins * 60 // 86400)


def fit_predict(F, y, tr, cols):
    X = design(F, cols); ok = tr & np.isfinite(y) & np.isfinite(X).all(1)
    beta = np.linalg.lstsq(X[ok], np.log(y[ok]), rcond=None)[0]
    yh = X @ beta; cst = np.nanmean((y[ok] ** 2) / np.exp(2 * yh[ok]))
    return cst * np.exp(2 * yh), beta, cst


def scale(var, y, tr):
    ok = tr & np.isfinite(var) & np.isfinite(y) & (var > 0)
    return var * np.nanmean(y[ok] ** 2 / var[ok])


def coin_run(c, rng):
    cl, hi, lo, m0 = load(c)
    F = features(cl, m0); M, day = masks(F["mins"]); tr, te, va = M["train"], M["test"], M["val"]
    cols = list(WIN)
    y15 = target(F["r2"], 15); y60 = target(F["r2"], 60)
    f15, beta, _ = fit_predict(F, y15, tr, cols)
    f15_noq, _, _ = fit_predict(F, y15, tr, [k for k in cols if k not in ("lrv1", "lrv5")])
    b0 = scale(F["b0sd"] ** 2 * 15 / 60, y15, tr)
    b1 = scale(np.exp(2 * F["lrv60"]) * 15 / 60, y15, tr)
    res = {"n_test": int((te & np.isfinite(y15)).sum()), "test_days": int(len(np.unique(day[te & np.isfinite(y15)])))}
    ok = te & np.isfinite(y15) & np.isfinite(f15) & np.isfinite(b0) & np.isfinite(b1) & np.isfinite(f15_noq)
    q = {k: qlike(y15[ok] ** 2, v[ok]) for k, v in (("har", f15), ("b0", b0), ("b1", b1), ("har_noq", f15_noq))}
    dd = day[ok]
    for base in ("b0", "b1", "har_noq"):
        d = q["har"] - q[base]
        dm = pd.Series(d).groupby(dd).mean().to_numpy()
        bs = np.array([rng.choice(dm, len(dm)).mean() for _ in range(B)])
        res[f"dq_{base}"] = dict(mean=float(d.mean()), up90=float(np.quantile(bs, .90)), days_better=float((dm < 0).mean()))
    res["qlike"] = {k: float(v.mean()) for k, v in q.items()}
    # 경보
    sig1 = np.sqrt(np.r_[np.nan, f15[:-1]]) / np.sqrt(15)                          # τ−1 예측의 1분 σ
    f5 = np.r_[np.full(5, np.nan), f15[:-5]]
    rv5 = np.exp(F["lrv5"])
    raw = (np.abs(F["r"]) >= 4 * sig1) | (rv5 >= 3 * np.sqrt(f5) * np.sqrt(5 / 15))
    hod = F["cell"] % 24                                                          # 사전등록: 같은 코인·UTC 시각 학습 중앙값
    med = pd.Series(y15[tr]).groupby(hod[tr]).median().reindex(range(24)).to_numpy()[hod]
    def alerts(raw_, mask):
        """감사 결함 1: 중복 제거는 과거로만(예측이 있는 분), 표적이 있는지는 그 뒤에 거른다."""
        a = np.zeros(len(raw_), bool); last = -10**9
        for i in np.flatnonzero(raw_ & mask & np.isfinite(f15)):
            if i - last > 30:
                a[i] = True; last = i
        return a & np.isfinite(y15)
    for sp, mk in (("test", te), ("val", va)):
        a = alerts(raw, mk); ia = np.flatnonzero(a)
        ratio = y15[ia] / med[ia]; cal = y15[ia] ** 2 / f15[ia]; calb0 = y15[ia] ** 2 / b0[ia]
        da = day[ia]
        bs = boot_days(da, ratio, rng) if len(ia) else np.array([np.nan])
        res[f"alert_{sp}"] = dict(n=int(len(ia)), per_day=float(len(ia) / max(len(np.unique(day[mk])), 1)), ratio_mean=float(ratio.mean()) if len(ia) else None,
                                  ratio_lo90=float(np.quantile(bs, .10)), calib_har=float(cal.mean()) if len(ia) else None, calib_b0=float(np.nanmean(calb0)) if len(ia) else None,
                                  ratio_gt2=float((ratio >= 2).mean()) if len(ia) else None)
    for k in (3, 5):
        rr = (np.abs(F["r"]) >= k * sig1) | (rv5 >= 3 * np.sqrt(f5) * np.sqrt(5 / 15))
        ia = np.flatnonzero(alerts(rr, te)); res[f"alert_test_k{k}"] = dict(n=int(len(ia)), ratio_mean=float((y15[ia] / med[ia]).mean()) if len(ia) else None, calib_har=float((y15[ia] ** 2 / f15[ia]).mean()) if len(ia) else None)
    # 앞 1시간 조용/격함
    f60, _, _ = fit_predict(F, y60, tr, cols)
    b060 = scale(F["b0sd"] ** 2, y60, tr)
    mins = F["mins"]
    s60, k60 = wsum(F["r2"], 60)                                                  # 지나간 60분 RV(완결된 시간 끝마다)
    past = np.where((k60 == 0) & ((mins + 1) % 60 == 0), np.sqrt(s60), np.nan)
    pidx = np.flatnonzero(np.isfinite(past))
    usual_h = pd.Series(past[pidx]).rolling(720, min_periods=600).median().to_numpy()
    last = np.searchsorted(pidx, np.arange(len(cl)), "right") - 1
    usual = np.where(last >= 0, usual_h[np.maximum(last, 0)], np.nan)
    okr = te & np.isfinite(y60) & np.isfinite(usual) & np.isfinite(f60) & np.isfinite(b060)
    def states(fv):
        rt = np.sqrt(fv) / usual; return rt <= 0.7, rt >= 1.5
    real_big = y60 > usual
    out = {}
    for nm, fv in (("har", f60), ("b0", b060)):
        qs, lb = states(fv)
        out[nm] = dict(quiet_acc=float((~real_big[okr & qs]).mean()), quiet_cov=float((okr & qs).sum() / okr.sum()),
                       loud_acc=float(real_big[okr & lb].mean()), loud_cov=float((okr & lb).sum() / okr.sum()))
    qh, lh = states(f60); qb, lbb = states(b060)
    dd = day[okr]; u = np.unique(dd); idx = {d: np.flatnonzero(dd == d) for d in u}
    rb, qh_, lh_, qb_, lb_ = real_big[okr], qh[okr], lh[okr], qb[okr], lbb[okr]
    dqs, dls = [], []
    for _ in range(B):
        p = np.concatenate([idx[d] for d in rng.choice(u, len(u))])
        dqs.append((~rb[p][qh_[p]]).mean() - (~rb[p][qb_[p]]).mean()); dls.append(rb[p][lh_[p]].mean() - rb[p][lb_[p]].mean())
    out["d_quiet_lo90"] = float(np.nanquantile(dqs, .10)); out["d_loud_lo90"] = float(np.nanquantile(dls, .10))
    res["regime60"] = out
    # 서술: 지금 국면(지난 4h 순이동 ≥ 1.5 × 30일 중앙값)
    lc = np.log(cl); net = np.abs(lc - np.r_[np.full(240, np.nan), lc[:-240]])
    hh = np.where((mins + 1) % 60 == 0, net, np.nan); hidx = np.flatnonzero(np.isfinite(hh))
    medn = pd.Series(hh[hidx]).rolling(720, min_periods=600).median().to_numpy()
    last = np.searchsorted(hidx, np.arange(len(cl)), "right") - 1
    trend = net >= 1.5 * np.where(last >= 0, medn[np.maximum(last, 0)], np.nan)
    tt = te & np.isfinite(net)
    res["now_regime"] = dict(trend_share=float(trend[tt].mean()), switches_per_day=float((np.diff(trend[tt].astype(int)) != 0).sum() / res["test_days"]))
    # 폭 띠: 15분 안 최대 |이동| ÷ σ̂ 분위(학습) → 시험 포함률
    hmax = pd.Series(hi[::-1]).rolling(15, min_periods=15).max().to_numpy()[::-1]; lmin = pd.Series(lo[::-1]).rolling(15, min_periods=15).min().to_numpy()[::-1]
    hmax = np.r_[hmax[1:], np.nan]; lmin = np.r_[lmin[1:], np.nan]                 # t+1..t+15
    exc = np.maximum(hmax / cl - 1, 1 - lmin / cl) / np.sqrt(f15)
    okx = np.isfinite(exc)
    q80, q95 = np.quantile(exc[tr & okx], [.8, .95])
    res["band"] = dict(q80_sigma=float(q80), q95_sigma=float(q95), cover80_test=float((exc[te & okx] <= q80).mean()), cover95_test=float((exc[te & okx] <= q95).mean()),
                       sigma15_med_test_pct=float(np.nanmedian(np.sqrt(f15[te])) * 100), b0_med_test_pct=float(np.nanmedian(np.sqrt(b0[te])) * 100))
    res["beta"] = dict(zip(["const", *cols], [float(b) for b in beta[:1 + len(cols)]]))
    return res


def run():
    rng = np.random.default_rng(SEED); R = {}
    for c in COINS:
        R[c] = coin_run(c, rng); print(c, json.dumps({k: R[c][k] for k in ("n_test", "test_days", "qlike", "dq_b0", "dq_b1", "dq_har_noq")}, ensure_ascii=False, default=float))
        print("   경보", json.dumps({k: R[c][k] for k in R[c] if k.startswith("alert")}, ensure_ascii=False, default=float))
        print("   국면60", json.dumps(R[c]["regime60"], ensure_ascii=False), "지금", R[c]["now_regime"], "띠", R[c]["band"])
    P = {}
    p1 = {c: R[c]["dq_b0"]["up90"] < 0 for c in COINS}; P["1_vs_current"] = dict(per_coin=p1, pass_=bool(p1["ETH"] and sum(p1.values()) >= 3))
    P["2_vs_persist"] = dict(eth=bool(R["ETH"]["dq_b1"]["up90"] < 0), per_coin={c: R[c]["dq_b1"]["up90"] < 0 for c in COINS})
    def al(a):                                                                    # 경보 0건이면 불통과
        ok_a = a["n"] > 0 and a["ratio_mean"] >= 1.5 and a["ratio_lo90"] > 1.2; ok_b = a["n"] > 0 and 0.8 <= a["calib_har"] <= 1.25
        return dict(a_ratio=bool(ok_a), b_calib=bool(ok_b), pass_=bool(ok_a and ok_b))
    P["3_alert"] = dict(eth=al(R["ETH"]["alert_test"]), per_coin={c: al(R[c]["alert_test"]) for c in COINS})
    g = R["ETH"]["regime60"]; P["4_regime60"] = dict(acc_ok=bool(g["har"]["quiet_acc"] >= .75 and g["har"]["loud_acc"] >= .75), beat_b0=bool(g["d_quiet_lo90"] > 0 and g["d_loud_lo90"] > 0))
    P["4_regime60"]["pass_"] = P["4_regime60"]["acc_ok"] and P["4_regime60"]["beat_b0"]
    R["verdict"] = P
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "result.json").write_text(json.dumps(R, indent=1, ensure_ascii=False, default=float))
    print("판정", json.dumps(P, ensure_ascii=False, default=float))


def export(path=ROOT / "dashboard/risk_radar_params_20261115.json"):
    """서빙용 동결 계수: 학습 구간(2024-01-08~2025-12) 적합 그대로. 코인별 15분·60분 β·눈금 c·15분 띠 분위."""
    P = dict(source="scripts/research_realtime_risk_radar_20261115.py export", prereg="docs/prereg/realtime_risk_radar_20261115.md",
             cols=list(WIN), cells=48, quiet=0.7, loud=1.5, alert_k=4.0, alert_k5=3.0, alert_dedupe_min=30, coins={})
    for c in COINS:
        cl, hi, lo, m0 = load(c); F = features(cl, m0); M, _ = masks(F["mins"]); tr = M["train"]
        y15, y60 = target(F["r2"], 15), target(F["r2"], 60)
        f15, b15, c15 = fit_predict(F, y15, tr, list(WIN)); _, b60, c60 = fit_predict(F, y60, tr, list(WIN))
        hmax = np.r_[pd.Series(hi[::-1]).rolling(15, min_periods=15).max().to_numpy()[::-1][1:], np.nan]
        lmin = np.r_[pd.Series(lo[::-1]).rolling(15, min_periods=15).min().to_numpy()[::-1][1:], np.nan]
        exc = np.maximum(hmax / cl - 1, 1 - lmin / cl) / np.sqrt(f15); okx = tr & np.isfinite(exc)
        q80, q95 = np.quantile(exc[okx], [.8, .95])
        v0 = F["b0sd"] ** 2 * 15 / 60; ok0 = tr & np.isfinite(v0) & np.isfinite(y15) & (v0 > 0)
        cb0 = float(np.nanmean(y15[ok0] ** 2 / v0[ok0]))                          # 기준선 B0(7일 1시간 σ 고정)의 학습 눈금 -- 연구 scale() 과 같은 값
        P["coins"][c] = dict(beta15=b15.tolist(), c15=float(c15), beta60=b60.tolist(), c60=float(c60), q80=float(q80), q95=float(q95), cb0=cb0)
        print(c, "c15", round(c15, 4), "q80", round(q80, 3), "q95", round(q95, 3))
    path.write_text(json.dumps(P, indent=1))
    print("→", path)


def selftest():
    rng = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, 30000))); m0 = 28_000_000 - 7     # 시간 경계 섞기
    F = features(c, m0)
    c2 = c.copy(); c2[20001:] *= np.exp(rng.normal(0, .05, len(c) - 20001)).cumprod()   # 미래를 쓰레기로
    G = features(c2, m0)
    for k in (*WIN, "b0sd", "cell"):
        a, b = F[k][:20001], G[k][:20001]
        assert np.allclose(a[np.isfinite(a)], b[np.isfinite(a)]) and (np.isfinite(a) == np.isfinite(b)).all(), k   # 피쳐는 과거만
    r = np.diff(np.log(c), prepend=np.nan); t = 15000
    assert np.isclose(target(r * r, 15)[t], np.sqrt((r[t + 1:t + 16] ** 2).sum()))
    assert np.isclose(np.exp(F["lrv60"][t]), np.sqrt((r[t - 59:t + 1] ** 2).sum()))
    mins = np.arange(len(c)) + m0; hc = np.flatnonzero((mins + 1) % 60 == 0); hc_t = hc[hc <= t][-169:]
    assert np.isclose(F["b0sd"][t], np.diff(np.log(c[hc_t])).std(ddof=1))
    assert abs(qlike(np.array([2.0]), np.array([2.0]))[0]) < 1e-12
    x = np.r_[np.ones(5), np.nan, np.ones(5)]; s, k = wsum(x, 3); assert s[4] == 3 and k[6] == 1 and k[8] == 0
    s, k = fwd(x, 3); assert s[0] == 3 and k[3] == 1 and k[5] == 0
    print("selftest OK")


if __name__ == "__main__":
    {"selftest": selftest, "run": run, "export": export}[sys.argv[1] if len(sys.argv) > 1 else "selftest"]()
