#!/usr/bin/env python3
"""SOXL 옵션 가설 (사전등록 docs/prereg/soxl_option_hypotheses_20261107.md).
    python scripts/research_soxl_option_hypotheses_20261107.py --selftest
    python scripts/research_soxl_option_hypotheses_20261107.py run     # 입력 tmp/soxl_option_20261107/{iv_history.csv,soxl_etf_60m.csv,SOXLUSDT.parquet}
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

D = Path("tmp/soxl_option_20261107")
BASKET = ["AMAT", "AMD", "AVGO", "INTC", "KLAC", "LRCX", "MU", "NVDA", "ON", "QCOM", "TXN"]
H, MIN_FIT, BOOT, SEED = 5, 40, 2000, 20261107
EVAL0 = pd.Timestamp("2026-06-01")
LAGS = list(range(21, 62, 5))   # 21·26·…·61 = 9판


def daily_rv(ts, close):
    """봉 수익 제곱 합. |r|>40% 봉은 액면 변경으로 뺀다."""
    s = pd.Series(np.asarray(close, float), index=pd.to_datetime(ts)).sort_index()
    r = np.log(s).diff().dropna()
    bad = int((r.abs() > np.log(1.4)).sum())
    r = r[r.abs() <= np.log(1.4)]
    return (r ** 2).groupby(r.index.normalize()).sum(), s.groupby(s.index.normalize()).last(), bad


def frame(rv, x):
    """rv: 일 RV · x: 결정일 d 에 쓸 값(이미 시각 규약대로 정렬). 표적 = d+1~d+H 합."""
    f = pd.DataFrame({"rv": rv}).sort_index()
    f = f[f.rv > 0]
    l = np.log(f.rv)
    f["r1"], f["r5"], f["r22"] = l, np.log(f.rv.rolling(5).mean()), np.log(f.rv.rolling(22).mean())
    f["y"] = np.log(f.rv[::-1].rolling(H).sum()[::-1].shift(-1))
    f["x"] = x.reindex(f.index)
    return f


def walk(f, cols, eval0=EVAL0):
    """확장 OLS. d 에서 쓰는 적합 행 = 표적이 d 이전에 끝난 행(위치 ≤ i−H−1). 반환: 평가일 오차²."""
    g = f.dropna(subset=["r1", "r5", "r22", "x"]).copy()
    idx = np.arange(len(f))
    pos = pd.Series(idx, index=f.index).reindex(g.index).to_numpy()
    y, X = g.y.to_numpy(), np.column_stack([np.ones(len(g))] + [g[c].to_numpy() for c in cols])
    out = {}
    for i, t in enumerate(g.index):
        if t < eval0 or not np.isfinite(y[i]):
            continue
        tr = (pos <= pos[i] - H - 1) & np.isfinite(y)
        if tr.sum() < MIN_FIT:
            continue
        b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
        out[t] = (y[i] - X[i] @ b) ** 2
    return pd.Series(out)


def weeks(s):
    return [g.to_numpy() for _, g in s.groupby(s.index.to_period("W"))]


def boot_mean(s):
    w, rng = weeks(s), np.random.default_rng(SEED)
    return np.array([np.concatenate([w[j] for j in rng.integers(0, len(w), len(w))]).mean() for _ in range(BOOT)])


def h1(rv, x, lags=LAGS, eval0=EVAL0):
    f = frame(rv, x)
    a = walk(f, ["r1", "r5", "r22"], eval0)
    b = walk(f, ["r1", "r5", "r22", "x"], eval0)
    d = (b - a).dropna()
    if len(d) < 60:
        return {"verdict": "보류(표본)", "n": int(len(d))}
    q = boot_mean(d)
    up = float(np.quantile(q, .90))
    pl = []
    for k in lags:
        fk = frame(rv, x.shift(k))
        ak, bk = walk(fk, ["r1", "r5", "r22"], eval0), walk(fk, ["r1", "r5", "r22", "x"], eval0)
        dk = (bk - ak).dropna()
        pl.append(float(dk.mean()) if len(dk) >= 40 else None)
    beat = sum(1 for v in pl if v is not None and d.mean() < v)
    n_pl = sum(1 for v in pl if v is not None)
    half = len(d) // 2
    return {"n_days": int(len(d)), "n_weeks": len(weeks(d)), "mse_A": float(a.reindex(d.index).mean()), "mse_B": float(b.reindex(d.index).mean()), "dmse": float(d.mean()), "upper90": up,
            "p": float((q >= 0).mean()), "placebo": pl, "placebo_beaten": f"{beat}/{n_pl}", "halves": [float(d.iloc[:half].mean()), float(d.iloc[half:].mean())],
            "verdict": "통과" if up < 0 and beat >= 8 else "불통과"}


def h2(px, z):
    fwd = np.log(px.shift(-H) / px)
    g = pd.DataFrame({"z": z.reindex(px.index), "f": fwd}).dropna()
    ic = lambda d: d.z.rank().corr(d.f.rank())
    wk, rng = [x for _, x in g.groupby(g.index.to_period("W"))], np.random.default_rng(SEED)
    b = np.array([ic(pd.concat([wk[j] for j in rng.integers(0, len(wk), len(wk))])) for _ in range(BOOT)])
    lo, hi = np.quantile(b, [.025, .975])
    up = g.f > 0
    return {"n_days": len(g), "n_weeks": len(wk), "ic": float(ic(g)), "ci95": [float(lo), float(hi)], "base_up": f"{int(up.sum())}/{len(g)}",
            "z_gt1_up": f"{int(up[g.z > 1].sum())}/{int((g.z > 1).sum())}", "z_lt_m1_up": f"{int(up[g.z < -1].sum())}/{int((g.z < -1).sum())}",
            "verdict": "통과" if lo > 0 or hi < 0 else "불통과"}


def basket(iv, syms=BASKET):
    p = iv[iv.act_symbol.isin(syms)].pivot(index="date", columns="act_symbol", values="iv_current").astype(float)
    p = p.where(p > 0)
    return np.log(p).mean(axis=1).where(p.notna().sum(axis=1) >= min(8, len(syms))).dropna()


def to_decision(b, days, lag=1):
    """거래일 달력 days 위로 옮긴 뒤 lag 일 늦춘다(d 결정 = d−1 행)."""
    return b.reindex(days.union(b.index)).ffill(limit=3).reindex(days).shift(lag)


def run():
    iv = pd.read_csv(D / "iv_history.csv", parse_dates=["date"])
    e = pd.read_csv(D / "soxl_etf_60m.csv", parse_dates=["timestamp"])
    e = e[e.timestamp >= "2025-11-01"]
    rv, px, bad = daily_rv(e.timestamp, e.close)
    rv26, px26 = rv[rv.index >= "2025-11-15"], px[px.index >= "2025-11-15"]
    days = rv26.index
    b = basket(iv)
    x = to_decision(b, days)
    rep = {"prereg": "docs/prereg/soxl_option_hypotheses_20261107.md", "etf_end": str(e.timestamp.max()), "split_bars_removed": bad, "basket_days": int(len(b)), "rv_days": int(len(rv26))}
    # 대조 먼저
    fut = np.log(rv26[::-1].rolling(H).sum()[::-1].shift(-1))
    rep["control_positive"] = h1(rv26, fut, lags=[])
    rng = np.random.default_rng(SEED)
    rnd = pd.Series(rng.permutation(x.dropna().to_numpy()), index=x.dropna().index)
    rep["control_negative"] = h1(rv26, rnd, lags=[])
    assert rep["control_positive"].get("upper90", 1) < 0, ("양성 대조 실패", rep["control_positive"])
    if rep["control_negative"].get("dmse", 1) < 0:
        rep["control_negative_warn"] = "무작위 판 ΔMSE 음수 -- 장치 점검 필요"
    rep["H1"] = h1(rv26, x)
    x2 = x[x.index >= "2026-01-01"]
    z = (x2 - x2.rolling(60).mean()) / x2.rolling(60).std()
    rep["H2"] = h2(px26[px26.index >= "2026-01-01"], z)
    # 보조
    aux = {}
    aux["same_day_row"] = h1(rv26, to_decision(b, days, lag=0), lags=[])
    aux["spy_iv"] = h1(rv26, to_decision(basket(iv, ["SPY"]), days), lags=[])
    f = frame(rv26, x)
    g = f.dropna(subset=["r1", "r5", "r22", "x", "y"])
    X = np.column_stack([np.ones(len(g)), g.r1, g.r5, g.r22, g.x])
    beta = np.linalg.lstsq(X, g.y, rcond=None)[0]
    aux["ols_all_beta_x"] = float(beta[4])
    p = pd.read_parquet(D / "SOXLUSDT.parquet")
    prv, _, pbad = daily_rv(p.timestamp, p.close)
    pdays = prv.index
    aux["perp_5m"] = h1(prv, to_decision(b, pdays), lags=[], eval0=pd.Timestamp("2026-08-01")) | {"split_bars_removed": pbad, "note": "무기한 05-15~ · 적합 행 확보로 평가 08-01~"}
    rep["aux"] = aux
    (D / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps(rep, ensure_ascii=False, indent=1))


def selftest():
    rng = np.random.default_rng(3)
    days = pd.bdate_range("2025-11-17", "2026-10-21")
    n = len(days)
    # 변동성 군집 + 미래를 아는 x / 무관한 x
    lv = np.zeros(n)
    for i in range(1, n):
        lv[i] = .9 * lv[i - 1] + rng.normal(0, .3)
    rv = pd.Series(np.exp(lv + rng.normal(0, .3, n)) * 1e-3, index=days)
    fut = np.log(rv[::-1].rolling(H).sum()[::-1].shift(-1))
    good = h1(rv, fut, lags=[])
    assert good["upper90"] < 0 and good["dmse"] < 0, good
    noise = pd.Series(rng.normal(size=n), index=days)
    bad = h1(rv, noise, lags=LAGS)
    assert bad["verdict"] == "불통과", bad
    # 표적이 끝나지 않은 행은 적합에 안 들어간다: 표적을 평가 뒤에서 망가뜨려도 평가 첫날 오차 불변
    f = frame(rv, noise)
    e0 = walk(f, ["r1", "r5", "r22"])
    f2 = f.copy()
    first = e0.index[0]
    loc = f2.index.get_loc(first)
    f2.iloc[loc - H:, f2.columns.get_loc("y")] = f2.iloc[loc - H:, f2.columns.get_loc("y")].where(f2.index[loc - H:] == first, 99.0)
    assert abs(walk(f2, ["r1", "r5", "r22"]).iloc[0] - e0.iloc[0]) < 1e-12
    # d 결정은 d−1 행
    b = pd.Series([1., 2., 3.], index=pd.bdate_range("2026-01-05", periods=3))
    assert to_decision(b, b.index).iloc[2] == 2.0
    # 표적은 d+1 부터: 오늘 RV 만 키우면 어제 표적은 변하고 오늘 표적은 그대로
    r2 = rv.copy(); r2.iloc[100] *= 100
    fa, fb = frame(rv, noise), frame(r2, noise)
    assert fa.y.iloc[100] == fb.y.iloc[100] and fa.y.iloc[99] != fb.y.iloc[99]
    # 액면 변경 봉 제거
    t = pd.date_range("2026-01-05 14:30", periods=4, freq="h")
    assert daily_rv(t, [10, 10.1, 2.0, 2.02])[2] == 1
    # H2: 부호가 뚜렷하면 통과, 무관하면 대개 불통과
    px = pd.Series(np.exp(np.cumsum(rng.normal(0, .02, n))), index=days)
    z = np.log(px.shift(-H) / px)
    assert h2(px, z)["verdict"] == "통과"
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "run" in sys.argv:
        run()
    else:
        sys.exit(__doc__)
