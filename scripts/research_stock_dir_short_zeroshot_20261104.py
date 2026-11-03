#!/usr/bin/env python3
"""주식(SNDK·SOXL) 15분·1시간 방향 -- ETH 65열 모형(학습 경계 2026-05-31 동결) 재학습 없이 적용.
사전등록: docs/prereg/stock_dir_short_zeroshot_20261104.md (4b0aa4ef). 로컬 CPU · 바이낸스 REST 없음(Vision CDN 정적 파일만).
    python3 scripts/research_stock_dir_short_zeroshot_20261104.py [--selftest]
"""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import live_eth_dir4h_20261015 as d4  # noqa: E402
import live_eth_dir_short_20261028 as L  # noqa: E402
from build_binance_vision_panel_20260915 import day  # noqa: E402

ROOT = HERE.parent; PANEL = ROOT / "data/live/dirshort_panel"; TMP = ROOT / "tmp/stock_dir_short_20261104"
END = pd.Timestamp("2026-11-01 23:55"); MONTH = pd.Timestamp("2026-06-01"); TEST0 = pd.Timestamp("2026-06-01")
STOCKS = {"SNDK": "2026-04-07", "SOXL": "2026-05-15"}; HS = {"15m": 3, "1h": 12}
B = 2000; RNG = np.random.default_rng(20261104); BAR = pd.Timedelta("5min")


def stock_panel(sym: str, start: str) -> pd.DataFrame:
    f = TMP / f"{sym}USDT.parquet"
    if f.exists():
        return pd.read_parquet(f)
    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(start, END.normalize(), freq="D")]
    with ThreadPoolExecutor(8) as ex:
        parts = [p for p in ex.map(lambda d: day(f"{sym}USDT", d), dates) if p is not None]
    df = pd.concat(parts).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df = df[df.timestamp <= END]; TMP.mkdir(parents=True, exist_ok=True); df.to_parquet(f)
    return df


def labels(close_by_bar: np.ndarray, h: int) -> np.ndarray:
    """close_by_bar[j] = 결정 j 의 결정가(= 직전 봉 종가, 결측 봉은 NaN). 라벨 = h봉 뒤 종가 / 결정가 − 1, 그 사이 봉이 전부 실제 봉일 때만."""
    c = close_by_bar; n = len(c); out = np.full(n, np.nan)
    ok = np.isfinite(c); cs = np.concatenate([[0], np.cumsum(ok)])
    idx = np.arange(n - h)
    full = (cs[idx + h + 1] - cs[idx]) == h + 1
    out[idx] = np.where(full, c[idx + h] / c[idx] - 1, np.nan)
    return out


def target_data(frames_all: dict, art: dict, T: str, panel: pd.DataFrame | None):
    """→ DataFrame(index = 결정 시각 t, 열 p15m·p1h·r3·r12·cov) (온전한 피쳐 ≥ 59/65 인 행만 예측)."""
    fr = {s: f[f.timestamp >= pd.Timestamp("2026-01-15")] for s, f in frames_all.items()}
    if panel is not None:
        fr[T] = panel
    ff = d4.frames_for(fr, T)
    t, X, _, _ = L.dataset(ff, None)
    e = ff["ETH"].drop_duplicates("timestamp").set_index("timestamp").close.astype(float).sort_index()
    g = pd.date_range(e.index[0], e.index[-1], freq="5min"); c = e.reindex(g)       # 결측 봉 = NaN (앞채움 안 함)
    cb = c.reindex(t - BAR).to_numpy()                                              # 행 j 의 결정가 = 봉(t−5분) 종가
    cov = np.isfinite(X).sum(1)
    P = {h: np.full(len(t), np.nan) for h in HS.values()}
    rows = np.where(cov >= 59)[0]
    pr = L.predict(art, X[rows])
    for h in HS.values():
        P[h][rows] = pr[h]
    df = pd.DataFrame({"p15m": P[3], "p1h": P[12], "r3": labels(cb, 3), "r12": labels(cb, 12), "cov": cov}, index=t)
    df["nan_by_feat"] = 0
    nanf = pd.Series(np.isnan(X).mean(0), index=d4.FEATS).sort_values(ascending=False)
    return df, nanf, X


def excess(df: pd.DataFrame, pcol: str, rcol: str, warm_days: int = 14):
    """인과 확장 20/80 분위(당일 0시 이전 자기 예측) 문턱 → 일별 집계 [n, up, dn, nt, ht, nb, hb] 와 풀링 초과."""
    d = df[[pcol, rcol]].dropna(subset=[pcol]); p = d[pcol]; day_ = p.index.normalize()
    first = p.index.min().normalize() + pd.Timedelta(days=warm_days)
    rows = []
    for dd in np.unique(day_[(day_ >= max(first, TEST0.normalize())) & (day_ <= END.normalize())]):
        past = p[p.index < dd].to_numpy(); hi, lo = np.quantile(past, .8), np.quantile(past, .2)
        m = day_ == dd; x = d[m]; ok = x[rcol].notna(); x = x[ok]
        if len(x) == 0:
            continue
        top, bot = x[pcol] >= hi, x[pcol] <= lo
        rows.append([dd, len(x), (x[rcol] > 0).sum(), (x[rcol] < 0).sum(), top.sum(), ((x[rcol] > 0) & top).sum(), bot.sum(), ((x[rcol] < 0) & bot).sum()])
    return np.array([r[1:] for r in rows], float), [str(r[0])[:10] for r in rows]


def stat(a: np.ndarray, idx=None):
    s = a.sum(0) if idx is None else a[idx].sum(0)
    if s[3] == 0 or s[5] == 0:
        return np.nan, np.nan, np.nan
    u, d = s[4] / s[3] - s[1] / s[0], s[6] / s[5] - s[2] / s[0]
    return (u + d) / 2, u, d


def evaluate(df, pcol, rcol, bootstrap=True):
    a, days = excess(df, pcol, rcol)
    if len(a) == 0:
        return {"days": 0}
    pooled, u, d = stat(a); s = a.sum(0)
    r = {"days": len(a), "rows": int(s[0]), "top": f"{int(s[4])}/{int(s[3])}", "bot": f"{int(s[6])}/{int(s[5])}", "base_up": s[1] / s[0], "base_dn": s[2] / s[0],
         "excess_top": u, "excess_bot": d, "pooled": pooled}
    if bootstrap:
        full = np.arange(len(a)); bs = np.array([stat(a, RNG.choice(full, len(full)))[0] for _ in range(B)]); bs = bs[np.isfinite(bs)]
        r.update(lo90=float(np.quantile(bs, .10)), p_le0=float((bs <= 0).mean()))
    return r


def holm(ps, alpha=0.10):
    order = np.argsort(ps); ok = [False] * len(ps)
    for rank, i in enumerate(order):
        if ps[i] <= alpha / (len(ps) - rank):
            ok[i] = True
        else:
            break
    return ok


def stat_dm(a: np.ndarray, idx=None):
    """같은 날 기준선 초과: Σ(적중 − 신호 수 × 그날 방향률) / Σ신호 수, 상·하위 평균. 일별 쏠림(추세일에 신호가 몰림)이 만드는 가짜 초과를 뺀다."""
    x = a if idx is None else a[idx]
    nt, ht, nb, hb = x[:, 3].sum(), x[:, 4].sum(), x[:, 5].sum(), x[:, 6].sum()
    if nt == 0 or nb == 0:
        return np.nan
    et = (x[:, 4] - x[:, 3] * x[:, 1] / x[:, 0]).sum() / nt
    eb = (x[:, 6] - x[:, 5] * x[:, 2] / x[:, 0]).sum() / nb
    return (et + eb) / 2


def evaluate_dm(df, pcol, rcol, bootstrap=True):
    a, days = excess(df, pcol, rcol)
    if len(a) == 0:
        return {"days": 0}
    r = {"days": len(a), "pooled_dm": stat_dm(a)}
    if bootstrap:
        full = np.arange(len(a)); bs = np.array([stat_dm(a, RNG.choice(full, len(full))) for _ in range(B)]); bs = bs[np.isfinite(bs)]
        r.update(lo90_dm=float(np.quantile(bs, .10)), p_le0_dm=float((bs <= 0).mean()))
    return r


def robust():
    """사후 강건성(사전등록 판정은 바꾸지 않는다): 같은 날 기준선 초과 + 같은 척도의 위약(하루 밀기) + 달/월 분할."""
    frames = {s: pd.read_parquet(PANEL / f"{s}USDT.parquet") for s in ["ETH"] + d4.COINS}
    import joblib; art = joblib.load(TMP / "eth65_20260601.joblib"); out = {}
    for T, start in [("SOL", None), *STOCKS.items()]:
        panel = stock_panel(T, start) if start else None
        df, _, _ = target_data(frames, art, T, panel); out[T] = {}
        for nm, h in HS.items():
            pc, rc = f"p{nm}", f"r{h}"; r = evaluate_dm(df, pc, rc)
            sh = df.copy(); sh[pc] = sh[pc].shift(288); r["placebo_1day_shift_dm"] = evaluate_dm(sh, pc, rc, bootstrap=False).get("pooled_dm")
            sh2 = df.copy(); sh2[pc] = sh2[pc].shift(288 * 7); r["placebo_7day_shift_dm"] = evaluate_dm(sh2, pc, rc, bootstrap=False).get("pooled_dm")
            half = {}                                                    # 시험 구간을 앞·뒤 절반으로 -- 한쪽 국면 몫이 아닌지
            mid = pd.Timestamp("2026-08-16")
            for k, d in (("06-01~08-15", df[df.index < mid]), ("08-16~11-01", df[df.index >= mid])):
                half[k] = evaluate_dm(d, pc, rc, bootstrap=False).get("pooled_dm")
            r["halves_dm"] = half; out[T][nm] = r
            print(T, nm, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, ensure_ascii=False, default=float), flush=True)
    (TMP / "results_robust.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))


def selftest():
    c = np.array([1, 2, np.nan, 4, 5, 6, 7, 8.0]); r3 = labels(c, 3)
    assert np.isnan(r3[0]) and np.isnan(r3[1]) and np.isnan(r3[2])      # 결측 봉이 낀 창은 라벨 없음
    assert abs(r3[3] - (7 / 4 - 1)) < 1e-12 and abs(r3[4] - (8 / 5 - 1)) < 1e-12 and np.isnan(r3[5])
    assert holm([.001, .5, .02, .9]) == [True, False, True, False]
    # 같은 날 기준선: 추세일(그날 오름률 90%)에 신호가 몰려 적중 90% 여도 초과 0
    a = np.array([[100, 80, 20, 50, 40, 10, 2], [100, 20, 80, 10, 2, 50, 40]], float)   # [n, up, dn, nt, ht, nb, hb] -- 신호 적중이 그날 방향률과 같다
    assert abs(stat_dm(a)) < 1e-9, stat_dm(a)
    print("selftest ok")


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    frames = {s: pd.read_parquet(PANEL / f"{s}USDT.parquet") for s in ["ETH"] + d4.COINS}
    art_f = TMP / "eth65_20260601.joblib"
    if art_f.exists():
        import joblib; art = joblib.load(art_f)
    else:
        import joblib; art = L.train(frames, None, MONTH, hz=(3, 12), nj=8); joblib.dump(art, art_f)
    print("모형 학습 경계:", art["train"], flush=True)
    res, primaries = {"model": art["train"]}, []
    for T, start in [("SOL", None), *STOCKS.items()]:
        panel = stock_panel(T, start) if start else None
        df, nanf, X = target_data(frames, art, T, panel)
        res[T] = {"rows": int(len(df)), "coverage_ok_share": float(df["p15m"].notna().mean()), "nan_share_top": {k: round(float(v), 3) for k, v in nanf.head(10).items()},
                  "first_pred": str(df["p15m"].first_valid_index()), "last": str(df.index.max())}
        for nm, h in HS.items():
            pc, rc = f"p{nm}", f"r{h}"
            r = evaluate(df, pc, rc); res[T][nm] = r
            # 위약: 예측을 하루(288봉) 밀기
            sh = df.copy(); sh[pc] = sh[pc].shift(288); res[T][nm]["placebo_1day_shift"] = evaluate(sh, pc, rc, bootstrap=False).get("pooled")
            if T in STOCKS:
                primaries.append((T, nm, r.get("p_le0", 1.0)))
        # 한 봉 밀기 점검(사전등록 대조 3): 피쳐 행을 한 봉 과거(+1)·한 봉 미래(−1)로 밀어 같은 모형·같은 지표 -- 누수 지문 = 정렬 값이 과거 값보다 «훨씬» 큼
        for k in (1, -1):
            Xs = np.full_like(X, np.nan)
            if k > 0:
                Xs[k:] = X[:-k]
            else:
                Xs[:k] = X[-k:]
            rows = np.where(np.isfinite(Xs).sum(1) >= 59)[0]; pr = L.predict(art, Xs[rows]); d2 = df.copy()
            for nm, h in HS.items():
                d2[f"p{nm}"] = np.nan; d2.iloc[rows, d2.columns.get_loc(f"p{nm}")] = pr[h]
                res[T][nm][f"shift{k:+d}"] = evaluate(d2, f"p{nm}", f"r{h}", bootstrap=False).get("pooled")
        print(T, json.dumps({k: v for k, v in res[T].items() if k != "nan_share_top"}, ensure_ascii=False, default=float), flush=True)
    ok = holm([p for _, _, p in primaries])
    for (T, nm, p), o in zip(primaries, ok):
        r = res[T][nm]; r["holm_pass"] = bool(o and r.get("lo90", -1) > 0 and r.get("days", 0) >= 30)
    (TMP / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    print("\nPRIMARY:", [(T, nm, round(res[T][nm].get("pooled", np.nan), 4), round(res[T][nm].get("lo90", np.nan), 4), res[T][nm].get("holm_pass")) for T, nm, _ in primaries])


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else robust() if "--robust" in sys.argv else main()
