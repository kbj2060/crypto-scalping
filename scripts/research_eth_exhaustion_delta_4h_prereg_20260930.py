#!/usr/bin/env python3
"""ETH «소진형 대량 델타» 봉 뒤 4시간 되돌림 — 사전등록 검정 (설계 동결 dfc3c990, 2026-09-30 20:09 KST).

설계서: docs/experiments/eth_exhaustion_delta_4h_reversal_prereg_20260930.md — 정의·판정은 거기 그대로다. 여기서 바꾸지 않는다.
  사건(봉 t 마감에 아는 것만, 기준선 = 직전 288봉·t 제외):
    E1 |델타| ≥ 20 × 중앙값 · E2 봉폭 ≥ 90분위 · E3 24h 신고가(매수)/신저가(매도) · E4 직전 1h 같은 방향
    대조군 = E2·E3·E4 같고 |델타| ≤ 3 × 중앙값 («E1만 없는 봉»)
    중복 제거: 사건 뒤 48봉 안의 사건은 버린다(대조군도 같은 규칙, 따로).
  라벨: 기준가 = 종가[t+1] · r = −sign(델타)·(종가[t+49]/기준가 − 1)·1e4 · rev = 1[r>0]. 수익(비용)은 판정 제외(사용자 지시).
  판정(CONFIRM 2022-01-01 ~ 2026-09-15 UTC, 날짜 블록 부트스트랩 5,000회):
    P1 평균 r > 0·CI 하한 > 0 · P1b P(rev) > 50%·CI 하한 > 50% · P2 연도 5개 중 4개↑ 평균 r > 0
    P3 사건 − 대조군 > 0·CI 하한 > 0 · P4 매수·매도 공격 둘 다 평균 r > 0
  결과를 쓰기 전 조건: 독립 재구성(numpy 경로)이 사건 집합·r 을 1e-9 로 재현 · --selfcheck 통과.

사용:  python scripts/research_eth_exhaustion_delta_4h_prereg_20260930.py --selfcheck
       python scripts/research_eth_exhaustion_delta_4h_prereg_20260930.py            # CONFIRM 1회
"""
from __future__ import annotations
import argparse, glob, json, sys, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path("/home/kbj20/crypto-scalping")          # 원천 데이터는 메인 체크아웃에 있다(워크트리 공유 안 됨)
K5 = MAIN / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
K5_DEC23 = MAIN / "binance_data/klines/ETHUSDT/ETHUSDT-5m-2023-12.csv"   # api 파일의 2023-12 한 달 누락을 vision 월 파일로
METRICS = MAIN / "binance_data/metrics"
OUT = ROOT / "docs/experiments/eth_exhaustion_delta_4h_reversal_prereg_20260930_result.json"
N, MULT, CTRL_MULT, RQ, H, DEDUP = 288, 20.0, 3.0, 0.90, 48, 48
CONFIRM = ("2022-01-01", "2026-09-15 23:55:00")
BOOT = 5000
RNG = np.random.default_rng(20260930)


def load5() -> pd.DataFrame:
    a = pd.read_csv(K5, usecols=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"], parse_dates=["timestamp"])
    b = pd.read_csv(K5_DEC23)
    b = pd.DataFrame({"timestamp": pd.to_datetime(b.open_time, unit="ms"), "open": b.open, "high": b.high, "low": b.low,
                      "close": b.close, "volume": b.volume, "taker_buy_base": b.taker_buy_volume})
    d = pd.concat([a, b]).drop_duplicates("timestamp", keep="first").sort_values("timestamp")
    d = d[(d.timestamp >= "2021-12-01") & (d.timestamp <= CONFIRM[1])]
    full = pd.date_range(d.timestamp.min(), d.timestamp.max(), freq="5min")
    d = d.set_index("timestamp").reindex(full)              # 빈 봉은 NaN -- 롤링 min_periods 로 그 주변 사건이 빠진다
    d.index.name = "timestamp"
    d["delta"] = 2 * d.taker_buy_base - d.volume
    return d


# ── 경로 A: pandas 롤링 ─────────────────────────────────────────────────────────────
def events_a(d: pd.DataFrame, mult=MULT) -> tuple[np.ndarray, np.ndarray]:
    ad = d.delta.abs()
    med = ad.shift(1).rolling(N, min_periods=N).median()
    rng = (d.high - d.low) / d.close
    rq = rng.shift(1).rolling(N, min_periods=N).quantile(RQ)
    hi24 = d.high.shift(1).rolling(N, min_periods=N).max(); lo24 = d.low.shift(1).rolling(N, min_periods=N).min()
    s = np.sign(d.delta)
    pre = np.sign(d.close.shift(1) - d.close.shift(13))
    e2 = rng >= rq
    e3 = ((s > 0) & (d.high >= hi24)) | ((s < 0) & (d.low <= lo24))
    e4 = (pre == s) & (s != 0)
    ev = (ad >= mult * med) & e2 & e3 & e4
    ct = (ad <= CTRL_MULT * med) & e2 & e3 & e4
    return ev.fillna(False).to_numpy(bool), ct.fillna(False).to_numpy(bool)


# ── 경로 B: numpy 창(독립 재구성) ────────────────────────────────────────────────────
def events_b(d: pd.DataFrame, mult=MULT) -> tuple[np.ndarray, np.ndarray]:
    dl = d.delta.to_numpy(float); ad = np.abs(dl)
    hi, lo, c = d.high.to_numpy(float), d.low.to_numpy(float), d.close.to_numpy(float)
    rng = (hi - lo) / c
    n = len(dl); ev = np.zeros(n, bool); ct = np.zeros(n, bool)
    for s0 in range(N, n, 50000):                        # 창 배열이 크니 조각으로
        s1 = min(n, s0 + 50000)
        idx = np.arange(s0, s1)
        W = lambda x: swv(x[s0 - N:s1 - 1], N)          # 행 i = x[t-N .. t-1], t = s0 + i
        wad, wr, wh, wl = W(ad), W(rng), W(hi), W(lo)
        ok = ~(np.isnan(wad).any(1) | np.isnan(wr).any(1) | np.isnan(wh).any(1) | np.isnan(wl).any(1))
        with np.errstate(invalid="ignore"):
            med = np.median(wad, axis=1); rq = np.quantile(wr, RQ, axis=1)   # pandas rolling.quantile 과 같은 linear 보간
            s = np.sign(dl[idx]); pre = np.sign(c[idx - 1] - c[idx - 13])
            e2 = rng[idx] >= rq
            e3 = ((s > 0) & (hi[idx] >= wh.max(1))) | ((s < 0) & (lo[idx] <= wl.min(1)))
            e4 = (pre == s) & (s != 0)
            base = ok & e2 & e3 & e4 & ~np.isnan(dl[idx])
            ev[idx] = base & (ad[idx] >= mult * med); ct[idx] = base & (ad[idx] <= CTRL_MULT * med)
    return ev, ct


def dedup(mask: np.ndarray, gap=DEDUP) -> np.ndarray:
    out, last = [], -10**9
    for k in np.flatnonzero(mask):
        if k - last > gap:
            out.append(k); last = k
    return np.array(out, int)


def labels(d: pd.DataFrame, ks: np.ndarray, h=H) -> pd.DataFrame:
    c = d.close.to_numpy(float); n = len(c)
    ks = ks[ks + 1 + h < n]
    s = np.sign(d.delta.to_numpy(float)[ks]); base = c[ks + 1]
    r = -s * (c[ks + 1 + h] / base - 1) * 1e4
    t = d.index[ks]
    df = pd.DataFrame({"k": ks, "t": t, "side": np.where(s > 0, "buy", "sell"), "r": r, "day": t.floor("D")})
    return df.dropna(subset=["r"]).reset_index(drop=True)


def boot(df: pd.DataFrame, f, n=BOOT):
    days = df.day.unique(); g = {k: v for k, v in df.groupby("day")}
    vals = []
    for _ in range(n):
        pick = RNG.choice(days, len(days), replace=True)
        vals.append(f(pd.concat([g[p] for p in pick])))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def boot_diff(a: pd.DataFrame, b: pd.DataFrame, n=BOOT):
    days = np.union1d(a.day.unique(), b.day.unique())
    ga = {k: v.r.to_numpy() for k, v in a.groupby("day")}; gb = {k: v.r.to_numpy() for k, v in b.groupby("day")}
    vals = []
    for _ in range(n):
        pick = RNG.choice(days, len(days), replace=True)
        xa = [ga[p] for p in pick if p in ga]; xb = [gb[p] for p in pick if p in gb]
        if xa and xb:
            vals.append(np.concatenate(xa).mean() - np.concatenate(xb).mean())
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def oi_delta(d: pd.DataFrame) -> pd.Series:
    """봉 [t,t+5) 마감 OI − 직전 봉 마감 OI. 스탬프 규약: 2024-03-04 이후 행 t = t+5분 스냅샷, 그 전 행 t = t 스냅샷."""
    parts = []
    for z in sorted(glob.glob(str(METRICS / "ETHUSDT-metrics-*.zip"))):
        with zipfile.ZipFile(z) as f:
            parts.append(pd.read_csv(f.open(f.namelist()[0]), usecols=["create_time", "sum_open_interest"]))
    m = pd.concat(parts); m["ts"] = pd.to_datetime(m.create_time)
    m = m.drop_duplicates("ts").set_index("ts").sum_open_interest.sort_index()
    snap_t = np.where(m.index >= pd.Timestamp("2024-03-04"), m.index + pd.Timedelta("5min"), m.index)   # 행 → 실제 스냅샷 시각
    snap = pd.Series(m.to_numpy(), index=pd.DatetimeIndex(snap_t)); snap = snap[~snap.index.duplicated(keep="last")]
    close_oi = snap.reindex(d.index + pd.Timedelta("5min")).to_numpy()                                   # 봉 t 마감 = t+5분 스냅샷
    s = pd.Series(close_oi, index=d.index)
    return s - s.shift(1)


def run() -> dict:
    d = load5()
    ev_a, ct_a = events_a(d); ev_b, ct_b = events_b(d)
    ka, kb = dedup(ev_a), dedup(ev_b); ca, cb = dedup(ct_a), dedup(ct_b)
    la, lb = labels(d, ka), labels(d, kb)
    recon = bool(np.array_equal(ka, kb) and np.array_equal(ca, cb) and np.allclose(la.r.to_numpy(), lb.r.to_numpy(), atol=1e-9, rtol=0))
    print(f"독립 재구성: 사건 {len(ka)} vs {len(kb)} · 대조 {len(ca)} vs {len(cb)} · 일치 {recon}", flush=True)
    if not recon:
        raise SystemExit("🔴 재구성 불일치 -- 결과를 쓰지 않는다")
    in_c = lambda df: df[(df.t >= CONFIRM[0]) & (df.t <= CONFIRM[1])].reset_index(drop=True)
    E, C = in_c(la), in_c(labels(d, ca))
    res = {"frozen_commit": "dfc3c990", "confirm": CONFIRM, "n_events": len(E), "n_days": int(E.day.nunique()), "n_control": len(C)}
    res["P1_mean_r"] = float(E.r.mean()); res["P1_ci"] = boot(E, lambda x: x.r.mean())
    res["P1b_prev"] = float((E.r > 0).mean()); res["P1b_ci"] = boot(E, lambda x: (x.r > 0).mean())
    yr = E.groupby(E.t.dt.year).r.agg(["mean", "count"]); res["P2_by_year"] = {int(k): [float(v["mean"]), int(v["count"])] for k, v in yr.iterrows()}
    res["control_mean_r"] = float(C.r.mean()); res["P3_diff"] = float(E.r.mean() - C.r.mean()); res["P3_ci"] = boot_diff(E, C)
    res["P4_buy"] = float(E[E.side == "buy"].r.mean()); res["P4_sell"] = float(E[E.side == "sell"].r.mean())
    res["P4_n"] = [int((E.side == "buy").sum()), int((E.side == "sell").sum())]
    res["pass"] = {"P1": res["P1_mean_r"] > 0 and res["P1_ci"][0] > 0,
                   "P1b": res["P1b_prev"] > 0.5 and res["P1b_ci"][0] > 0.5,
                   "P2": sum(v[0] > 0 for v in res["P2_by_year"].values()) >= 4,
                   "P3": res["P3_diff"] > 0 and res["P3_ci"][0] > 0,
                   "P4": res["P4_buy"] > 0 and res["P4_sell"] > 0}
    # ── 보고 전용(판정 아님)
    rep = {"r_minus_4bp": float((E.r - 4).mean())}
    for m in (30.0, 40.0):
        L = in_c(labels(d, dedup(events_a(d, m)[0]))); rep[f"mult{int(m)}"] = [float(L.r.mean()), int(len(L))]
    for h in (24, 96):
        rep[f"h{h * 5 // 60}h"] = float(in_c(labels(d, ka, h)).r.mean())
    try:
        doi = oi_delta(d).to_numpy()
        E2 = E.assign(doi=doi[E.k.to_numpy()])
        rep["oi_down"] = [float(E2[E2.doi < 0].r.mean()), int((E2.doi < 0).sum())]
        rep["oi_up"] = [float(E2[E2.doi >= 0].r.mean()), int((E2.doi >= 0).sum())]
        rep["oi_missing"] = int(E2.doi.isna().sum())
    except Exception as exc:                          # 보고 전용이라 실패해도 판정은 그대로
        rep["oi_error"] = str(exc)
    res["report_only"] = rep
    return res


def _selfcheck() -> None:
    """합성: 평탄 잡음 뒤 한 봉에 폭발 델타 + 신고가 + 선행 상승을 심고, 사건·라벨·중복 제거·미래 불변을 본다."""
    rng = np.random.default_rng(0); n = 2000
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    c = 2000 + np.cumsum(rng.normal(0, 0.5, n))
    d = pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": 100.0,
                      "taker_buy_base": 50 + rng.normal(0, 2, n)}, index=ts)
    k = 1500
    up = np.linspace(c[k - 13], c[k - 13] + 30, 13)
    d.iloc[k - 13:k, d.columns.get_loc("close")] = up
    d.iloc[k - 13:k, d.columns.get_loc("high")] = up + 0.5
    d.iloc[k - 13:k, d.columns.get_loc("low")] = up - 0.5
    top = d.high.iloc[k - N:k].max()
    d.iloc[k, [d.columns.get_loc(x) for x in ("high", "low", "close")]] = [top + 20, top - 5, top + 10]
    d.iloc[k, d.columns.get_loc("taker_buy_base")] = 5000; d.iloc[k, d.columns.get_loc("volume")] = 5100
    d["delta"] = 2 * d.taker_buy_base - d.volume
    ea, _ = events_a(d); eb, _ = events_b(d)
    assert ea[k] and eb[k], "심은 사건을 못 잡음"
    assert np.array_equal(ea, eb), "두 경로 사건 불일치"
    d2 = d.copy(); d2.iloc[k + 1:, d2.columns.get_loc("close")] += 100; d2["delta"] = 2 * d2.taker_buy_base - d2.volume
    assert events_a(d2)[0][k] and events_b(d2)[0][k], "미래 봉이 사건 판정에 새어 들어감"
    d3 = d.copy(); d3.iloc[k + 1 + H, d3.columns.get_loc("close")] = d3.close.iloc[k + 1] * (1 - 0.01)
    L = labels(d3, np.array([k])); assert abs(L.r.iloc[0] - 100.0) < 1e-6, L
    m = np.zeros(300, bool); m[[10, 30, 58, 59, 200]] = True
    assert dedup(m).tolist() == [10, 59, 200], dedup(m)
    print("selfcheck ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selfcheck", action="store_true"); a = ap.parse_args()
    if a.selfcheck:
        _selfcheck(); sys.exit(0)
    r = run()
    OUT.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(r, ensure_ascii=False, indent=1, default=str))
