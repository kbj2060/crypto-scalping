"""실측 청산으로 추정 청산맵 정확도 올리기 (2026-10-08, 사용자 «우리 청산데이터를 가지고 청산지도 정확도를 높이는 작업부터»).

평가 틀 = research_eth_liqmap_venue_leverage_validation_20261008 (매 정시 지도 → 다음 4h 지나간 0.2% 칸 안 스피어만 IC,
대조 = 시간 섞은 지도, 실측 = 바이낸스·OKX·Bybit 청산, 위치 = 바이낸스 마크 1초).

사전등록(결과 보기 전 고정):
  후보 A 현행(1일·6단·거래량) · B 14단 균등 · C 새 OI(ΔOI>0 달러)로 가중 · D 테이커 쪽 나눔(롱 ← 테이커 매수 거래대금, 숏 ← 매도)
       · E C×D(롱 ← ΔOI+ × 매수 몫, 숏 ← ΔOI+ × 매도 몫) · F 반감기 48h · G 14단 레버리지 가중을 TRAIN 실측에 NNLS 적합.
  TRAIN = 기준 시각 2026-09-21~09-30 · TEST = 10-01~10-07(4h 창이 끝나는 데까지).
  선택 = TRAIN 초과 IC(4h, 세 거래소 합) 1등 하나. 판정 = TEST 에서 그 하나 − 현행 짝 차이(일 블록 CI) 하한 > 0.
  🔴OI 시점: metrics 스탬프 t 값 = t+5분 스냅샷(2024-03-04~, 메모 binance_metrics_oi_stamp_switch). 봉 [h,h+1h) 의 ΔOI =
     스탬프 h+50분 − 스탬프 h−10분(= 스냅샷 h+55분 − h−5분) → 기준 시각 i 에 이미 알려진 값만.
실행: python scripts/research_eth_liqmap_calibrate_realized_20261008.py [--selftest]
"""
from __future__ import annotations

import io
import json
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_liqmap_venue_leverage_validation_20261008 as V  # noqa: E402

BASE = "https://data.binance.vision/data/futures/um"
ALL = ("binance", "okx", "bybit")
TRAIN_END = pd.Timestamp("2026-10-01", tz="UTC")
D0, D1 = date(2026, 8, 1), date(2026, 10, 7)


def _zip_csv(url: str) -> pd.DataFrame | None:
    try:
        raw = urllib.request.urlopen(url, timeout=60).read()
    except urllib.error.HTTPError:
        return None
    z = zipfile.ZipFile(io.BytesIO(raw))
    return pd.read_csv(z.open(z.namelist()[0]), header=None)


def klines() -> pd.DataFrame:
    """1시간봉 + 테이커 매수 거래대금(바이낸스 vision, REST 아님)."""
    f = V.WORK / "k1h_taker.parquet"
    if f.exists():
        return pd.read_parquet(f)
    parts = [_zip_csv(f"{BASE}/monthly/klines/ETHUSDT/1h/ETHUSDT-1h-{m}.zip") for m in ("2026-08", "2026-09")]
    d = date(2026, 10, 1)
    while d <= D1:
        parts.append(_zip_csv(f"{BASE}/daily/klines/ETHUSDT/1h/ETHUSDT-1h-{d}.zip")); d += timedelta(days=1)
    k = pd.concat([p for p in parts if p is not None])
    k = k[pd.to_numeric(k[0], errors="coerce").notna()].astype(float)
    k = pd.DataFrame({"t": k[0].astype("int64"), "high": k[2], "low": k[3], "close": k[4], "volume": k[5], "qv": k[7], "tbq": k[10]})
    k = k.drop_duplicates("t").sort_values("t").reset_index(drop=True)
    k["timestamp"] = pd.to_datetime(k.t, unit="ms", utc=True)
    k.to_parquet(f); return k


def oi_delta(k: pd.DataFrame) -> np.ndarray:
    """봉별 ΔOI 달러(위 시점 규약). 결측은 0."""
    f = V.WORK / "metrics_5m.parquet"
    if f.exists():
        m = pd.read_parquet(f)
    else:
        rows, d = [], D0 - timedelta(days=1)
        while d <= D1:
            x = _zip_csv(f"{BASE}/daily/metrics/ETHUSDT/ETHUSDT-metrics-{d}.zip")
            if x is not None:
                x = x[x[0] != "create_time"]
                rows.append(pd.DataFrame({"ts": pd.to_datetime(x[0], utc=True), "oi_usd": x[3].astype(float)}))
            d += timedelta(days=1)
        m = pd.concat(rows).drop_duplicates("ts").sort_values("ts"); m.to_parquet(f)
    s = m.set_index("ts").oi_usd
    end = s.reindex(k.timestamp + pd.Timedelta(minutes=50)).to_numpy()
    beg = s.reindex(k.timestamp - pd.Timedelta(minutes=10)).to_numpy()
    return np.nan_to_num(end - beg)


def fit_tier_weights(k, liq, origins, vol) -> np.ndarray:
    """G: TRAIN 기준 시각마다 지나간 칸의 실측 몫 ≈ Σ α_t × (레버리지 t 지도 몫). NNLS."""
    X, Y = [], []
    for i in origins:
        cp, mp = V.liq_map(k, i, 24, V.T14, "splice", vol, by_tier=True)
        rng, r = V.realized(k, liq, i, 4, cp, ALL)
        for s in ("long", "short"):
            bins = list(rng[s])
            y = np.array([r.get((s, b), 0.0) for b in bins])
            if len(bins) < 4 or y.sum() <= 0:
                continue
            x = np.array([[mp.get((t, s, b), 0.0) for t in range(len(V.T14))] for b in bins])
            tot = x.sum()
            if tot <= 0:
                continue
            X.append(x / tot); Y.append(y / y.sum())
    a, _ = nnls(np.vstack(X), np.concatenate(Y))
    return a / a.sum() if a.sum() > 0 else np.full(len(V.T14), 1 / len(V.T14))


def evaluate(k, liq, origins, spec) -> pd.DataFrame:
    df, _ = V.run(k, liq, origins, 4, ALL, spec["tiers"], "splice", 24, spec["vol"], tier_w=spec.get("tw"), halflife=spec.get("hl", V.LM.RECENCY_HALFLIFE_HOURS))
    df["ex"] = df.ic - df.ic_null
    return df


def selftest() -> None:
    ts = pd.date_range("2026-10-01", periods=6, freq="1h", tz="UTC")
    k = pd.DataFrame({"timestamp": ts})
    m = pd.DataFrame({"ts": pd.date_range("2026-09-30 23:00", periods=12 * 8, freq="5min", tz="UTC")})
    m["oi_usd"] = np.arange(len(m), dtype=float)
    s = m.set_index("ts").oi_usd
    end = s.reindex(k.timestamp + pd.Timedelta(minutes=50)).to_numpy(); beg = s.reindex(k.timestamp - pd.Timedelta(minutes=10)).to_numpy()
    assert np.allclose(end - beg, 12)                                       # 한 시간 = 5분 스탬프 12칸
    assert (k.timestamp + pd.Timedelta(minutes=50) + pd.Timedelta(minutes=5) <= k.timestamp + pd.Timedelta(hours=1)).all()   # 봉 끝 전에 알려짐
    print("selftest OK -- ΔOI 창 1시간 · 스냅샷이 봉 끝 이전")


def main() -> None:
    k = klines()
    liq = V.liquidations()
    doi = np.clip(oi_delta(k), 0, None)
    buy = k.tbq.to_numpy(); sell = (k.qv - k.tbq).to_numpy(); qv = k.qv.to_numpy()
    bshare = buy / np.where(qv > 0, qv, 1)
    t_lo = int(max(liq.ts_ms.min(), pd.Timestamp("2026-09-21", tz="UTC").value // 10**6))
    origins = [i for i in range(720, len(k) - 4) if int(k.t.iloc[i]) >= t_lo]
    tr = [i for i in origins if k.timestamp.iloc[i] < TRAIN_END]; te = [i for i in origins if k.timestamp.iloc[i] >= TRAIN_END]
    print("TRAIN", len(tr), "TEST", len(te), k.timestamp.iloc[te[0]], "~", k.timestamp.iloc[te[-1]], "· ΔOI>0 봉 몫", round(float((doi > 0).mean()), 3))
    vol = k.volume.to_numpy()
    eps = 1e-6 * np.nanmean(qv)
    specs = {
        "A 현행": {"tiers": V.T6, "vol": vol},
        "B 14단": {"tiers": V.T14, "vol": vol},
        "C ΔOI+": {"tiers": V.T6, "vol": doi + eps},
        "D 테이커 쪽": {"tiers": V.T6, "vol": (buy, sell)},
        "E ΔOI+×쪽": {"tiers": V.T6, "vol": (doi * bshare + eps, doi * (1 - bshare) + eps)},
        "F 반감기48h": {"tiers": V.T6, "vol": vol, "hl": 48.0},
    }
    tw = fit_tier_weights(k, liq, tr, vol)
    print("G 학습 레버리지 가중", dict(zip(V.T14, np.round(tw, 3))))
    specs["G 14단 학습가중"] = {"tiers": V.T14, "vol": vol, "tw": tw}
    res, train_ex = {}, {}
    for name, sp in specs.items():
        res[name] = {"tr": evaluate(k, liq, tr, sp), "te": evaluate(k, liq, te, sp)}
        d = res[name]["tr"]; train_ex[name] = d.ex.mean()
        print(f"{name:14s} TRAIN 초과IC {d.ex.mean():+.3f} IC {d.ic.mean():.3f} 상위20% {d.cap.mean():.3f}"
              f" | TEST 초과IC {res[name]['te'].ex.mean():+.3f} IC {res[name]['te'].ic.mean():.3f} 상위20% {res[name]['te'].cap.mean():.3f}", flush=True)
    pick = max((n for n in specs if n != "A 현행"), key=lambda n: train_ex[n])
    a, b = res["A 현행"]["te"], res[pick]["te"]
    m = a.merge(b, on=["i", "day", "side"], suffixes=("_a", "_b"))
    m["d"] = m.ex_b - m.ex_a; m["dic"] = m.ic_b - m.ic_a; m["dc"] = m.cap_b - m.cap_a
    print(f"\n선택(TRAIN 1등, 현행 제외) = {pick} (TRAIN {train_ex[pick]:+.3f} vs 현행 {train_ex['A 현행']:+.3f})")
    for col, lab in (("d", "초과IC 차"), ("dic", "IC 차"), ("dc", "상위20% 몫 차")):
        print(f"  TEST {lab}: {[round(x, 3) for x in V.boot(m, col)]}  (n {len(m)}, 일 {m.day.nunique()})")
    lo = V.boot(m, "d")[1]
    print("판정:", "통과(하한 > 0)" if lo > 0 else "불통과(하한 ≤ 0)")
    pd.DataFrame([{"variant": n, "train_ex": r["tr"].ex.mean(), "train_ic": r["tr"].ic.mean(), "test_ex": r["te"].ex.mean(),
                   "test_ic": r["te"].ic.mean(), "test_cap": r["te"].cap.mean()} for n, r in res.items()]).to_csv(V.WORK / "calibrate_results.csv", index=False)


def blend_memory(k, liq, maps: dict, lam: float, hours: int = 24) -> dict:
    """탐색(사후 추가, 사전등록 밖): 지도 + λ × 직전 hours 시간 실측 청산 밀도(같은 쪽·같은 상대 칸, 쪽마다 합 1로 맞춤)."""
    out = {}
    for i, (cp, mp) in maps.items():
        t0 = int(k.t.iloc[i])
        x = liq[(liq.ts_ms >= t0 - hours * 3600_000) & (liq.ts_ms < t0)]
        off = np.floor((x.mark / cp - 1) / V.EVAL_BIN).astype(int)
        r: dict = {}
        for sd, o, u in zip(x.side, off, x.usd):
            if (sd == "long" and o < 0) or (sd == "short" and o >= 0):
                r[(sd, int(o))] = r.get((sd, int(o)), 0.0) + u
        nm = {}
        for sd in ("long", "short"):
            tm = sum(v for (a, _), v in mp.items() if a == sd) or 1.0
            tr_ = sum(v for (a, _), v in r.items() if a == sd) or 1.0
            for key in set(kk for kk in mp if kk[0] == sd) | set(kk for kk in r if kk[0] == sd):
                nm[key] = mp.get(key, 0.0) / tm + lam * r.get(key, 0.0) / tr_
        out[i] = (cp, nm)
    return out


def memory() -> None:
    k = klines(); liq = V.liquidations(); vol = k.volume.to_numpy()
    t_lo = int(pd.Timestamp("2026-09-22", tz="UTC").value // 10**6)       # 직전 24h 실측이 있어야 하므로 하루 늦게 시작
    origins = [i for i in range(720, len(k) - 4) if int(k.t.iloc[i]) >= t_lo]
    tr = [i for i in origins if k.timestamp.iloc[i] < TRAIN_END]; te = [i for i in origins if k.timestamp.iloc[i] >= TRAIN_END]
    res = {}
    for part, og in (("tr", tr), ("te", te)):
        base = {i: V.liq_map(k, i, 24, V.T6, "splice", vol) for i in og}
        for lam in (0.0, 0.25, 0.5, 1.0, 2.0):
            df, _ = V.run(k, liq, og, 4, ALL, V.T6, "splice", 24, vol, maps=blend_memory(k, liq, base, lam))
            df["ex"] = df.ic - df.ic_null; res[(part, lam)] = df
            print(part, "λ", lam, f"초과IC {df.ex.mean():+.3f} IC {df.ic.mean():.3f} 상위20% {df.cap.mean():.3f}", flush=True)
    lam = max((0.25, 0.5, 1.0, 2.0), key=lambda x: res[("tr", x)].ex.mean())
    m = res[("te", 0.0)].merge(res[("te", lam)], on=["i", "day", "side"], suffixes=("_a", "_b"))
    m["d"] = m.ex_b - m.ex_a; m["dic"] = m.ic_b - m.ic_a; m["dc"] = m.cap_b - m.cap_a
    print("TRAIN 선택 λ", lam)
    for col, lab in (("d", "초과IC 차"), ("dic", "IC 차"), ("dc", "상위20% 몫 차")):
        print(f"  TEST {lab}: {[round(x, 3) for x in V.boot(m, col)]}  (n {len(m)}, 일 {m.day.nunique()})")


def forward(path: str, since: str = "2026-10-08") -> None:
    """전진 판정(사전등록 10-08, 판정일 = verdict_calendar «liqmap_fwd»): live_liqmap_scorer 기록에서 origin ≥ since 만.
    주 비교 = B(14단) − A(현행) 초과 IC 짝 차이, 일 블록 CI 하한 > 0 이면 통과. D·F 는 보고만(판정 아님)."""
    rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    rows = [r for r in rows if r["origin"] >= pd.Timestamp(since, tz="UTC").isoformat()]
    ts = np.array([pd.Timestamp(r["origin"]).value // 3_600_000_000_000 for r in rows])
    rs = np.random.default_rng(0)
    def as_map(r, v):
        return {(s, int(o)): x for s in ("long", "short") for o, x in r["maps"][v][s].items()}
    out = {v: [] for v in ("A", "B", "D", "F")}
    for j, r in enumerate(rows):
        rng = {"long": range(r["long_lo"], 0), "short": range(0, r["short_hi"] + 1)}
        real = {(s, int(o)): u for s in ("long", "short") for o, u in r["real"][s].items()}
        far = np.flatnonzero(np.abs(ts - ts[j]) >= 48)
        if len(far) < 30:
            continue
        pick = rs.choice(far, size=30, replace=False)
        for v in out:
            for s, ic, cap in V.score(as_map(r, v), rng, real):
                null = [x for q in pick for x in V.score(as_map(rows[q], v), rng, real) if x[0] == s]
                out[v].append({"i": j, "day": r["origin"][:10], "side": s, "ic": ic, "cap": cap,
                               "ex": ic - np.mean([x[1] for x in null])})
    df = {v: pd.DataFrame(x) for v, x in out.items()}
    print(f"기록 {len(rows)}줄 · 일 {len({r['origin'][:10] for r in rows})} · 점수 {len(df['A'])}")
    for v, d in df.items():
        print(v, f"초과IC {d.ex.mean():+.3f} IC {d.ic.mean():.3f} 상위20% {d.cap.mean():.3f}")
    for v in ("B", "D", "F"):
        m = df["A"].merge(df[v], on=["i", "day", "side"], suffixes=("_a", "_b")); m["d"] = m.ex_b - m.ex_a
        ci = V.boot(m, "d")
        print(f"{v} − A 초과IC 차 {[round(x, 3) for x in ci]}" + (("  → 판정: " + ("통과" if ci[1] > 0 else "불통과")) if v == "B" else "  (보고만)"))


if __name__ == "__main__":
    if "--forward" in sys.argv:
        forward(sys.argv[sys.argv.index("--forward") + 1])
    else:
        selftest() if "--selftest" in sys.argv else memory() if "--memory" in sys.argv else main()
