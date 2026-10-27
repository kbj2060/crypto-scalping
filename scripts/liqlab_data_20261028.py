"""liqlab 데이터 빌더(서버 실행) — binance.vision 원천만으로 4코인 5분 패널 + 20코인 일봉. 2026-10-28.

산출 data/liqlab/
  {SYM}_5m.parquet  (ETH·BTC·SOL·XRP, 2024-01-01 ~ 2026-10-23, 인덱스 = 5분봉 시작 UTC)
    o h l c qv tbqv n            — 1분봉 합성(5분 안 1분봉 수 k 도 저장)
    rv  = √Σ r1²(1분 로그수익) · ab = Σ|r1| · rr = Σ r1_t·r1_{t-1}(Roll 공분산 재료)
    im_s · im_s2 = Σ 1분 테이커 불균형(2·tb/q−1) 과 제곱합(굴림 분산 재료)
    d-1..d-5 · d1..d5 = 봉 끝 1초 전까지의 마지막 bookDepth 누적 명목(±1~5%) · d_age = 그 스냅샷 나이(초)
    fr = 봉 끝까지 정산된 마지막 펀딩비(bp)
  D1_{SYM}.parquet  (20코인 일봉 o h l c qv, 일 시작 UTC)
시각 규약: 행 τ 의 모든 값은 봉 τ 종가(τ+5분) 시점에 이미 알려진 것. 자가 점검이 이를 assert.
"""
import concurrent.futures as cf
import io
import pathlib
import sys
import urllib.error
import urllib.request
import zipfile

import numpy as np
import pandas as pd

BASE = "https://data.binance.vision/data/futures/um"
OUT = pathlib.Path("data/liqlab"); OUT.mkdir(parents=True, exist_ok=True)
RAW = OUT / "raw"; RAW.mkdir(exist_ok=True)
SYMS = ["ETHUSDT", "BTCUSDT", "SOLUSDT", "XRPUSDT"]
XS20 = ["ADAUSDT", "APTUSDT", "ARBUSDT", "ATOMUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "DOTUSDT", "ETHUSDT",
        "FILUSDT", "INJUSDT", "LINKUSDT", "LTCUSDT", "NEARUSDT", "OPUSDT", "SOLUSDT", "SUIUSDT", "TRXUSDT", "XRPUSDT"]
START, END = "2024-01-01", "2026-10-23"
MONTHS = pd.period_range(START, "2026-09", freq="M").strftime("%Y-%m")
OCT_DAYS = pd.date_range("2026-10-01", END).strftime("%Y-%m-%d")
DAYS = pd.date_range(START, END).strftime("%Y-%m-%d")


def fetch(path):
    """zip → DataFrame(첫 csv). 캐시. 404 → None."""
    cache = RAW / path.replace("/", "__")
    if cache.exists():
        raw = cache.read_bytes()
    else:
        for _ in range(4):
            try:
                raw = urllib.request.urlopen(f"{BASE}/{path}", timeout=120).read(); break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
            except Exception:
                pass
        else:
            raise RuntimeError(f"다운로드 실패 {path}")
        cache.write_bytes(raw)
    z = zipfile.ZipFile(io.BytesIO(raw))
    first = z.open(z.namelist()[0]).readline().decode()
    hdr = 0 if any(ch.isalpha() for ch in first.split(",")[0]) else None
    return pd.read_csv(z.open(z.namelist()[0]), header=hdr)


KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count",
         "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def klines(sym, tf):
    paths = [f"monthly/klines/{sym}/{tf}/{sym}-{tf}-{m}.zip" for m in MONTHS] + \
            [f"daily/klines/{sym}/{tf}/{sym}-{tf}-{d}.zip" for d in OCT_DAYS]
    with cf.ThreadPoolExecutor(8) as ex:
        parts = [p for p in ex.map(fetch, paths) if p is not None]
    k = pd.concat([p.set_axis(KCOLS, axis=1) for p in parts])
    k.index = pd.to_datetime(k.open_time.astype("int64"), unit="ms")
    k = k[~k.index.duplicated()].sort_index()
    return k.loc[START:END + " 23:59"]


def bookdepth(sym):
    with cf.ThreadPoolExecutor(12) as ex:
        parts = list(ex.map(fetch, [f"daily/bookDepth/{sym}/{sym}-bookDepth-{d}.zip" for d in DAYS]))
    missing = sum(p is None for p in parts)
    b = pd.concat([p for p in parts if p is not None])
    b["timestamp"] = pd.to_datetime(b["timestamp"])
    b = b[b["percentage"].abs().between(.5, 5.5)]
    w = b.pivot_table(index="timestamp", columns="percentage", values="notional", aggfunc="first")
    w.columns = [f"d{int(c)}" for c in w.columns]
    return w.sort_index(), missing


def funding(sym):
    paths = [f"monthly/fundingRate/{sym}/{sym}-fundingRate-{m}.zip" for m in MONTHS] + \
            [f"monthly/fundingRate/{sym}/{sym}-fundingRate-2026-10.zip"]
    f = pd.concat([p for p in map(fetch, paths) if p is not None])
    s = pd.Series(f.last_funding_rate.astype(float).to_numpy() * 1e4, index=pd.to_datetime(f.calc_time.astype("int64"), unit="ms"))
    return s[~s.index.duplicated()].sort_index()


def build(sym):
    k = klines(sym, "1m")
    c = k.close.astype(float); r1 = np.log(c).diff()
    q = k.quote_volume.astype(float); tb = k.taker_buy_quote_volume.astype(float)
    im = (2 * tb / q.replace(0, np.nan) - 1).fillna(0)
    one = pd.DataFrame({"o": k.open.astype(float), "h": k.high.astype(float), "l": k.low.astype(float), "c": c,
                        "qv": q, "tbqv": tb, "n": k["count"].astype(float), "r2": r1 ** 2, "ab": r1.abs(),
                        "rr": r1 * r1.shift(), "im_s": im, "im_s2": im ** 2, "k": 1.0})
    g = one.resample("5min")
    p = pd.DataFrame({"o": g.o.first(), "h": g.h.max(), "l": g.l.min(), "c": g.c.last()})
    for col in ("qv", "tbqv", "n", "r2", "ab", "rr", "im_s", "im_s2", "k"):
        p[col] = getattr(g, col).sum()
    p["rv"] = np.sqrt(p.pop("r2"))
    # 호가 깊이: 봉 τ 끝(τ+5분) 1초 전까지의 마지막 스냅샷
    bd, miss = bookdepth(sym)
    cut = p.index + pd.Timedelta("5min") - pd.Timedelta("1s")
    j = pd.merge_asof(pd.DataFrame({"cut": cut}), bd.reset_index().rename(columns={"timestamp": "snap"}),
                      left_on="cut", right_on="snap", direction="backward")
    for col in bd.columns:
        p[col] = j[col].to_numpy()
    p["d_age"] = (j.cut - j.snap).dt.total_seconds().to_numpy() + 1.0
    fr = funding(sym)
    jf = pd.merge_asof(pd.DataFrame({"cut": p.index + pd.Timedelta("5min")}),
                       pd.DataFrame({"t": fr.index, "fr": fr.to_numpy()}),
                       left_on="cut", right_on="t", direction="backward")
    p["fr"] = jf.fr.to_numpy(); p["fr_t"] = jf.t.to_numpy()
    p = p.dropna(subset=["c"])
    # ── 자가 점검(인과·완전성) ──
    end = p.index + pd.Timedelta("5min")
    ok = p.fr_t.notna()
    assert (p.fr_t[ok] <= end[ok]).all(), "펀딩 미래참조"
    ok = p.d_age.notna()
    assert ((end - pd.to_timedelta(p.d_age, unit="s"))[ok] < end[ok]).all() and (p.d_age[ok] >= 1).all(), "호가 깊이 미래참조"
    assert (p.k <= 5).all() and ((p.h >= p[["o", "c"]].max(1)) & (p.l <= p[["o", "c"]].min(1))).all()
    rep = {"sym": sym, "bars": len(p), "first": str(p.index[0]), "last": str(p.index[-1]),
           "full_k5": float((p.k == 5).mean()), "bd_missing_days": miss, "d_age_med": float(p.d_age.median()),
           "d_age_gt120": float((p.d_age > 120).mean()), "fr_nan": float(p.fr.isna().mean())}
    p.drop(columns="fr_t").to_parquet(OUT / f"{sym}_5m.parquet")
    print(rep, flush=True)
    return rep


def build_d1(sym):
    k = klines(sym, "1d")
    d = pd.DataFrame({"o": k.open, "h": k.high, "l": k.low, "c": k.close, "qv": k.quote_volume}).astype(float)
    d.to_parquet(OUT / f"D1_{sym}.parquet")
    return sym, len(d), str(d.index[0]), str(d.index[-1])


if __name__ == "__main__":
    which = sys.argv[1:] or ["4", "20"]
    if "4" in which:
        reps = [build(s) for s in SYMS]
        pd.DataFrame(reps).to_json(OUT / "build_report.json", orient="records", force_ascii=False, indent=1)
    if "20" in which:
        for r in map(build_d1, XS20):
            print("D1", r, flush=True)
