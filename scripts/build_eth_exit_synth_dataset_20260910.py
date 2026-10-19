#!/usr/bin/env python3
"""종합 청산 모델 **합성 포지션 데이터셋 빌더** (2026-09-10).

사용자 재량 진입은 어느 봉에나 올 수 있다 -> 모집단은 **모든 봉(stride) × 양측**의 합성 포지션.
진입 = 봉 t0 종가, 보유 상한 CAP=24봉(사용자 결정). 체크포인트 t = t0+k (k=1,3,..,23).
라벨  y_exit = 1 if side·(close[t0+CAP] − close[t]) < 0  -- "지금 나가는 것이 상한까지 드는 것보다 낫다".
      비용은 어느 시점에 나가도 한 번이라 라벨에서 상쇄된다. fwd_cap 은 정책 평가용 실측 이동(비율).
경계  피쳐는 봉 t 종가까지(대시보드 피쳐 프레임 행 t = 봉 t 자신), 라벨은 t+1 부터. 진입 봉 t0 의
      피쳐가 아니라 **체크포인트 봉 t** 의 피쳐를 쓴다(라이브도 매 봉 재계산).

입력  data/eth_5m_1year.csv(+Binance 로 최근 연장) · btc 동일 · TOTAL_ETHUSDT_metrics_2024_2026.csv(~08-22)
      · deribit_dvol/ETH_dvol_hourly.csv(+Deribit 로 연장) · 극점/변동성 아티팩트(data/live).
출력  tmp/eth_exit_synth_20260910/{bars.parquet, checkpoints.parquet, manifest.json}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import exit_synth_features_20260910 as EF  # noqa: E402

OUT = ROOT / "tmp/eth_exit_synth_20260910"
CAP, ENTRY_STRIDE, CK = 24, 4, list(range(1, 24, 2))
KLINES = "https://fapi.binance.com/fapi/v1/klines"
DVOL_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
        "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def log(m: str) -> None:
    print(f"[build {time.strftime('%H:%M:%S')}] {m}", flush=True)


def fetch_forward(symbol: str, start_ms: int) -> pd.DataFrame:
    rows = []
    while True:
        r = requests.get(KLINES, params={"symbol": symbol, "interval": "5m", "limit": 1500, "startTime": start_ms}, timeout=20)
        r.raise_for_status(); d = r.json()
        if not d:
            break
        rows += d; start_ms = int(d[-1][0]) + 300_000
        if len(d) < 1500:
            break
        time.sleep(0.2)
    f = pd.DataFrame(rows, columns=COLS)
    return f[f["close_time"] < int(time.time() * 1000)]


def load_klines(name: str, symbol: str) -> pd.DataFrame:
    cache = OUT / f"{name}_5m_ext.parquet"
    base = pd.read_csv(ROOT / f"data/{name}_5m_1year.csv", parse_dates=["timestamp"])
    if cache.exists():
        ext = pd.read_parquet(cache)
    else:
        ext = fetch_forward(symbol, int(base["close_time"].iloc[-1]) + 1)
        ext["timestamp"] = pd.to_datetime(ext["open_time"], unit="ms")
        ext.to_parquet(cache)
    kl = pd.concat([base, ext], ignore_index=True)
    for c in ("open", "high", "low", "close", "volume", "taker_buy_base"):
        kl[c] = kl[c].astype(float)
    kl = kl.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    log(f"{name}: {len(kl)} bars {kl.timestamp.iloc[0]} → {kl.timestamp.iloc[-1]}")
    return kl


def load_dvol() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data/derivatives/deribit_dvol/ETH_dvol_hourly.csv", parse_dates=["timestamp"])[["timestamp", "close"]]
    cache = OUT / "dvol_ext.parquet"
    if cache.exists():
        e = pd.read_parquet(cache)
    else:
        start = int((d["timestamp"].iloc[-1] + pd.Timedelta(hours=1)).timestamp() * 1000)
        rows = []
        while True:
            r = requests.get(DVOL_URL, params={"currency": "ETH", "start_timestamp": start,
                                               "end_timestamp": int(time.time() * 1000), "resolution": "3600"}, timeout=20)
            r.raise_for_status(); res = r.json()["result"]
            rows += res["data"]
            if res.get("continuation") is None or not res["data"]:
                break
            start = int(res["data"][-1][0]) + 3_600_000
        e = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "close"])
        e["timestamp"] = pd.to_datetime(e["ts"], unit="ms"); e = e[["timestamp", "close"]]
        e.to_parquet(cache)
    dv = pd.concat([d, e]).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    dv = dv.rename(columns={"close": "dvol"}).iloc[:-1]      # 형성 중인 시간봉 제외
    log(f"dvol: {len(dv)} h → {dv.timestamp.iloc[-1]}")
    return dv


def load_metrics():
    m = pd.read_csv(ROOT / "data/TOTAL_ETHUSDT_metrics_2024_2026.csv", parse_dates=["create_time"])
    ts = m["create_time"] - pd.Timedelta(minutes=5)               # +5분 보정 라벨 -> API timestamp(버킷 시작)
    met = {"retail": m["count_long_short_ratio"].to_numpy(float), "ttc": m["count_toptrader_long_short_ratio"].to_numpy(float),
           "ttp": m["sum_toptrader_long_short_ratio"].to_numpy(float), "tkv": m["sum_taker_long_short_vol_ratio"].to_numpy(float)}
    log(f"metrics: {len(m)} rows → {m.create_time.iloc[-1]}")
    return met, ts


def build_checkpoints(F: pd.DataFrame, kl: pd.DataFrame) -> pd.DataFrame:
    C = kl["close"].to_numpy(float); H = kl["high"].to_numpy(float); L = kl["low"].to_numpy(float)
    atr = F["atr"].to_numpy(float); n = len(C)
    t0 = np.arange(2200, n - CAP - 1, ENTRY_STRIDE)                 # 2200: 랭크 창(2016) 웜업
    rows = []
    for side in (1.0, -1.0):
        for k in CK:
            t = t0 + k
            e = C[t0]; u = side * (C[t] - e) / e
            # 진입 이후 봉 t0+1..t 의 고가/저가로 MFE/MAE (종가 기준 u 포함)
            fav = np.full(len(t0), -np.inf); adv = np.full(len(t0), np.inf)
            for j in range(1, k + 1):
                fav = np.maximum(fav, side * ((H if side > 0 else L)[t0 + j] - e) / e)
                adv = np.minimum(adv, side * ((L if side > 0 else H)[t0 + j] - e) / e)
            fwd_cap = side * (C[t0 + CAP] - C[t]) / C[t]
            fwd12 = side * (C[np.minimum(t + 12, n - 1)] - C[t]) / C[t]
            d = {"pid": (t0 * 2 + (side < 0)).astype(np.int64), "t": t, "t0": t0, "ts": F["ts"].to_numpy()[t],
                 "y_exit": (fwd_cap < 0).astype(np.int8), "fwd_cap": fwd_cap, "fwd12": fwd12}
            d.update(EF.position_features(np.full(len(t0), side), np.full(len(t0), k), CAP, u, fav, adv, atr[t]))
            rows.append(pd.DataFrame(d))
    ck = pd.concat(rows, ignore_index=True)
    return ck


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    eth = load_klines("eth", "ETHUSDT"); btc = load_klines("btc", "BTCUSDT")
    dv = load_dvol(); met, met_ts = load_metrics()
    ext, costw, vol = EF.load_artifacts()
    log(f"artifacts: extreme={ext is not None} costw={costw is not None} vol={vol is not None}")
    t = time.time()
    F = EF.bar_features(eth, btc, met, met_ts, dv, ext, costw, vol)
    log(f"bar features {F.shape} in {time.time() - t:.0f}s · ext fired bars {int(F.ext_bottom_age.eq(0).sum() + F.ext_top_age.eq(0).sum())}")
    F.to_parquet(OUT / "bars.parquet")
    ck = build_checkpoints(F, eth)
    ck = ck[np.isfinite(ck["pos_u"])].reset_index(drop=True)
    ck.to_parquet(OUT / "checkpoints.parquet")
    log(f"checkpoints {ck.shape} · positions {ck.pid.nunique()} · y_exit mean {ck.y_exit.mean():.3f}")
    (OUT / "manifest.json").write_text(json.dumps({
        "cap": CAP, "entry_stride": ENTRY_STRIDE, "checkpoints": CK, "n_bars": len(F), "n_rows": len(ck),
        "span": [str(F.ts.iloc[0]), str(F.ts.iloc[-1])], "feature_cols": EF.feature_cols(F), "pos_cols": EF.POS_COLS,
        "excluded": ["liquidation_map", "liq_burst", "v_rebound_tabpfn", "chip_metalabel_tabpfn", "regime_gbm3"],
        "boundary": "features at checkpoint bar t (close), label from close[t] to close[t0+CAP]",
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
