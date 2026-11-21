#!/usr/bin/env python3
"""rtserve 상황 카드 REST 5분봉 재료(src/ctx_rest.rs) 일치 시험의 입력·정답지 (2026-11-22).

정답 = server.py 의 make_app 안 닫힌 함수를 **옮겨 적은 것**(Read 로 본 줄 그대로 · 들여쓰기만):
  · trend_veto_rows 1613~1646 · kline_candles 1648~1661(vwap 은 market_ctx 를 import 해 그대로)
  · load_chart_klines_frames.produce 의 프레임 정리 3204~3214(ETH) · 3231~3244(BTC)
  · load_market_history_from_evidence_cache 의 행 조립 1691~1714 · load_5m_day.produce 의 거름 2434~2437
REST 본문은 합성(바이낸스 klines 모양 · 글자 숫자 · 형성 봉 · 같은 시각 중복). 네트워크 없음.
🔴서버 판 파이썬으로(pandas 2.2.3 -- rolling·ewm 합 차례):
    source <(bash rtmodel/tools/server_env.sh <venv> --env) && $PY rtserve/tools/gen_ctx_rest_fixture.py
출력: rtserve/tests/fixtures/gen/ctx_rest/ (커밋 안 함)
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rtserve" / "tests" / "fixtures" / "gen" / "ctx_rest"
shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True)
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import dashboard.market_ctx as mctx  # noqa: E402

R = random.Random(20261125)
TREND_VETO_N = 144
TREND_VETO_K = 1.0


def trend_veto_rows(df: pd.DataFrame, tail: int = TREND_VETO_N * 5) -> dict[int, dict[str, float | int]]:   # 1613~1646 옮겨 적음
    w = df.tail(tail)                       # ewm 워밍업 여유(720 봉). 1500 봉 전체는 낭비다
    if len(w) < TREND_VETO_N + 1:
        return {}
    c, h, l = w["close"], w["high"], w["low"]
    sma = c.rolling(TREND_VETO_N).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / TREND_VETO_N, adjust=False).mean()
    dev = ((c - sma) / sma).to_numpy()
    eps = (TREND_VETO_K * atr / c).to_numpy()
    drop = c.shift(TREND_VETO_N - 1).to_numpy()
    out: dict[int, dict[str, float | int]] = {}
    cur = 0
    for i, ts in enumerate(w["timestamp"]):
        if not (np.isfinite(dev[i]) and np.isfinite(eps[i])):
            continue
        if dev[i] > eps[i]:
            cur = 1
        elif dev[i] < -eps[i]:
            cur = -1                        # 밴드 안이면 직전 유지(히스테리시스)
        row: dict[str, float | int] = {"sma": round(float(sma.iloc[i]), 6),    # 6자리: XRP ATR ~0.003
                                       "atr": round(float(atr.iloc[i]), 6), "veto": cur}
        if np.isfinite(drop[i]):
            row["drop"], row["vn"] = round(float(drop[i]), 6), TREND_VETO_N
            row["vk"] = TREND_VETO_K
        out[int(ts.timestamp())] = row
    return out


def kline_candles(rows: list, now_ms: float, nd: int, veto_tail: int = TREND_VETO_N * 5) -> list[dict[str, float | int]]:   # 1648~1661 옮겨 적음
    closed = [r for r in rows if int(r[6]) < now_ms]
    veto = trend_veto_rows(pd.DataFrame({"timestamp": pd.to_datetime([int(r[0]) for r in closed], unit="ms", utc=True),
                                         "high": [float(r[2]) for r in closed], "low": [float(r[3]) for r in closed],
                                         "close": [float(r[4]) for r in closed]}), veto_tail)
    tss = [int(r[0]) // 1000 for r in rows]
    vwap = mctx.vwap_rows(tss, [float(r[2]) for r in rows], [float(r[3]) for r in rows],
                          [float(r[4]) for r in rows], [float(r[5]) for r in rows], nd=nd)
    return [{"time": t, "open": float(r[1]), "high": float(r[2]), "low": float(r[3]), "close": float(r[4]), "volume": float(r[5]),
             **veto.get(t, {}), **vwap.get(t, {}),
             **({"partial": True} if int(r[6]) >= now_ms else {})}
            for t, r in zip(tss, rows)]


def eth_frame(raw, now_ms):   # 3204~3214 옮겨 적음
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ("open", "high", "low", "close", "volume", "taker_buy_base"):
        df[c] = df[c].astype("float64")
    df["close_time"] = df["close_time"].astype("int64")
    df["timestamp"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    if len(df) and int(df.iloc[-1]["close_time"]) >= now_ms:
        df = df.iloc[:-1].reset_index(drop=True)  # drop the still-forming bar
    return df[["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"]]


def btc_frame(braw, now_ms):   # 3231~3244 옮겨 적음
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    bdf = pd.DataFrame(braw, columns=cols)
    for c in ("open", "high", "low", "close"):
        bdf[c] = bdf[c].astype("float64")
    bdf["close_time"] = bdf["close_time"].astype("int64")
    bdf["timestamp"] = pd.to_datetime(bdf["open_time"].astype("int64"), unit="ms", utc=True)
    bdf = bdf.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    if len(bdf) and int(bdf.iloc[-1]["close_time"]) >= now_ms:
        bdf = bdf.iloc[:-1].reset_index(drop=True)
    return bdf[["timestamp", "open", "high", "low", "close"]]


def evidence_rows(src):   # 1691~1714 옮겨 적음(vwap 칸 포함 -- 시험은 상황 카드가 읽는 칸만 본다)
    veto = trend_veto_rows(src)
    vwap: dict[int, dict[str, float]] = {}
    if "volume" in src.columns:
        tail = src.tail(600)
        tss = [int(t.timestamp()) for t in tail["timestamp"]]
        vwap = mctx.vwap_rows(tss, tail["high"].tolist(), tail["low"].tolist(), tail["close"].tolist(), tail["volume"].tolist())
    return [
        {
            "time": int(row["timestamp"].timestamp()),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            **({"volume": float(row["volume"])} if "volume" in src.columns else {}),
            **veto.get(int(row["timestamp"].timestamp()), {}),
            **vwap.get(int(row["timestamp"].timestamp()), {}),
        }
        for _, row in src.tail(200).iterrows()
    ]


def day5_produce(raw, now_s):   # 2434~2437 옮겨 적음
    return [{"time": int(r[0]) // 1000, "high": float(r[2]), "low": float(r[3]), "close": float(r[4]),
             "volume": float(r[5]), "taker": float(r[9])} for r in raw
            if int(r[0]) // 1000 + 300 <= now_s]


def klines(n, end_open_ms, px, vol=0.002, flat=False, dup=False, fmt=lambda x: f"{x:.2f}"):
    rows = []
    for k in range(n):
        o = end_open_ms - (n - 1 - k) * 300_000
        if not flat:
            px *= 1 + R.gauss(0, vol)
        hi, lo = px * (1 + abs(R.gauss(0, vol / 2))), px * (1 - abs(R.gauss(0, vol / 2)))
        v = R.uniform(10, 5000)
        rows.append([o, fmt(px), fmt(hi), fmt(lo), fmt(px), f"{v:.3f}", o + 299_999, "0", 100, f"{v * R.random():.3f}", "0", "0"])
    if dup:
        # 같은 시각 중복은 같은 값으로만: 원본 sort_values 는 불안정 정렬(quicksort)이라 값이 다른 중복은 어느 쪽이 남는지 정해져 있지 않다
        #   (바이낸스는 중복을 안 보낸다 -- Rust 는 «마지막»)
        rows.insert(R.randrange(len(rows) - 5), list(rows[-30]))
    return rows


def pick(r):
    return {k: r.get(k) for k in ("time", "high", "low", "close", "veto")} | {"partial": bool(r.get("partial"))}


cases = []
for i in range(24):
    now_ms = 1_763_700_000_000 + R.choice([R.randrange(0, 300_000), 299_999])   # 299_999 = 형성 봉 close_time 과 같은 ms(>= 경계)
    end_open = now_ms // 300_000 * 300_000
    n = R.choice([1500, 1500, 1000, 800, 721, 720, 300, 145, 144])
    kind = R.choice(["eth", "btc", "sol"])
    raw = klines(n, end_open if R.random() < 0.8 else end_open - 300_000, R.choice([3000.0, 95000.0, 2.15]),
                 vol=R.choice([0.0005, 0.002, 0.01]), flat=R.random() < 0.08, dup=R.random() < 0.3,
                 fmt=R.choice([lambda x: f"{x:.2f}", lambda x: f"{x:.4f}", repr]))
    if kind == "sol":
        out = [pick(r) for r in kline_candles(raw, now_ms, 6)[-200:]]
    else:
        src = eth_frame(raw, now_ms) if kind == "eth" else btc_frame(raw, now_ms)
        out = [pick(r) for r in evidence_rows(src)]
    now_s = now_ms / 1000 + R.choice([0.0, 0.5, 5.3])
    d5 = day5_produce(raw[-300:], now_s)
    cases.append({"kind": kind, "now_ms": now_ms, "now_s": now_s, "raw": raw, "candles": out, "day5": d5})
# rolling·ewm 직접(음수·상수 구간·NaN 없음)
series = []
for _ in range(60):
    n = R.choice([10, 150, 300, 720])
    base = R.choice([0.0, 1.0, -1.0, 3000.0])
    v = [base + (0.0 if R.random() < 0.2 else R.gauss(0, R.choice([1e-3, 1.0, 50.0]))) for _ in range(n)]
    if R.random() < 0.2:
        v = [abs(x) for x in v]
    s = pd.Series(v)
    series.append({"v": v, "roll": [None if np.isnan(x) else x for x in s.rolling(144).mean()], "ewm": s.ewm(alpha=1 / 144, adjust=False).mean().tolist()})
# 상수 구간(같은 값 연속 → 그 값) · 전부 음수 · 0 과 음수 섞음(부호 고정)
for v in ([0.1] * 300, [3000.3] * 200 + [R.uniform(2990, 3010) for _ in range(200)], [-R.uniform(1e-3, 1.0) for _ in range(300)],
          [R.choice([-1e-17, 0.0, -3e-17]) for _ in range(300)], [R.choice([1e-17, 0.0, 3e-17]) for _ in range(300)], [0.7] * 150 + [0.1] * 150):
    s = pd.Series(v)
    series.append({"v": v, "roll": [None if np.isnan(x) else x for x in s.rolling(144).mean()], "ewm": s.ewm(alpha=1 / 144, adjust=False).mean().tolist()})
(OUT / "rest.json").write_text(json.dumps({"cases": cases, "series": series}))
print(f"사례 {len(cases)} · 급수 {len(series)} → {OUT}")
