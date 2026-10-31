#!/usr/bin/env python3
"""ETH 박스 유지 확률 1h «굴림 Platt90» 월간 적합 (2026-10-26) -- docs/eth_box_hold_calibration_drift_prereg_20261025.md 8라운드·«배포 대상 점검».

기준 시각 T(매월 1일 00:00 UTC) 직전 90일의 1h 격자 행(5분마다 · 라벨이 T 전에 해결된 것만)에서 배포 모델(box_hold_hgb_20261016) 출력 p 로
Platt(로지스틱 on logit p) 를 맞춰 dashboard/box_hold_platt90_eth_1h.json 에 쓴다. 서빙(dashboard/box_hold.payload)은 1h 에만 p' = sig(a·logit p + b).
연구 기준: scripts/research_eth_box_hold_calibration_drift_20261025.py rolling("platt", …, N=90) · fit_platt · labels (패리티 test/test_box_hold_platt90_20261026.py).
피쳐는 서빙 features() 그대로(행마다 그 시점까지의 1분봉 NEED+60 개 = payload 와 같은 입력).

  python scripts/fit_box_hold_platt90_20261026.py                       # T = 이번 달 1일 · 1분봉 = data.binance.vision parquet(연구 원천)
  python scripts/fit_box_hold_platt90_20261026.py --rest --events <events.json>   # 서버: 바이낸스 공개 REST /fapi/v1/klines
  python scripts/fit_box_hold_platt90_20261026.py --month 2026-09-01 --out /tmp/x.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dashboard.box_hold import BUNDLES, NEED, PLATT90, PLATT90_MODEL, PLATT90_XRP, features, logit  # noqa: E402

MAIN = Path("/home/kbj20/crypto-scalping")
BV = MAIN / "data/binance_vision"
# 연구가 쓴 발표 시각 캐시({"major": [ms], "claims": [ms]} · FRED 날짜 + 관례 시각 · 2026-12-31 까지)
EVENTS = MAIN / ".claude/worktrees/chart-indicators-tuning-ecfa25/tmp/box_improve_20261016/events.json"
# 연구 build() 의 VIS(data.binance.vision 일 zip 09-14~10-23 · parquet 빈칸 09-15~10-04 를 메움) -- 연구를 돌린 워크트리
VIS = MAIN / ".claude/worktrees/agent-a231de71bb7c16d79/tmp/box_calib_20261025/vision_1m"
SYM = {"eth": "ETHUSDT", "xrp": "XRPUSDT"}   # 2026-10-31 --asset xrp(XRP 번들·요일×시각 표·출력 파일)
ASSET = "eth"
W, N = 60, 90
MIN = pd.Timedelta(minutes=1)


def load_vision(lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    """연구 load_1m() 과 같은 원천·순서(parquet 먼저, 겹치면 parquet)."""
    sym = SYM[ASSET]
    fs = sorted(glob.glob(f"{BV}/klines1m/{sym}-1m-*.parquet")) + sorted(glob.glob(f"{BV}/klines1m_daily/{sym}-1m-*.parquet"))
    fs = [f for f in fs if Path(f).stem[len(sym) + 4:len(sym) + 11] >= lo.strftime("%Y-%m")]          # 파일명 날짜(YYYY-MM[-DD]) ≥ 시작 달
    parts = [pd.read_parquet(f)[["t", "h", "l", "c", "v", "tb"]] for f in fs]
    for z in sorted(glob.glob(str(VIS / f"{sym}-*.zip"))):
        with zipfile.ZipFile(z) as zf:
            x = pd.read_csv(zf.open(zf.namelist()[0]))
        parts.append(pd.DataFrame({"t": x.open_time, "h": x.high, "l": x.low, "c": x.close, "v": x.volume, "tb": x.taker_buy_volume}))
    return pd.concat(parts)


def load_rest(lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    out, s, e = [], int(lo.value // 10 ** 6), int(hi.value // 10 ** 6)
    while s <= e:
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={SYM[ASSET]}&interval=1m&limit=1000&startTime={s}&endTime={e}"
        for i in range(5):   # 10-26 리뷰: 월 1회 cron 이라 한 번 끊기면 그 달 보정이 빠졌다 -- 5번까지(418/429 면 길게 쉼)
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    x = json.loads(r.read())
                break
            except Exception as exc:  # noqa: BLE001
                if i == 4:
                    raise
                time.sleep(120 if getattr(exc, "code", None) in (418, 429) else 10 * (i + 1))
        if not x:
            break
        out += x
        s = int(x[-1][0]) + 60_000
        time.sleep(0.5)   # 1000개 = 가중치 5 · 92일 ≈ 135번(1500개 가중치 10 보다 총량 적음) -- 서버·로컬 같은 IP 429 회피
    return pd.DataFrame({"t": [int(k[0]) for k in out], **{n: [float(k[j]) for k in out] for n, j in (("h", 2), ("l", 3), ("c", 4), ("v", 5), ("tb", 9))}})


def labels(h: np.ndarray, l: np.ndarray, c: np.ndarray, W: int) -> np.ndarray:
    """연구 labels() 그대로: y[i] = 다음 W분 고저가 c[i] ± ½(지난 W분 고저폭) 안. 못 정하면 −1."""
    mx, mn = sliding_window_view(h, W).max(1), sliding_window_view(l, W).min(1)
    y = np.full(len(c), -1, np.int8)
    i = np.arange(W - 1, len(c) - W)
    half = 0.5 * (mx[i - W + 1] - mn[i - W + 1])
    y[i] = ((mx[i + 1] < c[i] + half) & (mn[i + 1] > c[i] - half)).astype(np.int8)
    return y


def fit_rows(T: pd.Timestamp, rest: bool, events_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """T 에 맞출 행의 (ts, 배포 모델 p, y). 연구 rolling: ts ≥ T−90일 · ts+W+1분 ≤ T · 격자 = 봉 시작 분 %5 == 4 · 라벨 해결 · 창폭 > 0 · 피쳐 유한."""
    w0 = T - pd.Timedelta(days=N)
    lo, hi = w0 - (NEED + 60) * MIN, T - MIN                                      # 첫 행에 payload 와 같은 1분봉 NEED+60 개
    K = (load_rest if rest else load_vision)(lo, hi)
    K = K.drop_duplicates("t").sort_values("t")
    K = K[(K.t >= lo.value // 10 ** 6) & (K.t <= hi.value // 10 ** 6)].reset_index(drop=True)
    assert len(K) and K.t.iloc[0] == lo.value // 10 ** 6 and K.t.iloc[-1] == hi.value // 10 ** 6, "1분봉 범위 모자람"
    assert (np.diff(K.t) == 60_000).all(), "1분봉 빈칸"
    h, l, c = (K[k].to_numpy(float) for k in ("h", "l", "c"))
    y = labels(h, l, c, W)
    ts = pd.to_datetime(K.t, unit="ms")
    rng = pd.Series(h).rolling(W).max().to_numpy() - pd.Series(l).rolling(W).min().to_numpy()
    ii = np.where((K.t // 60_000 % 5 == 4) & (ts >= w0) & (ts + (W + 1) * MIN <= T) & (y >= 0) & (rng > 0))[0]
    ev = {k: np.asarray(v, np.int64) for k, v in json.loads(events_path.read_text()).items()}
    near = lambda v: v[(v >= lo.value // 10 ** 6 - 86_400_000) & (v <= T.value // 10 ** 6 + 86_400_000)]  # noqa: E731 -- 먼 발표는 피쳐에 안 닿음(속도)
    iso = lambda v: pd.Timestamp(int(v), unit="ms", tz="UTC").isoformat()  # noqa: E731
    evl = [{"time_utc": iso(v), "category": "fomc", "title_ko": "FOMC 금리결정 발표"} for v in near(ev["major"])] + \
          [{"time_utc": iso(v), "category": "econ", "detail": "release_id=180"} for v in near(ev["claims"])]   # 연구 build() 와 같은 변환
    k1m = pd.DataFrame({"timestamp": pd.to_datetime(K.t, unit="ms", utc=True), "high": h, "low": l, "close": c,
                        "volume": K.v.to_numpy(float), "taker_buy_base": K.tb.to_numpy(float)})
    bb = BUNDLES[ASSET]
    cf, prof = bb["scales"]["1h"], np.asarray(bb["prof"], float)   # 🔴XRP 는 XRP 요일×시각 표(서빙 payload 와 같다)
    X = pd.DataFrame([features(k1m.iloc[i - NEED - 59:i + 1].reset_index(drop=True), evl, prof)["1h"] for i in ii])[cf["cols"]]
    ok = np.isfinite(X.to_numpy()).all(1)                                          # 연구 dropna(inf→nan) = 서빙 ok False
    return ts.to_numpy()[ii][ok], cf["model"].predict_proba(X[ok])[:, 1], y[ii][ok].astype(float)


def fit_platt(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """연구 fit_platt 그대로 → (a = 기울기, b = 절편), p' = sig(a·logit p + b)."""
    m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p)[:, None], y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def fit(T: pd.Timestamp, rest: bool = False, events_path: Path = EVENTS) -> dict:
    ts, p, y = fit_rows(T, rest, events_path)
    a, b = fit_platt(p, y)
    return {"a": a, "b": b, "fit_start": str(T - pd.Timedelta(days=N)), "fit_end": str(T), "n_rows": int(len(y)),
            "created_utc": pd.Timestamp.now(tz="UTC").isoformat(), "model_file": PLATT90_MODEL[ASSET], "asset": ASSET,
            "source": "rest" if rest else "vision", "train_hold": float(y.mean()), "mean_p": float(p.mean())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", help="기준 T(YYYY-MM-01, UTC 00:00) -- 기본 이번 달 1일")
    ap.add_argument("--rest", action="store_true", help="1분봉을 바이낸스 공개 REST 로(서버)")
    ap.add_argument("--events", type=Path, default=EVENTS)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--asset", choices=sorted(SYM), default="eth")
    a = ap.parse_args()
    ASSET = a.asset
    a.out = a.out or (PLATT90 if ASSET == "eth" else PLATT90_XRP)
    T = pd.Timestamp(a.month) if a.month else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize().replace(day=1)
    assert T == T.normalize().replace(day=1), "T 는 달 1일 00:00"
    r = fit(T, a.rest, a.events)
    a.out.write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))
