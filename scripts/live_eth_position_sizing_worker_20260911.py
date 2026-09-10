#!/usr/bin/env python3
"""실시간 **크기 가늠자** 워커 (2026-09-11). 300초 주기. 상태파일만 쓴다.

사용자: *"그럼 이걸 내 대시보드에 어떻게 추가하는 게 좋을까?"*

## 무엇을 계산하나
① `atr_pct` 현재값과 **자기 이력 분위**
② **보정된 불리이탈 추정** `MAE_hat(q) = k_q · atr_pct · √H` — 계수 `k_q` 는 학습구간에서만 적합.
   [검증](../docs/what_magnitude_accuracy_means_for_sizing_20260911.md): 24/24 셀에서 상수 대조군 우위,
   표본외 보정격차 중앙 0.004(q=0.9 목표 초과 10% vs 실제 8.8~10.7%). 표본 40,749.
③ **변동성 등가 수량** = 기준수량 × (기준 atr_pct / 현재 atr_pct) — 역변동성 사이징 그대로.
④ **청산 도달 확률 곡선** — 지금과 비슷한 변동성 구간(atr_pct ±15%)의 과거 봉에서
   "진입 후 H시간 안에 역방향으로 d bp 이상 밀린 비율"을 거리 격자마다 그대로 센다.
   ⚠️모델 외삽이 아니라 **경험분포**다. 프런트가 포지션의 청산선 거리로 조회한다.
   [검증](../docs/sizing_rules_random_entry_20260911.md): 무작위 진입 82,167건에서 평균 명목 동일 조건에
   표준편차 −18% · 하위1% −22% · **50배 청산 도달률 13.9%→9.1%**.

## 🔴이 워커가 말하지 않는 것
**수익을 예측하지 않는다.** 실력이 0 이면 어떤 사이징도 손익 부호를 못 바꾼다(실측: 고정 −10.43 ·
역변동성 −10.46 · 재량 −10.28bp). 이건 **위험 눈금**이지 알파가 아니다. 카드 문구도 그렇게 쓴다.

## 왜 워커인가
대시보드 요청 경로에서 계산하면 스레드 풀이 고갈된다
([[feedback_dashboard_to_thread_pool_exhaustion_20260910]], 2026-09-10 실장애). 서버는 상태파일만 읽는다.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# 🔴경로를 하드코딩하지 않는다. 2026-09-11 배포에서 **개발 머신의 홈 경로**를 박아
#   서버(리눅스 계정이 다르다)에서 FileNotFoundError 로 워커가 못 떴다.
#   다른 라이브 워커와 같은 관례로 스크립트 위치에서 유도한다.
ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "data/live/eth_position_sizing_state.json"
CAL = ROOT / "data/live/eth_position_sizing_calib.json"
SYMBOL = "ETHUSDT"
PERIOD_S = 300
ATR_BARS = 288                      # 24시간
HOLDS = {"1h": 12, "4h": 48, "24h": 288}
# 청산 도달 확률을 조회할 거리 격자(bp). 50배 레버(≈200bp)·25배(≈400bp) 부근을 촘촘히.
DIST_GRID_BP = [50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500, 700, 1000, 1500, 2000]
VOL_BAND = 0.15                     # 현재 atr_pct 대비 ±15% 를 "비슷한 국면"으로 본다
QS = (0.5, 0.9, 0.95)
BASE_QTY_DEFAULT = 2.727            # 실계좌 관측 중앙 수량 — 사용자가 바꿀 수 있는 기준점
CALIB_END = "2025-08-31"            # 계수 적합 구간의 끝(그 뒤는 건드리지 않는다)


_TOUCH_CACHE: dict = {}


def log(m):
    print(f"[sizing {time.strftime('%H:%M:%S')}] {m}", flush=True)


def fetch_klines(limit=1500) -> pd.DataFrame | None:
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/klines",
                         params={"symbol": SYMBOL, "interval": "5m", "limit": limit}, timeout=20)
        r.raise_for_status()
        d = pd.DataFrame(r.json(), columns=["t", "o", "h", "l", "c", "v", "ct", "q",
                                            "n", "tb", "tq", "ig"])
    except Exception as e:
        log(f"klines 실패 {type(e).__name__}"); return None
    d["timestamp"] = pd.to_datetime(d["t"], unit="ms")
    for x in ("o", "h", "l", "c"):
        d[x] = pd.to_numeric(d[x])
    return d


def load_calib() -> dict:
    """`k_q` 는 학습구간에서 한 번만 적합해 파일로 굳힌다 — 매 주기 재적합하면 눈금이 흔들린다."""
    if CAL.exists():
        return json.loads(CAL.read_text())
    p = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
    d = pd.read_csv(p, usecols=["timestamp", "high", "low", "close"], parse_dates=["timestamp"])
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    d = d.loc[:CALIB_END]
    c = d["close"].to_numpy(float); hi = d["high"].to_numpy(float); lo = d["low"].to_numpy(float)
    atr = pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(ATR_BARS, min_periods=200).mean().to_numpy()
    ap = atr / np.maximum(c, 1e-9)
    out = {"calib_end": CALIB_END, "atr_bars": ATR_BARS, "k": {}, "atr_pct_ref": None}
    idx = np.arange(600, len(c) - max(HOLDS.values()) - 1, 12)
    idx = idx[np.isfinite(ap[idx]) & (ap[idx] > 0)]
    for name, H in HOLDS.items():
        mae_l = np.array([(c[i] - lo[i + 1:i + 1 + H].min()) / c[i] for i in idx])
        mae_s = np.array([(hi[i + 1:i + 1 + H].max() - c[i]) / c[i] for i in idx])
        base = ap[idx] * np.sqrt(H)
        out["k"][name] = {str(q): {"롱": float(np.quantile(mae_l / base, q)),
                                   "숏": float(np.quantile(mae_s / base, q))} for q in QS}
    out["atr_pct_ref"] = float(np.nanmedian(ap[idx]))       # 「평소 변동성」 기준점
    CAL.parent.mkdir(parents=True, exist_ok=True)
    CAL.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    log(f"계수 적합 완료(≤{CALIB_END}) · 기준 atr_pct {out['atr_pct_ref']:.5f}")
    return out


def compute(kl: pd.DataFrame, cal: dict, base_qty: float) -> dict:
    c = kl["c"].to_numpy(float)
    atr = pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(ATR_BARS, min_periods=200).mean().to_numpy()
    ap = float(atr[-1] / max(c[-1], 1e-9))
    hist = atr[np.isfinite(atr)] / c[np.isfinite(atr)]
    pct = float((hist <= ap).mean()) if len(hist) else float("nan")
    ref = float(cal["atr_pct_ref"])
    eq_qty = base_qty * (ref / ap) if ap > 0 else float("nan")
    out = {"ok": True, "generated_at": pd.Timestamp.utcnow().isoformat(),
           "price": float(c[-1]), "atr_pct": ap, "atr_pct_percentile": pct,
           "atr_pct_ref": ref, "base_qty": base_qty,
           "vol_equivalent_qty": eq_qty, "horizons": {}}
    for name, H in HOLDS.items():
        cell = {}
        for q in QS:
            k = cal["k"][name][str(q)]
            cell[str(q)] = {"롱_bp": float(k["롱"] * ap * np.sqrt(H) * 1e4),
                            "숏_bp": float(k["숏"] * ap * np.sqrt(H) * 1e4)}
        out["horizons"][name] = cell
    out["touch_prob"] = touch_probability(ap)
    return out


def touch_probability(cur_ap: float) -> dict:
    """⭐**청산에 닿을 확률** — 이 카드가 실제로 답해야 하는 질문.

    지금과 비슷한 변동성 구간(atr_pct ±15%)의 과거 봉에서, 진입 후 H시간 안에
    역방향으로 d bp 이상 밀린 **비율을 그대로 센다**. 모델도 분포 가정도 없다.
    캐시: atr_pct 는 24시간 평균이라 5분 주기로 거의 안 변한다. 밴드가 같으면 재사용."""
    key = round(cur_ap, 6)
    if _TOUCH_CACHE.get("key") == key:
        return _TOUCH_CACHE["val"]
    p = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
    d = pd.read_csv(p, usecols=["timestamp", "high", "low", "close"], parse_dates=["timestamp"])
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp")
    c = d["close"].to_numpy(float); hi = d["high"].to_numpy(float); lo = d["low"].to_numpy(float)
    atr = pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(ATR_BARS, min_periods=200).mean().to_numpy()
    ap = atr / np.maximum(c, 1e-9)
    band = np.flatnonzero((ap > cur_ap * (1 - VOL_BAND)) & (ap < cur_ap * (1 + VOL_BAND)))
    out = {"band_bars": int(len(band)), "vol_band_pct": VOL_BAND, "dist_grid_bp": DIST_GRID_BP,
           "horizons": {}}
    for name, H in HOLDS.items():
        idx = band[(band > 0) & (band < len(c) - H - 1)]
        if len(idx) < 500:
            continue
        mae_l = np.array([(c[i] - lo[i + 1:i + 1 + H].min()) / c[i] for i in idx]) * 1e4
        mae_s = np.array([(hi[i + 1:i + 1 + H].max() - c[i]) / c[i] for i in idx]) * 1e4
        out["horizons"][name] = {
            "n": int(len(idx)),
            "롱": [float((mae_l >= dd).mean()) for dd in DIST_GRID_BP],
            "숏": [float((mae_s >= dd).mean()) for dd in DIST_GRID_BP]}
    _TOUCH_CACHE.update(key=key, val=out)
    return out


def main(argv: list[str] | None = None) -> int:
    """`--loop` 없이 부르면 1회 계산 후 종료한다(스모크·수동 확인용).
    supervisor 는 `--loop` 를 붙인다 — pgrep 중복검사도 그 문자열에 의존한다."""
    argv = list(argv if argv is not None else sys.argv[1:])
    loop = "--loop" in argv
    cal = load_calib()
    base_qty = float(os.getenv("SIZING_BASE_QTY", BASE_QTY_DEFAULT))
    log(f"기준 수량 {base_qty} ETH · 주기 {PERIOD_S}s")
    while True:
        kl = fetch_klines()
        if kl is not None and len(kl) > ATR_BARS:
            try:
                st = compute(kl, cal, base_qty)
                STATE.parent.mkdir(parents=True, exist_ok=True)
                STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1))
                h = st["horizons"]["4h"]["0.9"]
                log(f"atr_pct {st['atr_pct']:.5f}(분위 {st['atr_pct_percentile']:.0%}) · "
                    f"변동성등가 수량 {st['vol_equivalent_qty']:.3f} ETH · "
                    f"4h 불리이탈 q90 롱 {h['롱_bp']:.0f}bp / 숏 {h['숏_bp']:.0f}bp")
            except Exception as e:
                log(f"계산 실패 {type(e).__name__}: {e}")
                STATE.write_text(json.dumps({"ok": False, "error": str(e)[:200]}, ensure_ascii=False))
        if not loop:
            return 0
        time.sleep(PERIOD_S - (time.time() % PERIOD_S))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
