#!/usr/bin/env python3
"""BTC 증거신호 7종 **라이브 메타라벨 스코어러** -- 대시보드 칩용 (판단 산출 전용).

## 이게 무엇인가

ETH의 `live_evidence_signal_metalabel_20260829.py`에 대응하는 BTC판. 2026-09-01에 BTC 7종의
그리드스크린 + TabPFN 메타라벨 검증이 끝났는데 **서빙 아티팩트가 없어** 라이브가 없었다.
2026-09-02에 동결 컨텍스트를 만들고(`build_btc_evidence_signal_frozen_contexts_20260902.py`)
이 스코어러를 붙인다.

## ⚠️ETH와 신호 정의는 같고, 라벨 파라미터는 BTC 전용이다

원시 신호 계산은 `compute_signals()`를 **ETH와 공유**한다(BTC 후보 빌드가 verbatim 재사용).
자산별로 다른 건 **메타라벨의 HIT정의/H/K/GAP**이고, 2026-09-01 그리드스크린이 BTC에서
독자 재선정했다:

    신호                       HIT정의        H     K      GAP    TRAIN hit률
    taker_delta_climax        종가기준       6     2.0     3      0.1388
    liquidity_sweep           터치+되돌림   20     2.0     6      0.1022
    kalman_deviation_meanrev  touch MFE     10     3.5     6      0.1425
    short_term_return_z       MAE 상한       6     2.0    12      0.3163
    orthogonal_combo          touch MFE      8     2.0     6      0.4271
    demarker_extreme          touch MFE      8     0.70    6      0.9003
    fib_extension_exhaustion  종가기준      10     2.75    6      0.1928

⚠️`demarker_extreme`의 hit률 0.9003은 K=0.70이 낮아 거의 다 맞는다는 뜻 -- 라이브 변별력이
낮으므로 표시할 때 감안할 것.

## ⚠️매매 신호가 아니다

ETH 증거신호와 같은 지위 -- **사람의 재량 판단을 위한 확률 이동 컨텍스트**다. 주문을 내지 않고
자동매매에 배선되지 않는다. BTC는 경제성 게이트를 통과한 모델이 아직 없다(2026-09-02
경제라벨 시도가 VAL에서 실패: 손익비 하한을 걸면 수익 조합이 없음).

Usage:
    from live_btc_evidence_signal_metalabel_20260902 import compute_btc_evidence_signals
    out = compute_btc_evidence_signals()
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CTX_DIR = ROOT / "data/labels/btc_5m_evidence_signal_live_contexts_20260902"
CTX_REPORT = CTX_DIR / "contexts_report.json"
SYMBOL = "BTCUSDT"
SCORE_TAIL_BARS = 3          # 최근 몇 봉을 채점할지
_MODELS: dict[str, Any] | None = None
_META: dict[str, Any] | None = None


def _meta() -> dict[str, Any]:
    global _META
    if _META is None:
        _META = json.loads(CTX_REPORT.read_text())
    return _META


def _load_models() -> dict[str, Any]:
    """신호별 동결 컨텍스트로 TabPFN을 적합해 캐시한다(첫 호출만 비용)."""
    global _MODELS
    if _MODELS is not None:
        return _MODELS
    from tabpfn import TabPFNClassifier
    out: dict[str, Any] = {}
    for name, info in _meta()["signals"].items():
        if "error" in info:
            continue
        csv = ROOT / info["artifact"]
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        feats = info["features"]
        if df["hit"].nunique() < 2:
            continue
        clf = TabPFNClassifier(device="cuda", random_state=20260902,
                               ignore_pretraining_limits=True)
        clf.fit(df[feats], df["hit"].to_numpy())
        out[name] = {"clf": clf, "features": feats,
                     "train_hit_rate": info["hit_rate"], "params": info["btc_params"]}
    _MODELS = out
    return out


def _build_frame() -> pd.DataFrame | None:
    """라이브 BTC 봉 -> 후보 빌드와 동일한 지표 프레임.

    ⚠️`build_btc_5m_evidence_signal_candidates_tier0_20260901.py`가 쓰는 것과 **같은 함수들**을
    재사용한다. 재구현하면 학습/추론 패리티가 조용히 깨진다.
    """
    from analyze_eth_broad_evidence_signal_sweep_20260814 import add_broad_indicators
    from analyze_eth_creative_reversal_evidence_signals_20260814 import add_creative_indicators
    from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import compute_indicators
    import requests

    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/klines",
                         params={"symbol": SYMBOL, "interval": "5m", "limit": 1500}, timeout=15)
        r.raise_for_status()
        raw = r.json()
    except Exception:                                          # noqa: BLE001
        return None
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "qv",
            "trades", "taker_buy_base", "tq", "ignore"]
    kl = pd.DataFrame(raw, columns=cols)
    for c in ("open", "high", "low", "close", "volume", "taker_buy_base"):
        kl[c] = kl[c].astype(float)
    kl["timestamp"] = pd.to_datetime(kl["open_time"], unit="ms", utc=True)
    kl = kl.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    now_ms = int(time.time() * 1000)
    if len(kl) and int(kl.iloc[-1]["close_time"]) >= now_ms:
        kl = kl.iloc[:-1].reset_index(drop=True)               # 형성중 봉 제거
    if len(kl) < 900:
        return None
    f = compute_indicators(kl)
    f = add_creative_indicators(f)
    f = add_broad_indicators(f)

    # ⚠️후보 빌드(build_btc_5m_evidence_signal_candidates_tier0_20260901.py:159~173)와
    # **동일한 공식**으로 파생 피쳐를 만든다. 여기가 어긋나면 학습/추론 패리티가 깨진다.
    bspec = importlib.util.spec_from_file_location(
        "btc_cand_build",
        ROOT / "scripts/build_btc_5m_evidence_signal_candidates_tier0_20260901.py")
    bmod = importlib.util.module_from_spec(bspec)
    bspec.loader.exec_module(bmod)
    sweep_impl = bmod.load_sweep_impl()

    ret3 = f["close"] / f["close"].shift(3) - 1.0
    f["ret3_z"] = (ret3 - ret3.rolling(288, min_periods=288).mean()) \
        / ret3.rolling(288, min_periods=288).std().replace(0.0, np.nan)
    causal = sweep_impl.add_causal_columns(kl[["timestamp", "open", "high", "low", "close"]].copy())
    f["sweep_level_low"] = causal["sweep_level_low"]
    f["sweep_level_high"] = causal["sweep_level_high"]
    f["atr"] = causal["atr"]
    f["atr_percentile_864"] = f["atr"].rolling(864, min_periods=864).rank(pct=True)
    f["range_width_pct"] = (f["sweep_level_high"] - f["sweep_level_low"]) / f["close"]
    f["hour_utc"] = f["timestamp"].dt.hour
    f["weekday"] = f["timestamp"].dt.weekday
    f["rsi"] = bmod.rsi_wilder(f["close"])

    # 신호 전용 파생 2개 -- 각 연구 스크립트의 계산부를 그대로 import
    dspec = importlib.util.spec_from_file_location(
        "btc_dem", ROOT / "scripts/research_btc_demarker_extreme_metalabel_tabpfn_20260901.py")
    dmod = importlib.util.module_from_spec(dspec); dspec.loader.exec_module(dmod)
    f["dem"] = dmod.compute_demarker(f["high"], f["low"]).to_numpy()
    kspec = importlib.util.spec_from_file_location(
        "btc_kal", ROOT / "scripts/research_btc_kalman_deviation_meanrev_metalabel_tabpfn_20260901.py")
    kmod = importlib.util.module_from_spec(kspec); kspec.loader.exec_module(kmod)
    f["kalman_dev_z"] = kmod.compute_kalman_dev_z(f["close"].to_numpy())

    # ⚠️Tier0 CSV에 없어 각 연구 스크립트가 자체 추가하던 4개 파생 피쳐
    # (atr_pct 재계산 / nyse_open_flag / er_24 / realized_vol_ratio).
    # **재구현하지 않고 원본 함수를 import**한다 -- 여기서 다시 쓰면 학습/추론 패리티가
    # 조용히 깨진다(스모크 테스트에서 실제로 missing features로 드러났던 지점).
    spec = importlib.util.spec_from_file_location(
        "btc_ls_prep",
        ROOT / "scripts/research_btc_liquidity_sweep_metalabel_tabpfn_20260901.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    if "atr" not in f.columns and "atr_price" in f.columns:
        f["atr"] = f["atr_price"]
    if "atr_pct" not in f.columns:
        f["atr_pct"] = f["atr"] / f["close"].clip(lower=1e-12)
    f = prep.add_missing_features(f)

    return f


def compute_btc_evidence_signals() -> dict[str, Any]:
    """최근 봉들의 신호별 발동 여부 + 메타라벨 확률. 주문은 내지 않는다."""
    empty = {"warmed_up": False, "error": None, "asset": SYMBOL, "signals": {},
             "note": "재량 참고용 컨텍스트 -- 매매 신호 아님, 주문 없음"}
    try:
        if not CTX_REPORT.exists():
            return {**empty, "error": "frozen_contexts_missing"}
        frame = _build_frame()
        if frame is None:
            return {**empty, "error": "klines_unavailable"}
        from live_evidence_signal_dashboard_20260823 import compute_signals
        sig = compute_signals(frame, btc_df=None, funding_df=None)

        models = _load_models()
        n = len(frame)
        lo = max(0, n - SCORE_TAIL_BARS)
        out: dict[str, Any] = {}
        for name, m in models.items():
            fired = {}
            for side in ("bottom", "top"):
                col = f"{side}_{name}"
                if col not in sig.columns:
                    continue
                arr = sig[col].fillna(False).to_numpy()
                idx = [i for i in range(lo, n) if arr[i]]
                if not idx:
                    continue
                rows = frame.iloc[idx].copy()
                rows["is_bottom"] = 1 if side == "bottom" else 0
                miss = [c for c in m["features"] if c not in rows.columns]
                if miss:
                    fired[side] = {"error": f"missing features: {miss[:4]}"}
                    continue
                X = rows[m["features"]].apply(pd.to_numeric, errors="coerce")
                if X.isna().any(axis=1).all():
                    continue
                ok = ~X.isna().any(axis=1)
                p = m["clf"].predict_proba(X.loc[ok])[:, 1]
                fired[side] = {
                    "n_fires": int(ok.sum()),
                    "latest_proba": round(float(p[-1]), 4) if len(p) else None,
                    "latest_bar_utc": str(frame["timestamp"].iloc[idx[-1]]),
                }
            out[name] = {"fired": fired, "train_hit_rate": m["train_hit_rate"],
                         "btc_params": m["params"]}
        return {**empty, "warmed_up": True, "signals": out,
                "latest_bar_utc": str(frame["timestamp"].iloc[-1]),
                "n_models": len(models)}
    except Exception as e:                                     # noqa: BLE001
        return {**empty, "error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    t0 = time.time()
    r = compute_btc_evidence_signals()
    r["cycle_sec"] = round(time.time() - t0, 2)
    print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
