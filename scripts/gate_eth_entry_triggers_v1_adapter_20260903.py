#!/usr/bin/env python3
"""L1 게이트용 어댑터 -- ETH 증거신호 8종 raw 트리거를 klines 프레임에서 다시 만든다 (2026-09-03).

`live_evidence_signal_dashboard_20260823.compute_signals(kl, btc_df)`가 라이브 정의(정본)이고,
`research_eth_causal_population_metalabel_prep_20260902.py`가 같은 함수로 v1 발동 모집단을 만들었다
("raw triggers -- NO cluster_dedup (causal)"). 그 주장을 L1이 기계적으로 확인한다.

반환: [timestamp, signal, side, known_ts]. raw 트리거는 봉 마감에 계산되므로 known_ts = timestamp.
BTC는 timestamp로 merge되므로(compute_signals 내부) 잘린 kl에 전체 BTC를 넘겨도 정렬이 어긋나지 않는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

BTC_PATH = ROOT / "data/splits/year_oos/btc_features_2024_2026.csv"
SIGNALS = ["taker_delta_z_climax", "short_term_return_z", "liquidity_sweep", "orthogonal_combo",
           "smt_divergence", "fib_extension_exhaustion", "demarker_extreme", "kalman_deviation_meanrev"]
_BTC: pd.DataFrame | None = None


def _btc() -> pd.DataFrame:
    global _BTC
    if _BTC is None:
        b = pd.read_csv(BTC_PATH, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
        _BTC = b.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return _BTC


def build_fires(kl: pd.DataFrame, signals=tuple(SIGNALS)) -> pd.DataFrame:
    from live_evidence_signal_dashboard_20260823 import compute_signals
    kl = kl.reset_index(drop=True)
    sig = compute_signals(kl, btc_df=_btc())
    ts = kl["timestamp"].to_numpy()
    rows = []
    for name in signals:
        for side in ("bottom", "top"):
            col = f"{side}_{name}"
            if col not in sig.columns:
                continue
            idx = np.flatnonzero(sig[col].fillna(False).to_numpy(dtype=bool))
            rows.append(pd.DataFrame({"timestamp": ts[idx], "signal": name, "side": side}))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["timestamp", "signal", "side"])
    out["known_ts"] = out["timestamp"]
    return out
