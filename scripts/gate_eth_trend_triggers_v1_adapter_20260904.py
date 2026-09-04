#!/usr/bin/env python3
"""L1 게이트용 어댑터 -- 추세 신호 v1 5종 raw 트리거를 klines(절단본)에서 다시 만든다 (2026-09-04).

정본 트리거 = `research_eth_trend_signals_v1_homer_pipeline_20260904.build_triggers(kl, aux)`. 보조 표(BTC·현물·OI·레짐 OOF·bookDepth)는
kl과 무관하게 한 번 로드되고 봉 timestamp로 merge_asof(backward)되므로, kl을 known_ts에서 잘라도 정렬이 어긋나지 않고 미래 행을 쓰지 않는다.
반환: [timestamp, signal, side, known_ts]. raw 트리거는 봉 마감에 계산되므로 known_ts = timestamp.
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("homer_trend", ROOT / "scripts/research_eth_trend_signals_v1_homer_pipeline_20260904.py"); HT = importlib.util.module_from_spec(_s); _s.loader.exec_module(HT)


def build_fires(kl: pd.DataFrame, signals=tuple(HT.SIGNALS)) -> pd.DataFrame:
    kl = kl.reset_index(drop=True).copy()
    if kl["timestamp"].dt.tz is not None:
        kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)
    T, _, _ = HT.build_triggers(kl, HT.load_aux()); ts = kl["timestamp"].to_numpy(); rows = []
    for name in signals:
        up, dn = T[name]
        for side, m in (("up", up), ("dn", dn)):
            idx = np.flatnonzero(m); rows.append(pd.DataFrame({"timestamp": ts[idx], "signal": name, "side": side}))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["timestamp", "signal", "side"]); out["known_ts"] = out["timestamp"]
    return out
