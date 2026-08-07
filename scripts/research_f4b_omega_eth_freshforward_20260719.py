"""F4-B 검증용: Omega4.6.1 ETH 단독을 진짜 fresh-forward bar-by-bar(5분봉)로 재실행.

저장된 최종 원장을 읽는 게 아니라, scripts/replay_portfolio_fresh_window_20260713.py의
_replay_concurrent_entry_floor()를 그대로 재사용해서 매 5분봉마다:
  1. 열린 포지션이 있으면 그 bar에서 종료 조건(TP/SL/exit-head) 평가
  2. 없으면 그 bar까지 causal하게 계산된 신호로 신규 진입 여부 결정
을 처음부터 끝까지 순차 실행한다. 라이브와 동일하게 duration-gate OFF,
ETH notional_multiplier=1.5 적용. enabled_assets=("eth",)로 SOL/BTC와의
자본(margin_cap) 경쟁을 제거해 "ETH 단독이었다면" 시나리오로 격리.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import replay_portfolio_fresh_window_20260713 as fw  # noqa: E402

OUT_DIR = ROOT / "data/research"
NEW_END = "2026-07-13"


def main():
    native = fw.native
    concurrent = fw.concurrent
    eth_retest = fw.eth_retest

    _orig_load_frame_current = eth_retest.load_frame_current

    def _patched(start: str, end: str) -> pd.DataFrame:
        return _orig_load_frame_current(start, NEW_END)

    eth_retest.load_frame_current = _patched
    try:
        device = eth_retest.DEVICE
        native.DURATION_THRESHOLDS = {k: -999.0 for k in native.DURATION_THRESHOLDS}
        print("stage=build_world (loads ETH/SOL/BTC frames+bundles, may take a while)...", flush=True)
        world = native._build_world("oos", device)
        print(f"world timestamps: {world['timestamps'][0]} .. {world['timestamps'][-1]}, n={len(world['timestamps'])}", flush=True)

        print("stage=bar-by-bar replay, ETH ONLY, no entry_floor (full range)...", flush=True)
        metrics, ledger, timeline, diag = fw._replay_concurrent_entry_floor(
            world, device=device, cap_mode="scale",
            asset_shares={"eth": 1.0, "btc": 0.0, "sol": 0.0},
            asset_notional_multipliers={"eth": 1.5, "btc": 1.0, "sol": 1.0},
            enabled_assets=("eth",),
            entry_floor=None,
        )
    finally:
        eth_retest.load_frame_current = _orig_load_frame_current

    print("=== ETH-only bar-by-bar fresh-forward result (live-matching config) ===")
    print(json.dumps(metrics, indent=2, default=str))

    ledger.to_csv(OUT_DIR / "omega461_eth_only_freshforward_ledger_20260719.csv", index=False)
    print("wrote", OUT_DIR / "omega461_eth_only_freshforward_ledger_20260719.csv", "rows:", len(ledger))


if __name__ == "__main__":
    main()
