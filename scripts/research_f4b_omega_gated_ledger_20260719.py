"""F4-B 정밀화: 실제 라이브 Omega4.6.1 duration-gate 적용 원장(145.34%/-10.13%/24트레이드)을
날짜정보 포함해서 재현. 무거운 torch/risk-sidecar 로드 없이 -- greedy_router_ledger_extended.csv
(이미 존재하는 no-gate 32트레이드 원장)에 ou_halflife 컬럼만 가벼운 CSV에서 병합해
scripts/replay_omega4_6_1_greedy_router_20260706.py의 게이트 로직을 그대로 재적용.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402

LEDGER_PATH = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/greedy_router_ledger_extended.csv"
DURATION_THRESHOLD = 0.005417
OUT_PATH = ROOT / "data/research/omega4_6_1_eth_gated_ledger_20260719.csv"


def main():
    print("Loading lightweight frame (CSV only, no torch/model deps)...")
    ext_frame = retest.load_frame_current("2026-01-01", "2026-06-30")
    market = ext_frame[["timestamp", "ou_halflife"]].copy()
    market["timestamp"] = pd.to_datetime(market["timestamp"])

    ledger = pd.read_csv(LEDGER_PATH)
    ledger["entry_timestamp_dt"] = pd.to_datetime(ledger["entry_timestamp"])
    merged = ledger.merge(
        market.rename(columns={"timestamp": "entry_timestamp_dt"}), on="entry_timestamp_dt", how="left"
    )
    n_missing = merged["ou_halflife"].isna().sum()
    print(f"rows missing ou_halflife after merge: {n_missing} / {len(merged)}")

    hit = merged["ou_halflife"] <= DURATION_THRESHOLD
    merged["duration_gate_skipped"] = hit
    gated_returns = merged["trade_return"].where(~hit, 0.0)

    import numpy as np
    curve = np.concatenate([[1.0], np.cumprod(1.0 + gated_returns.to_numpy())])
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1e-12) - 1.0
    n_active = int((~hit).sum())
    active_returns = merged.loc[~hit, "trade_return"]
    result = {
        "pnl_pct": float((curve[-1] - 1.0) * 100.0),
        "mdd_pct": float(dd.min() * 100.0),
        "trades_active": n_active,
        "trades_skipped": int(hit.sum()),
        "wr": float((active_returns > 0).mean()) if n_active else 0.0,
    }
    print("=== Reproduced gated result ===")
    print(result)
    print("Reference (2026-07-06 run, memory-recorded): pnl=145.34%, mdd=-10.13%, trades=24")

    merged.to_csv(OUT_PATH, index=False)
    print("wrote", OUT_PATH)


if __name__ == "__main__":
    main()
