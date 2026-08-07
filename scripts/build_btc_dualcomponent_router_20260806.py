"""BTC dual-component router (this session's version of ETH's live h48qual+zig075 priority
router): h48qual v1 (swing_transition_prob added, properly-retrained sidecar) has PRIORITY;
today's zigzag+pivot-transition 5m strategy fills bars where h48qual is NOT already in a position
(non-overlapping slot-filling -- a simplified but faithful approximation of ETH's greedy_replay:
no early-exit preemption of the secondary component, since that needs h48qual's full exit-head
apparatus which isn't reproduced here).

h48qual's ledger is filtered by its own reported duration-gate threshold (ou_halflife >
selected_duration_threshold) to match its "with_duration_gate" validated numbers exactly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
H48QUAL_DIR = ROOT / "tmp/causal_regen_20260516/btc_final_scale_map_swingtransition_properly_retrained_sidecar_20260806"
ZIGZAG_LEDGER_PATH = ROOT / "tmp/btc_1h_volregime_20260805/btc5m_zigzag_strategy_ledger_20260806.csv"
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
DURATION_THRESHOLD = 0.0054143218  # from h48qual v1's report.json selected_duration_threshold
ZIGZAG_NOTIONAL_SCALE = 0.35  # matches h48qual's own avg_notional (~0.32-0.40) -- the zigzag
# ledger's raw trade_return assumes 100% notional (unit exposure), which at its ~10 trades/day
# frequency makes naive full-notional compounding wildly overstate losses relative to how h48qual
# itself is actually sized. This scale makes the two ledgers' compounding comparable.


def compound_metrics(returns: pd.Series) -> dict:
    if returns.empty:
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0, "wr": 0.0}
    cash, peak, mdd, wins = 1.0, 1.0, 0.0, 0
    for r in returns.to_numpy(dtype=np.float64):
        cash *= 1.0 + float(r)
        wins += int(r > 0.0)
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
    return {"pnl": (cash - 1.0) * 100.0, "mdd": mdd * 100.0, "trades": len(returns), "wr": wins / len(returns)}


def main() -> int:
    ou = pd.read_parquet(PANEL_PATH, columns=["timestamp", "ou_halflife"]).rename(columns={"timestamp": "entry_timestamp"})

    h48_val = pd.read_csv(H48QUAL_DIR / "validation_ledger.csv", parse_dates=["entry_timestamp", "exit_timestamp"])
    h48_oos = pd.read_csv(H48QUAL_DIR / "oos_ledger.csv", parse_dates=["entry_timestamp", "exit_timestamp"])
    h48_val = h48_val.merge(ou, on="entry_timestamp", how="left")
    h48_oos = h48_oos.merge(ou, on="entry_timestamp", how="left")
    h48_val = h48_val[h48_val["ou_halflife"] > DURATION_THRESHOLD].reset_index(drop=True)
    h48_oos = h48_oos[h48_oos["ou_halflife"] > DURATION_THRESHOLD].reset_index(drop=True)
    h48_val["source_component"] = "h48qual"
    h48_oos["source_component"] = "h48qual"

    zz = pd.read_csv(ZIGZAG_LEDGER_PATH, parse_dates=["entry_timestamp", "exit_timestamp"])
    zz["trade_return"] = zz["trade_return"] * ZIGZAG_NOTIONAL_SCALE

    for split_name, h48_ledger, zz_split in [("validation", h48_val, "validation"), ("oos_extended", h48_oos, "oos_extended")]:
        zz_sub = zz[zz["split"] == zz_split].copy()
        h48_intervals = list(zip(h48_ledger["entry_timestamp"], h48_ledger["exit_timestamp"]))

        def overlaps_any(row) -> bool:
            for s, e in h48_intervals:
                if row["entry_timestamp"] < e and row["exit_timestamp"] > s:
                    return True
            return False

        zz_sub["blocked_by_h48qual"] = zz_sub.apply(overlaps_any, axis=1) if h48_intervals else False
        zz_kept = zz_sub[~zz_sub["blocked_by_h48qual"]].copy()

        combined = pd.concat([
            h48_ledger[["entry_timestamp", "exit_timestamp", "trade_return", "source_component"]],
            zz_kept[["entry_timestamp", "exit_timestamp", "trade_return", "source_component"]],
        ]).sort_values("entry_timestamp").reset_index(drop=True)

        h48_only = compound_metrics(h48_ledger["trade_return"])
        zz_only_all = compound_metrics(zz_sub["trade_return"])
        zz_kept_only = compound_metrics(zz_kept["trade_return"])
        router = compound_metrics(combined["trade_return"])

        print(f"\n===== {split_name} =====")
        print(f"h48qual alone (priority, duration-gated): {h48_only}")
        print(f"zigzag alone (ALL {len(zz_sub)} candidate trades, no router): {zz_only_all}")
        print(f"zigzag trades blocked by h48qual overlap: {int(zz_sub['blocked_by_h48qual'].sum())} / {len(zz_sub)}")
        print(f"zigzag kept (non-overlapping only): {zz_kept_only}")
        print(f"ROUTER (h48qual priority + zigzag fills gaps): {router}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
