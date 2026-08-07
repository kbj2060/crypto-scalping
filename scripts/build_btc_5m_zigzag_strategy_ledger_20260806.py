"""BTC 5m: regenerate the quality-weighted Layer A+B combined strategy's TRADE LEDGER (entry/exit
timestamps + returns, not just aggregate stats), extended through 2026-08-01 to match h48qual's
OOS-extended window -- needed to build a dual-component router (h48qual priority, this strategy as
secondary/gap-filler), same pattern as ETH's live h48qual+zig075 router.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
LAYERA_PRED_PATH = ROOT / "tmp/btc_1h_volregime_20260805/btc5m_layerA_pred.parquet"
LAYERB_PRED_PATH = ROOT / "tmp/btc_1h_volregime_20260805/btc5m_layerB_qualityweighted_pred.parquet"

VAL_START, OOS_START, OOS_END = "2025-09-01", "2026-01-01", "2026-08-02"
MAX_HOLD = 288
ROUND_TRIP_COST = 0.0010
TP_MULT, SL_MULT = 2.5, 1.2
TRAIL_VOL_BARS = 288
GATE_PCTL = 0.90


def main() -> int:
    panel = pd.read_parquet(PANEL_PATH, columns=["timestamp", "close", "high", "low"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    log_ret = np.log(panel["close"]).diff()
    panel["trailing_vol"] = log_ret.rolling(TRAIL_VOL_BARS, min_periods=TRAIL_VOL_BARS).std()

    layerA = pd.read_parquet(LAYERA_PRED_PATH)
    layerB = pd.read_parquet(LAYERB_PRED_PATH)
    df = panel.merge(layerA, on="timestamp", how="inner").merge(layerB, on="timestamp", how="inner")
    df = df.sort_values("timestamp").reset_index(drop=True)

    val_df = df[(df["timestamp"] >= VAL_START) & (df["timestamp"] < OOS_START)]
    gate_thresh = val_df["probA"].quantile(GATE_PCTL)
    print(f"gate threshold (fit on VAL): {gate_thresh:.4f}")

    all_trades = []
    for split_name, start, end in [("validation", VAL_START, OOS_START), ("oos_extended", OOS_START, OOS_END)]:
        sub = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)
        close, high, low = sub["close"].to_numpy(), sub["high"].to_numpy(), sub["low"].to_numpy()
        vol, probA, pred, ts = sub["trailing_vol"].to_numpy(), sub["probA"].to_numpy(), sub["pred"].to_numpy(), sub["timestamp"].to_numpy()
        n = len(sub)
        i = 0
        while i < n:
            if not np.isfinite(vol[i]) or vol[i] <= 0 or int(pred[i]) == 0 or probA[i] < gate_thresh:
                i += 1
                continue
            direction = 1 if int(pred[i]) == 1 else -1
            entry_price = close[i]
            v = vol[i]
            tp_price = entry_price * (1 + direction * TP_MULT * v)
            sl_price = entry_price * (1 - direction * SL_MULT * v)
            exit_price, exit_i = None, None
            j_end = min(i + MAX_HOLD, n - 1)
            for j in range(i + 1, j_end + 1):
                hit_tp = high[j] >= tp_price if direction == 1 else low[j] <= tp_price
                hit_sl = low[j] <= sl_price if direction == 1 else high[j] >= sl_price
                if hit_tp and hit_sl:
                    exit_price, exit_i = sl_price, j
                    break
                if hit_tp:
                    exit_price, exit_i = tp_price, j
                    break
                if hit_sl:
                    exit_price, exit_i = sl_price, j
                    break
            if exit_price is None:
                exit_price, exit_i = close[j_end], j_end
            ret = direction * (exit_price - entry_price) / entry_price - ROUND_TRIP_COST
            all_trades.append({
                "split": split_name, "entry_timestamp": pd.Timestamp(ts[i]), "exit_timestamp": pd.Timestamp(ts[exit_i]),
                "side": direction, "trade_return": ret, "source_component": "zigzag_pivot_5m",
            })
            i = exit_i + 1

    ledger = pd.DataFrame(all_trades)
    out_path = ROOT / "tmp/btc_1h_volregime_20260805/btc5m_zigzag_strategy_ledger_20260806.csv"
    ledger.to_csv(out_path, index=False)
    print(f"wrote {out_path}, {len(ledger)} trades")
    print(ledger.groupby("split")["trade_return"].agg(["count", "sum", "mean"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
