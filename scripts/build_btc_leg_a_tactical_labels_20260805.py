"""Build widened 5-minute tactical execution labels for Tau1-style Leg A."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from train_eval_btc_110branch_causal_20260804 import COST, load_frame  # noqa: E402

OUT = ROOT / "tmp/btc_leg_a_tactical_labels_20260805"
HORIZON, TP_FLOOR, SL_FLOOR, TP_ATR_MULT, SL_ATR_MULT = 96, .020, .010, 8.0, 4.0
NET_FLOOR, SIDE_EDGE = .004, .005


def atr_pct(frame: pd.DataFrame) -> np.ndarray:
    high, low, close = (frame[c].to_numpy(float) for c in ("high", "low", "close"))
    previous = np.r_[close[0], close[:-1]]
    tr = np.maximum.reduce([high - low, np.abs(high - previous), np.abs(low - previous)])
    return pd.Series(tr / close).rolling(14, min_periods=14).mean().to_numpy(float)


def first_hit(mask: np.ndarray) -> np.ndarray:
    return np.where(mask.any(axis=1), mask.argmax(axis=1), mask.shape[1])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    frame, _ = load_frame()
    high, low, close, open_px = (frame[c].to_numpy(float) for c in ("high", "low", "close", "open"))
    atr = atr_pct(frame)
    n = len(frame) - HORIZON
    decisions = np.arange(n)
    entries = open_px[1 : n + 1]
    highs = sliding_window_view(high[1:], HORIZON)
    lows = sliding_window_view(low[1:], HORIZON)
    closes = sliding_window_view(close[1:], HORIZON)
    tp = np.maximum(TP_FLOOR, TP_ATR_MULT * atr[:n])
    sl = np.maximum(SL_FLOOR, SL_ATR_MULT * atr[:n])
    valid = np.isfinite(tp) & np.isfinite(sl)
    long_tp, long_sl = first_hit(highs >= entries[:, None] * (1.0 + tp[:, None])), first_hit(lows <= entries[:, None] * (1.0 - sl[:, None]))
    short_tp, short_sl = first_hit(lows <= entries[:, None] * (1.0 - tp[:, None])), first_hit(highs >= entries[:, None] * (1.0 + sl[:, None]))
    long_timeout, short_timeout = closes[:, -1] / entries - 1.0, 1.0 - closes[:, -1] / entries
    long_move = np.where(long_tp < long_sl, tp, np.where(long_sl < HORIZON, -sl, long_timeout))
    short_move = np.where(short_tp < short_sl, tp, np.where(short_sl < HORIZON, -sl, short_timeout))
    long_net, short_net = long_move - COST, short_move - COST
    labels = np.zeros(n, dtype=np.int8)
    labels[(long_net >= NET_FLOOR) & (long_net - short_net >= SIDE_EDGE)] = 1
    labels[(short_net >= NET_FLOOR) & (short_net - long_net >= SIDE_EDGE)] = 2
    long_reason = np.where(long_tp < long_sl, "tp", np.where(long_sl < HORIZON, "sl", "timeout"))
    short_reason = np.where(short_tp < short_sl, "tp", np.where(short_sl < HORIZON, "sl", "timeout"))
    long_exit, short_exit = np.where(long_tp < long_sl, long_tp, np.minimum(long_sl, HORIZON - 1)), np.where(short_tp < short_sl, short_tp, np.minimum(short_sl, HORIZON - 1))
    chosen_exit = np.where(labels == 1, long_exit, np.where(labels == 2, short_exit, -1))
    chosen_reason = np.where(labels == 1, long_reason, np.where(labels == 2, short_reason, "flat"))
    exit_ts = frame.timestamp.to_numpy()[decisions + 1 + np.maximum(chosen_exit, 0)]
    exit_ts[chosen_exit < 0] = np.datetime64("NaT")
    result = pd.DataFrame({"decision_index": decisions[valid], "decision_timestamp": frame.timestamp.iloc[:n].to_numpy()[valid], "entry_timestamp": frame.timestamp.iloc[1 : n + 1].to_numpy()[valid], "exit_timestamp": exit_ts[valid], "label": labels[valid], "label_name": np.array(["FLAT", "LONG", "SHORT"])[labels[valid]], "exit_reason": chosen_reason[valid], "atr_pct": atr[:n][valid], "tp_move": tp[valid], "sl_move": sl[valid], "long_net": long_net[valid], "short_net": short_net[valid], "long_exit_reason": long_reason[valid], "short_exit_reason": short_reason[valid]})
    result.to_parquet(OUT / "labels.parquet", index=False)
    counts = result.label_name.value_counts().reindex(["FLAT", "LONG", "SHORT"], fill_value=0)
    report = {"label_contract": {"entry": "decision t+1 open", "horizon_bars": HORIZON, "tp_price_move": "max(2.00%, 8*ATR14)", "sl_price_move": "max(1.00%, 4*ATR14)", "intrabar_tie": "SL wins", "roundtrip_cost_rate": COST, "long_short_label": "net return >=0.40% and side advantage >=0.50%p", "future_path_used_only_as_training_target": True}, "rows": int(len(result)), "class_counts": {key: int(value) for key, value in counts.items()}, "class_ratios": {key: float(value / len(result)) for key, value in counts.items()}, "barrier_percent": {"tp_p50": float(result.tp_move.median() * 100), "sl_p50": float(result.sl_move.median() * 100)}, "net_return": {"long_p50": float(result.long_net.median()), "short_p50": float(result.short_net.median())}, "contracts": {"trade_ledgers_used_as_input": False, "future_rows_used_for_entry": False}}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
