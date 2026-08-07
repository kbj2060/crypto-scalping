"""Build BTC labels for the causal Tau1-trend continuation contract.

Tau1's trend scan is a causal candidate gate, not the supervised target.  The
target is whether that candidate side earns a buffered net return under the
same trailing execution contract used for later evaluation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from train_eval_btc_110branch_causal_20260804 import COST, load_frame  # noqa: E402

OUT = ROOT / "tmp/btc_tau1_continuation_labels_20260805"
WINDOWS = (3, 6, 12, 24, 36, 48)
THRESHOLD_LOOKBACK_HOURS, THRESHOLD_MIN_HOURS, THRESHOLD_QUANTILE = 720, 168, 0.60
HARD_STOP_ATR, TRAIL_ARM_ATR, TRAIL_GIVEBACK_ATR = 4.0, 4.0, 8.0
MAX_HOLD_HOURS, NET_PRICE_RETURN_FLOOR = 288, 0.010


def trend_scan(log_close: np.ndarray, windows: tuple[int, ...] = WINDOWS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal max-|t| linear trend scan, where every fitted window ends at t."""
    n = len(log_close)
    best_t, best_l, best_beta = np.zeros(n), np.full(n, -1, dtype=np.int16), np.zeros(n)
    for t in range(n):
        for length in windows:
            if t + 1 < length:
                continue
            y = log_close[t - length + 1 : t + 1]
            x = np.arange(length, dtype=float)
            xc, yc = x - x.mean(), y - y.mean()
            denom = float(np.dot(xc, xc))
            beta = float(np.dot(xc, yc) / denom)
            residual = y - (yc.mean() + beta * xc)
            sigma2 = float(np.dot(residual, residual) / max(length - 2, 1))
            t_value = beta / max(np.sqrt(sigma2 / denom), 1e-12)
            if abs(t_value) > abs(best_t[t]):
                best_t[t], best_l[t], best_beta[t] = t_value, length, beta
    return best_t, best_l, best_beta


def online_gate(abs_t: pd.Series) -> np.ndarray:
    """Use only completed earlier hourly t-values for the percentile threshold."""
    threshold = abs_t.shift(1).rolling(THRESHOLD_LOOKBACK_HOURS, min_periods=THRESHOLD_MIN_HOURS).quantile(THRESHOLD_QUANTILE)
    return (abs_t >= threshold).fillna(False).to_numpy(bool)


def atr_hourly(frame: pd.DataFrame, window: int = 24) -> np.ndarray:
    high, low, close = (frame[c].to_numpy(float) for c in ("high", "low", "close"))
    previous = np.r_[close[0], close[:-1]]
    true_range = np.maximum.reduce([high - low, np.abs(high - previous), np.abs(low - previous)])
    return pd.Series(true_range / close).rolling(window, min_periods=window).mean().to_numpy(float)


def trailing_outcome(*, side: int, entry: float, high: np.ndarray, low: np.ndarray, close: np.ndarray, atr_pct: float) -> tuple[float, str, int]:
    """Conservative intrabar hard-stop/trailing replay; result is raw price move."""
    hard, arm, trail = HARD_STOP_ATR * atr_pct, TRAIL_ARM_ATR * atr_pct, TRAIL_GIVEBACK_ATR * atr_pct
    if side > 0:
        peak, armed = entry, False
        for offset, (bar_high, bar_low) in enumerate(zip(high, low)):
            if bar_low <= entry * (1.0 - hard):
                return -hard, "hard_stop", offset
            peak = max(peak, float(bar_high))
            armed = armed or peak >= entry * (1.0 + arm)
            if armed and bar_low <= peak * (1.0 - trail):
                return peak * (1.0 - trail) / entry - 1.0, "trailing_stop", offset
        return float(close[-1] / entry - 1.0), "timeout", len(close) - 1
    trough, armed = entry, False
    for offset, (bar_high, bar_low) in enumerate(zip(high, low)):
        if bar_high >= entry * (1.0 + hard):
            return -hard, "hard_stop", offset
        trough = min(trough, float(bar_low))
        armed = armed or trough <= entry * (1.0 - arm)
        if armed and bar_high >= trough * (1.0 + trail):
            return 1.0 - trough * (1.0 + trail) / entry, "trailing_stop", offset
    return float(1.0 - close[-1] / entry), "timeout", len(close) - 1


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    frame, _ = load_frame()
    bars = frame.set_index("timestamp")[["open", "high", "low", "close"]].resample("1h", label="left", closed="left").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().reset_index()
    t_value, best_window, beta = trend_scan(np.log(bars.close.to_numpy(float)))
    abs_t = pd.Series(np.abs(t_value))
    online_threshold = abs_t.shift(1).rolling(THRESHOLD_LOOKBACK_HOURS, min_periods=THRESHOLD_MIN_HOURS).quantile(THRESHOLD_QUANTILE)
    gate = (abs_t >= online_threshold).fillna(False).to_numpy(bool)
    atr = atr_hourly(bars)
    five_ts = pd.DatetimeIndex(frame.timestamp)
    open5, high5, low5, close5 = (frame[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    rows: list[dict] = []
    for hour_i in np.flatnonzero(gate & np.isfinite(atr) & (best_window > 0)):
        decision_ts = pd.Timestamp(bars.timestamp.iloc[hour_i]) + pd.Timedelta(hours=1)
        decision_i = int(five_ts.searchsorted(decision_ts))
        if decision_i >= len(frame) or five_ts[decision_i] != decision_ts:
            # The final resampled hour can be incomplete relative to the 5m tape.
            # It has neither a next-open entry nor a complete future target.
            continue
        entry_i = decision_i + 1
        hold_bars = min(MAX_HOLD_HOURS, 4 * int(best_window[hour_i])) * 12
        final_i = entry_i + hold_bars - 1
        if final_i >= len(frame):
            continue
        side = 1 if beta[hour_i] > 0 else -1
        price_move, reason, exit_offset = trailing_outcome(side=side, entry=float(open5[entry_i]), high=high5[entry_i : final_i + 1], low=low5[entry_i : final_i + 1], close=close5[entry_i : final_i + 1], atr_pct=float(atr[hour_i]))
        net_price_return = price_move - COST
        label = 1 if side > 0 and net_price_return >= NET_PRICE_RETURN_FLOOR else 2 if side < 0 and net_price_return >= NET_PRICE_RETURN_FLOOR else 0
        rows.append({"decision_timestamp": decision_ts, "entry_timestamp": five_ts[entry_i], "side_candidate": side, "label": label, "label_name": ("FLAT", "LONG", "SHORT")[label], "trend_t_value": float(t_value[hour_i]), "trend_window_hours": int(best_window[hour_i]), "trend_beta": float(beta[hour_i]), "online_threshold": float(online_threshold.iloc[hour_i]), "atr_pct_1h": float(atr[hour_i]), "max_hold_bars": int(hold_bars), "exit_timestamp": five_ts[entry_i + exit_offset], "exit_reason": reason, "gross_price_move": price_move, "net_price_return": net_price_return})
    labels = pd.DataFrame(rows)
    labels.to_parquet(OUT / "labels.parquet", index=False)
    counts = labels.label_name.value_counts().reindex(["FLAT", "LONG", "SHORT"], fill_value=0)
    report = {"label_contract": {"candidate": "causal backward 1h max-|t| trend scan", "windows_hours": list(WINDOWS), "gate": "online prior-720h |t| 60th percentile; minimum 168h", "entry": "completed-hour timestamp +1h decision then next 5m open", "side": "sign of selected regression beta", "hard_stop": "4.0 * 1h ATR", "trailing": "arm at +4.0 ATR; giveback 8.0 ATR", "max_hold_hours": "min(288, 4*selected_window_hours)", "label": "LONG/SHORT only when candidate-side exact net price return >= 1.00%; otherwise FLAT", "roundtrip_cost_rate": COST}, "rows": int(len(labels)), "class_counts": {name: int(value) for name, value in counts.items()}, "class_ratios": {name: float(value / max(len(labels), 1)) for name, value in counts.items()}, "net_price_return": {"all_candidate_p50": float(labels.net_price_return.median()), "positive_label_p50": float(labels.loc[labels.label != 0, "net_price_return"].median()) if (labels.label != 0).any() else None}, "hold_hours": {"p50": float(labels.max_hold_bars.median() / 12), "p90": float(labels.max_hold_bars.quantile(.9) / 12)}, "exit_reasons": {str(key): int(value) for key, value in labels.exit_reason.value_counts().items()}, "contracts": {"future_path_used_only_as_training_target": True, "trend_gate_uses_only_current_and_past_rows": True, "trade_ledgers_used_as_input": False, "future_rows_used_for_entry": False}}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
