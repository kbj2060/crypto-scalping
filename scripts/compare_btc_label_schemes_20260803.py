"""
Compare BTC labeling schemes side-by-side on the new 5m+1h+metrics4 feature
frame (data/splits/year_oos/btc_features_5m1h_mtf_2024_2026.parquet), as input
to picking a label for the new BTC architecture.

Existing schemes (logic imported unchanged from their production scripts, not
reimplemented, to avoid silently drifting from the real definitions):
  - h48qual / triple-barrier (h48_conservative barrier config)
  - zigzag (v2 defaults)
  - trend-scan (already merged into the frame as mtf1h_ts_action)

New candidates for this comparison:
  - cusum_tb: CUSUM-filtered event sampling (only label bars where cumulative
    log-return since the last event exceeds an ATR-scaled threshold), then
    apply the same h48_conservative triple-barrier at those event bars only.
  - dc_1h: causal Directional-Change intrinsic-time segmentation on the 1h
    close series, threshold ATR-scaled (median mtf1h_atr_pct * multiplier).
  - confirmed_trend: ensemble label = h48_conservative triple-barrier action,
    kept only where it agrees in direction with the (already causal, already
    merged) 1h trend-scan action; disagreements are relabeled CASH.

This script only computes label statistics (frequency, direction split,
realized return, win rate, hold time) for a first-pass comparison. It is NOT a
Fresh-Forward validation and produces no PnL/promotion claim.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_omega1_2_triple_barrier_labels_btc_20260708 import (  # noqa: E402
    _atr_price_move,
    _reason_and_return,
)
from build_zigzag_action_labels_v2_20260604 import build_zigzag_action_labels  # noqa: E402

FRAME_PATH = ROOT / "data/splits/year_oos/btc_features_5m1h_mtf_2024_2026.parquet"
OUT_CSV = ROOT / "tmp/btc_label_scheme_comparison_20260803.csv"

TB_HORIZON = 48
TB_TP_MULT, TB_SL_MULT = 1.2, 0.8
TB_MIN_TP, TB_MIN_SL = 0.006, 0.004


def build_triple_barrier(frame: pd.DataFrame, candidate_idx: np.ndarray | None = None) -> pd.DataFrame:
    n = len(frame)
    open_px = frame["open"].to_numpy(dtype=np.float64)
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)
    atr = _atr_price_move(frame)
    ts = frame["timestamp"]

    idxs = candidate_idx if candidate_idx is not None else np.arange(0, n - TB_HORIZON - 2)
    rows = []
    for i in idxs:
        entry_i = i + 1
        end_i = entry_i + TB_HORIZON
        if end_i + 1 > n:
            continue
        entry = float(open_px[entry_i])
        vol = float(atr[i])
        tp_move = max(TB_MIN_TP, TB_TP_MULT * vol)
        sl_move = max(TB_MIN_SL, TB_SL_MULT * vol)
        fh, fl, fc = high[entry_i:end_i + 1], low[entry_i:end_i + 1], close[entry_i:end_i + 1]
        long_ret, long_reason, _, _, long_bars = _reason_and_return(
            side=1, entry=entry, future_high=fh, future_low=fl, future_close=fc,
            tp_move=tp_move, sl_move=sl_move)
        short_ret, short_reason, _, _, short_bars = _reason_and_return(
            side=-1, entry=entry, future_high=fh, future_low=fl, future_close=fc,
            tp_move=tp_move, sl_move=sl_move)
        long_q = long_ret - 0.0007 - 0.003 * int(long_reason == "sl")
        short_q = short_ret - 0.0007 - 0.003 * int(short_reason == "sl")
        if long_q > 0 and long_q >= short_q:
            action, ret, reason, bars = 1, long_ret, long_reason, long_bars
        elif short_q > 0:
            action, ret, reason, bars = 2, short_ret, short_reason, short_bars
        else:
            action, ret, reason, bars = 0, max(long_q, short_q), "none", 0
        rows.append({"i": i, "timestamp": ts.iloc[i], "action": action, "ret": ret,
                      "reason": reason, "bars": bars})
    return pd.DataFrame(rows)


def cusum_events(frame: pd.DataFrame, atr: np.ndarray, mult: float = 2.0) -> np.ndarray:
    close = frame["close"].to_numpy(dtype=np.float64)
    logret = np.diff(np.log(close), prepend=np.log(close[0]))
    s_pos = s_neg = 0.0
    events = []
    for i in range(1, len(close)):
        thresh = max(float(atr[i]), 0.001) * mult
        s_pos = max(0.0, s_pos + logret[i])
        s_neg = min(0.0, s_neg + logret[i])
        if s_pos > thresh or s_neg < -thresh:
            events.append(i)
            s_pos = s_neg = 0.0
    return np.array(events, dtype=np.int64)


def directional_change_1h(overlay_1h: pd.DataFrame, mult: float = 1.5) -> pd.DataFrame:
    close = overlay_1h["close"].to_numpy(dtype=np.float64)
    atr_pct = overlay_1h["mtf1h_atr_pct"].to_numpy(dtype=np.float64)
    theta = float(np.nanmedian(atr_pct)) * mult
    n = len(close)
    direction = 0  # 0 = undecided, 1 = up-run, -1 = down-run
    ext_price, ext_idx = close[0], 0
    events = []  # (confirm_idx, direction_confirmed)
    for i in range(1, n):
        price = close[i]
        if direction <= 0:
            if price < ext_price or direction == 0:
                if direction == 0 and price >= ext_price * (1 + theta):
                    events.append((i, 1))
                    direction, ext_price, ext_idx = 1, price, i
                    continue
                if price < ext_price:
                    ext_price, ext_idx = price, i
            elif price >= ext_price * (1 + theta):
                events.append((i, 1))
                direction, ext_price, ext_idx = 1, price, i
        elif direction >= 0:
            if price > ext_price:
                ext_price, ext_idx = price, i
            elif price <= ext_price * (1 - theta):
                events.append((i, -1))
                direction, ext_price, ext_idx = -1, price, i
    return pd.DataFrame(events, columns=["i", "direction"])


def summarize(name: str, n_total: int, bars_per_year: float, action: np.ndarray,
              ret: np.ndarray, reason: np.ndarray, bars: np.ndarray) -> dict:
    actionable = action != 0
    n_act = int(actionable.sum())
    resolved = np.isin(reason, ["tp", "sl"]) & actionable
    win = (reason == "tp") & actionable
    return {
        "scheme": name,
        "n_actionable": n_act,
        "pct_of_bars": round(100 * n_act / max(n_total, 1), 3),
        "implied_events_per_year": round(n_act / max(n_total, 1) * bars_per_year, 1),
        "pct_long": round(100 * (action == 1).sum() / max(n_act, 1), 1),
        "pct_short": round(100 * (action == 2).sum() / max(n_act, 1), 1),
        "win_rate_tp_vs_sl": round(100 * win.sum() / max(resolved.sum(), 1), 1),
        "mean_ret_pct": round(100 * np.nanmean(ret[actionable]) if n_act else 0.0, 3),
        "mean_hold_bars": round(float(np.nanmean(bars[actionable])) if n_act else 0.0, 1),
    }


def main():
    frame = pd.read_parquet(FRAME_PATH)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    n = len(frame)
    results = []

    # 1) h48qual / triple-barrier baseline (every bar)
    tb = build_triple_barrier(frame)
    results.append(summarize("h48qual_triple_barrier", n, 288 * 365,
                              tb["action"].to_numpy(), tb["ret"].to_numpy(),
                              tb["reason"].to_numpy(), tb["bars"].to_numpy()))

    # 2) zigzag (v2 defaults, same as production BTC config elsewhere in repo)
    zz = build_zigzag_action_labels(
        frame, min_reversal_pct=0.010, max_reversal_pct=0.018, min_wave_bars=8,
        transition_buffer=4, atr_window=14, atr_multiplier=1.0, mae_penalty=1.35,
        softmax_temperature=1.75, min_risk_floor=0.0010, min_edge_pct=0.0015,
        min_calmar=0.25, min_mfe_efficiency=0.45, min_phase=0.04, max_phase=0.82,
    )
    zz_action = zz["zigzag_action"].to_numpy()
    zz_ret = zz["zigzag_path_return"].to_numpy()
    zz_bars = zz["zigzag_wave_bars"].to_numpy()
    zz_reason = np.where(zz_ret > 0, "tp", np.where(zz_ret < 0, "sl", "none"))
    results.append(summarize("zigzag_v2", n, 288 * 365, zz_action, zz_ret, zz_reason, zz_bars))

    # 3) trend-scan (already merged, 1h resolution upsampled onto 5m rows)
    ts_action_5m = frame["mtf1h_ts_action"].fillna(0).to_numpy()
    # forward return proxy: realized close-to-close move over next 12 5m-bars (1h)
    close = frame["close"].to_numpy(dtype=np.float64)
    fwd_1h_ret = np.concatenate([(close[12:] / close[:-12] - 1.0), np.full(12, np.nan)])
    ts_ret = np.where(ts_action_5m == 2, -fwd_1h_ret, fwd_1h_ret)
    ts_reason = np.where(ts_ret > 0, "tp", np.where(ts_ret < 0, "sl", "none"))
    results.append(summarize("trend_scan_1h", n, 288 * 365, ts_action_5m, ts_ret, ts_reason,
                              np.full(n, 12.0)))

    # 4) NEW: CUSUM-filtered event triple-barrier
    atr = _atr_price_move(frame)
    events = cusum_events(frame, atr, mult=2.0)
    events = events[events < n - TB_HORIZON - 2]
    cusum_tb = build_triple_barrier(frame, candidate_idx=events)
    action_full = np.zeros(n); ret_full = np.zeros(n); reason_full = np.array(["none"] * n, dtype=object); bars_full = np.zeros(n)
    action_full[cusum_tb["i"]] = cusum_tb["action"]; ret_full[cusum_tb["i"]] = cusum_tb["ret"]
    reason_full[cusum_tb["i"]] = cusum_tb["reason"]; bars_full[cusum_tb["i"]] = cusum_tb["bars"]
    results.append(summarize("cusum_filtered_tb", n, 288 * 365, action_full, ret_full, reason_full, bars_full))

    # 5) NEW: Directional-Change intrinsic-time on 1h
    overlay_1h = frame[["mtf1h_ts_action", "mtf1h_atr_pct"]].copy()
    overlay_1h["close"] = close
    # reduce to unique 1h grid using the available_at merge already done (12 5m rows share 1h features);
    # take every 12th row as a proxy 1h series for DC purposes.
    hourly = overlay_1h.iloc[::12].reset_index(drop=True)
    dc = directional_change_1h(hourly, mult=1.5)
    dc_action_h = np.zeros(len(hourly))
    dc_action_h[dc["i"].to_numpy()] = np.where(dc["direction"] == 1, 1, 2)
    dc_action_5m = np.repeat(dc_action_h, 12)[:n]
    dc_ret = np.where(dc_action_5m == 2, -fwd_1h_ret, fwd_1h_ret)
    dc_reason = np.where(dc_ret > 0, "tp", np.where(dc_ret < 0, "sl", "none"))
    results.append(summarize("directional_change_1h", n, 288 * 365, dc_action_5m, dc_ret, dc_reason,
                              np.full(n, 12.0)))

    # 6) NEW: trend-scan x triple-barrier confirmed-trend ensemble
    idx = tb["i"].to_numpy()
    tb_action_arr = tb["action"].to_numpy()
    confirmed_action = np.where(
        (tb_action_arr == ts_action_5m[idx]) & (tb_action_arr != 0), tb_action_arr, 0)
    conf_action_full = np.zeros(n); conf_ret_full = np.zeros(n); conf_reason_full = np.array(["none"] * n, dtype=object); conf_bars_full = np.zeros(n)
    conf_action_full[idx] = confirmed_action
    conf_ret_full[idx] = tb["ret"].to_numpy()
    conf_reason_full[idx] = np.where(confirmed_action == 0, "none", tb["reason"].to_numpy())
    conf_bars_full[idx] = tb["bars"].to_numpy()
    results.append(summarize("confirmed_trend_ensemble", n, 288 * 365, conf_action_full, conf_ret_full,
                              conf_reason_full, conf_bars_full))

    out = pd.DataFrame(results)
    out.to_csv(OUT_CSV, index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
