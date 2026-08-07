"""Apply [[project-btc-oracle-label-selection-protocol-20260806]] step 3 (oracle/perfect-foresight
backtest through the SAME corrected TP/SL mechanics) to every distinct label methodology found in
this repo's history (see the labeling-methodology survey), plus zigzag. trend-scan was already
validated separately (build_btc_5m_trendscan_oracle_label_20260806.py: OOS win rate 29.3%, FAILED
the ceiling check) -- this script builds and oracle-validates the remaining candidates and prints
one consolidated comparison table against the already-validated triple-barrier (100% win rate,
44.3x OOS equity) and trend-scan results.

Candidates built here:
- zigzag: reuses the existing build_btc_5m_zigzag_and_pivot_labels_20260806.py output.
- cusum_tb: causal CUSUM event filter (ATR-scaled threshold, symmetric two-sided) -- direction is
  the sign of the triggering cumulative move at event bars; CASH elsewhere.
- directional_change: fixed-percent-threshold trend segmentation, structurally like zigzag's pivot
  state machine but WITHOUT zigzag's retrospective backfill -- label[i] is whatever trend state the
  causal single-pass state machine is in AT bar i (no waiting for pivot confirmation), which is
  what genuinely distinguishes DC from zigzag in the literature.
- path_utility: MFE - k*MAE - cost per side over the full horizon (k=1.1, matching this repo's
  historical MAE_PENALTY convention), whichever side clears a positive-utility floor wins.
- optimal_exit: proxy for "deep optimal stopping" -- under PERFECT foresight the optimal stopping
  policy is trivially "exit at the single best point in the horizon" (backward induction is only
  needed when you don't have the future path, i.e. for the live/predictive model, not for
  constructing the oracle target it's being distilled toward). Whichever side's best-achievable
  net return is higher (and positive) wins.
- meta_label: trend-scan AND triple-barrier must AGREE on direction, else CASH -- a concrete
  instantiation of "meta-labeling" (a secondary filter on a primary signal) using two already-built
  candidates, and directly tests the user's original "ensemble multiple oracle labels" idea.

All candidates trade through core/causal_futures_backtest.simulate_single_position with the SAME
corrected TP/SL basis (12-bar cumulative-return dispersion, 288-bar lookback, TP_MULT=2.5/
SL_MULT=1.2, horizon=288, fresh-entry gate, margin_fraction=0.30/leverage=3/cost=10bps) as the
validated triple-barrier label.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numba
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.causal_futures_backtest import simulate_single_position  # noqa: E402

PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
ZIGZAG_LABEL_PATH = ROOT / "data/splits/year_oos/btc_5m_zigzag_labels_20260806.parquet"
TRENDSCAN_LABEL_PATH = ROOT / "data/splits/year_oos/btc_5m_trendscan_oracle_labels_20260806.parquet"
TRIPBARRIER_LABEL_PATH = ROOT / "data/splits/year_oos/btc_5m_tripbarrier_tradeoutcome_labels_flatsmooth_20260806.parquet"
OUT_DIR = ROOT / "tmp/btc_oracle_label_logic_comparison_20260806"

CUMRET_BARS, VOL_LOOKBACK, TP_MULT, SL_MULT, HORIZON_BARS = 12, 288, 2.5, 1.2, 288
MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE = 0.30, 3.0, 0.0010
VAL_START, VAL_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-12-31")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31")

CUSUM_MIN_PCT = 0.006
CUSUM_ATR_MULT = 1.0
DC_THRESHOLD_PCT = 0.012
PATH_UTIL_K = 1.1
PATH_UTIL_COST = ROUNDTRIP_COST_RATE


@numba.njit(cache=True)
def _atr_pct(high, low, close, window):
    n = len(close)
    tr = np.empty(n, dtype=np.float64)
    prev = close[0]
    for i in range(n):
        a = high[i] - low[i]
        b = abs(high[i] - prev)
        c = abs(low[i] - prev)
        tr[i] = max(a, max(b, c))
        prev = close[i]
    atr = np.empty(n, dtype=np.float64)
    alpha = 2.0 / (window + 1.0)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = atr[i] / max(close[i], 1e-12)
    return out


@numba.njit(cache=True)
def _cusum_tb(close, atr_pct, min_pct, atr_mult):
    n = len(close)
    label = np.zeros(n, dtype=np.int8)
    s_pos, s_neg = 0.0, 0.0
    for i in range(1, n):
        ret = np.log(close[i] / close[i - 1])
        s_pos = max(0.0, s_pos + ret)
        s_neg = min(0.0, s_neg + ret)
        thr = max(min_pct, atr_pct[i] * atr_mult)
        if s_pos > thr:
            label[i] = 1
            s_pos, s_neg = 0.0, 0.0
        elif s_neg < -thr:
            label[i] = 2
            s_pos, s_neg = 0.0, 0.0
    return label


@numba.njit(cache=True)
def _directional_change(close, threshold_pct):
    n = len(close)
    label = np.zeros(n, dtype=np.int8)  # running causal trend state, no backfill
    trend = 0
    ext_price = close[0]
    for i in range(1, n):
        price = close[i]
        if trend == 0:
            if price >= ext_price * (1.0 + threshold_pct):
                trend = 1
                ext_price = price
            elif price <= ext_price * (1.0 - threshold_pct):
                trend = -1
                ext_price = price
            else:
                if price > ext_price:
                    ext_price = price
        elif trend == 1:
            if price > ext_price:
                ext_price = price
            elif price <= ext_price * (1.0 - threshold_pct):
                trend = -1
                ext_price = price
        else:
            if price < ext_price:
                ext_price = price
            elif price >= ext_price * (1.0 + threshold_pct):
                trend = 1
                ext_price = price
        label[i] = 1 if trend == 1 else (2 if trend == -1 else 0)
    return label


@numba.njit(cache=True)
def _path_based_labels(open_, high, low, horizon, k, cost):
    n = len(open_)
    path_util_label = np.zeros(n, dtype=np.int8)
    optimal_exit_label = np.zeros(n, dtype=np.int8)
    for i in range(n - 1):
        entry_i = i + 1
        entry = open_[entry_i]
        final_i = entry_i + horizon - 1
        if final_i >= n:
            final_i = n - 1
        if final_i < entry_i:
            continue
        max_high = high[entry_i]
        min_low = low[entry_i]
        for j in range(entry_i, final_i + 1):
            if high[j] > max_high:
                max_high = high[j]
            if low[j] < min_low:
                min_low = low[j]
        mfe_long = (max_high - entry) / entry
        mae_long = (entry - min_low) / entry
        mfe_short = (entry - min_low) / entry
        mae_short = (max_high - entry) / entry

        util_long = mfe_long - k * mae_long - cost
        util_short = mfe_short - k * mae_short - cost
        if util_long > util_short and util_long > 0.0:
            path_util_label[i] = 1
        elif util_short > util_long and util_short > 0.0:
            path_util_label[i] = 2

        net_long = mfe_long - cost
        net_short = mfe_short - cost
        if net_long > net_short and net_long > 0.0:
            optimal_exit_label[i] = 1
        elif net_short > net_long and net_short > 0.0:
            optimal_exit_label[i] = 2
    return path_util_label, optimal_exit_label


def _fresh_entry_mask(side_state: np.ndarray) -> np.ndarray:
    fresh = np.zeros(len(side_state), dtype=bool)
    fresh[0] = side_state[0] != 0
    fresh[1:] = (side_state[1:] != 0) & (side_state[1:] != side_state[:-1])
    return fresh


def _run(row_idx, side_state_full, tp_moves, sl_moves, panel):
    side_state = side_state_full[row_idx]
    fresh = _fresh_entry_mask(side_state)
    idx = row_idx[fresh]
    side = side_state[fresh]
    tp, sl = tp_moves[idx], sl_moves[idx]
    finite = np.isfinite(tp) & np.isfinite(sl)
    idx, side, tp, sl = idx[finite], side[finite], tp[finite], sl[finite]
    if len(idx) == 0:
        return None
    return simulate_single_position(
        timestamps=panel["timestamp"], open_px=panel["open"].to_numpy(dtype=np.float64),
        high=panel["high"].to_numpy(dtype=np.float64), low=panel["low"].to_numpy(dtype=np.float64),
        close=panel["close"].to_numpy(dtype=np.float64), decision_indices=idx, scores=side.astype(np.float64),
        tp_moves=tp, sl_moves=sl, upper_threshold=0.0, lower_threshold=0.0, horizon_bars=HORIZON_BARS,
        margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
    )


def _summarize(result) -> dict:
    if result is None or len(result.ledger) == 0:
        return {"n_trades": 0, "win_rate": None, "sum_ret_pct": None, "final_equity": None, "mdd_pct": None}
    ledger = result.ledger
    equity = result.equity
    running_max = np.maximum.accumulate(equity)
    mdd_pct = float(((equity - running_max) / running_max).min() * 100.0)
    return {
        "n_trades": int(len(ledger)),
        "win_rate": float((ledger["trade_return"] > 0).mean()),
        "sum_ret_pct": float(ledger["trade_return"].sum() * 100.0),
        "final_equity": float(equity[-1]),
        "mdd_pct": mdd_pct,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(PANEL_PATH, columns=["timestamp", "open", "high", "low", "close"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    open_ = panel["open"].to_numpy(dtype=np.float64)
    high = panel["high"].to_numpy(dtype=np.float64)
    low = panel["low"].to_numpy(dtype=np.float64)
    close = panel["close"].to_numpy(dtype=np.float64)
    n = len(panel)

    log_ret_1bar = np.diff(np.log(np.clip(close, 1e-9, None)), prepend=np.log(close[0]))
    cumret = pd.Series(log_ret_1bar).rolling(CUMRET_BARS).sum().to_numpy()
    vol = pd.Series(cumret).rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std().to_numpy()
    tp_moves_all, sl_moves_all = TP_MULT * vol, SL_MULT * vol

    atr_pct = _atr_pct(high, low, close, 14)

    candidates: dict[str, np.ndarray] = {}

    zz = pd.read_parquet(ZIGZAG_LABEL_PATH, columns=["timestamp", "zigzag_action"]).sort_values("timestamp").reset_index(drop=True)
    assert (zz["timestamp"].to_numpy() == panel["timestamp"].to_numpy()).all()
    candidates["zigzag"] = zz["zigzag_action"].to_numpy()

    ts_lab = pd.read_parquet(TRENDSCAN_LABEL_PATH, columns=["timestamp", "trendscan_action"]).sort_values("timestamp").reset_index(drop=True)
    assert (ts_lab["timestamp"].to_numpy() == panel["timestamp"].to_numpy()).all()
    candidates["trend_scan"] = ts_lab["trendscan_action"].to_numpy()

    tb = pd.read_parquet(TRIPBARRIER_LABEL_PATH, columns=["timestamp", "trade_outcome_action"]).sort_values("timestamp").reset_index(drop=True)
    assert (tb["timestamp"].to_numpy() == panel["timestamp"].to_numpy()).all()
    candidates["triple_barrier"] = tb["trade_outcome_action"].to_numpy()

    candidates["cusum_tb"] = _cusum_tb(close, atr_pct, CUSUM_MIN_PCT, CUSUM_ATR_MULT)
    candidates["directional_change"] = _directional_change(close, DC_THRESHOLD_PCT)
    path_util, optimal_exit = _path_based_labels(open_, high, low, HORIZON_BARS, PATH_UTIL_K, PATH_UTIL_COST)
    candidates["path_utility"] = path_util
    candidates["optimal_exit"] = optimal_exit

    meta_label = np.zeros(n, dtype=np.int8)
    ts_arr, tb_arr = candidates["trend_scan"], candidates["triple_barrier"]
    meta_label[(ts_arr == 1) & (tb_arr == 1)] = 1
    meta_label[(ts_arr == 2) & (tb_arr == 2)] = 2
    candidates["meta_label_agreement"] = meta_label

    ts = panel["timestamp"]
    splits = {
        "val": np.flatnonzero((ts >= VAL_START).to_numpy() & (ts <= VAL_END).to_numpy()),
        "oos": np.flatnonzero((ts >= OOS_START).to_numpy() & (ts <= OOS_END).to_numpy()),
    }

    all_results = []
    for name, action in candidates.items():
        side_state_full = np.where(action == 1, 1, np.where(action == 2, -1, 0))
        for split, row_idx in splits.items():
            result = _run(row_idx, side_state_full, tp_moves_all, sl_moves_all, panel)
            summary = _summarize(result)
            summary["label"] = name
            summary["split"] = split
            all_results.append(summary)

    print(f"{'label':<22}{'split':<6}{'n_trades':>9}{'win_rate':>10}{'sum_ret%':>10}{'equity':>9}{'mdd%':>8}")
    for r in all_results:
        wr = f"{r['win_rate']:.3f}" if r["win_rate"] is not None else "n/a"
        sr = f"{r['sum_ret_pct']:.1f}" if r["sum_ret_pct"] is not None else "n/a"
        eq = f"{r['final_equity']:.2f}x" if r["final_equity"] is not None else "n/a"
        mdd = f"{r['mdd_pct']:.1f}" if r["mdd_pct"] is not None else "n/a"
        print(f"{r['label']:<22}{r['split']:<6}{r['n_trades']:>9}{wr:>10}{sr:>10}{eq:>9}{mdd:>8}")

    (OUT_DIR / "oracle_comparison_summary.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT_DIR}/oracle_comparison_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
