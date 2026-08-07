"""Pure label-quality report, NO modeling: what's the economic ceiling of the causal triple-barrier
label itself (scripts/build_btc_5m_tripbarrier_tradeoutcome_labels_20260806.py) if a trader had
PERFECT foresight of it -- i.e. traded exactly the label's own side whenever it said LONG/SHORT?

This is the natural "teacher quality" check: the label's own argmax IS the perfect-information
target the model is being distilled toward, so this backtest (using the true label as the entry
signal instead of any model's prediction) is the ceiling the model's actual backtest performance
(win rate 35.6% OOS fresh-entry, gross pre-cost edge ~+0.054%/trade) should be measured against.

Same simulate_single_position mechanics as scripts/backtest_btc_tripbarrier_model_20260806.py
(same TP/SL vol basis, horizon, margin/leverage/cost) so the comparison is apples-to-apples.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.causal_futures_backtest import simulate_single_position  # noqa: E402

PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
LABEL_PATH = ROOT / "data/splits/year_oos/btc_5m_tripbarrier_tradeoutcome_labels_20260806.parquet"
OUT_DIR = ROOT / "tmp/btc_deepfeat_tripbarrier_backtest_20260806"

CUMRET_BARS, VOL_LOOKBACK, TP_MULT, SL_MULT, HORIZON_BARS = 12, 288, 2.5, 1.2, 288
MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE = 0.30, 3.0, 0.0010

VAL_START, VAL_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-12-31")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31")


def _fresh_entry_mask(side_state: np.ndarray) -> np.ndarray:
    fresh = np.zeros(len(side_state), dtype=bool)
    fresh[0] = side_state[0] != 0
    fresh[1:] = (side_state[1:] != 0) & (side_state[1:] != side_state[:-1])
    return fresh


def _run(row_idx, side_state_full, tp_moves, sl_moves, panel, fresh_only: bool):
    side_state = side_state_full[row_idx]
    mask = _fresh_entry_mask(side_state) if fresh_only else (side_state != 0)
    idx = row_idx[mask]
    side = side_state[mask]
    tp, sl = tp_moves[idx], sl_moves[idx]
    finite = np.isfinite(tp) & np.isfinite(sl)
    idx, side, tp, sl = idx[finite], side[finite], tp[finite], sl[finite]
    return simulate_single_position(
        timestamps=panel["timestamp"], open_px=panel["open"].to_numpy(dtype=np.float64),
        high=panel["high"].to_numpy(dtype=np.float64), low=panel["low"].to_numpy(dtype=np.float64),
        close=panel["close"].to_numpy(dtype=np.float64), decision_indices=idx, scores=side.astype(np.float64),
        tp_moves=tp, sl_moves=sl, upper_threshold=0.0, lower_threshold=0.0, horizon_bars=HORIZON_BARS,
        margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
    )


def _summarize(result, split, mode) -> dict:
    ledger = result.ledger
    n = len(ledger)
    if n == 0:
        return {"split": split, "mode": mode, "n_trades": 0}
    equity = result.equity
    running_max = np.maximum.accumulate(equity)
    mdd_pct = float(((equity - running_max) / running_max).min() * 100.0)
    return {
        "split": split, "mode": mode, "n_trades": n,
        "sum_ret_pct": float(ledger["trade_return"].sum() * 100.0),
        "mean_ret_pct": float(ledger["trade_return"].mean() * 100.0),
        "win_rate": float((ledger["trade_return"] > 0).mean()),
        "mdd_pct": mdd_pct, "final_equity": float(equity[-1]),
        "exit_reasons": ledger["reason"].value_counts().to_dict(),
    }


def main() -> int:
    panel = pd.read_parquet(PANEL_PATH, columns=["timestamp", "open", "high", "low", "close"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    close = panel["close"].to_numpy(dtype=np.float64)
    n = len(panel)

    labels = pd.read_parquet(LABEL_PATH)
    labels = labels.sort_values("timestamp").reset_index(drop=True)
    if not (panel["timestamp"].to_numpy() == labels["timestamp"].to_numpy()).all():
        raise RuntimeError("panel/label timestamp misalignment")
    true_hard = labels["trade_outcome_action"].to_numpy()
    side_state_true = np.where(true_hard == 1, 1, np.where(true_hard == 2, -1, 0))

    log_ret_1bar = np.diff(np.log(np.clip(close, 1e-9, None)), prepend=np.log(close[0]))
    cumret = pd.Series(log_ret_1bar).rolling(CUMRET_BARS).sum().to_numpy()
    vol = pd.Series(cumret).rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std().to_numpy()
    tp_moves_all, sl_moves_all = TP_MULT * vol, SL_MULT * vol

    ts = panel["timestamp"]
    splits = {
        "val": np.flatnonzero((ts >= VAL_START).to_numpy() & (ts <= VAL_END).to_numpy()),
        "oos": np.flatnonzero((ts >= OOS_START).to_numpy() & (ts <= OOS_END).to_numpy()),
    }

    results = []
    for split, row_idx in splits.items():
        counts = pd.Series(true_hard[row_idx]).value_counts(normalize=True).sort_index()
        print(f"{split} label balance: CASH={counts.get(0,0):.1%} LONG={counts.get(1,0):.1%} SHORT={counts.get(2,0):.1%}")
        for fresh_only, mode in [(False, "continuous_oracle"), (True, "fresh_entry_oracle")]:
            r = _run(row_idx, side_state_true, tp_moves_all, sl_moves_all, panel, fresh_only)
            summary = _summarize(r, split, mode)
            results.append(summary)
            print(json.dumps(summary, default=str))

    (OUT_DIR / "oracle_ceiling_summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"wrote {OUT_DIR}/oracle_ceiling_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
