"""Audit of the SOL survey's oracle logic, answering two challenges (2026-08-08):

(1) "Is the oracle logic broken?" -- run the IDENTICAL TB label + oracle replay on ETH raw 5m
    OHLC over the same VAL window. If ETH's oracle ceiling is also enormous, the ceiling is a
    property of perfect TP/SL-race foresight on any liquid asset, NOT a claim about available
    edge -- and "huge ceiling + zero 5m capture" coexists with ETH's WORKING live strategy
    (whose edge comes from a multi-day swing architecture, not 5m TB entries).

(2) "Is capture-zero a harness bug?" -- positive control: inject ONE deliberately leaked feature
    (the label's own race-conviction score difference) into the SOL LGBM cheap-gate pipeline.
    If the identical pipeline then prints a massively positive VAL, the training/replay
    machinery demonstrably converts real signal into PnL, so the observed capture-zero is a
    property of the data, not of the harness.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from core.causal_futures_backtest import simulate_single_position  # noqa: E402
from build_sol_5m_tripbarrier_tradeoutcome_labels_20260807 import (  # noqa: E402
    _triple_barrier_race, CUMRET_BARS, VOL_LOOKBACK, TP_MULT, SL_MULT, HORIZON_BARS,
)
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    RAW_LEVEL_COLS, PANEL_PATH, LABEL_PATH, MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE,
    TRAIN_END, VAL_START, VAL_END, SEED, replay, side_state_from_proba,
)

ETH_PATH = ROOT / "data/eth_5m_1year.csv"
OUT_PATH = ROOT / "tmp/sol_dl_rl_survey_20260807/oracle_logic_audit.json"


def oracle_on_ohlc(df: pd.DataFrame, lo: str, hi: str) -> dict:
    df = df.sort_values("timestamp").reset_index(drop=True)
    open_ = df["open"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    log_ret = np.diff(np.log(np.clip(close, 1e-9, None)), prepend=np.log(close[0]))
    cumret = pd.Series(log_ret).rolling(CUMRET_BARS).sum().to_numpy()
    vol = pd.Series(cumret).rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std().to_numpy()
    tp_move, sl_move = TP_MULT * vol, SL_MULT * vol
    label, ls, ss = _triple_barrier_race(open_, high, low, tp_move, sl_move, HORIZON_BARS)
    ts = df["timestamp"]
    mask = ((ts >= lo) & (ts <= hi)).to_numpy()
    idx = np.flatnonzero(mask & (label != 0) & np.isfinite(tp_move) & np.isfinite(sl_move))
    side = np.where(label[idx] == 1, 1.0, -1.0)
    res = simulate_single_position(
        timestamps=ts, open_px=open_, high=high, low=low, close=close,
        decision_indices=idx, scores=side, tp_moves=tp_move[idx], sl_moves=sl_move[idx],
        upper_threshold=0.0, lower_threshold=0.0, horizon_bars=HORIZON_BARS,
        margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
    )
    ledger = res.ledger
    counts = pd.Series(label[mask]).value_counts().to_dict()
    return {
        "window": [lo, hi], "n_trades": int(len(ledger)),
        "final_equity": float(res.equity[-1]),
        "sum_ret_pct": float(ledger["trade_return"].sum() * 100.0),
        "win_rate": float((ledger["trade_return"] > 0).mean()),
        "label_counts_in_window": {str(k): int(v) for k, v in counts.items()},
        "median_tp_move_pct": float(np.nanmedian(tp_move[mask]) * 100.0),
    }


def main() -> int:
    # (1) ETH oracle control
    eth = pd.read_csv(ETH_PATH, usecols=["timestamp", "open", "high", "low", "close"])
    eth["timestamp"] = pd.to_datetime(eth["timestamp"])
    eth_oracle = oracle_on_ohlc(eth, "2025-09-01", "2025-12-31 23:59:59")
    print("ETH oracle (identical logic):", json.dumps(eth_oracle), flush=True)

    # (2) SOL leak-injection positive control
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    labels = pd.read_parquet(LABEL_PATH)
    action = labels["trade_outcome_action"].to_numpy()
    tp_moves = labels["tp_move"].to_numpy(dtype=np.float64)
    sl_moves = labels["sl_move"].to_numpy(dtype=np.float64)
    feat_cols = [c for c in panel.columns if c != "timestamp" and c not in RAW_LEVEL_COLS]
    x = panel[feat_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    leak = (labels["trade_outcome_soft_long"].to_numpy() - labels["trade_outcome_soft_short"].to_numpy()).astype(np.float32)
    x_leak = np.column_stack([x, leak])

    ts = panel["timestamp"]
    train_mask = (ts <= TRAIN_END).to_numpy()
    purge_cut = np.flatnonzero(train_mask)[-HORIZON_BARS:]
    train_mask[purge_cut] = False
    train_mask &= np.isfinite(tp_moves)
    val_mask = ((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()

    clf = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=600, learning_rate=0.05,
                             num_leaves=63, min_child_samples=200, feature_fraction=0.8,
                             bagging_fraction=0.8, bagging_freq=1, reg_lambda=1.0,
                             random_state=SEED, n_jobs=-1, verbosity=-1)
    clf.fit(x_leak[train_mask], action[train_mask])
    proba_val = clf.predict_proba(x_leak[val_mask])
    acc = float((proba_val.argmax(axis=1) == action[val_mask]).mean())
    side_state = np.zeros(len(panel), dtype=np.int64)
    side_state[val_mask] = side_state_from_proba(proba_val, 0.55)
    r = replay(panel, side_state, tp_moves, sl_moves, val_mask)
    leak_result = {"val_accuracy_with_leak": acc, "val_replay_with_leak": r}
    print("SOL leak-injection control:", json.dumps(leak_result), flush=True)

    OUT_PATH.write_text(json.dumps({"eth_oracle_control": eth_oracle, "sol_leak_control": leak_result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
