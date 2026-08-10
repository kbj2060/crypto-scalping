"""Stage 1 cheap gate of docs/experiments/sol_dl_rl_architecture_survey_20260807.json.

LightGBM 3-class (CASH/LONG/SHORT) on the SOL flat feature panel against the corrected causal
triple-barrier trade-outcome label. This is the control every DL/RL candidate must beat.

Protocol:
- train <= 2025-08-31 minus a 288-bar purge (label horizon must not cross into VAL);
- entry-rule variants (argmax / side-prob thresholds) are selected on VAL ONLY;
- `--stage oos` replays the single frozen VAL-selected rule on OOS exactly once.

Usage:
  python scripts/train_eval_sol_tripbarrier_lgbm_cheapgate_20260807.py --stage val
  python scripts/train_eval_sol_tripbarrier_lgbm_cheapgate_20260807.py --stage oos
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.causal_futures_backtest import simulate_single_position  # noqa: E402

PANEL_PATH = ROOT / "data/splits/year_oos/sol_features_2024_2026.csv"
LABEL_PATH = ROOT / "data/splits/year_oos/sol_5m_tripbarrier_tradeoutcome_labels_20260807.parquet"
OUT_DIR = ROOT / "tmp/sol_dl_rl_survey_20260807/lgbm_cheapgate"

HORIZON_BARS = 288
MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE = 0.30, 3.0, 0.0010
SEED = 903174

TRAIN_END = pd.Timestamp("2025-08-31 23:59:59")
VAL_START, VAL_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-12-31 23:59:59")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31 23:59:59")

# Raw absolute-level columns: non-stationary across SOL's 166 -> 78 price regime, excluded from
# the flat model input (engineered features already encode their causal information relatively).
RAW_LEVEL_COLS = [
    "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "sum_open_interest_value", "close_btc", "volume_btc", "quote_volume_btc",
    "squeeze_power", "smart_money_flow",
]

ENTRY_RULES = [
    {"name": "argmax", "threshold": 0.0},
    {"name": "side_prob_040", "threshold": 0.40},
    {"name": "side_prob_045", "threshold": 0.45},
    {"name": "side_prob_050", "threshold": 0.50},
    {"name": "side_prob_055", "threshold": 0.55},
    {"name": "side_prob_060", "threshold": 0.60},
]


def load_frames():
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    labels = pd.read_parquet(LABEL_PATH)
    assert (labels["timestamp"].to_numpy() == panel["timestamp"].to_numpy()).all()
    feat_cols = [c for c in panel.columns if c != "timestamp" and c not in RAW_LEVEL_COLS]
    x = panel[feat_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    return panel, labels, feat_cols, x


def replay(panel, side_state, tp_moves, sl_moves, split_mask) -> dict:
    idx = np.flatnonzero(split_mask & (side_state != 0) & np.isfinite(tp_moves) & np.isfinite(sl_moves))
    if len(idx) == 0:
        return {"n_trades": 0}
    res = simulate_single_position(
        timestamps=panel["timestamp"],
        open_px=panel["open"].to_numpy(dtype=np.float64),
        high=panel["high"].to_numpy(dtype=np.float64),
        low=panel["low"].to_numpy(dtype=np.float64),
        close=panel["close"].to_numpy(dtype=np.float64),
        decision_indices=idx,
        scores=side_state[idx].astype(np.float64),
        tp_moves=tp_moves[idx],
        sl_moves=sl_moves[idx],
        upper_threshold=0.0,
        lower_threshold=0.0,
        horizon_bars=HORIZON_BARS,
        margin_fraction=MARGIN_FRACTION,
        leverage=LEVERAGE,
        roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
    )
    ledger = res.ledger
    equity = res.equity
    running_max = np.maximum.accumulate(equity)
    return {
        "n_trades": int(len(ledger)),
        "final_equity": float(equity[-1]),
        "pnl_pct": float((equity[-1] - 1.0) * 100.0),
        "win_rate": float((ledger["trade_return"] > 0).mean()),
        "mdd_pct": float(((equity - running_max) / running_max).min() * 100.0),
        "exit_reasons": {str(k): int(v) for k, v in ledger["reason"].value_counts().items()},
        "long_trades": int((ledger["side"] == 1).sum()),
        "short_trades": int((ledger["side"] == -1).sum()),
    }


def side_state_from_proba(proba: np.ndarray, threshold: float) -> np.ndarray:
    arg = proba.argmax(axis=1)
    side = np.where(arg == 1, 1, np.where(arg == 2, -1, 0))
    if threshold > 0.0:
        side_prob = np.take_along_axis(proba, arg[:, None], axis=1)[:, 0]
        side = np.where(side_prob >= threshold, side, 0)
    return side.astype(np.int64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["val", "oos"], required=True)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel, labels, feat_cols, x = load_frames()
    ts = panel["timestamp"]
    action = labels["trade_outcome_action"].to_numpy()
    tp_moves = labels["tp_move"].to_numpy(dtype=np.float64)
    sl_moves = labels["sl_move"].to_numpy(dtype=np.float64)

    train_mask = (ts <= TRAIN_END).to_numpy()
    # purge: drop the last HORIZON_BARS train rows so no label window crosses into VAL
    purge_cut = np.flatnonzero(train_mask)[-HORIZON_BARS:]
    train_mask[purge_cut] = False
    train_mask &= np.isfinite(tp_moves)  # vol warmup
    val_mask = ((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()
    oos_mask = ((ts >= OOS_START) & (ts <= OOS_END)).to_numpy()

    model_path = OUT_DIR / "lgbm_model.txt"
    if args.stage == "val":
        clf = lgb.LGBMClassifier(
            objective="multiclass", num_class=3, n_estimators=600, learning_rate=0.05,
            num_leaves=63, min_child_samples=200, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=-1,
        )
        clf.fit(x[train_mask], action[train_mask])
        clf.booster_.save_model(str(model_path))

        proba_val = clf.predict_proba(x[val_mask])
        proba_full = np.zeros((len(panel), 3))
        proba_full[val_mask] = proba_val

        acc = float((proba_val.argmax(axis=1) == action[val_mask]).mean())
        results = []
        for rule in ENTRY_RULES:
            side_state = np.zeros(len(panel), dtype=np.int64)
            side_state[val_mask] = side_state_from_proba(proba_val, rule["threshold"])
            r = replay(panel, side_state, tp_moves, sl_moves, val_mask)
            r["rule"] = rule["name"]
            r["threshold"] = rule["threshold"]
            results.append(r)
            print(json.dumps(r))

        eligible = [r for r in results if r.get("n_trades", 0) >= 15]
        best = max(eligible, key=lambda r: r["pnl_pct"]) if eligible else None
        out = {"stage": "val", "seed": SEED, "val_accuracy": acc, "n_features": len(feat_cols), "results": results, "selected_rule": best}
        (OUT_DIR / "val_results.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({"val_accuracy": acc, "selected_rule": best}, indent=2))
    else:
        prior = json.loads((OUT_DIR / "val_results.json").read_text())
        sel = prior["selected_rule"]
        if sel is None:
            print("no VAL-eligible rule; cheap gate FAILED, no OOS read")
            return 1
        booster = lgb.Booster(model_file=str(model_path))
        proba_oos = booster.predict(x[oos_mask])
        side_state = np.zeros(len(panel), dtype=np.int64)
        side_state[oos_mask] = side_state_from_proba(proba_oos, sel["threshold"])
        r = replay(panel, side_state, tp_moves, sl_moves, oos_mask)
        r["rule"] = sel["rule"]
        acc = float((proba_oos.argmax(axis=1) == action[oos_mask]).mean())
        out = {"stage": "oos", "selected_rule": sel["rule"], "oos_accuracy": acc, "oos_result": r}
        (OUT_DIR / "oos_result.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
