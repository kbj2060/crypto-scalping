"""Stage 0 of docs/experiments/sol_dl_rl_architecture_survey_20260807.json: oracle label-following
replay (the 4-step oracle-validation protocol from project-btc-oracle-label-selection-protocol).

An oracle that knows the SOL triple-barrier trade-outcome label perfectly enters label-side at
next bar open under the exact live execution/cost model. If even the oracle can't clear costs,
the label is dead and no model gets trained. TRAIN and VAL only -- OOS is not read at this stage.
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

PANEL_PATH = ROOT / "data/splits/year_oos/sol_features_2024_2026.csv"
LABEL_PATH = ROOT / "data/splits/year_oos/sol_5m_tripbarrier_tradeoutcome_labels_20260807.parquet"
OUT_PATH = ROOT / "tmp/sol_dl_rl_survey_20260807/oracle_ceiling.json"

HORIZON_BARS = 288
MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE = 0.30, 3.0, 0.0010

TRAIN_END = pd.Timestamp("2025-08-31 23:59:59")
VAL_START, VAL_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-12-31 23:59:59")


def _summarize(result, split: str) -> dict:
    ledger = result.ledger
    n = len(ledger)
    if n == 0:
        return {"split": split, "n_trades": 0}
    equity = result.equity
    running_max = np.maximum.accumulate(equity)
    return {
        "split": split,
        "n_trades": n,
        "final_equity": float(equity[-1]),
        "sum_ret_pct": float(ledger["trade_return"].sum() * 100.0),
        "win_rate": float((ledger["trade_return"] > 0).mean()),
        "mdd_pct": float(((equity - running_max) / running_max).min() * 100.0),
        "exit_reasons": {str(k): int(v) for k, v in ledger["reason"].value_counts().items()},
    }


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL_PATH, usecols=["timestamp", "open", "high", "low", "close"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    labels = pd.read_parquet(LABEL_PATH)
    assert (labels["timestamp"].to_numpy() == panel["timestamp"].to_numpy()).all()

    action = labels["trade_outcome_action"].to_numpy()
    tp_moves = labels["tp_move"].to_numpy(dtype=np.float64)
    sl_moves = labels["sl_move"].to_numpy(dtype=np.float64)
    ts = panel["timestamp"]

    results = []
    for split, lo, hi in (("train", ts.iloc[0], TRAIN_END), ("val", VAL_START, VAL_END)):
        in_split = (ts >= lo) & (ts <= hi)
        idx = np.flatnonzero(in_split.to_numpy() & (action != 0) & np.isfinite(tp_moves) & np.isfinite(sl_moves))
        side = np.where(action[idx] == 1, 1.0, -1.0)
        res = simulate_single_position(
            timestamps=ts,
            open_px=panel["open"].to_numpy(dtype=np.float64),
            high=panel["high"].to_numpy(dtype=np.float64),
            low=panel["low"].to_numpy(dtype=np.float64),
            close=panel["close"].to_numpy(dtype=np.float64),
            decision_indices=idx,
            scores=side,
            tp_moves=tp_moves[idx],
            sl_moves=sl_moves[idx],
            upper_threshold=0.0,
            lower_threshold=0.0,
            horizon_bars=HORIZON_BARS,
            margin_fraction=MARGIN_FRACTION,
            leverage=LEVERAGE,
            roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
        )
        results.append(_summarize(res, split))

    out = {"protocol": "oracle_label_following", "cost_model": {"roundtrip_cost_rate": ROUNDTRIP_COST_RATE, "margin_fraction": MARGIN_FRACTION, "leverage": LEVERAGE}, "results": results}
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
