"""Fresh-forward evaluation for the trained Tau1 Leg A checkpoint. Mirrors
scripts/eval_btc_tau1_leg_b_fresh_forward_20260805.py's structure/contracts, adapted
for Leg A's 5-minute native decisions and its own probability x flat-margin grid, per
the model-selection contract in docs/btc_new_architecture_session_summary_20260804.md
section 8 ("Leg A: max(P(LONG),P(SHORT)) >= {0.45..0.65} and >= P(FLAT) + {0.00,0.05,0.10}").

Unlike Leg B, Leg A has no external trend-scan direction gate -- the model's own
3-class softmax is the only candidate source. Exit is the SAME fixed TP/SL/timeout
barrier the training label was built from (reused from
scripts/build_btc_leg_a_tactical_labels_20260805.py, not reimplemented), replayed
causally bar-by-bar, single position at a time.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.btc_tau1_dual_leg_architecture_20260805 import (  # noqa: E402
    CALIBRATION_END, CHECKPOINT_END, LEG_A_SEQUENCE, LegANet, load_feature_frame,
)
from scripts.build_btc_leg_a_tactical_labels_20260805 import (  # noqa: E402
    HORIZON, SL_ATR_MULT, SL_FLOOR, TP_ATR_MULT, TP_FLOOR, atr_pct,
)
from scripts.train_eval_btc_110branch_causal_20260804 import COST, load_frame  # noqa: E402

CKPT = ROOT / "tmp/btc_tau1_dual_leg_training_20260805/leg_a_checkpoint.pt"
OUT = ROOT / "tmp/btc_tau1_leg_a_fresh_forward_20260805"
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")
PROB_THRESHOLDS = (.45, .50, .55, .60, .65)
FLAT_MARGINS = (.00, .05, .10)
MIN_CAL_TRADES, BOOTSTRAP_SAMPLES = 50, 2000


class SequenceDataset(Dataset):
    def __init__(self, market: np.ndarray, regime: np.ndarray, rows: np.ndarray) -> None:
        self.market, self.regime, self.rows = torch.from_numpy(market), torch.from_numpy(regime), torch.from_numpy(rows.astype(np.int64))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        end = int(self.rows[i]) + 1
        return self.market[end - LEG_A_SEQUENCE:end], self.regime[end - LEG_A_SEQUENCE:end]


def probabilities() -> tuple[pd.DataFrame, np.ndarray]:
    if not CKPT.exists():
        raise RuntimeError(f"Missing Leg A checkpoint: {CKPT}")
    saved = torch.load(CKPT, map_location="cpu", weights_only=False)
    frame, market_cols, regime_cols = load_feature_frame()
    if saved["market_columns"] != market_cols or saved["regime_columns"] != regime_cols:
        raise RuntimeError("Leg A checkpoint feature contract mismatch")
    market = np.clip((frame[market_cols].to_numpy(np.float32) - saved["market_mean"]) / saved["market_std"], -10, 10).astype(np.float32)
    regime = np.clip((frame[regime_cols].to_numpy(np.float32) - saved["regime_mean"]) / saved["regime_std"], -10, 10).astype(np.float32)
    finite_row = np.isfinite(market).all(1) & np.isfinite(regime).all(1)
    window_finite = pd.Series(finite_row).rolling(LEG_A_SEQUENCE, min_periods=LEG_A_SEQUENCE).min().fillna(0).astype(bool).to_numpy()
    rows = np.flatnonzero(window_finite)
    model = LegANet(); model.load_state_dict(saved["state"]); model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for x, r in DataLoader(SequenceDataset(market, regime, rows), batch_size=512):
            out.append(torch.softmax(model(x, r), dim=1).numpy())
    return frame.iloc[rows].reset_index(drop=True), np.concatenate(out)


def bootstrap_lower_bound(returns: np.ndarray) -> float:
    rng = np.random.default_rng(20260805)
    sampled = returns[rng.integers(0, len(returns), size=(BOOTSTRAP_SAMPLES, len(returns)))].mean(axis=1)
    return float(np.quantile(sampled, .05))


def replay(raw: pd.DataFrame, decisions: pd.DataFrame, prob_threshold: float, flat_margin: float) -> tuple[dict, pd.DataFrame]:
    index = pd.DatetimeIndex(pd.to_datetime(raw.timestamp, utc=True))
    open_px, high, low, close = (raw[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    atr = atr_pct(raw)
    n = len(raw)
    trades, next_free = [], 0
    for row in decisions.itertuples():
        best_side_prob = max(row.long_probability, row.short_probability)
        if best_side_prob < prob_threshold or best_side_prob - row.flat_probability < flat_margin:
            continue
        side = 1 if row.long_probability >= row.short_probability else -1
        decision_i = int(index.searchsorted(row.timestamp))
        if decision_i >= n or index[decision_i] != row.timestamp or decision_i < next_free:
            continue
        entry_i = decision_i + 1
        final_i = entry_i + HORIZON
        if final_i >= n or not np.isfinite(atr[decision_i]):
            continue
        entry = float(open_px[entry_i])
        tp_move = max(TP_FLOOR, TP_ATR_MULT * atr[decision_i])
        sl_move = max(SL_FLOOR, SL_ATR_MULT * atr[decision_i])
        tp_level = entry * (1.0 + tp_move) if side > 0 else entry * (1.0 - tp_move)
        sl_level = entry * (1.0 - sl_move) if side > 0 else entry * (1.0 + sl_move)
        reason, exit_i, move = "timeout", final_i, None
        for j in range(entry_i, final_i + 1):
            hit_sl = (low[j] <= sl_level) if side > 0 else (high[j] >= sl_level)
            hit_tp = (high[j] >= tp_level) if side > 0 else (low[j] <= tp_level)
            if hit_sl:
                reason, exit_i, move = "sl", j, -sl_move
                break
            if hit_tp:
                reason, exit_i, move = "tp", j, tp_move
                break
        if move is None:
            move = (close[final_i] / entry - 1.0) if side > 0 else (1.0 - close[final_i] / entry)
        net = move - COST
        trades.append({"decision_timestamp": row.timestamp, "entry_timestamp": index[entry_i], "exit_timestamp": index[exit_i],
                        "side": "LONG" if side > 0 else "SHORT", "probability": best_side_prob, "net_price_return": net, "exit_reason": reason})
        next_free = exit_i + 1
    ledger = pd.DataFrame(trades)
    returns = ledger.net_price_return.to_numpy(float) if len(ledger) else np.array([], dtype=float)
    metrics = {"prob_threshold": prob_threshold, "flat_margin": flat_margin, "trades": int(len(ledger)),
               "net_price_return_sum": float(returns.sum()), "mean_net_price_return": float(returns.mean()) if len(returns) else None,
               "bootstrap_p05_mean": bootstrap_lower_bound(returns) if len(returns) else None}
    return metrics, ledger


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    frame5m, probs = probabilities()
    raw, _ = load_frame()
    raw["timestamp"] = pd.to_datetime(raw.timestamp, utc=True)
    decisions_all = frame5m[["timestamp"]].assign(
        flat_probability=probs[:, 0], long_probability=probs[:, 1], short_probability=probs[:, 2])

    cal = decisions_all[(decisions_all.timestamp >= CHECKPOINT_END) & (decisions_all.timestamp < CALIBRATION_END)].copy()
    grid = list(itertools.product(PROB_THRESHOLDS, FLAT_MARGINS))
    candidates = [(*replay(raw, cal, thr, margin),) for thr, margin in grid]
    rows = [metrics for metrics, _ in candidates]
    pd.DataFrame(rows).to_csv(OUT / "calibration_candidates.csv", index=False)

    eligible = [(metrics, ledger) for metrics, ledger in candidates
                if metrics["trades"] >= MIN_CAL_TRADES and metrics["net_price_return_sum"] > 0 and metrics["bootstrap_p05_mean"] > 0]
    report = {"checkpoint": str(CKPT), "calibration_candidates": rows,
              "contracts": {"fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
                             "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
                             "thresholds_fit_on_calibration_only": True, "oos_used_for_selection": False}}
    if not eligible:
        report["result"] = "NO_CALIBRATION_CANDIDATE_PASSED_GATE"
        report["oos_metrics"] = None
    else:
        selected, cal_ledger = max(eligible, key=lambda item: item[0]["mean_net_price_return"])
        cal_ledger.to_csv(OUT / "selected_calibration_ledger.csv", index=False)
        oos = decisions_all[(decisions_all.timestamp >= CALIBRATION_END) & (decisions_all.timestamp < OOS_END)].copy()
        oos_metrics, oos_ledger = replay(raw, oos, selected["prob_threshold"], selected["flat_margin"])
        oos_ledger.to_csv(OUT / "oos_ledger.csv", index=False)
        report.update({"result": "CALIBRATION_GATE_PASSED", "selected_config": selected, "oos_metrics": oos_metrics})
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
