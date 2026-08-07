"""Fresh-forward evaluation for the trained Tau1 Leg B checkpoint."""
from __future__ import annotations

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
    CALIBRATION_END, CHECKPOINT_END, LEG_B_SEQUENCE, LegBNet, hourly_completed_features, load_feature_frame,
)
from scripts.build_btc_tau1_continuation_labels_20260805 import (  # noqa: E402
    COST, MAX_HOLD_HOURS, atr_hourly, online_gate, trailing_outcome, trend_scan,
)
from scripts.train_eval_btc_110branch_causal_20260804 import load_frame  # noqa: E402

CKPT = ROOT / "tmp/btc_tau1_dual_leg_training_20260805/leg_b_checkpoint.pt"
OUT = ROOT / "tmp/btc_tau1_leg_b_fresh_forward_20260805"
THRESHOLDS = (.45, .50, .55, .60, .65)
MIN_CAL_TRADES, BOOTSTRAP_SAMPLES = 20, 2000


class SequenceDataset(Dataset):
    def __init__(self, market: np.ndarray, regime: np.ndarray, rows: np.ndarray) -> None:
        self.market, self.regime, self.rows = torch.from_numpy(market), torch.from_numpy(regime), torch.from_numpy(rows.astype(np.int64))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        end = int(self.rows[i]) + 1
        return self.market[end - LEG_B_SEQUENCE:end], self.regime[end - LEG_B_SEQUENCE:end]


def probabilities() -> tuple[pd.DataFrame, np.ndarray]:
    if not CKPT.exists():
        raise RuntimeError(f"Missing Leg B checkpoint: {CKPT}")
    saved = torch.load(CKPT, map_location="cpu", weights_only=False)
    frame, market_cols, regime_cols = load_feature_frame()
    frame = hourly_completed_features(frame, market_cols, regime_cols)
    regime_cols = [f"regime_input_{column}" for column in regime_cols]
    if saved["market_columns"] != market_cols or saved["regime_columns"] != regime_cols:
        raise RuntimeError("Leg B checkpoint feature contract mismatch")
    market = np.clip((frame[market_cols].to_numpy(np.float32) - saved["market_mean"]) / saved["market_std"], -10, 10).astype(np.float32)
    regime = np.clip((frame[regime_cols].to_numpy(np.float32) - saved["regime_mean"]) / saved["regime_std"], -10, 10).astype(np.float32)
    rows = np.flatnonzero(np.arange(len(frame)) >= LEG_B_SEQUENCE - 1)
    model = LegBNet(); model.load_state_dict(saved["state"]); model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for x, r in DataLoader(SequenceDataset(market, regime, rows), batch_size=512):
            out.append(torch.softmax(model(x, r), dim=1).numpy())
    return frame.iloc[rows].reset_index(drop=True), np.concatenate(out)


def causal_candidates(raw: pd.DataFrame) -> pd.DataFrame:
    bars = raw.set_index("timestamp")[["open", "high", "low", "close"]].resample("1h", label="left", closed="left").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().reset_index()
    t_value, window, beta = trend_scan(np.log(bars.close.to_numpy(float)))
    gate = online_gate(pd.Series(np.abs(t_value)))
    atr = atr_hourly(bars)
    result = pd.DataFrame({"timestamp": pd.to_datetime(bars.timestamp, utc=True) + pd.Timedelta(hours=1), "side": np.where(beta > 0, 1, -1), "window": window, "atr": atr, "gate": gate})
    return result[result.gate & np.isfinite(result.atr) & (result.window > 0)].copy()


def bootstrap_lower_bound(returns: np.ndarray) -> float:
    rng = np.random.default_rng(20260805)
    sampled = returns[rng.integers(0, len(returns), size=(BOOTSTRAP_SAMPLES, len(returns)))].mean(axis=1)
    return float(np.quantile(sampled, .05))


def replay(raw: pd.DataFrame, decisions: pd.DataFrame, threshold: float) -> tuple[dict, pd.DataFrame]:
    index = pd.DatetimeIndex(pd.to_datetime(raw.timestamp, utc=True)); open_px, high, low, close = (raw[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    trades, next_free = [], 0
    for row in decisions.itertuples():
        probability = float(row.long_probability if row.side > 0 else row.short_probability)
        if probability < threshold:
            continue
        decision_i = int(index.searchsorted(row.timestamp))
        if decision_i >= len(raw) or index[decision_i] != row.timestamp or decision_i < next_free:
            continue
        entry_i = decision_i + 1
        hold = min(MAX_HOLD_HOURS, 4 * int(row.window)) * 12
        final_i = entry_i + hold - 1
        if final_i >= len(raw):
            continue
        move, reason, offset = trailing_outcome(side=int(row.side), entry=float(open_px[entry_i]), high=high[entry_i:final_i + 1], low=low[entry_i:final_i + 1], close=close[entry_i:final_i + 1], atr_pct=float(row.atr))
        net = move - COST
        exit_i = entry_i + offset
        trades.append({"decision_timestamp": row.timestamp, "entry_timestamp": index[entry_i], "exit_timestamp": index[exit_i], "side": "LONG" if row.side > 0 else "SHORT", "probability": probability, "net_price_return": net, "exit_reason": reason})
        next_free = exit_i + 1
    ledger = pd.DataFrame(trades)
    returns = ledger.net_price_return.to_numpy(float) if len(ledger) else np.array([], dtype=float)
    metrics = {"threshold": threshold, "trades": int(len(ledger)), "net_price_return_sum": float(returns.sum()), "mean_net_price_return": float(returns.mean()) if len(returns) else None, "bootstrap_p05_mean": bootstrap_lower_bound(returns) if len(returns) else None}
    return metrics, ledger


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    hourly, probs = probabilities()
    raw, _ = load_frame()
    raw["timestamp"] = pd.to_datetime(raw.timestamp, utc=True)
    joined = hourly[["timestamp"]].assign(flat_probability=probs[:, 0], short_probability=probs[:, 2], long_probability=probs[:, 1]).merge(causal_candidates(raw), on="timestamp", how="inner", validate="one_to_one")
    cal = joined[(joined.timestamp >= CHECKPOINT_END) & (joined.timestamp < CALIBRATION_END)].copy()
    candidates = [(*replay(raw, cal, threshold),) for threshold in THRESHOLDS]
    rows = [metrics for metrics, _ in candidates]
    pd.DataFrame(rows).to_csv(OUT / "calibration_candidates.csv", index=False)
    eligible = [(metrics, ledger) for metrics, ledger in candidates if metrics["trades"] >= MIN_CAL_TRADES and metrics["net_price_return_sum"] > 0 and metrics["bootstrap_p05_mean"] > 0]
    report = {"checkpoint": str(CKPT), "calibration_candidates": rows, "contracts": {"fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False, "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False, "thresholds_fit_on_calibration_only": True, "oos_used_for_selection": False}}
    if not eligible:
        report["result"] = "NO_CALIBRATION_CANDIDATE_PASSED_GATE"
        report["oos_metrics"] = None
    else:
        selected, cal_ledger = max(eligible, key=lambda item: item[0]["mean_net_price_return"])
        cal_ledger.to_csv(OUT / "selected_calibration_ledger.csv", index=False)
        oos = joined[(joined.timestamp >= CALIBRATION_END) & (joined.timestamp < pd.Timestamp("2026-04-01", tz="UTC"))].copy()
        oos_metrics, oos_ledger = replay(raw, oos, selected["threshold"])
        oos_ledger.to_csv(OUT / "oos_ledger.csv", index=False)
        report.update({"result": "CALIBRATION_GATE_PASSED", "selected_config": selected, "oos_metrics": oos_metrics})
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
