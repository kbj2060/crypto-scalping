"""BTC-110 event model: causal CUSUM candidates + symmetric 3-class triple-barrier labels."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from core.backtest_metrics import bar_level_performance  # noqa: E402
from core.causal_event_labels import causal_cusum_events, triple_barrier_direction  # noqa: E402
from core.causal_futures_backtest import purged_decision_mask, simulate_single_position  # noqa: E402
from train_eval_btc_110branch_causal_20260804 import (  # noqa: E402
    COST, HORIZON, LEVERAGE, MARGIN, ONCHAIN_COLS, DVOL_COLS, REGIME, load_frame,
)

OUT = ROOT / "tmp/btc110_cusum_tb_causal_20260804"
TRAIN_END, VAL_END, CAL_END, TEST_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-01"), pd.Timestamp("2026-08-01")
CUSUM_MULTS, SCORE_THRESHOLDS = [1.5, 2.0, 2.5], [.05, .10, .15, .20]
TP_MULT, SL_MULT, MIN_TP, MIN_SL = 1.2, .8, .006, .004
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EventNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        def branch(n, width): return nn.Sequential(nn.Linear(n, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(.1))
        self.market, self.context = branch(94, 64), branch(16, 32)
        self.fuse = nn.Sequential(nn.Linear(96, 64), nn.GELU(), nn.Dropout(.1))
        self.residual, self.norm, self.head = nn.Linear(64, 64), nn.LayerNorm(64), nn.Linear(64, 3)

    def forward(self, x):
        z = self.fuse(torch.cat([self.market(x[:, :94]), self.context(x[:, 94:])], 1))
        return self.head(torch.nn.functional.gelu(self.norm(z + self.residual(z))))


def atr_move(frame: pd.DataFrame) -> np.ndarray:
    high, low, close = frame.high.to_numpy(float), frame.low.to_numpy(float), frame.close.to_numpy(float)
    prev = np.r_[close[0], close[:-1]]
    tr = np.maximum.reduce([high - low, np.abs(high - prev), np.abs(low - prev)])
    return (pd.Series(tr).rolling(14, min_periods=4).mean().to_numpy() / close).astype(float)


def label_events(frame: pd.DataFrame, events: np.ndarray, atr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    high, low, close, op = (frame[c].to_numpy(float) for c in ("high", "low", "close", "open"))
    keep, labels = [], []
    for i in events:
        entry_i, end_i = int(i) + 1, int(i) + HORIZON
        if end_i >= len(frame): continue
        move = max(MIN_TP, TP_MULT * float(atr[i]))
        keep.append(i); labels.append(triple_barrier_direction(entry=float(op[entry_i]), high=high[entry_i:end_i + 1], low=low[entry_i:end_i + 1], close=close[entry_i:end_i + 1], move=move))
    return np.asarray(keep, int), np.asarray(labels, int)


def run_epoch(model, loader, optimiser=None):
    model.train(optimiser is not None); total = 0.; n = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if optimiser: optimiser.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        if optimiser: loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); optimiser.step()
        total += loss.item(); n += 1
    return total / max(n, 1)


def predict(model, x):
    model.eval(); values = []
    with torch.no_grad():
        for (b,) in DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=1024): values.append(torch.softmax(model(b.to(DEVICE)), 1).cpu().numpy())
    return np.concatenate(values)


def evaluate(frame, events, probs, atr, threshold):
    # score in [-1,1]; ±threshold gives an explicit neutral zone around FLAT.
    score = probs[:, 2] - probs[:, 1]
    tp, sl = np.maximum(MIN_TP, TP_MULT * atr[events]), np.maximum(MIN_SL, SL_MULT * atr[events])
    result = simulate_single_position(timestamps=frame.timestamp, open_px=frame.open.to_numpy(), high=frame.high.to_numpy(), low=frame.low.to_numpy(), close=frame.close.to_numpy(), decision_indices=events, scores=score, tp_moves=tp, sl_moves=sl, upper_threshold=threshold, lower_threshold=-threshold, horizon_bars=HORIZON, margin_fraction=MARGIN, leverage=LEVERAGE, roundtrip_cost_rate=COST)
    m = bar_level_performance(result.equity, result.ledger); m["mean_trade_return_pct"] = float(result.ledger.trade_return.mean() * 100) if len(result.ledger) else 0.; m["skipped_while_open"] = result.skipped_while_open
    return m, result.ledger


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); frame, cols = load_frame(); ts = pd.DatetimeIndex(frame.timestamp); atr = atr_move(frame)
    raw = frame[cols].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32); masks = {"train": purged_decision_mask(ts, start=ts[0], end=TRAIN_END, horizon_bars=HORIZON), "val": purged_decision_mask(ts, start=TRAIN_END, end=VAL_END, horizon_bars=HORIZON), "cal": purged_decision_mask(ts, start=VAL_END, end=CAL_END, horizon_bars=HORIZON), "test": purged_decision_mask(ts, start=CAL_END, end=TEST_END, horizon_bars=HORIZON)}
    train_rows = np.flatnonzero(masks["train"] & np.isfinite(raw).all(1)); mean, std = raw[train_rows].mean(0), raw[train_rows].std(0); std[std < 1e-6] = 1; x = np.clip((raw - mean) / std, -10, 10).astype(np.float32)
    candidates, validation = [], []
    for mult in CUSUM_MULTS:
        events, y = label_events(frame, causal_cusum_events(frame.close.to_numpy(), atr, mult), atr)
        valid = np.isfinite(raw[events]).all(1)
        events, y = events[valid], y[valid]
        groups = {name: np.flatnonzero(masks[name][events]) for name in masks}
        if any(len(groups[name]) == 0 for name in ("train", "val", "cal", "test")): raise RuntimeError(f"CUSUM {mult} produced an empty split")
        class_counts = np.bincount(y[groups["train"]], minlength=3)
        model = EventNet().to(DEVICE); opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        train = DataLoader(TensorDataset(torch.from_numpy(x[events[groups["train"]]]), torch.from_numpy(y[groups["train"]])), batch_size=256, shuffle=True)
        val = DataLoader(TensorDataset(torch.from_numpy(x[events[groups["val"]]]), torch.from_numpy(y[groups["val"]])), batch_size=512)
        best, bad, state, best_val = float("inf"), 0, None, None
        for epoch in range(1, 13):
            train_loss = run_epoch(model, train, opt); val_loss = run_epoch(model, val)
            print(f"cusum={mult} epoch={epoch} train_ce={train_loss:.5f} val_ce={val_loss:.5f}", flush=True)
            if val_loss < best - 1e-5: best, bad, state, best_val = val_loss, 0, {k:v.cpu().clone() for k,v in model.state_dict().items()}, val_loss
            else:
                bad += 1
                if bad >= 3: break
        model.load_state_dict(state); cal_p = predict(model, x[events[groups["cal"]]])
        validation.append({"cusum_multiplier": mult, "event_counts": {k:int(len(v)) for k,v in groups.items()}, "class_counts_train": class_counts.tolist(), "validation_cross_entropy": best_val})
        for threshold in SCORE_THRESHOLDS:
            metric, ledger = evaluate(frame, events[groups["cal"]], cal_p, atr, threshold)
            candidates.append((metric["pnl"], mult, threshold, state, events, groups, metric, ledger))
    _, mult, threshold, state, events, groups, cal_metric, cal_ledger = max(candidates, key=lambda z:z[0]); model = EventNet().to(DEVICE); model.load_state_dict(state); test_p = predict(model, x[events[groups["test"]]]); test_metric, test_ledger = evaluate(frame, events[groups["test"]], test_p, atr, threshold)
    cal_ledger.to_csv(OUT / "selected_calibration_ledger.csv", index=False); test_ledger.to_csv(OUT / "test_ledger.csv", index=False)
    report = {"architecture":"btc110_market_context_event_classifier", "layers":{"market":"94→64→LayerNorm→GELU→Dropout(0.1)","context":"16→32→LayerNorm→GELU→Dropout(0.1)","fusion":"96→64→GELU→Dropout(0.1)→residual(64)→LayerNorm→GELU","output":"64→3 [FLAT, SHORT, LONG]"}, "feature_contract":{"market_causalfix":94,"regime3_current":REGIME,"dvol":DVOL_COLS,"onchain":ONCHAIN_COLS,"total":110}, "label_contract":{"event":"causal CUSUM on close log-returns", "label":"symmetric 3-class triple barrier; intrabar dual-touch=FLAT", "entry":"event t+1 open", "directional_barrier":"max(0.6%, 1.2×ATR%)", "vertical_barrier_bars":HORIZON, "execution_stop":"max(0.4%, 0.8×ATR%)"}, "model_validation":validation, "selected_config":{"cusum_multiplier":mult,"score_threshold":threshold,"calibration_metrics":cal_metric}, "test_metrics":test_metric, "contracts":{"fresh_forward_bar_by_bar":True,"thresholds_fit_on_calibration_only":True,"trade_ledgers_used_as_input":False,"saved_parent_exit_timestamps_used":False,"future_rows_used_for_entry":False,"split_targets_purged":True,"single_position":True,"bar_level_mark_to_market":True,"regime3_pred_inputs_forbidden":True,"cusum_causal":True,"label_future_used_for_training_only":True}, "promotion_eligible":False,"promotion_blockers":["test period previously inspected","CUSUM event family previously failed; this is a distinct label/model diagnostic only"]}
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str)+"\n"); print(json.dumps({"selected":report["selected_config"],"test":test_metric},indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
