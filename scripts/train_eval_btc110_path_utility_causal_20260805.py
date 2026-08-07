"""BTC-110 path-utility labels with a causal 3-class execution backtest.

The MFE/MAE scan is a supervised target only.  At inference the model consumes
only features available at decision bar t, enters at t+1 open, and the backtest
does not read the target scan or any stored trade ledger.
"""
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
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from core.backtest_metrics import bar_level_performance  # noqa: E402
from core.causal_futures_backtest import purged_decision_mask, simulate_single_position  # noqa: E402
from train_eval_btc_110branch_causal_20260804 import COST, LEVERAGE, MARGIN, load_frame  # noqa: E402

OUT = ROOT / "tmp/btc110_path_utility_causal_20260805"
TRAIN_END = pd.Timestamp("2025-09-01")
VAL_END, CAL_END, TEST_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-01"), pd.Timestamp("2026-08-01")
LABEL_HORIZON = EXEC_HORIZON = 288
ADVERSE_WEIGHT, MIN_UTILITY = 1.25, 0.002
TP_MOVE, SL_MOVE = 0.012, 0.008
SCORE_THRESHOLDS = [0.05, 0.10, 0.15, 0.20]
MIN_CAL_TRADES, DEVICE = 30, "cuda" if torch.cuda.is_available() else "cpu"


class PathUtilityNet(nn.Module):
    """94 market + 16 context -> 3 logits (FLAT, SHORT, LONG)."""

    def __init__(self) -> None:
        super().__init__()

        def branch(width_in: int, width_out: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(width_in, width_out), nn.LayerNorm(width_out), nn.GELU(), nn.Dropout(0.10))

        self.market = branch(94, 64)
        self.context = branch(16, 32)
        self.fuse = nn.Sequential(nn.Linear(96, 64), nn.GELU(), nn.Dropout(0.10))
        self.residual = nn.Linear(64, 64)
        self.norm = nn.LayerNorm(64)
        self.head = nn.Linear(64, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.fuse(torch.cat([self.market(x[:, :94]), self.context(x[:, 94:])], dim=1))
        return self.head(torch.nn.functional.gelu(self.norm(z + self.residual(z))))


def path_utility_labels(frame: pd.DataFrame, *, horizon_bars: int = LABEL_HORIZON) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return FLAT=0/SHORT=1/LONG=2 labels from forward MFE/MAE utility.

    Label at decision bar i uses an assumed entry at i+1 open and bars i+1 through
    i+horizon.  The last incomplete targets remain -1 and are excluded by the
    split mask.  It is intentionally not a first-touch / triple-barrier label.
    """
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be positive")
    high, low, entry_open = (frame[c].to_numpy(dtype=float) for c in ("high", "low", "open"))
    n = len(frame)
    labels = np.full(n, -1, dtype=np.int64)
    long_utility, short_utility = np.full(n, np.nan), np.full(n, np.nan)
    for decision_i in range(n - horizon_bars):
        entry_i = decision_i + 1
        entry = entry_open[entry_i]
        path_high, path_low = high[entry_i : entry_i + horizon_bars], low[entry_i : entry_i + horizon_bars]
        long_mfe, long_mae = path_high.max() / entry - 1.0, 1.0 - path_low.min() / entry
        short_mfe, short_mae = long_mae, long_mfe
        lu = long_mfe - ADVERSE_WEIGHT * long_mae - COST
        su = short_mfe - ADVERSE_WEIGHT * short_mae - COST
        long_utility[decision_i], short_utility[decision_i] = lu, su
        if lu >= MIN_UTILITY and lu > su:
            labels[decision_i] = 2
        elif su >= MIN_UTILITY and su > lu:
            labels[decision_i] = 1
        else:
            labels[decision_i] = 0
    return labels, {"long_utility": long_utility, "short_utility": short_utility}


def _epoch(model: nn.Module, loader: DataLoader, optimizer=None) -> float:
    model.train(optimizer is not None)
    total, batches = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if optimizer is not None:
            optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        if optimizer is not None:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total, batches = total + loss.item(), batches + 1
    return total / max(batches, 1)


def _predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=1024):
            chunks.append(torch.softmax(model(batch.to(DEVICE)), dim=1).cpu().numpy())
    return np.concatenate(chunks)


def _evaluate(frame: pd.DataFrame, indices: np.ndarray, probabilities: np.ndarray, threshold: float):
    score = probabilities[:, 2] - probabilities[:, 1]
    result = simulate_single_position(
        timestamps=frame.timestamp,
        open_px=frame.open.to_numpy(), high=frame.high.to_numpy(), low=frame.low.to_numpy(), close=frame.close.to_numpy(),
        decision_indices=indices, scores=score,
        tp_moves=np.full(len(indices), TP_MOVE), sl_moves=np.full(len(indices), SL_MOVE),
        upper_threshold=threshold, lower_threshold=-threshold, horizon_bars=EXEC_HORIZON,
        margin_fraction=MARGIN, leverage=LEVERAGE, roundtrip_cost_rate=COST,
    )
    metrics = bar_level_performance(result.equity, result.ledger)
    metrics["mean_trade_return_pct"] = float(result.ledger.trade_return.mean() * 100.0) if len(result.ledger) else 0.0
    metrics["skipped_while_open"] = result.skipped_while_open
    return metrics, result.ledger


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    frame, columns = load_frame()
    timestamps = pd.DatetimeIndex(frame.timestamp)
    raw = frame[columns].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    labels, utility = path_utility_labels(frame)
    masks = {
        "train": purged_decision_mask(timestamps, start=timestamps[0], end=TRAIN_END, horizon_bars=LABEL_HORIZON),
        "val": purged_decision_mask(timestamps, start=TRAIN_END, end=VAL_END, horizon_bars=LABEL_HORIZON),
        "cal": purged_decision_mask(timestamps, start=VAL_END, end=CAL_END, horizon_bars=LABEL_HORIZON),
        "test": purged_decision_mask(timestamps, start=CAL_END, end=TEST_END, horizon_bars=LABEL_HORIZON),
    }
    valid = np.isfinite(raw).all(axis=1) & (labels >= 0)
    groups = {name: np.flatnonzero(mask & valid) for name, mask in masks.items()}
    if any(len(groups[name]) == 0 for name in groups):
        raise RuntimeError(f"empty split after target purge: {[name for name, rows in groups.items() if not len(rows)]}")

    train_rows = groups["train"]
    mean, std = raw[train_rows].mean(axis=0), raw[train_rows].std(axis=0)
    std[std < 1e-6] = 1.0
    x = np.clip((raw - mean) / std, -10.0, 10.0).astype(np.float32)
    model = PathUtilityNet().to(DEVICE)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x[train_rows]), torch.from_numpy(labels[train_rows])), batch_size=256, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x[groups["val"]]), torch.from_numpy(labels[groups["val"]])), batch_size=512)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_loss, bad_epochs, best_state, history = float("inf"), 0, None, []
    for epoch in range(1, 13):
        train_loss, val_loss = _epoch(model, train_loader, optimizer), _epoch(model, val_loader)
        history.append({"epoch": epoch, "train_cross_entropy": train_loss, "validation_cross_entropy": val_loss})
        print(f"epoch={epoch} train_ce={train_loss:.6f} val_ce={val_loss:.6f}", flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss, bad_epochs = val_loss, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= 3:
                break
    if best_state is None:
        raise RuntimeError("no validation checkpoint was saved")
    model.load_state_dict(best_state)

    calibration_probabilities = _predict(model, x[groups["cal"]])
    candidates: list[tuple[dict, pd.DataFrame]] = []
    for threshold in SCORE_THRESHOLDS:
        metrics, ledger = _evaluate(frame, groups["cal"], calibration_probabilities, threshold)
        candidates.append(({"score_threshold": threshold, **metrics}, ledger))
    candidates_table = pd.DataFrame([row for row, _ in candidates])
    candidates_table.to_csv(OUT / "calibration_candidates.csv", index=False)
    eligible = [(row, ledger) for row, ledger in candidates if row["pnl"] > 0.0 and row["trades"] >= MIN_CAL_TRADES]
    report = {
        "architecture": "btc110_path_utility_classifier",
        "layers": {
            "market": "94→64→LayerNorm→GELU→Dropout(0.1)",
            "context": "16→32→LayerNorm→GELU→Dropout(0.1)",
            "fusion": "96→64→GELU→Dropout(0.1)→residual(64)→LayerNorm→GELU",
            "output": "64→3 [FLAT, SHORT, LONG]",
        },
        "feature_contract": {"market_causalfix": 94, "context_regime3_dvol_onchain": 16, "total": 110},
        "label_contract": {
            "name": "forward_path_mfe_mae_net_utility",
            "entry_for_label": "decision t+1 open",
            "horizon_bars": LABEL_HORIZON,
            "long_utility": "long_MFE - 1.25*long_MAE - 0.0014",
            "short_utility": "short_MFE - 1.25*short_MAE - 0.0014",
            "minimum_utility": MIN_UTILITY,
            "classes": {"0": "FLAT", "1": "SHORT", "2": "LONG"},
            "future_path_used_only_as_training_target": True,
        },
        "execution_contract": {"entry": "decision t+1 open", "tp_price_move": TP_MOVE, "sl_price_move": SL_MOVE, "max_hold_bars": EXEC_HORIZON, "margin_fraction": MARGIN, "leverage": LEVERAGE, "notional": MARGIN * LEVERAGE, "roundtrip_cost_rate": COST},
        "model_validation": history,
        "label_distribution": {name: {str(cls): int((labels[rows] == cls).sum()) for cls in range(3)} for name, rows in groups.items()},
        "utility_summary_train": {name: float(np.nanquantile(utility[name][groups["train"]], .5)) for name in utility},
        "calibration_candidates": [row for row, _ in candidates],
        "contracts": {"fresh_forward_bar_by_bar": True, "thresholds_fit_on_calibration_only": True, "trade_ledgers_used_as_input": False, "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False, "split_targets_purged": True, "single_position": True, "bar_level_mark_to_market": True, "regime3_pred_inputs_forbidden": True, "test_used_for_selection": False},
        "promotion_eligible": False,
    }
    if not eligible:
        report.update({"result": "NO_CALIBRATION_CANDIDATE_PASSED_GATE", "test_metrics": None, "promotion_blockers": ["no positive calibration candidate with at least 30 trades"]})
        (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(report["result"])
        return 0
    selected, calibration_ledger = max(eligible, key=lambda item: item[0]["pnl"])
    calibration_ledger.to_csv(OUT / "selected_calibration_ledger.csv", index=False)
    test_metrics, test_ledger = _evaluate(frame, groups["test"], _predict(model, x[groups["test"]]), selected["score_threshold"])
    test_ledger.to_csv(OUT / "test_ledger.csv", index=False)
    report.update({"result": "CALIBRATION_GATE_PASSED", "selected_config": selected, "test_metrics": test_metrics, "promotion_blockers": ["test period previously inspected", "research-only artifact lineage"]})
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({"selected": selected, "test": test_metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
