"""BTC-110 temporal causal-TCN classifier on the fixed path-utility label."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from core.backtest_metrics import bar_level_performance  # noqa: E402
from core.causal_futures_backtest import purged_decision_mask, simulate_single_position  # noqa: E402
from train_eval_btc_110branch_causal_20260804 import COST, LEVERAGE, MARGIN, load_frame  # noqa: E402
from train_eval_btc110_path_utility_causal_20260805 import LABEL_HORIZON, path_utility_labels  # noqa: E402

OUT = ROOT / "tmp/btc110_temporal_tcn_causal_20260805"
TRAIN_END = pd.Timestamp("2025-09-01")
VAL_END, CAL_END, TEST_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-01"), pd.Timestamp("2026-08-01")
SEQUENCE_BARS, EXEC_HORIZON = 24, 288
TP_MOVE, SL_MOVE = 0.012, 0.008
SCORE_THRESHOLDS, MIN_CAL_TRADES = [0.05, 0.10, 0.15, 0.20], 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SequenceDataset(Dataset):
    def __init__(self, values: np.ndarray, indices: np.ndarray, labels: np.ndarray | None = None) -> None:
        self.values = torch.from_numpy(values)
        self.indices = torch.from_numpy(indices.astype(np.int64))
        self.labels = None if labels is None else torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, row: int):
        end = int(self.indices[row]) + 1
        sequence = self.values[end - SEQUENCE_BARS : end]
        return sequence if self.labels is None else (sequence, self.labels[row])


class CausalTCNBlock(nn.Module):
    """No right padding: output at each time step only reads its own past."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(0.10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(F.pad(x, (self.padding, 0)))
        z = self.norm(z.transpose(1, 2)).transpose(1, 2)
        return F.gelu(x + self.dropout(z))


class TemporalTCNNet(nn.Module):
    """24x110 -> causal market TCN plus current 16-column context -> 3 logits."""

    def __init__(self) -> None:
        super().__init__()
        self.market_projection = nn.Linear(94, 48)
        self.tcn = nn.Sequential(CausalTCNBlock(48, 1), CausalTCNBlock(48, 2), CausalTCNBlock(48, 4))
        self.context = nn.Sequential(nn.Linear(16, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(0.10))
        self.fusion = nn.Sequential(nn.Linear(80, 64), nn.GELU(), nn.Dropout(0.10))
        self.residual, self.norm, self.head = nn.Linear(64, 64), nn.LayerNorm(64), nn.Linear(64, 3)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        market = self.market_projection(sequence[:, :, :94]).transpose(1, 2)
        market_last = self.tcn(market)[:, :, -1]
        context_now = self.context(sequence[:, -1, 94:])
        z = self.fusion(torch.cat([market_last, context_now], dim=1))
        return self.head(F.gelu(self.norm(z + self.residual(z))))


def _epoch(model: nn.Module, loader: DataLoader, optimizer=None) -> float:
    model.train(optimizer is not None)
    total, batches = 0.0, 0
    for sequence, labels in loader:
        sequence, labels = sequence.to(DEVICE), labels.to(DEVICE)
        if optimizer is not None:
            optimizer.zero_grad()
        loss = F.cross_entropy(model(sequence), labels)
        if optimizer is not None:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total, batches = total + loss.item(), batches + 1
    return total / max(batches, 1)


def _predict(model: nn.Module, values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for sequence in DataLoader(SequenceDataset(values, indices), batch_size=1024):
            chunks.append(torch.softmax(model(sequence.to(DEVICE)), dim=1).cpu().numpy())
    return np.concatenate(chunks)


def _evaluate(frame: pd.DataFrame, indices: np.ndarray, probabilities: np.ndarray, threshold: float):
    result = simulate_single_position(
        timestamps=frame.timestamp, open_px=frame.open.to_numpy(), high=frame.high.to_numpy(), low=frame.low.to_numpy(), close=frame.close.to_numpy(),
        decision_indices=indices, scores=probabilities[:, 2] - probabilities[:, 1],
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
    labels, _ = path_utility_labels(frame, horizon_bars=LABEL_HORIZON)
    masks = {
        "train": purged_decision_mask(timestamps, start=timestamps[0], end=TRAIN_END, horizon_bars=LABEL_HORIZON),
        "val": purged_decision_mask(timestamps, start=TRAIN_END, end=VAL_END, horizon_bars=LABEL_HORIZON),
        "cal": purged_decision_mask(timestamps, start=VAL_END, end=CAL_END, horizon_bars=LABEL_HORIZON),
        "test": purged_decision_mask(timestamps, start=CAL_END, end=TEST_END, horizon_bars=LABEL_HORIZON),
    }
    feature_ready = np.isfinite(raw).all(axis=1)
    sequence_ready = np.convolve(feature_ready.astype(np.int16), np.ones(SEQUENCE_BARS, dtype=np.int16), mode="full")[: len(raw)] == SEQUENCE_BARS
    valid = feature_ready & sequence_ready & (labels >= 0)
    groups = {name: np.flatnonzero(mask & valid) for name, mask in masks.items()}
    if any(len(rows) == 0 for rows in groups.values()):
        raise RuntimeError(f"empty split after target/window purge: {[name for name, rows in groups.items() if not len(rows)]}")
    mean, std = raw[groups["train"]].mean(axis=0), raw[groups["train"]].std(axis=0)
    std[std < 1e-6] = 1.0
    values = np.clip((raw - mean) / std, -10.0, 10.0).astype(np.float32)

    model = TemporalTCNNet().to(DEVICE)
    train_loader = DataLoader(SequenceDataset(values, groups["train"], labels[groups["train"]]), batch_size=256, shuffle=True)
    val_loader = DataLoader(SequenceDataset(values, groups["val"], labels[groups["val"]]), batch_size=512)
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
    calibration_probabilities = _predict(model, values, groups["cal"])
    candidates: list[tuple[dict, pd.DataFrame]] = []
    for threshold in SCORE_THRESHOLDS:
        metrics, ledger = _evaluate(frame, groups["cal"], calibration_probabilities, threshold)
        candidates.append(({"score_threshold": threshold, **metrics}, ledger))
    pd.DataFrame([row for row, _ in candidates]).to_csv(OUT / "calibration_candidates.csv", index=False)
    eligible = [(row, ledger) for row, ledger in candidates if row["pnl"] > 0.0 and row["trades"] >= MIN_CAL_TRADES]
    report = {
        "architecture": "btc110_temporal_causal_tcn_classifier",
        "layers": {"input": "24 bars × 110 causal features", "market": "24×94→Linear(94→48)→causal TCN(k=3,dilation=1/2/4,residual,LayerNorm,GELU,Dropout(0.1))→last 48", "context": "current 16→32→LayerNorm→GELU→Dropout(0.1)", "fusion": "80→64→GELU→Dropout(0.1)→residual(64)→LayerNorm→GELU", "output": "64→3 [FLAT, SHORT, LONG]"},
        "feature_contract": {"market_causalfix": 94, "context_regime3_dvol_onchain": 16, "total": 110, "sequence_bars": SEQUENCE_BARS},
        "label_contract": {"name": "fixed_forward_path_mfe_mae_net_utility", "horizon_bars": LABEL_HORIZON, "future_path_used_only_as_training_target": True},
        "execution_contract": {"entry": "decision t+1 open", "tp_price_move": TP_MOVE, "sl_price_move": SL_MOVE, "max_hold_bars": EXEC_HORIZON, "margin_fraction": MARGIN, "leverage": LEVERAGE, "notional": MARGIN * LEVERAGE, "roundtrip_cost_rate": COST},
        "model_validation": history,
        "split_rows": {name: int(len(rows)) for name, rows in groups.items()},
        "calibration_candidates": [row for row, _ in candidates],
        "contracts": {"fresh_forward_bar_by_bar": True, "thresholds_fit_on_calibration_only": True, "trade_ledgers_used_as_input": False, "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False, "split_targets_purged": True, "sequence_uses_only_current_and_past_rows": True, "causal_tcn_has_no_right_padding": True, "single_position": True, "bar_level_mark_to_market": True, "regime3_pred_inputs_forbidden": True, "test_used_for_selection": False},
        "promotion_eligible": False,
    }
    if not eligible:
        report.update({"result": "NO_CALIBRATION_CANDIDATE_PASSED_GATE", "test_metrics": None, "promotion_blockers": ["no positive calibration candidate with at least 30 trades"]})
        (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(report["result"])
        return 0
    selected, calibration_ledger = max(eligible, key=lambda item: item[0]["pnl"])
    calibration_ledger.to_csv(OUT / "selected_calibration_ledger.csv", index=False)
    test_metrics, test_ledger = _evaluate(frame, groups["test"], _predict(model, values, groups["test"]), selected["score_threshold"])
    test_ledger.to_csv(OUT / "test_ledger.csv", index=False)
    report.update({"result": "CALIBRATION_GATE_PASSED", "selected_config": selected, "test_metrics": test_metrics, "promotion_blockers": ["test period previously inspected", "research-only artifact lineage"]})
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({"selected": selected, "test": test_metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
