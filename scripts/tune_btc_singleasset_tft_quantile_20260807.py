"""Validation-only single-asset BTC TFT, quantile head only (reduced from the multi-asset design).

Cross-asset attention and the regime/entry/exit classification heads are dropped for this pass.
The model consumes only BTC's own OHLCV-derived features plus BTC-wide global (on-chain/DVOL)
features and forecasts a distribution of horizon-ahead log returns. Interpretability (VSN weights)
is diagnostic only, never a trading signal or a promotion criterion by itself.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from core.causal_futures_backtest import purged_decision_mask, simulate_single_position
from core.multiasset_tft import GatedResidualNetwork, VariableSelectionNetwork
from scripts.train_rho1_panel_backbone_20260804 import FEATURE_COLS

ASSET_PATH = ROOT / "data/panel/features/BTCUSDT.parquet"
GLOBAL_PATH = ROOT / "data/splits/year_oos/btc_unified_raw_panel_20260804.parquet"
DEFAULT_OUT = ROOT / "tmp/btc_singleasset_tft_quantile_20260807"
GLOBAL_COLS = [
    "dvol_btc", "dvol_btc_pctrank_720h", "dvol_btc_roc_24h",
    "mvrv", "net_exchange_flow_pct_supply", "active_addr_roc_7d",
]
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
TRAIN_END = pd.Timestamp("2025-09-01")
VAL_END = pd.Timestamp("2026-01-01")
TEST_END = pd.Timestamp("2026-04-01")
ROUNDTRIP_COST_RATE = 0.001  # 10bps, matches this repo's standing cost convention


@dataclass(frozen=True)
class Config:
    name: str
    d_model: int
    dropout: float
    learning_rate: float


CONFIGS = (
    Config("small", d_model=32, dropout=0.10, learning_rate=3e-4),
    Config("medium", d_model=48, dropout=0.10, learning_rate=2e-4),
)


def _timestamp(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True).dt.tz_localize(None)


def pinball_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = target[:, None] - prediction
    q = torch.tensor(QUANTILES, device=prediction.device)[None]
    return torch.maximum(q * error, (q - 1.0) * error).mean()


class SingleAssetQuantileTFT(nn.Module):
    """TFT reduced to one asset and one quantile head (no cross-asset attention, no action heads)."""

    def __init__(
        self, *, n_asset_features: int, quantile_count: int,
        d_model: int = 64, n_heads: int = 4, dropout: float = 0.1,
        n_global_features: int = 0,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_asset_features = int(n_asset_features)
        self.n_global_features = int(n_global_features)
        self.asset_vsn = VariableSelectionNetwork(n_asset_features, d_model, d_model * 2, dropout)
        self.global_vsn = (
            VariableSelectionNetwork(n_global_features, d_model, d_model * 2, dropout)
            if n_global_features else None
        )
        self.temporal_lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.temporal_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.temporal_grn = GatedResidualNetwork(d_model, d_model * 2, d_model, dropout)
        fusion_input = d_model * 2 if n_global_features else d_model
        self.fusion_grn = GatedResidualNetwork(fusion_input, d_model * 2, d_model, dropout)
        self.quantile_base = nn.Linear(d_model, 1)
        self.quantile_steps = nn.Linear(d_model, quantile_count - 1)

    def forward(self, asset_history: torch.Tensor, global_history: torch.Tensor | None = None):
        if asset_history.ndim != 3 or asset_history.shape[-1] != self.n_asset_features:
            raise ValueError("asset_history must be [batch,time,feature]")
        if (global_history is None) != (self.global_vsn is None):
            raise ValueError("global_history must be supplied exactly when n_global_features is nonzero")
        selected, asset_weights = self.asset_vsn(asset_history)
        lstm, _ = self.temporal_lstm(selected)
        steps = asset_history.shape[1]
        causal_mask = torch.triu(torch.ones(steps, steps, device=lstm.device, dtype=torch.bool), diagonal=1)
        attended, _ = self.temporal_attention(lstm, lstm, lstm, attn_mask=causal_mask, need_weights=False)
        temporal = self.temporal_grn(attended[:, -1] + lstm[:, -1])
        fused, global_weights = temporal, None
        if self.global_vsn is not None:
            global_selected, global_weights = self.global_vsn(global_history)
            fused = self.fusion_grn(torch.cat([temporal, global_selected[:, -1]], dim=-1))
        base = self.quantile_base(fused)
        increments = F.softplus(self.quantile_steps(fused))
        quantiles = torch.cat([base, base + torch.cumsum(increments, dim=-1)], dim=-1)
        return quantiles, asset_weights, global_weights


class PanelDataset(Dataset):
    def __init__(self, store: "PanelStore", indices: np.ndarray) -> None:
        self.store, self.indices = store, np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, row: int):
        end = int(self.indices[row])
        start = end - self.store.lookback + 1
        return (
            torch.from_numpy(self.store.asset_features[start : end + 1]),
            torch.from_numpy(self.store.global_features[start : end + 1]),
            torch.tensor(self.store.forward_return[end]),
            torch.tensor(end, dtype=torch.int64),
        )


class PanelStore:
    def __init__(self, lookback: int, horizon: int) -> None:
        self.lookback, self.horizon = int(lookback), int(horizon)
        frame = pd.read_parquet(ASSET_PATH, columns=["timestamp", "open", "high", "low", "close", *FEATURE_COLS])
        frame["timestamp"] = _timestamp(frame["timestamp"])
        frame = frame.drop_duplicates("timestamp").set_index("timestamp").sort_index()
        self.timestamps = frame.index
        global_frame = pd.read_parquet(GLOBAL_PATH, columns=["timestamp", *GLOBAL_COLS])
        global_frame["timestamp"] = _timestamp(global_frame["timestamp"])
        global_frame = global_frame.drop_duplicates("timestamp").set_index("timestamp").reindex(self.timestamps)

        self.open = frame["open"].to_numpy(np.float64)
        self.high = frame["high"].to_numpy(np.float64)
        self.low = frame["low"].to_numpy(np.float64)
        self.close = frame["close"].to_numpy(np.float64)
        asset = frame[FEATURE_COLS].to_numpy(np.float32, copy=True)
        global_features = global_frame[GLOBAL_COLS].to_numpy(np.float32, copy=True)

        entry_open = np.roll(self.open, -1)
        future_close = np.roll(self.close, -self.horizon)
        self.forward_return = np.log(future_close / entry_open).astype(np.float32)
        self.forward_return[-self.horizon :] = np.nan

        train_mask = self.timestamps < TRAIN_END
        for values in (asset, global_features):
            mean = np.nanmean(values[train_mask], axis=0, keepdims=True)
            std = np.nanstd(values[train_mask], axis=0, keepdims=True)
            std = np.where(std > 1e-6, std, 1.0)
            values -= mean
            values /= std
            np.nan_to_num(values, copy=False, nan=0.0, posinf=10.0, neginf=-10.0)
            np.clip(values, -10.0, 10.0, out=values)
        self.asset_features = asset
        self.global_features = global_features

    def indices(self, start: pd.Timestamp, end: pd.Timestamp, stride: int) -> np.ndarray:
        mask = purged_decision_mask(self.timestamps, start=start, end=end, horizon_bars=self.horizon)
        eligible = np.arange(self.lookback - 1, len(self.timestamps) - self.horizon, stride)
        return eligible[mask[eligible] & np.isfinite(self.forward_return[eligible])]


def _epoch(model: SingleAssetQuantileTFT, loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer | None) -> dict[str, float]:
    model.train(optimizer is not None)
    total, batches = 0.0, 0
    for asset, global_, target, _ in loader:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        quantiles, _, _ = model(asset.to(device), global_.to(device))
        loss = pinball_loss(quantiles, target.to(device))
        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += float(loss.detach().cpu())
        batches += 1
    return {"pinball": total / max(batches, 1)}


def _collect(model: SingleAssetQuantileTFT, loader: DataLoader, device: torch.device):
    model.eval()
    all_q, all_y, all_idx = [], [], []
    with torch.no_grad():
        for asset, global_, target, end_idx in loader:
            quantiles, _, _ = model(asset.to(device), global_.to(device))
            all_q.append(quantiles.cpu().numpy())
            all_y.append(target.numpy())
            all_idx.append(end_idx.numpy())
    return np.concatenate(all_q), np.concatenate(all_y), np.concatenate(all_idx)


def _quantile_diagnostics(q: np.ndarray, y: np.ndarray) -> dict[str, float]:
    median = q[:, len(QUANTILES) // 2]
    coverage = {f"coverage_q{int(ql*100):02d}": float((y <= q[:, i]).mean()) for i, ql in enumerate(QUANTILES)}
    return {
        "rows": int(len(y)),
        "pinball": float(np.mean(np.maximum(np.asarray(QUANTILES)[None] * (y[:, None] - q), (np.asarray(QUANTILES)[None] - 1.0) * (y[:, None] - q)))),
        "median_direction_accuracy": float((np.sign(median) == np.sign(y)).mean()),
        **coverage,
    }


def _execution_backtest(store: PanelStore, q: np.ndarray, end_idx: np.ndarray) -> dict[str, float]:
    median = q[:, len(QUANTILES) // 2].astype(np.float64)
    order = np.argsort(end_idx)
    decision_indices, scores = end_idx[order], median[order]
    huge = np.full(len(decision_indices), 1.0)  # tp/sl set unreachable so every trade times out at horizon
    result = simulate_single_position(
        timestamps=store.timestamps, open_px=store.open, high=store.high, low=store.low, close=store.close,
        decision_indices=decision_indices, scores=scores, tp_moves=huge, sl_moves=huge,
        upper_threshold=0.0, lower_threshold=0.0, horizon_bars=store.horizon,
        margin_fraction=1.0, leverage=1.0, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
    )
    ledger = result.ledger
    if len(ledger) == 0:
        return {"trades": 0}
    returns = ledger["trade_return"].to_numpy(np.float64)
    win_rate = float((returns > 0).mean())
    return {
        "trades": int(len(ledger)),
        "skipped_while_open": int(result.skipped_while_open),
        "win_rate": win_rate,
        "pnl_sum_pct": float(returns.sum() * 100.0),
        "pnl_compound_pct": float((np.prod(1.0 + returns) - 1.0) * 100.0),
        "mdd_pct": float((result.equity / np.maximum.accumulate(result.equity) - 1.0).min() * 100.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lookback", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--config", choices=[cfg.name for cfg in CONFIGS], default="", help="run one named configuration")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = PanelStore(args.lookback, args.horizon)
    train_idx = store.indices(store.timestamps[0], TRAIN_END, args.stride)
    val_idx = store.indices(TRAIN_END, VAL_END, args.stride)
    test_idx = store.indices(VAL_END, TEST_END, args.stride)
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError("one or more required split has no eligible rows")
    train = DataLoader(PanelDataset(store, train_idx), batch_size=args.batch_size, shuffle=True)
    val = DataLoader(PanelDataset(store, val_idx), batch_size=args.batch_size * 2)
    test = DataLoader(PanelDataset(store, test_idx), batch_size=args.batch_size * 2)
    configs = [cfg for cfg in CONFIGS if not args.config or cfg.name == args.config]
    trials = []
    for cfg in configs:
        model = SingleAssetQuantileTFT(
            n_asset_features=len(FEATURE_COLS), n_global_features=len(GLOBAL_COLS),
            quantile_count=len(QUANTILES), d_model=cfg.d_model, n_heads=4, dropout=cfg.dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
        best_state, best_val_loss, best_val_metrics = None, float("inf"), None
        for epoch in range(1, args.epochs + 1):
            train_metrics = _epoch(model, train, device, optimizer)
            val_q, val_y, _ = _collect(model, val, device)
            val_metrics = _quantile_diagnostics(val_q, val_y)
            print(f"{cfg.name} epoch={epoch} train={train_metrics} val_pinball={val_metrics['pinball']:.5f} val_dir_acc={val_metrics['median_direction_accuracy']:.4f}", flush=True)
            if val_metrics["pinball"] < best_val_loss:
                best_val_loss, best_val_metrics = val_metrics["pinball"], val_metrics
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        assert best_state is not None
        model.load_state_dict(best_state)
        trials.append((best_val_loss, best_val_metrics, cfg, model))
    best_val_loss, best_val_metrics, best_cfg, best_model = min(trials, key=lambda row: row[0])
    val_q, val_y, val_end_idx = _collect(best_model, val, device)
    test_q, test_y, test_end_idx = _collect(best_model, test, device)
    report = {
        "research_only": True, "selection_scope": "validation_only", "test_used_for_selection": False,
        "architecture": "single_asset_quantile_only_tft", "dropped_from_multiasset_design": ["cross_asset_attention", "regime_head", "entry_head", "exit_head"],
        "target": "horizon_log_return", "entry_timing": "next_bar_open", "horizon_bars": args.horizon,
        "asset_features": FEATURE_COLS, "global_features": GLOBAL_COLS,
        "lookback": args.lookback, "stride": args.stride,
        "split_rows": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "trials": [{"config": asdict(cfg), "validation": metrics} for value, metrics, cfg, _ in trials],
        "selected_config": asdict(best_cfg),
        "validation": _quantile_diagnostics(val_q, val_y),
        "test": _quantile_diagnostics(test_q, test_y),
        "execution_backtest_no_tpsl_fixed_horizon": {
            "validation": _execution_backtest(store, val_q, val_end_idx),
            "test": _execution_backtest(store, test_q, test_end_idx),
        },
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save({"state_dict": best_model.cpu().state_dict(), "config": asdict(best_cfg), "report": report}, args.output_dir / "best.pt")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
