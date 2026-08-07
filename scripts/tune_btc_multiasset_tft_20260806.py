"""Validation-only architecture tuning for the research-only BTC multi-asset TFT.

It trains return quantiles plus entry, regime, and exit-risk supervised heads from causal panel inputs.
Both oracle outcomes are targets only: neither may enter the feature tensor or select a threshold.
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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from core.causal_futures_backtest import purged_decision_mask
from core.multiasset_tft import MultiAssetTFT
from scripts.train_rho1_panel_backbone_20260804 import FEATURE_COLS
import build_wave3_action_labels_20260531 as zigzag
FEATURES_DIR = ROOT / "data/panel/features"
GLOBAL_PATH = ROOT / "data/splits/year_oos/btc_unified_raw_panel_20260804.parquet"
LABEL_PATH = ROOT / "data/splits/year_oos/btc_5m_tripbarrier_tradeoutcome_labels_20260806.parquet"
DEFAULT_OUT = ROOT / "tmp/btc_multiasset_tft_20260806"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT"]
GLOBAL_COLS = [
    "dvol_btc", "dvol_btc_pctrank_720h", "dvol_btc_roc_24h",
    "mvrv", "net_exchange_flow_pct_supply", "active_addr_roc_7d",
]
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
TRAIN_END = pd.Timestamp("2025-09-01")
VAL_END = pd.Timestamp("2026-01-01")
TEST_END = pd.Timestamp("2026-04-01")
EXIT_HORIZON_BARS = 12


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
            torch.from_numpy(self.store.asset_ids),
            torch.tensor(0, dtype=torch.long),  # BTC is the first explicitly pinned panel symbol.
            torch.from_numpy(self.store.global_features[start : end + 1]),
            torch.tensor(self.store.forward_return[end]),
            torch.tensor(self.store.zigzag_action[end]),
            torch.tensor(self.store.exit_action[end]),
            torch.tensor(self.store.trade_outcome[end]),
        )


class PanelStore:
    def __init__(self, lookback: int, horizon: int) -> None:
        self.lookback, self.horizon = int(lookback), int(horizon)
        frames: list[pd.DataFrame] = []
        for symbol in SYMBOLS:
            path = FEATURES_DIR / f"{symbol}.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"required panel symbol is missing: {path}")
            frame = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close", *FEATURE_COLS])
            frame["timestamp"] = _timestamp(frame["timestamp"])
            frames.append(frame.set_index("timestamp").sort_index())
        timestamps = frames[0].index
        for frame in frames[1:]:
            timestamps = timestamps.intersection(frame.index)
        self.timestamps = pd.DatetimeIndex(timestamps.sort_values())
        if len(self.timestamps) == 0:
            raise ValueError("panel has no common timestamps")
        asset = np.stack([frame.reindex(self.timestamps)[FEATURE_COLS].to_numpy(np.float32) for frame in frames], axis=1)
        btc = frames[0].reindex(self.timestamps)
        global_frame = pd.read_parquet(GLOBAL_PATH, columns=["timestamp", *GLOBAL_COLS])
        global_frame["timestamp"] = _timestamp(global_frame["timestamp"])
        global_frame = global_frame.drop_duplicates("timestamp").set_index("timestamp").reindex(self.timestamps)
        self.global_features = global_frame[GLOBAL_COLS].to_numpy(np.float32)
        labels = pd.read_parquet(LABEL_PATH, columns=["timestamp", "trade_outcome_action"])
        labels["timestamp"] = _timestamp(labels["timestamp"])
        label_by_time = labels.drop_duplicates("timestamp").set_index("timestamp").reindex(self.timestamps)
        self.trade_outcome = label_by_time["trade_outcome_action"].fillna(-1).to_numpy(np.int64)
        pivots = zigzag._zigzag_pivots(
            btc.reset_index(), min_reversal_pct=0.0035, atr_window=14, atr_multiplier=1.0,
        )
        zigzag_action = np.zeros(len(self.timestamps), dtype=np.int64)
        for (begin, _, begin_type), (finish, _, finish_type) in zip(pivots, pivots[1:]):
            if finish - begin < 3:
                continue
            if begin_type == "L" and finish_type == "H":
                zigzag_action[begin:finish] = 1
            elif begin_type == "H" and finish_type == "L":
                zigzag_action[begin:finish] = 2
        for row in np.flatnonzero(zigzag_action[1:] != zigzag_action[:-1]) + 1:
            zigzag_action[max(0, row - 1) : min(len(zigzag_action), row + 2)] = 0
        self.zigzag_action = zigzag_action
        exit_action = np.zeros(len(zigzag_action), dtype=np.int64)  # 0=hold, 1=exit, 2=flip
        for row, side in enumerate(zigzag_action):
            if side == 0:
                continue
            future = zigzag_action[row + 1 : row + 1 + EXIT_HORIZON_BARS]
            changed = future[future != side]
            if len(changed) == 0:
                continue
            exit_action[row] = 2 if changed[0] != 0 else 1
        self.exit_action = exit_action
        close = btc["close"].to_numpy(np.float64)
        entry_open = np.roll(btc["open"].to_numpy(np.float64), -1)
        future_close = np.roll(close, -self.horizon)
        self.forward_return = np.log(future_close / entry_open).astype(np.float32)
        self.forward_return[-self.horizon :] = np.nan
        train_mask = self.timestamps < TRAIN_END
        for values in (asset, self.global_features):
            mean = np.nanmean(values[train_mask], axis=(0, 1) if values.ndim == 3 else 0, keepdims=True)
            std = np.nanstd(values[train_mask], axis=(0, 1) if values.ndim == 3 else 0, keepdims=True)
            std = np.where(std > 1e-6, std, 1.0)
            values -= mean
            values /= std
            np.nan_to_num(values, copy=False, nan=0.0, posinf=10.0, neginf=-10.0)
            np.clip(values, -10.0, 10.0, out=values)
        self.asset_features = asset
        self.asset_ids = np.arange(len(SYMBOLS), dtype=np.int64)

    def indices(self, start: pd.Timestamp, end: pd.Timestamp, stride: int) -> np.ndarray:
        mask = purged_decision_mask(self.timestamps, start=start, end=end, horizon_bars=self.horizon)
        eligible = np.arange(self.lookback - 1, len(self.timestamps) - self.horizon, stride)
        return eligible[mask[eligible] & np.isfinite(self.forward_return[eligible])]


def _epoch(model: MultiAssetTFT, loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer | None) -> dict[str, float]:
    model.train(optimizer is not None)
    totals = np.zeros(4, dtype=np.float64)
    batches = 0
    for asset, ids, target_idx, global_, target, regime_target, exit_target, entry_target in loader:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        output = model(asset.to(device), ids.to(device), target_idx.to(device), global_.to(device))
        pinball = pinball_loss(output.quantiles, target.to(device))
        regime_loss = torch.nn.functional.cross_entropy(output.regime_logits, regime_target.to(device))
        exit_loss = torch.nn.functional.cross_entropy(output.exit_logits, exit_target.to(device))
        entry_loss = torch.nn.functional.cross_entropy(output.entry_logits, entry_target.to(device))
        loss = pinball + 0.25 * (regime_loss + exit_loss + entry_loss)
        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        totals += [float(pinball.detach().cpu()), float(regime_loss.detach().cpu()), float(exit_loss.detach().cpu()), float(entry_loss.detach().cpu())]
        batches += 1
    values = totals / max(batches, 1)
    return {"pinball": float(values[0]), "regime_ce": float(values[1]), "exit_ce": float(values[2]), "entry_ce": float(values[3]), "selection_loss": float(values[0] + .25 * (values[1] + values[2] + values[3]))}


def _test_diagnostics(model: MultiAssetTFT, loader: DataLoader, device: torch.device) -> dict[str, float | int]:
    model.eval()
    all_q, all_y = [], []
    with torch.no_grad():
        for asset, ids, target_idx, global_, target, regime_target, exit_target, entry_target in loader:
            output = model(asset.to(device), ids.to(device), target_idx.to(device), global_.to(device))
            all_q.append(output.quantiles.cpu().numpy())
            all_y.append(np.column_stack([target.numpy(), regime_target.numpy(), exit_target.numpy(), entry_target.numpy(), output.regime_logits.argmax(dim=1).cpu().numpy(), output.exit_logits.argmax(dim=1).cpu().numpy(), output.entry_logits.argmax(dim=1).cpu().numpy()]))
    q = np.concatenate(all_q)
    y = np.concatenate(all_y)
    valid = y[:, 3] >= 0
    direction = np.where(q[:, 3] >= 0.0, 1, 2)
    return {
        "rows": int(len(q)),
        "pinball": float(np.mean(np.maximum(np.asarray(QUANTILES)[None] * (y[:, :1] - q), (np.asarray(QUANTILES)[None] - 1.0) * (y[:, :1] - q)))),
        "median_direction_accuracy": float((np.where(y[:, 0] >= 0, 1, 2) == direction).mean()),
        "regime_action_accuracy": float((y[:, 1] == y[:, 4]).mean()),
        "exit_action_accuracy": float((y[:, 2] == y[:, 5]).mean()),
        "entry_action_accuracy": float((y[valid, 3] == y[valid, 6]).mean()),
        "entry_direction_accuracy_ex_cash": float((direction[valid & (y[:, 3] > 0)] == y[valid & (y[:, 3] > 0), 3]).mean()) if (valid & (y[:, 3] > 0)).any() else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lookback", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--config", choices=[cfg.name for cfg in CONFIGS], default="", help="run one named configuration")
    parser.add_argument("--seed", type=int, default=20260806)
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
        model = MultiAssetTFT(n_asset_features=len(FEATURE_COLS), n_global_features=len(GLOBAL_COLS), n_assets=len(SYMBOLS), quantile_count=len(QUANTILES), d_model=cfg.d_model, n_heads=4, dropout=cfg.dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
        best_state, best_metrics, best_val = None, None, float("inf")
        for epoch in range(1, args.epochs + 1):
            train_metrics = _epoch(model, train, device, optimizer)
            val_metrics = _epoch(model, val, device, None)
            print(f"{cfg.name} epoch={epoch} train={train_metrics} val={val_metrics}", flush=True)
            if val_metrics["selection_loss"] < best_val:
                best_val, best_metrics, best_state = val_metrics["selection_loss"], val_metrics, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        assert best_state is not None
        model.load_state_dict(best_state)
        trials.append((best_val, best_metrics, cfg, model))
    best_val, best_metrics, best_cfg, best_model = min(trials, key=lambda row: row[0])
    report = {
        "research_only": True, "selection_scope": "validation_only", "test_used_for_selection": False,
        "oracle_targets": {"regime": "aggressive_zigzag_0.35pct_3bar", "entry": "causal_triple_barrier_24h", "exit": f"aggressive_zigzag_transition_within_{EXIT_HORIZON_BARS}_bars"}, "entry_timing": "next_bar_open",
        "symbols": SYMBOLS, "asset_features": FEATURE_COLS, "global_features": GLOBAL_COLS,
        "lookback": args.lookback, "horizon": args.horizon, "stride": args.stride,
        "split_rows": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "trials": [{"config": asdict(cfg), "validation": metrics} for value, metrics, cfg, _ in trials],
        "selected_config": asdict(best_cfg), "selected_validation": best_metrics,
        "test": _test_diagnostics(best_model, test, device),
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save({"state_dict": best_model.cpu().state_dict(), "config": asdict(best_cfg), "report": report}, args.output_dir / "best.pt")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
