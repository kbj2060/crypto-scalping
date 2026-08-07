"""Rho2: causal BTC temporal encoder plus contemporaneous cross-symbol attention.

This is the smallest architecture that tests the part of the original Rho1 design that was never
implemented: BTC's decision at bar i directly consumes other symbols' bar-i state. Membership is
the trailing-30-day quote-volume top 40 within the downloaded 60-symbol research universe.
Survivorship bias outside that downloaded universe remains explicit and blocks promotion.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from core.backtest_metrics import bar_level_performance  # noqa: E402
from core.causal_futures_backtest import (  # noqa: E402
    fit_tail_thresholds,
    purged_decision_mask,
    simulate_single_position,
)
import train_rho1_panel_backbone_20260804 as rho1  # noqa: E402
from build_btc_tau1_dvol_features_20260804 import TAU1_BTC_FEATURE_COLS  # noqa: E402

FEATURES_DIR = ROOT / "data/panel/features"
UNIVERSE_PATH = ROOT / "data/splits/panel_universe_symbols_20260804.json"
KLINES_DIR = ROOT / "binance_data/klines"
TAU1_FEATURE_PATH = ROOT / "data/splits/year_oos/btc_tau1_dvol_features_20260804.parquet"
CKPT_PATH = ROOT / "data/panel/ckpt/rho2_tau1_dvol_causal_best.pt"
OUT_DIR = ROOT / "tmp/rho2_tau1_dvol_causal_20260804"

TRAIN_END = pd.Timestamp("2025-09-01")
MODEL_VAL_START, MODEL_VAL_END = TRAIN_END, pd.Timestamp("2026-01-01")
CAL_START, CAL_END = MODEL_VAL_END, pd.Timestamp("2026-04-01")
TEST_START, TEST_END = CAL_END, pd.Timestamp("2026-08-01")
WINDOW_L, HORIZON_H = 96, 48
TOP_K, LIQUIDITY_WINDOW = 40, 30 * 288
QUANTILES = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], dtype=np.float32)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MARGIN_FRACTION, LEVERAGE = 0.30, 3.0
ROUNDTRIP_COST_RATE = (0.0005 + 0.0002) * 2.0
TB_MIN_TP, TB_MIN_SL = 0.006, 0.004
TPSL_CONFIGS = [("moderate", 0.75, 0.25), ("wide", 0.90, 0.10)]
TAIL_CONFIGS = [(0.80, 0.20), (0.90, 0.10), (0.95, 0.05)]


class PanelStore:
    def __init__(self) -> None:
        universe = json.loads(UNIVERSE_PATH.read_text())
        self.symbols = [row["symbol"] for row in universe["symbols"]]
        self.symbol_to_id = {symbol: i for i, symbol in enumerate(self.symbols)}
        self.btc_id = self.symbol_to_id["BTCUSDT"]

        btc = pd.read_parquet(FEATURES_DIR / "BTCUSDT.parquet")
        self.timestamps = pd.DatetimeIndex(btc["timestamp"])
        self.btc_high = btc["high"].to_numpy(dtype=np.float64)
        self.btc_low = btc["low"].to_numpy(dtype=np.float64)
        n_time, n_symbols, n_features = len(btc), len(self.symbols), len(rho1.FEATURE_COLS)
        self.features = np.full((n_time, n_symbols, n_features), np.nan, dtype=np.float32)
        self.open_px = np.full((n_time, n_symbols), np.nan, dtype=np.float64)
        self.close = np.full((n_time, n_symbols), np.nan, dtype=np.float64)
        liquidity = np.full((n_time, n_symbols), -np.inf, dtype=np.float64)

        for symbol_id, symbol in enumerate(self.symbols):
            frame = pd.read_parquet(
                FEATURES_DIR / f"{symbol}.parquet",
                columns=["timestamp", "open", "close", *rho1.FEATURE_COLS],
            ).set_index("timestamp").reindex(self.timestamps)
            self.features[:, symbol_id] = frame[rho1.FEATURE_COLS].to_numpy(dtype=np.float32)
            self.open_px[:, symbol_id] = frame["open"].to_numpy(dtype=np.float64)
            self.close[:, symbol_id] = frame["close"].to_numpy(dtype=np.float64)

            volume = pd.read_csv(
                KLINES_DIR / symbol / f"{symbol}-5m-api.csv",
                usecols=["timestamp", "quote_volume"],
            )
            volume["timestamp"] = pd.to_datetime(volume["timestamp"])
            quote_volume = volume.set_index("timestamp")["quote_volume"].reindex(self.timestamps)
            liquidity[:, symbol_id] = (
                quote_volume.rolling(LIQUIDITY_WINDOW, min_periods=7 * 288).sum().to_numpy()
            )
            print(f"loaded {symbol_id + 1:2d}/{n_symbols} {symbol}", flush=True)

        tau1 = pd.read_parquet(TAU1_FEATURE_PATH, columns=["timestamp", *TAU1_BTC_FEATURE_COLS])
        # An hour labelled H contains trades from [H, H+1), so all of its factors become
        # available only at H+1.  merge_asof prevents their use at an earlier 5m decision bar.
        tau1["timestamp"] = (
            pd.to_datetime(tau1["timestamp"]).astype("datetime64[ns]") + pd.Timedelta(hours=1)
        )
        decision_timestamps = pd.Series(self.timestamps).astype("datetime64[ns]")
        tau1 = pd.merge_asof(
            pd.DataFrame({"timestamp": decision_timestamps}), tau1.sort_values("timestamp"),
            on="timestamp", direction="backward",
        )
        self.tau1_features = tau1[TAU1_BTC_FEATURE_COLS].to_numpy(dtype=np.float32)

        safe_liquidity = np.nan_to_num(liquidity, nan=-np.inf)
        self.active_ids = np.argpartition(safe_liquidity, -TOP_K, axis=1)[:, -TOP_K:].astype(np.int64)
        has_btc = (self.active_ids == self.btc_id).any(axis=1)
        self.active_ids[~has_btc, -1] = self.btc_id

        future_close = np.roll(self.close, -HORIZON_H, axis=0)
        entry_open = np.roll(self.open_px, -1, axis=0)
        self.forward_returns = np.log(future_close / entry_open)
        self.forward_returns[-HORIZON_H:] = np.nan
        active_returns = np.take_along_axis(self.forward_returns, self.active_ids, axis=1)
        btc_returns = self.forward_returns[:, self.btc_id]
        self.rank_target = np.nanmean(active_returns <= btc_returns[:, None], axis=1).astype(np.float32)
        self.raw_target = btc_returns.astype(np.float32)

        train_mask = purged_decision_mask(
            self.timestamps, start=self.timestamps[0], end=TRAIN_END, horizon_bars=HORIZON_H
        )
        train_rows = np.flatnonzero(train_mask)[::12]
        sample = self.features[train_rows]
        self.feature_mean = np.nanmean(sample, axis=(0, 1)).astype(np.float32)
        self.feature_std = np.nanstd(sample, axis=(0, 1)).astype(np.float32)
        self.feature_std = np.where(self.feature_std > 1e-6, self.feature_std, 1.0)
        self.features = np.nan_to_num(
            (self.features - self.feature_mean[None, None, :])
            / self.feature_std[None, None, :],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clip(-10.0, 10.0).astype(np.float32)
        self.tau1_mean = np.nanmean(self.tau1_features[train_rows], axis=0).astype(np.float32)
        self.tau1_std = np.nanstd(self.tau1_features[train_rows], axis=0).astype(np.float32)
        self.tau1_std = np.where(self.tau1_std > 1e-6, self.tau1_std, 1.0)
        self.tau1_features = np.nan_to_num(
            (self.tau1_features - self.tau1_mean[None]) / self.tau1_std[None],
            nan=0.0, posinf=0.0, neginf=0.0,
        ).clip(-10.0, 10.0).astype(np.float32)

    def indices(self, start: pd.Timestamp, end: pd.Timestamp, stride: int) -> np.ndarray:
        mask = purged_decision_mask(
            self.timestamps, start=start, end=end, horizon_bars=HORIZON_H
        )
        valid = np.isfinite(self.raw_target) & np.isfinite(self.rank_target)
        idxs = np.arange(WINDOW_L - 1, len(self.timestamps) - HORIZON_H, stride)
        return idxs[mask[idxs] & valid[idxs]]


class CrossSymbolDataset(Dataset):
    def __init__(self, store: PanelStore, indices: np.ndarray) -> None:
        self.store = store
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        i = int(self.indices[item])
        active_ids = self.store.active_ids[i]
        return (
            torch.from_numpy(self.store.tau1_features[i - WINDOW_L + 1 : i + 1]),
            torch.from_numpy(self.store.features[i, active_ids]),
            torch.from_numpy(active_ids),
            torch.tensor(self.store.rank_target[i]),
            torch.tensor(self.store.raw_target[i]),
            torch.tensor(float(self.store.raw_target[i] > 0.0)),
        )


class Rho2CrossSymbol(nn.Module):
    def __init__(self, n_temporal_features: int, n_snapshot_features: int, n_symbols: int, btc_id: int, d_model: int = 64) -> None:
        super().__init__()
        self.btc_id = btc_id
        self.temporal_feature_proj = nn.Linear(n_temporal_features, d_model)
        self.snapshot_feature_proj = nn.Linear(n_snapshot_features, d_model)
        self.symbol_emb = nn.Embedding(n_symbols, d_model)
        self.temporal_pos = nn.Parameter(torch.randn(1, WINDOW_L, d_model) * 0.02)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 3,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        cross_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 3,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=2)
        self.cross_encoder = nn.TransformerEncoder(cross_layer, num_layers=2)
        self.fuse = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(0.1))
        self.rank_head = nn.Linear(d_model, 1)
        self.direction_head = nn.Linear(d_model, 1)
        self.quantile_head = nn.Linear(d_model, len(QUANTILES))

    def forward(
        self, btc_window: torch.Tensor, panel_snapshot: torch.Tensor, symbol_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        temporal = self.temporal_encoder(self.temporal_feature_proj(btc_window) + self.temporal_pos)[:, -1]
        cross = self.cross_encoder(self.snapshot_feature_proj(panel_snapshot) + self.symbol_emb(symbol_ids))
        btc_position = (symbol_ids == self.btc_id).to(torch.int64).argmax(dim=1)
        btc_cross = cross[torch.arange(len(cross), device=cross.device), btc_position]
        fused = self.fuse(torch.cat([temporal, btc_cross], dim=1))
        rank = torch.sigmoid(self.rank_head(fused).squeeze(1))
        direction_logit = self.direction_head(fused).squeeze(1)
        quantiles = torch.sort(self.quantile_head(fused), dim=1).values
        return direction_logit, rank, quantiles


def _pinball(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    errors = target[:, None] - pred
    q = torch.as_tensor(QUANTILES, device=pred.device)[None]
    return torch.maximum(q * errors, (q - 1.0) * errors).mean()


def _run_epoch(model: Rho2CrossSymbol, loader: DataLoader, optimizer=None) -> dict:
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(4, dtype=np.float64)
    for btc_window, panel_snapshot, symbol_ids, rank_target, raw_target, direction_target in loader:
        btc_window = btc_window.to(DEVICE)
        panel_snapshot = panel_snapshot.to(DEVICE)
        symbol_ids = symbol_ids.to(DEVICE)
        rank_target = rank_target.to(DEVICE)
        raw_target = raw_target.to(DEVICE)
        direction_target = direction_target.to(DEVICE)
        if training:
            optimizer.zero_grad()
        direction_logit, rank, quantiles = model(btc_window, panel_snapshot, symbol_ids)
        direction_loss = nn.functional.binary_cross_entropy_with_logits(
            direction_logit, direction_target
        )
        rank_loss = nn.functional.mse_loss(rank, rank_target)
        quantile_loss = _pinball(quantiles, raw_target)
        loss = direction_loss + 0.5 * rank_loss + 20.0 * quantile_loss
        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        totals += [loss.item(), direction_loss.item(), rank_loss.item(), quantile_loss.item()]
    totals /= max(len(loader), 1)
    return {
        "loss": float(totals[0]),
        "direction_bce": float(totals[1]),
        "rank_mse": float(totals[2]),
        "pinball": float(totals[3]),
    }


def _predict(model: Rho2CrossSymbol, dataset: CrossSymbolDataset) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    directions, quantiles = [], []
    model.eval()
    with torch.no_grad():
        for btc_window, panel_snapshot, symbol_ids, _, _, _ in loader:
            direction_logit, _, quantile = model(
                btc_window.to(DEVICE), panel_snapshot.to(DEVICE), symbol_ids.to(DEVICE)
            )
            directions.append(torch.sigmoid(direction_logit).cpu().numpy())
            quantiles.append(quantile.cpu().numpy())
    return np.concatenate(directions), np.concatenate(quantiles)


def _directional_moves(quantiles: np.ndarray, tp_q: float, sl_q: float):
    q_index = {round(float(q), 2): i for i, q in enumerate(QUANTILES)}
    tp_i, sl_i = q_index[round(tp_q, 2)], q_index[round(sl_q, 2)]
    long_tp = np.maximum(TB_MIN_TP, quantiles[:, tp_i])
    long_sl = np.maximum(TB_MIN_SL, -quantiles[:, sl_i])
    short_tp = np.maximum(TB_MIN_TP, -quantiles[:, sl_i])
    short_sl = np.maximum(TB_MIN_SL, quantiles[:, tp_i])
    return long_tp, long_sl, short_tp, short_sl


def _evaluate(
    store: PanelStore,
    indices: np.ndarray,
    scores: np.ndarray,
    moves,
    upper: float,
    lower: float,
):
    long_tp, long_sl, short_tp, short_sl = moves
    is_long = scores >= upper
    result = simulate_single_position(
        timestamps=store.timestamps,
        open_px=store.open_px[:, store.btc_id],
        high=store.btc_high,
        low=store.btc_low,
        close=store.close[:, store.btc_id],
        decision_indices=indices,
        scores=scores,
        tp_moves=np.where(is_long, long_tp, short_tp),
        sl_moves=np.where(is_long, long_sl, short_sl),
        upper_threshold=upper,
        lower_threshold=lower,
        horizon_bars=HORIZON_H,
        margin_fraction=MARGIN_FRACTION,
        leverage=LEVERAGE,
        roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
    )
    metrics = bar_level_performance(result.equity, result.ledger)
    metrics["mean_trade_return_pct"] = (
        float(result.ledger["trade_return"].mean() * 100.0) if len(result.ledger) else 0.0
    )
    metrics["skipped_while_open"] = result.skipped_while_open
    return metrics, result.ledger


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device={DEVICE}", flush=True)
    store = PanelStore()
    train_idx = store.indices(store.timestamps[0], TRAIN_END, stride=8)
    val_idx = store.indices(MODEL_VAL_START, MODEL_VAL_END, stride=4)
    cal_idx = store.indices(CAL_START, CAL_END, stride=1)
    test_idx = store.indices(TEST_START, TEST_END, stride=1)
    print(
        f"train={len(train_idx)} val={len(val_idx)} calibration={len(cal_idx)} test={len(test_idx)}",
        flush=True,
    )

    model = Rho2CrossSymbol(
        len(TAU1_BTC_FEATURE_COLS), len(rho1.FEATURE_COLS), len(store.symbols), store.btc_id
    ).to(DEVICE)
    if not args.eval_only:
        train_loader = DataLoader(
            CrossSymbolDataset(store, train_idx), batch_size=128, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            CrossSymbolDataset(store, val_idx), batch_size=256, shuffle=False, num_workers=0
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        best_val, bad_epochs = float("inf"), 0
        for epoch in range(1, 13):
            started = time.time()
            train_metrics = _run_epoch(model, train_loader, optimizer)
            val_metrics = _run_epoch(model, val_loader)
            print(
                f"epoch={epoch:02d} train={train_metrics} val={val_metrics} seconds={time.time()-started:.1f}",
                flush=True,
            )
            if val_metrics["loss"] < best_val - 1e-5:
                best_val, bad_epochs = val_metrics["loss"], 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "symbols": store.symbols,
                        "temporal_feature_cols": TAU1_BTC_FEATURE_COLS,
                        "snapshot_feature_cols": rho1.FEATURE_COLS,
                        "feature_mean": store.feature_mean,
                        "feature_std": store.feature_std,
                        "tau1_mean": store.tau1_mean,
                        "tau1_std": store.tau1_std,
                        "val_metrics": val_metrics,
                    },
                    CKPT_PATH,
                )
            else:
                bad_epochs += 1
                if bad_epochs >= 3:
                    break

    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    cal_scores, cal_quantiles = _predict(model, CrossSymbolDataset(store, cal_idx))
    test_scores, test_quantiles = _predict(model, CrossSymbolDataset(store, test_idx))

    candidates = []
    calibration_rows = []
    for tpsl, tp_q, sl_q in TPSL_CONFIGS:
        cal_moves = _directional_moves(cal_quantiles, tp_q, sl_q)
        for upper_q, lower_q in TAIL_CONFIGS:
            thresholds = fit_tail_thresholds(
                cal_scores, upper_quantile=upper_q, lower_quantile=lower_q
            )
            metrics, ledger = _evaluate(
                store, cal_idx, cal_scores, cal_moves, thresholds.upper, thresholds.lower
            )
            row = {
                "tpsl": tpsl, "tp_quantile": tp_q, "sl_quantile": sl_q,
                "upper_quantile": upper_q, "lower_quantile": lower_q,
                "upper_threshold": thresholds.upper, "lower_threshold": thresholds.lower,
                **metrics,
            }
            calibration_rows.append(row)
            candidates.append((metrics["pnl"], row, ledger))

    calibration = pd.DataFrame(calibration_rows).sort_values("pnl", ascending=False)
    calibration.to_csv(OUT_DIR / "calibration_candidates.csv", index=False)
    _, selected, selected_cal_ledger = max(candidates, key=lambda item: item[0])
    selected_cal_ledger.to_csv(OUT_DIR / "selected_calibration_ledger.csv", index=False)
    test_moves = _directional_moves(
        test_quantiles, selected["tp_quantile"], selected["sl_quantile"]
    )
    test_metrics, test_ledger = _evaluate(
        store, test_idx, test_scores, test_moves,
        selected["upper_threshold"], selected["lower_threshold"],
    )
    test_ledger.to_csv(OUT_DIR / "test_ledger.csv", index=False)

    report = {
        "architecture": "rho2_tau1_39_feature_btc_temporal_plus_point_in_time_crosssymbol_attention",
        "temporal_feature_contract": {
            "count": len(TAU1_BTC_FEATURE_COLS),
            "cols": TAU1_BTC_FEATURE_COLS,
            "tau1_cross_asset_mapping": "ETH is BTC's external reference; BTC-ETH spread replaces ETH-BTC spread",
            "dvol": "BTC Deribit DVOL, hour-close available at timestamp plus one hour",
        },
        "model_validation": checkpoint["val_metrics"],
        "selected_config": selected,
        "test_metrics": test_metrics,
        "contracts": {
            "fresh_forward_bar_by_bar": True,
            "thresholds_fit_on_calibration_only": True,
            "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False,
            "future_rows_used_for_entry": False,
            "split_targets_purged": True,
            "single_position": True,
            "bar_level_mark_to_market": True,
            "point_in_time_liquidity_top_k": TOP_K,
            "margin_fraction": MARGIN_FRACTION,
            "leverage": LEVERAGE,
            "notional": MARGIN_FRACTION * LEVERAGE,
            "roundtrip_cost_rate": ROUNDTRIP_COST_RATE,
        },
        "promotion_eligible": False,
        "promotion_blockers": [
            "download universe excludes contracts delisted before 2026-08-04",
            "test period was previously inspected for Rho1 Stage-1 pinball loss",
        ],
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(calibration.to_string(index=False))
    print("\nSELECTED", json.dumps(selected, indent=2, default=str))
    print("\nFROZEN TEST", json.dumps(test_metrics, indent=2))
    print(f"wrote {OUT_DIR / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
