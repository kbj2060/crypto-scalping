#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_eval_hf_clean_regime_core_loop_20260511 import (  # noqa: E402
    _audit_contract,
    _close,
    _days,
    _feature_cols,
    _fill_price,
    _read,
)
from ensemble.fully_learned_governor_policy import FullyLearnedGovernorConfig, prepare_features  # noqa: E402


MODEL_ID = "hf_v13_deep_edge_surface_parent_v39_20260512"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_deep_edge_surface_parent_v39_20260512"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_deep_edge_surface_parent_v39_20260512_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_deep_edge_surface_parent_v39_20260512_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_deep_edge_surface_parent_v39_20260512_grid.csv"

SEQ_LEN = 72
MAX_LABEL_HOLD = 96
TARGET_SCALE = 50.0
MARGIN110_COST1 = 139.4071
V31_COST1 = 277.0679629973942


@dataclass(frozen=True)
class EdgeRuntimeConfig:
    name: str
    score_th: float
    margin_th: float
    q90_weight: float
    q10_weight: float
    adverse_weight: float
    base_notional: float
    max_notional: float
    risk_budget: float
    base_tp: float
    base_sl: float
    max_hold: int
    cooldown: int
    tp_mult: float = 1.15
    sl_mult: float = 1.10
    trail: bool = True
    vol_throttle: bool = True


class DeepEdgeSurfaceTCN(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 96) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        dim = input_dim
        for dilation in (1, 2, 4, 8, 16):
            blocks.extend(
                [
                    nn.Conv1d(dim, hidden, kernel_size=3, padding=dilation, dilation=dilation),
                    nn.GELU(),
                    nn.Dropout(0.08),
                ]
            )
            dim = hidden
        self.tcn = nn.Sequential(*blocks)
        self.context = nn.Sequential(
            nn.Linear(hidden * 2, 160),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(160, 96),
            nn.GELU(),
        )
        self.head = nn.Linear(96, 8)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.tcn(seq.transpose(1, 2))
        global_ctx = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        recent_ctx = x[:, :, -1]
        ctx = self.context(torch.cat([global_ctx, recent_ctx], dim=-1))
        return self.head(ctx)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _parent_cfg() -> FullyLearnedGovernorConfig:
    return FullyLearnedGovernorConfig(
        notional_buckets=(0.23, 0.368, 0.575, 0.8625, 1.2075, 1.6675, 2.3, 3.105, 4.14),
        leverage_buckets=(1.5, 2.0, 3.0, 4.0, 5.0),
        take_profit_buckets=(0.007, 0.011, 0.018, 0.030, 0.050, 0.090, 0.180, 0.450, 0.900),
        stop_loss_buckets=(0.004, 0.006, 0.009, 0.014, 0.022, 0.035, 0.055),
        max_hold_buckets=(6, 12, 24, 48, 96, 192, 288),
        cooldown_buckets=(0, 1, 3, 6, 12, 24, 48),
        max_train_horizon_bars=288,
        cash_score=0.020,
        adverse_penalty=2.45,
        size_penalty=0.180,
        hold_penalty=0.042,
        turnover_bonus=0.0012,
        max_margin_fraction=1.10,
    )


def _runtime_grid() -> list[EdgeRuntimeConfig]:
    rows: list[EdgeRuntimeConfig] = []
    profiles = [
        ("balanced", 0.002, 0.0015, 0.35, 0.15, 0.45, 0.80, 1.40, 0.014, 0.034, 0.016, 48),
        ("precision", 0.005, 0.0030, 0.35, 0.20, 0.70, 0.70, 1.20, 0.012, 0.032, 0.014, 48),
        ("aggressive", 0.000, 0.0010, 0.45, 0.10, 0.35, 1.00, 1.80, 0.018, 0.040, 0.018, 60),
        ("convex", 0.003, 0.0020, 0.65, 0.10, 0.50, 0.90, 1.70, 0.016, 0.045, 0.018, 72),
        ("defensive", 0.004, 0.0025, 0.25, 0.25, 0.85, 0.70, 1.10, 0.010, 0.030, 0.012, 36),
    ]
    for name, score, margin, q90w, q10w, advw, base_n, max_n, risk, tp, sl, hold in profiles:
        for offset in (-0.0015, 0.0, 0.0015):
            rows.append(
                EdgeRuntimeConfig(
                    name=f"v39_{name}_s{score + offset:.4f}",
                    score_th=max(-0.001, score + offset),
                    margin_th=margin,
                    q90_weight=q90w,
                    q10_weight=q10w,
                    adverse_weight=advw,
                    base_notional=base_n,
                    max_notional=max_n,
                    risk_budget=risk,
                    base_tp=tp,
                    base_sl=sl,
                    max_hold=hold,
                    cooldown=12,
                )
            )
    return rows


def _seq_tensor(features: pd.DataFrame, indices: np.ndarray, cols: list[str]) -> np.ndarray:
    arr = features.loc[:, cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    pad = np.zeros((SEQ_LEN - 1, arr.shape[1]), dtype=np.float32)
    padded = np.vstack([pad, arr])
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=SEQ_LEN, axis=0)
    if windows.shape[1] == arr.shape[1]:
        windows = windows.transpose(0, 2, 1)
    return np.ascontiguousarray(windows[indices])


def _normalizer(seq: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mean": np.nanmean(seq, axis=(0, 1)).astype(np.float32),
        "std": (np.nanstd(seq, axis=(0, 1)) + 1e-6).astype(np.float32),
    }


def _apply_norm(seq: np.ndarray, norm: dict[str, np.ndarray]) -> np.ndarray:
    return ((seq - norm["mean"][None, None, :]) / norm["std"][None, None, :]).astype(np.float32)


def _build_edge_targets(frame: pd.DataFrame, indices: np.ndarray, *, fee: float, slip: float) -> np.ndarray:
    open_px = pd.to_numeric(frame["open"], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().to_numpy(dtype=np.float64)
    horizons = np.arange(1, MAX_LABEL_HOLD + 1, dtype=np.int64)
    targets = np.zeros((len(indices), 8), dtype=np.float32)
    cost = 2.0 * float(fee + slip)
    for r, idx in enumerate(indices):
        entry_i = min(int(idx) + 1, len(open_px) - 1)
        exit_i = np.minimum(entry_i + horizons, len(open_px) - 1)
        long_entry = open_px[entry_i] * (1.0 + slip)
        short_entry = open_px[entry_i] * (1.0 - slip)
        long_exit = open_px[exit_i] * (1.0 - slip)
        short_exit = open_px[exit_i] * (1.0 + slip)
        long_path = long_exit / max(long_entry, 1e-12) - 1.0 - cost
        short_path = (short_entry - short_exit) / max(short_entry, 1e-12) - cost
        targets[r, 0:3] = np.quantile(long_path, (0.10, 0.50, 0.90)).astype(np.float32)
        targets[r, 3:6] = np.quantile(short_path, (0.10, 0.50, 0.90)).astype(np.float32)
        targets[r, 6] = max(0.0, -float(np.min(long_path)))
        targets[r, 7] = max(0.0, -float(np.min(short_path)))
    return targets


def _fit_model(seq: np.ndarray, y: np.ndarray, norm: dict[str, np.ndarray], *, epochs: int, seed: int) -> DeepEdgeSurfaceTCN:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    x = _apply_norm(seq, norm)
    y_scaled = (y * TARGET_SCALE).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepEdgeSurfaceTCN(x.shape[-1]).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y_scaled)), batch_size=256, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    huber = nn.SmoothL1Loss()
    model.train()
    for _ in range(int(epochs)):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            base = huber(pred, yb)
            long_mono = F.relu(pred[:, 0] - pred[:, 1]).mean() + F.relu(pred[:, 1] - pred[:, 2]).mean()
            short_mono = F.relu(pred[:, 3] - pred[:, 4]).mean() + F.relu(pred[:, 4] - pred[:, 5]).mean()
            adv_pos = F.relu(-pred[:, 6]).mean() + F.relu(-pred[:, 7]).mean()
            loss = base + 0.05 * (long_mono + short_mono) + 0.03 * adv_pos
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model.cpu().eval()


def _predict_all(model: DeepEdgeSurfaceTCN, features: pd.DataFrame, cols: list[str], norm: dict[str, np.ndarray]) -> np.ndarray:
    indices = np.arange(len(features), dtype=np.int64)
    seq = _seq_tensor(features, indices, cols)
    x = _apply_norm(seq, norm)
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 1024):
            chunks.append((model(torch.from_numpy(x[start : start + 1024])).numpy() / TARGET_SCALE).astype(np.float32))
    return np.vstack(chunks)


def _safe_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(col, default))
    except Exception:
        return float(default)
    return value if np.isfinite(value) else float(default)


def _vol_anchor(row: pd.Series) -> float:
    bbw = abs(_safe_float(row, "bb_width", 0.0))
    gk = abs(_safe_float(row, "garman_klass_vol", 0.0))
    rs = abs(_safe_float(row, "rogers_satchell_vol", 0.0))
    pk = abs(_safe_float(row, "parkinson_vol", 0.0))
    volz = abs(_safe_float(row, "volatility_z", 0.0))
    rv = abs(_safe_float(row, "realized_vol_ratio", 1.0))
    base = max(0.0015, bbw * 0.15, gk * 2.5, rs * 2.5, pk * 2.5)
    return float(np.clip(base * (1.0 + 0.08 * min(volz, 3.0) + 0.05 * max(rv - 1.0, 0.0)), 0.0015, 0.030))


def _score_pair(pred_row: np.ndarray, cfg: EdgeRuntimeConfig, *, fee: float, slip: float) -> tuple[float, float]:
    cost_buffer = 2.0 * (fee + slip)
    lq10, lq50, lq90, sq10, sq50, sq90, ladv, sadv = [float(x) for x in pred_row]
    long_score = lq50 + cfg.q90_weight * max(lq90, 0.0) + cfg.q10_weight * lq10 - cfg.adverse_weight * max(ladv, 0.0) - cost_buffer
    short_score = sq50 + cfg.q90_weight * max(sq90, 0.0) + cfg.q10_weight * sq10 - cfg.adverse_weight * max(sadv, 0.0) - cost_buffer
    return float(long_score), float(short_score)


def _entry_params(pred_row: np.ndarray, cfg: EdgeRuntimeConfig, side: int, row: pd.Series) -> tuple[float, float, float]:
    if side > 0:
        q90 = max(float(pred_row[2]), 0.0)
        adverse = max(float(pred_row[6]), 0.003)
    else:
        q90 = max(float(pred_row[5]), 0.0)
        adverse = max(float(pred_row[7]), 0.003)
    risk_cap = cfg.risk_budget / max(adverse, 0.004)
    tail_scale = float(np.clip(0.85 + q90 / 0.035, 0.75, 1.45))
    notional = min(cfg.max_notional, risk_cap, cfg.base_notional * tail_scale)
    if cfg.vol_throttle:
        va = _vol_anchor(row)
        if va > 0.020:
            notional *= 0.50
        elif va > 0.014:
            notional *= 0.70
    take_profit = float(np.clip(max(cfg.base_tp, q90 * notional * cfg.tp_mult), cfg.base_tp * 0.75, 0.085))
    stop_loss = float(np.clip(max(cfg.base_sl, adverse * notional * cfg.sl_mult), cfg.base_sl * 0.65, 0.040))
    return float(max(notional, 0.0)), take_profit, stop_loss


def backtest(
    df: pd.DataFrame,
    pred: np.ndarray,
    cfg: EdgeRuntimeConfig,
    *,
    fee: float,
    slip: float,
    record: bool = False,
) -> dict[str, Any]:
    close = _close(df)
    cash = peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = entry_equity = 0.0
    entry_idx = 0
    notional = take_profit = stop_loss = 0.0
    cooldown = 0
    trades = wins = long_entries = short_entries = 0
    notional_sum = 0.0
    exits: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    open_record: dict[str, Any] | None = None
    mfe = mae = 0.0

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
        unreal = raw * notional
        return cash * (1.0 + unreal), unreal

    for i in range(0, len(df) - 2):
        eq, unreal = mark(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if pos != 0:
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            hold = i - entry_idx
            eff_sl = stop_loss
            if cfg.trail and mfe > 0.0:
                gap = max(_vol_anchor(df.iloc[entry_idx]) * notional * 0.80, 0.004)
                if hold >= 18:
                    gap = max(gap * 0.35, gap - 0.025 * (hold - 18) * gap)
                eff_sl = min(eff_sl, max(0.001, mfe - gap))
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "take_profit"
            elif eff_sl > 0.0 and unreal <= -abs(eff_sl):
                reason = "stop_loss"
            elif hold >= int(cfg.max_hold):
                reason = "max_hold"
            if reason:
                fill_i = min(i + 1, len(df) - 1)
                exit_px = _fill_price(df, fill_i, pos, slip, entry=False)
                raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * fee * notional
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] = exits.get(reason, 0) + 1
                if record and open_record is not None:
                    out = dict(open_record)
                    out.update(
                        {
                            "exit_signal_timestamp": str(df["timestamp"].iloc[i]),
                            "exit_fill_timestamp": str(df["timestamp"].iloc[fill_i]),
                            "exit_reason": reason,
                            "realized_net_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0),
                            "mfe_pct": float(mfe * 100.0),
                            "mae_pct": float(mae * 100.0),
                            "cash_after": float(cash),
                        }
                    )
                    records.append(out)
                pos = 0
                cooldown = int(cfg.cooldown)
                open_record = None
                continue
        if pos != 0:
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        long_score, short_score = _score_pair(pred[i], cfg, fee=fee, slip=slip)
        best = max(long_score, short_score)
        if best < cfg.score_th or abs(long_score - short_score) < cfg.margin_th:
            continue
        side = 1 if long_score > short_score else -1
        n, tp, sl = _entry_params(pred[i], cfg, side, df.iloc[i])
        if n <= 0.05:
            continue
        fill_i = min(i + 1, len(df) - 1)
        pos = side
        entry_price = _fill_price(df, fill_i, pos, slip, entry=True)
        entry_equity = cash
        entry_idx = i
        notional = float(n)
        take_profit = float(tp)
        stop_loss = float(sl)
        cash -= cash * fee * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        mfe = mae = 0.0
        if record:
            open_record = {
                "entry_signal_timestamp": str(df["timestamp"].iloc[i]),
                "entry_fill_timestamp": str(df["timestamp"].iloc[fill_i]),
                "side": "LONG" if pos > 0 else "SHORT",
                "entry_price": float(entry_price),
                "notional_exposure": float(notional),
                "long_score": float(long_score),
                "short_score": float(short_score),
                "take_profit": float(take_profit),
                "stop_loss": float(stop_loss),
                "fee_entry_pct": float(fee * notional * 100.0),
            }
    if pos != 0:
        fill_i = len(df) - 1
        exit_px = _fill_price(df, fill_i, pos, slip, entry=False)
        raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * fee * notional
        trades += 1
        wins += int(cash > entry_equity)
        exits["forced_end"] = exits.get("forced_end", 0) + 1
    entries = max(long_entries + short_entries, 1)
    out: dict[str, Any] = {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(df)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(notional_sum / entries),
        "avg_leverage": 1.0,
        "exits": exits,
    }
    if record:
        out["trade_records"] = records
    return out


def _selection_score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1["trades"]) < 20:
        return -1e9 + float(c1["pnl"])
    return float(c1["pnl"] + 0.35 * c2["pnl"] + 0.20 * c3["pnl"] - 0.45 * abs(c1["mdd"]) + 0.08 * min(c1["trades"], 160))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deep edge-surface parent V39: no CASH label, continuous long/short utility heads.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--seed", type=int, default=2031)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[{MODEL_ID}] loading data", flush=True)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    split_ts = pd.Timestamp("2025-10-01")
    train_df = train_all[train_all["timestamp"] < split_ts].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= split_ts].reset_index(drop=True)
    feature_cols = _feature_cols(train_all, eval_df)
    audit = _audit_contract(train_all, eval_df, feature_cols)
    cfg = _parent_cfg()
    full_train_features = prepare_features(train_df, side_hint=0, close=_close(train_df), feature_cols=feature_cols)
    valid = np.arange(SEQ_LEN - 1, max(SEQ_LEN, len(train_df) - MAX_LABEL_HOLD - 2), max(1, int(args.stride)), dtype=np.int64)
    print(f"[{MODEL_ID}] building edge-surface labels rows={len(valid)} cols={len(feature_cols)}", flush=True)
    seq = _seq_tensor(full_train_features, valid, feature_cols)
    y = _build_edge_targets(train_df, valid, fee=cfg.fee, slip=cfg.slip)
    norm = _normalizer(seq)
    print(f"[{MODEL_ID}] training epochs={args.epochs} device={'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)
    model = _fit_model(seq, y, norm, epochs=int(args.epochs), seed=int(args.seed))
    print(f"[{MODEL_ID}] predicting validation/OOS", flush=True)
    val_features = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=feature_cols)
    eval_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    val_pred = _predict_all(model, val_features, feature_cols, norm)
    eval_pred = _predict_all(model, eval_features, feature_cols, norm)

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for runtime in _runtime_grid():
        print(f"[{MODEL_ID}] validation {runtime.name}", flush=True)
        v1 = backtest(val_df, val_pred, runtime, fee=cfg.fee, slip=cfg.slip)
        v2 = backtest(val_df, val_pred, runtime, fee=cfg.fee * 2.0, slip=cfg.slip * 2.0)
        v3 = backtest(val_df, val_pred, runtime, fee=cfg.fee * 3.0, slip=cfg.slip * 3.0)
        row = {
            "runtime": asdict(runtime),
            "validation_cost1": v1,
            "validation_cost2": v2,
            "validation_cost3": v3,
            "selection_score": _selection_score(v1, v2, v3),
        }
        rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    if best is None:
        raise RuntimeError("no runtime candidates")
    selected = EdgeRuntimeConfig(**best["runtime"])
    print(f"[{MODEL_ID}] selected {selected.name}; running OOS", flush=True)
    metrics: dict[str, Any] = {}
    ledger_path = args.report_out.with_name(f"{args.report_out.stem}_cost1_ledger.csv")
    for mult in (1, 2, 3):
        r = backtest(eval_df, eval_pred, selected, fee=cfg.fee * mult, slip=cfg.slip * mult, record=(mult == 1))
        if mult == 1:
            ledger = pd.DataFrame(r.pop("trade_records", []))
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            ledger.to_csv(ledger_path, index=False)
        metrics[f"cost{mult}"] = r

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "deep_edge_surface_parent_v39.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "norm": norm,
            "target_scale": TARGET_SCALE,
            "seq_len": SEQ_LEN,
            "max_label_hold": MAX_LABEL_HOLD,
            "selected_runtime": asdict(selected),
            "label_columns": ["long_q10", "long_q50", "long_q90", "short_q10", "short_q50", "short_q90", "long_adverse", "short_adverse"],
        },
        model_path,
    )
    pd.DataFrame(
        [
            {
                **{f"rt_{k}": v for k, v in row["runtime"].items()},
                "selection_score": row["selection_score"],
                "val_pnl": row["validation_cost1"]["pnl"],
                "val_mdd": row["validation_cost1"]["mdd"],
                "val_trades": row["validation_cost1"]["trades"],
                "val_cost2_pnl": row["validation_cost2"]["pnl"],
                "val_cost3_pnl": row["validation_cost3"]["pnl"],
            }
            for row in rows
        ]
    ).to_csv(args.grid_out, index=False)
    blocking = list(audit.get("blocking", []))
    warnings = list(audit.get("warnings", []))
    if metrics["cost1"]["pnl"] <= 0.0:
        warnings.append("cost1_not_survived")
    if metrics["cost2"]["pnl"] <= 0.0:
        warnings.append("cost2_not_survived")
    if metrics["cost3"]["pnl"] <= 0.0:
        warnings.append("cost3_not_survived")
    final_audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if not blocking and metrics["cost1"]["pnl"] > MARGIN110_COST1 else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "feature_audit": audit,
        "baseline_margin110_cost1": MARGIN110_COST1,
        "baseline_v31_cost1": V31_COST1,
        "selected_runtime": asdict(selected),
        "metrics": metrics,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Deep Edge Surface Parent V39. The entry model does not learn a CASH class. It predicts long/short quantile utility and adverse risk from 72-bar feature sequences; CASH is a post-model threshold decision selected on 2025 Q4.",
        "split_policy": "train=2025 Jan-Sep, selection=2025 Q4, OOS=2026 fixed",
        "feature_count": len(feature_cols),
        "train_rows": int(len(valid)),
        "target_summary": {
            "long_q50_mean": float(np.mean(y[:, 1])),
            "short_q50_mean": float(np.mean(y[:, 4])),
            "long_q90_p95": float(np.quantile(y[:, 2], 0.95)),
            "short_q90_p95": float(np.quantile(y[:, 5], 0.95)),
            "long_adverse_mean": float(np.mean(y[:, 6])),
            "short_adverse_mean": float(np.mean(y[:, 7])),
        },
        "selection": best,
        "metrics": metrics,
        "audit": final_audit,
        "artifacts": {"model": str(model_path), "report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "ledger": str(ledger_path)},
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(final_audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "model": str(model_path), "selected": selected.name, "metrics": metrics, "verdict": final_audit["verdict"]}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
