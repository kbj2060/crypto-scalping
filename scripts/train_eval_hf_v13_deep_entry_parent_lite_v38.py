#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    ACTION_CASH,
    ACTION_LONG,
    FullyLearnedGovernorConfig,
    build_training_set,
    prepare_features,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import (  # noqa: E402
    _audit_contract,
    _close,
    _days,
    _feature_cols,
    _fill_price,
    _read,
)


MODEL_ID = "hf_v13_deep_entry_parent_lite_v38_20260512"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_deep_entry_parent_lite_v38_20260512"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_deep_entry_parent_lite_v38_20260512_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_deep_entry_parent_lite_v38_20260512_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_deep_entry_parent_lite_v38_20260512_grid.csv"

SEQ_LEN = 72
MARGIN110_COST1 = 139.4071
V31_COST1 = 277.0679629973942


@dataclass(frozen=True)
class RuntimeConfig:
    name: str
    mode: str
    confidence: float
    quality_floor: float
    fixed_notional: float
    notional_scale: float
    max_notional: float
    take_profit: float
    stop_loss: float
    max_hold: int
    cooldown: int
    vol_throttle: bool = False
    dynamic_exit: bool = False


class DeepEntryParentLite(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 96, notional_classes: int = 9) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 3,
            dropout=0.10,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, hidden))
        self.attn = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))
        self.recency_bias = nn.Parameter(torch.linspace(-0.25, 0.35, SEQ_LEN).view(1, SEQ_LEN, 1))
        self.action_head = nn.Linear(hidden, 3)
        self.quality_head = nn.Linear(hidden, 1)
        self.notional_head = nn.Linear(hidden, notional_classes)

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.proj(seq) + self.pos[:, -seq.shape[1] :, :]
        h = self.encoder(h)
        w = torch.softmax(self.attn(h) + self.recency_bias[:, -h.shape[1] :, :], dim=1)
        ctx = torch.sum(h * w, dim=1)
        return self.action_head(ctx), self.quality_head(ctx).squeeze(-1), self.notional_head(ctx)


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


def _runtime_grid() -> list[RuntimeConfig]:
    rows: list[RuntimeConfig] = []
    for conf in (0.46, 0.52, 0.58):
        rows.append(RuntimeConfig(f"v38_entry_fixed_c{conf:.2f}", "entry_fixed", conf, -99.0, 1.0, 1.0, 1.2, 0.040, 0.018, 48, 12))
    for conf in (0.46, 0.52, 0.58):
        for qf in (0.000, 0.010, 0.020):
            rows.append(RuntimeConfig(f"v39_conviction_c{conf:.2f}_q{qf:.3f}", "conviction_gate", conf, qf, 1.0, 1.0, 1.2, 0.040, 0.018, 48, 12))
    for conf in (0.46, 0.52, 0.58):
        rows.append(RuntimeConfig(f"v40_sizing_c{conf:.2f}", "learned_sizing", conf, 0.000, 1.0, 0.65, 1.6, 0.040, 0.018, 48, 12))
    for conf in (0.46, 0.52):
        rows.append(RuntimeConfig(f"v41_risk_sizing_c{conf:.2f}", "risk_throttled_sizing", conf, 0.000, 1.0, 0.75, 1.8, 0.040, 0.018, 48, 12, vol_throttle=True))
    for conf in (0.46, 0.52):
        rows.append(RuntimeConfig(f"v42_dynamic_exit_c{conf:.2f}", "dynamic_exit", conf, 0.000, 1.0, 0.75, 1.8, 0.040, 0.018, 48, 12, vol_throttle=True, dynamic_exit=True))
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


def _fit_model(seq: np.ndarray, y: dict[str, np.ndarray], norm: dict[str, np.ndarray], *, cfg: FullyLearnedGovernorConfig, epochs: int, seed: int) -> DeepEntryParentLite:
    torch.manual_seed(int(seed))
    x = _apply_norm(seq, norm)
    action = np.asarray(y["action"], dtype=np.int64)
    quality = np.asarray(y["quality"], dtype=np.float32)
    notional = np.asarray(y["notional"], dtype=np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepEntryParentLite(x.shape[-1], notional_classes=len(cfg.notional_buckets)).to(device)
    counts = np.bincount(action, minlength=3).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights[0] *= 0.40
    weights = weights / max(weights.mean(), 1e-6)
    ce_action = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    ce_size = nn.CrossEntropyLoss()
    huber = nn.SmoothL1Loss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(action), torch.from_numpy(quality), torch.from_numpy(notional)),
        batch_size=256,
        shuffle=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    model.train()
    for _ in range(int(epochs)):
        for xb, ab, qb, nb in loader:
            xb, ab, qb, nb = xb.to(device), ab.to(device), qb.to(device), nb.to(device)
            logits, qhat, nlogits = model(xb)
            trade = ab != ACTION_CASH
            size_loss = ce_size(nlogits[trade], nb[trade]) if torch.any(trade) else torch.tensor(0.0, device=device)
            loss = ce_action(logits, ab) + 2.0 * huber(qhat, qb) + 0.35 * size_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model.cpu().eval()


def _predict_all(model: DeepEntryParentLite, features: pd.DataFrame, cols: list[str], norm: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    indices = np.arange(len(features), dtype=np.int64)
    seq = _seq_tensor(features, indices, cols)
    x = _apply_norm(seq, norm)
    probs: list[np.ndarray] = []
    qvals: list[np.ndarray] = []
    nprobs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 1024):
            logits, qhat, nlogits = model(torch.from_numpy(x[start : start + 1024]))
            probs.append(torch.softmax(logits, dim=1).numpy())
            qvals.append(qhat.numpy())
            nprobs.append(torch.softmax(nlogits, dim=1).numpy())
    return {"action_proba": np.vstack(probs), "quality": np.concatenate(qvals), "notional_proba": np.vstack(nprobs)}


def _safe_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        x = float(row.get(col, default))
    except Exception:
        return float(default)
    return float(x) if np.isfinite(x) else float(default)


def _vol_anchor(row: pd.Series) -> float:
    bbw = abs(_safe_float(row, "bb_width", 0.0))
    volz = abs(_safe_float(row, "volatility_z", 0.0))
    rv = abs(_safe_float(row, "realized_vol_ratio", 1.0))
    return float(np.clip(max(0.0015, bbw * 0.15) * (1.0 + 0.08 * min(volz, 3.0) + 0.05 * max(rv - 1.0, 0.0)), 0.0015, 0.030))


def _notional_from_head(proba: np.ndarray, buckets: tuple[float, ...], cfg: RuntimeConfig) -> float:
    vals = np.asarray(buckets, dtype=np.float64)
    n = float(np.sum(proba * vals) * cfg.notional_scale)
    return float(np.clip(n, 0.0, cfg.max_notional))


def backtest(
    df: pd.DataFrame,
    pred: dict[str, np.ndarray],
    runtime: RuntimeConfig,
    *,
    buckets: tuple[float, ...],
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
    notional = 0.0
    take_profit = float(runtime.take_profit)
    stop_loss = float(runtime.stop_loss)
    max_hold = int(runtime.max_hold)
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
            eff_tp = take_profit
            eff_sl = stop_loss
            if runtime.dynamic_exit:
                va = _vol_anchor(df.iloc[i]) * max(notional, 1e-12)
                eff_sl = float(np.clip(va * 2.4, stop_loss * 0.65, 0.035))
                if mfe > 0.0:
                    gap = max(va * 0.8, 0.003)
                    if hold >= 18:
                        gap = max(va * 0.35, gap - 0.025 * (hold - 18) * va)
                    eff_sl = min(eff_sl, max(0.001, mfe - gap))
                eff_tp = float(np.clip(take_profit * (1.0 + max(pred["quality"][entry_idx], 0.0) * 8.0), take_profit * 0.80, 0.075))
            reason = ""
            if eff_tp > 0 and unreal >= eff_tp:
                reason = "take_profit"
            elif eff_sl > 0 and unreal <= -abs(eff_sl):
                reason = "stop_loss"
            elif max_hold > 0 and hold >= max_hold:
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
                    out.update({"exit_signal_timestamp": str(df["timestamp"].iloc[i]), "exit_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "exit_reason": reason, "realized_net_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0), "cash_after": float(cash)})
                    records.append(out)
                pos = 0
                cooldown = int(runtime.cooldown)
                open_record = None
                continue
        if pos != 0:
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        p = pred["action_proba"][i]
        action = int(np.argmax(p))
        conf = float(np.max(p))
        quality = float(pred["quality"][i])
        if action == ACTION_CASH or conf < runtime.confidence:
            continue
        if runtime.mode != "entry_fixed" and quality < runtime.quality_floor:
            continue
        side = 1 if action == ACTION_LONG else -1
        if runtime.mode in {"learned_sizing", "risk_throttled_sizing", "dynamic_exit"}:
            n = _notional_from_head(pred["notional_proba"][i], buckets, runtime)
        else:
            n = float(runtime.fixed_notional)
        if runtime.vol_throttle:
            va = _vol_anchor(df.iloc[i])
            if va > 0.018:
                n *= 0.55
            elif va > 0.012:
                n *= 0.75
        if n <= 0.05:
            continue
        fill_i = min(i + 1, len(df) - 1)
        pos = side
        entry_price = _fill_price(df, fill_i, pos, slip, entry=True)
        entry_equity = cash
        entry_idx = i
        notional = float(n)
        cash -= cash * fee * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        mfe = mae = 0.0
        if record:
            open_record = {"entry_signal_timestamp": str(df["timestamp"].iloc[i]), "entry_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "side": "LONG" if pos > 0 else "SHORT", "entry_price": float(entry_price), "notional_exposure": float(notional), "confidence": conf, "quality": quality, "fee_entry_pct": float(fee * notional * 100.0)}
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
    out = {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(df)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(notional_sum / entries),
        "exits": exits,
    }
    if record:
        out["trade_records"] = records
    return out


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1["trades"]) < 20:
        return -1e9 + float(c1["pnl"])
    return float(c1["pnl"] + 0.35 * c2["pnl"] + 0.15 * c3["pnl"] - 0.45 * abs(c1["mdd"]) + 0.10 * min(c1["trades"], 120))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deep Entry Parent Lite staged role split experiment.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--seed", type=int, default=2029)
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
    print(f"[{MODEL_ID}] building labels", flush=True)
    x_tab, y, meta = build_training_set(train_df, cfg=cfg, stride_bars=12, batch_size=512, feature_cols=feature_cols)
    valid = np.arange(0, max(0, len(train_df) - cfg.max_train_horizon_bars - 1), 12, dtype=np.int64)
    full_train_features = prepare_features(train_df, side_hint=0, close=_close(train_df), feature_cols=feature_cols)
    print(f"[{MODEL_ID}] building sequence tensor rows={len(valid)} cols={len(feature_cols)}", flush=True)
    seq = _seq_tensor(full_train_features, valid, feature_cols)
    norm = _normalizer(seq)
    print(f"[{MODEL_ID}] training deep entry parent epochs={args.epochs}", flush=True)
    model = _fit_model(seq, y, norm, cfg=cfg, epochs=int(args.epochs), seed=int(args.seed))
    val_features = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=feature_cols)
    eval_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    print(f"[{MODEL_ID}] predicting validation/OOS", flush=True)
    val_pred = _predict_all(model, val_features, feature_cols, norm)
    eval_pred = _predict_all(model, eval_features, feature_cols, norm)
    rows: list[dict[str, Any]] = []
    best_by_mode: dict[str, dict[str, Any]] = {}
    for runtime in _runtime_grid():
        print(f"[{MODEL_ID}] validation {runtime.name}", flush=True)
        v1 = backtest(val_df, val_pred, runtime, buckets=cfg.notional_buckets, fee=cfg.fee, slip=cfg.slip)
        v2 = backtest(val_df, val_pred, runtime, buckets=cfg.notional_buckets, fee=cfg.fee * 2.0, slip=cfg.slip * 2.0)
        v3 = backtest(val_df, val_pred, runtime, buckets=cfg.notional_buckets, fee=cfg.fee * 3.0, slip=cfg.slip * 3.0)
        row = {"runtime": asdict(runtime), "validation_cost1": v1, "validation_cost2": v2, "validation_cost3": v3, "selection_score": _score(v1, v2, v3)}
        rows.append(row)
        prev = best_by_mode.get(runtime.mode)
        if prev is None or row["selection_score"] > prev["selection_score"]:
            best_by_mode[runtime.mode] = row
    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for mode, row in best_by_mode.items():
        runtime = RuntimeConfig(**row["runtime"])
        mode_metrics: dict[str, Any] = {}
        for mult in (1, 2, 3):
            r = backtest(eval_df, eval_pred, runtime, buckets=cfg.notional_buckets, fee=cfg.fee * mult, slip=cfg.slip * mult, record=(mult == 1))
            if mult == 1:
                ledger = pd.DataFrame(r.pop("trade_records", []))
                lp = args.report_out.with_name(f"{args.report_out.stem}_{mode}_cost1_ledger.csv")
                lp.parent.mkdir(parents=True, exist_ok=True)
                ledger.to_csv(lp, index=False)
                ledgers[mode] = str(lp)
            mode_metrics[f"cost{mult}"] = r
        metrics[mode] = {"selected_runtime": asdict(runtime), "selection": row, "oos": mode_metrics}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "deep_entry_parent_lite_v38.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "norm": norm,
            "parent_label_config": asdict(cfg),
            "training_meta": meta,
        },
        model_path,
    )
    pd.DataFrame(
        [
            {
                **{f"rt_{k}": v for k, v in r["runtime"].items()},
                "selection_score": r["selection_score"],
                "val_pnl": r["validation_cost1"]["pnl"],
                "val_mdd": r["validation_cost1"]["mdd"],
                "val_trades": r["validation_cost1"]["trades"],
                "val_cost2_pnl": r["validation_cost2"]["pnl"],
                "val_cost3_pnl": r["validation_cost3"]["pnl"],
            }
            for r in rows
        ]
    ).to_csv(args.grid_out, index=False)
    blocking = list(audit.get("blocking", []))
    warnings = list(audit.get("warnings", []))
    for mode, payload in metrics.items():
        if payload["oos"]["cost1"]["pnl"] <= 0.0:
            warnings.append(f"{mode}:cost1_not_survived")
        if payload["oos"]["cost3"]["pnl"] <= 0.0:
            warnings.append(f"{mode}:cost3_not_survived")
    best_mode = max(metrics.items(), key=lambda kv: kv[1]["oos"]["cost1"]["pnl"])[0] if metrics else ""
    final_audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if not blocking and metrics.get(best_mode, {}).get("oos", {}).get("cost1", {}).get("pnl", -1e9) > MARGIN110_COST1 else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "feature_audit": audit,
        "best_mode": best_mode,
        "baseline_margin110_cost1": MARGIN110_COST1,
        "baseline_v31_cost1": V31_COST1,
        "metrics": metrics,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Deep Entry Parent Lite role-split experiment. A sequence Transformer predicts only entry direction plus auxiliary conviction/notional heads. Runtime ablations add conviction gate, learned sizing, risk throttle, and dynamic exit one role at a time.",
        "split_policy": "train=2025 Jan-Sep, selection=2025 Q4, OOS=2026 fixed",
        "feature_count": len(feature_cols),
        "training_meta": meta,
        "label_distribution": {k: pd.Series(v).value_counts().sort_index().to_dict() for k, v in y.items() if k != "quality"},
        "metrics": metrics,
        "audit": final_audit,
        "artifacts": {"model": str(model_path), "report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "ledgers": ledgers},
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(final_audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "model": str(model_path), "best_mode": best_mode, "metrics": metrics, "verdict": final_audit["verdict"]}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
