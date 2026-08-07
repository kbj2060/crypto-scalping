#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
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
    FEATURE_COLS,
    FullyLearnedGovernorConfig,
    build_training_set,
    predict_policy_frame,
    train_policy,
)


CLEAN_PREFIX = "clean_regime_2024_unsup_v4_"
MODEL_ID = "hf_v13_parent_dcn_hybrid_v36_20260512"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_OUT = ROOT / "data/ensemble/supervised/hf_v13_parent_dcn_hybrid_v36_20260512"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_parent_dcn_hybrid_v36_20260512_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_parent_dcn_hybrid_v36_20260512_audit.json"
DCN_DIM = 24


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


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path} missing timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df.reset_index(drop=True)


def _close(df: pd.DataFrame) -> np.ndarray:
    return (
        pd.to_numeric(df["close"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .to_numpy(dtype=np.float64)
    )


def _fill_price(df: pd.DataFrame, idx: int, side: int, slip: float, *, entry: bool) -> float:
    px = float(pd.to_numeric(df["open"], errors="coerce").ffill().iloc[int(np.clip(idx, 0, len(df) - 1))])
    if side > 0:
        return px * (1.0 + slip if entry else 1.0 - slip)
    return px * (1.0 - slip if entry else 1.0 + slip)


def _days(df: pd.DataFrame) -> float:
    return max((df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 86400.0, 1e-8)


def _feature_cols(train: pd.DataFrame, eval_df: pd.DataFrame) -> list[str]:
    common = set(train.columns) & set(eval_df.columns)
    clean = sorted(c for c in common if c.startswith(CLEAN_PREFIX))
    old_regime_ids = {
        "regime_bull_id",
        "regime_bear_id",
        "regime_chop_id",
        "regime_whipsaw_id",
        "regime_normal_id",
    }
    base = [c for c in FEATURE_COLS if c not in old_regime_ids]
    return base + [c for c in clean if c not in base]


def _is_forbidden_feature(col: str) -> bool:
    lower = col.lower()
    if lower.startswith(CLEAN_PREFIX):
        return False
    if any(token in lower for token in ("future", "target", "label", "realized", "cash_after", "hdb", "legacy")):
        return True
    if lower.startswith("hmm_"):
        return True
    if "regime" in lower and not (lower.startswith("patchtst_") or lower.startswith("ai_")):
        return True
    return False


def _audit_contract(train: pd.DataFrame, eval_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    blocking: list[str] = []
    warnings: list[str] = []
    missing_train = [c for c in feature_cols if c not in train.columns and c != "side_hint" and not c.startswith("mom_") and not c.startswith("abs_mom_")]
    missing_eval = [c for c in feature_cols if c not in eval_df.columns and c != "side_hint" and not c.startswith("mom_") and not c.startswith("abs_mom_")]
    forbidden = [c for c in feature_cols if _is_forbidden_feature(c)]
    overlap = len(set(train["timestamp"].astype("int64").tolist()) & set(eval_df["timestamp"].astype("int64").tolist()))
    if missing_train:
        warnings.append("missing_train_features_zero_filled:" + ",".join(missing_train[:20]))
    if missing_eval:
        warnings.append("missing_eval_features_zero_filled:" + ",".join(missing_eval[:20]))
    if forbidden:
        blocking.append("forbidden_feature_cols:" + ",".join(forbidden[:30]))
    if overlap:
        blocking.append(f"train_eval_timestamp_overlap:{overlap}")
    if not any(c.startswith(CLEAN_PREFIX) for c in feature_cols):
        blocking.append("clean_regime_features_missing")
    status = "pass" if not blocking else "fail"
    return {
        "status": status,
        "blocking": blocking,
        "warnings": warnings,
        "feature_count": len(feature_cols),
        "clean_regime_feature_count": len([c for c in feature_cols if c.startswith(CLEAN_PREFIX)]),
        "forbidden_feature_cols": forbidden,
        "train_range": [str(train["timestamp"].iloc[0]), str(train["timestamp"].iloc[-1])],
        "eval_range": [str(eval_df["timestamp"].iloc[0]), str(eval_df["timestamp"].iloc[-1])],
        "train_eval_timestamp_overlap": int(overlap),
    }


class DCNHybridEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = DCN_DIM, cross_layers: int = 3) -> None:
        super().__init__()
        self.cross_w = nn.ParameterList([nn.Parameter(torch.empty(input_dim)) for _ in range(cross_layers)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(cross_layers)])
        for w in self.cross_w:
            nn.init.normal_(w, mean=0.0, std=0.02)
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.embed = nn.Sequential(nn.Linear(input_dim + 64, 64), nn.GELU(), nn.Linear(64, embed_dim))
        self.action_head = nn.Linear(embed_dim, 3)
        self.quality_head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x0 = x
        xl = x
        for w, b in zip(self.cross_w, self.cross_b):
            dot = torch.sum(xl * w, dim=1, keepdim=True)
            xl = x0 * dot + b + xl
        deep = self.deep(x)
        emb = self.embed(torch.cat([xl, deep], dim=1))
        return emb, self.action_head(emb), self.quality_head(emb).squeeze(-1)


def _fit_dcn_encoder(x: pd.DataFrame, y: dict[str, np.ndarray], *, epochs: int = 35, seed: int = 42) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    arr = x.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    mean = np.nanmean(arr, axis=0).astype(np.float32)
    std = (np.nanstd(arr, axis=0) + 1e-6).astype(np.float32)
    xx = ((arr - mean[None, :]) / std[None, :]).astype(np.float32)
    action = np.asarray(y["action"], dtype=np.int64)
    quality = np.asarray(y["quality"], dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCNHybridEncoder(xx.shape[1]).to(device)
    ds = TensorDataset(torch.from_numpy(xx), torch.from_numpy(action), torch.from_numpy(quality))
    loader = DataLoader(ds, batch_size=512, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(weight=torch.tensor([0.35, 1.0, 1.0], dtype=torch.float32, device=device))
    huber = nn.SmoothL1Loss()
    model.train()
    for _ in range(int(epochs)):
        for xb, ab, qb in loader:
            xb, ab, qb = xb.to(device), ab.to(device), qb.to(device)
            _, logits, qhat = model(xb)
            loss = ce(logits, ab) + 2.0 * huber(qhat, qb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "mean": mean, "std": std, "input_cols": list(x.columns), "embed_dim": DCN_DIM}


def _dcn_features(payload: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    cols = list(payload["input_cols"])
    arr = frame.reindex(columns=cols).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    xx = ((arr - payload["mean"][None, :]) / payload["std"][None, :]).astype(np.float32)
    model = DCNHybridEncoder(len(cols), int(payload.get("embed_dim", DCN_DIM)))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(xx), 2048):
            emb, _, _ = model(torch.from_numpy(xx[start : start + 2048]))
            out.append(emb.numpy())
    emb_all = np.vstack(out).astype(np.float32)
    return pd.DataFrame(emb_all, columns=[f"dcn_cross_{i:02d}" for i in range(emb_all.shape[1])], index=frame.index)


def _augment_features(base_x: pd.DataFrame, dcn_payload: dict[str, Any]) -> pd.DataFrame:
    dcn = _dcn_features(dcn_payload, base_x)
    return pd.concat([base_x.reset_index(drop=True), dcn.reset_index(drop=True)], axis=1)


def _prepare_augmented_frame(frame: pd.DataFrame, feature_cols: list[str], dcn_payload: dict[str, Any]) -> pd.DataFrame:
    close = _close(frame)
    # Reuse the governor's own feature preparation through build-time columns.
    from ensemble.fully_learned_governor_policy import prepare_features

    prepared = prepare_features(frame, side_hint=0, close=close, feature_cols=feature_cols)
    dcn = _dcn_features(dcn_payload, prepared)
    out = frame.reset_index(drop=True).copy()
    for col in prepared.columns:
        if col not in out.columns:
            out[col] = prepared[col].to_numpy()
    for col in dcn.columns:
        out[col] = dcn[col].to_numpy()
    return out


def backtest_policy_frame(df: pd.DataFrame, bundle: dict[str, Any], *, fee: float, slip: float, record_trades: bool = False) -> dict[str, Any]:
    close = _close(df)
    decisions = predict_policy_frame(bundle, df, close=close)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_idx = 0
    notional = 0.0
    leverage = 1.0
    take_profit = 0.0
    stop_loss = 0.0
    max_hold = 0
    next_cooldown = 0
    cooldown_left = 0
    peak_unrealized = 0.0
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    action_counts: dict[str, int] = {"cash": 0, "long": 0, "short": 0}
    exits: dict[str, int] = {}
    notional_sum = 0.0
    leverage_sum = 0.0
    records: list[dict[str, Any]] = []
    open_record: dict[str, Any] | None = None

    def mark_equity(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        if pos > 0:
            raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12)
        else:
            raw = (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
        unreal = raw * notional
        return cash * (1.0 + unreal), unreal

    for i in range(0, len(df) - 2):
        eq, unreal = mark_equity(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if pos != 0:
            peak_unrealized = max(peak_unrealized, unreal)
            hold_bars = i - entry_idx
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "learned_take_profit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "learned_stop_loss"
            elif max_hold > 0 and hold_bars >= max_hold:
                reason = "learned_max_hold"
            if reason:
                fill_idx = min(i + 1, len(df) - 1)
                exit_price = _fill_price(df, fill_idx, pos, slip, entry=False)
                raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * fee * notional
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] = exits.get(reason, 0) + 1
                if record_trades and open_record is not None:
                    out = dict(open_record)
                    out.update(
                        {
                            "exit_signal_timestamp": str(df["timestamp"].iloc[i]),
                            "exit_fill_timestamp": str(df["timestamp"].iloc[fill_idx]),
                            "exit_reason": reason,
                            "realized_net_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0),
                            "peak_unrealized_pct": float(peak_unrealized * 100.0),
                            "fee_exit_pct": float(fee * notional * 100.0),
                        }
                    )
                    records.append(out)
                pos = 0
                notional = 0.0
                leverage = 1.0
                cooldown_left = int(next_cooldown)
                next_cooldown = 0
                peak_unrealized = 0.0
                open_record = None
                continue

        if pos == 0:
            if cooldown_left > 0:
                cooldown_left -= 1
                action_counts["cash"] += 1
                continue
            dec = decisions.iloc[i]
            if int(dec.action) == ACTION_CASH or int(dec.side) == 0:
                action_counts["cash"] += 1
                continue
            action_counts["long" if int(dec.action) == ACTION_LONG else "short"] += 1
            fill_idx = min(i + 1, len(df) - 1)
            pos = int(dec.side)
            entry_price = _fill_price(df, fill_idx, pos, slip, entry=True)
            entry_equity = cash
            entry_idx = i
            notional = float(dec.notional_exposure)
            leverage = float(dec.leverage)
            take_profit = float(dec.take_profit)
            stop_loss = float(dec.stop_loss)
            max_hold = int(dec.max_hold_bars)
            next_cooldown = int(dec.cooldown_bars)
            cash -= cash * fee * notional
            long_entries += int(pos > 0)
            short_entries += int(pos < 0)
            notional_sum += notional
            leverage_sum += leverage
            if record_trades:
                open_record = {
                    "entry_signal_timestamp": str(df["timestamp"].iloc[i]),
                    "entry_fill_timestamp": str(df["timestamp"].iloc[fill_idx]),
                    "side": "LONG" if pos > 0 else "SHORT",
                    "entry_price": float(entry_price),
                    "notional_exposure": float(notional),
                    "leverage": float(leverage),
                    "position_fraction": float(dec.position_fraction),
                    "take_profit": float(take_profit),
                    "stop_loss": float(stop_loss),
                    "max_hold_bars": int(max_hold),
                    "cooldown_bars": int(next_cooldown),
                    "quality_score": float(dec.quality_score),
                    "confidence": float(dec.confidence),
                    "fee_entry_pct": float(fee * notional * 100.0),
                }
    if pos != 0:
        fill_idx = len(df) - 1
        exit_price = _fill_price(df, fill_idx, pos, slip, entry=False)
        raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * fee * notional
        trades += 1
        wins += int(cash > entry_equity)
        exits["forced_end"] = exits.get("forced_end", 0) + 1
    n_entries = max(long_entries + short_entries, 1)
    out = {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(df)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(notional_sum / n_entries),
        "avg_leverage": float(leverage_sum / n_entries),
        "action_counts": action_counts,
        "exits": exits,
    }
    if record_trades:
        out["trade_records"] = records
    return out


def _cfgs(limit: int) -> list[tuple[str, FullyLearnedGovernorConfig, int, int]]:
    base_buckets = (0.10, 0.16, 0.24, 0.34, 0.48, 0.68, 0.95, 1.30, 1.80, 2.40, 3.00)
    candidates = [
        (
            "v13_clean_regime_h288",
            FullyLearnedGovernorConfig(
                notional_buckets=(0.20, 0.32, 0.50, 0.75, 1.05, 1.45, 2.00, 2.70, 3.60),
                leverage_buckets=(1.5, 2.0, 3.0, 4.0, 5.0),
                take_profit_buckets=(0.007, 0.011, 0.018, 0.030, 0.050, 0.090, 0.180, 0.450, 0.900),
                stop_loss_buckets=(0.004, 0.006, 0.009, 0.014, 0.022, 0.035, 0.055),
                max_hold_buckets=(6, 12, 24, 48, 96, 192, 288),
                cooldown_buckets=(0, 1, 3, 6, 12, 24, 48),
                max_train_horizon_bars=288,
                adverse_penalty=2.45,
                size_penalty=0.180,
                hold_penalty=0.042,
                turnover_bonus=0.0012,
                cash_score=0.020,
            ),
            12,
            512,
        ),
        (
            "balanced_clean_regime_h144",
            FullyLearnedGovernorConfig(
                notional_buckets=base_buckets,
                leverage_buckets=(1.5, 2.0, 3.0, 4.0, 5.0),
                take_profit_buckets=(0.004, 0.006, 0.009, 0.013, 0.020, 0.030, 0.050, 0.080, 0.130, 0.220),
                stop_loss_buckets=(0.003, 0.0045, 0.0065, 0.009, 0.013, 0.020, 0.030),
                max_hold_buckets=(3, 6, 12, 18, 24, 36, 48, 72, 96, 144),
                cooldown_buckets=(0, 0, 1, 2, 3, 6, 12),
                max_train_horizon_bars=144,
                adverse_penalty=1.55,
                size_penalty=0.110,
                hold_penalty=0.014,
                turnover_bonus=0.008,
                cash_score=-0.004,
            ),
            3,
            512,
        ),
        (
            "risk_tighter_clean_regime_h96",
            FullyLearnedGovernorConfig(
                notional_buckets=(0.08, 0.12, 0.18, 0.27, 0.40, 0.60, 0.90, 1.25, 1.70, 2.30),
                leverage_buckets=(1.5, 2.0, 3.0, 4.0),
                take_profit_buckets=(0.004, 0.006, 0.009, 0.014, 0.022, 0.035, 0.060, 0.100, 0.160),
                stop_loss_buckets=(0.003, 0.0045, 0.0065, 0.009, 0.013, 0.020),
                max_hold_buckets=(3, 6, 12, 18, 24, 36, 48, 72, 96),
                cooldown_buckets=(0, 1, 2, 3, 6, 12),
                max_train_horizon_bars=96,
                adverse_penalty=1.95,
                size_penalty=0.145,
                hold_penalty=0.016,
                turnover_bonus=0.010,
                cash_score=-0.002,
            ),
            3,
            512,
        ),
        (
            "turnover_clean_regime_h72",
            FullyLearnedGovernorConfig(
                notional_buckets=(0.08, 0.14, 0.22, 0.34, 0.52, 0.78, 1.15, 1.65, 2.25),
                leverage_buckets=(1.5, 2.0, 3.0, 4.0, 5.0),
                take_profit_buckets=(0.0035, 0.0055, 0.008, 0.012, 0.018, 0.028, 0.045, 0.070, 0.110),
                stop_loss_buckets=(0.003, 0.0045, 0.0065, 0.009, 0.013, 0.020),
                max_hold_buckets=(3, 6, 9, 12, 18, 24, 36, 48, 72),
                cooldown_buckets=(0, 0, 0, 1, 2, 3, 6),
                max_train_horizon_bars=72,
                adverse_penalty=1.35,
                size_penalty=0.095,
                hold_penalty=0.020,
                turnover_bonus=0.015,
                cash_score=-0.006,
            ),
            2,
            640,
        ),
    ]
    return candidates[: max(1, int(limit))]


def _score(row: dict[str, Any]) -> float:
    val = row["validation_cost1"]
    val2 = row["validation_cost2"]
    val3 = row["validation_cost3"]
    score = float(val["pnl"]) + 0.35 * float(val2["pnl"]) + 0.15 * float(val3["pnl"]) + 2.0 * float(val["mdd"])
    if float(val2["pnl"]) < 0.0:
        score -= abs(float(val2["pnl"])) * 1.5
    if float(val3["pnl"]) < 0.0:
        score -= abs(float(val3["pnl"])) * 2.0
    return float(score)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean-regime feature HF core loop experiment.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--max-candidates", type=int, default=3)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    split_ts = pd.Timestamp("2025-10-01")
    train_df = train_all[train_all["timestamp"] < split_ts].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= split_ts].reset_index(drop=True)
    feature_cols = _feature_cols(train_all, eval_df)
    audit = _audit_contract(train_all, eval_df, feature_cols)
    if audit["status"] != "pass":
        args.audit_out.parent.mkdir(parents=True, exist_ok=True)
        args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
        raise ValueError("feature audit failed: " + json.dumps(audit, ensure_ascii=False, default=_json_default))

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for idx, (name, cfg, stride, batch) in enumerate(_cfgs(args.max_candidates), start=1):
        print(f"[{MODEL_ID}] training {idx}/{args.max_candidates}: {name}", flush=True)
        x, y, meta = build_training_set(train_df, cfg=cfg, stride_bars=stride, batch_size=batch, feature_cols=feature_cols)
        print(f"[{MODEL_ID}] fitting DCN cross encoder rows={len(x)} cols={len(feature_cols)}", flush=True)
        dcn_payload = _fit_dcn_encoder(x, y, epochs=35, seed=int(args.random_state) + 1000 + idx)
        x_aug = _augment_features(x, dcn_payload)
        aug_feature_cols = list(x_aug.columns)
        bundle = train_policy(x_aug, y, cfg=cfg, random_state=int(args.random_state) + idx, feature_cols=aug_feature_cols)
        bundle["model_id"] = MODEL_ID
        bundle["candidate_name"] = name
        bundle["dcn_payload"] = dcn_payload
        bundle["base_feature_cols"] = feature_cols
        bundle["train_csv"] = str(args.train_csv)
        bundle["eval_csv"] = str(args.eval_csv)
        bundle["training_meta"] = meta
        bundle["feature_audit"] = audit
        model_path = args.out_dir / f"{name}.pkl"
        joblib.dump(bundle, model_path)
        train_aug_frame = _prepare_augmented_frame(train_df, feature_cols, dcn_payload)
        val_aug_frame = _prepare_augmented_frame(val_df, feature_cols, dcn_payload)
        row = {
            "name": name,
            "model": str(model_path),
            "config": asdict(cfg),
            "training_meta": meta,
            "label_distribution": bundle.get("label_distribution", {}),
            "train_cost1": backtest_policy_frame(train_aug_frame, bundle, fee=cfg.fee, slip=cfg.slip),
            "validation_cost1": backtest_policy_frame(val_aug_frame, bundle, fee=cfg.fee, slip=cfg.slip),
            "validation_cost2": backtest_policy_frame(val_aug_frame, bundle, fee=cfg.fee * 2.0, slip=cfg.slip * 2.0),
            "validation_cost3": backtest_policy_frame(val_aug_frame, bundle, fee=cfg.fee * 3.0, slip=cfg.slip * 3.0),
        }
        row["selection_score"] = _score(row)
        rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row

    if best is None:
        raise RuntimeError("no trained candidates")
    best_bundle = joblib.load(best["model"])
    cfg = FullyLearnedGovernorConfig(**dict(best_bundle["config"]))
    eval_aug_df = _prepare_augmented_frame(eval_df, list(best_bundle["base_feature_cols"]), best_bundle["dcn_payload"])
    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for mult in (1, 2, 3):
        result = backtest_policy_frame(eval_aug_df, best_bundle, fee=cfg.fee * mult, slip=cfg.slip * mult, record_trades=(mult == 1))
        if mult == 1:
            ledger = pd.DataFrame(result.pop("trade_records", []))
            ledger_path = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            ledger.to_csv(ledger_path, index=False)
            ledgers["cost1"] = str(ledger_path)
        metrics[f"cost{mult}"] = result

    grid_path = args.report_out.with_name(args.report_out.stem + "_grid.json")
    grid_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    report = {
        "model_id": MODEL_ID,
        "design": "Fully learned HF core retrained with explicit clean_regime_2024_unsup_v4_* inputs; old regime-v2 ids removed.",
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "split_policy": "train=2025-01-01..2025-09-30, selection=2025-10-01..2025-12-31, OOS=2026 fixed",
        "feature_contract": {
            "feature_count": len(feature_cols),
            "dcn_cross_feature_count": DCN_DIM,
            "clean_regime_features": [c for c in feature_cols if c.startswith(CLEAN_PREFIX)],
            "feature_cols": feature_cols,
        },
        "audit": audit,
        "selected": {k: v for k, v in best.items() if k not in {"config", "label_distribution"}},
        "selected_config": best["config"],
        "label_distribution": best["label_distribution"],
        "metrics": metrics,
        "artifacts": {
            "report": str(args.report_out),
            "audit": str(args.audit_out),
            "grid": str(grid_path),
            "model": best["model"],
            "ledgers": ledgers,
        },
    }
    audit["oos_cost_metrics"] = metrics
    audit["selected_model"] = best["model"]
    audit["verdict"] = "promote_candidate" if metrics["cost1"]["pnl"] >= 100.0 and metrics["cost2"]["pnl"] > 0.0 and metrics["cost3"]["pnl"] > 0.0 else "iterate"
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "selected": best["name"], "metrics": metrics, "verdict": audit["verdict"]}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
