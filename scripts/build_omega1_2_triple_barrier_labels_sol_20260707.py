#!/usr/bin/env python3
"""SOL copy of build_omega1_2_triple_barrier_labels_20260619.py.

Only TRAIN_CSV/EVAL_CSV/OUT_DIR/MODEL_ID are repointed at SOL data; the barrier
config grid, ATR/quality formulas, and split logic are identical to the ETH
recipe (faithful architecture replication, not a new hyperparameter search).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


MODEL_ID = "sol_omega1_2_triple_barrier_labels_20260707"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
SPLIT_TS = pd.Timestamp("2025-10-01")
TRAIN_CSV = ROOT / "data/splits/year_oos/sol_features_2025.csv"
EVAL_CSV = ROOT / "data/splits/year_oos/sol_features_2026.csv"
FEE_RATE = 0.0005
SLIP_RATE = 0.0002


@dataclass(frozen=True)
class BarrierConfig:
    name: str
    horizon: int
    tp_mult: float
    sl_mult: float
    min_tp: float
    min_sl: float


CONFIGS = (
    BarrierConfig("h24_conservative", 24, 1.2, 0.8, 0.006, 0.004),
    BarrierConfig("h24_balanced", 24, 1.6, 1.0, 0.006, 0.004),
    BarrierConfig("h24_runner", 24, 2.2, 1.2, 0.006, 0.004),
    BarrierConfig("h48_conservative", 48, 1.2, 0.8, 0.006, 0.004),
    BarrierConfig("h48_balanced", 48, 1.6, 1.0, 0.006, 0.004),
    BarrierConfig("h48_runner", 48, 2.2, 1.2, 0.006, 0.004),
    BarrierConfig("h96_conservative", 96, 1.2, 0.8, 0.006, 0.004),
    BarrierConfig("h96_balanced", 96, 1.6, 1.0, 0.006, 0.004),
    BarrierConfig("h96_runner", 96, 2.2, 1.2, 0.006, 0.004),
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _atr_price_move(frame: pd.DataFrame) -> np.ndarray:
    high = pd.to_numeric(frame["high"], errors="raise").astype(float)
    low = pd.to_numeric(frame["low"], errors="raise").astype(float)
    close = pd.to_numeric(frame["close"], errors="raise").astype(float)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # Shift by one bar so the label barrier width is known before the entry bar.
    atr = (true_range / close.replace(0.0, np.nan)).rolling(96, min_periods=24).mean().shift(1)
    return atr.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)


def _read_market_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    required = ["timestamp", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{path} missing required columns: {missing}")
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _reason_and_return(
    *,
    side: int,
    entry: float,
    future_high: np.ndarray,
    future_low: np.ndarray,
    future_close: np.ndarray,
    tp_move: float,
    sl_move: float,
) -> tuple[float, str, float, float, int]:
    if entry <= 0.0:
        return 0.0, "invalid_entry", 0.0, 0.0, 0

    if side > 0:
        tp_level = entry * (1.0 + tp_move)
        sl_level = entry * (1.0 - sl_move)
        rel_high = future_high / entry - 1.0
        rel_low = future_low / entry - 1.0
        mfe = float(np.nanmax(rel_high)) if len(rel_high) else 0.0
        mae = float(np.nanmin(rel_low)) if len(rel_low) else 0.0
        for bars, (hi, lo) in enumerate(zip(future_high, future_low), start=1):
            hit_sl = bool(lo <= sl_level)
            hit_tp = bool(hi >= tp_level)
            if hit_sl:
                return -float(sl_move), "sl", mae, mfe, bars
            if hit_tp:
                return float(tp_move), "tp", mae, mfe, bars
        return float(future_close[-1] / entry - 1.0), "timeout", mae, mfe, int(len(future_close))

    tp_level = entry * (1.0 - tp_move)
    sl_level = entry * (1.0 + sl_move)
    rel_high = 1.0 - future_low / entry
    rel_low = 1.0 - future_high / entry
    mfe = float(np.nanmax(rel_high)) if len(rel_high) else 0.0
    mae = float(np.nanmin(rel_low)) if len(rel_low) else 0.0
    for bars, (hi, lo) in enumerate(zip(future_high, future_low), start=1):
        hit_sl = bool(hi >= sl_level)
        hit_tp = bool(lo <= tp_level)
        if hit_sl:
            return -float(sl_move), "sl", mae, mfe, bars
        if hit_tp:
            return float(tp_move), "tp", mae, mfe, bars
    return float(1.0 - future_close[-1] / entry), "timeout", mae, mfe, int(len(future_close))


def _build_config_labels(frame: pd.DataFrame, cfg: BarrierConfig, *, fee_cost: float) -> pd.DataFrame:
    n = len(frame)
    open_px = pd.to_numeric(frame["open"], errors="raise").to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="raise").to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="raise").to_numpy(dtype=np.float64)
    close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
    atr = _atr_price_move(frame)
    ts = pd.to_datetime(frame["timestamp"], errors="raise")

    rows: list[dict[str, Any]] = []
    last_i = n - int(cfg.horizon) - 2
    for i in range(max(last_i, 0)):
        entry_i = i + 1
        end_i = entry_i + int(cfg.horizon)
        entry = float(open_px[entry_i])
        vol = float(atr[i])
        tp_move = max(float(cfg.min_tp), float(cfg.tp_mult) * vol)
        sl_move = max(float(cfg.min_sl), float(cfg.sl_mult) * vol)
        future_high = high[entry_i : end_i + 1]
        future_low = low[entry_i : end_i + 1]
        future_close = close[entry_i : end_i + 1]
        long_ret, long_reason, long_mae, long_mfe, long_bars = _reason_and_return(
            side=1,
            entry=entry,
            future_high=future_high,
            future_low=future_low,
            future_close=future_close,
            tp_move=tp_move,
            sl_move=sl_move,
        )
        short_ret, short_reason, short_mae, short_mfe, short_bars = _reason_and_return(
            side=-1,
            entry=entry,
            future_high=future_high,
            future_low=future_low,
            future_close=future_close,
            tp_move=tp_move,
            sl_move=sl_move,
        )
        long_quality = float(long_ret) - fee_cost - 0.20 * max(-float(long_mae), 0.0) - 0.003 * int(long_reason == "sl")
        short_quality = float(short_ret) - fee_cost - 0.20 * max(-float(short_mae), 0.0) - 0.003 * int(short_reason == "sl")
        if long_quality > 0.0 and long_quality >= short_quality:
            action = 1
            quality = long_quality
        elif short_quality > 0.0:
            action = 2
            quality = short_quality
        else:
            action = 0
            quality = max(long_quality, short_quality)
        rows.append(
            {
                "timestamp": ts.iloc[i],
                "entry_timestamp": ts.iloc[entry_i],
                "tb_action": int(action),
                "tb_quality": float(quality),
                "tb_long_ret": float(long_ret),
                "tb_short_ret": float(short_ret),
                "tb_long_quality": float(long_quality),
                "tb_short_quality": float(short_quality),
                "tb_long_reason": str(long_reason),
                "tb_short_reason": str(short_reason),
                "tb_long_mae": float(long_mae),
                "tb_short_mae": float(short_mae),
                "tb_long_mfe": float(long_mfe),
                "tb_short_mfe": float(short_mfe),
                "tb_long_bars": int(long_bars),
                "tb_short_bars": int(short_bars),
                "tb_tp_price_move": float(tp_move),
                "tb_sl_price_move": float(sl_move),
                "tb_atr_price_move": float(vol),
            }
        )
    return pd.DataFrame(rows)


def _prefixed(labels: pd.DataFrame, cfg: BarrierConfig) -> pd.DataFrame:
    keep = labels.copy()
    rename = {c: f"{c}_{cfg.name}" for c in keep.columns if c != "timestamp"}
    return keep.rename(columns=rename)


def _build_split_labels(frame: pd.DataFrame, *, fee_cost: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    out: pd.DataFrame | None = None
    audit: dict[str, Any] = {}
    for cfg in CONFIGS:
        labels = _build_config_labels(frame, cfg, fee_cost=fee_cost)
        pref = _prefixed(labels, cfg)
        out = pref if out is None else out.merge(pref, on="timestamp", how="inner", validate="one_to_one")
        counts = labels["tb_action"].value_counts().sort_index().to_dict()
        audit[cfg.name] = {
            "horizon": int(cfg.horizon),
            "tp_mult": float(cfg.tp_mult),
            "sl_mult": float(cfg.sl_mult),
            "min_tp": float(cfg.min_tp),
            "min_sl": float(cfg.min_sl),
            "rows": int(len(labels)),
            "action_counts": {str(int(k)): int(v) for k, v in counts.items()},
            "action_rates": {str(int(k)): float(v / max(len(labels), 1)) for k, v in counts.items()},
            "long_reason_counts": labels["tb_long_reason"].value_counts().sort_index().to_dict(),
            "short_reason_counts": labels["tb_short_reason"].value_counts().sort_index().to_dict(),
            "quality_mean": float(labels["tb_quality"].mean()) if len(labels) else 0.0,
            "tp_price_move_mean": float(labels["tb_tp_price_move"].mean()) if len(labels) else 0.0,
            "sl_price_move_mean": float(labels["tb_sl_price_move"].mean()) if len(labels) else 0.0,
        }
    if out is None:
        raise RuntimeError("no triple-barrier labels generated")
    return out, audit


def _select_by_validation(audit: dict[str, Any]) -> str:
    scored: list[tuple[float, str]] = []
    for name, data in audit.items():
        rates = {int(k): float(v) for k, v in data["action_rates"].items()}
        cash = rates.get(0, 0.0)
        long = rates.get(1, 0.0)
        short = rates.get(2, 0.0)
        active = long + short
        balance_penalty = abs(long - short)
        sparse_penalty = abs(active - 0.25)
        quality = float(data["quality_mean"])
        score = quality - 0.05 * balance_penalty - 0.03 * sparse_penalty - 0.01 * max(cash - 0.90, 0.0)
        scored.append((score, name))
    scored.sort(reverse=True)
    return scored[0][1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_2025 = _read_market_frame(TRAIN_CSV)
    eval_2026 = _read_market_frame(EVAL_CSV)
    fee_cost = float(FEE_RATE + SLIP_RATE) * 2.0 * 3.0

    split_frames = {
        "train": train_2025.loc[pd.to_datetime(train_2025["timestamp"], errors="raise") < SPLIT_TS].reset_index(drop=True),
        "validation": train_2025.loc[pd.to_datetime(train_2025["timestamp"], errors="raise") >= SPLIT_TS].reset_index(drop=True),
        "oos": eval_2026.reset_index(drop=True),
    }
    artifacts: dict[str, str] = {}
    audits: dict[str, Any] = {}
    for split, frame in split_frames.items():
        labels, audit = _build_split_labels(frame, fee_cost=fee_cost)
        path = out_dir / f"{split}_triple_barrier_labels.csv"
        labels.to_csv(path, index=False)
        artifacts[split] = str(path)
        audits[split] = audit

    selected = _select_by_validation(audits["validation"])
    report = {
        "model_id": MODEL_ID,
        "label_contract": {
            "entry": "next_bar_open",
            "barrier_hit": "intrabar_high_low",
            "same_bar_tp_sl_policy": "stop_loss_first",
            "volatility_source": "past-only ATR price move, rolling=96, min_periods=24, shifted by one bar",
            "fee_cost_in_quality": fee_cost,
            "source_frames": {"train_2025": str(TRAIN_CSV), "eval_2026": str(EVAL_CSV)},
            "action_map": {"0": "CASH", "1": "LONG", "2": "SHORT"},
            "configs": [cfg.__dict__ for cfg in CONFIGS],
        },
        "selected_by_validation_distribution_only": selected,
        "split_rows": {k: int(len(v)) for k, v in split_frames.items()},
        "artifacts": artifacts,
        "audit": audits,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "selected": selected, "artifacts": artifacts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
