#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _read_frame(path: Path, *, expected_year: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    required = {"timestamp", "high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    years = sorted(frame["timestamp"].dt.year.dropna().astype(int).unique().tolist())
    if years != [int(expected_year)]:
        raise RuntimeError(f"{path} year guard failed: expected={[int(expected_year)]} actual={years}")
    return frame


def _atr_pct(frame: pd.DataFrame, window: int) -> np.ndarray:
    high = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=np.float64)
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=np.float64)
    prev = np.roll(close, 1)
    prev[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    atr = pd.Series(tr).ewm(span=int(window), adjust=False, min_periods=1).mean().to_numpy(dtype=np.float64)
    return atr / np.maximum(close, 1e-12)


def _filter_alternating(pivots: list[tuple[int, float, str]]) -> list[tuple[int, float, str]]:
    if not pivots:
        return []
    out = [pivots[0]]
    for cur in pivots[1:]:
        prev = out[-1]
        if cur[2] == prev[2]:
            if cur[2] == "H" and cur[1] > prev[1]:
                out[-1] = cur
            elif cur[2] == "L" and cur[1] < prev[1]:
                out[-1] = cur
        else:
            out.append(cur)
    return out


def _zigzag_pivots(frame: pd.DataFrame, *, min_reversal_pct: float, atr_window: int, atr_multiplier: float) -> list[tuple[int, float, str]]:
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=np.float64)
    atr_pct = _atr_pct(frame, atr_window)
    n = len(close)
    if n == 0:
        return []

    def _threshold(i: int) -> float:
        atr = float(atr_pct[min(max(int(i), 0), n - 1)])
        return max(float(min_reversal_pct), atr * float(atr_multiplier))

    trend = 0
    low_idx = high_idx = 0
    low_price = high_price = float(close[0])
    pivots: list[tuple[int, float, str]] = []

    for i in range(1, n):
        price = float(close[i])
        if not np.isfinite(price):
            continue
        if trend == 0:
            if price < low_price:
                low_idx, low_price = i, price
            if price > high_price:
                high_idx, high_price = i, price
            thr = _threshold(i)
            if high_price / max(low_price, 1e-12) - 1.0 >= thr:
                if low_idx < high_idx:
                    pivots.append((int(low_idx), float(low_price), "L"))
                    trend = 1
                    high_idx, high_price = i, price
                else:
                    pivots.append((int(high_idx), float(high_price), "H"))
                    trend = -1
                    low_idx, low_price = i, price
        elif trend == 1:
            if price > high_price:
                high_idx, high_price = i, price
            drop = high_price / max(price, 1e-12) - 1.0
            if drop >= _threshold(i):
                pivots.append((int(high_idx), float(high_price), "H"))
                trend = -1
                low_idx, low_price = i, price
        else:
            if price < low_price:
                low_idx, low_price = i, price
            rise = price / max(low_price, 1e-12) - 1.0
            if rise >= _threshold(i):
                pivots.append((int(low_idx), float(low_price), "L"))
                trend = 1
                high_idx, high_price = i, price

    if trend == 1 and (not pivots or pivots[-1][0] != high_idx):
        pivots.append((int(high_idx), float(high_price), "H"))
    elif trend == -1 and (not pivots or pivots[-1][0] != low_idx):
        pivots.append((int(low_idx), float(low_price), "L"))
    return _filter_alternating(pivots)


def build_zigzag_action_labels(
    frame: pd.DataFrame,
    *,
    min_reversal_pct: float,
    min_wave_bars: int,
    transition_buffer: int,
    atr_window: int,
    atr_multiplier: float,
    mae_penalty: float,
    softmax_temperature: float,
    min_risk_floor: float,
) -> pd.DataFrame:
    n = len(frame)
    labels = np.zeros(n, dtype=np.int8)
    segment_id = np.full(n, -1, dtype=np.int32)
    wave_ret = np.zeros(n, dtype=np.float32)
    wave_bars = np.zeros(n, dtype=np.int16)
    is_buffer = np.zeros(n, dtype=np.int8)
    path_return = np.zeros(n, dtype=np.float32)
    path_mae = np.zeros(n, dtype=np.float32)
    path_mfe = np.zeros(n, dtype=np.float32)
    path_calmar = np.zeros(n, dtype=np.float32)
    path_edge = np.zeros(n, dtype=np.float32)
    soft = np.zeros((n, 3), dtype=np.float32)
    soft[:, 0] = 1.0
    atr_pct = _atr_pct(frame, atr_window)
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=np.float64)
    pivots = _zigzag_pivots(
        frame,
        min_reversal_pct=float(min_reversal_pct),
        atr_window=int(atr_window),
        atr_multiplier=float(atr_multiplier),
    )

    sid = 0
    for start, end in zip(pivots, pivots[1:]):
        idx_s, val_s, type_s = start
        idx_e, val_e, type_e = end
        if idx_e <= idx_s:
            continue
        bars = int(idx_e - idx_s)
        if bars < int(min_wave_bars):
            continue
        if type_s == "L" and type_e == "H":
            side = 1
        elif type_s == "H" and type_e == "L":
            side = 2
        else:
            continue
        labels[int(idx_s) : int(idx_e)] = side
        segment_id[int(idx_s) : int(idx_e)] = sid
        wave_ret[int(idx_s) : int(idx_e)] = np.float32(
            (float(val_e) / max(float(val_s), 1e-12) - 1.0) * (1.0 if side == 1 else -1.0)
        )
        wave_bars[int(idx_s) : int(idx_e)] = np.int16(min(bars, np.iinfo(np.int16).max))
        for i in range(int(idx_s), int(idx_e)):
            entry = float(close[i])
            if not np.isfinite(entry) or entry <= 0.0:
                continue
            hi = high[i : int(idx_e) + 1]
            lo = low[i : int(idx_e) + 1]
            end_px = float(close[int(idx_e)])
            if side == 1:
                ret = end_px / max(entry, 1e-12) - 1.0
                mae = max(0.0, (entry - float(np.nanmin(lo))) / max(entry, 1e-12))
                mfe = max(0.0, (float(np.nanmax(hi)) - entry) / max(entry, 1e-12))
                side_idx = 1
            else:
                ret = entry / max(end_px, 1e-12) - 1.0
                mae = max(0.0, (float(np.nanmax(hi)) - entry) / max(entry, 1e-12))
                mfe = max(0.0, (entry - float(np.nanmin(lo))) / max(entry, 1e-12))
                side_idx = 2
            risk_floor = max(float(min_risk_floor), float(atr_pct[i]))
            calmar = ret / max(mae, risk_floor)
            edge = ret - float(mae_penalty) * mae
            score = edge / max(risk_floor, 1e-12)
            temp = max(float(softmax_temperature), 1e-6)
            logits = np.zeros(3, dtype=np.float64)
            logits[side_idx] = score / temp
            logits[0] = max(0.0, (mae - ret) / max(risk_floor, 1e-12)) / temp
            logits -= float(np.max(logits))
            probs = np.exp(logits)
            probs /= max(float(np.sum(probs)), 1e-12)
            path_return[i] = np.float32(ret)
            path_mae[i] = np.float32(mae)
            path_mfe[i] = np.float32(mfe)
            path_calmar[i] = np.float32(calmar)
            path_edge[i] = np.float32(edge)
            soft[i, :] = probs.astype(np.float32)
        sid += 1

    buf = int(max(0, transition_buffer))
    if buf > 0:
        change = np.flatnonzero(labels != np.roll(labels, 1))
        change = change[change > 0]
        for idx in change:
            lo = max(0, int(idx) - buf)
            hi = min(n, int(idx) + buf + 1)
            labels[lo:hi] = 0
            is_buffer[lo:hi] = 1
            soft[lo:hi, :] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            path_return[lo:hi] = 0.0
            path_mae[lo:hi] = 0.0
            path_mfe[lo:hi] = 0.0
            path_calmar[lo:hi] = 0.0
            path_edge[lo:hi] = 0.0

    inactive = labels == 0
    soft[inactive, :] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    path_return[inactive] = 0.0
    path_mae[inactive] = 0.0
    path_mfe[inactive] = 0.0
    path_calmar[inactive] = 0.0
    path_edge[inactive] = 0.0

    out = frame[["timestamp", "open", "high", "low", "close"]].copy()
    out["zigzag_action"] = labels
    out["zigzag_action_name"] = pd.Series(labels).map({0: "CASH", 1: "LONG", 2: "SHORT"}).to_numpy()
    out["zigzag_segment_id"] = segment_id
    out["zigzag_wave_return"] = wave_ret
    out["zigzag_wave_bars"] = wave_bars
    out["zigzag_transition_buffer"] = is_buffer
    out["zigzag_atr_pct"] = atr_pct.astype(np.float32)
    out["zigzag_path_return"] = path_return
    out["zigzag_path_mae"] = path_mae
    out["zigzag_path_mfe"] = path_mfe
    out["zigzag_path_calmar"] = path_calmar
    out["zigzag_path_edge"] = path_edge
    out["zigzag_soft_cash"] = soft[:, 0]
    out["zigzag_soft_long"] = soft[:, 1]
    out["zigzag_soft_short"] = soft[:, 2]
    return out


def _summary(labels: pd.DataFrame) -> dict[str, Any]:
    counts = labels["zigzag_action"].value_counts().sort_index().to_dict()
    total = max(len(labels), 1)
    soft_cols = ["zigzag_soft_cash", "zigzag_soft_long", "zigzag_soft_short"]
    return {
        "rows": int(len(labels)),
        "counts": {str(k): int(v) for k, v in counts.items()},
        "ratios": {str(k): float(v) / total for k, v in counts.items()},
        "segments": int(labels["zigzag_segment_id"].max() + 1) if len(labels) else 0,
        "buffer_rows": int(labels["zigzag_transition_buffer"].sum()),
        "soft_mean": {col: float(labels[col].mean()) for col in soft_cols},
        "active_soft_side_mean": float(
            labels.loc[labels["zigzag_action"] != 0, ["zigzag_soft_long", "zigzag_soft_short"]].max(axis=1).mean()
        )
        if int((labels["zigzag_action"] != 0).sum()) > 0
        else 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--input-2024", type=Path, default=ROOT / "data/splits/year_oos/training_features_2024.csv")
    p.add_argument("--input-2025", type=Path, default=ROOT / "data/splits/year_oos/training_features_2025.csv")
    p.add_argument("--input-2026", type=Path, default=ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv")
    p.add_argument("--min-wave-bars", type=int, default=8)
    p.add_argument("--transition-buffer", type=int, default=2)
    p.add_argument("--atr-window", type=int, default=14)
    p.add_argument("--atr-multiplier", type=float, default=1.0)
    p.add_argument("--zigzag-reversal-pct", type=float, default=0.010)
    p.add_argument("--mae-penalty", type=float, default=1.25)
    p.add_argument("--softmax-temperature", type=float, default=1.75)
    p.add_argument("--min-risk-floor", type=float, default=0.0010)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit: dict[str, Any] = {
        "type": "zigzag_3class_risk_adjusted_soft_action_labels",
        "params": {
            "method": "zigzag",
            "min_wave_bars": int(args.min_wave_bars),
            "transition_buffer": int(args.transition_buffer),
            "atr_window": int(args.atr_window),
            "atr_multiplier": float(args.atr_multiplier),
            "zigzag_reversal_pct": float(args.zigzag_reversal_pct),
            "mae_penalty": float(args.mae_penalty),
            "softmax_temperature": float(args.softmax_temperature),
            "min_risk_floor": float(args.min_risk_floor),
        },
        "artifacts": {},
        "summaries": {},
        "contract": {
            "label_mapping": {"0": "CASH", "1": "LONG", "2": "SHORT"},
            "label_column": "zigzag_action",
            "soft_label_columns": ["zigzag_soft_cash", "zigzag_soft_long", "zigzag_soft_short"],
            "removed_columns": ["wave3_action"],
            "zigzag_confirmed_pivot_segments": True,
            "legacy_swing_wave3_retired": True,
            "legacy_tp_sl_action_score_retired": True,
            "transition_buffer_is_cash": True,
            "risk_adjusted_soft_labels": True,
            "uses_future_only_for_offline_labeling": True,
        },
    }
    for year, path in [(2024, args.input_2024), (2025, args.input_2025), (2026, args.input_2026)]:
        frame = _read_frame(path, expected_year=year)
        labels = build_zigzag_action_labels(
            frame,
            min_reversal_pct=float(args.zigzag_reversal_pct),
            min_wave_bars=int(args.min_wave_bars),
            transition_buffer=int(args.transition_buffer),
            atr_window=int(args.atr_window),
            atr_multiplier=float(args.atr_multiplier),
            mae_penalty=float(args.mae_penalty),
            softmax_temperature=float(args.softmax_temperature),
            min_risk_floor=float(args.min_risk_floor),
        )
        out = args.out_dir / f"zigzag_action_labels_{year}.csv"
        labels.to_csv(out, index=False)
        audit["artifacts"][str(year)] = str(out)
        audit["summaries"][str(year)] = _summary(labels)
    audit_path = args.out_dir / "zigzag_action_label_audit.json"
    audit["artifacts"]["audit"] = str(audit_path)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
