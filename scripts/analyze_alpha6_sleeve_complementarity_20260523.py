#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha6_catboost_5head_policy_20260522 import (  # noqa: E402
    DEFAULT_FEATURE_CSV,
    DEFAULT_LABEL_DIR,
    DEFAULT_SPEC_DIR,
    _days,
    _fill_price,
    _label_frame,
    _numeric_matrix,
    _read_feature_frame,
    _read_spec,
)
from scripts.alpha6_catboost_entry_quality_exit_policy_20260522 import (  # noqa: E402
    CONTEXT_COLS,
    TARGET_BUCKET_TO_HORIZON,
    _bucket_horizon,
    _exit_close_prob,
    _exit_state_vec,
    _threshold_for_bucket,
)
from scripts.train_alpha6_dsac_ensemble_router_20260523 import (  # noqa: E402
    MODEL_SPECS,
    _predict_bundle,
    _target_horizon_bucket,
)


@dataclass
class Expert:
    name: str
    prefix: Path
    bundle: dict[str, Any]
    summary: dict[str, Any]
    dec: pd.DataFrame
    x: np.ndarray
    entry_threshold: float
    exit_threshold: float | tuple[float, ...]


def _parse_exit_threshold(raw: Any) -> float | tuple[float, ...]:
    if isinstance(raw, str) and "," in raw:
        return tuple(float(x.strip()) for x in raw.split(",") if x.strip())
    if isinstance(raw, (list, tuple)):
        return tuple(float(x) for x in raw)
    return float(raw)


def _load_frame(variant: str) -> pd.DataFrame:
    spec = _read_spec(DEFAULT_SPEC_DIR, variant)
    feat, _, _ = _read_feature_frame(DEFAULT_FEATURE_CSV, list(spec["features"]), CONTEXT_COLS)
    frame = feat.merge(_label_frame(DEFAULT_LABEL_DIR), on="timestamp", how="inner")
    return frame.sort_values("timestamp").reset_index(drop=True)


def _load_experts(variant: str) -> tuple[pd.DataFrame, list[Expert]]:
    full = _load_frame(variant)
    val_mask = full["dataset_split"].astype(str).str.lower().to_numpy() != "train"
    frame = full.loc[val_mask].reset_index(drop=True)
    experts: list[Expert] = []
    for name, prefix in MODEL_SPECS:
        bundle = joblib.load(f"{prefix}_bundle.joblib")
        summary = json.loads(Path(f"{prefix}_summary.json").read_text())
        pred, x_all = _predict_bundle(bundle, full)
        pred = pred.loc[val_mask].reset_index(drop=True)
        x = np.asarray(x_all[val_mask], dtype=np.float64)
        dec = pd.DataFrame(
            {
                "action": pred["action"].to_numpy(dtype=np.int64),
                "quality_score": pred["quality"].to_numpy(dtype=np.float64),
                "confidence": pred["confidence"].to_numpy(dtype=np.float64),
                "target_bucket": pred["target_bucket"].to_numpy(dtype=np.int64),
                "target_horizon": pred["target_horizon"].to_numpy(dtype=np.int64),
                "notional": np.full(len(pred), float(bundle.get("config", {}).get("fixed_notional", 0.25))),
            }
        )
        best = summary["best"]
        experts.append(
            Expert(
                name=name,
                prefix=Path(prefix),
                bundle=bundle,
                summary=summary,
                dec=dec,
                x=x,
                entry_threshold=float(best["entry_threshold"]),
                exit_threshold=_parse_exit_threshold(best.get("exit_threshold", 0.55)),
            )
        )
    return frame, experts


def _empty_summary() -> dict[str, Any]:
    return {
        "pnl": 0.0,
        "mdd": 0.0,
        "trades": 0,
        "wr": 0.0,
        "trades_per_day": 0.0,
        "long_entries": 0,
        "short_entries": 0,
        "avg_notional": 0.0,
        "exit_model_closes": 0,
        "missed_entries": 0,
        "exits": {},
        "active_bars": 0,
    }


def _replay_single(
    frame: pd.DataFrame,
    expert: Expert,
    *,
    fee: float,
    slip: float,
    min_exit_hold: int,
    state_horizon: int,
    exit_on_flip: bool,
    allowed_entry_mask: np.ndarray | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    open_px = pd.to_numeric(frame["open"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    _ = high, low, open_px
    atr = pd.to_numeric(frame.get("atr14_pct", 0.003), errors="coerce").fillna(0.003).to_numpy(dtype=np.float64)
    dec = expert.dec.reset_index(drop=True)
    x_val = expert.x
    exit_model = expert.bundle["exit_model"]
    expected = expert.bundle.get("expected_return_by_bucket") or {k: 0.01 for k in TARGET_BUCKET_TO_HORIZON}
    exit_meta = expert.bundle.get("exit_meta", {})
    regime_drift = bool(exit_meta.get("regime_drift", False))
    capture_ratio = bool(exit_meta.get("capture_ratio", False))
    if allowed_entry_mask is None:
        allowed_entry_mask = np.ones(len(frame), dtype=bool)
    else:
        allowed_entry_mask = np.asarray(allowed_entry_mask, dtype=bool)

    cash = 1.0
    peak = 1.0
    mdd = 0.0
    side = 0
    entry = 0.0
    entry_i = 0
    entry_fill_i = 0
    entry_equity = 1.0
    hold = 0
    mae = mfe = 0.0
    exposure = 0.0
    target_horizon = int(state_horizon)
    target_bucket = 4
    trades = wins = long_entries = short_entries = exit_model_closes = missed_entries = 0
    exposure_sum = 0.0
    exits: dict[str, int] = {}
    trade_rows: list[dict[str, Any]] = []
    active_mask = np.zeros(len(frame), dtype=bool)

    def equity(i: int) -> float:
        if side == 0:
            return cash
        px = close[int(np.clip(i, 0, len(close) - 1))]
        raw = (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)
        return cash * (1.0 + raw * exposure)

    def enter(i: int, new_side: int, notional: float, horizon: int, bucket: int) -> None:
        nonlocal side, entry, entry_i, entry_fill_i, entry_equity, hold, mae, mfe, exposure
        nonlocal target_horizon, target_bucket, cash, exposure_sum, long_entries, short_entries
        fill_i = min(i + 1, len(frame) - 1)
        side = int(new_side)
        entry_i = int(i)
        entry_fill_i = int(fill_i)
        exposure = float(np.clip(notional, 0.01, 2.0))
        target_horizon = int(np.clip(horizon, 2, state_horizon))
        target_bucket = int(np.clip(bucket, 0, 4))
        entry = _fill_price(frame, fill_i, side, slip, entry=True)
        entry_equity = cash
        cash -= cash * fee * exposure
        hold = 0
        mae = mfe = 0.0
        exposure_sum += exposure
        long_entries += int(side > 0)
        short_entries += int(side < 0)

    def exit_pos(i: int, reason: str) -> None:
        nonlocal side, entry, cash, hold, mae, mfe, exposure, target_horizon, target_bucket, trades, wins
        fill_i = min(i + 1, len(frame) - 1)
        fill_px = _fill_price(frame, fill_i, side, slip, entry=False)
        raw = (fill_px - entry) / max(entry, 1e-12) if side > 0 else (entry - fill_px) / max(entry, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * exposure)
        cash -= before * fee * exposure
        pnl_pct = (cash / max(entry_equity, 1e-12) - 1.0) * 100.0
        trades += 1
        wins += int(cash > entry_equity)
        exits[reason] = exits.get(reason, 0) + 1
        active_mask[entry_i : int(i) + 1] = True
        trade_rows.append(
            {
                "expert": expert.name,
                "entry_idx": int(entry_i),
                "exit_idx": int(i),
                "entry_time": str(frame.iloc[entry_i]["timestamp"]),
                "exit_time": str(frame.iloc[int(i)]["timestamp"]),
                "side": "LONG" if side > 0 else "SHORT",
                "entry_px": float(entry),
                "exit_px": float(fill_px),
                "entry_fill_idx": int(entry_fill_i),
                "exit_fill_idx": int(fill_i),
                "hold_bars": int(hold),
                "target_horizon": int(target_horizon),
                "target_bucket": int(target_bucket),
                "exposure": float(exposure),
                "raw_ret": float(raw),
                "pnl_pct_on_equity": float(pnl_pct),
                "mae": float(mae),
                "mfe": float(mfe),
                "reason": reason,
            }
        )
        side = 0
        entry = 0.0
        hold = 0
        mae = mfe = exposure = 0.0
        target_horizon = int(state_horizon)
        target_bucket = 4

    for i in range(len(frame) - 2):
        row = dec.iloc[i]
        desired = int(row.action) if float(row.quality_score) >= expert.entry_threshold and bool(allowed_entry_mask[i]) else 0
        closed_this_bar = False
        if side != 0:
            hold += 1
            px = close[i]
            raw = (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)
            mae = max(mae, max(0.0, -raw * exposure))
            mfe = max(mfe, max(0.0, raw * exposure))
            if hold >= int(min_exit_hold):
                state = _exit_state_vec(
                    frame,
                    side=side,
                    entry_idx=entry_i,
                    current_idx=i,
                    entry_px=entry,
                    px=px,
                    hold=hold,
                    horizon=int(target_horizon),
                    mae=mae,
                    mfe=mfe,
                    target_bucket=target_bucket,
                    regime_drift=regime_drift,
                    capture_ratio=capture_ratio,
                    expected_return=float(expected.get(target_bucket, 0.01)),
                )
                close_prob = _exit_close_prob(exit_model, x_val[i], state)
                if close_prob >= _threshold_for_bucket(expert.exit_threshold, target_bucket):
                    exit_model_closes += 1
                    exit_pos(i, "exit_model")
                    closed_this_bar = True
                elif exit_on_flip and desired != 0 and ((desired == 1 and side < 0) or (desired == 2 and side > 0)):
                    exit_pos(i, "model_flip")
                    closed_this_bar = True
        eq = equity(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if side == 0 and desired != 0 and not closed_this_bar:
            enter(
                i,
                1 if desired == 1 else -1,
                float(row.notional),
                int(getattr(row, "target_horizon", state_horizon)),
                int(getattr(row, "target_bucket", 4)),
            )
    if side != 0:
        exit_pos(len(frame) - 2, "end")
    if trades == 0:
        summary = _empty_summary()
    else:
        summary = {
            "pnl": float((cash - 1.0) * 100.0),
            "mdd": float(mdd * 100.0),
            "trades": int(trades),
            "wr": float(wins / max(trades, 1)),
            "trades_per_day": float(trades / _days(frame)),
            "long_entries": int(long_entries),
            "short_entries": int(short_entries),
            "missed_entries": int(missed_entries),
            "avg_notional": float(exposure_sum / max(trades, 1)),
            "exit_model_closes": int(exit_model_closes),
            "exits": exits,
            "active_bars": int(active_mask.sum()),
        }
    return summary, pd.DataFrame(trade_rows), active_mask


def _replay_priority_stack(
    frame: pd.DataFrame,
    primary: Expert,
    secondary: Expert,
    *,
    fee: float,
    slip: float,
    min_exit_hold: int,
    state_horizon: int,
    exit_on_flip: bool,
    preempt_secondary: bool,
    switch_on_preempt: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    experts = [primary, secondary]
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    active = -1
    side = 0
    entry = 0.0
    entry_i = 0
    entry_fill_i = 0
    entry_equity = 1.0
    hold = 0
    mae = mfe = exposure = 0.0
    target_horizon = int(state_horizon)
    target_bucket = 4
    trades = wins = long_entries = short_entries = exit_model_closes = 0
    exposure_sum = 0.0
    exits: dict[str, int] = {}
    trade_rows: list[dict[str, Any]] = []

    def desired_for(e: Expert, i: int) -> int:
        row = e.dec.iloc[i]
        return int(row.action) if float(row.quality_score) >= e.entry_threshold else 0

    def equity(i: int) -> float:
        if side == 0:
            return cash
        px = close[int(np.clip(i, 0, len(close) - 1))]
        raw = (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)
        return cash * (1.0 + raw * exposure)

    def enter(i: int, expert_idx: int, new_side: int) -> None:
        nonlocal active, side, entry, entry_i, entry_fill_i, entry_equity, hold, mae, mfe, exposure
        nonlocal target_horizon, target_bucket, cash, exposure_sum, long_entries, short_entries
        e = experts[expert_idx]
        row = e.dec.iloc[i]
        fill_i = min(i + 1, len(frame) - 1)
        active = int(expert_idx)
        side = int(new_side)
        entry_i = int(i)
        entry_fill_i = int(fill_i)
        exposure = float(np.clip(float(row.notional), 0.01, 2.0))
        target_horizon = int(np.clip(int(getattr(row, "target_horizon", state_horizon)), 2, state_horizon))
        target_bucket = int(np.clip(int(getattr(row, "target_bucket", 4)), 0, 4))
        entry = _fill_price(frame, fill_i, side, slip, entry=True)
        entry_equity = cash
        cash -= cash * fee * exposure
        hold = 0
        mae = mfe = 0.0
        exposure_sum += exposure
        long_entries += int(side > 0)
        short_entries += int(side < 0)

    def exit_pos(i: int, reason: str) -> None:
        nonlocal active, side, entry, cash, hold, mae, mfe, exposure, target_horizon, target_bucket, trades, wins
        e = experts[active]
        fill_i = min(i + 1, len(frame) - 1)
        fill_px = _fill_price(frame, fill_i, side, slip, entry=False)
        raw = (fill_px - entry) / max(entry, 1e-12) if side > 0 else (entry - fill_px) / max(entry, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * exposure)
        cash -= before * fee * exposure
        pnl_pct = (cash / max(entry_equity, 1e-12) - 1.0) * 100.0
        trades += 1
        wins += int(cash > entry_equity)
        exits[reason] = exits.get(reason, 0) + 1
        trade_rows.append(
            {
                "expert": e.name,
                "entry_idx": int(entry_i),
                "exit_idx": int(i),
                "entry_time": str(frame.iloc[entry_i]["timestamp"]),
                "exit_time": str(frame.iloc[int(i)]["timestamp"]),
                "side": "LONG" if side > 0 else "SHORT",
                "entry_px": float(entry),
                "exit_px": float(fill_px),
                "entry_fill_idx": int(entry_fill_i),
                "exit_fill_idx": int(fill_i),
                "hold_bars": int(hold),
                "target_horizon": int(target_horizon),
                "target_bucket": int(target_bucket),
                "exposure": float(exposure),
                "raw_ret": float(raw),
                "pnl_pct_on_equity": float(pnl_pct),
                "mae": float(mae),
                "mfe": float(mfe),
                "reason": reason,
            }
        )
        active = -1
        side = 0
        entry = 0.0
        hold = 0
        mae = mfe = exposure = 0.0
        target_horizon = int(state_horizon)
        target_bucket = 4

    for i in range(len(frame) - 2):
        primary_desired = desired_for(primary, i)
        secondary_desired = desired_for(secondary, i)
        closed_this_bar = False
        if side != 0:
            e = experts[active]
            hold += 1
            px = close[i]
            raw = (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)
            mae = max(mae, max(0.0, -raw * exposure))
            mfe = max(mfe, max(0.0, raw * exposure))
            active_desired = primary_desired if active == 0 else secondary_desired
            if active == 1 and preempt_secondary and primary_desired != 0:
                exit_pos(i, "primary_preempt")
                closed_this_bar = True
                if switch_on_preempt:
                    enter(i, 0, 1 if primary_desired == 1 else -1)
            elif hold >= int(min_exit_hold):
                expected = e.bundle.get("expected_return_by_bucket") or {k: 0.01 for k in TARGET_BUCKET_TO_HORIZON}
                exit_meta = e.bundle.get("exit_meta", {})
                state = _exit_state_vec(
                    frame,
                    side=side,
                    entry_idx=entry_i,
                    current_idx=i,
                    entry_px=entry,
                    px=px,
                    hold=hold,
                    horizon=int(target_horizon),
                    mae=mae,
                    mfe=mfe,
                    target_bucket=target_bucket,
                    regime_drift=bool(exit_meta.get("regime_drift", False)),
                    capture_ratio=bool(exit_meta.get("capture_ratio", False)),
                    expected_return=float(expected.get(target_bucket, 0.01)),
                )
                close_prob = _exit_close_prob(e.bundle["exit_model"], e.x[i], state)
                if close_prob >= _threshold_for_bucket(e.exit_threshold, target_bucket):
                    exit_model_closes += 1
                    exit_pos(i, "exit_model")
                    closed_this_bar = True
                elif exit_on_flip and active_desired != 0 and ((active_desired == 1 and side < 0) or (active_desired == 2 and side > 0)):
                    exit_pos(i, "model_flip")
                    closed_this_bar = True
        eq = equity(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if side == 0 and not closed_this_bar:
            if primary_desired != 0:
                enter(i, 0, 1 if primary_desired == 1 else -1)
            elif secondary_desired != 0:
                enter(i, 1, 1 if secondary_desired == 1 else -1)
    if side != 0:
        exit_pos(len(frame) - 2, "end")
    if trades == 0:
        return _empty_summary(), pd.DataFrame(trade_rows)
    return (
        {
            "pnl": float((cash - 1.0) * 100.0),
            "mdd": float(mdd * 100.0),
            "trades": int(trades),
            "wr": float(wins / max(trades, 1)),
            "trades_per_day": float(trades / _days(frame)),
            "long_entries": int(long_entries),
            "short_entries": int(short_entries),
            "avg_notional": float(exposure_sum / max(trades, 1)),
            "exit_model_closes": int(exit_model_closes),
            "exits": exits,
        },
        pd.DataFrame(trade_rows),
    )


def _replay_priority_stack_many(
    frame: pd.DataFrame,
    experts: list[Expert],
    *,
    fee: float,
    slip: float,
    min_exit_hold: int,
    state_horizon: int,
    exit_on_flip: bool,
    switch_on_preempt: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    active = -1
    side = 0
    entry = 0.0
    entry_i = 0
    entry_fill_i = 0
    entry_equity = 1.0
    hold = 0
    mae = mfe = exposure = 0.0
    target_horizon = int(state_horizon)
    target_bucket = 4
    trades = wins = long_entries = short_entries = exit_model_closes = 0
    exposure_sum = 0.0
    exits: dict[str, int] = {}
    trade_rows: list[dict[str, Any]] = []

    def desired_for(e: Expert, i: int) -> int:
        row = e.dec.iloc[i]
        return int(row.action) if float(row.quality_score) >= e.entry_threshold else 0

    def first_desired(desired: list[int], *, before_idx: int | None = None) -> int:
        limit = len(desired) if before_idx is None else max(0, int(before_idx))
        for idx in range(limit):
            if desired[idx] != 0:
                return idx
        return -1

    def equity(i: int) -> float:
        if side == 0:
            return cash
        px = close[int(np.clip(i, 0, len(close) - 1))]
        raw = (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)
        return cash * (1.0 + raw * exposure)

    def enter(i: int, expert_idx: int, new_side: int) -> None:
        nonlocal active, side, entry, entry_i, entry_fill_i, entry_equity, hold, mae, mfe, exposure
        nonlocal target_horizon, target_bucket, cash, exposure_sum, long_entries, short_entries
        e = experts[expert_idx]
        row = e.dec.iloc[i]
        fill_i = min(i + 1, len(frame) - 1)
        active = int(expert_idx)
        side = int(new_side)
        entry_i = int(i)
        entry_fill_i = int(fill_i)
        exposure = float(np.clip(float(row.notional), 0.01, 2.0))
        target_horizon = int(np.clip(int(getattr(row, "target_horizon", state_horizon)), 2, state_horizon))
        target_bucket = int(np.clip(int(getattr(row, "target_bucket", 4)), 0, 4))
        entry = _fill_price(frame, fill_i, side, slip, entry=True)
        entry_equity = cash
        cash -= cash * fee * exposure
        hold = 0
        mae = mfe = 0.0
        exposure_sum += exposure
        long_entries += int(side > 0)
        short_entries += int(side < 0)

    def exit_pos(i: int, reason: str) -> None:
        nonlocal active, side, entry, cash, hold, mae, mfe, exposure, target_horizon, target_bucket, trades, wins
        e = experts[active]
        fill_i = min(i + 1, len(frame) - 1)
        fill_px = _fill_price(frame, fill_i, side, slip, entry=False)
        raw = (fill_px - entry) / max(entry, 1e-12) if side > 0 else (entry - fill_px) / max(entry, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * exposure)
        cash -= before * fee * exposure
        pnl_pct = (cash / max(entry_equity, 1e-12) - 1.0) * 100.0
        trades += 1
        wins += int(cash > entry_equity)
        exits[reason] = exits.get(reason, 0) + 1
        trade_rows.append(
            {
                "expert": e.name,
                "entry_idx": int(entry_i),
                "exit_idx": int(i),
                "entry_time": str(frame.iloc[entry_i]["timestamp"]),
                "exit_time": str(frame.iloc[int(i)]["timestamp"]),
                "side": "LONG" if side > 0 else "SHORT",
                "entry_px": float(entry),
                "exit_px": float(fill_px),
                "entry_fill_idx": int(entry_fill_i),
                "exit_fill_idx": int(fill_i),
                "hold_bars": int(hold),
                "target_horizon": int(target_horizon),
                "target_bucket": int(target_bucket),
                "exposure": float(exposure),
                "raw_ret": float(raw),
                "pnl_pct_on_equity": float(pnl_pct),
                "mae": float(mae),
                "mfe": float(mfe),
                "reason": reason,
            }
        )
        active = -1
        side = 0
        entry = 0.0
        hold = 0
        mae = mfe = exposure = 0.0
        target_horizon = int(state_horizon)
        target_bucket = 4

    for i in range(len(frame) - 2):
        desired = [desired_for(e, i) for e in experts]
        closed_this_bar = False
        if side != 0:
            e = experts[active]
            hold += 1
            px = close[i]
            raw = (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)
            mae = max(mae, max(0.0, -raw * exposure))
            mfe = max(mfe, max(0.0, raw * exposure))
            preempt_idx = first_desired(desired, before_idx=active)
            if preempt_idx >= 0:
                exit_pos(i, "higher_priority_preempt")
                closed_this_bar = True
                if switch_on_preempt:
                    enter(i, preempt_idx, 1 if desired[preempt_idx] == 1 else -1)
            elif hold >= int(min_exit_hold):
                expected = e.bundle.get("expected_return_by_bucket") or {k: 0.01 for k in TARGET_BUCKET_TO_HORIZON}
                exit_meta = e.bundle.get("exit_meta", {})
                state = _exit_state_vec(
                    frame,
                    side=side,
                    entry_idx=entry_i,
                    current_idx=i,
                    entry_px=entry,
                    px=px,
                    hold=hold,
                    horizon=int(target_horizon),
                    mae=mae,
                    mfe=mfe,
                    target_bucket=target_bucket,
                    regime_drift=bool(exit_meta.get("regime_drift", False)),
                    capture_ratio=bool(exit_meta.get("capture_ratio", False)),
                    expected_return=float(expected.get(target_bucket, 0.01)),
                )
                close_prob = _exit_close_prob(e.bundle["exit_model"], e.x[i], state)
                if close_prob >= _threshold_for_bucket(e.exit_threshold, target_bucket):
                    exit_model_closes += 1
                    exit_pos(i, "exit_model")
                    closed_this_bar = True
                elif exit_on_flip and desired[active] != 0 and ((desired[active] == 1 and side < 0) or (desired[active] == 2 and side > 0)):
                    exit_pos(i, "model_flip")
                    closed_this_bar = True
        eq = equity(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if side == 0 and not closed_this_bar:
            idx = first_desired(desired)
            if idx >= 0:
                enter(i, idx, 1 if desired[idx] == 1 else -1)
    if side != 0:
        exit_pos(len(frame) - 2, "end")
    if trades == 0:
        return _empty_summary(), pd.DataFrame(trade_rows)
    return (
        {
            "pnl": float((cash - 1.0) * 100.0),
            "mdd": float(mdd * 100.0),
            "trades": int(trades),
            "wr": float(wins / max(trades, 1)),
            "trades_per_day": float(trades / _days(frame)),
            "long_entries": int(long_entries),
            "short_entries": int(short_entries),
            "avg_notional": float(exposure_sum / max(trades, 1)),
            "exit_model_closes": int(exit_model_closes),
            "exits": exits,
        },
        pd.DataFrame(trade_rows),
    )


def _summary_row(name: str, mode: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "mode": mode,
        "pnl": round(float(summary["pnl"]), 6),
        "mdd": round(float(summary["mdd"]), 6),
        "calmar": round(float(summary["pnl"]) / max(abs(float(summary["mdd"])), 1e-12), 6),
        "trades": int(summary["trades"]),
        "trades_per_day": round(float(summary["trades_per_day"]), 6),
        "wr": round(float(summary["wr"]), 6),
        "long_entries": int(summary["long_entries"]),
        "short_entries": int(summary["short_entries"]),
        "avg_notional": round(float(summary["avg_notional"]), 6),
        "exit_model_closes": int(summary["exit_model_closes"]),
        "exits": json.dumps(summary["exits"], sort_keys=True, ensure_ascii=False),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="current_tail111")
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--min-exit-hold", type=int, default=2)
    ap.add_argument("--state-horizon", type=int, default=96)
    ap.add_argument("--exit-on-flip", action="store_true")
    ap.add_argument("--no-preempt", action="store_true")
    ap.add_argument("--no-switch-on-preempt", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "tmp/causal_regen_20260516/alpha6_sleeve_complementarity_20260523")
    args = ap.parse_args()

    frame, experts = _load_experts(args.variant)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fee = 0.0004 * float(args.cost_mult)
    slip = 0.00015 * float(args.cost_mult)
    primary = experts[0]

    rows: list[dict[str, Any]] = []
    trade_paths: dict[str, Path] = {}
    active_masks: dict[str, np.ndarray] = {}
    for e in experts:
        summary, trades, active = _replay_single(
            frame,
            e,
            fee=fee,
            slip=slip,
            min_exit_hold=args.min_exit_hold,
            state_horizon=args.state_horizon,
            exit_on_flip=bool(args.exit_on_flip),
        )
        rows.append(_summary_row(e.name, "standalone", summary))
        path = args.out_dir / f"{e.name}_standalone_trades.csv"
        trades.to_csv(path, index=False)
        trade_paths[f"{e.name}_standalone"] = path
        active_masks[e.name] = active

    primary_active = active_masks[primary.name]
    primary_flat = ~primary_active
    for e in experts[1:]:
        summary, trades, _ = _replay_single(
            frame,
            e,
            fee=fee,
            slip=slip,
            min_exit_hold=args.min_exit_hold,
            state_horizon=args.state_horizon,
            exit_on_flip=bool(args.exit_on_flip),
            allowed_entry_mask=primary_flat,
        )
        rows.append(_summary_row(e.name, "diagnostic_primary_flat_only", summary))
        trades.to_csv(args.out_dir / f"{e.name}_primary_flat_only_trades.csv", index=False)

    for e in experts[1:]:
        summary, trades = _replay_priority_stack(
            frame,
            primary,
            e,
            fee=fee,
            slip=slip,
            min_exit_hold=args.min_exit_hold,
            state_horizon=args.state_horizon,
            exit_on_flip=bool(args.exit_on_flip),
            preempt_secondary=not bool(args.no_preempt),
            switch_on_preempt=not bool(args.no_switch_on_preempt),
        )
        rows.append(_summary_row(f"{primary.name}+{e.name}", "deployable_primary_priority", summary))
        trades.to_csv(args.out_dir / f"{primary.name}_plus_{e.name}_priority_trades.csv", index=False)

    orders = {
        "primary_coverage_sam_high": [0, 1, 5, 2],
        "primary_coverage_high_sam": [0, 1, 2, 5],
        "primary_all_diag_order": [0, 1, 5, 2, 4, 3],
        "primary_all_original_order": list(range(len(experts))),
    }
    for order_name, idxs in orders.items():
        ordered = [experts[i] for i in idxs]
        summary, trades = _replay_priority_stack_many(
            frame,
            ordered,
            fee=fee,
            slip=slip,
            min_exit_hold=args.min_exit_hold,
            state_horizon=args.state_horizon,
            exit_on_flip=bool(args.exit_on_flip),
            switch_on_preempt=not bool(args.no_switch_on_preempt),
        )
        rows.append(_summary_row(order_name, "deployable_multi_priority", summary))
        trades.to_csv(args.out_dir / f"{order_name}_trades.csv", index=False)

    rank = pd.DataFrame(rows).sort_values(["mode", "pnl"], ascending=[True, False]).reset_index(drop=True)
    rank.to_csv(args.out_dir / "sleeve_complementarity.csv", index=False)
    (args.out_dir / "sleeve_complementarity.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    print(f"[out] {args.out_dir}", flush=True)
    print(rank.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
