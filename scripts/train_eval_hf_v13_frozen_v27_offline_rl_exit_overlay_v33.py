#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, predict_policy_frame
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _days, _fill_price, _read
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import DeepAlphaTCN, _json_default
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig, _predict_cost_runner


MODEL_ID = "hf_v13_frozen_v27_offline_rl_exit_overlay_v33_20260511"
DEFAULT_PARENT = ROOT / "data/ensemble/supervised/hf_v13_clean_regime_margin110_20260511/v13_clean_regime_margin110.pkl"
DEFAULT_JACKPOT = ROOT / "data/ensemble/supervised/hf_v13_jackpot_runner_v21_2_20260511/v21_2_jackpot_runner.pkl"
DEFAULT_V27 = ROOT / "data/ensemble/supervised/hf_v13_deep_alpha_candidate_expansion_v27_20260511/v27_deep_alpha_candidate_expansion.pt"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_frozen_v27_offline_rl_exit_overlay_v33_20260511"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_frozen_v27_offline_rl_exit_overlay_v33_20260511_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_frozen_v27_offline_rl_exit_overlay_v33_20260511_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_frozen_v27_offline_rl_exit_overlay_v33_20260511_grid.csv"
SEQ_LEN = 72
V27_COST1 = 226.82447187089713
V27_COST2 = 123.11659362616143
V27_COST3 = 14.22783363158393
REVERSAL_FEATURES = [
    "deep_edge",
    "deep_margin",
    "side",
    "hold",
    "unreal",
    "mfe",
    "mae",
    "vol_anchor",
    "volatility_z",
    "realized_vol_ratio",
    "bb_width",
    "net_taker_ratio",
    "trade_intensity",
    "oi_change_rate",
    "funding_rate",
    "teacher_uncertainty",
    "teacher_tail_warning",
    "ai_dir_entropy",
    "clean_regime_transition_risk",
    "clean_regime_confidence",
    "clean_regime_entropy",
]


@dataclass(frozen=True)
class OverlayConfig:
    name: str
    edge_th: float
    margin_th: float
    notional: float
    cooldown: int
    close_p_th: float
    min_hold: int
    base_tp: float
    base_sl: float
    base_hold: int


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _safe_row_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        x = float(row.get(col, default))
    except Exception:
        return float(default)
    return float(x) if np.isfinite(x) else float(default)


def _vol_anchor(row: pd.Series) -> float:
    bbw = abs(_safe_row_float(row, "bb_width", 0.0))
    gk = abs(_safe_row_float(row, "garman_klass_vol", 0.0))
    rs = abs(_safe_row_float(row, "rogers_satchell_vol", 0.0))
    pk = abs(_safe_row_float(row, "parkinson_vol", 0.0))
    volz = abs(_safe_row_float(row, "volatility_z", 0.0))
    rv = abs(_safe_row_float(row, "realized_vol_ratio", 1.0))
    base = max(0.0015, bbw * 0.15, gk * 2.5, rs * 2.5, pk * 2.5)
    scale = base * (1.0 + 0.08 * min(volz, 3.0) + 0.05 * max(rv - 1.0, 0.0))
    return _clip(scale, 0.0015, 0.030)


def _seq_at(df: pd.DataFrame, idx: int, cols: list[str]) -> np.ndarray:
    start = max(0, idx - SEQ_LEN + 1)
    arr = (
        df.loc[start:idx, cols]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    if len(arr) < SEQ_LEN:
        arr = np.vstack([np.zeros((SEQ_LEN - len(arr), len(cols)), dtype=np.float32), arr])
    return arr[-SEQ_LEN:]


def _apply_norm(seqs: np.ndarray, norm: dict[str, np.ndarray]) -> np.ndarray:
    return ((seqs - norm["mean"][None, None, :]) / norm["std"][None, None, :]).astype(np.float32)


def _predict_all(model: DeepAlphaTCN, df: pd.DataFrame, seq_cols: list[str], norm: dict[str, np.ndarray]) -> np.ndarray:
    arr = (
        df.loc[:, seq_cols]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    pad = np.zeros((SEQ_LEN - 1, arr.shape[1]), dtype=np.float32)
    padded = np.vstack([pad, arr])
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=SEQ_LEN, axis=0)
    if windows.shape[1] == arr.shape[1]:
        windows = windows.transpose(0, 2, 1)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(df), 1024):
            seqs = np.ascontiguousarray(windows[start : start + 1024])
            x = _apply_norm(seqs, norm)
            out.append(model(torch.from_numpy(x)).numpy())
    return np.vstack(out).astype(np.float32)


def _load_v27(path: Path) -> tuple[dict[str, Any], DeepAlphaTCN]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = DeepAlphaTCN(len(payload["seq_cols"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return payload, model


def _overlay_grid() -> list[OverlayConfig]:
    return [
        OverlayConfig("v33_rl_p055_hold1", 0.010, 0.004, 1.2, 12, 0.55, 1, 0.045, 0.022, 48),
        OverlayConfig("v33_rl_p060_hold2", 0.010, 0.004, 1.2, 12, 0.60, 2, 0.045, 0.022, 48),
        OverlayConfig("v33_rl_p065_hold3", 0.010, 0.004, 1.2, 12, 0.65, 3, 0.045, 0.022, 48),
        OverlayConfig("v33_rl_p060_n1", 0.010, 0.004, 1.0, 12, 0.60, 2, 0.045, 0.022, 48),
        OverlayConfig("v33_rl_p055_n1", 0.010, 0.004, 1.0, 12, 0.55, 1, 0.045, 0.022, 48),
        OverlayConfig("v33_rl_p062_precision", 0.012, 0.005, 1.0, 12, 0.62, 2, 0.045, 0.022, 48),
    ]


def _deep_state_row(
    frame: pd.DataFrame,
    i: int,
    side: int,
    edge: float,
    margin: float,
    hold: int,
    unreal: float,
    mfe: float,
    mae: float,
) -> dict[str, float]:
    row = frame.iloc[i]
    vol_anchor = _vol_anchor(row)
    return {
        "deep_edge": float(edge),
        "deep_margin": float(margin),
        "side": float(side),
        "hold": float(hold),
        "unreal": float(unreal),
        "mfe": float(mfe),
        "mae": float(mae),
        "vol_anchor": float(vol_anchor),
        "volatility_z": _safe_row_float(row, "volatility_z", 0.0),
        "realized_vol_ratio": _safe_row_float(row, "realized_vol_ratio", 1.0),
        "bb_width": _safe_row_float(row, "bb_width", 0.0),
        "net_taker_ratio": _safe_row_float(row, "net_taker_ratio", 0.0),
        "trade_intensity": _safe_row_float(row, "trade_intensity", 0.0),
        "oi_change_rate": _safe_row_float(row, "oi_change_rate", 0.0),
        "funding_rate": _safe_row_float(row, "last_funding_rate", 0.0),
        "teacher_uncertainty": _safe_row_float(row, "teacher_uncertainty", 0.0),
        "teacher_tail_warning": _safe_row_float(row, "teacher_tail_warning", 0.0),
        "ai_dir_entropy": _safe_row_float(row, "ai_dir_entropy", 0.0),
        "clean_regime_transition_risk": _safe_row_float(row, "clean_regime_2024_unsup_v4_transition_risk", 0.0),
        "clean_regime_confidence": _safe_row_float(row, "clean_regime_2024_unsup_v4_confidence", 0.0),
        "clean_regime_entropy": _safe_row_float(row, "clean_regime_2024_unsup_v4_entropy", 0.0),
    }


def _state_array(state: dict[str, float]) -> np.ndarray:
    return np.asarray([[state.get(c, 0.0) for c in REVERSAL_FEATURES]], dtype=np.float32)


def _predict_close_prob(model: Any, state: dict[str, float]) -> float:
    x: Any = _state_array(state)
    if hasattr(model, "steps"):
        for _, step in model.steps[:-1]:
            x = step.transform(x)
        return float(model.steps[-1][1].predict_proba(x)[0, 1])
    return float(model.predict_proba(x)[0, 1])


def _offline_rl_episode_labels(
    episode_states: list[dict[str, float]],
    *,
    gamma: float = 0.985,
    step_penalty: float = 0.00012,
    close_margin: float = 0.00045,
) -> tuple[list[dict[str, float]], list[int]]:
    if not episode_states:
        return [], []
    labels = [0 for _ in episode_states]
    values = [0.0 for _ in episode_states]
    next_value = -1e9
    for idx in range(len(episode_states) - 1, -1, -1):
        state = episode_states[idx]
        close_value = float(state["unreal"])
        hold_value = gamma * next_value - step_penalty if idx < len(episode_states) - 1 else -1e9
        labels[idx] = int(close_value >= hold_value - close_margin)
        values[idx] = max(close_value, hold_value)
        next_value = values[idx]
    return episode_states, labels


def _collect_reversal_training(
    frame: pd.DataFrame,
    decisions: pd.DataFrame,
    deep_q: np.ndarray,
    base_cfg: OverlayConfig,
    *,
    fee: float,
    slip: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    close = _close(frame)
    fee_eff = fee * 2.0
    slip_eff = slip * 2.0
    pos = 0
    owner = ""
    entry_price = 0.0
    entry_idx = 0
    take_profit = stop_loss = 0.0
    max_hold = 0
    cooldown = deep_cooldown = 0
    edge = margin = 0.0
    notional = 0.0
    mfe = mae = 0.0
    rows: list[dict[str, float]] = []
    labels: list[int] = []
    episode_states: list[dict[str, float]] = []

    def flush_episode() -> None:
        nonlocal episode_states
        if not episode_states:
            return
        out_rows, out_labels = _offline_rl_episode_labels(episode_states)
        rows.extend(out_rows)
        labels.extend(out_labels)
        episode_states = []

    for i in range(0, len(frame) - 2):
        if pos != 0:
            px = float(close[i])
            raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
            unreal = raw * notional
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            hold = i - entry_idx
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "tp"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "sl"
            elif max_hold > 0 and hold >= max_hold:
                reason = "hold"
            if owner == "deep_alpha" and hold >= 1:
                episode_states.append(_deep_state_row(frame, i, pos, edge, margin, hold, unreal, mfe, mae))
            if reason:
                if owner == "deep_alpha":
                    flush_episode()
                pos = 0
                owner = ""
                cooldown = base_cfg.cooldown
                continue
        if pos != 0:
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        if deep_cooldown > 0:
            deep_cooldown -= 1
        dec = decisions.iloc[i]
        if int(dec.action) != ACTION_CASH and int(dec.side) != 0:
            pos = int(dec.side)
            owner = "v21_2"
            fill_i = min(i + 1, len(frame) - 1)
            entry_price = _fill_price(frame, fill_i, pos, slip_eff, entry=True)
            entry_idx = i
            notional = float(dec.notional_exposure)
            take_profit = float(dec.take_profit)
            stop_loss = float(dec.stop_loss)
            max_hold = int(dec.max_hold_bars)
            mfe = mae = 0.0
            continue
        if i >= SEQ_LEN:
            ql, qs = float(deep_q[i, 0]), float(deep_q[i, 1])
            side = 1 if ql > qs else -1
            e = max(ql, qs)
            m = abs(ql - qs)
            if e >= base_cfg.edge_th and m >= base_cfg.margin_th:
                pos = side
                owner = "deep_alpha"
                fill_i = min(i + 1, len(frame) - 1)
                entry_price = _fill_price(frame, fill_i, pos, slip_eff, entry=True)
                entry_idx = i
                notional = base_cfg.notional
                take_profit = base_cfg.base_tp
                stop_loss = base_cfg.base_sl
                max_hold = base_cfg.base_hold
                edge = e
                margin = m
                mfe = mae = 0.0
                deep_cooldown = base_cfg.cooldown
    if pos != 0 and owner == "deep_alpha":
        flush_episode()
    if not rows:
        raise RuntimeError("no offline RL training rows")
    return pd.DataFrame(rows), np.asarray(labels, dtype=np.int64)


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1["trades"]) < 20:
        return -1e9 + float(c1["pnl"])
    return float(c1["pnl"] + 0.35 * c2["pnl"] + 0.20 * c3["pnl"] - 0.35 * abs(c1["mdd"]) + 0.18 * min(c1.get("deep_entries", 0), 90))


def backtest(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    deep_q: np.ndarray,
    overlay_model: Any,
    cfg: OverlayConfig,
    *,
    fee: float,
    slip: float,
    cost_mult: float = 1.0,
    decisions: pd.DataFrame | None = None,
    record: bool = False,
) -> dict[str, Any]:
    close = _close(df)
    if decisions is None:
        decisions = predict_policy_frame(bundle, df, close=close)
    fee_eff = fee * cost_mult
    slip_eff = slip * cost_mult
    cash = peak = 1.0
    mdd = 0.0
    pos = 0
    owner = ""
    entry_price = entry_equity = 0.0
    entry_idx = 0
    parent_notional = notional = 0.0
    take_profit = stop_loss = 0.0
    max_hold = 0
    cooldown = next_cooldown = deep_cooldown = 0
    add_done = False
    mfe = mae = 0.0
    entry_edge = entry_margin = 0.0
    trades = wins = long_entries = short_entries = deep_entries = 0
    notional_sum = leverage_sum = 0.0
    exits: dict[str, int] = {}
    actions: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    open_record: dict[str, Any] | None = None

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
        unreal = raw * notional
        return cash * (1.0 + unreal), unreal

    for i in range(0, len(df) - 2):
        eq, unreal = mark(i)
        peak = max(peak, eq)
        dd_abs = max(0.0, 1.0 - eq / max(peak, 1e-12))
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if pos != 0:
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            hold = i - entry_idx
            reason = ""
            if owner == "deep_alpha" and hold >= cfg.min_hold:
                p_close = _predict_close_prob(
                    overlay_model,
                    _deep_state_row(df, i, pos, entry_edge, entry_margin, hold, unreal, mfe, mae),
                )
                if p_close >= cfg.close_p_th:
                    reason = "deep_alpha_rl_exit_overlay"
            if not reason:
                if take_profit > 0.0 and unreal >= take_profit:
                    reason = f"{owner}_take_profit"
                elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                    reason = f"{owner}_stop_loss"
                elif max_hold > 0 and hold >= max_hold:
                    reason = f"{owner}_max_hold"
            if owner == "v21_2" and not reason and not add_done and add_cfg.full_add_frac > 0.0 and unreal >= add_cfg.min_unrealized and hold >= add_cfg.min_bars_since_entry and dd_abs <= add_cfg.dd_block:
                state = {"parent_notional": parent_notional, "notional": notional, "bars_since_entry": hold, "unrealized": unreal, "mfe": mfe, "mae": mae, "drawdown_abs": dd_abs, "take_profit": take_profit, "stop_loss": stop_loss, "max_hold": max_hold}
                x = _feature_frame(df, bundle, decisions, i, state)
                _, _, _, q90, p_jackpot, p_bad, p_cost3 = _predict_cost_runner(jackpot_model, x)
                if p_jackpot >= add_cfg.jackpot_p and q90 >= add_cfg.jackpot_q90 and p_bad <= add_cfg.bad_cap and p_cost3 >= 0.40:
                    fill_i = min(i + 1, len(df) - 1)
                    delta = max(0.0, min(parent_notional * add_cfg.full_add_frac, parent_notional * add_cfg.max_total_mult - notional))
                    add_px = _fill_price(df, fill_i, pos, slip_eff, entry=True)
                    new_notional = notional + delta
                    entry_price = (entry_price * notional + add_px * delta) / max(new_notional, 1e-12)
                    before = cash
                    cash -= before * fee_eff * delta
                    notional = new_notional
                    actions["v21_add_on"] = actions.get("v21_add_on", 0) + 1
                else:
                    actions["v21_reject"] = actions.get("v21_reject", 0) + 1
                add_done = True
            if reason:
                fill_i = min(i + 1, len(df) - 1)
                exit_px = _fill_price(df, fill_i, pos, slip_eff, entry=False)
                raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * fee_eff * notional
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] = exits.get(reason, 0) + 1
                if record and open_record is not None:
                    out = dict(open_record)
                    out.update({"exit_signal_timestamp": str(df["timestamp"].iloc[i]), "exit_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "exit_reason": reason, "realized_net_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0), "final_notional_exposure": float(notional), "mfe_pct": float(mfe * 100.0), "mae_pct": float(mae * 100.0), "fee_exit_pct": float(fee_eff * notional * 100.0), "cash_after": float(cash)})
                    records.append(out)
                pos = 0
                owner = ""
                cooldown = int(next_cooldown)
                next_cooldown = 0
                deep_cooldown = max(deep_cooldown, int(cfg.cooldown))
                add_done = False
                open_record = None
                continue
        if pos != 0:
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        if deep_cooldown > 0:
            deep_cooldown -= 1
        dec = decisions.iloc[i]
        if int(dec.action) != ACTION_CASH and int(dec.side) != 0:
            fill_i = min(i + 1, len(df) - 1)
            pos = int(dec.side)
            owner = "v21_2"
            entry_price = _fill_price(df, fill_i, pos, slip_eff, entry=True)
            entry_equity = cash
            entry_idx = i
            parent_notional = min(float(dec.notional_exposure), add_cfg.max_entry_notional)
            notional = parent_notional
            take_profit = float(dec.take_profit)
            stop_loss = float(dec.stop_loss)
            max_hold = int(dec.max_hold_bars)
            next_cooldown = int(dec.cooldown_bars)
            cash -= cash * fee_eff * notional
            long_entries += int(pos > 0)
            short_entries += int(pos < 0)
            notional_sum += notional
            leverage_sum += float(dec.leverage)
            mfe = mae = 0.0
            add_done = False
            actions["v21_entry"] = actions.get("v21_entry", 0) + 1
            if record:
                open_record = {"entry_signal_timestamp": str(df["timestamp"].iloc[i]), "entry_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "owner": owner, "side": "LONG" if pos > 0 else "SHORT", "entry_price": float(entry_price), "notional_exposure": float(notional), "leverage": float(dec.leverage), "take_profit": float(take_profit), "stop_loss": float(stop_loss), "max_hold_bars": int(max_hold), "fee_entry_pct": float(fee_eff * notional * 100.0)}
            continue
        if deep_cooldown <= 0 and i >= SEQ_LEN:
            ql, qs = float(deep_q[i, 0]), float(deep_q[i, 1])
            side = 1 if ql > qs else -1
            edge = max(ql, qs)
            margin = abs(ql - qs)
            if edge >= cfg.edge_th and margin >= cfg.margin_th:
                fill_i = min(i + 1, len(df) - 1)
                pos = side
                owner = "deep_alpha"
                entry_price = _fill_price(df, fill_i, pos, slip_eff, entry=True)
                entry_equity = cash
                entry_idx = i
                parent_notional = float(cfg.notional)
                notional = float(cfg.notional)
                take_profit = float(cfg.base_tp)
                stop_loss = float(cfg.base_sl)
                max_hold = int(cfg.base_hold)
                next_cooldown = int(cfg.cooldown)
                entry_edge = edge
                entry_margin = margin
                cash -= cash * fee_eff * notional
                long_entries += int(pos > 0)
                short_entries += int(pos < 0)
                deep_entries += 1
                notional_sum += notional
                leverage_sum += max(notional, 1.0)
                mfe = mae = 0.0
                add_done = True
                actions["deep_entry"] = actions.get("deep_entry", 0) + 1
                if record:
                    open_record = {"entry_signal_timestamp": str(df["timestamp"].iloc[i]), "entry_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "owner": owner, "side": "LONG" if pos > 0 else "SHORT", "entry_price": float(entry_price), "notional_exposure": float(notional), "deep_q_long": ql, "deep_q_short": qs, "deep_edge": float(edge), "deep_margin": float(margin), "take_profit": float(take_profit), "stop_loss": float(stop_loss), "max_hold_bars": int(max_hold), "fee_entry_pct": float(fee_eff * notional * 100.0)}
    if pos != 0:
        fill_i = len(df) - 1
        exit_px = _fill_price(df, fill_i, pos, slip_eff, entry=False)
        raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * fee_eff * notional
        trades += 1
        wins += int(cash > entry_equity)
        exits["forced_end"] = exits.get("forced_end", 0) + 1
    n = max(long_entries + short_entries, 1)
    out = {"pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(trades), "wr": float(wins / max(trades, 1)), "trades_per_day": float(trades / _days(df)), "deep_entries": int(deep_entries), "long_entries": int(long_entries), "short_entries": int(short_entries), "avg_notional": float(notional_sum / n), "avg_leverage": float(leverage_sum / n), "exits": exits, "runner_actions": actions}
    if record:
        out["trade_records"] = records
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V33 frozen V27 with constrained offline-RL close/hold exit overlay.")
    p.add_argument("--parent-model", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--jackpot-model", type=Path, default=DEFAULT_JACKPOT)
    p.add_argument("--v27-model", type=Path, default=DEFAULT_V27)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print("[v33] loading models and data", flush=True)
    bundle = joblib.load(args.parent_model)
    jackpot_payload = joblib.load(args.jackpot_model)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = _load_v27(args.v27_model)
    base = dict(bundle["config"])
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    train = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    feature_audit = _audit_contract(train_all, eval_df, list(bundle.get("feature_cols") or []))
    print("[v33] predicting frozen V27 deep utilities", flush=True)
    train_q = _predict_all(v27_model, train, v27_payload["seq_cols"], v27_payload["norm"])
    val_q = _predict_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = _predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    print("[v33] predicting frozen parent policy decisions", flush=True)
    train_dec = predict_policy_frame(bundle, train, close=_close(train))
    val_dec = predict_policy_frame(bundle, val, close=_close(val))
    eval_dec = predict_policy_frame(bundle, eval_df, close=_close(eval_df))
    base_cfg = OverlayConfig("train_base", 0.010, 0.004, 1.2, 12, 0.60, 2, 0.045, 0.022, 48)
    print("[v33] collecting offline-RL close/hold labels", flush=True)
    x_train, y_train = _collect_reversal_training(train, train_dec, train_q, base_cfg, fee=float(base["fee"]), slip=float(base["slip"]))
    overlay_model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=500, C=0.75, class_weight="balanced", random_state=2027),
    )
    print(f"[v33] fitting offline-RL policy rows={len(x_train)} close_rate={float(np.mean(y_train)):.4f}", flush=True)
    overlay_model.fit(x_train.loc[:, REVERSAL_FEATURES].to_numpy(dtype=np.float32), y_train)

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    print("[v33] running validation grid", flush=True)
    for cfg in _overlay_grid():
        print(f"[v33] validation {cfg.name}", flush=True)
        v1 = backtest(val, bundle, jackpot_model, add_cfg, val_q, overlay_model, cfg, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=1.0, decisions=val_dec)
        v2 = backtest(val, bundle, jackpot_model, add_cfg, val_q, overlay_model, cfg, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=2.0, decisions=val_dec)
        v3 = backtest(val, bundle, jackpot_model, add_cfg, val_q, overlay_model, cfg, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=3.0, decisions=val_dec)
        row = {"config": asdict(cfg), "validation_cost1": v1, "validation_cost2": v2, "validation_cost3": v3, "selection_score": _score(v1, v2, v3)}
        rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    assert best is not None
    selected = OverlayConfig(**best["config"])
    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    print(f"[v33] selected {selected.name}; running 2026 fixed OOS", flush=True)
    for mult in (1, 2, 3):
        r = backtest(eval_df, bundle, jackpot_model, add_cfg, eval_q, overlay_model, selected, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=float(mult), decisions=eval_dec, record=(mult == 1))
        if mult == 1:
            ledger = pd.DataFrame(r.pop("trade_records", []))
            lp = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
            lp.parent.mkdir(parents=True, exist_ok=True)
            ledger.to_csv(lp, index=False)
            ledgers["cost1"] = str(lp)
        metrics[f"cost{mult}"] = r
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "v33_offline_rl_exit_overlay.pkl"
    joblib.dump({"overlay_model": overlay_model, "train_columns": list(x_train.columns), "selected_config": asdict(selected), "v27_model": str(args.v27_model)}, model_path)
    pd.DataFrame([{**{f"cfg_{k}": v for k, v in r["config"].items()}, "score": r["selection_score"], "val_pnl": r["validation_cost1"]["pnl"], "val_mdd": r["validation_cost1"]["mdd"], "val_trades": r["validation_cost1"]["trades"], "val_deep_entries": r["validation_cost1"].get("deep_entries", 0), "val_c2_pnl": r["validation_cost2"]["pnl"], "val_c3_pnl": r["validation_cost3"]["pnl"]} for r in rows]).to_csv(args.grid_out, index=False)
    blocking: list[str] = []
    warnings: list[str] = []
    if feature_audit["status"] != "pass":
        blocking.extend(feature_audit.get("blocking", []))
    warnings.extend(feature_audit.get("warnings", []))
    if metrics["cost1"]["pnl"] <= V27_COST1:
        warnings.append("oos_cost1_did_not_beat_v27")
    if metrics["cost2"]["pnl"] <= 0.0:
        warnings.append("cost2_not_survived")
    if metrics["cost3"]["pnl"] <= 0.0:
        warnings.append("cost3_not_survived")
    verdict = "promote" if not blocking and metrics["cost1"]["pnl"] > V27_COST1 and metrics["cost2"]["pnl"] > 0.0 and metrics["cost3"]["pnl"] > 0.0 else "iterate"
    audit = {"status": "pass" if not blocking else "fail", "verdict": verdict, "blocking": blocking, "warnings": warnings, "selection_uses_2026": False, "selection_window": "2025-10-01..2025-12-31", "oos_window": "2026 fixed OOS only after selection", "policy": "frozen_v27_offline_rl_exit_overlay_v33", "v27_entry_frozen": True, "v21_2_preserved": True, "deep_sleeve_only_when_parent_cash": True, "rl_formulation": "constrained offline dynamic-programming close/hold labels over V27 deep_alpha episodes", "feature_audit": feature_audit, "train_rows": int(len(x_train)), "close_rate": float(np.mean(y_train)), "selected_config": asdict(selected), "metrics": metrics, "baseline_v27": {"cost1": V27_COST1, "cost2": V27_COST2, "cost3": V27_COST3}}
    report = {"model_id": MODEL_ID, "design": "V33 freezes the trained V27 entry model and V21.2 jackpot parent. A constrained offline-RL policy learns a binary Hold/Close action for open deep_alpha positions using backward dynamic-programming labels over 2025 training episodes.", "selected_config": asdict(selected), "selection_result": best, "metrics": metrics, "audit": audit, "artifacts": {"model": str(model_path), "report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "ledgers": ledgers}}
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "model": str(model_path), "selected": asdict(selected), "metrics": metrics, "verdict": verdict}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
