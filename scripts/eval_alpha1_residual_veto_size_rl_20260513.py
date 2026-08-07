#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, predict_policy_frame  # noqa: E402
from scripts import eval_alpha1_rl_exit_and_sizing_20260513 as alpha1  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _fill_price, _read  # noqa: E402
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig, _predict_cost_runner  # noqa: E402


MODEL_ID = "alpha1_residual_veto_size_rl_20260513"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha1_residual_veto_size_rl_20260513"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha1_residual_veto_size_rl_20260513_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha1_residual_veto_size_rl_20260513_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha1_residual_veto_size_rl_20260513_grid.csv"

FEATURES = [
    "is_parent",
    "is_deep",
    "side",
    "base_notional",
    "leverage",
    "take_profit",
    "stop_loss",
    "max_hold",
    "cooldown",
    "quality_score",
    "confidence",
    "deep_edge",
    "deep_margin",
    "vol_anchor",
    "volatility_z",
    "realized_vol_ratio",
    "bb_width",
    "net_taker_ratio",
    "taker_acceleration",
    "trade_intensity",
    "oi_change_rate",
    "last_funding_rate",
    "ai_dir_entropy",
    "ai_reward_risk",
    "ai_adverse_risk",
    "clean_regime_2024_unsup_v4_confidence",
    "clean_regime_2024_unsup_v4_entropy",
    "clean_regime_2024_unsup_v4_transition_risk",
]

ACTION_MULTS = {
    "skip": 0.0,
    "size_0_5": 0.5,
    "size_0_75": 0.75,
    "keep_1_0": 1.0,
}


@dataclass(frozen=True)
class ResidualConfig:
    name: str
    min_pred: float
    complexity_penalty: float
    parent_allow_skip: bool
    deep_allow_skip: bool


def _grid() -> list[ResidualConfig]:
    return [
        ResidualConfig("veto_mid_all", -0.0015, 0.00005, True, True),
        ResidualConfig("veto_tight_all", 0.0000, 0.00010, True, True),
        ResidualConfig("deep_only_mid", -0.0015, 0.00005, False, True),
        ResidualConfig("deep_only_tight", 0.0000, 0.00010, False, True),
        ResidualConfig("size_only_no_skip", -0.0040, 0.00005, False, False),
    ]


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1.get("trades", 0)) < 20:
        return -1e9 + float(c1.get("pnl", 0.0))
    return float(c1["pnl"] + 0.55 * c2["pnl"] + 0.35 * c3["pnl"] - 0.35 * abs(c1["mdd"]))


def _safe(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        x = float(row.get(col, default))
    except Exception:
        return float(default)
    return float(x) if np.isfinite(x) else float(default)


def _base_state(df: pd.DataFrame, i: int) -> dict[str, float]:
    row = df.iloc[i]
    out: dict[str, float] = {}
    for col in FEATURES:
        out[col] = _safe(row, col, 0.0)
    return out


def _parent_state(df: pd.DataFrame, dec: pd.Series, i: int) -> dict[str, float]:
    state = _base_state(df, i)
    state.update(
        {
            "is_parent": 1.0,
            "is_deep": 0.0,
            "side": float(int(dec.side)),
            "base_notional": float(dec.notional_exposure),
            "leverage": float(dec.leverage),
            "take_profit": float(dec.take_profit),
            "stop_loss": float(dec.stop_loss),
            "max_hold": float(dec.max_hold_bars),
            "cooldown": float(dec.cooldown_bars),
            "quality_score": float(getattr(dec, "quality_score", 0.0)),
            "confidence": float(getattr(dec, "confidence", 0.0)),
            "vol_anchor": float(v31._vol_anchor(df.iloc[i])),
        }
    )
    return state


def _deep_state(df: pd.DataFrame, i: int, side: int, edge: float, margin: float) -> dict[str, float]:
    state = _base_state(df, i)
    state.update(
        {
            "is_parent": 0.0,
            "is_deep": 1.0,
            "side": float(side),
            "base_notional": float(alpha1.ALPHA1_CFG.notional),
            "leverage": float(max(alpha1.ALPHA1_CFG.notional, 1.0)),
            "take_profit": float(alpha1.ALPHA1_CFG.base_tp),
            "stop_loss": float(alpha1.ALPHA1_CFG.base_sl),
            "max_hold": float(alpha1.ALPHA1_CFG.base_hold),
            "cooldown": float(alpha1.ALPHA1_CFG.cooldown),
            "quality_score": float(edge),
            "confidence": float(abs(margin)),
            "deep_edge": float(edge),
            "deep_margin": float(margin),
            "vol_anchor": float(v31._vol_anchor(df.iloc[i])),
        }
    )
    return state


def _x(state: dict[str, float]) -> np.ndarray:
    return np.asarray([[float(state.get(c, 0.0)) for c in FEATURES]], dtype=np.float32)


def _predict_v27_fast(model: torch.nn.Module, df: pd.DataFrame, seq_cols: list[str], norm: dict[str, np.ndarray]) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    arr = (
        df.loc[:, seq_cols]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    pad = np.zeros((v31.SEQ_LEN - 1, arr.shape[1]), dtype=np.float32)
    padded = np.vstack([pad, arr])
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=v31.SEQ_LEN, axis=0)
    if windows.shape[1] == arr.shape[1]:
        windows = windows.transpose(0, 2, 1)
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    outs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(df), 4096):
            seqs = np.ascontiguousarray(windows[start : start + 4096])
            x = ((seqs - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
            outs.append(model(torch.from_numpy(x).to(device)).detach().cpu().numpy())
    return np.vstack(outs).astype(np.float32)


def _effective_v31_thresholds(*, entry_edge: float, entry_vol_anchor: float, mfe: float, hold: int) -> tuple[float, float]:
    return alpha1._effective_v31_thresholds(alpha1.ALPHA1_CFG, entry_edge=entry_edge, entry_vol_anchor=entry_vol_anchor, mfe=mfe, hold=hold)


def _simulate_trade(
    df: pd.DataFrame,
    close: np.ndarray,
    signal_i: int,
    *,
    side: int,
    notional: float,
    owner: str,
    take_profit: float,
    stop_loss: float,
    max_hold: int,
    entry_edge: float,
    fee: float,
    slip: float,
) -> tuple[float, str]:
    if notional <= 0.0 or side == 0:
        return 0.0, "skip"
    fill_i = min(signal_i + 1, len(df) - 1)
    entry_px = _fill_price(df, fill_i, side, slip, entry=True)
    cash = 1.0 - fee * notional
    mfe = 0.0
    entry_vol_anchor = v31._vol_anchor(df.iloc[signal_i]) * notional
    for j in range(signal_i + 1, min(signal_i + max(1, int(max_hold)) + 3, len(df) - 1)):
        px = float(close[j])
        raw = (px * (1.0 - slip) - entry_px) / max(entry_px, 1e-12) if side > 0 else (entry_px - px * (1.0 + slip)) / max(entry_px, 1e-12)
        unreal = raw * notional
        mfe = max(mfe, unreal)
        hold = j - signal_i
        if owner == "deep_alpha":
            tp, sl = _effective_v31_thresholds(entry_edge=entry_edge, entry_vol_anchor=entry_vol_anchor, mfe=mfe, hold=hold)
            reason = "take_profit" if unreal >= tp else "stop_loss" if unreal <= -abs(sl) else "max_hold" if hold >= alpha1.ALPHA1_CFG.base_hold else ""
        else:
            reason = "take_profit" if take_profit > 0.0 and unreal >= take_profit else "stop_loss" if stop_loss > 0.0 and unreal <= -abs(stop_loss) else "max_hold" if max_hold > 0 and hold >= max_hold else ""
        if reason:
            exit_i = min(j + 1, len(df) - 1)
            exit_px = _fill_price(df, exit_i, side, slip, entry=False)
            raw2 = (exit_px - entry_px) / max(entry_px, 1e-12) if side > 0 else (entry_px - exit_px) / max(entry_px, 1e-12)
            out = cash * (1.0 + raw2 * notional)
            out -= cash * fee * notional
            return float(out - 1.0), reason
    return 0.0, "no_exit"


def _collect_candidates(df: pd.DataFrame, decisions: pd.DataFrame, deep_q: np.ndarray, *, fee: float, slip: float) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    rows: list[dict[str, float]] = []
    rewards: dict[str, list[float]] = {k: [] for k in ACTION_MULTS}
    meta = {"parent_candidates": 0, "deep_candidates": 0, "stop_loss_labels": 0}
    close = _close(df)
    cooldown = deep_cooldown = 0
    for i in range(0, len(df) - 3):
        if cooldown > 0:
            cooldown -= 1
            continue
        if deep_cooldown > 0:
            deep_cooldown -= 1
        dec = decisions.iloc[i]
        state: dict[str, float] | None = None
        owner = ""
        side = 0
        base_notional = 0.0
        tp = sl = edge = 0.0
        hold = 0
        if int(dec.action) != ACTION_CASH and int(dec.side) != 0:
            state = _parent_state(df, dec, i)
            owner = "parent"
            side = int(dec.side)
            base_notional = float(dec.notional_exposure)
            tp = float(dec.take_profit)
            sl = float(dec.stop_loss)
            hold = int(dec.max_hold_bars)
            cooldown = int(dec.cooldown_bars)
            meta["parent_candidates"] += 1
        elif deep_cooldown <= 0 and i >= v31.SEQ_LEN:
            ql, qs = float(deep_q[i, 0]), float(deep_q[i, 1])
            side = 1 if ql > qs else -1
            edge = max(ql, qs)
            margin = abs(ql - qs)
            if edge >= alpha1.ALPHA1_CFG.edge_th and margin >= alpha1.ALPHA1_CFG.margin_th:
                state = _deep_state(df, i, side, edge, margin)
                owner = "deep_alpha"
                base_notional = float(alpha1.ALPHA1_CFG.notional)
                tp = float(alpha1.ALPHA1_CFG.base_tp)
                sl = float(alpha1.ALPHA1_CFG.base_sl)
                hold = int(alpha1.ALPHA1_CFG.base_hold)
                deep_cooldown = int(alpha1.ALPHA1_CFG.cooldown)
                meta["deep_candidates"] += 1
        if state is None:
            continue
        rows.append(state)
        for name, mult in ACTION_MULTS.items():
            reward, reason = _simulate_trade(
                df,
                close,
                i,
                side=side,
                notional=base_notional * float(mult),
                owner=owner,
                take_profit=tp,
                stop_loss=sl,
                max_hold=hold,
                entry_edge=edge,
                fee=fee,
                slip=slip,
            )
            rewards[name].append(float(reward))
            if name == "keep_1_0" and reason == "stop_loss":
                meta["stop_loss_labels"] += 1
    x = np.asarray([[float(r.get(c, 0.0)) for c in FEATURES] for r in rows], dtype=np.float32)
    y = {k: np.asarray(v, dtype=np.float32) for k, v in rewards.items()}
    meta["rows"] = int(len(rows))
    return x, y, meta


def _train_policy(train: pd.DataFrame, decisions: pd.DataFrame, deep_q: np.ndarray, *, fee: float, slip: float) -> tuple[dict[str, Any], dict[str, Any]]:
    print(f"[{MODEL_ID}] collecting residual candidate labels", flush=True)
    x, y_by_action, meta = _collect_candidates(train, decisions, deep_q, fee=fee * 3.0, slip=slip * 3.0)
    print(f"[{MODEL_ID}] collected rows={len(x)} parent={meta['parent_candidates']} deep={meta['deep_candidates']}", flush=True)
    models = {}
    for name, y in y_by_action.items():
        print(f"[{MODEL_ID}] fitting critic action={name}", flush=True)
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            HistGradientBoostingRegressor(
                max_iter=90,
                learning_rate=0.055,
                max_leaf_nodes=31,
                l2_regularization=0.12,
                random_state=713 + len(name),
            ),
        )
        model.fit(x, y)
        models[name] = model
    return models, meta | {"actions": ACTION_MULTS, "features": FEATURES}


def _choose_action(models: dict[str, Any], state: dict[str, float], cfg: ResidualConfig) -> tuple[str, float]:
    xx = _x(state)
    preds = {name: float(model.predict(xx)[0]) for name, model in models.items()}
    is_parent = float(state.get("is_parent", 0.0)) > 0.5
    is_deep = float(state.get("is_deep", 0.0)) > 0.5
    if (is_parent and not cfg.parent_allow_skip) or (is_deep and not cfg.deep_allow_skip):
        preds["skip"] = -1e6
    for name, mult in ACTION_MULTS.items():
        if mult > 0:
            preds[name] -= float(cfg.complexity_penalty) * max(float(mult) - 0.5, 0.0)
    best = max(preds, key=preds.get)
    if preds[best] < float(cfg.min_pred):
        best = "skip" if ((is_parent and cfg.parent_allow_skip) or (is_deep and cfg.deep_allow_skip)) else "size_0_5"
    return best, float(ACTION_MULTS[best])


def backtest_residual(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    deep_q: np.ndarray,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    decisions: pd.DataFrame,
    models: dict[str, Any] | None = None,
    cfg: ResidualConfig | None = None,
) -> dict[str, Any]:
    close = _close(df)
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
    entry_edge = entry_margin = entry_vol_anchor = 0.0
    trades = wins = long_entries = short_entries = deep_entries = 0
    notional_sum = leverage_sum = 0.0
    exits: dict[str, int] = {}
    actions: dict[str, int] = {}
    residual_actions: dict[str, int] = {}

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
            if owner == "deep_alpha":
                effective_tp, effective_sl = _effective_v31_thresholds(entry_edge=entry_edge, entry_vol_anchor=entry_vol_anchor, mfe=mfe, hold=hold)
                if unreal >= effective_tp:
                    reason = "deep_alpha_take_profit"
                elif unreal <= -abs(effective_sl):
                    reason = "deep_alpha_stop_loss"
                elif hold >= int(alpha1.ALPHA1_CFG.base_hold):
                    reason = "deep_alpha_max_hold"
            if owner == "v21_2" and not reason:
                if take_profit > 0.0 and unreal >= take_profit:
                    reason = "v21_2_take_profit"
                elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                    reason = "v21_2_stop_loss"
                elif max_hold > 0 and hold >= max_hold:
                    reason = "v21_2_max_hold"
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
                pos = 0
                owner = ""
                cooldown = int(next_cooldown)
                next_cooldown = 0
                deep_cooldown = max(deep_cooldown, int(alpha1.ALPHA1_CFG.cooldown))
                add_done = False
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
            state = _parent_state(df, dec, i)
            act, mult = ("keep_1_0", 1.0) if models is None or cfg is None else _choose_action(models, state, cfg)
            residual_actions[f"parent_{act}"] = residual_actions.get(f"parent_{act}", 0) + 1
            if mult <= 0.0:
                continue
            fill_i = min(i + 1, len(df) - 1)
            pos = int(dec.side)
            owner = "v21_2"
            entry_price = _fill_price(df, fill_i, pos, slip_eff, entry=True)
            entry_equity = cash
            entry_idx = i
            parent_notional = min(float(dec.notional_exposure) * mult, add_cfg.max_entry_notional)
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
            continue
        if deep_cooldown <= 0 and i >= v31.SEQ_LEN:
            ql, qs = float(deep_q[i, 0]), float(deep_q[i, 1])
            side = 1 if ql > qs else -1
            edge = max(ql, qs)
            margin = abs(ql - qs)
            if edge >= alpha1.ALPHA1_CFG.edge_th and margin >= alpha1.ALPHA1_CFG.margin_th:
                state = _deep_state(df, i, side, edge, margin)
                act, mult = ("keep_1_0", 1.0) if models is None or cfg is None else _choose_action(models, state, cfg)
                residual_actions[f"deep_{act}"] = residual_actions.get(f"deep_{act}", 0) + 1
                if mult <= 0.0:
                    deep_cooldown = int(alpha1.ALPHA1_CFG.cooldown)
                    continue
                fill_i = min(i + 1, len(df) - 1)
                pos = side
                owner = "deep_alpha"
                entry_price = _fill_price(df, fill_i, pos, slip_eff, entry=True)
                entry_equity = cash
                entry_idx = i
                parent_notional = float(alpha1.ALPHA1_CFG.notional) * mult
                notional = parent_notional
                take_profit = float(alpha1.ALPHA1_CFG.base_tp)
                stop_loss = float(alpha1.ALPHA1_CFG.base_sl)
                max_hold = int(alpha1.ALPHA1_CFG.base_hold)
                next_cooldown = int(alpha1.ALPHA1_CFG.cooldown)
                entry_edge = edge
                entry_margin = margin
                entry_vol_anchor = v31._vol_anchor(df.iloc[i]) * float(notional)
                cash -= cash * fee_eff * notional
                long_entries += int(pos > 0)
                short_entries += int(pos < 0)
                deep_entries += 1
                notional_sum += notional
                leverage_sum += max(notional, 1.0)
                mfe = mae = 0.0
                add_done = True
                actions["deep_entry"] = actions.get("deep_entry", 0) + 1
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
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / v31._days(df)),
        "deep_entries": int(deep_entries),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(notional_sum / n),
        "avg_leverage": float(leverage_sum / n),
        "exits": exits,
        "runner_actions": actions,
        "residual_actions": residual_actions,
    }


def main() -> int:
    print(f"[{MODEL_ID}] loading artifacts", flush=True)
    parent = joblib.load(v31.DEFAULT_PARENT)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    base = dict(parent["config"])
    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    train = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    parent_audit = _audit_contract(train_all, eval_df, list(parent.get("feature_cols") or []))
    print(f"[{MODEL_ID}] predicting frozen V27 utilities", flush=True)
    train_q = _predict_v27_fast(v27_model, train, v27_payload["seq_cols"], v27_payload["norm"])
    val_q = _predict_v27_fast(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = _predict_v27_fast(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    print(f"[{MODEL_ID}] predicting alpha1 parent decisions", flush=True)
    train_dec = predict_policy_frame(parent, train, close=_close(train))
    val_dec = predict_policy_frame(parent, val, close=_close(val))
    eval_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))
    models, train_meta = _train_policy(train, train_dec, train_q, fee=float(base["fee"]), slip=float(base["slip"]))

    grid_rows: list[dict[str, Any]] = []
    selected: ResidualConfig | None = None
    best_score = -1e18
    for cfg in _grid():
        print(f"[{MODEL_ID}] selection config={cfg.name}", flush=True)
        v1 = backtest_residual(val, parent, jackpot_model, add_cfg, val_q, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=1.0, decisions=val_dec, models=models, cfg=cfg)
        v2 = backtest_residual(val, parent, jackpot_model, add_cfg, val_q, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=2.0, decisions=val_dec, models=models, cfg=cfg)
        v3 = backtest_residual(val, parent, jackpot_model, add_cfg, val_q, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=3.0, decisions=val_dec, models=models, cfg=cfg)
        score = _score(v1, v2, v3)
        grid_rows.append({**asdict(cfg), "score": score, "val_pnl": v1["pnl"], "val_mdd": v1["mdd"], "val_trades": v1["trades"], "val_deep_entries": v1["deep_entries"], "val_c2_pnl": v2["pnl"], "val_c3_pnl": v3["pnl"]})
        if score > best_score:
            best_score = score
            selected = cfg
    assert selected is not None
    experiments = []
    for name, kwargs in [
        ("alpha1", {"models": None, "cfg": None}),
        (f"alpha1_residual::{selected.name}", {"models": models, "cfg": selected}),
    ]:
        print(f"[{MODEL_ID}] OOS experiment={name}", flush=True)
        metrics = {
            f"cost{mult}": backtest_residual(
                eval_df,
                parent,
                jackpot_model,
                add_cfg,
                eval_q,
                fee=float(base["fee"]),
                slip=float(base["slip"]),
                cost_mult=float(mult),
                decisions=eval_dec,
                **kwargs,
            )
            for mult in (1, 2, 3)
        }
        experiments.append({"name": name, "metrics": metrics, "score": _score(metrics["cost1"], metrics["cost2"], metrics["cost3"])})
        print(f"[{MODEL_ID}] {name} cost1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f} cost2={metrics['cost2']['pnl']:.2f} cost3={metrics['cost3']['pnl']:.2f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model_id": MODEL_ID, "models": models, "selected_config": asdict(selected), "features": FEATURES, "actions": ACTION_MULTS, "train_meta": train_meta}, OUT_DIR / "residual_veto_size_rl.pkl")
    pd.DataFrame(grid_rows).sort_values("score", ascending=False).to_csv(GRID_OUT, index=False)
    blocking = list(parent_audit.get("blocking", []))
    warnings = list(parent_audit.get("warnings", []))
    best = max(experiments, key=lambda e: e["score"])
    if best["metrics"]["cost1"]["pnl"] <= alpha1.ALPHA1_BASELINE["cost1"]["pnl"]:
        warnings.append("best_did_not_beat_alpha1_cost1")
    if best["metrics"]["cost2"]["pnl"] <= alpha1.ALPHA1_BASELINE["cost2"]["pnl"] * 0.9:
        warnings.append("best_cost2_below_90pct_alpha1")
    if best["metrics"]["cost3"]["pnl"] <= alpha1.ALPHA1_BASELINE["cost3"]["pnl"]:
        warnings.append("best_cost3_not_improved")
    if best["metrics"]["cost1"]["deep_entries"] < int(alpha1.ALPHA1_BASELINE.get("cost1", {}).get("deep_entries", 0) or 0):
        warnings.append("deep_entry_count_check_manual_required")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if not blocking and best["name"] != "alpha1" and best["metrics"]["cost1"]["pnl"] > alpha1.ALPHA1_BASELINE["cost1"]["pnl"] and best["metrics"]["cost2"]["pnl"] > 0.0 and best["metrics"]["cost3"]["pnl"] > alpha1.ALPHA1_BASELINE["cost3"]["pnl"] else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "parent_full_replacement": False,
        "direction_change_allowed": False,
        "new_entry_generation_allowed": False,
        "actions": ACTION_MULTS,
        "selected_config": asdict(selected),
        "train_meta": train_meta,
        "parent_audit": parent_audit,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Alpha1 residual event-level offline RL critic. Alpha1 parent/V27 generate candidates; controller may only keep, size down, or skip. Direction changes and new entries are forbidden.",
        "selected_config": asdict(selected),
        "experiments": experiments,
        "audit": audit,
        "artifacts": {"model": str(OUT_DIR / "residual_veto_size_rl.pkl"), "report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "grid": str(GRID_OUT)},
    }
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "best": best}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
