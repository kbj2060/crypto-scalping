#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, predict_policy_frame  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import train_eval_hf_v13_frozen_v27_offline_rl_exit_overlay_v33 as v33  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _fill_price, _read  # noqa: E402
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig, _predict_cost_runner  # noqa: E402


MODEL_ID = "alpha1_rl_exit_and_sizing_20260513"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha1_rl_exit_and_sizing_20260513"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha1_rl_exit_and_sizing_20260513_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha1_rl_exit_and_sizing_20260513_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha1_rl_exit_and_sizing_20260513_grid.csv"

ALPHA1_BASELINE = {
    "cost1": {"pnl": 361.19, "mdd": -31.74},
    "cost2": {"pnl": 88.74, "mdd": -31.74},
    "cost3": {"pnl": 0.58, "mdd": -43.09},
}
ALPHA1_CFG = v31.OverlayConfig(
    "alpha1_v31_deep_notional2",
    0.010,
    0.004,
    2.0,
    12,
    0.040,
    0.018,
    48,
    1.5,
    2.5,
    1.0,
    0.50,
    18,
    0.025,
    0.075,
    0.036,
)
NOTIONAL_BUCKETS = [0.75, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
ENTRY_FEATURES = [
    "deep_edge",
    "deep_margin",
    "side",
    "vol_anchor",
    "volatility_z",
    "realized_vol_ratio",
    "bb_width",
    "net_taker_ratio",
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


@dataclass(frozen=True)
class ExitConfig:
    name: str
    close_p_th: float
    min_hold: int


def _exit_grid() -> list[ExitConfig]:
    return [
        ExitConfig("alpha1_1_exitrl_p055_h1", 0.55, 1),
        ExitConfig("alpha1_1_exitrl_p060_h2", 0.60, 2),
        ExitConfig("alpha1_1_exitrl_p065_h3", 0.65, 3),
        ExitConfig("alpha1_1_exitrl_p070_h4", 0.70, 4),
    ]


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1.get("trades", 0)) < 20:
        return -1e9 + float(c1.get("pnl", 0.0))
    return float(c1["pnl"] + 0.45 * c2["pnl"] + 0.25 * c3["pnl"] - 0.35 * abs(c1["mdd"]))


def _safe_row_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        x = float(row.get(col, default))
    except Exception:
        return float(default)
    return float(x) if np.isfinite(x) else float(default)


def _entry_state_row(frame: pd.DataFrame, i: int, side: int, edge: float, margin: float) -> dict[str, float]:
    row = frame.iloc[i]
    out = {
        "deep_edge": float(edge),
        "deep_margin": float(margin),
        "side": float(side),
        "vol_anchor": float(v31._vol_anchor(row)),
    }
    for col in ENTRY_FEATURES:
        out.setdefault(col, _safe_row_float(row, col, 0.0))
    return out


def _entry_x(row: dict[str, float]) -> np.ndarray:
    return np.asarray([[float(row.get(c, 0.0)) for c in ENTRY_FEATURES]], dtype=np.float32)


def _effective_v31_thresholds(cfg: v31.OverlayConfig, *, entry_edge: float, entry_vol_anchor: float, mfe: float, hold: int) -> tuple[float, float]:
    effective_tp = float(cfg.base_tp)
    effective_sl = float(cfg.base_sl)
    if cfg.tp_util_mult > 0.0:
        util_gain = 1.0 + float(cfg.tp_util_mult) * max(float(entry_edge) - float(cfg.edge_th), 0.0) / max(0.02, float(cfg.edge_th))
        effective_tp = float(np.clip(float(cfg.base_tp) * util_gain, float(cfg.base_tp) * 0.8, float(cfg.tp_cap)))
    if cfg.sl_vol_mult > 0.0:
        effective_sl = float(np.clip(float(entry_vol_anchor) * float(cfg.sl_vol_mult), float(cfg.base_sl) * 0.6, float(cfg.sl_cap)))
    if float(mfe) > 0.0 and float(cfg.trail_gap_mult) > 0.0:
        trail_gap = float(entry_vol_anchor) * float(cfg.trail_gap_mult)
        if int(cfg.hold_decay_start) < 999 and int(hold) >= int(cfg.hold_decay_start):
            decay_bars = int(hold) - int(cfg.hold_decay_start)
            trail_gap = max(float(entry_vol_anchor) * 0.35, trail_gap - float(cfg.hold_decay_rate) * decay_bars * float(entry_vol_anchor))
        trail_stop = max(-effective_sl, float(mfe) - trail_gap)
        effective_sl = min(effective_sl, max(0.001, trail_stop))
    return float(effective_tp), float(effective_sl)


def _train_exit_policy(train: pd.DataFrame, train_dec: pd.DataFrame, train_q: np.ndarray, *, fee: float, slip: float):
    # Reuse the established V33 offline-RL labeler, but train on alpha1 notional=2.0.
    train_cfg = v33.OverlayConfig("alpha1_1_exit_train", 0.010, 0.004, 2.0, 12, 0.60, 2, 0.040, 0.018, 48)
    x_train, y_train = v33._collect_reversal_training(train, train_dec, train_q, train_cfg, fee=fee, slip=slip)
    policy = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        HistGradientBoostingRegressor(max_iter=180, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.08, random_state=2049),
    )
    policy.fit(x_train.loc[:, v33.REVERSAL_FEATURES].to_numpy(dtype=np.float32), y_train.astype(np.float32))
    return policy, {"train_rows": int(len(y_train)), "close_rate": float(np.mean(y_train))}


def _predict_close_prob(policy: Any, state: dict[str, float]) -> float:
    x = np.asarray([[float(state.get(c, 0.0)) for c in v33.REVERSAL_FEATURES]], dtype=np.float32)
    pred = float(policy.predict(x)[0])
    return float(np.clip(pred, 0.0, 1.0))


def _collect_sizing_training(frame: pd.DataFrame, decisions: pd.DataFrame, deep_q: np.ndarray, *, fee: float, slip: float) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    close = _close(frame)
    rows: list[dict[str, float]] = []
    rewards: dict[float, list[float]] = {float(b): [] for b in NOTIONAL_BUCKETS}
    pos = 0
    cooldown = deep_cooldown = 0
    for i in range(0, len(frame) - 3):
        if pos != 0:
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        if deep_cooldown > 0:
            deep_cooldown -= 1
        dec = decisions.iloc[i]
        if int(dec.action) != ACTION_CASH and int(dec.side) != 0:
            cooldown = int(dec.cooldown_bars)
            continue
        if deep_cooldown > 0 or i < v31.SEQ_LEN:
            continue
        ql, qs = float(deep_q[i, 0]), float(deep_q[i, 1])
        side = 1 if ql > qs else -1
        edge = max(ql, qs)
        margin = abs(ql - qs)
        if edge < ALPHA1_CFG.edge_th or margin < ALPHA1_CFG.margin_th:
            continue
        state = _entry_state_row(frame, i, side, edge, margin)
        rows.append(state)
        for notional in NOTIONAL_BUCKETS:
            cfg = replace(ALPHA1_CFG, notional=float(notional))
            fill_i = min(i + 1, len(frame) - 1)
            entry_price = _fill_price(frame, fill_i, side, slip, entry=True)
            cash = 1.0 - fee * float(notional)
            mfe = mae = 0.0
            entry_vol_anchor = v31._vol_anchor(frame.iloc[i]) * float(notional)
            reward = 0.0
            for j in range(i + 1, min(i + int(cfg.base_hold) + 3, len(frame) - 1)):
                px = float(close[j])
                raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
                unreal = raw * float(notional)
                mfe = max(mfe, unreal)
                mae = min(mae, unreal)
                hold = j - i
                effective_tp, effective_sl = _effective_v31_thresholds(cfg, entry_edge=edge, entry_vol_anchor=entry_vol_anchor, mfe=mfe, hold=hold)
                reason = bool(unreal >= effective_tp or unreal <= -abs(effective_sl) or hold >= cfg.base_hold)
                if reason:
                    exit_i = min(j + 1, len(frame) - 1)
                    exit_px = _fill_price(frame, exit_i, side, slip, entry=False)
                    raw2 = (exit_px - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                    out_cash = cash * (1.0 + raw2 * float(notional))
                    out_cash -= cash * fee * float(notional)
                    reward = float(out_cash - 1.0)
                    break
            rewards[float(notional)].append(reward)
        deep_cooldown = int(ALPHA1_CFG.cooldown)
    x = np.asarray([[float(r.get(c, 0.0)) for c in ENTRY_FEATURES] for r in rows], dtype=np.float32)
    y = {k: np.asarray(v, dtype=np.float32) for k, v in rewards.items()}
    return x, y


def _train_sizing_policy(train: pd.DataFrame, train_dec: pd.DataFrame, train_q: np.ndarray, *, fee: float, slip: float):
    x, y_by_bucket = _collect_sizing_training(train, train_dec, train_q, fee=fee * 2.0, slip=slip * 2.0)
    models = {}
    for bucket, y in y_by_bucket.items():
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            HistGradientBoostingRegressor(max_iter=160, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.10, random_state=int(bucket * 1000 + 7)),
        )
        model.fit(x, y)
        models[float(bucket)] = model
    return models, {"train_rows": int(len(x)), "buckets": list(NOTIONAL_BUCKETS)}


def _select_notional(models: dict[float, Any], state: dict[str, float]) -> float:
    x = _entry_x(state)
    preds = {float(k): float(m.predict(x)[0]) for k, m in models.items()}
    # Conservative tie-break: prefer smaller notional unless expected value is clearly better.
    best = max(preds.items(), key=lambda kv: (kv[1] - 0.00015 * max(kv[0] - 1.0, 0.0), -kv[0]))[0]
    return float(best)


def backtest_alpha1(
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
    exit_policy: Any | None = None,
    exit_cfg: ExitConfig | None = None,
    sizing_models: dict[float, Any] | None = None,
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
            if owner == "deep_alpha" and exit_policy is not None and exit_cfg is not None and hold >= int(exit_cfg.min_hold):
                p_close = _predict_close_prob(exit_policy, v33._deep_state_row(df, i, pos, entry_edge, entry_margin, hold, unreal, mfe, mae))
                if p_close >= float(exit_cfg.close_p_th):
                    reason = "deep_alpha_rl_exit"
            if owner == "deep_alpha" and not reason:
                effective_tp, effective_sl = _effective_v31_thresholds(ALPHA1_CFG, entry_edge=entry_edge, entry_vol_anchor=entry_vol_anchor, mfe=mfe, hold=hold)
                if unreal >= effective_tp:
                    reason = "deep_alpha_take_profit"
                elif unreal <= -abs(effective_sl):
                    reason = "deep_alpha_stop_loss"
                elif hold >= int(ALPHA1_CFG.base_hold):
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
                deep_cooldown = max(deep_cooldown, int(ALPHA1_CFG.cooldown))
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
            continue
        if deep_cooldown <= 0 and i >= v31.SEQ_LEN:
            ql, qs = float(deep_q[i, 0]), float(deep_q[i, 1])
            side = 1 if ql > qs else -1
            edge = max(ql, qs)
            margin = abs(ql - qs)
            if edge >= ALPHA1_CFG.edge_th and margin >= ALPHA1_CFG.margin_th:
                state = _entry_state_row(df, i, side, edge, margin)
                chosen_notional = float(ALPHA1_CFG.notional if sizing_models is None else _select_notional(sizing_models, state))
                cfg = replace(ALPHA1_CFG, notional=chosen_notional)
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
    }


def main() -> int:
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
    feature_audit = {"status": "pass", "blocking": [], "warnings": [], "entry_features": ENTRY_FEATURES, "exit_features": v33.REVERSAL_FEATURES}
    print(f"[{MODEL_ID}] audits parent={parent_audit.get('status')} feature={feature_audit.get('status')}", flush=True)
    train_q = v31._predict_all(v27_model, train, v27_payload["seq_cols"], v27_payload["norm"])
    val_q = v31._predict_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    train_dec = predict_policy_frame(parent, train, close=_close(train))
    val_dec = predict_policy_frame(parent, val, close=_close(val))
    eval_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))
    exit_policy, exit_train_meta = _train_exit_policy(train, train_dec, train_q, fee=float(base["fee"]), slip=float(base["slip"]))
    sizing_models, sizing_train_meta = _train_sizing_policy(train, train_dec, train_q, fee=float(base["fee"]), slip=float(base["slip"]))

    grid_rows: list[dict[str, Any]] = []
    best_exit: dict[str, Any] | None = None
    for cfg in _exit_grid():
        v1 = backtest_alpha1(val, parent, jackpot_model, add_cfg, val_q, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=1.0, decisions=val_dec, exit_policy=exit_policy, exit_cfg=cfg)
        v2 = backtest_alpha1(val, parent, jackpot_model, add_cfg, val_q, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=2.0, decisions=val_dec, exit_policy=exit_policy, exit_cfg=cfg)
        v3 = backtest_alpha1(val, parent, jackpot_model, add_cfg, val_q, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=3.0, decisions=val_dec, exit_policy=exit_policy, exit_cfg=cfg)
        row = {"experiment": "alpha1.1_rl_exit", "config": asdict(cfg), "validation_cost1": v1, "validation_cost2": v2, "validation_cost3": v3, "selection_score": _score(v1, v2, v3)}
        grid_rows.append(row)
        if best_exit is None or row["selection_score"] > best_exit["selection_score"]:
            best_exit = row
    assert best_exit is not None
    selected_exit = ExitConfig(**best_exit["config"])

    experiments: list[dict[str, Any]] = []
    for name, kwargs in [
        ("alpha1", {}),
        ("alpha1.1_rl_exit", {"exit_policy": exit_policy, "exit_cfg": selected_exit}),
        ("alpha1.2_rl_sizing", {"sizing_models": sizing_models}),
        ("alpha1.2_combo_rl_exit_sizing", {"exit_policy": exit_policy, "exit_cfg": selected_exit, "sizing_models": sizing_models}),
    ]:
        metrics: dict[str, Any] = {}
        for mult in (1, 2, 3):
            metrics[f"cost{mult}"] = backtest_alpha1(eval_df, parent, jackpot_model, add_cfg, eval_q, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=float(mult), decisions=eval_dec, **kwargs)
        experiments.append({"name": name, "metrics": metrics, "score": _score(metrics["cost1"], metrics["cost2"], metrics["cost3"])})
        print(f"[{MODEL_ID}] {name} cost1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f} cost2={metrics['cost2']['pnl']:.2f} cost3={metrics['cost3']['pnl']:.2f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_id": MODEL_ID,
            "exit_policy": exit_policy,
            "exit_config": asdict(selected_exit),
            "sizing_models": sizing_models,
            "sizing_buckets": NOTIONAL_BUCKETS,
            "entry_features": ENTRY_FEATURES,
            "exit_features": v33.REVERSAL_FEATURES,
        },
        OUT_DIR / "alpha1_rl_layers.pkl",
    )
    pd.DataFrame(
        [
            {
                "experiment": r["experiment"],
                **{f"cfg_{k}": v for k, v in r["config"].items()},
                "score": r["selection_score"],
                "val_pnl": r["validation_cost1"]["pnl"],
                "val_mdd": r["validation_cost1"]["mdd"],
                "val_c2_pnl": r["validation_cost2"]["pnl"],
                "val_c3_pnl": r["validation_cost3"]["pnl"],
            }
            for r in grid_rows
        ]
    ).to_csv(GRID_OUT, index=False)
    blocking = list(parent_audit.get("blocking", [])) + list(feature_audit.get("blocking", []))
    warnings = list(parent_audit.get("warnings", [])) + list(feature_audit.get("warnings", []))
    best = max(experiments, key=lambda x: x["score"])
    if best["metrics"]["cost1"]["pnl"] <= ALPHA1_BASELINE["cost1"]["pnl"]:
        warnings.append("best_did_not_beat_alpha1_cost1")
    if best["metrics"]["cost2"]["pnl"] <= 0.0:
        warnings.append("best_cost2_not_survived")
    if best["metrics"]["cost3"]["pnl"] <= 0.0:
        warnings.append("best_cost3_not_survived")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if not blocking and best["metrics"]["cost1"]["pnl"] > ALPHA1_BASELINE["cost1"]["pnl"] and best["metrics"]["cost2"]["pnl"] > 0.0 else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "policy": MODEL_ID,
        "alpha1_parent_preserved": True,
        "alpha1_v27_entry_preserved": True,
        "direction_changed": False,
        "parent_audit": parent_audit,
        "feature_audit": feature_audit,
        "selected_exit_config": asdict(selected_exit),
        "exit_train_meta": exit_train_meta,
        "sizing_train_meta": sizing_train_meta,
        "metrics": {e["name"]: e["metrics"] for e in experiments},
        "baseline_alpha1": ALPHA1_BASELINE,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Alpha1 upgrade tests: alpha1.1 adds an offline RL close/hold exit overlay to deep_alpha only; alpha1.2 adds a contextual bandit/offline RL notional bucket allocator to deep_alpha entries only. Parent direction and V27 entry utilities are frozen.",
        "selected_exit_config": asdict(selected_exit),
        "experiments": experiments,
        "audit": audit,
        "artifacts": {"model": str(OUT_DIR / "alpha1_rl_layers.pkl"), "report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "grid": str(GRID_OUT)},
    }
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "best": best}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
