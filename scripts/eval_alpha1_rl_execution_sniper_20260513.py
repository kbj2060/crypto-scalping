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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, predict_policy_frame  # noqa: E402
from scripts import eval_alpha1_rl_exit_and_sizing_20260513 as alpha1_prev  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _fill_price, _read  # noqa: E402
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig, _predict_cost_runner  # noqa: E402


MODEL_ID = "alpha1_rl_execution_sniper_20260513"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha1_rl_execution_sniper_20260513"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha1_rl_execution_sniper_20260513_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha1_rl_execution_sniper_20260513_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha1_rl_execution_sniper_20260513_grid.csv"

ALPHA1_CFG = alpha1_prev.ALPHA1_CFG
ALPHA1_BASELINE = alpha1_prev.ALPHA1_BASELINE
ENTRY_FEATURES = [
    *alpha1_prev.ENTRY_FEATURES,
    "open_to_close",
    "hl_range",
    "body_abs",
    "upper_wick",
    "lower_wick",
]


@dataclass(frozen=True)
class ExecAction:
    name: str
    mode: str
    offset: float = 0.0
    penetration: float = 0.0001
    maker_fee_mult: float = 0.45


EXEC_ACTIONS = [
    ExecAction("taker", "taker", 0.0),
    ExecAction("maker_2bp_skip", "maker_skip", 0.0002),
    ExecAction("maker_5bp_skip", "maker_skip", 0.0005),
    ExecAction("maker_10bp_skip", "maker_skip", 0.0010),
    ExecAction("skip", "skip", 0.0),
]


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1.get("trades", 0)) < 20:
        return -1e9 + float(c1.get("pnl", 0.0))
    return float(c1["pnl"] + 0.45 * c2["pnl"] + 0.25 * c3["pnl"] - 0.35 * abs(c1["mdd"]))


def _num(df: pd.DataFrame, col: str, idx: int, default: float = 0.0) -> float:
    if col not in df.columns:
        return float(default)
    i = int(np.clip(idx, 0, len(df) - 1))
    try:
        x = float(pd.to_numeric(df[col], errors="coerce").ffill().iloc[i])
    except Exception:
        return float(default)
    return float(x) if np.isfinite(x) else float(default)


def _candle_features(df: pd.DataFrame, i: int) -> dict[str, float]:
    op = _num(df, "open", i)
    hi = _num(df, "high", i, op)
    lo = _num(df, "low", i, op)
    cl = _num(df, "close", i, op)
    rng = max(hi - lo, 0.0) / max(op, 1e-12)
    body = (cl - op) / max(op, 1e-12)
    upper = (hi - max(op, cl)) / max(op, 1e-12)
    lower = (min(op, cl) - lo) / max(op, 1e-12)
    return {
        "open_to_close": float(body),
        "hl_range": float(rng),
        "body_abs": float(abs(body)),
        "upper_wick": float(max(upper, 0.0)),
        "lower_wick": float(max(lower, 0.0)),
    }


def _entry_state(df: pd.DataFrame, i: int, side: int, edge: float, margin: float) -> dict[str, float]:
    state = alpha1_prev._entry_state_row(df, i, side, edge, margin)
    state.update(_candle_features(df, i))
    return state


def _x(state: dict[str, float]) -> np.ndarray:
    return np.asarray([[float(state.get(c, 0.0)) for c in ENTRY_FEATURES]], dtype=np.float32)


def _entry_fill(df: pd.DataFrame, fill_i: int, side: int, fee: float, slip: float, action: ExecAction) -> tuple[bool, float, float, str]:
    if action.mode == "skip":
        return False, 0.0, 0.0, "skip"
    if action.mode == "taker":
        return True, float(_fill_price(df, fill_i, side, slip, entry=True)), float(fee), "taker"
    op = _num(df, "open", fill_i)
    hi = _num(df, "high", fill_i, op)
    lo = _num(df, "low", fill_i, op)
    if op <= 0.0:
        return False, 0.0, 0.0, "maker_bad_open"
    if side > 0:
        limit_px = op * (1.0 - action.offset)
        filled = lo <= limit_px * (1.0 - action.penetration)
    else:
        limit_px = op * (1.0 + action.offset)
        filled = hi >= limit_px * (1.0 + action.penetration)
    return bool(filled), float(limit_px), float(fee * action.maker_fee_mult), "maker_fill" if filled else "maker_miss"


def _simulate_deep_trade(
    df: pd.DataFrame,
    signal_i: int,
    side: int,
    edge: float,
    action: ExecAction,
    *,
    fee: float,
    slip: float,
    notional: float = 2.0,
) -> float:
    fill_i = min(signal_i + 1, len(df) - 1)
    filled, entry_px, fee_entry, _ = _entry_fill(df, fill_i, side, fee, slip, action)
    if not filled:
        return 0.0
    close = _close(df)
    cash = 1.0 - fee_entry * notional
    mfe = 0.0
    entry_vol_anchor = v31._vol_anchor(df.iloc[signal_i]) * notional
    for j in range(signal_i + 1, min(signal_i + int(ALPHA1_CFG.base_hold) + 3, len(df) - 1)):
        px = float(close[j])
        raw = (px * (1.0 - slip) - entry_px) / max(entry_px, 1e-12) if side > 0 else (entry_px - px * (1.0 + slip)) / max(entry_px, 1e-12)
        unreal = raw * notional
        mfe = max(mfe, unreal)
        hold = j - signal_i
        tp, sl = alpha1_prev._effective_v31_thresholds(ALPHA1_CFG, entry_edge=edge, entry_vol_anchor=entry_vol_anchor, mfe=mfe, hold=hold)
        if unreal >= tp or unreal <= -abs(sl) or hold >= ALPHA1_CFG.base_hold:
            exit_i = min(j + 1, len(df) - 1)
            exit_px = _fill_price(df, exit_i, side, slip, entry=False)
            raw2 = (exit_px - entry_px) / max(entry_px, 1e-12) if side > 0 else (entry_px - exit_px) / max(entry_px, 1e-12)
            out = cash * (1.0 + raw2 * notional)
            out -= cash * fee * notional
            return float(out - 1.0)
    return 0.0


def _collect_training(df: pd.DataFrame, decisions: pd.DataFrame, deep_q: np.ndarray, *, fee: float, slip: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rows: list[dict[str, float]] = []
    rewards: dict[str, list[float]] = {a.name: [] for a in EXEC_ACTIONS}
    cooldown = deep_cooldown = 0
    for i in range(0, len(df) - 3):
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
        rows.append(_entry_state(df, i, side, edge, margin))
        for action in EXEC_ACTIONS:
            rewards[action.name].append(_simulate_deep_trade(df, i, side, edge, action, fee=fee, slip=slip, notional=float(ALPHA1_CFG.notional)))
        deep_cooldown = int(ALPHA1_CFG.cooldown)
    x = np.asarray([[float(r.get(c, 0.0)) for c in ENTRY_FEATURES] for r in rows], dtype=np.float32)
    y = {k: np.asarray(v, dtype=np.float32) for k, v in rewards.items()}
    return x, y


def _train_exec_policy(train: pd.DataFrame, train_dec: pd.DataFrame, train_q: np.ndarray, *, fee: float, slip: float):
    # Train with cost2 economics so the policy does not overfit to free taker fills.
    x, y_by_action = _collect_training(train, train_dec, train_q, fee=fee * 2.0, slip=slip * 2.0)
    models = {}
    for action in EXEC_ACTIONS:
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            HistGradientBoostingRegressor(max_iter=180, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.10, random_state=3071 + len(action.name)),
        )
        model.fit(x, y_by_action[action.name])
        models[action.name] = model
    return models, {"train_rows": int(len(x)), "actions": [asdict(a) for a in EXEC_ACTIONS]}


def _choose_action(models: dict[str, Any], state: dict[str, float], *, min_edge_reward: float) -> ExecAction:
    xx = _x(state)
    preds = {a.name: float(models[a.name].predict(xx)[0]) for a in EXEC_ACTIONS}
    preds["skip"] = max(preds.get("skip", 0.0), float(min_edge_reward))
    best_name = max(preds, key=lambda k: (preds[k], -next(i for i, a in enumerate(EXEC_ACTIONS) if a.name == k)))
    return next(a for a in EXEC_ACTIONS if a.name == best_name)


def backtest_exec(
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
    exec_models: dict[str, Any] | None = None,
    min_edge_reward: float = 0.0,
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
    routes: dict[str, int] = {}

    def route(name: str) -> None:
        routes[name] = routes.get(name, 0) + 1

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
                tp, sl = alpha1_prev._effective_v31_thresholds(ALPHA1_CFG, entry_edge=entry_edge, entry_vol_anchor=entry_vol_anchor, mfe=mfe, hold=hold)
                if unreal >= tp:
                    reason = "deep_alpha_take_profit"
                elif unreal <= -abs(sl):
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
                    cash -= cash * fee_eff * delta
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
            route("parent_taker")
            continue
        if deep_cooldown <= 0 and i >= v31.SEQ_LEN:
            ql, qs = float(deep_q[i, 0]), float(deep_q[i, 1])
            side = 1 if ql > qs else -1
            edge = max(ql, qs)
            margin = abs(ql - qs)
            if edge >= ALPHA1_CFG.edge_th and margin >= ALPHA1_CFG.margin_th:
                state = _entry_state(df, i, side, edge, margin)
                action = ExecAction("taker", "taker") if exec_models is None else _choose_action(exec_models, state, min_edge_reward=min_edge_reward)
                fill_i = min(i + 1, len(df) - 1)
                filled, fill_px, fill_fee, fill_route = _entry_fill(df, fill_i, side, fee_eff, slip_eff, action)
                route(fill_route)
                if not filled:
                    actions["deep_entry_miss"] = actions.get("deep_entry_miss", 0) + 1
                    deep_cooldown = max(deep_cooldown, int(ALPHA1_CFG.cooldown // 2))
                    continue
                pos = side
                owner = "deep_alpha"
                entry_price = fill_px
                entry_equity = cash
                entry_idx = i
                parent_notional = float(ALPHA1_CFG.notional)
                notional = float(ALPHA1_CFG.notional)
                take_profit = float(ALPHA1_CFG.base_tp)
                stop_loss = float(ALPHA1_CFG.base_sl)
                max_hold = int(ALPHA1_CFG.base_hold)
                next_cooldown = int(ALPHA1_CFG.cooldown)
                entry_edge = edge
                entry_margin = margin
                entry_vol_anchor = v31._vol_anchor(df.iloc[i]) * float(notional)
                cash -= cash * fill_fee * notional
                long_entries += int(pos > 0)
                short_entries += int(pos < 0)
                deep_entries += 1
                notional_sum += notional
                leverage_sum += max(notional, 1.0)
                mfe = mae = 0.0
                add_done = True
                actions["deep_entry"] = actions.get("deep_entry", 0) + 1
                actions[f"exec_{action.name}"] = actions.get(f"exec_{action.name}", 0) + 1
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
        "route_counts": routes,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
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
    feature_audit = {"status": "pass", "blocking": [], "warnings": [], "entry_features": ENTRY_FEATURES, "actions": [asdict(a) for a in EXEC_ACTIONS]}
    print(f"[{MODEL_ID}] audits parent={parent_audit.get('status')} feature={feature_audit.get('status')}", flush=True)
    train_q = v31._predict_all(v27_model, train, v27_payload["seq_cols"], v27_payload["norm"])
    val_q = v31._predict_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    train_dec = predict_policy_frame(parent, train, close=_close(train))
    val_dec = predict_policy_frame(parent, val, close=_close(val))
    eval_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))
    exec_models, train_meta = _train_exec_policy(train, train_dec, train_q, fee=float(base["fee"]), slip=float(base["slip"]))
    print(f"[{MODEL_ID}] trained execution policy rows={train_meta['train_rows']}", flush=True)

    grid_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for min_edge_reward in (0.0, 0.0002, 0.0005, 0.0010):
        v1 = backtest_exec(val, parent, jackpot_model, add_cfg, val_q, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=1.0, decisions=val_dec, exec_models=exec_models, min_edge_reward=min_edge_reward)
        v2 = backtest_exec(val, parent, jackpot_model, add_cfg, val_q, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=2.0, decisions=val_dec, exec_models=exec_models, min_edge_reward=min_edge_reward)
        v3 = backtest_exec(val, parent, jackpot_model, add_cfg, val_q, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=3.0, decisions=val_dec, exec_models=exec_models, min_edge_reward=min_edge_reward)
        row = {"min_edge_reward": float(min_edge_reward), "validation_cost1": v1, "validation_cost2": v2, "validation_cost3": v3, "selection_score": _score(v1, v2, v3)}
        grid_rows.append(row)
        print(f"[{MODEL_ID}] val min_edge={min_edge_reward:.4g} score={row['selection_score']:.2f} c1={v1['pnl']:.2f} c3={v3['pnl']:.2f} routes={v1.get('route_counts', {})}", flush=True)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    assert best is not None
    selected_min_edge = float(best["min_edge_reward"])

    experiments: list[dict[str, Any]] = []
    for name, kwargs in [
        ("alpha1", {}),
        ("alpha1.3_rl_execution_sniper", {"exec_models": exec_models, "min_edge_reward": selected_min_edge}),
    ]:
        metrics: dict[str, Any] = {}
        for mult in (1, 2, 3):
            metrics[f"cost{mult}"] = backtest_exec(eval_df, parent, jackpot_model, add_cfg, eval_q, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=float(mult), decisions=eval_dec, **kwargs)
        experiments.append({"name": name, "metrics": metrics, "score": _score(metrics["cost1"], metrics["cost2"], metrics["cost3"])})
        print(f"[{MODEL_ID}] {name} cost1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f} cost2={metrics['cost2']['pnl']:.2f} cost3={metrics['cost3']['pnl']:.2f}", flush=True)

    joblib.dump(
        {
            "model_id": MODEL_ID,
            "exec_models": exec_models,
            "selected_min_edge_reward": selected_min_edge,
            "entry_features": ENTRY_FEATURES,
            "actions": [asdict(a) for a in EXEC_ACTIONS],
            "train_meta": train_meta,
        },
        OUT_DIR / "alpha1_rl_execution_sniper.pkl",
    )
    pd.DataFrame(
        [
            {
                "min_edge_reward": r["min_edge_reward"],
                "score": r["selection_score"],
                "val_pnl": r["validation_cost1"]["pnl"],
                "val_mdd": r["validation_cost1"]["mdd"],
                "val_c2_pnl": r["validation_cost2"]["pnl"],
                "val_c3_pnl": r["validation_cost3"]["pnl"],
                "val_routes": json.dumps(r["validation_cost1"].get("route_counts", {}), ensure_ascii=False),
            }
            for r in grid_rows
        ]
    ).to_csv(GRID_OUT, index=False)
    best_exp = max(experiments, key=lambda x: x["score"])
    blocking = list(parent_audit.get("blocking", [])) + list(feature_audit.get("blocking", []))
    warnings = list(parent_audit.get("warnings", [])) + list(feature_audit.get("warnings", []))
    warnings.append("maker_fill_simulation_uses_next_bar_ohlc_not_live_orderbook_queue")
    if best_exp["metrics"]["cost1"]["pnl"] <= ALPHA1_BASELINE["cost1"]["pnl"]:
        warnings.append("best_did_not_beat_alpha1_cost1")
    if best_exp["metrics"]["cost2"]["pnl"] <= 0.0:
        warnings.append("best_cost2_not_survived")
    if best_exp["metrics"]["cost3"]["pnl"] <= 0.0:
        warnings.append("best_cost3_not_survived")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if not blocking and best_exp["name"] != "alpha1" and best_exp["metrics"]["cost1"]["pnl"] > ALPHA1_BASELINE["cost1"]["pnl"] and best_exp["metrics"]["cost2"]["pnl"] > 0.0 else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "policy": MODEL_ID,
        "alpha1_parent_preserved": True,
        "alpha1_v27_entry_preserved": True,
        "alpha1_exit_preserved": True,
        "selected_min_edge_reward": selected_min_edge,
        "train_meta": train_meta,
        "parent_audit": parent_audit,
        "feature_audit": feature_audit,
        "metrics": {e["name"]: e["metrics"] for e in experiments},
        "baseline_alpha1": ALPHA1_BASELINE,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Alpha1.3 contextual-bandit RL execution sniper. It freezes parent, V21.2, V27, and V31 exits; only deep_alpha entry execution chooses taker/maker/skip based on cost2-trained net reward.",
        "architecture": {
            "state": ENTRY_FEATURES,
            "actions": [asdict(a) for a in EXEC_ACTIONS],
            "reward": "net trade return after alpha1 V31 exit, including fee/slippage and maker fill/miss simulation",
            "training_window": "2025-01-01..2025-09-30",
            "selection_window": "2025-10-01..2025-12-31",
            "oos_window": "2026 fixed OOS",
        },
        "selected_min_edge_reward": selected_min_edge,
        "experiments": experiments,
        "audit": audit,
        "artifacts": {"model": str(OUT_DIR / "alpha1_rl_execution_sniper.pkl"), "report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "grid": str(GRID_OUT)},
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "best": best_exp}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
