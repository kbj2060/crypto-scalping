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
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, ACTION_LONG, predict_policy_frame  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _days, _fill_price, _read  # noqa: E402
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig, _predict_cost_runner  # noqa: E402


MODEL_ID = "hf_v13_jackpot_learned_exit_v21_4_20260511"
DEFAULT_PARENT = ROOT / "data/ensemble/supervised/hf_v13_clean_regime_margin110_20260511/v13_clean_regime_margin110.pkl"
DEFAULT_JACKPOT = ROOT / "data/ensemble/supervised/hf_v13_jackpot_runner_v21_2_20260511/v21_2_jackpot_runner.pkl"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_jackpot_learned_exit_v21_4_20260511"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_jackpot_learned_exit_v21_4_20260511_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_jackpot_learned_exit_v21_4_20260511_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_jackpot_learned_exit_v21_4_20260511_grid.csv"
V21_2_COST1 = 199.5442148936891
V21_2_COST2 = 113.24305052028865
V21_2_COST3 = 24.714228358176072


@dataclass(frozen=True)
class ExitConfig:
    name: str
    exit_prob: float
    adv_floor: float
    min_exit_age: int
    eval_stride: int
    safety_max_hold: int
    max_entry_notional: float


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


def _grid() -> list[ExitConfig]:
    return [
        ExitConfig("v21_4_balanced_stride6", 0.60, -0.001, 6, 6, 576, 2.75),
        ExitConfig("v21_4_fast_profit_lock_stride6", 0.55, -0.004, 3, 6, 576, 2.75),
        ExitConfig("v21_4_slow_runner_stride12", 0.70, 0.000, 12, 12, 864, 2.75),
    ]


def _addon_cfg(jackpot_payload: dict[str, Any]) -> CostRunnerConfig:
    return CostRunnerConfig(**dict(jackpot_payload["selected_config"]))


def _positive_prob(clf: Any, xx: pd.DataFrame) -> float:
    try:
        classes = np.asarray(clf[-1].classes_, dtype=int)
        prob = clf.predict_proba(xx)[0]
        idx = int(np.flatnonzero(classes == 1)[0]) if np.any(classes == 1) else 0
        return float(prob[idx])
    except Exception:
        return 0.0


def _exit_feature_frame(
    frame: pd.DataFrame,
    bundle: dict[str, Any],
    decisions: pd.DataFrame,
    idx: int,
    state: dict[str, float | int],
    exit_cols: list[str] | None = None,
) -> pd.DataFrame:
    x = _feature_frame(frame, bundle, decisions, idx, state)
    extra = {
        "exit_side": float(state["side"]),
        "exit_parent_notional": float(state["parent_notional"]),
        "exit_notional": float(state["notional"]),
        "exit_notional_mult": float(state["notional"]) / max(float(state["parent_notional"]), 1e-12),
        "exit_bars_since_entry": float(state["bars_since_entry"]),
        "exit_unrealized": float(state["unrealized"]),
        "exit_mfe": float(state["mfe"]),
        "exit_mae": float(state["mae"]),
        "exit_giveback": float(state["mfe"]) - float(state["unrealized"]),
        "exit_recovery": float(state["unrealized"]) - float(state["mae"]),
        "exit_drawdown_abs": float(state["drawdown_abs"]),
        "exit_parent_take_profit": float(state["take_profit"]),
        "exit_parent_stop_loss": float(state["stop_loss"]),
        "exit_parent_max_hold": float(state["max_hold"]),
        "exit_hold_ratio": float(state["bars_since_entry"]) / max(float(state["max_hold"]), 1.0),
    }
    for k, v in extra.items():
        x[k] = v
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if exit_cols is not None:
        x = x.reindex(columns=exit_cols, fill_value=0.0)
    return x


def _exit_utility(
    frame: pd.DataFrame,
    *,
    exit_i: int,
    pos: int,
    entry_price: float,
    notional: float,
    fee_eff: float,
    slip_eff: float,
) -> float:
    exit_px = _fill_price(frame, int(np.clip(exit_i, 0, len(frame) - 1)), pos, slip_eff, entry=False)
    raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
    return float(raw * notional - fee_eff * notional)


def _label_exit_snapshot(
    frame: pd.DataFrame,
    close: np.ndarray,
    *,
    snapshot_idx: int,
    entry_idx: int,
    pos: int,
    entry_price: float,
    notional: float,
    fee_eff: float,
    slip_eff: float,
    horizon: int,
) -> tuple[int, float]:
    now_i = min(snapshot_idx + 1, len(frame) - 1)
    now_u = _exit_utility(frame, exit_i=now_i, pos=pos, entry_price=entry_price, notional=notional, fee_eff=fee_eff, slip_eff=slip_eff)
    end_i = min(len(frame) - 1, max(now_i, entry_idx + horizon))
    px = close[now_i : end_i + 1].astype(np.float64, copy=False)
    if pos > 0:
        raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12)
    else:
        raw = (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
    utils_arr = raw * notional - fee_eff * notional
    best_future = float(np.max(utils_arr)) if len(utils_arr) else now_u
    worst_future = float(np.min(utils_arr)) if len(utils_arr) else now_u
    advantage = now_u - best_future
    exit_label = int(advantage >= -0.0025 or (now_u > 0.0 and worst_future <= now_u - 0.018))
    return exit_label, advantage


def _fit_exit_model(
    frame: pd.DataFrame,
    bundle: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    *,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    decisions = predict_policy_frame(bundle, frame, close=_close(frame))
    close = _close(frame)
    rows: list[pd.DataFrame] = []
    labels: list[int] = []
    advs: list[float] = []
    pos = 0
    entry_price = 0.0
    entry_idx = 0
    parent_notional = notional = 0.0
    take_profit = stop_loss = 0.0
    max_hold = 0
    cash = peak = 1.0
    mfe = mae = 0.0
    add_done = False
    max_snapshots = 1000
    for i in range(0, len(frame) - 2):
        if pos != 0:
            px = float(close[i])
            raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
            unreal = raw * notional
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            hold = i - entry_idx
            state = {
                "side": pos,
                "parent_notional": parent_notional,
                "notional": notional,
                "bars_since_entry": hold,
                "unrealized": unreal,
                "mfe": mfe,
                "mae": mae,
                "drawdown_abs": max(0.0, 1.0 - eq / max(peak, 1e-12)),
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "max_hold": max_hold,
            }
            if hold >= 2 and hold % 6 == 0:
                label, adv = _label_exit_snapshot(
                    frame,
                    close,
                    snapshot_idx=i,
                    entry_idx=entry_idx,
                    pos=pos,
                    entry_price=entry_price,
                    notional=notional,
                    fee_eff=fee,
                    slip_eff=slip,
                    horizon=max(96, min(864, max_hold if max_hold > 0 else 288)),
                )
                rows.append(_exit_feature_frame(frame, bundle, decisions, i, state))
                labels.append(label)
                advs.append(adv)
                if len(labels) >= max_snapshots:
                    break
            if not add_done and unreal >= add_cfg.min_unrealized and hold >= add_cfg.min_bars_since_entry and state["drawdown_abs"] <= add_cfg.dd_block:
                x_add = _feature_frame(frame, bundle, decisions, i, state)
                _, _, _, q90, p_jackpot, p_bad, p_cost3 = _predict_cost_runner(jackpot_model, x_add)
                is_add = p_jackpot >= add_cfg.jackpot_p and q90 >= add_cfg.jackpot_q90 and p_bad <= add_cfg.bad_cap and p_cost3 >= 0.40
                delta = max(0.0, min(parent_notional * add_cfg.full_add_frac, parent_notional * add_cfg.max_total_mult - notional)) if is_add else 0.0
                if delta > 1e-12:
                    add_i = min(i + 1, len(frame) - 1)
                    add_px = _fill_price(frame, add_i, pos, slip, entry=True)
                    new_notional = notional + delta
                    entry_price = (entry_price * notional + add_px * delta) / max(new_notional, 1e-12)
                    cash -= cash * fee * delta
                    notional = new_notional
                add_done = True
            if hold >= max(96, min(864, max_hold if max_hold > 0 else 288)):
                exit_i = min(i + 1, len(frame) - 1)
                exit_px = _fill_price(frame, exit_i, pos, slip, entry=False)
                raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * fee * notional
                pos = 0
                add_done = False
                continue
        if pos == 0:
            dec = decisions.iloc[i]
            if int(dec.action) == ACTION_CASH or int(dec.side) == 0:
                continue
            fill_i = min(i + 1, len(frame) - 1)
            pos = int(dec.side)
            entry_price = _fill_price(frame, fill_i, pos, slip, entry=True)
            entry_idx = i
            parent_notional = min(float(dec.notional_exposure), add_cfg.max_entry_notional)
            notional = parent_notional
            take_profit = float(dec.take_profit)
            stop_loss = float(dec.stop_loss)
            max_hold = int(dec.max_hold_bars)
            cash -= cash * fee * notional
            mfe = mae = 0.0
            add_done = False
    if not rows:
        raise RuntimeError("no learned-exit snapshots")
    x = pd.concat(rows, ignore_index=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = np.asarray(labels, dtype=np.int64)
    adv = np.asarray(advs, dtype=np.float64)
    if len(np.unique(y)) < 2:
        clf = make_pipeline(SimpleImputer(strategy="median"), DummyClassifier(strategy="constant", constant=int(y[0])))
    else:
        clf = make_pipeline(SimpleImputer(strategy="median"), DecisionTreeClassifier(max_depth=7, min_samples_leaf=35, random_state=2140))
    reg = make_pipeline(SimpleImputer(strategy="median"), DecisionTreeRegressor(max_depth=7, min_samples_leaf=35, random_state=2141))
    clf.fit(x, y)
    reg.fit(x, adv)
    return {
        "classifier": clf,
        "advantage_regressor": reg,
        "feature_cols": list(x.columns),
        "snapshot_count": int(len(x)),
        "exit_label_rate": float(y.mean()),
        "advantage_mean": float(adv.mean()),
        "advantage_p10": float(np.quantile(adv, 0.10)),
        "advantage_p90": float(np.quantile(adv, 0.90)),
    }


def _predict_exit(model: dict[str, Any], x: pd.DataFrame) -> tuple[float, float]:
    xx = x.reindex(columns=list(model["feature_cols"]), fill_value=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    p = _positive_prob(model["classifier"], xx)
    adv = float(model["advantage_regressor"].predict(xx)[0])
    return p, adv


def backtest(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    jackpot_model: dict[str, Any],
    exit_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    exit_cfg: ExitConfig,
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
    entry_price = entry_equity = 0.0
    entry_idx = 0
    parent_notional = notional = 0.0
    leverage = 1.0
    take_profit = stop_loss = 0.0
    max_hold = 0
    cooldown = next_cooldown = 0
    add_done = False
    mfe = mae = 0.0
    trades = wins = long_entries = short_entries = 0
    notional_sum = leverage_sum = 0.0
    exits: dict[str, int] = {}
    runner_actions: dict[str, int] = {}
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
            exit_prob = exit_adv = 0.0
            state = {
                "side": pos,
                "parent_notional": parent_notional,
                "notional": notional,
                "bars_since_entry": hold,
                "unrealized": unreal,
                "mfe": mfe,
                "mae": mae,
                "drawdown_abs": dd_abs,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "max_hold": max_hold,
            }
            if hold >= exit_cfg.min_exit_age and hold % max(exit_cfg.eval_stride, 1) == 0:
                x_exit = _exit_feature_frame(df, bundle, decisions, i, state, list(exit_model["feature_cols"]))
                exit_prob, exit_adv = _predict_exit(exit_model, x_exit)
                if exit_prob >= exit_cfg.exit_prob and exit_adv >= exit_cfg.adv_floor:
                    reason = "learned_exit"
            if not reason and hold >= exit_cfg.safety_max_hold:
                reason = "safety_max_hold"
            if not reason and not add_done and add_cfg.full_add_frac > 0.0 and unreal >= add_cfg.min_unrealized and hold >= add_cfg.min_bars_since_entry and dd_abs <= add_cfg.dd_block:
                x_add = _feature_frame(df, bundle, decisions, i, state)
                _, _, _, q90, p_jackpot, p_bad, p_cost3 = _predict_cost_runner(jackpot_model, x_add)
                is_add = p_jackpot >= add_cfg.jackpot_p and q90 >= add_cfg.jackpot_q90 and p_bad <= add_cfg.bad_cap and p_cost3 >= 0.40
                delta = max(0.0, min(parent_notional * add_cfg.full_add_frac, parent_notional * add_cfg.max_total_mult - notional)) if is_add else 0.0
                if delta > 1e-12:
                    fill_i = min(i + 1, len(df) - 1)
                    add_px = _fill_price(df, fill_i, pos, slip_eff, entry=True)
                    new_notional = notional + delta
                    entry_price = (entry_price * notional + add_px * delta) / max(new_notional, 1e-12)
                    before = cash
                    cash -= before * fee_eff * delta
                    notional = new_notional
                    runner_actions["add_on"] = runner_actions.get("add_on", 0) + 1
                    if record and open_record is not None:
                        open_record.update({"add_on_timestamp": str(df["timestamp"].iloc[fill_i]), "add_on_delta_notional": float(delta), "add_on_price": float(add_px), "add_fee_pct": float(fee_eff * delta * 100.0)})
                else:
                    runner_actions["reject"] = runner_actions.get("reject", 0) + 1
                add_done = True
            if reason:
                fill_i = min(i + 1, len(df) - 1)
                exit_price = _fill_price(df, fill_i, pos, slip_eff, entry=False)
                raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * fee_eff * notional
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] = exits.get(reason, 0) + 1
                if record and open_record is not None:
                    out = dict(open_record)
                    out.update({"exit_signal_timestamp": str(df["timestamp"].iloc[i]), "exit_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "exit_reason": reason, "exit_prob": float(exit_prob), "exit_advantage": float(exit_adv), "realized_net_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0), "final_notional_exposure": float(notional), "mfe_pct": float(mfe * 100.0), "mae_pct": float(mae * 100.0), "fee_exit_pct": float(fee_eff * notional * 100.0), "cash_after": float(cash)})
                    records.append(out)
                pos = 0
                cooldown = int(next_cooldown)
                next_cooldown = 0
                add_done = False
                open_record = None
                continue
        if pos != 0:
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        dec = decisions.iloc[i]
        if int(dec.action) == ACTION_CASH or int(dec.side) == 0:
            continue
        fill_i = min(i + 1, len(df) - 1)
        pos = int(dec.side)
        entry_price = _fill_price(df, fill_i, pos, slip_eff, entry=True)
        entry_equity = cash
        entry_idx = i
        parent_notional = min(float(dec.notional_exposure), exit_cfg.max_entry_notional)
        notional = parent_notional
        leverage = float(dec.leverage)
        take_profit = float(dec.take_profit)
        stop_loss = float(dec.stop_loss)
        max_hold = int(dec.max_hold_bars)
        next_cooldown = int(dec.cooldown_bars)
        cash -= cash * fee_eff * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        leverage_sum += leverage
        mfe = mae = 0.0
        add_done = False
        if record:
            open_record = {"entry_signal_timestamp": str(df["timestamp"].iloc[i]), "entry_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "side": "LONG" if pos > 0 else "SHORT", "entry_price": float(entry_price), "parent_notional_exposure": float(dec.notional_exposure), "notional_exposure": float(notional), "leverage": float(leverage), "position_fraction": float(notional / max(leverage, 1e-12)), "parent_take_profit": float(take_profit), "parent_stop_loss": float(stop_loss), "parent_max_hold_bars": int(max_hold), "fee_entry_pct": float(fee_eff * notional * 100.0)}
    if pos != 0:
        fill_i = len(df) - 1
        exit_price = _fill_price(df, fill_i, pos, slip_eff, entry=False)
        raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * fee_eff * notional
        trades += 1
        wins += int(cash > entry_equity)
        exits["forced_end"] = exits.get("forced_end", 0) + 1
    n = max(long_entries + short_entries, 1)
    out = {"pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(trades), "wr": float(wins / max(trades, 1)), "trades_per_day": float(trades / _days(df)), "long_entries": int(long_entries), "short_entries": int(short_entries), "avg_notional": float(notional_sum / n), "avg_leverage": float(leverage_sum / n), "exits": exits, "runner_actions": runner_actions}
    if record:
        out["trade_records"] = records
    return out


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1["trades"]) < 20:
        return -1e9 + float(c1["pnl"])
    return float(c1["pnl"] + 0.40 * float(c2["pnl"]) + 0.15 * float(c3["pnl"]) - 0.20 * abs(float(c1["mdd"])))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V21.4 learned exit governor with TP/SL disabled as direct exits.")
    p.add_argument("--parent-model", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--jackpot-model", type=Path, default=DEFAULT_JACKPOT)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bundle = joblib.load(args.parent_model)
    jackpot_payload = joblib.load(args.jackpot_model)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = _addon_cfg(jackpot_payload)
    base = dict(bundle["config"])
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    train = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    feature_audit = _audit_contract(train_all, eval_df, list(bundle.get("feature_cols") or []))
    exit_model = _fit_exit_model(train, bundle, jackpot_model, add_cfg, fee=float(base["fee"]), slip=float(base["slip"]))
    val_dec = predict_policy_frame(bundle, val, close=_close(val))
    eval_dec = predict_policy_frame(bundle, eval_df, close=_close(eval_df))
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for cfg in _grid():
        v1 = backtest(val, bundle, jackpot_model, exit_model, add_cfg, cfg, fee=float(base["fee"]), slip=float(base["slip"]), decisions=val_dec, cost_mult=1.0)
        v2 = backtest(val, bundle, jackpot_model, exit_model, add_cfg, cfg, fee=float(base["fee"]), slip=float(base["slip"]), decisions=val_dec, cost_mult=2.0)
        v3 = backtest(val, bundle, jackpot_model, exit_model, add_cfg, cfg, fee=float(base["fee"]), slip=float(base["slip"]), decisions=val_dec, cost_mult=3.0)
        row = {"config": asdict(cfg), "validation_cost1": v1, "validation_cost2": v2, "validation_cost3": v3, "selection_score": _score(v1, v2, v3)}
        rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    selected = ExitConfig(**best["config"])
    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for mult in (1, 2, 3):
        r = backtest(eval_df, bundle, jackpot_model, exit_model, add_cfg, selected, fee=float(base["fee"]), slip=float(base["slip"]), decisions=eval_dec, cost_mult=float(mult), record=(mult == 1))
        if mult == 1:
            ledger = pd.DataFrame(r.pop("trade_records", []))
            lp = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
            lp.parent.mkdir(parents=True, exist_ok=True)
            ledger.to_csv(lp, index=False)
            ledgers["cost1"] = str(lp)
        metrics[f"cost{mult}"] = r
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "v21_4_jackpot_learned_exit.pkl"
    joblib.dump({"model_id": MODEL_ID, "parent_model": str(args.parent_model), "jackpot_model": str(args.jackpot_model), "exit_model": exit_model, "add_config": asdict(add_cfg), "selected_config": asdict(selected)}, model_path)
    args.grid_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**{f"cfg_{k}": v for k, v in r["config"].items()}, "score": r["selection_score"], "val_pnl": r["validation_cost1"]["pnl"], "val_mdd": r["validation_cost1"]["mdd"], "val_trades": r["validation_cost1"]["trades"], "val_c2_pnl": r["validation_cost2"]["pnl"], "val_c3_pnl": r["validation_cost3"]["pnl"]} for r in rows]).to_csv(args.grid_out, index=False)
    blocking: list[str] = []
    warnings: list[str] = []
    if feature_audit["status"] != "pass":
        blocking.extend(feature_audit["blocking"])
    warnings.extend(feature_audit.get("warnings", []))
    if metrics["cost1"]["pnl"] <= V21_2_COST1:
        warnings.append("oos_cost1_did_not_beat_v21_2")
    if metrics["cost2"]["pnl"] <= 0.0:
        warnings.append("cost2_not_survived")
    if metrics["cost3"]["pnl"] <= 0.0:
        warnings.append("cost3_not_survived")
    verdict = "promote" if not blocking and metrics["cost1"]["pnl"] > V21_2_COST1 and metrics["cost2"]["pnl"] > 0.0 and metrics["cost3"]["pnl"] > 0.0 else "iterate"
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": verdict,
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "tp_sl_direct_exit_disabled": True,
        "safety_exit_remaining": "safety_max_hold only",
        "feature_audit": feature_audit,
        "exit_model_meta": {k: v for k, v in exit_model.items() if k not in {"classifier", "advantage_regressor", "feature_cols"}},
        "selected_config": asdict(selected),
        "metrics": metrics,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "V21.2 parent entry + jackpot add-on retained; parent TP/SL are removed as direct close triggers. A learned exit classifier/regressor decides close timing from in-position state, with only a wide safety max-hold fallback.",
        "parent_model": str(args.parent_model),
        "jackpot_model": str(args.jackpot_model),
        "model": str(model_path),
        "split_policy": "Exit model trained on 2025 Jan-Sep; exit threshold selected on 2025 Oct-Dec; 2026 fixed OOS is used only after selection.",
        "add_config": asdict(add_cfg),
        "selected_config": asdict(selected),
        "selection_result": best,
        "metrics": metrics,
        "audit": audit,
        "artifacts": {"model": str(model_path), "report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "ledgers": ledgers},
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "model": str(model_path), "selected": asdict(selected), "metrics": metrics, "verdict": verdict}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
