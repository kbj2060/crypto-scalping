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
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, ACTION_LONG, predict_policy_frame  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _days, _fill_price, _read  # noqa: E402


MODEL_ID = "hf_v13_convex_runner_pyramid_v18_20260511"
DEFAULT_MODEL = ROOT / "data/ensemble/supervised/hf_v13_clean_regime_margin110_20260511/v13_clean_regime_margin110.pkl"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_convex_runner_pyramid_v18_20260511"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_convex_runner_pyramid_v18_20260511_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_convex_runner_pyramid_v18_20260511_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_convex_runner_pyramid_v18_20260511_grid.csv"


@dataclass(frozen=True)
class PyramidConfig:
    name: str
    pred_utility_threshold: float
    min_unrealized: float
    min_bars_since_entry: int
    add_frac: float
    max_total_mult: float
    dd_block: float
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


def _grid() -> list[PyramidConfig]:
    out: list[PyramidConfig] = []
    idx = 0
    for th in (0.0000, 0.0015, 0.0030):
        for add_frac in (0.25, 0.50):
            for max_mult in (1.35, 1.60, 1.85):
                out.append(PyramidConfig(f"v18_pyr_{idx}", th, 0.004, 3, add_frac, max_mult, 0.30, 2.75))
                idx += 1
    out.append(PyramidConfig("v18_parent_noop", 99.0, 99.0, 999, 0.0, 1.0, 99.0, 99.0))
    return out


def _feature_frame(frame: pd.DataFrame, bundle: dict[str, Any], decisions: pd.DataFrame, idx: int, state: dict[str, float]) -> pd.DataFrame:
    cols = list(bundle.get("feature_cols") or [])
    row = frame.iloc[[idx]].reindex(columns=cols).replace([np.inf, -np.inf], np.nan).copy()
    if row.isna().all(axis=None):
        row = frame.iloc[[idx]].select_dtypes(include=[np.number]).copy()
    dec = decisions.iloc[idx]
    extra = {
        "parent_side": float(dec.side),
        "parent_notional": float(state["parent_notional"]),
        "current_notional": float(state["notional"]),
        "bars_since_entry": float(state["bars_since_entry"]),
        "unrealized_pct": float(state["unrealized"]),
        "mfe_so_far": float(state["mfe"]),
        "mae_so_far": float(state["mae"]),
        "drawdown_abs": float(state["drawdown_abs"]),
        "parent_take_profit": float(state["take_profit"]),
        "parent_stop_loss": float(state["stop_loss"]),
        "parent_max_hold_bars": float(state["max_hold"]),
        "parent_confidence": float(getattr(dec, "confidence", 0.0)),
        "parent_quality_score": float(getattr(dec, "quality_score", 0.0)),
    }
    for k, v in extra.items():
        row[k] = v
    return row.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _fit_runner_model(train_df: pd.DataFrame, bundle: dict[str, Any], *, fee: float, slip: float) -> dict[str, Any]:
    decisions = predict_policy_frame(bundle, train_df, close=_close(train_df))
    close = _close(train_df)
    rows: list[pd.DataFrame] = []
    y: list[float] = []
    pos = 0
    entry_price = entry_equity = 0.0
    entry_idx = 0
    parent_notional = notional = 0.0
    take_profit = stop_loss = 0.0
    max_hold = 0
    mfe = mae = 0.0
    cash = peak = 1.0
    for i in range(0, len(train_df) - 2):
        if pos != 0:
            px = float(close[i])
            raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
            unreal = raw * notional
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            state = {
                "parent_notional": parent_notional,
                "notional": notional,
                "bars_since_entry": i - entry_idx,
                "unrealized": unreal,
                "mfe": mfe,
                "mae": mae,
                "drawdown_abs": max(0.0, 1.0 - eq / max(peak, 1e-12)),
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "max_hold": max_hold,
            }
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "tp"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "sl"
            elif max_hold > 0 and i - entry_idx >= max_hold:
                reason = "hold"
            if unreal >= 0.004 and (i - entry_idx) >= 3:
                # Counterfactual utility of adding 25% parent notional now until the
                # same parent exit. This trains the direction of convex add-on value.
                exit_i = min(i + 1, len(train_df) - 1)
                for j in range(i, min(entry_idx + max_hold + 1, len(train_df) - 1)):
                    pxj = float(close[j])
                    rawj = (pxj * (1.0 - slip) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - pxj * (1.0 + slip)) / max(entry_price, 1e-12)
                    uj = rawj * notional
                    if (take_profit > 0.0 and uj >= take_profit) or (stop_loss > 0.0 and uj <= -abs(stop_loss)) or (max_hold > 0 and j - entry_idx >= max_hold):
                        exit_i = min(j + 1, len(train_df) - 1)
                        break
                add_i = min(i + 1, len(train_df) - 1)
                add_px = _fill_price(train_df, add_i, pos, slip, entry=True)
                exit_px = _fill_price(train_df, exit_i, pos, slip, entry=False)
                add_raw = (exit_px - add_px) / max(add_px, 1e-12) if pos > 0 else (add_px - exit_px) / max(add_px, 1e-12)
                target = add_raw * (0.25 * parent_notional) - fee * (0.25 * parent_notional) * 2.0
                rows.append(_feature_frame(train_df, bundle, decisions, i, state))
                y.append(float(target))
            if reason:
                exit_i = min(i + 1, len(train_df) - 1)
                exit_price = _fill_price(train_df, exit_i, pos, slip, entry=False)
                raw_exit = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * fee * notional
                pos = 0
                continue
        if pos == 0:
            dec = decisions.iloc[i]
            if int(dec.action) == ACTION_CASH or int(dec.side) == 0:
                continue
            fill_i = min(i + 1, len(train_df) - 1)
            pos = int(dec.side)
            entry_price = _fill_price(train_df, fill_i, pos, slip, entry=True)
            entry_equity = cash
            entry_idx = i
            parent_notional = float(dec.notional_exposure)
            notional = parent_notional
            take_profit = float(dec.take_profit)
            stop_loss = float(dec.stop_loss)
            max_hold = int(dec.max_hold_bars)
            cash -= cash * fee * notional
            mfe = mae = 0.0
    if not rows:
        raise RuntimeError("no runner snapshots generated")
    x = pd.concat(rows, ignore_index=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    yy = np.asarray(y, dtype=np.float64)
    reg = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(max_iter=220, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.08, min_samples_leaf=10, random_state=1818))
    clf = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_iter=180, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=0.08, min_samples_leaf=10, random_state=1819))
    reg.fit(x, yy)
    clf.fit(x, (yy > 0.0).astype(int))
    return {"regressor": reg, "classifier": clf, "feature_cols": list(x.columns), "snapshot_count": int(len(x)), "target_mean": float(yy.mean()), "target_p75": float(np.quantile(yy, 0.75)), "target_p95": float(np.quantile(yy, 0.95))}


def _predict_utility(model: dict[str, Any], x: pd.DataFrame) -> tuple[float, float]:
    xx = x.reindex(columns=list(model["feature_cols"])).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pred = float(model["regressor"].predict(xx)[0])
    clf = model["classifier"]
    try:
        classes = np.asarray(clf[-1].classes_, dtype=int)
        proba = clf.predict_proba(xx)[0]
        idx = int(np.flatnonzero(classes == 1)[0]) if np.any(classes == 1) else 0
        p = float(proba[idx])
    except Exception:
        p = 0.0
    return pred, p


def backtest(df: pd.DataFrame, bundle: dict[str, Any], runner_model: dict[str, Any], cfg: PyramidConfig, *, fee: float, slip: float, cost_mult: float = 1.0, decisions: pd.DataFrame | None = None, record: bool = False) -> dict[str, Any]:
    close = _close(df)
    if decisions is None:
        decisions = predict_policy_frame(bundle, df, close=close)
    fee_eff = fee * float(cost_mult)
    slip_eff = slip * float(cost_mult)
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
    action_counts = {"cash": 0, "long": 0, "short": 0}
    exits: dict[str, int] = {}
    pyramid: dict[str, int] = {}
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
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "learned_take_profit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "learned_stop_loss"
            elif max_hold > 0 and hold >= max_hold:
                reason = "learned_max_hold"
            if not reason and not add_done and cfg.add_frac > 0.0 and unreal >= cfg.min_unrealized and hold >= cfg.min_bars_since_entry and dd_abs <= cfg.dd_block:
                state = {"parent_notional": parent_notional, "notional": notional, "bars_since_entry": hold, "unrealized": unreal, "mfe": mfe, "mae": mae, "drawdown_abs": dd_abs, "take_profit": take_profit, "stop_loss": stop_loss, "max_hold": max_hold}
                pred, p = _predict_utility(runner_model, _feature_frame(df, bundle, decisions, i, state))
                cap = parent_notional * cfg.max_total_mult
                delta = max(0.0, min(parent_notional * cfg.add_frac, cap - notional))
                if delta > 1e-12 and pred >= cfg.pred_utility_threshold and p >= 0.50:
                    fill_i = min(i + 1, len(df) - 1)
                    add_px = _fill_price(df, fill_i, pos, slip_eff, entry=True)
                    new_notional = notional + delta
                    entry_price = (entry_price * notional + add_px * delta) / max(new_notional, 1e-12)
                    before = cash
                    cash -= before * fee_eff * delta
                    notional = new_notional
                    add_done = True
                    pyramid["add_on"] = pyramid.get("add_on", 0) + 1
                    if record and open_record is not None:
                        open_record.update({"add_on_timestamp": str(df["timestamp"].iloc[fill_i]), "add_on_delta_notional": float(delta), "add_on_price": float(add_px), "add_pred_utility": float(pred), "add_win_prob": float(p), "add_fee_pct": float(fee_eff * delta * 100.0)})
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
                    out.update({"exit_signal_timestamp": str(df["timestamp"].iloc[i]), "exit_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "exit_reason": reason, "realized_net_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0), "final_notional_exposure": float(notional), "mfe_pct": float(mfe * 100.0), "mae_pct": float(mae * 100.0), "fee_exit_pct": float(fee_eff * notional * 100.0), "cash_after": float(cash)})
                    records.append(out)
                pos = 0
                notional = parent_notional = 0.0
                leverage = 1.0
                cooldown = int(next_cooldown)
                next_cooldown = 0
                add_done = False
                open_record = None
                continue
        if pos != 0:
            continue
        if cooldown > 0:
            cooldown -= 1
            action_counts["cash"] += 1
            continue
        dec = decisions.iloc[i]
        if int(dec.action) == ACTION_CASH or int(dec.side) == 0:
            action_counts["cash"] += 1
            continue
        fill_i = min(i + 1, len(df) - 1)
        pos = int(dec.side)
        entry_price = _fill_price(df, fill_i, pos, slip_eff, entry=True)
        entry_equity = cash
        entry_idx = i
        parent_notional = min(float(dec.notional_exposure), cfg.max_entry_notional)
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
        action_counts["long" if int(dec.action) == ACTION_LONG else "short"] += 1
        if record:
            open_record = {"entry_signal_timestamp": str(df["timestamp"].iloc[i]), "entry_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "side": "LONG" if pos > 0 else "SHORT", "entry_price": float(entry_price), "parent_notional_exposure": float(dec.notional_exposure), "notional_exposure": float(notional), "leverage": float(leverage), "position_fraction": float(notional / max(leverage, 1e-12)), "take_profit": float(take_profit), "stop_loss": float(stop_loss), "max_hold_bars": int(max_hold), "fee_entry_pct": float(fee_eff * notional * 100.0)}
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
    out = {"pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(trades), "wr": float(wins / max(trades, 1)), "trades_per_day": float(trades / _days(df)), "long_entries": int(long_entries), "short_entries": int(short_entries), "avg_notional": float(notional_sum / n), "avg_leverage": float(leverage_sum / n), "action_counts": action_counts, "exits": exits, "pyramid_actions": pyramid}
    if record:
        out["trade_records"] = records
    return out


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any], parent_val_pnl: float) -> float:
    pnl = float(c1["pnl"])
    if int(c1["trades"]) < 20:
        return -1e9 + pnl
    penalty = 0.0
    if pnl < parent_val_pnl:
        penalty += (parent_val_pnl - pnl) * 0.8
    if float(c2["pnl"]) <= 0.0:
        penalty += abs(float(c2["pnl"])) * 1.5
    if float(c3["pnl"]) <= -10.0:
        penalty += abs(float(c3["pnl"])) * 0.8
    return float(pnl + 0.45 * float(c2["pnl"]) + 0.20 * float(c3["pnl"]) - 0.45 * abs(float(c1["mdd"])) - penalty)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V18 convex runner pyramid: supervised counterfactual add-on utility head.")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bundle = joblib.load(args.model)
    cfg_base = dict(bundle["config"])
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    train = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    feature_audit = _audit_contract(train_all, eval_df, list(bundle.get("feature_cols") or []))
    runner_model = _fit_runner_model(train, bundle, fee=float(cfg_base["fee"]), slip=float(cfg_base["slip"]))
    val_dec = predict_policy_frame(bundle, val, close=_close(val))
    eval_dec = predict_policy_frame(bundle, eval_df, close=_close(eval_df))
    parent_val = backtest(val, bundle, runner_model, PyramidConfig("parent_ref", 99.0, 99.0, 999, 0.0, 1.0, 99.0, 99.0), fee=float(cfg_base["fee"]), slip=float(cfg_base["slip"]), decisions=val_dec)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for cfg in _grid():
        v1 = backtest(val, bundle, runner_model, cfg, fee=float(cfg_base["fee"]), slip=float(cfg_base["slip"]), decisions=val_dec, cost_mult=1.0)
        v2 = backtest(val, bundle, runner_model, cfg, fee=float(cfg_base["fee"]), slip=float(cfg_base["slip"]), decisions=val_dec, cost_mult=2.0)
        v3 = backtest(val, bundle, runner_model, cfg, fee=float(cfg_base["fee"]), slip=float(cfg_base["slip"]), decisions=val_dec, cost_mult=3.0)
        row = {"config": asdict(cfg), "validation_cost1": v1, "validation_cost2": v2, "validation_cost3": v3, "selection_score": _score(v1, v2, v3, float(parent_val["pnl"]))}
        rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    if best is None:
        raise RuntimeError("empty grid")
    selected = PyramidConfig(**best["config"])
    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for mult in (1, 2, 3):
        r = backtest(eval_df, bundle, runner_model, selected, fee=float(cfg_base["fee"]), slip=float(cfg_base["slip"]), decisions=eval_dec, cost_mult=float(mult), record=(mult == 1))
        if mult == 1:
            ledger = pd.DataFrame(r.pop("trade_records", []))
            lp = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
            lp.parent.mkdir(parents=True, exist_ok=True)
            ledger.to_csv(lp, index=False)
            ledgers["cost1"] = str(lp)
        metrics[f"cost{mult}"] = r
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "v18_convex_runner_pyramid.pkl"
    joblib.dump({"model_id": MODEL_ID, "base_model": str(args.model), "runner_model": runner_model, "selected_config": asdict(selected), "selection_policy": "2025 Oct-Dec validation only; 2026 fixed OOS"}, model_path)
    args.grid_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**{f"cfg_{k}": v for k, v in r["config"].items()}, "score": r["selection_score"], "val_pnl": r["validation_cost1"]["pnl"], "val_mdd": r["validation_cost1"]["mdd"], "val_trades": r["validation_cost1"]["trades"], "val_c2_pnl": r["validation_cost2"]["pnl"], "val_c3_pnl": r["validation_cost3"]["pnl"]} for r in rows]).to_csv(args.grid_out, index=False)
    blocking: list[str] = []
    warnings: list[str] = []
    if feature_audit["status"] != "pass":
        blocking.extend(feature_audit["blocking"])
    warnings.extend(feature_audit.get("warnings", []))
    if metrics["cost1"]["pnl"] <= 139.40707050411612:
        warnings.append("oos_cost1_did_not_beat_margin110")
    if metrics["cost2"]["pnl"] <= 0.0:
        warnings.append("cost2_not_survived")
    if metrics["cost3"]["pnl"] <= 0.0:
        warnings.append("cost3_not_survived")
    audit = {"status": "pass" if not blocking else "fail", "verdict": "promote" if not blocking and metrics["cost1"]["pnl"] > 139.40707050411612 and metrics["cost2"]["pnl"] > 0.0 and metrics["cost3"]["pnl"] > 0.0 else "iterate", "blocking": blocking, "warnings": warnings, "selection_uses_2026": False, "selection_window": "2025-10-01..2025-12-31", "oos_window": "2026 fixed OOS only after selection", "runner_model_meta": {k: v for k, v in runner_model.items() if k not in {"regressor", "classifier", "feature_cols"}}, "parent_validation_reference": parent_val, "selected_config": asdict(selected), "feature_audit": feature_audit, "metrics": metrics}
    report = {"model_id": MODEL_ID, "design": "v13 margin110 parent plus supervised counterfactual convex runner add-on head. Entry/side preserved; add-on only when open trade is already positive.", "base_model": str(args.model), "model": str(model_path), "split_policy": "Runner utility model trained on 2025 Jan-Sep; config selected on 2025 Oct-Dec; 2026 fixed OOS not used for selection.", "selected_config": asdict(selected), "selection_result": best, "metrics": metrics, "audit": audit, "artifacts": {"model": str(model_path), "report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "ledgers": ledgers}}
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "model": str(model_path), "selected": asdict(selected), "metrics": metrics, "verdict": audit["verdict"]}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
