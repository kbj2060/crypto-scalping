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
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import PyramidConfig, _feature_frame, _predict_utility  # noqa: E402


MODEL_ID = "hf_v13_cost_stressed_runner_v20_20260511"
DEFAULT_MODEL = ROOT / "data/ensemble/supervised/hf_v13_clean_regime_margin110_20260511/v13_clean_regime_margin110.pkl"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_cost_stressed_runner_v20_20260511"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_cost_stressed_runner_v20_20260511_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_cost_stressed_runner_v20_20260511_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_cost_stressed_runner_v20_20260511_grid.csv"


@dataclass(frozen=True)
class CostRunnerConfig:
    name: str
    edge_th: float
    p_th: float
    min_unrealized: float
    min_bars_since_entry: int
    add_frac: float
    max_total_mult: float
    max_entry_notional: float
    dd_block: float


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


def _grid() -> list[CostRunnerConfig]:
    rows: list[CostRunnerConfig] = []
    i = 0
    for edge in (0.002, 0.006, 0.012, 0.020):
        for p_th in (0.55, 0.65, 0.75):
            for add_frac in (0.20, 0.25):
                rows.append(CostRunnerConfig(f"v20_cost_runner_{i}", edge, p_th, 0.004, 3, add_frac, 1.35, 2.75, 0.30))
                i += 1
    rows.append(CostRunnerConfig("v20_parent_noop", 99.0, 1.0, 99.0, 999, 0.0, 1.0, 99.0, 99.0))
    return rows


def _addon_utility(frame: pd.DataFrame, close: np.ndarray, *, pos: int, entry_idx: int, snapshot_idx: int, entry_price: float, current_notional: float, parent_notional: float, take_profit: float, stop_loss: float, max_hold: int, add_frac: float, fee: float, slip: float, cost_mult: float) -> float:
    fee_eff = fee * cost_mult
    slip_eff = slip * cost_mult
    exit_i = min(snapshot_idx + 1, len(frame) - 1)
    for j in range(snapshot_idx, min(entry_idx + max_hold + 1, len(frame) - 1)):
        pxj = float(close[j])
        rawj = (pxj * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - pxj * (1.0 + slip_eff)) / max(entry_price, 1e-12)
        uj = rawj * current_notional
        if (take_profit > 0.0 and uj >= take_profit) or (stop_loss > 0.0 and uj <= -abs(stop_loss)) or (max_hold > 0 and j - entry_idx >= max_hold):
            exit_i = min(j + 1, len(frame) - 1)
            break
    add_i = min(snapshot_idx + 1, len(frame) - 1)
    delta = float(add_frac * parent_notional)
    add_px = _fill_price(frame, add_i, pos, slip_eff, entry=True)
    exit_px = _fill_price(frame, exit_i, pos, slip_eff, entry=False)
    add_raw = (exit_px - add_px) / max(add_px, 1e-12) if pos > 0 else (add_px - exit_px) / max(add_px, 1e-12)
    return float(add_raw * delta - fee_eff * delta * 2.0)


def _fit_cost_runner(frame: pd.DataFrame, bundle: dict[str, Any], *, fee: float, slip: float) -> dict[str, Any]:
    decisions = predict_policy_frame(bundle, frame, close=_close(frame))
    close = _close(frame)
    rows: list[pd.DataFrame] = []
    targets: list[float] = []
    pos = 0
    entry_price = 0.0
    entry_idx = 0
    parent_notional = notional = 0.0
    take_profit = stop_loss = 0.0
    max_hold = 0
    cash = peak = 1.0
    mfe = mae = 0.0
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
            state = {"parent_notional": parent_notional, "notional": notional, "bars_since_entry": hold, "unrealized": unreal, "mfe": mfe, "mae": mae, "drawdown_abs": max(0.0, 1.0 - eq / max(peak, 1e-12)), "take_profit": take_profit, "stop_loss": stop_loss, "max_hold": max_hold}
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "tp"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "sl"
            elif max_hold > 0 and hold >= max_hold:
                reason = "hold"
            if unreal >= 0.004 and hold >= 3:
                u1 = _addon_utility(frame, close, pos=pos, entry_idx=entry_idx, snapshot_idx=i, entry_price=entry_price, current_notional=notional, parent_notional=parent_notional, take_profit=take_profit, stop_loss=stop_loss, max_hold=max_hold, add_frac=0.25, fee=fee, slip=slip, cost_mult=1.0)
                u2 = _addon_utility(frame, close, pos=pos, entry_idx=entry_idx, snapshot_idx=i, entry_price=entry_price, current_notional=notional, parent_notional=parent_notional, take_profit=take_profit, stop_loss=stop_loss, max_hold=max_hold, add_frac=0.25, fee=fee, slip=slip, cost_mult=2.0)
                u3 = _addon_utility(frame, close, pos=pos, entry_idx=entry_idx, snapshot_idx=i, entry_price=entry_price, current_notional=notional, parent_notional=parent_notional, take_profit=take_profit, stop_loss=stop_loss, max_hold=max_hold, add_frac=0.25, fee=fee, slip=slip, cost_mult=3.0)
                target = min(u1, 0.75 * u2, 0.55 * u3)
                x = _feature_frame(frame, bundle, decisions, i, state)
                rows.append(x)
                targets.append(float(target))
            if reason:
                exit_i = min(i + 1, len(frame) - 1)
                exit_px = _fill_price(frame, exit_i, pos, slip, entry=False)
                raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * fee * notional
                pos = 0
                continue
        if pos == 0:
            dec = decisions.iloc[i]
            if int(dec.action) == ACTION_CASH or int(dec.side) == 0:
                continue
            fill_i = min(i + 1, len(frame) - 1)
            pos = int(dec.side)
            entry_price = _fill_price(frame, fill_i, pos, slip, entry=True)
            entry_idx = i
            parent_notional = float(dec.notional_exposure)
            notional = parent_notional
            take_profit = float(dec.take_profit)
            stop_loss = float(dec.stop_loss)
            max_hold = int(dec.max_hold_bars)
            cash -= cash * fee * notional
            mfe = mae = 0.0
    if not rows:
        raise RuntimeError("no cost-stressed runner snapshots")
    x = pd.concat(rows, ignore_index=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = np.asarray(targets, dtype=np.float64)
    reg = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(max_iter=220, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.12, min_samples_leaf=12, random_state=2020))
    clf = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_iter=200, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.12, min_samples_leaf=12, random_state=2021))
    reg.fit(x, y)
    clf.fit(x, (y > 0.0).astype(int))
    return {"regressor": reg, "classifier": clf, "feature_cols": list(x.columns), "snapshot_count": int(len(x)), "positive_rate": float((y > 0.0).mean()), "target_mean": float(y.mean()), "target_p75": float(np.quantile(y, 0.75)), "target_p95": float(np.quantile(y, 0.95))}


def _predict_cost_runner(model: dict[str, Any], x: pd.DataFrame) -> tuple[float, float]:
    xx = x.reindex(columns=list(model["feature_cols"])).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    edge = float(model["regressor"].predict(xx)[0])
    clf = model["classifier"]
    try:
        classes = np.asarray(clf[-1].classes_, dtype=int)
        prob = clf.predict_proba(xx)[0]
        idx = int(np.flatnonzero(classes == 1)[0]) if np.any(classes == 1) else 0
        p = float(prob[idx])
    except Exception:
        p = 0.0
    return edge, p


def backtest(df: pd.DataFrame, bundle: dict[str, Any], model: dict[str, Any], cfg: CostRunnerConfig, *, fee: float, slip: float, cost_mult: float = 1.0, decisions: pd.DataFrame | None = None, record: bool = False) -> dict[str, Any]:
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
    action_counts = {"cash": 0, "long": 0, "short": 0}
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
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "learned_take_profit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "learned_stop_loss"
            elif max_hold > 0 and hold >= max_hold:
                reason = "learned_max_hold"
            if not reason and not add_done and cfg.add_frac > 0.0 and unreal >= cfg.min_unrealized and hold >= cfg.min_bars_since_entry and dd_abs <= cfg.dd_block:
                state = {"parent_notional": parent_notional, "notional": notional, "bars_since_entry": hold, "unrealized": unreal, "mfe": mfe, "mae": mae, "drawdown_abs": dd_abs, "take_profit": take_profit, "stop_loss": stop_loss, "max_hold": max_hold}
                x = _feature_frame(df, bundle, decisions, i, state)
                edge, p = _predict_cost_runner(model, x)
                cap = parent_notional * cfg.max_total_mult
                delta = max(0.0, min(parent_notional * cfg.add_frac, cap - notional))
                if delta > 1e-12 and edge >= cfg.edge_th and p >= cfg.p_th:
                    fill_i = min(i + 1, len(df) - 1)
                    add_px = _fill_price(df, fill_i, pos, slip_eff, entry=True)
                    new_notional = notional + delta
                    entry_price = (entry_price * notional + add_px * delta) / max(new_notional, 1e-12)
                    before = cash
                    cash -= before * fee_eff * delta
                    notional = new_notional
                    runner_actions["add_on"] = runner_actions.get("add_on", 0) + 1
                    if record and open_record is not None:
                        open_record.update({"add_on_timestamp": str(df["timestamp"].iloc[fill_i]), "add_on_delta_notional": float(delta), "add_on_price": float(add_px), "cost_runner_edge": float(edge), "cost_runner_prob": float(p), "add_fee_pct": float(fee_eff * delta * 100.0)})
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
                    out.update({"exit_signal_timestamp": str(df["timestamp"].iloc[i]), "exit_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "exit_reason": reason, "realized_net_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0), "final_notional_exposure": float(notional), "mfe_pct": float(mfe * 100.0), "mae_pct": float(mae * 100.0), "fee_exit_pct": float(fee_eff * notional * 100.0), "cash_after": float(cash)})
                    records.append(out)
                pos = 0
                parent_notional = notional = 0.0
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
    out = {"pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(trades), "wr": float(wins / max(trades, 1)), "trades_per_day": float(trades / _days(df)), "long_entries": int(long_entries), "short_entries": int(short_entries), "avg_notional": float(notional_sum / n), "avg_leverage": float(leverage_sum / n), "action_counts": action_counts, "exits": exits, "runner_actions": runner_actions}
    if record:
        out["trade_records"] = records
    return out


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    pnl = float(c1["pnl"])
    if int(c1["trades"]) < 20:
        return -1e9 + pnl
    return float(pnl + 0.45 * float(c2["pnl"]) + 0.20 * float(c3["pnl"]) - 0.25 * abs(float(c1["mdd"])))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V20 cost-stressed marginal utility runner.")
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
    base = dict(bundle["config"])
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    train = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    feature_audit = _audit_contract(train_all, eval_df, list(bundle.get("feature_cols") or []))
    model = _fit_cost_runner(train, bundle, fee=float(base["fee"]), slip=float(base["slip"]))
    val_dec = predict_policy_frame(bundle, val, close=_close(val))
    eval_dec = predict_policy_frame(bundle, eval_df, close=_close(eval_df))
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for cfg in _grid():
        v1 = backtest(val, bundle, model, cfg, fee=float(base["fee"]), slip=float(base["slip"]), decisions=val_dec, cost_mult=1.0)
        v2 = backtest(val, bundle, model, cfg, fee=float(base["fee"]), slip=float(base["slip"]), decisions=val_dec, cost_mult=2.0)
        v3 = backtest(val, bundle, model, cfg, fee=float(base["fee"]), slip=float(base["slip"]), decisions=val_dec, cost_mult=3.0)
        row = {"config": asdict(cfg), "validation_cost1": v1, "validation_cost2": v2, "validation_cost3": v3, "selection_score": _score(v1, v2, v3)}
        rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    selected = CostRunnerConfig(**best["config"])
    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for mult in (1, 2, 3):
        r = backtest(eval_df, bundle, model, selected, fee=float(base["fee"]), slip=float(base["slip"]), decisions=eval_dec, cost_mult=float(mult), record=(mult == 1))
        if mult == 1:
            ledger = pd.DataFrame(r.pop("trade_records", []))
            lp = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
            lp.parent.mkdir(parents=True, exist_ok=True)
            ledger.to_csv(lp, index=False)
            ledgers["cost1"] = str(lp)
        metrics[f"cost{mult}"] = r
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "v20_cost_stressed_runner.pkl"
    joblib.dump({"model_id": MODEL_ID, "base_model": str(args.model), "cost_runner": model, "selected_config": asdict(selected)}, model_path)
    args.grid_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**{f"cfg_{k}": v for k, v in r["config"].items()}, "score": r["selection_score"], "val_pnl": r["validation_cost1"]["pnl"], "val_mdd": r["validation_cost1"]["mdd"], "val_trades": r["validation_cost1"]["trades"], "val_c2_pnl": r["validation_cost2"]["pnl"], "val_c3_pnl": r["validation_cost3"]["pnl"], "val_add": r["validation_cost1"]["runner_actions"].get("add_on", 0), "val_reject": r["validation_cost1"]["runner_actions"].get("reject", 0)} for r in rows]).to_csv(args.grid_out, index=False)
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
    audit = {"status": "pass" if not blocking else "fail", "verdict": "promote" if not blocking and metrics["cost1"]["pnl"] > 139.40707050411612 and metrics["cost2"]["pnl"] > 0.0 else "iterate", "blocking": blocking, "warnings": warnings, "selection_uses_2026": False, "selection_window": "2025-10-01..2025-12-31", "oos_window": "2026 fixed OOS only after selection", "model_meta": {k: v for k, v in model.items() if k not in {"regressor", "classifier", "feature_cols"}}, "feature_audit": feature_audit, "selected_config": asdict(selected), "metrics": metrics}
    report = {"model_id": MODEL_ID, "design": "v20 cost-stressed marginal add-on utility runner. Label is conservative across cost1/2/3 add-on utility.", "base_model": str(args.model), "model": str(model_path), "split_policy": "Cost runner trained on 2025 Jan-Sep; config selected on 2025 Oct-Dec; 2026 fixed OOS not used for selection.", "selected_config": asdict(selected), "selection_result": best, "metrics": metrics, "audit": audit, "artifacts": {"model": str(model_path), "report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "ledgers": ledgers}}
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "model": str(model_path), "selected": asdict(selected), "metrics": metrics, "verdict": audit["verdict"]}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
