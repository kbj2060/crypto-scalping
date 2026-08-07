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

from ensemble.fully_learned_governor_policy import ACTION_CASH, prepare_features, predict_policy_frame  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _days, _fill_price, _read  # noqa: E402


MODEL_ID = "hf_v13_clean_regime_conformal_downside_veto_20260511"
DEFAULT_MODEL = ROOT / "data/ensemble/supervised/hf_v13_clean_regime_validation_selected_exposure_20260511/v13_clean_regime_validation_selected_exposure.pkl"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_clean_regime_conformal_downside_veto_20260511"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_clean_regime_conformal_downside_veto_20260511_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_clean_regime_conformal_downside_veto_20260511_audit.json"


@dataclass(frozen=True)
class VetoConfig:
    loss_prob_veto: float
    loss_prob_scale: float
    pred_net_floor: float
    winner_protect_floor: float
    scale_mult: float
    long_trend_bias_floor: float
    max_notional: float


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


def _model_features(frame: pd.DataFrame, bundle: dict[str, Any], decisions: pd.DataFrame) -> pd.DataFrame:
    cols = list(bundle.get("feature_cols") or [])
    x = prepare_features(frame, side_hint=0, close=_close(frame), feature_cols=cols).reset_index(drop=True)
    d = decisions.reset_index(drop=True)
    x["parent_side"] = pd.to_numeric(d["side"], errors="coerce").fillna(0.0)
    x["parent_notional"] = pd.to_numeric(d["notional_exposure"], errors="coerce").fillna(0.0)
    x["parent_leverage"] = pd.to_numeric(d["leverage"], errors="coerce").fillna(0.0)
    x["parent_position_fraction"] = pd.to_numeric(d["position_fraction"], errors="coerce").fillna(0.0)
    x["parent_take_profit"] = pd.to_numeric(d["take_profit"], errors="coerce").fillna(0.0)
    x["parent_stop_loss"] = pd.to_numeric(d["stop_loss"], errors="coerce").fillna(0.0)
    x["parent_max_hold_bars"] = pd.to_numeric(d["max_hold_bars"], errors="coerce").fillna(0.0)
    x["parent_quality_score"] = pd.to_numeric(d["quality_score"], errors="coerce").fillna(0.0)
    x["parent_confidence"] = pd.to_numeric(d["confidence"], errors="coerce").fillna(0.0)
    return x.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _simulate_one(frame: pd.DataFrame, i: int, dec: pd.Series, *, fee: float, slip: float) -> dict[str, Any]:
    side = int(dec.side)
    fill_idx = min(i + 1, len(frame) - 1)
    entry_price = _fill_price(frame, fill_idx, side, slip, entry=True)
    notional = float(dec.notional_exposure)
    take_profit = float(dec.take_profit)
    stop_loss = float(dec.stop_loss)
    max_hold = int(dec.max_hold_bars)
    cash = 1.0 - fee * notional
    peak_unreal = 0.0
    min_unreal = 0.0
    exit_i = min(fill_idx + max(1, max_hold), len(frame) - 1)
    reason = "max_hold"
    close = _close(frame)
    for j in range(fill_idx, min(fill_idx + max(1, max_hold) + 1, len(frame))):
        px = float(close[j])
        raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
        unreal = raw * notional
        peak_unreal = max(peak_unreal, unreal)
        min_unreal = min(min_unreal, unreal)
        if take_profit > 0.0 and unreal >= take_profit:
            exit_i, reason = min(j + 1, len(frame) - 1), "take_profit"
            break
        if stop_loss > 0.0 and unreal <= -abs(stop_loss):
            exit_i, reason = min(j + 1, len(frame) - 1), "stop_loss"
            break
    exit_price = _fill_price(frame, exit_i, side, slip, entry=False)
    raw = (exit_price - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
    before = cash
    cash = cash * (1.0 + raw * notional)
    cash -= before * fee * notional
    return {
        "signal_idx": int(i),
        "timestamp": str(frame["timestamp"].iloc[i]),
        "side": int(side),
        "realized_net_pct": float((cash - 1.0) * 100.0),
        "min_unrealized_pct": float(min_unreal * 100.0),
        "peak_unrealized_pct": float(peak_unreal * 100.0),
        "exit_reason": reason,
    }


def _candidate_table(frame: pd.DataFrame, bundle: dict[str, Any], *, fee: float, slip: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    decisions = predict_policy_frame(bundle, frame, close=_close(frame))
    features = _model_features(frame, bundle, decisions)
    active_idx = np.flatnonzero((pd.to_numeric(decisions["action"], errors="coerce").fillna(0).to_numpy(int) != ACTION_CASH) & (pd.to_numeric(decisions["side"], errors="coerce").fillna(0).to_numpy(int) != 0))
    active_idx = active_idx[active_idx < len(frame) - 2]
    rows = [_simulate_one(frame, int(i), decisions.iloc[int(i)], fee=fee, slip=slip) for i in active_idx]
    labels = pd.DataFrame(rows)
    x = features.iloc[active_idx].reset_index(drop=True)
    labels = labels.reset_index(drop=True)
    return x, labels, decisions


def _fit_models(x: pd.DataFrame, labels: pd.DataFrame) -> dict[str, Any]:
    y_loss = (pd.to_numeric(labels["realized_net_pct"], errors="coerce").fillna(0.0).to_numpy(float) <= -1.25).astype(int)
    y_win = (pd.to_numeric(labels["realized_net_pct"], errors="coerce").fillna(0.0).to_numpy(float) >= 2.00).astype(int)
    y_net = pd.to_numeric(labels["realized_net_pct"], errors="coerce").fillna(0.0).to_numpy(float)
    clf = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(max_iter=180, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.10, min_samples_leaf=12, random_state=1701),
    )
    reg = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(max_iter=180, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.10, min_samples_leaf=12, random_state=1702),
    )
    win = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(max_iter=180, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.10, min_samples_leaf=12, random_state=1703),
    )
    clf.fit(x, y_loss)
    win.fit(x, y_win)
    reg.fit(x, y_net)
    return {"loss_classifier": clf, "winner_classifier": win, "net_regressor": reg, "feature_cols": list(x.columns), "loss_threshold_pct": -1.25, "winner_threshold_pct": 2.00}


def _predict_risk(model: dict[str, Any], x: pd.DataFrame) -> pd.DataFrame:
    cols = list(model["feature_cols"])
    xx = x.reindex(columns=cols).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    clf = model["loss_classifier"]
    if hasattr(clf[-1], "classes_") and len(clf[-1].classes_) == 1:
        p_loss = np.full(len(xx), float(clf[-1].classes_[0]), dtype=float)
    else:
        classes = np.asarray(clf[-1].classes_, dtype=int)
        proba = clf.predict_proba(xx)
        idx = int(np.flatnonzero(classes == 1)[0]) if np.any(classes == 1) else 0
        p_loss = proba[:, idx]
    pred_net = np.asarray(model["net_regressor"].predict(xx), dtype=float)
    win = model["winner_classifier"]
    if hasattr(win[-1], "classes_") and len(win[-1].classes_) == 1:
        p_win = np.full(len(xx), float(win[-1].classes_[0]), dtype=float)
    else:
        classes = np.asarray(win[-1].classes_, dtype=int)
        proba = win.predict_proba(xx)
        idx = int(np.flatnonzero(classes == 1)[0]) if np.any(classes == 1) else 0
        p_win = proba[:, idx]
    return pd.DataFrame({"loss_prob": p_loss, "winner_prob": p_win, "pred_net_pct": pred_net}, index=x.index)


def _grid() -> list[VetoConfig]:
    return [
        VetoConfig(float(veto), float(scale), float(floor), float(winner), float(mult), float(trend), float(cap))
        for veto in (0.72, 0.82)
        for scale in (0.58,)
        for floor in (-0.50, -0.20)
        for winner in (0.50, 0.65)
        for mult in (0.70,)
        for trend in (0.00,)
        for cap in (2.30,)
    ]


def backtest(
    frame: pd.DataFrame,
    bundle: dict[str, Any],
    risk_model: dict[str, Any],
    cfg: VetoConfig,
    *,
    fee: float,
    slip: float,
    decisions: pd.DataFrame | None = None,
    risk: pd.DataFrame | None = None,
    record: bool = False,
) -> dict[str, Any]:
    close = _close(frame)
    if decisions is None:
        decisions = predict_policy_frame(bundle, frame, close=close)
    if risk is None:
        risk = _predict_risk(risk_model, _model_features(frame, bundle, decisions))
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_idx = 0
    notional = 0.0
    leverage = 1.0
    take_profit = 0.0
    stop_loss = 0.0
    max_hold = 0
    cooldown = 0
    next_cooldown = 0
    trades = wins = long_entries = short_entries = 0
    notional_sum = leverage_sum = 0.0
    blocks: dict[str, int] = {}
    exits: dict[str, int] = {}
    ledger: list[dict[str, Any]] = []
    open_record: dict[str, Any] | None = None

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
        unreal = raw * notional
        return cash * (1.0 + unreal), unreal

    for i in range(0, len(frame) - 2):
        eq, unreal = mark(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if pos != 0:
            hold_bars = i - entry_idx
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "learned_take_profit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "learned_stop_loss"
            elif max_hold > 0 and hold_bars >= max_hold:
                reason = "learned_max_hold"
            if reason:
                fill_idx = min(i + 1, len(frame) - 1)
                exit_price = _fill_price(frame, fill_idx, pos, slip, entry=False)
                raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * fee * notional
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] = exits.get(reason, 0) + 1
                if record and open_record is not None:
                    out = dict(open_record)
                    out.update({"exit_signal_timestamp": str(frame["timestamp"].iloc[i]), "exit_fill_timestamp": str(frame["timestamp"].iloc[fill_idx]), "exit_reason": reason, "trade_pnl_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0), "cash_after": float(cash)})
                    ledger.append(out)
                pos = 0
                cooldown = int(next_cooldown)
                next_cooldown = 0
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
        row = frame.iloc[i]
        rr = risk.iloc[i]
        if int(dec.side) > 0 and float(row.get("clean_regime_2024_unsup_v4_trend_bias", 0.0) or 0.0) < cfg.long_trend_bias_floor:
            blocks["long_trend_bias_floor"] = blocks.get("long_trend_bias_floor", 0) + 1
            continue
        protected = float(rr.winner_prob) >= cfg.winner_protect_floor
        if (float(rr.loss_prob) >= cfg.loss_prob_veto or float(rr.pred_net_pct) <= cfg.pred_net_floor) and not protected:
            blocks["downside_veto"] = blocks.get("downside_veto", 0) + 1
            continue
        size_mult = cfg.scale_mult if float(rr.loss_prob) >= cfg.loss_prob_scale and not protected else 1.0
        fill_idx = min(i + 1, len(frame) - 1)
        pos = int(dec.side)
        entry_price = _fill_price(frame, fill_idx, pos, slip, entry=True)
        entry_equity = cash
        entry_idx = i
        notional = min(float(dec.notional_exposure) * float(size_mult), float(cfg.max_notional))
        leverage = float(dec.leverage)
        take_profit = float(dec.take_profit) * max(float(size_mult), 0.60)
        stop_loss = float(dec.stop_loss)
        max_hold = int(dec.max_hold_bars)
        next_cooldown = int(dec.cooldown_bars)
        cash -= cash * fee * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        leverage_sum += leverage
        if record:
            open_record = {
                "entry_signal_timestamp": str(frame["timestamp"].iloc[i]),
                "entry_fill_timestamp": str(frame["timestamp"].iloc[fill_idx]),
                "side": "LONG" if pos > 0 else "SHORT",
                "entry_price": float(entry_price),
                "notional_exposure": float(notional),
                "leverage": float(leverage),
                "loss_prob": float(rr.loss_prob),
                "winner_prob": float(rr.winner_prob),
                "pred_net_pct": float(rr.pred_net_pct),
                "size_mult": float(size_mult),
            }
    if pos != 0:
        fill_idx = len(frame) - 1
        exit_price = _fill_price(frame, fill_idx, pos, slip, entry=False)
        raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * fee * notional
        trades += 1
        wins += int(cash > entry_equity)
        exits["forced_end"] = exits.get("forced_end", 0) + 1
    n = max(long_entries + short_entries, 1)
    out = {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "trades_per_day": float(trades / _days(frame)),
        "wr": float(wins / max(trades, 1)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(notional_sum / n),
        "avg_leverage": float(leverage_sum / n),
        "block_reason_counts": blocks,
        "exits": exits,
    }
    if record:
        out["ledger"] = ledger
    return out


def _score(r1: dict[str, Any], r2: dict[str, Any], r3: dict[str, Any]) -> float:
    pnl = float(r1["pnl"])
    mdd = abs(float(r1["mdd"]))
    if int(r1["trades"]) < 20:
        return -1e9 + pnl
    return float(pnl + 0.25 * float(r2["pnl"]) + 0.10 * float(r3["pnl"]) - 1.8 * mdd - max(0.0, 100.0 - pnl) * 4.0 - max(0.0, mdd - 15.0) * 12.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate validation-selected conformal downside veto over v13 clean-regime policy.")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bundle = joblib.load(args.model)
    cfg = dict(bundle["config"])
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    fit = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    feature_audit = _audit_contract(train_all, eval_df, list(bundle.get("feature_cols") or []))

    fit_x, fit_y, _ = _candidate_table(fit, bundle, fee=float(cfg["fee"]), slip=float(cfg["slip"]))
    risk_model = _fit_models(fit_x, fit_y)
    val_decisions = predict_policy_frame(bundle, val, close=_close(val))
    eval_decisions = predict_policy_frame(bundle, eval_df, close=_close(eval_df))
    val_risk = _predict_risk(risk_model, _model_features(val, bundle, val_decisions))
    eval_risk = _predict_risk(risk_model, _model_features(eval_df, bundle, eval_decisions))
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for veto_cfg in _grid():
        r1 = backtest(val, bundle, risk_model, veto_cfg, fee=float(cfg["fee"]), slip=float(cfg["slip"]), decisions=val_decisions, risk=val_risk)
        r2 = backtest(val, bundle, risk_model, veto_cfg, fee=float(cfg["fee"]) * 2.0, slip=float(cfg["slip"]) * 2.0, decisions=val_decisions, risk=val_risk)
        r3 = backtest(val, bundle, risk_model, veto_cfg, fee=float(cfg["fee"]) * 3.0, slip=float(cfg["slip"]) * 3.0, decisions=val_decisions, risk=val_risk)
        row = {"config": asdict(veto_cfg), "validation_cost1": r1, "validation_cost2": r2, "validation_cost3": r3, "selection_score": _score(r1, r2, r3)}
        rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    if best is None:
        raise RuntimeError("no config selected")
    selected = VetoConfig(**best["config"])
    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for mult in (1, 2, 3):
        result = backtest(eval_df, bundle, risk_model, selected, fee=float(cfg["fee"]) * mult, slip=float(cfg["slip"]) * mult, decisions=eval_decisions, risk=eval_risk, record=(mult == 1))
        if mult == 1:
            ledger = pd.DataFrame(result.pop("ledger", []))
            ledger_path = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            ledger.to_csv(ledger_path, index=False)
            ledgers["cost1"] = str(ledger_path)
        metrics[f"cost{mult}"] = result
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "v13_conformal_downside_veto.pkl"
    joblib.dump({"model_id": MODEL_ID, "base_model": str(args.model), "risk_model": risk_model, "selected_config": asdict(selected), "selection_policy": "Risk model fit on 2025 Jan-Sep candidates; veto config selected on 2025 Oct-Dec validation only; 2026 fixed OOS not used for selection."}, model_path)
    grid_path = args.report_out.with_name(args.report_out.stem + "_validation_grid.json")
    grid_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    blocking: list[str] = []
    warnings: list[str] = []
    if feature_audit["status"] != "pass":
        blocking.extend(feature_audit["blocking"])
    warnings.extend(feature_audit.get("warnings", []))
    if metrics["cost1"]["pnl"] < 100.0:
        warnings.append("oos_pnl_below_100pct")
    if metrics["cost1"]["mdd"] < -15.0:
        warnings.append("oos_mdd_target_not_met")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote_candidate" if not blocking and metrics["cost1"]["pnl"] >= 100.0 and metrics["cost1"]["mdd"] >= -15.0 else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "fit_window": "2025-01-01..2025-09-30",
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed, used only after veto config selection",
        "selected_config": asdict(selected),
        "fit_candidate_rows": int(len(fit_y)),
        "feature_audit": feature_audit,
        "metrics": metrics,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Loss-candidate classifier + net-return critic + validation-selected downside veto/size-down over v13 clean-regime policy.",
        "base_model": str(args.model),
        "model": str(model_path),
        "split_policy": "Risk model fit on 2025 Jan-Sep; config selected on 2025 Oct-Dec; 2026 fixed OOS not used for selection.",
        "selected_config": asdict(selected),
        "selection_score": best["selection_score"],
        "selection_result": {k: v for k, v in best.items() if k != "selection_score"},
        "metrics": metrics,
        "audit": audit,
        "artifacts": {"model": str(model_path), "report": str(args.report_out), "audit": str(args.audit_out), "validation_grid": str(grid_path), "ledgers": ledgers},
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "model": str(model_path), "selected_config": asdict(selected), "metrics": metrics, "verdict": audit["verdict"]}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
