#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_1_tp_runner_20260610 as runner  # noqa: E402
import train_eval_omega1_2_1_age_lifecycle_labels_20260611 as age  # noqa: E402
import train_eval_omega1_2_1_exit_only_rl_editor_20260610 as base  # noqa: E402
import train_eval_omega1_2_1_tp_runner_meta_selector_20260610 as meta  # noqa: E402


MODEL_ID = "omega1_2_1_age_defensive_actions_20260612"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
AGE_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_1_age_lifecycle_labels_20260611"
TP_RUNNER_BUNDLE = ROOT / "data/ensemble/supervised/omega1_2_1_tp_runner_meta_selector_20260610/tp_runner_meta_selector.joblib"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _tp_runner_extend_allowed(bundle: dict[str, Any] | None, frame: pd.DataFrame, state: pd.DataFrame, pos: base.Position, i: int, unreal: float) -> bool:
    if not bundle:
        return False
    template = meta.RunnerTemplate(**bundle["template"])
    return meta._selector_allowed(
        bundle.get("model"),
        list(bundle.get("feature_cols", [])),
        frame,
        state,
        pos,
        int(i),
        float(unreal),
        template=template,
        proba_min=float(bundle.get("proba_min", 2.0)),
    )


def _rule_signal(pos: base.Position, i: int, unreal: float, *, mode: str) -> bool:
    age_bars = max(int(i) - int(pos.entry_i), 0)
    mfe = max(float(pos.mfe), float(unreal))
    giveback = (mfe - float(unreal)) / max(abs(mfe), 1e-8) if mfe > 0.0 else 0.0
    tp_progress = float(unreal) / max(float(pos.take_profit), 1e-8)
    if mode == "rule_floor_gb45_prog60":
        return age_bars >= 3 and unreal >= 0.020 and tp_progress >= 0.60 and giveback >= 0.45
    if mode == "rule_floor_gb35_prog75":
        return age_bars >= 3 and unreal >= 0.025 and tp_progress >= 0.75 and giveback >= 0.35
    if mode == "rule_breakeven_prog55":
        return age_bars >= 3 and unreal >= 0.020 and tp_progress >= 0.55
    if mode == "rule_tp_downshift_prog70":
        return age_bars >= 3 and unreal >= 0.025 and tp_progress >= 0.70 and giveback >= 0.35
    return False


def _model_signal(
    model: Any | None,
    feature_cols: list[str],
    frame: pd.DataFrame,
    state: pd.DataFrame,
    pos: base.Position,
    i: int,
    unreal: float,
    *,
    proba_min: float,
    min_unreal: float,
    min_age: int,
) -> bool:
    if model is None:
        return False
    if unreal < float(min_unreal) or max(int(i) - int(pos.entry_i), 0) < int(min_age):
        return False
    feat = age._feature_row(frame, state, pos, i, unreal)
    x = np.asarray([[float(feat[c]) for c in feature_cols]], dtype=np.float64)
    if hasattr(model, "predict_proba"):
        p = float(model.predict_proba(x)[0, 1])
    else:
        p = float(model.predict(x)[0])
    return p >= float(proba_min)


def _apply_defensive_action(pos: base.Position, unreal: float, *, action: str) -> tuple[base.Position, bool, str]:
    out = base.Position(**pos.__dict__)
    if getattr(out, "tightened", 0):
        return out, False, ""
    if action == "floor_raise":
        # Keep the trade open, but protect part of current MFE.
        floor = max(0.001, min(float(unreal) * 0.55, float(out.take_profit) * 0.75))
        out.floor_unreal = max(float(out.floor_unreal), floor)
        out.tightened = 1
        return out, True, "age_floor_raise"
    if action == "breakeven_lock":
        out.floor_unreal = max(float(out.floor_unreal), 0.001)
        out.tightened = 1
        return out, True, "age_breakeven_lock"
    if action == "tp_downshift":
        # Lower TP only to a level still above current unrealized PnL.
        new_tp = max(float(unreal) + 0.006, float(out.take_profit) * 0.80)
        if new_tp < float(out.take_profit):
            out.take_profit = new_tp
            out.floor_unreal = max(float(out.floor_unreal), max(0.001, float(unreal) * 0.45))
            out.tightened = 1
            return out, True, "age_tp_downshift"
    return out, False, ""


def _metrics(cash: float, equity_curve: list[float], trades: list[float], reasons: dict[str, int], actions: dict[str, int], long_entries: int, short_entries: int) -> dict[str, Any]:
    eq = np.asarray(equity_curve if equity_curve else [1.0], dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    dd = (eq / np.maximum(peak, 1e-12) - 1.0) * 100.0
    arr = np.asarray(trades, dtype=np.float64)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(dd.min()),
        "trades": int(len(trades)),
        "wr": float(np.mean(arr > 0.0)) if len(arr) else 0.0,
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exit_reasons": dict(reasons),
        "defensive_actions": dict(actions),
    }


def _simulate(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    state: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    tp_bundle: dict[str, Any] | None,
    rule_mode: str,
    model: Any | None,
    feature_cols: list[str],
    model_action: str,
    proba_min: float,
    min_unreal: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    arrays = base._arrays(frame)
    active = np.asarray(base.omega._active(dec), dtype=bool)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    equity_curve = [cash]
    trades: list[float] = []
    rows: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    actions: dict[str, int] = {}
    pos = base.Position()
    long_entries = short_entries = 0
    extensions = 0
    template = meta.RunnerTemplate(**tp_bundle["template"]) if tp_bundle else meta.TEMPLATES[0]
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = base._unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, unreal)
            pos.mae = min(pos.mae, unreal)
            equity_curve.append(cash * (1.0 + unreal))
            reason = ""
            if pos.take_profit > 0.0 and unreal >= pos.take_profit:
                if tp_bundle and extensions < int(template.max_extensions) and _tp_runner_extend_allowed(tp_bundle, frame, state, pos, i, unreal):
                    extensions += 1
                    old_tp = float(pos.take_profit)
                    pos.floor_unreal = max(float(pos.floor_unreal), old_tp * float(template.floor_frac))
                    pos.take_profit = old_tp * float(template.extend_mult)
                else:
                    reason = "take_profit"
            elif pos.floor_unreal > -abs(pos.stop_loss) and unreal <= pos.floor_unreal:
                reason = "meta_runner_profit_lock_exit"
            elif pos.stop_loss > 0.0 and unreal <= -abs(pos.stop_loss):
                reason = "stop_loss"
            elif rule_mode and _rule_signal(pos, i, unreal, mode=rule_mode):
                action = "tp_downshift" if "tp_downshift" in rule_mode else ("breakeven_lock" if "breakeven" in rule_mode else "floor_raise")
                pos, changed, action_name = _apply_defensive_action(pos, unreal, action=action)
                if changed:
                    actions[action_name] = actions.get(action_name, 0) + 1
            elif _model_signal(model, feature_cols, frame, state, pos, i, unreal, proba_min=proba_min, min_unreal=min_unreal, min_age=3):
                pos, changed, action_name = _apply_defensive_action(pos, unreal, action=model_action)
                if changed:
                    actions[action_name] = actions.get(action_name, 0) + 1
            if reason:
                close_pos = base.Position(**pos.__dict__)
                cash, pos, _ = base._close_fraction(cash, arrays, close_pos, i, 1.0, fee_eff, slip_eff)
                net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
                trades.append(net_pct)
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append(runner._ledger_row(frame, arrays, close_pos, i, cash, net_pct, reason, extensions))
                extensions = 0
            continue
        equity_curve.append(cash)
        if bool(active[i]):
            side = int(dec.iloc[int(i)].get("side", 0) or 0)
            cash, pos, entered = base._enter(cash, arrays, dec, i, fee_eff, slip_eff)
            if entered:
                long_entries += int(side > 0)
                short_entries += int(side < 0)
                extensions = 0
    if pos.side != 0:
        close_pos = base.Position(**pos.__dict__)
        cash, pos, _ = base._close_fraction(cash, arrays, close_pos, len(frame) - 1, 1.0, fee_eff, slip_eff)
        net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
        trades.append(net_pct)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        rows.append(runner._ledger_row(frame, arrays, close_pos, len(frame) - 1, cash, net_pct, "forced_end", extensions))
    return _metrics(cash, equity_curve, trades, reasons, actions, long_entries, short_entries), pd.DataFrame(rows)


def _row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(metrics["pnl"]),
        f"{prefix}_mdd": float(metrics["mdd"]),
        f"{prefix}_wr": float(metrics["wr"]),
        f"{prefix}_trades": int(metrics["trades"]),
        f"{prefix}_long": int(metrics["long_entries"]),
        f"{prefix}_short": int(metrics["short_entries"]),
        f"{prefix}_reasons": metrics["exit_reasons"],
        f"{prefix}_actions": metrics["defensive_actions"],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = runner._build()
    print(json.dumps({"stage": "build_done", "sec": round(time.time() - t0, 3)}), flush=True)
    labels = pd.read_csv(AGE_LABEL_DIR / "validation_age_lifecycle_labels.csv")
    tp_bundle = joblib.load(TP_RUNNER_BUNDLE) if TP_RUNNER_BUNDLE.exists() else None
    model, feature_cols, model_diag = age._train_model(labels, kind="et", seed=260613)
    configs: list[dict[str, Any]] = [
        {"variant": "baseline_no_runner", "tp_bundle": None, "rule_mode": "", "model": None, "feature_cols": [], "model_action": "", "proba_min": 2.0, "min_unreal": 999.0},
        {"variant": "tp_runner_only", "tp_bundle": tp_bundle, "rule_mode": "", "model": None, "feature_cols": [], "model_action": "", "proba_min": 2.0, "min_unreal": 999.0},
        {"variant": "tp_runner_rule_floor_gb45_prog60", "tp_bundle": tp_bundle, "rule_mode": "rule_floor_gb45_prog60", "model": None, "feature_cols": [], "model_action": "", "proba_min": 2.0, "min_unreal": 999.0},
        {"variant": "tp_runner_rule_breakeven_prog55", "tp_bundle": tp_bundle, "rule_mode": "rule_breakeven_prog55", "model": None, "feature_cols": [], "model_action": "", "proba_min": 2.0, "min_unreal": 999.0},
        {"variant": "tp_runner_rule_tp_downshift_prog70", "tp_bundle": tp_bundle, "rule_mode": "rule_tp_downshift_prog70", "model": None, "feature_cols": [], "model_action": "", "proba_min": 2.0, "min_unreal": 999.0},
        {"variant": "tp_runner_model_floor_p085_u040", "tp_bundle": tp_bundle, "rule_mode": "", "model": model, "feature_cols": feature_cols, "model_action": "floor_raise", "proba_min": 0.85, "min_unreal": 0.040},
    ]
    rows: list[dict[str, Any]] = []
    ledgers: dict[str, dict[str, pd.DataFrame]] = {}
    for idx, cfg in enumerate(configs):
        print(json.dumps({"stage": "simulate_start", "variant": cfg["variant"], "sec": round(time.time() - t0, 3)}), flush=True)
        row = {"variant_id": int(idx), "variant": str(cfg["variant"])}
        split_ledgers: dict[str, pd.DataFrame] = {}
        for split in ("validation", "oos"):
            metrics, ledger = _simulate(
                data[split]["frame"],
                data[split]["dec"],
                data[split]["state"],
                fee=float(data[split]["fee"]),
                slip=float(data[split]["slip"]),
                cost_mult=3.0,
                tp_bundle=cfg["tp_bundle"],
                rule_mode=str(cfg["rule_mode"]),
                model=cfg["model"],
                feature_cols=list(cfg["feature_cols"]),
                model_action=str(cfg["model_action"]),
                proba_min=float(cfg["proba_min"]),
                min_unreal=float(cfg["min_unreal"]),
            )
            row.update(_row(split, metrics))
            split_ledgers[split] = ledger
        ledgers[str(idx)] = split_ledgers
        rows.append(row)
    ranking = pd.DataFrame(rows)
    base_oos = float(ranking.loc[ranking["variant"].eq("baseline_no_runner"), "oos_pnl"].iloc[0])
    base_val = float(ranking.loc[ranking["variant"].eq("baseline_no_runner"), "validation_pnl"].iloc[0])
    ranking["delta_oos_pnl"] = ranking["oos_pnl"] - base_oos
    ranking["delta_validation_pnl"] = ranking["validation_pnl"] - base_val
    ranking["score"] = ranking["oos_pnl"] + 0.45 * ranking["validation_pnl"] + 0.25 * ranking["oos_mdd"] + 0.20 * ranking["validation_mdd"]
    ranking = ranking.sort_values(["oos_pnl", "validation_pnl", "oos_mdd"], ascending=[False, False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "defensive_actions_ranking.csv", index=False)
    for variant_id in sorted(set([0, 1] + [int(x) for x in ranking["variant_id"].head(5).tolist()])):
        for split, ledger in ledgers[str(variant_id)].items():
            ledger.to_csv(OUT_DIR / f"{split}_variant{variant_id}_ledger.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "method": "Use age-lifecycle labels to trigger non-closing defensive actions: floor raise, breakeven lock, or TP downshift. Parent entries and true-leverage contract are frozen.",
        "label_source": str(AGE_LABEL_DIR / "validation_age_lifecycle_labels.csv"),
        "label_diag": {
            "rows": int(len(labels)),
            "positive": int(labels["label_exit_now"].sum()),
            "positive_rate": float(labels["label_exit_now"].mean()),
            "edge_mean": float(labels["edge_pct"].mean()),
            "edge_median": float(labels["edge_pct"].median()),
        },
        "model_diag": model_diag,
        "top": ranking.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "defensive_actions_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "top": ranking.head(10).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
