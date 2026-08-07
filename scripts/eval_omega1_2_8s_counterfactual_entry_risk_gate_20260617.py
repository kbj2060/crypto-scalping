#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_8b_full_retrain_numeric_cash_sleeve_leverage_only_20260616 as exp  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_8s_counterfactual_entry_risk_gate_20260617"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


@dataclass(frozen=True)
class GateVariant:
    name: str
    stop_max: float
    mae_max: float
    net_lb_min: float
    min_fallback_entries: int


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


def _reason_count(reasons: Any, key: str) -> int:
    if not isinstance(reasons, dict):
        return 0
    return int(reasons.get(key, 0) or 0)


def _base_sleeve_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        **metrics,
        "primary_entries": int(metrics["long_entries"] + metrics["short_entries"]),
        "fallback_entries": 0,
        "primary_takeovers": 0,
        "exit_reasons": dict(metrics.get("exit_reasons") or {}),
    }


def _fit_regressor(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=140,
        learning_rate=0.035,
        max_leaf_nodes=9,
        l2_regularization=2.0,
        random_state=int(seed),
    )


def _fit_predict_entry_risk(
    x_val: pd.DataFrame,
    x_oos: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    side: str,
    seed: int,
    net_cal_q: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if side not in {"long", "short"}:
        raise RuntimeError(f"invalid side: {side}")
    idx = labels["i"].to_numpy(dtype=np.int64)
    y_stop = np.zeros(len(x_val), dtype=np.float64)
    y_mae = np.zeros(len(x_val), dtype=np.float64)
    y_net = np.zeros(len(x_val), dtype=np.float64)
    y_stop[idx] = labels[f"{side}_stop"].to_numpy(dtype=np.float64)
    y_mae[idx] = np.maximum(-labels[f"{side}_mae"].to_numpy(dtype=np.float64), 0.0)
    y_net[idx] = labels[f"{side}_net"].to_numpy(dtype=np.float64)

    val_stop = np.zeros(len(x_val), dtype=np.float64)
    val_mae = np.zeros(len(x_val), dtype=np.float64)
    val_net_lb = np.zeros(len(x_val), dtype=np.float64)
    folds_meta: list[dict[str, Any]] = []
    for fold_id, (tr, va) in enumerate(exp._chron_folds(idx)):
        x_tr = x_val.iloc[tr].to_numpy(dtype=np.float64)
        x_va = x_val.iloc[va].to_numpy(dtype=np.float64)
        stop_model = _fit_regressor(seed + fold_id * 100 + 1)
        mae_model = _fit_regressor(seed + fold_id * 100 + 2)
        net_model = _fit_regressor(seed + fold_id * 100 + 3)
        stop_model.fit(x_tr, y_stop[tr])
        mae_model.fit(x_tr, y_mae[tr])
        net_model.fit(x_tr, y_net[tr])
        stop_pred_tr = stop_model.predict(x_tr).astype(np.float64)
        mae_pred_tr = mae_model.predict(x_tr).astype(np.float64)
        net_pred_tr = net_model.predict(x_tr).astype(np.float64)
        stop_pred_va = stop_model.predict(x_va).astype(np.float64)
        mae_pred_va = mae_model.predict(x_va).astype(np.float64)
        net_pred_va = net_model.predict(x_va).astype(np.float64)
        net_offset = float(np.quantile(np.abs(y_net[tr] - net_pred_tr), net_cal_q))
        val_stop[va] = np.clip(stop_pred_va, 0.0, 1.0)
        val_mae[va] = np.maximum(mae_pred_va, 0.0)
        val_net_lb[va] = net_pred_va - net_offset
        folds_meta.append(
            {
                "fold": int(fold_id),
                "train_rows": int(len(tr)),
                "val_rows": int(len(va)),
                "stop_mean_train": float(np.mean(y_stop[tr])),
                "mae_mean_train": float(np.mean(y_mae[tr])),
                "net_abs_resid_q": net_offset,
                "mae_train_mae": float(np.mean(np.abs(y_mae[tr] - mae_pred_tr))),
                "stop_train_mae": float(np.mean(np.abs(y_stop[tr] - stop_pred_tr))),
            }
        )

    x_train = x_val.iloc[idx].to_numpy(dtype=np.float64)
    stop_model = _fit_regressor(seed + 9001)
    mae_model = _fit_regressor(seed + 9002)
    net_model = _fit_regressor(seed + 9003)
    stop_model.fit(x_train, y_stop[idx])
    mae_model.fit(x_train, y_mae[idx])
    net_model.fit(x_train, y_net[idx])
    net_pred_train = net_model.predict(x_train).astype(np.float64)
    final_net_offset = float(np.quantile(np.abs(y_net[idx] - net_pred_train), net_cal_q))
    out = {
        "val_stop": val_stop,
        "val_mae": val_mae,
        "val_net_lb": val_net_lb,
        "oos_stop": np.clip(stop_model.predict(x_oos.to_numpy(dtype=np.float64)).astype(np.float64), 0.0, 1.0),
        "oos_mae": np.maximum(mae_model.predict(x_oos.to_numpy(dtype=np.float64)).astype(np.float64), 0.0),
        "oos_net_lb": net_model.predict(x_oos.to_numpy(dtype=np.float64)).astype(np.float64) - final_net_offset,
    }
    diag = {
        "side": side,
        "net_cal_q": float(net_cal_q),
        "label_rows": int(len(labels)),
        "stop_rate": float(np.mean(y_stop[idx])) if len(idx) else 0.0,
        "mae_mean": float(np.mean(y_mae[idx])) if len(idx) else 0.0,
        "net_mean": float(np.mean(y_net[idx])) if len(idx) else 0.0,
        "folds": folds_meta,
        "final_net_abs_resid_q": final_net_offset,
    }
    return out, diag


def _base_numeric_actions(
    x_val: pd.DataFrame,
    x_oos: pd.DataFrame,
    path_labels: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    ev_labels, ev_diag = exp._utility_from_path_labels(
        path_labels,
        exp.RISK,
        {"stop_penalty": 0.0, "mae_penalty": 0.0, "time_penalty": 0.0},
    )
    utility_labels, utility_diag = exp._utility_from_path_labels(path_labels, exp.RISK, exp.UTILITY_CFGS[0])
    ev_vl, ev_vs, ev_ol, ev_os, ev_fit = exp._fit_predict_lower_bound(
        x_val,
        x_oos,
        ev_labels,
        "long_net",
        "short_net",
        seed=280000,
        cal_q=0.80,
    )
    u_vl, u_vs, u_ol, u_os, utility_fit = exp._fit_predict_lower_bound(
        x_val,
        x_oos,
        utility_labels,
        "long_utility",
        "short_utility",
        seed=281000,
        cal_q=0.50,
    )
    val_ev_a, val_ev_c = exp._actions_from_scores(ev_vl, ev_vs, 0.003)
    oos_ev_a, oos_ev_c = exp._actions_from_scores(ev_ol, ev_os, 0.003)
    val_a, val_c, val_filter = exp._apply_agreement(
        val_ev_a,
        val_ev_c,
        u_vl,
        u_vs,
        utility_min=-0.001,
        margin_min=0.0,
    )
    oos_a, oos_c, oos_filter = exp._apply_agreement(
        oos_ev_a,
        oos_ev_c,
        u_ol,
        u_os,
        utility_min=-0.001,
        margin_min=0.0,
    )
    diag = {
        "ev_labels": ev_diag,
        "utility_labels": utility_diag,
        "ev_fit": ev_fit,
        "utility_fit": utility_fit,
        "validation_filter": val_filter,
        "oos_filter": oos_filter,
    }
    return val_a, val_c, oos_a, oos_c, diag


def _apply_gate(
    actions: np.ndarray,
    conf: np.ndarray,
    long_risk: dict[str, np.ndarray],
    short_risk: dict[str, np.ndarray],
    split: str,
    variant: GateVariant,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    stop = np.where(actions == sleeve.ACTION_LONG, long_risk[f"{split}_stop"], np.where(actions == sleeve.ACTION_SHORT, short_risk[f"{split}_stop"], 0.0))
    mae = np.where(actions == sleeve.ACTION_LONG, long_risk[f"{split}_mae"], np.where(actions == sleeve.ACTION_SHORT, short_risk[f"{split}_mae"], 0.0))
    net_lb = np.where(actions == sleeve.ACTION_LONG, long_risk[f"{split}_net_lb"], np.where(actions == sleeve.ACTION_SHORT, short_risk[f"{split}_net_lb"], 0.0))
    active = np.isin(actions, [sleeve.ACTION_LONG, sleeve.ACTION_SHORT])
    keep = active & (stop <= float(variant.stop_max)) & (mae <= float(variant.mae_max)) & (net_lb >= float(variant.net_lb_min))
    gated_actions = np.where(keep, actions, sleeve.ACTION_CASH).astype(np.int64)
    gated_conf = np.where(keep, conf, 0.0).astype(np.float64)
    diag = {
        "active_rows_before": int(active.sum()),
        "kept_rows": int(keep.sum()),
        "blocked_rows": int((active & ~keep).sum()),
        "keep_rate": float(keep.sum() / max(active.sum(), 1)),
        "kept_stop_pred_mean": float(np.mean(stop[keep])) if keep.any() else 0.0,
        "blocked_stop_pred_mean": float(np.mean(stop[active & ~keep])) if (active & ~keep).any() else 0.0,
        "kept_mae_pred_mean": float(np.mean(mae[keep])) if keep.any() else 0.0,
        "blocked_mae_pred_mean": float(np.mean(mae[active & ~keep])) if (active & ~keep).any() else 0.0,
        "kept_net_lb_mean": float(np.mean(net_lb[keep])) if keep.any() else 0.0,
        "blocked_net_lb_mean": float(np.mean(net_lb[active & ~keep])) if (active & ~keep).any() else 0.0,
    }
    return gated_actions, gated_conf, diag


def _variants() -> list[GateVariant]:
    return [
        GateVariant("gate_stop040_mae020_netlb000_min20", 0.40, 0.020, 0.000, 20),
        GateVariant("gate_stop035_mae020_netlb000_min20", 0.35, 0.020, 0.000, 20),
        GateVariant("gate_stop030_mae018_netlb000_min20", 0.30, 0.018, 0.000, 20),
        GateVariant("gate_stop025_mae016_netlb000_min15", 0.25, 0.016, 0.000, 15),
        GateVariant("gate_stop020_mae014_netlb000_min10", 0.20, 0.014, 0.000, 10),
        GateVariant("gate_stop035_mae020_netlb001_min20", 0.35, 0.020, 0.001, 20),
        GateVariant("gate_stop030_mae018_netlb001_min15", 0.30, 0.018, 0.001, 15),
        GateVariant("gate_stop025_mae016_netlb001_min10", 0.25, 0.016, 0.001, 10),
        GateVariant("gate_stop040_mae024_netlb000_min25", 0.40, 0.024, 0.000, 25),
        GateVariant("gate_stop045_mae028_netlb000_min25", 0.45, 0.028, 0.000, 25),
    ]


def _row(
    variant: str,
    family: str,
    val_m: dict[str, Any],
    oos_m: dict[str, Any],
    base_val: dict[str, Any],
    base_oos: dict[str, Any],
    diag: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "variant": variant,
        "family": family,
    }
    row.update(sleeve._metric_row("val", val_m))
    row.update(sleeve._metric_row("oos", oos_m))
    row["val_delta_pnl"] = float(row["val_pnl"] - float(base_val["pnl"]))
    row["oos_delta_pnl"] = float(row["oos_pnl"] - float(base_oos["pnl"]))
    row["val_fallback_stop_loss"] = _reason_count(row["val_reasons"], "fallback_stop_loss")
    row["val_fallback_take_profit"] = _reason_count(row["val_reasons"], "fallback_take_profit")
    row["val_fallback_primary_takeover"] = _reason_count(row["val_reasons"], "fallback_primary_takeover")
    row["oos_fallback_stop_loss"] = _reason_count(row["oos_reasons"], "fallback_stop_loss")
    row["oos_fallback_take_profit"] = _reason_count(row["oos_reasons"], "fallback_take_profit")
    row["oos_fallback_primary_takeover"] = _reason_count(row["oos_reasons"], "fallback_primary_takeover")
    row["val_fallback_stop_rate"] = float(row["val_fallback_stop_loss"] / max(int(row["val_fallback_entries"]), 1))
    row["val_mdd_improvement"] = float(row["val_mdd"] - float(base_val["mdd"]))
    row["val_stop_loss_reduction"] = float(_reason_count(base_val.get("exit_reasons", {}), "fallback_stop_loss") - row["val_fallback_stop_loss"])
    row["trade_collapse_penalty"] = max(0.0, 20.0 - float(row["val_fallback_entries"]))
    row["selection_score_val_only"] = (
        row["val_delta_pnl"]
        + 0.25 * row["val_mdd_improvement"]
        - 1.25 * row["val_fallback_stop_loss"]
        - 5.0 * row["val_fallback_stop_rate"]
        - 0.30 * row["trade_collapse_penalty"]
        + 0.02 * row["val_fallback_entries"]
    )
    row["diag"] = diag
    return row


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"stage": "build_payloads", "model_id": MODEL_ID}, ensure_ascii=True), flush=True)
    val_payload, oos_payload, meta = exp._build_payloads()
    x_val = val_payload["features"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_oos = oos_payload["features"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if list(x_val.columns) != list(x_oos.columns):
        raise RuntimeError("validation/oos feature columns mismatch")

    fee = float(meta["fee"])
    slip = float(meta["slip"])
    parent_val = _base_sleeve_metrics(omega._metrics(val_payload["frame"], val_payload["dec"], fee=fee, slip=slip, cost_mult=3.0))
    parent_oos = _base_sleeve_metrics(omega._metrics(oos_payload["frame"], oos_payload["dec"], fee=fee, slip=slip, cost_mult=3.0))

    print(json.dumps({"stage": "counterfactual_path_labels"}, ensure_ascii=True), flush=True)
    path_labels, path_diag = exp._path_label_table(val_payload, exp.RISK)

    print(json.dumps({"stage": "base_numeric_actions"}, ensure_ascii=True), flush=True)
    val_base_a, val_base_c, oos_base_a, oos_base_c, base_numeric_diag = _base_numeric_actions(x_val, x_oos, path_labels)
    base_numeric_val = sleeve._metrics_with_fallback(val_payload["frame"], val_payload["dec"], exp.RISK, val_base_a, val_base_c, 0.0, fee=fee, slip=slip, cost_mult=3.0)
    base_numeric_oos = sleeve._metrics_with_fallback(oos_payload["frame"], oos_payload["dec"], exp.RISK, oos_base_a, oos_base_c, 0.0, fee=fee, slip=slip, cost_mult=3.0)

    print(json.dumps({"stage": "fit_entry_risk_long"}, ensure_ascii=True), flush=True)
    long_risk, long_diag = _fit_predict_entry_risk(x_val, x_oos, path_labels, side="long", seed=282100, net_cal_q=0.65)
    print(json.dumps({"stage": "fit_entry_risk_short"}, ensure_ascii=True), flush=True)
    short_risk, short_diag = _fit_predict_entry_risk(x_val, x_oos, path_labels, side="short", seed=282200, net_cal_q=0.65)

    rows: list[dict[str, Any]] = [
        _row(
            "parent_only_baseline",
            "control_parent",
            parent_val,
            parent_oos,
            parent_val,
            parent_oos,
            {"description": "original parent primary-only metrics"},
        ),
        _row(
            "oof_base_numeric_no_gate",
            "control_omega1_2_8b_numeric",
            base_numeric_val,
            base_numeric_oos,
            base_numeric_val,
            base_numeric_oos,
            base_numeric_diag,
        ),
    ]

    for variant in _variants():
        print(json.dumps({"stage": "eval_gate", "variant": variant.name}, ensure_ascii=True), flush=True)
        val_a, val_c, val_gate_diag = _apply_gate(val_base_a, val_base_c, long_risk, short_risk, "val", variant)
        oos_a, oos_c, oos_gate_diag = _apply_gate(oos_base_a, oos_base_c, long_risk, short_risk, "oos", variant)
        val_m = sleeve._metrics_with_fallback(val_payload["frame"], val_payload["dec"], exp.RISK, val_a, val_c, 0.0, fee=fee, slip=slip, cost_mult=3.0)
        oos_m = sleeve._metrics_with_fallback(oos_payload["frame"], oos_payload["dec"], exp.RISK, oos_a, oos_c, 0.0, fee=fee, slip=slip, cost_mult=3.0)
        row = _row(
            variant.name,
            "counterfactual_entry_risk_gate",
            val_m,
            oos_m,
            base_numeric_val,
            base_numeric_oos,
            {
                "thresholds": {
                    "stop_max": float(variant.stop_max),
                    "mae_max": float(variant.mae_max),
                    "net_lb_min": float(variant.net_lb_min),
                    "min_fallback_entries": int(variant.min_fallback_entries),
                },
                "validation_gate": val_gate_diag,
                "oos_gate": oos_gate_diag,
            },
        )
        if int(row["val_fallback_entries"]) < int(variant.min_fallback_entries):
            row["selection_score_val_only"] = float(row["selection_score_val_only"]) - 10.0
            row["trade_count_blocker"] = True
        else:
            row["trade_count_blocker"] = False
        rows.append(row)

    ranking = pd.DataFrame(rows).sort_values(["selection_score_val_only", "val_delta_pnl", "val_pnl"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "counterfactual_entry_risk_gate_ranking.csv", index=False)
    gate_rows = ranking[ranking["family"].eq("counterfactual_entry_risk_gate")].copy()
    selected = gate_rows.iloc[0].to_dict() if len(gate_rows) else ranking.iloc[0].to_dict()
    best_oos = gate_rows.sort_values(["oos_pnl", "oos_delta_pnl"], ascending=False).iloc[0].to_dict() if len(gate_rows) else ranking.iloc[0].to_dict()

    blockers: list[str] = []
    bad_features = [
        c
        for c in x_val.columns
        if c == "tp_sl_action_score" or c.startswith("clean_regime4_") or c.startswith("regime4_pred_") or c.startswith("teacher_")
    ]
    if bad_features:
        blockers.append(f"forbidden feature columns: {bad_features[:20]}")
    if not len(gate_rows):
        blockers.append("no gate candidates produced")
    if len(gate_rows) and str(selected.get("variant")) == "oof_base_numeric_no_gate":
        blockers.append("gate candidate did not outrank no-gate control")

    report = {
        "model_id": MODEL_ID,
        "status": "redteam_pass_oof_counterfactual_entry_risk_eval" if not blockers else "redteam_fail",
        "method": "Entry risk gate is trained from all validation cash bar x side counterfactual replay labels. Validation gate predictions are chronological OOF; OOS is diagnostic only. TP/SL/notional/leverage stay fixed at omega1.2.8b risk.",
        "risk": exp.RISK.__dict__,
        "selection_policy": "validation_oof_only; OOS diagnostic only; no live export in this experiment",
        "baseline": {
            "parent_only_validation": parent_val,
            "parent_only_oos": parent_oos,
            "oof_base_numeric_validation": base_numeric_val,
            "oof_base_numeric_oos": base_numeric_oos,
        },
        "diagnostics": {
            "parent_artifact": meta["parent_dir"],
            "feature_count": int(x_val.shape[1]),
            "features": list(x_val.columns),
            "path_labels": path_diag,
            "base_numeric": base_numeric_diag,
            "entry_risk_long": long_diag,
            "entry_risk_short": short_diag,
        },
        "selected_by_validation_oof": selected,
        "best_by_oos_diagnostic": best_oos,
        "top20": ranking.head(20).to_dict(orient="records"),
        "redteam_pass": not blockers,
        "redteam_blockers": blockers,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "counterfactual_entry_risk_gate_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(OUT_DIR / "report.json"),
                "status": report["status"],
                "selected_by_validation_oof": selected,
                "best_by_oos_diagnostic": best_oos,
            },
            indent=2,
            ensure_ascii=True,
            default=_json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
