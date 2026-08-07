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


MODEL_ID = "omega1_2_8t_pairwise_bad_entry_veto_20260617"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


@dataclass(frozen=True)
class VetoVariant:
    name: str
    bad_prob_max: float
    expected_delta_min: float
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
        max_iter=120,
        learning_rate=0.035,
        max_leaf_nodes=7,
        l2_regularization=3.0,
        random_state=int(seed),
    )


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
    return val_a, val_c, oos_a, oos_c, {
        "ev_labels": ev_diag,
        "utility_labels": utility_diag,
        "ev_fit": ev_fit,
        "utility_fit": utility_fit,
        "validation_filter": val_filter,
        "oos_filter": oos_filter,
    }


def _candidate_training_table(
    x_val: pd.DataFrame,
    path_labels: pd.DataFrame,
    actions: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    label_by_i = path_labels.set_index("i", drop=False)
    rows: list[dict[str, Any]] = []
    active_i = np.flatnonzero(np.isin(actions, [sleeve.ACTION_LONG, sleeve.ACTION_SHORT]))
    for i in active_i:
        if int(i) not in label_by_i.index:
            continue
        side = "long" if int(actions[i]) == sleeve.ACTION_LONG else "short"
        row = label_by_i.loc[int(i)]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        net = float(row[f"{side}_net"])
        mae = float(max(-float(row[f"{side}_mae"]), 0.0))
        stop = int(row[f"{side}_stop"])
        reason = str(row[f"{side}_reason"])
        bad = int(stop == 1 or net < 0.0 or mae >= 0.018)
        rec: dict[str, Any] = {
            "i": int(i),
            "side": 1 if side == "long" else -1,
            "net": net,
            "mae": mae,
            "stop": stop,
            "bad": bad,
            "reason": reason,
        }
        for col, val in x_val.iloc[int(i)].items():
            rec[str(col)] = float(val)
        rows.append(rec)
    table = pd.DataFrame(rows)
    diag = {
        "candidate_rows": int(len(table)),
        "bad_rate": float(table["bad"].mean()) if len(table) else 0.0,
        "stop_rate": float(table["stop"].mean()) if len(table) else 0.0,
        "net_mean": float(table["net"].mean()) if len(table) else 0.0,
        "mae_mean": float(table["mae"].mean()) if len(table) else 0.0,
        "reason_counts": table["reason"].value_counts().sort_index().to_dict() if len(table) else {},
    }
    return table, diag


def _make_candidate_features(x: pd.DataFrame, actions: np.ndarray, feature_cols: list[str]) -> pd.DataFrame:
    active = np.flatnonzero(np.isin(actions, [sleeve.ACTION_LONG, sleeve.ACTION_SHORT]))
    rows = x.iloc[active].copy()
    rows["side"] = np.where(actions[active] == sleeve.ACTION_LONG, 1.0, -1.0)
    return rows[feature_cols], active


def _fit_predict_bad_veto(
    x_val: pd.DataFrame,
    x_oos: pd.DataFrame,
    path_labels: pd.DataFrame,
    val_actions: np.ndarray,
    oos_actions: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    train_table, table_diag = _candidate_training_table(x_val, path_labels, val_actions)
    if len(train_table) < 50:
        raise RuntimeError(f"not enough active validation candidates for pairwise bad-entry training: {len(train_table)}")
    feature_cols = list(x_val.columns) + ["side"]
    idx = train_table["i"].to_numpy(dtype=np.int64)
    y_bad = train_table["bad"].to_numpy(dtype=np.float64)
    y_delta = train_table["net"].to_numpy(dtype=np.float64)

    val_bad_prob = np.zeros(len(x_val), dtype=np.float64)
    val_expected_delta = np.zeros(len(x_val), dtype=np.float64)
    folds_meta: list[dict[str, Any]] = []
    full_train_index = pd.Index(train_table["i"].astype(int))
    for fold_id, (tr_idx, va_idx) in enumerate(exp._chron_folds(idx)):
        tr_set = set(int(i) for i in tr_idx.tolist())
        va_set = set(int(i) for i in va_idx.tolist())
        tr_mask = full_train_index.isin(tr_set)
        va_mask = full_train_index.isin(va_set)
        if int(tr_mask.sum()) < 30 or int(va_mask.sum()) == 0:
            continue
        x_tr = train_table.loc[tr_mask, feature_cols].to_numpy(dtype=np.float64)
        x_va = train_table.loc[va_mask, feature_cols].to_numpy(dtype=np.float64)
        yb_tr = y_bad[tr_mask]
        yd_tr = y_delta[tr_mask]
        bad_model = _fit_regressor(283000 + fold_id * 100 + 1)
        delta_model = _fit_regressor(283000 + fold_id * 100 + 2)
        bad_model.fit(x_tr, yb_tr)
        delta_model.fit(x_tr, yd_tr)
        va_i = train_table.loc[va_mask, "i"].to_numpy(dtype=np.int64)
        val_bad_prob[va_i] = np.clip(bad_model.predict(x_va).astype(np.float64), 0.0, 1.0)
        val_expected_delta[va_i] = delta_model.predict(x_va).astype(np.float64)
        folds_meta.append(
            {
                "fold": int(fold_id),
                "train_candidates": int(tr_mask.sum()),
                "val_candidates": int(va_mask.sum()),
                "train_bad_rate": float(np.mean(yb_tr)),
                "train_net_mean": float(np.mean(yd_tr)),
            }
        )

    bad_model = _fit_regressor(283901)
    delta_model = _fit_regressor(283902)
    x_train = train_table[feature_cols].to_numpy(dtype=np.float64)
    bad_model.fit(x_train, y_bad)
    delta_model.fit(x_train, y_delta)
    x_oos_active, oos_active = _make_candidate_features(x_oos, oos_actions, feature_cols)
    oos_bad_prob = np.zeros(len(x_oos), dtype=np.float64)
    oos_expected_delta = np.zeros(len(x_oos), dtype=np.float64)
    if len(oos_active):
        oos_bad_prob[oos_active] = np.clip(bad_model.predict(x_oos_active.to_numpy(dtype=np.float64)).astype(np.float64), 0.0, 1.0)
        oos_expected_delta[oos_active] = delta_model.predict(x_oos_active.to_numpy(dtype=np.float64)).astype(np.float64)
    return {
        "val_bad_prob": val_bad_prob,
        "val_expected_delta": val_expected_delta,
        "oos_bad_prob": oos_bad_prob,
        "oos_expected_delta": oos_expected_delta,
    }, {
        "candidate_table": table_diag,
        "feature_count": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "folds": folds_meta,
    }


def _apply_veto(
    actions: np.ndarray,
    conf: np.ndarray,
    pred: dict[str, np.ndarray],
    split: str,
    variant: VetoVariant,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    active = np.isin(actions, [sleeve.ACTION_LONG, sleeve.ACTION_SHORT])
    bad_prob = pred[f"{split}_bad_prob"]
    expected_delta = pred[f"{split}_expected_delta"]
    keep = active & (bad_prob <= float(variant.bad_prob_max)) & (expected_delta >= float(variant.expected_delta_min))
    out_a = np.where(keep, actions, sleeve.ACTION_CASH).astype(np.int64)
    out_c = np.where(keep, conf, 0.0).astype(np.float64)
    diag = {
        "active_rows_before": int(active.sum()),
        "kept_rows": int(keep.sum()),
        "blocked_rows": int((active & ~keep).sum()),
        "keep_rate": float(keep.sum() / max(active.sum(), 1)),
        "kept_bad_prob_mean": float(np.mean(bad_prob[keep])) if keep.any() else 0.0,
        "blocked_bad_prob_mean": float(np.mean(bad_prob[active & ~keep])) if (active & ~keep).any() else 0.0,
        "kept_expected_delta_mean": float(np.mean(expected_delta[keep])) if keep.any() else 0.0,
        "blocked_expected_delta_mean": float(np.mean(expected_delta[active & ~keep])) if (active & ~keep).any() else 0.0,
    }
    return out_a, out_c, diag


def _variants() -> list[VetoVariant]:
    return [
        VetoVariant("badprob070_delta000_min6", 0.70, 0.000, 6),
        VetoVariant("badprob060_delta000_min6", 0.60, 0.000, 6),
        VetoVariant("badprob050_delta000_min5", 0.50, 0.000, 5),
        VetoVariant("badprob040_delta000_min4", 0.40, 0.000, 4),
        VetoVariant("badprob060_delta001_min5", 0.60, 0.001, 5),
        VetoVariant("badprob050_delta001_min4", 0.50, 0.001, 4),
        VetoVariant("badprob040_delta001_min3", 0.40, 0.001, 3),
        VetoVariant("badprob080_delta000_min8", 0.80, 0.000, 8),
        VetoVariant("badprob090_delta000_min8", 0.90, 0.000, 8),
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
    row = {"variant": variant, "family": family}
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
    row["trade_collapse_penalty"] = max(0.0, 5.0 - float(row["val_fallback_entries"]))
    row["selection_score_val_only"] = (
        row["val_delta_pnl"]
        + 0.25 * row["val_mdd_improvement"]
        - 1.50 * row["val_fallback_stop_loss"]
        - 4.0 * row["val_fallback_stop_rate"]
        - 0.80 * row["trade_collapse_penalty"]
        + 0.03 * row["val_fallback_entries"]
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

    print(json.dumps({"stage": "fit_pairwise_bad_entry_veto"}, ensure_ascii=True), flush=True)
    pred, veto_diag = _fit_predict_bad_veto(x_val, x_oos, path_labels, val_base_a, oos_base_a)

    rows = [
        _row("parent_only_baseline", "control_parent", parent_val, parent_oos, parent_val, parent_oos, {}),
        _row("oof_base_numeric_no_veto", "control_omega1_2_8b_numeric", base_numeric_val, base_numeric_oos, base_numeric_val, base_numeric_oos, base_numeric_diag),
    ]
    for variant in _variants():
        print(json.dumps({"stage": "eval_veto", "variant": variant.name}, ensure_ascii=True), flush=True)
        val_a, val_c, val_diag = _apply_veto(val_base_a, val_base_c, pred, "val", variant)
        oos_a, oos_c, oos_diag = _apply_veto(oos_base_a, oos_base_c, pred, "oos", variant)
        val_m = sleeve._metrics_with_fallback(val_payload["frame"], val_payload["dec"], exp.RISK, val_a, val_c, 0.0, fee=fee, slip=slip, cost_mult=3.0)
        oos_m = sleeve._metrics_with_fallback(oos_payload["frame"], oos_payload["dec"], exp.RISK, oos_a, oos_c, 0.0, fee=fee, slip=slip, cost_mult=3.0)
        row = _row(
            variant.name,
            "pairwise_bad_entry_veto",
            val_m,
            oos_m,
            base_numeric_val,
            base_numeric_oos,
            {"thresholds": variant.__dict__, "validation_veto": val_diag, "oos_veto": oos_diag},
        )
        if int(row["val_fallback_entries"]) < int(variant.min_fallback_entries):
            row["selection_score_val_only"] = float(row["selection_score_val_only"]) - 10.0
            row["trade_count_blocker"] = True
        else:
            row["trade_count_blocker"] = False
        rows.append(row)

    ranking = pd.DataFrame(rows).sort_values(["selection_score_val_only", "val_delta_pnl", "val_pnl"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "pairwise_bad_entry_veto_ranking.csv", index=False)
    veto_rows = ranking[ranking["family"].eq("pairwise_bad_entry_veto")].copy()
    selected = veto_rows.iloc[0].to_dict() if len(veto_rows) else ranking.iloc[0].to_dict()
    best_oos = veto_rows.sort_values(["oos_pnl", "oos_delta_pnl"], ascending=False).iloc[0].to_dict() if len(veto_rows) else ranking.iloc[0].to_dict()

    blockers: list[str] = []
    bad_features = [
        c
        for c in x_val.columns
        if c == "tp_sl_action_score" or c.startswith("clean_regime4_") or c.startswith("regime4_pred_") or c.startswith("teacher_")
    ]
    if bad_features:
        blockers.append(f"forbidden feature columns: {bad_features[:20]}")
    if not len(veto_rows):
        blockers.append("no veto candidates produced")
    if len(veto_rows) and float(selected.get("val_delta_pnl", 0.0)) <= 0.0 and int(selected.get("val_fallback_stop_loss", 0)) >= _reason_count(base_numeric_val.get("exit_reasons", {}), "fallback_stop_loss"):
        blockers.append("best veto did not improve validation pnl or fallback stop-loss count versus no-veto control")

    report = {
        "model_id": MODEL_ID,
        "status": "redteam_pass_oof_pairwise_bad_entry_veto_eval" if not blockers else "redteam_fail",
        "method": "Baseline-selected entry candidates are labeled as bad/good using counterfactual selected-side replay. Validation predictions are chronological OOF candidate-only veto predictions. TP/SL/notional/leverage remain fixed.",
        "risk": exp.RISK.__dict__,
        "selection_policy": "validation_oof_only; OOS diagnostic only; no live export",
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
            "pairwise_veto": veto_diag,
        },
        "selected_by_validation_oof": selected,
        "best_by_oos_diagnostic": best_oos,
        "top20": ranking.head(20).to_dict(orient="records"),
        "redteam_pass": not blockers,
        "redteam_blockers": blockers,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "pairwise_bad_entry_veto_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "status": report["status"], "selected": selected, "best_oos": best_oos}, indent=2, ensure_ascii=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
