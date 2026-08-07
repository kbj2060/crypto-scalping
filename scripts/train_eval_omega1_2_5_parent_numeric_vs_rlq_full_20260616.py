#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_5_parent_numeric_vs_rlq_20260616 as probe  # noqa: E402


MODEL_ID = "omega1_2_5_parent_numeric_vs_rlq_full_20260616"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


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


def _regressor(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=260,
        learning_rate=0.03,
        max_leaf_nodes=15,
        l2_regularization=2.0,
        random_state=int(seed),
    )


def _fit_oof_and_oos(
    x_val: pd.DataFrame,
    labels: pd.DataFrame,
    target: str,
    x_oos: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    idx = labels["i"].to_numpy(dtype=np.int64)
    y = labels[target].to_numpy(dtype=np.float64)
    order = np.argsort(idx)
    idx = idx[order]
    y = y[order]
    n = len(idx)
    val_pred = np.full(len(x_val), np.nan, dtype=np.float64)
    folds: list[dict[str, Any]] = []
    for fold_id, (train_frac, end_frac) in enumerate(((0.35, 0.50), (0.50, 0.65), (0.65, 0.80), (0.80, 1.00))):
        train_end = int(n * train_frac)
        val_end = int(n * end_frac)
        if train_end < 500 or val_end <= train_end:
            folds.append({"fold": fold_id, "skipped": "insufficient_rows", "train_rows": train_end, "val_rows": max(0, val_end - train_end)})
            continue
        train_pos = np.arange(0, train_end, dtype=np.int64)
        val_pos = np.arange(train_end, val_end, dtype=np.int64)
        model = _regressor(seed + fold_id)
        model.fit(x_val.iloc[idx[train_pos]].to_numpy(dtype=np.float64), y[train_pos])
        pred = model.predict(x_val.iloc[idx[val_pos]].to_numpy(dtype=np.float64)).astype(np.float64)
        val_pred[idx[val_pos]] = pred
        folds.append(
            {
                "fold": fold_id,
                "train_rows": int(len(train_pos)),
                "val_rows": int(len(val_pos)),
                "target_mean_train": float(np.mean(y[train_pos])),
                "target_mean_val": float(np.mean(y[val_pos])),
                "pred_mean_val": float(np.mean(pred)),
            }
        )
    final_model = _regressor(seed + 100)
    final_model.fit(x_val.iloc[idx].to_numpy(dtype=np.float64), y)
    oos_pred = final_model.predict(x_oos.to_numpy(dtype=np.float64)).astype(np.float64)
    return val_pred, oos_pred, folds


def _actions_from_oof_scores(long_s: np.ndarray, short_s: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(long_s) & np.isfinite(short_s)
    long_clean = np.where(valid, long_s, -np.inf)
    short_clean = np.where(valid, short_s, -np.inf)
    action, conf = probe._actions_from_scores(long_clean, short_clean, threshold)
    action = np.where(valid, action, probe.ACTION_CASH).astype(np.int64)
    conf = np.where(valid, conf, 0.0).astype(np.float64)
    return action, conf


def _eval_parent(
    name: str,
    data: dict[str, Any],
    cfg: Any,
    risk: Any,
    val_action: np.ndarray,
    val_conf: np.ndarray,
    oos_action: np.ndarray,
    oos_conf: np.ndarray,
    base_val: dict[str, Any],
    base_oos: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    val_dec = probe._decisions_from_actions(data["validation"], val_action, val_conf, risk)
    oos_dec = probe._decisions_from_actions(data["oos"], oos_action, oos_conf, risk)
    val_payload = probe._with_dec(data["validation"], val_dec)
    oos_payload = probe._with_dec(data["oos"], oos_dec)
    val_m, val_ledger = probe.base._simulate_combo(val_payload, cfg, None, None, None, 1.0)
    oos_m, oos_ledger = probe.base._simulate_combo(oos_payload, cfg, None, None, None, 1.0)
    row = probe._row(name, "parent_only_oof_validation_final_refit_oos", val_m, val_ledger, oos_m, oos_ledger, base_val, base_oos)
    return row, val_ledger, oos_ledger


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = probe._runner_cfg()
    risk = [r for r in probe.base.RISKS if r.name == probe.RISK_NAME][0]
    data = probe.base.legacy_runner._build()
    x_val = probe._parent_features(data["validation"])
    x_oos = probe._parent_features(data["oos"])
    if list(x_val.columns) != list(x_oos.columns):
        raise RuntimeError("parent feature columns mismatch")

    base_val, base_val_ledger = probe.base._simulate_combo(data["validation"], cfg, None, None, None, 1.0)
    base_oos, base_oos_ledger = probe.base._simulate_combo(data["oos"], cfg, None, None, None, 1.0)
    base_val_ledger.to_csv(OUT_DIR / "baseline_validation_ledger.csv", index=False)
    base_oos_ledger.to_csv(OUT_DIR / "baseline_oos_ledger.csv", index=False)

    diagnostics: dict[str, Any] = {
        "mode": "full_oof_validation_final_refit_oos_no_subsample",
        "validation_rows": int(len(x_val)),
        "oos_rows": int(len(x_oos)),
        "feature_count": int(x_val.shape[1]),
        "features": list(x_val.columns),
    }
    rows: list[dict[str, Any]] = []
    ledgers: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    util_labels, util_diag = probe._utility_labels_all(data["validation"], risk)
    diagnostics["utility_labels"] = util_diag
    ulv, uoo, folds = _fit_oof_and_oos(x_val, util_labels, "long_utility", x_oos, 266001)
    usv, uso, folds_s = _fit_oof_and_oos(x_val, util_labels, "short_utility", x_oos, 266101)
    diagnostics["utility_oof_folds_long"] = folds
    diagnostics["utility_oof_folds_short"] = folds_s
    for thr in (0.000, 0.001, 0.002):
        va, vc = _actions_from_oof_scores(ulv, usv, thr)
        oa, oc = probe._actions_from_scores(uoo, uso, thr)
        row, vl, ol = _eval_parent(f"utility_thr{thr:.3f}_full_oof", data, cfg, risk, va, vc, oa, oc, base_val, base_oos)
        rows.append(row)
        ledgers[row["candidate"]] = (vl, ol)

    critic, router, dsac_meta = probe._load_dsac_critic()
    rlq_labels, rlq_diag = probe._rlq_labels(data["validation"], critic, router)
    diagnostics["rlq_source"] = dsac_meta
    diagnostics["rlq_labels"] = rlq_diag
    rlv, roo, folds = _fit_oof_and_oos(x_val, rlq_labels, "long_adv", x_oos, 266201)
    rsv, rso, folds_s = _fit_oof_and_oos(x_val, rlq_labels, "short_adv", x_oos, 266301)
    diagnostics["rlq_oof_folds_long"] = folds
    diagnostics["rlq_oof_folds_short"] = folds_s
    positives = np.r_[roo[roo > 0], rso[rso > 0]]
    q_thresholds = [0.0]
    if len(positives):
        q_thresholds.extend([float(np.quantile(positives, q)) for q in (0.25, 0.50)])
    for thr in q_thresholds:
        va, vc = _actions_from_oof_scores(rlv, rsv, thr)
        oa, oc = probe._actions_from_scores(roo, rso, thr)
        row, vl, ol = _eval_parent(f"rlq_thr{thr:.6f}_full_oof", data, cfg, risk, va, vc, oa, oc, base_val, base_oos)
        rows.append(row)
        ledgers[row["candidate"]] = (vl, ol)

    ranking = pd.DataFrame(rows)
    ranking["selection_score_val_only"] = ranking["val_delta_pnl"].fillna(0.0) - 0.25 * ranking["val_mdd"].abs().fillna(0.0)
    ranking = ranking.sort_values(["selection_score_val_only", "val_delta_pnl", "val_pnl"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "parent_numeric_vs_rlq_full_oof_ranking.csv", index=False)
    selected = ranking.iloc[0].to_dict()
    best_oos = ranking.sort_values(["oos_delta_pnl", "oos_pnl"], ascending=False).iloc[0].to_dict()
    for prefix, row in (("selected", selected), ("best_oos_diagnostic", best_oos)):
        cand = str(row["candidate"])
        if cand in ledgers:
            v, o = ledgers[cand]
            v.to_csv(OUT_DIR / f"{prefix}_validation_ledger.csv", index=False)
            o.to_csv(OUT_DIR / f"{prefix}_oos_ledger.csv", index=False)

    blockers: list[str] = []
    forbidden = [c for c in x_val.columns if c in probe.base.FORBIDDEN_FEATURE_EXACT or c.startswith(probe.base.FORBIDDEN_FEATURE_PREFIXES)]
    if forbidden:
        blockers.append(f"forbidden parent feature columns: {forbidden[:20]}")
    if not np.isfinite(ranking["oos_pnl"].to_numpy(dtype=np.float64)).all():
        blockers.append("non-finite OOS PnL in ranking")

    report = {
        "model_id": MODEL_ID,
        "status": "redteam_pass_full_eval" if not blockers else "redteam_fail",
        "baseline_model_id": probe.base.BASELINE_ID,
        "method": "Full no-subsample parent relabel evaluation: expanding OOF validation, final refit on all validation labels, OOS diagnostic test.",
        "selection_policy": "validation_oof_only_no_oos_selection; OOS is diagnostic",
        "baseline": {"validation": base_val, "oos": base_oos},
        "diagnostics": diagnostics,
        "selected_by_validation_oof": selected,
        "best_by_oos_diagnostic": best_oos,
        "ranking": ranking.to_dict(orient="records"),
        "redteam_pass": not blockers,
        "redteam_blockers": blockers,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "parent_numeric_vs_rlq_full_oof_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "status": report["status"], "selected": selected, "best_oos_diagnostic": best_oos}, indent=2, ensure_ascii=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
