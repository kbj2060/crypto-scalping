#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_3_cash_sleeve_upgrade_20260615 as sleeve_up  # noqa: E402
import train_eval_omega1_2_5_parent_numeric_vs_rlq_20260616 as rlq_probe  # noqa: E402


MODEL_ID = "omega1_2_5_rlq_cash_sleeve_full_20260616"
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


def _runner_cfg() -> Any:
    report = json.loads(rlq_probe.base.BASELINE_REPORT.read_text(encoding="utf-8"))
    sc = report["selected_config"]
    return rlq_probe.base.repair.RunnerConfig(
        int(sc["candidate_id"]),
        str(sc["mode"]),
        float(sc["quality_min"]),
        float(sc["extend_mult"]),
        float(sc["floor_frac"]),
        int(sc["max_extensions"]),
    )


def _regressor(kind: str, seed: int) -> Any:
    if kind == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.03,
            max_leaf_nodes=13,
            l2_regularization=2.0,
            random_state=int(seed),
        )
    if kind == "extra":
        return ExtraTreesRegressor(
            n_estimators=480,
            max_depth=8,
            min_samples_leaf=24,
            random_state=int(seed),
            n_jobs=-1,
        )
    raise RuntimeError(f"unknown regressor kind: {kind}")


def _chron_folds(idx: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    n = len(idx)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_frac, end_frac in ((0.35, 0.50), (0.50, 0.65), (0.65, 0.80), (0.80, 1.00)):
        train_end = int(n * train_frac)
        val_end = int(n * end_frac)
        if train_end >= 500 and val_end > train_end:
            folds.append((idx[:train_end], idx[train_end:val_end]))
    return folds


def _fit_rlq_regressor(
    kind: str,
    x_val: pd.DataFrame,
    target: np.ndarray,
    idx: np.ndarray,
    x_oos: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    val_pred = np.full(len(x_val), np.nan, dtype=np.float64)
    folds_meta: list[dict[str, Any]] = []
    for fold_id, (tr, va) in enumerate(_chron_folds(idx)):
        model = _regressor(kind, seed + fold_id)
        model.fit(x_val.iloc[tr].to_numpy(dtype=np.float64), target[tr])
        pred = model.predict(x_val.iloc[va].to_numpy(dtype=np.float64)).astype(np.float64)
        val_pred[va] = pred
        folds_meta.append(
            {
                "fold": int(fold_id),
                "train_rows": int(len(tr)),
                "val_rows": int(len(va)),
                "target_mean_train": float(np.mean(target[tr])),
                "target_mean_val": float(np.mean(target[va])),
                "pred_mean_val": float(np.mean(pred)),
            }
        )
    final_model = _regressor(kind, seed + 100)
    final_model.fit(x_val.iloc[idx].to_numpy(dtype=np.float64), target[idx])
    oos_pred = final_model.predict(x_oos.to_numpy(dtype=np.float64)).astype(np.float64)
    return val_pred, oos_pred, {"folds": folds_meta}


def _actions_from_rlq(long_adv: np.ndarray, short_adv: np.ndarray, q_min: float) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(long_adv) & np.isfinite(short_adv)
    long_clean = np.where(valid, long_adv, -np.inf)
    short_clean = np.where(valid, short_adv, -np.inf)
    best_long = long_clean >= short_clean
    best = np.where(best_long, long_clean, short_clean)
    action = np.where(best > float(q_min), np.where(best_long, rlq_probe.ACTION_LONG, rlq_probe.ACTION_SHORT), rlq_probe.ACTION_CASH).astype(np.int64)
    conf = np.where(valid, np.clip((best - float(q_min)) / 0.02, 0.0, 1.0), 0.0).astype(np.float64)
    return action, conf


def _row(candidate: str, family: str, risk: Any, q_min: float, val_m: dict[str, Any], val_ledger: pd.DataFrame, oos_m: dict[str, Any], oos_ledger: pd.DataFrame, base_val: dict[str, Any], base_oos: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "candidate": candidate,
        "family": family,
        "risk": risk.name,
        "q_min": float(q_min),
    }
    row.update(sleeve_up._row("val", val_m, val_ledger))
    row.update(sleeve_up._row("oos", oos_m, oos_ledger))
    row["val_delta_pnl"] = float(row["val_pnl"] - float(base_val["pnl"]))
    row["oos_delta_pnl"] = float(row["oos_pnl"] - float(base_oos["pnl"]))
    return row


def _cash_train_idx(payload: dict[str, Any], risk: Any) -> np.ndarray:
    dec = payload["dec"].reset_index(drop=True)
    active = rlq_probe.base._active(dec)
    max_i = len(dec) - int(risk.max_hold_bars) - 3
    return np.flatnonzero((~active) & (np.arange(len(dec)) < max_i)).astype(np.int64)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _runner_cfg()
    data = rlq_probe.base.legacy_runner._build()
    x_val = sleeve_up._enhanced_features(data["validation"])
    x_oos = sleeve_up._enhanced_features(data["oos"])
    if list(x_val.columns) != list(x_oos.columns):
        raise RuntimeError("validation/oos sleeve feature columns mismatch")

    base_val, base_val_ledger = rlq_probe.base._simulate_combo(data["validation"], cfg, None, None, None, 1.0)
    base_oos, base_oos_ledger = rlq_probe.base._simulate_combo(data["oos"], cfg, None, None, None, 1.0)
    base_val_ledger.to_csv(OUT_DIR / "baseline_validation_ledger.csv", index=False)
    base_oos_ledger.to_csv(OUT_DIR / "baseline_oos_ledger.csv", index=False)

    critic, router, dsac_meta = rlq_probe._load_dsac_critic()
    rlq_labels, rlq_diag = rlq_probe._rlq_labels(data["validation"], critic, router)
    long_target = np.full(len(x_val), np.nan, dtype=np.float64)
    short_target = np.full(len(x_val), np.nan, dtype=np.float64)
    label_idx = rlq_labels["i"].to_numpy(dtype=np.int64)
    long_target[label_idx] = rlq_labels["long_adv"].to_numpy(dtype=np.float64)
    short_target[label_idx] = rlq_labels["short_adv"].to_numpy(dtype=np.float64)

    diagnostics: dict[str, Any] = {
        "mode": "cash_sleeve_only_rlq_full_validation_oof_final_refit_oos",
        "baseline_model_id": rlq_probe.base.BASELINE_ID,
        "validation_rows": int(len(x_val)),
        "oos_rows": int(len(x_oos)),
        "feature_count": int(x_val.shape[1]),
        "features": list(x_val.columns),
        "rlq_source": dsac_meta,
        "rlq_labels": rlq_diag,
        "baseline": {"validation": base_val, "oos": base_oos},
    }
    rows: list[dict[str, Any]] = [
        {
            "candidate": "baseline_tp_runner_clean_repair",
            "family": "baseline",
            "risk": "none",
            "q_min": 0.0,
            **sleeve_up._row("val", base_val, base_val_ledger),
            **sleeve_up._row("oos", base_oos, base_oos_ledger),
            "val_delta_pnl": 0.0,
            "oos_delta_pnl": 0.0,
        }
    ]
    ledgers: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for risk in rlq_probe.base.RISKS:
        idx = _cash_train_idx(data["validation"], risk)
        idx = idx[np.isfinite(long_target[idx]) & np.isfinite(short_target[idx])]
        diagnostics[f"{risk.name}_cash_rows"] = int(len(idx))
        if len(idx) < 500:
            continue
        preds: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for kind in ("hgb", "extra"):
            vl, ol, diag_l = _fit_rlq_regressor(kind, x_val, long_target, idx, x_oos, seed=267101)
            vs, os, diag_s = _fit_rlq_regressor(kind, x_val, short_target, idx, x_oos, seed=267501)
            diagnostics[f"{risk.name}_{kind}_long"] = diag_l
            diagnostics[f"{risk.name}_{kind}_short"] = diag_s
            preds[kind] = (vl, vs, ol, os)

        threshold_values = [0.0, 0.0005, 0.0010, 0.0020, 0.0030]
        pos = np.r_[long_target[idx][long_target[idx] > 0.0], short_target[idx][short_target[idx] > 0.0]]
        if len(pos):
            threshold_values.extend(float(np.quantile(pos, q)) for q in (0.25, 0.50, 0.75))
        threshold_values = sorted(set(round(float(v), 8) for v in threshold_values))

        for kind, (vl, vs, ol, os) in preds.items():
            for q_min in threshold_values:
                val_a, val_c = _actions_from_rlq(vl, vs, q_min)
                oos_a, oos_c = _actions_from_rlq(ol, os, q_min)
                val_m, val_ledger = rlq_probe.base._simulate_combo(data["validation"], cfg, risk, val_a, val_c, 0.0)
                oos_m, oos_ledger = rlq_probe.base._simulate_combo(data["oos"], cfg, risk, oos_a, oos_c, 0.0)
                name = f"{risk.name}_rlq_{kind}_qmin{q_min:.8f}"
                rows.append(_row(name, f"rlq_{kind}", risk, q_min, val_m, val_ledger, oos_m, oos_ledger, base_val, base_oos))
                ledgers[name] = (val_ledger, oos_ledger)

        if "hgb" in preds and "extra" in preds:
            hv_l, hv_s, ho_l, ho_s = preds["hgb"]
            ev_l, ev_s, eo_l, eo_s = preds["extra"]
            for q_min in threshold_values:
                hv_a, hv_c = _actions_from_rlq(hv_l, hv_s, q_min)
                ev_a, ev_c = _actions_from_rlq(ev_l, ev_s, q_min)
                ho_a, ho_c = _actions_from_rlq(ho_l, ho_s, q_min)
                eo_a, eo_c = _actions_from_rlq(eo_l, eo_s, q_min)
                val_agree = (hv_a == ev_a) & np.isin(hv_a, [rlq_probe.ACTION_LONG, rlq_probe.ACTION_SHORT])
                oos_agree = (ho_a == eo_a) & np.isin(ho_a, [rlq_probe.ACTION_LONG, rlq_probe.ACTION_SHORT])
                val_a = np.where(val_agree, hv_a, rlq_probe.ACTION_CASH).astype(np.int64)
                oos_a = np.where(oos_agree, ho_a, rlq_probe.ACTION_CASH).astype(np.int64)
                val_c = np.where(val_agree, np.minimum(hv_c, ev_c), 0.0).astype(np.float64)
                oos_c = np.where(oos_agree, np.minimum(ho_c, eo_c), 0.0).astype(np.float64)
                val_m, val_ledger = rlq_probe.base._simulate_combo(data["validation"], cfg, risk, val_a, val_c, 0.0)
                oos_m, oos_ledger = rlq_probe.base._simulate_combo(data["oos"], cfg, risk, oos_a, oos_c, 0.0)
                name = f"{risk.name}_rlq_agree_qmin{q_min:.8f}"
                rows.append(_row(name, "rlq_agreement", risk, q_min, val_m, val_ledger, oos_m, oos_ledger, base_val, base_oos))
                ledgers[name] = (val_ledger, oos_ledger)

    ranking = pd.DataFrame(rows)
    ranking["selection_score_val_only"] = (
        ranking["val_fallback_pnl"].fillna(0.0)
        - 35.0 * ranking["val_fallback_stop_rate"].fillna(0.0)
        + 0.20 * ranking["val_fallback_trades"].fillna(0.0)
        + 0.25 * ranking["val_delta_pnl"].fillna(0.0)
    )
    ranking = ranking.sort_values(["selection_score_val_only", "val_fallback_pnl", "val_delta_pnl"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "rlq_cash_sleeve_full_ranking.csv", index=False)
    selected = ranking.iloc[0].to_dict()
    best_oos = ranking.sort_values(["oos_fallback_pnl", "oos_delta_pnl", "oos_fallback_trades"], ascending=False).iloc[0].to_dict()

    for prefix, row in (("selected", selected), ("best_oos_diagnostic", best_oos)):
        candidate = str(row["candidate"])
        if candidate in ledgers:
            val_ledger, oos_ledger = ledgers[candidate]
            val_ledger.to_csv(OUT_DIR / f"{prefix}_validation_ledger.csv", index=False)
            oos_ledger.to_csv(OUT_DIR / f"{prefix}_oos_ledger.csv", index=False)
            val_ledger[val_ledger["sleeve"] == "fallback"].to_csv(OUT_DIR / f"{prefix}_validation_fallback_only_ledger.csv", index=False)
            oos_ledger[oos_ledger["sleeve"] == "fallback"].to_csv(OUT_DIR / f"{prefix}_oos_fallback_only_ledger.csv", index=False)

    redteam_blockers: list[str] = []
    forbidden = [c for c in x_val.columns if c in rlq_probe.base.FORBIDDEN_FEATURE_EXACT or c.startswith(rlq_probe.base.FORBIDDEN_FEATURE_PREFIXES)]
    if forbidden:
        redteam_blockers.append(f"forbidden sleeve feature columns: {forbidden[:20]}")
    if len(x_val) != len(data["validation"]["dec"]):
        redteam_blockers.append("validation feature/decision row count mismatch")
    if len(x_oos) != len(data["oos"]["dec"]):
        redteam_blockers.append("oos feature/decision row count mismatch")
    if len(ranking) <= 1:
        redteam_blockers.append("no RLQ cash sleeve candidates produced")
    if str(selected["candidate"]) == "baseline_tp_runner_clean_repair":
        redteam_blockers.append("validation selection returned baseline instead of RLQ sleeve candidate")

    report = {
        "model_id": MODEL_ID,
        "status": "redteam_pass_full_eval" if not redteam_blockers else "redteam_fail",
        "method": "Train only the CASH fallback sleeve on DSAC critic Q-value advantage labels. Parent Omega TP-runner decisions are unchanged.",
        "selection_policy": "validation_oof_only_no_oos_selection; OOS is diagnostic",
        "redteam_policy": "PnL is diagnostic. FAIL is limited to feature/data/artifact contract defects or no candidate generation.",
        "baseline": {"validation": base_val, "oos": base_oos},
        "diagnostics": diagnostics,
        "selected_by_validation": selected,
        "best_by_oos_diagnostic": best_oos,
        "top20": ranking.head(20).to_dict(orient="records"),
        "redteam_pass": not redteam_blockers,
        "redteam_blockers": redteam_blockers,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "rlq_cash_sleeve_full_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "status": report["status"], "selected": selected, "best_oos_diagnostic": best_oos}, indent=2, ensure_ascii=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
