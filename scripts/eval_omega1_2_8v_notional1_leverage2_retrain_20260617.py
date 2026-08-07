#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_8b_full_retrain_numeric_cash_sleeve_leverage_only_20260616 as exp  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_8v_notional1_leverage2_retrain_20260617"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
RISK = sleeve.FallbackRisk("notional1_tp052_sl028_lev2_h192", 0.052, 0.028, 1.0, 2.0, 192)


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


def _metric_row(
    candidate: str,
    family: str,
    cfg_id: int | None,
    cal_q: float | None,
    ev_min: float | None,
    utility_min: float | None,
    margin_min: float | None,
    val_m: dict[str, Any],
    oos_m: dict[str, Any],
    base_val: dict[str, Any],
    base_oos: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "candidate": candidate,
        "family": family,
        "utility_cfg_id": cfg_id,
        "cal_q": None if cal_q is None else float(cal_q),
        "ev_min": None if ev_min is None else float(ev_min),
        "utility_min": None if utility_min is None else float(utility_min),
        "margin_min": None if margin_min is None else float(margin_min),
    }
    row.update(sleeve._metric_row("val", val_m))
    row.update(sleeve._metric_row("oos", oos_m))
    row["val_delta_pnl"] = float(row["val_pnl"] - float(base_val["pnl"]))
    row["oos_delta_pnl"] = float(row["oos_pnl"] - float(base_oos["pnl"]))
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

    print(json.dumps({"stage": "path_labels", "risk": RISK.__dict__}, ensure_ascii=True), flush=True)
    path_labels, path_diag = exp._path_label_table(val_payload, RISK)
    ev_labels, ev_diag = exp._utility_from_path_labels(
        path_labels,
        RISK,
        {"stop_penalty": 0.0, "mae_penalty": 0.0, "time_penalty": 0.0},
    )

    rows: list[dict[str, Any]] = [
        _metric_row(
            "parent_only_baseline",
            "control_parent",
            None,
            None,
            None,
            None,
            None,
            parent_val,
            parent_oos,
            parent_val,
            parent_oos,
        )
    ]
    diagnostics: dict[str, Any] = {
        "risk": RISK.__dict__,
        "parent_artifact": meta["parent_dir"],
        "feature_count": int(x_val.shape[1]),
        "features": list(x_val.columns),
        "parent_baseline": {"validation": parent_val, "oos": parent_oos},
        "path_labels": path_diag,
        "ev_labels": ev_diag,
    }

    utility_preds: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for cfg_id, cfg in enumerate(exp.UTILITY_CFGS):
        print(json.dumps({"stage": "fit_utility", "cfg_id": int(cfg_id), "config": cfg}, ensure_ascii=True), flush=True)
        labels, diag = exp._utility_from_path_labels(path_labels, RISK, cfg)
        vl, vs, ol, os, fit_diag = exp._fit_predict_lower_bound(
            x_val,
            x_oos,
            labels,
            "long_utility",
            "short_utility",
            seed=285000 + cfg_id * 100,
            cal_q=0.50,
        )
        utility_preds[cfg_id] = (vl, vs, ol, os)
        diagnostics[f"utility_cfg_{cfg_id}"] = {"config": cfg, "labels": diag, "fit": fit_diag}

    for cal_q in (0.50, 0.65, 0.80):
        print(json.dumps({"stage": "fit_ev", "cal_q": float(cal_q)}, ensure_ascii=True), flush=True)
        ev_vl, ev_vs, ev_ol, ev_os, ev_fit_diag = exp._fit_predict_lower_bound(
            x_val,
            x_oos,
            ev_labels,
            "long_net",
            "short_net",
            seed=284000,
            cal_q=cal_q,
        )
        diagnostics[f"ev_lower_bound_cal_q{cal_q:.2f}"] = ev_fit_diag
        for ev_min in (0.001, 0.002, 0.003, 0.004):
            val_ev_a, val_ev_c = exp._actions_from_scores(ev_vl, ev_vs, ev_min)
            oos_ev_a, oos_ev_c = exp._actions_from_scores(ev_ol, ev_os, ev_min)
            val_m = sleeve._metrics_with_fallback(val_payload["frame"], val_payload["dec"], RISK, val_ev_a, val_ev_c, 0.0, fee=fee, slip=slip, cost_mult=3.0)
            oos_m = sleeve._metrics_with_fallback(oos_payload["frame"], oos_payload["dec"], RISK, oos_ev_a, oos_ev_c, 0.0, fee=fee, slip=slip, cost_mult=3.0)
            rows.append(_metric_row(f"notional1_ev_cal{cal_q:.2f}_ev{ev_min:.3f}", "ev_lower_bound_only", None, cal_q, ev_min, None, None, val_m, oos_m, parent_val, parent_oos))
            for cfg_id, (uvl, uvs, uol, uos) in utility_preds.items():
                for utility_min in (-0.001, 0.0, 0.001, 0.002):
                    for margin_min in (0.0, 0.001, 0.002):
                        val_a, val_c, val_filter = exp._apply_agreement(
                            val_ev_a,
                            val_ev_c,
                            uvl,
                            uvs,
                            utility_min=utility_min,
                            margin_min=margin_min,
                        )
                        oos_a, oos_c, oos_filter = exp._apply_agreement(
                            oos_ev_a,
                            oos_ev_c,
                            uol,
                            uos,
                            utility_min=utility_min,
                            margin_min=margin_min,
                        )
                        cand = f"notional1_ev_cal{cal_q:.2f}_ev{ev_min:.3f}_cfg{cfg_id}_u{utility_min:.3f}_m{margin_min:.3f}"
                        diagnostics[f"{cand}_filter"] = {"validation": val_filter, "oos": oos_filter}
                        val_m = sleeve._metrics_with_fallback(val_payload["frame"], val_payload["dec"], RISK, val_a, val_c, 0.0, fee=fee, slip=slip, cost_mult=3.0)
                        oos_m = sleeve._metrics_with_fallback(oos_payload["frame"], oos_payload["dec"], RISK, oos_a, oos_c, 0.0, fee=fee, slip=slip, cost_mult=3.0)
                        rows.append(_metric_row(cand, "ev_lower_bound_numeric_agreement_veto", cfg_id, cal_q, ev_min, utility_min, margin_min, val_m, oos_m, parent_val, parent_oos))

    ranking = pd.DataFrame(rows)
    ranking["val_fallback_stop_loss"] = ranking["val_reasons"].apply(lambda x: _reason_count(x, "fallback_stop_loss"))
    ranking["val_fallback_take_profit"] = ranking["val_reasons"].apply(lambda x: _reason_count(x, "fallback_take_profit"))
    ranking["val_fallback_max_hold"] = ranking["val_reasons"].apply(lambda x: _reason_count(x, "fallback_max_hold"))
    ranking["val_fallback_primary_takeover"] = ranking["val_reasons"].apply(lambda x: _reason_count(x, "fallback_primary_takeover"))
    ranking["oos_fallback_stop_loss"] = ranking["oos_reasons"].apply(lambda x: _reason_count(x, "fallback_stop_loss"))
    ranking["oos_fallback_take_profit"] = ranking["oos_reasons"].apply(lambda x: _reason_count(x, "fallback_take_profit"))
    ranking["oos_fallback_max_hold"] = ranking["oos_reasons"].apply(lambda x: _reason_count(x, "fallback_max_hold"))
    ranking["oos_fallback_primary_takeover"] = ranking["oos_reasons"].apply(lambda x: _reason_count(x, "fallback_primary_takeover"))
    ranking["val_fallback_stop_rate"] = ranking["val_fallback_stop_loss"] / ranking["val_fallback_entries"].replace(0, np.nan)
    ranking["val_fallback_stop_rate"] = ranking["val_fallback_stop_rate"].fillna(0.0)
    ranking["val_wr_drop_vs_parent"] = (float(parent_val["wr"]) - ranking["val_wr"].fillna(0.0)).clip(lower=0.0)
    ranking["selection_score_val_only"] = (
        ranking["val_delta_pnl"].fillna(0.0)
        + 0.04 * ranking["val_fallback_entries"].fillna(0.0)
        + 8.0 * ranking["val_wr"].fillna(0.0)
        + 0.20 * ranking["val_mdd"].fillna(0.0)
        - 1.50 * ranking["val_fallback_stop_loss"].fillna(0.0)
        - 0.35 * ranking["val_fallback_max_hold"].fillna(0.0)
        - 0.50 * ranking["val_fallback_primary_takeover"].fillna(0.0)
        - 18.0 * ranking["val_wr_drop_vs_parent"].fillna(0.0)
        - 6.0 * ranking["val_fallback_stop_rate"].fillna(0.0)
    )
    ranking = ranking.sort_values(["selection_score_val_only", "val_delta_pnl", "val_pnl"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "notional1_leverage2_ranking.csv", index=False)

    hybrid = ranking[ranking["family"].eq("ev_lower_bound_numeric_agreement_veto")].copy()
    selected = hybrid.iloc[0].to_dict() if len(hybrid) else ranking.iloc[0].to_dict()
    best_oos = hybrid.sort_values(["oos_pnl", "oos_delta_pnl"], ascending=False).iloc[0].to_dict() if len(hybrid) else ranking.sort_values(["oos_pnl", "oos_delta_pnl"], ascending=False).iloc[0].to_dict()

    blockers: list[str] = []
    bad_features = [
        c
        for c in x_val.columns
        if c == "tp_sl_action_score" or c.startswith("clean_regime4_") or c.startswith("regime4_pred_") or c.startswith("teacher_")
    ]
    if bad_features:
        blockers.append(f"forbidden feature columns: {bad_features[:20]}")
    if len(hybrid) == 0:
        blockers.append("no hybrid candidates produced")

    report = {
        "model_id": MODEL_ID,
        "status": "redteam_pass_notional1_leverage2_oof_eval" if not blockers else "redteam_fail",
        "method": "Omega 1.2.8b cash-sleeve structure retrained with fixed risk notional=1.0 and leverage=2.0. TP/SL/max_hold unchanged. Selection uses validation only; OOS is diagnostic.",
        "selection_policy": "validation_oof_only; OOS diagnostic only; no live export",
        "risk": RISK.__dict__,
        "baseline": {"parent_only_validation": parent_val, "parent_only_oos": parent_oos},
        "diagnostics": diagnostics,
        "selected_by_validation": selected,
        "best_by_oos_diagnostic": best_oos,
        "top20_hybrid": hybrid.head(20).to_dict(orient="records"),
        "top20_all": ranking.head(20).to_dict(orient="records"),
        "redteam_pass": not blockers,
        "redteam_blockers": blockers,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "notional1_leverage2_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "status": report["status"], "selected": selected, "best_oos": best_oos}, indent=2, ensure_ascii=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
