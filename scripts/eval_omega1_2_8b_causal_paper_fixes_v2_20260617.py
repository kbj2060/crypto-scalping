#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_8b_paper_fixes_20260617 as paper  # noqa: E402
import export_omega1_2_8b_live_bundle_20260616 as exporter  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_8b_full_retrain_numeric_cash_sleeve_leverage_only_20260616 as exp  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_8b_causal_paper_fixes_v2_20260617"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


@dataclass(frozen=True)
class Variant:
    name: str
    paper_motive: str
    ev_extra_offset: float = 0.0
    ev_min_delta: float = 0.0
    utility_min: float = -0.001
    margin_min: float = 0.0
    support_min_fraction: float | None = None
    support_max_z: float | None = None
    router_conf_min: float = 0.0


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


def _support_masks(
    x_val: pd.DataFrame,
    x_oos: pd.DataFrame,
    ev_labels: pd.DataFrame,
    *,
    min_fraction: float,
    max_z: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    idx = ev_labels["i"].to_numpy(dtype=np.int64)
    val_mask = np.zeros(len(x_val), dtype=bool)
    fold_diag: list[dict[str, Any]] = []
    for fold_id, (tr, va) in enumerate(exp._chron_folds(idx)):
        train_set = set(int(i) for i in tr.tolist())
        train_labels = ev_labels[ev_labels["i"].astype(int).isin(train_set)].reset_index(drop=True)
        profile = exporter._feature_support_profile(x_val, train_labels)
        fold_pass, diag = paper._support_pass(x_val.iloc[va], profile, min_fraction=min_fraction, max_z=max_z)
        val_mask[va] = fold_pass
        fold_diag.append(
            {
                "fold": int(fold_id),
                "train_rows": int(len(tr)),
                "val_rows": int(len(va)),
                "support_rows": int(profile["rows"]),
                "pass_rate": float(diag["pass_rate"]),
                "min_fraction": float(min_fraction),
                "max_z": float(max_z),
            }
        )
    full_profile = exporter._feature_support_profile(x_val, ev_labels)
    oos_mask, oos_diag = paper._support_pass(x_oos, full_profile, min_fraction=min_fraction, max_z=max_z)
    return val_mask, oos_mask, {"folds": fold_diag, "oos": oos_diag, "full_profile_rows_for_oos": int(full_profile["rows"])}


def _actions(
    x: pd.DataFrame,
    ev_long: np.ndarray,
    ev_short: np.ndarray,
    utility_long: np.ndarray,
    utility_short: np.ndarray,
    variant: Variant,
    *,
    support_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    long_s = np.asarray(ev_long, dtype=np.float64) - float(variant.ev_extra_offset)
    short_s = np.asarray(ev_short, dtype=np.float64) - float(variant.ev_extra_offset)
    ev_min = 0.003 + float(variant.ev_min_delta)
    best_long = long_s >= short_s
    best = np.where(best_long, long_s, short_s)
    action = np.where(best > ev_min, np.where(best_long, sleeve.ACTION_LONG, sleeve.ACTION_SHORT), sleeve.ACTION_CASH)
    long_ok = (action == sleeve.ACTION_LONG) & (utility_long > float(variant.utility_min)) & ((utility_long - utility_short) >= float(variant.margin_min))
    short_ok = (action == sleeve.ACTION_SHORT) & (utility_short > float(variant.utility_min)) & ((utility_short - utility_long) >= float(variant.margin_min))
    action = np.where(long_ok | short_ok, action, sleeve.ACTION_CASH).astype(np.int64)
    if support_mask is not None:
        action = np.where(support_mask, action, sleeve.ACTION_CASH).astype(np.int64)
    if variant.router_conf_min > 0.0:
        router_conf = x["router_confidence"].to_numpy(dtype=np.float64)
        action = np.where(router_conf >= float(variant.router_conf_min), action, sleeve.ACTION_CASH).astype(np.int64)
    conf = np.clip((best - ev_min) / 0.02, 0.0, 1.0)
    conf = np.where(action != sleeve.ACTION_CASH, conf, 0.0).astype(np.float64)
    return action, conf


def _row(
    variant: Variant,
    val_m: dict[str, Any],
    oos_m: dict[str, Any],
    base_val: dict[str, Any],
    base_oos: dict[str, Any],
    diag: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "variant": variant.name,
        "paper_motive": variant.paper_motive,
        "ev_extra_offset": float(variant.ev_extra_offset),
        "ev_min_delta": float(variant.ev_min_delta),
        "utility_min": float(variant.utility_min),
        "margin_min": float(variant.margin_min),
        "support_min_fraction": variant.support_min_fraction,
        "support_max_z": variant.support_max_z,
        "router_conf_min": float(variant.router_conf_min),
    }
    row.update(sleeve._metric_row("val", val_m))
    row.update(sleeve._metric_row("oos", oos_m))
    row["val_delta_pnl"] = float(row["val_pnl"] - float(base_val["pnl"]))
    row["oos_delta_pnl"] = float(row["oos_pnl"] - float(base_oos["pnl"]))
    row["val_fallback_stop_loss"] = _reason_count(row["val_reasons"], "fallback_stop_loss")
    row["oos_fallback_stop_loss"] = _reason_count(row["oos_reasons"], "fallback_stop_loss")
    row["val_fallback_primary_takeover"] = _reason_count(row["val_reasons"], "fallback_primary_takeover")
    row["oos_fallback_primary_takeover"] = _reason_count(row["oos_reasons"], "fallback_primary_takeover")
    row["val_wr_drop_vs_baseline"] = max(float(base_val["wr"]) - float(row["val_wr"]), 0.0)
    row["val_fallback_stop_rate"] = float(row["val_fallback_stop_loss"] / max(int(row["val_fallback_entries"]), 1))
    row["selection_score_val_only"] = (
        row["val_delta_pnl"]
        + 0.04 * row["val_fallback_entries"]
        + 8.0 * row["val_wr"]
        + 0.20 * row["val_mdd"]
        - 1.50 * row["val_fallback_stop_loss"]
        - 0.50 * row["val_fallback_primary_takeover"]
        - 18.0 * row["val_wr_drop_vs_baseline"]
        - 6.0 * row["val_fallback_stop_rate"]
    )
    row["diag"] = diag
    return row


def _variants() -> list[Variant]:
    return [
        Variant("causal_oof_base_numeric", "Control: OOF EV/utility lower-bound sleeve without paper gates"),
        Variant("causal_spci_extra_0005", "SPCI: stricter sequential residual lower-bound", ev_extra_offset=0.0005),
        Variant("causal_spci_extra_0010", "SPCI: stricter sequential residual lower-bound", ev_extra_offset=0.0010),
        Variant("causal_spci_extra_0015", "SPCI: stricter sequential residual lower-bound", ev_extra_offset=0.0015),
        Variant("causal_cql_support_092_8", "CQL: fold-prefix behavior-support action blocking", support_min_fraction=0.92, support_max_z=8.0),
        Variant("causal_cql_support_095_6", "CQL: stricter fold-prefix behavior-support action blocking", support_min_fraction=0.95, support_max_z=6.0),
        Variant("causal_cql_support_098_4", "CQL: high-confidence in-support only", support_min_fraction=0.98, support_max_z=4.0),
        Variant("causal_mmdrex_router_050", "MM-DREX: dynamic router confidence gate", router_conf_min=0.50),
        Variant("causal_mmdrex_router_055", "MM-DREX: dynamic router confidence gate", router_conf_min=0.55),
        Variant("causal_spci_cql_0010_095_6", "SPCI + CQL causal combination", ev_extra_offset=0.0010, support_min_fraction=0.95, support_max_z=6.0),
        Variant("causal_spci_cql_router", "SPCI + CQL + router confidence causal combination", ev_extra_offset=0.0010, support_min_fraction=0.95, support_max_z=6.0, router_conf_min=0.50),
        Variant("causal_conservative_all", "SPCI + CQL + utility margin + router confidence", ev_extra_offset=0.0010, utility_min=0.001, margin_min=0.001, support_min_fraction=0.95, support_max_z=6.0, router_conf_min=0.50),
    ]


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
    base_val = _base_sleeve_metrics(omega._metrics(val_payload["frame"], val_payload["dec"], fee=fee, slip=slip, cost_mult=3.0))
    base_oos = _base_sleeve_metrics(omega._metrics(oos_payload["frame"], oos_payload["dec"], fee=fee, slip=slip, cost_mult=3.0))

    print(json.dumps({"stage": "path_labels"}, ensure_ascii=True), flush=True)
    path_labels, path_diag = exp._path_label_table(val_payload, exp.RISK)
    ev_labels, ev_diag = exp._utility_from_path_labels(path_labels, exp.RISK, {"stop_penalty": 0.0, "mae_penalty": 0.0, "time_penalty": 0.0})
    utility_labels, utility_diag = exp._utility_from_path_labels(path_labels, exp.RISK, exp.UTILITY_CFGS[0])

    print(json.dumps({"stage": "fit_oof_ev"}, ensure_ascii=True), flush=True)
    ev_vl, ev_vs, ev_ol, ev_os, ev_fit_diag = exp._fit_predict_lower_bound(x_val, x_oos, ev_labels, "long_net", "short_net", seed=280000, cal_q=0.80)
    print(json.dumps({"stage": "fit_oof_utility"}, ensure_ascii=True), flush=True)
    u_vl, u_vs, u_ol, u_os, utility_fit_diag = exp._fit_predict_lower_bound(x_val, x_oos, utility_labels, "long_utility", "short_utility", seed=281000, cal_q=0.50)

    support_cache: dict[tuple[float, float], tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = [
        {
            "variant": "parent_only_baseline",
            "paper_motive": "Control: primary parent only",
            **sleeve._metric_row("val", base_val),
            **sleeve._metric_row("oos", base_oos),
            "val_delta_pnl": 0.0,
            "oos_delta_pnl": 0.0,
            "selection_score_val_only": 0.0,
        }
    ]
    for variant in _variants():
        print(json.dumps({"stage": "eval_variant", "variant": variant.name}, ensure_ascii=True), flush=True)
        support_val = None
        support_oos = None
        support_diag: dict[str, Any] = {"enabled": False}
        if variant.support_min_fraction is not None and variant.support_max_z is not None:
            key = (float(variant.support_min_fraction), float(variant.support_max_z))
            if key not in support_cache:
                support_cache[key] = _support_masks(x_val, x_oos, ev_labels, min_fraction=key[0], max_z=key[1])
            support_val, support_oos, support_diag = support_cache[key]
        val_a, val_c = _actions(x_val, ev_vl, ev_vs, u_vl, u_vs, variant, support_mask=support_val)
        oos_a, oos_c = _actions(x_oos, ev_ol, ev_os, u_ol, u_os, variant, support_mask=support_oos)
        val_m = sleeve._metrics_with_fallback(val_payload["frame"], val_payload["dec"], exp.RISK, val_a, val_c, 0.0, fee=fee, slip=slip, cost_mult=3.0)
        oos_m = sleeve._metrics_with_fallback(oos_payload["frame"], oos_payload["dec"], exp.RISK, oos_a, oos_c, 0.0, fee=fee, slip=slip, cost_mult=3.0)
        rows.append(_row(variant, val_m, oos_m, base_val, base_oos, {"support": support_diag}))

    ranking = pd.DataFrame(rows).sort_values(["selection_score_val_only", "val_delta_pnl", "val_pnl"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "causal_paper_fix_ranking.csv", index=False)
    candidate_rows = ranking[ranking["variant"].ne("parent_only_baseline")].copy()
    selected = candidate_rows.iloc[0].to_dict() if len(candidate_rows) else ranking.iloc[0].to_dict()
    best_oos = candidate_rows.sort_values(["oos_pnl", "oos_delta_pnl"], ascending=False).iloc[0].to_dict() if len(candidate_rows) else ranking.iloc[0].to_dict()
    blockers: list[str] = []
    bad = [c for c in x_val.columns if c == "tp_sl_action_score" or c.startswith("clean_regime4_") or c.startswith("regime4_pred_") or c.startswith("teacher_")]
    if bad:
        blockers.append(f"forbidden feature columns: {bad[:20]}")
    if str(selected.get("variant")) == "parent_only_baseline":
        blockers.append("no paper-fix candidate selected")
    report = {
        "model_id": MODEL_ID,
        "status": "redteam_pass_causal_oof_eval" if not blockers else "redteam_fail",
        "method": "HF paper ideas re-evaluated with chronological OOF validation. Validation never uses final exported bundle, full-validation residual offsets, or full-validation support profiles.",
        "selection_policy": "validation_oof_only; OOS is diagnostic and not used for selection",
        "papers": {
            "SPCI": "https://huggingface.co/papers/2212.03463",
            "C-CQL": "https://huggingface.co/papers/2301.01298",
            "MM-DREX": "https://huggingface.co/papers/2509.05080",
            "TimeRFT": "https://huggingface.co/papers/2605.00015",
            "CopulaCPTS": "https://huggingface.co/papers/2212.03281",
        },
        "baseline": {"validation": base_val, "oos": base_oos},
        "diagnostics": {
            "parent_artifact": meta["parent_dir"],
            "risk": exp.RISK.__dict__,
            "feature_count": int(x_val.shape[1]),
            "features": list(x_val.columns),
            "path_labels": path_diag,
            "ev_labels": ev_diag,
            "utility_labels": utility_diag,
            "ev_fit": ev_fit_diag,
            "utility_fit": utility_fit_diag,
            "support_profiles": {f"{k[0]}_{k[1]}": v[2] for k, v in support_cache.items()},
        },
        "selected_by_validation_oof": selected,
        "best_by_oos_diagnostic": best_oos,
        "top20": ranking.head(20).to_dict(orient="records"),
        "redteam_pass": not blockers,
        "redteam_blockers": blockers,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "causal_paper_fix_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "status": report["status"], "selected_by_validation_oof": selected, "best_by_oos_diagnostic": best_oos}, indent=2, ensure_ascii=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
