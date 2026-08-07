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

import train_eval_omega1_2_8_full_retrain_numeric_cash_sleeve_20260616 as hgb_exp  # noqa: E402
import train_eval_omega1_2_9_full_retrain_numeric_tabm_cash_sleeve_20260616 as tabm_exp  # noqa: E402


MODEL_ID = "omega1_2_10_tabm_stabilization_suite_20260616"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
HGB_REPORT = ROOT / "tmp/causal_regen_20260516/omega1_2_8_full_retrain_numeric_cash_sleeve_20260616/report.json"
TABM_REPORT = ROOT / "tmp/causal_regen_20260516/omega1_2_9_full_retrain_numeric_tabm_cash_sleeve_20260616/report.json"
TABM_RANKING = ROOT / "tmp/causal_regen_20260516/omega1_2_9_full_retrain_numeric_tabm_cash_sleeve_20260616/full_retrain_numeric_tabm_cash_sleeve_ranking.csv"


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


def _load_controls() -> dict[str, Any]:
    return {
        "hgb": json.loads(HGB_REPORT.read_text(encoding="utf-8")) if HGB_REPORT.exists() else None,
        "tabm": json.loads(TABM_REPORT.read_text(encoding="utf-8")) if TABM_REPORT.exists() else None,
    }


def _metric_row(
    candidate: str,
    family: str,
    val_m: dict[str, Any],
    oos_m: dict[str, Any],
    base_val: dict[str, Any],
    base_oos: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {"candidate": candidate, "family": family}
    row.update(extra or {})
    row.update(hgb_exp.sleeve._metric_row("val", val_m))
    row.update(hgb_exp.sleeve._metric_row("oos", oos_m))
    row["val_delta_pnl"] = float(row["val_pnl"] - float(base_val["pnl"]))
    row["oos_delta_pnl"] = float(row["oos_pnl"] - float(base_oos["pnl"]))
    return row


def _evaluate_actions(
    val_payload: dict[str, Any],
    oos_payload: dict[str, Any],
    val_a: np.ndarray,
    val_c: np.ndarray,
    oos_a: np.ndarray,
    oos_c: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fee = float(val_payload["fee"])
    slip = float(val_payload["slip"])
    val_m = hgb_exp.sleeve._metrics_with_fallback(
        val_payload["frame"],
        val_payload["dec"],
        hgb_exp.RISK,
        val_a,
        val_c,
        0.0,
        fee=fee,
        slip=slip,
        cost_mult=3.0,
    )
    oos_m = hgb_exp.sleeve._metrics_with_fallback(
        oos_payload["frame"],
        oos_payload["dec"],
        hgb_exp.RISK,
        oos_a,
        oos_c,
        0.0,
        fee=fee,
        slip=slip,
        cost_mult=3.0,
    )
    return val_m, oos_m


def _apply_uncertainty_veto(
    ev_action: np.ndarray,
    ev_conf: np.ndarray,
    util_long_stack: np.ndarray,
    util_short_stack: np.ndarray,
    *,
    utility_min: float,
    margin_min: float,
    max_std: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mean_l = util_long_stack.mean(axis=0)
    mean_s = util_short_stack.mean(axis=0)
    std_l = util_long_stack.std(axis=0)
    std_s = util_short_stack.std(axis=0)
    long_ok = (
        (ev_action == hgb_exp.sleeve.ACTION_LONG)
        & (mean_l > float(utility_min))
        & ((mean_l - mean_s) >= float(margin_min))
        & (std_l <= float(max_std))
    )
    short_ok = (
        (ev_action == hgb_exp.sleeve.ACTION_SHORT)
        & (mean_s > float(utility_min))
        & ((mean_s - mean_l) >= float(margin_min))
        & (std_s <= float(max_std))
    )
    keep = long_ok | short_ok
    support = np.where(ev_action == hgb_exp.sleeve.ACTION_LONG, mean_l, np.where(ev_action == hgb_exp.sleeve.ACTION_SHORT, mean_s, 0.0))
    support_std = np.where(ev_action == hgb_exp.sleeve.ACTION_LONG, std_l, np.where(ev_action == hgb_exp.sleeve.ACTION_SHORT, std_s, np.inf))
    action = np.where(keep, ev_action, hgb_exp.sleeve.ACTION_CASH).astype(np.int64)
    conf = np.where(keep, np.minimum(ev_conf, np.clip((support - float(utility_min)) / 0.02, 0.0, 1.0)), 0.0).astype(np.float64)
    active = np.isin(ev_action, [hgb_exp.sleeve.ACTION_LONG, hgb_exp.sleeve.ACTION_SHORT])
    diag = {
        "ev_active_rows": int(active.sum()),
        "kept_rows": int(keep.sum()),
        "veto_rows": int((active & ~keep).sum()),
        "uncertainty_veto_rows": int((active & (support_std > float(max_std))).sum()),
        "keep_rate_on_ev_active": float(keep.sum() / max(active.sum(), 1)),
        "utility_min": float(utility_min),
        "margin_min": float(margin_min),
        "max_std": float(max_std),
    }
    return action, conf, diag


def _rerank_tabm_with_stronger_fallback_penalty() -> dict[str, Any]:
    if not TABM_RANKING.exists():
        return {"available": False, "reason": f"missing {TABM_RANKING}"}
    ranking = pd.read_csv(TABM_RANKING)
    hybrid = ranking[ranking["family"].eq("tabm_numeric_agreement_veto")].copy()
    if len(hybrid) == 0:
        return {"available": False, "reason": "no tabm hybrid rows"}
    for penalty in (0.40, 0.80, 1.20):
        hybrid[f"fallback_penalty_score_{penalty:.2f}"] = (
            hybrid["val_delta_pnl"].fillna(0.0)
            - float(penalty) * hybrid["val_fallback_entries"].fillna(0.0)
            + 8.0 * hybrid["val_wr"].fillna(0.0)
            + 0.20 * hybrid["val_mdd"].fillna(0.0)
        )
    rows = []
    for penalty in (0.40, 0.80, 1.20):
        col = f"fallback_penalty_score_{penalty:.2f}"
        best = hybrid.sort_values([col, "val_delta_pnl", "val_pnl"], ascending=False).iloc[0].to_dict()
        rows.append({"penalty": float(penalty), "selected": best})
    pd.DataFrame([{"penalty": r["penalty"], **r["selected"]} for r in rows]).to_csv(OUT_DIR / "tabm_strong_fallback_penalty_rerank.csv", index=False)
    return {"available": True, "rows": rows}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    controls = _load_controls()
    print(json.dumps({"stage": "build_payloads", "model_id": MODEL_ID}, ensure_ascii=True), flush=True)
    val_payload, oos_payload, meta = hgb_exp._build_payloads()
    x_val = val_payload["features"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_oos = oos_payload["features"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    fee = float(meta["fee"])
    slip = float(meta["slip"])
    base_val = hgb_exp.omega._metrics(val_payload["frame"], val_payload["dec"], fee=fee, slip=slip, cost_mult=3.0)
    base_oos = hgb_exp.omega._metrics(oos_payload["frame"], oos_payload["dec"], fee=fee, slip=slip, cost_mult=3.0)
    base_val_sleeve = {**base_val, "primary_entries": base_val["long_entries"] + base_val["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}
    base_oos_sleeve = {**base_oos, "primary_entries": base_oos["long_entries"] + base_oos["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}

    path_labels, path_diag = hgb_exp._path_label_table(val_payload, hgb_exp.RISK)
    ev_labels, ev_diag = hgb_exp._utility_from_path_labels(path_labels, hgb_exp.RISK, {"stop_penalty": 0.0, "mae_penalty": 0.0, "time_penalty": 0.0})
    util_labels_cfg1, util_diag_cfg1 = hgb_exp._utility_from_path_labels(path_labels, hgb_exp.RISK, hgb_exp.UTILITY_CFGS[1])

    rows: list[dict[str, Any]] = [
        {
            "candidate": "full_retrain_primary_only",
            "family": "baseline",
            **hgb_exp.sleeve._metric_row("val", base_val_sleeve),
            **hgb_exp.sleeve._metric_row("oos", base_oos_sleeve),
            "val_delta_pnl": 0.0,
            "oos_delta_pnl": 0.0,
        }
    ]
    diagnostics: dict[str, Any] = {
        "mode": "tabm_stabilization_suite",
        "baseline_model_id": hgb_exp.BASELINE_ID,
        "parent_artifact": meta["parent_dir"],
        "risk": hgb_exp.asdict(hgb_exp.RISK),
        "feature_count": int(x_val.shape[1]),
        "path_labels": path_diag,
        "ev_labels": ev_diag,
        "utility_cfg1_labels": util_diag_cfg1,
        "controls": {
            "hgb_selected": controls["hgb"]["selected_by_validation"] if controls["hgb"] else None,
            "hgb_best_oos": controls["hgb"]["best_by_oos_diagnostic"] if controls["hgb"] else None,
            "tabm_selected": controls["tabm"]["selected_by_validation"] if controls["tabm"] else None,
            "tabm_best_oos": controls["tabm"]["best_by_oos_diagnostic"] if controls["tabm"] else None,
        },
    }

    # 1. HGB owns candidate direction; TabM is only a second veto.
    print(json.dumps({"stage": "test_1_hgb_owner_tabm_veto"}, ensure_ascii=True), flush=True)
    hgb_ev_cache: dict[tuple[float, float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    tabm_util_vl, tabm_util_vs, tabm_util_ol, tabm_util_os, tabm_util_diag = tabm_exp._fit_predict_lower_bound_tabm(
        x_val,
        x_oos,
        util_labels_cfg1,
        "long_utility",
        "short_utility",
        seed=301100,
        cal_q=0.50,
    )
    diagnostics["test_1_tabm_utility_fit"] = tabm_util_diag
    for cal_q, ev_min in ((0.50, 0.001), (0.50, 0.004), (0.65, 0.003)):
        ev_vl, ev_vs, ev_ol, ev_os, ev_fit = hgb_exp._fit_predict_lower_bound(x_val, x_oos, ev_labels, "long_net", "short_net", seed=280000, cal_q=cal_q)
        diagnostics[f"test_1_hgb_ev_fit_cal{cal_q:.2f}"] = ev_fit
        hgb_ev_cache[(cal_q, ev_min)] = (ev_vl, ev_vs, ev_ol, ev_os)
        val_ev_a, val_ev_c = hgb_exp._actions_from_scores(ev_vl, ev_vs, ev_min)
        oos_ev_a, oos_ev_c = hgb_exp._actions_from_scores(ev_ol, ev_os, ev_min)
        for utility_min in (0.0, 0.001, 0.002):
            for margin_min in (0.0, 0.001):
                val_a, val_c, val_filter = hgb_exp._apply_agreement(
                    val_ev_a,
                    val_ev_c,
                    tabm_util_vl,
                    tabm_util_vs,
                    utility_min=utility_min,
                    margin_min=margin_min,
                )
                oos_a, oos_c, oos_filter = hgb_exp._apply_agreement(
                    oos_ev_a,
                    oos_ev_c,
                    tabm_util_ol,
                    tabm_util_os,
                    utility_min=utility_min,
                    margin_min=margin_min,
                )
                val_m, oos_m = _evaluate_actions(val_payload, oos_payload, val_a, val_c, oos_a, oos_c)
                cand = f"hgb_owner_tabm_veto_cal{cal_q:.2f}_ev{ev_min:.3f}_u{utility_min:.3f}_m{margin_min:.3f}"
                diagnostics[f"{cand}_filter"] = {"validation": val_filter, "oos": oos_filter}
                rows.append(
                    _metric_row(
                        cand,
                        "test1_hgb_owner_tabm_veto",
                        val_m,
                        oos_m,
                        base_val_sleeve,
                        base_oos_sleeve,
                        {"cal_q": cal_q, "ev_min": ev_min, "utility_min": utility_min, "margin_min": margin_min},
                    )
                )

    # 2. Stronger fallback-entry penalty: rerank prior plain TabM grid.
    print(json.dumps({"stage": "test_2_stronger_fallback_penalty"}, ensure_ascii=True), flush=True)
    rerank = _rerank_tabm_with_stronger_fallback_penalty()
    diagnostics["test_2_stronger_fallback_penalty"] = rerank

    # 3. More conservative TabM residual calibration.
    print(json.dumps({"stage": "test_3_conservative_tabm_calibration"}, ensure_ascii=True), flush=True)
    for cal_q in (0.80, 0.90):
        ev_vl, ev_vs, ev_ol, ev_os, ev_fit = tabm_exp._fit_predict_lower_bound_tabm(
            x_val,
            x_oos,
            ev_labels,
            "long_net",
            "short_net",
            seed=302000 + int(cal_q * 100),
            cal_q=cal_q,
        )
        diagnostics[f"test_3_tabm_ev_fit_cal{cal_q:.2f}"] = ev_fit
        for ev_min in (0.001, 0.002, 0.004):
            val_ev_a, val_ev_c = hgb_exp._actions_from_scores(ev_vl, ev_vs, ev_min)
            oos_ev_a, oos_ev_c = hgb_exp._actions_from_scores(ev_ol, ev_os, ev_min)
            for utility_min in (0.0, 0.001, 0.002):
                val_a, val_c, val_filter = hgb_exp._apply_agreement(
                    val_ev_a,
                    val_ev_c,
                    tabm_util_vl,
                    tabm_util_vs,
                    utility_min=utility_min,
                    margin_min=0.001,
                )
                oos_a, oos_c, oos_filter = hgb_exp._apply_agreement(
                    oos_ev_a,
                    oos_ev_c,
                    tabm_util_ol,
                    tabm_util_os,
                    utility_min=utility_min,
                    margin_min=0.001,
                )
                val_m, oos_m = _evaluate_actions(val_payload, oos_payload, val_a, val_c, oos_a, oos_c)
                cand = f"tabm_conservative_cal{cal_q:.2f}_ev{ev_min:.3f}_u{utility_min:.3f}_m0.001"
                diagnostics[f"{cand}_filter"] = {"validation": val_filter, "oos": oos_filter}
                rows.append(
                    _metric_row(
                        cand,
                        "test3_conservative_tabm_calibration",
                        val_m,
                        oos_m,
                        base_val_sleeve,
                        base_oos_sleeve,
                        {"cal_q": cal_q, "ev_min": ev_min, "utility_min": utility_min, "margin_min": 0.001},
                    )
                )

    # 4. Seed ensemble uncertainty veto. HGB owns EV; TabM utility ensemble only vetoes uncertain rows.
    print(json.dumps({"stage": "test_4_seed_ensemble_uncertainty_veto"}, ensure_ascii=True), flush=True)
    util_stacks_val_l = [tabm_util_vl]
    util_stacks_val_s = [tabm_util_vs]
    util_stacks_oos_l = [tabm_util_ol]
    util_stacks_oos_s = [tabm_util_os]
    for seed in (301200, 301300):
        vl, vs, ol, os, diag = tabm_exp._fit_predict_lower_bound_tabm(
            x_val,
            x_oos,
            util_labels_cfg1,
            "long_utility",
            "short_utility",
            seed=seed,
            cal_q=0.50,
        )
        diagnostics[f"test_4_tabm_utility_seed_{seed}"] = diag
        util_stacks_val_l.append(vl)
        util_stacks_val_s.append(vs)
        util_stacks_oos_l.append(ol)
        util_stacks_oos_s.append(os)
    val_l_stack = np.stack(util_stacks_val_l, axis=0)
    val_s_stack = np.stack(util_stacks_val_s, axis=0)
    oos_l_stack = np.stack(util_stacks_oos_l, axis=0)
    oos_s_stack = np.stack(util_stacks_oos_s, axis=0)
    for cal_q, ev_min in ((0.50, 0.001), (0.50, 0.004)):
        ev_vl, ev_vs, ev_ol, ev_os = hgb_ev_cache.get((cal_q, ev_min), (None, None, None, None))
        if ev_vl is None:
            ev_vl, ev_vs, ev_ol, ev_os, _fit = hgb_exp._fit_predict_lower_bound(x_val, x_oos, ev_labels, "long_net", "short_net", seed=280000, cal_q=cal_q)
        val_ev_a, val_ev_c = hgb_exp._actions_from_scores(ev_vl, ev_vs, ev_min)
        oos_ev_a, oos_ev_c = hgb_exp._actions_from_scores(ev_ol, ev_os, ev_min)
        for utility_min in (0.0, 0.001):
            for max_std in (0.0015, 0.0030, 0.0060):
                val_a, val_c, val_filter = _apply_uncertainty_veto(
                    val_ev_a,
                    val_ev_c,
                    val_l_stack,
                    val_s_stack,
                    utility_min=utility_min,
                    margin_min=0.001,
                    max_std=max_std,
                )
                oos_a, oos_c, oos_filter = _apply_uncertainty_veto(
                    oos_ev_a,
                    oos_ev_c,
                    oos_l_stack,
                    oos_s_stack,
                    utility_min=utility_min,
                    margin_min=0.001,
                    max_std=max_std,
                )
                val_m, oos_m = _evaluate_actions(val_payload, oos_payload, val_a, val_c, oos_a, oos_c)
                cand = f"hgb_owner_tabm_ensemble_uncert_cal{cal_q:.2f}_ev{ev_min:.3f}_u{utility_min:.3f}_std{max_std:.4f}"
                diagnostics[f"{cand}_filter"] = {"validation": val_filter, "oos": oos_filter}
                rows.append(
                    _metric_row(
                        cand,
                        "test4_seed_ensemble_uncertainty_veto",
                        val_m,
                        oos_m,
                        base_val_sleeve,
                        base_oos_sleeve,
                        {"cal_q": cal_q, "ev_min": ev_min, "utility_min": utility_min, "margin_min": 0.001, "max_std": max_std, "ensemble_size": 3},
                    )
                )

    ranking = pd.DataFrame(rows)
    ranking["selection_score_val_only"] = (
        ranking["val_delta_pnl"].fillna(0.0)
        - 0.80 * ranking["val_fallback_entries"].fillna(0.0)
        + 8.0 * ranking["val_wr"].fillna(0.0)
        + 0.20 * ranking["val_mdd"].fillna(0.0)
    )
    ranking = ranking.sort_values(["selection_score_val_only", "val_delta_pnl", "val_pnl"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "tabm_stabilization_suite_ranking.csv", index=False)

    family_bests = []
    for family, group in ranking[ranking["family"].ne("baseline")].groupby("family"):
        val_best = group.sort_values(["selection_score_val_only", "val_delta_pnl"], ascending=False).iloc[0].to_dict()
        oos_best = group.sort_values(["oos_pnl", "oos_delta_pnl"], ascending=False).iloc[0].to_dict()
        family_bests.append({"family": family, "selected_by_validation": val_best, "best_by_oos_diagnostic": oos_best})
    best_val = ranking[ranking["family"].ne("baseline")].iloc[0].to_dict()
    best_oos = ranking[ranking["family"].ne("baseline")].sort_values(["oos_pnl", "oos_delta_pnl"], ascending=False).iloc[0].to_dict()

    report = {
        "model_id": MODEL_ID,
        "status": "redteam_pass_tabm_stabilization_suite",
        "method": "Four TabM stabilization tests on the full-retrained numeric cash sleeve: HGB-owner TabM veto, stronger fallback penalty rerank, conservative TabM calibration, and seed-ensemble uncertainty veto.",
        "selection_policy": "validation_only_no_oos_selection_for_candidate_rows; OOS diagnostic; test_2 is rerank-only on prior TabM grid",
        "baseline": {"validation": base_val_sleeve, "oos": base_oos_sleeve},
        "diagnostics": diagnostics,
        "selected_by_validation": best_val,
        "best_by_oos_diagnostic": best_oos,
        "family_bests": family_bests,
        "top20": ranking.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "tabm_stabilization_suite_ranking.csv"),
            "rerank": str(OUT_DIR / "tabm_strong_fallback_penalty_rerank.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "status": report["status"], "selected": best_val, "best_oos_diagnostic": best_oos}, indent=2, ensure_ascii=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
