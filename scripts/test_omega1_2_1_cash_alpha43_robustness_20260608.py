#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import plot_omega1_2_1_cash_alpha43_sleeve_trade_charts_20260608 as chart  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_label_family_20260606 as label_family  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_1_cash_alpha43_robustness_20260608"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


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


def _metric_row(prefix: str, m: dict[str, Any]) -> dict[str, Any]:
    return sleeve._metric_row(prefix, m)


def _fit_preds_for_label(
    label_name: str,
    atr_mult: float,
    max_hold: int,
    val_frame: pd.DataFrame,
    val_dec: pd.DataFrame,
    val_features: pd.DataFrame,
    oos_features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    val_cash = ~omega._active(val_dec)
    if label_name == "tb_custom":
        y_val, valid_mask, label_diag = label_family._triple_barrier_labels(val_frame, atr_mult=atr_mult, max_hold=max_hold, min_barrier=0.0035)
    elif label_name == "tb_atr08_h48":
        y_val, valid_mask, label_diag = label_family._label_family("tb_atr08_h48", val_frame, val_dec, val_cash, 2025)
    else:
        raise RuntimeError(f"unknown label_name: {label_name}")
    train_mask = val_cash & valid_mask
    if int(np.count_nonzero(train_mask)) < 500 or len(np.unique(y_val[train_mask])) < 2:
        raise RuntimeError(f"insufficient training labels: {label_name} atr={atr_mult} hold={max_hold}")
    val_action, val_conf, oof_diag = label_family._predict_oof("hgb", val_features, y_val, train_mask, seed=260608 + int(100 * atr_mult) + int(max_hold))
    oos_action, oos_conf, _fitted = label_family._fit_predict("hgb", val_features, y_val, train_mask, oos_features, seed=260608 + int(100 * atr_mult) + int(max_hold))
    return val_action, val_conf, oos_action, oos_conf, {"label_diag": label_diag, "oof_diag": oof_diag}


def _evaluate(
    rows: list[dict[str, Any]],
    *,
    family: str,
    risk: sleeve.FallbackRisk,
    threshold: float,
    val_frame: pd.DataFrame,
    val_dec: pd.DataFrame,
    val_action: np.ndarray,
    val_conf: np.ndarray,
    oos_frame: pd.DataFrame,
    oos_dec: pd.DataFrame,
    oos_action: np.ndarray,
    oos_conf: np.ndarray,
    fee: float,
    slip: float,
    cost_mult: float,
) -> None:
    val_m = sleeve._metrics_with_fallback(val_frame, val_dec, risk, val_action, val_conf, threshold, fee=fee, slip=slip, cost_mult=cost_mult)
    oos_m = sleeve._metrics_with_fallback(oos_frame, oos_dec, risk, oos_action, oos_conf, threshold, fee=fee, slip=slip, cost_mult=cost_mult)
    rows.append(
        {
            "family": family,
            "risk": risk.name,
            "threshold": float(threshold),
            "cost_mult": float(cost_mult),
            **_metric_row("val", val_m),
            **_metric_row("oos", oos_m),
        }
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_frame, val_dec, val_features = chart._build_split(frames, "validation")
    oos_frame, oos_dec, oos_features = chart._build_split(frames, "oos")

    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}

    # 1) Local threshold/risk robustness around the selected label/model.
    val_action, val_conf, oos_action, oos_conf, diag = _fit_preds_for_label("tb_atr08_h48", 0.8, 48, val_frame, val_dec, val_features, oos_features)
    diagnostics["tb_atr08_h48"] = diag
    for notional in (0.25, 0.30, 0.35):
        for hold in (96, 144, 192):
            risk = sleeve.FallbackRisk(f"tp026_sl014_n{notional:g}_h{hold}", 0.026, 0.014, float(notional), 2.0, int(hold))
            for threshold in (0.50, 0.55, 0.60, 0.65, 0.70):
                for cost_mult in (3.0, 4.0, 5.0):
                    _evaluate(
                        rows,
                        family="risk_threshold_local_tb08h48",
                        risk=risk,
                        threshold=threshold,
                        val_frame=val_frame,
                        val_dec=val_dec,
                        val_action=val_action,
                        val_conf=val_conf,
                        oos_frame=oos_frame,
                        oos_dec=oos_dec,
                        oos_action=oos_action,
                        oos_conf=oos_conf,
                        fee=fee,
                        slip=slip,
                        cost_mult=cost_mult,
                    )

    # 2) Label robustness: nearby ATR/horizon labels, selected risk shape only.
    for atr_mult in (0.6, 0.8, 1.0):
        for label_hold in (24, 48, 72):
            name = f"label_atr{atr_mult:g}_h{label_hold}"
            try:
                va, vc, oa, oc, ldiag = _fit_preds_for_label("tb_custom", atr_mult, label_hold, val_frame, val_dec, val_features, oos_features)
            except RuntimeError as exc:
                diagnostics[f"{name}_skip"] = str(exc)
                continue
            diagnostics[name] = ldiag
            risk = sleeve.FallbackRisk("tp026_sl014_n0.30_h192", 0.026, 0.014, 0.30, 2.0, 192)
            for threshold in (0.50, 0.55, 0.60, 0.65):
                _evaluate(
                    rows,
                    family=name,
                    risk=risk,
                    threshold=threshold,
                    val_frame=val_frame,
                    val_dec=val_dec,
                    val_action=va,
                    val_conf=vc,
                    oos_frame=oos_frame,
                    oos_dec=oos_dec,
                    oos_action=oa,
                    oos_conf=oc,
                    fee=fee,
                    slip=slip,
                    cost_mult=3.0,
                )

    ranking = pd.DataFrame(rows)
    ranking["val_delta_pnl"] = ranking["val_pnl"] - sleeve.AGGRESSIVE_VAL["pnl"]
    ranking["oos_delta_pnl"] = ranking["oos_pnl"] - sleeve.AGGRESSIVE_OOS["pnl"]
    ranking["val_delta_mdd"] = ranking["val_mdd"] - sleeve.AGGRESSIVE_VAL["mdd"]
    ranking["oos_delta_mdd"] = ranking["oos_mdd"] - sleeve.AGGRESSIVE_OOS["mdd"]
    ranking["promotable_like"] = (
        (ranking["cost_mult"] == 3.0)
        & (ranking["val_pnl"] > sleeve.AGGRESSIVE_VAL["pnl"])
        & (ranking["oos_pnl"] > sleeve.AGGRESSIVE_OOS["pnl"])
        & (ranking["val_mdd"] >= -12.0)
        & (ranking["oos_mdd"] >= -10.0)
    )
    ranking = ranking.sort_values(["cost_mult", "oos_pnl", "val_pnl"], ascending=[True, False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "robustness_ranking.csv", index=False)
    cost3 = ranking[ranking["cost_mult"] == 3.0].copy()
    report = {
        "model_id": MODEL_ID,
        "base_candidate": chart.CANDIDATE,
        "diagnostics": diagnostics,
        "top_cost3": cost3.sort_values(["oos_pnl", "val_pnl"], ascending=False).head(20).to_dict(orient="records"),
        "selected_candidate_cost3": cost3[
            (cost3["family"] == "risk_threshold_local_tb08h48")
            & (cost3["risk"] == "tp026_sl014_n0.3_h192")
            & (cost3["threshold"] == 0.55)
        ].to_dict(orient="records"),
        "cost_stress_selected": ranking[
            (ranking["family"] == "risk_threshold_local_tb08h48")
            & (ranking["risk"] == "tp026_sl014_n0.3_h192")
            & (ranking["threshold"] == 0.55)
        ].sort_values("cost_mult").to_dict(orient="records"),
        "promotable_like_count": int(ranking["promotable_like"].sum()),
        "artifacts": {"ranking": str(OUT_DIR / "robustness_ranking.csv"), "report": str(OUT_DIR / "report.json")},
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "top_cost3": report["top_cost3"][:8], "cost_stress_selected": report["cost_stress_selected"]}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
