#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_2_tp_runner_cash_sleeve_20260615 as base  # noqa: E402
import train_eval_omega1_2_3_cash_sleeve_upgrade_20260615 as upgrade  # noqa: E402


MODEL_ID = "omega1_2_4_ev_calibrated_cash_sleeve_probe_20260616"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
BUNDLE_PATH = ROOT / "data/ensemble/supervised/omega1_2_3_ev_hgb_cash_sleeve_20260615/ev_hgb_cash_sleeve.joblib"


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


def _selected_runner_cfg() -> base.repair.RunnerConfig:
    baseline_report = json.loads(base.BASELINE_REPORT.read_text(encoding="utf-8"))
    selected_cfg = baseline_report["selected_config"]
    return base.repair.RunnerConfig(
        int(selected_cfg["candidate_id"]),
        str(selected_cfg["mode"]),
        float(selected_cfg["quality_min"]),
        float(selected_cfg["extend_mult"]),
        float(selected_cfg["floor_frac"]),
        int(selected_cfg["max_extensions"]),
    )


def _metric_row(name: str, family: str, val_m: dict[str, Any], val_ledger: pd.DataFrame, oos_m: dict[str, Any], oos_ledger: pd.DataFrame, base_val: dict[str, Any], base_oos: dict[str, Any]) -> dict[str, Any]:
    row = {"candidate": name, "family": family}
    row.update(upgrade._row("val", val_m, val_ledger))
    row.update(upgrade._row("oos", oos_m, oos_ledger))
    row["val_delta_pnl"] = float(row["val_pnl"] - float(base_val["pnl"]))
    row["oos_delta_pnl"] = float(row["oos_pnl"] - float(base_oos["pnl"]))
    return row


def _fit_stop_probs(x_val: pd.DataFrame, x_oos: pd.DataFrame, labels: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    idx = labels["i"].to_numpy(dtype=np.int64)
    y_stop = np.zeros(len(x_val), dtype=np.int64)
    y_stop[idx] = labels["best_stop"].to_numpy(dtype=np.int64)
    if len(np.unique(y_stop[idx])) < 2:
        return None, None, {"skipped": "single_class_stop_label"}
    val_stop, oos_stop, diag = upgrade._fit_predict_binary("hgb", x_val, y_stop, idx, x_oos, seed=263001)
    return val_stop, oos_stop, diag


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _selected_runner_cfg()
    data = base.legacy_runner._build()
    x_val_all = upgrade._enhanced_features(data["validation"])
    x_oos_all = upgrade._enhanced_features(data["oos"])
    bundle = joblib.load(BUNDLE_PATH)
    feature_cols = list(bundle["feature_cols"])
    risk = base.SleeveRisk("base_tp026_sl014_n0405_h192", 0.026, 0.014, 0.405, 2.0, 192)
    ev_min_base = float(bundle["ev_min"])

    base_val, base_val_ledger = base._simulate_combo(data["validation"], cfg, None, None, None, 1.0)
    base_oos, base_oos_ledger = base._simulate_combo(data["oos"], cfg, None, None, None, 1.0)

    x_val = x_val_all[feature_cols]
    x_oos = x_oos_all[feature_cols]
    val_long = bundle["long_model"].predict(x_val.to_numpy(dtype=np.float64)).astype(np.float64)
    val_short = bundle["short_model"].predict(x_val.to_numpy(dtype=np.float64)).astype(np.float64)
    oos_long = bundle["long_model"].predict(x_oos.to_numpy(dtype=np.float64)).astype(np.float64)
    oos_short = bundle["short_model"].predict(x_oos.to_numpy(dtype=np.float64)).astype(np.float64)

    labels, label_diag = upgrade._label_table(data["validation"], risk, ev_min_base)
    idx = labels["i"].to_numpy(dtype=np.int64)
    long_resid = np.abs(labels["long_net"].to_numpy(dtype=np.float64) - val_long[idx])
    short_resid = np.abs(labels["short_net"].to_numpy(dtype=np.float64) - val_short[idx])
    val_stop, oos_stop, stop_diag = _fit_stop_probs(x_val_all, x_oos_all, labels)

    rows: list[dict[str, Any]] = []
    ledgers: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    raw_val_a, raw_val_c = upgrade._actions_from_ev(val_long, val_short, ev_min_base)
    raw_oos_a, raw_oos_c = upgrade._actions_from_ev(oos_long, oos_short, ev_min_base)
    val_m, val_ledger = base._simulate_combo(data["validation"], cfg, risk, raw_val_a, raw_val_c, 0.0)
    oos_m, oos_ledger = base._simulate_combo(data["oos"], cfg, risk, raw_oos_a, raw_oos_c, 0.0)
    rows.append(_metric_row("omega1_2_3_raw_ev002", "raw", val_m, val_ledger, oos_m, oos_ledger, base_val, base_oos))
    ledgers["omega1_2_3_raw_ev002"] = (val_ledger, oos_ledger)

    for q in (0.50, 0.60, 0.70, 0.80, 0.90):
        q_long = float(np.quantile(long_resid, q))
        q_short = float(np.quantile(short_resid, q))
        cal_val_long = val_long - q_long
        cal_val_short = val_short - q_short
        cal_oos_long = oos_long - q_long
        cal_oos_short = oos_short - q_short
        for ev_min in (0.000, 0.001, 0.002):
            val_a, val_c = upgrade._actions_from_ev(cal_val_long, cal_val_short, ev_min)
            oos_a, oos_c = upgrade._actions_from_ev(cal_oos_long, cal_oos_short, ev_min)
            name = f"cal_q{int(q*100):02d}_ev{ev_min:.3f}"
            val_m, val_ledger = base._simulate_combo(data["validation"], cfg, risk, val_a, val_c, 0.0)
            oos_m, oos_ledger = base._simulate_combo(data["oos"], cfg, risk, oos_a, oos_c, 0.0)
            rows.append(_metric_row(name, "ev_lower_bound", val_m, val_ledger, oos_m, oos_ledger, base_val, base_oos))
            ledgers[name] = (val_ledger, oos_ledger)
            if val_stop is not None and oos_stop is not None:
                for stop_max in (0.35, 0.45, 0.55):
                    va2, vc2 = upgrade._apply_veto(val_a, val_c, val_stop, stop_max)
                    oa2, oc2 = upgrade._apply_veto(oos_a, oos_c, oos_stop, stop_max)
                    veto_name = f"{name}_stop{stop_max:.2f}"
                    val_m, val_ledger = base._simulate_combo(data["validation"], cfg, risk, va2, vc2, 0.0)
                    oos_m, oos_ledger = base._simulate_combo(data["oos"], cfg, risk, oa2, oc2, 0.0)
                    rows.append(_metric_row(veto_name, "ev_lower_bound_stop_veto", val_m, val_ledger, oos_m, oos_ledger, base_val, base_oos))
                    ledgers[veto_name] = (val_ledger, oos_ledger)

    ranking = pd.DataFrame(rows)
    ranking["selection_score_val_only"] = (
        ranking["val_fallback_pnl"].fillna(0.0)
        + 0.20 * ranking["val_delta_pnl"].fillna(0.0)
        + 0.08 * ranking["val_fallback_trades"].fillna(0.0)
        - 35.0 * ranking["val_fallback_stop_rate"].fillna(0.0)
    )
    ranking = ranking.sort_values(
        ["selection_score_val_only", "val_fallback_pnl", "val_delta_pnl"],
        ascending=False,
    ).reset_index(drop=True)
    selected = ranking.iloc[0].to_dict()
    stop_focused = ranking.sort_values(
        ["oos_fallback_stop_rate", "oos_fallback_pnl", "oos_delta_pnl"],
        ascending=[True, False, False],
    ).iloc[0].to_dict()
    ranking.to_csv(OUT_DIR / "calibration_ranking.csv", index=False)
    for key, row in (("selected", selected), ("stop_focused_oos_diagnostic", stop_focused)):
        cand = str(row["candidate"])
        if cand in ledgers:
            val_ledger, oos_ledger = ledgers[cand]
            val_ledger.to_csv(OUT_DIR / f"{key}_validation_ledger.csv", index=False)
            oos_ledger.to_csv(OUT_DIR / f"{key}_oos_ledger.csv", index=False)
            val_ledger[val_ledger["sleeve"] == "fallback"].to_csv(OUT_DIR / f"{key}_validation_fallback_only_ledger.csv", index=False)
            oos_ledger[oos_ledger["sleeve"] == "fallback"].to_csv(OUT_DIR / f"{key}_oos_fallback_only_ledger.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "status": "probe_complete",
        "base_model_id": "omega1_2_3_ev_hgb_cash_sleeve_20260615",
        "selection_policy": "validation_only_no_oos_selection",
        "risk": risk.__dict__,
        "bundle_path": str(BUNDLE_PATH),
        "feature_count": int(len(feature_cols)),
        "label_diagnostics": label_diag,
        "stop_model_diagnostics": stop_diag,
        "residual_quantiles": {
            "long": {str(q): float(np.quantile(long_resid, q)) for q in (0.5, 0.6, 0.7, 0.8, 0.9)},
            "short": {str(q): float(np.quantile(short_resid, q)) for q in (0.5, 0.6, 0.7, 0.8, 0.9)},
        },
        "baseline": {"validation": base_val, "oos": base_oos},
        "selected_by_validation": selected,
        "stop_focused_oos_diagnostic": stop_focused,
        "top20": ranking.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "calibration_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": selected, "stop_focused_oos_diagnostic": stop_focused}, indent=2, ensure_ascii=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
