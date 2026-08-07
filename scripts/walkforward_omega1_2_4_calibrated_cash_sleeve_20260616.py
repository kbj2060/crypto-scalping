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

import train_eval_omega1_2_2_tp_runner_cash_sleeve_20260615 as base  # noqa: E402
import train_eval_omega1_2_3_cash_sleeve_upgrade_20260615 as up  # noqa: E402
from walkforward_omega1_2_3_ev_hgb_cash_sleeve_20260615 import (  # noqa: E402
    _concat_payloads,
    _fallback_only,
    _slice_payload,
)


MODEL_ID = "omega1_2_4_calibrated_cash_sleeve_walkforward_20260616"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
BASELINE_ID = base.BASELINE_ID
RISK_NAME = "base_tp026_sl014_n0405_h192"
MIN_EDGE = 0.002
CAL_Q = 0.50
CAL_EV_MIN = 0.002


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


def _fit_ev_calibrated(
    train_payload: dict[str, Any],
    test_payload: dict[str, Any],
    risk: base.SleeveRisk,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x_train = up._enhanced_features(train_payload)
    x_test = up._enhanced_features(test_payload)
    labels, label_diag = up._label_table(train_payload, risk, MIN_EDGE)
    if len(labels) < 500:
        raise RuntimeError(f"not enough train labels: {len(labels)}")
    idx = labels["i"].to_numpy(dtype=np.int64)
    y_long = labels["long_net"].to_numpy(dtype=np.float64)
    y_short = labels["short_net"].to_numpy(dtype=np.float64)
    long_model = up._model("hgb", "regressor", 262000)
    short_model = up._model("hgb", "regressor", 262500)
    long_model.fit(x_train.iloc[idx].to_numpy(dtype=np.float64), y_long)
    short_model.fit(x_train.iloc[idx].to_numpy(dtype=np.float64), y_short)
    train_long = long_model.predict(x_train.iloc[idx].to_numpy(dtype=np.float64)).astype(np.float64)
    train_short = short_model.predict(x_train.iloc[idx].to_numpy(dtype=np.float64)).astype(np.float64)
    long_q = float(np.quantile(np.abs(y_long - train_long), CAL_Q))
    short_q = float(np.quantile(np.abs(y_short - train_short), CAL_Q))
    test_long = long_model.predict(x_test.to_numpy(dtype=np.float64)).astype(np.float64) - long_q
    test_short = short_model.predict(x_test.to_numpy(dtype=np.float64)).astype(np.float64) - short_q
    diag = {
        "label_diag": label_diag,
        "calibration": {
            "quantile": float(CAL_Q),
            "long_abs_residual_q": long_q,
            "short_abs_residual_q": short_q,
            "ev_min": float(CAL_EV_MIN),
        },
    }
    return test_long, test_short, diag


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_report = json.loads(base.BASELINE_REPORT.read_text(encoding="utf-8"))
    sc = baseline_report["selected_config"]
    cfg = base.repair.RunnerConfig(
        int(sc["candidate_id"]),
        str(sc["mode"]),
        float(sc["quality_min"]),
        float(sc["extend_mult"]),
        float(sc["floor_frac"]),
        int(sc["max_extensions"]),
    )
    risk = [r for r in base.RISKS if r.name == RISK_NAME][0]
    raw = base.legacy_runner._build()
    full = _concat_payloads([raw["validation"], raw["oos"]])
    windows = [
        ("wf_2025_10_to_2025_11", "2025-10-01", "2025-11-01", "2025-11-01", "2025-12-01"),
        ("wf_2025_10_11_to_2025_12", "2025-10-01", "2025-12-01", "2025-12-01", "2026-01-01"),
        ("wf_2025_q4_to_2026_01", "2025-10-01", "2026-01-01", "2026-01-01", "2026-02-01"),
        ("wf_2025_q4_2026_01_to_2026_02", "2025-10-01", "2026-02-01", "2026-02-01", "2026-03-01"),
    ]
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {"risk": asdict(risk), "folds": {}}
    for name, train_start, train_end, test_start, test_end in windows:
        print(json.dumps({"stage": "fold", "name": name, "train": [train_start, train_end], "test": [test_start, test_end]}, ensure_ascii=False), flush=True)
        train_payload = _slice_payload(full, train_start, train_end)
        test_payload = _slice_payload(full, test_start, test_end)
        test_long, test_short, diag = _fit_ev_calibrated(train_payload, test_payload, risk)
        base_m, base_ledger = base._simulate_combo(test_payload, cfg, None, None, None, 1.0)
        action, conf = up._actions_from_ev(test_long, test_short, CAL_EV_MIN)
        combo_m, ledger = base._simulate_combo(test_payload, cfg, risk, action, conf, 0.0)
        fb = _fallback_only(ledger)
        row = {
            "fold": name,
            "baseline_pnl": float(base_m["pnl"]),
            "baseline_mdd": float(base_m["mdd"]),
            "baseline_trades": int(base_m["trades"]),
            "combo_pnl": float(combo_m["pnl"]),
            "combo_delta_pnl": float(combo_m["pnl"] - base_m["pnl"]),
            "combo_mdd": float(combo_m["mdd"]),
            "combo_trades": int(combo_m["trades"]),
            "fallback": fb,
        }
        row.update({f"fallback_{k}": v for k, v in fb.items() if k != "reasons"})
        rows.append(row)
        diagnostics["folds"][name] = diag
        base_ledger.to_csv(OUT_DIR / f"{name}_baseline_ledger.csv", index=False)
        ledger.to_csv(OUT_DIR / f"{name}_cal_q50_ev002_ledger.csv", index=False)
        ledger[ledger["sleeve"] == "fallback"].to_csv(OUT_DIR / f"{name}_cal_q50_ev002_fallback_only_ledger.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "walkforward_cal_q50_ev002.csv", index=False)
    aggregate = {
        "fold_count": int(len(df)),
        "positive_delta_folds": int((df["combo_delta_pnl"] > 0.0).sum()),
        "total_combo_delta_pnl_points": float(df["combo_delta_pnl"].sum()),
        "total_fallback_pnl_points": float(df["fallback_pnl"].sum()),
        "total_fallback_trades": int(df["fallback_trades"].sum()),
        "mean_fallback_wr": float(df["fallback_wr"].mean()) if len(df) else 0.0,
        "mean_fallback_stop_rate": float(df["fallback_stop_rate"].mean()) if len(df) else 0.0,
        "folds": df.to_dict(orient="records"),
    }
    blockers: list[str] = []
    if int(aggregate["positive_delta_folds"]) < 3:
        blockers.append("calibrated candidate did not improve at least 3 of 4 walk-forward folds")
    if int(aggregate["total_fallback_trades"]) <= 0:
        blockers.append("calibrated candidate produced no fallback trades in walk-forward")
    report = {
        "model_id": MODEL_ID,
        "candidate": "cal_q50_ev0.002",
        "status": "walkforward_pass_shadow_candidate" if not blockers else "walkforward_fail",
        "baseline_model_id": BASELINE_ID,
        "base_candidate_model_id": "omega1_2_3_ev_hgb_cash_sleeve_20260615",
        "method": "Monthly expanding-window walk-forward for median absolute residual EV lower-bound calibration. Calibration quantile and EV threshold are fixed from prior validation probe.",
        "selection_policy": "fixed_cal_q50_ev002_from_validation_probe; OOS and WF are diagnostics",
        "diagnostics": diagnostics,
        "aggregate": aggregate,
        "redteam_blockers": blockers,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "report": str(OUT_DIR / "report.json"),
            "grid": str(OUT_DIR / "walkforward_cal_q50_ev002.csv"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "status": report["status"], "aggregate": aggregate, "redteam_blockers": blockers}, indent=2, ensure_ascii=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
