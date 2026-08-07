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


MODEL_ID = "omega1_2_3_ev_hgb_cash_sleeve_walkforward_20260615"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
BASELINE_ID = base.BASELINE_ID
RISK_NAME = "base_tp026_sl014_n0405_h192"
MIN_EDGE = 0.002
EV_MIN_GRID = (0.002, 0.003, 0.004, 0.005, 0.006)
SELECTED_EV_MIN = 0.002


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


def _concat_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not payloads:
        raise RuntimeError("no payloads to concatenate")
    return {
        "frame": pd.concat([p["frame"] for p in payloads], ignore_index=True),
        "dec": pd.concat([p["dec"] for p in payloads], ignore_index=True),
        "state": pd.concat([p["state"] for p in payloads], ignore_index=True),
        "fee": float(payloads[0]["fee"]),
        "slip": float(payloads[0]["slip"]),
    }


def _slice_payload(payload: dict[str, Any], start: str, end: str) -> dict[str, Any]:
    frame = payload["frame"].reset_index(drop=True)
    ts = pd.to_datetime(frame["timestamp"])
    mask = (ts >= pd.Timestamp(start)) & (ts < pd.Timestamp(end))
    idx = np.flatnonzero(mask.to_numpy())
    if len(idx) == 0:
        raise RuntimeError(f"empty payload slice: {start} to {end}")
    return {
        "frame": payload["frame"].iloc[idx].reset_index(drop=True),
        "dec": payload["dec"].iloc[idx].reset_index(drop=True),
        "state": payload["state"].iloc[idx].reset_index(drop=True),
        "fee": float(payload["fee"]),
        "slip": float(payload["slip"]),
    }


def _fallback_only(ledger: pd.DataFrame) -> dict[str, Any]:
    fb = ledger[ledger["sleeve"] == "fallback"].copy() if len(ledger) else ledger.copy()
    rets = fb["net_trade_return_pct"].astype(float).to_numpy() if len(fb) else np.asarray([], dtype=np.float64)
    eq = [1.0]
    for ret in rets:
        eq.append(eq[-1] * (1.0 + float(ret) / 100.0))
    eq_arr = np.asarray(eq, dtype=np.float64)
    dd = (eq_arr / np.maximum(np.maximum.accumulate(eq_arr), 1e-12) - 1.0) * 100.0
    wins = rets[rets > 0.0]
    losses = rets[rets <= 0.0]
    return {
        "trades": int(len(rets)),
        "pnl": float((eq_arr[-1] - 1.0) * 100.0),
        "mdd": float(dd.min()) if len(dd) else 0.0,
        "wr": float(np.mean(rets > 0.0)) if len(rets) else 0.0,
        "avg": float(np.mean(rets)) if len(rets) else 0.0,
        "pf": float(wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 1e-12 else None,
        "stop_rate": float(fb["exit_reason"].eq("fallback_stop_loss").mean()) if len(fb) else 0.0,
        "reasons": fb["exit_reason"].value_counts().sort_index().to_dict() if len(fb) else {},
    }


def _fit_ev(train_payload: dict[str, Any], test_payload: dict[str, Any], risk: base.SleeveRisk) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    x_train = up._enhanced_features(train_payload)
    x_test = up._enhanced_features(test_payload)
    labels, label_diag = up._label_table(train_payload, risk, MIN_EDGE)
    if len(labels) < 500:
        raise RuntimeError(f"not enough train labels: {len(labels)}")
    idx = labels["i"].to_numpy(dtype=np.int64)
    y_long = np.zeros(len(x_train), dtype=np.float64)
    y_short = np.zeros(len(x_train), dtype=np.float64)
    y_long[idx] = labels["long_net"].to_numpy(dtype=np.float64)
    y_short[idx] = labels["short_net"].to_numpy(dtype=np.float64)
    long_model = up._model("hgb", "regressor", 262000)
    short_model = up._model("hgb", "regressor", 262500)
    long_model.fit(x_train.iloc[idx].to_numpy(dtype=np.float64), y_long[idx])
    short_model.fit(x_train.iloc[idx].to_numpy(dtype=np.float64), y_short[idx])
    train_long = long_model.predict(x_train.to_numpy(dtype=np.float64)).astype(np.float64)
    train_short = short_model.predict(x_train.to_numpy(dtype=np.float64)).astype(np.float64)
    test_long = long_model.predict(x_test.to_numpy(dtype=np.float64)).astype(np.float64)
    test_short = short_model.predict(x_test.to_numpy(dtype=np.float64)).astype(np.float64)
    return train_long, train_short, test_long, test_short, label_diag


def _eval_fold(
    fold_name: str,
    train_payload: dict[str, Any],
    test_payload: dict[str, Any],
    cfg: base.repair.RunnerConfig,
    risk: base.SleeveRisk,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _train_long, _train_short, test_long, test_short, label_diag = _fit_ev(train_payload, test_payload, risk)
    base_m, base_ledger = base._simulate_combo(test_payload, cfg, None, None, None, 1.0)
    rows = []
    for ev_min in EV_MIN_GRID:
        action, conf = up._actions_from_ev(test_long, test_short, float(ev_min))
        combo_m, ledger = base._simulate_combo(test_payload, cfg, risk, action, conf, 0.0)
        fb = _fallback_only(ledger)
        row = {
            "fold": fold_name,
            "ev_min": float(ev_min),
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
        if abs(float(ev_min) - SELECTED_EV_MIN) < 1e-12:
            ledger.to_csv(OUT_DIR / f"{fold_name}_ev004_ledger.csv", index=False)
            ledger[ledger["sleeve"] == "fallback"].to_csv(OUT_DIR / f"{fold_name}_ev004_fallback_only_ledger.csv", index=False)
    base_ledger.to_csv(OUT_DIR / f"{fold_name}_baseline_ledger.csv", index=False)
    return rows, {"label_diag": label_diag, "baseline": base_m}


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
    all_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {"risk": asdict(risk), "selected_ev_min": SELECTED_EV_MIN, "folds": {}}
    for name, train_start, train_end, test_start, test_end in windows:
        print(json.dumps({"stage": "fold", "name": name, "train": [train_start, train_end], "test": [test_start, test_end]}, ensure_ascii=False), flush=True)
        train_payload = _slice_payload(full, train_start, train_end)
        test_payload = _slice_payload(full, test_start, test_end)
        rows, diag = _eval_fold(name, train_payload, test_payload, cfg, risk)
        diagnostics["folds"][name] = diag
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "walkforward_ev_min_grid.csv", index=False)
    selected_df = df[np.isclose(df["ev_min"], SELECTED_EV_MIN)].copy()
    aggregate = {
        "fold_count": int(len(selected_df)),
        "positive_delta_folds": int((selected_df["combo_delta_pnl"] > 0.0).sum()),
        "total_combo_delta_pnl_points": float(selected_df["combo_delta_pnl"].sum()),
        "total_fallback_pnl_points": float(selected_df["fallback_pnl"].sum()),
        "total_fallback_trades": int(selected_df["fallback_trades"].sum()),
        "mean_fallback_wr": float(selected_df["fallback_wr"].mean()) if len(selected_df) else 0.0,
        "mean_fallback_stop_rate": float(selected_df["fallback_stop_rate"].mean()) if len(selected_df) else 0.0,
        "folds": selected_df.to_dict(orient="records"),
    }
    redteam_blockers: list[str] = []
    if int(aggregate["positive_delta_folds"]) < 3:
        redteam_blockers.append("selected ev_min did not improve at least 3 of 4 walk-forward folds")
    if int(aggregate["total_fallback_trades"]) <= 0:
        redteam_blockers.append("selected ev_min produced no fallback trades in walk-forward")
    report = {
        "model_id": MODEL_ID,
        "candidate_model_id": "omega1_2_3_ev_hgb_cash_sleeve_20260615",
        "baseline_model_id": BASELINE_ID,
        "status": "walkforward_pass_shadow_candidate" if not redteam_blockers else "walkforward_fail",
        "method": "Monthly expanding-window walk-forward for EV-HGB cash sleeve. Train on prior months and test next month; fixed risk base_tp026_sl014_n0405_h192 and selected ev_min=0.004.",
        "selection_policy": "fixed_candidate_from_prior_oos_diagnostic; this script verifies stability only",
        "diagnostics": diagnostics,
        "selected_ev_min_aggregate": aggregate,
        "ev_min_grid": df.to_dict(orient="records"),
        "redteam_blockers": redteam_blockers,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "report": str(OUT_DIR / "report.json"),
            "grid": str(OUT_DIR / "walkforward_ev_min_grid.csv"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "status": report["status"], "aggregate": aggregate, "redteam_blockers": redteam_blockers}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
