#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_2_tp_runner_cash_sleeve_20260615 as base  # noqa: E402
import train_eval_omega1_2_3_cash_sleeve_upgrade_20260615 as up  # noqa: E402


MODEL_ID = "omega1_2_4_numeric_utility_cash_sleeve_20260616"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
LIVE_OUT_DIR = ROOT / "data/ensemble/supervised" / MODEL_ID
RISK_NAME = "base_tp026_sl014_n0405_h192"
UTILITY_THRESHOLDS = (0.000, 0.001, 0.002, 0.003, 0.004)


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


def _runner_cfg() -> base.repair.RunnerConfig:
    baseline_report = json.loads(base.BASELINE_REPORT.read_text(encoding="utf-8"))
    sc = baseline_report["selected_config"]
    return base.repair.RunnerConfig(
        int(sc["candidate_id"]),
        str(sc["mode"]),
        float(sc["quality_min"]),
        float(sc["extend_mult"]),
        float(sc["floor_frac"]),
        int(sc["max_extensions"]),
    )


def _utility_label_table(
    payload: dict[str, Any],
    risk: base.SleeveRisk,
    *,
    stop_penalty: float,
    mae_penalty: float,
    time_penalty: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = payload["frame"].reset_index(drop=True)
    dec = payload["dec"].reset_index(drop=True)
    arrays = base.repair._arrays(frame)
    active = base._active(dec)
    fee_eff = float(payload["fee"]) * 3.0
    slip_eff = float(payload["slip"]) * 3.0
    rows: list[dict[str, Any]] = []
    max_hold = max(int(risk.max_hold_bars), 1)
    for i in np.flatnonzero(~active):
        if i >= len(frame) - int(risk.max_hold_bars) - 3:
            continue
        long_d = up._simulate_label_detail(frame, arrays, active, int(i), 1, risk, fee_eff, slip_eff)
        short_d = up._simulate_label_detail(frame, arrays, active, int(i), -1, risk, fee_eff, slip_eff)

        def utility(d: dict[str, Any]) -> float:
            adverse = abs(min(float(d["mae"]), 0.0))
            time_frac = min(float(d["bars_to_takeover"]) / float(max_hold), 1.0)
            return float(d["net"]) - float(stop_penalty) * int(d["stop"]) - float(mae_penalty) * adverse - float(time_penalty) * time_frac

        long_u = utility(long_d)
        short_u = utility(short_d)
        best_u = max(long_u, short_u, 0.0)
        rows.append(
            {
                "i": int(i),
                "long_net": float(long_d["net"]),
                "short_net": float(short_d["net"]),
                "long_utility": long_u,
                "short_utility": short_u,
                "cash_utility": 0.0,
                "best_utility": float(best_u),
                "long_stop": int(long_d["stop"]),
                "short_stop": int(short_d["stop"]),
                "long_mae": float(long_d["mae"]),
                "short_mae": float(short_d["mae"]),
                "long_reason": str(long_d["reason"]),
                "short_reason": str(short_d["reason"]),
            }
        )
    labels = pd.DataFrame(rows)
    diag = {
        "valid_cash_rows": int(len(labels)),
        "positive_long_utility": int((labels["long_utility"] > 0.0).sum()) if len(labels) else 0,
        "positive_short_utility": int((labels["short_utility"] > 0.0).sum()) if len(labels) else 0,
        "best_utility_mean": float(labels["best_utility"].mean()) if len(labels) else 0.0,
        "long_stop_rate": float(labels["long_stop"].mean()) if len(labels) else 0.0,
        "short_stop_rate": float(labels["short_stop"].mean()) if len(labels) else 0.0,
        "stop_penalty": float(stop_penalty),
        "mae_penalty": float(mae_penalty),
        "time_penalty": float(time_penalty),
    }
    return labels, diag


def _actions_from_utility(long_u: np.ndarray, short_u: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    best_long = long_u >= short_u
    best = np.where(best_long, long_u, short_u)
    action = np.where(best > float(threshold), np.where(best_long, base.ACTION_LONG, base.ACTION_SHORT), base.ACTION_CASH).astype(np.int64)
    conf = np.clip((best - float(threshold)) / 0.02, 0.0, 1.0).astype(np.float64)
    return action, conf


def _metric_row(
    candidate: str,
    utility_cfg: dict[str, float],
    threshold: float,
    val_m: dict[str, Any],
    val_ledger: pd.DataFrame,
    oos_m: dict[str, Any],
    oos_ledger: pd.DataFrame,
    base_val: dict[str, Any],
    base_oos: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "candidate": candidate,
        "utility_cfg": dict(utility_cfg),
        "threshold": float(threshold),
    }
    row.update(up._row("val", val_m, val_ledger))
    row.update(up._row("oos", oos_m, oos_ledger))
    row["val_delta_pnl"] = float(row["val_pnl"] - float(base_val["pnl"]))
    row["oos_delta_pnl"] = float(row["oos_pnl"] - float(base_oos["pnl"]))
    return row


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _runner_cfg()
    risk = [r for r in base.RISKS if r.name == RISK_NAME][0]
    data = base.legacy_runner._build()
    x_val = up._enhanced_features(data["validation"])
    x_oos = up._enhanced_features(data["oos"])
    base_val, base_val_ledger = base._simulate_combo(data["validation"], cfg, None, None, None, 1.0)
    base_oos, base_oos_ledger = base._simulate_combo(data["oos"], cfg, None, None, None, 1.0)
    base_val_ledger.to_csv(OUT_DIR / "validation_baseline_replay_ledger.csv", index=False)
    base_oos_ledger.to_csv(OUT_DIR / "oos_baseline_replay_ledger.csv", index=False)

    rows: list[dict[str, Any]] = []
    ledgers: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    diagnostics: dict[str, Any] = {
        "feature_count": int(x_val.shape[1]),
        "features": list(x_val.columns),
        "risk": asdict(risk),
        "selected_tp_runner_config": asdict(cfg),
    }
    utility_grid = [
        {"stop_penalty": 0.0, "mae_penalty": 0.0, "time_penalty": 0.0},
        {"stop_penalty": 0.003, "mae_penalty": 0.0, "time_penalty": 0.0},
        {"stop_penalty": 0.006, "mae_penalty": 0.0, "time_penalty": 0.0},
        {"stop_penalty": 0.003, "mae_penalty": 0.20, "time_penalty": 0.0},
        {"stop_penalty": 0.003, "mae_penalty": 0.20, "time_penalty": 0.001},
    ]
    selected_models: dict[str, Any] = {}
    for cfg_id, utility_cfg in enumerate(utility_grid):
        labels, label_diag = _utility_label_table(data["validation"], risk, **utility_cfg)
        diagnostics[f"utility_cfg_{cfg_id}"] = {"config": utility_cfg, "labels": label_diag}
        idx = labels["i"].to_numpy(dtype=np.int64)
        if len(idx) < 500:
            continue
        y_long = np.zeros(len(x_val), dtype=np.float64)
        y_short = np.zeros(len(x_val), dtype=np.float64)
        y_long[idx] = labels["long_utility"].to_numpy(dtype=np.float64)
        y_short[idx] = labels["short_utility"].to_numpy(dtype=np.float64)
        val_long, oos_long, long_diag = up._fit_predict_regressor("hgb", x_val, y_long, idx, x_oos, seed=264001 + cfg_id * 10)
        val_short, oos_short, short_diag = up._fit_predict_regressor("hgb", x_val, y_short, idx, x_oos, seed=264002 + cfg_id * 10)
        diagnostics[f"utility_cfg_{cfg_id}"]["long_model"] = long_diag
        diagnostics[f"utility_cfg_{cfg_id}"]["short_model"] = short_diag
        for threshold in UTILITY_THRESHOLDS:
            val_a, val_c = _actions_from_utility(val_long, val_short, threshold)
            oos_a, oos_c = _actions_from_utility(oos_long, oos_short, threshold)
            val_m, val_ledger = base._simulate_combo(data["validation"], cfg, risk, val_a, val_c, 0.0)
            oos_m, oos_ledger = base._simulate_combo(data["oos"], cfg, risk, oos_a, oos_c, 0.0)
            candidate = f"utility_cfg{cfg_id}_thr{threshold:.3f}"
            rows.append(_metric_row(candidate, utility_cfg, threshold, val_m, val_ledger, oos_m, oos_ledger, base_val, base_oos))
            ledgers[candidate] = (val_ledger, oos_ledger)
        selected_models[f"utility_cfg{cfg_id}"] = {
            "utility_cfg": utility_cfg,
            "long_seed": 264001 + cfg_id * 10,
            "short_seed": 264002 + cfg_id * 10,
        }

    ranking = pd.DataFrame(rows)
    ranking["selection_score_val_only"] = (
        ranking["val_fallback_pnl"].fillna(0.0)
        + 0.25 * ranking["val_delta_pnl"].fillna(0.0)
        + 0.08 * ranking["val_fallback_trades"].fillna(0.0)
        - 30.0 * ranking["val_fallback_stop_rate"].fillna(0.0)
    )
    ranking = ranking.sort_values(
        ["selection_score_val_only", "val_fallback_pnl", "val_delta_pnl"],
        ascending=False,
    ).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "numeric_utility_ranking.csv", index=False)
    selected = ranking.iloc[0].to_dict()
    best_oos = ranking.sort_values(["oos_delta_pnl", "oos_fallback_pnl"], ascending=False).iloc[0].to_dict()
    for prefix, row in (("selected", selected), ("best_oos_diagnostic", best_oos)):
        cand = str(row["candidate"])
        if cand in ledgers:
            val_ledger, oos_ledger = ledgers[cand]
            val_ledger.to_csv(OUT_DIR / f"{prefix}_validation_ledger.csv", index=False)
            oos_ledger.to_csv(OUT_DIR / f"{prefix}_oos_ledger.csv", index=False)
            val_ledger[val_ledger["sleeve"] == "fallback"].to_csv(OUT_DIR / f"{prefix}_validation_fallback_only_ledger.csv", index=False)
            oos_ledger[oos_ledger["sleeve"] == "fallback"].to_csv(OUT_DIR / f"{prefix}_oos_fallback_only_ledger.csv", index=False)

    redteam_blockers: list[str] = []
    if len(ranking) == 0:
        redteam_blockers.append("no numeric utility candidates produced")
    if list(x_val.columns) != list(x_oos.columns):
        redteam_blockers.append("validation/oos feature columns mismatch")
    forbidden = [c for c in x_val.columns if c in base.FORBIDDEN_FEATURE_EXACT or c.startswith(base.FORBIDDEN_FEATURE_PREFIXES)]
    if forbidden:
        redteam_blockers.append(f"forbidden feature columns present: {forbidden[:20]}")

    report = {
        "model_id": MODEL_ID,
        "status": "redteam_pass_numeric_utility_probe" if not redteam_blockers else "redteam_fail",
        "baseline_model_id": base.BASELINE_ID,
        "method": "Numeric utility regression cash sleeve. No hard action labels are trained; long and short utility are continuous labels relative to cash_utility=0.",
        "selection_policy": "validation_only_no_oos_selection",
        "redteam_policy": "FAIL is limited to logical defects, data/feature contract violations, forbidden feature leakage, or failed candidate generation. OOS is diagnostic.",
        "diagnostics": diagnostics,
        "baseline": {"validation": base_val, "oos": base_oos},
        "selected_by_validation": selected,
        "best_by_oos_diagnostic": best_oos,
        "top20": ranking.head(20).to_dict(orient="records"),
        "redteam_pass": not redteam_blockers,
        "redteam_blockers": redteam_blockers,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "numeric_utility_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "status": report["status"], "selected": selected, "best_oos_diagnostic": best_oos}, indent=2, ensure_ascii=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
