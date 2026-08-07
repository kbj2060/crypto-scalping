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

import train_eval_omega1_2_3_cash_sleeve_upgrade_20260615 as sleeve_up  # noqa: E402
import train_eval_omega1_2_5_parent_numeric_vs_rlq_20260616 as rlq_probe  # noqa: E402


MODEL_ID = "omega1_2_6_ev_lower_bound_rlq_agreement_20260616"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
RISK_NAME = "base_tp026_sl014_n0405_h192"
MIN_EDGE = 0.002


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


def _fit_ev_lower_bound(
    x_val: pd.DataFrame,
    x_oos: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    cal_q: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    idx = labels["i"].to_numpy(dtype=np.int64)
    y_long = np.zeros(len(x_val), dtype=np.float64)
    y_short = np.zeros(len(x_val), dtype=np.float64)
    y_long[idx] = labels["long_net"].to_numpy(dtype=np.float64)
    y_short[idx] = labels["short_net"].to_numpy(dtype=np.float64)
    val_long, oos_long, long_diag = sleeve_up._fit_predict_regressor("hgb", x_val, y_long, idx, x_oos, seed=262000)
    val_short, oos_short, short_diag = sleeve_up._fit_predict_regressor("hgb", x_val, y_short, idx, x_oos, seed=262500)

    train_long_pred = sleeve_up._model("hgb", "regressor", 262000)
    train_short_pred = sleeve_up._model("hgb", "regressor", 262500)
    train_long_pred.fit(x_val.iloc[idx].to_numpy(dtype=np.float64), y_long[idx])
    train_short_pred.fit(x_val.iloc[idx].to_numpy(dtype=np.float64), y_short[idx])
    long_resid_q = float(np.quantile(np.abs(y_long[idx] - train_long_pred.predict(x_val.iloc[idx].to_numpy(dtype=np.float64))), cal_q))
    short_resid_q = float(np.quantile(np.abs(y_short[idx] - train_short_pred.predict(x_val.iloc[idx].to_numpy(dtype=np.float64))), cal_q))
    diag = {
        "long_model": long_diag,
        "short_model": short_diag,
        "calibration": {
            "quantile": float(cal_q),
            "long_abs_residual_q": long_resid_q,
            "short_abs_residual_q": short_resid_q,
        },
    }
    return val_long - long_resid_q, val_short - short_resid_q, oos_long - long_resid_q, oos_short - short_resid_q, diag


def _fit_rlq_advantage(
    x_val: pd.DataFrame,
    x_oos: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    idx = labels["i"].to_numpy(dtype=np.int64)
    long_target = np.full(len(x_val), np.nan, dtype=np.float64)
    short_target = np.full(len(x_val), np.nan, dtype=np.float64)
    long_target[idx] = labels["long_adv"].to_numpy(dtype=np.float64)
    short_target[idx] = labels["short_adv"].to_numpy(dtype=np.float64)
    train_idx = idx[np.isfinite(long_target[idx]) & np.isfinite(short_target[idx])]
    val_long, oos_long, long_diag = sleeve_up._fit_predict_regressor("hgb", x_val, long_target, train_idx, x_oos, seed=268101)
    val_short, oos_short, short_diag = sleeve_up._fit_predict_regressor("hgb", x_val, short_target, train_idx, x_oos, seed=268501)
    diag = {
        "train_rows": int(len(train_idx)),
        "long_model": long_diag,
        "short_model": short_diag,
    }
    return val_long, val_short, oos_long, oos_short, diag


def _apply_rlq_filter(
    ev_action: np.ndarray,
    ev_conf: np.ndarray,
    rlq_long: np.ndarray,
    rlq_short: np.ndarray,
    *,
    q_min: float,
    margin_min: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    long_ok = (ev_action == rlq_probe.ACTION_LONG) & (rlq_long > float(q_min)) & ((rlq_long - rlq_short) >= float(margin_min))
    short_ok = (ev_action == rlq_probe.ACTION_SHORT) & (rlq_short > float(q_min)) & ((rlq_short - rlq_long) >= float(margin_min))
    keep = long_ok | short_ok
    out_action = np.where(keep, ev_action, rlq_probe.ACTION_CASH).astype(np.int64)
    rlq_support = np.where(ev_action == rlq_probe.ACTION_LONG, rlq_long, np.where(ev_action == rlq_probe.ACTION_SHORT, rlq_short, 0.0))
    out_conf = np.where(keep, np.minimum(ev_conf, np.clip((rlq_support - float(q_min)) / 0.02, 0.0, 1.0)), 0.0).astype(np.float64)
    active = np.isin(ev_action, [rlq_probe.ACTION_LONG, rlq_probe.ACTION_SHORT])
    diag = {
        "ev_active_rows": int(active.sum()),
        "kept_rows": int(keep.sum()),
        "veto_rows": int((active & ~keep).sum()),
        "keep_rate_on_ev_active": float(keep.sum() / max(active.sum(), 1)),
        "q_min": float(q_min),
        "margin_min": float(margin_min),
    }
    return out_action, out_conf, diag


def _row(
    candidate: str,
    family: str,
    cal_q: float,
    ev_min: float,
    q_min: float | None,
    margin_min: float | None,
    val_m: dict[str, Any],
    val_ledger: pd.DataFrame,
    oos_m: dict[str, Any],
    oos_ledger: pd.DataFrame,
    base_val: dict[str, Any],
    base_oos: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "candidate": candidate,
        "family": family,
        "risk": RISK_NAME,
        "cal_q": float(cal_q),
        "ev_min": float(ev_min),
        "q_min": None if q_min is None else float(q_min),
        "margin_min": None if margin_min is None else float(margin_min),
    }
    row.update(sleeve_up._row("val", val_m, val_ledger))
    row.update(sleeve_up._row("oos", oos_m, oos_ledger))
    row["val_delta_pnl"] = float(row["val_pnl"] - float(base_val["pnl"]))
    row["oos_delta_pnl"] = float(row["oos_pnl"] - float(base_oos["pnl"]))
    return row


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _runner_cfg()
    data = rlq_probe.base.legacy_runner._build()
    risk = [r for r in rlq_probe.base.RISKS if r.name == RISK_NAME][0]
    x_val = sleeve_up._enhanced_features(data["validation"])
    x_oos = sleeve_up._enhanced_features(data["oos"])
    if list(x_val.columns) != list(x_oos.columns):
        raise RuntimeError("validation/oos sleeve feature columns mismatch")

    base_val, base_val_ledger = rlq_probe.base._simulate_combo(data["validation"], cfg, None, None, None, 1.0)
    base_oos, base_oos_ledger = rlq_probe.base._simulate_combo(data["oos"], cfg, None, None, None, 1.0)
    base_val_ledger.to_csv(OUT_DIR / "baseline_validation_ledger.csv", index=False)
    base_oos_ledger.to_csv(OUT_DIR / "baseline_oos_ledger.csv", index=False)

    ev_labels, ev_label_diag = sleeve_up._label_table(data["validation"], risk, MIN_EDGE)
    critic, router, dsac_meta = rlq_probe._load_dsac_critic()
    rlq_labels, rlq_label_diag = rlq_probe._rlq_labels(data["validation"], critic, router)
    rlq_val_long, rlq_val_short, rlq_oos_long, rlq_oos_short, rlq_fit_diag = _fit_rlq_advantage(x_val, x_oos, rlq_labels)

    diagnostics: dict[str, Any] = {
        "mode": "deterministic_ev_lower_bound_cash_sleeve_with_rlq_agreement_veto",
        "baseline_model_id": rlq_probe.base.BASELINE_ID,
        "selected_tp_runner_config": asdict(cfg),
        "risk": asdict(risk),
        "min_edge": float(MIN_EDGE),
        "feature_count": int(x_val.shape[1]),
        "features": list(x_val.columns),
        "ev_labels": ev_label_diag,
        "rlq_source": dsac_meta,
        "rlq_labels": rlq_label_diag,
        "rlq_fit": rlq_fit_diag,
        "baseline": {"validation": base_val, "oos": base_oos},
    }

    rows: list[dict[str, Any]] = [
        {
            "candidate": "baseline_tp_runner_clean_repair",
            "family": "baseline",
            "risk": "none",
            "cal_q": None,
            "ev_min": None,
            "q_min": None,
            "margin_min": None,
            **sleeve_up._row("val", base_val, base_val_ledger),
            **sleeve_up._row("oos", base_oos, base_oos_ledger),
            "val_delta_pnl": 0.0,
            "oos_delta_pnl": 0.0,
        }
    ]
    ledgers: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for cal_q in (0.50, 0.65, 0.80):
        ev_val_long, ev_val_short, ev_oos_long, ev_oos_short, ev_fit_diag = _fit_ev_lower_bound(x_val, x_oos, ev_labels, cal_q=cal_q)
        diagnostics[f"ev_lower_bound_cal_q{cal_q:.2f}"] = ev_fit_diag
        for ev_min in (0.001, 0.002, 0.003, 0.004):
            val_ev_a, val_ev_c = sleeve_up._actions_from_ev(ev_val_long, ev_val_short, ev_min)
            oos_ev_a, oos_ev_c = sleeve_up._actions_from_ev(ev_oos_long, ev_oos_short, ev_min)

            val_m, val_ledger = rlq_probe.base._simulate_combo(data["validation"], cfg, risk, val_ev_a, val_ev_c, 0.0)
            oos_m, oos_ledger = rlq_probe.base._simulate_combo(data["oos"], cfg, risk, oos_ev_a, oos_ev_c, 0.0)
            ev_name = f"ev_lower_bound_cal{cal_q:.2f}_ev{ev_min:.3f}"
            rows.append(_row(ev_name, "ev_lower_bound_only", cal_q, ev_min, None, None, val_m, val_ledger, oos_m, oos_ledger, base_val, base_oos))
            ledgers[ev_name] = (val_ledger, oos_ledger)

            for q_min in (-0.002, 0.0, 0.001, 0.002):
                for margin_min in (0.0, 0.001, 0.002):
                    val_a, val_c, val_filter_diag = _apply_rlq_filter(
                        val_ev_a,
                        val_ev_c,
                        rlq_val_long,
                        rlq_val_short,
                        q_min=q_min,
                        margin_min=margin_min,
                    )
                    oos_a, oos_c, oos_filter_diag = _apply_rlq_filter(
                        oos_ev_a,
                        oos_ev_c,
                        rlq_oos_long,
                        rlq_oos_short,
                        q_min=q_min,
                        margin_min=margin_min,
                    )
                    cand = f"ev_lower_bound_cal{cal_q:.2f}_ev{ev_min:.3f}_rlq_q{q_min:.3f}_m{margin_min:.3f}"
                    diagnostics[f"{cand}_filter"] = {"validation": val_filter_diag, "oos": oos_filter_diag}
                    val_m, val_ledger = rlq_probe.base._simulate_combo(data["validation"], cfg, risk, val_a, val_c, 0.0)
                    oos_m, oos_ledger = rlq_probe.base._simulate_combo(data["oos"], cfg, risk, oos_a, oos_c, 0.0)
                    rows.append(_row(cand, "ev_lower_bound_rlq_agreement_veto", cal_q, ev_min, q_min, margin_min, val_m, val_ledger, oos_m, oos_ledger, base_val, base_oos))
                    ledgers[cand] = (val_ledger, oos_ledger)

    ranking = pd.DataFrame(rows)
    ranking["selection_score_val_only"] = (
        ranking["val_fallback_pnl"].fillna(0.0)
        + 0.25 * ranking["val_delta_pnl"].fillna(0.0)
        + 0.08 * ranking["val_fallback_trades"].fillna(0.0)
        - 30.0 * ranking["val_fallback_stop_rate"].fillna(0.0)
    )
    ranking = ranking.sort_values(["selection_score_val_only", "val_fallback_pnl", "val_delta_pnl"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "ev_lower_bound_rlq_agreement_ranking.csv", index=False)
    hybrid_ranking = ranking[ranking["family"].eq("ev_lower_bound_rlq_agreement_veto")].copy()
    selected = hybrid_ranking.iloc[0].to_dict() if len(hybrid_ranking) else ranking.iloc[0].to_dict()
    best_oos = (
        hybrid_ranking.sort_values(["oos_fallback_pnl", "oos_delta_pnl", "oos_fallback_trades"], ascending=False).iloc[0].to_dict()
        if len(hybrid_ranking)
        else ranking.sort_values(["oos_fallback_pnl", "oos_delta_pnl", "oos_fallback_trades"], ascending=False).iloc[0].to_dict()
    )
    best_control = ranking[~ranking["family"].eq("ev_lower_bound_rlq_agreement_veto")].head(5).to_dict(orient="records")

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
        redteam_blockers.append("no hybrid candidates produced")
    if not ranking["family"].eq("ev_lower_bound_rlq_agreement_veto").any():
        redteam_blockers.append("no RLQ agreement/veto candidates produced")

    report = {
        "model_id": MODEL_ID,
        "status": "redteam_pass_hybrid_eval" if not redteam_blockers else "redteam_fail",
        "method": "Existing deterministic EV lower-bound cash sleeve remains the action owner. RLQ does not create fallback entries; it only agrees with the EV side or vetoes it using DSAC critic advantage predictions.",
        "selection_policy": "hybrid_validation_only_no_oos_selection; EV-only rows are controls, OOS is diagnostic",
        "redteam_policy": "FAIL is limited to feature/data/artifact contract defects or no candidate generation. PnL is diagnostic.",
        "diagnostics": diagnostics,
        "baseline": {"validation": base_val, "oos": base_oos},
        "selected_by_validation": selected,
        "best_by_oos_diagnostic": best_oos,
        "best_ev_only_controls": best_control,
        "top20_hybrid": hybrid_ranking.head(20).to_dict(orient="records"),
        "top20_all_including_controls": ranking.head(20).to_dict(orient="records"),
        "redteam_pass": not redteam_blockers,
        "redteam_blockers": redteam_blockers,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "ev_lower_bound_rlq_agreement_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "status": report["status"], "selected": selected, "best_oos_diagnostic": best_oos}, indent=2, ensure_ascii=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
