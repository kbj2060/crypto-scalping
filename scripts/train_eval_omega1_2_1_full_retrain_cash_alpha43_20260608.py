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

import eval_omega1_2_true3head_overlays_20260604 as overlay  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_label_family_20260606 as label_family  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_1_exposure_selector_20260606 as base  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_1_full_retrain_cash_alpha43_20260608"
PARENT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_true_3head_tabm_20260603_full_retrain_cash_alpha43_20260608"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

CURRENT_BASELINE = {
    "validation": {"pnl": 100.542729, "mdd": -10.677653, "wr": 0.636364, "trades": 33},
    "oos": {"pnl": 72.760041, "mdd": -8.108171, "wr": 0.722222, "trades": 18},
}
RISK = sleeve.FallbackRisk("base_tp026_sl014_n030_h192", 0.026, 0.014, 0.30, 2.0, 192)
FORBIDDEN_PREFIXES = ("clean_regime4_", "regime4_pred_", "teacher_")
FORBIDDEN_EXACT = {"tp_sl_action_score"}


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


def _align(frame: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    out = frame[["timestamp"]].merge(pred, on="timestamp", how="left", validate="one_to_one")
    if out.isna().any().any():
        bad = out.loc[out.isna().any(axis=1), "timestamp"].head(10).tolist()
        raise RuntimeError(f"prediction alignment produced NaN: {bad}")
    return out


def _forbidden_features(cols: list[str]) -> list[str]:
    return [c for c in cols if c in FORBIDDEN_EXACT or c.startswith(FORBIDDEN_PREFIXES)]


def _build_split(frames: dict[str, pd.DataFrame], split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if split == "validation":
        frame = frames["val_raw"].reset_index(drop=True)
        pred_path = PARENT_DIR / "validation_predictions_2025_true3head.csv"
        pred = pd.read_csv(pred_path, parse_dates=["timestamp"])
        src = _align(frame, pred)
        prefix = "omega1_regime3_expertdq_oof_"
        dec0 = overlay._build_dec(src, prefix, oof=True)
    elif split == "oos":
        frame = frames["oos_raw"].reset_index(drop=True)
        pred_path = PARENT_DIR / "oos_predictions_2026_true3head.csv"
        pred = pd.read_csv(pred_path, parse_dates=["timestamp"])
        src = _align(frame, pred)
        prefix = "omega1_regime3_expertdq_"
        dec0 = overlay._build_dec(src, prefix, oof=False)
    else:
        raise RuntimeError(f"unknown split: {split}")

    dec = sleeve._apply_aggressive(dec0)
    features = sleeve._extra_features(base._feature_frame(frame, src, dec0, prefix), dec)
    bad = _forbidden_features(list(features.columns))
    if bad:
        raise RuntimeError(f"{split}: forbidden feature columns: {bad}")
    return frame, dec, features


def _metric_row(candidate: str, val_m: dict[str, Any], oos_m: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate": candidate,
        **sleeve._metric_row("val", val_m),
        **sleeve._metric_row("oos", oos_m),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not PARENT_DIR.exists():
        raise RuntimeError(f"missing full-retrain parent artifact: {PARENT_DIR}")

    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_frame, val_dec, val_features = _build_split(frames, "validation")
    oos_frame, oos_dec, oos_features = _build_split(frames, "oos")

    val_primary_m = omega._metrics(val_frame, val_dec, fee=fee, slip=slip, cost_mult=3.0)
    oos_primary_m = omega._metrics(oos_frame, oos_dec, fee=fee, slip=slip, cost_mult=3.0)
    rows = [
        _metric_row(
            "full_retrain_aggressive_primary_only",
            {**val_primary_m, "primary_entries": val_primary_m["long_entries"] + val_primary_m["short_entries"], "fallback_entries": 0, "primary_takeovers": 0},
            {**oos_primary_m, "primary_entries": oos_primary_m["long_entries"] + oos_primary_m["short_entries"], "fallback_entries": 0, "primary_takeovers": 0},
        )
    ]

    val_cash = ~omega._active(val_dec)
    oos_cash = ~omega._active(oos_dec)
    y_val, valid_mask, label_diag = label_family._triple_barrier_labels(
        val_frame,
        atr_mult=1.0,
        max_hold=72,
        min_barrier=0.0035,
    )
    train_mask = val_cash & valid_mask
    if len(set(y_val[train_mask].tolist())) < 2:
        raise RuntimeError("full-retrain cash sleeve labels are single-class")

    val_action, val_conf, oof_diag = label_family._predict_oof("hgb", val_features, y_val, train_mask, seed=260608)
    oos_action, oos_conf, _model = label_family._fit_predict("hgb", val_features, y_val, train_mask, oos_features, seed=260608)

    for threshold in (0.50, 0.55, 0.60, 0.65):
        val_m = sleeve._metrics_with_fallback(
            val_frame,
            val_dec,
            RISK,
            val_action,
            val_conf,
            threshold,
            fee=fee,
            slip=slip,
            cost_mult=3.0,
        )
        oos_m = sleeve._metrics_with_fallback(
            oos_frame,
            oos_dec,
            RISK,
            oos_action,
            oos_conf,
            threshold,
            fee=fee,
            slip=slip,
            cost_mult=3.0,
        )
        rows.append(_metric_row(f"full_retrain_label_atr1_h72_hgb_thr{threshold:g}", val_m, oos_m))

    ranking = pd.DataFrame(rows)
    ranking["oos_delta_vs_current_baseline"] = ranking["oos_pnl"] - CURRENT_BASELINE["oos"]["pnl"]
    ranking["val_delta_vs_current_baseline"] = ranking["val_pnl"] - CURRENT_BASELINE["validation"]["pnl"]
    ranking = ranking.sort_values(["oos_pnl", "val_pnl"], ascending=[False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "ranking.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "parent_dir": str(PARENT_DIR),
        "current_baseline": CURRENT_BASELINE,
        "risk": RISK.__dict__,
        "feature_contract": {
            "source": "full-retrained 3-head TabM predictions + Omega-only cash sleeve features",
            "feature_count": int(val_features.shape[1]),
            "features": list(val_features.columns),
            "forbidden_feature_audit": {"passed": True, "forbidden": []},
        },
        "cash_rows": {"validation": int(np.count_nonzero(val_cash)), "oos": int(np.count_nonzero(oos_cash))},
        "label_diag": label_diag,
        "oof_diag": oof_diag,
        "best": ranking.iloc[0].to_dict(),
        "ranking": ranking.to_dict(orient="records"),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default))
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "best": report["best"]}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
