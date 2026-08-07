#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_1_cash_fallback_label_family_20260606 as label_family  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_1_full_retrain_cash_alpha43_20260608 as full  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_1_cash_alpha43_exit_feature_ablation_20260609"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
SEEDS = (260000, 260001, 260002, 260003, 260004, 260005, 260006, 260007, 260008, 260009, 260608, 260780)
THRESHOLD = 0.55
RISK = sleeve.FallbackRisk("tp026_sl014_n0.30_h192", 0.026, 0.014, 0.30, 2.0, 192)


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


def _load_models(device: torch.device) -> tuple[dict[str, tuple[threehead.ThreeHeadTabM, dict[str, Any]]], list[str]]:
    bundle_path = full.PARENT_DIR / "true_3head_tabm_bundle.pt"
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    return threehead._load_payloads(bundle["models"], device=device), list(bundle["base_cols"])


def _entry_exit_features(
    features: pd.DataFrame,
    dec: pd.DataFrame,
    loaded: dict[str, tuple[threehead.ThreeHeadTabM, dict[str, Any]]],
    base_cols: list[str],
    device: torch.device,
) -> pd.DataFrame:
    x = features.reindex(columns=base_cols, fill_value=0.0).copy()
    for col in threehead.POS_COLS:
        x[col] = 0.0
    out = np.zeros(len(features), dtype=np.float64)
    expert_series = dec["router_expert"].astype(str).replace({"chop_expert": "chop"})
    for expert, (model, scaler) in loaded.items():
        idx = np.flatnonzero(expert_series.eq(expert).to_numpy())
        if len(idx) == 0:
            continue
        pred = threehead._predict_loaded_exit(model, scaler, x.iloc[idx], device=device)
        out[idx] = pred[:, 1]
    enriched = features.copy()
    enriched["exit_head_entry_risk"] = out
    enriched["exit_head_entry_safe"] = 1.0 - out
    enriched["exit_head_entry_risk_x_quality"] = out * pd.to_numeric(features["quality_for_action"], errors="raise").to_numpy(dtype=np.float64)
    return enriched.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _evaluate(
    *,
    name: str,
    val_features: pd.DataFrame,
    oos_features: pd.DataFrame,
    val_frame: pd.DataFrame,
    val_dec: pd.DataFrame,
    oos_frame: pd.DataFrame,
    oos_dec: pd.DataFrame,
    y_val: np.ndarray,
    train_mask: np.ndarray,
    fee: float,
    slip: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        val_action, val_conf, _oof = label_family._predict_oof("hgb", val_features, y_val, train_mask, seed=int(seed))
        oos_action, oos_conf, _fitted = label_family._fit_predict("hgb", val_features, y_val, train_mask, oos_features, seed=int(seed))
        val_m = sleeve._metrics_with_fallback(val_frame, val_dec, RISK, val_action, val_conf, THRESHOLD, fee=fee, slip=slip, cost_mult=3.0)
        oos_m = sleeve._metrics_with_fallback(oos_frame, oos_dec, RISK, oos_action, oos_conf, THRESHOLD, fee=fee, slip=slip, cost_mult=3.0)
        rows.append(
            {
                "variant": name,
                "seed": int(seed),
                "threshold": float(THRESHOLD),
                "val_pnl": float(val_m["pnl"]),
                "val_mdd": float(val_m["mdd"]),
                "val_wr": float(val_m["wr"]),
                "val_trades": int(val_m["trades"]),
                "oos_pnl": float(oos_m["pnl"]),
                "oos_mdd": float(oos_m["mdd"]),
                "oos_wr": float(oos_m["wr"]),
                "oos_trades": int(oos_m["trades"]),
                "oos_reasons": oos_m["exit_reasons"],
            }
        )
    return rows


def _summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, g in detail.groupby("variant", sort=False):
        rows.append(
            {
                "variant": variant,
                "runs": int(len(g)),
                "val_pnl_mean": float(g["val_pnl"].mean()),
                "val_pnl_median": float(g["val_pnl"].median()),
                "val_pnl_min": float(g["val_pnl"].min()),
                "val_pnl_max": float(g["val_pnl"].max()),
                "oos_pnl_mean": float(g["oos_pnl"].mean()),
                "oos_pnl_median": float(g["oos_pnl"].median()),
                "oos_pnl_min": float(g["oos_pnl"].min()),
                "oos_pnl_max": float(g["oos_pnl"].max()),
                "oos_mdd_worst": float(g["oos_mdd"].min()),
                "oos_wr_mean": float(g["oos_wr"].mean()),
                "oos_trades_mean": float(g["oos_trades"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_frame, val_dec, val_features = full._build_split(frames, "validation")
    oos_frame, oos_dec, oos_features = full._build_split(frames, "oos")
    val_cash = ~omega._active(val_dec)
    y_val, valid_mask, label_diag = label_family._triple_barrier_labels(
        val_frame,
        atr_mult=1.0,
        max_hold=24,
        min_barrier=0.0035,
    )
    train_mask = val_cash & valid_mask

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded, base_cols = _load_models(device)
    val_exit_features = _entry_exit_features(val_features, val_dec, loaded, base_cols, device)
    oos_exit_features = _entry_exit_features(oos_features, oos_dec, loaded, base_cols, device)

    rows = []
    rows.extend(
        _evaluate(
            name="without_exit_feature",
            val_features=val_features,
            oos_features=oos_features,
            val_frame=val_frame,
            val_dec=val_dec,
            oos_frame=oos_frame,
            oos_dec=oos_dec,
            y_val=y_val,
            train_mask=train_mask,
            fee=fee,
            slip=slip,
        )
    )
    rows.extend(
        _evaluate(
            name="with_exit_entry_risk_feature",
            val_features=val_exit_features,
            oos_features=oos_exit_features,
            val_frame=val_frame,
            val_dec=val_dec,
            oos_frame=oos_frame,
            oos_dec=oos_dec,
            y_val=y_val,
            train_mask=train_mask,
            fee=fee,
            slip=slip,
        )
    )
    detail = pd.DataFrame(rows)
    summary = _summary(detail)
    detail.to_csv(OUT_DIR / "exit_feature_ablation_detail.csv", index=False)
    summary.to_csv(OUT_DIR / "exit_feature_ablation_summary.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "label": {"atr_mult": 1.0, "max_hold": 24, "min_barrier": 0.0035, "diag": label_diag},
        "threshold": THRESHOLD,
        "risk": RISK.__dict__,
        "summary": summary.to_dict(orient="records"),
        "top_single_runs": detail.sort_values(["oos_pnl", "val_pnl"], ascending=[False, False]).head(20).to_dict(orient="records"),
        "artifacts": {
            "detail": str(OUT_DIR / "exit_feature_ablation_detail.csv"),
            "summary": str(OUT_DIR / "exit_feature_ablation_summary.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "summary": report["summary"]}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
