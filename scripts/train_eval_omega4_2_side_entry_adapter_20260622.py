#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as risk_exp  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as omega4  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


MODEL_ID = "omega4_2_side_entry_adapter_20260622"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
BASELINE_BUNDLE = risk_exp.BASELINE_BUNDLE


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _train_side_classifier(x: pd.DataFrame, y: np.ndarray, *, seed: int) -> HistGradientBoostingClassifier:
    weights = compute_sample_weight(class_weight="balanced", y=np.asarray(y, dtype=np.int64))
    model = HistGradientBoostingClassifier(
        max_iter=260,
        learning_rate=0.035,
        l2_regularization=0.08,
        max_leaf_nodes=15,
        min_samples_leaf=32,
        random_state=int(seed),
    )
    model.fit(x, np.asarray(y, dtype=np.int64), sample_weight=weights)
    return model


def _parent_score_columns(src: pd.DataFrame) -> pd.DataFrame:
    out = src.drop(columns=["timestamp"], errors="ignore").copy()
    out = out.rename(
        columns={
            c: c.replace("omega1_regime3_expertdq_oof_", "parent_").replace("omega1_regime3_expertdq_", "parent_")
            for c in out.columns
        }
    )
    return out


def _adapter_feature_frame(frame: pd.DataFrame, src: pd.DataFrame, dec: pd.DataFrame, atr_pct: np.ndarray) -> pd.DataFrame:
    x = risk_exp._risk_feature_frame(
        frame,
        src,
        dec,
        [],
        atr_pct=np.asarray(atr_pct, dtype=np.float64),
        feature_mode="parent_outputs",
    )
    return x.astype(np.float32)


def _predict_side_prob(model: HistGradientBoostingClassifier, x: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(x)
    if proba.shape[1] != 2:
        raise RuntimeError(f"unexpected side classifier probability shape: {proba.shape}")
    out = np.asarray(proba[:, 1], dtype=np.float64)
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite side adapter probabilities")
    return out


def _adapter_decisions(
    base_dec: pd.DataFrame,
    features: pd.DataFrame,
    *,
    p_long: np.ndarray,
    p_short: np.ndarray,
    threshold: float,
    gap: float,
    score_mode: str,
    parent_gated: bool,
) -> pd.DataFrame:
    out = base_dec.copy().reset_index(drop=True)
    q_long = pd.to_numeric(features["parent_quality_p_long"], errors="raise").to_numpy(dtype=np.float64)
    q_short = pd.to_numeric(features["parent_quality_p_short"], errors="raise").to_numpy(dtype=np.float64)
    dir_long = pd.to_numeric(features["parent_dir_p_long"], errors="raise").to_numpy(dtype=np.float64)
    dir_short = pd.to_numeric(features["parent_dir_p_short"], errors="raise").to_numpy(dtype=np.float64)
    if score_mode == "prob":
        long_score = np.asarray(p_long, dtype=np.float64)
        short_score = np.asarray(p_short, dtype=np.float64)
    elif score_mode == "prob_quality":
        long_score = np.asarray(p_long, dtype=np.float64) * q_long
        short_score = np.asarray(p_short, dtype=np.float64) * q_short
    elif score_mode == "prob_quality_direction":
        long_score = np.asarray(p_long, dtype=np.float64) * q_long * dir_long
        short_score = np.asarray(p_short, dtype=np.float64) * q_short * dir_short
    else:
        raise RuntimeError(f"unknown side adapter score mode: {score_mode}")

    choose_long = (long_score >= float(threshold)) & ((long_score - short_score) >= float(gap))
    choose_short = (short_score >= float(threshold)) & ((short_score - long_score) >= float(gap))
    if bool(parent_gated):
        base_side = pd.to_numeric(base_dec["side"], errors="raise").to_numpy(dtype=np.int64)
        choose_long &= base_side > 0
        choose_short &= base_side < 0
    action = np.zeros(len(out), dtype=np.int64)
    action[choose_long] = omega.ACTION_LONG
    action[choose_short] = omega.ACTION_SHORT
    side = np.where(action == omega.ACTION_LONG, 1, np.where(action == omega.ACTION_SHORT, -1, 0)).astype(np.int64)
    active = action != omega.ACTION_CASH
    out["action"] = action
    out["side"] = side
    out["notional_exposure"] = np.where(active, float(omega.BASE_TEMPLATE["notional"]), 0.0)
    out["leverage"] = np.where(active, float(omega.BASE_TEMPLATE["leverage"]), 1.0)
    out["position_fraction"] = np.where(active, float(omega.BASE_TEMPLATE["notional"]), 0.0)
    out["take_profit"] = np.where(active, float(omega.BASE_TEMPLATE["take_profit"]), 0.0)
    out["stop_loss"] = np.where(active, float(omega.BASE_TEMPLATE["stop_loss"]), 0.0)
    out["quality_score"] = np.where(side > 0, q_long, np.where(side < 0, q_short, 0.0))
    out["confidence"] = np.maximum(long_score, short_score)
    out["adapter_long_score"] = long_score
    out["adapter_short_score"] = short_score
    out["adapter_score_gap"] = np.abs(long_score - short_score)
    return out


def _select_rows(rows: list[dict[str, Any]], *, baseline_trades: int, max_mdd_abs: float, min_trade_ratio: float) -> dict[str, Any]:
    floor = int(np.floor(int(baseline_trades) * float(min_trade_ratio)))
    eligible = [r for r in rows if int(r["validation_trades"]) >= floor and float(r["validation_mdd"]) >= -abs(float(max_mdd_abs))]
    if not eligible:
        eligible = [r for r in rows if int(r["validation_trades"]) >= floor]
    if not eligible:
        eligible = rows
    return max(eligible, key=lambda r: (float(r["validation_pnl"]), float(r["oos_pnl"]), float(r["validation_mdd"])))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-bundle", type=Path, default=BASELINE_BUNDLE)
    ap.add_argument("--quality-threshold", type=float, default=0.70)
    ap.add_argument("--exit-threshold", type=float, default=0.70)
    ap.add_argument("--atr-window", type=int, default=192)
    ap.add_argument("--tp-mult", type=float, default=12.0)
    ap.add_argument("--sl-mult", type=float, default=6.0)
    ap.add_argument("--min-tp", type=float, default=0.075)
    ap.add_argument("--min-sl", type=float, default=0.040)
    ap.add_argument("--max-tp", type=float, default=0.22)
    ap.add_argument("--max-sl", type=float, default=0.12)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--max-validation-mdd-abs", type=float, default=8.0)
    ap.add_argument("--min-trade-ratio", type=float, default=0.70)
    ap.add_argument("--thresholds", default="0.22,0.26,0.30,0.34,0.38,0.42,0.46,0.50,0.54,0.58")
    ap.add_argument("--gaps", default="0.00,0.02,0.04,0.06,0.08,0.10")
    ap.add_argument("--score-modes", default="prob,prob_quality,prob_quality_direction")
    ap.add_argument("--precheck-only", action="store_true")
    ap.add_argument("--parent-gated", action="store_true")
    ap.add_argument("--seed", type=int, default=260622)
    ap.add_argument("--out-suffix", default="parent_output_side_hgb")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    args = ap.parse_args()

    omega.BASE_TEMPLATE["max_hold"] = 0
    omega.BASE_TEMPLATE["cooldown"] = 0
    device = parent._device(str(args.device))
    out_dir = OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}" if str(args.out_suffix).strip() else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("stage=load_bundle", flush=True)
    bundle = torch.load(Path(args.baseline_bundle), map_location=device, weights_only=False)
    models: dict[str, dict[str, Any]] = bundle["models"]
    base_cols = list(bundle["base_cols"])
    loaded = parent._load_payloads(models, device=device)

    print("stage=prepare_frames", flush=True)
    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=omega4.LABEL_DIR,
        quality_mode="same_as_direction",
        quality_label_dir=None,
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )
    fee, slip = omega._load_fee_slip()

    print("stage=predict_parent", flush=True)
    x_train, train_src, train_dec_base = risk_exp._predict_decisions(
        frames["train_raw"], oof=True, models=models, base_cols=base_cols, quality_threshold=float(args.quality_threshold), device=device
    )
    x_val, val_src, val_dec_base = risk_exp._predict_decisions(
        frames["val_raw"], oof=True, models=models, base_cols=base_cols, quality_threshold=float(args.quality_threshold), device=device
    )
    x_oos, oos_src, oos_dec_base = risk_exp._predict_decisions(
        frames["oos_raw"], oof=False, models=models, base_cols=base_cols, quality_threshold=float(args.quality_threshold), device=device
    )

    print("stage=atr_contract", flush=True)
    train_dec_base, train_atr_diag = atr_eval._apply_atr_safety_sltp(
        train_dec_base, frames["train_raw"], atr_window=int(args.atr_window), tp_mult=float(args.tp_mult), sl_mult=float(args.sl_mult),
        min_tp=float(args.min_tp), min_sl=float(args.min_sl), max_tp=float(args.max_tp), max_sl=float(args.max_sl)
    )
    val_dec_base, val_atr_diag = atr_eval._apply_atr_safety_sltp(
        val_dec_base, frames["val_raw"], atr_window=int(args.atr_window), tp_mult=float(args.tp_mult), sl_mult=float(args.sl_mult),
        min_tp=float(args.min_tp), min_sl=float(args.min_sl), max_tp=float(args.max_tp), max_sl=float(args.max_sl)
    )
    oos_dec_base, oos_atr_diag = atr_eval._apply_atr_safety_sltp(
        oos_dec_base, frames["oos_raw"], atr_window=int(args.atr_window), tp_mult=float(args.tp_mult), sl_mult=float(args.sl_mult),
        min_tp=float(args.min_tp), min_sl=float(args.min_sl), max_tp=float(args.max_tp), max_sl=float(args.max_sl)
    )
    train_atr = atr_eval._atr_pct(frames["train_raw"], int(args.atr_window))
    val_atr = atr_eval._atr_pct(frames["val_raw"], int(args.atr_window))
    oos_atr = atr_eval._atr_pct(frames["oos_raw"], int(args.atr_window))

    print("stage=features", flush=True)
    train_feat = _adapter_feature_frame(frames["train_raw"], train_src, train_dec_base, train_atr)
    val_feat = _adapter_feature_frame(frames["val_raw"], val_src, val_dec_base, val_atr)
    oos_feat = _adapter_feature_frame(frames["oos_raw"], oos_src, oos_dec_base, oos_atr)
    feature_cols = list(train_feat.columns)
    val_feat = val_feat.reindex(columns=feature_cols).astype(np.float32)
    oos_feat = oos_feat.reindex(columns=feature_cols).astype(np.float32)

    print("stage=train_side_adapters", flush=True)
    y_action = pd.to_numeric(frames["train_raw"]["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    y_long = (y_action == omega.ACTION_LONG).astype(np.int64)
    y_short = (y_action == omega.ACTION_SHORT).astype(np.int64)
    long_model = _train_side_classifier(train_feat, y_long, seed=int(args.seed) + 101)
    short_model = _train_side_classifier(train_feat, y_short, seed=int(args.seed) + 202)
    p_val_long = _predict_side_prob(long_model, val_feat)
    p_val_short = _predict_side_prob(short_model, val_feat)
    p_oos_long = _predict_side_prob(long_model, oos_feat)
    p_oos_short = _predict_side_prob(short_model, oos_feat)

    print("stage=baseline_replay", flush=True)
    thresholds = [float(x) for x in str(args.thresholds).split(",") if str(x).strip()]
    gaps = [float(x) for x in str(args.gaps).split(",") if str(x).strip()]
    score_modes = [str(x).strip() for x in str(args.score_modes).split(",") if str(x).strip()]
    if bool(args.precheck_only):
        rows = []
        for score_mode in score_modes:
            for threshold in thresholds:
                for gap in gaps:
                    val_dec = _adapter_decisions(
                        val_dec_base, val_feat, p_long=p_val_long, p_short=p_val_short,
                        threshold=float(threshold), gap=float(gap), score_mode=score_mode, parent_gated=bool(args.parent_gated)
                    )
                    oos_dec = _adapter_decisions(
                        oos_dec_base, oos_feat, p_long=p_oos_long, p_short=p_oos_short,
                        threshold=float(threshold), gap=float(gap), score_mode=score_mode, parent_gated=bool(args.parent_gated)
                    )
                    rows.append(
                        {
                            "score_mode": score_mode,
                            "threshold": float(threshold),
                            "gap": float(gap),
                            "validation_active_rows": int(omega._active(val_dec).sum()),
                            "validation_long_rows": int((pd.to_numeric(val_dec["side"], errors="raise").to_numpy(dtype=np.int64) > 0).sum()),
                            "validation_short_rows": int((pd.to_numeric(val_dec["side"], errors="raise").to_numpy(dtype=np.int64) < 0).sum()),
                            "oos_active_rows": int(omega._active(oos_dec).sum()),
                            "oos_long_rows": int((pd.to_numeric(oos_dec["side"], errors="raise").to_numpy(dtype=np.int64) > 0).sum()),
                            "oos_short_rows": int((pd.to_numeric(oos_dec["side"], errors="raise").to_numpy(dtype=np.int64) < 0).sum()),
                        }
                    )
        df = pd.DataFrame(rows).sort_values(["validation_active_rows", "oos_active_rows"], ascending=[False, False])
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "side_adapter_precheck.csv", index=False)
        print(json.dumps({"precheck": str(out_dir / "side_adapter_precheck.csv"), "top": df.head(30).to_dict(orient="records")}, ensure_ascii=False, indent=2), flush=True)
        return 0

    print("stage=baseline_replay", flush=True)
    val_base_m, _ = risk_exp._replay_with_risk(
        frames["val_raw"], x_val, val_dec_base, loaded, risk_margin_fraction=None, exit_threshold=float(args.exit_threshold),
        fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device
    )
    oos_base_m, _ = risk_exp._replay_with_risk(
        frames["oos_raw"], x_oos, oos_dec_base, loaded, risk_margin_fraction=None, exit_threshold=float(args.exit_threshold),
        fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device
    )

    print("stage=grid_eval", flush=True)
    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for score_mode in score_modes:
        for threshold in thresholds:
            for gap in gaps:
                val_dec = _adapter_decisions(
                    val_dec_base, val_feat, p_long=p_val_long, p_short=p_val_short,
                    threshold=float(threshold), gap=float(gap), score_mode=score_mode, parent_gated=bool(args.parent_gated)
                )
                oos_dec = _adapter_decisions(
                    oos_dec_base, oos_feat, p_long=p_oos_long, p_short=p_oos_short,
                    threshold=float(threshold), gap=float(gap), score_mode=score_mode, parent_gated=bool(args.parent_gated)
                )
                val_dec, _ = atr_eval._apply_atr_safety_sltp(
                    val_dec, frames["val_raw"], atr_window=int(args.atr_window), tp_mult=float(args.tp_mult), sl_mult=float(args.sl_mult),
                    min_tp=float(args.min_tp), min_sl=float(args.min_sl), max_tp=float(args.max_tp), max_sl=float(args.max_sl)
                )
                oos_dec, _ = atr_eval._apply_atr_safety_sltp(
                    oos_dec, frames["oos_raw"], atr_window=int(args.atr_window), tp_mult=float(args.tp_mult), sl_mult=float(args.sl_mult),
                    min_tp=float(args.min_tp), min_sl=float(args.min_sl), max_tp=float(args.max_tp), max_sl=float(args.max_sl)
                )
                val_m, _ = risk_exp._replay_with_risk(
                    frames["val_raw"], x_val, val_dec, loaded, risk_margin_fraction=None, exit_threshold=float(args.exit_threshold),
                    fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device
                )
                oos_m, _ = risk_exp._replay_with_risk(
                    frames["oos_raw"], x_oos, oos_dec, loaded, risk_margin_fraction=None, exit_threshold=float(args.exit_threshold),
                    fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device
                )
                name = f"{score_mode}_thr{threshold:g}_gap{gap:g}".replace(".", "p")
                row = {
                    "variant": name,
                    "score_mode": score_mode,
                    "threshold": float(threshold),
                    "gap": float(gap),
                    "validation_pnl": float(val_m["pnl"]),
                    "validation_mdd": float(val_m["mdd"]),
                    "validation_trades": int(val_m["trades"]),
                    "validation_wr": float(val_m["wr"]),
                    "oos_pnl": float(oos_m["pnl"]),
                    "oos_mdd": float(oos_m["mdd"]),
                    "oos_trades": int(oos_m["trades"]),
                    "oos_wr": float(oos_m["wr"]),
                }
                rows.append(row)
                results[name] = {"config": row, "validation": val_m, "oos": oos_m}

    selected = _select_rows(
        rows,
        baseline_trades=int(val_base_m["trades"]),
        max_mdd_abs=float(args.max_validation_mdd_abs),
        min_trade_ratio=float(args.min_trade_ratio),
    )
    ranking = pd.DataFrame(rows).sort_values(["validation_pnl", "oos_pnl", "validation_mdd"], ascending=[False, False, False])
    ranking.to_csv(out_dir / "side_adapter_ranking.csv", index=False)
    with (out_dir / "side_entry_adapter.pkl").open("wb") as f:
        pickle.dump(
            {
                "long_model": long_model,
                "short_model": short_model,
                "feature_columns": feature_cols,
                "selected": selected,
                "contract": "Omega 4.2 parent weights and exit head unchanged; side adapter replaces final entry action only.",
            },
            f,
        )

    report = {
        "model_id": MODEL_ID,
        "base_model": "omega4_2_atr192_tp12_sl6_floor_tp075_sl040_exit070_20260622",
        "design": "Frozen Omega 4.2 shared parent. Separate long-vs-rest and short-vs-rest HGB classifiers are trained on parent output features, then an arbiter picks long/short/cash. Exit head and ATR safety SLTP are unchanged.",
        "contract": {
            "entry_changed": "side adapter replaces final_action",
            "exit_changed": False,
            "risk_sizing_changed": False,
            "parent_gated": bool(args.parent_gated),
            "quality_threshold_reference": float(args.quality_threshold),
            "exit_threshold": float(args.exit_threshold),
            "atr_window": int(args.atr_window),
            "tp_mult": float(args.tp_mult),
            "sl_mult": float(args.sl_mult),
            "min_tp": float(args.min_tp),
            "min_sl": float(args.min_sl),
            "max_tp": float(args.max_tp),
            "max_sl": float(args.max_sl),
        },
        "training": {
            "features": "parent output / decision / ATR runtime features only",
            "feature_count": int(len(feature_cols)),
            "train_rows": int(len(train_feat)),
            "long_positive_rate": float(y_long.mean()),
            "short_positive_rate": float(y_short.mean()),
        },
        "baseline": {"validation": val_base_m, "oos": oos_base_m},
        "atr_diag": {"train": train_atr_diag, "validation": val_atr_diag, "oos": oos_atr_diag},
        "selected": {"variant": selected["variant"], "config": selected, "validation": results[selected["variant"]]["validation"], "oos": results[selected["variant"]]["oos"]},
        "top_validation": ranking.head(12).to_dict(orient="records"),
        "artifacts": {"out_dir": str(out_dir), "report": str(out_dir / "report.json"), "ranking": str(out_dir / "side_adapter_ranking.csv"), "adapter": str(out_dir / "side_entry_adapter.pkl")},
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "baseline": report["baseline"], "selected": report["selected"]}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
