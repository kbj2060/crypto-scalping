#!/usr/bin/env python3
"""Train ETH 3-head TabM on split-local Oracle labels and freeze on validation."""

from __future__ import annotations

import argparse
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

import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as trainer  # noqa: E402
import build_hmm_confluence_meta_labels_20260724 as oracle_base  # noqa: E402


omega = trainer.omega
parent = trainer.parent
hard = trainer.hard
exit_head = trainer.exit_head
MODEL_ID = "eth_split_oracle_3head_noleak_20260724"
LABEL_DIR = ROOT / "tmp/causal_regen_20260516/eth_split_oracle_strategy_labels_20260724"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
QUALITY_TARGET_COLUMN: str | None = None
TRAIN_END = pd.Timestamp("2026-01-01")
VALIDATION_END = pd.Timestamp("2026-04-01")
MARKET_DIR = ROOT / "data/splits/year_oos"
CURRENT_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
CMAMBA_DIR = ROOT / "data/ensemble/supervised/regime3_cryptomamba_h6_sidecar_20260601"
RISK_DIR = ROOT / "data/ensemble/supervised/regime3_stability_risk_h6_20260530"


def _year_paths(year: int) -> tuple[Path, Path, Path, Path]:
    suffix = "training_features_2026_rebuilt" if year == 2026 else f"training_features_{year}"
    return (
        MARKET_DIR / f"{suffix}.csv",
        CURRENT_DIR / f"{suffix}_regime3_current_sensitive_hmm_wide24.csv",
        CMAMBA_DIR / f"{suffix}_regime3_cryptomamba_h6_sidecar_20260601.csv",
        RISK_DIR / f"{suffix}_regime3_stability_risk_h6.csv",
    )


def load_year(year: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    market_path, current_path, cmamba_path, risk_path = _year_paths(year)
    frame = omega._read(market_path)
    frame, current = omega._overlay_required(
        frame, current_path, omega.REGIME3_CURRENT_COLS, tag=f"{year}_current"
    )
    frame, cmamba = omega._overlay_required(
        frame, cmamba_path, omega.REGIME3_CMAMBA_COLS, tag=f"{year}_cmamba"
    )
    inferred_risk = oracle_base.infer_transition_risk(frame, oracle_base.RISK_ARTIFACT)
    frame, aligned_risk = omega._align(frame, inferred_risk, f"{year}_risk_reinferred")
    missing_risk = sorted(set(omega.REGIME3_RISK_COLS) - set(aligned_risk.columns))
    if missing_risk:
        raise RuntimeError(f"{year} inferred risk missing columns: {missing_risk}")
    for column in omega.REGIME3_RISK_COLS:
        frame[column] = pd.to_numeric(aligned_risk[column], errors="raise").to_numpy(dtype=np.float64)
    risk = {
        "artifact": str(oracle_base.RISK_ARTIFACT),
        "rows": int(len(aligned_risk)),
        "stale_csv_not_used": str(risk_path),
    }
    labels = pd.read_csv(
        LABEL_DIR / f"zigzag_action_labels_{year}.csv",
        parse_dates=["timestamp"],
        low_memory=False,
    )
    required = {"timestamp", "zigzag_action", "oracle_label_valid", "oracle_split"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise RuntimeError(f"{year} split Oracle labels missing columns: {missing}")
    frame, aligned = omega._align(frame, labels, f"{year}_oracle_labels")
    frame = frame.copy()
    for column in ("zigzag_action", "oracle_label_valid", "oracle_split"):
        frame[column] = aligned[column].to_numpy()
    if QUALITY_TARGET_COLUMN is None:
        quality_source = frame["zigzag_action"]
    else:
        if QUALITY_TARGET_COLUMN not in aligned.columns:
            raise RuntimeError(f"{year} labels missing explicit quality target: {QUALITY_TARGET_COLUMN}")
        quality_source = aligned[QUALITY_TARGET_COLUMN]
    frame["omega4_quality_action"] = pd.to_numeric(
        quality_source, errors="raise"
    ).to_numpy(dtype=np.int64)
    return frame, {"current": current, "cmamba": cmamba, "risk": risk}


def prepare_frames() -> dict[str, Any]:
    yearly: dict[int, pd.DataFrame] = {}
    overlays: dict[str, Any] = {}
    for year in (2024, 2025, 2026):
        yearly[year], overlays[str(year)] = load_year(year)
    train_all = pd.concat([yearly[2024], yearly[2025]], ignore_index=True)
    eval_all = yearly[2026].reset_index(drop=True)
    feature_cols = omega._numeric_feature_cols(train_all, eval_all)
    direct_target_columns = {"zigzag_action", "omega4_quality_action", "oracle_label_valid"}
    leaked = sorted(direct_target_columns & set(feature_cols))
    if leaked != ["omega4_quality_action"]:
        raise RuntimeError(f"unexpected direct-target feature audit result: {leaked}")
    feature_cols = [column for column in feature_cols if column not in direct_target_columns]
    remaining_target_like = [
        column for column in feature_cols
        if column.startswith("oracle_") or column in {"zigzag_action", "omega4_quality_action"}
    ]
    if remaining_target_like:
        raise RuntimeError(f"target columns entered model features: {remaining_target_like}")
    valid_train = train_all["oracle_label_valid"].astype(bool)
    if not (train_all.loc[valid_train, "oracle_split"] == "train").all():
        raise RuntimeError("non-train Oracle labels entered the weight-update frame")
    train = train_all.loc[valid_train & (train_all["timestamp"] < TRAIN_END)].reset_index(drop=True)
    validation = eval_all.loc[
        (eval_all["timestamp"] < VALIDATION_END)
        & (eval_all["oracle_split"] == "validation")
    ].reset_index(drop=True)
    oos = eval_all.loc[
        (eval_all["timestamp"] >= VALIDATION_END)
        & (eval_all["oracle_split"] == "oos")
    ].reset_index(drop=True)
    if train.empty or validation.empty or oos.empty:
        raise RuntimeError("empty Train/Validation/OOS frame after split alignment")
    return {
        "train": train,
        "validation": validation,
        "oos": oos,
        "feature_cols": feature_cols,
        "excluded_direct_target_columns": leaked,
        "overlays": overlays,
    }


def apply_notional_scale(decisions: pd.DataFrame, *, scale: float, cap: float = 0.90) -> pd.DataFrame:
    out = decisions.copy().reset_index(drop=True)
    active_idx = np.flatnonzero(omega._active(out))
    if not len(active_idx):
        return out
    base_notional = pd.to_numeric(
        out.loc[active_idx, "notional_exposure"], errors="raise"
    ).to_numpy(dtype=np.float64)
    leverage = pd.to_numeric(out.loc[active_idx, "leverage"], errors="raise").to_numpy(dtype=np.float64)
    new_notional = np.minimum(base_notional * float(scale), float(cap))
    ratio = new_notional / np.maximum(base_notional, 1.0e-12)
    out.loc[active_idx, "notional_exposure"] = new_notional
    out.loc[active_idx, "position_fraction"] = new_notional / np.maximum(leverage, 1.0e-12)
    out.loc[active_idx, "take_profit"] = pd.to_numeric(
        out.loc[active_idx, "take_profit"], errors="raise"
    ).to_numpy(dtype=np.float64) * ratio
    out.loc[active_idx, "stop_loss"] = pd.to_numeric(
        out.loc[active_idx, "stop_loss"], errors="raise"
    ).to_numpy(dtype=np.float64) * ratio
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-exit-samples", type=int, default=30000)
    parser.add_argument("--quality-thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75")
    parser.add_argument("--sizing-scales", default="0.50,0.75,1.00,1.25,1.50,2.00")
    parser.add_argument("--cost-mult", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=260724)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--skip-oos", action="store_true")
    args = parser.parse_args()

    trainer._seed_everything(args.seed)
    device = parent._device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = prepare_frames()
    train = frames["train"]
    validation = frames["validation"]
    oos = frames["oos"]
    base_cols = list(frames["feature_cols"])
    fee, slip = omega._load_fee_slip()

    x_train = parent._base_input(train, base_cols)
    y_train = train["zigzag_action"].to_numpy(dtype=np.int64)
    y_quality = train["omega4_quality_action"].to_numpy(dtype=np.int64)
    train_state = parent._base_input(train, base_cols)
    x_exit_raw, y_exit, exit_frame, exit_diag = trainer._build_exit_dataset_entry_label_terminal_giveback(
        train,
        train_state,
        risk_margin=None,
        risk_leverage=None,
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        max_samples=int(args.max_exit_samples),
        terminal_window=3,
        adverse_unreal=-0.010,
        min_mfe_for_giveback=0.006,
        giveback_min=0.65,
    )
    x_exit = parent._exit_input_from_position_rows(x_exit_raw, base_cols)

    models: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        payload = trainer._fit_expert_omega4(
            x_train,
            y_train,
            y_quality,
            train,
            x_exit,
            y_exit,
            exit_frame,
            expert_idx=idx,
            seed=int(args.seed),
            epochs=int(args.epochs),
            device=device,
            model_path=OUT_DIR / "models" / f"{expert}_3head_tabm.pt",
            direction_class_weights={},
            quality_class_weights={},
        )
        models[expert] = payload
        summaries[expert] = {
            "epochs_ran": int(payload["epochs_ran"]),
            "best_validation_loss_within_train": float(payload["best_validation_loss"]),
        }

    def predict(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = parent._base_input(frame, base_cols)
        preds = {
            expert: parent._predict_payload(models[expert], x, device=device)
            for expert in hard.EXPERT_NAMES
        }
        route = hard._route_id(frame)
        return (
            parent._routed(preds, route, "direction", 3),
            parent._routed(preds, route, "quality", 3),
        )

    train_direction, train_quality = predict(train)
    val_direction, val_quality = predict(validation)
    q_values = [float(value) for value in args.quality_thresholds.split(",")]
    scales = [float(value) for value in args.sizing_scales.split(",")]
    validation_rows: list[dict[str, Any]] = []
    for q in q_values:
        val_prediction = parent._prediction_output(
            validation, val_direction, val_quality, threshold=q, prefix="omega1_regime3_expertdq_oof"
        )
        base_decisions = parent._to_decisions(val_prediction, oof=True)
        for scale in scales:
            metrics = omega._metrics(
                validation,
                apply_notional_scale(base_decisions, scale=scale),
                fee=fee,
                slip=slip,
                cost_mult=float(args.cost_mult),
            )
            validation_rows.append({"quality_threshold": q, "notional_scale": scale, **metrics})
    ranking = pd.DataFrame(validation_rows).sort_values(
        ["pnl", "mdd", "trades"], ascending=[False, False, False]
    ).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "validation_threshold_sizing_ranking.csv", index=False)
    selected = ranking.iloc[0].to_dict()
    chosen_q = float(selected["quality_threshold"])
    chosen_scale = float(selected["notional_scale"])

    q_tag = f"q{int(round(chosen_q * 100)):03d}"
    train_prediction = parent._prediction_output(
        train, train_direction, train_quality, threshold=chosen_q, prefix="omega1_regime3_expertdq_oof"
    )
    val_prediction = parent._prediction_output(
        validation, val_direction, val_quality, threshold=chosen_q, prefix="omega1_regime3_expertdq_oof"
    )
    train_prediction.to_csv(OUT_DIR / f"train_predictions_{q_tag}.csv", index=False)
    val_prediction.to_csv(OUT_DIR / f"validation_predictions_{q_tag}.csv", index=False)

    torch.save(
        {"models": models, "base_cols": base_cols, "pos_cols": parent.POS_COLS, "config": parent.CFG.__dict__},
        OUT_DIR / "true_3head_tabm_bundle.pt",
    )

    if bool(args.skip_oos):
        report = {
            "model_id": MODEL_ID,
            "status": "validation_selected_oos_not_evaluated",
            "label_dir": str(LABEL_DIR),
            "quality_target_column": QUALITY_TARGET_COLUMN,
            "selection": {
                "source": "validation_realized_performance_only",
                "objective": "maximum_validation_pnl_then_lower_drawdown",
                "quality_threshold": chosen_q,
                "notional_scale": chosen_scale,
                "validation_metrics": selected,
            },
            "feature_audit": {
                "feature_count": int(len(base_cols)),
                "excluded_direct_target_columns": frames["excluded_direct_target_columns"],
                "target_columns_in_model_input": False,
            },
            "weight_updates_use_train_labels_only": True,
            "oos_evaluated": False,
            "promotion_eligible": False,
            "promotion_blocker": "OOS intentionally withheld until validation-only variant selection",
        }
        (OUT_DIR / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=trainer._json_default) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"report": str(OUT_DIR / "report.json"), "selection": report["selection"], "oos_evaluated": False}, ensure_ascii=False), flush=True)
        return 0

    # OOS is first touched for performance only after q and sizing are frozen above.
    oos_direction, oos_quality = predict(oos)
    oos_prediction_oof = parent._prediction_output(
        oos, oos_direction, oos_quality, threshold=chosen_q, prefix="omega1_regime3_expertdq_oof"
    )
    oos_prediction = oos_prediction_oof.rename(
        columns={column: column.replace("omega1_regime3_expertdq_oof_", "omega1_regime3_expertdq_") for column in oos_prediction_oof.columns}
    )
    oos_prediction.to_csv(OUT_DIR / f"oos_predictions_{q_tag}.csv", index=False)
    oos_decisions = apply_notional_scale(
        parent._to_decisions(oos_prediction, oof=False), scale=chosen_scale
    )
    oos_metrics = omega._metrics(
        oos, oos_decisions, fee=fee, slip=slip, cost_mult=float(args.cost_mult)
    )

    report = {
        "model_id": MODEL_ID,
        "split_contract": {
            "train": [str(train["timestamp"].iloc[0]), str(train["timestamp"].iloc[-1])],
            "validation": [str(validation["timestamp"].iloc[0]), str(validation["timestamp"].iloc[-1])],
            "oos": [str(oos["timestamp"].iloc[0]), str(oos["timestamp"].iloc[-1])],
        },
        "label_contract": {
            "source": str(LABEL_DIR),
            "quality_target_column": QUALITY_TARGET_COLUMN,
            "weight_update_rows": int(len(train)),
            "weight_updates_use_train_labels_only": True,
            "validation_oracle_labels_used_for_selection": False,
            "oos_oracle_labels_used": False,
        },
        "feature_audit": {
            "feature_count": int(len(base_cols)),
            "excluded_direct_target_columns": frames["excluded_direct_target_columns"],
            "target_columns_in_model_input": False,
        },
        "selection": {
            "source": "validation_realized_performance_only",
            "objective": "maximum_validation_pnl_then_lower_drawdown",
            "quality_threshold": chosen_q,
            "notional_scale": chosen_scale,
            "fixed_leverage": float(omega.BASE_TEMPLATE["leverage"]),
            "base_notional": float(omega.BASE_TEMPLATE["notional"]),
            "base_margin_fraction": float(omega.BASE_TEMPLATE["notional"] / omega.BASE_TEMPLATE["leverage"]),
            "validation_metrics": selected,
        },
        "oos": oos_metrics,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "oos_evaluations_after_freeze": 1,
        "summaries": summaries,
        "exit_label": {"source": "train Oracle trajectory only", "diag": exit_diag},
        "prediction_artifacts": {
            q_tag: {
                "train": str(OUT_DIR / f"train_predictions_{q_tag}.csv"),
                "validation": str(OUT_DIR / f"validation_predictions_{q_tag}.csv"),
                "oos": str(OUT_DIR / f"oos_predictions_{q_tag}.csv"),
            }
        },
        "promotion_eligible": False,
        "promotion_blocker": "negative fresh-forward OOS; candidate rejected (Omega integrity audit schema also not satisfied)",
    }
    (OUT_DIR / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=trainer._json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selection": report["selection"], "oos": oos_metrics}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
