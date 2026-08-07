#!/usr/bin/env python3
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

import eval_omega1_2_true3head_overlays_20260604 as overlay  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_1_full_retrain_cash_alpha43_20260608 as full_parent  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_tabm_exit_head_20260603 as exit_head  # noqa: E402
import train_omega1_direction_head_direction_only_20260602 as direction_labels  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


MODEL_ID = "omega1_2_asymmetric_direction_cleanup_20260618"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
CURRENT_PARENT_VAL = {
    "pnl": 100.54272942091158,
    "mdd": -10.677652697162888,
    "trades": 33,
    "wr": 0.6363636363636364,
    "long_entries": 3,
    "short_entries": 30,
    "exit_reasons": {"take_profit": 21, "stop_loss": 12},
}
CURRENT_PARENT_OOS = {
    "pnl": 72.76004148106665,
    "mdd": -8.108170708968387,
    "trades": 18,
    "wr": 0.7222222222222222,
    "long_entries": 2,
    "short_entries": 16,
    "exit_reasons": {"take_profit": 13, "stop_loss": 5},
}


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _router_names(frame: pd.DataFrame) -> np.ndarray:
    ids = hard._route_id(frame)
    names = np.asarray([hard.EXPERT_NAMES[int(i)] for i in ids], dtype=object)
    return np.where(names == "chop", "chop_expert", names)


def _oracle_decision(frame: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    action = np.asarray(y, dtype=np.int64)
    active = action != omega.ACTION_CASH
    side = np.where(action == omega.ACTION_LONG, 1, np.where(action == omega.ACTION_SHORT, -1, 0)).astype(np.int64)
    dec = pd.DataFrame(
        {
            "action": action,
            "side": side,
            "notional_exposure": np.where(active, float(omega.BASE_TEMPLATE["notional"]), 0.0),
            "leverage": np.where(active, float(omega.BASE_TEMPLATE["leverage"]), 1.0),
            "position_fraction": np.where(active, float(omega.BASE_TEMPLATE["notional"]), 0.0),
            "take_profit": np.where(active, float(omega.BASE_TEMPLATE["take_profit"]), 0.0),
            "stop_loss": np.where(active, float(omega.BASE_TEMPLATE["stop_loss"]), 0.0),
            "max_hold_bars": np.where(active, int(omega.BASE_TEMPLATE["max_hold"]), 0).astype(np.int64),
            "cooldown_bars": np.where(active, int(omega.BASE_TEMPLATE["cooldown"]), 0).astype(np.int64),
            "quality_score": np.where(active, 1.0, 0.0),
            "confidence": np.where(active, 1.0, 0.0),
            "router_expert": _router_names(frame),
        }
    )
    return omega._apply_expert_scale(dec)


def _read_zigzag_labels(year: int) -> pd.DataFrame:
    path = direction_labels.LABEL_DIR / f"zigzag_action_labels_{int(year)}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    labels = pd.read_csv(path, usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"], low_memory=False)
    labels = labels.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    y = pd.to_numeric(labels["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    invalid = sorted(set(np.unique(y).tolist()) - {0, 1, 2})
    if invalid:
        raise RuntimeError(f"zigzag labels {year} invalid classes: {invalid}")
    return labels


def _prepare_frames_fast(*, disable_tp_sl: bool) -> dict[str, Any]:
    omega.BASE_TEMPLATE["max_hold"] = 0
    omega.BASE_TEMPLATE["cooldown"] = 0
    train_all, eval_df, overlay_report = omega._load_omega_frames()
    feature_cols = omega._numeric_feature_cols(train_all, eval_df)
    train_all, train_labels = omega._align(train_all, _read_zigzag_labels(2025), "omega train labels")
    eval_df, eval_labels = omega._align(eval_df, _read_zigzag_labels(2026), "omega oos labels")
    train_all = train_all.copy()
    eval_df = eval_df.copy()
    train_all["zigzag_action"] = pd.to_numeric(train_labels["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    eval_df["zigzag_action"] = pd.to_numeric(eval_labels["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    train_raw = train_all[train_all["timestamp"] < threehead.SPLIT_TS].reset_index(drop=True)
    val_raw = train_all[train_all["timestamp"] >= threehead.SPLIT_TS].reset_index(drop=True)

    tabm_2025 = omega._read(omega.TABM_2025)
    train_df, train_src = omega._align(train_raw, tabm_2025, "train")
    train_fixed = omega._to_fixed_decisions(train_src, oof=True)
    if disable_tp_sl:
        train_fixed = exit_head._disable_tp_sl(train_fixed)
    s_train_label = threehead._base_input(train_df, feature_cols)
    return {
        "train_raw": train_raw,
        "val_raw": val_raw,
        "oos_raw": eval_df.reset_index(drop=True),
        "train_df": train_df,
        "train_fixed": train_fixed,
        "s_train_label": s_train_label,
        "feature_cols": feature_cols,
        "overlay_report": overlay_report,
    }


def _filter_to_parent_prediction_span(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    if split == "validation":
        pred_path = full_parent.PARENT_DIR / "validation_predictions_2025_true3head.csv"
    elif split == "oos":
        pred_path = full_parent.PARENT_DIR / "oos_predictions_2026_true3head.csv"
    else:
        raise RuntimeError(f"unknown split: {split}")
    pred_ts = pd.read_csv(pred_path, usecols=["timestamp"], parse_dates=["timestamp"])
    keep = frame["timestamp"].isin(set(pred_ts["timestamp"]))
    out = frame.loc[keep].reset_index(drop=True)
    if len(out) != len(pred_ts):
        raise RuntimeError(f"{split} parent timestamp filter mismatch: frame={len(out)} pred={len(pred_ts)}")
    return out


def _purify_direction_labels(
    frame: pd.DataFrame,
    y: np.ndarray,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    mode: str,
    mae_floor: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    dec = _oracle_decision(frame, y)
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    y_out = np.asarray(y, dtype=np.int64).copy()
    active_idx = np.flatnonzero(omega._active(dec))
    rows: list[dict[str, Any]] = []
    relabeled = 0
    for i in active_idx:
        if int(i) >= len(frame) - 3:
            continue
        _score, meta = omega._simulate_trade(frame, arrays, int(i), dec.iloc[int(i)], fee=fee, slip=slip, cost_mult=cost_mult)
        if int(meta.get("active", 0)) != 1:
            continue
        reason = str(meta.get("exit_reason", ""))
        net = float(meta.get("net", 0.0))
        mae = float(meta.get("mae", 0.0))
        if mode == "stop_loss_to_cash":
            toxic = reason == "stop_loss"
        elif mode == "negative_net_to_cash":
            toxic = net < 0.0
        elif mode == "stop_loss_or_bad_mae_to_cash":
            toxic = reason == "stop_loss" or mae <= -abs(float(mae_floor))
        else:
            raise RuntimeError(f"unknown cleanup mode: {mode}")
        if toxic:
            y_out[int(i)] = omega.ACTION_CASH
            relabeled += 1
        rows.append({"exit_reason": reason, "net": net, "mae": mae, "toxic": bool(toxic)})
    labels = pd.DataFrame(rows)
    diag = {
        "mode": mode,
        "mae_floor": float(mae_floor),
        "active_rows": int(len(active_idx)),
        "simulated_rows": int(len(labels)),
        "relabeled_to_cash": int(relabeled),
        "relabeled_rate_of_simulated": float(relabeled / max(len(labels), 1)),
        "original_class_counts": pd.Series(y).value_counts().sort_index().to_dict(),
        "purified_class_counts": pd.Series(y_out).value_counts().sort_index().to_dict(),
        "exit_reason_counts": labels["exit_reason"].value_counts().sort_index().to_dict() if not labels.empty else {},
        "net_mean": float(labels["net"].mean()) if not labels.empty else 0.0,
        "mae_mean": float(labels["mae"].mean()) if not labels.empty else 0.0,
    }
    return y_out, diag


def _build_operational_dec(src: pd.DataFrame, prefix: str, *, oof: bool) -> pd.DataFrame:
    return sleeve._apply_aggressive(overlay._build_dec(src, prefix, oof=oof))


def _metric_row(candidate: str, val_m: dict[str, Any], oos_m: dict[str, Any], base_val: dict[str, Any], base_oos: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"candidate": candidate, **params}
    row.update(sleeve._metric_row("val", {**val_m, "primary_entries": val_m["long_entries"] + val_m["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}))
    row.update(sleeve._metric_row("oos", {**oos_m, "primary_entries": oos_m["long_entries"] + oos_m["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}))
    row["val_delta_vs_current"] = float(row["val_pnl"] - float(base_val["pnl"]))
    row["oos_delta_vs_current"] = float(row["oos_pnl"] - float(base_oos["pnl"]))
    val_reasons = row["val_reasons"] if isinstance(row["val_reasons"], dict) else {}
    row["val_stop_loss"] = int(val_reasons.get("stop_loss", 0))
    row["selection_score_val_only"] = (
        row["val_delta_vs_current"]
        + 10.0 * float(row["val_wr"])
        + 0.25 * float(row["val_mdd"])
        - 0.75 * float(row["val_stop_loss"])
        - 0.05 * max(0.0, float(row["val_trades"]) - 80.0)
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=28)
    ap.add_argument("--mode", choices=["stop_loss_to_cash", "negative_net_to_cash", "stop_loss_or_bad_mae_to_cash"], default="stop_loss_to_cash")
    ap.add_argument("--quality-thresholds", default="0.45,0.55,0.65,0.75,0.80,0.85,0.90")
    ap.add_argument("--exit-edge-min", type=float, default=0.0020)
    ap.add_argument("--exit-hold-offsets", default="1,2,3,6,12,24,48,96,192,384")
    ap.add_argument("--max-exit-samples", type=int, default=0)
    ap.add_argument("--max-train-rows", type=int, default=0)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--mae-floor", type=float, default=0.020)
    ap.add_argument("--seed", type=int, default=260618)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--skip-train-if-present", action="store_true")
    args = ap.parse_args()

    threehead._seed_everything(int(args.seed))
    device = threehead._device(str(args.device))
    out_dir = OUT_DIR / str(args.mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = _prepare_frames_fast(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    base_cols = list(frames["feature_cols"])
    train_raw = frames["train_raw"]
    val_raw = _filter_to_parent_prediction_span(frames["val_raw"], "validation")
    oos_raw = _filter_to_parent_prediction_span(frames["oos_raw"], "oos")

    hold_offsets = [int(x.strip()) for x in str(args.exit_hold_offsets).split(",") if x.strip()]
    models: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    model_paths = {expert: out_dir / "models" / f"{expert}_3head_tabm.pt" for expert in hard.EXPERT_NAMES}
    all_present = all(p.exists() for p in model_paths.values())
    cleanup_diag: dict[str, Any]
    exit_diag: dict[str, Any]
    if bool(args.skip_train_if_present) and all_present:
        cleanup_diag = {"mode": str(args.mode), "loaded_existing": True, "note": "cleanup diagnostics were generated during the original training run"}
        exit_diag = {"loaded_existing": True}
        for expert in hard.EXPERT_NAMES:
            payload = torch.load(model_paths[expert], map_location="cpu", weights_only=False)
            models[expert] = payload
            summaries[expert] = {
                "model": str(model_paths[expert]),
                "epochs_ran": int(payload.get("epochs_ran", -1)),
                "best_validation_loss": float(payload.get("best_validation_loss", float("nan"))),
                "loaded_existing": True,
            }
    else:
        x_train = threehead._base_input(train_raw, base_cols)
        y_train_original = train_raw["zigzag_action"].to_numpy(dtype=np.int64)
        y_train, cleanup_diag = _purify_direction_labels(
            train_raw,
            y_train_original,
            fee=fee,
            slip=slip,
            cost_mult=float(args.cost_mult),
            mode=str(args.mode),
            mae_floor=float(args.mae_floor),
        )
        print(json.dumps({"stage": "cleanup_done", "diag": cleanup_diag}, ensure_ascii=False, default=_json_default), flush=True)
        train_fit_frame = train_raw
        if int(args.max_train_rows) > 0:
            limit = int(args.max_train_rows)
            x_train = x_train.iloc[:limit].reset_index(drop=True)
            y_train = y_train[:limit]
            train_fit_frame = train_raw.iloc[:limit].reset_index(drop=True)

        print(json.dumps({"stage": "build_exit_dataset", "max_exit_samples": int(args.max_exit_samples)}, ensure_ascii=True), flush=True)
        x_exit_raw, y_exit, frame_exit, exit_diag = exit_head._build_exit_dataset_independent(
            frames["train_df"],
            frames["s_train_label"],
            frames["train_fixed"],
            fee=fee,
            slip=slip,
            cost_mult=float(args.cost_mult),
            exit_edge_min=float(args.exit_edge_min),
            hold_offsets=hold_offsets,
            max_samples=int(args.max_exit_samples),
        )
        x_exit = threehead._exit_input_from_position_rows(x_exit_raw, base_cols)

        for idx, expert in enumerate(hard.EXPERT_NAMES):
            model_path = model_paths[expert]
            print(json.dumps({"stage": "fit_asymmetric_expert", "expert": expert, "mode": args.mode}, ensure_ascii=True), flush=True)
            payload = threehead._fit_expert_3head(
                x_train,
                y_train,
                train_fit_frame,
                x_exit,
                y_exit,
                frame_exit,
                expert_idx=idx,
                seed=int(args.seed),
                epochs=int(args.epochs),
                device=device,
                model_path=model_path,
            )
            models[expert] = payload
            summaries[expert] = {
                "model": str(model_path),
                "epochs_ran": int(payload["epochs_ran"]),
                "best_validation_loss": float(payload["best_validation_loss"]),
            }
    def predict_src(frame: pd.DataFrame, *, threshold: float, oof: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        x = threehead._base_input(frame, base_cols)
        preds = {expert: threehead._predict_payload(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
        route = hard._route_id(frame)
        direction = threehead._routed(preds, route, "direction", 3)
        quality = threehead._routed(preds, route, "quality", 3)
        prefix = "omega1_regime3_expertdq_oof" if oof else "omega1_regime3_expertdq"
        src = threehead._prediction_output(frame, direction, quality, threshold=float(threshold), prefix=prefix)
        return src, x

    if not full_parent.PARENT_DIR.exists():
        raise RuntimeError(f"missing parent artifact: {full_parent.PARENT_DIR}")
    current_val_m = dict(CURRENT_PARENT_VAL)
    current_oos_m = dict(CURRENT_PARENT_OOS)

    rows: list[dict[str, Any]] = []
    thresholds = [float(x.strip()) for x in str(args.quality_thresholds).split(",") if x.strip()]
    saved_src = False
    for q in thresholds:
        val_src, _x_val = predict_src(val_raw, threshold=q, oof=True)
        oos_src, _x_oos = predict_src(oos_raw, threshold=q, oof=False)
        val_dec = _build_operational_dec(val_src, "omega1_regime3_expertdq_oof_", oof=True)
        oos_dec = _build_operational_dec(oos_src, "omega1_regime3_expertdq_", oof=False)
        val_m = omega._metrics(val_raw, val_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        oos_m = omega._metrics(oos_raw, oos_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        rows.append(_metric_row(
            f"{args.mode}_q{q:.2f}".replace(".", "p"),
            val_m,
            oos_m,
            current_val_m,
            current_oos_m,
            {"mode": str(args.mode), "quality_threshold": q},
        ))
        if not saved_src and abs(q - 0.80) < 1e-9:
            val_src.to_csv(out_dir / "validation_predictions_2025_asym_true3head.csv", index=False)
            oos_src.to_csv(out_dir / "oos_predictions_2026_asym_true3head.csv", index=False)
            saved_src = True

    ranking = pd.DataFrame(rows).sort_values(["selection_score_val_only", "val_delta_vs_current", "val_pnl"], ascending=False).reset_index(drop=True)
    ranking.to_csv(out_dir / "asymmetric_direction_cleanup_ranking.csv", index=False)
    selected = ranking.iloc[0].to_dict()
    best_oos = ranking.sort_values(["oos_pnl", "oos_delta_vs_current"], ascending=False).iloc[0].to_dict()
    torch.save({"models": models, "base_cols": base_cols, "pos_cols": threehead.POS_COLS, "config": threehead.CFG.__dict__}, out_dir / "asymmetric_true_3head_tabm_bundle.pt")
    report = {
        "model_id": MODEL_ID,
        "status": "redteam_pass_asymmetric_direction_cleanup_eval",
        "method": "Relabel toxic directional training examples to cash using train-only trade simulation, retrain the same 3-head TabM experts, then evaluate through the operational overlay + aggressive sleeve path.",
        "mode": str(args.mode),
        "parent_dir": str(full_parent.PARENT_DIR),
        "current_quality_gate_parent": {"validation": current_val_m, "oos": current_oos_m},
        "selected_by_validation": selected,
        "best_by_oos_diagnostic": best_oos,
        "top30": ranking.head(30).to_dict(orient="records"),
        "cleanup_diagnostics": cleanup_diag,
        "exit_label": {"exit_edge_min": float(args.exit_edge_min), "hold_offsets": hold_offsets, "diag": exit_diag},
        "summaries": summaries,
        "artifacts": {
            "out_dir": str(out_dir),
            "ranking": str(out_dir / "asymmetric_direction_cleanup_ranking.csv"),
            "report": str(out_dir / "report.json"),
            "bundle": str(out_dir / "asymmetric_true_3head_tabm_bundle.pt"),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "selected": selected, "best_oos": best_oos, "cleanup": cleanup_diag}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
