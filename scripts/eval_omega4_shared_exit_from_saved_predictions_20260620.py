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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_quality_regression_20260621 as qreg  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


SPLIT_TS = pd.Timestamp("2025-10-01")


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _load_frame(split: str) -> pd.DataFrame:
    if split == "validation":
        frame = omega._read(omega.TRAIN_CSV)
        frame, _ = omega._overlay_required(frame, omega.REGIME3_CURRENT_2025, omega.REGIME3_CURRENT_COLS, tag="val_regime3_current")
        frame, _ = omega._overlay_required(frame, omega.REGIME3_CMAMBA_2025, omega.REGIME3_CMAMBA_COLS, tag="val_regime3_cmamba")
        frame, _ = omega._overlay_required(frame, omega.REGIME3_RISK_2025, omega.REGIME3_RISK_COLS, tag="val_regime3_risk")
        return frame[frame["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    if split == "oos":
        frame = omega._read(omega.EVAL_CSV)
        frame, _ = omega._overlay_required(frame, omega.REGIME3_CURRENT_2026, omega.REGIME3_CURRENT_COLS, tag="oos_regime3_current")
        frame, _ = omega._overlay_required(frame, omega.REGIME3_CMAMBA_2026, omega.REGIME3_CMAMBA_COLS, tag="oos_regime3_cmamba")
        frame, _ = omega._overlay_required(frame, omega.REGIME3_RISK_2026, omega.REGIME3_RISK_COLS, tag="oos_regime3_risk")
        return frame.reset_index(drop=True)
    raise RuntimeError(f"unknown split: {split}")


def _align_prediction_frame(frame: pd.DataFrame, pred_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = omega._read(pred_path)
    out_frame, out_pred = omega._align(frame, pred, f"saved_pred_{pred_path.name}")
    return out_frame, out_pred


def _predict_frame_from_bundle(
    frame: pd.DataFrame,
    models: dict[str, dict[str, Any]],
    base_cols: list[str],
    *,
    quality_threshold: float,
    oof: bool,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = parent._base_input(frame, base_cols)
    preds = {expert: parent._predict_payload(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
    route = hard._route_id(frame)
    direction = parent._routed(preds, route, "direction", 3)
    quality = parent._routed(preds, route, "quality", 3)
    prefix = "omega1_regime3_expertdq_oof" if oof else "omega1_regime3_expertdq"
    pred = parent._prediction_output(frame, direction, quality, threshold=float(quality_threshold), prefix=prefix)
    return frame, pred


def _load_bundle(bundle_path: Path, *, device: torch.device) -> tuple[dict[str, dict[str, Any]], list[str], str]:
    payload = torch.load(bundle_path, map_location=device, weights_only=False)
    return payload["models"], list(payload["base_cols"]), str(payload.get("model_class", "ThreeHeadTabM"))


@torch.no_grad()
def _predict_exit_prob_any(model_payload: dict[str, Any], x: pd.DataFrame, *, model_class: str, device: torch.device) -> np.ndarray:
    if model_class == "ThreeHeadQualityRegTabM":
        pred = qreg._predict_payload(model_payload, x, device=device)
        return pred["exit"][:, 1]
    model = parent.ThreeHeadTabM(int(model_payload["n_features"]), cfg=parent.CFG).to(device)
    model.load_state_dict(model_payload["state_dict"])
    model.eval()
    x_np = parent._standardize_apply(x, model_payload["scaler"])
    out: list[np.ndarray] = []
    for start in range(0, len(x_np), 8192):
        xb = torch.from_numpy(x_np[start : start + 8192]).to(device)
        probs = torch.softmax(model(xb)["exit"], dim=-1).mean(dim=1)
        out.append(probs.detach().cpu().numpy()[:, 1])
    return np.concatenate(out, axis=0).astype(np.float64)


def _metrics_with_shared_exit_any(
    frame: pd.DataFrame,
    base_x: pd.DataFrame,
    dec: pd.DataFrame,
    models: dict[str, dict[str, Any]],
    *,
    model_class: str,
    threshold: float,
    fee: float,
    slip: float,
    cost_mult: float,
    device: torch.device,
) -> dict[str, Any]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_i = 0
    notional = 0.0
    leverage = 1.0
    take_profit = 0.0
    stop_loss = 0.0
    mfe = 0.0
    mae = 0.0
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    reasons: dict[str, int] = {}
    route = hard._route_id(frame)
    for i in range(0, len(frame) - 2):
        if pos != 0:
            px = float(arrays["close"][i])
            raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
            unreal = raw * notional
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            eq = cash * (1.0 + unreal)
        else:
            unreal = 0.0
            eq = cash
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if pos != 0:
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "stop_loss"
            else:
                xrow = base_x.iloc[[i]].copy().reset_index(drop=True)
                hold = max(int(i) - int(entry_i), 0)
                giveback = (float(mfe) - float(unreal)) / max(abs(float(mfe)), 1e-8) if mfe > 0 else 0.0
                vals = {
                    "pos_side": float(pos),
                    "pos_hold_bars": float(hold),
                    "pos_unrealized": float(unreal),
                    "pos_mfe": float(mfe),
                    "pos_mae": float(mae),
                    "pos_giveback": float(np.clip(giveback, 0.0, 10.0)),
                    "pos_dist_to_tp": float(take_profit - unreal),
                    "pos_dist_to_sl": float(unreal + abs(stop_loss)),
                    "pos_notional": float(notional),
                    "pos_leverage": float(leverage),
                    "pos_exposure": float(notional * leverage),
                    "pos_tp": float(take_profit),
                    "pos_sl": float(stop_loss),
                }
                for col, val in vals.items():
                    xrow[col] = val
                expert = hard.EXPERT_NAMES[int(route[i])]
                prob = float(_predict_exit_prob_any(models[expert], xrow, model_class=model_class, device=device)[0])
                if prob >= float(threshold):
                    reason = "exit_head"
            if reason:
                filled, exit_px, exit_fee, _route = omega._try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * exit_fee * notional
                trades += 1
                wins += int(cash > entry_equity)
                reasons[reason] = reasons.get(reason, 0) + 1
                pos = 0
                continue
        if pos != 0 or not bool(active[i]):
            continue
        row = dec.iloc[i]
        side = int(row.get("side", 0) or 0)
        if side == 0:
            continue
        filled, px, entry_fee, _route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        pos = side
        entry_price = float(px)
        entry_equity = cash
        entry_i = min(int(i) + 1, len(frame) - 1)
        notional = float(row.get("notional_exposure", 0.0) or 0.0)
        leverage = float(row.get("leverage", 1.0) or 1.0)
        take_profit = float(row.get("take_profit", 0.0) or 0.0)
        stop_loss = float(row.get("stop_loss", 0.0) or 0.0)
        cash -= cash * entry_fee * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        mfe = 0.0
        mae = 0.0
    if pos != 0:
        exit_px = omega._fill_price(arrays, len(frame) - 1, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        trades += 1
        wins += int(cash > entry_equity)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / trades) if trades else 0.0,
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exit_reasons": reasons,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--validation-pred", type=Path, default=None)
    ap.add_argument("--oos-pred", type=Path, default=None)
    ap.add_argument("--quality-threshold", type=float, default=0.45)
    ap.add_argument("--exit-thresholds", default="0.45,0.50,0.60,0.70,0.80,0.90")
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    omega.BASE_TEMPLATE["max_hold"] = 0
    omega.BASE_TEMPLATE["cooldown"] = 0
    device = parent._device(str(args.device))
    if args.verbose:
        print("loading bundle", flush=True)
    models, base_cols, model_class = _load_bundle(Path(args.bundle), device=device)
    if args.verbose:
        print("loading payloads", flush=True)
    loaded = parent._load_payloads(models, device=device) if model_class != "ThreeHeadQualityRegTabM" else None
    fee, slip = omega._load_fee_slip()
    results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    split_inputs: list[tuple[str, Path | None]] = [("oos", Path(args.oos_pred) if args.oos_pred is not None else None)]
    if args.validation_pred is not None:
        split_inputs.insert(0, ("validation", Path(args.validation_pred)))
    split_data: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for split, pred_path in split_inputs:
        if args.verbose:
            print(f"loading frame {split}", flush=True)
        frame = _load_frame(split)
        if pred_path is None:
            if args.verbose:
                print(f"predicting bundle {split} q={args.quality_threshold}", flush=True)
            frame, pred = _predict_frame_from_bundle(
                frame,
                models,
                base_cols,
                quality_threshold=float(args.quality_threshold),
                oof=(split == "validation"),
                device=device,
            )
        else:
            if args.verbose:
                print(f"aligning predictions {split}", flush=True)
            frame, pred = _align_prediction_frame(frame, pred_path)
        dec = parent._to_decisions(pred, oof=(split == "validation"))
        x_base = parent._base_input(frame, base_cols)
        split_data[split] = (frame, x_base, dec)
    no_exit: dict[str, Any] = {}
    for split, (frame, _x_base, dec) in split_data.items():
        no_exit[split] = omega._metrics(frame, dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    results["no_exit_head"] = no_exit

    for thr in [float(x.strip()) for x in str(args.exit_thresholds).split(",") if x.strip()]:
        if args.verbose:
            print(f"evaluating exit threshold {thr}", flush=True)
        key = f"exit_thr_{thr:.2f}".replace(".", "p")
        results[key] = {}
        row: dict[str, Any] = {"variant": key, "exit_threshold": thr}
        for split, (frame, x_base, dec) in split_data.items():
            if loaded is None:
                metrics = _metrics_with_shared_exit_any(frame, x_base, dec, models, model_class=model_class, threshold=thr, fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)
            else:
                metrics = parent._metrics_with_shared_exit(frame, x_base, dec, loaded, threshold=thr, fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)
            results[key][split] = metrics
            row[f"{split}_pnl"] = float(metrics["pnl"])
            row[f"{split}_mdd"] = float(metrics["mdd"])
            row[f"{split}_wr"] = float(metrics["wr"])
            row[f"{split}_trades"] = int(metrics["trades"])
        rows.append(row)
    rows.sort(key=lambda r: (float(r.get("oos_pnl", -1.0e9)), float(r.get("validation_pnl", -1.0e9))), reverse=True)
    report = {
        "bundle": str(args.bundle),
        "model_class": model_class,
        "quality_threshold": float(args.quality_threshold),
        "prediction_inputs": {split: str(path) if path is not None else "computed_from_bundle" for split, path in split_inputs},
        "results": results,
        "ranking_by_oos_pnl": rows,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.out_json), "top": rows[:5], "no_exit_head": no_exit}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
