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

import train_eval_omega1_2_tabm_3head_20260603 as th  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_tabm_exit_head_20260603 as exit_head  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


MODEL_ID = "omega1_2_pathlabel_3head_20260604"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
THR_GRID = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
DEFAULT_TP = 0.026
DEFAULT_SL = 0.012
SCALE_MAP = {"bull": 0.65, "bear": 0.90, "chop_expert": 0.90}
BASE_SCALES = {"bull": 0.75, "bear": 0.90, "chop_expert": 0.90}


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if (name == "cuda" or (name == "auto" and torch.cuda.is_available())) else "cpu")


def _path_net(
    close: np.ndarray,
    *,
    idx: int,
    side: int,
    notional: float,
    fee_eff: float,
    slip_eff: float,
    take_profit: float,
    stop_loss: float,
    forward_window: int,
) -> float:
    if idx >= len(close) - 2:
        return 0.0
    entry_idx = min(idx + 1, len(close) - 1)
    entry = float(close[entry_idx]) * (1.0 + slip_eff if side > 0 else 1.0 - slip_eff)
    cash = 1.0 - fee_eff * notional
    end = min(idx + int(forward_window), len(close) - 1)
    exit_px = float(close[end]) * (1.0 - slip_eff if side > 0 else 1.0 + slip_eff)
    for j in range(entry_idx, end + 1):
        px = float(close[j])
        raw = (px * (1.0 - slip_eff) - entry) / max(entry, 1e-12) if side > 0 else (entry - px * (1.0 + slip_eff)) / max(entry, 1e-12)
        unreal = raw * notional
        if unreal <= -abs(float(stop_loss)):
            exit_px = px * (1.0 - slip_eff if side > 0 else 1.0 + slip_eff)
            break
        if unreal >= float(take_profit):
            exit_px = px * (1.0 - slip_eff if side > 0 else 1.0 + slip_eff)
            break
    raw_exit = (exit_px - entry) / max(entry, 1e-12) if side > 0 else (entry - exit_px) / max(entry, 1e-12)
    cash = cash * (1.0 + raw_exit * notional)
    cash -= cash * fee_eff * notional
    return float(cash - 1.0)


def _build_path_labels(
    frame: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    take_profit: float,
    stop_loss: float,
    forward_window: int,
    min_edge: float,
) -> np.ndarray:
    close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
    route = hard._route_id(frame)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    labels = np.zeros(len(frame), dtype=np.int64)
    for i in range(len(frame) - 2):
        expert = hard.EXPERT_NAMES[int(route[i])]
        notional = float(omega.BASE_TEMPLATE["notional"]) * float(SCALE_MAP.get(expert, 1.0)) / float(BASE_SCALES.get(expert, 1.0))
        long_net = _path_net(
            close,
            idx=i,
            side=1,
            notional=notional,
            fee_eff=fee_eff,
            slip_eff=slip_eff,
            take_profit=take_profit,
            stop_loss=stop_loss,
            forward_window=forward_window,
        )
        short_net = _path_net(
            close,
            idx=i,
            side=-1,
            notional=notional,
            fee_eff=fee_eff,
            slip_eff=slip_eff,
            take_profit=take_profit,
            stop_loss=stop_loss,
            forward_window=forward_window,
        )
        if long_net > short_net and long_net >= float(min_edge):
            labels[i] = omega.ACTION_LONG
        elif short_net > long_net and short_net >= float(min_edge):
            labels[i] = omega.ACTION_SHORT
        else:
            labels[i] = omega.ACTION_CASH
    return labels


def _scale_dec(dec: pd.DataFrame, *, take_profit: float, stop_loss: float) -> pd.DataFrame:
    out = dec.copy()
    active = omega._active(out)
    for expert, scale in SCALE_MAP.items():
        mask = active & out["router_expert"].astype(str).eq(expert)
        ratio = float(scale) / float(BASE_SCALES[expert])
        out.loc[mask, "notional_exposure"] = pd.to_numeric(out.loc[mask, "notional_exposure"], errors="raise") * ratio
        out.loc[mask, "position_fraction"] = pd.to_numeric(out.loc[mask, "position_fraction"], errors="raise") * ratio
    active = omega._active(out)
    out.loc[active, "take_profit"] = float(take_profit)
    out.loc[active, "stop_loss"] = float(stop_loss)
    out.loc[active, "max_hold_bars"] = 0
    out.loc[active, "cooldown_bars"] = 0
    return out


def _set_threshold(src: pd.DataFrame, prefix: str, threshold: float) -> pd.DataFrame:
    out = src.copy()
    q = pd.to_numeric(out[f"{prefix}quality_for_action"], errors="raise").to_numpy(dtype=np.float64)
    action = pd.to_numeric(out[f"{prefix}dir_action"], errors="raise").to_numpy(dtype=np.int64)
    out[f"{prefix}quality_threshold"] = float(threshold)
    out[f"{prefix}final_action"] = np.where(q >= float(threshold), action, omega.ACTION_CASH).astype(np.int64)
    return out


def _decisions(src: pd.DataFrame, prefix: str, *, threshold: float, oof: bool, take_profit: float, stop_loss: float) -> pd.DataFrame:
    return _scale_dec(
        omega._to_fixed_decisions(_set_threshold(src, prefix, threshold), oof=oof),
        take_profit=take_profit,
        stop_loss=stop_loss,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=36)
    ap.add_argument("--forward-window", type=int, default=72)
    ap.add_argument("--min-edge", type=float, default=0.0010)
    ap.add_argument("--exit-edge-min", type=float, default=0.0020)
    ap.add_argument("--exit-thresholds", default="0.90,0.95,0.98")
    ap.add_argument("--max-exit-samples", type=int, default=30000)
    ap.add_argument("--tp", type=float, default=DEFAULT_TP)
    ap.add_argument("--sl", type=float, default=DEFAULT_SL)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260604)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()
    device = _device(args.device)
    th._seed_everything(int(args.seed))
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{args.out_suffix.strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = th._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    base_cols = list(frames["feature_cols"])
    x_train = th._base_input(frames["train_raw"], base_cols)
    y_train = _build_path_labels(
        frames["train_raw"],
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        take_profit=float(args.tp),
        stop_loss=float(args.sl),
        forward_window=int(args.forward_window),
        min_edge=float(args.min_edge),
    )
    hold_offsets = [1, 2, 3, 6, 12, 24, 48, 96, 192, 384]
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
    x_exit = th._exit_input_from_position_rows(x_exit_raw, base_cols)
    models: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        payload = th._fit_expert_3head(
            x_train,
            y_train,
            frames["train_raw"],
            x_exit,
            y_exit,
            frame_exit,
            expert_idx=idx,
            seed=int(args.seed),
            epochs=int(args.epochs),
            device=device,
            model_path=out_dir / "models" / f"{expert}_pathlabel_3head_tabm.pt",
        )
        models[expert] = payload
        summaries[expert] = {"epochs_ran": int(payload["epochs_ran"]), "best_validation_loss": float(payload["best_validation_loss"])}

    def predict(frame: pd.DataFrame, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        x = th._base_input(frame, base_cols)
        preds = {expert: th._predict_payload(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
        route = hard._route_id(frame)
        direction = th._routed(preds, route, "direction", 3)
        quality = th._routed(preds, route, "quality", 3)
        src = th._prediction_output(frame, direction, quality, threshold=0.50, prefix=prefix.rstrip("_"))
        return x, src

    _x_val, val_src = predict(frames["val_raw"], "omega1_regime3_expertdq_oof_")
    _x_oos, oos_src_oof = predict(frames["oos_raw"], "omega1_regime3_expertdq_oof_")
    oos_src = oos_src_oof.rename(columns={c: c.replace("omega1_regime3_expertdq_oof_", "omega1_regime3_expertdq_") for c in oos_src_oof.columns})
    loaded_models = th._load_payloads(models, device=device)
    exit_thresholds = [float(x) for x in str(args.exit_thresholds).split(",") if str(x).strip()]
    rows: list[dict[str, Any]] = []
    for thr in THR_GRID:
        val_dec = _decisions(val_src, "omega1_regime3_expertdq_oof_", threshold=thr, oof=True, take_profit=float(args.tp), stop_loss=float(args.sl))
        oos_dec = _decisions(oos_src, "omega1_regime3_expertdq_", threshold=thr, oof=False, take_profit=float(args.tp), stop_loss=float(args.sl))
        val = omega._metrics(frames["val_raw"], val_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        oos = omega._metrics(frames["oos_raw"], oos_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        rows.append({"variant": "no_exit", "threshold": thr, "exit_threshold": 0.0, "val_pnl": val["pnl"], "val_mdd": val["mdd"], "val_wr": val["wr"], "val_trades": val["trades"], "oos_pnl": oos["pnl"], "oos_mdd": oos["mdd"], "oos_wr": oos["wr"], "oos_trades": oos["trades"], "val_exit_reasons": val.get("exit_reasons", {}), "oos_exit_reasons": oos.get("exit_reasons", {})})
        for exit_thr in exit_thresholds:
            val_exit = th._metrics_with_shared_exit(frames["val_raw"], _x_val, val_dec, loaded_models, threshold=float(exit_thr), fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)
            oos_exit = th._metrics_with_shared_exit(frames["oos_raw"], _x_oos, oos_dec, loaded_models, threshold=float(exit_thr), fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)
            rows.append({"variant": "exit_head", "threshold": thr, "exit_threshold": float(exit_thr), "val_pnl": val_exit["pnl"], "val_mdd": val_exit["mdd"], "val_wr": val_exit["wr"], "val_trades": val_exit["trades"], "oos_pnl": oos_exit["pnl"], "oos_mdd": oos_exit["mdd"], "oos_wr": oos_exit["wr"], "oos_trades": oos_exit["trades"], "val_exit_reasons": val_exit.get("exit_reasons", {}), "oos_exit_reasons": oos_exit.get("exit_reasons", {})})
    ranking = pd.DataFrame(rows).sort_values(["val_pnl", "val_wr"], ascending=False)
    ranking.to_csv(out_dir / "ranking.csv", index=False)
    torch.save({"models": models, "base_cols": base_cols, "pos_cols": th.POS_COLS, "config": th.CFG.__dict__}, out_dir / "pathlabel_3head_tabm_bundle.pt")
    report = {
        "model_id": MODEL_ID,
        "design": "Per-expert true 3-head TabM retrained with Cost3 TP/SL path labels for Direction/Quality instead of zigzag labels. Runtime still uses no max_hold/cooldown.",
        "label": {"forward_window": int(args.forward_window), "min_edge": float(args.min_edge), "tp": float(args.tp), "sl": float(args.sl), "class_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y_train, minlength=3))}},
        "exit_label": {"exit_edge_min": float(args.exit_edge_min), "diag": exit_diag},
        "summaries": summaries,
        "ranking": rows,
        "artifacts": {"out_dir": str(out_dir), "ranking": str(out_dir / "ranking.csv")},
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(ranking.to_string(index=False))
    print(json.dumps({"report": str(out_dir / "report.json"), "ranking": str(out_dir / "ranking.csv")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
