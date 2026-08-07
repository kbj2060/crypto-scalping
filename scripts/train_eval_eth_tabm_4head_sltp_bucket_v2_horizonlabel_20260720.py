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
import train_eval_omega1_2_tabm_exit_head_20260603 as exit_head  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as omega4  # noqa: E402
import eval_omega4_1_atr_safety_sltp_20260622 as atr_safety  # noqa: E402
import train_eval_eth_tabm_4head_sltp_20260720 as reg_variant  # noqa: E402
import train_eval_eth_tabm_4head_sltp_bucket_20260720 as bucket_v1  # noqa: E402


MODEL_ID = "eth_tabm_4head_sltp_bucket_v2_horizonlabel_20260720"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

MIN_TP = bucket_v1.MIN_TP
MAX_TP = bucket_v1.MAX_TP
MIN_SL = bucket_v1.MIN_SL
MAX_SL = bucket_v1.MAX_SL
N_LEVELS = bucket_v1.N_LEVELS
TP_LEVELS = bucket_v1.TP_LEVELS
SL_LEVELS = bucket_v1.SL_LEVELS


def _long_horizon_mfe_mae(frame: pd.DataFrame, action: np.ndarray, *, max_horizon_bars: int) -> tuple[np.ndarray, np.ndarray]:
    """For each active (long/short) row, scan forward up to max_horizon_bars closes and record the
    best-favorable / worst-adverse raw price move. Unlike zigzag_path_mfe/mae (scoped to the short
    zigzag-pivot segment, which turned out to almost never reach the 7.5-22%/4-12% live safety
    envelope -- see the v1 diagnosis), this measures the full realistic forward opportunity over a
    horizon long enough to actually span that envelope (empirically checked: 2016 bars / 1 week at
    5m puts ~18-25% of TP targets and ~50% of SL targets above the floor, vs 1.6%/13% at 288 bars).
    Training-label only; never used as a live inference input."""
    close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
    n = len(close)
    mfe = np.zeros(n, dtype=np.float64)
    mae = np.zeros(n, dtype=np.float64)
    active_idx = np.flatnonzero(action != 0)
    for i in active_idx:
        side = 1 if action[i] == 1 else -1
        entry = close[i]
        end = min(int(i) + int(max_horizon_bars), n - 1)
        path = close[i + 1 : end + 1]
        if len(path) == 0:
            continue
        moves = (path - entry) / entry if side > 0 else (entry - path) / entry
        mfe[i] = float(moves.max())
        mae[i] = float(moves.min())
    return mfe, mae


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction-label-dir", type=Path, default=ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_parent72_loose_20260620")
    ap.add_argument("--quality-mode", default="same_as_direction")
    ap.add_argument("--quality-threshold", type=float, default=0.75)
    ap.add_argument("--exit-threshold", type=float, default=0.97)
    ap.add_argument("--epochs", type=int, default=28)
    ap.add_argument("--exit-edge-min", type=float, default=0.0020)
    ap.add_argument("--exit-hold-offsets", default="1,2,3,6,12,24,48,96,192,384")
    ap.add_argument("--max-exit-samples", type=int, default=0)
    ap.add_argument("--max-train-rows", type=int, default=0)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260720)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--tp-capture-frac", type=float, default=0.70)
    ap.add_argument("--sl-capture-frac", type=float, default=0.90)
    ap.add_argument("--sltp-loss-weight", type=float, default=0.35)
    ap.add_argument("--label-horizon-bars", type=int, default=2016)
    ap.add_argument("--atr-window", type=int, default=192)
    ap.add_argument("--atr-tp-mult", type=float, default=12.0)
    ap.add_argument("--atr-sl-mult", type=float, default=6.0)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    omega4._seed_everything(int(args.seed))
    device = parent._device(str(args.device))
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=Path(args.direction_label_dir),
        quality_mode=str(args.quality_mode),
        quality_label_dir=None,
        quality_min_edge=0.0,
        quality_max_mae=1.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )
    fee, slip = omega._load_fee_slip()
    base_cols = list(frames["feature_cols"])
    train_raw = frames["train_raw"]
    val_raw = frames["val_raw"]
    oos_raw = frames["oos_raw"]

    x_train = parent._base_input(train_raw, base_cols)
    y_train = train_raw["zigzag_action"].to_numpy(dtype=np.int64)
    y_quality = train_raw["omega4_quality_action"].to_numpy(dtype=np.int64)
    train_mfe, train_mae = _long_horizon_mfe_mae(train_raw, y_train, max_horizon_bars=int(args.label_horizon_bars))
    tp_bucket, sl_bucket = bucket_v1._sltp_bucket_targets(y_train, train_mfe, train_mae, tp_capture_frac=float(args.tp_capture_frac), sl_capture_frac=float(args.sl_capture_frac))
    label_diag = {
        "label_horizon_bars": int(args.label_horizon_bars),
        "active_rows": int((y_train != 0).sum()),
        "tp_bucket_counts": {str(k): int(v) for k, v in zip(*np.unique(tp_bucket[y_train != 0], return_counts=True))},
        "sl_bucket_counts": {str(k): int(v) for k, v in zip(*np.unique(sl_bucket[y_train != 0], return_counts=True))},
        "mfe_abs_p50": float(np.quantile(np.abs(train_mfe[y_train != 0]), 0.50)),
        "mfe_abs_p90": float(np.quantile(np.abs(train_mfe[y_train != 0]), 0.90)),
        "mae_abs_p50": float(np.quantile(np.abs(train_mae[y_train != 0]), 0.50)),
        "mae_abs_p90": float(np.quantile(np.abs(train_mae[y_train != 0]), 0.90)),
    }

    if int(args.max_train_rows) > 0:
        limit = int(args.max_train_rows)
        x_train = x_train.iloc[:limit].reset_index(drop=True)
        y_train = y_train[:limit]
        y_quality = y_quality[:limit]
        tp_bucket = tp_bucket[:limit]
        sl_bucket = sl_bucket[:limit]
        train_fit_frame = train_raw.iloc[:limit].reset_index(drop=True)
    else:
        train_fit_frame = train_raw

    hold_offsets = [int(x.strip()) for x in str(args.exit_hold_offsets).split(",") if x.strip()]
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
    x_exit = parent._exit_input_from_position_rows(x_exit_raw, base_cols)

    cfg = bucket_v1.FourHeadBucketConfig(sltp_loss_weight=float(args.sltp_loss_weight), tp_capture_frac=float(args.tp_capture_frac), sl_capture_frac=float(args.sl_capture_frac))
    models: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        payload = bucket_v1._fit_expert_4head_bucket(
            x_train, y_train, y_quality, tp_bucket, sl_bucket,
            train_fit_frame, x_exit, y_exit, frame_exit,
            expert_idx=idx, seed=int(args.seed), epochs=int(args.epochs), device=device,
            model_path=out_dir / "models" / f"{expert}_4head_bucket_tabm.pt",
            cfg=cfg,
        )
        models[expert] = payload
        summaries[expert] = {"epochs_ran": int(payload["epochs_ran"]), "best_validation_loss": float(payload["best_validation_loss"])}

    def predict_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, dict[str, np.ndarray]]]:
        x = parent._base_input(frame, base_cols)
        preds = {expert: bucket_v1._predict_sltp_bucket(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
        route = hard._route_id(frame)
        direction = parent._routed(preds, route, "direction", 3)
        quality = parent._routed(preds, route, "quality", 3)
        out = parent._prediction_output(frame, direction, quality, threshold=float(args.quality_threshold), prefix="omega1_regime3_expertdq_oof")
        return x, out, route, preds

    x_val, val_src, val_route, val_sltp_preds = predict_frame(val_raw)
    x_oos, oos_src_oof, oos_route, oos_sltp_preds = predict_frame(oos_raw)
    oos_src = oos_src_oof.rename(columns={c: c.replace("omega1_regime3_expertdq_oof_", "omega1_regime3_expertdq_") for c in oos_src_oof.columns})
    val_dec_base = parent._to_decisions(val_src, oof=True)
    oos_dec_base = parent._to_decisions(oos_src, oof=False)

    loaded_models = bucket_v1._load_payloads_4head_bucket(models, device=device)

    val_dec_atr, val_atr_diag = atr_safety._apply_atr_safety_sltp(
        val_dec_base, val_raw, atr_window=int(args.atr_window), tp_mult=float(args.atr_tp_mult), sl_mult=float(args.atr_sl_mult),
        min_tp=MIN_TP, min_sl=MIN_SL, max_tp=MAX_TP, max_sl=MAX_SL,
    )
    oos_dec_atr, oos_atr_diag = atr_safety._apply_atr_safety_sltp(
        oos_dec_base, oos_raw, atr_window=int(args.atr_window), tp_mult=float(args.atr_tp_mult), sl_mult=float(args.atr_sl_mult),
        min_tp=MIN_TP, min_sl=MIN_SL, max_tp=MAX_TP, max_sl=MAX_SL,
    )
    val_dec_learned, val_learned_diag = reg_variant._apply_learned_sltp(val_dec_base, val_sltp_preds, val_route)
    oos_dec_learned, oos_learned_diag = reg_variant._apply_learned_sltp(oos_dec_base, oos_sltp_preds, oos_route)

    val_m_atr = parent._metrics_with_shared_exit(val_raw, x_val, val_dec_atr, loaded_models, threshold=float(args.exit_threshold), fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)
    oos_m_atr = parent._metrics_with_shared_exit(oos_raw, x_oos, oos_dec_atr, loaded_models, threshold=float(args.exit_threshold), fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)
    val_m_learned = parent._metrics_with_shared_exit(val_raw, x_val, val_dec_learned, loaded_models, threshold=float(args.exit_threshold), fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)
    oos_m_learned = parent._metrics_with_shared_exit(oos_raw, x_oos, oos_dec_learned, loaded_models, threshold=float(args.exit_threshold), fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)

    report = {
        "model_id": MODEL_ID,
        "design": "Same FourHeadTabMBucket architecture as bucket_v1, but the TP/SL bucket labels are now derived from a long-horizon (2016 bars = ~1 week at 5m) forward-path MFE/MAE scan instead of the short zigzag-pivot-segment mfe/mae. v1's diagnosis: zigzag_path_mfe/mae almost never exceeded the safety floor (99.98% TP / 100% SL rows fell in the floor bucket), so no loss function could learn a differentiated target. Empirically re-checked before this run: at 2016 bars, ~18-25% of TP targets and ~50% of SL targets land above the floor.",
        "caveats": [
            "fresh_forward_bar_by_bar=true for the backtest replay itself, but train/val/oos split is this script family's legacy convention (train<2025-10-01, val>=2025-10-01 in 2025, oos=full 2026), NOT the project's canonical 2025-09-01..12-31 / 2026-01-01..03-31 split -- rerun on canonical split before treating as promotion evidence.",
            "trade_ledgers_used_as_input=false; the long-horizon mfe/mae scan is a training-label construction only (standard supervised-learning label design, same category as the h48 quality rule's own barrier-diagnostic labels already used elsewhere in this codebase), never a model input at inference.",
            "v1 predecessors (train_eval_eth_tabm_4head_sltp_20260720.py regression, train_eval_eth_tabm_4head_sltp_bucket_20260720.py bucket) both collapsed to the ATR floor because the zigzag-segment mfe/mae label source itself almost never exceeded the floor -- this run isolates whether that was the true bottleneck by fixing only the label source, keeping architecture/loss/exit_threshold identical to bucket_v1.",
            "direction/quality/exit heads are retrained from scratch (new architecture can't warm-start from live h48qual/zig075 bundles) -- not a drop-in live replacement even if the sltp mechanism wins here.",
        ],
        "quality_threshold": float(args.quality_threshold),
        "exit_threshold": float(args.exit_threshold),
        "exit_label_diag": exit_diag,
        "sltp_label_diag": label_diag,
        "sltp_targets": {
            "tp_capture_frac": float(args.tp_capture_frac), "sl_capture_frac": float(args.sl_capture_frac), "sltp_loss_weight": float(args.sltp_loss_weight),
            "label_horizon_bars": int(args.label_horizon_bars), "n_levels": N_LEVELS, "tp_levels": TP_LEVELS.tolist(), "sl_levels": SL_LEVELS.tolist(),
        },
        "summaries": summaries,
        "results": {
            "baseline_atr_fixed_formula": {"validation": val_m_atr, "oos": oos_m_atr, "validation_atr_diag": val_atr_diag, "oos_atr_diag": oos_atr_diag},
            "learned_sltp_bucket_head": {"validation": val_m_learned, "oos": oos_m_learned, "validation_sltp_diag": val_learned_diag, "oos_sltp_diag": oos_learned_diag},
        },
        "artifacts": {"out_dir": str(out_dir), "report": str(out_dir / "report.json")},
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=omega._json_default) + "\n", encoding="utf-8")
    torch.save({"models": models, "base_cols": base_cols, "pos_cols": parent.POS_COLS, "config": cfg.__dict__}, out_dir / "eth_4head_bucket_v2_tabm_bundle.pt")
    print(json.dumps({
        "report": str(out_dir / "report.json"),
        "label_diag": label_diag,
        "baseline_atr": {"validation": val_m_atr, "oos": oos_m_atr},
        "learned_sltp_bucket": {"validation": val_m_learned, "oos": oos_m_learned},
    }, ensure_ascii=False, indent=2, default=omega._json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
