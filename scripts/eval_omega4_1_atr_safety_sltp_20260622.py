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
import train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622 as price_exit  # noqa: E402


MODEL_ID = "omega4_1_atr_safety_sltp_20260622"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
BASELINE_BUNDLE = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_3head_parent72_loose_entry_quality_20260620_smoke_loose_entry_loose_quality_terminal_giveback_exit_e2_train15k_exit15k_q070"
    / "true_3head_tabm_bundle.pt"
)
PREDICTION_DIR = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_frozen_risk_heads_margin_leverage_20260622_e8_fulltrain_q070_price_move_contract"
)
SPLIT_TS = pd.Timestamp("2025-10-01")


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _atr_pct(frame: pd.DataFrame, window: int) -> np.ndarray:
    high = pd.to_numeric(frame["high"], errors="raise").to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="raise").to_numpy(dtype=np.float64)
    close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    atr = pd.Series(tr).rolling(window=max(int(window), 1), min_periods=1).mean().to_numpy(dtype=np.float64)
    out = atr / np.maximum(close, 1.0e-12)
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite ATR percent")
    return out


def _apply_atr_safety_sltp(
    dec: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    atr_window: int,
    tp_mult: float,
    sl_mult: float,
    min_tp: float,
    min_sl: float,
    max_tp: float,
    max_sl: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = dec.copy().reset_index(drop=True)
    if len(out) != len(frame):
        raise RuntimeError(f"decision/frame length mismatch: {len(out)} vs {len(frame)}")
    active = omega._active(out)
    atr = _atr_pct(frame, int(atr_window))
    tp = np.clip(np.maximum(float(min_tp), atr * float(tp_mult)), 0.0, float(max_tp))
    sl = np.clip(np.maximum(float(min_sl), atr * float(sl_mult)), 0.0, float(max_sl))
    out.loc[active, "take_profit"] = tp[active]
    out.loc[active, "stop_loss"] = sl[active]
    out.loc[~active, ["take_profit", "stop_loss"]] = 0.0
    active_tp = tp[active]
    active_sl = sl[active]
    diag = {
        "atr_window": int(atr_window),
        "tp_mult": float(tp_mult),
        "sl_mult": float(sl_mult),
        "min_tp": float(min_tp),
        "min_sl": float(min_sl),
        "max_tp": float(max_tp),
        "max_sl": float(max_sl),
        "active_rows": int(active.sum()),
        "atr_pct_p50": float(np.quantile(atr[active], 0.50)) if bool(active.any()) else 0.0,
        "atr_pct_p90": float(np.quantile(atr[active], 0.90)) if bool(active.any()) else 0.0,
        "tp_p50": float(np.quantile(active_tp, 0.50)) if len(active_tp) else 0.0,
        "tp_p90": float(np.quantile(active_tp, 0.90)) if len(active_tp) else 0.0,
        "sl_p50": float(np.quantile(active_sl, 0.50)) if len(active_sl) else 0.0,
        "sl_p90": float(np.quantile(active_sl, 0.90)) if len(active_sl) else 0.0,
    }
    return out, diag


def _load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_all, eval_df, _overlay = omega._load_omega_frames()
    val_raw = train_all[pd.to_datetime(train_all["timestamp"], errors="raise") >= SPLIT_TS].reset_index(drop=True)
    oos_raw = eval_df.reset_index(drop=True)
    return val_raw, oos_raw


def _load_decisions() -> tuple[pd.DataFrame, pd.DataFrame]:
    val_src = pd.read_csv(PREDICTION_DIR / "validation_predictions_baseline_entry_q070.csv")
    oos_src = pd.read_csv(PREDICTION_DIR / "oos_predictions_baseline_entry_q070.csv")
    return parent._to_decisions(val_src, oof=True), parent._to_decisions(oos_src, oof=False)


def _assert_aligned(name: str, frame: pd.DataFrame, pred_path: Path) -> None:
    pred_ts = pd.read_csv(pred_path, usecols=["timestamp"], parse_dates=["timestamp"])["timestamp"].reset_index(drop=True)
    frame_ts = pd.to_datetime(frame["timestamp"], errors="raise").reset_index(drop=True)
    if len(pred_ts) != len(frame_ts) or not frame_ts.equals(pd.to_datetime(pred_ts, errors="raise")):
        raise RuntimeError(f"{name}: frame and prediction timestamps are not aligned")


def _parse_configs(text: str) -> list[dict[str, float]]:
    configs: list[dict[str, float]] = []
    for raw in str(text).split(";"):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(",")
        if len(parts) != 3:
            raise RuntimeError(f"invalid config '{raw}', expected window,tp_mult,sl_mult")
        configs.append({"atr_window": int(parts[0]), "tp_mult": float(parts[1]), "sl_mult": float(parts[2])})
    if not configs:
        raise RuntimeError("empty ATR config list")
    return configs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-bundle", type=Path, default=BASELINE_BUNDLE)
    ap.add_argument("--configs", default="14,6,3;48,6,3;48,8,4;96,8,4;48,10,5;96,10,5")
    ap.add_argument("--quality-threshold", type=float, default=0.70)
    ap.add_argument("--exit-threshold", type=float, default=0.70)
    ap.add_argument("--min-tp", type=float, default=0.026)
    ap.add_argument("--min-sl", type=float, default=0.014)
    ap.add_argument("--max-tp", type=float, default=0.12)
    ap.add_argument("--max-sl", type=float, default=0.06)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--out-suffix", default="q070_exit070")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    args = ap.parse_args()

    omega.BASE_TEMPLATE["max_hold"] = 0
    omega.BASE_TEMPLATE["cooldown"] = 0
    device = parent._device(str(args.device))
    out_dir = OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}" if str(args.out_suffix).strip() else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("stage=load_frames", flush=True)
    val_raw, oos_raw = _load_frames()
    _assert_aligned("validation", val_raw, PREDICTION_DIR / "validation_predictions_baseline_entry_q070.csv")
    _assert_aligned("oos", oos_raw, PREDICTION_DIR / "oos_predictions_baseline_entry_q070.csv")
    print("stage=load_decisions", flush=True)
    val_base_dec, oos_base_dec = _load_decisions()
    print("stage=load_baseline_bundle", flush=True)
    bundle = torch.load(Path(args.baseline_bundle), map_location=device, weights_only=False)
    base_cols = list(bundle["base_cols"])
    loaded = parent._load_payloads(bundle["models"], device=device)
    x_val = parent._base_input(val_raw, base_cols)
    x_oos = parent._base_input(oos_raw, base_cols)
    fee, slip = omega._load_fee_slip()

    configs = _parse_configs(str(args.configs))
    results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for cfg in configs:
        name = f"atr{int(cfg['atr_window'])}_tp{cfg['tp_mult']:g}_sl{cfg['sl_mult']:g}".replace(".", "p")
        print(f"stage=evaluate {name}", flush=True)
        val_dec, val_diag = _apply_atr_safety_sltp(
            val_base_dec,
            val_raw,
            atr_window=int(cfg["atr_window"]),
            tp_mult=float(cfg["tp_mult"]),
            sl_mult=float(cfg["sl_mult"]),
            min_tp=float(args.min_tp),
            min_sl=float(args.min_sl),
            max_tp=float(args.max_tp),
            max_sl=float(args.max_sl),
        )
        oos_dec, oos_diag = _apply_atr_safety_sltp(
            oos_base_dec,
            oos_raw,
            atr_window=int(cfg["atr_window"]),
            tp_mult=float(cfg["tp_mult"]),
            sl_mult=float(cfg["sl_mult"]),
            min_tp=float(args.min_tp),
            min_sl=float(args.min_sl),
            max_tp=float(args.max_tp),
            max_sl=float(args.max_sl),
        )
        val_m = price_exit._metrics_shared_exit_price_move_sltp(
            val_raw,
            x_val,
            val_dec,
            loaded,
            threshold=float(args.exit_threshold),
            fee=fee,
            slip=slip,
            cost_mult=float(args.cost_mult),
            device=device,
        )
        oos_m = price_exit._metrics_shared_exit_price_move_sltp(
            oos_raw,
            x_oos,
            oos_dec,
            loaded,
            threshold=float(args.exit_threshold),
            fee=fee,
            slip=slip,
            cost_mult=float(args.cost_mult),
            device=device,
        )
        results[name] = {
            "config": cfg,
            "validation_atr_diag": val_diag,
            "oos_atr_diag": oos_diag,
            "validation": val_m,
            "oos": oos_m,
        }
        rows.append(
            {
                "variant": name,
                "atr_window": int(cfg["atr_window"]),
                "tp_mult": float(cfg["tp_mult"]),
                "sl_mult": float(cfg["sl_mult"]),
                "validation_pnl": float(val_m["pnl"]),
                "validation_mdd": float(val_m["mdd"]),
                "validation_trades": int(val_m["trades"]),
                "validation_wr": float(val_m["wr"]),
                "oos_pnl": float(oos_m["pnl"]),
                "oos_mdd": float(oos_m["mdd"]),
                "oos_trades": int(oos_m["trades"]),
                "oos_wr": float(oos_m["wr"]),
                "val_sl_p50": float(val_diag["sl_p50"]),
                "val_tp_p50": float(val_diag["tp_p50"]),
                "oos_sl_p50": float(oos_diag["sl_p50"]),
                "oos_tp_p50": float(oos_diag["tp_p50"]),
            }
        )

    rows_by_validation = sorted(rows, key=lambda r: (float(r["validation_pnl"]), float(r["validation_mdd"])), reverse=True)
    rows_by_oos = sorted(rows, key=lambda r: (float(r["oos_pnl"]), float(r["validation_pnl"])), reverse=True)
    pd.DataFrame(rows).to_csv(out_dir / "ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "baseline_bundle": str(args.baseline_bundle),
        "source_predictions": str(PREDICTION_DIR),
        "contract": {
            "entry": "Omega4.1 baseline direction/quality q=0.70",
            "exit": "existing Omega4.1 exit head threshold",
            "sltp": "entry-time ATR percent safety barriers; SLTP hit compares raw price_move only",
            "pnl": "realized price_move * notional",
            "quality_threshold": float(args.quality_threshold),
            "exit_threshold": float(args.exit_threshold),
            "min_tp": float(args.min_tp),
            "min_sl": float(args.min_sl),
            "max_tp": float(args.max_tp),
            "max_sl": float(args.max_sl),
        },
        "reference": {
            "existing_exit070_baseline": {
                "validation": {"pnl": 3.2756516763214893, "mdd": -7.817488798061978, "trades": 149, "wr": 0.6711409395973155},
                "oos": {"pnl": 7.513325496582635, "mdd": -5.61401353413885, "trades": 100, "wr": 0.63},
                "note": "previous account-threshold SLTP replay contract",
            },
            "fixed_price_move_sltp_exit070": {
                "validation": {"pnl": -2.981096790793014, "mdd": -8.62776452380153, "trades": 239, "wr": 0.5146443514644351},
                "oos": {"pnl": 2.001072984565999, "mdd": -9.410992452606415, "trades": 134, "wr": 0.5298507462686567},
            },
        },
        "results": results,
        "ranking_by_validation": rows_by_validation,
        "ranking_by_oos": rows_by_oos,
        "artifacts": {"out_dir": str(out_dir), "ranking": str(out_dir / "ranking.csv"), "report": str(out_dir / "report.json")},
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "top_validation": rows_by_validation[:5], "top_oos": rows_by_oos[:5]}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
