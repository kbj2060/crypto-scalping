#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_1_aggressive_compensated_parent72_loose_zigzag_20260620"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
PREDICTION_DIRS = {
    "loose_smoke_e4_train30k_exit12k": ROOT
    / "tmp/causal_regen_20260516/omega1_2_true_3head_tabm_parent72_loose_zigzag_entry_20260620_smoke_e4_train30k_exit12k",
    "loose_full_e28_exit30k": ROOT
    / "tmp/causal_regen_20260516/omega1_2_true_3head_tabm_parent72_loose_zigzag_entry_20260620_e28_fulltrain_exit30k",
}
AGGRESSIVE_VAL = {"pnl": 100.54272942091158, "mdd": -10.677652697162888, "wr": 0.6363636363636364, "trades": 33}
AGGRESSIVE_OOS = {"pnl": 72.76004148106665, "mdd": -8.108170708968387, "wr": 0.7222222222222222, "trades": 18}


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _load_predictions(pred_dir: Path, split: str) -> pd.DataFrame:
    name = "validation_predictions_2025_true3head.csv" if split == "validation" else "oos_predictions_2026_true3head.csv"
    path = pred_dir / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)


def _decisions_from_predictions(pred_dir: Path, split: str) -> pd.DataFrame:
    src = _load_predictions(pred_dir, split)
    return threehead._to_decisions(src, oof=(split == "validation"))


def _apply_compensated(dec: pd.DataFrame, active_idx: np.ndarray, *, scale: float, cap: float) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    if len(active_idx) == 0:
        return out
    base_notional = pd.to_numeric(out.loc[active_idx, "notional_exposure"], errors="raise").to_numpy(dtype=np.float64)
    new_notional = np.minimum(base_notional * float(scale), float(cap))
    ratio = new_notional / np.maximum(base_notional, 1.0e-12)
    out.loc[active_idx, "notional_exposure"] = new_notional
    out.loc[active_idx, "position_fraction"] = new_notional
    out.loc[active_idx, "take_profit"] = pd.to_numeric(out.loc[active_idx, "take_profit"], errors="raise").to_numpy(dtype=np.float64) * ratio
    out.loc[active_idx, "stop_loss"] = pd.to_numeric(out.loc[active_idx, "stop_loss"], errors="raise").to_numpy(dtype=np.float64) * ratio
    return out


def _eval_variant(pred_dir: Path, frames: dict[str, Any], fee: float, slip: float) -> dict[str, Any]:
    val_dec_base = _decisions_from_predictions(pred_dir, "validation")
    oos_dec_base = _decisions_from_predictions(pred_dir, "oos")
    val_active = np.flatnonzero(omega._active(val_dec_base))
    oos_active = np.flatnonzero(omega._active(oos_dec_base))
    val_dec = _apply_compensated(val_dec_base, val_active, scale=2.0, cap=0.90)
    oos_dec = _apply_compensated(oos_dec_base, oos_active, scale=2.0, cap=0.90)
    return {
        "base_no_compensation": {
            "validation": omega._metrics(frames["val_raw"], val_dec_base, fee=fee, slip=slip, cost_mult=3.0),
            "oos": omega._metrics(frames["oos_raw"], oos_dec_base, fee=fee, slip=slip, cost_mult=3.0),
        },
        "aggressive_compensated_scale200_cap090": {
            "validation": omega._metrics(frames["val_raw"], val_dec, fee=fee, slip=slip, cost_mult=3.0),
            "oos": omega._metrics(frames["oos_raw"], oos_dec, fee=fee, slip=slip, cost_mult=3.0),
        },
        "active_rows": {"validation": int(len(val_active)), "oos": int(len(oos_active))},
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    results = {}
    for name, pred_dir in PREDICTION_DIRS.items():
        if not pred_dir.exists():
            raise RuntimeError(f"missing prediction dir for {name}: {pred_dir}")
        results[name] = _eval_variant(pred_dir, frames, fee, slip)
    report = {
        "model_id": MODEL_ID,
        "source_model": "omega1_2_1_aggressive_compensated_scale200_cap090",
        "change": "Replace original parent predictions with parent retrained on parent72-loose zigzag labels, then apply the same aggressive compensated scale=2.0 cap=0.90 contract.",
        "risk_contract": {
            "base_notional": float(omega.BASE_TEMPLATE["notional"]),
            "base_take_profit": float(omega.BASE_TEMPLATE["take_profit"]),
            "base_stop_loss": float(omega.BASE_TEMPLATE["stop_loss"]),
            "base_leverage": float(omega.BASE_TEMPLATE["leverage"]),
            "compensated_scale": 2.0,
            "notional_cap": 0.90,
            "cost_multiplier": 3.0,
        },
        "baseline": {
            "omega1_2_1_aggressive_compensated_scale200_cap090": {
                "validation": AGGRESSIVE_VAL,
                "oos": AGGRESSIVE_OOS,
            }
        },
        "results": results,
        "artifacts": {"out_dir": str(OUT_DIR), "report": str(OUT_DIR / "report.json")},
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "results": results}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
