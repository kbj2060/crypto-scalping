#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import eval_omega1_2_1_exit_hazard_guard_20260610 as base


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_2_1_mid_barrier_20260616"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _apply_mid_barrier(dec: pd.DataFrame) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = np.flatnonzero(base.omega._active(out))
    if len(active) == 0:
        return out

    base_notional = pd.to_numeric(out.loc[active, "notional_exposure"], errors="raise").to_numpy(dtype=np.float64)
    margin_notional = np.minimum(base_notional * base.COMPENSATED_SCALE, base.MARGIN_CAP)
    effective_exposure = margin_notional * base.TRUE_LEVERAGE

    out.loc[active, "notional_exposure"] = effective_exposure
    out.loc[active, "position_fraction"] = margin_notional
    out.loc[active, "leverage"] = base.TRUE_LEVERAGE
    out.loc[active, "take_profit"] = base.BASE_TP * base.TRUE_LEVERAGE
    out.loc[active, "stop_loss"] = base.BASE_SL * base.TRUE_LEVERAGE
    out.loc[active, "max_hold_bars"] = 0
    out.loc[active, "cooldown_bars"] = 0
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base._apply_true_leverage_price_barrier = _apply_mid_barrier

    fee, slip = base.omega._load_fee_slip()
    splits = base._build_decisions()
    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}

    for split, (frame, dec) in splits.items():
        split_metrics, ledger = base._simulate_exit_guard(
            frame,
            dec,
            fee=fee,
            slip=slip,
            cost_mult=3.0,
            lock_trigger=999.0,
            lock_floor=-999.0,
            giveback_frac=0.0,
            emergency_loss=999.0,
        )
        metrics[split] = split_metrics
        ledger_path = OUT_DIR / f"{split}_mid_barrier_ledger.csv"
        ledger.to_csv(ledger_path, index=False)
        ledgers[split] = str(ledger_path)

    report = {
        "model_id": MODEL_ID,
        "baseline_model": "omega1_2_1_tp_runner_clean_repair_20260613",
        "variant": "mid_barrier_tp052_sl028",
        "risk_contract": {
            "base_tp": base.BASE_TP,
            "base_sl": base.BASE_SL,
            "take_profit": base.BASE_TP * base.TRUE_LEVERAGE,
            "stop_loss": base.BASE_SL * base.TRUE_LEVERAGE,
            "compensated_scale": base.COMPENSATED_SCALE,
            "margin_cap": base.MARGIN_CAP,
            "true_leverage": base.TRUE_LEVERAGE,
            "barrier_scale": base.TRUE_LEVERAGE,
            "removed_ratio_from_barrier": True,
        },
        "cost_accounting": {
            "fee": fee,
            "slip": slip,
            "cost_mult": 3.0,
            "notional_exposure": "effective account exposure",
            "position_fraction": "margin notional",
        },
        "metrics": metrics,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "report": str(OUT_DIR / "report.json"),
            "ledgers": ledgers,
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")

    print(json.dumps({
        "report": str(OUT_DIR / "report.json"),
        "variant": report["variant"],
        "risk_contract": report["risk_contract"],
        "metrics": metrics,
    }, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
