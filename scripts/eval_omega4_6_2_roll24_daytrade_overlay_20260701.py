#!/usr/bin/env python3
"""24h roll overlay for the Omega 4.6.2 v4 loss-cluster candidate."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V1_MODULE_PATH = ROOT / "scripts/eval_omega4_6_2_loss_cluster_governor_20260701.py"
MODEL_ID = "omega4_6_2_roll24_daytrade_overlay_20260701"
REFERENCE_MODEL_ID = "omega4_6_2_loss_cluster_governor_v4_fine_exposure_20260701"
REFERENCE_REPORT = ROOT / "tmp/causal_regen_20260516" / REFERENCE_MODEL_ID / "report.json"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_roll24_daytrade_overlay_20260701.md"
MAX_ROLL_HOURS = 24.0
EPS = 1.0e-12


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def segment_mae_mfe(window: pd.DataFrame, side: int, entry_price: float) -> tuple[float, float]:
    if side > 0:
        mfe = float(window["high"].astype(float).max() / entry_price - 1.0)
        mae = float(window["low"].astype(float).min() / entry_price - 1.0)
    else:
        mfe = float(entry_price / window["low"].astype(float).min() - 1.0)
        mae = float(entry_price / window["high"].astype(float).max() - 1.0)
    return mfe, mae


def split_roll24(df: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    market = market.copy()
    market["timestamp"] = pd.to_datetime(market["timestamp"], errors="raise")
    close_by_ts = dict(zip(market["timestamp"], market["close"].astype(float)))
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        if float(row["notional"]) <= EPS:
            rows.append(row.to_dict())
            continue
        side = int(row["side"])
        original_entry = pd.Timestamp(row["entry_timestamp"])
        original_exit = pd.Timestamp(row["exit_timestamp"])
        cur_entry = original_entry
        seg = 0
        roundtrip_cost = float(row["raw_exit_price_move"]) - float(row["net_per_notional"])
        while cur_entry < original_exit:
            target_exit = min(cur_entry + pd.Timedelta(hours=MAX_ROLL_HOURS), original_exit)
            window = market[
                (market["timestamp"] >= cur_entry) & (market["timestamp"] <= target_exit)
            ].copy()
            if window.empty:
                break
            if cur_entry not in close_by_ts:
                cur_entry = pd.Timestamp(window["timestamp"].iloc[0])
            exit_ts = pd.Timestamp(window["timestamp"].iloc[-1])
            entry_price = float(close_by_ts[cur_entry])
            exit_price = float(close_by_ts[exit_ts])
            raw_move = float(side) * (exit_price / entry_price - 1.0)
            net_per_notional = raw_move - roundtrip_cost
            mfe, mae = segment_mae_mfe(window, side, entry_price)
            out = row.to_dict()
            hold_bars = int(round((exit_ts - cur_entry).total_seconds() / 300.0))
            synthetic_entry_i = int(row["entry_i"]) + seg * 100_000
            out.update(
                {
                    "entry_i": synthetic_entry_i,
                    "exit_i": synthetic_entry_i + hold_bars,
                    "entry_timestamp": cur_entry.strftime("%Y-%m-%d %H:%M:%S"),
                    "exit_timestamp": exit_ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "entry_timestamp_dt": cur_entry,
                    "exit_timestamp_dt": exit_ts,
                    "reason": "roll24_final" if exit_ts >= original_exit else "roll24_time_exit",
                    "raw_exit_price_move": raw_move,
                    "mfe_price_move": mfe,
                    "mae_price_move": mae,
                    "net_per_notional": net_per_notional,
                    "trade_return": net_per_notional * float(row["notional"]),
                    "win": int(net_per_notional > 0.0),
                    "hold_hours": float((exit_ts - cur_entry).total_seconds() / 3600.0),
                    "roll24_parent_entry_timestamp": original_entry.strftime("%Y-%m-%d %H:%M:%S"),
                    "roll24_parent_exit_timestamp": original_exit.strftime("%Y-%m-%d %H:%M:%S"),
                    "roll24_segment": seg,
                    "roll24_roundtrip_cost": roundtrip_cost,
                }
            )
            rows.append(out)
            if exit_ts >= original_exit:
                break
            cur_entry = exit_ts + pd.Timedelta(minutes=5)
            seg += 1
    return pd.DataFrame(rows)


def write_markdown(report: dict[str, Any]) -> None:
    selected = report["selected_variant"]
    reference = report["reference_variant"]
    text = f"""# Omega 4.6.2 Roll24 Daytrade Overlay - 2026-07-01

## Method

This overlay freezes the v4 candidate and splits every active position into 24h-or-less roll segments. Each roll pays the same estimated roundtrip cost as the parent trade segment.

## Result

- Status: `{report["status"]}`
- Reference model: `{report["reference_model_id"]}`

| Metric | Reference Val | Roll24 Val | Reference OOS | Roll24 OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `{reference["validation_pnl"]:.4f}` | `{selected["validation_pnl"]:.4f}` | `{reference["oos_pnl"]:.4f}` | `{selected["oos_pnl"]:.4f}` |
| MDD % | `{reference["validation_mdd"]:.4f}` | `{selected["validation_mdd"]:.4f}` | `{reference["oos_mdd"]:.4f}` | `{selected["oos_mdd"]:.4f}` |
| Trades | `{reference["validation_trades"]}` | `{selected["validation_trades"]}` | `{reference["oos_trades"]}` | `{selected["oos_trades"]}` |
| Avg hold h | `{reference["validation_avg_hold_hours"]:.4f}` | `{selected["validation_avg_hold_hours"]:.4f}` | `{reference["oos_avg_hold_hours"]:.4f}` | `{selected["oos_avg_hold_hours"]:.4f}` |
| Max hold h | `{reference["validation_max_hold_hours"]:.4f}` | `{selected["validation_max_hold_hours"]:.4f}` | `{reference["oos_max_hold_hours"]:.4f}` | `{selected["oos_max_hold_hours"]:.4f}` |

## Artifacts

- Report: `{report["artifacts"]["report"]}`
- Validation ledger: `{report["artifacts"]["selected_validation_ledger"]}`
- OOS ledger: `{report["artifacts"]["selected_oos_ledger"]}`
"""
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD.write_text(text, encoding="utf-8")


def main() -> int:
    v1 = load_module("omega462_loss_cluster_v1_for_roll24", V1_MODULE_PATH)
    stop_mod = v1.load_module("omega462_stop_mod_for_roll24", v1.STOP_MODULE_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runtime = stop_mod.read_json(stop_mod.resolve_path(v1.DEFAULT_RUNTIME))
    train_csv, eval_csv = stop_mod.component_train_eval_paths(runtime)
    train_market = stop_mod.load_market(train_csv)
    eval_market = stop_mod.load_market(eval_csv)
    reference_report = read_json(REFERENCE_REPORT)
    reference = reference_report["selected_variant"]
    artifacts = reference_report["artifacts"]
    val = pd.read_csv(artifacts["selected_validation_ledger"])
    oos = pd.read_csv(artifacts["selected_oos_ledger"])
    roll_val = split_roll24(val, train_market)
    roll_oos = split_roll24(oos, eval_market)
    val_metrics = stop_mod.metrics(roll_val)
    oos_metrics = stop_mod.metrics(roll_oos)

    selected = {
        "overlay": "roll24_daytrade",
        "max_roll_hours": MAX_ROLL_HOURS,
        **{f"validation_{k}": v for k, v in val_metrics.items() if not isinstance(v, (dict, list))},
        "validation_reason_counts": json.dumps(val_metrics["reason_counts"], ensure_ascii=False, sort_keys=True),
        **{f"oos_{k}": v for k, v in oos_metrics.items() if not isinstance(v, (dict, list))},
        "oos_reason_counts": json.dumps(oos_metrics["reason_counts"], ensure_ascii=False, sort_keys=True),
    }
    daytrade_pass = bool(
        val_metrics["pnl"] >= 100.0
        and oos_metrics["pnl"] >= 100.0
        and val_metrics["mdd"] >= -20.0
        and oos_metrics["mdd"] >= -20.0
        and val_metrics["max_hold_hours"] <= 24.0
        and oos_metrics["max_hold_hours"] <= 24.0
    )
    pnl_upgrade_vs_reference = bool(
        val_metrics["pnl"] > reference["validation_pnl"] and oos_metrics["pnl"] > reference["oos_pnl"]
    )
    status = (
        "DAYTRADE_HOLD_PASS_PNL_LOWER_THAN_REFERENCE"
        if daytrade_pass and not pnl_upgrade_vs_reference
        else "DAYTRADE_HOLD_AND_PNL_PASS"
        if daytrade_pass
        else "DAYTRADE_HOLD_FAIL"
    )
    val_out = OUT_DIR / "validation_roll24_daytrade_ledger.csv"
    oos_out = OUT_DIR / "oos_roll24_daytrade_ledger.csv"
    report_path = OUT_DIR / "report.json"
    roll_val.to_csv(val_out, index=False)
    roll_oos.to_csv(oos_out, index=False)
    report = {
        "model_id": MODEL_ID,
        "reference_model_id": REFERENCE_MODEL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_scope": "fixed_v4_overlay; OOS readout only",
        "reference_variant": reference,
        "selected_variant": selected,
        "status": status,
        "daytrade_pass": daytrade_pass,
        "pnl_upgrade_vs_reference": pnl_upgrade_vs_reference,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "selected_validation_ledger": str(val_out),
            "selected_oos_ledger": str(oos_out),
            "report": str(report_path),
            "audit_md": str(AUDIT_MD),
        },
    }
    write_json(report_path, report)
    write_markdown(report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "status": status,
                "daytrade_pass": daytrade_pass,
                "pnl_upgrade_vs_reference": pnl_upgrade_vs_reference,
                "validation": val_metrics,
                "oos": oos_metrics,
            },
            ensure_ascii=False,
            indent=2,
            default=json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
