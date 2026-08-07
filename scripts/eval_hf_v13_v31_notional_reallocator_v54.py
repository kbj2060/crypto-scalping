#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, predict_policy_frame  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "hf_v13_v31_notional_reallocator_v54_20260513"
DEFAULT_PARENT = v31.DEFAULT_PARENT
DEFAULT_JACKPOT = v31.DEFAULT_JACKPOT
DEFAULT_V27 = v31.DEFAULT_V27
DEFAULT_TRAIN = v31.DEFAULT_TRAIN
DEFAULT_EVAL = v31.DEFAULT_EVAL
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_v31_notional_reallocator_v54_20260513"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_v31_notional_reallocator_v54_20260513_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_v31_notional_reallocator_v54_20260513_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_v31_notional_reallocator_v54_20260513_grid.csv"

V31_BASELINE = {
    "cost1": {"pnl": 277.0679629973942, "mdd": -31.74},
    "cost2": {"pnl": 112.79326141840412, "mdd": -31.46},
    "cost3": {"pnl": 20.933695032758784, "mdd": -43.09},
}


@dataclass(frozen=True)
class ReallocatorConfig:
    name: str
    parent_mult: float
    parent_cap: float
    deep_mult: float
    addon_frac: float
    addon_total_mult: float
    addon_cap: float
    tpsl_power: float
    quality_floor: float
    confidence_floor: float
    vol_throttle: float


def _configs() -> list[ReallocatorConfig]:
    rows: list[ReallocatorConfig] = []
    for parent_mult in (1.15, 1.30, 1.50, 1.75, 2.00):
        for tpsl_power in (0.50, 0.75, 1.00):
            rows.append(
                ReallocatorConfig(
                    f"v54_pm{parent_mult:.2f}_tp{tpsl_power:.2f}_balanced",
                    parent_mult,
                    4.14,
                    1.0,
                    0.20,
                    1.35,
                    4.14,
                    tpsl_power,
                    -99.0,
                    0.0,
                    0.0,
                )
            )
    for parent_mult in (1.50, 1.75, 2.00):
        rows.append(
            ReallocatorConfig(
                f"v54_pm{parent_mult:.2f}_addon35_aggressive",
                parent_mult,
                4.50,
                1.15,
                0.35,
                1.60,
                4.50,
                1.00,
                -99.0,
                0.0,
                0.0,
            )
        )
    for parent_mult in (1.50, 1.75, 2.00):
        rows.append(
            ReallocatorConfig(
                f"v54_pm{parent_mult:.2f}_volguard",
                parent_mult,
                4.14,
                1.0,
                0.20,
                1.35,
                4.14,
                1.00,
                -99.0,
                0.0,
                0.018,
            )
        )
    rows.append(ReallocatorConfig("v54_baseline_identity", 1.0, 2.75, 1.0, 0.20, 1.35, 2.75, 1.0, -99.0, 0.0, 0.0))
    return rows


def _safe_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        x = float(row.get(col, default))
    except Exception:
        return float(default)
    return float(x) if np.isfinite(x) else float(default)


def _vol_anchor(row: pd.Series) -> float:
    bbw = abs(_safe_float(row, "bb_width", 0.0))
    gk = abs(_safe_float(row, "garman_klass_vol", 0.0))
    rs = abs(_safe_float(row, "rogers_satchell_vol", 0.0))
    pk = abs(_safe_float(row, "parkinson_vol", 0.0))
    volz = abs(_safe_float(row, "volatility_z", 0.0))
    rv = abs(_safe_float(row, "realized_vol_ratio", 1.0))
    base = max(0.0015, bbw * 0.15, gk * 2.5, rs * 2.5, pk * 2.5)
    return float(np.clip(base * (1.0 + 0.08 * min(volz, 3.0) + 0.05 * max(rv - 1.0, 0.0)), 0.0015, 0.030))


def _scale_decisions(df: pd.DataFrame, decisions: pd.DataFrame, cfg: ReallocatorConfig) -> pd.DataFrame:
    out = decisions.copy()
    trade = (out["action"].to_numpy(dtype=np.int64) != ACTION_CASH) & (out["side"].to_numpy(dtype=np.int64) != 0)
    if cfg.quality_floor > -90:
        trade &= out["quality_score"].to_numpy(dtype=np.float64) >= float(cfg.quality_floor)
    if cfg.confidence_floor > 0:
        trade &= out["confidence"].to_numpy(dtype=np.float64) >= float(cfg.confidence_floor)
    mult = np.ones(len(out), dtype=np.float64)
    mult[trade] = float(cfg.parent_mult)
    if cfg.vol_throttle > 0:
        vol = np.asarray([_vol_anchor(df.iloc[i]) for i in range(len(df))], dtype=np.float64)
        hot = vol > float(cfg.vol_throttle)
        mult[hot & trade] = 1.0 + (mult[hot & trade] - 1.0) * 0.45
    old_notional = out["notional_exposure"].to_numpy(dtype=np.float64)
    new_notional = np.minimum(old_notional * mult, float(cfg.parent_cap))
    scale = np.divide(new_notional, np.maximum(old_notional, 1e-12), out=np.ones_like(new_notional), where=old_notional > 0)
    out["notional_exposure"] = new_notional
    out["leverage"] = np.minimum(np.maximum(out["leverage"].to_numpy(dtype=np.float64), new_notional), 5.0)
    out["position_fraction"] = np.divide(new_notional, np.maximum(out["leverage"].to_numpy(dtype=np.float64), 1e-12))
    # Preserve approximate price-level TP/SL when notional is amplified.
    tpsl_scale = np.power(scale, float(cfg.tpsl_power))
    out["take_profit"] = out["take_profit"].to_numpy(dtype=np.float64) * tpsl_scale
    out["stop_loss"] = out["stop_loss"].to_numpy(dtype=np.float64) * tpsl_scale
    cash = ~trade
    out.loc[cash, ["side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[cash, "leverage"] = 1.0
    return out


def _scale_overlay(overlay: v31.OverlayConfig, cfg: ReallocatorConfig) -> v31.OverlayConfig:
    if cfg.deep_mult == 1.0:
        return overlay
    return replace(
        overlay,
        name=f"{overlay.name}_deepx{cfg.deep_mult:.2f}",
        notional=float(overlay.notional) * float(cfg.deep_mult),
        base_tp=float(overlay.base_tp) * float(cfg.deep_mult) ** float(cfg.tpsl_power),
        base_sl=float(overlay.base_sl) * float(cfg.deep_mult) ** float(cfg.tpsl_power),
        tp_cap=float(overlay.tp_cap) * float(cfg.deep_mult) ** float(cfg.tpsl_power),
        sl_cap=float(overlay.sl_cap) * float(cfg.deep_mult) ** float(cfg.tpsl_power),
    )


def _scale_add_cfg(add_cfg: CostRunnerConfig, cfg: ReallocatorConfig) -> CostRunnerConfig:
    return replace(
        add_cfg,
        name=f"{add_cfg.name}_{cfg.name}",
        full_add_frac=float(cfg.addon_frac),
        max_total_mult=float(cfg.addon_total_mult),
        max_entry_notional=float(cfg.addon_cap),
    )


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1.get("trades", 0)) < 20:
        return -1e9 + float(c1.get("pnl", 0.0))
    mdd_penalty = 0.20 * max(abs(float(c1["mdd"])) - 45.0, 0.0)
    return float(c1["pnl"] + 0.30 * c2["pnl"] + 0.10 * c3["pnl"] - 0.20 * abs(c1["mdd"]) - mdd_penalty)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V54 V31 conviction/notional reallocator.")
    p.add_argument("--parent-model", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--jackpot-model", type=Path, default=DEFAULT_JACKPOT)
    p.add_argument("--v27-model", type=Path, default=DEFAULT_V27)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[{MODEL_ID}] loading models/data", flush=True)
    parent = joblib.load(args.parent_model)
    jackpot_payload = joblib.load(args.jackpot_model)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg0 = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(args.v27_model)
    base = dict(parent["config"])
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    feature_audit = _audit_contract(train_all, eval_df, list(parent.get("feature_cols") or []))
    val_q = v31._predict_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    val_dec0 = predict_policy_frame(parent, val, close=_close(val))
    eval_dec0 = predict_policy_frame(parent, eval_df, close=_close(eval_df))
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    overlays = [o for o in v31._grid() if o.name in {"v31_notional1_time_decay", "v31_tight_after_24", "v31_ref", "v31_precision"}]
    for cfg in _configs():
        val_dec = _scale_decisions(val, val_dec0, cfg)
        add_cfg = _scale_add_cfg(add_cfg0, cfg)
        for overlay0 in overlays:
            overlay = _scale_overlay(overlay0, cfg)
            print(f"[{MODEL_ID}] validation cfg={cfg.name} overlay={overlay.name}", flush=True)
            v1 = v31.backtest(val, parent, jackpot_model, add_cfg, val_q, overlay, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=1.0, decisions=val_dec)
            v2 = v31.backtest(val, parent, jackpot_model, add_cfg, val_q, overlay, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=2.0, decisions=val_dec)
            v3 = v31.backtest(val, parent, jackpot_model, add_cfg, val_q, overlay, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=3.0, decisions=val_dec)
            row = {"config": asdict(cfg), "overlay": asdict(overlay), "validation_cost1": v1, "validation_cost2": v2, "validation_cost3": v3, "selection_score": _score(v1, v2, v3)}
            rows.append(row)
            if best is None or row["selection_score"] > best["selection_score"]:
                best = row
    assert best is not None
    selected_cfg = ReallocatorConfig(**best["config"])
    selected_overlay = v31.OverlayConfig(**best["overlay"])
    eval_dec = _scale_decisions(eval_df, eval_dec0, selected_cfg)
    selected_add = _scale_add_cfg(add_cfg0, selected_cfg)
    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for mult in (1, 2, 3):
        r = v31.backtest(eval_df, parent, jackpot_model, selected_add, eval_q, selected_overlay, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=float(mult), decisions=eval_dec, record=(mult == 1))
        if mult == 1:
            ledger = pd.DataFrame(r.pop("trade_records", []))
            lp = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
            lp.parent.mkdir(parents=True, exist_ok=True)
            ledger.to_csv(lp, index=False)
            ledgers["cost1"] = str(lp)
        metrics[f"cost{mult}"] = r
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "v54_notional_reallocator_manifest.json"
    manifest_path.write_text(json.dumps({"model_id": MODEL_ID, "selected_config": asdict(selected_cfg), "selected_overlay": asdict(selected_overlay), "parent_model": str(args.parent_model), "jackpot_model": str(args.jackpot_model), "v27_model": str(args.v27_model)}, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    pd.DataFrame(
        [
            {
                **{f"cfg_{k}": v for k, v in r["config"].items()},
                "overlay_name": r["overlay"]["name"],
                "selection_score": r["selection_score"],
                "val_pnl": r["validation_cost1"]["pnl"],
                "val_mdd": r["validation_cost1"]["mdd"],
                "val_trades": r["validation_cost1"]["trades"],
                "val_c2_pnl": r["validation_cost2"]["pnl"],
                "val_c3_pnl": r["validation_cost3"]["pnl"],
            }
            for r in rows
        ]
    ).to_csv(args.grid_out, index=False)
    blocking = list(feature_audit.get("blocking", []))
    warnings = list(feature_audit.get("warnings", []))
    if metrics["cost1"]["pnl"] <= V31_BASELINE["cost1"]["pnl"]:
        warnings.append("oos_cost1_did_not_beat_v31")
    if metrics["cost1"]["pnl"] <= 500.0:
        warnings.append("target_pnl_500_not_reached")
    if metrics["cost2"]["pnl"] <= 0:
        warnings.append("cost2_not_survived")
    if metrics["cost3"]["pnl"] <= -50:
        warnings.append("cost3_catastrophic")
    verdict = "promote" if not blocking and metrics["cost1"]["pnl"] > 500.0 and metrics["cost2"]["pnl"] > 0 else "iterate"
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": verdict,
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "policy": "v31_notional_reallocator_v54",
        "entry_owner_frozen": True,
        "direction_changed": False,
        "feature_audit": feature_audit,
        "selected_config": asdict(selected_cfg),
        "selected_overlay": asdict(selected_overlay),
        "metrics": metrics,
        "baseline_v31": V31_BASELINE,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "V54 keeps V31 parent/V27 directions frozen and reallocates notional/TP/SL/add-on capacity on 2025 Q4 selection. TP/SL are scaled with notional to avoid accidentally tightening price-level exits.",
        "selected_config": asdict(selected_cfg),
        "selected_overlay": asdict(selected_overlay),
        "selection_result": best,
        "metrics": metrics,
        "audit": audit,
        "artifacts": {"manifest": str(manifest_path), "report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "ledgers": ledgers},
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "manifest": str(manifest_path), "selected_config": asdict(selected_cfg), "selected_overlay": asdict(selected_overlay), "metrics": metrics, "verdict": verdict}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
