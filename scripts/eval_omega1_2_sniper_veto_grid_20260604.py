#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as tabm  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_sniper_veto_grid_20260604"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
BASE_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080"
PRACTICAL = {
    "thr_bull": 0.72,
    "thr_bear": 0.64,
    "thr_chop": 0.65,
    "scale_bull": 0.65,
    "scale_bear": 0.90,
    "scale_chop": 0.90,
    "tp": 0.026,
    "sl": 0.012,
}


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _prefix(oof: bool) -> str:
    return "omega1_regime3_expertdq_oof_" if oof else "omega1_regime3_expertdq_"


def _read_predictions(path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    pred = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    if not pred["timestamp"].equals(frame["timestamp"]):
        raise RuntimeError(f"timestamp contract mismatch: {path}")
    return pred


def _decisions(pred: pd.DataFrame, *, oof: bool, cfg: dict[str, float]) -> pd.DataFrame:
    prefix = _prefix(oof)
    action = pd.to_numeric(pred[f"{prefix}dir_action"], errors="raise").to_numpy(dtype=np.int64)
    quality = pd.to_numeric(pred[f"{prefix}quality_for_action"], errors="raise").to_numpy(dtype=np.float64)
    router_conf = pd.to_numeric(pred[f"{prefix}router_confidence"], errors="raise").to_numpy(dtype=np.float64)
    router_margin = pd.to_numeric(pred[f"{prefix}router_margin"], errors="raise").to_numpy(dtype=np.float64)
    dir_conf = pd.to_numeric(pred[f"{prefix}dir_confidence"], errors="raise").to_numpy(dtype=np.float64)
    side_edge = np.abs(pd.to_numeric(pred[f"{prefix}dir_side_edge"], errors="raise").to_numpy(dtype=np.float64))
    expert = pred[f"{prefix}router_expert"].astype(str).replace({"chop_expert": "chop"}).to_numpy()
    thresholds = {
        "bull": float(PRACTICAL["thr_bull"]) + float(cfg["quality_delta"]),
        "bear": float(PRACTICAL["thr_bear"]) + float(cfg["quality_delta"]),
        "chop": float(PRACTICAL["thr_chop"]) + float(cfg["quality_delta"]),
    }
    scales = {
        "bull": float(PRACTICAL["scale_bull"]),
        "bear": float(PRACTICAL["scale_bear"]),
        "chop": float(PRACTICAL["scale_chop"]),
    }
    thr = np.array([thresholds[str(x)] for x in expert], dtype=np.float64)
    scale = np.array([scales[str(x)] for x in expert], dtype=np.float64)
    passed = (
        (quality >= thr)
        & (router_conf >= float(cfg["min_router_conf"]))
        & (router_margin >= float(cfg["min_router_margin"]))
        & (dir_conf >= float(cfg["min_dir_conf"]))
        & (side_edge >= float(cfg["min_abs_side_edge"]))
    )
    final = np.where(passed, action, omega.ACTION_CASH).astype(np.int64)
    active = final != omega.ACTION_CASH
    side = np.where(final == omega.ACTION_LONG, 1, np.where(final == omega.ACTION_SHORT, -1, 0)).astype(np.int64)
    notional = np.where(active, 0.45 * scale, 0.0)
    return pd.DataFrame(
        {
            "action": final,
            "side": side,
            "notional_exposure": notional,
            "leverage": np.where(active, 2.0, 1.0),
            "position_fraction": notional,
            "take_profit": np.where(active, float(PRACTICAL["tp"]), 0.0),
            "stop_loss": np.where(active, float(PRACTICAL["sl"]), 0.0),
            "max_hold_bars": 0,
            "cooldown_bars": 0,
            "quality_score": quality,
            "confidence": dir_conf,
            "router_expert": np.where(expert == "chop", "chop_expert", expert),
        }
    )


def _metrics(frame: pd.DataFrame, pred: pd.DataFrame, *, oof: bool, cfg: dict[str, float], fee: float, slip: float, cost_mult: float) -> dict[str, Any]:
    dec = _decisions(pred, oof=oof, cfg=cfg)
    return omega._metrics(frame, dec, fee=fee, slip=slip, cost_mult=cost_mult)


def _monthly_metrics(frame: pd.DataFrame, pred: pd.DataFrame, *, oof: bool, cfg: dict[str, float], fee: float, slip: float, cost_mult: float) -> list[dict[str, Any]]:
    month = pd.to_datetime(frame["timestamp"], errors="raise").dt.to_period("M").astype(str)
    out: list[dict[str, Any]] = []
    for m in sorted(month.unique()):
        mask = month.eq(m).to_numpy()
        vals = _metrics(frame.loc[mask].reset_index(drop=True), pred.loc[mask].reset_index(drop=True), oof=oof, cfg=cfg, fee=fee, slip=slip, cost_mult=cost_mult)
        out.append({"month": m, **vals})
    return out


def _score(vals: dict[str, Any], monthly: list[dict[str, Any]]) -> float:
    pnl = float(vals["pnl"])
    mdd = float(vals["mdd"])
    wr = float(vals["wr"])
    trades = int(vals["trades"])
    min_month = min(float(x["pnl"]) for x in monthly) if monthly else 0.0
    months_pos = sum(float(x["pnl"]) > 0.0 for x in monthly)
    trade_penalty = 25.0 if trades < 12 else 0.0
    return pnl + 2.0 * mdd + 12.0 * wr + 0.35 * min_month + 2.0 * months_pos - trade_penalty


def _parse_reasons(text: str) -> dict[str, int]:
    try:
        val = ast.literal_eval(str(text))
        return val if isinstance(val, dict) else {}
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    out_dir = OUT_DIR if not args.out_suffix.strip() else OUT_DIR.parent / f"{MODEL_ID}_{args.out_suffix.strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = tabm._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_pred = _read_predictions(BASE_DIR / "validation_predictions_2025_true3head.csv", frames["val_raw"])
    oos_pred = _read_predictions(BASE_DIR / "oos_predictions_2026_true3head.csv", frames["oos_raw"])
    grid = []
    for quality_delta in [0.0, 0.02, 0.04, 0.06, 0.08]:
        for min_router_conf in [0.0, 0.55, 0.60, 0.65]:
            for min_router_margin in [0.0, 0.15, 0.25, 0.35]:
                for min_dir_conf in [0.0, 0.55, 0.60, 0.65]:
                    for min_abs_side_edge in [0.0, 0.15, 0.25, 0.35]:
                        grid.append(
                            {
                                "quality_delta": quality_delta,
                                "min_router_conf": min_router_conf,
                                "min_router_margin": min_router_margin,
                                "min_dir_conf": min_dir_conf,
                                "min_abs_side_edge": min_abs_side_edge,
                            }
                        )
    rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    for cfg in grid:
        val = _metrics(frames["val_raw"], val_pred, oof=True, cfg=cfg, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        val_monthly = _monthly_metrics(frames["val_raw"], val_pred, oof=True, cfg=cfg, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        sc = _score(val, val_monthly)
        oos = _metrics(frames["oos_raw"], oos_pred, oof=False, cfg=cfg, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        oos_monthly = _monthly_metrics(frames["oos_raw"], oos_pred, oof=False, cfg=cfg, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        row = {
            **cfg,
            "score": float(sc),
            "val_pnl": val["pnl"],
            "val_mdd": val["mdd"],
            "val_wr": val["wr"],
            "val_trades": val["trades"],
            "val_min_month_pnl": float(min(x["pnl"] for x in val_monthly)),
            "val_months_pos": int(sum(float(x["pnl"]) > 0.0 for x in val_monthly)),
            "oos_pnl": oos["pnl"],
            "oos_mdd": oos["mdd"],
            "oos_wr": oos["wr"],
            "oos_trades": oos["trades"],
            "oos_min_month_pnl": float(min(x["pnl"] for x in oos_monthly)),
            "oos_months_pos": int(sum(float(x["pnl"]) > 0.0 for x in oos_monthly)),
            "val_exit_reasons": val.get("exit_reasons", {}),
            "oos_exit_reasons": oos.get("exit_reasons", {}),
        }
        rows.append(row)
        # Only keep detailed monthly rows for compact high-level candidates.
        if row["val_trades"] >= 12 and row["val_pnl"] > 20.0:
            for x in val_monthly:
                monthly_rows.append({"split": "val", **cfg, "month": x["month"], "pnl": x["pnl"], "mdd": x["mdd"], "wr": x["wr"], "trades": x["trades"], "exit_reasons": x.get("exit_reasons", {})})
            for x in oos_monthly:
                monthly_rows.append({"split": "oos", **cfg, "month": x["month"], "pnl": x["pnl"], "mdd": x["mdd"], "wr": x["wr"], "trades": x["trades"], "exit_reasons": x.get("exit_reasons", {})})

    ranking = pd.DataFrame(rows).sort_values(["score", "val_pnl"], ascending=False)
    ranking.to_csv(out_dir / "sniper_veto_grid.csv", index=False)
    pd.DataFrame(monthly_rows).to_csv(out_dir / "sniper_veto_monthly_candidates.csv", index=False)
    top = ranking.head(20).copy()
    report = {
        "model_id": MODEL_ID,
        "design": "Sniper development for Omega1.2 practical baseline. Model and risk template are fixed; only additional validation-selected veto filters are tested.",
        "base_dir": str(BASE_DIR),
        "practical": PRACTICAL,
        "grid_size": int(len(grid)),
        "top_by_validation_score": json.loads(top.to_json(orient="records")),
        "baseline_reference": {
            "val_pnl": 55.55076403452504,
            "val_mdd": -5.024365041644074,
            "val_wr": 0.6097560975609756,
            "val_trades": 41,
            "oos_pnl": 26.770616593092477,
            "oos_mdd": -5.16898834513716,
            "oos_wr": 0.631578947368421,
            "oos_trades": 19,
        },
        "artifacts": {
            "out_dir": str(out_dir),
            "ranking": str(out_dir / "sniper_veto_grid.csv"),
            "monthly_candidates": str(out_dir / "sniper_veto_monthly_candidates.csv"),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(ranking.head(25).to_string(index=False))
    print(json.dumps({"report": str(out_dir / "report.json"), "ranking": str(out_dir / "sniper_veto_grid.csv")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
