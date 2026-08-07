#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


MODEL_ID = "omega1_2_practical_walk_forward_20260604"
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


def _read_predictions(path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    pred = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    if not pred["timestamp"].equals(frame["timestamp"]):
        raise RuntimeError(f"timestamp contract mismatch: {path}")
    return pred


def _prefix(oof: bool) -> str:
    return "omega1_regime3_expertdq_oof_" if oof else "omega1_regime3_expertdq_"


def _config_grid() -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for tb in [0.70, 0.72, 0.74, 0.75]:
        for trb in [0.64, 0.66]:
            for tc in [0.64, 0.65, 0.70]:
                for sb in [0.55, 0.65, 0.75]:
                    for sc in [0.90, 1.05]:
                        for sl in [0.012, 0.014, 0.016]:
                            rows.append(
                                {
                                    "thr_bull": tb,
                                    "thr_bear": trb,
                                    "thr_chop": tc,
                                    "scale_bull": sb,
                                    "scale_bear": 0.90,
                                    "scale_chop": sc,
                                    "tp": 0.026,
                                    "sl": sl,
                                }
                            )
    rows.append(dict(PRACTICAL))
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def _decisions(pred: pd.DataFrame, *, oof: bool, cfg: dict[str, float]) -> pd.DataFrame:
    prefix = _prefix(oof)
    action = pd.to_numeric(pred[f"{prefix}dir_action"], errors="raise").to_numpy(dtype=np.int64)
    q = pd.to_numeric(pred[f"{prefix}quality_for_action"], errors="raise").to_numpy(dtype=np.float64)
    expert = pred[f"{prefix}router_expert"].astype(str).replace({"chop_expert": "chop"}).to_numpy()
    thresholds = {
        "bull": float(cfg["thr_bull"]),
        "bear": float(cfg["thr_bear"]),
        "chop": float(cfg["thr_chop"]),
    }
    scales = {
        "bull": float(cfg["scale_bull"]),
        "bear": float(cfg["scale_bear"]),
        "chop": float(cfg["scale_chop"]),
    }
    thr = np.array([thresholds[str(x)] for x in expert], dtype=np.float64)
    final = np.where(q >= thr, action, omega.ACTION_CASH).astype(np.int64)
    active = final != omega.ACTION_CASH
    side = np.where(final == omega.ACTION_LONG, 1, np.where(final == omega.ACTION_SHORT, -1, 0)).astype(np.int64)
    scale = np.array([scales[str(x)] for x in expert], dtype=np.float64)
    notional = np.where(active, 0.45 * scale, 0.0)
    return pd.DataFrame(
        {
            "action": final,
            "side": side,
            "notional_exposure": notional,
            "leverage": np.where(active, 2.0, 1.0),
            "position_fraction": notional,
            "take_profit": np.where(active, float(cfg["tp"]), 0.0),
            "stop_loss": np.where(active, float(cfg["sl"]), 0.0),
            "max_hold_bars": 0,
            "cooldown_bars": 0,
            "quality_score": q,
            "confidence": pd.to_numeric(pred[f"{prefix}dir_confidence"], errors="raise").to_numpy(dtype=np.float64),
            "router_expert": np.where(expert == "chop", "chop_expert", expert),
        }
    )


def _slice_month(frame: pd.DataFrame, pred: pd.DataFrame, month: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    period = pd.to_datetime(frame["timestamp"], errors="raise").dt.to_period("M").astype(str)
    mask = period.eq(month).to_numpy()
    return frame.loc[mask].reset_index(drop=True), pred.loc[mask].reset_index(drop=True)


def _metrics(frame: pd.DataFrame, pred: pd.DataFrame, *, oof: bool, cfg: dict[str, float], fee: float, slip: float, cost_mult: float) -> dict[str, Any]:
    if len(frame) == 0:
        return {"pnl": 0.0, "mdd": 0.0, "wr": 0.0, "trades": 0, "exit_reasons": {}}
    dec = _decisions(pred, oof=oof, cfg=cfg)
    return omega._metrics(frame, dec, fee=fee, slip=slip, cost_mult=cost_mult)


def _combined_metrics(items: list[tuple[pd.DataFrame, pd.DataFrame, bool]], *, cfg: dict[str, float], fee: float, slip: float, cost_mult: float) -> dict[str, Any]:
    frames = []
    preds = []
    oof_flags = []
    for frame, pred, oof in items:
        frames.append(frame)
        preds.append(pred)
        oof_flags.append(oof)
    if len(set(oof_flags)) != 1:
        # Metrics can concatenate only when the prediction prefix is common.
        vals = [_metrics(f, p, oof=o, cfg=cfg, fee=fee, slip=slip, cost_mult=cost_mult) for f, p, o in items]
        return {
            "pnl_sum": float(sum(v["pnl"] for v in vals)),
            "mdd_min": float(min(v["mdd"] for v in vals)),
            "trades": int(sum(v["trades"] for v in vals)),
            "wr_weighted": float(sum(v["wr"] * v["trades"] for v in vals) / max(sum(v["trades"] for v in vals), 1)),
        }
    return _metrics(pd.concat(frames, ignore_index=True), pd.concat(preds, ignore_index=True), oof=oof_flags[0], cfg=cfg, fee=fee, slip=slip, cost_mult=cost_mult)


def _score(metrics: dict[str, Any]) -> float:
    pnl = float(metrics.get("pnl", metrics.get("pnl_sum", 0.0)))
    mdd = float(metrics.get("mdd", metrics.get("mdd_min", 0.0)))
    trades = int(metrics.get("trades", 0))
    penalty = 20.0 if trades < 8 else 0.0
    return pnl + 2.0 * mdd - penalty


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
    monthly_items: dict[str, tuple[pd.DataFrame, pd.DataFrame, bool]] = {}
    for month in ["2025-10", "2025-11", "2025-12"]:
        monthly_items[month] = (*_slice_month(frames["val_raw"], val_pred, month), True)
    for month in ["2026-01", "2026-02"]:
        monthly_items[month] = (*_slice_month(frames["oos_raw"], oos_pred, month), False)

    grid = _config_grid()
    wf_months = ["2025-11", "2025-12", "2026-01", "2026-02"]
    history_for_month = {
        "2025-11": ["2025-10"],
        "2025-12": ["2025-10", "2025-11"],
        "2026-01": ["2025-10", "2025-11", "2025-12"],
        "2026-02": ["2025-10", "2025-11", "2025-12", "2026-01"],
    }
    rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    for month, item in monthly_items.items():
        vals = _metrics(item[0], item[1], oof=item[2], cfg=PRACTICAL, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        fixed_rows.append({"month": month, "mode": "fixed_practical", **PRACTICAL, **{f"test_{k}": v for k, v in vals.items() if k != "exit_reasons"}, "test_exit_reasons": vals.get("exit_reasons", {})})

    for month in wf_months:
        history = [monthly_items[m] for m in history_for_month[month]]
        best: tuple[float, int, dict[str, Any]] | None = None
        for idx, cfg_row in grid.iterrows():
            cfg = {k: float(cfg_row[k]) for k in grid.columns}
            hist = _combined_metrics(history, cfg=cfg, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
            sc = _score(hist)
            if best is None or sc > best[0]:
                best = (float(sc), int(idx), cfg)
        assert best is not None
        test_item = monthly_items[month]
        test = _metrics(test_item[0], test_item[1], oof=test_item[2], cfg=best[2], fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        hist = _combined_metrics(history, cfg=best[2], fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        rows.append(
            {
                "month": month,
                "history_months": ",".join(history_for_month[month]),
                "selected_grid_idx": int(best[1]),
                "selection_score": float(best[0]),
                **best[2],
                **{f"history_{k}": v for k, v in hist.items() if k != "exit_reasons"},
                **{f"test_{k}": v for k, v in test.items() if k != "exit_reasons"},
                "test_exit_reasons": test.get("exit_reasons", {}),
            }
        )

    pd.DataFrame(fixed_rows).to_csv(out_dir / "fixed_practical_monthly.csv", index=False)
    pd.DataFrame(rows).to_csv(out_dir / "walk_forward_monthly.csv", index=False)
    wf_summary = {
        "test_months": wf_months,
        "walk_forward_pnl_sum": float(sum(r["test_pnl"] for r in rows)),
        "walk_forward_mdd_min": float(min(r["test_mdd"] for r in rows)),
        "walk_forward_trades": int(sum(r["test_trades"] for r in rows)),
        "walk_forward_wr_weighted": float(sum(r["test_wr"] * r["test_trades"] for r in rows) / max(sum(r["test_trades"] for r in rows), 1)),
    }
    fixed_eval_rows = [r for r in fixed_rows if r["month"] in wf_months]
    fixed_summary = {
        "test_months": wf_months,
        "fixed_pnl_sum": float(sum(r["test_pnl"] for r in fixed_eval_rows)),
        "fixed_mdd_min": float(min(r["test_mdd"] for r in fixed_eval_rows)),
        "fixed_trades": int(sum(r["test_trades"] for r in fixed_eval_rows)),
        "fixed_wr_weighted": float(sum(r["test_wr"] * r["test_trades"] for r in fixed_eval_rows) / max(sum(r["test_trades"] for r in fixed_eval_rows), 1)),
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Fixed-model monthly walk-forward. Each test month selects threshold/scale/TP/SL only from prior completed months; no future month is used.",
        "grid_size": int(len(grid)),
        "fixed_practical": PRACTICAL,
        "walk_forward_summary": wf_summary,
        "fixed_summary_same_months": fixed_summary,
        "artifacts": {
            "out_dir": str(out_dir),
            "walk_forward_monthly": str(out_dir / "walk_forward_monthly.csv"),
            "fixed_practical_monthly": str(out_dir / "fixed_practical_monthly.csv"),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(fixed_rows).to_string(index=False))
    print(json.dumps({"report": str(out_dir / "report.json"), "walk_forward_summary": wf_summary, "fixed_summary_same_months": fixed_summary}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
