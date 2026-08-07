#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_1_exposure_selector_20260606 as exposure  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_1_aggressive_primary_edge_risk_grid_20260607"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _metric_row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(metrics["pnl"]),
        f"{prefix}_mdd": float(metrics["mdd"]),
        f"{prefix}_wr": float(metrics["wr"]),
        f"{prefix}_trades": int(metrics["trades"]),
        f"{prefix}_long": int(metrics["long_entries"]),
        f"{prefix}_short": int(metrics["short_entries"]),
        f"{prefix}_avg_notional": float(metrics["avg_notional"]),
        f"{prefix}_avg_leverage": float(metrics["avg_leverage"]),
        f"{prefix}_reasons": metrics["exit_reasons"],
    }


def _build_split(frames: dict[str, pd.DataFrame], split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame, src, dec, prefix = exposure._build_split(frames, split)
    dec = sleeve._apply_aggressive(dec)
    feat = exposure._feature_frame(frame, src, dec, prefix)
    side = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.float64)
    long_edge = pd.to_numeric(feat["dir_p_long"], errors="raise").to_numpy(dtype=np.float64) - pd.to_numeric(feat["dir_p_short"], errors="raise").to_numpy(dtype=np.float64)
    feat["side_aligned_dir_edge"] = np.where(side > 0, long_edge, np.where(side < 0, -long_edge, 0.0))
    feat["side_quality_prob"] = np.where(
        side > 0,
        pd.to_numeric(feat["quality_p_long"], errors="raise").to_numpy(dtype=np.float64),
        np.where(side < 0, pd.to_numeric(feat["quality_p_short"], errors="raise").to_numpy(dtype=np.float64), pd.to_numeric(feat["quality_p_cash"], errors="raise").to_numpy(dtype=np.float64)),
    )
    bad = [c for c in feat.columns if c.startswith("clean_regime4_") or c.startswith("regime4_pred_") or c.startswith("teacher_") or c == "tp_sl_action_score"]
    if bad:
        raise RuntimeError(f"{split}: forbidden feature columns: {bad[:40]}")
    return frame, dec, feat


def _scale_rows(dec: pd.DataFrame, idx: np.ndarray, scale: float, *, cap: float) -> None:
    if len(idx) == 0:
        return
    base = pd.to_numeric(dec.loc[idx, "notional_exposure"], errors="raise").to_numpy(dtype=np.float64)
    new = np.minimum(base * float(scale), float(cap))
    ratio = new / np.maximum(base, 1.0e-12)
    dec.loc[idx, "notional_exposure"] = new
    dec.loc[idx, "position_fraction"] = new
    dec.loc[idx, "take_profit"] = pd.to_numeric(dec.loc[idx, "take_profit"], errors="raise").to_numpy(dtype=np.float64) * ratio
    dec.loc[idx, "stop_loss"] = pd.to_numeric(dec.loc[idx, "stop_loss"], errors="raise").to_numpy(dtype=np.float64) * ratio


def _apply_policy(
    dec: pd.DataFrame,
    feat: pd.DataFrame,
    *,
    mode: str,
    low_edge: float,
    shrink: float,
    high_edge: float,
    boost: float,
    cap: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = dec.copy().reset_index(drop=True)
    active = omega._active(out)
    edge = pd.to_numeric(feat["side_aligned_dir_edge"], errors="raise").to_numpy(dtype=np.float64)
    low_idx = np.flatnonzero(active & (edge < float(low_edge)))
    high_idx = np.flatnonzero(active & (edge >= float(high_edge)))
    if mode == "veto_low_edge":
        out.loc[low_idx, "action"] = 0
        out.loc[low_idx, "side"] = 0
        out.loc[low_idx, "position_fraction"] = 0.0
        out.loc[low_idx, "notional_exposure"] = 0.0
    elif mode == "shrink_low_edge":
        _scale_rows(out, low_idx, shrink, cap=cap)
    elif mode == "shrink_low_edge_boost_high_edge":
        _scale_rows(out, low_idx, shrink, cap=cap)
        _scale_rows(out, high_idx, boost, cap=cap)
    else:
        raise RuntimeError(f"unknown mode: {mode}")
    return out, {"low_rows": int(len(low_idx)), "high_rows": int(len(high_idx))}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_frame, val_dec, val_feat = _build_split(frames, "validation")
    oos_frame, oos_dec, oos_feat = _build_split(frames, "oos")

    baseline_val = omega._metrics(val_frame, val_dec, fee=fee, slip=slip, cost_mult=3.0)
    baseline_oos = omega._metrics(oos_frame, oos_dec, fee=fee, slip=slip, cost_mult=3.0)
    rows: list[dict[str, Any]] = [
        {
            "candidate": "baseline_aggressive",
            "mode": "baseline",
            "low_edge": None,
            "shrink": None,
            "high_edge": None,
            "boost": None,
            "cap": None,
            "val_low_rows": 0,
            "val_high_rows": 0,
            "oos_low_rows": 0,
            "oos_high_rows": 0,
            **_metric_row("val", baseline_val),
            **_metric_row("oos", baseline_oos),
        }
    ]

    for low_edge in (0.35, 0.40, 0.45, 0.50, 0.55):
        for mode in ("veto_low_edge", "shrink_low_edge"):
            for shrink in ((0.0,) if mode == "veto_low_edge" else (0.55, 0.70, 0.85)):
                val_sel, val_counts = _apply_policy(val_dec, val_feat, mode=mode, low_edge=low_edge, shrink=shrink, high_edge=9.0, boost=1.0, cap=0.90)
                oos_sel, oos_counts = _apply_policy(oos_dec, oos_feat, mode=mode, low_edge=low_edge, shrink=shrink, high_edge=9.0, boost=1.0, cap=0.90)
                rows.append(
                    {
                        "candidate": f"{mode}_edge{low_edge:.2f}_shrink{shrink:.2f}",
                        "mode": mode,
                        "low_edge": float(low_edge),
                        "shrink": float(shrink),
                        "high_edge": None,
                        "boost": None,
                        "cap": 0.90,
                        "val_low_rows": val_counts["low_rows"],
                        "val_high_rows": val_counts["high_rows"],
                        "oos_low_rows": oos_counts["low_rows"],
                        "oos_high_rows": oos_counts["high_rows"],
                        **_metric_row("val", omega._metrics(val_frame, val_sel, fee=fee, slip=slip, cost_mult=3.0)),
                        **_metric_row("oos", omega._metrics(oos_frame, oos_sel, fee=fee, slip=slip, cost_mult=3.0)),
                    }
                )

    for low_edge in (0.35, 0.40, 0.45, 0.50):
        for high_edge in (0.55, 0.60, 0.65):
            if high_edge <= low_edge:
                continue
            for shrink in (0.70, 0.85):
                for boost in (1.05, 1.10, 1.1111111111):
                    val_sel, val_counts = _apply_policy(val_dec, val_feat, mode="shrink_low_edge_boost_high_edge", low_edge=low_edge, shrink=shrink, high_edge=high_edge, boost=boost, cap=0.90)
                    oos_sel, oos_counts = _apply_policy(oos_dec, oos_feat, mode="shrink_low_edge_boost_high_edge", low_edge=low_edge, shrink=shrink, high_edge=high_edge, boost=boost, cap=0.90)
                    rows.append(
                        {
                            "candidate": f"shrink_edge{low_edge:.2f}_{shrink:.2f}_boost_edge{high_edge:.2f}_{boost:.3f}",
                            "mode": "shrink_low_edge_boost_high_edge",
                            "low_edge": float(low_edge),
                            "shrink": float(shrink),
                            "high_edge": float(high_edge),
                            "boost": float(boost),
                            "cap": 0.90,
                            "val_low_rows": val_counts["low_rows"],
                            "val_high_rows": val_counts["high_rows"],
                            "oos_low_rows": oos_counts["low_rows"],
                            "oos_high_rows": oos_counts["high_rows"],
                            **_metric_row("val", omega._metrics(val_frame, val_sel, fee=fee, slip=slip, cost_mult=3.0)),
                            **_metric_row("oos", omega._metrics(oos_frame, oos_sel, fee=fee, slip=slip, cost_mult=3.0)),
                        }
                    )

    df = pd.DataFrame(rows)
    df["score"] = df["oos_pnl"] - df["val_mdd"].abs() * 0.10 - df["oos_mdd"].abs() * 0.25
    df = df.sort_values(["oos_pnl", "val_pnl", "oos_wr"], ascending=[False, False, False]).reset_index(drop=True)
    df.to_csv(OUT_DIR / "edge_risk_grid_ranking.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "baseline": {"validation": baseline_val, "oos": baseline_oos},
        "top_by_oos_pnl": df.head(15).to_dict(orient="records"),
        "top_stable": df.loc[(df["oos_mdd"] >= baseline_oos["mdd"] - 1.0) & (df["val_mdd"] >= baseline_val["mdd"] - 1.0)].head(15).to_dict(orient="records"),
        "ranking_csv": str(OUT_DIR / "edge_risk_grid_ranking.csv"),
    }
    (OUT_DIR / "edge_risk_grid_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
