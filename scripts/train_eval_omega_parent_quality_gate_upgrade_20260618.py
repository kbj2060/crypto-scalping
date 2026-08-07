#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_true3head_overlays_20260604 as overlay  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_1_full_retrain_cash_alpha43_20260608 as full_parent  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega_parent_quality_gate_upgrade_20260618"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
PARENT_MODEL_ID = "omega1_2_true_3head_tabm_20260603_full_retrain_cash_alpha43_20260608"

CURRENT_THR = {"bull": 0.72, "bear": 0.64, "chop": 0.65}
GLOBAL_GRID = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
BULL_GRID = (0.58, 0.65, 0.72, 0.78)
BEAR_GRID = (0.52, 0.58, 0.64, 0.70)
CHOP_GRID = (0.55, 0.60, 0.65, 0.72)
SIDE_GRID = (0.55, 0.60, 0.65, 0.70, 0.75)


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


def _prediction_paths(split: str) -> tuple[str, Path, bool]:
    if split == "validation":
        return "omega1_regime3_expertdq_oof_", full_parent.PARENT_DIR / "validation_predictions_2025_true3head.csv", True
    if split == "oos":
        return "omega1_regime3_expertdq_", full_parent.PARENT_DIR / "oos_predictions_2026_true3head.csv", False
    raise RuntimeError(f"unknown split: {split}")


def _load_split(frames: dict[str, pd.DataFrame], split: str) -> tuple[pd.DataFrame, pd.DataFrame, str, bool]:
    prefix, pred_path, oof = _prediction_paths(split)
    if split == "validation":
        frame = frames["val_raw"].reset_index(drop=True)
    else:
        frame = frames["oos_raw"].reset_index(drop=True)
    pred = pd.read_csv(pred_path, parse_dates=["timestamp"])
    src = full_parent._align(frame, pred)
    return frame, src, prefix, oof


def _threshold_array(src: pd.DataFrame, prefix: str, cfg: dict[str, Any]) -> np.ndarray:
    expert = src[f"{prefix}router_expert"].astype(str).replace({"chop_expert": "chop"}).to_numpy()
    action = pd.to_numeric(src[f"{prefix}dir_action"], errors="raise").to_numpy(dtype=np.int64)
    family = str(cfg["family"])
    if family == "current_control":
        return np.asarray([CURRENT_THR.get(str(x), CURRENT_THR["chop"]) for x in expert], dtype=np.float64)
    if family == "global":
        return np.full(len(src), float(cfg["global_thr"]), dtype=np.float64)
    if family == "regime":
        return np.asarray([float(cfg[f"{x}_thr"]) for x in expert], dtype=np.float64)
    if family == "side":
        return np.where(action == omega.ACTION_LONG, float(cfg["long_thr"]), float(cfg["short_thr"])).astype(np.float64)
    if family == "regime_side":
        out = np.empty(len(src), dtype=np.float64)
        for regime in ("bull", "bear", "chop"):
            for side_name, side_action in (("long", omega.ACTION_LONG), ("short", omega.ACTION_SHORT)):
                mask = (expert == regime) & (action == side_action)
                out[mask] = float(cfg[f"{regime}_{side_name}_thr"])
            out[(expert == regime) & (action == omega.ACTION_CASH)] = float(cfg[f"{regime}_cash_thr"])
        return out
    raise RuntimeError(f"unknown family: {family}")


def _build_dec(src: pd.DataFrame, prefix: str, *, oof: bool, cfg: dict[str, Any]) -> pd.DataFrame:
    out = src.copy()
    q = pd.to_numeric(out[f"{prefix}quality_for_action"], errors="raise").to_numpy(dtype=np.float64)
    action = pd.to_numeric(out[f"{prefix}dir_action"], errors="raise").to_numpy(dtype=np.int64)
    thr = _threshold_array(out, prefix, cfg)
    out[f"{prefix}quality_threshold"] = thr
    out[f"{prefix}final_action"] = np.where(q >= thr, action, omega.ACTION_CASH).astype(np.int64)
    dec = omega._to_fixed_decisions(out, oof=oof)
    active = omega._active(dec)
    for expert, scale in overlay.SCALE_MAP.items():
        key = "chop_expert" if expert == "chop" else expert
        mask = active & dec["router_expert"].astype(str).eq(key)
        ratio = float(scale) / float(overlay.BASE_SCALES[key])
        dec.loc[mask, "notional_exposure"] = pd.to_numeric(dec.loc[mask, "notional_exposure"], errors="raise") * ratio
        dec.loc[mask, "position_fraction"] = pd.to_numeric(dec.loc[mask, "position_fraction"], errors="raise") * ratio
    active = omega._active(dec)
    dec.loc[active, "take_profit"] = overlay.TP
    dec.loc[active, "stop_loss"] = overlay.SL
    dec.loc[active, "max_hold_bars"] = 0
    dec.loc[active, "cooldown_bars"] = 0
    return sleeve._apply_aggressive(dec)


def _metric_row(candidate: str, cfg: dict[str, Any], val_m: dict[str, Any], oos_m: dict[str, Any], base_val: dict[str, Any], base_oos: dict[str, Any]) -> dict[str, Any]:
    row = {"candidate": candidate, **cfg}
    row.update(sleeve._metric_row("val", {**val_m, "primary_entries": val_m["long_entries"] + val_m["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}))
    row.update(sleeve._metric_row("oos", {**oos_m, "primary_entries": oos_m["long_entries"] + oos_m["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}))
    row["val_delta_pnl_vs_current"] = float(row["val_pnl"] - base_val["pnl"])
    row["oos_delta_pnl_vs_current"] = float(row["oos_pnl"] - base_oos["pnl"])
    row["val_trade_delta_vs_current"] = int(row["val_trades"] - base_val["trades"])
    row["oos_trade_delta_vs_current"] = int(row["oos_trades"] - base_oos["trades"])
    row["selection_score_val_only"] = (
        float(row["val_pnl"])
        + 0.20 * float(row["val_mdd"])
        + 8.0 * float(row["val_wr"])
        - 0.06 * max(0, int(row["val_trades"]) - int(base_val["trades"]))
    )
    return row


def _configs() -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = [("current_control", {"family": "current_control"})]
    for q in GLOBAL_GRID:
        out.append((f"global_q{q:.2f}", {"family": "global", "global_thr": float(q)}))
    for bull, bear, chop in itertools.product(BULL_GRID, BEAR_GRID, CHOP_GRID):
        out.append(
            (
                f"regime_b{bull:.2f}_r{bear:.2f}_c{chop:.2f}",
                {"family": "regime", "bull_thr": float(bull), "bear_thr": float(bear), "chop_thr": float(chop)},
            )
        )
    for long_thr, short_thr in itertools.product(SIDE_GRID, SIDE_GRID):
        out.append((f"side_l{long_thr:.2f}_s{short_thr:.2f}", {"family": "side", "long_thr": float(long_thr), "short_thr": float(short_thr)}))
    return out


def _action_counts(src: pd.DataFrame, prefix: str, cfg: dict[str, Any]) -> dict[str, Any]:
    q = pd.to_numeric(src[f"{prefix}quality_for_action"], errors="raise").to_numpy(dtype=np.float64)
    action = pd.to_numeric(src[f"{prefix}dir_action"], errors="raise").to_numpy(dtype=np.int64)
    thr = _threshold_array(src, prefix, cfg)
    final = np.where(q >= thr, action, omega.ACTION_CASH).astype(np.int64)
    return {
        "dir_action": {str(k): int(v) for k, v in pd.Series(action).value_counts().sort_index().to_dict().items()},
        "final_action": {str(k): int(v) for k, v in pd.Series(final).value_counts().sort_index().to_dict().items()},
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"stage": "load", "model_id": MODEL_ID}, ensure_ascii=True), flush=True)
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_frame, val_src, val_prefix, val_oof = _load_split(frames, "validation")
    oos_frame, oos_src, oos_prefix, oos_oof = _load_split(frames, "oos")

    current_cfg = {"family": "current_control"}
    current_val_dec = _build_dec(val_src, val_prefix, oof=val_oof, cfg=current_cfg)
    current_oos_dec = _build_dec(oos_src, oos_prefix, oof=oos_oof, cfg=current_cfg)
    base_val = omega._metrics(val_frame, current_val_dec, fee=fee, slip=slip, cost_mult=3.0)
    base_oos = omega._metrics(oos_frame, current_oos_dec, fee=fee, slip=slip, cost_mult=3.0)

    rows: list[dict[str, Any]] = []
    for name, cfg in _configs():
        val_dec = _build_dec(val_src, val_prefix, oof=val_oof, cfg=cfg)
        oos_dec = _build_dec(oos_src, oos_prefix, oof=oos_oof, cfg=cfg)
        val_m = omega._metrics(val_frame, val_dec, fee=fee, slip=slip, cost_mult=3.0)
        oos_m = omega._metrics(oos_frame, oos_dec, fee=fee, slip=slip, cost_mult=3.0)
        rows.append(_metric_row(name, cfg, val_m, oos_m, base_val, base_oos))

    ranking = pd.DataFrame(rows).sort_values(["selection_score_val_only", "val_pnl", "val_mdd"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "quality_gate_ranking.csv", index=False)
    selected = ranking.iloc[0].to_dict()
    best_oos = ranking.sort_values(["oos_pnl", "oos_delta_pnl_vs_current"], ascending=False).iloc[0].to_dict()
    selected_cfg = {k: selected[k] for k in selected.keys() if k in {"family", "global_thr", "bull_thr", "bear_thr", "chop_thr", "long_thr", "short_thr"}}
    report = {
        "model_id": MODEL_ID,
        "parent_model_id": PARENT_MODEL_ID,
        "status": "redteam_pass_quality_gate_eval",
        "method": "Replay full-retrained parent predictions. Change only quality gate thresholds before the existing cash_alpha43 overlay/aggressive parent-only decision builder. Select by validation score only; OOS is diagnostic.",
        "current_thresholds": CURRENT_THR,
        "current_control": _metric_row("current_control", current_cfg, base_val, base_oos, base_val, base_oos),
        "selected_by_validation": selected,
        "selected_action_counts": {
            "validation": _action_counts(val_src, val_prefix, selected_cfg),
            "oos": _action_counts(oos_src, oos_prefix, selected_cfg),
        },
        "best_by_oos_diagnostic": best_oos,
        "top20": ranking.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "quality_gate_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": selected, "current": report["current_control"], "best_oos": best_oos}, indent=2, ensure_ascii=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
