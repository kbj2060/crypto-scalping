#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_asymmetric_direction_cleanup_20260618 as fast_frames  # noqa: E402
import train_eval_omega1_2_tabm_4head_price_risk_20260618 as base4  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


MODEL_ID = "omega1_2_true_4head_price_move_exit_eval_20260618"
SOURCE_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_true_4head_price_risk_notional_bucket_conservative_tabm_20260618"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _load_models() -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for expert in hard.EXPERT_NAMES:
        path = SOURCE_DIR / "models" / f"{expert}_4head_price_risk_tabm.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        models[expert] = torch.load(path, map_location="cpu", weights_only=False)
    return models


def _read_required(path: Path, cols: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    wanted = set(cols) | {"timestamp"}
    frame = pd.read_csv(path, usecols=lambda c: c in wanted, parse_dates=["timestamp"], low_memory=False)
    if "timestamp" not in frame.columns:
        raise RuntimeError(f"{path} missing timestamp")
    return frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _merge_overlay(base: pd.DataFrame, path: Path, needed: set[str], tag: str) -> pd.DataFrame:
    cols = needed - set(base.columns)
    if not cols:
        return base
    src = _read_required(path, cols)
    have = [c for c in cols if c in src.columns]
    if not have:
        return base
    before = len(base)
    out = base.merge(src[["timestamp", *have]], on="timestamp", how="left", validate="one_to_one")
    if len(out) != before:
        raise RuntimeError(f"{tag}: row count changed")
    missing = [c for c in have if out[c].isna().any()]
    if missing:
        raise RuntimeError(f"{tag}: missing overlay values for {missing[:10]}")
    return out


def _load_split_frame(base_cols: list[str], split: str) -> pd.DataFrame:
    needed = set(base_cols) | {"open", "high", "low", "close"}
    if split == "validation":
        frame = _read_required(omega.TRAIN_CSV, needed)
        frame = _merge_overlay(frame, omega.REGIME3_CURRENT_2025, needed, "val_regime3_current")
        frame = _merge_overlay(frame, omega.REGIME3_CMAMBA_2025, needed, "val_regime3_cmamba")
        frame = _merge_overlay(frame, omega.REGIME3_RISK_2025, needed, "val_regime3_risk")
        frame = frame[frame["timestamp"] >= threehead.SPLIT_TS].reset_index(drop=True)
        return fast_frames._filter_to_parent_prediction_span(frame, "validation")
    if split == "oos":
        frame = _read_required(omega.EVAL_CSV, needed)
        frame = _merge_overlay(frame, omega.REGIME3_CURRENT_2026, needed, "oos_regime3_current")
        frame = _merge_overlay(frame, omega.REGIME3_CMAMBA_2026, needed, "oos_regime3_cmamba")
        frame = _merge_overlay(frame, omega.REGIME3_RISK_2026, needed, "oos_regime3_risk")
        return fast_frames._filter_to_parent_prediction_span(frame.reset_index(drop=True), "oos")
    raise RuntimeError(f"unknown split: {split}")


def _predict_frame(
    frame: pd.DataFrame,
    models: dict[str, dict[str, Any]],
    base_cols: list[str],
    *,
    threshold: float,
    oof: bool,
    device: torch.device,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    x = threehead._base_input(frame, base_cols)
    preds = {expert: base4._predict_payload(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
    route = hard._route_id(frame)
    direction = threehead._routed(preds, route, "direction", 3)
    quality = threehead._routed(preds, route, "quality", 3)
    risk_unit = threehead._routed(preds, route, "risk_unit", len(base4.RISK_COLS))
    notional_prob = threehead._routed(preds, route, "notional", len(base4.NOTIONAL_BUCKETS))
    notional_bucket = np.argmax(notional_prob, axis=1).astype(np.int64)
    risk = base4._unit_to_risk(risk_unit.astype(np.float32)).astype(np.float64)
    prefix = "omega1_regime3_expertdq_oof" if oof else "omega1_regime3_expertdq"
    src = threehead._prediction_output(frame, direction, quality, threshold=float(threshold), prefix=prefix)
    dec = omega._to_fixed_decisions(src, oof=oof)
    return base4._apply_price_risk(dec, risk, notional_bucket), risk, notional_bucket


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = threehead._device("auto")
    fee, slip = omega._load_fee_slip()
    models = _load_models()
    first = models[hard.EXPERT_NAMES[0]]
    base_cols = list(first["input_columns"])
    val_raw = _load_split_frame(base_cols, "validation")
    oos_raw = _load_split_frame(base_cols, "oos")

    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for q in (0.45, 0.55, 0.65, 0.75, 0.80, 0.85, 0.90):
        val_dec, val_risk, val_bucket = _predict_frame(val_raw, models, base_cols, threshold=q, oof=True, device=device)
        oos_dec, oos_risk, oos_bucket = _predict_frame(oos_raw, models, base_cols, threshold=q, oof=False, device=device)
        val_m = base4._metrics_price_move_exit(val_raw, val_dec, fee=fee, slip=slip, cost_mult=3.0)
        oos_m = base4._metrics_price_move_exit(oos_raw, oos_dec, fee=fee, slip=slip, cost_mult=3.0)
        name = f"price_move_exit_q{q:.2f}".replace(".", "p")
        reports[name] = {
            "validation": val_m,
            "oos": oos_m,
            "validation_risk_distribution": {
                "tp_price_move_mean": float(np.mean(val_risk[:, 0])),
                "sl_price_move_mean": float(np.mean(val_risk[:, 1])),
                "notional_mean": float(np.mean(base4.NOTIONAL_BUCKETS[val_bucket])),
            },
            "oos_risk_distribution": {
                "tp_price_move_mean": float(np.mean(oos_risk[:, 0])),
                "sl_price_move_mean": float(np.mean(oos_risk[:, 1])),
                "notional_mean": float(np.mean(base4.NOTIONAL_BUCKETS[oos_bucket])),
            },
        }
        rows.append(base4._metric_row(name, val_m, oos_m, base4.CURRENT_PARENT_VAL, base4.CURRENT_PARENT_OOS))

    ranking = pd.DataFrame(rows).sort_values(["selection_score_val_only", "val_delta_vs_current", "val_pnl"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "ranking.csv", index=False)
    selected = ranking.iloc[0].to_dict()
    best_oos = ranking.sort_values(["oos_pnl", "oos_delta_vs_current"], ascending=False).iloc[0].to_dict()
    report = {
        "model_id": MODEL_ID,
        "source_dir": str(SOURCE_DIR),
        "status": "redteam_pass_price_move_exit_eval",
        "risk_contract": {
            "model_outputs": ["tp_price_move", "sl_price_move", "notional_bucket"],
            "entry": "direction/quality gates decide side; selected bucket value is notional_exposure",
            "exit": "price_move >= tp_price_move or price_move <= -sl_price_move",
            "pnl": "account PnL = realized price_move * notional",
            "leverage": "fixed bookkeeping value 1.0; not learned and not used in exits",
            "notional_buckets": base4.NOTIONAL_BUCKETS.tolist(),
        },
        "current_quality_gate_parent": {"validation": base4.CURRENT_PARENT_VAL, "oos": base4.CURRENT_PARENT_OOS},
        "selected_by_validation": selected,
        "best_by_oos_diagnostic": best_oos,
        "results": reports,
        "top30": ranking.head(30).to_dict(orient="records"),
        "artifacts": {"out_dir": str(OUT_DIR), "ranking": str(OUT_DIR / "ranking.csv"), "report": str(OUT_DIR / "report.json")},
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": selected, "best_oos": best_oos}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
