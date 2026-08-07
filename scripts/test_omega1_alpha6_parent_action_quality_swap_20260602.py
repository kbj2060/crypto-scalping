#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import _json_default  # noqa: E402
from scripts.eval_omega1_regime3_expertdq_risk_replay_20260602 import ACTIVE_SCALES, ACTIVE_TEMPLATE  # noqa: E402
from scripts.train_eval_omega1_expertdq_dsac_proposal_overlay_20260602 import _fast_replay_metrics  # noqa: E402
from scripts.train_eval_omega1_expertdq_dsac_risk_allocator_20260602 import (  # noqa: E402
    ACTION_CASH,
    ACTION_LONG,
    ACTION_SHORT,
    _active,
    _load_variant_frames,
    _to_decisions,
    _zero_row,
)


MODEL_ID = "omega1_alpha6_parent_action_quality_swap_20260602"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
DEFAULT_ALPHA6_BUNDLE = ROOT / "data/ensemble/supervised/alpha6_entry_quality_exit_5bucket_main_20260522/current_tail111_bundle.joblib"


def _class_prob(model: Any, x: np.ndarray, cls: int) -> np.ndarray:
    proba = np.asarray(model.predict_proba(x), dtype=np.float64)
    classes = np.asarray(model.classes_, dtype=int)
    if int(cls) not in classes:
        return np.zeros(len(x), dtype=np.float64)
    return proba[:, int(np.flatnonzero(classes == int(cls))[0])]


def _predict_alpha6(bundle: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    cols = list(bundle["feature_cols"])
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise RuntimeError(f"Alpha6 feature contract mismatch. Missing columns: {missing[:30]}")
    x_raw = frame.loc[:, cols].copy()
    for col in cols:
        x_raw[col] = pd.to_numeric(x_raw[col], errors="coerce")
    x_raw = x_raw.replace([np.inf, -np.inf], np.nan)
    x = bundle["pipeline"].transform(x_raw)
    models = bundle["entry_models"]
    action_model = models["action_model"]
    cash_p = _class_prob(action_model, x, ACTION_CASH)
    long_p = _class_prob(action_model, x, ACTION_LONG)
    short_p = _class_prob(action_model, x, ACTION_SHORT)
    proba = np.vstack([cash_p, long_p, short_p]).T
    action = np.argmax(proba, axis=1).astype(np.int64)
    quality = np.asarray(models["quality_model"].predict(x), dtype=np.float64)
    return pd.DataFrame(
        {
            "alpha6_action": action,
            "alpha6_side": np.where(action == ACTION_LONG, 1, np.where(action == ACTION_SHORT, -1, 0)).astype(np.int64),
            "alpha6_cash_prob": cash_p,
            "alpha6_long_prob": long_p,
            "alpha6_short_prob": short_p,
            "alpha6_confidence": np.max(proba, axis=1),
            "alpha6_quality": quality,
        }
    )


def _template_decisions(pred: pd.DataFrame) -> pd.DataFrame:
    action = pred["alpha6_action"].to_numpy(dtype=np.int64)
    side = pred["alpha6_side"].to_numpy(dtype=np.int64)
    active = (action != ACTION_CASH) & (side != 0)
    dec = pd.DataFrame(
        {
            "action": action,
            "side": side,
            "notional_exposure": np.where(active, float(ACTIVE_TEMPLATE["notional"]), 0.0),
            "leverage": np.where(active, float(ACTIVE_TEMPLATE["leverage"]), 1.0),
            "position_fraction": np.where(active, float(ACTIVE_TEMPLATE["notional"]) / max(float(ACTIVE_TEMPLATE["leverage"]), 1e-8), 0.0),
            "take_profit": np.where(active, float(ACTIVE_TEMPLATE["take_profit"]), 0.0),
            "stop_loss": np.where(active, float(ACTIVE_TEMPLATE["stop_loss"]), 0.0),
            "max_hold_bars": np.where(active, int(ACTIVE_TEMPLATE["max_hold"]), 0).astype(np.int64),
            "cooldown_bars": np.where(active, int(ACTIVE_TEMPLATE["cooldown"]), 0).astype(np.int64),
            "quality_score": pred["alpha6_quality"].to_numpy(dtype=np.float64),
            "confidence": pred["alpha6_confidence"].to_numpy(dtype=np.float64),
        }
    )
    return dec


def _overlay_on_omega_risk(base: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    out = base.copy().reset_index(drop=True)
    action = pred["alpha6_action"].to_numpy(dtype=np.int64)
    side = pred["alpha6_side"].to_numpy(dtype=np.int64)
    alpha_active = (action != ACTION_CASH) & (side != 0)
    base_active = _active(out)
    out.loc[:, "action"] = action
    out.loc[:, "side"] = side
    out.loc[:, "quality_score"] = pred["alpha6_quality"].to_numpy(dtype=np.float64)
    out.loc[:, "confidence"] = pred["alpha6_confidence"].to_numpy(dtype=np.float64)
    for idx in np.flatnonzero(~alpha_active):
        out.iloc[int(idx)] = _zero_row(out.iloc[int(idx)])
    fill = alpha_active & (~base_active)
    if bool(np.any(fill)):
        out.loc[fill, "notional_exposure"] = float(ACTIVE_TEMPLATE["notional"])
        out.loc[fill, "leverage"] = float(ACTIVE_TEMPLATE["leverage"])
        out.loc[fill, "position_fraction"] = float(ACTIVE_TEMPLATE["notional"]) / max(float(ACTIVE_TEMPLATE["leverage"]), 1e-8)
        out.loc[fill, "take_profit"] = float(ACTIVE_TEMPLATE["take_profit"])
        out.loc[fill, "stop_loss"] = float(ACTIVE_TEMPLATE["stop_loss"])
        out.loc[fill, "max_hold_bars"] = int(ACTIVE_TEMPLATE["max_hold"])
        out.loc[fill, "cooldown_bars"] = int(ACTIVE_TEMPLATE["cooldown"])
    return out


def _score(row: pd.Series) -> float:
    trades = int(row.get("trades", 0) or 0)
    if trades < 30:
        return -1e9 + float(row.get("pnl", 0.0) or 0.0)
    return float(row.get("pnl", 0.0) + 130.0 * row.get("wr", 0.0) - 0.45 * abs(row.get("mdd", 0.0)) + 0.015 * trades)


def _metrics_row(split: str, variant: str, frame: pd.DataFrame, dec: pd.DataFrame, *, fee: float, slip: float, cost_mult: float) -> dict[str, Any]:
    overlays = np.ones(len(frame), dtype=np.int64)
    metrics = _fast_replay_metrics(frame, dec, overlays, fee=fee, slip=slip, cost_mult=cost_mult)
    usage = metrics.pop("usage")
    row = {"split": split, "variant": variant, "cost": 3, **metrics, "usage_json": json.dumps(usage, ensure_ascii=False)}
    row["selection_score"] = _score(pd.Series(row))
    return row


def _counts(name: str, dec: pd.DataFrame) -> dict[str, Any]:
    action = pd.to_numeric(dec["action"], errors="coerce").fillna(0).astype(int)
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).astype(int)
    return {
        f"{name}_action_counts": {str(k): int(v) for k, v in action.value_counts().sort_index().to_dict().items()},
        f"{name}_side_counts": {str(k): int(v) for k, v in side.value_counts().sort_index().to_dict().items()},
        f"{name}_active_rows": int(np.sum(_active(dec))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="soft_floor_0p00")
    ap.add_argument("--alpha6-bundle", type=Path, default=DEFAULT_ALPHA6_BUNDLE)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    args = ap.parse_args()

    out_dir = OUT_DIR / str(args.variant) / args.alpha6_bundle.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(args.alpha6_bundle)
    train_df, val_df, oos_df, train_src, val_src, oos_src, overlay = _load_variant_frames(str(args.variant))
    train_omega = _to_decisions(train_src, oof=True)
    val_omega = _to_decisions(val_src, oof=True)
    oos_omega = _to_decisions(oos_src, oof=False)

    train_pred = _predict_alpha6(bundle, train_df)
    val_pred = _predict_alpha6(bundle, val_df)
    oos_pred = _predict_alpha6(bundle, oos_df)
    train_template = _template_decisions(train_pred)
    val_template = _template_decisions(val_pred)
    oos_template = _template_decisions(oos_pred)
    train_overlay = _overlay_on_omega_risk(train_omega, train_pred)
    val_overlay = _overlay_on_omega_risk(val_omega, val_pred)
    oos_overlay = _overlay_on_omega_risk(oos_omega, oos_pred)

    parent_cfg = joblib.load(v31.DEFAULT_PARENT)["config"]
    fee = float(parent_cfg["fee"])
    slip = float(parent_cfg["slip"])
    rows = [
        _metrics_row("val", "omega1_expertdq_original", val_df, val_omega, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
        _metrics_row("oos", "omega1_expertdq_original", oos_df, oos_omega, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
        _metrics_row("val", "alpha6_action_quality_template_risk", val_df, val_template, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
        _metrics_row("oos", "alpha6_action_quality_template_risk", oos_df, oos_template, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
        _metrics_row("val", "alpha6_action_quality_overlay_omega_risk", val_df, val_overlay, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
        _metrics_row("oos", "alpha6_action_quality_overlay_omega_risk", oos_df, oos_overlay, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
    ]
    grid = pd.DataFrame(rows)
    grid_path = out_dir / "grid.csv"
    grid.to_csv(grid_path, index=False)
    summary = {
        "model_id": MODEL_ID,
        "variant": str(args.variant),
        "alpha6_bundle": str(args.alpha6_bundle),
        "design": "Replace Omega1 action and quality heads with Alpha6 parent action_model and quality_model. Evaluate both template risk and overlay-on-Omega-risk modes.",
        "selection_uses_2026": False,
        "legacy_compat_alias": False,
        "risk_template": ACTIVE_TEMPLATE,
        "expert_scales_reference": ACTIVE_SCALES,
        "alpha6_feature_cols": list(bundle["feature_cols"]),
        "alpha6_missing_features_in_bundle": list(bundle.get("missing_features", [])),
        "overlay": overlay,
        "counts": {
            **_counts("omega_train", train_omega),
            **_counts("alpha6_template_train", train_template),
            **_counts("alpha6_overlay_train", train_overlay),
            **_counts("omega_val", val_omega),
            **_counts("alpha6_template_val", val_template),
            **_counts("alpha6_overlay_val", val_overlay),
            **_counts("omega_oos", oos_omega),
            **_counts("alpha6_template_oos", oos_template),
            **_counts("alpha6_overlay_oos", oos_overlay),
        },
        "grid": rows,
        "artifacts": {"summary": str(out_dir / "summary.json"), "grid": str(grid_path)},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(out_dir / "summary.json"), "grid": rows}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
