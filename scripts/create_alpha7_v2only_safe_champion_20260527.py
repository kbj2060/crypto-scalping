#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, predict_policy_frame  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    FALLBACK_PARENT,
    FALLBACK_SUMMARY,
    PRIMARY_PARENT,
    PRIMARY_SUMMARY,
    _combine_primary_fallback,
    _combo_metrics,
    _load_best_scale_runtime,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read  # noqa: E402


MODEL_ID = "alpha7_v2only_safe_champion_20260527"
V2_PREFIX = "clean_regime4_state24_sticky090_v2_"
LEGACY_PREFIX = "clean_regime4_2024_unsup_v1_"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
TRAIN_2025 = ROOT / "tmp/causal_regen_20260516/alpha7_mamba_iqn_feature_contract_inputs_20260527/trade_candidates_2025_alpha6_current_tail111_exact.csv"
EVAL_2026 = ROOT / "tmp/causal_regen_20260516/alpha7_mamba_iqn_feature_contract_inputs_20260527/trade_candidates_2026_alpha6_current_tail111_exact.csv"
SPLIT_TS = pd.Timestamp("2025-10-01")

DERIVED_COLS = {"side_hint"}
DERIVED_PREFIXES = ("mom_", "abs_mom_")


@dataclass(frozen=True)
class OverlayCfg:
    name: str
    quality_min: float | None = None
    confidence_min: float | None = None
    current_conf_min: float | None = None
    current_instability_max: float | None = None
    current_whipsaw_max: float | None = None
    pred_conf_min: float | None = None
    pred_instability_max: float | None = None
    pred_whipsaw_max: float | None = None
    directional_bias_abs_min: float | None = None
    max_notional: float | None = None
    notional_scale: float = 1.0
    side_mode: str = "all"


def _is_derived(col: str) -> bool:
    return col in DERIVED_COLS or col.startswith(DERIVED_PREFIXES)


def _assert_failfast_contract(frame: pd.DataFrame, feature_cols: list[str], *, model_name: str) -> None:
    legacy_features = [c for c in feature_cols if str(c).startswith(LEGACY_PREFIX)]
    if legacy_features:
        raise RuntimeError(f"{model_name} contains forbidden legacy regime features: {legacy_features[:8]}")
    missing = [c for c in feature_cols if c not in frame.columns and not _is_derived(str(c))]
    if missing:
        raise RuntimeError(f"{model_name} missing required feature columns: {missing[:16]}")


def _assert_frame(frame: pd.DataFrame, *, split: str) -> None:
    legacy_cols = [c for c in frame.columns if str(c).startswith(LEGACY_PREFIX)]
    if legacy_cols:
        raise RuntimeError(f"{split} frame contains forbidden legacy regime columns: {legacy_cols[:8]}")
    v2_cols = [c for c in frame.columns if str(c).startswith(V2_PREFIX)]
    if len(v2_cols) < 20:
        raise RuntimeError(f"{split} frame has insufficient v2 regime columns: {len(v2_cols)}")
    for col in (
        f"{V2_PREFIX}confidence",
        f"{V2_PREFIX}instability_prob",
        f"{V2_PREFIX}whipsaw_prob",
        f"{V2_PREFIX}directional_bias",
        "regime4_pred_confidence",
        "regime4_pred_instability_prob",
        "regime4_pred_whipsaw_prob",
        "regime4_pred_directional_bias",
    ):
        if col not in frame.columns:
            raise RuntimeError(f"{split} frame missing filter column: {col}")


def _active(dec: pd.DataFrame) -> np.ndarray:
    action = pd.to_numeric(dec["action"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    return (action != ACTION_CASH) & (side != 0)


def _apply_overlay(frame: pd.DataFrame, dec: pd.DataFrame, cfg: OverlayCfg) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    frame = frame.reset_index(drop=True)
    active = _active(out)
    keep = active.copy()
    side = pd.to_numeric(out["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)

    if cfg.quality_min is not None:
        keep &= pd.to_numeric(out["quality_score"], errors="coerce").fillna(-999.0).to_numpy(dtype=np.float64) >= cfg.quality_min
    if cfg.confidence_min is not None:
        keep &= pd.to_numeric(out["confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) >= cfg.confidence_min
    if cfg.current_conf_min is not None:
        keep &= pd.to_numeric(frame[f"{V2_PREFIX}confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) >= cfg.current_conf_min
    if cfg.current_instability_max is not None:
        keep &= pd.to_numeric(frame[f"{V2_PREFIX}instability_prob"], errors="coerce").fillna(9.0).to_numpy(dtype=np.float64) <= cfg.current_instability_max
    if cfg.current_whipsaw_max is not None:
        keep &= pd.to_numeric(frame[f"{V2_PREFIX}whipsaw_prob"], errors="coerce").fillna(9.0).to_numpy(dtype=np.float64) <= cfg.current_whipsaw_max
    if cfg.pred_conf_min is not None:
        keep &= pd.to_numeric(frame["regime4_pred_confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) >= cfg.pred_conf_min
    if cfg.pred_instability_max is not None:
        keep &= pd.to_numeric(frame["regime4_pred_instability_prob"], errors="coerce").fillna(9.0).to_numpy(dtype=np.float64) <= cfg.pred_instability_max
    if cfg.pred_whipsaw_max is not None:
        keep &= pd.to_numeric(frame["regime4_pred_whipsaw_prob"], errors="coerce").fillna(9.0).to_numpy(dtype=np.float64) <= cfg.pred_whipsaw_max
    if cfg.directional_bias_abs_min is not None:
        cur_bias = pd.to_numeric(frame[f"{V2_PREFIX}directional_bias"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        pred_bias = pd.to_numeric(frame["regime4_pred_directional_bias"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        bias = 0.5 * cur_bias + 0.5 * pred_bias
        keep &= np.where(side > 0, bias >= cfg.directional_bias_abs_min, np.where(side < 0, bias <= -cfg.directional_bias_abs_min, False))
    if cfg.side_mode == "long":
        keep &= side > 0
    elif cfg.side_mode == "short":
        keep &= side < 0
    elif cfg.side_mode != "all":
        raise RuntimeError(f"unknown side_mode: {cfg.side_mode}")

    out.loc[active & ~keep, ["action", "side"]] = 0
    active2 = _active(out)
    notional = pd.to_numeric(out["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    notional = np.where(active2, notional * float(cfg.notional_scale), 0.0)
    if cfg.max_notional is not None:
        notional = np.where(active2, np.minimum(notional, float(cfg.max_notional)), 0.0)
    out["notional_exposure"] = notional
    lev = pd.to_numeric(out["leverage"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    out["position_fraction"] = np.where(active2, np.clip(notional / np.maximum(lev, 1e-12), 0.0, 1.0), 0.0)
    return out


def _predict_combo(frame: pd.DataFrame, primary: dict[str, Any], fallback: dict[str, Any]) -> pd.DataFrame:
    primary_rt = _load_best_scale_runtime(PRIMARY_SUMMARY)
    fallback_rt = _load_best_scale_runtime(FALLBACK_SUMMARY)
    primary_dec = predict_policy_frame(primary, frame, close=_close(frame)).reset_index(drop=True)
    fallback_dec = predict_policy_frame(fallback, frame, close=_close(frame)).reset_index(drop=True)
    if primary_rt is not None:
        from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2

        primary_dec = alpha2._scale_parent_notional(primary_dec, primary_rt).reset_index(drop=True)
    if fallback_rt is not None:
        from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2

        fallback_dec = alpha2._scale_parent_notional(fallback_dec, fallback_rt).reset_index(drop=True)
    return _combine_primary_fallback(primary_dec, fallback_dec)


def _metric_row(split: str, cfg: OverlayCfg, metrics: dict[str, Any]) -> dict[str, Any]:
    cost3 = metrics["cost3"]
    return {
        "split": split,
        **asdict(cfg),
        "pnl": float(cost3["pnl"]),
        "mdd": float(cost3["mdd"]),
        "trades": int(cost3["trades"]),
        "trades_per_day": float(cost3["trades_per_day"]),
        "wr": float(cost3["wr"]),
        "avg_notional": float(cost3.get("avg_notional", 0.0)),
        "score": float(cost3["pnl"]) + 120.0 * float(cost3["wr"]) - 0.8 * abs(float(cost3["mdd"])),
    }


def _grid() -> list[OverlayCfg]:
    out = [OverlayCfg("base")]
    for q in (None, 0.02, 0.025, 0.03):
        for c in (None, 0.55, 0.65, 0.75):
            if q is None and c is None:
                continue
            out.append(OverlayCfg(f"dec_q{q}_c{c}", quality_min=q, confidence_min=c))
    for inst in (0.45, 0.55, 0.65):
        for whip in (0.45, 0.55, 0.65):
            out.append(OverlayCfg(f"cur_risk_i{inst}_w{whip}", current_instability_max=inst, current_whipsaw_max=whip))
            out.append(OverlayCfg(f"pred_risk_i{inst}_w{whip}", pred_instability_max=inst, pred_whipsaw_max=whip))
    for bias in (0.02, 0.05, 0.08):
        out.append(OverlayCfg(f"bias_align_{bias}", directional_bias_abs_min=bias))
    for cap in (1.25, 1.5, 2.0):
        out.append(OverlayCfg(f"cap_{cap}", max_notional=cap))
    for scale in (0.6, 0.75, 0.9, 1.15):
        out.append(OverlayCfg(f"scale_{scale}", notional_scale=scale))
    for side in ("long", "short"):
        out.append(OverlayCfg(f"{side}_only", side_mode=side))
    for q in (0.02, 0.025):
        for inst in (0.55, 0.65):
            for whip in (0.55, 0.65):
                out.append(
                    OverlayCfg(
                        f"precision_q{q}_i{inst}_w{whip}",
                        quality_min=q,
                        confidence_min=0.55,
                        current_instability_max=inst,
                        current_whipsaw_max=whip,
                        max_notional=2.0,
                    )
                )
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    primary = joblib.load(PRIMARY_PARENT)
    fallback = joblib.load(FALLBACK_PARENT)
    train_all = _read(TRAIN_2025)
    eval_df = _read(EVAL_2026)
    val_df = train_all[pd.to_datetime(train_all["timestamp"], errors="coerce") >= SPLIT_TS].reset_index(drop=True)

    for split, frame in (("val", val_df), ("oos", eval_df)):
        _assert_frame(frame, split=split)
    for frame_name, frame in (("val", val_df), ("oos", eval_df)):
        _assert_failfast_contract(frame, list(primary["feature_cols"]), model_name=f"primary/{frame_name}")
        _assert_failfast_contract(frame, list(fallback["feature_cols"]), model_name=f"fallback/{frame_name}")

    val_base = _predict_combo(val_df, primary, fallback)
    eval_base = _predict_combo(eval_df, primary, fallback)

    rows: list[dict[str, Any]] = []
    best_val: tuple[float, OverlayCfg, dict[str, Any], pd.DataFrame] | None = None
    for cfg in _grid():
        val_dec = _apply_overlay(val_df, val_base, cfg)
        val_metrics = _combo_metrics(val_df, val_dec)
        row = _metric_row("val", cfg, val_metrics)
        rows.append(row)
        if int(row["trades"]) < 30:
            score = -1e9 + float(row["pnl"])
        else:
            score = float(row["score"])
        if best_val is None or score > best_val[0]:
            best_val = (score, cfg, val_metrics, val_dec)
    assert best_val is not None

    _, best_cfg, best_val_metrics, best_val_dec = best_val
    eval_dec = _apply_overlay(eval_df, eval_base, best_cfg)
    eval_metrics = _combo_metrics(eval_df, eval_dec)
    rows.append(_metric_row("oos", best_cfg, eval_metrics))

    grid = pd.DataFrame(rows)
    grid.to_csv(OUT_DIR / "grid.csv", index=False)
    best_val_dec.to_csv(OUT_DIR / "validation_decisions.csv", index=False)
    eval_dec.to_csv(OUT_DIR / "oos_decisions.csv", index=False)
    summary = {
        "model_id": MODEL_ID,
        "design": "Alpha7 v2-only primary + alpha43 no-legacy fallback, with validation-selected fail-fast overlay. No legacy regime aliasing, no 01965/diagnostic/OOS-selected candidate.",
        "selected_overlay": asdict(best_cfg),
        "validation_costs": best_val_metrics,
        "oos_costs_after_val_selection": eval_metrics,
        "audit": {
            "selection_uses_2026": False,
            "selection_window": "2025-10-01..2025-12-31",
            "oos_window": "2026 fixed OOS",
            "legacy_regime_columns_allowed": False,
            "compat_alias_added": False,
            "diagnostic_oracle_candidates_allowed": False,
            "excluded_candidates": [
                "01965_random_alpha7_combo_primary_fallback",
                "01965 IQN OOS-selected overlays",
                "alpha7 high-turnover s1",
                "Mamba/IQN/CatBoost OOS-diagnostic veto",
            ],
            "feature_contract": {
                "current_regime_prefix": V2_PREFIX,
                "primary_feature_count": len(primary["feature_cols"]),
                "fallback_feature_count": len(fallback["feature_cols"]),
            },
        },
        "artifacts": {
            "grid": str((OUT_DIR / "grid.csv").relative_to(ROOT)),
            "validation_decisions": str((OUT_DIR / "validation_decisions.csv").relative_to(ROOT)),
            "oos_decisions": str((OUT_DIR / "oos_decisions.csv").relative_to(ROOT)),
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(OUT_DIR / "summary.json"), "selected_overlay": asdict(best_cfg), "oos_cost3": eval_metrics["cost3"]}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
