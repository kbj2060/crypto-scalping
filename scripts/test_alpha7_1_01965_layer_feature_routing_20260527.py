#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.backtest_alpha3_exit_guard_persistence_20260527 import backtest_signal_limit_exit_guard  # noqa: E402
from scripts.loop_alpha3_1_alpha6_alpha7_combo_search_until_0800_20260527 import (  # noqa: E402
    _apply_decision_mods,
    _decision_sources,
    _default_limit_cfg,
    _guard,
    _load_frames,
    _load_stack,
    _overlay,
    _score,
    _sl_ratio,
)
from scripts.precision_retest_01965_alpha7_combo_20260527 import CANDIDATE, _cfg_from_results  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "alpha7_1_01965_layer_feature_routing_20260527"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
SUMMARY_OUT = OUT_DIR / "summary.json"
GRID_OUT = OUT_DIR / "grid.csv"
TOP_COSTS_OUT = OUT_DIR / "top_costs.csv"
FEATURE_SPEC_OUT = OUT_DIR / "feature_layer_specs.json"

ACTIVE_2025 = ROOT / "tmp/causal_regen_20260516/certified/features_2025.csv"
ACTIVE_2026 = ROOT / "tmp/causal_regen_20260516/certified/features_2026.csv"
UTILITY_CSV = ROOT / "data/ensemble/reports/active_live_feature_analysis_20260527/active_live_feature_scores.csv"

ENTRY_FEATURES = [
    "m7_expected_ret",
    "m7_quant_up",
    "m7_quant_dn",
    "ai_reward_risk",
    "ai_dir_edge",
    "teacher_side_margin",
    "m7_trend_xgb_up",
    "m7_trend_xgb_dn",
    "timesnet_cycle_delta",
]

DEEP_FEATURES = [
    "teacher_side_margin",
    "teacher_uncertainty",
    "teacher_tail_warning",
    "m7_qwidth",
    "m7_confidence",
    "ai_vol_regime_pct",
    "ai_adverse_risk",
    "timesnet_cycle_delta",
    "timesnet_cycle_sin",
]

EXIT_FEATURES = [
    "m7_qwidth",
    "teacher_uncertainty",
    "teacher_tail_warning",
    "m7_tail_risk",
    "m7_confidence",
    "ai_vol_regime_pct",
    "ai_adverse_risk",
    "m7_quality_pred",
    "m7_hold_pred",
]

META_FEATURES = [
    "teacher_side_margin",
    "m7_composite_score",
    "m7_action",
    "m7_confidence",
    "m7_expected_ret",
    "ai_dir_edge",
    "m7_qwidth",
    "teacher_tail_warning",
]

DRIFT_RISK_FORBIDDEN = [
    "m7_entry_long_price",
    "m7_entry_short_price",
    "m7_tp_price",
    "m7_sl_price",
]


@dataclass(frozen=True)
class Variant:
    name: str
    entry_core: bool = False
    deep_context: bool = False
    exit_risk: bool = False
    meta_conflict: bool = False


VARIANTS = [
    Variant("baseline_01965"),
    Variant("entry_future_core", entry_core=True),
    Variant("deep_context_core", deep_context=True),
    Variant("exit_risk_context", exit_risk=True),
    Variant("meta_conflict_throttle", meta_conflict=True),
    Variant("entry_plus_exit", entry_core=True, exit_risk=True),
    Variant("deep_plus_exit", deep_context=True, exit_risk=True),
    Variant("entry_deep_exit", entry_core=True, deep_context=True, exit_risk=True),
    Variant("layered_full", entry_core=True, deep_context=True, exit_risk=True, meta_conflict=True),
]


def _active(dec: pd.DataFrame) -> np.ndarray:
    action = pd.to_numeric(dec["action"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    return (action != ACTION_CASH) & (side != 0)


def _safe_arr(frame: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in frame.columns:
        raise KeyError(f"required active/live feature missing: {col}")
    return (
        pd.to_numeric(frame[col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
        .to_numpy(dtype=np.float64)
    )


def _side_support(frame: pd.DataFrame) -> np.ndarray:
    return (
        0.35 * np.tanh(_safe_arr(frame, "teacher_side_margin") * 4.0)
        + 0.25 * np.tanh(_safe_arr(frame, "m7_expected_ret") * 220.0)
        + 0.20 * np.tanh(_safe_arr(frame, "ai_dir_edge") * 3.0)
        + 0.20 * np.tanh((_safe_arr(frame, "m7_trend_xgb_up") - _safe_arr(frame, "m7_trend_xgb_dn")) * 3.0)
    )


def _risk_score(frame: pd.DataFrame) -> np.ndarray:
    return np.clip(
        0.24 * np.clip(_safe_arr(frame, "teacher_tail_warning") / 3.0, 0.0, 1.0)
        + 0.22 * np.clip(_safe_arr(frame, "teacher_uncertainty") / 3.0, 0.0, 1.0)
        + 0.18 * np.clip(_safe_arr(frame, "m7_qwidth") / 0.02, 0.0, 1.0)
        + 0.16 * np.clip(_safe_arr(frame, "m7_tail_risk") / 0.03, 0.0, 1.0)
        + 0.12 * np.clip(_safe_arr(frame, "ai_vol_regime_pct"), 0.0, 1.0)
        + 0.08 * np.clip(_safe_arr(frame, "ai_adverse_risk"), 0.0, 1.0),
        0.0,
        1.0,
    )


def _augment_active_features(frame: pd.DataFrame, active_path: Path) -> pd.DataFrame:
    needed = sorted(set(ENTRY_FEATURES + DEEP_FEATURES + EXIT_FEATURES + META_FEATURES))
    missing_forbidden = [c for c in needed if c in DRIFT_RISK_FORBIDDEN]
    if missing_forbidden:
        raise RuntimeError(f"drift-risk raw price features are forbidden in this test: {missing_forbidden}")
    active = pd.read_csv(active_path, usecols=["timestamp", *needed])
    left = frame.copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True, errors="raise").dt.tz_convert(None)
    active["timestamp"] = pd.to_datetime(active["timestamp"], utc=True, errors="raise").dt.tz_convert(None)
    active = active.drop_duplicates("timestamp", keep="last")
    overlap = [c for c in needed if c in left.columns]
    if overlap:
        left = left.drop(columns=overlap)
    out = left.merge(active, on="timestamp", how="left", validate="one_to_one")
    missing = [c for c in needed if c not in out.columns or out[c].isna().any()]
    if missing:
        raise RuntimeError(f"active feature merge failed; missing/NaN: {missing[:20]}")
    return out.reset_index(drop=True)


def _load_active_feature_frames(val_df: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _augment_active_features(val_df, ACTIVE_2025), _augment_active_features(eval_df, ACTIVE_2026)


def _apply_entry_core(dec: pd.DataFrame, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = dec.copy().reset_index(drop=True)
    active = _active(out)
    side = pd.to_numeric(out["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    support = _side_support(frame)
    aligned = side * support
    weak = active & (aligned < -0.10)
    soft = active & (aligned >= -0.10) & (aligned < 0.02)
    notional = pd.to_numeric(out["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    notional = np.where(soft, notional * 0.70, notional)
    out["notional_exposure"] = notional
    out.loc[weak, ["action", "side"]] = 0
    return out, {"entry_weak_block": int(weak.sum()), "entry_soft_throttle": int(soft.sum())}


def _apply_meta_conflict(dec: pd.DataFrame, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = dec.copy().reset_index(drop=True)
    active = _active(out)
    side = pd.to_numeric(out["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    support = _side_support(frame)
    risk = _risk_score(frame)
    composite = np.tanh(_safe_arr(frame, "m7_composite_score") * 2.0)
    conflict = active & (side * support < -0.03) & (side * composite < -0.03)
    high_risk_conflict = conflict & (risk >= 0.55)
    mild_conflict = conflict & ~high_risk_conflict
    notional = pd.to_numeric(out["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    notional = np.where(mild_conflict, notional * 0.55, notional)
    out["notional_exposure"] = notional
    out.loc[high_risk_conflict, ["action", "side"]] = 0
    return out, {"meta_conflict_block": int(high_risk_conflict.sum()), "meta_conflict_throttle": int(mild_conflict.sum())}


def _apply_deep_context(q: np.ndarray, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    out = np.array(q, copy=True)
    support = _side_support(frame)
    risk = _risk_score(frame)
    long_scale = np.clip(1.0 + 0.35 * support - 0.45 * risk, 0.35, 1.35)
    short_scale = np.clip(1.0 - 0.35 * support - 0.45 * risk, 0.35, 1.35)
    out[:, 0] = out[:, 0] * long_scale
    out[:, 1] = out[:, 1] * short_scale
    return out.astype(np.float32), {
        "deep_avg_long_scale": float(np.mean(long_scale)),
        "deep_avg_short_scale": float(np.mean(short_scale)),
        "deep_high_risk_rows": int((risk >= 0.70).sum()),
    }


def _apply_exit_risk_frame(market_frame: pd.DataFrame, feature_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = market_frame.copy()
    risk = _risk_score(feature_frame)
    base_risk_off = (
        pd.to_numeric(out["risk_off_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if "risk_off_prob" in out.columns
        else np.zeros(len(out), dtype=np.float32)
    )
    base_whipsaw = (
        pd.to_numeric(out["whipsaw_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if "whipsaw_prob" in out.columns
        else np.zeros(len(out), dtype=np.float32)
    )
    # backtest_alpha3_exit_guard_persistence._regime_bad reads these columns.
    out["risk_off_prob"] = np.maximum(base_risk_off, risk)
    out["whipsaw_prob"] = np.maximum(
        base_whipsaw,
        np.clip(0.70 * risk + 0.30 * np.clip(_safe_arr(feature_frame, "m7_qwidth") / 0.02, 0.0, 1.0), 0.0, 1.0),
    )
    return out, {"exit_high_risk_rows": int((risk >= 0.55).sum()), "exit_risk_mean": float(np.mean(risk))}


def _prepare_variant(
    variant: Variant,
    *,
    market_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    q: np.ndarray,
    dec: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, Any]]:
    out_df = market_df
    out_q = q
    out_dec = dec.copy().reset_index(drop=True)
    audit: dict[str, Any] = {}
    if variant.entry_core:
        out_dec, row = _apply_entry_core(out_dec, feature_df)
        audit.update(row)
    if variant.meta_conflict:
        out_dec, row = _apply_meta_conflict(out_dec, feature_df)
        audit.update(row)
    if variant.deep_context:
        out_q, row = _apply_deep_context(out_q, feature_df)
        audit.update(row)
    if variant.exit_risk:
        out_df, row = _apply_exit_risk_frame(out_df, feature_df)
        audit.update(row)
    audit["active_after_variant"] = int(_active(out_dec).sum())
    return out_df, out_q, out_dec, audit


def _eval(
    *,
    df: pd.DataFrame,
    q: np.ndarray,
    dec: pd.DataFrame,
    stack: dict[str, Any],
    cfg: dict[str, Any],
    split: str,
    variant: str,
    cost_mult: int,
) -> dict[str, Any]:
    res = backtest_signal_limit_exit_guard(
        df.reset_index(drop=True),
        stack["parent"],
        stack["runner"],
        stack["add_cfg"],
        q,
        dec.reset_index(drop=True),
        _overlay(stack["overlay"], cfg),
        _default_limit_cfg(),
        _guard(cfg),
        fee=stack["fee"],
        slip=stack["slip"],
        cost_mult=float(cost_mult),
    )
    return {
        "split": split,
        "variant": variant,
        "cost": int(cost_mult),
        "pnl": float(res["pnl"]),
        "mdd": float(res["mdd"]),
        "wr": float(res["wr"]),
        "trades": int(res["trades"]),
        "trades_per_day": float(res["trades_per_day"]),
        "sl_ratio": float(_sl_ratio(res)),
        "score": float(_score(res)),
        "deep_entries": int(res.get("deep_entries", 0)),
        "long_entries": int(res.get("long_entries", 0)),
        "short_entries": int(res.get("short_entries", 0)),
        "avg_notional": float(res.get("avg_notional", 0.0)),
        "avg_leverage": float(res.get("avg_leverage", 0.0)),
        "exits": json.dumps(res.get("exits", {}), ensure_ascii=False, sort_keys=True),
    }


def _selection_score(row: pd.Series) -> float:
    trades = int(row["trades"])
    if trades < 50:
        return -1e9 + float(row["pnl"])
    return float(row["pnl"]) + 1.5 * float(row["mdd"]) + 55.0 * float(row["wr"]) - 0.02 * max(0, trades - 220)


def _write_feature_specs() -> None:
    utility_rows = pd.read_csv(UTILITY_CSV).set_index("feature")
    spec = {
        "model_id": MODEL_ID,
        "base_candidate": CANDIDATE,
        "source_analysis": "docs/subagents/model_architect.md::Active/Live Feature Utility Memory - 2026-05-27",
        "artifacts": {
            "utility_scores": str(UTILITY_CSV),
            "m7_teacher_provenance": "data/ensemble/reports/m7_teacher_live_provenance_20260527_audit.json",
        },
        "forbidden_drift_risk_features": DRIFT_RISK_FORBIDDEN,
        "layers": {
            "entry_parent_meta": ENTRY_FEATURES,
            "deep_sequence_context": DEEP_FEATURES,
            "exit_risk_context": EXIT_FEATURES,
            "meta_conflict_throttle": META_FEATURES,
        },
        "feature_scores": {
            name: {
                c: {
                    "future_score_0_100": float(utility_rows.loc[c, "future_score_0_100"]),
                    "current_score_0_100": float(utility_rows.loc[c, "current_score_0_100"]),
                    "utility_bucket": str(utility_rows.loc[c, "utility_bucket"]),
                }
                for c in cols
                if c in utility_rows.index
            }
            for name, cols in {
                "entry_parent_meta": ENTRY_FEATURES,
                "deep_sequence_context": DEEP_FEATURES,
                "exit_risk_context": EXIT_FEATURES,
                "meta_conflict_throttle": META_FEATURES,
            }.items()
        },
    }
    FEATURE_SPEC_OUT.write_text(json.dumps(spec, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _cfg_from_results()
    if cfg.get("source") != "alpha7_combo_primary_fallback":
        raise RuntimeError(f"01965 source contract changed: {cfg.get('source')}")
    _write_feature_specs()

    stack = _load_stack()
    val_df, eval_df = _load_frames()
    val_feature_df, eval_feature_df = _load_active_feature_frames(val_df, eval_df)
    sources = _decision_sources(val_df, eval_df, stack["parent"])
    base_val = _apply_decision_mods(sources[str(cfg["source"])][0], cfg)
    base_eval = _apply_decision_mods(sources[str(cfg["source"])][1], cfg)
    val_q = v27._predict_all(stack["deep_model"], val_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    eval_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])

    rows: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}
    prepared: dict[tuple[str, str], tuple[pd.DataFrame, np.ndarray, pd.DataFrame]] = {}
    for variant in VARIANTS:
        for split, df, feature_df, q, dec in (
            ("val", val_df, val_feature_df, val_q, base_val),
            ("oos", eval_df, eval_feature_df, eval_q, base_eval),
        ):
            v_df, v_q, v_dec, audit = _prepare_variant(variant, market_df=df, feature_df=feature_df, q=q, dec=dec)
            audits[f"{split}:{variant.name}"] = audit
            prepared[(split, variant.name)] = (v_df, v_q, v_dec)
            row = _eval(df=v_df, q=v_q, dec=v_dec, stack=stack, cfg=cfg, split=split, variant=variant.name, cost_mult=3)
            row.update({f"audit_{k}": v for k, v in audit.items()})
            rows.append(row)

    grid = pd.DataFrame(rows)
    val_grid = grid[grid["split"].eq("val")].copy()
    val_grid["selection_score"] = val_grid.apply(_selection_score, axis=1)
    selected = val_grid.sort_values(["selection_score", "pnl", "wr"], ascending=False).head(3)
    selected_variants = list(dict.fromkeys(["baseline_01965", *selected["variant"].astype(str).tolist()]))

    full_rows: list[dict[str, Any]] = []
    for variant_name in selected_variants:
        for split in ("val", "oos"):
            v_df, v_q, v_dec = prepared[(split, variant_name)]
            for cost in (1, 2, 3):
                full_rows.append(_eval(df=v_df, q=v_q, dec=v_dec, stack=stack, cfg=cfg, split=split, variant=variant_name, cost_mult=cost))
    top_costs = pd.DataFrame(full_rows)
    grid.to_csv(GRID_OUT, index=False)
    top_costs.to_csv(TOP_COSTS_OUT, index=False)

    summary = {
        "model_id": MODEL_ID,
        "base_candidate": CANDIDATE,
        "design": "Layer-role feature routing test on Alpha7.1-01965 using active/live 2026-05-27 feature utility analysis.",
        "selection_uses_2026": False,
        "selection_rule": "top 3 variants selected by validation cost3 selection_score; OOS read after selection",
        "feature_spec": str(FEATURE_SPEC_OUT),
        "grid": str(GRID_OUT),
        "top_costs": str(TOP_COSTS_OUT),
        "variants": [variant.__dict__ for variant in VARIANTS],
        "selected_variants": selected_variants,
        "val_cost3": val_grid.sort_values("selection_score", ascending=False).to_dict(orient="records"),
        "selected_metrics": top_costs.to_dict(orient="records"),
        "variant_audits": audits,
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY_OUT), "grid": str(GRID_OUT), "selected": selected_variants}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
