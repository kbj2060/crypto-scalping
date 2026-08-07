#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, predict_policy_frame  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_alpha3_regime4_state24_v2_full_retrain_20260526 as alpha3_full  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    FALLBACK_SUMMARY,
    PRIMARY_EVAL_CSV,
    PRIMARY_SUMMARY,
    PRIMARY_TRAIN_CSV,
    _combine_primary_fallback,
    _load_best_scale_runtime,
    _predict_scaled,
)
from scripts.backtest_alpha3_exit_guard_persistence_20260527 import (  # noqa: E402
    ExitGuardConfig,
    _default_limit_cfg,
    backtest_signal_limit_exit_guard,
)
from scripts.rebuild_alpha7_v2_only_live_20260526 import _rename_clean4_v2  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read  # noqa: E402


MODEL_ID = "alpha3_1_alpha6_alpha7_combo_loop_20260527"
BASE_REPORT = ROOT / "data/ensemble/reports/alpha3_regime4_state24_v2_full_retrain_20260526_summary.json"
ALPHA7_LIVE_DIR = ROOT / "data/ensemble/supervised/alpha7_v2_only_live_20260526"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
RESULTS_JSONL = OUT_DIR / "results.jsonl"
RESULTS_CSV = OUT_DIR / "results.csv"
BEST_JSON = OUT_DIR / "best.json"
SUMMARY_JSON = OUT_DIR / "summary.json"

DERIVABLE_FEATURES = {
    "side_hint",
    "mom_21d",
    "abs_mom_21d",
    "mom_3d",
    "abs_mom_3d",
    "mom_1d",
    "abs_mom_1d",
}

HISTORICAL_ZERO_FEATURES = {
    "patchtst_pred",
    "patchtst_confidence",
}


def _now() -> datetime:
    return datetime.now()


def _parse_until(value: str) -> datetime:
    value = value.strip()
    if ":" in value and len(value) <= 5:
        hour, minute = [int(x) for x in value.split(":", 1)]
        out = _now().replace(hour=hour, minute=minute, second=0, microsecond=0)
        if out <= _now():
            out += timedelta(days=1)
        return out
    return datetime.fromisoformat(value)


def _score(c3: dict[str, Any]) -> float:
    if int(c3.get("trades", 0)) < 20:
        return -1e9 + float(c3.get("pnl", 0.0))
    return (
        float(c3["pnl"])
        + 2.0 * float(c3["mdd"])
        + 40.0 * float(c3["wr"])
        - 0.03 * float(c3["trades"])
    )


def _sl_ratio(c3: dict[str, Any]) -> float:
    exits = dict(c3.get("exits", {}))
    sl = sum(int(v) for k, v in exits.items() if "stop_loss" in str(k))
    return float(sl / max(int(c3.get("trades", 0)), 1))


def _merge_state24(base: pd.DataFrame, side_path: Path) -> pd.DataFrame:
    side = alpha3_full._rename_state24_sidecar(_read(side_path))
    merged, _ = alpha3_full._merge_state24(base, side)
    return merged


def _augment_with_alpha7_features(frame: pd.DataFrame, alpha7_frame: pd.DataFrame) -> pd.DataFrame:
    left = frame.copy()
    right = alpha7_frame.copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    right = right.dropna(subset=["timestamp"]).drop_duplicates("timestamp", keep="last")
    extra_cols = [c for c in right.columns if c not in left.columns and c != "timestamp"]
    out = left.merge(right[["timestamp", *extra_cols]], on="timestamp", how="left")
    needed = [c for c in extra_cols if c.startswith("regime4_pred_") or c == "tp_sl_action_score"]
    bad = [c for c in needed if c not in out.columns or out[c].isna().any()]
    if bad:
        raise RuntimeError(f"alpha7 feature augmentation failed; missing/NaN columns: {bad[:20]}")
    return out


def _assert_parent_contract(parent: dict[str, Any], frame: pd.DataFrame, *, name: str) -> None:
    missing = [c for c in list(parent["feature_cols"]) if c not in frame.columns]
    allowed = DERIVABLE_FEATURES | HISTORICAL_ZERO_FEATURES
    non_derivable = [c for c in missing if c not in allowed]
    if non_derivable:
        raise RuntimeError(f"{name} feature contract mismatch: {non_derivable[:30]}")


def _load_stack() -> dict[str, Any]:
    rep = json.loads(BASE_REPORT.read_text(encoding="utf-8"))
    exp = dict(rep["experiments"][-1])
    parent = joblib.load(exp["artifacts"]["parent"])
    runner_payload = joblib.load(exp["artifacts"]["runner"])
    runner = runner_payload["cost_runner"]
    add_cfg = alpha3_full.v21.CostRunnerConfig(**dict(exp["selected_runner_config"]))
    overlay = alpha3_full.v31.OverlayConfig(**dict(exp["selected_overlay"]))
    deep_payload = torch.load(exp["artifacts"]["deep_scout"], map_location="cpu", weights_only=False)
    deep_model = v27.DeepAlphaTCN(len(deep_payload["seq_cols"]))
    deep_model.load_state_dict(deep_payload["state_dict"])
    deep_model = deep_model.cpu().eval()
    return {
        "parent": parent,
        "runner": runner,
        "add_cfg": add_cfg,
        "overlay": overlay,
        "deep_model": deep_model,
        "deep_payload": deep_payload,
        "fee": float(parent["config"]["fee"]),
        "slip": float(parent["config"]["slip"]),
    }


def _load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_all = _merge_state24(_read(v31.DEFAULT_TRAIN), alpha3_full.SIDE_CLEAN4_2025)
    eval_df = _merge_state24(_read(v31.DEFAULT_EVAL), alpha3_full.SIDE_CLEAN4_2026)
    a7_train = _rename_clean4_v2(_read(PRIMARY_TRAIN_CSV))
    a7_eval = _rename_clean4_v2(_read(PRIMARY_EVAL_CSV))
    train_all = _augment_with_alpha7_features(train_all, a7_train)
    eval_df = _augment_with_alpha7_features(eval_df, a7_eval)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    return val_df.reset_index(drop=True), eval_df.reset_index(drop=True)


def _decision_sources(val_df: pd.DataFrame, eval_df: pd.DataFrame, a3_parent: dict[str, Any]) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    a7_primary = joblib.load(ALPHA7_LIVE_DIR / "primary_parent.pkl")
    a7_fallback = joblib.load(ALPHA7_LIVE_DIR / "fallback_alpha43_no_legacy_parent.pkl")
    _assert_parent_contract(a3_parent, val_df, name="alpha3.1")
    _assert_parent_contract(a7_primary, val_df, name="alpha7_primary")
    _assert_parent_contract(a7_fallback, val_df, name="alpha7_fallback")

    p_rt = _load_best_scale_runtime(PRIMARY_SUMMARY)
    f_rt = _load_best_scale_runtime(FALLBACK_SUMMARY)
    a3_val = predict_policy_frame(a3_parent, val_df, close=_close(val_df)).reset_index(drop=True)
    a3_eval = predict_policy_frame(a3_parent, eval_df, close=_close(eval_df)).reset_index(drop=True)
    p_val = _predict_scaled(a7_primary, val_df, p_rt)
    p_eval = _predict_scaled(a7_primary, eval_df, p_rt)
    f_val = _predict_scaled(a7_fallback, val_df, f_rt)
    f_eval = _predict_scaled(a7_fallback, eval_df, f_rt)
    c_val = _combine_primary_fallback(p_val, f_val)
    c_eval = _combine_primary_fallback(p_eval, f_eval)
    return {
        "alpha3_1_parent_direct": (a3_val, a3_eval),
        "alpha7_primary": (p_val, p_eval),
        "alpha7_combo_primary_fallback": (c_val, c_eval),
        "alpha3_1_when_alpha7_agrees": (_agree_gate(a3_val, c_val), _agree_gate(a3_eval, c_eval)),
        "alpha3_1_plus_alpha7_cash_fallback": (_cash_fallback(a3_val, c_val), _cash_fallback(a3_eval, c_eval)),
    }


def _active(dec: pd.DataFrame) -> pd.Series:
    return (pd.to_numeric(dec["action"], errors="coerce").fillna(0).astype(int) != ACTION_CASH) & (
        pd.to_numeric(dec["side"], errors="coerce").fillna(0).astype(int) != 0
    )


def _agree_gate(primary: pd.DataFrame, confirmer: pd.DataFrame) -> pd.DataFrame:
    out = primary.copy().reset_index(drop=True)
    confirmer = confirmer.reset_index(drop=True)
    same_side = _active(out) & _active(confirmer) & (
        pd.to_numeric(out["side"], errors="coerce").fillna(0).astype(int)
        == pd.to_numeric(confirmer["side"], errors="coerce").fillna(0).astype(int)
    )
    out.loc[~same_side, ["action", "side"]] = 0
    return out


def _cash_fallback(primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    out = primary.copy().reset_index(drop=True)
    fallback = fallback.reset_index(drop=True)
    mask = (~_active(out)) & _active(fallback)
    for col in fallback.columns:
        if col in out.columns:
            out.loc[mask, col] = fallback.loc[mask, col].to_numpy()
    return out


def _apply_decision_mods(dec: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    q_min = float(cfg["entry_quality_min"])
    conf_min = float(cfg["entry_conf_min"])
    if q_min > -100.0:
        out.loc[pd.to_numeric(out["quality_score"], errors="coerce").fillna(-999.0) < q_min, ["action", "side"]] = 0
    if conf_min > 0.0:
        out.loc[pd.to_numeric(out["confidence"], errors="coerce").fillna(0.0) < conf_min, ["action", "side"]] = 0
    out["notional_exposure"] = (
        pd.to_numeric(out["notional_exposure"], errors="coerce").fillna(0.0) * float(cfg["parent_notional_mult"])
    ).clip(lower=0.0, upper=float(cfg["parent_notional_cap"]))
    out["take_profit"] = (
        pd.to_numeric(out["take_profit"], errors="coerce").fillna(0.0) * float(cfg["parent_tp_mult"])
    ).clip(lower=0.001, upper=1.5)
    out["stop_loss"] = (
        pd.to_numeric(out["stop_loss"], errors="coerce").fillna(0.0) * float(cfg["parent_sl_mult"])
    ).clip(lower=0.001, upper=0.30)
    out["max_hold_bars"] = (
        pd.to_numeric(out["max_hold_bars"], errors="coerce").fillna(0).astype(float) * float(cfg["parent_hold_mult"])
    ).clip(lower=1, upper=float(cfg["parent_hold_cap"])).round().astype(int)
    if bool(cfg["alpha6_bucketize_hold"]):
        buckets = np.array([6, 12, 24, 48, 96], dtype=np.int64)
        mh = pd.to_numeric(out["max_hold_bars"], errors="coerce").fillna(24).to_numpy(dtype=np.float64)
        out["max_hold_bars"] = buckets[np.argmin(np.abs(mh[:, None] - buckets[None, :]), axis=1)]
    return out


def _guard(cfg: dict[str, Any]) -> ExitGuardConfig:
    return ExitGuardConfig(
        name=str(cfg["name"]),
        hard_sl_mult=float(cfg["hard_sl_mult"]),
        soft_sl_mult=float(cfg["soft_sl_mult"]),
        early_bars=int(cfg["early_bars"]),
        early_sl_mult=float(cfg["early_sl_mult"]),
        soft_min_hold=int(cfg["soft_min_hold"]),
        soft_persist_bars=int(cfg["soft_persist_bars"]),
        regime_bad_th=float(cfg["regime_bad_th"]),
        flow_bad_th=float(cfg["flow_bad_th"]),
        giveback_trigger=float(cfg["giveback_trigger"]),
        giveback_min_mfe=float(cfg["giveback_min_mfe"]),
        giveback_min_hold=int(cfg["giveback_min_hold"]),
        entry_quality_min=-999.0,
        entry_conf_min=0.0,
        same_side_entry_gap=int(cfg["same_side_entry_gap"]),
        cooldown_after_hard_stop=int(cfg["cooldown_after_hard_stop"]),
        cooldown_after_soft_stop=int(cfg["cooldown_after_soft_stop"]),
        cooldown_after_giveback=int(cfg["cooldown_after_giveback"]),
    )


def _overlay(base: v31.OverlayConfig, cfg: dict[str, Any]) -> v31.OverlayConfig:
    return replace(
        base,
        name=f"{base.name}_{cfg['name']}",
        notional=float(np.clip(base.notional * float(cfg["deep_notional_mult"]), 0.10, 3.0)),
        base_tp=float(np.clip(base.base_tp * float(cfg["deep_tp_mult"]), 0.005, 0.20)),
        base_sl=float(np.clip(base.base_sl * float(cfg["deep_sl_mult"]), 0.004, 0.12)),
        base_hold=int(np.clip(round(base.base_hold * float(cfg["deep_hold_mult"])), 6, 192)),
        trail_activation=float(np.clip(float(cfg["deep_trail_activation"]), 0.0, 0.08)),
    )


def _candidate(rng: random.Random, i: int, source_names: list[str]) -> dict[str, Any]:
    presets = [
        {
            "tag": "alpha3_1_current",
            "source": "alpha3_1_parent_direct",
            "entry_quality_min": -999.0,
            "entry_conf_min": 0.0,
            "parent_notional_mult": 1.0,
            "parent_notional_cap": 9.0,
            "parent_tp_mult": 1.0,
            "parent_sl_mult": 1.0,
            "parent_hold_mult": 1.0,
            "parent_hold_cap": 864,
            "alpha6_bucketize_hold": False,
            "hard_sl_mult": 1.45,
            "soft_sl_mult": 1.0,
            "early_bars": 18,
            "early_sl_mult": 1.35,
            "soft_min_hold": 3,
            "soft_persist_bars": 3,
            "regime_bad_th": 0.50,
            "flow_bad_th": 0.02,
            "giveback_trigger": 0.72,
            "giveback_min_mfe": 0.014,
            "giveback_min_hold": 3,
            "same_side_entry_gap": 0,
            "cooldown_after_hard_stop": 0,
            "cooldown_after_soft_stop": 0,
            "cooldown_after_giveback": 0,
            "deep_notional_mult": 1.0,
            "deep_tp_mult": 1.0,
            "deep_sl_mult": 1.0,
            "deep_hold_mult": 1.0,
            "deep_trail_activation": 0.0,
        },
        {
            "tag": "alpha6_style_safer_horizon",
            "source": "alpha3_1_parent_direct",
            "entry_quality_min": 0.0015,
            "entry_conf_min": 0.0,
            "parent_notional_mult": 0.85,
            "parent_notional_cap": 2.50,
            "parent_tp_mult": 1.25,
            "parent_sl_mult": 1.65,
            "parent_hold_mult": 1.50,
            "parent_hold_cap": 96,
            "alpha6_bucketize_hold": True,
            "hard_sl_mult": 1.75,
            "soft_sl_mult": 1.15,
            "early_bars": 24,
            "early_sl_mult": 1.65,
            "soft_min_hold": 6,
            "soft_persist_bars": 3,
            "regime_bad_th": 0.60,
            "flow_bad_th": 0.03,
            "giveback_trigger": 0.82,
            "giveback_min_mfe": 0.020,
            "giveback_min_hold": 6,
            "same_side_entry_gap": 3,
            "cooldown_after_hard_stop": 6,
            "cooldown_after_soft_stop": 4,
            "cooldown_after_giveback": 3,
            "deep_notional_mult": 0.85,
            "deep_tp_mult": 1.25,
            "deep_sl_mult": 1.45,
            "deep_hold_mult": 1.40,
            "deep_trail_activation": 0.018,
        },
        {
            "tag": "alpha7_agreement_low_turnover",
            "source": "alpha3_1_when_alpha7_agrees",
            "entry_quality_min": -999.0,
            "entry_conf_min": 0.0,
            "parent_notional_mult": 1.0,
            "parent_notional_cap": 3.0,
            "parent_tp_mult": 1.10,
            "parent_sl_mult": 1.50,
            "parent_hold_mult": 1.25,
            "parent_hold_cap": 144,
            "alpha6_bucketize_hold": False,
            "hard_sl_mult": 1.65,
            "soft_sl_mult": 1.10,
            "early_bars": 24,
            "early_sl_mult": 1.55,
            "soft_min_hold": 4,
            "soft_persist_bars": 3,
            "regime_bad_th": 0.55,
            "flow_bad_th": 0.025,
            "giveback_trigger": 0.80,
            "giveback_min_mfe": 0.018,
            "giveback_min_hold": 5,
            "same_side_entry_gap": 4,
            "cooldown_after_hard_stop": 6,
            "cooldown_after_soft_stop": 4,
            "cooldown_after_giveback": 3,
            "deep_notional_mult": 0.75,
            "deep_tp_mult": 1.15,
            "deep_sl_mult": 1.35,
            "deep_hold_mult": 1.25,
            "deep_trail_activation": 0.018,
        },
    ]
    if i < len(presets):
        cfg = dict(presets[i])
    else:
        cfg = {
            "tag": "random",
            "source": rng.choice(source_names),
            "entry_quality_min": rng.choice([-999.0, 0.0005, 0.0010, 0.0015, 0.0020, 0.0025]),
            "entry_conf_min": rng.choice([0.0, 0.50, 0.55, 0.58, 0.60]),
            "parent_notional_mult": rng.choice([0.55, 0.70, 0.85, 1.0, 1.15]),
            "parent_notional_cap": rng.choice([0.25, 1.0, 1.5, 2.0, 2.5, 3.0, 9.0]),
            "parent_tp_mult": rng.choice([0.90, 1.0, 1.10, 1.25, 1.45, 1.70]),
            "parent_sl_mult": rng.choice([1.0, 1.25, 1.50, 1.75, 2.00, 2.40]),
            "parent_hold_mult": rng.choice([0.75, 1.0, 1.25, 1.50, 2.0]),
            "parent_hold_cap": rng.choice([48, 96, 144, 288, 864]),
            "alpha6_bucketize_hold": rng.choice([False, True]),
            "hard_sl_mult": rng.choice([1.20, 1.35, 1.50, 1.65, 1.85, 2.10, 2.40]),
            "soft_sl_mult": rng.choice([0.95, 1.05, 1.15, 1.30]),
            "early_bars": rng.choice([12, 18, 24, 36]),
            "early_sl_mult": rng.choice([1.20, 1.40, 1.60, 1.90, 2.20]),
            "soft_min_hold": rng.choice([3, 4, 6, 9, 12]),
            "soft_persist_bars": rng.choice([2, 3, 4]),
            "regime_bad_th": rng.choice([0.45, 0.50, 0.55, 0.60, 0.70]),
            "flow_bad_th": rng.choice([0.00, 0.01, 0.02, 0.03, 0.05]),
            "giveback_trigger": rng.choice([0.65, 0.72, 0.80, 0.88, 0.96]),
            "giveback_min_mfe": rng.choice([0.010, 0.014, 0.018, 0.024, 0.032]),
            "giveback_min_hold": rng.choice([3, 5, 8, 12]),
            "same_side_entry_gap": rng.choice([0, 2, 4, 6, 8]),
            "cooldown_after_hard_stop": rng.choice([0, 3, 6, 12]),
            "cooldown_after_soft_stop": rng.choice([0, 3, 6]),
            "cooldown_after_giveback": rng.choice([0, 2, 4, 6]),
            "deep_notional_mult": rng.choice([0.50, 0.75, 1.0, 1.20]),
            "deep_tp_mult": rng.choice([0.90, 1.0, 1.15, 1.35, 1.60]),
            "deep_sl_mult": rng.choice([1.0, 1.25, 1.50, 1.80, 2.20]),
            "deep_hold_mult": rng.choice([0.75, 1.0, 1.25, 1.50, 2.0]),
            "deep_trail_activation": rng.choice([0.0, 0.012, 0.018, 0.026, 0.036]),
        }
    cfg["name"] = f"{i:05d}_{cfg['tag']}_{cfg['source']}"
    return cfg


def _eval_one(
    *,
    val_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    stack: dict[str, Any],
    val_q: np.ndarray,
    eval_q: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    val_dec0, eval_dec0 = sources[str(cfg["source"])]
    val_dec = _apply_decision_mods(val_dec0, cfg)
    eval_dec = _apply_decision_mods(eval_dec0, cfg)
    guard = _guard(cfg)
    overlay = _overlay(stack["overlay"], cfg)
    limit_cfg = _default_limit_cfg()
    val_c3 = backtest_signal_limit_exit_guard(
        val_df,
        stack["parent"],
        stack["runner"],
        stack["add_cfg"],
        val_q,
        val_dec,
        overlay,
        limit_cfg,
        guard,
        fee=stack["fee"],
        slip=stack["slip"],
        cost_mult=3.0,
    )
    eval_c3 = backtest_signal_limit_exit_guard(
        eval_df,
        stack["parent"],
        stack["runner"],
        stack["add_cfg"],
        eval_q,
        eval_dec,
        overlay,
        limit_cfg,
        guard,
        fee=stack["fee"],
        slip=stack["slip"],
        cost_mult=3.0,
    )
    return {
        "name": cfg["name"],
        "source": cfg["source"],
        "tag": cfg["tag"],
        "val_score": float(_score(val_c3)),
        "val_pnl": float(val_c3["pnl"]),
        "val_mdd": float(val_c3["mdd"]),
        "val_wr": float(val_c3["wr"]),
        "val_trades": int(val_c3["trades"]),
        "val_sl_ratio": float(_sl_ratio(val_c3)),
        "oos_score": float(_score(eval_c3)),
        "oos_pnl": float(eval_c3["pnl"]),
        "oos_mdd": float(eval_c3["mdd"]),
        "oos_wr": float(eval_c3["wr"]),
        "oos_trades": int(eval_c3["trades"]),
        "oos_sl_ratio": float(_sl_ratio(eval_c3)),
        "oos_deep_entries": int(eval_c3.get("deep_entries", 0)),
        "oos_long_entries": int(eval_c3.get("long_entries", 0)),
        "oos_short_entries": int(eval_c3.get("short_entries", 0)),
        "oos_exits": eval_c3.get("exits", {}),
        "config": cfg,
    }


def _write_rows(rows: list[dict[str, Any]]) -> None:
    flat = []
    for r in rows:
        row = {k: v for k, v in r.items() if k not in {"config", "oos_exits"}}
        row["oos_exits"] = json.dumps(r.get("oos_exits", {}), ensure_ascii=False, sort_keys=True)
        for ck, cv in dict(r.get("config", {})).items():
            if ck not in row:
                row[f"cfg_{ck}"] = cv
        flat.append(row)
    pd.DataFrame(flat).sort_values(["oos_pnl", "val_score"], ascending=[False, False]).to_csv(RESULTS_CSV, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Loop Alpha3.1 x Alpha6 x Alpha7 layer combinations until a wall-clock cutoff.")
    ap.add_argument("--until", default="08:00")
    ap.add_argument("--max-iterations", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=7527)
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    until = _parse_until(str(args.until))
    rng = random.Random(int(args.seed))

    stack = _load_stack()
    val_df, eval_df = _load_frames()
    sources = _decision_sources(val_df, eval_df, stack["parent"])
    val_q = v27._predict_all(stack["deep_model"], val_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    eval_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])

    existing: list[dict[str, Any]] = []
    if RESULTS_JSONL.exists():
        for line in RESULTS_JSONL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing.append(json.loads(line))
    start_i = len(existing)
    best_oos = max(existing, key=lambda r: float(r["oos_pnl"])) if existing else None
    best_val = max(existing, key=lambda r: float(r["val_score"])) if existing else None

    with RESULTS_JSONL.open("a", encoding="utf-8") as fh:
        i = start_i
        while i < int(args.max_iterations) and _now() < until:
            cfg = _candidate(rng, i, list(sources))
            try:
                row = _eval_one(
                    val_df=val_df,
                    eval_df=eval_df,
                    sources=sources,
                    stack=stack,
                    val_q=val_q,
                    eval_q=eval_q,
                    cfg=cfg,
                )
            except Exception as exc:
                row = {
                    "name": cfg["name"],
                    "source": cfg.get("source"),
                    "tag": cfg.get("tag"),
                    "error": repr(exc),
                    "config": cfg,
                }
            row["iteration"] = int(i)
            row["finished_at"] = _now().isoformat(timespec="seconds")
            fh.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
            fh.flush()
            if "error" not in row:
                existing.append(row)
                best_oos = row if best_oos is None or float(row["oos_pnl"]) > float(best_oos["oos_pnl"]) else best_oos
                best_val = row if best_val is None or float(row["val_score"]) > float(best_val["val_score"]) else best_val
                if i % 5 == 0:
                    _write_rows(existing)
                    BEST_JSON.write_text(
                        json.dumps(
                            {
                                "model_id": MODEL_ID,
                                "best_oos_pnl_observed": best_oos,
                                "best_validation_selected": best_val,
                                "selection_warning": "best_oos_pnl_observed is OOS-snooped and must not be live-promoted without a new untouched OOS window.",
                            },
                            ensure_ascii=False,
                            indent=2,
                            default=_json_default,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                print(
                    json.dumps(
                        {
                            "i": i,
                            "name": row["name"],
                            "val_pnl": row["val_pnl"],
                            "oos_pnl": row["oos_pnl"],
                            "oos_mdd": row["oos_mdd"],
                            "best_oos_pnl": None if best_oos is None else best_oos["oos_pnl"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            else:
                print(json.dumps({"i": i, "name": row["name"], "error": row["error"]}, ensure_ascii=False), flush=True)
            i += 1
            if float(args.sleep) > 0.0:
                time.sleep(float(args.sleep))

    if existing:
        _write_rows(existing)
    summary = {
        "model_id": MODEL_ID,
        "started_existing_rows": start_i,
        "finished_rows": len(existing),
        "until": until.isoformat(timespec="seconds"),
        "best_oos_pnl_observed": best_oos,
        "best_validation_selected": best_val,
        "source_models": {
            "alpha3_1": "Alpha3.1 no_teacher_parent_direct parent + V21.2/V27/V31/guard runner",
            "alpha6": "Layer ideas only: 5-bucket horizon, quality gate, capped notional, exit guard/giveback. Direct artifact not mixed because current common frame lacks required Alpha6 features.",
            "alpha7": "Actual v2-only primary and primary+alpha4.3 fallback decisions are included as decision sources and gates.",
        },
        "selection_warning": "OOS-PnL best is an alpha-discovery observation, not a promotion metric. Use validation-selected row or run a fresh untouched OOS before live use.",
        "results_jsonl": str(RESULTS_JSONL),
        "results_csv": str(RESULTS_CSV),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    BEST_JSON.write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "best_oos_pnl_observed": best_oos,
                "best_validation_selected": best_val,
                "selection_warning": summary["selection_warning"],
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(SUMMARY_JSON), "best": str(BEST_JSON), "rows": len(existing)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
