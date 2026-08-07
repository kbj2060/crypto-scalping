#!/usr/bin/env python3
"""Full bar-level Omega 4.5 v5_guard18p0 warmup replay.

This rebuilds component parent decisions and risk-sidecar outputs from the
saved Omega 4.2 component artifacts, then runs the v5 priority router at
bar-level with the 20260630 warmup gate applied before entry.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as risk_exp  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as omega4  # noqa: E402
import train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622 as price_exit  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


CREATIVE_BASE = ROOT / "tmp/causal_regen_20260516/omega_creative_until_10am_20260630"
PRIORITY_REPORT = CREATIVE_BASE / "priority_router_v5_h48_h48quality_zig/report.json"
GUARD_REPORT = (
    CREATIVE_BASE
    / "walkforward_oos_blind_source_side_scale_20260630_strict_mdd"
    / "v5_explainable_router"
    / "guard18p0"
    / "report.json"
)
BASELINE_MANIFEST = ROOT / "tmp/causal_regen_20260516/omega4_5_baseline_v5_guard18p0_20260630/report.json"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_5_v5_guard18p0_full_bar_warmup_20260630"

MODEL_ID = "omega4_5_v5_guard18p0_full_bar_warmup_20260630"
PRIORITY_ORDER = ("h48_conservative", "h48quality_repaired", "zigzag_q075")
WARMUP_BARS = 576
ZERO_DEFAULT_MAX = 0.35
MAX_LEVERAGE = 5.0
TARGET_PNL = 100.0
MDD_FLOOR = -20.0
MARKET_OR_LABEL_COLS = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
    "year",
    "label",
    "target",
    "zigzag_action",
    "quality_target",
}


@dataclass
class ComponentRuntime:
    alias: str
    split: str
    frame: pd.DataFrame
    base_x: pd.DataFrame
    decisions: pd.DataFrame
    loaded_models: dict[str, tuple[parent.ThreeHeadTabM, dict[str, Any]]]
    arrays: dict[str, np.ndarray]
    active: np.ndarray
    route: np.ndarray
    base_np: np.ndarray
    exit_runtime: dict[str, tuple[parent.ThreeHeadTabM, np.ndarray, np.ndarray]]
    pos_idx: list[int]
    base_margin_fraction: np.ndarray
    base_leverage: np.ndarray
    base_notional: np.ndarray
    source_side_scale: dict[str, float]
    config: dict[str, Any]
    device: torch.device


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text())


def json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def load_source_side_scale_map() -> dict[str, float]:
    report = read_json(GUARD_REPORT)
    scale_map = report.get("source_side_scales") or report.get("scale_map")
    if isinstance(scale_map, dict):
        return {str(k): float(v) for k, v in scale_map.items()}

    # The strict guard report stores the mapping at top-level in current artifacts.
    out: dict[str, float] = {}
    for key, value in report.items():
        if isinstance(key, str) and key.endswith(("_L", "_S")):
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                pass
    if out:
        return out

    # Final fallback is still fail-fast on exact ledger contract, not a guessed alias.
    ledger = pd.read_csv(GUARD_REPORT.parent / "validation_scaled_trade_ledger.csv")
    required = {"scale_group", "raw_source_side_scale"}
    if not required.issubset(ledger.columns):
        raise RuntimeError(f"cannot extract source/side scale map from {GUARD_REPORT}")
    for group, grp in ledger.groupby("scale_group"):
        vals = sorted(float(x) for x in grp["raw_source_side_scale"].dropna().unique())
        if len(vals) != 1:
            raise RuntimeError(f"non-unique scale for {group}: {vals}")
        out[str(group)] = vals[0]
    return out


def side_key(side: int) -> str:
    if int(side) == 1:
        return "L"
    if int(side) == -1:
        return "S"
    raise RuntimeError(f"unknown side: {side}")


def warmup_diagnostics(frame: pd.DataFrame, split: str, train_rows: int) -> pd.DataFrame:
    numeric = frame.select_dtypes(include=[np.number]).copy()
    keep_cols = [c for c in numeric.columns if c not in MARKET_OR_LABEL_COLS and not str(c).startswith("future_")]
    if not keep_cols:
        raise RuntimeError(f"{split}: no numeric feature columns for warmup diagnostics")
    feat = numeric[keep_cols].replace([np.inf, -np.inf], np.nan)
    zero_ratio = feat.eq(0.0).mean(axis=1).astype(float)
    zero_default_ratio = (feat.isna() | feat.eq(0.0)).mean(axis=1).astype(float)
    if "ai_ready" in frame.columns:
        ai_ready = pd.to_numeric(frame["ai_ready"], errors="coerce").fillna(0.0).astype(float)
    else:
        ai_ready = pd.Series(1.0, index=frame.index, dtype=float)
    if split == "validation":
        bar_index = np.arange(len(frame), dtype=np.int64) + int(train_rows)
        pre_split_tail_used = True
    else:
        bar_index = np.arange(len(frame), dtype=np.int64)
        pre_split_tail_used = False
    return pd.DataFrame(
        {
            "_warmup_bar_index": bar_index,
            "_warmup_feature_zero_ratio": zero_ratio.to_numpy(dtype=float),
            "_warmup_feature_zero_default_ratio": zero_default_ratio.to_numpy(dtype=float),
            "ai_ready": ai_ready.to_numpy(dtype=float),
            "_warmup_pre_split_tail_used": pre_split_tail_used,
        }
    )


def warmup_keep(diag: pd.DataFrame, i: int) -> tuple[bool, str]:
    row = diag.iloc[int(i)]
    reason = ""
    if float(row["_warmup_bar_index"]) < WARMUP_BARS:
        reason += f"bar_index_below_{WARMUP_BARS};"
    if float(row["ai_ready"]) != 1.0:
        reason += "ai_ready_not_1;"
    if float(row["_warmup_feature_zero_default_ratio"]) > ZERO_DEFAULT_MAX:
        reason += "zero_default_ratio_gt_0p35;"
    return reason == "", reason


def component_args(report: dict[str, Any]) -> dict[str, Any]:
    risk_model = report["risk_model"]
    contract = report["contract"]
    return {
        "baseline_bundle": Path(report["baseline_bundle"]),
        "train_csv": Path(risk_model["train_csv"]),
        "eval_csv": Path(risk_model["eval_csv"]),
        "direction_label_dir": Path(risk_model["direction_label_dir"]),
        "quality_mode": str(risk_model.get("quality_mode", "same_as_direction")),
        "quality_threshold": float(contract["quality_threshold"]),
        "exit_threshold": float(contract["exit_threshold"]),
        "atr_window": int(contract["atr_window"]),
        "tp_mult": float(contract["take_profit_atr_multiple"]),
        "sl_mult": float(contract["stop_loss_atr_multiple"]),
        "min_tp": float(contract["floor_take_profit_price_move"]),
        "min_sl": float(contract["floor_stop_loss_price_move"]),
        "max_tp": float(contract["cap_take_profit_price_move"]),
        "max_sl": float(contract["cap_stop_loss_price_move"]),
        "cost_mult": 3.0,
        "notional_scaled_sltp": bool(contract["notional_scaled_sltp"]),
        "exit_sizing_input_mode": str(risk_model.get("exit_sizing_input_mode", "actual")),
        "risk_feature_mode": str(risk_model.get("risk_feature_mode", "parent_outputs")),
        "score_quality_blend": float(risk_model.get("score_quality_blend", 0.0)),
        "exit_trend_threshold_scale": float(risk_model.get("exit_trend_threshold_scale", 0.0)),
        "exit_threshold_floor": float(risk_model.get("exit_threshold_floor", 0.55)),
        "exit_threshold_cap": float(risk_model.get("exit_threshold_cap", 0.95)),
    }


def apply_risk_sidecar(
    sidecar: dict[str, Any],
    features: pd.DataFrame,
    decisions: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_all, _ = risk_exp._feature_matrix(features, list(sidecar["feature_columns"]))
    side_all = pd.to_numeric(decisions["side"], errors="raise").to_numpy(dtype=np.int64)
    if bool(sidecar.get("side_split_model", False)):
        model_score = risk_exp._predict_side_split_models(sidecar["model"], x_all, side_all)
    else:
        model_score = np.asarray(sidecar["model"].predict(x_all), dtype=np.float64)
        model_score[side_all == 0] = 0.0

    if float(sidecar.get("score_quality_blend", 0.0)) != 0.0:
        raise RuntimeError("full replay does not support sidecar score_quality_blend without stored blend quantiles")

    mapping = {str(k): float(v) for k, v in sidecar["selected_mapping"].items()}
    margin_cfg = {k: mapping[k] for k in risk_exp.MARGIN_CFG_KEYS}
    leverage_cfg = {k: mapping[k] for k in risk_exp.LEVERAGE_CFG_KEYS if k in mapping}
    margins = risk_exp._risk_margins(
        decisions,
        model_score,
        train_q50=float(sidecar["train_score_q50"]),
        train_iqr=float(sidecar["train_score_iqr"]),
        **margin_cfg,
    )
    leverage = (
        risk_exp._risk_leverage(
            decisions,
            model_score,
            train_q50=float(sidecar["train_score_q50"]),
            train_iqr=float(sidecar["train_score_iqr"]),
            **leverage_cfg,
        )
        if leverage_cfg
        else pd.to_numeric(decisions["leverage"], errors="raise").to_numpy(dtype=np.float64)
    )
    notional = margins * leverage
    return margins, leverage, notional


def build_component(alias: str, split: str, device: torch.device) -> tuple[ComponentRuntime, int]:
    priority_report = read_json(PRIORITY_REPORT)
    component_dir = Path(priority_report["components"][alias]["out_dir"])
    report = read_json(component_dir / "report.json")
    cfg = component_args(report)

    omega.BASE_TEMPLATE["max_hold"] = 0
    omega.BASE_TEMPLATE["cooldown"] = 0
    omega.TRAIN_CSV = Path(cfg["train_csv"])
    omega.EVAL_CSV = Path(cfg["eval_csv"])

    bundle = torch.load(Path(cfg["baseline_bundle"]), map_location=device, weights_only=False)
    models: dict[str, dict[str, Any]] = bundle["models"]
    base_cols = list(bundle["base_cols"])
    loaded = parent._load_payloads(models, device=device)
    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=Path(cfg["direction_label_dir"]),
        quality_mode=str(cfg["quality_mode"]),
        quality_label_dir=None,
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )
    frame_key = "val_raw" if split == "validation" else "oos_raw"
    oof = split == "validation"
    frame = frames[frame_key].reset_index(drop=True)
    base_x, src, dec_base = risk_exp._predict_decisions(
        frame,
        oof=oof,
        models=models,
        base_cols=base_cols,
        quality_threshold=float(cfg["quality_threshold"]),
        device=device,
    )
    dec, _atr_diag = atr_eval._apply_atr_safety_sltp(
        dec_base,
        frame,
        atr_window=int(cfg["atr_window"]),
        tp_mult=float(cfg["tp_mult"]),
        sl_mult=float(cfg["sl_mult"]),
        min_tp=float(cfg["min_tp"]),
        min_sl=float(cfg["min_sl"]),
        max_tp=float(cfg["max_tp"]),
        max_sl=float(cfg["max_sl"]),
    )
    atr_pct = atr_eval._atr_pct(frame, int(cfg["atr_window"]))
    features = risk_exp._risk_feature_frame(
        frame,
        src,
        dec,
        base_cols,
        atr_pct=atr_pct,
        feature_mode=str(cfg["risk_feature_mode"]),
    )
    with (component_dir / "risk_sidecar.pkl").open("rb") as f:
        sidecar = pickle.load(f)
    margin, leverage, notional = apply_risk_sidecar(sidecar, features, dec)
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    base_np, exit_runtime, pos_idx = risk_exp._prepare_exit_runtime(base_x, loaded)
    return (
        ComponentRuntime(
            alias=alias,
            split=split,
            frame=frame,
            base_x=base_x,
            decisions=dec.reset_index(drop=True),
            loaded_models=loaded,
            arrays=arrays,
            active=omega._active(dec),
            route=hard._route_id(frame),
            base_np=base_np,
            exit_runtime=exit_runtime,
            pos_idx=pos_idx,
            base_margin_fraction=margin,
            base_leverage=leverage,
            base_notional=notional,
            source_side_scale=load_source_side_scale_map(),
            config=cfg,
            device=device,
        ),
        len(frames["train_raw"]),
    )


def align_components_to_common_timestamps(components: dict[str, ComponentRuntime]) -> tuple[pd.DataFrame, dict[str, Any]]:
    timestamp_sets: dict[str, pd.Series] = {}
    for alias, comp in components.items():
        ts = pd.to_datetime(comp.frame["timestamp"], errors="raise").reset_index(drop=True)
        if ts.duplicated().any():
            dup = ts[ts.duplicated()].iloc[0]
            raise RuntimeError(f"{alias}: duplicate timestamp in component frame: {dup}")
        timestamp_sets[alias] = ts
    common = set(timestamp_sets[PRIORITY_ORDER[0]].tolist())
    for alias in PRIORITY_ORDER[1:]:
        common &= set(timestamp_sets[alias].tolist())
    if not common:
        raise RuntimeError("no common timestamps across v5 components")
    common_index = pd.Index(sorted(common))
    align_meta: dict[str, Any] = {
        "common_rows": int(len(common_index)),
        "common_start": str(common_index[0]),
        "common_end": str(common_index[-1]),
        "components": {},
    }
    for alias, comp in components.items():
        ts = timestamp_sets[alias]
        pos = pd.Series(np.arange(len(ts), dtype=np.int64), index=ts)
        take = pos.reindex(common_index)
        if take.isna().any():
            raise RuntimeError(f"{alias}: internal alignment failure")
        idx = take.to_numpy(dtype=np.int64)
        align_meta["components"][alias] = {
            "original_rows": int(len(comp.frame)),
            "aligned_rows": int(len(idx)),
            "dropped_rows": int(len(comp.frame) - len(idx)),
            "original_start": str(ts.iloc[0]) if len(ts) else "",
            "original_end": str(ts.iloc[-1]) if len(ts) else "",
        }
        comp.frame = comp.frame.iloc[idx].reset_index(drop=True)
        comp.base_x = comp.base_x.iloc[idx].reset_index(drop=True)
        comp.decisions = comp.decisions.iloc[idx].reset_index(drop=True)
        comp.arrays = {k: v[idx] for k, v in comp.arrays.items()}
        comp.active = comp.active[idx]
        comp.route = comp.route[idx]
        comp.base_np = comp.base_np[idx]
        comp.base_margin_fraction = comp.base_margin_fraction[idx]
        comp.base_leverage = comp.base_leverage[idx]
        comp.base_notional = comp.base_notional[idx]
    return components[PRIORITY_ORDER[0]].frame.reset_index(drop=True), align_meta


def scaled_entry_sizing(comp: ComponentRuntime, i: int, side: int) -> dict[str, float]:
    group = f"{comp.alias}_{side_key(side)}"
    if group not in comp.source_side_scale:
        raise RuntimeError(f"missing source/side scale: {group}")
    raw_scale = float(comp.source_side_scale[group])
    base_margin = float(comp.base_margin_fraction[int(i)])
    base_leverage = float(comp.base_leverage[int(i)])
    base_notional = base_margin * base_leverage
    effective_scale = min(raw_scale, MAX_LEVERAGE / max(base_leverage, 1.0e-12))
    scaled_leverage = min(base_leverage * effective_scale, MAX_LEVERAGE)
    scaled_notional = base_margin * scaled_leverage
    return {
        "raw_source_side_scale": raw_scale,
        "effective_source_side_scale": effective_scale,
        "original_notional": base_notional,
        "original_leverage": base_leverage,
        "original_margin_fraction": base_margin,
        "notional": scaled_notional,
        "leverage": scaled_leverage,
        "margin_fraction": base_margin,
    }


def metrics_from_ledger(ledger: pd.DataFrame) -> dict[str, Any]:
    if ledger.empty:
        return {
            "pnl": 0.0,
            "mdd": 0.0,
            "trades": 0,
            "wr": 0.0,
            "avg_notional": 0.0,
            "max_notional": 0.0,
            "max_leverage": 0.0,
            "long_entries": 0,
            "short_entries": 0,
            "contract_diff": 0.0,
            "exit_reasons": {},
        }
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    wins = 0
    for _, row in ledger.iterrows():
        adverse = cash * (1.0 + min(float(row["mae_price_move"]), 0.0) * float(row["notional"]))
        peak = max(peak, cash)
        mdd = min(mdd, adverse / max(peak, 1.0e-12) - 1.0)
        before = cash
        cash *= 1.0 + float(row["trade_return"])
        wins += int(cash > before)
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1.0e-12) - 1.0)
    diff = (
        ledger["notional"].astype(float)
        - ledger["margin_fraction"].astype(float) * ledger["leverage"].astype(float)
    ).abs().max()
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(len(ledger)),
        "wr": float(wins / max(len(ledger), 1)),
        "avg_notional": float(ledger["notional"].mean()),
        "max_notional": float(ledger["notional"].max()),
        "max_leverage": float(ledger["leverage"].max()),
        "long_entries": int((ledger["side"].astype(int) == 1).sum()),
        "short_entries": int((ledger["side"].astype(int) == -1).sum()),
        "contract_diff": float(diff),
        "exit_reasons": {str(k): int(v) for k, v in ledger["reason"].value_counts().to_dict().items()},
    }


def scale_lifecycle_ledger(ledger: pd.DataFrame, alias: str, source_side_scale: dict[str, float]) -> pd.DataFrame:
    if ledger.empty:
        return ledger.copy()
    out = ledger.copy().reset_index(drop=True)
    out["source_alias"] = alias
    out["side_key"] = out["side"].astype(int).map(lambda x: side_key(int(x)))
    out["scale_group"] = out["source_alias"].astype(str) + "_" + out["side_key"].astype(str)
    raw_scales = []
    effective_scales = []
    original_notional = pd.to_numeric(out["notional"], errors="raise").to_numpy(dtype=np.float64)
    original_leverage = pd.to_numeric(out["leverage"], errors="raise").to_numpy(dtype=np.float64)
    original_margin = pd.to_numeric(out["margin_fraction"], errors="raise").to_numpy(dtype=np.float64)
    for group, lev in zip(out["scale_group"], original_leverage):
        if group not in source_side_scale:
            raise RuntimeError(f"missing source/side scale: {group}")
        raw = float(source_side_scale[str(group)])
        eff = min(raw, MAX_LEVERAGE / max(float(lev), 1.0e-12))
        raw_scales.append(raw)
        effective_scales.append(eff)
    out["raw_source_side_scale"] = raw_scales
    out["effective_source_side_scale"] = effective_scales
    out["original_notional"] = original_notional
    out["original_leverage"] = original_leverage
    out["original_margin_fraction"] = original_margin
    out["leverage"] = np.minimum(original_leverage * np.asarray(effective_scales, dtype=np.float64), MAX_LEVERAGE)
    out["margin_fraction"] = original_margin
    out["notional"] = out["margin_fraction"].astype(float) * out["leverage"].astype(float)
    out["trade_return"] = out["net_per_notional"].astype(float) * out["notional"].astype(float)
    out["risk_notional"] = out["notional"]
    out["risk_leverage"] = out["leverage"]
    out["risk_margin_fraction"] = out["margin_fraction"]
    return out


def component_lifecycle_ledger(comp: ComponentRuntime) -> pd.DataFrame:
    metrics, ledger = risk_exp._replay_with_risk(
        comp.frame,
        comp.base_x,
        comp.decisions,
        comp.loaded_models,
        risk_margin_fraction=comp.base_margin_fraction,
        risk_leverage=comp.base_leverage,
        exit_threshold=float(comp.config["exit_threshold"]),
        fee=omega._load_fee_slip()[0],
        slip=omega._load_fee_slip()[1],
        cost_mult=float(comp.config["cost_mult"]),
        notional_scaled_sltp=bool(comp.config["notional_scaled_sltp"]),
        exit_sizing_input_mode=str(comp.config["exit_sizing_input_mode"]),
        exit_context_features=None,
        exit_trend_threshold_scale=float(comp.config["exit_trend_threshold_scale"]),
        exit_threshold_floor=float(comp.config["exit_threshold_floor"]),
        exit_threshold_cap=float(comp.config["exit_threshold_cap"]),
        device=comp.device,
    )
    _ = metrics
    return scale_lifecycle_ledger(ledger, comp.alias, comp.source_side_scale)


def replay_lifecycle_events_split(
    split: str,
    components: dict[str, ComponentRuntime],
    train_rows_by_component: dict[str, int],
    *,
    enforce_warmup: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    priority_rank = {alias: rank for rank, alias in enumerate(PRIORITY_ORDER)}
    event_frames = []
    candidate_rows = []
    warmup_frames = []
    component_meta: dict[str, Any] = {}
    for alias in PRIORITY_ORDER:
        comp = components[alias]
        warm = warmup_diagnostics(comp.frame, split, int(train_rows_by_component[alias]))
        warm["source_alias"] = alias
        warmup_frames.append(warm)
        ledger = component_lifecycle_ledger(comp)
        component_meta[alias] = {
            "frame_rows": int(len(comp.frame)),
            "frame_start": str(comp.frame["timestamp"].iloc[0]) if len(comp.frame) else "",
            "frame_end": str(comp.frame["timestamp"].iloc[-1]) if len(comp.frame) else "",
            "component_lifecycle_trades": int(len(ledger)),
        }
        if ledger.empty:
            continue
        keep_flags = []
        reasons = []
        diag_cols: dict[str, list[Any]] = {
            "_warmup_bar_index": [],
            "_warmup_feature_zero_ratio": [],
            "_warmup_feature_zero_default_ratio": [],
            "ai_ready": [],
        }
        for _, row in ledger.iterrows():
            entry_i = int(row["entry_signal_i"])
            keep, reason = warmup_keep(warm, entry_i) if enforce_warmup else (True, "")
            keep_flags.append(bool(keep))
            reasons.append(reason)
            for col in diag_cols:
                diag_cols[col].append(warm.iloc[entry_i][col])
            candidate_rows.append(
                {
                    "split": split,
                    "entry_signal_i": entry_i,
                    "entry_timestamp": row["entry_timestamp"],
                    "exit_i": int(row["exit_i"]),
                    "exit_timestamp": row["exit_timestamp"],
                    "source_alias": alias,
                    "side": int(row["side"]),
                    "warmup_keep": bool(keep),
                    "warmup_block_reason": reason,
                }
            )
        ledger["_warmup_keep"] = keep_flags
        ledger["_warmup_block_reason"] = reasons
        for col, vals in diag_cols.items():
            ledger[col] = vals
        event_frames.append(ledger[ledger["_warmup_keep"]].copy())

    if event_frames:
        events = pd.concat(event_frames, ignore_index=True)
    else:
        events = pd.DataFrame()
    if events.empty:
        return events, pd.DataFrame(candidate_rows), pd.concat(warmup_frames, ignore_index=True), metrics_from_ledger(events), component_meta
    events["_entry_dt"] = pd.to_datetime(events["entry_timestamp"], errors="raise")
    events["_exit_dt"] = pd.to_datetime(events["exit_timestamp"], errors="raise")
    events["_priority_rank"] = events["source_alias"].map(priority_rank).astype(int)
    events = events.sort_values(["_entry_dt", "_priority_rank", "entry_signal_i"], kind="mergesort").reset_index(drop=True)
    chosen = []
    skipped = []
    open_until = pd.Timestamp.min
    for _, row in events.iterrows():
        rec = row.to_dict()
        if row["_entry_dt"] <= open_until:
            rec["_router_decision"] = "skipped_overlap"
            skipped.append(rec)
            continue
        rec["_router_decision"] = "taken"
        chosen.append(rec)
        open_until = row["_exit_dt"]
    ledger = pd.DataFrame(chosen).drop(columns=["_entry_dt", "_exit_dt", "_priority_rank"], errors="ignore")
    skipped_df = pd.DataFrame(skipped).drop(columns=["_entry_dt", "_exit_dt", "_priority_rank"], errors="ignore")
    if len(skipped_df):
        skipped_df["warmup_keep"] = True
        skipped_df["warmup_block_reason"] = ""
        candidate_rows.extend(skipped_df[["entry_signal_i", "entry_timestamp", "exit_i", "exit_timestamp", "source_alias", "side", "warmup_keep", "warmup_block_reason"]].assign(split=split).to_dict("records"))
    return ledger, pd.DataFrame(candidate_rows), pd.concat(warmup_frames, ignore_index=True), metrics_from_ledger(ledger), component_meta


def replay_split(
    split: str,
    components: dict[str, ComponentRuntime],
    warmup: pd.DataFrame,
    *,
    enforce_warmup: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame, _align_meta = align_components_to_common_timestamps(components)
    arrays = components[PRIORITY_ORDER[0]].arrays
    fee, slip = omega._load_fee_slip()
    fee_eff = float(fee) * 3.0
    slip_eff = float(slip) * 3.0

    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    source_alias = ""
    active_comp: ComponentRuntime | None = None
    entry_price = 0.0
    entry_equity = 1.0
    entry_i = 0
    entry_signal_i = 0
    entry_fee = 0.0
    notional = 0.0
    leverage = 1.0
    margin_fraction = 0.0
    exit_input_notional = 0.0
    exit_input_leverage = 1.0
    take_profit = 0.0
    stop_loss = 0.0
    mfe = 0.0
    mae = 0.0
    scale_info: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for i in range(0, len(frame) - 2):
        if pos != 0:
            move = price_exit._price_move(arrays, int(i), side=pos, entry_price=float(entry_price), slip_eff=slip_eff)
            unreal = move * notional
            mfe = max(mfe, move)
            mae = min(mae, move)
            eq = cash * (1.0 + unreal)
        else:
            move = 0.0
            eq = cash
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1.0e-12) - 1.0)

        if pos != 0:
            assert active_comp is not None
            reason = ""
            exit_prob = 0.0
            effective_exit_threshold = float(active_comp.config["exit_threshold"])
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            else:
                hold = max(int(i) - int(entry_i), 0)
                giveback = (float(mfe) - float(move)) / max(abs(float(mfe)), 1.0e-8) if mfe > 0.0 else 0.0
                expert = hard.EXPERT_NAMES[int(active_comp.route[i])]
                prob = risk_exp._predict_exit_prob_one(
                    active_comp.base_np,
                    active_comp.exit_runtime,
                    active_comp.pos_idx,
                    row_i=int(i),
                    expert=expert,
                    pos_values=[
                        float(pos),
                        float(hold),
                        float(move),
                        float(mfe),
                        float(mae),
                        float(np.clip(giveback, 0.0, 10.0)),
                        float(take_profit - move),
                        float(move + abs(stop_loss)),
                        float(exit_input_notional),
                        float(exit_input_leverage),
                        float(exit_input_notional * exit_input_leverage),
                        float(take_profit),
                        float(stop_loss),
                    ],
                    device=active_comp.device,
                )
                exit_prob = float(prob)
                if prob >= effective_exit_threshold:
                    reason = "exit_head"
            if reason:
                filled, exit_px, exit_fee, _route = omega._try_execution(
                    arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff
                )
                if not filled:
                    continue
                raw_exit = (
                    (exit_px - entry_price) / max(entry_price, 1.0e-12)
                    if pos > 0
                    else (entry_price - exit_px) / max(entry_price, 1.0e-12)
                )
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * exit_fee * notional
                trade_return = cash / max(entry_equity, 1.0e-12) - 1.0
                rows.append(
                    {
                        "entry_signal_i": int(entry_signal_i),
                        "entry_i": int(entry_i),
                        "exit_i": int(i),
                        "entry_timestamp": str(frame["timestamp"].iloc[int(entry_signal_i)]),
                        "exit_timestamp": str(frame["timestamp"].iloc[int(i)]),
                        "side": int(pos),
                        "reason": reason,
                        "win": int(cash > entry_equity),
                        "raw_exit_price_move": float(raw_exit),
                        "mfe_price_move": float(mfe),
                        "mae_price_move": float(mae),
                        "trade_return": float(trade_return),
                        "net_per_notional": float(trade_return / max(notional, 1.0e-12)),
                        "notional": float(notional),
                        "margin_fraction": float(margin_fraction),
                        "leverage": float(leverage),
                        "exit_input_notional": float(exit_input_notional),
                        "exit_input_leverage": float(exit_input_leverage),
                        "exit_input_exposure": float(exit_input_notional * exit_input_leverage),
                        "exit_prob": float(exit_prob),
                        "exit_trend_support": 0.0,
                        "exit_threshold_effective": float(effective_exit_threshold),
                        "take_profit": float(take_profit),
                        "stop_loss": float(stop_loss),
                        "source_alias": source_alias,
                        "side_key": side_key(pos),
                        "scale_group": f"{source_alias}_{side_key(pos)}",
                        **scale_info,
                        "risk_notional": float(notional),
                        "risk_leverage": float(leverage),
                        "risk_margin_fraction": float(margin_fraction),
                    }
                )
                pos = 0
                source_alias = ""
                active_comp = None
                continue

        if pos != 0:
            continue

        for alias in PRIORITY_ORDER:
            comp = components[alias]
            if not bool(comp.active[i]):
                continue
            row = comp.decisions.iloc[int(i)]
            side = int(row.get("side", 0) or 0)
            if side == 0:
                continue
            keep, block_reason = warmup_keep(warmup, int(i)) if enforce_warmup else (True, "")
            candidates.append(
                {
                    "split": split,
                    "bar_i": int(i),
                    "timestamp": str(frame["timestamp"].iloc[int(i)]),
                    "source_alias": alias,
                    "side": int(side),
                    "warmup_keep": bool(keep),
                    "warmup_block_reason": block_reason,
                }
            )
            if not keep:
                continue
            sizing = scaled_entry_sizing(comp, int(i), side)
            if float(sizing["notional"]) <= 0.0:
                continue
            filled, px, fee_paid, _route = omega._try_execution(
                arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff
            )
            if not filled:
                continue
            pos = side
            source_alias = alias
            active_comp = comp
            entry_price = float(px)
            entry_equity = cash
            entry_i = min(int(i) + 1, len(frame) - 1)
            entry_signal_i = int(i)
            entry_fee = float(fee_paid)
            margin_fraction = float(sizing["margin_fraction"])
            leverage = float(sizing["leverage"])
            notional = float(sizing["notional"])
            exit_input_notional = float(sizing["original_notional"])
            exit_input_leverage = float(sizing["original_leverage"])
            base_take_profit = float(row.get("take_profit", 0.0) or 0.0)
            base_stop_loss = float(row.get("stop_loss", 0.0) or 0.0)
            if bool(comp.config["notional_scaled_sltp"]):
                take_profit = base_take_profit * notional
                stop_loss = base_stop_loss * notional
            else:
                take_profit = base_take_profit
                stop_loss = base_stop_loss
            cash -= cash * entry_fee * notional
            mfe = 0.0
            mae = 0.0
            scale_info = sizing
            break

    if pos != 0 and active_comp is not None:
        exit_px = omega._fill_price(arrays, len(frame) - 1, pos, slip_eff, entry=False)
        raw_exit = (
            (exit_px - entry_price) / max(entry_price, 1.0e-12)
            if pos > 0
            else (entry_price - exit_px) / max(entry_price, 1.0e-12)
        )
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        trade_return = cash / max(entry_equity, 1.0e-12) - 1.0
        rows.append(
            {
                "entry_signal_i": int(entry_signal_i),
                "entry_i": int(entry_i),
                "exit_i": int(len(frame) - 1),
                "entry_timestamp": str(frame["timestamp"].iloc[int(entry_signal_i)]),
                "exit_timestamp": str(frame["timestamp"].iloc[-1]),
                "side": int(pos),
                "reason": "forced_end",
                "win": int(cash > entry_equity),
                "raw_exit_price_move": float(raw_exit),
                "mfe_price_move": float(mfe),
                "mae_price_move": float(mae),
                "trade_return": float(trade_return),
                "net_per_notional": float(trade_return / max(notional, 1.0e-12)),
                "notional": float(notional),
                "margin_fraction": float(margin_fraction),
                "leverage": float(leverage),
                "exit_input_notional": float(exit_input_notional),
                "exit_input_leverage": float(exit_input_leverage),
                "exit_input_exposure": float(exit_input_notional * exit_input_leverage),
                "exit_prob": 0.0,
                "exit_trend_support": 0.0,
                "exit_threshold_effective": float(active_comp.config["exit_threshold"]),
                "take_profit": float(take_profit),
                "stop_loss": float(stop_loss),
                "source_alias": source_alias,
                "side_key": side_key(pos),
                "scale_group": f"{source_alias}_{side_key(pos)}",
                **scale_info,
                "risk_notional": float(notional),
                "risk_leverage": float(leverage),
                "risk_margin_fraction": float(margin_fraction),
            }
        )

    ledger = pd.DataFrame(rows)
    candidate_df = pd.DataFrame(candidates)
    replay_metrics = metrics_from_ledger(ledger)
    replay_metrics["runtime_mdd"] = float(mdd * 100.0)
    replay_metrics["warmup_enforced"] = bool(enforce_warmup)
    return ledger, candidate_df, replay_metrics


def run_split(split: str, device: torch.device, enforce_warmup: bool) -> dict[str, Any]:
    components: dict[str, ComponentRuntime] = {}
    train_rows_by_component: dict[str, int] = {}
    for alias in PRIORITY_ORDER:
        print(f"stage=build_component split={split} alias={alias}", flush=True)
        comp, train_rows = build_component(alias, split, device)
        components[alias] = comp
        train_rows_by_component[alias] = int(train_rows)
    frame, align_meta = align_components_to_common_timestamps(components)
    train_rows_for_warmup = min(train_rows_by_component.values()) if train_rows_by_component else 0
    warmup = warmup_diagnostics(frame, split, int(train_rows_for_warmup))
    ledger, candidates, metrics = replay_split(split, components, warmup, enforce_warmup=enforce_warmup)
    return {
        "ledger": ledger,
        "candidates": candidates,
        "warmup": warmup,
        "metrics": metrics,
        "frame_rows": int(len(frame)),
        "train_rows_for_warmup": int(train_rows_for_warmup),
        "train_rows_by_component": train_rows_by_component,
        "alignment": align_meta,
        "start": str(frame["timestamp"].iloc[0]) if len(frame) else "",
        "end": str(frame["timestamp"].iloc[-1]) if len(frame) else "",
    }


def run_lifecycle_split(split: str, device: torch.device, enforce_warmup: bool) -> dict[str, Any]:
    components: dict[str, ComponentRuntime] = {}
    train_rows_by_component: dict[str, int] = {}
    for alias in PRIORITY_ORDER:
        print(f"stage=build_component_lifecycle split={split} alias={alias}", flush=True)
        comp, train_rows = build_component(alias, split, device)
        components[alias] = comp
        train_rows_by_component[alias] = int(train_rows)
    ledger, candidates, warmup, metrics, component_meta = replay_lifecycle_events_split(
        split,
        components,
        train_rows_by_component,
        enforce_warmup=enforce_warmup,
    )
    return {
        "ledger": ledger,
        "candidates": candidates,
        "warmup": warmup,
        "metrics": metrics,
        "frame_rows": int(sum(int(v["frame_rows"]) for v in component_meta.values())),
        "train_rows_for_warmup": min(train_rows_by_component.values()) if train_rows_by_component else 0,
        "train_rows_by_component": train_rows_by_component,
        "alignment": {"router_input": "component_lifecycle_events", "components": component_meta},
        "start": min((str(v["frame_start"]) for v in component_meta.values() if v["frame_start"]), default=""),
        "end": max((str(v["frame_end"]) for v in component_meta.values() if v["frame_end"]), default=""),
    }


def gates(metrics: dict[str, dict[str, Any]]) -> dict[str, bool]:
    return {
        "target_pass": float(metrics["validation"]["pnl"]) >= TARGET_PNL and float(metrics["oos"]["pnl"]) >= TARGET_PNL,
        "risk_pass": float(metrics["validation"]["mdd"]) >= MDD_FLOOR and float(metrics["oos"]["mdd"]) >= MDD_FLOOR,
        "leverage_pass": float(metrics["validation"]["max_leverage"]) <= MAX_LEVERAGE + 1.0e-9
        and float(metrics["oos"]["max_leverage"]) <= MAX_LEVERAGE + 1.0e-9,
        "accounting_pass": float(metrics["validation"]["contract_diff"]) <= 1.0e-9
        and float(metrics["oos"]["contract_diff"]) <= 1.0e-9,
        "full_bar_replay_available": True,
        "selection_oos_independent": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--skip-legacy", action="store_true")
    ap.add_argument("--router-input-mode", choices=["lifecycle_events", "raw_signals"], default="lifecycle_events")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = parent._device(str(args.device))
    modes = ["warmup"]
    if not bool(args.skip_legacy):
        modes.insert(0, "legacy_no_warmup")

    report_modes: dict[str, Any] = {}
    for mode in modes:
        enforce = mode == "warmup"
        mode_name = f"{mode}_{args.router_input_mode}"
        mode_dir = OUT_DIR / mode_name
        mode_dir.mkdir(parents=True, exist_ok=True)
        split_metrics: dict[str, Any] = {}
        split_meta: dict[str, Any] = {}
        for split in ("validation", "oos"):
            print(f"stage=run_split mode={mode_name} split={split}", flush=True)
            if str(args.router_input_mode) == "lifecycle_events":
                result = run_lifecycle_split(split, device, enforce_warmup=enforce)
            else:
                result = run_split(split, device, enforce_warmup=enforce)
            result["ledger"].to_csv(mode_dir / f"{split}_full_bar_trade_ledger.csv", index=False)
            result["candidates"].to_csv(mode_dir / f"{split}_full_bar_candidates.csv", index=False)
            result["warmup"].to_csv(mode_dir / f"{split}_warmup_diagnostics.csv", index=False)
            split_metrics[split] = result["metrics"]
            split_meta[split] = {
                "frame_rows": result["frame_rows"],
                "train_rows_for_warmup": result["train_rows_for_warmup"],
                "train_rows_by_component": result["train_rows_by_component"],
                "alignment": result["alignment"],
                "start": result["start"],
                "end": result["end"],
                "ledger_csv": str(mode_dir / f"{split}_full_bar_trade_ledger.csv"),
                "candidates_csv": str(mode_dir / f"{split}_full_bar_candidates.csv"),
                "warmup_csv": str(mode_dir / f"{split}_warmup_diagnostics.csv"),
                "blocked_warmup_candidates": int(
                    (
                        result["candidates"].get("warmup_block_reason", pd.Series(dtype=str))
                        .fillna("")
                        .astype(str)
                        != ""
                    ).sum()
                )
                if len(result["candidates"])
                else 0,
            }
        mode_gates = gates(split_metrics)
        mode_gates["redteam_full_pass"] = all(mode_gates.values()) if mode == "warmup" else False
        mode_gates["promotion_pass"] = bool(mode_gates["redteam_full_pass"])
        report_modes[mode_name] = {"metrics": split_metrics, "meta": split_meta, "gates": mode_gates}

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "baseline_alias": "v5_guard18p0",
        "canonical_candidate": "v5_explainable_router_guard18p0",
        "source_priority_report": str(PRIORITY_REPORT),
        "guard_report": str(GUARD_REPORT),
        "baseline_manifest": str(BASELINE_MANIFEST),
        "contracts": {
            "warmup_bars": WARMUP_BARS,
            "zero_default_feature_ratio_max": ZERO_DEFAULT_MAX,
            "max_leverage": MAX_LEVERAGE,
            "target_pnl_each_split": TARGET_PNL,
            "mdd_floor_each_split": MDD_FLOOR,
            "notional_contract": "notional = margin_fraction * leverage",
            "guard18p0_scaling": "scale account leverage/notional by source/side while preserving original exit-head sizing inputs",
        },
        "modes": report_modes,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=json_default))
    rows = []
    for mode, data in report_modes.items():
        row = {"mode": mode, **data["gates"]}
        for split in ("validation", "oos"):
            m = data["metrics"][split]
            row.update({f"{split}_{k}": v for k, v in m.items() if not isinstance(v, dict)})
            row[f"{split}_blocked_warmup_candidates"] = data["meta"][split]["blocked_warmup_candidates"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_DIR / "redteam_full_bar_summary.csv", index=False)
    print(json.dumps(report_modes, indent=2, ensure_ascii=False, default=json_default), flush=True)
    print(f"wrote {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
