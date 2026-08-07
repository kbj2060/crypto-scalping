#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from features.dsac_pure_rl_kernel import decide_pure_rl_action
from features.m7 import trend_signal_from_m7
from features.schema import STATE_CONF as DSAC_STATE_CONF, STATE_PRED as DSAC_STATE_PRED
from ensemble.train_rl_dsac_unified_controller import (
    DSAC_STATE_DIM as CONTROLLER_STATE_DIM,
    GaussianActor as ControllerActor,
    DSACRouter as ControllerRouter,
    _controller_bucket_from_action,
)
import trading_bot as tb


@dataclass
class Metrics:
    pnl_pct: float
    mdd_pct: float
    trades: int
    wr_pct: float
    long_entries: int
    short_entries: int
    avg_hold_bars: float
    avg_fraction: float
    avg_leverage: float
    avg_exposure: float


def _safe_float(v, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not np.isfinite(x):
        return float(default)
    return x


def _load_frame(csv_path: str, start: str | None, end: str | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if start:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end:
        df = df[df["timestamp"] <= pd.Timestamp(end)]
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"]).reset_index(drop=True)
    defaults = {
        "open": None,
        "high": None,
        "low": None,
        "volume": 1.0,
        "regime_bull": 0.0,
        "regime_bear": 0.0,
        "regime_chop": 0.0,
        "regime_whipsaw": 0.0,
        "regime_normal": 0.0,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = df["close"] if default is None else default
    for col in ("open", "high", "low"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df["close"])
    return df


def _build_controller_router(ckpt_path: str) -> ControllerRouter:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    actor = ControllerActor(state_dim=CONTROLLER_STATE_DIM).to("cpu")
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return ControllerRouter(actor, device="cpu", hmm_detector=None, mtf_features=None)


def _controller_bucket_relaxed(raw_action: float) -> tuple[float, str]:
    x = float(np.clip(raw_action, -1.0, 1.0))
    ax = abs(x)
    if ax < 0.10:
        bucket = 0.0
    elif ax < 0.22:
        bucket = 0.5
    elif ax < 0.34:
        bucket = 0.75
    elif ax < 0.48:
        bucket = 1.0
    elif ax < 0.62:
        bucket = 1.25
    elif ax < 0.78:
        bucket = 1.5
    else:
        bucket = 2.0
    return float(bucket), f"ctrl_relaxed_{bucket:.2f}x"


def _regime_name_from_dict(regime: dict | None, row: pd.Series) -> str:
    if isinstance(regime, dict):
        for k, v in regime.items():
            if str(k).startswith("regime_") and float(v) == 1.0:
                return str(k).replace("regime_", "")
    for name in ("bull", "bear", "chop", "whipsaw", "normal"):
        if _safe_float(row.get(f"regime_{name}", 0.0), 0.0) >= 0.5:
            return name
    return "normal"


def simulate(df: pd.DataFrame, ckpt_path: str, *, mode: str, disable_same_side_resize: bool = False) -> dict:
    with tempfile.TemporaryDirectory(prefix="tb_native_bt_2026_") as tmpdir:
        os.environ["DSAC_LIVE_STATE_PATH"] = os.path.join(tmpdir, "live_state.json")
        os.environ["DSAC_COMPACT_LIVE_STATE_PATH"] = os.path.join(tmpdir, "compact_live_state.json")

        meta_router = tb.DSACTrendRouter()
        meta_router.live_state_path = os.path.join(tmpdir, "meta_router_live_state.json")
        tb._reset_virtual_router_state(meta_router)
        meta_router._save_live_state()

        dsac_router = tb.DSACSignalRouter(
            model_path=ckpt_path if mode == "base" else os.path.join(_ROOT_DIR, "data/ensemble/ckpt/best_dsac_agents.pth")
        )
        compact_router = None
        controller_router = None
        if mode == "compact":
            compact_router = tb.DSACSignalRouter(model_path=ckpt_path)
        elif mode in {"controller", "controller_relaxed", "controller_stable"}:
            compact_router = tb.DSACSignalRouter(model_path=os.path.join(_ROOT_DIR, "data/ensemble/ckpt/best_dsac_unified.pth"))
            controller_router = _build_controller_router(ckpt_path)

        def _sync_router(router: tb.DSACSignalRouter) -> None:
            router.pos = meta_router.pos
            router.entry_price = meta_router.entry_price
            router.hold_count = meta_router.hold_count
            router._set_runtime_sizing(
                fraction=meta_router.position_fraction,
                leverage_mult=meta_router.execution_leverage,
            )
            router.current_equity = meta_router.cur_equity
            router.peak_equity = meta_router.peak_equity
            router.adaptive_enter_offset = meta_router.adaptive_enter_offset
            router.adaptive_agreement_offset = meta_router.adaptive_agreement_offset

        balance = 1.0
        eq_curve = [balance]
        trades = wins = 0
        long_entries = short_entries = 0
        hold_bars: list[int] = []
        fractions: list[float] = []
        leverages: list[float] = []
        exposures: list[float] = []
        regime_pnl: dict[str, list[float]] = defaultdict(list)
        trade_rows: list[dict] = []
        controller_bucket_counts: dict[str, int] = defaultdict(int)
        exec_leverage_counts: dict[str, int] = defaultdict(int)
        exposure_band_counts: dict[str, int] = defaultdict(int)
        controller_trade_bucket_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl_sum": 0.0})
        lots: list[dict] = []

        def _lot_side() -> str | None:
            if not lots:
                return None
            return str(lots[0].get("side") or "").upper() or None

        def _lot_total_exposure() -> float:
            return float(sum(float(x.get("exposure", 0.0) or 0.0) for x in lots))

        def _lot_weighted_entry() -> float:
            total = _lot_total_exposure()
            if total <= 1e-12:
                return 0.0
            return float(sum(float(x.get("entry_price", 0.0) or 0.0) * float(x.get("exposure", 0.0) or 0.0) for x in lots) / total)

        def _lot_mtm_pnl_frac(price: float) -> float:
            total = 0.0
            for lot in lots:
                total += float(meta_router._trade_math(str(lot.get("side", "")), float(lot.get("entry_price", 0.0) or 0.0), float(price), float(lot.get("exposure", 0.0) or 0.0)).get("pnl_frac", 0.0))
            return float(total)

        def _sync_meta_router_from_lots(current_price: float) -> None:
            side = _lot_side()
            total = _lot_total_exposure()
            if side is None or total <= 1e-12:
                meta_router.pos = None
                meta_router.entry_price = 0.0
                meta_router._set_position_sizing(exposure=0.0)
                meta_router.cur_equity = 1.0
                meta_router.peak_equity = 1.0
                return
            meta_router.pos = side
            meta_router.entry_price = _lot_weighted_entry()
            frac = float(np.clip(min(total, 1.0), 0.0, 1.0))
            lev_mult = float(np.clip(total / max(frac, 1e-8), 1.0, meta_router.exposure_cap)) if frac > 0.0 else 1.0
            meta_router._set_position_sizing(fraction=frac, leverage_mult=lev_mult)
            meta_router.cur_equity = 1.0 + _lot_mtm_pnl_frac(current_price)
            meta_router.peak_equity = max(float(meta_router.peak_equity or 1.0), float(meta_router.cur_equity or 1.0))

        def _close_lots_partial(close_exposure: float, price: float) -> float:
            remaining = float(max(close_exposure, 0.0))
            realized = 0.0
            while remaining > 1e-12 and lots:
                lot = lots[0]
                lot_exp = float(lot.get("exposure", 0.0) or 0.0)
                take = min(lot_exp, remaining)
                realized += float(
                    meta_router._trade_math(
                        str(lot.get("side", "")),
                        float(lot.get("entry_price", 0.0) or 0.0),
                        float(price),
                        float(take),
                    ).get("pnl_frac", 0.0)
                )
                lot_exp -= take
                remaining -= take
                if lot_exp <= 1e-12:
                    lots.pop(0)
                else:
                    lot["exposure"] = float(lot_exp)
            return float(realized)

        for i in range(60, len(df)):
            processed_df = df.iloc[max(0, i - 300) : i + 1].copy()
            last_row = processed_df.iloc[-1]
            current_price = float(last_row["close"])
            current_time_kst = pd.Timestamp(last_row["timestamp"])

            _sync_meta_router_from_lots(current_price)

            _sync_router(dsac_router)
            if compact_router is not None:
                _sync_router(compact_router)

            row_dict = last_row.to_dict()
            row_dict.setdefault("m7_prob_dn", _safe_float(row_dict.get("prob_dn", row_dict.get("m7_trend_xgb_dn", 0.0))))
            row_dict.setdefault("m7_prob_up", _safe_float(row_dict.get("prob_up", row_dict.get("m7_trend_xgb_up", 0.0))))
            nf_preds = dict(row_dict)
            pred_fallback = _safe_float(nf_preds.get("pred_patchtst", 0.0))
            conf_fallback = float(np.clip(_safe_float(nf_preds.get("conf_patchtst", 0.5), 0.5), 0.0, 1.0))
            for c in DSAC_STATE_PRED:
                nf_preds.setdefault(c, pred_fallback)
            for c in DSAC_STATE_CONF:
                nf_preds.setdefault(c, conf_fallback)

            trend_signal = trend_signal_from_m7(row_dict)

            prev_pos = meta_router.pos
            prev_hold = meta_router.hold_count
            prev_entry = float(meta_router.entry_price)
            prev_exposure = float(meta_router.current_leverage or 0.0)
            prev_fraction = float(meta_router.position_fraction or 0.0)
            prev_exec_lev = float(meta_router.execution_leverage or 1.0)
            current_side = _lot_side()
            current_total_before = _lot_total_exposure() if current_side is not None else 0.0

            if mode == "compact":
                assert compact_router is not None
                compact_action, compact_lev, compact_info, _, compact_regime = compact_router.decide(
                    processed_df, nf_preds, m7_signal=trend_signal
                )
                base_action, base_lev, base_info, _, _ = dsac_router.decide(
                    processed_df, nf_preds, m7_signal=trend_signal
                )
                compact_action, compact_lev, compact_info = tb.refine_compact_native_decision(
                    compact_action=int(compact_action),
                    compact_lev=float(compact_lev),
                    compact_info=compact_info,
                    base_action=int(base_action),
                    base_lev=float(base_lev),
                    base_info=base_info,
                    processed_df=processed_df,
                    trend_signal=trend_signal,
                    current_pos=meta_router.pos,
                    exposure_cap=float(meta_router.exposure_cap),
                )
                regime_name = _regime_name_from_dict(compact_regime, last_row).upper()
                final_action = int(compact_action)
                target_exposure = float(np.clip(compact_lev, 0.0, meta_router.exposure_cap))
                target_fraction, target_exec_leverage = tb._decode_exposure_bucket(
                    target_exposure, cap=meta_router.exposure_cap
                )
                source = "DSAC_COMPACT|UNIFIED_NATIVE|" + str((compact_info or {}).get("overlay_tag", "raw"))
                applied_bucket_tag = "compact_native"
            elif mode in {"controller", "controller_relaxed", "controller_stable"}:
                assert compact_router is not None and controller_router is not None
                base_action, base_lev, base_info, _, _ = dsac_router.decide(
                    processed_df, nf_preds, m7_signal=trend_signal
                )
                compact_action, compact_lev, compact_info, _, compact_regime = compact_router.decide(
                    processed_df, nf_preds, m7_signal=trend_signal
                )

                current_unr = float(meta_router._net_pnl_frac(current_price)) if meta_router.pos in {"LONG", "SHORT"} else 0.0
                current_eq = float(balance * (1.0 + current_unr)) if meta_router.pos in {"LONG", "SHORT"} else float(balance)
                peak_eq = float(max(current_eq, max(eq_curve) if eq_curve else 1.0))
                raw_drawdown = float(min((current_eq / max(peak_eq, 1e-8)) - 1.0, 0.0))
                effective_hold_count = int(meta_router.hold_count + 1) if meta_router.pos is not None else 0
                pos_dict = {
                    "type": meta_router.pos,
                    "entry_price": float(meta_router.entry_price or 0.0),
                    "unrealized": float(current_unr),
                    "mdd": raw_drawdown,
                    "hold_count": float(effective_hold_count),
                    "hold_norm": min(effective_hold_count / 96.0, 1.0),
                    "margin_usage": float(np.clip(meta_router.position_fraction if meta_router.pos is not None else 0.0, 0.0, 1.0)),
                }

                ctrl_state = controller_router._state_tensor(row_dict, pos_dict)
                with torch.no_grad():
                    ctrl_raw = float(controller_router.actor.deterministic(ctrl_state).cpu().item())
                if mode == "controller_relaxed":
                    bucket_mult, bucket_tag = _controller_bucket_relaxed(ctrl_raw)
                else:
                    bucket_mult, bucket_tag = _controller_bucket_from_action(ctrl_raw)

                unified_action = int(compact_action)
                unified_kelly = float(np.clip(compact_lev, 0.0, 1.0))
                base_veto = bool(int(base_action) in (1, 2) and int(compact_action) in (1, 2) and int(base_action) != int(compact_action) and float(base_lev) >= 0.20)

                target_exposure = 0.0
                if unified_action in (1, 2):
                    target_exposure = unified_kelly * float(bucket_mult)
                if base_veto and float(bucket_mult) > 1.0:
                    target_exposure = unified_kelly * 1.0
                if int(base_action) == 0 and float(bucket_mult) > 2.0:
                    target_exposure = unified_kelly * 1.5
                if meta_router.pos is not None and current_unr < 0.0:
                    target_exposure = min(target_exposure, max(unified_kelly, 0.75))
                if mode == "controller_relaxed":
                    if float(bucket_mult) >= 1.25 and target_exposure > 1e-8:
                        target_exposure = max(target_exposure, min(1.0, unified_kelly + 0.18))
                    if float(bucket_mult) >= 1.5 and target_exposure > 1e-8:
                        target_exposure = max(target_exposure, min(1.25, unified_kelly + 0.30))
                    if float(bucket_mult) >= 2.0 and target_exposure > 1e-8:
                        target_exposure = max(target_exposure, min(1.50, unified_kelly + 0.45))
                elif mode == "controller_stable":
                    current_exposure = float(np.clip(meta_router.current_leverage or 0.0, 0.0, meta_router.exposure_cap))
                    same_side = (
                        (meta_router.pos == "LONG" and unified_action == 1)
                        or (meta_router.pos == "SHORT" and unified_action == 2)
                    )
                    hard_flip = (
                        (meta_router.pos == "LONG" and unified_action == 2)
                        or (meta_router.pos == "SHORT" and unified_action == 1)
                    )
                    # Suppress tiny fresh entries that create churn.
                    if meta_router.pos is None and target_exposure < 0.22:
                        target_exposure = 0.0
                    # Keep same-side positions stable unless controller wants a materially different size.
                    if same_side and abs(target_exposure - current_exposure) < 0.18:
                        target_exposure = current_exposure
                    # Hold existing position a bit longer unless the model is explicitly flipping or fully flat.
                    if meta_router.pos is not None and not hard_flip and meta_router.hold_count < 3:
                        if target_exposure <= 1e-8:
                            target_exposure = current_exposure
                        elif target_exposure < current_exposure * 0.75:
                            target_exposure = current_exposure
                    # Do not shrink to very small residual exposures; either keep meaningful size or exit.
                    if 0.0 < target_exposure < 0.30:
                        target_exposure = 0.30

                target_exposure = float(np.clip(target_exposure, 0.0, meta_router.exposure_cap))
                final_action = int(unified_action if target_exposure > 1e-8 else 0)
                target_fraction = float(np.clip(min(target_exposure, 1.0), 0.0, 1.0))
                target_exec_leverage = (
                    float(np.clip(target_exposure / max(target_fraction, 1e-8), 1.0, meta_router.exposure_cap))
                    if target_fraction > 0.0 else 1.0
                )
                regime_name = _regime_name_from_dict(compact_regime, last_row).upper()
                source = "DSAC_CONTROLLER|UNIFIED_BUCKET|" + str(bucket_tag)
                controller_bucket_counts[str(bucket_tag)] += 1
                applied_bucket_tag = str(bucket_tag)
            else:
                dsac_action, dsac_lev, info, _, regime = dsac_router.decide(
                    processed_df, nf_preds, m7_signal=trend_signal
                )
                regime_name = _regime_name_from_dict(regime, last_row).upper()
                action_val = float(info.get("primary_raw", info.get("raw_action", 0.0)))
                pure_rl = decide_pure_rl_action(
                    action_val=action_val,
                    current_pos=meta_router.pos,
                    live_unrealized_pnl=float(meta_router._net_pnl_frac(current_price)) if meta_router.pos in {"LONG", "SHORT"} else 0.0,
                    alpha_focus_enabled=False,
                    alpha_focus_row=row_dict,
                    alpha_focus_regime=regime_name.lower(),
                    alpha_focus_sizing_fn=None,
                    alpha_focus_exposure_cap=float(meta_router.exposure_cap),
                    oos_parity_mode=False,
                    dsac_action=int(dsac_action),
                    dsac_lev=float(dsac_lev),
                    source_pure="DSAC_PURE_RL",
                )
                final_action = int(pure_rl.final_action)
                target_exposure = float(np.clip(pure_rl.kelly, 0.0, meta_router.exposure_cap))
                target_fraction = float(np.clip(target_exposure, 0.0, 1.0))
                target_exec_leverage = 1.0
                source = str(pure_rl.source)
                applied_bucket_tag = "base_native"

            if final_action == 0 or target_exposure <= 0.0:
                target_fraction = 0.0
                target_exec_leverage = 1.0
                target_exposure = 0.0

            desired_pos = "LONG" if final_action == 1 and target_exposure > 0.0 else ("SHORT" if final_action == 2 and target_exposure > 0.0 else None)
            if disable_same_side_resize and current_side is not None and desired_pos == current_side:
                target_exposure = float(current_total_before)
                target_fraction = float(np.clip(min(target_exposure, 1.0), 0.0, 1.0))
                target_exec_leverage = (
                    float(np.clip(target_exposure / max(target_fraction, 1e-8), 1.0, meta_router.exposure_cap))
                    if target_fraction > 0.0 else 1.0
                )

            if prev_pos is not None:
                eq_curve.append(balance * (1.0 + _lot_mtm_pnl_frac(current_price)))
            else:
                eq_curve.append(balance)

            meta_router._update_pos(
                final_action,
                current_price,
                timestamp_kst=current_time_kst,
                leverage=target_exposure,
                fraction=target_fraction,
                leverage_mult=target_exec_leverage,
                trend_signal=trend_signal,
            )
            meta_router.update_adaptive_gate(final_action=int(final_action), in_position=(meta_router.pos is not None))
            desired_pos = "LONG" if final_action == 1 and target_exposure > 0.0 else ("SHORT" if final_action == 2 and target_exposure > 0.0 else None)
            current_side = _lot_side()
            resize_realized = 0.0

            if current_side is not None and desired_pos != current_side:
                resize_realized += _close_lots_partial(_lot_total_exposure(), current_price)
            if resize_realized != 0.0:
                balance *= (1.0 + resize_realized)

            if desired_pos is not None:
                current_total = _lot_total_exposure() if _lot_side() == desired_pos else 0.0
                delta_exposure = float(target_exposure - current_total)
                if delta_exposure > 1e-12:
                    lots.append({
                        "side": desired_pos,
                        "entry_price": float(current_price),
                        "exposure": float(delta_exposure),
                        "opened_at": str(current_time_kst),
                    })
                    if current_side == desired_pos and prev_pos == desired_pos:
                        trade_rows.append({
                            "ts": str(current_time_kst),
                            "side": desired_pos,
                            "entry_price": float(current_price),
                            "exit_price": current_price,
                            "hold_bars": int(prev_hold),
                            "prev_exposure": prev_exposure,
                            "new_exposure": float(target_exposure),
                            "delta_exposure": float(delta_exposure),
                            "prev_fraction": prev_fraction,
                            "new_fraction": float(np.clip(min(target_exposure, 1.0), 0.0, 1.0)),
                            "prev_exec_leverage": prev_exec_lev,
                            "new_exec_leverage": float(np.clip(target_exposure / max(min(target_exposure, 1.0), 1e-8), 1.0, meta_router.exposure_cap)) if target_exposure > 0.0 else 1.0,
                            "pnl_frac": 0.0,
                            "source": source,
                            "event": "resize_add",
                        })
                elif delta_exposure < -1e-12 and _lot_side() == desired_pos:
                    reduced = abs(delta_exposure)
                    partial_realized = _close_lots_partial(reduced, current_price)
                    balance *= (1.0 + partial_realized)
                    trade_rows.append({
                        "ts": str(current_time_kst),
                        "side": desired_pos,
                        "entry_price": prev_entry,
                        "exit_price": current_price,
                        "hold_bars": int(prev_hold),
                        "prev_exposure": prev_exposure,
                        "new_exposure": float(target_exposure),
                        "delta_exposure": float(delta_exposure),
                        "prev_fraction": prev_fraction,
                        "new_fraction": float(np.clip(min(target_exposure, 1.0), 0.0, 1.0)),
                        "prev_exec_leverage": prev_exec_lev,
                        "new_exec_leverage": float(np.clip(target_exposure / max(min(target_exposure, 1.0), 1e-8), 1.0, meta_router.exposure_cap)) if target_exposure > 0.0 else 1.0,
                        "pnl_frac": float(partial_realized),
                        "source": source,
                        "event": "resize_reduce",
                    })

            _sync_meta_router_from_lots(current_price)
            new_pos = meta_router.pos
            new_exposure = float(meta_router.current_leverage or 0.0)
            new_fraction = float(meta_router.position_fraction or 0.0)
            new_exec_lev = float(meta_router.execution_leverage or 1.0)

            if prev_pos is None and meta_router.pos is not None:
                if meta_router.pos == "LONG":
                    long_entries += 1
                else:
                    short_entries += 1
                fractions.append(float(meta_router.position_fraction))
                leverages.append(float(meta_router.execution_leverage))
                exposures.append(float(meta_router.current_leverage))
                exec_leverage_counts[f"{float(meta_router.execution_leverage):.2f}x"] += 1
                exposure_band_counts[f"{round(float(meta_router.current_leverage), 2):.2f}x"] += 1
            elif prev_pos is not None and meta_router.pos != prev_pos:
                realized = float(resize_realized)
                trades += 1
                wins += int(realized > 0.0)
                hold_bars.append(prev_hold)
                regime_pnl[regime_name].append(realized)
                trade_rows.append({
                    "ts": str(current_time_kst),
                    "side": prev_pos,
                    "entry_price": prev_entry,
                    "exit_price": current_price,
                    "hold_bars": int(prev_hold),
                    "exposure": prev_exposure,
                    "pnl_frac": realized,
                    "source": source,
                    "event": "flip" if meta_router.pos is not None else "exit",
                })
                st = controller_trade_bucket_stats[applied_bucket_tag]
                st["trades"] += 1
                st["wins"] += int(realized > 0.0)
                st["pnl_sum"] += float(realized) * 100.0
                if meta_router.pos is not None:
                    if meta_router.pos == "LONG":
                        long_entries += 1
                    else:
                        short_entries += 1
                    fractions.append(float(meta_router.position_fraction))
                    leverages.append(float(meta_router.execution_leverage))
                    exposures.append(float(meta_router.current_leverage))
                    exec_leverage_counts[f"{float(meta_router.execution_leverage):.2f}x"] += 1
                    exposure_band_counts[f"{round(float(meta_router.current_leverage), 2):.2f}x"] += 1

        if meta_router.pos is not None:
            final_price = float(df.iloc[-1]["close"])
            realized = float(_close_lots_partial(_lot_total_exposure(), final_price))
            balance *= (1.0 + realized)
            trades += 1
            wins += int(realized > 0.0)
            hold_bars.append(meta_router.hold_count)
            regime_pnl[_regime_name_from_dict(None, df.iloc[-1])].append(realized)
            trade_rows.append({
                "ts": str(df.iloc[-1]["timestamp"]),
                "side": meta_router.pos,
                "entry_price": float(meta_router.entry_price),
                "exit_price": final_price,
                "hold_bars": int(meta_router.hold_count),
                "exposure": float(meta_router.current_leverage),
                "pnl_frac": realized,
                "source": "final_mark",
                "event": "final_mark",
            })
            st = controller_trade_bucket_stats[applied_bucket_tag]
            st["trades"] += 1
            st["wins"] += int(realized > 0.0)
            st["pnl_sum"] += float(realized) * 100.0
            eq_curve.append(balance)

        eq = np.asarray(eq_curve, dtype=float)
        peak = np.maximum.accumulate(eq)
        drawdown = (eq / np.maximum(peak, 1e-12)) - 1.0
        metrics = Metrics(
            pnl_pct=float((balance - 1.0) * 100.0),
            mdd_pct=float(drawdown.min() * 100.0),
            trades=int(trades),
            wr_pct=float((wins / trades) * 100.0 if trades else 0.0),
            long_entries=int(long_entries),
            short_entries=int(short_entries),
            avg_hold_bars=float(np.mean(hold_bars) if hold_bars else 0.0),
            avg_fraction=float(np.mean(fractions) if fractions else 0.0),
            avg_leverage=float(np.mean(leverages) if leverages else 0.0),
            avg_exposure=float(np.mean(exposures) if exposures else 0.0),
        )
        regime_summary = {
            k: {
                "trades": int(len(v)),
                "pnl_pct_sum": float(sum(v) * 100.0),
                "avg_pnl_pct": float(np.mean(v) * 100.0 if v else 0.0),
            }
            for k, v in sorted(regime_pnl.items())
        }
        return {
            "metrics": asdict(metrics),
            "regime_summary": regime_summary,
            "controller_bucket_counts": {k: int(v) for k, v in sorted(controller_bucket_counts.items())},
            "controller_trade_bucket_stats": {
                k: {
                    "trades": int(v["trades"]),
                    "wr_pct": float((100.0 * v["wins"] / v["trades"]) if v["trades"] else 0.0),
                    "pnl_pct_sum": float(v["pnl_sum"]),
                    "avg_pnl_pct": float(v["pnl_sum"] / v["trades"]) if v["trades"] else 0.0,
                }
                for k, v in sorted(controller_trade_bucket_stats.items())
            },
            "execution_leverage_counts": {k: int(v) for k, v in sorted(exec_leverage_counts.items())},
            "total_exposure_counts": {k: int(v) for k, v in sorted(exposure_band_counts.items())},
            "trade_rows_tail": trade_rows[-20:],
            "equity_final": float(balance),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "mode": mode,
            "disable_same_side_resize": bool(disable_same_side_resize),
            "ckpt_path": ckpt_path,
            "script": os.path.abspath(__file__),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", default="data/rl_training_2026_unified.csv")
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--mode", choices=["base", "compact", "controller", "controller_relaxed", "controller_stable"], required=True)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--report-path", default="")
    ap.add_argument("--disable-same-side-resize", action="store_true")
    args = ap.parse_args()

    df = _load_frame(args.csv_path, args.start, args.end)
    report = simulate(df, args.ckpt_path, mode=args.mode, disable_same_side_resize=bool(args.disable_same_side_resize))
    report_path = args.report_path or os.path.join(
        _ROOT_DIR, "data", "ensemble", "reports", f"backtest_trading_bot_native_2026_{args.mode}.json"
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"report_path={report_path}")


if __name__ == "__main__":
    main()
