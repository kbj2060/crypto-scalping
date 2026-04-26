#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from features.dsac_pure_rl_kernel import decide_pure_rl_action


@dataclass
class Metrics:
    pnl_pct: float
    mdd_pct: float
    trades: int
    wr_pct: float
    long_entries: int
    short_entries: int
    avg_hold_bars: float
    avg_lev: float


def _safe_float(v, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not np.isfinite(x):
        return float(default)
    return x


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _alpha_focus_strict_sizing(
    action: int,
    kelly: float,
    last_row,
    regime_name: str,
    *,
    exposure_cap: float = 3.0,
) -> dict:
    target_exposure = float(np.clip(float(kelly), 0.0, max(float(exposure_cap), 1.0)))
    if int(action) not in (1, 2) or target_exposure <= 0.0:
        return {
            "target_fraction": float(np.clip(target_exposure, 0.0, 1.0)),
            "target_exposure": target_exposure,
            "leverage_mult": 1.0,
            "flow_score": 0.0,
            "tag": "neutral",
        }
    conf = float(np.clip(float(last_row.get("m7_confidence", 0.0) or 0.0), 0.0, 1.0))
    qwidth = abs(float(last_row.get("m7_qwidth", 0.0) or 0.0))
    vol_z = abs(float(last_row.get("volatility_z", 0.0) or 0.0))
    smf = float(last_row.get("smart_money_flow", 0.0) or 0.0)
    whale = float(last_row.get("whale_conviction", 0.0) or 0.0)
    funding_div = float(last_row.get("funding_price_divergence", 0.0) or 0.0)
    side_sign = 1.0 if int(action) == 1 else -1.0
    aligned = (int(action) == 1 and regime_name == "bull") or (int(action) == 2 and regime_name == "bear")
    bad = regime_name in {"chop", "whipsaw"} or qwidth > 0.010 or vol_z > 1.6 or conf < 0.40
    very_bad = regime_name == "whipsaw" or qwidth > 0.014 or vol_z > 2.4
    flow_score = side_sign * (0.55 * smf + 0.35 * whale + 0.10 * funding_div)

    mult = 1.0
    tag = "base"
    if very_bad:
        mult *= 0.992
        tag = "very_bad"
    elif bad:
        mult *= 0.998
        tag = "bad"

    if flow_score > 0.07 and conf > 0.66 and target_exposure > 0.88 and qwidth < 0.010:
        mult *= 1.12
        tag = f"{tag}_boost12"
    elif flow_score > 0.04 and aligned and target_exposure > 0.78:
        mult *= 1.05
        tag = f"{tag}_boost05"

    target_exposure = float(np.clip(target_exposure * mult, 0.05, 1.0))
    return {
        "target_fraction": float(np.clip(target_exposure, 0.0, 1.0)),
        "target_exposure": target_exposure,
        "leverage_mult": 1.0,
        "flow_score": float(flow_score),
        "tag": str(tag),
    }


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
        "sig_whale": 0.0,
        "nif_whale": 0.0,
        "shadow_toxicity_score": 0.0,
        "shadow_queue_collapse": 0.0,
        "obi": 0.0,
        "taker_buy_ratio": 0.5,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = df["close"] if default is None else default
    for col in ("open", "high", "low"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df["close"])
    return df


def _adverse_score(
    processed_df: pd.DataFrame,
    row: pd.Series,
    action: int,
    current_price: float,
    limit_price: float,
) -> float:
    prev_close = float(processed_df.iloc[-2]["close"]) if len(processed_df) >= 2 else float(current_price)
    ret_1 = (float(current_price) - prev_close) / max(abs(prev_close), 1e-8)
    toxicity = float(row.get("shadow_toxicity_score", 0.0) or 0.0)
    queue_collapse = float(row.get("shadow_queue_collapse", 0.0) or 0.0)
    qwidth = float(row.get("m7_qwidth", 0.0) or 0.0)
    trade_intensity = float(row.get("trade_intensity", 0.0) or 0.0)
    limit_gap = abs(float(limit_price) - float(current_price)) / max(abs(float(current_price)), 1e-8)
    if action == 1:
        adverse_move = max(0.0, -ret_1)
    else:
        adverse_move = max(0.0, ret_1)
    score = (
        0.34 * min(toxicity, 1.5)
        + 0.24 * min(queue_collapse, 1.5)
        + 0.18 * min(adverse_move / 0.0035, 1.5)
        + 0.14 * min(qwidth / 0.012, 1.5)
        + 0.10 * min(trade_intensity / 2.0, 1.5)
        + 0.12 * min(limit_gap / 0.006, 1.5)
    )
    return float(score)


def simulate(
    df: pd.DataFrame,
    ckpt_path: str,
    hybrid_execution: bool,
    fill_mode: str = "next_close",
    realistic_limit_fill: bool = False,
    limit_ttl_bars: int = 2,
    adverse_filter: bool = False,
    adverse_place_th: float = 0.78,
    adverse_cancel_th: float = 0.95,
) -> tuple[Metrics, dict]:
    with tempfile.TemporaryDirectory(prefix="replay_pure_rl_") as tmpdir:
        os.environ["DSAC_LIVE_STATE_PATH"] = os.path.join(tmpdir, "live_state.json")
        os.environ["FUSE_ADAPT_STATE_PATH"] = os.path.join(tmpdir, "adapt_state.json")

        from features.m7 import trend_signal_from_m7
        from features.schema import STATE_CONF as DSAC_STATE_CONF, STATE_PRED as DSAC_STATE_PRED
        from trading_bot import DSACSignalRouter, DSACTrendRouter, _hybrid_execution_overlay

        dsac_router = DSACSignalRouter(model_path=ckpt_path)
        meta_router = DSACTrendRouter()
        meta_router.online_adapt = False
        meta_router._save_live_state = lambda *args, **kwargs: None
        oos_parity_mode = str(os.getenv("DSAC_OOS_PARITY_MODE", "false")).strip().lower() in {"1", "true", "yes", "on"}
        alpha_focus_strict_enable = _env_flag("DSAC_ALPHA_FOCUS_STRICT_ENABLE", True)

        def _sync_dsac_with_meta() -> None:
            dsac_router.pos = meta_router.pos
            dsac_router.entry_price = meta_router.entry_price
            dsac_router.hold_count = meta_router.hold_count
            dsac_router.current_leverage = meta_router.current_leverage
            dsac_router.current_equity = meta_router.cur_equity
            dsac_router.peak_equity = meta_router.peak_equity

        balance = 1.0
        eq_curve = [balance]
        trades = wins = 0
        long_entries = short_entries = 0
        hold_bars: list[int] = []
        levs: list[float] = []
        trade_rows: list[dict] = []
        hybrid_counter: Counter[str] = Counter()
        pending_counter: Counter[str] = Counter()
        pending_order: dict | None = None
        current_entry_kind: str = "market"
        current_entry_display_price: float = 0.0

        for i in tqdm(range(60, len(df) - 1), desc="replay-pure-rl", unit="bar"):
            processed_df = df.iloc[max(0, i - 300) : i + 1].copy()
            last_row = processed_df.iloc[-1]
            current_price = float(last_row["close"])
            next_row = df.iloc[i + 1]
            next_open = float(next_row.get("open", next_row["close"]) or next_row["close"])
            next_close = float(next_row["close"])
            if fill_mode == "current":
                fill_price = float(current_price)
            elif fill_mode == "next_open":
                fill_price = float(next_open)
            else:
                fill_price = float(next_close)
            current_time_kst = pd.Timestamp(last_row["timestamp"])

            _sync_dsac_with_meta()

            if hybrid_execution and realistic_limit_fill and meta_router.pos is None and pending_order is not None:
                bar_open = float(last_row.get("open", current_price) or current_price)
                bar_high = float(last_row.get("high", current_price) or current_price)
                bar_low = float(last_row.get("low", current_price) or current_price)
                limit_price = float(pending_order.get("limit_price", current_price) or current_price)
                pending_action = int(pending_order.get("action", 0) or 0)
                if adverse_filter:
                    live_adverse = _adverse_score(processed_df, last_row, pending_action, current_price, limit_price)
                    if live_adverse >= float(adverse_cancel_th):
                        pending_counter["adverse_cancel"] += 1
                        pending_counter[f"adverse_cancel_{pending_order.get('mode', 'unknown')}"] += 1
                        pending_order = None
                        live_adverse = None
                filled_price: float | None = None
                if pending_order is not None:
                    if pending_action == 1:
                        if bar_open <= limit_price:
                            filled_price = bar_open
                        elif bar_low <= limit_price <= bar_high:
                            filled_price = limit_price
                    elif pending_action == 2:
                        if bar_open >= limit_price:
                            filled_price = bar_open
                        elif bar_low <= limit_price <= bar_high:
                            filled_price = limit_price
                if pending_order is not None and filled_price is not None:
                    fill_signal = dict(pending_order.get("trend_signal", {}) or {})
                    if pending_action == 1:
                        fill_signal["hybrid_entry_long_price"] = float(filled_price)
                    else:
                        fill_signal["hybrid_entry_short_price"] = float(filled_price)
                    meta_router._update_pos(
                        pending_action,
                        float(filled_price),
                        timestamp_kst=pending_order.get("placed_ts", current_time_kst),
                        leverage=float(pending_order.get("kelly", 0.0) or 0.0),
                        trend_signal=fill_signal,
                    )
                    if meta_router.pos == "LONG":
                        meta_router.entry_price = float(filled_price) / max(1.0 + meta_router.trade_slip, 1e-8)
                    elif meta_router.pos == "SHORT":
                        meta_router.entry_price = float(filled_price) / max(1.0 - meta_router.trade_slip, 1e-8)
                    current_entry_kind = "limit_anchor"
                    current_entry_display_price = float(filled_price)
                    pending_counter["filled"] += 1
                    pending_counter[f"filled_{pending_order.get('mode', 'unknown')}"] += 1
                    pending_order = None
                    _sync_dsac_with_meta()
                else:
                    if pending_order is not None:
                        age = i - int(pending_order.get("placed_index", i))
                        if age >= max(1, int(limit_ttl_bars)):
                            pending_counter["ttl_cancel"] += 1
                            pending_counter[f"ttl_cancel_{pending_order.get('mode', 'unknown')}"] += 1
                            pending_order = None

            row_dict = last_row.to_dict()
            if "m7_prob_dn" not in row_dict:
                row_dict["m7_prob_dn"] = _safe_float(row_dict.get("prob_dn", row_dict.get("m7_trend_xgb_dn", 0.0)))
            if "m7_prob_fl" not in row_dict:
                row_dict["m7_prob_fl"] = _safe_float(row_dict.get("prob_flat", row_dict.get("m7_trend_xgb_fl", 0.0)))
            if "m7_prob_up" not in row_dict:
                row_dict["m7_prob_up"] = _safe_float(row_dict.get("prob_up", row_dict.get("m7_trend_xgb_up", 0.0)))
            m7_defaults = {
                "m7_trend_xgb_dn": 1.0 / 3.0,
                "m7_trend_xgb_fl": 1.0 / 3.0,
                "m7_trend_xgb_up": 1.0 / 3.0,
                "m7_confidence": 0.0,
                "m7_action": 0.0,
                "m7_size": 0.0,
                "m7_gate_block": 0.0,
                "m7_hdb_label": -1.0,
                "m7_hdb_prob": 0.0,
                "m7_iso_score": 0.0,
                "m7_iso_pred": 1.0,
                "m7_qwidth": 0.0,
                "m7_quality_pred": 0.0,
                "m7_hold_pred": 0.0,
                "m7_target_hold": 0.0,
                "m7_entry_long_offset": 0.0,
                "m7_entry_short_offset": 0.0,
                "m7_entry_long_price": 0.0,
                "m7_entry_short_price": 0.0,
                "m7_tp_offset": 0.0,
                "m7_sl_offset": 0.0,
                "m7_iso_anom": 0.0,
                "m7_vae_error": 0.0,
                "m7_vae_threshold": 0.0,
                "m7_vae_anom": 0.0,
                "m7_q10": 0.0,
                "m7_q50": 0.0,
                "m7_q90": 0.0,
                "m7_tp_price": 0.0,
                "m7_sl_price": 0.0,
                "m7_gmm_cluster": -1.0,
                "m7_gmm_conf": 0.0,
                "m7_gmm_vol_rank": 0.5,
                "m7_expected_ret": 0.0,
                "m7_tail_risk": 0.0,
                "m7_composite_score": 0.0,
            }
            for k, v in m7_defaults.items():
                row_dict.setdefault(k, v)
            nf_preds = dict(row_dict)
            pred_fallback = _safe_float(nf_preds.get("pred_patchtst", 0.0))
            conf_fallback = float(np.clip(_safe_float(nf_preds.get("conf_patchtst", 0.5), 0.5), 0.0, 1.0))
            for c in DSAC_STATE_PRED:
                nf_preds.setdefault(c, pred_fallback)
            for c in DSAC_STATE_CONF:
                nf_preds.setdefault(c, conf_fallback)

            trend_signal = trend_signal_from_m7(row_dict)
            dsac_action, dsac_lev, info, _, _ = dsac_router.decide(processed_df, nf_preds, m7_signal=trend_signal)

            action_val = float(info.get("primary_raw", info.get("raw_action", 0.0)))
            regime_name = next(
                (
                    str(k).replace("regime_", "")
                    for k, v in row_dict.items()
                    if str(k).startswith("regime_") and float(_safe_float(v, 0.0)) == 1.0
                ),
                "normal",
            )
            pure_rl = decide_pure_rl_action(
                action_val=action_val,
                current_pos=meta_router.pos,
                live_unrealized_pnl=float(meta_router._net_pnl_frac(current_price)) if meta_router.pos in {"LONG", "SHORT"} else 0.0,
                alpha_focus_enabled=alpha_focus_strict_enable,
                alpha_focus_row=row_dict,
                alpha_focus_regime=regime_name,
                alpha_focus_sizing_fn=_alpha_focus_strict_sizing,
                alpha_focus_exposure_cap=1.0,
                oos_parity_mode=oos_parity_mode,
                dsac_action=int(dsac_action),
                dsac_lev=float(dsac_lev),
            )
            fa = int(pure_rl.final_action)
            kelly = float(pure_rl.kelly)
            source = str(pure_rl.source)

            prev_pos = meta_router.pos
            prev_hold = meta_router.hold_count
            prev_entry = current_entry_display_price if current_entry_display_price > 0.0 else meta_router.entry_price
            prev_lev = meta_router.current_leverage
            closed_entry_kind = current_entry_kind

            if hybrid_execution and prev_pos is None and pending_order is None and fa in (1, 2):
                nif = float(last_row.get("nif_whale", last_row.get("sig_whale", 0.0)) or 0.0)
                ms_signal = {
                    "signal_bias": 1 if nif > 0.0 else (-1 if nif < 0.0 else 0),
                    "shadow_toxicity_score": float(last_row.get("shadow_toxicity_score", 0.0) or 0.0),
                    "shadow_queue_collapse": float(last_row.get("shadow_queue_collapse", 0.0) or 0.0),
                    "obi": float(last_row.get("obi", 0.0) or 0.0),
                    "taker_buy_ratio": float(last_row.get("taker_buy_ratio", 0.5) or 0.5),
                }
                overlay = _hybrid_execution_overlay(
                    final_action=int(fa),
                    current_price=float(current_price),
                    processed_df=processed_df,
                    trend_signal=trend_signal,
                    ms_signal=ms_signal,
                    current_time_kst=current_time_kst,
                )
                hybrid_counter[str(overlay.get("mode", "inactive"))] += 1
                fa = int(overlay.get("action", fa))
                if fa in (1, 2):
                    trend_signal = dict(trend_signal or {})
                    mode = str(overlay.get("mode", "market") or "market")
                    if realistic_limit_fill and mode == "limit_anchor":
                        limit_price = float(overlay.get("entry_price", current_price) or current_price)
                        adverse = _adverse_score(processed_df, last_row, int(fa), current_price, limit_price)
                        if adverse_filter and adverse >= float(adverse_place_th):
                            pending_counter["adverse_skip"] += 1
                            pending_counter[f"adverse_skip_{mode}"] += 1
                            fa = 0
                            kelly = 0.0
                            source = f"{source}|EXEC_SKIP:{mode}"
                            continue
                        pending_order = {
                            "action": int(fa),
                            "kelly": float(kelly),
                            "limit_price": limit_price,
                            "trend_signal": dict(trend_signal),
                            "placed_index": i,
                            "mode": mode,
                            "placed_ts": str(current_time_kst),
                        }
                        pending_counter["placed"] += 1
                        pending_counter[f"placed_{mode}"] += 1
                        fa = 0
                        kelly = 0.0
                        source = f"{source}|EXEC_PENDING:{mode}"
                    else:
                        entry_price = float(overlay.get("entry_price", next_price) or next_price)
                        if fill_mode == "current":
                            entry_price = float(overlay.get("entry_price", current_price) or current_price)
                        elif fill_mode == "next_open":
                            entry_price = float(overlay.get("entry_price", next_open) or next_open)
                        else:
                            entry_price = float(overlay.get("entry_price", next_close) or next_close)
                        if realistic_limit_fill and mode == "trend_taker":
                            entry_price = float(next_row.get("open", entry_price) or entry_price)
                        if fa == 1:
                            trend_signal["hybrid_entry_long_price"] = entry_price
                        else:
                            trend_signal["hybrid_entry_short_price"] = entry_price
                        source = f"{source}|EXEC:{mode}"
                else:
                    kelly = 0.0
                    source = f"{source}|EXEC_WAIT:{overlay.get('reason', '')}"

            if prev_pos is not None:
                eq_curve.append(balance * (1.0 + meta_router._net_pnl_frac(current_price)))
            else:
                eq_curve.append(balance)

            meta_router._update_pos(
                fa,
                fill_price,
                timestamp_kst=current_time_kst,
                leverage=kelly,
                trend_signal=trend_signal,
            )

            if prev_pos is None and meta_router.pos is not None:
                current_entry_kind = "trend_taker" if ("EXEC:trend_taker" in source) else "market"
                current_entry_display_price = float(meta_router.entry_price)
            elif prev_pos is not None and meta_router.pos is None:
                current_entry_kind = "market"
                current_entry_display_price = 0.0
            elif prev_pos is not None and meta_router.pos is not None and prev_pos != meta_router.pos:
                current_entry_kind = "market"
                current_entry_display_price = float(meta_router.entry_price)

            if prev_pos is None and meta_router.pos == "LONG":
                long_entries += 1
                levs.append(meta_router.current_leverage)
            elif prev_pos is None and meta_router.pos == "SHORT":
                short_entries += 1
                levs.append(meta_router.current_leverage)
            elif prev_pos is not None and meta_router.pos is not None and prev_pos != meta_router.pos:
                realized = float(meta_router.last_realized_pnl or 0.0)
                if closed_entry_kind == "limit_anchor":
                    realized += float(meta_router.trade_fee) * float(prev_lev)
                balance *= (1.0 + realized)
                trades += 1
                wins += int(realized > 0.0)
                hold_bars.append(prev_hold)
                trade_rows.append(
                    {
                        "ts": str(df.iloc[i + 1]["timestamp"]),
                        "side": prev_pos,
                        "entry_price": prev_entry,
                        "exit_price": fill_price,
                        "lev": prev_lev,
                        "hold_bars": prev_hold,
                        "pnl_frac": realized,
                        "exit_source": source + f"|FLIP|ENTRY_KIND:{closed_entry_kind}",
                    }
                )
            elif prev_pos is not None and meta_router.pos is None:
                realized = float(meta_router.last_realized_pnl or 0.0)
                if closed_entry_kind == "limit_anchor":
                    realized += float(meta_router.trade_fee) * float(prev_lev)
                balance *= (1.0 + realized)
                trades += 1
                wins += int(realized > 0.0)
                hold_bars.append(prev_hold)
                trade_rows.append(
                    {
                        "ts": str(df.iloc[i + 1]["timestamp"]),
                        "side": prev_pos,
                        "entry_price": prev_entry,
                        "exit_price": fill_price,
                        "lev": prev_lev,
                        "hold_bars": prev_hold,
                        "pnl_frac": realized,
                        "exit_source": source + f"|ENTRY_KIND:{closed_entry_kind}",
                    }
                )

        if meta_router.pos is not None:
            final_price = float(df.iloc[-1]["close"])
            realized = float(meta_router._net_pnl_frac(final_price))
            if current_entry_kind == "limit_anchor":
                realized += float(meta_router.trade_fee) * float(meta_router.current_leverage)
            balance *= (1.0 + realized)
            trades += 1
            wins += int(realized > 0.0)
            hold_bars.append(meta_router.hold_count)
            trade_rows.append(
                {
                    "ts": str(df.iloc[-1]["timestamp"]),
                    "side": meta_router.pos,
                    "entry_price": current_entry_display_price if current_entry_display_price > 0.0 else meta_router.entry_price,
                    "exit_price": final_price,
                    "lev": meta_router.current_leverage,
                    "hold_bars": meta_router.hold_count,
                    "pnl_frac": realized,
                    "exit_source": f"FINAL_MARK|ENTRY_KIND:{current_entry_kind}",
                }
            )
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
            avg_lev=float(np.mean(levs) if levs else 0.0),
        )
        extra = {
            "final_balance": balance,
            "hybrid_execution": bool(hybrid_execution),
            "fill_mode": str(fill_mode),
            "realistic_limit_fill": bool(realistic_limit_fill),
            "limit_ttl_bars": int(limit_ttl_bars),
            "adverse_filter": bool(adverse_filter),
            "adverse_place_th": float(adverse_place_th),
            "adverse_cancel_th": float(adverse_cancel_th),
            "hybrid_mode_counts": dict(hybrid_counter),
            "pending_order_counts": dict(pending_counter),
            "trades": trade_rows,
        }
        return metrics, extra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", required=True)
    ap.add_argument("--ckpt-path", default="/home/kbj20/crypto-scalping/data/ensemble/ckpt/best_dsac_agents.pth")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--hybrid-execution", action="store_true")
    ap.add_argument("--fill-mode", choices=["current", "next_open", "next_close"], default="next_close")
    ap.add_argument("--realistic-limit-fill", action="store_true")
    ap.add_argument("--limit-ttl-bars", type=int, default=2)
    ap.add_argument("--adverse-filter", action="store_true")
    ap.add_argument("--adverse-place-th", type=float, default=0.78)
    ap.add_argument("--adverse-cancel-th", type=float, default=0.95)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    df = _load_frame(args.csv_path, args.start, args.end)
    metrics, extra = simulate(
        df,
        args.ckpt_path,
        hybrid_execution=bool(args.hybrid_execution),
        fill_mode=str(args.fill_mode),
        realistic_limit_fill=bool(args.realistic_limit_fill),
        limit_ttl_bars=int(args.limit_ttl_bars),
        adverse_filter=bool(args.adverse_filter),
        adverse_place_th=float(args.adverse_place_th),
        adverse_cancel_th=float(args.adverse_cancel_th),
    )
    payload = {
        "csv_path": args.csv_path,
        "ckpt_path": args.ckpt_path,
        "start": str(df["timestamp"].iloc[0]) if len(df) else None,
        "end": str(df["timestamp"].iloc[-1]) if len(df) else None,
        "rows": int(len(df)),
        "hybrid_execution": bool(args.hybrid_execution),
        "fill_mode": str(args.fill_mode),
        "realistic_limit_fill": bool(args.realistic_limit_fill),
        "limit_ttl_bars": int(args.limit_ttl_bars),
        "adverse_filter": bool(args.adverse_filter),
        "adverse_place_th": float(args.adverse_place_th),
        "adverse_cancel_th": float(args.adverse_cancel_th),
        "metrics": asdict(metrics),
        "extra": extra,
    }
    out_json = args.out_json or os.path.join(
        _ROOT_DIR,
        "data/ensemble/reports",
        f"replay_trading_bot_pure_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    print(json.dumps(payload["metrics"], indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
