#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)


@dataclass
class SimMetrics:
    pnl_pct: float
    mdd_pct: float
    trades: int
    wr_pct: float
    long_entries: int
    short_entries: int
    avg_hold_long: float
    avg_hold_short: float
    avg_lev_long: float
    avg_lev_short: float
    cross_exit: int
    ambiguity_exit: int
    reduce_hits: int
    max_hold_exit: int
    trail_exit: int
    step_stop_exit: int
    trend_exit: int
    cooldown_block: int
    trend_flat_block: int
    trend_mismatch_block: int
    iso_vae_block: int


def _safe_float(v, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not np.isfinite(x):
        return float(default)
    return x


def _load_frame(csv_path: str, start: str | None, end: str | None) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise KeyError("timestamp column missing")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if start:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end:
        df = df[df["timestamp"] <= pd.Timestamp(end)]
    if "close" not in df.columns:
        raise KeyError("close column missing")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"]).reset_index(drop=True)
    # Dual specialist live router expects a wider live feature schema than RL CSV carries.
    base_defaults = {
        "open": None,
        "high": None,
        "low": None,
        "volume": 1.0,
        "whale_retail_ratio": 0.0,
        "whale_conviction": 0.0,
        "smart_money_flow": 0.0,
        "last_funding_rate": 0.0,
        "net_taker_ratio": 0.0,
        "taker_acceleration": 0.0,
        "rsi": 50.0,
        "wick_ratio": 0.0,
        "hurst_48": 0.0,
        "hurst_288": 0.0,
        "ofi_acceleration": 0.0,
        "trade_intensity": 0.0,
        "funding_price_divergence": 0.0,
        "short_squeeze_risk": 0.0,
        "long_squeeze_risk": 0.0,
        "big_trade_ratio": 0.0,
        "funding_roc_12": 0.0,
        "funding_roc_288": 0.0,
        "cvp_cluster_position": 0.0,
        "fibonacci_level": 0.0,
        "count_toptrader_long_short_ratio": 0.0,
        "count_long_short_ratio": 0.0,
        "btc_corr_60": 0.0,
        "eth_btc_ratio_change": 0.0,
        "mtf_trend_1h": 0.0,
        "mtf_trend_4h": 0.0,
        "rogers_satchell_vol": 0.0,
        "amihud_illiquidity_z": 0.0,
        "squeeze_power": 0.0,
        "garman_klass_vol": 0.0,
        "funding_z_score": 0.0,
        "volatility_z": 0.0,
        "hma_slope": 0.0,
        "jump_z": 0.0,
        "evt_excess_z": 0.0,
        "jump_flag": 0.0,
        "evt_tail_flag": 0.0,
        "funding_pressure": 0.0,
        "garch_vol_z": 0.0,
    }
    try:
        from features.schema import STATE_ALPHA as _STATE_ALPHA, STATE_SYNTH as _STATE_SYNTH
        for _c in list(_STATE_ALPHA) + list(_STATE_SYNTH):
            base_defaults.setdefault(str(_c), 0.0)
    except Exception:
        pass
    for col, default in base_defaults.items():
        if col not in df.columns:
            if default is None:
                df[col] = df["close"]
            else:
                df[col] = default
    for col in ("open", "high", "low"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df["close"])
    return df


def _simulate_dual(
    df: pd.DataFrame,
    long_ckpt: str | None,
    short_ckpt: str | None,
    mode: str = "classic",
) -> tuple[SimMetrics, dict]:
    with tempfile.TemporaryDirectory(prefix="dual_dsac_bt_") as tmpdir:
        os.environ["DSAC_LIVE_STATE_PATH"] = os.path.join(tmpdir, "live_state.json")
        os.environ["FUSE_ADAPT_STATE_PATH"] = os.path.join(tmpdir, "adapt_state.json")

        from features.m7 import trend_signal_from_m7
        from features.schema import STATE_PRED as DSAC_STATE_PRED, STATE_CONF as DSAC_STATE_CONF
        from trading_bot import DSACSignalRouter, DSACTrendRouter

        # trading_bot.DSACSignalRouter now uses a single checkpoint path.
        dsac_router = DSACSignalRouter(model_path=(long_ckpt or short_ckpt))
        meta_router = DSACTrendRouter()
        meta_router.online_adapt = False
        # Backtest loop performance: disable per-step live-state fsync.
        meta_router._save_live_state = lambda *args, **kwargs: None
        flat_override_min_votes = int(os.getenv("DSAC_FLAT_OVERRIDE_MIN_VOTES", "3"))
        flat_override_kelly_mult = float(os.getenv("DSAC_FLAT_OVERRIDE_KELLY_MULT", "0.60"))

        def _sync_dsac_with_meta() -> None:
            dsac_router.pos = meta_router.pos
            dsac_router.entry_price = meta_router.entry_price
            dsac_router.hold_count = meta_router.hold_count
            dsac_router.current_leverage = meta_router.current_leverage
            dsac_router.current_equity = meta_router.cur_equity
            dsac_router.peak_equity = meta_router.peak_equity

        balance = 1.0
        eq_curve: list[float] = [balance]
        long_entries = short_entries = 0
        cross_exit = ambiguity_exit = reduce_hits = 0
        max_hold_exit = trail_exit = step_stop_exit = trend_exit = 0
        cooldown_block = trend_flat_block = trend_mismatch_block = iso_vae_block = 0
        wins = trades = 0
        lev_long: list[float] = []
        lev_short: list[float] = []
        hold_long: list[int] = []
        hold_short: list[int] = []
        trade_rows: list[dict] = []

        # Proposed-mode tunables (env-driven for sweep without code edits)
        p_jump_z_th = float(os.getenv("P_JUMP_Z_TH", "3.0"))
        p_chop_std_th = float(os.getenv("P_CHOP_STD_TH", "1.2"))
        p_th_bull_long = float(os.getenv("P_TH_BULL_LONG", "0.15"))
        p_th_bull_short = float(os.getenv("P_TH_BULL_SHORT", "0.35"))
        p_th_bear_long = float(os.getenv("P_TH_BEAR_LONG", "0.35"))
        p_th_bear_short = float(os.getenv("P_TH_BEAR_SHORT", "0.15"))
        p_th_normal = float(os.getenv("P_TH_NORMAL", "0.22"))
        p_th_chop = float(os.getenv("P_TH_CHOP", "0.40"))
        p_kelly_cap = float(os.getenv("P_KELLY_CAP", "0.35"))
        p_kelly_min = float(os.getenv("P_KELLY_MIN", "0.05"))
        p_quality_mult = float(os.getenv("P_QUALITY_MULT", "0.40"))
        p_agree_yes_mult = float(os.getenv("P_AGREE_YES_MULT", "0.90"))
        p_agree_no_base = float(os.getenv("P_AGREE_NO_BASE", "0.30"))
        p_agree_no_excess = float(os.getenv("P_AGREE_NO_EXCESS", "0.20"))
        p_hard_stop = float(os.getenv("P_HARD_STOP", "0.025"))
        p_m7_opp_exit = float(os.getenv("P_M7_OPP_EXIT", "0.60"))
        p_opp_pressure_exit = float(os.getenv("P_OPP_PRESSURE_EXIT", "1.15"))
        p_trail_arm = float(os.getenv("P_TRAIL_ARM", "0.012"))
        p_trail_gap = float(os.getenv("P_TRAIL_GAP", "0.008"))
        p_reduce_net_edge = float(os.getenv("P_REDUCE_NET_EDGE", "0.05"))
        p_reduce_mult = float(os.getenv("P_REDUCE_MULT", "0.65"))
        p_min_tp_offset = float(os.getenv("P_MIN_TP_OFFSET", "0.0"))
        p_min_sl_offset = float(os.getenv("P_MIN_SL_OFFSET", "0.0"))
        p_enable_tpsl = str(os.getenv("P_ENABLE_TPSL", "true")).strip().lower() in {"1", "true", "yes", "on"}

        proposed_no_go_counter: Counter[str] = Counter()
        proposed_exit_counter: Counter[str] = Counter()
        proposed_entry_counter: Counter[str] = Counter()

        for i in range(60, len(df) - 1):
            start_i = max(0, i - 300)
            processed_df = df.iloc[start_i : i + 1].copy()
            last_row = processed_df.iloc[-1]
            current_price = float(last_row["close"])
            next_price = float(df.iloc[i + 1]["close"])

            _sync_dsac_with_meta()

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
                "m7_iso_anom": 0.0,
                "m7_vae_error": 0.0,
                "m7_vae_threshold": 0.0,
                "m7_vae_anom": 0.0,
                "m7_q10": 0.0,
                "m7_q50": 0.0,
                "m7_q90": 0.0,
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
                if k not in row_dict:
                    row_dict[k] = v
            nf_preds = dict(row_dict)
            pred_fallback = _safe_float(nf_preds.get("pred_patchtst", 0.0))
            conf_fallback = float(np.clip(_safe_float(nf_preds.get("conf_patchtst", 0.5), 0.5), 0.0, 1.0))
            for c in DSAC_STATE_PRED:
                if c not in nf_preds:
                    nf_preds[c] = pred_fallback
            for c in DSAC_STATE_CONF:
                if c not in nf_preds:
                    nf_preds[c] = conf_fallback
            trend_signal = trend_signal_from_m7(row_dict)
            dsac_action, dsac_lev, info, _, _ = dsac_router.decide(processed_df, nf_preds, m7_signal=trend_signal)
            long_raw = float(info.get("_long_raw", info.get("long_edge", 0.0)))
            short_raw = float(info.get("_short_raw", info.get("short_edge", 0.0)))
            if _safe_float(last_row.get("regime_bull", 0.0)) >= 0.5:
                regime_name = "bull"
            elif _safe_float(last_row.get("regime_bear", 0.0)) >= 0.5:
                regime_name = "bear"
            elif _safe_float(last_row.get("regime_chop", 0.0)) >= 0.5:
                regime_name = "chop"
            elif _safe_float(last_row.get("regime_whipsaw", 0.0)) >= 0.5:
                regime_name = "whipsaw"
            else:
                regime_name = "normal"

            garch_vol_z = float(last_row.get("garch_vol_z", 0.0))
            btc_3bar_ret = float(last_row.get("btc_3bar_ret", 0.0))
            iso_anom = bool(float((trend_signal or {}).get("m7_iso_anom", 0.0)) >= 0.5)
            vae_anom = bool(float((trend_signal or {}).get("m7_vae_anom", 0.0)) >= 0.5)
            vae_err = float((trend_signal or {}).get("m7_vae_error", 0.0))
            vae_th = float((trend_signal or {}).get("m7_vae_threshold", 0.0))
            vae_ratio = (vae_err / max(vae_th, 1e-8)) if vae_th > 1e-8 else (1.0 if vae_anom else 0.0)

            if mode == "proposed":
                # With specialists_only=True, primary_raw/primary_lev are zeroed in the router.
                # Use the preserved primary_model_raw/primary_model_kelly for accurate signals.
                primary_raw = float(info.get("primary_model_raw", info.get("primary_raw", info.get("raw_action", 0.0))))
                primary_kelly = float(np.clip(
                    info.get("primary_model_kelly", info.get("primary_kelly", dsac_lev)), 0.0, 1.0
                ))
                long_logit = float(info.get("long_logit", long_raw))
                short_logit = float(info.get("short_logit", short_raw))
                long_std = max(1e-6, float(info.get("long_std", 1.0)))
                short_std = max(1e-6, float(info.get("short_std", 1.0)))
                avg_std = max(1e-6, 0.5 * (long_std + short_std))
                m7_up = float((trend_signal or {}).get("m7_prob_up", (trend_signal or {}).get("prob_up", 0.0)))
                m7_dn = float((trend_signal or {}).get("m7_prob_dn", (trend_signal or {}).get("prob_dn", 0.0)))
                m7_conf = float((trend_signal or {}).get("m7_confidence", 0.0))
                m7_quality = float((trend_signal or {}).get("m7_quality_pred", 0.0))
                agreement_count = int(info.get("agreement_count", 0))
                regime_bias = 1.0 if regime_name == "bull" else (-1.0 if regime_name == "bear" else 0.0)
                direction = (
                    0.35 * primary_raw
                    + 0.25 * ((long_logit - short_logit) / avg_std)
                    + 0.25 * ((m7_up - m7_dn) * m7_conf)
                    + 0.15 * regime_bias
                )

                if regime_name == "bull":
                    th_long, th_short = p_th_bull_long, p_th_bull_short
                elif regime_name == "bear":
                    th_long, th_short = p_th_bear_long, p_th_bear_short
                elif regime_name in ("chop", "whipsaw"):
                    th_long, th_short = p_th_chop, p_th_chop
                else:
                    th_long, th_short = p_th_normal, p_th_normal

                hard_no_go = False
                if int((trend_signal or {}).get("m7_gate_block", 0)) == 1:
                    hard_no_go = True
                    proposed_no_go_counter["m7_gate_block"] += 1
                elif bool(float((trend_signal or {}).get("m7_iso_anom", 0.0)) >= 0.5) and bool(float((trend_signal or {}).get("m7_vae_anom", 0.0)) >= 0.5):
                    hard_no_go = True
                    proposed_no_go_counter["iso_vae"] += 1
                elif abs(float(last_row.get("jump_z", 0.0))) > p_jump_z_th:
                    hard_no_go = True
                    proposed_no_go_counter["jump_z"] += 1
                elif int(float(last_row.get("evt_tail_flag", 0.0)) >= 0.5) == 1:
                    hard_no_go = True
                    proposed_no_go_counter["evt_tail"] += 1
                elif regime_name == "chop" and long_std > p_chop_std_th and short_std > p_chop_std_th:
                    hard_no_go = True
                    proposed_no_go_counter["chop_high_std"] += 1

                fa = 0
                kelly = 0.0
                source = "PROPOSED"
                if meta_router.pos is None:
                    if not hard_no_go:
                        target = 0
                        if direction >= th_long:
                            target = 1
                        elif direction <= -th_short:
                            target = 2
                        if target != 0:
                            is_specialists_only = getattr(dsac_router, "specialists_only", True)
                            if is_specialists_only:
                                # Primary kelly is zeroed by specialists_only; use specialist's kelly
                                if target == 1:
                                    kelly = float(np.clip(info.get("_long_kelly", info.get("primary_model_kelly", 0.15)), p_kelly_min, p_kelly_cap))
                                else:
                                    kelly = float(np.clip(info.get("_short_kelly", info.get("primary_model_kelly", 0.15)), p_kelly_min, p_kelly_cap))
                            else:
                                kelly = primary_kelly
                            kelly *= float(np.clip(1.0 + p_quality_mult * m7_quality, 0.6, 1.4))
                            # specialists_only=True: vote_pool=(long,short) → max agreement_count=1
                            # (short_vote never equals +1, long_vote never equals -1)
                            # So tiers 2 and 3 are unreachable; re-map accordingly:
                            #   ≥1 = specialist confirmed   → full kelly
                            #    0 = direction signal only  → reduced kelly (not zero)
                            if is_specialists_only:
                                if agreement_count >= 1:
                                    kelly *= p_agree_yes_mult  # specialist explicitly confirmed direction
                                else:
                                    # Specialist uncertain; direction score crossed threshold.
                                    # Scale by how strongly direction exceeded the bar.
                                    dir_excess = float(np.clip(abs(direction) / max(th_long, th_short, 1e-6) - 1.0, 0.0, 1.0))
                                    kelly *= float(np.clip(p_agree_no_base + p_agree_no_excess * dir_excess, 0.20, 0.60))
                            else:
                                if agreement_count >= 3:
                                    kelly *= 1.0
                                elif agreement_count == 2:
                                    kelly *= 0.75
                                elif agreement_count == 1:
                                    kelly *= 0.45
                                else:
                                    kelly = 0.0
                            kelly = float(np.clip(kelly, 0.0, p_kelly_cap))
                            if kelly > 0.0:
                                fa = target
                                if target == 1:
                                    proposed_entry_counter["long"] += 1
                                elif target == 2:
                                    proposed_entry_counter["short"] += 1
                else:
                    side = 1 if meta_router.pos == "LONG" else 2
                    fa = side
                    kelly = float(np.clip(max(meta_router.current_leverage, 0.05), 0.0, 0.35))
                    live_unr = float(meta_router._net_pnl_frac(current_price))
                    own_support = float(info.get("own_support", 0.0))
                    opp_pressure = float(info.get("opp_pressure", 0.0))
                    net_edge = float(info.get("net_edge", 0.0))
                    # Exit/Reduce priority: hard risk, opposite M7, opp pressure, trailing, TP/SL
                    force_exit = False
                    if live_unr <= -p_hard_stop:
                        force_exit = True
                        source = "PROPOSED_HARD_STOP"
                        proposed_exit_counter["hard_stop"] += 1
                    else:
                        opposite_prob = m7_dn if meta_router.pos == "LONG" else m7_up
                        if opposite_prob > p_m7_opp_exit:
                            force_exit = True
                            source = "PROPOSED_M7_REVERSE_EXIT"
                            proposed_exit_counter["m7_reverse"] += 1
                        elif opp_pressure > p_opp_pressure_exit:
                            force_exit = True
                            source = "PROPOSED_OPP_PRESSURE_EXIT"
                            proposed_exit_counter["opp_pressure"] += 1
                        else:
                            peak_gain = float(meta_router.peak_equity - 1.0)
                            draw_from_peak = float(meta_router.peak_equity - meta_router.cur_equity)
                            if peak_gain >= p_trail_arm and draw_from_peak >= p_trail_gap:
                                force_exit = True
                                source = "PROPOSED_TRAIL_EXIT"
                                proposed_exit_counter["trail"] += 1
                    tp_offset = max(float((trend_signal or {}).get("m7_tp_offset", 0.0)), p_min_tp_offset)
                    sl_offset = max(float((trend_signal or {}).get("m7_sl_offset", 0.0)), p_min_sl_offset)
                    if p_enable_tpsl and meta_router.entry_price > 0.0:
                        if meta_router.pos == "LONG":
                            tp_px = meta_router.entry_price * (1.0 + max(tp_offset, 0.0))
                            sl_px = max(
                                meta_router.entry_price * (1.0 - max(sl_offset, 0.0)),
                                meta_router.entry_price * (1.0 - 0.025),
                            )
                            if next_price >= tp_px or next_price <= sl_px:
                                force_exit = True
                                source = "PROPOSED_M7_TPSL_EXIT"
                                proposed_exit_counter["tpsl"] += 1
                        else:
                            tp_px = meta_router.entry_price * (1.0 - max(tp_offset, 0.0))
                            sl_px = min(
                                meta_router.entry_price * (1.0 + max(sl_offset, 0.0)),
                                meta_router.entry_price * (1.0 + 0.025),
                            )
                            if next_price <= tp_px or next_price >= sl_px:
                                force_exit = True
                                source = "PROPOSED_M7_TPSL_EXIT"
                                proposed_exit_counter["tpsl"] += 1
                    if force_exit:
                        fa = 0
                        kelly = 0.0
                    else:
                        # keep EMA-based hold condition; reduce when condition weakens
                        if not (own_support > 0.0 and net_edge >= p_reduce_net_edge):
                            kelly = float(np.clip(kelly * p_reduce_mult, 0.0, p_kelly_cap))

            elif mode == "pure_rl":
                if getattr(dsac_router, "specialists_only", False):
                    action_val = float(info.get("raw_action", info.get("primary_raw", 0.0)))
                else:
                    action_val = float(info.get("primary_raw", info.get("raw_action", 0.0)))
                abs_action = abs(action_val)
                pos_th = float(os.getenv("DSAC_PURE_RL_POS_TH", "0.12"))
                close_th = float(os.getenv("DSAC_PURE_RL_CLOSE_TH", "0.03"))
                flip_th = float(os.getenv("DSAC_PURE_RL_FLIP_TH", str(max(pos_th * 1.5, pos_th))))
                flip_kelly_mult = float(os.getenv("DSAC_PURE_RL_FLIP_KELLY_MULT", "0.85"))
                max_kelly = float(os.getenv("DSAC_PURE_RL_MAX_KELLY", "1.0"))
                force_close = str(os.getenv("DSAC_PURE_RL_FORCE_CLOSE", "false")).strip().lower() in {"1", "true", "yes", "on"}
                fa = 0
                kelly = 0.0
                source = "DSAC_PURE_RL"
                if meta_router.pos is None:
                    if action_val > pos_th:
                        fa, kelly = 1, min(abs_action, max_kelly)
                    elif action_val < -pos_th:
                        fa, kelly = 2, min(abs_action, max_kelly)
                elif meta_router.pos == "LONG":
                    live_unr = float(meta_router._net_pnl_frac(current_price))
                    if force_close and live_unr <= -0.025:
                        fa, kelly = 0, 0.0
                        source = "DSAC_PURE_RL_FORCE_CLOSE"
                    elif abs_action < close_th:
                        fa, kelly = 0, 0.0
                    elif action_val < -flip_th:
                        fa, kelly = 2, min(abs_action, max_kelly) * flip_kelly_mult
                    else:
                        fa, kelly = 1, min(abs_action, max_kelly)
                else:
                    live_unr = float(meta_router._net_pnl_frac(current_price))
                    if force_close and live_unr <= -0.025:
                        fa, kelly = 0, 0.0
                        source = "DSAC_PURE_RL_FORCE_CLOSE"
                    elif abs_action < close_th:
                        fa, kelly = 0, 0.0
                    elif action_val > flip_th:
                        fa, kelly = 1, min(abs_action, max_kelly) * flip_kelly_mult
                    else:
                        fa, kelly = 2, min(abs_action, max_kelly)
                kelly = float(np.clip(kelly, 0.0, 1.0))
            else:
                # vol_scale removed from runtime router; keep neutral scaling in backtest.
                kelly = float(np.clip(dsac_lev, 0.0, 1.0))
                fa = int(dsac_action)
                source = "DSAC_ONLY"
            if mode not in ("pure_rl", "proposed") and meta_router.pos is not None:
                live_unr = float(meta_router._net_pnl_frac(current_price))
                position_signal = str(info.get("position_signal", "HOLD"))
                position_reason = str(info.get("position_reason", ""))
                trend_exit_flag, _, trend_exit_reason = meta_router.update_trend_mismatch(processed_df, trend_signal)
                step_floor = float(meta_router.step_stop_floor()) if hasattr(meta_router, "step_stop_floor") else -0.025
                trailing_stop_hit = bool(meta_router.should_trailing_stop()) if hasattr(meta_router, "should_trailing_stop") else False
                dsac_only_max_hold = int(getattr(meta_router, "dsac_only_max_hold", 96))
                if live_unr <= step_floor:
                    fa = 0
                    kelly = 0.0
                    source = "DSAC_STEP_STOP" if meta_router.peak_equity >= 1.006 else "DSAC_ONLY_HARD_STOP"
                    step_stop_exit += 1
                elif trailing_stop_hit:
                    fa = 0
                    kelly = 0.0
                    source = "DSAC_ONLY_TRAILING_STOP"
                    trail_exit += 1
                elif meta_router.hold_count >= max(1, dsac_only_max_hold):
                    fa = 0
                    kelly = 0.0
                    source = "DSAC_ONLY_MAX_HOLD"
                    max_hold_exit += 1
                elif position_signal == "EXIT":
                    fa = 0
                    kelly = 0.0
                    source = f"DSAC_LOGIT_EXIT:{position_reason or 'RULE'}"
                    cross_exit += 1
                elif position_signal == "REDUCE":
                    fa = 1 if meta_router.pos == "LONG" else 2
                    kelly = float(np.clip(kelly, 0.0, 1.0))
                    source = f"DSAC_LOGIT_REDUCE:{position_reason or 'RULE'}"
                    reduce_hits += 1
                elif trend_exit_flag:
                    fa = 0
                    kelly = 0.0
                    source = trend_exit_reason
                    trend_exit += 1
            elif mode not in ("pure_rl", "proposed"):
                meta_router.trend_mismatch_streak = 0

            if mode not in ("pure_rl", "proposed") and meta_router.cooldown_bars_left > 0 and meta_router.pos is None and fa != 0:
                fa = 0
                kelly = 0.0
                source = "DSAC_ONLY_COOLDOWN"
                cooldown_block += 1

            if mode not in ("pure_rl", "proposed") and fa != 0 and meta_router.pos is None:
                signal_side = 1 if fa == 1 else -1
                trend_dir = int((trend_signal or {}).get("trend_dir", 1))
                trend_side = 1 if trend_dir == 2 else (-1 if trend_dir == 0 else 0)
                agreement_count = int(info.get("agreement_count", 0))
                if trend_side == 0:
                    if agreement_count >= max(1, flat_override_min_votes):
                        kelly = float(np.clip(kelly * flat_override_kelly_mult, 0.0, 1.0))
                        source = "DSAC_TREND_FLAT_OVERRIDE"
                    else:
                        fa = 0
                        kelly = 0.0
                        source = "DSAC_ONLY_TREND_FLAT_BLOCK"
                        trend_flat_block += 1
                elif signal_side != trend_side:
                    fa = 0
                    kelly = 0.0
                    source = "DSAC_ONLY_TREND_MISMATCH_BLOCK"
                    trend_mismatch_block += 1

            if mode not in ("pure_rl", "proposed") and fa != 0 and meta_router.pos is None:
                if iso_anom and vae_anom and vae_ratio >= meta_router.dsac_only_vae_block_ratio:
                    fa = 0
                    kelly = 0.0
                    source = "DSAC_ONLY_ISO_VAE_BLOCK"
                    iso_vae_block += 1

            prev_pos = meta_router.pos
            prev_hold = meta_router.hold_count
            prev_entry = meta_router.entry_price
            prev_lev = meta_router.current_leverage

            if prev_pos is not None:
                eq_curve.append(balance * (1.0 + meta_router._net_pnl_frac(current_price)))
            else:
                eq_curve.append(balance)

            meta_router._update_pos(fa, next_price, kelly, trend_signal)

            if prev_pos is None and meta_router.pos == "LONG":
                long_entries += 1
                lev_long.append(meta_router.current_leverage)
            elif prev_pos is None and meta_router.pos == "SHORT":
                short_entries += 1
                lev_short.append(meta_router.current_leverage)
            elif prev_pos is not None and meta_router.pos is not None and prev_pos != meta_router.pos:
                # Flip: count close leg and new entry.
                realized = float(meta_router.last_realized_pnl or 0.0)
                balance *= (1.0 + realized)
                trades += 1
                wins += int(realized > 0.0)
                if prev_pos == "LONG":
                    hold_long.append(prev_hold)
                else:
                    hold_short.append(prev_hold)
                trade_rows.append(
                    {
                        "ts": str(df.iloc[i + 1]["timestamp"]),
                        "side": prev_pos,
                        "entry_price": prev_entry,
                        "exit_price": next_price,
                        "lev": prev_lev,
                        "hold_bars": prev_hold,
                        "pnl_frac": realized,
                        "exit_source": source + "|FLIP",
                    }
                )
                if meta_router.pos == "LONG":
                    long_entries += 1
                    lev_long.append(meta_router.current_leverage)
                else:
                    short_entries += 1
                    lev_short.append(meta_router.current_leverage)
                meta_router.record_outcome(realized)

            if prev_pos is not None and meta_router.pos is None:
                realized = float(meta_router.last_realized_pnl or 0.0)
                balance *= (1.0 + realized)
                trades += 1
                wins += int(realized > 0.0)
                if prev_pos == "LONG":
                    hold_long.append(prev_hold)
                else:
                    hold_short.append(prev_hold)
                trade_rows.append(
                    {
                        "ts": str(df.iloc[i + 1]["timestamp"]),
                        "side": prev_pos,
                        "entry_price": prev_entry,
                        "exit_price": next_price,
                        "lev": prev_lev,
                        "hold_bars": prev_hold,
                        "pnl_frac": realized,
                        "exit_source": source,
                    }
                )
                meta_router.record_outcome(realized)
            elif prev_pos is not None and meta_router.pos is not None and fa in (1, 2):
                if meta_router.pos == "LONG":
                    lev_long.append(meta_router.current_leverage)
                else:
                    lev_short.append(meta_router.current_leverage)

            if mode not in ("pure_rl", "proposed"):
                meta_router.decrement_cooldown()

        # Terminal close for fair realized-PnL accounting.
        if meta_router.pos is not None:
            terminal_price = float(df.iloc[-1]["close"])
            realized = float(meta_router._net_pnl_frac(terminal_price))
            prev_pos = str(meta_router.pos)
            prev_hold = int(meta_router.hold_count)
            prev_entry = float(meta_router.entry_price)
            prev_lev = float(meta_router.current_leverage)
            balance *= (1.0 + realized)
            trades += 1
            wins += int(realized > 0.0)
            if prev_pos == "LONG":
                hold_long.append(prev_hold)
            else:
                hold_short.append(prev_hold)
            trade_rows.append(
                {
                    "ts": str(df.iloc[-1]["timestamp"]),
                    "side": prev_pos,
                    "entry_price": prev_entry,
                    "exit_price": terminal_price,
                    "lev": prev_lev,
                    "hold_bars": prev_hold,
                    "pnl_frac": realized,
                    "exit_source": "TERMINAL_CLOSE",
                }
            )

        eq = np.asarray(eq_curve, dtype=np.float64)
        peak = np.maximum.accumulate(np.maximum(eq, 1e-12))
        dd = eq / peak - 1.0
        mdd_pct = float(np.min(dd) * 100.0) if dd.size else 0.0
        pnl_pct = float((balance - 1.0) * 100.0)
        wr_pct = float(100.0 * wins / trades) if trades > 0 else 0.0

        metrics = SimMetrics(
            pnl_pct=pnl_pct,
            mdd_pct=mdd_pct,
            trades=trades,
            wr_pct=wr_pct,
            long_entries=long_entries,
            short_entries=short_entries,
            avg_hold_long=float(np.mean(hold_long)) if hold_long else 0.0,
            avg_hold_short=float(np.mean(hold_short)) if hold_short else 0.0,
            avg_lev_long=float(np.mean(lev_long)) if lev_long else 0.0,
            avg_lev_short=float(np.mean(lev_short)) if lev_short else 0.0,
            cross_exit=cross_exit,
            ambiguity_exit=ambiguity_exit,
            reduce_hits=reduce_hits,
            max_hold_exit=max_hold_exit,
            trail_exit=trail_exit,
            step_stop_exit=step_stop_exit,
            trend_exit=trend_exit,
            cooldown_block=cooldown_block,
            trend_flat_block=trend_flat_block,
            trend_mismatch_block=trend_mismatch_block,
            iso_vae_block=iso_vae_block,
        )
        extra = {"trades": trade_rows, "final_balance": balance}
        if mode == "proposed":
            extra["proposed_diag"] = {
                "no_go_counts": dict(proposed_no_go_counter),
                "entry_counts": dict(proposed_entry_counter),
                "exit_counts": dict(proposed_exit_counter),
                "params": {
                    "jump_z_th": p_jump_z_th,
                    "chop_std_th": p_chop_std_th,
                    "th_bull_long": p_th_bull_long,
                    "th_bull_short": p_th_bull_short,
                    "th_bear_long": p_th_bear_long,
                    "th_bear_short": p_th_bear_short,
                    "th_normal": p_th_normal,
                    "th_chop": p_th_chop,
                    "kelly_cap": p_kelly_cap,
                    "kelly_min": p_kelly_min,
                    "quality_mult": p_quality_mult,
                    "agree_yes_mult": p_agree_yes_mult,
                    "agree_no_base": p_agree_no_base,
                    "agree_no_excess": p_agree_no_excess,
                    "hard_stop": p_hard_stop,
                    "m7_opp_exit": p_m7_opp_exit,
                    "opp_pressure_exit": p_opp_pressure_exit,
                    "trail_arm": p_trail_arm,
                    "trail_gap": p_trail_gap,
                    "reduce_net_edge": p_reduce_net_edge,
                    "reduce_mult": p_reduce_mult,
                    "min_tp_offset": p_min_tp_offset,
                    "min_sl_offset": p_min_sl_offset,
                    "enable_tpsl": p_enable_tpsl,
                },
            }
        return metrics, extra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", required=True)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--long-ckpt", default="data/ensemble/ckpt/best_dsac_long_agents.pth")
    ap.add_argument("--short-ckpt", default="data/ensemble/ckpt/best_dsac_short_agents.pth")
    ap.add_argument("--mode", choices=["classic", "pure_rl", "proposed"], default="pure_rl")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    df = _load_frame(args.csv_path, args.start, args.end)
    if args.max_rows and args.max_rows > 0:
        df = df.head(int(args.max_rows)).reset_index(drop=True)
    metrics, extra = _simulate_dual(df, args.long_ckpt, args.short_ckpt, mode=args.mode)

    payload = {
        "csv_path": args.csv_path,
        "start": str(df["timestamp"].iloc[0]) if len(df) else None,
        "end": str(df["timestamp"].iloc[-1]) if len(df) else None,
        "rows": int(len(df)),
        "long_ckpt": args.long_ckpt,
        "short_ckpt": args.short_ckpt,
        "mode": args.mode,
        "metrics": asdict(metrics),
        "extra": extra,
    }

    out_json = args.out_json
    if not out_json:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_json = os.path.join("data/ensemble/metrics", f"dual_specialist_backtest_{ts}.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)

    print(json.dumps(payload["metrics"], indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
