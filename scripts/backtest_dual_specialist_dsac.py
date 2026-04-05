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
    }
    for col, default in base_defaults.items():
        if col not in df.columns:
            if default is None:
                df[col] = df["close"]
            else:
                df[col] = default
    for col in ("open", "high", "low"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df["close"])
    return df


def _simulate_dual(df: pd.DataFrame, long_ckpt: str | None, short_ckpt: str | None) -> tuple[SimMetrics, dict]:
    with tempfile.TemporaryDirectory(prefix="dual_dsac_bt_") as tmpdir:
        os.environ["DSAC_LIVE_STATE_PATH"] = os.path.join(tmpdir, "live_state.json")
        os.environ["FUSE_ADAPT_STATE_PATH"] = os.path.join(tmpdir, "adapt_state.json")

        from features.m7 import trend_signal_from_m7
        from features.schema import STATE_PRED as DSAC_STATE_PRED, STATE_CONF as DSAC_STATE_CONF
        from trading_bot import DSACSignalRouter, DSACTrendRouter

        dsac_router = DSACSignalRouter(long_path=long_ckpt, short_path=short_ckpt)
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

            garch_vol_z = float(last_row.get("garch_vol_z", 0.0))
            btc_3bar_ret = float(last_row.get("btc_3bar_ret", 0.0))
            iso_anom = bool(float((trend_signal or {}).get("m7_iso_anom", 0.0)) >= 0.5)
            vae_anom = bool(float((trend_signal or {}).get("m7_vae_anom", 0.0)) >= 0.5)
            vae_err = float((trend_signal or {}).get("m7_vae_error", 0.0))
            vae_th = float((trend_signal or {}).get("m7_vae_threshold", 0.0))
            vae_ratio = (vae_err / max(vae_th, 1e-8)) if vae_th > 1e-8 else (1.0 if vae_anom else 0.0)

            kelly = float(np.clip(dsac_lev * meta_router.vol_scale(garch_vol_z, 0.0), 0.0, 1.0))
            fa = int(dsac_action)
            source = "DSAC_ONLY"
            long_raw = float(info.get("_long_raw", info.get("long_edge", 0.0)))
            short_raw = float(info.get("_short_raw", info.get("short_edge", 0.0)))

            if meta_router.pos is not None:
                live_unr = float(meta_router._net_pnl_frac(current_price))
                position_signal = str(info.get("position_signal", "HOLD"))
                position_reason = str(info.get("position_reason", ""))
                trend_exit_flag, _, trend_exit_reason = meta_router.update_trend_mismatch(processed_df, trend_signal)
                step_floor = meta_router.step_stop_floor()
                if live_unr <= step_floor:
                    fa = 0
                    kelly = 0.0
                    source = "DSAC_STEP_STOP" if meta_router.peak_equity >= 1.006 else "DSAC_ONLY_HARD_STOP"
                    step_stop_exit += 1
                elif meta_router.should_trailing_stop():
                    fa = 0
                    kelly = 0.0
                    source = "DSAC_ONLY_TRAILING_STOP"
                    trail_exit += 1
                elif meta_router.hold_count >= max(1, meta_router.dsac_only_max_hold):
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
            else:
                meta_router.trend_mismatch_streak = 0

            if meta_router.cooldown_bars_left > 0 and meta_router.pos is None and fa != 0:
                fa = 0
                kelly = 0.0
                source = "DSAC_ONLY_COOLDOWN"
                cooldown_block += 1

            if fa != 0 and meta_router.pos is None:
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

            if fa != 0 and meta_router.pos is None:
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

            meta_router.decrement_cooldown()

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
        return metrics, {"trades": trade_rows, "final_balance": balance}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", required=True)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--long-ckpt", default="data/ensemble/ckpt/best_dsac_long_agents.pth")
    ap.add_argument("--short-ckpt", default="data/ensemble/ckpt/best_dsac_short_agents.pth")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    df = _load_frame(args.csv_path, args.start, args.end)
    metrics, extra = _simulate_dual(df, args.long_ckpt, args.short_ckpt)

    payload = {
        "csv_path": args.csv_path,
        "start": str(df["timestamp"].iloc[0]) if len(df) else None,
        "end": str(df["timestamp"].iloc[-1]) if len(df) else None,
        "rows": int(len(df)),
        "long_ckpt": args.long_ckpt,
        "short_ckpt": args.short_ckpt,
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
