#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)


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
        "regime_bull": 0.0,
        "regime_bear": 0.0,
        "regime_chop": 0.0,
        "regime_whipsaw": 0.0,
        "m7_confidence": 0.0,
        "m7_qwidth": 0.0,
    }
    try:
        from features.schema import STATE_ALPHA as _STATE_ALPHA, STATE_SYNTH as _STATE_SYNTH
        for _c in list(_STATE_ALPHA) + list(_STATE_SYNTH):
            defaults.setdefault(str(_c), 0.0)
    except Exception:
        pass
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = df["close"] if default is None else default
    for col in ("open", "high", "low"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df["close"])
    return df


def _m7_defaults() -> dict[str, float]:
    return {
        "m7_trend_xgb_dn": 1.0 / 3.0,
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


def _regime_name(row: pd.Series) -> str:
    if _safe_float(row.get("regime_bull", 0.0)) >= 0.5:
        return "bull"
    if _safe_float(row.get("regime_bear", 0.0)) >= 0.5:
        return "bear"
    if _safe_float(row.get("regime_chop", 0.0)) >= 0.5:
        return "chop"
    if _safe_float(row.get("regime_whipsaw", 0.0)) >= 0.5:
        return "whipsaw"
    return "normal"


def _sizing(config: str, side: str, kelly: float, row: pd.Series) -> tuple[float, float, float]:
    k = float(np.clip(kelly, 0.0, 1.0))
    if config in {
        "current_coupled",
        "current_mild_guard",
        "current_quality_tilt",
        "current_adaptive_mild",
        "current_quality_guarded",
        "current_selective_boost",
        "current_drawdown_guard",
        "current_alpha_focus",
        "current_alpha_focus_strict",
    }:
        return k, 1.0, k

    regime = _regime_name(row)
    confidence = max(0.0, min(1.0, _safe_float(row.get("m7_confidence", 0.0), 0.0)))
    qwidth = max(0.0, _safe_float(row.get("m7_qwidth", 0.0), 0.0))
    vol_z = abs(_safe_float(row.get("volatility_z", 0.0), 0.0))
    trend_bonus = 0.0
    if (side == "LONG" and regime == "bull") or (side == "SHORT" and regime == "bear"):
        trend_bonus = 1.0
    elif regime in {"chop", "whipsaw"}:
        trend_bonus = -0.7
    elif regime == "normal":
        trend_bonus = 0.2
    vol_penalty = min(1.0, qwidth / 0.012 + vol_z / 4.0)

    if config == "split_conservative":
        fraction = np.clip(0.08 + 0.22 * k + 0.04 * confidence, 0.05, 0.25)
        leverage = np.clip(1.0 + 0.30 * trend_bonus + 0.25 * confidence - 0.30 * vol_penalty, 1.0, 1.6)
        exposure = min(float(fraction * leverage), 0.40)
    elif config == "split_cons_tight":
        fraction = np.clip(0.07 + 0.18 * k + 0.03 * confidence, 0.05, 0.22)
        leverage = np.clip(1.0 + 0.22 * trend_bonus + 0.18 * confidence - 0.32 * vol_penalty, 1.0, 1.40)
        exposure = min(float(fraction * leverage), 0.28)
    else:
        raise ValueError(config)

    if fraction > 0.0:
        leverage = exposure / fraction
    return float(fraction), float(leverage), float(exposure)


def simulate(df: pd.DataFrame, ckpt_path: str, config: str) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"replay_kelly_{config}_") as tmpdir:
        os.environ["DSAC_LIVE_STATE_PATH"] = os.path.join(tmpdir, "live_state.json")
        os.environ["FUSE_ADAPT_STATE_PATH"] = os.path.join(tmpdir, "adapt_state.json")

        from features.m7 import trend_signal_from_m7
        from features.schema import STATE_CONF as DSAC_STATE_CONF, STATE_PRED as DSAC_STATE_PRED
        from trading_bot import DSACSignalRouter, DSACTrendRouter

        dsac_router = DSACSignalRouter(model_path=ckpt_path)
        meta_router = DSACTrendRouter()
        meta_router.online_adapt = False
        meta_router._save_live_state = lambda *args, **kwargs: None

        def _sync() -> None:
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
        fractions: list[float] = []
        leverages: list[float] = []
        exposures: list[float] = []
        trade_rows: list[dict] = []
        recent_pnls: list[float] = []
        peak_balance = 1.0

        for i in range(60, len(df) - 1):
            processed_df = df.iloc[max(0, i - 300) : i + 1].copy()
            last_row = processed_df.iloc[-1]
            current_price = float(last_row["close"])
            next_price = float(df.iloc[i + 1]["close"])

            _sync()

            row_dict = last_row.to_dict()
            row_dict.setdefault("m7_prob_dn", _safe_float(row_dict.get("prob_dn", row_dict.get("m7_trend_xgb_dn", 0.0))))
            row_dict.setdefault("m7_prob_up", _safe_float(row_dict.get("prob_up", row_dict.get("m7_trend_xgb_up", 0.0))))
            for k, v in _m7_defaults().items():
                row_dict.setdefault(k, v)
            nf_preds = dict(row_dict)
            pred_fallback = _safe_float(nf_preds.get("pred_patchtst", 0.0))
            conf_fallback = float(np.clip(_safe_float(nf_preds.get("conf_patchtst", 0.5), 0.5), 0.0, 1.0))
            for c in DSAC_STATE_PRED:
                nf_preds.setdefault(c, pred_fallback)
            for c in DSAC_STATE_CONF:
                nf_preds.setdefault(c, conf_fallback)

            trend_signal = trend_signal_from_m7(row_dict)
            _, _, info, _, _ = dsac_router.decide(processed_df, nf_preds, m7_signal=trend_signal)

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
            if meta_router.pos is None:
                if action_val > pos_th:
                    fa, kelly = 1, min(abs_action, max_kelly)
                elif action_val < -pos_th:
                    fa, kelly = 2, min(abs_action, max_kelly)
            elif meta_router.pos == "LONG":
                live_unr = float(meta_router._net_pnl_frac(current_price))
                if force_close and live_unr <= -0.025:
                    fa, kelly = 0, 0.0
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
                elif abs_action < close_th:
                    fa, kelly = 0, 0.0
                elif action_val > flip_th:
                    fa, kelly = 1, min(abs_action, max_kelly) * flip_kelly_mult
                else:
                    fa, kelly = 2, min(abs_action, max_kelly)

            if meta_router.pos is not None:
                eq_curve.append(balance * (1.0 + meta_router._net_pnl_frac(current_price)))
            else:
                eq_curve.append(balance)

            prev_pos = meta_router.pos
            prev_hold = meta_router.hold_count
            prev_entry = meta_router.entry_price
            prev_exposure = meta_router.current_leverage
            prev_fraction = float(getattr(meta_router, "_debug_fraction", prev_exposure) or prev_exposure)
            prev_leverage = float(getattr(meta_router, "_debug_leverage_mult", 1.0) or 1.0)

            if fa in (1, 2):
                side = "LONG" if fa == 1 else "SHORT"
                if config.startswith("current_") and config != "current_coupled":
                    regime = _regime_name(last_row)
                    conf = _safe_float(last_row.get("m7_confidence", 0.0), 0.0)
                    qwidth = _safe_float(last_row.get("m7_qwidth", 0.0), 0.0)
                    vol_z = abs(_safe_float(last_row.get("volatility_z", 0.0), 0.0))
                    smf = _safe_float(last_row.get("smart_money_flow", 0.0), 0.0)
                    whale = _safe_float(last_row.get("whale_conviction", 0.0), 0.0)
                    funding_div = _safe_float(last_row.get("funding_price_divergence", 0.0), 0.0)
                    side_sign = 1.0 if side == "LONG" else -1.0
                    aligned = (side == "LONG" and regime == "bull") or (side == "SHORT" and regime == "bear")
                    bad = regime in {"chop", "whipsaw"} or qwidth > 0.010 or vol_z > 1.6 or conf < 0.40
                    very_bad = regime == "whipsaw" or qwidth > 0.014 or vol_z > 2.4
                    good = aligned and conf > 0.58 and qwidth < 0.0065 and vol_z < 1.0
                    flow_good = side_sign * (0.55 * smf + 0.35 * whale + 0.10 * funding_div) > 0.0
                    loss_streak = 0
                    for p in reversed(recent_pnls[-4:]):
                        if p < 0:
                            loss_streak += 1
                        else:
                            break
                    drawdown = 1.0 - (balance / max(peak_balance, 1e-8))
                    mult = 1.0
                    if config == "current_mild_guard":
                        if very_bad:
                            mult *= 0.90
                        elif bad:
                            mult *= 0.96
                        elif good and flow_good:
                            mult *= 1.03
                    elif config == "current_quality_tilt":
                        if very_bad:
                            mult *= 0.88
                        elif bad:
                            mult *= 0.95
                        elif good and flow_good:
                            mult *= 1.06
                        elif aligned and conf > 0.50:
                            mult *= 1.02
                    elif config == "current_adaptive_mild":
                        if very_bad:
                            mult *= 0.86
                        elif bad:
                            mult *= 0.94
                        elif good and flow_good:
                            mult *= 1.05
                        if loss_streak >= 2:
                            mult *= 0.90
                        if drawdown >= 0.04:
                            mult *= 0.92
                    elif config == "current_quality_guarded":
                        if very_bad:
                            mult *= 0.91
                        elif bad:
                            mult *= 0.97
                        elif good and flow_good:
                            mult *= 1.05
                        elif aligned and conf > 0.52:
                            mult *= 1.02
                        if loss_streak >= 2:
                            mult *= 0.95
                        if drawdown >= 0.03:
                            mult *= 0.96
                    elif config == "current_selective_boost":
                        if very_bad:
                            mult *= 0.94
                        elif bad:
                            mult *= 0.99
                        elif good and flow_good:
                            mult *= 1.08
                        elif aligned and conf > 0.55 and qwidth < 0.0075:
                            mult *= 1.03
                    elif config == "current_drawdown_guard":
                        if very_bad:
                            mult *= 0.92
                        elif bad:
                            mult *= 0.98
                        elif good and flow_good:
                            mult *= 1.04
                        if drawdown >= 0.025:
                            mult *= 0.95
                        if drawdown >= 0.04:
                            mult *= 0.93
                    elif config == "current_alpha_focus":
                        flow_score = side_sign * (0.55 * smf + 0.35 * whale + 0.10 * funding_div)
                        strong_base = kelly > 0.72
                        if very_bad:
                            mult *= 0.988
                        elif bad:
                            mult *= 0.995
                        if flow_score > 0.09 and kelly > 0.92:
                            mult *= 1.18
                        elif flow_score > 0.05 and conf > 0.62 and kelly > 0.82:
                            mult *= 1.10
                        elif strong_base and flow_score > 0.03 and (aligned or conf > 0.70):
                            mult *= 1.05
                        if flow_score < -0.04 and not aligned:
                            mult *= 0.985
                    elif config == "current_alpha_focus_strict":
                        flow_score = side_sign * (0.55 * smf + 0.35 * whale + 0.10 * funding_div)
                        if very_bad:
                            mult *= 0.992
                        elif bad:
                            mult *= 0.998
                        if flow_score > 0.07 and conf > 0.66 and kelly > 0.88 and qwidth < 0.010:
                            mult *= 1.12
                        elif flow_score > 0.04 and aligned and kelly > 0.78:
                            mult *= 1.05
                    exposure = float(np.clip(kelly * mult, 0.05, 1.0))
                    fraction, leverage_mult = exposure, 1.0
                else:
                    fraction, leverage_mult, exposure = _sizing(config, side, kelly, last_row)
                kelly = exposure
            else:
                fraction, leverage_mult, exposure = 0.0, 1.0, 0.0

            meta_router._update_pos(fa, next_price, kelly, trend_signal=trend_signal)

            if meta_router.pos is not None:
                meta_router._debug_fraction = float(fraction)
                meta_router._debug_leverage_mult = float(leverage_mult)
            else:
                meta_router._debug_fraction = 0.0
                meta_router._debug_leverage_mult = 1.0

            if prev_pos is None and meta_router.pos == "LONG":
                long_entries += 1
                fractions.append(float(fraction))
                leverages.append(float(leverage_mult))
                exposures.append(float(exposure))
            elif prev_pos is None and meta_router.pos == "SHORT":
                short_entries += 1
                fractions.append(float(fraction))
                leverages.append(float(leverage_mult))
                exposures.append(float(exposure))
            elif prev_pos is not None and meta_router.pos is not None and prev_pos != meta_router.pos:
                realized = float(meta_router.last_realized_pnl or 0.0)
                balance *= (1.0 + realized)
                trades += 1
                wins += int(realized > 0.0)
                hold_bars.append(prev_hold)
                trade_rows.append(
                    {
                        "ts": str(df.iloc[i + 1]["timestamp"]),
                        "side": prev_pos,
                        "entry_price": prev_entry,
                        "exit_price": next_price,
                        "exposure": prev_exposure,
                        "fraction": prev_fraction,
                        "leverage": prev_leverage,
                        "hold_bars": prev_hold,
                        "pnl_frac": realized,
                        "event": "flip",
                    }
                )
                recent_pnls.append(realized)
                peak_balance = max(peak_balance, balance)
                if meta_router.pos == "LONG":
                    long_entries += 1
                else:
                    short_entries += 1
                fractions.append(float(fraction))
                leverages.append(float(leverage_mult))
                exposures.append(float(exposure))
            elif prev_pos is not None and meta_router.pos is None:
                realized = float(meta_router.last_realized_pnl or 0.0)
                balance *= (1.0 + realized)
                trades += 1
                wins += int(realized > 0.0)
                hold_bars.append(prev_hold)
                trade_rows.append(
                    {
                        "ts": str(df.iloc[i + 1]["timestamp"]),
                        "side": prev_pos,
                        "entry_price": prev_entry,
                        "exit_price": next_price,
                        "exposure": prev_exposure,
                        "fraction": prev_fraction,
                        "leverage": prev_leverage,
                        "hold_bars": prev_hold,
                        "pnl_frac": realized,
                        "event": "exit",
                    }
                )
                recent_pnls.append(realized)
                peak_balance = max(peak_balance, balance)

        if meta_router.pos is not None:
            final_price = float(df.iloc[-1]["close"])
            realized = float(meta_router._net_pnl_frac(final_price))
            balance *= (1.0 + realized)
            trades += 1
            wins += int(realized > 0.0)
            hold_bars.append(meta_router.hold_count)
            trade_rows.append(
                {
                    "ts": str(df.iloc[-1]["timestamp"]),
                    "side": meta_router.pos,
                    "entry_price": meta_router.entry_price,
                    "exit_price": final_price,
                    "exposure": meta_router.current_leverage,
                    "fraction": float(getattr(meta_router, "_debug_fraction", meta_router.current_leverage) or meta_router.current_leverage),
                    "leverage": float(getattr(meta_router, "_debug_leverage_mult", 1.0) or 1.0),
                    "hold_bars": meta_router.hold_count,
                    "pnl_frac": realized,
                    "event": "final_mark",
                }
            )
            recent_pnls.append(realized)
            peak_balance = max(peak_balance, balance)
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
        return {
            "config": config,
            "metrics": asdict(metrics),
            "trade_count": len(trade_rows),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", required=True)
    ap.add_argument("--ckpt-path", default="/home/kbj20/crypto-scalping/data/ensemble/ckpt/best_dsac_agents.pth")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--out-json", default="")
    ap.add_argument(
        "--configs",
        default="current_coupled,current_mild_guard,current_quality_tilt,current_adaptive_mild,split_conservative,split_cons_tight",
        help="Comma-separated config names to run",
    )
    args = ap.parse_args()

    df = _load_frame(args.csv_path, args.start, args.end)
    configs = [c.strip() for c in str(args.configs or "").split(",") if c.strip()]
    results = [simulate(df, args.ckpt_path, cfg) for cfg in configs]
    payload = {
        "csv_path": args.csv_path,
        "ckpt_path": args.ckpt_path,
        "start": str(df["timestamp"].iloc[0]) if len(df) else None,
        "end": str(df["timestamp"].iloc[-1]) if len(df) else None,
        "rows": int(len(df)),
        "configs": configs,
        "results": results,
    }
    out_json = args.out_json or os.path.join(
        _ROOT_DIR,
        "data/ensemble/reports",
        f"backtest_replay_engine_kelly_leverage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
