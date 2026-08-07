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
class TradeEvent:
    side: str
    entry_ts: str
    exit_ts: str
    entry_price: float
    exit_price: float
    hold_bars: int
    entry_kelly: float
    entry_row: dict


@dataclass
class Metrics:
    pnl_pct: float
    mdd_pct: float
    trades: int
    wr_pct: float
    avg_exposure: float
    avg_fraction: float
    avg_leverage: float
    long_entries: int
    short_entries: int


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
    defaults = {"open": None, "high": None, "low": None, "volume": 1.0}
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


def _build_trade_events(df: pd.DataFrame, ckpt_path: str) -> list[TradeEvent]:
    with tempfile.TemporaryDirectory(prefix="split_events_") as tmpdir:
        os.environ["DSAC_LIVE_STATE_PATH"] = os.path.join(tmpdir, "live_state.json")
        os.environ["FUSE_ADAPT_STATE_PATH"] = os.path.join(tmpdir, "adapt_state.json")

        from features.m7 import trend_signal_from_m7
        from features.schema import STATE_CONF as DSAC_STATE_CONF, STATE_PRED as DSAC_STATE_PRED
        from trading_bot import DSACSignalRouter

        router = DSACSignalRouter(model_path=ckpt_path)
        pos_th = float(os.getenv("DSAC_PURE_RL_POS_TH", "0.15"))
        close_th = float(os.getenv("DSAC_PURE_RL_CLOSE_TH", "0.00"))
        flip_th = float(os.getenv("DSAC_PURE_RL_FLIP_TH", str(max(pos_th * 1.5, pos_th))))
        flip_kelly_mult = float(os.getenv("DSAC_PURE_RL_FLIP_KELLY_MULT", "0.85"))
        max_kelly = float(os.getenv("DSAC_PURE_RL_MAX_KELLY", "1.0"))
        force_close = str(os.getenv("DSAC_PURE_RL_FORCE_CLOSE", "false")).strip().lower() in {"1", "true", "yes", "on"}

        open_side: str | None = None
        entry_price = 0.0
        entry_ts = ""
        entry_row: dict | None = None
        entry_kelly = 0.0
        hold_bars = 0
        events: list[TradeEvent] = []

        for i in range(60, len(df) - 1):
            processed_df = df.iloc[max(0, i - 300) : i + 1].copy()
            last_row = processed_df.iloc[-1]
            current_price = float(last_row["close"])
            next_price = float(df.iloc[i + 1]["close"])
            next_ts = str(df.iloc[i + 1]["timestamp"])

            router.pos = open_side
            router.entry_price = entry_price if open_side else 0.0
            router.hold_count = hold_bars if open_side else 0
            router.current_leverage = 0.0
            router.current_equity = 1.0
            router.peak_equity = 1.0

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
            _, _, info, _, _ = router.decide(processed_df, nf_preds, m7_signal=trend_signal)

            action_val = float(info.get("primary_raw", info.get("raw_action", 0.0)))
            abs_action = abs(action_val)
            kelly = min(abs_action, max_kelly)
            fa = 0
            if open_side is None:
                if action_val > pos_th:
                    fa = 1
                elif action_val < -pos_th:
                    fa = 2
            elif open_side == "LONG":
                if force_close and False:
                    fa = 0
                elif abs_action < close_th:
                    fa = 0
                elif action_val < -flip_th:
                    fa = 2
                    kelly *= flip_kelly_mult
                else:
                    fa = 1
            else:
                if force_close and False:
                    fa = 0
                elif abs_action < close_th:
                    fa = 0
                elif action_val > flip_th:
                    fa = 1
                    kelly *= flip_kelly_mult
                else:
                    fa = 2

            if open_side is not None:
                hold_bars += 1

            target_side = "LONG" if fa == 1 else ("SHORT" if fa == 2 else None)

            if open_side is not None and target_side != open_side:
                events.append(
                    TradeEvent(
                        side=open_side,
                        entry_ts=entry_ts,
                        exit_ts=next_ts,
                        entry_price=float(entry_price),
                        exit_price=float(next_price),
                        hold_bars=int(hold_bars),
                        entry_kelly=float(entry_kelly),
                        entry_row=dict(entry_row or {}),
                    )
                )
                open_side = None
                entry_price = 0.0
                entry_ts = ""
                entry_row = None
                entry_kelly = 0.0
                hold_bars = 0

            if open_side is None and target_side is not None:
                open_side = target_side
                entry_price = float(next_price)
                entry_ts = next_ts
                entry_row = dict(last_row.to_dict())
                entry_kelly = float(kelly)
                hold_bars = 0

        if open_side is not None:
            events.append(
                TradeEvent(
                    side=open_side,
                    entry_ts=entry_ts,
                    exit_ts=str(df.iloc[-1]["timestamp"]),
                    entry_price=float(entry_price),
                    exit_price=float(df.iloc[-1]["close"]),
                    hold_bars=int(hold_bars),
                    entry_kelly=float(entry_kelly),
                    entry_row=dict(entry_row or {}),
                )
            )
        return events


def _regime_name(row: dict) -> str:
    if _safe_float(row.get("regime_bull", 0.0)) >= 0.5:
        return "bull"
    if _safe_float(row.get("regime_bear", 0.0)) >= 0.5:
        return "bear"
    if _safe_float(row.get("regime_chop", 0.0)) >= 0.5:
        return "chop"
    if _safe_float(row.get("regime_whipsaw", 0.0)) >= 0.5:
        return "whipsaw"
    return "normal"


def _sizing(config: str, side: str, kelly: float, row: dict) -> tuple[float, float, float]:
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
    k = float(np.clip(kelly, 0.0, 1.0))
    vol_penalty = min(1.0, qwidth / 0.012 + vol_z / 4.0)

    if config == "baseline_coupled":
        fraction = k
        leverage = 1.0
        exposure = fraction
    elif config == "split_conservative":
        fraction = np.clip(0.08 + 0.22 * k + 0.04 * confidence, 0.05, 0.25)
        leverage = np.clip(1.0 + 0.30 * trend_bonus + 0.25 * confidence - 0.30 * vol_penalty, 1.0, 1.6)
        exposure = min(float(fraction * leverage), 0.40)
    elif config == "split_cons_tight":
        fraction = np.clip(0.07 + 0.18 * k + 0.03 * confidence, 0.05, 0.22)
        leverage = np.clip(1.0 + 0.22 * trend_bonus + 0.18 * confidence - 0.32 * vol_penalty, 1.0, 1.40)
        exposure = min(float(fraction * leverage), 0.28)
    elif config == "split_cons_lowcap":
        fraction = np.clip(0.075 + 0.20 * k + 0.035 * confidence, 0.05, 0.235)
        leverage = np.clip(1.0 + 0.26 * trend_bonus + 0.20 * confidence - 0.31 * vol_penalty, 1.0, 1.50)
        exposure = min(float(fraction * leverage), 0.32)
    elif config == "split_cons_midcap":
        fraction = np.clip(0.085 + 0.24 * k + 0.045 * confidence, 0.05, 0.265)
        leverage = np.clip(1.0 + 0.32 * trend_bonus + 0.24 * confidence - 0.28 * vol_penalty, 1.0, 1.65)
        exposure = min(float(fraction * leverage), 0.38)
    elif config == "split_cons_conftrend":
        fraction = np.clip(0.07 + 0.19 * k + 0.07 * confidence, 0.05, 0.24)
        leverage = np.clip(1.0 + 0.36 * trend_bonus + 0.22 * confidence - 0.34 * vol_penalty, 1.0, 1.55)
        exposure = min(float(fraction * leverage), 0.34)
    elif config == "split_balanced":
        fraction = np.clip(0.10 + 0.28 * k + 0.05 * confidence, 0.06, 0.35)
        leverage = np.clip(1.1 + 0.45 * trend_bonus + 0.35 * confidence - 0.25 * vol_penalty, 1.0, 2.0)
        exposure = min(float(fraction * leverage), 0.60)
    elif config == "split_aggressive":
        fraction = np.clip(0.12 + 0.34 * k + 0.06 * confidence, 0.08, 0.42)
        leverage = np.clip(1.2 + 0.60 * trend_bonus + 0.45 * confidence - 0.20 * vol_penalty, 1.0, 2.4)
        exposure = min(float(fraction * leverage), 0.85)
    else:
        raise ValueError(config)

    if exposure <= 0.0:
        return 0.0, 1.0, 0.0
    if fraction > 0.0:
        leverage = exposure / fraction
    return float(fraction), float(leverage), float(exposure)


def _net_trade_return(side: str, entry_price: float, exit_price: float, exposure: float, fee: float, slip: float) -> float:
    if side == "LONG":
        entry_exec = entry_price * (1.0 + slip)
        exit_exec = exit_price * (1.0 - slip)
        gross = (exit_exec - entry_exec) / max(entry_exec, 1e-8)
    else:
        entry_exec = entry_price * (1.0 - slip)
        exit_exec = exit_price * (1.0 + slip)
        gross = (entry_exec - exit_exec) / max(abs(entry_exec), 1e-8)
    return float(gross * exposure - (2.0 * fee * exposure))


def _evaluate(events: list[TradeEvent], config: str) -> dict:
    fee = float(os.getenv("LIVE_FEE_RATE", "0.0005"))
    slip = float(os.getenv("LIVE_SLIP_RATE", "0.0002"))
    balance = 1.0
    eq_curve = [balance]
    trades = wins = long_entries = short_entries = 0
    exposures: list[float] = []
    fractions: list[float] = []
    leverages: list[float] = []

    for ev in events:
        fraction, leverage, exposure = _sizing(config, ev.side, ev.entry_kelly, ev.entry_row)
        pnl_frac = _net_trade_return(ev.side, ev.entry_price, ev.exit_price, exposure, fee, slip)
        balance *= (1.0 + pnl_frac)
        eq_curve.append(balance)
        trades += 1
        wins += int(pnl_frac > 0.0)
        exposures.append(exposure)
        fractions.append(fraction)
        leverages.append(leverage)
        if ev.side == "LONG":
            long_entries += 1
        else:
            short_entries += 1

    eq = np.asarray(eq_curve, dtype=float)
    peak = np.maximum.accumulate(eq)
    drawdown = (eq / np.maximum(peak, 1e-12)) - 1.0
    metrics = Metrics(
        pnl_pct=float((balance - 1.0) * 100.0),
        mdd_pct=float(drawdown.min() * 100.0),
        trades=int(trades),
        wr_pct=float((wins / trades) * 100.0 if trades else 0.0),
        avg_exposure=float(np.mean(exposures) if exposures else 0.0),
        avg_fraction=float(np.mean(fractions) if fractions else 0.0),
        avg_leverage=float(np.mean(leverages) if leverages else 0.0),
        long_entries=int(long_entries),
        short_entries=int(short_entries),
    )
    return {"config": config, "metrics": asdict(metrics), "trade_count": len(events)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", required=True)
    ap.add_argument("--ckpt-path", default="/home/kbj20/crypto-scalping/data/ensemble/ckpt/best_dsac_agents.pth")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    df = _load_frame(args.csv_path, args.start, args.end)
    events = _build_trade_events(df, args.ckpt_path)
    configs = [
        "baseline_coupled",
        "split_conservative",
        "split_cons_tight",
        "split_cons_lowcap",
        "split_cons_midcap",
        "split_cons_conftrend",
        "split_balanced",
        "split_aggressive",
    ]
    results = [_evaluate(events, cfg) for cfg in configs]
    payload = {
        "csv_path": args.csv_path,
        "ckpt_path": args.ckpt_path,
        "start": str(df["timestamp"].iloc[0]) if len(df) else None,
        "end": str(df["timestamp"].iloc[-1]) if len(df) else None,
        "rows": int(len(df)),
        "fixed_trade_timing_count": len(events),
        "results": results,
    }
    out_json = args.out_json or os.path.join(
        _ROOT_DIR,
        "data/ensemble/reports",
        f"backtest_fraction_leverage_split_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nFixed trade timing count: {len(events)}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
