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
    base_kelly: float
    entry_row: dict


@dataclass
class Metrics:
    pnl_pct: float
    mdd_pct: float
    trades: int
    wr_pct: float
    avg_exposure: float
    avg_multiplier: float


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
        "smart_money_flow": 0.0,
        "whale_conviction": 0.0,
        "funding_price_divergence": 0.0,
        "volatility_z": 0.0,
        "m7_confidence": 0.0,
        "m7_qwidth": 0.0,
        "regime_bull": 0.0,
        "regime_bear": 0.0,
        "regime_chop": 0.0,
        "regime_whipsaw": 0.0,
    }
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


def _extract_current_coupled_events(df: pd.DataFrame, ckpt_path: str) -> list[TradeEvent]:
    with tempfile.TemporaryDirectory(prefix="reweight_events_") as tmpdir:
        os.environ["DSAC_LIVE_STATE_PATH"] = os.path.join(tmpdir, "live_state.json")
        os.environ["FUSE_ADAPT_STATE_PATH"] = os.path.join(tmpdir, "adapt_state.json")

        from features.m7 import trend_signal_from_m7
        from features.schema import STATE_CONF as DSAC_STATE_CONF, STATE_PRED as DSAC_STATE_PRED
        from trading_bot import DSACSignalRouter, DSACTrendRouter

        router = DSACSignalRouter(model_path=ckpt_path)
        meta = DSACTrendRouter()
        meta.online_adapt = False
        meta._save_live_state = lambda *args, **kwargs: None

        def _sync() -> None:
            router.pos = meta.pos
            router.entry_price = meta.entry_price
            router.hold_count = meta.hold_count
            router.current_leverage = meta.current_leverage
            router.current_equity = meta.cur_equity
            router.peak_equity = meta.peak_equity

        pos_th = float(os.getenv("DSAC_PURE_RL_POS_TH", "0.12"))
        close_th = float(os.getenv("DSAC_PURE_RL_CLOSE_TH", "0.03"))
        flip_th = float(os.getenv("DSAC_PURE_RL_FLIP_TH", str(max(pos_th * 1.5, pos_th))))
        flip_kelly_mult = float(os.getenv("DSAC_PURE_RL_FLIP_KELLY_MULT", "0.85"))
        max_kelly = float(os.getenv("DSAC_PURE_RL_MAX_KELLY", "1.0"))
        force_close = str(os.getenv("DSAC_PURE_RL_FORCE_CLOSE", "false")).strip().lower() in {"1", "true", "yes", "on"}

        current_event: TradeEvent | None = None
        events: list[TradeEvent] = []

        for i in range(60, len(df) - 1):
            processed_df = df.iloc[max(0, i - 300) : i + 1].copy()
            last_row = processed_df.iloc[-1]
            current_price = float(last_row["close"])
            next_price = float(df.iloc[i + 1]["close"])
            next_ts = str(df.iloc[i + 1]["timestamp"])

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
            _, _, info, _, _ = router.decide(processed_df, nf_preds, m7_signal=trend_signal)

            action_val = float(info.get("primary_raw", info.get("raw_action", 0.0)))
            abs_action = abs(action_val)
            fa = 0
            kelly = 0.0
            if meta.pos is None:
                if action_val > pos_th:
                    fa, kelly = 1, min(abs_action, max_kelly)
                elif action_val < -pos_th:
                    fa, kelly = 2, min(abs_action, max_kelly)
            elif meta.pos == "LONG":
                live_unr = float(meta._net_pnl_frac(current_price))
                if force_close and live_unr <= -0.025:
                    fa, kelly = 0, 0.0
                elif abs_action < close_th:
                    fa, kelly = 0, 0.0
                elif action_val < -flip_th:
                    fa, kelly = 2, min(abs_action, max_kelly) * flip_kelly_mult
                else:
                    fa, kelly = 1, min(abs_action, max_kelly)
            else:
                live_unr = float(meta._net_pnl_frac(current_price))
                if force_close and live_unr <= -0.025:
                    fa, kelly = 0, 0.0
                elif abs_action < close_th:
                    fa, kelly = 0, 0.0
                elif action_val > flip_th:
                    fa, kelly = 1, min(abs_action, max_kelly) * flip_kelly_mult
                else:
                    fa, kelly = 2, min(abs_action, max_kelly)

            prev_pos = meta.pos
            prev_hold = meta.hold_count
            prev_entry = meta.entry_price
            meta._update_pos(fa, next_price, kelly, trend_signal)
            new_pos = meta.pos

            if prev_pos is None and new_pos in {"LONG", "SHORT"}:
                current_event = TradeEvent(
                    side=new_pos,
                    entry_ts=next_ts,
                    exit_ts="",
                    entry_price=float(next_price),
                    exit_price=0.0,
                    hold_bars=0,
                    base_kelly=float(kelly),
                    entry_row=dict(last_row.to_dict()),
                )
            elif prev_pos is not None and new_pos is not None and prev_pos != new_pos:
                if current_event is not None:
                    current_event.exit_ts = next_ts
                    current_event.exit_price = float(next_price)
                    current_event.hold_bars = int(prev_hold)
                    events.append(current_event)
                current_event = TradeEvent(
                    side=new_pos,
                    entry_ts=next_ts,
                    exit_ts="",
                    entry_price=float(next_price),
                    exit_price=0.0,
                    hold_bars=0,
                    base_kelly=float(kelly),
                    entry_row=dict(last_row.to_dict()),
                )
            elif prev_pos is not None and new_pos is None:
                if current_event is not None:
                    current_event.exit_ts = next_ts
                    current_event.exit_price = float(next_price)
                    current_event.hold_bars = int(prev_hold)
                    events.append(current_event)
                    current_event = None

        if current_event is not None:
            current_event.exit_ts = str(df.iloc[-1]["timestamp"])
            current_event.exit_price = float(df.iloc[-1]["close"])
            current_event.hold_bars = int(meta.hold_count)
            events.append(current_event)
        return events


def _multiplier(policy: str, ev: TradeEvent, recent_pnls: list[float], peak_balance: float, balance: float) -> float:
    row = ev.entry_row
    regime = _regime_name(row)
    conf = _safe_float(row.get("m7_confidence", 0.0), 0.0)
    qwidth = _safe_float(row.get("m7_qwidth", 0.0), 0.0)
    vol_z = abs(_safe_float(row.get("volatility_z", 0.0), 0.0))
    smf = _safe_float(row.get("smart_money_flow", 0.0), 0.0)
    whale = _safe_float(row.get("whale_conviction", 0.0), 0.0)
    funding_div = _safe_float(row.get("funding_price_divergence", 0.0), 0.0)
    side_sign = 1.0 if ev.side == "LONG" else -1.0
    aligned = (ev.side == "LONG" and regime == "bull") or (ev.side == "SHORT" and regime == "bear")
    risk_bad = regime in {"chop", "whipsaw"} or qwidth > 0.010 or vol_z > 1.6 or conf < 0.38
    very_bad = regime in {"whipsaw"} or qwidth > 0.014 or vol_z > 2.3
    quality_good = aligned and conf > 0.58 and qwidth < 0.0065 and vol_z < 1.1
    flow_good = side_sign * (0.55 * smf + 0.35 * whale + 0.10 * funding_div) > 0.0
    drawdown = 1.0 - (balance / max(peak_balance, 1e-8))
    loss_streak = 0
    for p in reversed(recent_pnls[-4:]):
        if p < 0:
            loss_streak += 1
        else:
            break

    if policy == "baseline":
        return 1.0
    if policy == "mild_guard":
        if very_bad:
            return 0.82
        if risk_bad:
            return 0.90
        if quality_good and flow_good:
            return 1.05
        return 1.0
    if policy == "quality_tilt":
        if very_bad:
            return 0.78
        if quality_good and flow_good:
            return 1.10
        if aligned and conf > 0.50:
            return 1.03
        if risk_bad:
            return 0.88
        return 1.0
    if policy == "adaptive_shield":
        m = 1.0
        if quality_good and flow_good:
            m *= 1.08
        if risk_bad:
            m *= 0.88
        if very_bad:
            m *= 0.82
        if loss_streak >= 2:
            m *= 0.80
        if drawdown >= 0.04:
            m *= 0.85
        return float(np.clip(m, 0.70, 1.12))
    if policy == "barbell":
        if quality_good and flow_good:
            return 1.15
        if very_bad:
            return 0.70
        if risk_bad:
            return 0.84
        return 0.98
    if policy == "anti_chop":
        if regime in {"chop", "whipsaw"}:
            return 0.72
        if qwidth > 0.010 or vol_z > 1.5:
            return 0.86
        if aligned and conf > 0.55:
            return 1.06
        return 1.0
    raise ValueError(policy)


def _evaluate(events: list[TradeEvent], policy: str) -> dict:
    fee = float(os.getenv("LIVE_FEE_RATE", "0.0005"))
    slip = float(os.getenv("LIVE_SLIP_RATE", "0.0002"))
    balance = 1.0
    peak = 1.0
    eq_curve = [balance]
    trades = wins = 0
    exposures: list[float] = []
    multipliers: list[float] = []
    recent_pnls: list[float] = []

    for ev in events:
        m = _multiplier(policy, ev, recent_pnls, peak, balance)
        exposure = float(np.clip(ev.base_kelly * m, 0.05, 1.0))
        pnl_frac = _net_trade_return(ev.side, ev.entry_price, ev.exit_price, exposure, fee, slip)
        balance *= (1.0 + pnl_frac)
        peak = max(peak, balance)
        eq_curve.append(balance)
        trades += 1
        wins += int(pnl_frac > 0.0)
        exposures.append(exposure)
        multipliers.append(m)
        recent_pnls.append(pnl_frac)

    eq = np.asarray(eq_curve, dtype=float)
    peak_arr = np.maximum.accumulate(eq)
    drawdown = (eq / np.maximum(peak_arr, 1e-12)) - 1.0
    metrics = Metrics(
        pnl_pct=float((balance - 1.0) * 100.0),
        mdd_pct=float(drawdown.min() * 100.0),
        trades=int(trades),
        wr_pct=float((wins / trades) * 100.0 if trades else 0.0),
        avg_exposure=float(np.mean(exposures) if exposures else 0.0),
        avg_multiplier=float(np.mean(multipliers) if multipliers else 0.0),
    )
    return {"policy": policy, "metrics": asdict(metrics)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", required=True)
    ap.add_argument("--ckpt-path", default="/home/kbj20/crypto-scalping/data/ensemble/ckpt/best_dsac_agents.pth")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    df = _load_frame(args.csv_path, args.start, args.end)
    events = _extract_current_coupled_events(df, args.ckpt_path)
    results = [_evaluate(events, p) for p in ["baseline", "mild_guard", "quality_tilt", "adaptive_shield", "barbell", "anti_chop"]]
    payload = {
        "csv_path": args.csv_path,
        "ckpt_path": args.ckpt_path,
        "start": str(df["timestamp"].iloc[0]) if len(df) else None,
        "end": str(df["timestamp"].iloc[-1]) if len(df) else None,
        "rows": int(len(df)),
        "fixed_events": len(events),
        "results": results,
    }
    out_json = args.out_json or os.path.join(
        _ROOT_DIR,
        "data/ensemble/reports",
        f"explore_replay_reweight_policies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nFixed events: {len(events)}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
