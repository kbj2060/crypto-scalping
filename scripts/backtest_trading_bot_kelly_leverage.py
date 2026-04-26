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
class Position:
    side: str
    entry_price: float
    fraction: float
    leverage: float
    exposure: float
    hold_bars: int = 0


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
        "nif_whale": 0.0,
        "sig_whale": 0.0,
        "m7_confidence": 0.0,
        "m7_qwidth": 0.0,
        "volatility_z": 0.0,
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
    k = float(np.clip(kelly, 0.0, 1.0))

    if config == "current_coupled":
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
    else:
        raise ValueError(config)

    if exposure <= 0.0:
        return 0.0, 1.0, 0.0
    if fraction > 0.0:
        leverage = exposure / fraction
    return float(fraction), float(leverage), float(exposure)


def simulate(df: pd.DataFrame, ckpt_path: str, config: str) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"bot_kelly_{config}_") as tmpdir:
        os.environ["DSAC_LIVE_STATE_PATH"] = os.path.join(tmpdir, "live_state.json")
        os.environ["FUSE_ADAPT_STATE_PATH"] = os.path.join(tmpdir, "adapt_state.json")

        from features.m7 import trend_signal_from_m7
        from features.schema import STATE_CONF as DSAC_STATE_CONF, STATE_PRED as DSAC_STATE_PRED
        from trading_bot import DSACSignalRouter

        router = DSACSignalRouter(model_path=ckpt_path)
        fee = float(os.getenv("LIVE_FEE_RATE", "0.0005"))
        slip = float(os.getenv("LIVE_SLIP_RATE", "0.0002"))
        pos_th = float(os.getenv("DSAC_PURE_RL_POS_TH", "0.15"))
        close_th = float(os.getenv("DSAC_PURE_RL_CLOSE_TH", "0.00"))
        flip_th = float(os.getenv("DSAC_PURE_RL_FLIP_TH", str(max(pos_th * 1.5, pos_th))))
        flip_kelly_mult = float(os.getenv("DSAC_PURE_RL_FLIP_KELLY_MULT", "0.85"))
        max_kelly = float(os.getenv("DSAC_PURE_RL_MAX_KELLY", "1.0"))
        force_close = str(os.getenv("DSAC_PURE_RL_FORCE_CLOSE", "false")).strip().lower() in {"1", "true", "yes", "on"}

        pos: Position | None = None
        balance = 1.0
        eq_curve = [balance]
        trades = wins = 0
        long_entries = short_entries = 0
        hold_bars: list[int] = []
        fractions: list[float] = []
        leverages: list[float] = []
        exposures: list[float] = []
        trade_rows: list[dict] = []

        for i in range(60, len(df) - 1):
            processed_df = df.iloc[max(0, i - 300) : i + 1].copy()
            last_row = processed_df.iloc[-1]
            current_price = float(last_row["close"])
            next_price = float(df.iloc[i + 1]["close"])

            router.pos = pos.side if pos else None
            router.entry_price = pos.entry_price if pos else 0.0
            router.hold_count = pos.hold_bars if pos else 0
            router.current_leverage = pos.exposure if pos else 0.0
            router.current_equity = 1.0
            router.peak_equity = 1.0

            row_dict = last_row.to_dict()
            row_dict.setdefault("m7_prob_dn", _safe_float(row_dict.get("prob_dn", row_dict.get("m7_trend_xgb_dn", 0.0))))
            row_dict.setdefault("m7_prob_fl", _safe_float(row_dict.get("prob_flat", row_dict.get("m7_trend_xgb_fl", 0.0))))
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
            base_kelly = min(abs_action, max_kelly)

            if pos is not None:
                pos.hold_bars += 1
                eq_curve.append(balance * (1.0 + _net_trade_return(pos.side, pos.entry_price, current_price, pos.exposure, fee, slip)))
            else:
                eq_curve.append(balance)

            fa = 0
            used_kelly = base_kelly
            if pos is None:
                if action_val > pos_th:
                    fa = 1
                elif action_val < -pos_th:
                    fa = 2
            elif pos.side == "LONG":
                live_unr = _net_trade_return("LONG", pos.entry_price, current_price, pos.exposure, fee, slip)
                if force_close and live_unr <= -0.025:
                    fa = 0
                elif abs_action < close_th:
                    fa = 0
                elif action_val < -flip_th:
                    fa = 2
                    used_kelly *= flip_kelly_mult
                else:
                    fa = 1
            else:
                live_unr = _net_trade_return("SHORT", pos.entry_price, current_price, pos.exposure, fee, slip)
                if force_close and live_unr <= -0.025:
                    fa = 0
                elif abs_action < close_th:
                    fa = 0
                elif action_val > flip_th:
                    fa = 1
                    used_kelly *= flip_kelly_mult
                else:
                    fa = 2

            target_side = "LONG" if fa == 1 else ("SHORT" if fa == 2 else None)
            prev_side = pos.side if pos else None

            if pos is not None and target_side != pos.side:
                realized = _net_trade_return(pos.side, pos.entry_price, next_price, pos.exposure, fee, slip)
                balance *= (1.0 + realized)
                trades += 1
                wins += int(realized > 0.0)
                hold_bars.append(pos.hold_bars)
                trade_rows.append(
                    {
                        "ts": str(df.iloc[i + 1]["timestamp"]),
                        "side": pos.side,
                        "entry_price": pos.entry_price,
                        "exit_price": next_price,
                        "fraction": pos.fraction,
                        "leverage": pos.leverage,
                        "exposure": pos.exposure,
                        "hold_bars": pos.hold_bars,
                        "pnl_frac": realized,
                        "event": "flip" if target_side else "exit",
                    }
                )
                pos = None

            if pos is None and target_side is not None:
                fraction, leverage, exposure = _sizing(config, target_side, used_kelly, last_row)
                if exposure > 0.0:
                    pos = Position(
                        side=target_side,
                        entry_price=float(next_price),
                        fraction=fraction,
                        leverage=leverage,
                        exposure=exposure,
                        hold_bars=0,
                    )
                    fractions.append(fraction)
                    leverages.append(leverage)
                    exposures.append(exposure)
                    if prev_side != target_side:
                        if target_side == "LONG":
                            long_entries += 1
                        else:
                            short_entries += 1

        if pos is not None:
            final_price = float(df.iloc[-1]["close"])
            realized = _net_trade_return(pos.side, pos.entry_price, final_price, pos.exposure, fee, slip)
            balance *= (1.0 + realized)
            trades += 1
            wins += int(realized > 0.0)
            hold_bars.append(pos.hold_bars)
            trade_rows.append(
                {
                    "ts": str(df.iloc[-1]["timestamp"]),
                    "side": pos.side,
                    "entry_price": pos.entry_price,
                    "exit_price": final_price,
                    "fraction": pos.fraction,
                    "leverage": pos.leverage,
                    "exposure": pos.exposure,
                    "hold_bars": pos.hold_bars,
                    "pnl_frac": realized,
                    "event": "final_mark",
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
    args = ap.parse_args()

    df = _load_frame(args.csv_path, args.start, args.end)
    configs = ["current_coupled", "split_conservative", "split_cons_tight"]
    results = [simulate(df, args.ckpt_path, cfg) for cfg in configs]
    payload = {
        "csv_path": args.csv_path,
        "ckpt_path": args.ckpt_path,
        "start": str(df["timestamp"].iloc[0]) if len(df) else None,
        "end": str(df["timestamp"].iloc[-1]) if len(df) else None,
        "rows": int(len(df)),
        "results": results,
    }
    out_json = args.out_json or os.path.join(
        _ROOT_DIR,
        "data/ensemble/reports",
        f"backtest_trading_bot_kelly_leverage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
