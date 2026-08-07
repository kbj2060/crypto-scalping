#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from scripts.backtest_replay_engine_kelly_leverage import Metrics, _load_frame, _m7_defaults, _regime_name, _safe_float


def simulate_profile(df: pd.DataFrame, ckpt_path: str, profile: str) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"fixed2x_{profile}_") as tmpdir:
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

            regime = _regime_name(last_row)
            conf = _safe_float(last_row.get("m7_confidence", 0.0), 0.0)
            qwidth = _safe_float(last_row.get("m7_qwidth", 0.0), 0.0)
            vol_z = abs(_safe_float(last_row.get("volatility_z", 0.0), 0.0))
            toxicity = _safe_float(last_row.get("shadow_toxicity_score", last_row.get("toxicity", 0.0)), 0.0)
            queue_collapse = _safe_float(last_row.get("shadow_queue_collapse", 0.0), 0.0)

            fa = 0
            kelly = 0.0
            if meta_router.pos is None:
                if action_val > pos_th:
                    fa, kelly = 1, min(abs_action, max_kelly)
                elif action_val < -pos_th:
                    fa, kelly = 2, min(abs_action, max_kelly)
            elif meta_router.pos == "LONG":
                live_unr = float(meta_router._net_pnl_frac(current_price))
                if abs_action < close_th:
                    fa, kelly = 0, 0.0
                elif action_val < -flip_th:
                    fa, kelly = 2, min(abs_action, max_kelly) * flip_kelly_mult
                else:
                    fa, kelly = 1, min(abs_action, max_kelly)
                if profile in {"stop_loss_guard", "toxicity_exit", "whipsaw_filter"} and live_unr <= -0.030:
                    fa, kelly = 0, 0.0
                if profile in {"toxicity_exit", "whipsaw_filter"} and toxicity >= 0.80 and queue_collapse >= 0.60 and live_unr <= -0.010:
                    fa, kelly = 0, 0.0
            else:
                live_unr = float(meta_router._net_pnl_frac(current_price))
                if abs_action < close_th:
                    fa, kelly = 0, 0.0
                elif action_val > flip_th:
                    fa, kelly = 1, min(abs_action, max_kelly) * flip_kelly_mult
                else:
                    fa, kelly = 2, min(abs_action, max_kelly)
                if profile in {"stop_loss_guard", "toxicity_exit", "whipsaw_filter"} and live_unr <= -0.030:
                    fa, kelly = 0, 0.0
                if profile in {"toxicity_exit", "whipsaw_filter"} and toxicity >= 0.80 and queue_collapse >= 0.60 and live_unr <= -0.010:
                    fa, kelly = 0, 0.0

            if meta_router.pos is not None:
                eq_curve.append(balance * (1.0 + meta_router._net_pnl_frac(current_price)))
            else:
                eq_curve.append(balance)

            prev_pos = meta_router.pos
            prev_hold = meta_router.hold_count

            # Entry veto only, leverage itself stays fixed at 2x
            if prev_pos is None and fa in (1, 2):
                if profile == "whipsaw_filter":
                    if regime in {"whipsaw", "chop"} and (qwidth > 0.009 or conf < 0.42 or vol_z > 1.5):
                        fa, kelly = 0, 0.0
                elif profile == "trend_confirm":
                    entering_long = fa == 1
                    aligned = (entering_long and regime == "bull") or ((not entering_long) and regime == "bear")
                    if not aligned and (conf < 0.52 or qwidth > 0.0085):
                        fa, kelly = 0, 0.0

            if fa in (1, 2):
                exposure = float(np.clip(kelly * 2.0, 0.05, 2.0))
                fraction = float(np.clip(kelly, 0.05, 1.0))
                leverage_mult = 2.0
                kelly = exposure
            else:
                fraction, leverage_mult, exposure = 0.0, 1.0, 0.0

            meta_router._update_pos(fa, next_price, kelly, trend_signal=trend_signal)
            meta_router._debug_fraction = float(fraction) if meta_router.pos is not None else 0.0
            meta_router._debug_leverage_mult = float(leverage_mult) if meta_router.pos is not None else 1.0

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
                peak_balance = max(peak_balance, balance)

        if meta_router.pos is not None:
            final_price = float(df.iloc[-1]["close"])
            realized = float(meta_router._net_pnl_frac(final_price))
            balance *= (1.0 + realized)
            trades += 1
            wins += int(realized > 0.0)
            hold_bars.append(meta_router.hold_count)
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
        return {"profile": profile, "metrics": asdict(metrics)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", default="/home/kbj20/crypto-scalping/data/splits/year_oos/rl_training_2026_m7_supervised_redesign_clean.csv")
    ap.add_argument("--ckpt-path", default="/home/kbj20/crypto-scalping/data/ensemble/ckpt/best_dsac_agents.pth")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    df = _load_frame(args.csv_path, None, None)
    profiles = ["plain_2x", "stop_loss_guard", "toxicity_exit", "whipsaw_filter", "trend_confirm"]
    results = [simulate_profile(df, args.ckpt_path, p) for p in profiles]
    payload = {
        "csv_path": args.csv_path,
        "ckpt_path": args.ckpt_path,
        "rows": int(len(df)),
        "results": results,
    }
    out_json = args.out_json or os.path.join(
        _ROOT_DIR,
        "data/ensemble/reports",
        f"explore_mdd_guards_fixed_2x_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
