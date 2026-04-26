#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from scripts.backtest_replay_engine_kelly_leverage import Metrics, _load_frame, _m7_defaults, _regime_name, _safe_float


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _controller(row: pd.Series, side: str, balance: float, peak_balance: float, recent_pnls: list[float], cfg: dict) -> float:
    regime = _regime_name(row)
    conf = _safe_float(row.get("m7_confidence", 0.0), 0.0)
    qwidth = _safe_float(row.get("m7_qwidth", 0.0), 0.0)
    vol_z = abs(_safe_float(row.get("volatility_z", 0.0), 0.0))
    smf = _safe_float(row.get("smart_money_flow", 0.0), 0.0)
    whale = _safe_float(row.get("whale_conviction", 0.0), 0.0)
    funding_div = _safe_float(row.get("funding_price_divergence", 0.0), 0.0)
    toxicity = _safe_float(row.get("shadow_toxicity_score", row.get("toxicity", 0.0)), 0.0)
    queue_collapse = _safe_float(row.get("shadow_queue_collapse", 0.0), 0.0)
    aftershock = _safe_float(row.get("shadow_aftershock_prob", row.get("aftershock", 0.0)), 0.0)
    regime_conf = _safe_float(row.get("shadow_regime_conf", 0.0), 0.0)
    side_sign = 1.0 if side == "LONG" else -1.0

    aligned = (side == "LONG" and regime == "bull") or (side == "SHORT" and regime == "bear")
    whipsaw = regime in {"whipsaw", "chop"}
    flow_score = side_sign * (0.55 * smf + 0.35 * whale + 0.10 * funding_div)
    drawdown = 1.0 - (balance / max(peak_balance, 1e-8))
    loss_streak = 0
    for p in reversed(recent_pnls[-4:]):
        if p < 0:
            loss_streak += 1
        else:
            break

    core = (
        cfg["aligned_w"] * (1.0 if aligned else -0.35)
        + cfg["conf_w"] * np.clip(conf - 0.50, -0.50, 0.50)
        + cfg["flow_w"] * np.clip(flow_score, -1.0, 1.0)
        + cfg["regime_conf_w"] * np.clip(regime_conf - 0.50, -0.50, 0.50)
        - cfg["qwidth_w"] * min(qwidth / 0.012, 1.5)
        - cfg["vol_w"] * min(vol_z / 2.5, 1.5)
        - cfg["tox_w"] * min(toxicity / 0.9, 1.5)
        - cfg["aftershock_w"] * min(aftershock / 0.8, 1.5)
        - cfg["queue_w"] * min(queue_collapse / 0.9, 1.5)
        - cfg["drawdown_w"] * min(drawdown / 0.06, 1.5)
        - cfg["loss_w"] * loss_streak
        - (cfg["whipsaw_penalty"] if whipsaw else 0.0)
    )

    lev = 1.0 + cfg["amp"] * _sigmoid(core + cfg["bias"])
    lev = min(lev, cfg["max_lev"])
    if conf > cfg["bonus_conf"] and flow_score > cfg["bonus_flow"] and aligned and drawdown < cfg["bonus_dd"] and toxicity < cfg["bonus_tox"] and qwidth < cfg["bonus_qwidth"]:
        lev = min(cfg["max_lev"], lev + cfg["bonus_add"])
    if drawdown >= cfg["cap_dd"]:
        lev = min(lev, cfg["dd_cap_lev"])
    if whipsaw:
        lev = min(lev, cfg["whipsaw_cap_lev"])
    return float(np.clip(lev, 1.0, cfg["max_lev"]))


def simulate(df: pd.DataFrame, ckpt_path: str, cfg: dict) -> dict:
    with tempfile.TemporaryDirectory(prefix="tune_lev_ctrl_") as tmpdir:
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

            if fa in (1, 2):
                side = "LONG" if fa == 1 else "SHORT"
                lev_mult = _controller(last_row, side, balance, peak_balance, recent_pnls, cfg)
                fraction = float(np.clip(kelly, 0.05, 1.0))
                exposure = float(np.clip(kelly * lev_mult, 0.05, lev_mult))
                kelly = exposure
            else:
                fraction, lev_mult, exposure = 0.0, 1.0, 0.0

            meta_router._update_pos(fa, next_price, kelly, trend_signal=trend_signal)
            meta_router._debug_fraction = float(fraction) if meta_router.pos is not None else 0.0
            meta_router._debug_leverage_mult = float(lev_mult) if meta_router.pos is not None else 1.0

            if prev_pos is None and meta_router.pos == "LONG":
                long_entries += 1
                fractions.append(float(fraction))
                leverages.append(float(lev_mult))
                exposures.append(float(exposure))
            elif prev_pos is None and meta_router.pos == "SHORT":
                short_entries += 1
                fractions.append(float(fraction))
                leverages.append(float(lev_mult))
                exposures.append(float(exposure))
            elif prev_pos is not None and meta_router.pos is not None and prev_pos != meta_router.pos:
                realized = float(meta_router.last_realized_pnl or 0.0)
                balance *= (1.0 + realized)
                trades += 1
                wins += int(realized > 0.0)
                hold_bars.append(prev_hold)
                recent_pnls.append(realized)
                peak_balance = max(peak_balance, balance)
                if meta_router.pos == "LONG":
                    long_entries += 1
                else:
                    short_entries += 1
                fractions.append(float(fraction))
                leverages.append(float(lev_mult))
                exposures.append(float(exposure))
            elif prev_pos is not None and meta_router.pos is None:
                realized = float(meta_router.last_realized_pnl or 0.0)
                balance *= (1.0 + realized)
                trades += 1
                wins += int(realized > 0.0)
                hold_bars.append(prev_hold)
                recent_pnls.append(realized)
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
        return {"config": cfg, "metrics": asdict(metrics)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", default="/home/kbj20/crypto-scalping/data/splits/year_oos/rl_training_2026_m7_supervised_redesign_clean.csv")
    ap.add_argument("--ckpt-path", default="/home/kbj20/crypto-scalping/data/ensemble/ckpt/best_dsac_agents.pth")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    df = _load_frame(args.csv_path, None, None)
    grid = [
        {
            "name": f"convex_tune_{i}",
            "aligned_w": a,
            "conf_w": c,
            "flow_w": f,
            "regime_conf_w": 0.30,
            "qwidth_w": qw,
            "vol_w": vw,
            "tox_w": tw,
            "aftershock_w": 0.65,
            "queue_w": 0.55,
            "drawdown_w": dw,
            "loss_w": 0.45,
            "whipsaw_penalty": wp,
            "amp": amp,
            "bias": bias,
            "max_lev": 2.0,
            "bonus_conf": bconf,
            "bonus_flow": bflow,
            "bonus_dd": bdd,
            "bonus_tox": 0.45,
            "bonus_qwidth": 0.006,
            "bonus_add": badd,
            "cap_dd": capdd,
            "dd_cap_lev": ddlev,
            "whipsaw_cap_lev": whlev,
        }
        for i, (a, c, f, qw, vw, tw, dw, wp, amp, bias, bconf, bflow, bdd, badd, capdd, ddlev, whlev) in enumerate(
            product(
                [1.20],
                [1.25],
                [0.90],
                [0.90],
                [0.85],
                [0.80],
                [1.00],
                [0.40],
                [1.10],
                [-0.05],
                [0.68],
                [0.05],
                [0.015, 0.025],
                [0.10, 0.18],
                [0.03],
                [1.25, 1.40],
                [1.02],
            )
        )
    ]

    results = []
    iterator = grid
    if tqdm is not None:
        iterator = tqdm(grid, desc="lev-controller-grid", ncols=100)
    for cfg in iterator:
        results.append(simulate(df, args.ckpt_path, cfg))

    results.sort(
        key=lambda x: (
            x["metrics"]["pnl_pct"] - 0.8 * abs(x["metrics"]["mdd_pct"]),
            x["metrics"]["pnl_pct"],
        ),
        reverse=True,
    )

    payload = {
        "csv_path": args.csv_path,
        "ckpt_path": args.ckpt_path,
        "rows": int(len(df)),
        "top10": results[:10],
        "results": results,
    }
    out_json = args.out_json or os.path.join(
        _ROOT_DIR,
        "data/ensemble/reports",
        f"tune_leverage_controller_oos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    print(json.dumps(payload["top10"][:3], indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
