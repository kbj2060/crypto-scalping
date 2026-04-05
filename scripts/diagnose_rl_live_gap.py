#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from features.m7 import trend_signal_from_m7
from features.schema import STATE_CONF, STATE_PRED
from scripts.backtest_dual_specialist_dsac import _load_frame, _simulate_dual
from trading_bot import DSACSignalRouter
from ensemble.train_rl_dsac_agent import DSACCompactTradingEnv
from ensemble.train_rl_dsac_long_agent import LongSpecialistEnv
from ensemble.train_rl_dsac_short_agent import ShortSpecialistEnv


def _safe_cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _summary_diff(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    mae = np.array([r["mae"] for r in rows], dtype=np.float64)
    mxe = np.array([r["max_abs"] for r in rows], dtype=np.float64)
    cos = np.array([r["cosine"] for r in rows], dtype=np.float64)
    return {
        "count": int(len(rows)),
        "mae_mean": float(np.mean(mae)),
        "mae_p95": float(np.percentile(mae, 95)),
        "max_abs_mean": float(np.mean(mxe)),
        "max_abs_p95": float(np.percentile(mxe, 95)),
        "cosine_mean": float(np.mean(cos)),
        "cosine_p05": float(np.percentile(cos, 5)),
    }


def _diag_state_gap(df: pd.DataFrame, ckpt_long: str, ckpt_short: str, n_samples: int = 300) -> dict:
    work_df = df.copy()
    for c, v in {
        "m7_tp_price": 0.0,
        "m7_sl_price": 0.0,
        "m7_target_hold": 0.0,
        "current_spread": 0.0005,
        "spread": 0.0005,
        "m7_trend_xgb_dn": 1.0 / 3.0,
        "m7_trend_xgb_fl": 1.0 / 3.0,
        "m7_trend_xgb_up": 1.0 / 3.0,
        "m7_quality_pred": 0.0,
        "m7_hold_pred": 0.0,
        "m7_q10": 0.0,
        "m7_q50": 0.0,
        "m7_q90": 0.0,
        "m7_qwidth": 0.0,
        "m7_gmm_cluster": -1.0,
        "m7_gmm_conf": 0.0,
        "m7_gmm_vol_rank": 0.5,
        "m7_iso_score": 0.0,
        "m7_iso_anom": 0.0,
        "m7_vae_error": 0.0,
        "m7_vae_anom": 0.0,
        "m7_entry_short_offset": 0.0,
        "m7_entry_long_offset": 0.0,
        "m7_tp_offset": 0.0,
        "m7_sl_offset": 0.0,
    }.items():
        if c not in work_df.columns:
            work_df[c] = v

    dsac = DSACSignalRouter(long_path=ckpt_long, short_path=ckpt_short)
    env_p = DSACCompactTradingEnv(work_df.copy(), phase="val", side_mode="both")
    env_l = LongSpecialistEnv(work_df.copy(), phase="val")
    env_s = ShortSpecialistEnv(work_df.copy(), phase="val")

    pos_flat = {
        "type": None,
        "entry_price": 0.0,
        "unrealized": 0.0,
        "mdd": 0.0,
        "hold_count": 0.0,
        "hold_norm": 0.0,
        "margin_usage": 0.0,
    }

    start = min(120, max(0, len(df) - 1))
    end = max(start + 1, len(work_df) - 2)
    idxs = np.linspace(start, end, num=min(n_samples, max(1, end - start + 1)), dtype=int).tolist()

    rows_p: list[dict] = []
    rows_l: list[dict] = []
    rows_s: list[dict] = []

    for idx in idxs:
        row = work_df.iloc[idx].to_dict()
        row.setdefault("m7_prob_dn", float(row.get("m7_trend_xgb_dn", row.get("prob_dn", 1.0 / 3.0))))
        row.setdefault("m7_prob_fl", float(row.get("m7_trend_xgb_fl", row.get("prob_flat", 1.0 / 3.0))))
        row.setdefault("m7_prob_up", float(row.get("m7_trend_xgb_up", row.get("prob_up", 1.0 / 3.0))))

        tp = np.asarray(env_p._build_state(idx), dtype=np.float64)
        tl = np.asarray(env_l._build_state(idx), dtype=np.float64)
        ts = np.asarray(env_s._build_state(idx), dtype=np.float64)
        lp = np.asarray(dsac.primary_router._build_compact_state(row, pos_flat), dtype=np.float64)
        ll = np.asarray(dsac.long_router._build_compact_state(row, pos_flat), dtype=np.float64)
        ls = np.asarray(dsac.short_router._build_compact_state(row, pos_flat), dtype=np.float64)

        if tp.shape == lp.shape:
            d = np.abs(tp - lp)
            rows_p.append({"idx": idx, "mae": float(np.mean(d)), "max_abs": float(np.max(d)), "cosine": _safe_cos(tp, lp)})
        if tl.shape == ll.shape:
            d = np.abs(tl - ll)
            rows_l.append({"idx": idx, "mae": float(np.mean(d)), "max_abs": float(np.max(d)), "cosine": _safe_cos(tl, ll)})
        if ts.shape == ls.shape:
            d = np.abs(ts - ls)
            rows_s.append({"idx": idx, "mae": float(np.mean(d)), "max_abs": float(np.max(d)), "cosine": _safe_cos(ts, ls)})

    return {
        "primary": _summary_diff(rows_p),
        "long_specialist": _summary_diff(rows_l),
        "short_specialist": _summary_diff(rows_s),
    }


def _diag_action_dist(df: pd.DataFrame, ckpt_long: str, ckpt_short: str, max_rows: int = 5000) -> dict:
    dsac = DSACSignalRouter(long_path=ckpt_long, short_path=ckpt_short)
    n = min(len(df), max_rows)
    start = max(60, len(df) - n)

    c_final = Counter()
    c_primary = Counter()
    c_long = Counter()
    c_short = Counter()
    c_tuple = Counter()
    c_hold = Counter()
    direction = []
    agreement = []
    conviction = []

    for i in range(start, len(df) - 1):
        processed = df.iloc[max(0, i - 300): i + 1].copy()
        row = processed.iloc[-1].to_dict()
        row.setdefault("m7_prob_dn", float(row.get("m7_trend_xgb_dn", row.get("prob_dn", 1.0 / 3.0))))
        row.setdefault("m7_prob_fl", float(row.get("m7_trend_xgb_fl", row.get("prob_flat", 1.0 / 3.0))))
        row.setdefault("m7_prob_up", float(row.get("m7_trend_xgb_up", row.get("prob_up", 1.0 / 3.0))))
        row.setdefault("m7_trend_xgb_dn", float(row.get("m7_prob_dn", row.get("prob_dn", 1.0 / 3.0))))
        row.setdefault("m7_trend_xgb_fl", float(row.get("m7_prob_fl", row.get("prob_flat", 1.0 / 3.0))))
        row.setdefault("m7_trend_xgb_up", float(row.get("m7_prob_up", row.get("prob_up", 1.0 / 3.0))))
        for k, v in {
            "m7_action": 0.0,
            "m7_confidence": 0.0,
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
        }.items():
            row.setdefault(k, v)
        nf = dict(row)
        pred_fb = float(nf.get("pred_patchtst", 0.0))
        conf_fb = float(np.clip(float(nf.get("conf_patchtst", 0.5)), 0.0, 1.0))
        for col in STATE_PRED:
            nf.setdefault(col, pred_fb)
        for col in STATE_CONF:
            nf.setdefault(col, conf_fb)
        m7 = trend_signal_from_m7(row)
        fa, _, info, _, _ = dsac.decide(processed, nf, m7_signal=m7)
        pa = int(info.get("primary_action", 0))
        la = int(info.get("_long_action", 0))
        sa = int(info.get("_short_action", 0))
        c_final[fa] += 1
        c_primary[pa] += 1
        c_long[la] += 1
        c_short[sa] += 1
        c_tuple[(pa, la, sa)] += 1
        if int(fa) == 0:
            c_hold[str(info.get("hold_reason", "router_hold"))] += 1
        direction.append(float(info.get("direction_score", 0.0)))
        agreement.append(float(info.get("agreement", 0.0)))
        conviction.append(float(info.get("conviction", 0.0)))

    def _dist(x: np.ndarray) -> dict:
        if x.size == 0:
            return {}
        return {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "p05": float(np.percentile(x, 5)),
            "p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95)),
        }

    return {
        "samples": int(len(direction)),
        "final_action_counts": dict(c_final),
        "primary_action_counts": dict(c_primary),
        "long_action_counts": dict(c_long),
        "short_action_counts": dict(c_short),
        "top_action_tuples": [{"tuple": list(k), "count": int(v)} for k, v in c_tuple.most_common(12)],
        "top_hold_reasons": [{"reason": k, "count": int(v)} for k, v in c_hold.most_common(12)],
        "direction_score": _dist(np.asarray(direction, dtype=np.float64)),
        "agreement": _dist(np.asarray(agreement, dtype=np.float64)),
        "conviction": _dist(np.asarray(conviction, dtype=np.float64)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", required=True)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--long-ckpt", default="data/ensemble/ckpt/best_dsac_long_agents.pth")
    ap.add_argument("--short-ckpt", default="data/ensemble/ckpt/best_dsac_short_agents.pth")
    ap.add_argument("--mode", choices=["classic", "pure_rl"], default="pure_rl")
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--max-rows", type=int, default=5000)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    df = _load_frame(args.csv_path, args.start, args.end)
    state_gap = _diag_state_gap(df, args.long_ckpt, args.short_ckpt, n_samples=max(32, args.samples))
    replay_metrics, replay_extra = _simulate_dual(df, args.long_ckpt, args.short_ckpt, mode=args.mode)
    action_dist = _diag_action_dist(df, args.long_ckpt, args.short_ckpt, max_rows=max(500, args.max_rows))

    payload = {
        "csv_path": args.csv_path,
        "rows": int(len(df)),
        "start": str(df["timestamp"].iloc[0]) if len(df) else None,
        "end": str(df["timestamp"].iloc[-1]) if len(df) else None,
        "mode": args.mode,
        "diag_1_state_gap": state_gap,
        "diag_2_deterministic_replay": {
            "metrics": replay_metrics.__dict__,
            "final_balance": float(replay_extra.get("final_balance", 1.0)),
            "trade_count_logged": int(len(replay_extra.get("trades", []))),
        },
        "diag_3_action_distribution": action_dist,
    }

    out_json = args.out_json
    if not out_json:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_json = os.path.join("data/ensemble/metrics", f"rl_live_gap_diag_{ts}.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "diag_1_state_gap": payload["diag_1_state_gap"],
        "diag_2_metrics": payload["diag_2_deterministic_replay"]["metrics"],
        "diag_3_action_counts": payload["diag_3_action_distribution"]["final_action_counts"],
        "diag_3_top_hold_reasons": payload["diag_3_action_distribution"]["top_hold_reasons"][:5],
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
