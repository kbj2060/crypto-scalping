"""Rev7 maker-first execution for the SOL survey
(docs/experiments/sol_dl_rl_architecture_survey_rev7_maker_exec_20260808.json).

Frozen inputs: the saved unconditional LGBM control booster
(tmp/sol_dl_rl_survey_20260807/lgbm_cheapgate/lgbm_model.txt) and the TB label's tp/sl moves.
New input: SOLUSDT 1m klines (scripts/download_klines_sol_1m_20260808.py).

Fill model (ported from scripts/stage0_kappa1_maker_fill_audit_20260807.py, conservative):
  post-only limit at the decision 5m bar CLOSE, active from the next 1m bar; if that 1m bar's
  OPEN already crosses the limit -> NO FILL (post-only reject); otherwise filled only when a 1m
  bar trades strictly THROUGH the limit (buy: low < limit, sell: high > limit) within a
  15-minute cancel window; no queue-position credit.

Filled trades enter at the limit price and run TP/SL (2.5/1.2 sigma off the limit) bar-by-bar on
1m data, 1440-minute horizon, SL before TP in every bar, non-overlapping single position.
Cost: 7bps roundtrip (2 maker entry + 5 taker exit) vs the 10bps taker baseline.

Per rule we also record the taker-counterfactual PnL of the UNFILLED entries (adverse-selection
kill signal) and the fill rate (<50% kills the line).

Usage: --stage {val, oos}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    RAW_LEVEL_COLS, PANEL_PATH, LABEL_PATH, ENTRY_RULES,
    TRAIN_END, VAL_START, VAL_END, OOS_START, OOS_END,
    MARGIN_FRACTION, LEVERAGE, side_state_from_proba, replay as taker_replay,
)

OUT_DIR = ROOT / "tmp/sol_dl_rl_survey_20260807/maker_exec_rev7"
LGBM_MODEL = ROOT / "tmp/sol_dl_rl_survey_20260807/lgbm_cheapgate/lgbm_model.txt"
M1_PATH = ROOT / "binance_data/klines/SOLUSDT/SOLUSDT-1m-api.csv"

NOTIONAL = MARGIN_FRACTION * LEVERAGE
MAKER_ROUNDTRIP = 0.0007  # 2bps maker entry + 5bps taker exit, on price move
TAKER_ROUNDTRIP = 0.0010
CANCEL_MIN = 15
HORIZON_MIN = 288 * 5
FILL_KILL = 0.50


def maker_replay(dec_rows, sides, limits, tp_m, sl_m, m1_open, m1_high, m1_low, m1_close, dec_to_m1):
    """Chronological non-overlapping maker replay on 1m bars. Returns summary + per-decision fill
    flags for the adverse-selection counterfactual."""
    n1 = len(m1_open)
    cash = 1.0
    equity = [1.0]
    occupied_until = -1  # 1m index
    filled_flags = np.zeros(len(dec_rows), dtype=bool)
    attempted = 0
    n_tp = n_sl = n_to = 0
    for k in range(len(dec_rows)):
        j0 = dec_to_m1[k]  # first 1m bar after decision close
        if j0 < 0 or j0 >= n1:
            continue
        if j0 <= occupied_until:
            continue
        attempted += 1
        s = sides[k]
        L = limits[k]
        # post-only reject if first bar's open already crosses
        if (s > 0 and m1_open[j0] < L) or (s < 0 and m1_open[j0] > L):
            continue
        fill_j = -1
        for j in range(j0, min(j0 + CANCEL_MIN, n1)):
            if s > 0 and m1_low[j] < L:
                fill_j = j
                break
            if s < 0 and m1_high[j] > L:
                fill_j = j
                break
        if fill_j < 0:
            continue
        filled_flags[k] = True
        tp_lvl = L * (1.0 + s * tp_m[k])
        sl_lvl = L * (1.0 - s * sl_m[k])
        final_j = min(fill_j + HORIZON_MIN - 1, n1 - 1)
        move, code, exit_j = None, "timeout", final_j
        for j in range(fill_j, final_j + 1):
            if s > 0:
                if m1_low[j] <= sl_lvl:
                    move, code, exit_j = -sl_m[k], "sl", j
                    break
                if m1_high[j] >= tp_lvl:
                    move, code, exit_j = tp_m[k], "tp", j
                    break
            else:
                if m1_high[j] >= sl_lvl:
                    move, code, exit_j = -sl_m[k], "sl", j
                    break
                if m1_low[j] <= tp_lvl:
                    move, code, exit_j = tp_m[k], "tp", j
                    break
        if move is None:
            move = (m1_close[final_j] / L - 1.0) * s
        r = move * NOTIONAL - MAKER_ROUNDTRIP * NOTIONAL
        cash *= 1.0 + r
        equity.append(cash)
        occupied_until = exit_j
        n_tp += code == "tp"
        n_sl += code == "sl"
        n_to += code == "timeout"
    equity = np.array(equity)
    running_max = np.maximum.accumulate(equity)
    n_fills = int(filled_flags.sum())
    return {
        "n_attempted": int(attempted),
        "n_trades": n_fills,
        "fill_rate": float(n_fills / attempted) if attempted else 0.0,
        "pnl_pct": float((cash - 1.0) * 100.0),
        "mdd_pct": float(((equity - running_max) / running_max).min() * 100.0) if len(equity) > 1 else 0.0,
        "n_tp": int(n_tp), "n_sl": int(n_sl), "n_timeout": int(n_to),
    }, filled_flags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["val", "oos"], required=True)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    labels = pd.read_parquet(LABEL_PATH)
    tp_moves = labels["tp_move"].to_numpy(dtype=np.float64)
    sl_moves = labels["sl_move"].to_numpy(dtype=np.float64)
    feat_cols = [c for c in panel.columns if c != "timestamp" and c not in RAW_LEVEL_COLS]
    x = panel[feat_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    ts = panel["timestamp"]
    close5 = panel["close"].to_numpy(dtype=np.float64)

    m1 = pd.read_csv(M1_PATH, usecols=["timestamp", "open", "high", "low", "close"])
    m1["timestamp"] = pd.to_datetime(m1["timestamp"])
    m1 = m1.sort_values("timestamp").reset_index(drop=True)
    m1_ts = m1["timestamp"].to_numpy()
    m1_open = m1["open"].to_numpy(dtype=np.float64)
    m1_high = m1["high"].to_numpy(dtype=np.float64)
    m1_low = m1["low"].to_numpy(dtype=np.float64)
    m1_close = m1["close"].to_numpy(dtype=np.float64)

    booster = lgb.Booster(model_file=str(LGBM_MODEL))
    eval_mask = ((ts >= VAL_START) & (ts <= VAL_END)).to_numpy() if args.stage == "val" else ((ts >= OOS_START) & (ts <= OOS_END)).to_numpy()
    e_idx = np.flatnonzero(eval_mask)
    proba = booster.predict(x[e_idx])
    months = ts.dt.to_period("M").astype(str).to_numpy()

    results = []
    for rule in ENTRY_RULES:
        side_e = side_state_from_proba(proba, rule["threshold"])
        keep = (side_e != 0) & np.isfinite(tp_moves[e_idx]) & np.isfinite(sl_moves[e_idx])
        dec_rows = e_idx[keep]
        sides = side_e[keep].astype(np.float64)
        limits = close5[dec_rows]  # decision 5m bar close
        tp_m = tp_moves[dec_rows]
        sl_m = sl_moves[dec_rows]
        # first 1m bar strictly after the decision 5m bar's close time (bar open time + 5m)
        dec_close_times = (ts.iloc[dec_rows] + pd.Timedelta(minutes=5)).to_numpy()
        dec_to_m1 = np.searchsorted(m1_ts, dec_close_times, side="left")
        dec_to_m1[dec_to_m1 >= len(m1_ts)] = -1

        summary, filled = maker_replay(dec_rows, sides, limits, tp_m, sl_m, m1_open, m1_high, m1_low, m1_close, dec_to_m1)

        # adverse-selection counterfactual: taker replay of the UNFILLED decisions only
        unfilled_rows = dec_rows[~filled]
        side_full = np.zeros(len(panel), dtype=np.int64)
        side_full[unfilled_rows] = sides[~filled].astype(np.int64)
        cf = taker_replay(panel, side_full, tp_moves, sl_moves, eval_mask)
        # monthly stability of the maker replay
        mon = {}
        for m in sorted(set(months[dec_rows])):
            sub = months[dec_rows] == m
            msum, _ = maker_replay(dec_rows[sub], sides[sub], limits[sub], tp_m[sub], sl_m[sub], m1_open, m1_high, m1_low, m1_close, dec_to_m1[sub])
            mon[m] = msum["pnl_pct"]
        n_pos_m = sum(v > 0 for v in mon.values())
        rec = {"rule": rule["name"], "threshold": rule["threshold"], **summary,
               "unfilled_taker_counterfactual_pnl_pct": cf.get("pnl_pct", 0.0),
               "unfilled_n_trades": cf.get("n_trades", 0),
               "monthly": mon, "n_pos_months": int(n_pos_m)}
        results.append(rec)
        print(json.dumps({k: rec[k] for k in ("rule", "n_trades", "fill_rate", "pnl_pct", "mdd_pct", "n_pos_months", "unfilled_taker_counterfactual_pnl_pct")}), flush=True)

    if args.stage == "val":
        eligible = [r for r in results if r["n_trades"] >= 15 and r["pnl_pct"] > 0 and r["n_pos_months"] >= 3 and r["fill_rate"] >= FILL_KILL]
        best = max(eligible, key=lambda r: r["pnl_pct"]) if eligible else None
        out = {"stage": "val", "results": results,
               "selected": None if best is None else {k: best[k] for k in ("rule", "threshold", "pnl_pct", "n_trades", "fill_rate", "mdd_pct", "n_pos_months", "unfilled_taker_counterfactual_pnl_pct")},
               "earns_oos_read": best is not None}
        (OUT_DIR / "val_results.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({"selected": out["selected"], "earns_oos_read": out["earns_oos_read"]}, indent=2))
    else:
        prior = json.loads((OUT_DIR / "val_results.json").read_text())
        if not prior.get("earns_oos_read"):
            print(json.dumps({"oos": "REFUSED -- maker VAL gate failed"}))
            return 1
        sel_rule = prior["selected"]["rule"]
        rec = next(r for r in results if r["rule"] == sel_rule)
        rec["adopted"] = bool(rec["pnl_pct"] > 0)
        (OUT_DIR / "oos_results.json").write_text(json.dumps({"stage": "oos", **rec}, indent=2))
        print(json.dumps({k: rec[k] for k in ("rule", "n_trades", "fill_rate", "pnl_pct", "mdd_pct", "adopted")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
