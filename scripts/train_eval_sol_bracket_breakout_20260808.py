"""Rev5 volatility-breakout bracket line for the SOL survey
(docs/experiments/sol_dl_rl_architecture_survey_rev5_bracket_20260808.json).

Decision primitive: at bar i, arm stop-entries at ref*(1 +/- b*sigma_i) where ref = open[i+1] and
sigma_i is the label family's 12-bar-cumret dispersion (causal). If a stop is touched within W
bars the market has chosen the side; the trade then runs the standard TB exit (TP 2.5 sigma_i /
SL 1.2 sigma_i / 288-bar horizon from entry). The learned component is only a binary continuation
gate P(TP-first | triggered setup).

Stages:
  --stage stage0  mechanical replay, no model: raw ungated economics + oracle-gated ceiling for
                  each (b, W) on TRAIN and VAL (OOS untouched)
  --stage val     train per-(b,W) LightGBM gates on purged train, select (b, W, thr) on VAL with
                  the pre-registered monthly-stability screen
  --stage oos     single frozen OOS read of the selected config

Conventions: same-bar double-trigger setups are skipped (deterministic, conservative); SL is
checked before TP in every bar including the entry bar; entries fill at the stop price; costs are
the standard 10bps roundtrip on notional; non-overlapping single position.
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
    RAW_LEVEL_COLS, PANEL_PATH, LABEL_PATH,
    TRAIN_END, VAL_START, VAL_END, OOS_START, OOS_END,
)

OUT_DIR = ROOT / "tmp/sol_dl_rl_survey_20260807/bracket_rev5"
HORIZON = 288
TP_MULT, SL_MULT = 2.5, 1.2
MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE = 0.30, 3.0, 0.0010
NOTIONAL = MARGIN_FRACTION * LEVERAGE
B_GRID = [0.75, 1.0]
W_GRID = [6, 12]
THR_GRID = [0.45, 0.50, 0.55, 0.60]
SEED = 903174

# outcome codes
NO_TRIGGER, TP, SL, TIMEOUT, AMBIGUOUS = 0, 1, 2, 3, 4


def build_bracket_outcomes(open_v, high_v, low_v, close_v, sigma, b: float, W: int):
    """Per-bar mechanical bracket outcome, independent of position overlap (for labels/oracle).
    Returns outcome code, side (+1/-1/0), entry_offset (bars from decision bar to entry bar),
    exit_offset (bars from decision bar to exit bar), price_move (signed, on entry price)."""
    n = len(open_v)
    outcome = np.zeros(n, dtype=np.int8)
    side = np.zeros(n, dtype=np.int8)
    entry_off = np.full(n, -1, dtype=np.int32)
    exit_off = np.full(n, -1, dtype=np.int32)
    price_move = np.full(n, np.nan)

    for i in range(n - 2):
        s = sigma[i]
        if not np.isfinite(s) or s <= 0:
            continue
        ref = open_v[i + 1]
        up = ref * (1.0 + b * s)
        dn = ref * (1.0 - b * s)
        trig_side, trig_j = 0, -1
        last_trig = min(i + W, n - 1)
        for j in range(i + 1, last_trig + 1):
            hit_up = high_v[j] >= up
            hit_dn = low_v[j] <= dn
            if hit_up and hit_dn:
                trig_side, trig_j = 9, j  # ambiguous
                break
            if hit_up:
                trig_side, trig_j = 1, j
                break
            if hit_dn:
                trig_side, trig_j = -1, j
                break
        if trig_side == 0:
            continue
        if trig_side == 9:
            outcome[i] = AMBIGUOUS
            continue
        entry = up if trig_side == 1 else dn
        tp_lvl = entry * (1.0 + trig_side * TP_MULT * s)
        sl_lvl = entry * (1.0 - trig_side * SL_MULT * s)
        final_j = min(trig_j + HORIZON - 1, n - 1)
        res_move, res_code, res_j = None, TIMEOUT, final_j
        for j in range(trig_j, final_j + 1):
            if trig_side == 1:
                if low_v[j] <= sl_lvl:
                    res_move, res_code, res_j = -SL_MULT * s, SL, j
                    break
                if high_v[j] >= tp_lvl:
                    res_move, res_code, res_j = TP_MULT * s, TP, j
                    break
            else:
                if high_v[j] >= sl_lvl:
                    res_move, res_code, res_j = -SL_MULT * s, SL, j
                    break
                if low_v[j] <= tp_lvl:
                    res_move, res_code, res_j = TP_MULT * s, TP, j
                    break
        if res_move is None:
            res_move = (close_v[final_j] / entry - 1.0) * trig_side
        outcome[i] = res_code
        side[i] = trig_side
        entry_off[i] = trig_j - i
        exit_off[i] = res_j - i
        price_move[i] = res_move
    return outcome, side, entry_off, exit_off, price_move


def replay_bracket(dec_idx, outcome, entry_off, exit_off, price_move, gate_mask):
    """Non-overlapping chronological replay over pre-computed per-bar outcomes. A setup is armed
    only if gate_mask; skipped if a position is open when its ENTRY would occur."""
    cash = 1.0
    equity = [1.0]
    occupied_through = -1
    n_trades = n_tp = n_sl = n_to = 0
    for i in dec_idx:
        if not gate_mask[i]:
            continue
        if outcome[i] in (NO_TRIGGER, AMBIGUOUS):
            continue
        e = i + entry_off[i]
        if e <= occupied_through:
            continue
        r = price_move[i] * NOTIONAL - ROUNDTRIP_COST_RATE * NOTIONAL
        cash *= 1.0 + r
        equity.append(cash)
        occupied_through = i + exit_off[i]
        n_trades += 1
        n_tp += outcome[i] == TP
        n_sl += outcome[i] == SL
        n_to += outcome[i] == TIMEOUT
    equity = np.array(equity)
    running_max = np.maximum.accumulate(equity)
    return {
        "n_trades": int(n_trades),
        "pnl_pct": float((cash - 1.0) * 100.0),
        "mdd_pct": float(((equity - running_max) / running_max).min() * 100.0) if len(equity) > 1 else 0.0,
        "n_tp": int(n_tp), "n_sl": int(n_sl), "n_timeout": int(n_to),
    }


def monthly_pnls(dec_idx, ts, outcome, entry_off, exit_off, price_move, gate_mask):
    months = ts.dt.to_period("M").astype(str).to_numpy()
    out = {}
    for m in sorted(set(months[dec_idx])):
        sub = dec_idx[months[dec_idx] == m]
        out[m] = replay_bracket(sub, outcome, entry_off, exit_off, price_move, gate_mask)["pnl_pct"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage0", "val", "oos"], required=True)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    labels = pd.read_parquet(LABEL_PATH)
    sigma = (labels["tp_move"].to_numpy(dtype=np.float64) / TP_MULT)  # recover sigma from label tp_move
    open_v = panel["open"].to_numpy(dtype=np.float64)
    high_v = panel["high"].to_numpy(dtype=np.float64)
    low_v = panel["low"].to_numpy(dtype=np.float64)
    close_v = panel["close"].to_numpy(dtype=np.float64)
    ts = panel["timestamp"]

    feat_cols = [c for c in panel.columns if c != "timestamp" and c not in RAW_LEVEL_COLS]
    raw_x = panel[feat_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)

    train_mask = (ts <= TRAIN_END).to_numpy()
    val_mask = ((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()
    oos_mask = ((ts >= OOS_START) & (ts <= OOS_END)).to_numpy()

    combos = [(b, W) for b in B_GRID for W in W_GRID]
    cache = {}
    for b, W in combos:
        cpath = OUT_DIR / f"outcomes_b{b}_W{W}.npz"
        if cpath.exists():
            z = np.load(cpath)
            cache[(b, W)] = (z["outcome"], z["side"], z["entry_off"], z["exit_off"], z["price_move"])
        else:
            print(f"computing bracket outcomes b={b} W={W}", flush=True)
            res = build_bracket_outcomes(open_v, high_v, low_v, close_v, sigma, b, W)
            np.savez(cpath, outcome=res[0], side=res[1], entry_off=res[2], exit_off=res[3], price_move=res[4])
            cache[(b, W)] = res

    if args.stage == "stage0":
        report = []
        for (b, W), (outcome, side, e_off, x_off, pmove) in cache.items():
            row = {"b": b, "W": W}
            for split, mask in (("train", train_mask), ("val", val_mask)):
                dec = np.flatnonzero(mask)
                all_gate = np.ones(len(panel), dtype=bool)
                raw = replay_bracket(dec, outcome, e_off, x_off, pmove, all_gate)
                oracle_gate = outcome == TP
                orc = replay_bracket(dec, outcome, e_off, x_off, pmove, oracle_gate)
                trig = np.flatnonzero(mask & np.isin(outcome, (TP, SL, TIMEOUT)))
                row[split] = {
                    "raw": raw, "oracle": orc,
                    "n_setups_triggered": int(len(trig)),
                    "tp_rate_among_triggered": float((outcome[trig] == TP).mean()) if len(trig) else 0.0,
                }
            report.append(row)
            print(json.dumps({"b": b, "W": W,
                              "train_raw_pnl": row["train"]["raw"]["pnl_pct"],
                              "val_raw_pnl": row["val"]["raw"]["pnl_pct"],
                              "val_oracle_pnl": row["val"]["oracle"]["pnl_pct"],
                              "val_tp_rate": row["val"]["tp_rate_among_triggered"]}), flush=True)
        (OUT_DIR / "stage0.json").write_text(json.dumps(report, indent=2))
    elif args.stage == "val":
        table = []
        for (b, W), (outcome, side, e_off, x_off, pmove) in cache.items():
            purge = HORIZON + W
            tr = train_mask.copy()
            tr_idx = np.flatnonzero(tr)
            tr[tr_idx[-purge - 288:]] = False  # purge + embargo
            triggered_train = np.flatnonzero(tr & np.isin(outcome, (TP, SL, TIMEOUT)))
            y = (outcome[triggered_train] == TP).astype(int)
            clf = lgb.LGBMClassifier(objective="binary", n_estimators=500, learning_rate=0.05,
                                     num_leaves=63, min_child_samples=200, feature_fraction=0.8,
                                     bagging_fraction=0.8, bagging_freq=1, reg_lambda=1.0,
                                     random_state=SEED, n_jobs=-1, verbosity=-1)
            clf.fit(raw_x[triggered_train], y)
            clf.booster_.save_model(str(OUT_DIR / f"gate_b{b}_W{W}.txt"))
            p_all = np.zeros(len(panel))
            v_idx = np.flatnonzero(val_mask)
            p_all[v_idx] = clf.booster_.predict(raw_x[v_idx])
            dec = np.flatnonzero(val_mask)
            for thr in THR_GRID:
                gate = p_all >= thr
                r = replay_bracket(dec, outcome, e_off, x_off, pmove, gate)
                mon = monthly_pnls(dec, ts, outcome, e_off, x_off, pmove, gate)
                pos_months = sum(v > 0 for v in mon.values())
                table.append({"b": b, "W": W, "thr": thr, **r,
                              "monthly_pnl": mon, "n_pos_months": int(pos_months)})
                print(json.dumps({k: table[-1][k] for k in ("b", "W", "thr", "n_trades", "pnl_pct", "mdd_pct", "n_pos_months")}), flush=True)
        eligible = [r for r in table if r["n_trades"] >= 15 and r["pnl_pct"] > 0 and r["n_pos_months"] >= 3]
        best = max(eligible, key=lambda r: r["pnl_pct"]) if eligible else None
        out = {"stage": "val", "table": table,
               "selected": None if best is None else {k: best[k] for k in ("b", "W", "thr", "pnl_pct", "n_trades", "mdd_pct", "n_pos_months")},
               "earns_oos_read": best is not None}
        (OUT_DIR / "val_results.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({"selected": out["selected"], "earns_oos_read": out["earns_oos_read"]}, indent=2))
    else:
        prior = json.loads((OUT_DIR / "val_results.json").read_text())
        if not prior.get("earns_oos_read"):
            print(json.dumps({"oos": "REFUSED -- no config passed the VAL stability screen"}))
            return 1
        sel = prior["selected"]
        b, W = sel["b"], sel["W"]
        outcome, side, e_off, x_off, pmove = cache[(b, W)]
        booster = lgb.Booster(model_file=str(OUT_DIR / f"gate_b{b}_W{W}.txt"))
        p_all = np.zeros(len(panel))
        o_idx = np.flatnonzero(oos_mask)
        p_all[o_idx] = booster.predict(raw_x[o_idx])
        gate = p_all >= sel["thr"]
        dec = np.flatnonzero(oos_mask)
        r = replay_bracket(dec, outcome, e_off, x_off, pmove, gate)
        mon = monthly_pnls(dec, ts, outcome, e_off, x_off, pmove, gate)
        out = {"stage": "oos", "selected": sel, **r, "monthly_pnl": mon,
               "adopted": bool(r["pnl_pct"] > 0)}
        (OUT_DIR / "oos_results.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
