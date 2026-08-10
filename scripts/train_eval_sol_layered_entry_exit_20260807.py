"""Rev2 layered stack for the SOL survey
(docs/experiments/sol_dl_rl_architecture_survey_rev2_layers_20260807.json).

Frozen parent: 5-seed TabM-flat entry models + rule side_prob_055 from
scripts/train_eval_sol_deepfeat_candidates_20260807.py (VAL +0.65% / OOS +4.08% seed-mean).

Layer A (entry quality): two binary LightGBM heads predicting per-side TP-first
  (targets: soft_long > soft_cash, soft_short > soft_cash), trained on purged train.
  Gate grid q in {0 (control), 0.40, 0.45, 0.50, 0.55}; a LONG entry needs P_long_tp >= q etc.
Layer B (trailing exit): close-based activation & ratchet trailing stop on top of the frozen
  layer-A choice. Grid: activation a in {0.5, 0.75} (of tp_move progress on close),
  trail distance d in {0.5, 1.0} (of sl_move). Intrabar exit priority SL -> trail -> TP,
  matching core/causal_futures_backtest's SL-before-TP convention.

Both layers are selected on seed-mean VAL only; each is adopted only if it beats the level below
it; `--stage oos` replays the final adopted stack exactly once.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from core.causal_futures_backtest import simulate_single_position  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    RAW_LEVEL_COLS, PANEL_PATH, LABEL_PATH, HORIZON_BARS,
    TRAIN_END, VAL_START, VAL_END, OOS_START, OOS_END,
)
import train_eval_sol_deepfeat_candidates_20260807 as dl  # noqa: E402

OUT_DIR = ROOT / "tmp/sol_dl_rl_survey_20260807/layered_rev2"
DL_DIR = ROOT / "tmp/sol_dl_rl_survey_20260807/dl_tabm_flat"
SEEDS = [903174, 42517, 6688211, 15093, 771442]
ENTRY_THRESHOLD = 0.55  # frozen parent rule
BASELINE_VAL_PNL = 0.6507  # frozen parent seed-mean VAL
MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE = 0.30, 3.0, 0.0010
QUALITY_GRID = [0.0, 0.40, 0.45, 0.50, 0.55]
TRAIL_GRID = [(0.5, 0.5), (0.5, 1.0), (0.75, 0.5), (0.75, 1.0)]
HEAD_SEED = 903174


def _mdd(equity: np.ndarray) -> float:
    running_max = np.maximum.accumulate(equity)
    return float(((equity - running_max) / running_max).min() * 100.0)


def replay_plain(panel, idx, side, tp_moves, sl_moves):
    if len(idx) == 0:
        return {"n_trades": 0, "pnl_pct": 0.0, "mdd_pct": 0.0, "win_rate": 0.0}
    res = simulate_single_position(
        timestamps=panel["timestamp"],
        open_px=panel["open"].to_numpy(dtype=np.float64),
        high=panel["high"].to_numpy(dtype=np.float64),
        low=panel["low"].to_numpy(dtype=np.float64),
        close=panel["close"].to_numpy(dtype=np.float64),
        decision_indices=idx, scores=side.astype(np.float64),
        tp_moves=tp_moves[idx], sl_moves=sl_moves[idx],
        upper_threshold=0.0, lower_threshold=0.0, horizon_bars=HORIZON_BARS,
        margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
    )
    ledger = res.ledger
    return {
        "n_trades": int(len(ledger)),
        "pnl_pct": float((res.equity[-1] - 1.0) * 100.0),
        "mdd_pct": _mdd(res.equity),
        "win_rate": float((ledger["trade_return"] > 0).mean()) if len(ledger) else 0.0,
        "exit_reasons": {str(k): int(v) for k, v in ledger["reason"].value_counts().items()} if len(ledger) else {},
    }


def replay_trailing(panel, idx, side, tp_moves, sl_moves, act: float, dist: float):
    """simulate_single_position semantics + close-ratchet trailing stop.
    Long: once close >= entry*(1+act*tp_move), trail = peak_close*(1-dist*sl_move), ratcheting on
    higher closes; exit when bar low <= trail (checked after hard SL, before TP)."""
    open_v = panel["open"].to_numpy(dtype=np.float64)
    high_v = panel["high"].to_numpy(dtype=np.float64)
    low_v = panel["low"].to_numpy(dtype=np.float64)
    close_v = panel["close"].to_numpy(dtype=np.float64)
    n = len(open_v)
    notional = MARGIN_FRACTION * LEVERAGE
    account_cost = ROUNDTRIP_COST_RATE * notional
    cash = 1.0
    equity_curve = [1.0]
    occupied_through = -1
    trades = []
    for k in range(len(idx)):
        i = int(idx[k])
        s = int(side[k])
        entry_i = i + 1
        if entry_i >= n or entry_i <= occupied_through:
            continue
        tp_m, sl_m = float(tp_moves[i]), float(sl_moves[i])
        if not (np.isfinite(tp_m) and np.isfinite(sl_m)):
            continue
        entry = open_v[entry_i]
        final_i = min(entry_i + HORIZON_BARS - 1, n - 1)
        trail_active = False
        trail_level = np.nan
        peak = entry
        price_move, reason, exit_i = None, None, final_i
        for j in range(entry_i, final_i + 1):
            if s > 0:
                if low_v[j] <= entry * (1.0 - sl_m):
                    price_move, reason, exit_i = -sl_m, "sl", j
                    break
                if trail_active and low_v[j] <= trail_level:
                    price_move, reason, exit_i = trail_level / entry - 1.0, "trail", j
                    break
                if high_v[j] >= entry * (1.0 + tp_m):
                    price_move, reason, exit_i = tp_m, "tp", j
                    break
                if close_v[j] > peak:
                    peak = close_v[j]
                if not trail_active and close_v[j] >= entry * (1.0 + act * tp_m):
                    trail_active = True
                if trail_active:
                    trail_level = max(trail_level if np.isfinite(trail_level) else -np.inf, peak * (1.0 - dist * sl_m))
            else:
                if high_v[j] >= entry * (1.0 + sl_m):
                    price_move, reason, exit_i = -sl_m, "sl", j
                    break
                if trail_active and high_v[j] >= trail_level:
                    price_move, reason, exit_i = 1.0 - trail_level / entry, "trail", j
                    break
                if low_v[j] <= entry * (1.0 - tp_m):
                    price_move, reason, exit_i = tp_m, "tp", j
                    break
                if close_v[j] < peak:
                    peak = close_v[j]
                if not trail_active and close_v[j] <= entry * (1.0 - act * tp_m):
                    trail_active = True
                if trail_active:
                    trail_level = min(trail_level if np.isfinite(trail_level) else np.inf, peak * (1.0 + dist * sl_m))
        if price_move is None:
            price_move = close_v[final_i] / entry - 1.0 if s > 0 else 1.0 - close_v[final_i] / entry
            reason, exit_i = "timeout", final_i
        trade_return = price_move * notional - account_cost
        cash *= 1.0 + trade_return
        equity_curve.append(cash)
        occupied_through = exit_i
        trades.append(trade_return)
    trades = np.array(trades)
    equity = np.array(equity_curve)
    return {
        "n_trades": int(len(trades)),
        "pnl_pct": float((cash - 1.0) * 100.0),
        "mdd_pct": _mdd(equity) if len(equity) > 1 else 0.0,
        "win_rate": float((trades > 0).mean()) if len(trades) else 0.0,
    }


def entry_decisions(proba: np.ndarray, rows: np.ndarray, q_long: np.ndarray, q_short: np.ndarray, q: float):
    arg = proba.argmax(axis=1)
    side_prob = np.take_along_axis(proba, arg[:, None], axis=1)[:, 0]
    side = np.where(arg == 1, 1, np.where(arg == 2, -1, 0))
    side = np.where(side_prob >= ENTRY_THRESHOLD, side, 0)
    if q > 0:
        side = np.where((side == 1) & (q_long[rows] < q), 0, side)
        side = np.where((side == -1) & (q_short[rows] < q), 0, side)
    keep = side != 0
    return rows[keep], side[keep]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["val", "oos"], required=True)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    panel, x_std, soft, action, tp_moves, sl_moves, train_mask, val_mask, oos_mask, feat_cols = dl.build_data()
    labels = pd.read_parquet(LABEL_PATH)
    raw = pd.read_csv(PANEL_PATH, low_memory=False)
    raw_x = raw[feat_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)

    y_long = (labels["trade_outcome_soft_long"].to_numpy() > labels["trade_outcome_soft_cash"].to_numpy()).astype(int)
    y_short = (labels["trade_outcome_soft_short"].to_numpy() > labels["trade_outcome_soft_cash"].to_numpy()).astype(int)

    head_path_l = OUT_DIR / "quality_long.txt"
    head_path_s = OUT_DIR / "quality_short.txt"
    if args.stage == "val":
        params = dict(objective="binary", n_estimators=500, learning_rate=0.05, num_leaves=63,
                      min_child_samples=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                      reg_lambda=1.0, random_state=HEAD_SEED, n_jobs=-1, verbosity=-1)
        clf_l = lgb.LGBMClassifier(**params).fit(raw_x[train_mask], y_long[train_mask])
        clf_s = lgb.LGBMClassifier(**params).fit(raw_x[train_mask], y_short[train_mask])
        clf_l.booster_.save_model(str(head_path_l))
        clf_s.booster_.save_model(str(head_path_s))
    booster_l = lgb.Booster(model_file=str(head_path_l))
    booster_s = lgb.Booster(model_file=str(head_path_s))

    q_long = np.zeros(len(panel))
    q_short = np.zeros(len(panel))
    eval_mask = val_mask if args.stage == "val" else oos_mask
    q_long[eval_mask] = booster_l.predict(raw_x[eval_mask])
    q_short[eval_mask] = booster_s.predict(raw_x[eval_mask])

    rows_eval = np.flatnonzero(eval_mask)
    per_seed_proba = {}
    for seed in SEEDS:
        if args.stage == "val":
            proba = np.load(DL_DIR / f"val_proba_seed{seed}.npy")
        else:
            model = dl.FlatTabM(x_std.shape[1]).to(device)
            model.load_state_dict(torch.load(DL_DIR / f"model_seed{seed}.pt", map_location=device))
            proba = dl.predict_rows(model, "tabm_flat", x_std, rows_eval, device)
        per_seed_proba[seed] = proba

    if args.stage == "val":
        # ---- Layer A selection ----
        table_a = []
        for q in QUALITY_GRID:
            per = []
            for seed in SEEDS:
                idx, side = entry_decisions(per_seed_proba[seed], rows_eval, q_long, q_short, q)
                per.append(replay_plain(panel, idx, side, tp_moves, sl_moves))
            table_a.append({
                "q": q,
                "seed_mean_pnl_pct": float(np.mean([r["pnl_pct"] for r in per])),
                "n_pos_seeds": int(sum(r["pnl_pct"] > 0 for r in per)),
                "seed_mean_trades": float(np.mean([r["n_trades"] for r in per])),
                "per_seed": per,
            })
            print(json.dumps({k: table_a[-1][k] for k in ("q", "seed_mean_pnl_pct", "n_pos_seeds", "seed_mean_trades")}), flush=True)
        eligible = [r for r in table_a if r["seed_mean_trades"] >= 15]
        best_a = max(eligible, key=lambda r: r["seed_mean_pnl_pct"]) if eligible else None
        adopt_a = bool(best_a and best_a["q"] > 0 and best_a["seed_mean_pnl_pct"] > BASELINE_VAL_PNL)
        frozen_q = best_a["q"] if adopt_a else 0.0
        level_a_val = best_a["seed_mean_pnl_pct"] if adopt_a else BASELINE_VAL_PNL

        # ---- Layer B selection on top of frozen layer A ----
        table_b = []
        for act, dist in TRAIL_GRID:
            per = []
            for seed in SEEDS:
                idx, side = entry_decisions(per_seed_proba[seed], rows_eval, q_long, q_short, frozen_q)
                per.append(replay_trailing(panel, idx, side, tp_moves, sl_moves, act, dist))
            table_b.append({
                "act": act, "dist": dist,
                "seed_mean_pnl_pct": float(np.mean([r["pnl_pct"] for r in per])),
                "n_pos_seeds": int(sum(r["pnl_pct"] > 0 for r in per)),
                "seed_mean_trades": float(np.mean([r["n_trades"] for r in per])),
                "per_seed": per,
            })
            print(json.dumps({k: table_b[-1][k] for k in ("act", "dist", "seed_mean_pnl_pct", "n_pos_seeds")}), flush=True)
        eligible_b = [r for r in table_b if r["seed_mean_trades"] >= 15]
        best_b = max(eligible_b, key=lambda r: r["seed_mean_pnl_pct"]) if eligible_b else None
        adopt_b = bool(best_b and best_b["seed_mean_pnl_pct"] > level_a_val)

        stack_val = best_b["seed_mean_pnl_pct"] if adopt_b else level_a_val
        earns_oos = bool((adopt_a or adopt_b) and stack_val > BASELINE_VAL_PNL)
        out = {
            "stage": "val", "baseline_val_pnl": BASELINE_VAL_PNL,
            "layer_a": {"table": table_a, "adopted": adopt_a, "frozen_q": frozen_q, "val_pnl": level_a_val},
            "layer_b": {"table": table_b, "adopted": adopt_b,
                         "frozen_trail": ([best_b["act"], best_b["dist"]] if adopt_b else None)},
            "final_stack_val_pnl": stack_val, "earns_oos_read": earns_oos,
        }
        (OUT_DIR / "val_results.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({"layer_a_adopted": adopt_a, "frozen_q": frozen_q, "layer_a_val": level_a_val,
                          "layer_b_adopted": adopt_b, "final_stack_val_pnl": stack_val,
                          "earns_oos_read": earns_oos}, indent=2))
    else:
        prior = json.loads((OUT_DIR / "val_results.json").read_text())
        if not prior.get("earns_oos_read"):
            print(json.dumps({"oos": "REFUSED -- layered stack did not beat baseline on VAL"}))
            return 1
        frozen_q = prior["layer_a"]["frozen_q"]
        trail = prior["layer_b"]["frozen_trail"]
        per = []
        for seed in SEEDS:
            idx, side = entry_decisions(per_seed_proba[seed], rows_eval, q_long, q_short, frozen_q)
            if trail is not None:
                r = replay_trailing(panel, idx, side, tp_moves, sl_moves, trail[0], trail[1])
            else:
                r = replay_plain(panel, idx, side, tp_moves, sl_moves)
            per.append({"seed": seed, **r})
            print(json.dumps(per[-1]), flush=True)
        pnls = [r["pnl_pct"] for r in per]
        out = {"stage": "oos", "frozen_q": frozen_q, "frozen_trail": trail,
               "seed_mean_pnl_pct": float(np.mean(pnls)), "n_pos_seeds": int(sum(p > 0 for p in pnls)),
               "per_seed": per}
        (OUT_DIR / "oos_results.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({k: out[k] for k in ("frozen_q", "frozen_trail", "seed_mean_pnl_pct", "n_pos_seeds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
