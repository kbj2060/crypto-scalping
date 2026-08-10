"""BTC regime-conditioned entry line (docs/experiments/btc_regime_conditioned_entry_20260808.json).

Stages (all constants inherited from the SOL-survey pipeline; BTC panel built by
scripts/build_btc_panel_for_regime_line_20260808.py):
  --stage stage0   oracle label-following ceiling on train+VAL (label validation)
  --stage stageR   within-regime top-20 sign-stability gate on the full panel
                   (>=60% train->VAL agreement in >=2 of 3 D2 regimes)
  --stage control  unconditional LGBM control (asset-matched baseline; no OOS for it)
  --stage val      Stage 2 retraining+tuning: per-regime experts, chop forced CASH
                   (+ one chop-expert control config), grid {hp_default, hp_conservative} x
                   {full features, per-regime top-20} x 6 entry rules; VAL-only selection
  --stage oos      single frozen OOS read of the selected config
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from core.causal_futures_backtest import simulate_single_position  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    RAW_LEVEL_COLS, ENTRY_RULES, HORIZON_BARS, SEED,
    MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE,
    TRAIN_END, VAL_START, VAL_END, OOS_START, OOS_END,
    replay, side_state_from_proba,
)

PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_2024_2026_regimeline.csv"
LABEL_PATH = ROOT / "data/splits/year_oos/btc_5m_tripbarrier_tradeoutcome_labels_regimeline_20260808.parquet"
OUT_DIR = ROOT / "tmp/btc_regime_conditioned_20260808"
REGIME_NAMES = ["bear", "chop", "bull"]
TOP_K = 20
GATE_AGREE, GATE_MIN_REGIMES = 0.60, 2
HP_GRID = {"hp_default": dict(num_leaves=63, min_child_samples=200),
           "hp_conservative": dict(num_leaves=31, min_child_samples=500)}
FEATSET_GRID = ["full", "top20"]


def auc_binary(x, y):
    m = np.isfinite(x)
    x, y = x[m], y[m]
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 < 50 or n0 < 50:
        return np.nan
    r = rankdata(x)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def load_all():
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    labels = pd.read_parquet(LABEL_PATH)
    n = min(len(panel), len(labels))
    panel, labels = panel.iloc[:n].reset_index(drop=True), labels.iloc[:n]
    assert (labels["timestamp"].to_numpy() == panel["timestamp"].to_numpy()).all()
    action = labels["trade_outcome_action"].to_numpy()
    tp_moves = labels["tp_move"].to_numpy(dtype=np.float64)
    sl_moves = labels["sl_move"].to_numpy(dtype=np.float64)
    feat_cols = [c for c in panel.columns if c != "timestamp" and c not in RAW_LEVEL_COLS]
    x = panel[feat_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    ts = panel["timestamp"]
    close = panel["close"].to_numpy(dtype=np.float64)
    r288 = np.full(len(close), np.nan)
    r288[288:] = close[288:] / close[:-288] - 1.0
    regime = np.full(len(close), 1, dtype=np.int8)
    regime[r288 > 0.04] = 2
    regime[r288 < -0.04] = 0
    train_mask = (ts <= TRAIN_END).to_numpy()
    tr_all = np.flatnonzero(train_mask)
    train_mask[tr_all[-HORIZON_BARS:]] = False
    train_mask &= np.isfinite(tp_moves)
    val_mask = ((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()
    oos_mask = ((ts >= OOS_START) & (ts <= OOS_END)).to_numpy()
    return panel, ts, x, feat_cols, action, tp_moves, sl_moves, regime, train_mask, val_mask, oos_mask


def per_regime_auc(x, action, idx, regime, r):
    sub = idx[regime[idx] == r]
    a = action[sub]
    nz = a != 0
    out = np.full(x.shape[1], np.nan)
    if nz.sum() < 200:
        return out
    y = (a[nz] == 1).astype(int)
    for f in range(x.shape[1]):
        out[f] = auc_binary(x[sub, f][nz].astype(np.float64), y)
    return out


def lgbm(params: dict, seed_off: int = 0):
    return lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=600, learning_rate=0.05,
                              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, reg_lambda=1.0,
                              random_state=SEED + seed_off, n_jobs=-1, verbosity=-1, **params)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage0", "stageR", "control", "val", "oos"], required=True)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel, ts, x, feat_cols, action, tp_moves, sl_moves, regime, train_mask, val_mask, oos_mask = load_all()
    tr_idx = np.flatnonzero(train_mask)
    v_idx = np.flatnonzero(val_mask)
    months = ts.dt.to_period("M").astype(str).to_numpy()

    if args.stage == "stage0":
        out = {}
        for split, mask in (("train", train_mask), ("val", val_mask)):
            idx = np.flatnonzero(mask & (action != 0) & np.isfinite(tp_moves) & np.isfinite(sl_moves))
            side = np.where(action[idx] == 1, 1.0, -1.0)
            res = simulate_single_position(
                timestamps=ts, open_px=panel["open"].to_numpy(dtype=np.float64),
                high=panel["high"].to_numpy(dtype=np.float64), low=panel["low"].to_numpy(dtype=np.float64),
                close=panel["close"].to_numpy(dtype=np.float64), decision_indices=idx, scores=side,
                tp_moves=tp_moves[idx], sl_moves=sl_moves[idx], upper_threshold=0.0, lower_threshold=0.0,
                horizon_bars=HORIZON_BARS, margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE,
                roundtrip_cost_rate=ROUNDTRIP_COST_RATE)
            ledger = res.ledger
            out[split] = {"n_trades": int(len(ledger)), "win_rate": float((ledger["trade_return"] > 0).mean()),
                          "sum_ret_pct": float(ledger["trade_return"].sum() * 100.0)}
        (OUT_DIR / "stage0.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
    elif args.stage == "stageR":
        subs = np.array_split(tr_idx, 3)
        result = {}
        for r in range(3):
            auc_tr = per_regime_auc(x, action, tr_idx, regime, r)
            auc_v = per_regime_auc(x, action, v_idx, regime, r)
            dev = np.abs(np.nan_to_num(auc_tr, nan=0.5) - 0.5)
            top = np.argsort(-dev)[:TOP_K]
            s_tr = np.sign(auc_tr[top] - 0.5)
            agree = float(np.mean(s_tr == np.sign(np.nan_to_num(auc_v[top], nan=0.5) - 0.5)))
            signed = float(np.nanmean((np.nan_to_num(auc_v[top], nan=0.5) - 0.5) * s_tr))
            sub_ok = []
            for s_idx in subs:
                auc_s = per_regime_auc(x, action, s_idx, regime, r)
                sub_ok.append(np.sign(np.nan_to_num(auc_s[top], nan=0.5) - 0.5))
            stability = float(np.mean((np.stack(sub_ok, axis=1) == s_tr[:, None]).all(axis=1)))
            result[REGIME_NAMES[r]] = {"val_sign_agreement": round(agree, 3), "val_signed_edge": round(signed, 4),
                                       "train_subwindow_stability": round(stability, 3),
                                       "top3": [feat_cols[i] for i in top[:3]],
                                       "occupancy_train": round(float((regime[tr_idx] == r).mean()), 3)}
            print(json.dumps({REGIME_NAMES[r]: result[REGIME_NAMES[r]]}), flush=True)
        n_pass = sum(v["val_sign_agreement"] >= GATE_AGREE for v in result.values())
        gate_pass = bool(n_pass >= GATE_MIN_REGIMES)
        out = {"per_regime": result, "n_regimes_passing": int(n_pass), "gate_pass": gate_pass}
        (OUT_DIR / "stageR.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({"gate_pass": gate_pass, "n_regimes_passing": n_pass}, indent=2))
    elif args.stage == "control":
        clf = lgbm(HP_GRID["hp_default"])
        clf.fit(x[tr_idx], action[tr_idx])
        clf.booster_.save_model(str(OUT_DIR / "control_lgbm.txt"))
        proba = clf.predict_proba(x[v_idx])
        results = []
        for rule in ENTRY_RULES:
            side_state = np.zeros(len(panel), dtype=np.int64)
            side_state[v_idx] = side_state_from_proba(proba, rule["threshold"])
            rres = replay(panel, side_state, tp_moves, sl_moves, val_mask)
            results.append({"rule": rule["name"], **{k: rres.get(k) for k in ("n_trades", "pnl_pct", "win_rate", "mdd_pct")}})
            print(json.dumps(results[-1]), flush=True)
        eligible = [r for r in results if (r["n_trades"] or 0) >= 15]
        best = max(eligible, key=lambda r: r["pnl_pct"]) if eligible else None
        out = {"results": results, "best_val_pnl": None if best is None else best["pnl_pct"]}
        (OUT_DIR / "control.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({"control_best_val_pnl": out["best_val_pnl"]}, indent=2))
    elif args.stage == "val":
        stager = json.loads((OUT_DIR / "stageR.json").read_text())
        if not stager.get("gate_pass"):
            print(json.dumps({"verdict": "REFUSED -- Stage R gate failed"}))
            return 1
        control_pnl = json.loads((OUT_DIR / "control.json").read_text())["best_val_pnl"]
        tops = {}
        for r in range(3):
            auc_tr = per_regime_auc(x, action, tr_idx, regime, r)
            dev = np.abs(np.nan_to_num(auc_tr, nan=0.5) - 0.5)
            tops[r] = np.argsort(-dev)[:TOP_K]
        table = []
        for hp_name, hp in HP_GRID.items():
            for featset in FEATSET_GRID:
                proba = np.zeros((len(panel), 3))
                for r in range(3):
                    rows = tr_idx[regime[tr_idx] == r]
                    cols = tops[r] if featset == "top20" else np.arange(x.shape[1])
                    clf = lgbm(hp, seed_off=r)
                    clf.fit(x[rows][:, cols], action[rows])
                    clf.booster_.save_model(str(OUT_DIR / f"expert_{hp_name}_{featset}_{REGIME_NAMES[r]}.txt"))
                    sub = v_idx[regime[v_idx] == r]
                    if len(sub):
                        proba[sub] = clf.booster_.predict(x[sub][:, cols])
                for chop_policy in (["force_cash", "expert"] if (hp_name == "hp_default" and featset == "full") else ["force_cash"]):
                    for rule in ENTRY_RULES:
                        side_state = np.zeros(len(panel), dtype=np.int64)
                        side_state[v_idx] = side_state_from_proba(proba[v_idx], rule["threshold"])
                        if chop_policy == "force_cash":
                            side_state[v_idx[regime[v_idx] == 1]] = 0
                        rres = replay(panel, side_state, tp_moves, sl_moves, val_mask)
                        mon = {}
                        for m in sorted(set(months[v_idx])):
                            mon[m] = replay(panel, side_state, tp_moves, sl_moves, val_mask & (months == m)).get("pnl_pct", 0.0)
                        rec = {"hp": hp_name, "featset": featset, "chop": chop_policy, "rule": rule["name"],
                               "threshold": rule["threshold"],
                               **{k: rres.get(k) for k in ("n_trades", "pnl_pct", "win_rate", "mdd_pct")},
                               "monthly": mon, "n_pos_months": int(sum(v_ > 0 for v_ in mon.values()))}
                        table.append(rec)
                        print(json.dumps({k: rec[k] for k in ("hp", "featset", "chop", "rule", "n_trades", "pnl_pct", "n_pos_months")}), flush=True)
        eligible = [r for r in table if (r["n_trades"] or 0) >= 15 and (r["pnl_pct"] or 0) > 0
                    and r["n_pos_months"] >= 3 and (r["pnl_pct"] or 0) > control_pnl]
        best = max(eligible, key=lambda r: r["pnl_pct"]) if eligible else None
        out = {"control_best_val_pnl": control_pnl, "table": table,
               "selected": None if best is None else {k: best[k] for k in ("hp", "featset", "chop", "rule", "threshold", "pnl_pct", "n_trades", "mdd_pct", "n_pos_months")},
               "earns_oos_read": best is not None}
        (OUT_DIR / "val_results.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({"selected": out["selected"], "earns_oos_read": out["earns_oos_read"]}, indent=2))
    else:
        prior = json.loads((OUT_DIR / "val_results.json").read_text())
        if not prior.get("earns_oos_read"):
            print(json.dumps({"oos": "REFUSED -- VAL gate failed"}))
            return 1
        sel = prior["selected"]
        tops = {}
        for r in range(3):
            auc_tr = per_regime_auc(x, action, tr_idx, regime, r)
            dev = np.abs(np.nan_to_num(auc_tr, nan=0.5) - 0.5)
            tops[r] = np.argsort(-dev)[:TOP_K]
        o_idx = np.flatnonzero(oos_mask)
        proba = np.zeros((len(panel), 3))
        for r in range(3):
            cols = tops[r] if sel["featset"] == "top20" else np.arange(x.shape[1])
            booster = lgb.Booster(model_file=str(OUT_DIR / f"expert_{sel['hp']}_{sel['featset']}_{REGIME_NAMES[r]}.txt"))
            sub = o_idx[regime[o_idx] == r]
            if len(sub):
                proba[sub] = booster.predict(x[sub][:, cols])
        side_state = np.zeros(len(panel), dtype=np.int64)
        side_state[o_idx] = side_state_from_proba(proba[o_idx], sel["threshold"])
        if sel["chop"] == "force_cash":
            side_state[o_idx[regime[o_idx] == 1]] = 0
        rres = replay(panel, side_state, tp_moves, sl_moves, oos_mask)
        mon = {}
        for m in sorted(set(months[o_idx])):
            mon[m] = replay(panel, side_state, tp_moves, sl_moves, oos_mask & (months == m)).get("pnl_pct", 0.0)
        out = {"stage": "oos", "selected": sel, **rres, "monthly": mon,
               "adopted": bool((rres.get("pnl_pct") or 0) > 0)}
        (OUT_DIR / "oos_results.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
