"""Rev6 regime-conditional entry stack for the SOL survey
(docs/experiments/sol_dl_rl_architecture_survey_rev6_regime_moe_20260808.json).

Stage R -- training-free causal trend-regime detectors, selected on TRAIN-only sub-window
stability, then ONE VAL check as the pre-registered gate:
  D1: 288-bar close return, theta=2%   (bull > +2%, bear < -2%, else chop)
  D2: 288-bar close return, theta=4%
  D3: mtf_trend_4h split at train quantiles (33%/66%)
Gate: within-regime top-20 direction features must keep their train sign on VAL for >=60% of
features in >=2 of 3 regimes (unconditional baseline: 0/20).

Stage M -- only if R passes: per-regime LightGBM 3-class experts on the TB label, causally
routed; frozen parent rule family; VAL selection (PnL>0, >=15 trades, >=3/4 months positive,
beats unconditional control -6.90%); single OOS read.

Usage: --stage {stageR, val, oos}
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
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    RAW_LEVEL_COLS, PANEL_PATH, LABEL_PATH, SEED, HORIZON_BARS,
    TRAIN_END, VAL_START, VAL_END, OOS_START, OOS_END, replay, side_state_from_proba,
)

OUT_DIR = ROOT / "tmp/sol_dl_rl_survey_20260807/regime_moe_rev6"
TOP_K = 20
GATE_AGREE = 0.60
GATE_MIN_REGIMES = 2
MIN_EXPERT_ROWS = 5000
THRESHOLDS = [0.0, 0.40, 0.45, 0.50, 0.55, 0.60]
CONTROL_VAL_PNL = -6.90
REGIME_NAMES = ["bear", "chop", "bull"]


def auc_binary(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x)
    x, y = x[m], y[m]
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 < 50 or n0 < 50:
        return np.nan
    r = rankdata(x)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def detectors(panel: pd.DataFrame, train_mask: np.ndarray) -> dict[str, np.ndarray]:
    close = panel["close"].to_numpy(dtype=np.float64)
    r288 = np.full(len(close), np.nan)
    r288[288:] = close[288:] / close[:-288] - 1.0
    out = {}
    for name, theta in (("D1_r288_t2", 0.02), ("D2_r288_t4", 0.04)):
        reg = np.full(len(close), 1, dtype=np.int8)  # chop
        reg[r288 > theta] = 2
        reg[r288 < -theta] = 0
        reg[~np.isfinite(r288)] = 1
        out[name] = reg
    m4 = pd.to_numeric(panel["mtf_trend_4h"], errors="coerce").to_numpy(dtype=np.float64)
    q33, q66 = np.nanquantile(m4[train_mask], [0.33, 0.66])
    reg = np.full(len(close), 1, dtype=np.int8)
    reg[m4 > q66] = 2
    reg[m4 < q33] = 0
    out["D3_mtf4h_q"] = reg
    return out


def within_regime_sign_table(x, action, idx_windows: dict[str, np.ndarray], regime: np.ndarray, feat_cols):
    """Per regime: top-K train features by |AUC-0.5| and their sign across windows."""
    result = {}
    for r, rname in enumerate(REGIME_NAMES):
        aucs = {}
        for wname, idx in idx_windows.items():
            sub = idx[regime[idx] == r]
            a = action[sub]
            nz = a != 0
            vals = []
            for f_i in range(x.shape[1]):
                vals.append(auc_binary(x[sub, f_i][nz].astype(np.float64), (a[nz][a[nz] != 0] == 1).astype(int)) if nz.sum() > 100 else np.nan)
            aucs[wname] = np.array(vals)
        dev = np.abs(np.nan_to_num(aucs["train"], nan=0.5) - 0.5)
        top = np.argsort(-dev)[:TOP_K]
        sign_tr = np.sign(aucs["train"][top] - 0.5)
        result[rname] = {
            "top_features": [feat_cols[i] for i in top],
            "train_auc": aucs["train"][top],
            "windows": {w: aucs[w][top] for w in idx_windows if w != "train"},
            "sign_train": sign_tr,
        }
    return result


def train_stability_score(table) -> float:
    """Mean across regimes of the fraction of top-K features whose sign holds in all 3 train subs."""
    scores = []
    for rname, rec in table.items():
        signs = np.stack([np.sign(np.nan_to_num(rec["windows"][f"tr_sub{k}"], nan=0.5) - 0.5) for k in (1, 2, 3)], axis=1)
        ok = (signs == rec["sign_train"][:, None]).all(axis=1)
        scores.append(float(ok.mean()))
    return float(np.mean(scores))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stageR", "val", "oos"], required=True)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    labels = pd.read_parquet(LABEL_PATH)
    action = labels["trade_outcome_action"].to_numpy()
    tp_moves = labels["tp_move"].to_numpy(dtype=np.float64)
    sl_moves = labels["sl_move"].to_numpy(dtype=np.float64)
    feat_cols = [c for c in panel.columns if c != "timestamp" and c not in RAW_LEVEL_COLS]
    x = panel[feat_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)

    ts = panel["timestamp"]
    train_mask = (ts <= TRAIN_END).to_numpy()
    purge_cut = np.flatnonzero(train_mask)[-HORIZON_BARS:]
    train_mask[purge_cut] = False
    train_mask &= np.isfinite(tp_moves)
    val_mask = ((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()
    oos_mask = ((ts >= OOS_START) & (ts <= OOS_END)).to_numpy()

    dets = detectors(panel, train_mask)
    tr_idx = np.flatnonzero(train_mask)
    subs = np.array_split(tr_idx, 3)

    if args.stage == "stageR":
        # 1) detector selection on TRAIN-only sub-window stability
        train_windows = {"train": tr_idx, "tr_sub1": subs[0], "tr_sub2": subs[1], "tr_sub3": subs[2]}
        det_scores = {}
        det_tables = {}
        for dname, regime in dets.items():
            table = within_regime_sign_table(x, action, train_windows, regime, feat_cols)
            det_tables[dname] = table
            det_scores[dname] = train_stability_score(table)
            counts = {REGIME_NAMES[r]: int((regime[tr_idx] == r).sum()) for r in range(3)}
            print(json.dumps({"detector": dname, "train_stability": det_scores[dname], "train_regime_counts": counts}), flush=True)
        selected = max(det_scores, key=det_scores.get)

        # 2) ONE VAL check for the selected detector = the gate
        regime = dets[selected]
        gate_windows = {"train": tr_idx, "tr_sub1": subs[0], "tr_sub2": subs[1], "tr_sub3": subs[2], "val": np.flatnonzero(val_mask)}
        table = within_regime_sign_table(x, action, gate_windows, regime, feat_cols)
        per_regime_agree = {}
        for rname, rec in table.items():
            sign_val = np.sign(np.nan_to_num(rec["windows"]["val"], nan=0.5) - 0.5)
            per_regime_agree[rname] = float((sign_val == rec["sign_train"]).mean())
        n_pass = sum(v >= GATE_AGREE for v in per_regime_agree.values())
        gate_pass = n_pass >= GATE_MIN_REGIMES
        out = {
            "stage": "stageR", "detector_train_scores": det_scores, "selected_detector": selected,
            "val_gate_agreement_per_regime": per_regime_agree,
            "gate_threshold": GATE_AGREE, "n_regimes_passing": int(n_pass), "gate_pass": bool(gate_pass),
            "top_features_per_regime": {r: rec["top_features"] for r, rec in table.items()},
        }
        (OUT_DIR / "stageR.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({k: out[k] for k in ("selected_detector", "val_gate_agreement_per_regime", "gate_pass")}, indent=2))
        return 0

    stager = json.loads((OUT_DIR / "stageR.json").read_text())
    if not stager.get("gate_pass"):
        print(json.dumps({"stage": args.stage, "verdict": "REFUSED -- Stage R sign-stability gate failed; no expert models trained"}))
        return 1
    regime = dets[stager["selected_detector"]]

    if args.stage == "val":
        experts = {}
        for r in range(3):
            rows = tr_idx[regime[tr_idx] == r]
            if len(rows) < MIN_EXPERT_ROWS:
                raise RuntimeError(f"regime {REGIME_NAMES[r]}: only {len(rows)} train rows")
            clf = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=600, learning_rate=0.05,
                                     num_leaves=63, min_child_samples=200, feature_fraction=0.8,
                                     bagging_fraction=0.8, bagging_freq=1, reg_lambda=1.0,
                                     random_state=SEED + r, n_jobs=-1, verbosity=-1)
            clf.fit(x[rows], action[rows])
            clf.booster_.save_model(str(OUT_DIR / f"expert_{REGIME_NAMES[r]}.txt"))
            experts[r] = clf.booster_
        v_idx = np.flatnonzero(val_mask)
        proba = np.zeros((len(panel), 3))
        for r in range(3):
            sub = v_idx[regime[v_idx] == r]
            if len(sub):
                proba[sub] = experts[r].predict(x[sub])
        months = ts.dt.to_period("M").astype(str).to_numpy()
        results = []
        for thr in THRESHOLDS:
            side_state = np.zeros(len(panel), dtype=np.int64)
            side_state[v_idx] = side_state_from_proba(proba[v_idx], thr)
            rres = replay(panel, side_state, tp_moves, sl_moves, val_mask)
            mon = {}
            for m in sorted(set(months[v_idx])):
                mmask = val_mask & (months == m)
                mon[m] = replay(panel, side_state, tp_moves, sl_moves, mmask).get("pnl_pct", 0.0)
            n_pos_m = sum(v > 0 for v in mon.values())
            results.append({"threshold": thr, **{k: rres.get(k) for k in ("n_trades", "pnl_pct", "win_rate", "mdd_pct")}, "monthly": mon, "n_pos_months": int(n_pos_m)})
            print(json.dumps({k: results[-1][k] for k in ("threshold", "n_trades", "pnl_pct", "n_pos_months")}), flush=True)
        eligible = [r for r in results if (r["n_trades"] or 0) >= 15 and (r["pnl_pct"] or 0) > 0 and r["n_pos_months"] >= 3 and (r["pnl_pct"] or 0) > CONTROL_VAL_PNL]
        best = max(eligible, key=lambda r: r["pnl_pct"]) if eligible else None
        out = {"stage": "val", "detector": stager["selected_detector"], "results": results,
               "selected": None if best is None else {k: best[k] for k in ("threshold", "pnl_pct", "n_trades", "mdd_pct", "n_pos_months")},
               "earns_oos_read": best is not None}
        (OUT_DIR / "val_results.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({"selected": out["selected"], "earns_oos_read": out["earns_oos_read"]}, indent=2))
    else:
        prior = json.loads((OUT_DIR / "val_results.json").read_text())
        if not prior.get("earns_oos_read"):
            print(json.dumps({"oos": "REFUSED -- Stage M VAL gate failed"}))
            return 1
        sel = prior["selected"]
        experts = {r: lgb.Booster(model_file=str(OUT_DIR / f"expert_{REGIME_NAMES[r]}.txt")) for r in range(3)}
        o_idx = np.flatnonzero(oos_mask)
        proba = np.zeros((len(panel), 3))
        for r in range(3):
            sub = o_idx[regime[o_idx] == r]
            if len(sub):
                proba[sub] = experts[r].predict(x[sub])
        side_state = np.zeros(len(panel), dtype=np.int64)
        side_state[o_idx] = side_state_from_proba(proba[o_idx], sel["threshold"])
        rres = replay(panel, side_state, tp_moves, sl_moves, oos_mask)
        out = {"stage": "oos", "selected": sel, **rres, "adopted": bool((rres.get("pnl_pct") or 0) > 0)}
        (OUT_DIR / "oos_results.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
