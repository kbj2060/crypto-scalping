#!/usr/bin/env python3
"""TabPFN validation for the FEEDER-ROLE-OPTIMIZED 8-trigger V자반등 candidate pool (data/labels/
eth_5m_v_rebound_multitrigger_feeder_optimized_20260901/ -- orthogonal_combo dropped, kalman_
deviation_meanrev/demarker_extreme cluster-deduped GAP=12, see research_eth_v_rebound_multitrigger_
feeder_role_screen_20260901.py + build_eth_5m_v_rebound_multitrigger_labels_feeder_optimized_
20260901.py for the audit/rebuild this follows from).

Runs BOTH stages of this project's established gate in one script, since the fix here is a
provable-safe candidate-pool cleanup (not a novel/risky formula -- see feeder-role-audit memory)
and the gate criterion is the SAME one the original 9-trigger model itself had to clear:

  Stage 1 (VAL+OOS only): 4-seed TabPFN stability check, embargoed Fresh-Forward split, SAME 4
  seeds/methodology as research_eth_v_rebound_multitrigger_tabpfn_seed_stability_20260831.py.
  GATE: all 4 seeds must beat v7b sweep-only's own AUC (VAL>0.7342, OOS>0.7621) on BOTH splits --
  exactly the bar the ORIGINAL 9-trigger model needed to clear before it was allowed to touch
  HOLDOUT. If this fails, the script stops WITHOUT touching HOLDOUT.

  Stage 2 (only if Stage 1 passes): HOLDOUT (>=2026-04-01) touched EXACTLY ONCE, classification +
  economics together (SL/ARM/Trail grid selected on VAL+OOS only, HOLDOUT gets one evaluation with
  that fixed config) -- same combined-single-touch convention as research_eth_v_rebound_
  multitrigger_holdout_20260831.py, whose build_candidates()/economics logic is reused verbatim
  (imported from backtest_eth_sweep_v_rebound_v7b_trailing_costgate_20260830.py, unchanged).

This is a NEW, separate HOLDOUT touch for a NEW candidate-pool definition -- distinct from (and not
re-spending) the original 9-trigger pool's own already-spent HOLDOUT look.

Must run in the quant_ai conda env on the SERVER (GPU required):
  ~/anaconda3/envs/quant_ai/bin/python3 scripts/research_eth_v_rebound_multitrigger_feeder_optimized_tabpfn_full_20260901.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_feeder_optimized_20260901/eth_5m_v_rebound_multitrigger_feeder_optimized_features_tier0.csv"
KLINES = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT_DIR = ROOT / "data/research/eth_v_rebound_multitrigger_feeder_optimized_holdout_20260901"

TRAIL_SCRIPT = ROOT / "scripts/backtest_eth_sweep_v_rebound_v7b_trailing_costgate_20260830.py"
_spec = importlib.util.spec_from_file_location("v7b_trailing_costgate_20260830b", TRAIL_SCRIPT)
_trail = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_trail)

VAL_START, VAL_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
OOS_START, OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31 23:59:59", tz="UTC")
HOLDOUT_START = pd.Timestamp("2026-04-01", tz="UTC")
LABEL_WINDOW = pd.Timedelta(minutes=60)
SEEDS = [20260829, 141592, 271828, 577215]  # same 4 seeds reused project-wide for this exact model family
FORWARD_BARS = 200
STANDARD_COST_BP = _trail.STANDARD_COST_BP

V7B_BASELINE = {"val_auc": 0.7342, "oos_auc": 0.7621, "holdout_auc": 0.7788}
ORIGINAL_9TRIGGER_BENCHMARK = {
    "source": "data/labels/eth_5m_v_rebound_multitrigger_20260831 lineage (cheap_gate/seed_stability/holdout)",
    "val_auc_mean": 0.8289, "oos_auc_mean": 0.8125,
    "holdout_auc": 0.8465,
    "economics": {"val_opt_bp": 11.97, "oos_opt_bp": 20.96, "holdout_opt_bp": 9.28,
                  "val_win_rate": 0.838, "oos_win_rate": 0.867, "holdout_win_rate": 0.857},
}

FEATURE_COLUMNS = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday",
    "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile", "rsi",
]


def embargoed_split(df: pd.DataFrame) -> dict:
    ts = df["timestamp"]
    window_end = ts + LABEL_WINDOW
    return {
        "train": df.loc[(ts < VAL_START) & (window_end < VAL_START)],
        "val": df.loc[(ts >= VAL_START) & (ts <= VAL_END) & (window_end < OOS_START)],
        "oos": df.loc[(ts >= OOS_START) & (ts <= OOS_END) & (ts < HOLDOUT_START)],
        "holdout": df.loc[ts >= HOLDOUT_START],
    }


def evaluate(proba: np.ndarray, y: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    naive_acc = float(max(y.mean(), 1.0 - y.mean()))
    accuracy = float((pred == y).mean())
    return {
        "n": int(len(y)), "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "auc": round(float(roc_auc_score(y, proba)), 4),
        "naive_majority_class_accuracy": round(naive_acc, 4),
        "beats_naive_accuracy": bool(accuracy > naive_acc),
    }


def build_candidates(called: pd.DataFrame, kl: pd.DataFrame, ts_to_idx: pd.Series) -> pd.DataFrame:
    rows = []
    for _, ev in called.iterrows():
        idx = ts_to_idx.get(ev["timestamp"])
        if idx is None or idx + FORWARD_BARS + 1 >= len(kl):
            continue
        entry_bar = kl.iloc[idx + 1]
        side = "long" if ev["is_downside"] == 1 else "short"
        fwd = kl.iloc[idx + 1: idx + 1 + FORWARD_BARS][["timestamp", "open", "high", "low", "close"]]
        rows.append({
            "idx": int(ev["idx"]), "event_ts": ev["timestamp"], "split": ev["split"],
            "side": side, "model_proba": float(ev["model_proba"]), "label": int(ev["label"]),
            "atr": float(ev["atr"]), "entry_ts": entry_bar["timestamp"], "entry_price": float(entry_bar["open"]),
            "fwd_open": fwd["open"].tolist(), "fwd_high": fwd["high"].tolist(),
            "fwd_low": fwd["low"].tolist(), "fwd_close": fwd["close"].tolist(),
        })
    return pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURES_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[df["outcome"].isin(["V자반등", "지지/횡보"])].copy()
    df["label"] = (df["outcome"] == "V자반등").astype(int)
    df = df.dropna(subset=FEATURE_COLUMNS + ["label"]).reset_index(drop=True)

    parts = embargoed_split(df)
    for name in ("train", "val", "oos", "holdout"):
        p = parts[name]
        print(f"{name}: n={len(p)} label_rate={p['label'].mean():.4f}", flush=True)
    over_limit = len(parts["train"]) > 10000
    print(f"train n={len(parts['train'])} ignore_pretraining_limits={over_limit}\n", flush=True)

    print("=== STAGE 1: 4-seed stability check, VAL+OOS only (HOLDOUT not touched yet) ===", flush=True)
    per_seed = []
    val_probas, oos_probas = [], []
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed, ignore_pretraining_limits=over_limit)
        clf.fit(parts["train"][FEATURE_COLUMNS], parts["train"]["label"].to_numpy())
        row = {"seed": seed}
        p_val = clf.predict_proba(parts["val"][FEATURE_COLUMNS])[:, 1]
        p_oos = clf.predict_proba(parts["oos"][FEATURE_COLUMNS])[:, 1]
        val_probas.append(p_val); oos_probas.append(p_oos)
        row["val_auc"] = evaluate(p_val, parts["val"]["label"].to_numpy())["auc"]
        row["oos_auc"] = evaluate(p_oos, parts["oos"]["label"].to_numpy())["auc"]
        print(f"  seed={seed}: VAL_AUC={row['val_auc']:.4f} OOS_AUC={row['oos_auc']:.4f}", flush=True)
        per_seed.append(row)

    val_aucs = np.array([r["val_auc"] for r in per_seed])
    oos_aucs = np.array([r["oos_auc"] for r in per_seed])
    stability = {
        "seeds": SEEDS,
        "val_auc_mean": round(float(val_aucs.mean()), 4), "val_auc_std": round(float(val_aucs.std()), 4),
        "oos_auc_mean": round(float(oos_aucs.mean()), 4), "oos_auc_std": round(float(oos_aucs.std()), 4),
        "sign_consistent_vs_v7b": bool((val_aucs > V7B_BASELINE["val_auc"]).all() and (oos_aucs > V7B_BASELINE["oos_auc"]).all()),
        "per_seed": per_seed,
    }
    print(f"\nVAL AUC mean={stability['val_auc_mean']:.4f} std={stability['val_auc_std']:.4f} | "
          f"OOS AUC mean={stability['oos_auc_mean']:.4f} std={stability['oos_auc_std']:.4f}", flush=True)
    print(f"vs original 9-trigger model (VAL mean {ORIGINAL_9TRIGGER_BENCHMARK['val_auc_mean']}/"
          f"OOS mean {ORIGINAL_9TRIGGER_BENCHMARK['oos_auc_mean']}): "
          f"VAL delta {stability['val_auc_mean']-ORIGINAL_9TRIGGER_BENCHMARK['val_auc_mean']:+.4f}, "
          f"OOS delta {stability['oos_auc_mean']-ORIGINAL_9TRIGGER_BENCHMARK['oos_auc_mean']:+.4f}", flush=True)
    print(f"GATE (all 4 seeds beat v7b sweep-only VAL>{V7B_BASELINE['val_auc']}/OOS>{V7B_BASELINE['oos_auc']}): "
          f"{stability['sign_consistent_vs_v7b']}", flush=True)

    report = {
        "signal": "v_rebound_feeder_optimized", "asset": "ETHUSDT",
        "change_summary": "orthogonal_combo dropped (0 net-new) + kalman_deviation_meanrev/demarker_extreme "
                           "cluster-deduped GAP=12 -- see research_eth_v_rebound_multitrigger_feeder_role_screen_20260901.py",
        "v7b_baseline": V7B_BASELINE, "original_9trigger_benchmark": ORIGINAL_9TRIGGER_BENCHMARK,
        "stage1_stability": stability,
        "holdout_touched": False,
    }

    if not stability["sign_consistent_vs_v7b"]:
        print("\n*** GATE FAILED -- stopping WITHOUT touching HOLDOUT. ***", flush=True)
        (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 1

    print("\n=== GATE PASSED -- proceeding to STAGE 2: HOLDOUT touch (classification+economics, ONE-TIME) ===", flush=True)
    holdout_probas = []
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed, ignore_pretraining_limits=over_limit)
        clf.fit(parts["train"][FEATURE_COLUMNS], parts["train"]["label"].to_numpy())
        holdout_probas.append(clf.predict_proba(parts["holdout"][FEATURE_COLUMNS])[:, 1])
        print(f"  seed={seed} holdout scored", flush=True)

    classification = {}
    scored = {}
    all_probas = {"val": val_probas, "oos": oos_probas, "holdout": holdout_probas}
    for name in ("val", "oos", "holdout"):
        p = parts[name].copy()
        p["model_proba"] = np.mean(all_probas[name], axis=0)
        p["split"] = name
        scored[name] = p
        r = evaluate(p["model_proba"].to_numpy(), p["label"].to_numpy())
        classification[name] = r
        print(f"  {name:7s} n={r['n']:5d} auc={r['auc']:.4f} bal_acc={r['balanced_accuracy']:.4f}", flush=True)

    print(f"\nvs original 9-trigger HOLDOUT AUC {ORIGINAL_9TRIGGER_BENCHMARK['holdout_auc']}: "
          f"delta {classification['holdout']['auc']-ORIGINAL_9TRIGGER_BENCHMARK['holdout_auc']:+.4f}", flush=True)

    print("\n=== building trade candidates (called=proba>=0.5) for economics ===", flush=True)
    kl = pd.read_csv(KLINES, usecols=["timestamp", "open", "high", "low", "close"])
    kl["timestamp"] = pd.to_datetime(kl["timestamp"], utc=True)
    kl = kl.dropna().sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts_to_idx = pd.Series(kl.index.to_numpy(), index=kl["timestamp"].to_numpy())

    all_candidates = {}
    for name in ("val", "oos", "holdout"):
        called = scored[name][scored[name]["model_proba"] >= 0.5].copy()
        cand = build_candidates(called, kl, ts_to_idx)
        all_candidates[name] = cand
        print(f"  {name}: called {len(called)}/{len(scored[name])}, with forward data {len(cand)}, "
              f"precision(label==1|called)={called['label'].mean():.4f}", flush=True)

    val_oos = pd.concat([all_candidates["val"], all_candidates["oos"]], ignore_index=True)
    print(f"\n=== SL-race + trailing-stop grid search on VAL+OOS ONLY (n={len(val_oos)}) ===", flush=True)
    _trail.sl_race_diagnostic(val_oos)
    best = _trail.grid_search(val_oos)
    sl, arm, trail_mult = best[0][0], best[0][1], best[0][2]
    print(f"\n=== SELECTED config (by VAL+OOS only): SL={sl} ARM={arm} Trail={trail_mult} ===", flush=True)

    print("\n=== economics by split, FIXED config, HOLDOUT evaluated ONCE ===", flush=True)
    economics = {}
    for name in ("val", "oos", "holdout"):
        cand = all_candidates[name]
        opt = cand.apply(lambda r: _trail.simulate_trailing(r, sl, arm, trail_mult, False), axis=1)
        pess = cand.apply(lambda r: _trail.simulate_trailing(r, sl, arm, trail_mult, True), axis=1)
        opt_bp = float(opt.mean() * 1e4 - STANDARD_COST_BP)
        pess_bp = float(pess.mean() * 1e4 - STANDARD_COST_BP)
        win_rate = float((opt > 0).mean())
        economics[name] = {"n": int(len(cand)), "opt_bp": round(opt_bp, 2), "pess_bp": round(pess_bp, 2),
                            "win_rate": round(win_rate, 4)}
        print(f"  {name:7s} n={len(cand):5d} opt={opt_bp:+7.2f}bp pess={pess_bp:+7.2f}bp win_rate={win_rate:.1%}", flush=True)

    print(f"\nvs original 9-trigger economics (VAL+{ORIGINAL_9TRIGGER_BENCHMARK['economics']['val_opt_bp']}bp/"
          f"OOS+{ORIGINAL_9TRIGGER_BENCHMARK['economics']['oos_opt_bp']}bp/"
          f"HOLDOUT+{ORIGINAL_9TRIGGER_BENCHMARK['economics']['holdout_opt_bp']}bp): "
          f"HOLDOUT delta {economics['holdout']['opt_bp']-ORIGINAL_9TRIGGER_BENCHMARK['economics']['holdout_opt_bp']:+.2f}bp", flush=True)

    report["holdout_touched"] = True
    report["classification"] = classification
    report["selected_config"] = {"sl": sl, "arm": arm, "trail": trail_mult, "selected_on": "val+oos only"}
    report["economics"] = economics
    report["note"] = ("HOLDOUT touched exactly once for both classification and economics in this run "
                       "-- this is a NEW pool definition's OWN HOLDOUT spend, separate from the original "
                       "9-trigger pool's already-spent HOLDOUT look. Do not re-run with different parameters.")
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    for name in ("val", "oos", "holdout"):
        all_candidates[name].to_pickle(OUT_DIR / f"candidates_{name}.pkl")
    print(f"\nWrote {OUT_DIR / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
