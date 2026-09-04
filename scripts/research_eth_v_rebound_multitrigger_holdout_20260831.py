#!/usr/bin/env python3
"""HOLDOUT spend (classification + economics together, ONE-TIME per this project's convention --
same pattern as eth_kalman_demarker_final_complete_20260831's "홀드아웃(분류+경제성 동시, 1회성
소진)") for the 9-trigger V자반등 candidate pool. Cheap_gate (VAL AUC 0.8296/OOS 0.8119, single
seed) and N-seed stability (4 seeds, VAL mean=0.8289 std=0.0007 / OOS mean=0.8125 std=0.0004, all
4 beat v7b sweep-only on both splits) both already passed -- see memory eth_v_rebound_sweep_gated_
recall_gap_20260831 for the full chain. HOLDOUT (>=2026-04-01) has NOT been touched until this run.

Discipline (verbatim from this project's established cost-gate convention, imported not
reimplemented): SL/ARM/Trail grid selection happens on VAL+OOS ONLY
(backtest_eth_sweep_v_rebound_v7b_trailing_costgate_20260830.py::simulate_trailing/grid_search,
imported unchanged) -- HOLDOUT gets exactly ONE evaluation with the config already fixed by that
VAL+OOS search, never used for tuning. Candidate-building methodology (4-seed-averaged proba,
called=proba>=0.5, entry=next bar's open, 200 forward bars) reused from research_eth_sweep_v_
rebound_v7b_costgate_candidates_20260830.py, extended to also produce a HOLDOUT candidate set
(that script only went to VAL+OOS since v7b's own holdout was a separate, earlier, classification-
only spend -- this project's later signals (fib_ext, kalman/demarker) combined classification+
economics into a single HOLDOUT touch, which this script follows).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
CHEAP_GATE_SCRIPT = ROOT / "scripts/research_eth_v_rebound_multitrigger_tabpfn_cheap_gate_20260831.py"
_spec = importlib.util.spec_from_file_location("v_rebound_cheap_gate_20260831b", CHEAP_GATE_SCRIPT)
_cg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cg)

TRAIL_SCRIPT = ROOT / "scripts/backtest_eth_sweep_v_rebound_v7b_trailing_costgate_20260830.py"
_spec2 = importlib.util.spec_from_file_location("v7b_trailing_costgate_20260830", TRAIL_SCRIPT)
_trail = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_trail)

KLINES = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT_DIR = ROOT / "data/research/eth_v_rebound_multitrigger_holdout_20260831"
HOLDOUT_START = pd.Timestamp("2026-04-01", tz="UTC")
SEEDS = [20260829, 141592, 271828, 577215]
FORWARD_BARS = 200
STANDARD_COST_BP = _trail.STANDARD_COST_BP


def split_with_holdout(df: pd.DataFrame) -> dict:
    parts = _cg.embargoed_split(df)
    ts = df["timestamp"]
    parts["holdout"] = df.loc[ts >= HOLDOUT_START].reset_index(drop=True)
    return parts


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
    df = pd.read_csv(_cg.FEATURES_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[df["outcome"].isin(["V자반등", "지지/횡보"])].copy()
    df["label"] = (df["outcome"] == "V자반등").astype(int)
    df = df.dropna(subset=_cg.FEATURE_COLUMNS + ["label"]).reset_index(drop=True)

    parts = split_with_holdout(df)
    for name in ("train", "val", "oos", "holdout"):
        p = parts[name]
        print(f"{name}: n={len(p)} label_rate={p['label'].mean():.4f}", flush=True)
    over_limit = len(parts["train"]) > 10000

    print("\n=== fitting 4-seed ensemble on TRAIN, scoring VAL/OOS/HOLDOUT (HOLDOUT TOUCHED HERE) ===", flush=True)
    probas = {"val": [], "oos": [], "holdout": []}
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed, ignore_pretraining_limits=over_limit)
        clf.fit(parts["train"][_cg.FEATURE_COLUMNS], parts["train"]["label"].to_numpy())
        for name in ("val", "oos", "holdout"):
            probas[name].append(clf.predict_proba(parts[name][_cg.FEATURE_COLUMNS])[:, 1])
        print(f"  seed={seed} done", flush=True)

    classification = {}
    scored = {}
    for name in ("val", "oos", "holdout"):
        p = parts[name].copy()
        p["model_proba"] = np.mean(probas[name], axis=0)
        p["split"] = name
        scored[name] = p
        r = _cg.evaluate(p["model_proba"].to_numpy(), p["label"].to_numpy())
        classification[name] = r
        print(f"  {name:7s} n={r['n']:5d} auc={r['auc']:.4f} bal_acc={r['balanced_accuracy']:.4f} "
              f"beats_naive={r['beats_naive_accuracy']}", flush=True)

    print(f"\nvs v7b(sweep-only) VAL={0.7342:.4f} OOS={0.7621:.4f} HOLDOUT={0.7788:.4f}: "
          f"VAL delta {classification['val']['auc']-0.7342:+.4f}, OOS delta {classification['oos']['auc']-0.7621:+.4f}, "
          f"HOLDOUT delta {classification['holdout']['auc']-0.7788:+.4f}", flush=True)

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
        print(f"  {name}: called {len(called)}/{len(scored[name])}, "
              f"with forward data {len(cand)}, precision(label==1|called)={called['label'].mean():.4f}", flush=True)

    val_oos = pd.concat([all_candidates["val"], all_candidates["oos"]], ignore_index=True)
    print(f"\n=== SL-race + trailing-stop grid search on VAL+OOS ONLY (n={len(val_oos)}) -- "
          f"HOLDOUT excluded from parameter selection ===", flush=True)
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

    report = {
        "seeds": SEEDS, "train_n": int(len(parts["train"])), "ignore_pretraining_limits": over_limit,
        "classification": classification,
        "classification_vs_v7b": {
            "val_delta": round(classification["val"]["auc"] - 0.7342, 4),
            "oos_delta": round(classification["oos"]["auc"] - 0.7621, 4),
            "holdout_delta": round(classification["holdout"]["auc"] - 0.7788, 4),
        },
        "selected_config": {"sl": sl, "arm": arm, "trail": trail_mult, "selected_on": "val+oos only"},
        "economics": economics,
        "note": "HOLDOUT touched exactly once for both classification and economics in this run. "
                "Do not re-run with different parameters -- that would be a second holdout look.",
    }
    (OUT_DIR / "holdout_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    for name in ("val", "oos", "holdout"):
        all_candidates[name].to_pickle(OUT_DIR / f"candidates_{name}.pkl")
    print(f"\nWrote {OUT_DIR / 'holdout_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
