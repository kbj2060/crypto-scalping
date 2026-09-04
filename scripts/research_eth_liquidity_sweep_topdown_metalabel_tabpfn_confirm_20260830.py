#!/usr/bin/env python3
"""TabPFN confirmation pass for liquidity_sweep "top/down" metalabel (Homer signal #2 redo) --
takes the top HORIZON x CLUSTER_GAP candidates from the GBM grid screen
(research_eth_liquidity_sweep_topdown_metalabel_gridscreen_20260830.py) and re-evaluates each with
the project's actual standard model (TabPFN, 4 seeds) on VAL/OOS only -- HOLDOUT (2026-04-01+)
stays reserved/untouched until one config is finally picked (matches this project's single-touch
holdout discipline, docs/homer/README.md 6)).

GBM has consistently underestimated the real signal in every prior Homer signal (V_REBOUND: GBM
0.622/0.643 vs TabPFN 0.642/0.657) -- the grid screen's ranking is trusted, its absolute AUC
values are not assumed to transfer to TabPFN.

Runs on the GPU server (quant_ai env, CUDA required for TabPFN) via handoff.sh.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402
from research_eth_liquidity_sweep_topdown_metalabel_gridscreen_20260830 import build_fires  # noqa: E402
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (  # noqa: E402
    FEATURE_COLUMNS,
    build_indicator_frame,
)

KLINES_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT_DIR = ROOT / "data/labels/eth_5m_liquidity_sweep_topdown_metalabel_20260830"
REPORT_DIR = ROOT / "tmp/eth_liquidity_sweep_topdown_metalabel_20260830"

VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")
SEEDS = [20260829, 141592, 271828, 577215]  # same 4 seeds used across every Homer signal so far

CANDIDATES = [
    {"horizon": 30, "gap": 12},
    {"horizon": 24, "gap": 6},
    {"horizon": 16, "gap": 12},
]


def log(msg: str) -> None:
    print(f"[liq_sweep_topdown_tabpfn_confirm] {msg}", flush=True)


def load_klines() -> pd.DataFrame:
    df = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    assert df["timestamp"].diff().dropna().eq(pd.Timedelta(minutes=5)).all(), "gap/dup in klines"
    return df


def evaluate(proba: np.ndarray, y: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    naive_pred = np.full_like(y, np.bincount(y).argmax())
    return {
        "auc": round(float(roc_auc_score(y, proba)), 4),
        "accuracy": round(float((pred == y).mean()), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "naive_majority_accuracy": round(float((naive_pred == y).mean()), 4),
    }


def run_tabpfn_panel(train: pd.DataFrame, eval_df: pd.DataFrame, tag: str) -> dict:
    from tabpfn import TabPFNClassifier
    seed_rows = []
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed)
        clf.fit(train[FEATURE_COLUMNS], train["hit"].to_numpy().astype(int))
        proba = clf.predict_proba(eval_df[FEATURE_COLUMNS])[:, 1]
        r = evaluate(proba, eval_df["hit"].to_numpy().astype(int))
        r["seed"] = seed
        seed_rows.append(r)
        log(f"    [{tag}] seed={seed}: auc={r['auc']:.4f} acc={r['accuracy']:.4f} "
            f"bal_acc={r['balanced_accuracy']:.4f} (naive={r['naive_majority_accuracy']:.4f})")
    table = pd.DataFrame(seed_rows)
    return {
        "n_eval": int(len(eval_df)),
        "auc_mean": round(float(table["auc"].mean()), 4), "auc_std": round(float(table["auc"].std(ddof=1)), 4),
        "accuracy_mean": round(float(table["accuracy"].mean()), 4),
        "balanced_accuracy_mean": round(float(table["balanced_accuracy"].mean()), 4),
        "naive_majority_accuracy": seed_rows[0]["naive_majority_accuracy"],
        "per_seed": seed_rows,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log("loading klines + building indicator frame + signals (once, shared across candidates)...")
    klines = load_klines()
    sig = compute_signals(klines, btc_df=None, funding_df=None).reset_index(drop=True)
    ind = build_indicator_frame(klines)
    assert len(sig) == len(ind) and (sig["timestamp"].to_numpy() == ind["timestamp"].to_numpy()).all()
    log(f"{len(klines)} bars ready")

    results = []
    for cand in CANDIDATES:
        horizon, gap = cand["horizon"], cand["gap"]
        tag = f"H{horizon}_GAP{gap}"
        log(f"\n=== candidate {tag} ===")
        fires = build_fires(klines, ind, sig, horizon, gap)
        n_before = len(fires)
        fires = fires.dropna(subset=FEATURE_COLUMNS + ["hit"]).reset_index(drop=True)
        ts = fires["timestamp"]
        train = fires.loc[ts < VAL_START].reset_index(drop=True)
        val = fires.loc[(ts >= VAL_START) & (ts < OOS_START)].reset_index(drop=True)
        oos = fires.loc[(ts >= OOS_START) & (ts < HOLDOUT_START)].reset_index(drop=True)
        log(f"  n_fires={n_before}->{len(fires)} usable, hit_rate={fires['hit'].mean():.4f}, "
            f"TRAIN n={len(train)} VAL n={len(val)} OOS n={len(oos)} "
            f"(HOLDOUT n={len(fires.loc[ts >= HOLDOUT_START])}, NOT touched)")

        fires.to_csv(OUT_DIR / f"eth_5m_liquidity_sweep_topdown_metalabel_features_{tag}.csv", index=False)

        log(f"  VAL panel ({tag}, 4 seeds):")
        val_result = run_tabpfn_panel(train, val, "VAL")
        log(f"  VAL -> AUC {val_result['auc_mean']:.4f}+/-{val_result['auc_std']:.4f}")
        log(f"  OOS panel ({tag}, 4 seeds):")
        oos_result = run_tabpfn_panel(train, oos, "OOS")
        log(f"  OOS -> AUC {oos_result['auc_mean']:.4f}+/-{oos_result['auc_std']:.4f}")

        results.append({
            "horizon": horizon, "gap": gap, "tag": tag,
            "n_fires_raw": n_before, "n_usable": len(fires),
            "n_train": len(train), "n_val": len(val), "n_oos": len(oos),
            "hit_rate": round(float(fires["hit"].mean()), 4),
            "val": val_result, "oos": oos_result,
            "val_oos_gap": round(abs(val_result["auc_mean"] - oos_result["auc_mean"]), 4),
            "min_val_oos": round(min(val_result["auc_mean"], oos_result["auc_mean"]), 4),
        })

    log("\n=== SUMMARY (all candidates, VAL/OOS only, HOLDOUT untouched) ===")
    for r in sorted(results, key=lambda x: -x["min_val_oos"]):
        log(f"  {r['tag']}: VAL={r['val']['auc_mean']:.4f} OOS={r['oos']['auc_mean']:.4f} "
            f"min={r['min_val_oos']:.4f} |gap|={r['val_oos_gap']:.4f} hit_rate={r['hit_rate']:.3f} n={r['n_usable']}")

    out_path = REPORT_DIR / "tabpfn_confirm_report.json"
    out_path.write_text(json.dumps({"feature_columns": FEATURE_COLUMNS, "seeds": SEEDS, "candidates": results}, indent=2, default=str))
    log(f"\nreport saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
