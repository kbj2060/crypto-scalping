#!/usr/bin/env python3
"""N-seed stability check for the 9-trigger V자반등 TabPFN cheap_gate (VAL AUC 0.8296 / OOS AUC
0.8119 at seed=20260831 -- research_eth_v_rebound_multitrigger_tabpfn_cheap_gate_20260831.py).
Reuses that script's split/eval logic verbatim (imported, not copy-pasted); only difference is
looping the fit+predict over the SAME 4 seeds this project has reused across every other signal's
own stability check (v7b, fib_extension_exhaustion, etc.) for direct comparability -- not a fresh
random draw, deliberately the established fixed set.

HOLDOUT still untouched (VAL+OOS only). Reports mean/std per split, and per this project's own
"tabm_hp_low_signal_pattern"/CLAUDE.md discipline: std should be much smaller than the mean effect
size to trust the result as signal rather than seed-variance noise.
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
_spec = importlib.util.spec_from_file_location("v_rebound_cheap_gate_20260831", CHEAP_GATE_SCRIPT)
_cg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cg)

OUT_DIR = ROOT / "tmp/eth_v_rebound_multitrigger_tabpfn_seed_stability_20260831"
SEEDS = [20260829, 141592, 271828, 577215]  # same 4 seeds reused project-wide, not a fresh draw


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(_cg.FEATURES_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[df["outcome"].isin(["V자반등", "지지/횡보"])].copy()
    df["label"] = (df["outcome"] == "V자반등").astype(int)
    df = df.dropna(subset=_cg.FEATURE_COLUMNS + ["label"]).reset_index(drop=True)

    parts = _cg.embargoed_split(df)
    for name, part in parts.items():
        print(f"{name}: n={len(part)} label_rate={part['label'].mean():.4f}", flush=True)
    over_limit = len(parts["train"]) > 10000
    print(f"train n={len(parts['train'])} ignore_pretraining_limits={over_limit}\n", flush=True)

    per_seed = []
    for seed in SEEDS:
        print(f"=== seed={seed} ===", flush=True)
        clf = TabPFNClassifier(device="cuda", random_state=seed, ignore_pretraining_limits=over_limit)
        clf.fit(parts["train"][_cg.FEATURE_COLUMNS], parts["train"]["label"].to_numpy())
        row = {"seed": seed}
        for name in ("val", "oos"):
            proba = clf.predict_proba(parts[name][_cg.FEATURE_COLUMNS])[:, 1]
            r = _cg.evaluate(proba, parts[name]["label"].to_numpy())
            row[f"{name}_auc"] = r["auc"]
            row[f"{name}_bal_acc"] = r["balanced_accuracy"]
        print(f"  VAL_AUC={row['val_auc']:.4f} OOS_AUC={row['oos_auc']:.4f} "
              f"VAL_bal_acc={row['val_bal_acc']:.4f} OOS_bal_acc={row['oos_bal_acc']:.4f}", flush=True)
        per_seed.append(row)

    val_aucs = np.array([r["val_auc"] for r in per_seed])
    oos_aucs = np.array([r["oos_auc"] for r in per_seed])
    summary = {
        "n_seeds": len(SEEDS), "seeds": SEEDS,
        "val_auc_mean": round(float(val_aucs.mean()), 4), "val_auc_std": round(float(val_aucs.std()), 4),
        "oos_auc_mean": round(float(oos_aucs.mean()), 4), "oos_auc_std": round(float(oos_aucs.std()), 4),
        "val_auc_min": round(float(val_aucs.min()), 4), "val_auc_max": round(float(val_aucs.max()), 4),
        "oos_auc_min": round(float(oos_aucs.min()), 4), "oos_auc_max": round(float(oos_aucs.max()), 4),
        "sign_consistent_vs_v7b": bool((val_aucs > 0.7342).all() and (oos_aucs > 0.7621).all()),
    }
    print(f"\n=== summary ({len(SEEDS)} seeds) ===")
    print(f"VAL AUC: mean={summary['val_auc_mean']:.4f} std={summary['val_auc_std']:.4f} "
          f"range=[{summary['val_auc_min']:.4f},{summary['val_auc_max']:.4f}]")
    print(f"OOS AUC: mean={summary['oos_auc_mean']:.4f} std={summary['oos_auc_std']:.4f} "
          f"range=[{summary['oos_auc_min']:.4f},{summary['oos_auc_max']:.4f}]")
    print(f"all {len(SEEDS)} seeds beat v7b(sweep-only) on both VAL(0.7342) and OOS(0.7621): "
          f"{summary['sign_consistent_vs_v7b']}")

    report = {"per_seed": per_seed, "summary": summary}
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT_DIR / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
