"""Full BTC parent hyperparameter tuning under the redesigned-JM regime overlay.

The first retrain froze every tunable at the LIVE model's values, including `quality_threshold=0.55`
-- a value the incumbent had itself VAL-selected under the OLD regime. Under the new regime VAL
actually prefers 0.50, so the sidecar was run at a threshold this candidate never chose and its
rejection said nothing about the candidate. This script does the tuning properly.

Swept (the label contract is NOT touched -- direction/quality/exit label definitions stay exactly
at the live contract, so this tunes the model, not the target):
    seed          5 randomly drawn values, not a fixed-increment ladder, per the project's
                  seed-diversity gate; the live seed 260620 is included as the regression anchor
    epochs        3 | 4 (live) | 6
    max_train_rows 30000 (live) | 45000

Selection, VAL only:
    for each (epochs, train_rows, quality_threshold) average VAL pnl and VAL mdd ACROSS SEEDS, then
    take the highest mean VAL pnl subject to mean VAL mdd >= -8.0. The drawdown floor is applied
    here, at parent selection, because it is the same floor the downstream sidecar enforces -- there
    is no point selecting a parent the risk stage will reject, which is exactly what happened on the
    first pass.

Averaging across seeds rather than taking the best single run is deliberate: a per-seed maximum is
an initialisation draw, and this project has already been burned once by a clustered-seed result
(2026-08-01 Sigma3-1h audit).

OOS is computed by the parent script but is never read here.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/ensemble/reports/jm_redesign_20260810"
RUN_ROOT = ROOT / "tmp/causal_regen_20260516"
BASE = "btc_omega4_3head_parent72_loose_entry_quality_swingtransition_20260806_h48qual_"
SCRIPT = ROOT / "scripts/train_eval_omega4_3head_parent72_btc_regime_jmredesign_20260810.py"

SEEDS = (260620, 481003, 26611, 903174, 155827)   # 260620 = live anchor
EPOCHS = (3, 4, 6)
TRAIN_ROWS = (30000, 45000)
VAL_MDD_FLOOR = -8.0


def suffix(epochs: int, rows: int, seed: int) -> str:
    return f"h48qual_regime_jmredesign_20260810_e{epochs}_r{rows}_s{seed}"


def run_one(epochs: int, rows: int, seed: int, python: str) -> Path | None:
    # the parent script composes its own output dir from --out-suffix, so resolve it by globbing
    # after the run rather than trying to predict the prefix
    cmd = [python, str(SCRIPT), "--epochs", str(epochs), "--max-train-rows", str(rows),
           "--seed", str(seed), "--out-suffix", suffix(epochs, rows, seed)]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAILED e{epochs} r{rows} s{seed}: {r.stderr.strip().splitlines()[-1][:160]}")
        return None
    hits = sorted(RUN_ROOT.glob(f"*{suffix(epochs, rows, seed)}"))
    print(f"  ok e{epochs} r{rows} s{seed}  ({time.time() - t0:.0f}s) -> {hits[-1].name if hits else '?'}")
    return hits[-1] if hits else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=str(ROOT / "venv/bin/python"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    combos = list(product(EPOCHS, TRAIN_ROWS, SEEDS))
    print(f"=== BTC parent tuning: {len(combos)} runs "
          f"({len(EPOCHS)} epochs x {len(TRAIN_ROWS)} train_rows x {len(SEEDS)} seeds)")
    if args.dry_run:
        for e, r, s in combos:
            print("  ", suffix(e, r, s))
        return

    rows = []
    t0 = time.time()
    for i, (e, r, s) in enumerate(combos, 1):
        print(f"[{i}/{len(combos)}] epochs={e} train_rows={r} seed={s}", flush=True)
        d = run_one(e, r, s, args.python)
        if d is None:
            continue
        rank = pd.read_csv(d / "quality_threshold_ranking.csv")
        for _, row in rank.iterrows():
            rows.append({"epochs": e, "train_rows": r, "seed": s, "dir": d.name,
                         "quality_threshold": float(row["quality_threshold"]),
                         "val_pnl": float(row["validation_pnl"]),
                         "val_mdd": float(row["validation_mdd"]),
                         "val_trades": int(row["validation_trades"]),
                         "oos_pnl": float(row["oos_pnl"]),
                         "oos_mdd": float(row["oos_mdd"])})
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "tune_btc_parent_runs.csv", index=False)

    agg = (df.groupby(["epochs", "train_rows", "quality_threshold"])
             .agg(val_pnl_mean=("val_pnl", "mean"), val_pnl_std=("val_pnl", "std"),
                  val_pnl_min=("val_pnl", "min"), val_mdd_mean=("val_mdd", "mean"),
                  val_mdd_worst=("val_mdd", "min"), val_trades_mean=("val_trades", "mean"),
                  oos_pnl_mean=("oos_pnl", "mean"), n=("seed", "count"))
             .reset_index())
    agg.to_csv(OUT_DIR / "tune_btc_parent_aggregate.csv", index=False)

    eligible = agg[agg["val_mdd_mean"] >= VAL_MDD_FLOOR]
    print(f"\n=== {len(eligible)}/{len(agg)} configs meet mean VAL MDD >= {VAL_MDD_FLOOR}")
    show = (eligible if len(eligible) else agg).sort_values("val_pnl_mean", ascending=False)
    print(show.head(12).to_string(index=False,
          columns=["epochs", "train_rows", "quality_threshold", "val_pnl_mean", "val_pnl_std",
                   "val_mdd_mean", "val_mdd_worst", "val_trades_mean", "oos_pnl_mean", "n"],
          float_format=lambda v: f"{v:8.3f}"))
    if len(eligible):
        best = show.iloc[0]
        print(f"\nSELECTED (VAL only): epochs={int(best['epochs'])} "
              f"train_rows={int(best['train_rows'])} q={best['quality_threshold']:.2f}  "
              f"VAL pnl {best['val_pnl_mean']:.2f}+-{best['val_pnl_std']:.2f} "
              f"mdd {best['val_mdd_mean']:.2f}")
        (OUT_DIR / "tune_btc_parent_selected.json").write_text(json.dumps({
            "epochs": int(best["epochs"]), "train_rows": int(best["train_rows"]),
            "quality_threshold": float(best["quality_threshold"]),
            "val_pnl_mean": float(best["val_pnl_mean"]),
            "val_mdd_mean": float(best["val_mdd_mean"]),
            "seeds": list(SEEDS), "selection": "mean VAL pnl s.t. mean VAL mdd >= -8, seeds averaged",
        }, indent=2))
    print(f"\ntotal {time.time() - t0:.0f}s  -> {OUT_DIR}/tune_btc_parent_*.csv")


if __name__ == "__main__":
    main()
