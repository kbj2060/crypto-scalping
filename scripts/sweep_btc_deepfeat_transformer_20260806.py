"""Hyperparameter sweep for the BTC transformer deep-feature encoder
(docs/btc_deepfeat_cnn_transformer_zigzag_soft_label_20260806.md). User chose to keep pushing the
transformer line despite a raw-feature GBDT baseline beating it standalone (65.5%/63.4% val/OOS
acc vs transformer's 63.8%/60.2%) -- this sweep tunes window/d_model/n_layers/dropout to see how
much of that gap the encoder itself can close before any strategy/backtest work.

Random search (not grid) over a modest budget, each config launched as an isolated subprocess of
train_btc_deepfeat_encoders_20260806.py (process isolation avoids GPU memory / RNG-state carryover
between runs). Model selection is strictly on VAL soft-CE loss -- OOS is reported for the selected
best but never used to pick it, per this repo's Fresh-Forward discipline.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts/train_btc_deepfeat_encoders_20260806.py"
OUT_ROOT = ROOT / "tmp/btc_deepfeat_transformer_sweep_20260806"

N_CONFIGS = 16
SEED = 20260806

WINDOW_CHOICES = [24, 48, 96]
D_MODEL_CHOICES = [32, 48, 64, 96]  # all divisible by N_HEADS=4
N_LAYERS_CHOICES = [1, 2, 3]
DROPOUT_CHOICES = [0.15, 0.2, 0.25, 0.3, 0.35]
N_HEADS = 4


def _sample_configs(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    seen: set[tuple] = set()
    configs = []
    while len(configs) < n:
        cfg = {
            "window": int(rng.choice(WINDOW_CHOICES)),
            "d_model": int(rng.choice(D_MODEL_CHOICES)),
            "n_layers": int(rng.choice(N_LAYERS_CHOICES)),
            "dropout": float(rng.choice(DROPOUT_CHOICES)),
        }
        key = tuple(cfg.values())
        if key in seen:
            continue
        seen.add(key)
        configs.append(cfg)
    return configs


def _run_config(idx: int, cfg: dict) -> dict:
    out_dir = OUT_ROOT / f"cfg_{idx:02d}"
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--arch", "transformer",
        "--window", str(cfg["window"]),
        "--d-model", str(cfg["d_model"]),
        "--n-heads", str(N_HEADS),
        "--n-layers", str(cfg["n_layers"]),
        "--dropout", str(cfg["dropout"]),
        "--epochs", "40",
        "--patience", "8",
        "--out-dir", str(out_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if result.returncode != 0:
        return {**cfg, "idx": idx, "status": "failed", "stderr_tail": result.stderr[-2000:]}
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    return {
        **cfg,
        "idx": idx,
        "status": "ok",
        "val_soft_ce_loss": metrics["val"]["soft_ce_loss"],
        "val_hard_top1_acc": metrics["val"]["hard_top1_acc"],
        "oos_soft_ce_loss": metrics["oos"]["soft_ce_loss"],
        "oos_hard_top1_acc": metrics["oos"]["hard_top1_acc"],
        "best_epoch_n": len(metrics["history"]),
    }


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    configs = _sample_configs(N_CONFIGS, SEED)
    results = []
    for i, cfg in enumerate(configs):
        print(f"=== config {i+1}/{len(configs)}: {cfg} ===", flush=True)
        row = _run_config(i, cfg)
        print(json.dumps(row), flush=True)
        results.append(row)

    ok_results = [r for r in results if r["status"] == "ok"]
    if not ok_results:
        raise RuntimeError("all sweep configs failed")
    best = min(ok_results, key=lambda r: r["val_soft_ce_loss"])

    summary = {
        "n_configs": len(configs),
        "n_ok": len(ok_results),
        "gbdt_raw_baseline": {"val_hard_top1_acc": 0.6548824930414072, "oos_hard_top1_acc": 0.6338704014356493},
        "prior_default_transformer": {"val_hard_top1_acc": 0.6376653562512554, "oos_hard_top1_acc": 0.6023485350914837},
        "best_config_by_val_soft_ce_loss": best,
        "all_results": results,
    }
    (OUT_ROOT / "sweep_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"BEST": best}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
