"""Test whether averaging the 5 genuinely-random-seed parent bundles' predictions together
(a real ensemble, not just comparing individual seeds) produces more stable VAL/OOS performance
than any single seed -- for SOL and BTC's Omega4.6.1 parent (q0.45 threshold, same 5 seeds used
throughout this seed-variance investigation: 260620/260728/260729/260730/260731).

Reuses the exact same decision/metrics pipeline each training script uses internally
(cat_dq._prediction_output -> parent._to_decisions -> asset omega._metrics), substituting the
seed-averaged direction/quality probability matrices for a single model's output. This is not a
new backtest engine -- it's the same one, fed an averaged input.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

SEEDS = [260620, 260728, 260729, 260730, 260731]
QUALITY_THRESHOLD = 0.45


def load_avg_proba(pred_dir_template: str, split: str, prefix: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    dir_stack, qual_stack, ts_ref = [], [], None
    for seed in SEEDS:
        path = ROOT / pred_dir_template.format(seed=seed) / f"{split}_predictions_q045.csv"
        df = pd.read_csv(path)
        ts = df["timestamp"].to_numpy()
        if ts_ref is None:
            ts_ref = ts
        else:
            assert (ts == ts_ref).all(), f"timestamp mismatch for seed {seed} in {split}"
        dir_stack.append(df[[f"{prefix}dir_p_cash", f"{prefix}dir_p_long", f"{prefix}dir_p_short"]].to_numpy(dtype=np.float64))
        qual_stack.append(df[[f"{prefix}quality_p_cash", f"{prefix}quality_p_long", f"{prefix}quality_p_short"]].to_numpy(dtype=np.float64))
    avg_dir = np.mean(dir_stack, axis=0)
    avg_qual = np.mean(qual_stack, axis=0)
    return pd.DataFrame({"timestamp": ts_ref}), avg_dir, avg_qual


def run_asset(asset: str, script_module: str, omega_module: str, pred_dir_template: str) -> dict:
    sol_main = importlib.import_module(script_module)
    parent = sol_main.parent
    omega = importlib.import_module(omega_module)

    frames = sol_main._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=sol_main.LABEL_DIR,
        quality_mode="hard_rule",
        quality_label_dir=None,
        quality_min_edge=0.0010,
        quality_max_mae=0.0100,
        quality_min_mfe_mae=1.20,
        quality_max_hold_bars=288,
    )
    val_raw, oos_raw = frames["val_raw"], frames["oos_raw"]
    fee, slip = omega._load_fee_slip()

    out = {}
    for split, raw, prefix, oof in [
        ("validation", val_raw, "omega1_regime3_expertdq_oof_", True),
        ("oos", oos_raw, "omega1_regime3_expertdq_", False),
    ]:
        _, avg_dir, avg_qual = load_avg_proba(pred_dir_template, split, prefix)
        assert len(avg_dir) == len(raw), f"{asset} {split}: pred rows {len(avg_dir)} != raw rows {len(raw)}"
        pred_out = parent._prediction_output(raw, avg_dir, avg_qual, threshold=QUALITY_THRESHOLD, prefix=prefix.rstrip("_"))
        dec = parent._to_decisions(pred_out, oof=oof)
        m = omega._metrics(raw, dec, fee=fee, slip=slip, cost_mult=3.0)
        out[split] = {"pnl": m["pnl"], "mdd": m["mdd"], "trades": m["trades"], "wr": m["wr"]}
    return out


ASSET_SPECS = {
    "ETH_baseline": (
        "train_eval_omega4_3head_parent72_loose_entry_quality_20260620",
        "train_eval_omega1_2_tabm_diffusion_risk_20260603",
        "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_seedtest_{seed}_baseline",
    ),
    "ETH_swa": (
        "train_eval_omega4_3head_parent72_loose_entry_quality_20260620",
        "train_eval_omega1_2_tabm_diffusion_risk_20260603",
        "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_seedtest_{seed}_swa",
    ),
    "SOL": (
        "train_eval_omega4_3head_parent72_loose_entry_quality_sol_20260707",
        "train_eval_omega1_2_tabm_diffusion_risk_sol_20260707",
        "tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_sol_seedtest_{seed}",
    ),
    "BTC": (
        "train_eval_omega4_3head_parent72_loose_entry_quality_btc_20260708",
        "train_eval_omega1_2_tabm_diffusion_risk_btc_20260708",
        "tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_20260708_btc_seedtest_{seed}",
    ),
}


def main() -> None:
    # IMPORTANT (found 2026-07-31): running multiple assets' run_asset() in ONE process silently
    # corrupts later assets' results (SOL/BTC numbers changed when run after ETH in-process --
    # root cause not fully isolated, suspected shared-module state in `parent`/`hard`, both of
    # which are singleton-imported across the ETH/SOL/BTC training scripts). Each asset is
    # therefore run in its OWN subprocess for guaranteed isolation -- do not "optimize" this back
    # into a single process without first finding and fixing the actual shared-state bug.
    import subprocess

    if len(sys.argv) > 1 and sys.argv[1] in ASSET_SPECS:
        label = sys.argv[1]
        script_module, omega_module, pred_dir_template = ASSET_SPECS[label]
        r = run_asset(label, script_module, omega_module, pred_dir_template)
        print(json.dumps(r, indent=2, default=str))
        return

    results = {}
    for label in ASSET_SPECS:
        out = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), label],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        if out.returncode != 0:
            raise RuntimeError(f"{label} subprocess failed:\n{out.stdout}\n{out.stderr}")
        json_start = out.stdout.index("{")
        results[label] = json.loads(out.stdout[json_start:])
    print(json.dumps(results, indent=2, default=str))
    with open(ROOT / "data/research/seed_ensemble_average_20260731.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
