"""Chain the BTC risk sidecar onto the finished parent tuning sweep.

Waits for scripts/tune_btc_parent_regime_jmredesign_20260810.py to write its aggregate, then runs
the sidecar for the VAL-selected (epochs, train_rows, quality_threshold) configuration across ALL
five seeds rather than one representative bundle.

Running every seed is the point, not thoroughness theatre: the two sidecar rejections so far missed
the `validation_mdd >= -8.0` floor by 2.6 and 2.6 points on a validation replay of only 17 trades,
which is few enough that seed-to-seed variation could straddle the gate. One bundle would tell us
whether that bundle passes; five tell us whether the CONFIGURATION passes, which is what a
promotion claim needs.

If the sweep found no configuration meeting the mean VAL drawdown floor, the sidecar is still run
for the best-by-VAL-pnl configuration. A pre-filter is a prediction about what the sidecar will do;
the sidecar's own verdict is the evidence, and reporting "we never asked" would repeat the mistake
this whole re-run exists to correct.

Waits on the aggregate CSV rather than on a process name -- the earlier chain in this session hung
forever because `pgrep -f <script>` matched the waiting shell's own command line.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/ensemble/reports/jm_redesign_20260810"
RUN_ROOT = ROOT / "tmp/causal_regen_20260516"
AGG = OUT_DIR / "tune_btc_parent_aggregate.csv"
SELECTED = OUT_DIR / "tune_btc_parent_selected.json"
SIDECAR = ROOT / "scripts/train_eval_omega4_2_risk_sidecar_btc_regime_jmredesign_20260810.py"
SEEDS = (260620, 481003, 26611, 903174, 155827)
TAG = "jmredesign_20260810"


def parent_dir(epochs: int, rows: int, seed: int) -> Path | None:
    hits = sorted(RUN_ROOT.glob(f"*{TAG}_e{epochs}_r{rows}_s{seed}"))
    return hits[-1] if hits else None


def wait_for_sweep(timeout_s: int) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if AGG.exists():
            time.sleep(5)   # let the writer finish
            return True
        time.sleep(30)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=str(ROOT / "venv/bin/python"))
    ap.add_argument("--timeout-s", type=int, default=6 * 3600)
    args = ap.parse_args()

    print(f"waiting for {AGG.name} ...", flush=True)
    if not wait_for_sweep(args.timeout_s):
        raise SystemExit(f"parent sweep did not finish within {args.timeout_s}s")

    agg = pd.read_csv(AGG)
    if SELECTED.exists():
        sel = json.loads(SELECTED.read_text())
        note = "VAL-selected (mean VAL pnl s.t. mean VAL mdd >= -8)"
    else:
        best = agg.sort_values("val_pnl_mean", ascending=False).iloc[0]
        sel = {"epochs": int(best["epochs"]), "train_rows": int(best["train_rows"]),
               "quality_threshold": float(best["quality_threshold"]),
               "val_pnl_mean": float(best["val_pnl_mean"]),
               "val_mdd_mean": float(best["val_mdd_mean"])}
        note = "NO config met the VAL drawdown floor; using best mean VAL pnl and asking the sidecar anyway"
    q = sel["quality_threshold"]
    tag_q = f"q{int(round(q * 100)):03d}"
    print(f"\n=== {note}")
    print(f"    epochs={sel['epochs']} train_rows={sel['train_rows']} quality={q:.2f} ({tag_q})"
          f"  VAL pnl {sel['val_pnl_mean']:.2f} mdd {sel['val_mdd_mean']:.2f}\n", flush=True)

    results = []
    for seed in SEEDS:
        d = parent_dir(sel["epochs"], sel["train_rows"], seed)
        if d is None:
            print(f"  [skip] seed={seed}: no parent dir")
            continue
        if not (d / f"validation_predictions_{tag_q}.csv").exists():
            print(f"  [skip] seed={seed}: {d.name} has no {tag_q} predictions")
            continue
        suffix = f"regime_{TAG}_e{sel['epochs']}_r{sel['train_rows']}_s{seed}_{tag_q}"
        cmd = [args.python, str(SIDECAR),
               "--baseline-bundle", str(d / "true_3head_tabm_bundle.pt"),
               "--precomputed-prediction-dir", str(d),
               "--precomputed-prediction-tag", tag_q,
               "--quality-threshold", f"{q:.2f}",
               "--out-suffix", suffix]
        t0 = time.time()
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        ok = r.returncode == 0
        msg = ""
        if not ok:
            tail = [ln for ln in r.stderr.strip().splitlines() if ln.strip()]
            msg = tail[-1][:200] if tail else "unknown failure"
        print(f"  seed={seed} {'PASS' if ok else 'reject'} ({time.time() - t0:.0f}s) {msg}", flush=True)
        results.append({"seed": seed, "parent_dir": d.name, "out_suffix": suffix,
                        "passed": ok, "message": msg})

    n_pass = sum(r["passed"] for r in results)
    print(f"\n=== {n_pass}/{len(results)} seeds produced a sidecar that clears its own gates")
    (OUT_DIR / "sidecar_after_tuning_results.json").write_text(json.dumps(
        {"selection": sel, "selection_note": note, "results": results}, indent=2))
    print(f"-> {OUT_DIR / 'sidecar_after_tuning_results.json'}")


if __name__ == "__main__":
    main()
