"""Final-layer tuning for the BTC redesigned-JM candidate: duration gate, scale map, exit threshold.

These are the last untuned layers, and they only became reachable once the sidecar passed (at the
relaxed -20% validation-drawdown floor the user set; the live contract is -8%, so everything here
is exploratory until the incumbent is measured under the same floor).

Pipeline per seed: the frozen parent bundle -> extended-window inference (stage 2, already done by
infer_btc_predictions_ext_regime_jmredesign_20260810.py) -> this replay, which applies

    duration_gate_threshold   entry gate on predicted hold duration (live: 0.0054143218)
    long_scale / short_scale  asymmetric position scaling (live: 0.5 / 2.5)
    exit_threshold            exit-head firing level (live: 0.95)

Selection is VAL-only, as in every other stage. The live BTC promotion was judged on the number
this stage produces (+24.23% VAL / +10.76% OOS-extended), so this is the first output actually
comparable to it -- the sidecar's own `selected` metrics are pre-gate, pre-scale-map and were not.

The grid deliberately includes the live values (0.5/2.5, 0.95, and the live duration gate) so the
candidate is never handed a configuration the incumbent was not also allowed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from itertools import product
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/ensemble/reports/jm_redesign_20260810"
RUN_ROOT = ROOT / "tmp/causal_regen_20260516"
REPLAY = ROOT / "scripts/apply_final_scale_map_btc_regime_jmredesign_20260810.py"
INFER = ROOT / "scripts/infer_btc_predictions_ext_regime_jmredesign_20260810.py"
TAG = "jmredesign_20260810"

EPOCHS, ROWS, QUALITY = 3, 30000, 0.50
QTAG = f"q{int(round(QUALITY * 100)):03d}"
SEEDS = (260620, 481003, 26611, 903174, 155827)

LONG_SCALE = (0.5, 1.0, 1.5)
SHORT_SCALE = (1.0, 1.5, 2.0, 2.5)
EXIT_THRESHOLD = (0.90, 0.95)
DURATION_GATE = (None, 0.0054143218)      # None = no gate; the second is the live BTC value


def parent_dir(seed: int) -> Path | None:
    hits = [p for p in sorted(RUN_ROOT.glob(
        f"btc_omega4_3head_parent72_*{TAG}_e{EPOCHS}_r{ROWS}_s{seed}")) if p.is_dir()]
    return hits[-1] if hits else None


def sidecar_pkl(seed: int) -> Path | None:
    hits = [p for p in sorted(RUN_ROOT.glob(f"*mdd20_s{seed}")) if p.is_dir()]
    return (hits[-1] / "risk_sidecar.pkl") if hits else None


def ext_dir(seed: int) -> Path:
    return RUN_ROOT / f"btc_parent_ext_{TAG}_e{EPOCHS}_r{ROWS}_s{seed}"


def ensure_ext(seed: int, python: str) -> bool:
    d = ext_dir(seed)
    if (d / f"oos_predictions_{QTAG}.csv").exists():
        return True
    r = subprocess.run([python, str(INFER), "--seed", str(seed),
                        "--quality-threshold", f"{QUALITY:.2f}",
                        "--epochs", str(EPOCHS), "--train-rows", str(ROWS)],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        tail = [ln for ln in r.stderr.strip().splitlines() if ln.strip()]
        print(f"  [ext-infer FAILED seed={seed}] {tail[-1][:170] if tail else ''}", flush=True)
        return False
    return True


def replay(seed: int, ls: float, ss: float, ex: float, dg: float | None, python: str) -> dict:
    p, sc = parent_dir(seed), sidecar_pkl(seed)
    tagbits = f"s{seed}_L{ls}_S{ss}_x{ex}_d{'none' if dg is None else 'live'}"
    out = RUN_ROOT / f"btc_final_{TAG}_{tagbits}"
    cmd = [python, str(REPLAY),
           "--baseline-bundle", str(p / "true_3head_tabm_bundle.pt"),
           "--sidecar-pkl", str(sc),
           "--precomputed-prediction-dir", str(ext_dir(seed)),
           "--precomputed-prediction-tag", QTAG,
           "--quality-threshold", f"{QUALITY:.2f}",
           "--long-scale", str(ls), "--short-scale", str(ss),
           "--exit-threshold", str(ex), "--device", "cpu",
           "--out-dir", str(out)]
    if dg is not None:
        cmd += ["--duration-gate-threshold", str(dg)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        tail = [ln for ln in r.stderr.strip().splitlines() if ln.strip()]
        return {"ok": False, "msg": tail[-1][:180] if tail else "unknown"}
    rep = out / "report.json"
    if not rep.exists():
        return {"ok": False, "msg": "no report.json"}
    d = json.loads(rep.read_text())
    res = {"ok": True}
    for split in ("validation", "oos"):
        blk = d.get(split) or d.get(f"{split}_replay") or {}
        if isinstance(blk, dict):
            res[f"{split}_pnl"] = blk.get("pnl")
            res[f"{split}_mdd"] = blk.get("mdd")
            res[f"{split}_trades"] = blk.get("trades")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=str(ROOT / "venv/bin/python"))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== stage 2: extended-window inference per seed", flush=True)
    ready = [s for s in args.seeds if ensure_ext(s, args.python)]
    print(f"    {len(ready)}/{len(args.seeds)} seeds have extended predictions\n", flush=True)

    combos = list(product(LONG_SCALE, SHORT_SCALE, EXIT_THRESHOLD, DURATION_GATE))
    print(f"=== stage 4: {len(combos)} final-layer combos x {len(ready)} seeds", flush=True)
    rows = []
    t0 = time.time()
    for i, (ls, ss, ex, dg) in enumerate(combos, 1):
        for s in ready:
            res = replay(s, ls, ss, ex, dg, args.python)
            rows.append({"seed": s, "long_scale": ls, "short_scale": ss, "exit_threshold": ex,
                         "duration_gate": dg, **res})
        done = [r for r in rows if r["long_scale"] == ls and r["short_scale"] == ss
                and r["exit_threshold"] == ex and r["duration_gate"] == dg and r.get("ok")]
        if done:
            vp = sum(r.get("validation_pnl") or 0 for r in done) / len(done)
            op = sum(r.get("oos_pnl") or 0 for r in done) / len(done)
            print(f"  [{i}/{len(combos)}] L={ls} S={ss} exit={ex} dur={'live' if dg else 'none'}"
                  f"  VAL {vp:+7.2f}  OOS {op:+7.2f}  ({len(done)}/{len(ready)} seeds)", flush=True)
        else:
            print(f"  [{i}/{len(combos)}] L={ls} S={ss} exit={ex} "
                  f"dur={'live' if dg else 'none'}  all seeds failed", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "tune_btc_final_layers_runs.csv", index=False)
    ok = df[df["ok"] == True]  # noqa: E712
    if len(ok):
        agg = (ok.groupby(["long_scale", "short_scale", "exit_threshold", "duration_gate"],
                          dropna=False)
                 .agg(val_pnl=("validation_pnl", "mean"), val_mdd=("validation_mdd", "mean"),
                      oos_pnl=("oos_pnl", "mean"), oos_mdd=("oos_mdd", "mean"),
                      n=("seed", "count")).reset_index())
        agg.to_csv(OUT_DIR / "tune_btc_final_layers_aggregate.csv", index=False)
        best = agg.sort_values("val_pnl", ascending=False).iloc[0]
        print("\n=== top 8 by mean VAL pnl (selection is VAL-only)")
        print(agg.sort_values("val_pnl", ascending=False).head(8).to_string(
            index=False, float_format=lambda v: f"{v:8.3f}"))
        print(f"\nSELECTED: L={best['long_scale']} S={best['short_scale']} "
              f"exit={best['exit_threshold']} dur={best['duration_gate']}  "
              f"VAL {best['val_pnl']:+.2f}/{best['val_mdd']:.2f}  "
              f"OOS {best['oos_pnl']:+.2f}/{best['oos_mdd']:.2f}")
    print(f"\ntotal {time.time() - t0:.0f}s -> {OUT_DIR}/tune_btc_final_layers_*.csv")


if __name__ == "__main__":
    main()
