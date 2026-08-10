"""BTC risk-sidecar tuning under the redesigned-JM regime, at the VAL-selected parent config.

The "0/5 seeds reject" result was produced with every sidecar knob at its default, and one of those
defaults is very likely the binding constraint rather than the model:

    --full-replay-top-k 1

The failure is `selected risk mapping failed final validation replay constraint`, i.e. the SINGLE
top-ranked mapping missed the -8% validation drawdown floor and the run aborted. With top-k > 1 the
search falls through to the next-ranked mappings, and the misses were small (best seed -9.66 vs the
-8.00 floor). Reporting a rejection without having tried that is the same shortcut this re-run
exists to correct.

Two stages, because the full grid across all seeds is ~4 hours:
  stage 1  sweep the sidecar knobs on ONE seed -- 903174, the seed whose default-knob drawdown came
           closest to the floor (-9.66) and therefore the most informative probe
  stage 2  any knob combination that passes is then re-run across ALL FIVE seeds, because a single
           passing bundle is an initialisation draw, not a passing configuration

Swept: full_replay_top_k, selection_objective, log_tail_penalty, model_kind. The label contract,
the ATR/TP/SL template, quality/exit thresholds and the parent bundle are all held fixed.

IMPORTANT on interpretation: loosening a selection knob until the candidate passes is only
meaningful if the incumbent is judged under the SAME knobs. The BTC live-regime baseline cannot be
re-run right now (its 2025/2026 regime CSVs vanished from disk mid-session), so any pass found here
is provisional and must be re-tested head-to-head once that baseline is restorable.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/ensemble/reports/jm_redesign_20260810"
RUN_ROOT = ROOT / "tmp/causal_regen_20260516"
SIDECAR = ROOT / "scripts/train_eval_omega4_2_risk_sidecar_btc_regime_jmredesign_20260810.py"
TAG = "jmredesign_20260810"

PARENT_EPOCHS, PARENT_ROWS, QUALITY = 3, 30000, 0.50
QTAG = f"q{int(round(QUALITY * 100)):03d}"
PROBE_SEED = 903174                      # closest to the floor under default knobs (-9.66)
ALL_SEEDS = (260620, 481003, 26611, 903174, 155827)

TOP_K = (5, 15, 40)                      # 1 is the already-tested default
OBJECTIVE = ("log_risk", "pnl")
TAIL_PENALTY = (0.5, 1.0)
MODEL_KIND = ("hgb", "extra_trees")

MDD_RE = re.compile(r"mdd=(-?\d+\.?\d*)")
TRADES_RE = re.compile(r"trades=(\d+)")


def parent_dir(seed: int) -> Path | None:
    hits = sorted(RUN_ROOT.glob(f"*{TAG}_e{PARENT_EPOCHS}_r{PARENT_ROWS}_s{seed}"))
    return hits[-1] if hits else None


def run(seed: int, top_k: int, obj: str, tail: float, kind: str, python: str) -> dict:
    d = parent_dir(seed)
    if d is None:
        return {"seed": seed, "ok": False, "msg": "no parent dir"}
    suffix = f"regime_{TAG}_tune_s{seed}_k{top_k}_{obj}_t{tail}_{kind}"
    cmd = [python, str(SIDECAR),
           "--baseline-bundle", str(d / "true_3head_tabm_bundle.pt"),
           "--precomputed-prediction-dir", str(d),
           "--precomputed-prediction-tag", QTAG,
           "--quality-threshold", f"{QUALITY:.2f}",
           "--full-replay-top-k", str(top_k),
           "--selection-objective", obj,
           "--log-tail-penalty", str(tail),
           "--model-kind", kind,
           "--out-suffix", suffix]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    ok = r.returncode == 0
    err = ""
    mdd = trades = None
    if not ok:
        lines = [ln for ln in r.stderr.strip().splitlines() if ln.strip()]
        err = lines[-1][:220] if lines else "unknown"
        m, t_ = MDD_RE.search(err), TRADES_RE.search(err)
        mdd = float(m.group(1)) if m else None
        trades = int(t_.group(1)) if t_ else None
    return {"seed": seed, "top_k": top_k, "objective": obj, "tail_penalty": tail,
            "model_kind": kind, "ok": ok, "mdd": mdd, "trades": trades,
            "seconds": round(time.time() - t0), "out_suffix": suffix, "msg": err}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=str(ROOT / "venv/bin/python"))
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    combos = list(product(TOP_K, OBJECTIVE, TAIL_PENALTY, MODEL_KIND))
    print(f"=== stage 1: {len(combos)} sidecar knob combos on probe seed {PROBE_SEED}\n"
          f"    parent e{PARENT_EPOCHS} r{PARENT_ROWS} {QTAG}", flush=True)
    stage1 = []
    for i, (k, o, t, mk) in enumerate(combos, 1):
        res = run(PROBE_SEED, k, o, t, mk, args.python)
        stage1.append(res)
        status = "PASS" if res["ok"] else f"reject mdd={res['mdd']}"
        print(f"  [{i}/{len(combos)}] top_k={k:<3} {o:<8} tail={t} {mk:<12} "
              f"{status}  ({res['seconds']}s)", flush=True)

    winners = [r for r in stage1 if r["ok"]]
    print(f"\n=== stage 1: {len(winners)}/{len(stage1)} combos pass on the probe seed")
    stage2 = []
    if winners:
        best = winners[0]
        print(f"=== stage 2: re-running top_k={best['top_k']} {best['objective']} "
              f"tail={best['tail_penalty']} {best['model_kind']} across all {len(ALL_SEEDS)} seeds",
              flush=True)
        for s in ALL_SEEDS:
            res = run(s, best["top_k"], best["objective"], best["tail_penalty"],
                      best["model_kind"], args.python)
            stage2.append(res)
            print(f"  seed={s} {'PASS' if res['ok'] else 'reject mdd=' + str(res['mdd'])} "
                  f"({res['seconds']}s)", flush=True)
        n = sum(r["ok"] for r in stage2)
        print(f"\n=== stage 2: {n}/{len(stage2)} seeds pass")
    else:
        print("no knob combination clears the floor on the probe seed; stage 2 skipped")

    (OUT_DIR / "tune_btc_sidecar_results.json").write_text(json.dumps(
        {"parent": {"epochs": PARENT_EPOCHS, "train_rows": PARENT_ROWS, "quality": QUALITY},
         "probe_seed": PROBE_SEED, "stage1": stage1, "stage2": stage2}, indent=2))
    print(f"-> {OUT_DIR / 'tune_btc_sidecar_results.json'}")


if __name__ == "__main__":
    main()
