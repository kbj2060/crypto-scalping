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

Combinations run CONCURRENTLY (--jobs, default 3). Each combination is an independent subprocess
writing to its own --out-suffix directory, so concurrency changes wall-clock only, never a result.
Per-process thread counts are deliberately left alone: pinning OMP/MKL threads would be faster
still, but thread count can perturb float reduction order, and these runs have to stay numerically
comparable with the default-knob 5-seed rejection they are being measured against.

Stage 2's candidate is chosen by the live contract's own selection key -- validation
log_risk_utility, then validation mdd, then validation pnl -- read back out of each passing run's
report.json. The previous revision took whichever combination happened to come first in grid
iteration order, which is not a selection rule. Every passing combination's metrics are recorded
either way, so the choice stays auditable.

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
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# the project venv was lost on 2026-08-10; runs use the quant_ai conda env instead
PYTHON_BIN = str(Path.home() / "anaconda3/envs/quant_ai/bin/python")
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

SIDECAR_MODEL_ID = "btc_omega4_2_trade_risk_sidecar_20260708"   # out_dir stem the sidecar writes to


def _selection_key(r: dict) -> tuple:
    """The live contract's own key: validation_only + log_risk, highest first."""
    v = r.get("validation") or {}
    return (float(v.get("log_risk_utility", float("-inf"))),
            float(v.get("mdd", float("-inf"))),
            float(v.get("pnl", float("-inf"))))


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
    validation = oos = None
    if ok:
        # read the run's own report so passing combinations can be ranked, not just counted
        report = RUN_ROOT / f"{SIDECAR_MODEL_ID}_{suffix}" / "report.json"
        try:
            sel = json.loads(report.read_text())["selected"]
            validation, oos = sel.get("validation"), sel.get("oos")
            mdd = float(validation["mdd"])
            trades = int(validation["trades"])
        except Exception as exc:                      # a pass with no readable report is suspect
            err = f"passed but report unreadable: {type(exc).__name__}: {exc}"[:220]
    else:
        lines = [ln for ln in r.stderr.strip().splitlines() if ln.strip()]
        err = lines[-1][:220] if lines else "unknown"
        m, t_ = MDD_RE.search(err), TRADES_RE.search(err)
        mdd = float(m.group(1)) if m else None
        trades = int(t_.group(1)) if t_ else None
    return {"seed": seed, "top_k": top_k, "objective": obj, "tail_penalty": tail,
            "model_kind": kind, "ok": ok, "mdd": mdd, "trades": trades,
            "validation": validation, "oos": oos,
            "seconds": round(time.time() - t0), "out_suffix": suffix, "msg": err}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=PYTHON_BIN)
    ap.add_argument("--jobs", type=int, default=3,
                    help="concurrent sidecar subprocesses; each run saturates ~4-5 of 12 cores")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = max(1, int(args.jobs))

    combos = list(product(TOP_K, OBJECTIVE, TAIL_PENALTY, MODEL_KIND))
    print(f"=== stage 1: {len(combos)} sidecar knob combos on probe seed {PROBE_SEED}, "
          f"{jobs} at a time\n    parent e{PARENT_EPOCHS} r{PARENT_ROWS} {QTAG}", flush=True)
    t_stage1 = time.time()
    done = 0

    def stage1_run(combo: tuple) -> dict:
        nonlocal done
        k, o, t, mk = combo
        res = run(PROBE_SEED, k, o, t, mk, args.python)
        done += 1
        status = "PASS" if res["ok"] else f"reject mdd={res['mdd']}"
        print(f"  [{done}/{len(combos)}] top_k={k:<3} {o:<8} tail={t} {mk:<12} "
              f"{status}  ({res['seconds']}s)", flush=True)
        return res

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        stage1 = list(pool.map(stage1_run, combos))

    winners = sorted([r for r in stage1 if r["ok"]], key=_selection_key, reverse=True)
    print(f"\n=== stage 1: {len(winners)}/{len(stage1)} combos pass on the probe seed "
          f"({round(time.time() - t_stage1)}s wall)")
    for r in winners:
        v = r.get("validation") or {}
        print(f"    top_k={r['top_k']:<3} {r['objective']:<8} tail={r['tail_penalty']} "
              f"{r['model_kind']:<12} val_logrisk={v.get('log_risk_utility')} "
              f"val_mdd={v.get('mdd')} val_pnl={v.get('pnl')}", flush=True)

    stage2 = []
    if winners:
        best = winners[0]
        print(f"\n=== stage 2: re-running top_k={best['top_k']} {best['objective']} "
              f"tail={best['tail_penalty']} {best['model_kind']} across all {len(ALL_SEEDS)} seeds "
              f"(ranked first by validation log_risk_utility)", flush=True)
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            stage2 = list(pool.map(
                lambda s: run(s, best["top_k"], best["objective"], best["tail_penalty"],
                              best["model_kind"], args.python), ALL_SEEDS))
        for res in stage2:
            print(f"  seed={res['seed']} "
                  f"{'PASS' if res['ok'] else 'reject mdd=' + str(res['mdd'])} "
                  f"({res['seconds']}s)", flush=True)
        n = sum(r["ok"] for r in stage2)
        print(f"\n=== stage 2: {n}/{len(stage2)} seeds pass")
    else:
        print("no knob combination clears the floor on the probe seed; stage 2 skipped")

    (OUT_DIR / "tune_btc_sidecar_results.json").write_text(json.dumps(
        {"parent": {"epochs": PARENT_EPOCHS, "train_rows": PARENT_ROWS, "quality": QUALITY},
         "probe_seed": PROBE_SEED, "jobs": jobs,
         "stage2_selection": "validation log_risk_utility, then validation mdd, then validation pnl",
         "stage1": stage1, "stage2": stage2}, indent=2))
    print(f"-> {OUT_DIR / 'tune_btc_sidecar_results.json'}")


if __name__ == "__main__":
    main()
