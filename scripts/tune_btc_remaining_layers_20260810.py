"""Remaining-layer tuning for the BTC redesigned-JM stack, after the sidecar knob axis died.

What the sidecar sweep established (17 combos, probe seed 903174) before it was stopped:
    model_kind=hgb          -> validation MDD -9.6622, ALWAYS
    model_kind=extra_trees  -> validation MDD -9.8798, ALWAYS
`selection_objective` (log_risk vs pnl) and `log_tail_penalty` (0.5 vs 1.0) do not move the number
by even 1e-4 -- re-ranking the eligible mappings leaves the same one on top. `full_replay_top_k` is
inert by construction (only the top-1 mapping is checked against the floor; top-k is reported, not
fallen back to) and at 40 it cost 1308s per run, so the remaining combos were pure waste.

So the sidecar's *selection* stage cannot fix a -9.7% drawdown against a -8.0% floor. The drawdown
is a property of the parent's trade ledger, which means the remaining leverage is upstream. Two
stages here, cheapest-first:

  stage A  the two untested risk models (random_forest, gradient_boosting). Completes the
           model_kind axis; ~4 min each.
  stage B  parent DIRECTION/QUALITY class weights. This is the first genuinely untouched parent
           axis that changes which trades get taken -- and therefore the drawdown -- without
           touching any label definition. Up-weighting CASH makes the parent more selective; a
           smaller, higher-conviction ledger is the standard route to a lower drawdown.

Not touched here, deliberately: the exit-head parameters (exit_edge_min, exit_terminal_window,
exit_adverse_unreal, exit_min_mfe_for_giveback, exit_giveback_min). Those DEFINE the exit head's
label, so sweeping them changes the target rather than the model, and the incumbent's values are
part of the live contract. If stage B fails they are the next candidate, but they must then be
flagged as a contract change and offered to the incumbent too.

Everything is measured against the same `validation_mdd >= -8.0` floor the live sidecar enforces.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/ensemble/reports/jm_redesign_20260810"
RUN_ROOT = ROOT / "tmp/causal_regen_20260516"
SIDECAR = ROOT / "scripts/train_eval_omega4_2_risk_sidecar_btc_regime_jmredesign_20260810.py"
PARENT = ROOT / "scripts/train_eval_omega4_3head_parent72_btc_regime_jmredesign_20260810.py"
TAG = "jmredesign_20260810"

EPOCHS, ROWS, QUALITY = 3, 30000, 0.50
QTAG = f"q{int(round(QUALITY * 100)):03d}"
PROBE_SEED = 903174
MDD_RE = re.compile(r"mdd=(-?\d+\.?\d*)")

STAGE_A_MODELS = ("random_forest", "gradient_boosting")
# Class-weight strings are "<int_class>:<weight>". Read off the live report: the QUALITY head's
# class 0 is CASH (36,518 of 78,624 train rows, matching its 0.5355 active ratio), and the quality
# head is the gate that decides whether a signal becomes a trade at all. Up-weighting CASH there
# makes the parent more selective, which shrinks the ledger toward higher-conviction trades -- the
# standard route to a lower drawdown. Direction-head weights are left at the live default, since
# rebalancing long-vs-short is a different intervention with no drawdown rationale.
STAGE_B_QUALITY_WEIGHTS = ("", "0:1.5", "0:2.0", "0:3.0")


def parent_dir(seed: int, wtag: str = "") -> Path | None:
    pat = f"*{TAG}_e{EPOCHS}_r{ROWS}_s{seed}" + (f"_{wtag}" if wtag else "")
    hits = [p for p in sorted(RUN_ROOT.glob(pat)) if p.is_dir()]
    return hits[-1] if hits else None


def run_sidecar(d: Path, suffix: str, model_kind: str, python: str) -> dict:
    cmd = [python, str(SIDECAR),
           "--baseline-bundle", str(d / "true_3head_tabm_bundle.pt"),
           "--precomputed-prediction-dir", str(d),
           "--precomputed-prediction-tag", QTAG,
           "--quality-threshold", f"{QUALITY:.2f}",
           "--model-kind", model_kind,
           "--out-suffix", suffix]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    ok = r.returncode == 0
    mdd, err = None, ""
    if not ok:
        lines = [ln for ln in r.stderr.strip().splitlines() if ln.strip()]
        err = lines[-1][:200] if lines else "unknown"
        m = MDD_RE.search(err)
        mdd = float(m.group(1)) if m else None
    return {"ok": ok, "mdd": mdd, "seconds": round(time.time() - t0), "msg": err}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=str(ROOT / "venv/bin/python"))
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"stage_a": [], "stage_b": []}

    d = parent_dir(PROBE_SEED)
    print(f"=== stage A: remaining risk models on {d.name if d else '?'}", flush=True)
    for mk in STAGE_A_MODELS:
        res = run_sidecar(d, f"regime_{TAG}_mk_{mk}_s{PROBE_SEED}", mk, args.python)
        verdict = "PASS" if res["ok"] else f"reject mdd={res['mdd']}"
        print(f"  {mk:<18} {verdict}  ({res['seconds']}s)", flush=True)
        report["stage_a"].append({"model_kind": mk, **res})

    if any(r["ok"] for r in report["stage_a"]):
        print("\nstage A found a passing risk model; stage B not needed for a first read")
    else:
        print(f"\n=== stage B: parent class-weight sweep (seed {PROBE_SEED}), then sidecar each",
              flush=True)
        for w in STAGE_B_QUALITY_WEIGHTS:
            wtag = ("q" + w.replace(":", "_").replace(".", "")) if w else "qbase"
            suffix = f"h48qual_regime_{TAG}_e{EPOCHS}_r{ROWS}_s{PROBE_SEED}_{wtag}"
            cmd = [args.python, str(PARENT), "--epochs", str(EPOCHS),
                   "--max-train-rows", str(ROWS), "--seed", str(PROBE_SEED),
                   "--out-suffix", suffix]
            if w:
                cmd += ["--quality-class-weights", w]
            t0 = time.time()
            pr = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            if pr.returncode != 0:
                tail = [ln for ln in pr.stderr.strip().splitlines() if ln.strip()]
                print(f"  weights={w or 'live-default':<14} PARENT FAILED: {tail[-1][:160] if tail else ''}",
                      flush=True)
                report["stage_b"].append({"weights": w, "parent_ok": False})
                continue
            hits = [p for p in sorted(RUN_ROOT.glob(f"*{suffix}")) if p.is_dir()]
            pd_ = hits[-1] if hits else None
            if pd_ is None or not (pd_ / f"validation_predictions_{QTAG}.csv").exists():
                print(f"  weights={w or 'live-default':<14} no {QTAG} predictions", flush=True)
                report["stage_b"].append({"weights": w, "parent_ok": True, "sidecar": None})
                continue
            res = run_sidecar(pd_, f"regime_{TAG}_{wtag}_s{PROBE_SEED}", "hgb", args.python)
            verdict = "PASS" if res["ok"] else f"reject mdd={res['mdd']}"
            label = w or "live-default"
            print(f"  weights={label:<14} parent {time.time() - t0 - res['seconds']:.0f}s"
                  f" | sidecar {verdict}", flush=True)
            report["stage_b"].append({"weights": w, "parent_ok": True, "parent_dir": pd_.name,
                                      "sidecar": res})

    (OUT_DIR / "tune_btc_remaining_layers.json").write_text(json.dumps(report, indent=2))
    print(f"\n-> {OUT_DIR / 'tune_btc_remaining_layers.json'}")


if __name__ == "__main__":
    main()
