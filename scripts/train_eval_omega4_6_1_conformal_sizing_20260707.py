"""Omega4.6.1 -- next-version candidate 2: conformal-style recalibration of the L4 risk-sidecar's
confidence transform (does NOT touch the live-wired omega4_6_1_duration_ou_halflife_risk_gate_20260630
base -- separate frozen-independent candidate: omega4_6_1_conformal_sizing_20260707).

Diagnostic finding (scripts/diagnose_risk_sidecar_calibration_20260707.py) that motivates this:
the L4 sidecar's raw risk score, when checked against REALIZED trade outcomes (using each
component run alone so there's more data than the tiny combined-greedy trade counts):
  - h48qual: VAL spearman(score,return) = -0.043 (p=0.83, pure noise) but OOS = -0.406 (p=0.036,
    SIGNIFICANT AND NEGATIVE) -- the sizing model is backwards for h48qual: it sizes UP exactly
    when it should size DOWN, and this isn't visible on VAL at all.
  - zig075: VAL spearman = +0.002 (raw corr +0.297) but margin-vs-win spearman is positive in
    BOTH windows (+0.161 VAL / +0.250 OOS) -- a real, if modest, signal.

The current L4 formula is `unit = sigmoid(temp * z)` where z is a historical-quantile z-score of
the raw HGB score, with temp/min_scale/max_scale/floor/cap frozen from original 2026-06-30 tuning
-- NOT fit against realized trade outcomes at all, just a generic distributional rescaling.

Candidate fix: replace `unit = sigmoid(temp * z)` with `unit = sigmoid(a + b * z)` where (a, b) are
fit via regularized logistic regression of z -> win on the VAL-window alone-component ledger (more
samples than the tiny combined-greedy counts). This is the standard conformal/Platt-style
recalibration idea: don't invent a new signal, recalibrate the SIGN and SLOPE of the existing one
against realized outcomes. Everything else (min_scale/max_scale/floor/cap/leverage params, side
scales, L7 SCALE_MAP, duration gate) stays frozen exactly as in the live model.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as sidecar  # noqa: E402
import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import replay_omega4_6_1_greedy_val_20260706 as valmod  # noqa: E402
from test_omega4_6_1_drop_h48qual_20260706 import _metrics  # noqa: E402
from diagnose_risk_sidecar_calibration_20260707 import score_for, prep_val_pred  # noqa: E402

MODEL_ID = "omega4_6_1_conformal_sizing_20260707"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_conformal_sizing_20260707"
DEVICE = retest.DEVICE


def fit_calibration(z_train: np.ndarray, win_train: np.ndarray) -> tuple[float, float]:
    """Regularized 1D logistic recalibration z -> P(win). Returns (intercept, coef)."""
    if len(np.unique(win_train)) < 2:
        return 0.0, 0.0
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(z_train.reshape(-1, 1), win_train)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def recalibrated_component(comp: dict, z: np.ndarray, pkl: dict, calib: tuple[float, float]) -> dict:
    """Return a copy of `comp` with margin/leverage recomputed using the VAL-fit (a,b) calibration
    instead of the frozen sigmoid(temp*z), keeping every other sizing parameter identical."""
    a, b = calib
    unit = 1.0 / (1.0 + np.exp(-(a + b * z)))
    m = pkl["selected_mapping"]
    dec = comp["dec"]
    side = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    leverage0 = pd.to_numeric(dec["leverage"], errors="raise").to_numpy(dtype=np.float64)
    notional0 = pd.to_numeric(dec["notional_exposure"], errors="raise").to_numpy(dtype=np.float64)
    base_margin = notional0 / np.maximum(leverage0, 1e-12)
    scale = float(m["min_scale"]) + (float(m["max_scale"]) - float(m["min_scale"])) * unit
    margin = np.clip(base_margin * scale, float(m["floor"]), float(m["cap"]))
    margin[side > 0] *= float(m["long_scale"])
    margin[side < 0] *= float(m["short_scale"])
    margin = np.clip(margin, float(m["floor"]), float(m["cap"]))
    margin[~omega._active(dec)] = 0.0

    unit_l = 1.0 / (1.0 + np.exp(-(a + b * z) * (float(m["leverage_temp"]) / max(float(m["temp"]), 1e-8))))
    lev = float(m["leverage_min"]) + (float(m["leverage_max"]) - float(m["leverage_min"])) * unit_l
    lev[side > 0] *= float(m["long_leverage_scale"])
    lev[side < 0] *= float(m["short_leverage_scale"])
    lev = np.clip(lev, float(m["leverage_floor"]), float(m["leverage_cap"]))
    lev[~omega._active(dec)] = 0.0

    comp2 = dict(comp)
    comp2["margin"] = margin
    comp2["leverage"] = lev
    return comp2


def build_val():
    frame = valmod.load_val_frame()
    comps, paths, decs = {}, {}, {}
    for cname, cfg in retest.COMPONENTS.items():
        frame, tmp = prep_val_pred(cname, cfg, frame)
        paths[cname] = tmp
    for cname, cfg in retest.COMPONENTS.items():
        comps[cname] = greedy.prepare_component(frame, paths[cname], cfg, DEVICE)
    return frame, comps, paths


def build_oos():
    frame = retest.load_frame_current("2026-01-01", "2026-06-30")
    comps, paths = {}, {}
    for cname, cfg in retest.COMPONENTS.items():
        pred_csv = retest.EXT_PRED_DIR / cname / f"oos_predictions_{cfg['q_tag']}.csv"
        paths[cname] = pred_csv
        comps[cname] = greedy.prepare_component(frame, pred_csv, cfg, DEVICE)
    return frame, comps, paths


def run_router(frame, comps, fee, slip):
    greedy.PRIORITY = ("h48qual", "zig075")
    return greedy.greedy_replay(frame, comps, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=DEVICE)


def report(name: str, frame, ledger) -> dict:
    ng = _metrics(ledger, frame, apply_gate=False)
    wg = _metrics(ledger, frame, apply_gate=True)
    print(f"  {name:34s} no_gate: pnl={ng['pnl']:+7.2f}% mdd={ng['mdd']:+6.2f}% n={ng['trades']:2d} wr={ng['wr']:.3f}  |  "
          f"gate: pnl={wg['pnl']:+7.2f}% mdd={wg['mdd']:+6.2f}% n={wg['trades']:2d} wr={wg['wr']:.3f}")
    return {"no_gate": ng, "with_gate": wg}


def main() -> int:
    fee, slip = omega._load_fee_slip()

    print("### Step 1: fit calibration (a,b) per component on VAL alone-ledgers ###")
    val_frame, val_comp, val_paths = build_val()
    calib = {}
    for cname, cfg in retest.COMPONENTS.items():
        dec, score, z, pkl = score_for(val_frame, val_paths[cname], cfg)
        greedy.PRIORITY = (cname,)
        comp_alone = greedy.prepare_component(val_frame, val_paths[cname], cfg, DEVICE)
        _, lg = greedy.greedy_replay(val_frame, {cname: comp_alone}, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=DEVICE)
        z_train = lg["entry_signal_i"].map(lambda i: z[i]).to_numpy()
        win_train = lg["win"].to_numpy()
        a, b = fit_calibration(z_train, win_train)
        calib[cname] = (a, b, pkl)
        print(f"  {cname:8s}: fit on n={len(lg)} VAL trades -> intercept={a:+.4f} slope={b:+.4f} "
              f"(slope>0 means score positively predicts win per VAL)")

    print("\n### Step 2: apply calibration, run genuine combined greedy router ###")
    print("\n################ VAL 2025-10-01..12-31 (SELECTION window) ################")
    _, baseline_val = run_router(val_frame, val_comp, fee, slip)
    base_val_res = report("BASELINE (frozen sigmoid sizing)", val_frame, baseline_val)

    val_comp_recal = {}
    for cname, cfg in retest.COMPONENTS.items():
        _, score, z, pkl = score_for(val_frame, val_paths[cname], cfg)
        a, b, _ = calib[cname]
        val_comp_recal[cname] = recalibrated_component(val_comp[cname], z, pkl, (a, b))
    _, recal_val = run_router(val_frame, val_comp_recal, fee, slip)
    recal_val_res = report("CONFORMAL-RECALIBRATED sizing", val_frame, recal_val)

    print("\n################ OOS 2026-01-01..06-30 (one-shot confirm, frozen calibration from VAL) ################")
    oos_frame, oos_comp, oos_paths = build_oos()
    _, baseline_oos = run_router(oos_frame, oos_comp, fee, slip)
    base_oos_res = report("BASELINE (frozen sigmoid sizing)", oos_frame, baseline_oos)

    oos_comp_recal = {}
    for cname, cfg in retest.COMPONENTS.items():
        _, score, z, pkl = score_for(oos_frame, oos_paths[cname], cfg)
        a, b, _ = calib[cname]
        oos_comp_recal[cname] = recalibrated_component(oos_comp[cname], z, pkl, (a, b))
    _, recal_oos = run_router(oos_frame, oos_comp_recal, fee, slip)
    recal_oos_res = report("CONFORMAL-RECALIBRATED sizing", oos_frame, recal_oos)

    print("\n################ VERDICT ################")
    vb, vr = base_val_res["with_gate"]["pnl"], recal_val_res["with_gate"]["pnl"]
    ob, orr = base_oos_res["with_gate"]["pnl"], recal_oos_res["with_gate"]["pnl"]
    print(f"VAL:  recalibrated {vr:+.2f}% vs baseline {vb:+.2f}%  -> {'IMPROVED' if vr>vb else 'WORSE/EQUAL'}")
    print(f"OOS:  recalibrated {orr:+.2f}% vs baseline {ob:+.2f}%  -> {'IMPROVED' if orr>ob else 'WORSE/EQUAL'}")
    both = (vr > vb) and (orr > ob)
    print(f"CONSISTENT IMPROVEMENT ACROSS BOTH WINDOWS: {both}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import json
    (OUT_DIR / "result.json").write_text(json.dumps({
        "model_id": MODEL_ID,
        "calibration": {k: {"intercept": v[0], "slope": v[1]} for k, v in calib.items()},
        "val": {"baseline": base_val_res, "recalibrated": recal_val_res},
        "oos": {"baseline": base_oos_res, "recalibrated": recal_oos_res},
        "consistent_improvement": bool(both),
    }, indent=2))
    recal_oos.to_csv(OUT_DIR / "oos_ledger_conformal.csv", index=False)
    recal_val.to_csv(OUT_DIR / "val_ledger_conformal.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
