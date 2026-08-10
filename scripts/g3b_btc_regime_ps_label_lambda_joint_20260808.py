"""G3b — corrected re-run: featset x lambda selected JOINTLY on VAL  (2026-08-08)

Contract: docs/experiments/btc_regime_ps_label_learnability_g3b_20260808.json
Supersedes the halted G3 (docs/experiments/btc_regime_ps_label_learnability_g3_20260808.json).

WHAT WAS WRONG.  G3 evaluated the 52.4-bar median-run floor at Stage 1 under an INHERITED
lambda=2, while lambda is the knob that CONTROLS run length and was only to be swept at Stage 2.
Every cell landed at 28-33 bars and the run halted before lambda was ever tuned.  Applying a
persistence floor before the persistence knob is chosen does not serve the floor's purpose — it
tests one arbitrary lambda.

WHAT CHANGES: only WHERE the gate is evaluated.  The floor VALUE is unchanged at 52.4 bars, and
lambda is extended to {1,2,4,8,16,32} because runs sat at 28-33 at lambda=2 — extended before any
lambda result was seen.  Mode is fixed to `state`, which beat `geo` in all three feature sets on
G3's recorded VAL screen.  OOS was never read by G3, so this contract still makes ONE clean read.

The value gate is left as pre-registered at "> 0" rather than tightened mid-arc, but note up front
that it is WEAK: the no-learning czz0.5 returns +533% traded on OOS, so a detector can pass this
gate while being far worse than a definitional rule.  The verdict reports traded return against
every baseline and says so plainly if that happens.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from chart_btc_jm_regime_verification_20260808 import causal_zigzag  # noqa: E402
from refine_btc_regime_classifier_theta005_20260808 import (  # noqa: E402
    MIN_COVERAGE, PANEL_PATH, PURGE, SCORE_SCALES, jump_decode_proba, summarize, to_named,
)
from reselect_btc_regime_classifier_zigzag_only_20260808 import zigzag_geometry  # noqa: E402
from stage0_btc_regime_label_design_20260808 import BEAR, BULL, CHOP, COST  # noqa: E402
from stage0c_btc_regime_label_oracle_dp_20260808 import trade  # noqa: E402
from stage0e_btc_regime_label_pagan_sossounov_20260808 import ps_label, ps_pivots  # noqa: E402
from test_statistical_jump_model_regimes_20260808 import contiguous_runs, zigzag_oracle  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    SEED, TRAIN_END, VAL_START, VAL_END, OOS_START, OOS_END,
)

OUT_DIR = ROOT / "tmp/btc_regime_ps_g3b_20260808"
OUT_PARQUET = ROOT / "data/research/btc_regime_ps_g3b_states_20260808.parquet"
PS_P, PS_A = 48, 0.02
FEATSETS = {
    "S2_fine5": [0.001, 0.002, 0.0035, 0.005, 0.008],
    "S_coarse5": [0.0035, 0.005, 0.008, 0.012, 0.020],
    "S_wide7": [0.001, 0.002, 0.0035, 0.005, 0.008, 0.012, 0.020],
}
LAMBDAS = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
CZZ_BASELINES = [0.005, 0.008, 0.012, 0.020, 0.030]
N_SEEDS = 5
INCUMBENT_VAL, INCUMBENT_OOS, REPRO_TOL = 70.1, 68.0, 0.3


def summ(named: np.ndarray, tgt: np.ndarray, idx: np.ndarray, logret: np.ndarray) -> dict:
    d = np.where(named == 2, 1, np.where(named == 0, -1, 0)).astype(np.int8)
    st3 = np.where(d == 1, BULL, np.where(d == -1, BEAR, CHOP)).astype(np.int8)
    runs = [e - s + 1 for s, e, _ in contiguous_runs(named[idx])]
    m = idx[(d[idx] != 0) & (tgt[idx] != 0)]
    return {"agree": round(float(np.mean(d[m] == tgt[m])) * 100, 1) if len(m) >= 50 else None,
            "coverage_pct": round(float((d[idx] != 0).mean()) * 100, 1),
            "median_run_bars": float(np.median(runs)) if runs else None,
            "n_switches": int(np.sum(st3[idx][1:] != st3[idx][:-1])),
            "traded_return_pct": trade(st3, logret, idx, COST)}


def main() -> int:
    argparse.ArgumentParser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_PATH, low_memory=False, usecols=["timestamp", "close"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    ts, close = panel["timestamp"], panel["close"].to_numpy(dtype=np.float64)
    logret = np.diff(np.log(close), prepend=np.log(close[0]))
    logret[0] = 0.0

    piv = ps_pivots(close, 2 * PS_P, PS_P, 4 * PS_P, PS_A)
    ps = ps_label(close, piv, net_gate=True)
    tgt = np.where(ps == BULL, 1, np.where(ps == BEAR, -1, 0)).astype(np.int8)
    runs_ps = [e - s + 1 for s, e, _ in contiguous_runs(np.where(ps == BULL, 2, np.where(ps == BEAR, 0, 1)))]
    med_ps = float(np.median(runs_ps))
    purge = int(max(PURGE, 4 * med_ps))
    run_floor = med_ps / 4.0

    v_idx = np.flatnonzero(((ts >= VAL_START) & (ts <= VAL_END)).to_numpy())
    o_idx = np.flatnonzero(((ts >= OOS_START) & (ts <= OOS_END)).to_numpy())
    windows = {"val_2025Q4": v_idx, "oos_2026Q1": o_idx}
    train_all = np.flatnonzero((ts <= TRAIN_END).to_numpy())
    print(json.dumps({"run_floor_bars": round(run_floor, 1), "purge_bars": purge,
                      "ps_ceiling_traded": {w: trade(ps, logret, i, COST) for w, i in windows.items()}}),
          flush=True)

    all_th = sorted({t for s in FEATSETS.values() for t in s} | set(CZZ_BASELINES))
    geo = {t: zigzag_geometry(close, t) for t in all_th}

    # ---- Stage 0 regression gate
    y_inc = zigzag_oracle(close, threshold=0.005)[0]
    tri = train_all[:-PURGE]
    tri = tri[y_inc[tri] != 0]
    x_inc = np.column_stack([geo[t][0] for t in FEATSETS["S2_fine5"]]).astype(np.float32)
    seeds_inc = sorted(int(s) for s in np.random.default_rng(SEED + 1).choice(1_000_000, size=5, replace=False))
    pr = []
    for s in seeds_inc:
        c = lgb.LGBMClassifier(objective="binary", n_estimators=700, learning_rate=0.05, num_leaves=63,
                               min_child_samples=200, feature_fraction=0.8, bagging_fraction=0.8,
                               bagging_freq=1, reg_lambda=1.0, random_state=s, n_jobs=-1, verbosity=-1)
        c.fit(x_inc[tri], (y_inc[tri] == 1).astype(int))
        pr.append(c.predict_proba(x_inc)[:, 1])
    inc = to_named(jump_decode_proba(np.mean(pr, axis=0), 0.5))
    oracles = {t: zigzag_oracle(close, threshold=t)[0] for t in SCORE_SCALES}
    iv = summarize(inc, oracles, v_idx)["agree"]["0.005"]
    io = summarize(inc, oracles, o_idx)["agree"]["0.005"]
    ok = abs(iv - INCUMBENT_VAL) <= REPRO_TOL and abs(io - INCUMBENT_OOS) <= REPRO_TOL
    print(json.dumps({"STAGE0_incumbent": {"val": iv, "oos": io, "reproduced": ok}}), flush=True)
    out: dict = {"contract": "docs/experiments/btc_regime_ps_label_learnability_g3b_20260808.json",
                 "stage_0_regression_gate": {"val": iv, "oos": io, "reproduced": bool(ok)},
                 "run_floor_bars": round(run_floor, 1)}
    if not ok:
        out["halted"] = "Stage 0 regression gate failed"
        (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2))
        return 1

    # ---- baselines
    base = {}
    for t in CZZ_BASELINES:
        cz = causal_zigzag(close, threshold=t)
        nm = np.where(cz == 1, 2, np.where(cz == -1, 0, 1)).astype(np.int8)
        base[f"czz{t*100:g}"] = {w: summ(nm, tgt, i, logret) for w, i in windows.items()}
    eb = {k: v for k, v in base.items()
          if v["val_2025Q4"]["coverage_pct"] >= MIN_COVERAGE
          and (v["val_2025Q4"]["median_run_bars"] or 0) >= run_floor}
    best_base = max(eb, key=lambda k: eb[k]["val_2025Q4"]["agree"]) if eb else None
    out.update({"baselines": base, "best_eligible_baseline": best_base,
                "ps_ceiling_traded": {w: trade(ps, logret, i, COST) for w, i in windows.items()}})
    print(json.dumps({"best_baseline": best_base,
                      "oos": None if not best_base else base[best_base]["oos_2026Q1"]}), flush=True)

    # ---- train each featset once (state mode), cache probabilities
    tr = train_all[:-purge]
    tr = tr[tgt[tr] != 0]
    y = (tgt[tr] == 1).astype(int)
    print(json.dumps({"train_rows": int(len(tr)), "bull_frac": round(float(y.mean()), 3)}), flush=True)
    seeds = sorted(int(s) for s in np.random.default_rng(SEED + 11).choice(1_000_000, size=N_SEEDS, replace=False))
    probs: dict[str, np.ndarray] = {}
    for fs, th in FEATSETS.items():
        cache = OUT_DIR / f"probs_{fs}.npy"
        if cache.exists():
            probs[fs] = np.load(cache)
            print(json.dumps({"loaded_cached_probs": fs}), flush=True)
            continue
        xm = np.column_stack([geo[t][0] for t in th]).astype(np.float32)
        pp = []
        for s in seeds:
            c = lgb.LGBMClassifier(objective="binary", n_estimators=700, learning_rate=0.05, num_leaves=63,
                                   min_child_samples=200, feature_fraction=0.8, bagging_fraction=0.8,
                                   bagging_freq=1, reg_lambda=1.0, random_state=s, n_jobs=-1, verbosity=-1)
            c.fit(xm[tr], y)
            pp.append(c.predict_proba(xm)[:, 1])
        probs[fs] = np.mean(pp, axis=0)
        np.save(cache, probs[fs])
        print(json.dumps({"trained": fs}), flush=True)

    # ---- JOINT featset x lambda selection on VAL, floor applied HERE
    grid: dict[str, dict] = {}
    for fs in FEATSETS:
        for lam in LAMBDAS:
            nm = to_named(jump_decode_proba(probs[fs], lam))
            s = summ(nm, tgt, v_idx, logret)
            grid[f"{fs}|lam{lam:g}"] = s
            print(f"  {fs:11}|lam{lam:<5g} VAL agree {s['agree']}  cov {s['coverage_pct']}  "
                  f"run {s['median_run_bars']}  sw {s['n_switches']}  traded {s['traded_return_pct']}%",
                  flush=True)
    elig = {k: v for k, v in grid.items()
            if v["coverage_pct"] >= MIN_COVERAGE and (v["median_run_bars"] or 0) >= run_floor
            and v["agree"] is not None}
    win = max(elig, key=lambda k: elig[k]["agree"]) if elig else None
    no_floor_win = max(grid, key=lambda k: (grid[k]["agree"] or -1))
    out.update({"grid_val": grid, "val_winner": win, "n_eligible": len(elig),
                "would_have_won_with_no_run_floor": no_floor_win})
    print(json.dumps({"VAL_WINNER": win, "n_eligible": len(elig),
                      "no_floor_winner": no_floor_win}, indent=2), flush=True)
    if win is None:
        out["halted"] = "no eligible cell even with lambda swept jointly"
        (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
        return 1

    # ---- single OOS read
    fs, lam = win.split("|lam")
    final = to_named(jump_decode_proba(probs[fs], float(lam)))
    rep = {w: summ(final, tgt, i, logret) for w, i in windows.items()}
    o = rep["oos_2026Q1"]
    bb = base[best_base]["oos_2026Q1"] if best_base else None
    gates = {"eligible": bool(o["coverage_pct"] >= MIN_COVERAGE and (o["median_run_bars"] or 0) >= run_floor),
             "beats_no_learning_baseline_agreement": bool(bb and o["agree"] is not None and o["agree"] > bb["agree"]),
             "traded_return_positive": bool((o["traded_return_pct"] or -1) > 0)}
    out.update({"final": {"featset": fs, "lambda": float(lam)}, "measured": rep, "gates": gates,
                "adopt": bool(all(gates.values())),
                "traded_vs_baselines_oos": {k: v["oos_2026Q1"]["traded_return_pct"] for k, v in base.items()},
                "beats_best_baseline_traded": bool(bb and (o["traded_return_pct"] or -1) > bb["traded_return_pct"]),
                "capture_vs_ps_ceiling": (round(o["traded_return_pct"] / out["ps_ceiling_traded"]["oos_2026Q1"], 4)
                                          if out["ps_ceiling_traded"]["oos_2026Q1"] else None)})
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({"FINAL": out["final"], "oos": o, "baseline_oos": bb, "gates": gates,
                      "ADOPT": out["adopt"],
                      "beats_best_baseline_traded": out["beats_best_baseline_traded"]}, indent=2), flush=True)
    pd.DataFrame({"timestamp": ts, "close": close, "ps_label": ps,
                  "p_bull": probs[fs], "detector_state": final}).to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {OUT_PARQUET}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
