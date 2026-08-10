"""G3 — can a CAUSAL detector recover the Pagan-Sossounov regime label?  (2026-08-08)

Contract: docs/experiments/btc_regime_ps_label_learnability_g3_20260808.json

Stage 0e produced the first label in this arc that is both DESCRIPTIVE (pure path geometry, not
profit-defined) and LAG-ROBUST (retains 87% of its value at 3-bar lag against the incumbent's 13%).
G3 asks the only question that decides whether that matters.

HYPOTHESIS.  Every detector line closed today trained on a 16-bar-run / 8,173-switch target.  The
PS label is 209.5-bar runs / 939 switches — a far easier object to nowcast.  So detector accuracy
may have been limited by TARGET GRANULARITY rather than by feature information.

The architecture is held FIXED at the frozen detector's own recipe (multi-scale causal zigzag
states -> seed-bagged LGBM probability -> jump-penalized causal DP decode).  Only the target
changes, so any difference is attributable to the label.

HEADLINE GATE, and the thing every earlier detector line never measured: the decoded state is
TRADED DIRECTLY on OOS.  Agreement percentages have been repeatedly shown in this project to move
without moving PnL, so a detector that wins on agreement and loses money does not pass.
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

OUT_DIR = ROOT / "tmp/btc_regime_ps_g3_20260808"
OUT_PARQUET = ROOT / "data/research/btc_regime_ps_g3_states_20260808.parquet"
PS_P, PS_A = 48, 0.02
FEATSETS = {
    "S2_fine5": [0.001, 0.002, 0.0035, 0.005, 0.008],
    "S_coarse5": [0.0035, 0.005, 0.008, 0.012, 0.020],
    "S_wide7": [0.001, 0.002, 0.0035, 0.005, 0.008, 0.012, 0.020],
}
MODES = ["state", "geo"]
CZZ_BASELINES = [0.005, 0.008, 0.012, 0.020, 0.030]
STAGE1_LAM = 2.0
LAMBDAS = [0.5, 1.0, 2.0, 4.0, 8.0]
N_SEEDS = 5
INCUMBENT_VAL, INCUMBENT_OOS, REPRO_TOL = 70.1, 68.0, 0.3


def agree(det: np.ndarray, tgt: np.ndarray, idx: np.ndarray) -> float | None:
    m = idx[(det[idx] != 0) & (tgt[idx] != 0)]
    return None if len(m) < 50 else round(float(np.mean(det[m] == tgt[m])) * 100, 1)


def dir_of(named: np.ndarray) -> np.ndarray:
    return np.where(named == 2, 1, np.where(named == 0, -1, 0)).astype(np.int8)


def summ(named: np.ndarray, tgt: np.ndarray, idx: np.ndarray, logret: np.ndarray) -> dict:
    runs = [e - s + 1 for s, e, _ in contiguous_runs(named[idx])]
    d = dir_of(named)
    st3 = np.where(d == 1, BULL, np.where(d == -1, BEAR, CHOP)).astype(np.int8)
    return {"agree": agree(d, tgt, idx),
            "coverage_pct": round(float((d[idx] != 0).mean()) * 100, 1),
            "median_run_bars": float(np.median(runs)) if runs else None,
            "n_switches": int(np.sum(st3[idx][1:] != st3[idx][:-1])),
            "traded_return_pct": trade(st3, logret, idx, COST)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["screen", "full"], default="full")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_PATH, low_memory=False, usecols=["timestamp", "close"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    ts, close = panel["timestamp"], panel["close"].to_numpy(dtype=np.float64)
    n = len(close)
    logret = np.diff(np.log(close), prepend=np.log(close[0]))
    logret[0] = 0.0

    piv = ps_pivots(close, 2 * PS_P, PS_P, 4 * PS_P, PS_A)
    ps = ps_label(close, piv, net_gate=True)
    tgt = np.where(ps == BULL, 1, np.where(ps == BEAR, -1, 0)).astype(np.int8)
    runs_ps = [e - s + 1 for s, e, _ in contiguous_runs(np.where(ps == BULL, 2, np.where(ps == BEAR, 0, 1)))]
    med_ps = float(np.median(runs_ps))
    purge = int(max(PURGE, 4 * med_ps))
    run_floor = med_ps / 4.0
    print(json.dumps({"ps_turning_points": len(piv), "ps_median_run_bars": med_ps,
                      "purge_bars": purge, "run_floor_bars": round(run_floor, 1),
                      "ps_ceiling_traded_oos_pct": None}), flush=True)

    v_idx = np.flatnonzero(((ts >= VAL_START) & (ts <= VAL_END)).to_numpy())
    o_idx = np.flatnonzero(((ts >= OOS_START) & (ts <= OOS_END)).to_numpy())
    windows = {"val_2025Q4": v_idx, "oos_2026Q1": o_idx}
    train_all = np.flatnonzero((ts <= TRAIN_END).to_numpy())

    all_th = sorted({t for s in FEATSETS.values() for t in s} | set(CZZ_BASELINES))
    geo = {t: zigzag_geometry(close, t) for t in all_th}
    print(json.dumps({"geometry_built": len(geo)}), flush=True)

    out: dict = {"contract": "docs/experiments/btc_regime_ps_label_learnability_g3_20260808.json",
                 "ps_label": {"P": PS_P, "A": PS_A, "median_run_bars": med_ps,
                              "turning_points": len(piv), "purge_bars": purge}}

    # ---- Stage 0 regression gate: reproduce the frozen incumbent on ITS target
    y_inc = zigzag_oracle(close, threshold=0.005)[0]
    tri = train_all[:-PURGE]
    tri = tri[y_inc[tri] != 0]
    x_inc = np.column_stack([geo[t][0] for t in FEATSETS["S2_fine5"]]).astype(np.float32)
    seeds_inc = sorted(int(s) for s in np.random.default_rng(SEED + 1).choice(1_000_000, size=5, replace=False))
    ps_probs = []
    for s in seeds_inc:
        c = lgb.LGBMClassifier(objective="binary", n_estimators=700, learning_rate=0.05, num_leaves=63,
                               min_child_samples=200, feature_fraction=0.8, bagging_fraction=0.8,
                               bagging_freq=1, reg_lambda=1.0, random_state=s, n_jobs=-1, verbosity=-1)
        c.fit(x_inc[tri], (y_inc[tri] == 1).astype(int))
        ps_probs.append(c.predict_proba(x_inc)[:, 1])
    inc_state = to_named(jump_decode_proba(np.mean(ps_probs, axis=0), 0.5))
    oracles = {t: zigzag_oracle(close, threshold=t)[0] for t in SCORE_SCALES}
    iv = summarize(inc_state, oracles, v_idx)["agree"]["0.005"]
    io = summarize(inc_state, oracles, o_idx)["agree"]["0.005"]
    ok = abs(iv - INCUMBENT_VAL) <= REPRO_TOL and abs(io - INCUMBENT_OOS) <= REPRO_TOL
    print(json.dumps({"STAGE0_incumbent": {"val": iv, "oos": io, "reproduced": ok}}), flush=True)
    out["stage_0_regression_gate"] = {"val": iv, "oos": io, "reproduced": bool(ok)}
    if not ok:
        out["halted"] = "Stage 0 regression gate failed"
        (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2))
        return 1

    # ---- ceiling + no-learning baselines, all scored against the PS target
    ceil = {w: trade(ps, logret, idx, COST) for w, idx in windows.items()}
    base = {}
    for t in CZZ_BASELINES:
        cz = causal_zigzag(close, threshold=t)
        nm = np.where(cz == 1, 2, np.where(cz == -1, 0, 1)).astype(np.int8)
        base[f"czz{t*100:g}"] = {w: summ(nm, tgt, idx, logret) for w, idx in windows.items()}
        b = base[f"czz{t*100:g}"]
        print(f"  baseline czz{t*100:g}   VAL agree {b['val_2025Q4']['agree']}  "
              f"OOS agree {b['oos_2026Q1']['agree']}  OOS traded {b['oos_2026Q1']['traded_return_pct']}%  "
              f"run {b['oos_2026Q1']['median_run_bars']}", flush=True)
    elig_base = {k: v for k, v in base.items()
                 if v["val_2025Q4"]["coverage_pct"] >= MIN_COVERAGE
                 and (v["val_2025Q4"]["median_run_bars"] or 0) >= run_floor}
    best_base = max(elig_base, key=lambda k: elig_base[k]["val_2025Q4"]["agree"]) if elig_base else None
    out.update({"ps_ceiling_traded_pct": ceil, "baselines": base,
                "best_eligible_baseline_on_val": best_base})
    print(json.dumps({"PS_CEILING_traded_pct": ceil, "best_baseline": best_base}, indent=2), flush=True)

    # ---- train on the PS target
    tr = train_all[:-purge]
    tr = tr[tgt[tr] != 0]
    y = (tgt[tr] == 1).astype(int)
    print(json.dumps({"train_rows": int(len(tr)), "bull_frac": round(float(y.mean()), 3)}), flush=True)

    def build(fs: str, mode: str) -> np.ndarray:
        th = FEATSETS[fs]
        cols = [geo[t][0] for t in th]
        if mode == "geo":
            cols += [geo[t][1] for t in th] + [geo[t][2] for t in th]
        xm = np.column_stack(cols).astype(np.float32)
        seeds = sorted(int(s) for s in np.random.default_rng(SEED + 11).choice(1_000_000, size=N_SEEDS, replace=False))
        pp = []
        for s in seeds:
            c = lgb.LGBMClassifier(objective="binary", n_estimators=700, learning_rate=0.05, num_leaves=63,
                                   min_child_samples=200, feature_fraction=0.8, bagging_fraction=0.8,
                                   bagging_freq=1, reg_lambda=1.0, random_state=s, n_jobs=-1, verbosity=-1)
            c.fit(xm[tr], y)
            pp.append(c.predict_proba(xm)[:, 1])
        return np.mean(pp, axis=0)

    stage1, probs = {}, {}
    for fs in FEATSETS:
        for md in MODES:
            p = build(fs, md)
            probs[(fs, md)] = p
            s = summ(to_named(jump_decode_proba(p, STAGE1_LAM)), tgt, v_idx, logret)
            stage1[f"{fs}|{md}"] = s
            print(f"  {fs}|{md:5}  VAL agree {s['agree']}  cov {s['coverage_pct']}  "
                  f"run {s['median_run_bars']}  traded {s['traded_return_pct']}%", flush=True)
    elig1 = {k: v for k, v in stage1.items()
             if v["coverage_pct"] >= MIN_COVERAGE and (v["median_run_bars"] or 0) >= run_floor
             and v["agree"] is not None}
    win1 = max(elig1, key=lambda k: elig1[k]["agree"]) if elig1 else None
    out.update({"stage1": stage1, "stage1_winner": win1})
    print(json.dumps({"STAGE1_WINNER": win1, "n_eligible": len(elig1)}, indent=2), flush=True)
    if win1 is None or args.stage == "screen":
        out["halted"] = "no eligible Stage-1 cell" if win1 is None else "screen only"
        (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
        return 0 if win1 else 1

    fs, md = win1.split("|")
    p = probs[(fs, md)]
    stage2 = {f"lam{L:g}": summ(to_named(jump_decode_proba(p, L)), tgt, v_idx, logret) for L in LAMBDAS}
    for k, v in stage2.items():
        print(f"  {k:8} VAL agree {v['agree']}  cov {v['coverage_pct']}  run {v['median_run_bars']}  "
              f"traded {v['traded_return_pct']}%", flush=True)
    elig2 = {k: v for k, v in stage2.items()
             if v["coverage_pct"] >= MIN_COVERAGE and (v["median_run_bars"] or 0) >= run_floor
             and v["agree"] is not None}
    win2 = max(elig2, key=lambda k: elig2[k]["agree"]) if elig2 else None
    out.update({"stage2": stage2, "stage2_winner": win2})
    if win2 is None:
        out["halted"] = "no eligible Stage-2 cell"
        (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
        return 1

    lam = float(win2[3:])
    final = to_named(jump_decode_proba(p, lam))
    rep = {w: summ(final, tgt, idx, logret) for w, idx in windows.items()}
    bb = base[best_base]["oos_2026Q1"]["agree"] if best_base else None
    oo = rep["oos_2026Q1"]
    gates = {
        "eligible": bool(oo["coverage_pct"] >= MIN_COVERAGE and (oo["median_run_bars"] or 0) >= run_floor),
        "beats_no_learning_baseline": bool(bb is not None and oo["agree"] is not None and oo["agree"] > bb),
        "traded_return_positive": bool((oo["traded_return_pct"] or -1) > 0),
    }
    out.update({"final": {"featset": fs, "mode": md, "lambda": lam}, "measured": rep,
                "baseline_oos_agree": bb,
                "capture_vs_ceiling": (round(oo["traded_return_pct"] / ceil["oos_2026Q1"], 4)
                                       if ceil["oos_2026Q1"] else None),
                "gates": gates, "adopt": bool(all(gates.values()))})
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({"FINAL": out["final"], "oos": oo, "baseline_oos_agree": bb,
                      "ceiling_oos_pct": ceil["oos_2026Q1"], "gates": gates,
                      "ADOPT": out["adopt"]}, indent=2), flush=True)
    pd.DataFrame({"timestamp": ts, "close": close, "ps_label": ps,
                  "p_bull": p, "detector_state": final}).to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {OUT_PARQUET}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
