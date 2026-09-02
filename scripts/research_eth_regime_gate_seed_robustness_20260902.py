#!/usr/bin/env python3
"""Seed robustness for the 2026-09-02 regime-gate PnL finding. User: "잘 나온 것들도 시드 검증해줘".

WHAT ACTUALLY CARRIES A SEED HERE (and what does not)
  * NOT the backtest: simulate_single_position is deterministic, and the cost gate consumes each
    fire's SIDE, not its TabPFN metalabel probability -- so the metalabel seeds never enter.
  * NOT the ensemble/top-k ordering: given fixed per-signal configs those arms are deterministic.
  * ⭐THE REGIME CLASSIFIER DOES. HistGradientBoostingClassifier enables early stopping
    automatically above 10k samples, and its internal validation split is drawn from random_state.
    With 262k TRAIN rows that split -- and therefore the chop predictions, and therefore the whole
    gate -- moves with the seed. That is the exposure this script measures.

GATE (CLAUDE.md Seed-Diversity): N>=5 GENUINELY RANDOM seeds, drawn from a master RNG rather than
as fixed increments off one base, and the seed list is recorded. The policy exists because a
2026-08-01 Sigma3-1h "5-seed ensemble" built as base+5,+10,+15,+20 matched a truly diverse
re-ensemble on VAL (+22.99% vs +23.85%) and then FLIPPED SIGN on OOS (+24.32% -> -13.57%).

CLAIMS UNDER TEST (from docs/experiments/eth_regime_gated_costgate_and_ensemble_pnl_20260902.md)
  C1  orthogonal_combo BENEFITS from chop gating   (genuine 63 -> 87, bp +7.07/+22.44 -> +15.82/+24.55)
  C2  orthogonal_combo's nonchop arm COLLAPSES     (genuine 28) -- the coherence evidence
  C3  taker_delta_z_climax is HURT by chop gating  (genuine 71 -> 15, nonchop 72 is the best arm)
  C4  smt_divergence is hurt too                   (68 -> 32, nonchop 69)
A claim survives only if it holds with the SAME SIGN on every seed.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from core.causal_futures_backtest import purged_decision_mask  # noqa: E402
from research_eth_regime_gated_costgate_ensemble_20260902 import (  # noqa: E402
    HOLDOUT_START, KLINES_PATH, MIN_WINDOW_N, OOS_START, REGIME_ARTIFACT, SIGNALS, VAL_START,
    genuine_from, run_grid,
)
from research_eth_regime_s12k3_label_train_20260902 import (  # noqa: E402
    GBM3_HP, TRAIN_END, TRAIN_START, load_frame, s12k3_label,
)

MASTER_SEED = 20260902
N_SEEDS = 5
OUT_DIR = ROOT / "tmp/eth_regime_gate_seed_robustness_20260902"


def log(m: str) -> None:
    print(f"[seed_robust] {m}", flush=True)


def main() -> int:
    payload = joblib.load(REGIME_ARTIFACT)
    feat_cols, medians = payload["feature_cols"], payload["feature_medians"]

    df = load_frame()
    ts_r = df["timestamp"]
    tr = ((ts_r >= TRAIN_START) & (ts_r <= TRAIN_END)).to_numpy()
    x = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    for c in feat_cols:
        x[c] = x[c].replace([np.inf, -np.inf], np.nan).fillna(medians.get(c, 0.0))
    y, t1, t2 = s12k3_label(df, tr)
    log(f"TRAIN {int(tr.sum()):,} rows | label S12_K3 T1={t1:.6f} T2={t2:.6f}")

    master = np.random.default_rng(MASTER_SEED)
    seeds = master.integers(0, 2**31 - 1, size=N_SEEDS).tolist()   # genuinely random, NOT base+k
    log(f"seeds (randomly drawn per CLAUDE.md gate): {seeds}")

    klines = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    ts = klines["timestamp"]
    o, h, l, c = (klines[k].to_numpy() for k in ("open", "high", "low", "close"))

    fires_cache = {}
    for name, cfg in SIGNALS.items():
        f = pd.read_csv(ROOT / cfg["fires"], parse_dates=["timestamp"])
        fires_cache[name] = f.loc[f["timestamp"] < HOLDOUT_START].sort_values("pos").reset_index(drop=True)

    results = {}
    for si, seed in enumerate(seeds):
        m = HistGradientBoostingClassifier(random_state=int(seed), **GBM3_HP).fit(x[tr], y[tr])
        pred = m.predict(x)
        chop_ts = set(ts_r[pred == 2])
        log(f"\n--- seed {seed} ({si+1}/{N_SEEDS}) chop share {float((pred==2).mean()):.4f} ---")
        per_seed = {}
        for name, cfg in SIGNALS.items():
            fires = fires_cache[name]
            in_chop = fires["timestamp"].isin(chop_ts).to_numpy()
            horizon = cfg["horizon"]
            ev = purged_decision_mask(ts, start=VAL_START, end=OOS_START, horizon_bars=horizon)
            eo = purged_decision_mask(ts, start=OOS_START, end=HOLDOUT_START, horizon_bars=horizon)
            vset, oset = set(np.flatnonzero(ev).tolist()), set(np.flatnonzero(eo).tolist())
            arms = {"chop": in_chop, "nonchop": ~in_chop}
            row = {}
            for aname, keep in arms.items():
                dec = fires.loc[keep, "pos"].to_numpy(np.int64)
                sc = np.where(fires.loc[keep, "side"].to_numpy() == "bottom", 1.0, -1.0)
                atr = fires.loc[keep, "atr_pct"].to_numpy(float)
                vm = np.array([d in vset for d in dec]); om = np.array([d in oset for d in dec])
                if vm.sum() < MIN_WINDOW_N or om.sum() < MIN_WINDOW_N:
                    row[aname] = {"n_genuine": None, "note": "insufficient"}
                    continue
                real = run_grid(ts, o, h, l, c, dec, sc, atr, horizon, vm, om)
                flip = run_grid(ts, o, h, l, c, dec, -sc, atr, horizon, vm, om)
                gen = genuine_from(real, flip)
                best = max(gen, key=lambda g: min(g["val_bp"], g["oos_bp"])) if gen else None
                row[aname] = {"n": int(keep.sum()), "n_genuine": len(gen),
                              "best_val_bp": best["val_bp"] if best else None,
                              "best_oos_bp": best["oos_bp"] if best else None}
            per_seed[name] = row
            ch, nc = row["chop"], row["nonchop"]
            log(f"   {name:24s} chop 진짜={str(ch.get('n_genuine')):>4s} "
                f"(bp {ch.get('best_val_bp')}/{ch.get('best_oos_bp')})  |  "
                f"nonchop 진짜={str(nc.get('n_genuine')):>4s} "
                f"(bp {nc.get('best_val_bp')}/{nc.get('best_oos_bp')})")
        results[str(seed)] = per_seed

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "seed_robustness.json").write_text(json.dumps(
        {"seeds": seeds, "master_seed": MASTER_SEED, "results": results}, indent=2, ensure_ascii=False))

    log("\n=== 시드별 부호 일치 검정 ===")
    for name in SIGNALS:
        chops = [results[str(s)][name]["chop"].get("n_genuine") for s in seeds]
        noncs = [results[str(s)][name]["nonchop"].get("n_genuine") for s in seeds]
        ok = [c is not None and n is not None for c, n in zip(chops, noncs)]
        if not all(ok):
            log(f"  {name:24s} 일부 시드 표본부족 -- chop={chops} nonchop={noncs}"); continue
        helps = [c > n for c, n in zip(chops, noncs)]
        verdict = "✅ chop>nonchop 전 시드 일치" if all(helps) else (
                  "✅ nonchop>chop 전 시드 일치" if not any(helps) else "❌ 시드간 부호 불일치")
        log(f"  {name:24s} chop={chops} nonchop={noncs}  -> {verdict}")
    log(f"\nWrote {OUT_DIR/'seed_robustness.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
