"""Axis 1a across several regime labels, including the LIVE one  (2026-08-09)

Framework: docs/experiments/btc_regime_classifier_evaluation_framework_20260809.json
Extends scripts/axis1a_btc_regime_ps_label_validity_20260809.py (PS label, 0/9 properties passed).

CANDIDATES
  ps_net|P48|A2          Pagan-Sossounov, chop 1.3% — carried through for comparison
  zigzag_net|th0.5|m0    chop 57%  — the cost-gated zigzag family, real chop
  zigzag_net|th2|m0      chop 39%
  regime3_live_argmax    THE LIVE STATES the trading stack actually reads

HONEST SCOPE NOTE ON regime3.  Its label-construction code (`balancedish_adx16_slope15_bb012`) is
not in the repository — only the fitted artifact and its consumers — and guessing the ADX/Bollinger
windows would mean fabricating a label.  So what is tested here is the argmax of the live
`regime3_current_sensitive_wide24_{bull,bear,chop}_prob` columns, i.e. the DETECTOR'S STATES, not
the ground-truth label.  A failure therefore cannot distinguish "the label is empty" from "the
detector destroys it".  It is still the more directly useful object, because those states are what
the live stack consumes.

SECOND CONTRAST ADDED.  For a chop-heavy label the decisive validity question is not bull-vs-bear
but CHOP vs DIRECTIONAL: chop is supposed to be the quiet, trendless state, so it should differ in
volatility / activity / autocorrelation from the directional states.  If it does not, the third
state is a name without content.  Run only where chop occupancy >= 5%.

Same disciplines as before: no returns tested (circular), permutation of the LABELLING by circular
shift (preserves run structure and occupancy), never pool windows, sign must persist.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from axis1a_btc_regime_ps_label_validity_20260809 import (  # noqa: E402
    EFFECT_MIN, N_PERM, properties, stat_for,
)
from refine_btc_regime_classifier_theta005_20260808 import PANEL_PATH  # noqa: E402
from stage0_btc_regime_label_design_20260808 import BEAR, BULL, CHOP, label_family, zigzag_waves  # noqa: E402
from stage0e_btc_regime_label_pagan_sossounov_20260808 import ps_label, ps_pivots  # noqa: E402
from test_statistical_jump_model_regimes_20260808 import contiguous_runs  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    TRAIN_END, VAL_START, VAL_END, OOS_START, OOS_END,
)

OUT_DIR = ROOT / "tmp/btc_regime_axis1a_20260809"
R3_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
R3_PREFIX = "regime3_current_sensitive_wide24_"
PROPS = ["volatility", "abs_move", "jump_intensity", "kurtosis",
         "autocorr_lag1", "autocorr_lag6", "autocorr_lag12", "volume_z", "trade_z"]
MIN_CHOP_OCC = 0.05


def load_regime3(ts: pd.Series) -> np.ndarray | None:
    frames = []
    for f in sorted(R3_DIR.glob("training_features_202*_regime3_current_sensitive_hmm_wide24.csv")):
        d = pd.read_csv(f, usecols=["timestamp", f"{R3_PREFIX}bull_prob",
                                    f"{R3_PREFIX}bear_prob", f"{R3_PREFIX}chop_prob"])
        frames.append(d)
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    m = pd.DataFrame({"timestamp": ts}).merge(d, on="timestamp", how="left")
    pr = m[[f"{R3_PREFIX}bull_prob", f"{R3_PREFIX}bear_prob", f"{R3_PREFIX}chop_prob"]].to_numpy(float)
    st = np.full(len(ts), CHOP, dtype=np.int8)
    ok = np.isfinite(pr).all(axis=1)
    am = np.argmax(pr[ok], axis=1)
    sub = np.where(am == 0, BULL, np.where(am == 1, BEAR, CHOP)).astype(np.int8)
    st[ok] = sub
    print(json.dumps({"regime3_matched_pct": round(float(ok.mean()) * 100, 1)}), flush=True)
    return st


def contrast_states(st: np.ndarray, which: str) -> np.ndarray:
    """Map to a two-state vector reusing BULL/BEAR slots so stat_for() works unchanged."""
    if which == "bull_vs_bear":
        return st
    out = np.full(len(st), 0, dtype=np.int8)          # 0 = excluded
    out[(st == BULL) | (st == BEAR)] = BULL           # 'directional' takes the BULL slot
    out[st == CHOP] = BEAR                            # 'chop' takes the BEAR slot
    return out


def run_battery(name: str, st: np.ndarray, r: np.ndarray, p: dict, windows: dict,
                rng: np.random.Generator) -> dict:
    res: dict[str, dict] = {}
    for prop in PROPS:
        res[prop] = {}
        for wname, idx in windows.items():
            obs = stat_for(prop, r, p, st, idx)
            if obs is None:
                res[prop][wname] = {"insufficient": True}
                continue
            draws = [v for v in (stat_for(prop, r, p, np.roll(st, int(rng.integers(len(st)))), idx)
                                 for _ in range(N_PERM)) if v is not None and np.isfinite(v)]
            d = np.asarray(draws)
            pct = float((d < obs).mean()) * 100 if len(d) else np.nan
            res[prop][wname] = {"effect": round(obs, 4), "percentile_in_null": round(pct, 1),
                                "clears_null_two_sided_95": bool(pct >= 97.5 or pct <= 2.5),
                                "n_draws": len(d)}
        e = {w: res[prop][w].get("effect") for w in windows}
        ok = all(v is not None for v in e.values())
        sign_ok = bool(ok and len({int(np.sign(v)) for v in e.values()}) == 1 and all(v != 0 for v in e.values()))
        oos = res[prop]["oos_2026Q1"]
        v = {"sign_persists": sign_ok,
             "clears_null_oos": bool(oos.get("clears_null_two_sided_95", False)),
             "effect_non_trivial_oos": bool(oos.get("effect") is not None and abs(oos["effect"]) >= EFFECT_MIN)}
        v["PASS"] = bool(all(v.values()))
        res[prop]["_verdict"] = v
        print(f"    {prop:16} tr {str(e['train']):>9} val {str(e['val_2025Q4']):>9} "
              f"oos {str(e['oos_2026Q1']):>9}  pct {oos.get('percentile_in_null')}  "
              f"{'PASS' if v['PASS'] else 'fail'}", flush=True)
    return res


def main() -> int:
    argparse.ArgumentParser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_PATH, low_memory=False,
                        usecols=["timestamp", "close", "volume", "trades"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    ts = panel["timestamp"]
    close = panel["close"].to_numpy(dtype=np.float64)
    p = properties(close, panel["volume"].to_numpy(float), panel["trades"].to_numpy(float))
    r = p["logret"]
    windows = {
        "train": np.flatnonzero((ts <= TRAIN_END).to_numpy()),
        "val_2025Q4": np.flatnonzero(((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()),
        "oos_2026Q1": np.flatnonzero(((ts >= OOS_START) & (ts <= OOS_END)).to_numpy()),
    }

    labels: dict[str, np.ndarray] = {}
    labels["ps_net|P48|A2"] = ps_label(close, ps_pivots(close, 96, 48, 192, 0.02), net_gate=True)
    zz = label_family(close, {th: zigzag_waves(close, th) for th in (0.005, 0.010, 0.020)})
    labels["zigzag_net|th0.5|m0"] = zz["zigzag_net|th0.5|m0"]
    labels["zigzag_net|th2|m0"] = zz["zigzag_net|th2|m0"]
    r3 = load_regime3(ts)
    if r3 is not None:
        labels["regime3_live_argmax"] = r3

    rng = np.random.default_rng(20260809)
    out: dict = {"framework": "docs/experiments/btc_regime_classifier_evaluation_framework_20260809.json",
                 "axis": "1a — descriptive validity, multiple labels",
                 "regime3_scope_caveat": "argmax of the LIVE detector's probabilities, not the "
                                         "ground-truth label (its construction code is not in the "
                                         "repo); a failure cannot separate 'label empty' from "
                                         "'detector destroys it'",
                 "labels": {}}
    for name, st in labels.items():
        runs = [e - s + 1 for s, e, _ in contiguous_runs(np.where(st == BULL, 2, np.where(st == BEAR, 0, 1)))]
        occ = {"bull": round(float((st == BULL).mean()), 3), "bear": round(float((st == BEAR).mean()), 3),
               "chop": round(float((st == CHOP).mean()), 3)}
        blk: dict = {"occupancy": occ, "median_run_bars": float(np.median(runs)) if runs else None,
                     "contrasts": {}}
        print(f"\n=== {name}  occ {occ}  run {blk['median_run_bars']}", flush=True)
        for which in ("bull_vs_bear", "chop_vs_directional"):
            if which == "chop_vs_directional" and occ["chop"] < MIN_CHOP_OCC:
                blk["contrasts"][which] = {"skipped": f"chop occupancy {occ['chop']} < {MIN_CHOP_OCC}"}
                print(f"  [{which}] skipped — chop {occ['chop']}", flush=True)
                continue
            print(f"  [{which}]", flush=True)
            res = run_battery(name, contrast_states(st, which), r, p, windows, rng)
            passing = [k for k in PROPS if res[k]["_verdict"]["PASS"]]
            blk["contrasts"][which] = {"properties": res, "passing": passing, "PASS": bool(passing)}
            print(f"  -> passing: {passing or 'NONE'}", flush=True)
        blk["ANY_CONTRAST_PASSES"] = any(c.get("PASS") for c in blk["contrasts"].values() if isinstance(c, dict))
        out["labels"][name] = blk

    out["summary"] = {k: {"passes": v["ANY_CONTRAST_PASSES"],
                          "passing_by_contrast": {c: (d.get("passing") if isinstance(d, dict) else None)
                                                  for c, d in v["contrasts"].items()}}
                      for k, v in out["labels"].items()}
    (OUT_DIR / "axis1a_multi_label.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\n" + json.dumps(out["summary"], indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {OUT_DIR / 'axis1a_multi_label.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
