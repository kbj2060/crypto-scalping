"""Axis 1a — descriptive validity of the Pagan-Sossounov regime LABEL  (2026-08-09)

Framework: docs/experiments/btc_regime_classifier_evaluation_framework_20260809.json

Axis 1a is a PRECONDITION, not a formality: it asks whether the label's own states differ in
measurable statistical properties, and whether the difference PERSISTS out of sample.  If they do
not, no classifier can fix it and axes 2-3 would be measuring fidelity to a meaningless target, so
no classifier is trained.

NO RETURNS ARE TESTED.  A bull phase goes up by construction, so any mean-return difference is
circular.  Every property here is direction-agnostic — volatility, autocorrelation, absolute move
size, jump intensity, kurtosis, activity — and asks whether bull and bear differ in CHARACTER.

Three disciplines carried over from the 2026-08-08 arc, each earned by a specific failure:
  PERMUTE THE LABELLING, not the blocks.  A circular shift preserves run structure and occupancy
    exactly while destroying alignment, which is the null that answers "how special is THIS
    labelling" — the question a block bootstrap never asks.
  NEVER POOL WINDOWS.  A pooled variance ratio of 0.881 hid a 67x reversal (VAL 0.097 / OOS 6.49).
    Every effect is reported separately for train, VAL and OOS.
  SIGN MUST PERSIST.  A property whose sign flips between windows is not a regime property, no
    matter how significant it looks in any one of them.

Gate: at least one property must clear the permutation null two-sided at 95% ON OOS, keep its sign
across all three windows, and carry a non-trivial effect size.  The sign requirement across three
windows is itself a strong multiple-comparison control.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from refine_btc_regime_classifier_theta005_20260808 import PANEL_PATH  # noqa: E402
from stage0_btc_regime_label_design_20260808 import BEAR, BULL  # noqa: E402
from stage0e_btc_regime_label_pagan_sossounov_20260808 import ps_label, ps_pivots  # noqa: E402
from test_statistical_jump_model_regimes_20260808 import contiguous_runs  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    TRAIN_END, VAL_START, VAL_END, OOS_START, OOS_END,
)

OUT_DIR = ROOT / "tmp/btc_regime_axis1a_20260809"
PS_P, PS_A = 48, 0.02
N_PERM = 300
VOL_LOOKBACK = 288
EFFECT_MIN = 0.15          # |Cohen's d| or |log variance ratio| below this is called trivial


def properties(close: np.ndarray, vol: np.ndarray, trades: np.ndarray) -> dict[str, np.ndarray]:
    """Per-bar, direction-agnostic quantities. Nothing here is a signed return."""
    r = np.diff(np.log(close), prepend=np.log(close[0]))
    r[0] = 0.0
    s = pd.Series(r)
    sigma = s.rolling(VOL_LOOKBACK, min_periods=48).std().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        jump = (np.abs(r) > 3.0 * sigma).astype(np.float64)
    jump[~np.isfinite(sigma)] = np.nan
    lv = np.log1p(pd.Series(vol).rolling(VOL_LOOKBACK, min_periods=48).mean().to_numpy())
    vz = (np.log1p(vol) - lv) / (pd.Series(np.log1p(vol)).rolling(VOL_LOOKBACK, min_periods=48).std().to_numpy() + 1e-12)
    tz = (np.log1p(trades) - np.log1p(pd.Series(trades).rolling(VOL_LOOKBACK, min_periods=48).mean().to_numpy())) \
        / (pd.Series(np.log1p(trades)).rolling(VOL_LOOKBACK, min_periods=48).std().to_numpy() + 1e-12)
    return {"logret": r, "abs_move": np.abs(r), "jump_3sigma": jump, "volume_z": vz, "trade_z": tz}


def stat_for(prop: str, r: np.ndarray, p: dict[str, np.ndarray], st: np.ndarray,
             idx: np.ndarray) -> float | None:
    """Signed effect: positive means the BULL state has more of the property."""
    b = idx[st[idx] == BULL]
    s = idx[st[idx] == BEAR]
    if len(b) < 200 or len(s) < 200:
        return None
    if prop == "volatility":
        vb, vs = np.nanvar(r[b], ddof=1), np.nanvar(r[s], ddof=1)
        return float(np.log(vb / vs)) if vs > 0 else None
    if prop == "kurtosis":
        return float(stats.kurtosis(r[b], nan_policy="omit") - stats.kurtosis(r[s], nan_policy="omit"))
    if prop.startswith("autocorr_lag"):
        k = int(prop.split("lag")[1])

        def ac(ii: np.ndarray) -> float:
            m = ii[(ii - k) >= 0]
            m = m[st[m - k] == st[m]]
            return float(np.corrcoef(r[m], r[m - k])[0, 1]) if len(m) > 200 else np.nan
        a, c = ac(b), ac(s)
        return None if not (np.isfinite(a) and np.isfinite(c)) else float(a - c)
    x = p[{"abs_move": "abs_move", "jump_intensity": "jump_3sigma",
           "volume_z": "volume_z", "trade_z": "trade_z"}[prop]]
    xb, xs = x[b], x[s]
    xb, xs = xb[np.isfinite(xb)], xs[np.isfinite(xs)]
    if len(xb) < 200 or len(xs) < 200:
        return None
    pooled = np.sqrt((np.var(xb, ddof=1) + np.var(xs, ddof=1)) / 2.0)
    return float((xb.mean() - xs.mean()) / pooled) if pooled > 0 else None


def main() -> int:
    argparse.ArgumentParser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_PATH, low_memory=False,
                        usecols=["timestamp", "close", "volume", "trades"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    ts = panel["timestamp"]
    close = panel["close"].to_numpy(dtype=np.float64)
    st = ps_label(close, ps_pivots(close, 2 * PS_P, PS_P, 4 * PS_P, PS_A), net_gate=True)
    p = properties(close, panel["volume"].to_numpy(float), panel["trades"].to_numpy(float))
    r = p["logret"]

    runs = [e - s + 1 for s, e, _ in contiguous_runs(np.where(st == BULL, 2, np.where(st == BEAR, 0, 1)))]
    windows = {
        "train": np.flatnonzero((ts <= TRAIN_END).to_numpy()),
        "val_2025Q4": np.flatnonzero(((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()),
        "oos_2026Q1": np.flatnonzero(((ts >= OOS_START) & (ts <= OOS_END)).to_numpy()),
    }
    print(json.dumps({"label": f"ps_net|P{PS_P}|A{PS_A*100:g}",
                      "median_run_bars": float(np.median(runs)),
                      "occupancy": {"bull": round(float((st == BULL).mean()), 3),
                                    "bear": round(float((st == BEAR).mean()), 3),
                                    "chop": round(float((st != BULL) & (st != BEAR)).mean() if False else
                                                  float(((st != BULL) & (st != BEAR)).mean()), 3)},
                      "n_perm": N_PERM}, ensure_ascii=False), flush=True)

    props = ["volatility", "abs_move", "jump_intensity", "kurtosis",
             "autocorr_lag1", "autocorr_lag6", "autocorr_lag12", "volume_z", "trade_z"]
    rng = np.random.default_rng(20260809)
    res: dict[str, dict] = {}
    for prop in props:
        res[prop] = {}
        for wname, idx in windows.items():
            obs = stat_for(prop, r, p, st, idx)
            if obs is None:
                res[prop][wname] = {"insufficient": True}
                continue
            draws = []
            for _ in range(N_PERM):
                sh = np.roll(st, int(rng.integers(len(st))))
                v = stat_for(prop, r, p, sh, idx)
                if v is not None and np.isfinite(v):
                    draws.append(v)
            d = np.asarray(draws)
            pct = float((d < obs).mean()) * 100
            res[prop][wname] = {"effect": round(obs, 4),
                                "null_mean": round(float(d.mean()), 4),
                                "null_p2.5": round(float(np.percentile(d, 2.5)), 4),
                                "null_p97.5": round(float(np.percentile(d, 97.5)), 4),
                                "percentile_in_null": round(pct, 1),
                                "clears_null_two_sided_95": bool(pct >= 97.5 or pct <= 2.5),
                                "n_draws": len(d)}
        e = {w: res[prop][w].get("effect") for w in windows}
        ok = all(v is not None for v in e.values())
        sign_ok = bool(ok and len({int(np.sign(v)) for v in e.values()}) == 1 and all(v != 0 for v in e.values()))
        oos = res[prop]["oos_2026Q1"]
        res[prop]["_verdict"] = {
            "sign_persists_all_windows": sign_ok,
            "clears_null_on_oos": bool(oos.get("clears_null_two_sided_95", False)),
            "effect_non_trivial_oos": bool(oos.get("effect") is not None and abs(oos["effect"]) >= EFFECT_MIN),
        }
        res[prop]["_verdict"]["PASS"] = bool(all(res[prop]["_verdict"].values()))
        v = res[prop]["_verdict"]
        print(f"  {prop:16} tr {str(e['train']):>8}  val {str(e['val_2025Q4']):>8}  "
              f"oos {str(e['oos_2026Q1']):>8}  oos_pctile {oos.get('percentile_in_null')}  "
              f"sign{'+' if v['sign_persists_all_windows'] else '-'} "
              f"null{'+' if v['clears_null_on_oos'] else '-'} "
              f"size{'+' if v['effect_non_trivial_oos'] else '-'}  "
              f"{'PASS' if v['PASS'] else 'fail'}", flush=True)

    passing = [k for k in props if res[k]["_verdict"]["PASS"]]
    out = {"framework": "docs/experiments/btc_regime_classifier_evaluation_framework_20260809.json",
           "axis": "1a — label descriptive validity",
           "label": f"ps_net|P{PS_P}|A{PS_A*100:g}",
           "no_returns_tested": "mean return is excluded as circular: a bull phase rises by construction",
           "effect_min": EFFECT_MIN, "n_perm": N_PERM,
           "properties": res, "passing_properties": passing,
           "gate": "at least one property with sign persistent across train/VAL/OOS AND clearing the "
                   "permutation null two-sided at 95% on OOS AND a non-trivial effect size",
           "PASS": bool(passing),
           "meaning": ("the label's states are statistically distinguishable in a way that survives "
                       "out of sample — axis 1b and axes 2-3 may proceed" if passing else
                       "the label's states are NOT distinguishable out of sample; no classifier can "
                       "fix that, so no classifier is trained on this label")}
    (OUT_DIR / "axis1a_ps_label.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({"passing_properties": passing, "PASS": out["PASS"]}, indent=2), flush=True)
    print(f"wrote {OUT_DIR / 'axis1a_ps_label.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
