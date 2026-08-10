"""Can regime-specific features be CONSTRUCTED rather than discovered? (2026-08-08)

Every previous pass searched the existing panel for features whose bull-vs-bear differential
persists and found none beyond chance.  This flips the question: build features that are
regime-specific BY CONSTRUCTION, then test whether the constructed differential persists.

Technical constraint that shapes the design: within-regime AUC is rank-based, so any FIXED
monotone transform of a feature (plain z-scoring, min-max, log) leaves it exactly unchanged.
Simply "normalising per regime" is therefore a no-op for the diagnostic and for per-regime trees.
Only two kinds of construction escape that:

  BLOCK A  regime-relative, TIME-VARYING normalisation.  z = (x_t - mean_r(<t)) / std_r(<t) where
           the moments are expanding over PAST bars of the SAME regime only.  Because the
           reference moments move, this is not a fixed monotone map and it does change ranks.
           Reads "how extreme is this bar for this regime, given what this regime has looked like
           so far".  Causal by construction.
  BLOCK B  regime-ANCHORED quantities -- genuinely new numbers that do not exist in the panel
           because they are defined relative to the current regime segment: bars since the regime
           began, log return since it began, best/worst excursion inside it, where price sits in
           the regime's own range so far, realised-vol ratio inside vs before, and how unstable
           the regime has been lately (flip count).  All computed forward-only within a segment.

Gate: jm_lam32 (median run 132 bars — long enough for an anchor to mean something, and the
closest of the candidates to the 288-bar label horizon).

TESTS, in order, with a pre-registered stop:
  1 DIFFERENTIAL  the same train-fold + circular-shift permutation protocol used for the panel
                  features. Question: did construction actually create a persistent differential?
                  STOP if the constructed block's qualifier rate is not above its own permutation
                  null — that would mean the construction failed at its stated purpose.
  2 CONTRIBUTION  VAL-only: does adding the constructed block to an unconditional pooled model
                  improve anything (AUC-style separation and replay PnL) versus the same model
                  without it?  Reported for both, no adoption.
No OOS is read in this script.  The entry axis is closed; this is a feature-construction study.
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
from analyze_btc_regime_feature_differential_20260808 import regime_auc  # noqa: E402
from train_eval_btc_persistent_differential_moe_20260808 import (  # noqa: E402
    K_FOLDS, MIN_ABS_DELTA, N_PERM, qualifiers,
)
from train_eval_btc_regime_conditioned_entry_20260808 import load_all  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    ENTRY_RULES, SEED, replay, side_state_from_proba,
)

ZOO_PATH = ROOT / "data/research/btc_regime_classifier_zoo_20260808.parquet"
OUT_DIR = ROOT / "tmp/btc_regime_biased_features_20260808"
OUT_PARQUET = ROOT / "data/research/btc_regime_biased_features_20260808.parquet"
BASE_FOR_RELATIVE = ["vwap_dist_288", "mean_reversion_z", "fibonacci_level", "cvd_288",
                     "whale_retail_ratio", "crowding_pressure", "funding_pressure",
                     "sum_toptrader_long_short_ratio", "turtle_signal", "volume_profile_signal"]
N_SEEDS = 3
HP = dict(num_leaves=63, min_child_samples=200)


def expanding_regime_z(x_col: np.ndarray, regime: np.ndarray) -> np.ndarray:
    """(x_t - mean_r(<t)) / std_r(<t) using only PAST bars of the same regime. Vectorised per
    regime by walking that regime's own subsequence with cumulative sums."""
    out = np.zeros(len(x_col), dtype=np.float32)
    v = np.nan_to_num(x_col, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    for r in np.unique(regime):
        idx = np.flatnonzero(regime == r)
        if len(idx) < 50:
            continue
        s = v[idx]
        n = np.arange(len(s), dtype=np.float64)
        csum = np.concatenate([[0.0], np.cumsum(s)])[:-1]
        csq = np.concatenate([[0.0], np.cumsum(s ** 2)])[:-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(n > 0, csum / np.maximum(n, 1), 0.0)
            var = np.where(n > 1, csq / np.maximum(n, 1) - mean ** 2, 0.0)
            sd = np.sqrt(np.clip(var, 1e-12, None))
            z = np.where(n > 30, (s - mean) / sd, 0.0)
        out[idx] = np.clip(z, -8, 8).astype(np.float32)
    return out


def regime_anchored(close: np.ndarray, regime: np.ndarray) -> dict[str, np.ndarray]:
    n = len(close)
    logc = np.log(np.maximum(close, 1e-12))
    starts = np.flatnonzero(np.diff(regime) != 0) + 1
    seg_start = np.zeros(n, dtype=np.int64)
    b = 0
    for s in starts:
        seg_start[b:s] = b
        b = s
    seg_start[b:] = b
    bars_in = np.arange(n) - seg_start
    ret_since = logc - logc[seg_start]
    run_max = np.empty(n)
    run_min = np.empty(n)
    bounds = np.concatenate([[0], starts, [n]])
    for i in range(len(bounds) - 1):
        a, e = bounds[i], bounds[i + 1]
        if e > a:
            run_max[a:e] = np.maximum.accumulate(logc[a:e])
            run_min[a:e] = np.minimum.accumulate(logc[a:e])
    mfe = run_max - logc[seg_start]
    mae = run_min - logc[seg_start]
    rng = np.maximum(run_max - run_min, 1e-9)
    pos_in_range = (logc - run_min) / rng
    lr = np.diff(logc, prepend=logc[0])
    vol_short = pd.Series(lr).rolling(96, min_periods=24).std().to_numpy()
    vol_long = pd.Series(lr).rolling(576, min_periods=96).std().to_numpy()
    vol_ratio = np.nan_to_num(vol_short / np.where(vol_long > 0, vol_long, np.nan), nan=1.0)
    flips = pd.Series((np.diff(regime, prepend=regime[0]) != 0).astype(float)).rolling(
        576, min_periods=96).sum().to_numpy()
    return {
        "rg_bars_since_start": np.log1p(bars_in).astype(np.float32),
        "rg_ret_since_start": ret_since.astype(np.float32),
        "rg_mfe_since_start": mfe.astype(np.float32),
        "rg_mae_since_start": mae.astype(np.float32),
        "rg_pos_in_range": np.nan_to_num(pos_in_range, nan=0.5).astype(np.float32),
        "rg_giveback": np.nan_to_num((mfe - ret_since) / np.maximum(np.abs(mfe), 1e-6),
                                     nan=0.0, posinf=0.0, neginf=0.0).clip(-5, 5).astype(np.float32),
        "rg_vol_ratio": np.clip(vol_ratio, 0, 8).astype(np.float32),
        "rg_flip_count_576": np.nan_to_num(flips, nan=0.0).astype(np.float32),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["build", "differential", "contribution"], required=True)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel, ts, x, feat_cols, action, tp_moves, sl_moves, _d2, train_mask, val_mask, oos_mask = load_all()
    zoo = pd.read_parquet(ZOO_PATH)
    assert len(zoo) == len(panel)
    regime = zoo["jm"].to_numpy().astype(np.int8)
    close = panel["close"].to_numpy(dtype=np.float64)
    tr_idx, v_idx = np.flatnonzero(train_mask), np.flatnonzero(val_mask)
    folds = np.array_split(tr_idx, K_FOLDS)

    if args.stage == "build":
        cols: dict[str, np.ndarray] = {}
        for name in BASE_FOR_RELATIVE:
            if name not in feat_cols:
                print(json.dumps({"skipped_missing": name}), flush=True)
                continue
            cols[f"rr_{name}"] = expanding_regime_z(x[:, feat_cols.index(name)], regime)
        cols.update(regime_anchored(close, regime))
        df = pd.DataFrame({"timestamp": ts, **cols})
        df.to_parquet(OUT_PARQUET, index=False)
        print(json.dumps({"n_constructed": len(cols), "names": list(cols)}, indent=2), flush=True)
        print(f"wrote {OUT_PARQUET}")
    elif args.stage == "differential":
        built = pd.read_parquet(OUT_PARQUET)
        names = [c for c in built.columns if c != "timestamp"]
        xn = built[names].to_numpy(dtype=np.float32)

        def deltas_for(xmat, reg):
            return np.array([np.nan_to_num(regime_auc(xmat, action, f, reg, 2), nan=0.5)
                             - np.nan_to_num(regime_auc(xmat, action, f, reg, 0), nan=0.5)
                             for f in folds])
        d_real = deltas_for(xn, regime)
        q_real = qualifiers(d_real)
        rng = np.random.default_rng(SEED)
        null = []
        for i in range(N_PERM):
            shift = int(rng.integers(len(regime) // 8, len(regime) - len(regime) // 8))
            null.append(int(len(qualifiers(deltas_for(xn, np.roll(regime, shift))))))
            print(json.dumps({"perm": i, "n_qualifiers": null[-1]}), flush=True)
        rate_real = len(q_real) / len(names)
        rate_null = float(np.mean(null)) / len(names)
        out = {"n_constructed": len(names), "min_abs_delta": MIN_ABS_DELTA, "k_folds": K_FOLDS,
               "n_qualifiers_real": int(len(q_real)), "qualifier_rate_real": round(rate_real, 3),
               "permutation_null": {"counts": null, "mean": round(float(np.mean(null)), 1),
                                    "max": int(max(null)), "rate_mean": round(rate_null, 3)},
               "qualifiers": [{"feature": names[i],
                               "median_delta": round(float(np.median(d_real[:, i])), 4),
                               "per_fold": [round(float(v), 4) for v in d_real[:, i]]} for i in q_real],
               "max_abs_delta": round(float(np.abs(np.median(d_real, axis=0)).max()), 4),
               "construction_succeeded": bool(len(q_real) > max(null))}
        (OUT_DIR / "differential.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(json.dumps({k: out[k] for k in ("n_constructed", "n_qualifiers_real", "qualifier_rate_real",
                                              "permutation_null", "max_abs_delta",
                                              "construction_succeeded")}, indent=2, ensure_ascii=False), flush=True)
        print(json.dumps({"qualifier_names": [q["feature"] for q in out["qualifiers"]]}), flush=True)
    else:
        built = pd.read_parquet(OUT_PARQUET)
        names = [c for c in built.columns if c != "timestamp"]
        xn = built[names].to_numpy(dtype=np.float32)
        seeds = sorted(int(s) for s in np.random.default_rng(SEED + 4).choice(1_000_000, size=N_SEEDS, replace=False))
        res = {}
        for tag, xm in (("panel_only", x), ("panel_plus_constructed", np.column_stack([x, xn]))):
            proba = np.zeros((len(panel), 3))
            for s in seeds:
                clf = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=600,
                                         learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8,
                                         bagging_freq=1, reg_lambda=1.0, random_state=s, n_jobs=-1,
                                         verbosity=-1, **HP)
                clf.fit(xm[tr_idx], action[tr_idx])
                proba[v_idx] += clf.booster_.predict(xm[v_idx])
                if tag == "panel_plus_constructed":
                    gains = dict(zip(list(feat_cols) + names, clf.booster_.feature_importance("gain")))
                    tot = sum(gains.values())
                    res.setdefault("constructed_gain_share", []).append(
                        round(float(sum(gains[n] for n in names) / max(tot, 1e-9)), 4))
            proba /= N_SEEDS
            rows = []
            for rule in ENTRY_RULES:
                ss = np.zeros(len(panel), dtype=np.int64)
                ss[v_idx] = side_state_from_proba(proba[v_idx], rule["threshold"])
                rr = replay(panel, ss, tp_moves, sl_moves, val_mask)
                rows.append({"rule": rule["name"], **{k: rr.get(k) for k in ("n_trades", "pnl_pct", "win_rate")}})
                print(json.dumps({tag: rows[-1]}), flush=True)
            elig = [r for r in rows if (r["n_trades"] or 0) >= 15]
            res[tag] = {"rows": rows, "best_val_pnl": max((r["pnl_pct"] for r in elig), default=None)}
        res["delta_best_val_pnl"] = (None if res["panel_only"]["best_val_pnl"] is None
                                     else round(res["panel_plus_constructed"]["best_val_pnl"]
                                                - res["panel_only"]["best_val_pnl"], 2))
        res["seeds"] = seeds
        (OUT_DIR / "contribution.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(json.dumps({"panel_only_best": res["panel_only"]["best_val_pnl"],
                          "panel_plus_constructed_best": res["panel_plus_constructed"]["best_val_pnl"],
                          "delta": res["delta_best_val_pnl"],
                          "constructed_gain_share": res.get("constructed_gain_share")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
