"""Regime experts built ONLY from features whose bull-vs-bear differential is stable (2026-08-08).

Idea under test: the differential analysis showed ΔAUC = AUC_bull − AUC_bear collapses to random
on OOS *in aggregate*.  But an aggregate collapse is consistent with a small subset of features
having a genuinely stable differential, drowned out by ~120 noisy ones.  If such a subset exists,
experts built on it alone are the strongest possible version of the regime-conditioned idea.

METHODOLOGICAL TRAP, avoided explicitly: selecting features because their differential held on OOS
and then scoring on OOS is circular.  Feature selection here uses TRAIN ONLY -- the train window is
cut into K=4 contiguous folds and a feature qualifies when the SIGN of its ΔAUC is identical in all
four.  VAL is untouched during selection and OOS is read exactly once at the end.

NULL COMPARISON, mandatory before proceeding: with no true differential, a feature's per-fold sign
is close to a coin flip, so ~1/8 of features would qualify by chance -- ~16 of 130.  Rather than
lean on that binomial (folds are not independent; features are autocorrelated), the null is
measured by PERMUTATION: circularly shift the regime vector by a large random offset, which
preserves both the feature autocorrelation and the regime run-length structure while destroying
the regime-feature alignment, then recount qualifiers.  R=10 shifts.

PRE-REGISTERED STOPPING RULE: if the real qualifier count is not above the permutation null's
maximum, the subset is indistinguishable from chance and the line STOPS AT STAGE F -- no experts
are trained, no VAL grid is run, no OOS read is taken.  Reporting "we found N stable features"
without this comparison would be the multiple-comparisons error this project has already been
burned by (the 60-symbol screen's #1 rank).

Gate: jm_lam32 (median run 132 bars, the best within-regime sign stability measured, and the run
length closest to the 288-bar label horizon among the candidates).

If Stage F passes:
  --stage val  per-regime LGBM experts on the qualifying features only, 5 random seeds, standard
               grid (entry rules x bear policy) and the project's standard gates
               (n_trades>=15, pnl>0, >=3/4 positive months, beats the unconditional control,
               >=4/5 seeds VAL-positive, >=60% family VAL-positive)
  --stage oos  single frozen read
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
from analyze_btc_regime_feature_differential_20260808 import is_carrier, regime_auc  # noqa: E402
from train_eval_btc_regime_conditioned_entry_20260808 import load_all, REGIME_NAMES  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    ENTRY_RULES, SEED, replay, side_state_from_proba,
)

ZOO_PATH = ROOT / "data/research/btc_regime_classifier_zoo_20260808.parquet"
OUT_DIR = ROOT / "tmp/btc_persistent_differential_moe_20260808"
CLOSED_LINE_DIR = ROOT / "tmp/btc_regime_conditioned_20260808"
K_FOLDS = 4
N_PERM = 10
MIN_ABS_DELTA = 0.01
HP = dict(num_leaves=31, min_child_samples=500)
N_SEEDS = 5
BEAR_POLICIES = ["expert", "long_only", "short_only"]
FAMILY_MIN_POS_FRAC, SEED_MIN_POS = 0.60, 4


def draw_seeds():
    return sorted(int(s) for s in np.random.default_rng(SEED + 3).choice(1_000_000, size=N_SEEDS, replace=False))


def lgbm(seed: int):
    return lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=600, learning_rate=0.05,
                              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, reg_lambda=1.0,
                              random_state=seed, n_jobs=-1, verbosity=-1, **HP)


def fold_deltas(x, action, folds, regime):
    """ΔAUC per feature per train fold. Uses train rows only."""
    out = []
    for f in folds:
        a_bull = regime_auc(x, action, f, regime, 2)
        a_bear = regime_auc(x, action, f, regime, 0)
        out.append(np.nan_to_num(a_bull, nan=0.5) - np.nan_to_num(a_bear, nan=0.5))
    return np.array(out)


def qualifiers(deltas, min_abs=MIN_ABS_DELTA):
    signs = np.sign(deltas)
    same = (np.abs(signs.sum(axis=0)) == deltas.shape[0]) & (signs != 0).all(axis=0)
    strong = np.abs(np.median(deltas, axis=0)) >= min_abs
    return np.flatnonzero(same & strong)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stageF", "val", "oos"], required=True)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "models").mkdir(exist_ok=True)
    panel, ts, x, feat_cols, action, tp_moves, sl_moves, _d2, train_mask, val_mask, oos_mask = load_all()
    zoo = pd.read_parquet(ZOO_PATH)
    assert len(zoo) == len(panel)
    regime = zoo["jm"].to_numpy().astype(np.int8)
    tr_idx = np.flatnonzero(train_mask)
    v_idx = np.flatnonzero(val_mask)
    months = ts.dt.to_period("M").astype(str).to_numpy()
    seeds = draw_seeds()
    folds = np.array_split(tr_idx, K_FOLDS)

    if args.stage == "stageF":
        deltas = fold_deltas(x, action, folds, regime)
        qual = qualifiers(deltas)
        rng = np.random.default_rng(SEED)
        null_counts = []
        for i in range(N_PERM):
            shift = int(rng.integers(len(regime) // 8, len(regime) - len(regime) // 8))
            perm_regime = np.roll(regime, shift)
            null_counts.append(int(len(qualifiers(fold_deltas(x, action, folds, perm_regime)))))
            print(json.dumps({"perm": i, "shift": shift, "n_qualifiers": null_counts[-1]}), flush=True)
        out = {"gate": "jm_lam32", "k_folds": K_FOLDS, "min_abs_delta": MIN_ABS_DELTA,
               "n_features": len(feat_cols), "n_qualifiers_real": int(len(qual)),
               "permutation_null": {"counts": null_counts, "mean": round(float(np.mean(null_counts)), 1),
                                    "max": int(max(null_counts))},
               "binomial_expectation_if_independent": round(len(feat_cols) * 2 * 0.5 ** K_FOLDS, 1),
               "qualifiers": [{"feature": feat_cols[i],
                               "median_delta": round(float(np.median(deltas[:, i])), 4),
                               "per_fold": [round(float(v), 4) for v in deltas[:, i]],
                               "carrier": is_carrier(feat_cols[i])} for i in qual],
               "proceed": bool(len(qual) > max(null_counts))}
        (OUT_DIR / "stageF.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(json.dumps({"n_qualifiers_real": out["n_qualifiers_real"],
                          "null_mean": out["permutation_null"]["mean"],
                          "null_max": out["permutation_null"]["max"],
                          "binomial_expectation": out["binomial_expectation_if_independent"],
                          "PROCEED": out["proceed"],
                          "qualifier_names": [q["feature"] for q in out["qualifiers"]]}, indent=2,
                         ensure_ascii=False))
    elif args.stage == "val":
        sf = json.loads((OUT_DIR / "stageF.json").read_text())
        if not sf["proceed"]:
            print(json.dumps({"verdict": "REFUSED -- Stage F qualifier count is within the "
                                         "permutation null; the 'stable differential' subset is "
                                         "indistinguishable from chance"}, ensure_ascii=False))
            return 1
        cols = np.array([feat_cols.index(q["feature"]) for q in sf["qualifiers"]])
        control = json.loads((CLOSED_LINE_DIR / "control.json").read_text())["best_val_pnl"]
        probas = [np.zeros((len(panel), 3)) for _ in seeds]
        for r in (0, 2):
            rows = tr_idx[regime[tr_idx] == r]
            sub = np.flatnonzero(regime == r)
            for si, s in enumerate(seeds):
                clf = lgbm(s)
                clf.fit(x[rows][:, cols], action[rows])
                clf.booster_.save_model(str(OUT_DIR / "models" / f"{REGIME_NAMES[r]}_seed{s}.txt"))
                probas[si][sub] = clf.booster_.predict(x[sub][:, cols])
            print(json.dumps({"trained": REGIME_NAMES[r], "n_rows": int(len(rows)),
                              "n_features": int(len(cols))}), flush=True)
        bag = sum(probas) / N_SEEDS
        table = []
        for bear in BEAR_POLICIES:
            for rule in ENTRY_RULES:
                def side_of(proba):
                    ss = np.zeros(len(panel), dtype=np.int64)
                    ss[v_idx] = side_state_from_proba(proba[v_idx], rule["threshold"])
                    ss[v_idx[regime[v_idx] == 1]] = 0
                    b = v_idx[regime[v_idx] == 0]
                    if bear == "long_only":
                        ss[b] = np.where(ss[b] == 1, 1, 0)
                    elif bear == "short_only":
                        ss[b] = np.where(ss[b] == -1, -1, 0)
                    return ss
                ss = side_of(bag)
                rr = replay(panel, ss, tp_moves, sl_moves, val_mask)
                mon = {m: replay(panel, ss, tp_moves, sl_moves, val_mask & (months == m)).get("pnl_pct", 0.0)
                       for m in sorted(set(months[v_idx]))}
                per_seed = [replay(panel, side_of(probas[i]), tp_moves, sl_moves, val_mask).get("pnl_pct", 0.0)
                            for i in range(N_SEEDS)]
                rec = {"bear": bear, "rule": rule["name"], "threshold": rule["threshold"],
                       **{k: rr.get(k) for k in ("n_trades", "pnl_pct", "win_rate", "mdd_pct")},
                       "monthly": mon, "n_pos_months": int(sum(v > 0 for v in mon.values())),
                       "per_seed_val_pnl": [round(p, 2) for p in per_seed],
                       "n_seeds_pos": int(sum(p > 0 for p in per_seed))}
                table.append(rec)
                print(json.dumps({k: rec[k] for k in ("bear", "rule", "n_trades", "pnl_pct",
                                                      "n_pos_months", "n_seeds_pos")}), flush=True)
        fam = float(np.mean([(r["pnl_pct"] or 0) > 0 for r in table]))
        eligible = [r for r in table if (r["n_trades"] or 0) >= 15 and (r["pnl_pct"] or 0) > 0
                    and r["n_pos_months"] >= 3 and (r["pnl_pct"] or 0) > control
                    and r["n_seeds_pos"] >= SEED_MIN_POS and fam >= FAMILY_MIN_POS_FRAC]
        best = max(eligible, key=lambda r: r["pnl_pct"]) if eligible else None
        out = {"seeds": seeds, "n_features_used": int(len(cols)),
               "features": [q["feature"] for q in sf["qualifiers"]],
               "control_val_pnl": control, "family_pos_frac": round(fam, 2),
               "table": table, "n_eligible": len(eligible),
               "selected": None if best is None else {k: best[k] for k in
                    ("bear", "rule", "threshold", "pnl_pct", "n_trades", "mdd_pct", "n_pos_months",
                     "per_seed_val_pnl", "n_seeds_pos")},
               "earns_oos_read": best is not None}
        (OUT_DIR / "val_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(json.dumps({"n_eligible": out["n_eligible"], "selected": out["selected"],
                          "earns_oos_read": out["earns_oos_read"]}, indent=2))
    else:
        prior = json.loads((OUT_DIR / "val_results.json").read_text())
        if not prior.get("earns_oos_read"):
            print(json.dumps({"oos": "REFUSED -- VAL gates failed"}))
            return 1
        sf = json.loads((OUT_DIR / "stageF.json").read_text())
        cols = np.array([feat_cols.index(q["feature"]) for q in sf["qualifiers"]])
        sel = prior["selected"]
        o_idx = np.flatnonzero(oos_mask)
        probas = [np.zeros((len(panel), 3)) for _ in seeds]
        for r in (0, 2):
            sub = o_idx[regime[o_idx] == r]
            if not len(sub):
                continue
            for si, s in enumerate(seeds):
                b = lgb.Booster(model_file=str(OUT_DIR / "models" / f"{REGIME_NAMES[r]}_seed{s}.txt"))
                probas[si][sub] = b.predict(x[sub][:, cols])
        bag = sum(probas) / N_SEEDS

        def side_of(proba):
            ss = np.zeros(len(panel), dtype=np.int64)
            ss[o_idx] = side_state_from_proba(proba[o_idx], sel["threshold"])
            ss[o_idx[regime[o_idx] == 1]] = 0
            b = o_idx[regime[o_idx] == 0]
            if sel["bear"] == "long_only":
                ss[b] = np.where(ss[b] == 1, 1, 0)
            elif sel["bear"] == "short_only":
                ss[b] = np.where(ss[b] == -1, -1, 0)
            return ss
        ss = side_of(bag)
        rr = replay(panel, ss, tp_moves, sl_moves, oos_mask)
        mon = {m: replay(panel, ss, tp_moves, sl_moves, oos_mask & (months == m)).get("pnl_pct", 0.0)
               for m in sorted(set(months[o_idx]))}
        per_seed = [round(replay(panel, side_of(probas[i]), tp_moves, sl_moves, oos_mask).get("pnl_pct", 0.0), 2)
                    for i in range(N_SEEDS)]
        out = {"stage": "oos", "selected": sel, "seeds": seeds, **rr, "monthly": mon,
               "per_seed_oos_pnl": per_seed, "n_seeds_pos_oos": int(sum(p > 0 for p in per_seed)),
               "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
               "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
               "adopted": bool((rr.get("pnl_pct") or 0) > 0 and sum(p > 0 for p in per_seed) >= 4
                               and sum(v > 0 for v in mon.values()) >= 2)}
        (OUT_DIR / "oos_results.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
