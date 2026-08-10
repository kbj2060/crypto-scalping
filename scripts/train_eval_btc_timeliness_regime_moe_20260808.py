"""BTC regime-expert MoE gated by the TIMELINESS-FIRST detector (2026-08-08).

Reopening argument, stated honestly.  The regime-conditioned 5m TB entry axis was closed earlier
today: the D2 gate flipped OOS -19.5%, and the JM/czz-gated MoE rerun scored 0/108 eligible at VAL,
which was taken as proof that detector quality was not the binding failure.  What is new here is
not "a better detector" but a detector with a DIFFERENT CAPABILITY: the timeliness-first overlay
nowcasts (agreement peaks at lag 0, detection lag 2 bars, first-quintile agreement 45% vs the
stability model's 30.8%).  Every previous gate was a slow, multi-day partition; this one answers
"has the wave just turned", which is an entry-TIMING question rather than a market-state question.

Mechanical caveat recorded up front, because it shapes the design: the timeliness detector's
median run is 4 bars while the TB labels have a 288-bar horizon, so the regime will flip dozens of
times inside a single trade.  Partitioning experts by entry-bar regime is therefore NOT the
natural use of this gate.  The axis that matches its capability is FRESH-TURN entry timing -- only
act within N bars of a detected flip -- so that is added as an explicit grid axis, and the
plain partition is kept only as the apples-to-apples comparison against the closed line.

Stages (constants and metrics identical to the closed line so numbers are directly comparable):
  --stage stageR   within-regime top-20 sign-stability on the timeliness gate
  --stage val      per-regime LGBM experts (hp_conservative, 5 random seeds, probability bag)
                   x entry rules x bear policy x FRESH-TURN in {off, 12 bars, 48 bars}
  --stage oos      single frozen OOS read of the selected config

VAL selection gates (all required, same as the closed line plus the seed gate):
  n_trades>=15, pnl>0, >=3/4 positive months, beats BOTH controls,
  >=60% of the config's family VAL-positive, >=4/5 seeds individually VAL-positive.
Fresh-forward: fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false.
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
from train_eval_btc_regime_conditioned_entry_20260808 import (  # noqa: E402
    load_all, per_regime_auc, REGIME_NAMES, TOP_K,
)
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    ENTRY_RULES, SEED, replay, side_state_from_proba,
)

STATES_PATH = ROOT / "data/research/btc_regime_theta005_timeliness_20260808.parquet"
OUT_DIR = ROOT / "tmp/btc_timeliness_regime_moe_20260808"
CLOSED_LINE_DIR = ROOT / "tmp/btc_regime_conditioned_20260808"
HP = dict(num_leaves=31, min_child_samples=500)
N_SEEDS = 5
FEATSETS = ["full", "top20"]
BEAR_POLICIES = ["expert", "long_only", "short_only"]
FRESH_TURN = [0, 12, 48]
FAMILY_MIN_POS_FRAC, SEED_MIN_POS = 0.60, 4


def draw_seeds():
    return sorted(int(s) for s in np.random.default_rng(SEED + 2).choice(1_000_000, size=N_SEEDS, replace=False))


def lgbm(seed: int):
    return lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=600, learning_rate=0.05,
                              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, reg_lambda=1.0,
                              random_state=seed, n_jobs=-1, verbosity=-1, **HP)


def load_gate(ts: pd.Series):
    st = pd.read_parquet(STATES_PATH)
    assert len(st) == len(ts) and (pd.to_datetime(st["timestamp"]).to_numpy() == ts.to_numpy()).all(), \
        "timeliness states misaligned with the panel"
    regime = st["timeliness_first"].to_numpy().astype(np.int8)   # 0 bear / 1 chop / 2 bull
    flip = np.zeros(len(regime), dtype=np.int64)
    last = 0
    for i in range(1, len(regime)):
        if regime[i] != regime[i - 1]:
            last = i
        flip[i] = i - last
    return regime, flip


def apply_policies(proba, n, idx, regime, threshold, bear_policy, flip=None, fresh=0):
    side = np.zeros(n, dtype=np.int64)
    side[idx] = side_state_from_proba(proba[idx], threshold)
    side[idx[regime[idx] == 1]] = 0
    bear = idx[regime[idx] == 0]
    if bear_policy == "long_only":
        side[bear] = np.where(side[bear] == 1, 1, 0)
    elif bear_policy == "short_only":
        side[bear] = np.where(side[bear] == -1, -1, 0)
    if fresh > 0:
        side[idx[flip[idx] > fresh]] = 0
    return side


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stageR", "val", "timing", "timing_oos", "oos"], required=True)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "models").mkdir(exist_ok=True)
    panel, ts, x, feat_cols, action, tp_moves, sl_moves, _d2, train_mask, val_mask, oos_mask = load_all()
    regime, flip = load_gate(ts)
    tr_idx = np.flatnonzero(train_mask)
    v_idx = np.flatnonzero(val_mask)
    months = ts.dt.to_period("M").astype(str).to_numpy()
    seeds = draw_seeds()

    if args.stage == "stageR":
        res = {"seeds": seeds, "gate": "timeliness_first"}
        occ = {REGIME_NAMES[r]: round(float((regime[tr_idx] == r).mean()), 3) for r in range(3)}
        for r in range(3):
            auc_tr = per_regime_auc(x, action, tr_idx, regime, r)
            auc_v = per_regime_auc(x, action, v_idx, regime, r)
            dev = np.abs(np.nan_to_num(auc_tr, nan=0.5) - 0.5)
            top = np.argsort(-dev)[:TOP_K]
            s_tr = np.sign(auc_tr[top] - 0.5)
            agreev = float(np.mean(s_tr == np.sign(np.nan_to_num(auc_v[top], nan=0.5) - 0.5)))
            res[REGIME_NAMES[r]] = {"val_sign_agreement": round(agreev, 3), "occupancy_train": occ[REGIME_NAMES[r]],
                                    "n_train_rows": int((regime[tr_idx] == r).sum()),
                                    "top3": [feat_cols[i] for i in top[:3]]}
            print(json.dumps({REGIME_NAMES[r]: res[REGIME_NAMES[r]]}), flush=True)
        n_pass = sum(res[n]["val_sign_agreement"] >= 0.60 for n in ("bear", "bull"))
        res["gate_pass"] = bool(n_pass >= 1)
        res["median_regime_run_bars"] = float(np.median(np.diff(np.flatnonzero(np.diff(regime) != 0))))
        (OUT_DIR / "stageR.json").write_text(json.dumps(res, indent=2))
        print(json.dumps({"gate_pass": res["gate_pass"], "median_regime_run_bars": res["median_regime_run_bars"]},
                         indent=2))
    elif args.stage == "val":
        stager = json.loads((OUT_DIR / "stageR.json").read_text())
        if not stager.get("gate_pass"):
            print(json.dumps({"verdict": "REFUSED -- Stage R gate failed on the timeliness gate"}))
            return 1
        control_uncond = json.loads((CLOSED_LINE_DIR / "control.json").read_text())["best_val_pnl"]
        xr = np.column_stack([x, regime.astype(np.float32), np.log1p(flip).astype(np.float32)])
        proba_sum = np.zeros((len(panel), 3))
        for s in seeds:
            clf = lgbm(s)
            clf.fit(xr[tr_idx], action[tr_idx])
            proba_sum[v_idx] += clf.booster_.predict(xr[v_idx])
        proba_feat = proba_sum / N_SEEDS
        cf = []
        for rule in ENTRY_RULES:
            ss = np.zeros(len(panel), dtype=np.int64)
            ss[v_idx] = side_state_from_proba(proba_feat[v_idx], rule["threshold"])
            r_ = replay(panel, ss, tp_moves, sl_moves, val_mask)
            cf.append(r_.get("pnl_pct") if (r_.get("n_trades") or 0) >= 15 else None)
        control_feat = max([c for c in cf if c is not None], default=-1e9)
        control_pnl = max(control_uncond, control_feat)
        print(json.dumps({"control_uncond": control_uncond, "control_feat": control_feat}), flush=True)

        tops = {}
        for r in (0, 2):
            auc_tr = per_regime_auc(x, action, tr_idx, regime, r)
            dev = np.abs(np.nan_to_num(auc_tr, nan=0.5) - 0.5)
            tops[r] = np.argsort(-dev)[:TOP_K]
        table = []
        for featset in FEATSETS:
            probas = [np.zeros((len(panel), 3)) for _ in seeds]
            for r in (0, 2):
                rows = tr_idx[regime[tr_idx] == r]
                cols = tops[r] if featset == "top20" else np.arange(x.shape[1])
                sub = np.flatnonzero(regime == r)
                for si, s in enumerate(seeds):
                    clf = lgbm(s)
                    clf.fit(x[rows][:, cols], action[rows])
                    clf.booster_.save_model(str(OUT_DIR / "models" / f"{featset}_{REGIME_NAMES[r]}_seed{s}.txt"))
                    probas[si][sub] = clf.booster_.predict(x[sub][:, cols])
                print(json.dumps({"trained": f"{featset}_{REGIME_NAMES[r]}", "n_rows": int(len(rows))}), flush=True)
            bag = sum(probas) / N_SEEDS
            for bear in BEAR_POLICIES:
                for fresh in FRESH_TURN:
                    for rule in ENTRY_RULES:
                        ss = apply_policies(bag, len(panel), v_idx, regime, rule["threshold"], bear, flip, fresh)
                        rr = replay(panel, ss, tp_moves, sl_moves, val_mask)
                        mon = {m: replay(panel, ss, tp_moves, sl_moves, val_mask & (months == m)).get("pnl_pct", 0.0)
                               for m in sorted(set(months[v_idx]))}
                        per_seed = []
                        for si in range(N_SEEDS):
                            s2 = apply_policies(probas[si], len(panel), v_idx, regime, rule["threshold"],
                                                bear, flip, fresh)
                            per_seed.append(replay(panel, s2, tp_moves, sl_moves, val_mask).get("pnl_pct", 0.0))
                        rec = {"featset": featset, "bear": bear, "fresh_turn": fresh, "rule": rule["name"],
                               "threshold": rule["threshold"],
                               **{k: rr.get(k) for k in ("n_trades", "pnl_pct", "win_rate", "mdd_pct")},
                               "monthly": mon, "n_pos_months": int(sum(v > 0 for v in mon.values())),
                               "per_seed_val_pnl": [round(p, 2) for p in per_seed],
                               "n_seeds_pos": int(sum(p > 0 for p in per_seed))}
                        table.append(rec)
                        print(json.dumps({k: rec[k] for k in ("featset", "bear", "fresh_turn", "rule",
                                                              "n_trades", "pnl_pct", "n_pos_months",
                                                              "n_seeds_pos")}), flush=True)

        def fam(rec):
            f = [r for r in table if (r["featset"], r["fresh_turn"]) == (rec["featset"], rec["fresh_turn"])]
            return float(np.mean([(r["pnl_pct"] or 0) > 0 for r in f]))
        eligible = [r for r in table if (r["n_trades"] or 0) >= 15 and (r["pnl_pct"] or 0) > 0
                    and r["n_pos_months"] >= 3 and (r["pnl_pct"] or 0) > control_pnl
                    and r["n_seeds_pos"] >= SEED_MIN_POS and fam(r) >= FAMILY_MIN_POS_FRAC]
        best = max(eligible, key=lambda r: r["pnl_pct"]) if eligible else None
        out = {"seeds": seeds, "control_uncond_val_pnl": control_uncond, "control_feat_val_pnl": control_feat,
               "n_cells": len(table), "n_eligible": len(eligible), "table": table,
               "selected": None if best is None else {k: best[k] for k in
                    ("featset", "bear", "fresh_turn", "rule", "threshold", "pnl_pct", "n_trades",
                     "mdd_pct", "n_pos_months", "per_seed_val_pnl", "n_seeds_pos")},
               "earns_oos_read": best is not None}
        (OUT_DIR / "val_results.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({"n_cells": out["n_cells"], "n_eligible": out["n_eligible"],
                          "selected": out["selected"], "earns_oos_read": out["earns_oos_read"]}, indent=2))
    elif args.stage in ("timing", "timing_oos"):
        # The FRESH-TURN axis is a timing FILTER, not a partition, so the Stage R partition gate
        # does not govern it -- this arm was declared in the module docstring as the axis matching
        # the detector's actual capability, and it runs on ONE unconditional model.
        # Grid: entry rule x fresh-turn window x alignment policy
        #   align="any"   take the model's side regardless of the detector
        #   align="with"  only take entries whose side matches the current detector direction
        #   align="turn"  only take entries whose side matches the direction just turned INTO
        # fresh=0 with align="any" IS the unconditional control, so the control is inside the grid.
        is_oos = args.stage == "timing_oos"
        split_mask, s_idx = (oos_mask, np.flatnonzero(oos_mask)) if is_oos else (val_mask, v_idx)
        model_dir = OUT_DIR / "models"
        probas = [np.zeros((len(panel), 3)) for _ in seeds]
        for si, s in enumerate(seeds):
            mp = model_dir / f"uncond_seed{s}.txt"
            if mp.exists():
                booster = lgb.Booster(model_file=str(mp))
            else:
                clf = lgbm(s)
                clf.fit(x[tr_idx], action[tr_idx])
                clf.booster_.save_model(str(mp))
                booster = clf.booster_
            probas[si][s_idx] = booster.predict(x[s_idx])
            print(json.dumps({"uncond_seed_done": s}), flush=True)
        bag = sum(probas) / N_SEEDS
        reg_dir = np.where(regime == 2, 1, np.where(regime == 0, -1, 0)).astype(np.int64)

        def timing_side(proba, idx, threshold, fresh, align):
            side = np.zeros(len(panel), dtype=np.int64)
            side[idx] = side_state_from_proba(proba[idx], threshold)
            if fresh > 0:
                side[idx[flip[idx] > fresh]] = 0
            if align in ("with", "turn"):
                side[idx] = np.where(side[idx] == reg_dir[idx], side[idx], 0)
            return side

        if is_oos:
            prior = json.loads((OUT_DIR / "timing_val.json").read_text())
            if not prior.get("earns_oos_read"):
                print(json.dumps({"oos": "REFUSED -- timing VAL gates failed"}))
                return 1
            sel = prior["selected"]
            ss = timing_side(bag, s_idx, sel["threshold"], sel["fresh_turn"], sel["align"])
            rr = replay(panel, ss, tp_moves, sl_moves, split_mask)
            mon = {m: replay(panel, ss, tp_moves, sl_moves, split_mask & (months == m)).get("pnl_pct", 0.0)
                   for m in sorted(set(months[s_idx]))}
            per_seed = [round(replay(panel, timing_side(probas[i], s_idx, sel["threshold"], sel["fresh_turn"],
                                                        sel["align"]), tp_moves, sl_moves,
                                     split_mask).get("pnl_pct", 0.0), 2) for i in range(N_SEEDS)]
            out = {"stage": "timing_oos", "selected": sel, "seeds": seeds, **rr, "monthly": mon,
                   "per_seed_oos_pnl": per_seed, "n_seeds_pos_oos": int(sum(p > 0 for p in per_seed)),
                   "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
                   "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
                   "adopted": bool((rr.get("pnl_pct") or 0) > 0 and sum(p > 0 for p in per_seed) >= 4
                                   and sum(v > 0 for v in mon.values()) >= 2)}
            (OUT_DIR / "timing_oos.json").write_text(json.dumps(out, indent=2))
            print(json.dumps(out, indent=2))
            return 0

        table = []
        for align in ("any", "with"):
            for fresh in (0, 6, 12, 24, 48):
                for rule in ENTRY_RULES:
                    ss = timing_side(bag, s_idx, rule["threshold"], fresh, align)
                    rr = replay(panel, ss, tp_moves, sl_moves, split_mask)
                    mon = {m: replay(panel, ss, tp_moves, sl_moves, split_mask & (months == m)).get("pnl_pct", 0.0)
                           for m in sorted(set(months[s_idx]))}
                    per_seed = [replay(panel, timing_side(probas[i], s_idx, rule["threshold"], fresh, align),
                                       tp_moves, sl_moves, split_mask).get("pnl_pct", 0.0) for i in range(N_SEEDS)]
                    rec = {"align": align, "fresh_turn": fresh, "rule": rule["name"],
                           "threshold": rule["threshold"],
                           **{k: rr.get(k) for k in ("n_trades", "pnl_pct", "win_rate", "mdd_pct")},
                           "monthly": mon, "n_pos_months": int(sum(v > 0 for v in mon.values())),
                           "per_seed_val_pnl": [round(p, 2) for p in per_seed],
                           "n_seeds_pos": int(sum(p > 0 for p in per_seed))}
                    table.append(rec)
                    print(json.dumps({k: rec[k] for k in ("align", "fresh_turn", "rule", "n_trades",
                                                          "pnl_pct", "n_pos_months", "n_seeds_pos")}), flush=True)
        ctrl = [r for r in table if r["align"] == "any" and r["fresh_turn"] == 0 and (r["n_trades"] or 0) >= 15]
        control_pnl = max((r["pnl_pct"] for r in ctrl), default=-1e9)

        def fam2(rec):
            f = [r for r in table if (r["align"], r["fresh_turn"]) == (rec["align"], rec["fresh_turn"])]
            return float(np.mean([(r["pnl_pct"] or 0) > 0 for r in f]))
        eligible = [r for r in table if (r["n_trades"] or 0) >= 15 and (r["pnl_pct"] or 0) > 0
                    and r["n_pos_months"] >= 3 and (r["pnl_pct"] or 0) > control_pnl
                    and r["n_seeds_pos"] >= SEED_MIN_POS and fam2(r) >= FAMILY_MIN_POS_FRAC
                    and not (r["align"] == "any" and r["fresh_turn"] == 0)]
        best = max(eligible, key=lambda r: r["pnl_pct"]) if eligible else None
        out = {"seeds": seeds, "arm": "fresh-turn entry TIMING on one unconditional model",
               "control_any_fresh0_val_pnl": control_pnl, "n_cells": len(table),
               "n_eligible": len(eligible), "table": table,
               "selected": None if best is None else {k: best[k] for k in
                    ("align", "fresh_turn", "rule", "threshold", "pnl_pct", "n_trades", "mdd_pct",
                     "n_pos_months", "per_seed_val_pnl", "n_seeds_pos")},
               "earns_oos_read": best is not None}
        (OUT_DIR / "timing_val.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({"control_val_pnl": control_pnl, "n_eligible": out["n_eligible"],
                          "selected": out["selected"], "earns_oos_read": out["earns_oos_read"]}, indent=2))
    else:
        prior = json.loads((OUT_DIR / "val_results.json").read_text())
        if not prior.get("earns_oos_read"):
            print(json.dumps({"oos": "REFUSED -- VAL gates failed; no OOS read earned"}))
            return 1
        sel = prior["selected"]
        tops = {}
        for r in (0, 2):
            auc_tr = per_regime_auc(x, action, tr_idx, regime, r)
            dev = np.abs(np.nan_to_num(auc_tr, nan=0.5) - 0.5)
            tops[r] = np.argsort(-dev)[:TOP_K]
        o_idx = np.flatnonzero(oos_mask)
        probas = [np.zeros((len(panel), 3)) for _ in seeds]
        for r in (0, 2):
            cols = tops[r] if sel["featset"] == "top20" else np.arange(x.shape[1])
            sub = o_idx[regime[o_idx] == r]
            if not len(sub):
                continue
            for si, s in enumerate(seeds):
                b = lgb.Booster(model_file=str(OUT_DIR / "models" / f"{sel['featset']}_{REGIME_NAMES[r]}_seed{s}.txt"))
                probas[si][sub] = b.predict(x[sub][:, cols])
        bag = sum(probas) / N_SEEDS
        ss = apply_policies(bag, len(panel), o_idx, regime, sel["threshold"], sel["bear"], flip, sel["fresh_turn"])
        rr = replay(panel, ss, tp_moves, sl_moves, oos_mask)
        mon = {m: replay(panel, ss, tp_moves, sl_moves, oos_mask & (months == m)).get("pnl_pct", 0.0)
               for m in sorted(set(months[o_idx]))}
        per_seed = [round(replay(panel, apply_policies(probas[si], len(panel), o_idx, regime, sel["threshold"],
                                                       sel["bear"], flip, sel["fresh_turn"]),
                                 tp_moves, sl_moves, oos_mask).get("pnl_pct", 0.0), 2) for si in range(N_SEEDS)]
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
