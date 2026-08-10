"""BTC Jump-Model-gated regime MoE line (contract docs/experiments/btc_jm_regime_moe_20260808.json).

Reopening argument vs the closed D2-rule regime-conditioned entry line (OOS -19.5%,
docs/btc_regime_conditioned_entry_line_20260808.md): the D2 gate starved the experts
(9.4% trend occupancy, 5-7k rows) and flickered at boundaries; the causal Statistical
Jump Model partition (k3, lambda 32/128) has 3-5x expert data, persistent runs, and
69.5% zigzag-oracle agreement vs the HMM's 51%.  This line tests whether GATE QUALITY
was the binding failure.  Differences from the closed design, all pre-registered here:
  - gate: JM causal states (lam32 primary, lam128 variant) from
    data/research/btc_jm_regime_states_20260808.parquet (train-only fit, causal decode)
  - experts: LGBM hp_conservative only (leaves31/mcs500 -- the closed line's winner family)
  - seed bagging: N=5 truly random seeds per expert, probas averaged; per-seed PnL reported
  - transition purge: optionally drop the first 24 bars after a regime switch from training
  - bear policy grid: expert / long_only (JM-bear contrarian bounce) / short_only
  - chop: always force-cash (dead in every prior audit)
  - controls: unconditional LGBM (from the closed line) AND single LGBM with the JM regime
    appended as a feature (partition-vs-feature control)
VAL selection gates (all required): n_trades>=15, pnl>0, >=3/4 positive months, beats both
controls, >=60% of the config's 18-variant family VAL-positive, >=4/5 seeds VAL-positive.
Exactly one OOS read for the selected config.  Fresh-forward: fresh_forward_bar_by_bar=true,
trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_used=false,
future_rows_used_for_entry=false.
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

STATES_PATH = ROOT / "data/research/btc_jm_regime_states_20260808.parquet"
OUT_DIR = ROOT / "tmp/btc_jm_regime_moe_20260808"
CLOSED_LINE_DIR = ROOT / "tmp/btc_regime_conditioned_20260808"
HP_CONSERVATIVE = dict(num_leaves=31, min_child_samples=500)
N_SEEDS = 5
GATES = {"czz4": "czz4", "czz4_chop3": "czz4_chop3", "jm_lam32": "jm_lam32"}
FEATSETS = ["full", "top20"]
PURGES = {"czz4": [0], "czz4_chop3": [0], "jm_lam32": [0, 24]}
BEAR_POLICIES = ["expert", "long_only", "short_only"]
FAMILY_MIN_POS_FRAC = 0.60
SEED_MIN_POS = 4


def draw_seeds():
    rng = np.random.default_rng(SEED)
    return sorted(int(s) for s in rng.choice(1_000_000, size=N_SEEDS, replace=False))


def lgbm(seed: int):
    return lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=600, learning_rate=0.05,
                              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, reg_lambda=1.0,
                              random_state=seed, n_jobs=-1, verbosity=-1, **HP_CONSERVATIVE)


def purge_transition_rows(regime: np.ndarray, idx: np.ndarray, purge: int) -> np.ndarray:
    if purge <= 0:
        return idx
    switch = np.flatnonzero(np.diff(regime) != 0) + 1
    bad = np.zeros(len(regime), dtype=bool)
    for s in switch:
        bad[s: s + purge] = True
    return idx[~bad[idx]]


def load_jm(ts: pd.Series) -> dict[str, np.ndarray]:
    st = pd.read_parquet(STATES_PATH)
    n = min(len(st), len(ts))
    assert (st["timestamp"].to_numpy()[:n] == ts.to_numpy()[:n]).all(), "JM states misaligned with panel"
    out = {}
    for col in ("jm_lam32", "czz4", "czz4_chop3"):
        r = np.full(len(ts), 1, dtype=np.int8)
        r[:n] = st[col].to_numpy()[:n]
        out[col] = r
    return out


def apply_policies(proba, panel_len, idx, regime, rule_threshold, bear_policy, czz_dir=None):
    side_state = np.zeros(panel_len, dtype=np.int64)
    side_state[idx] = side_state_from_proba(proba[idx], rule_threshold)
    side_state[idx[regime[idx] == 1]] = 0  # chop force-cash
    bear_rows = idx[regime[idx] == 0]
    if bear_policy == "long_only":
        side_state[bear_rows] = np.where(side_state[bear_rows] == 1, 1, 0)
    elif bear_policy == "short_only":
        side_state[bear_rows] = np.where(side_state[bear_rows] == -1, -1, 0)
    if czz_dir is not None:  # consensus filter: entry side must match the czz4 wave direction
        side_state[idx] = np.where(side_state[idx] == czz_dir[idx], side_state[idx], 0)
    return side_state


def train_expert_set(x, action, tr_idx, regime, featset, purge, tops, seeds, tag, model_dir):
    """Returns per-seed OOF-style proba arrays over the full panel (filled where regime matches)."""
    probas = [np.zeros((x.shape[0], 3)) for _ in seeds]
    for r in (0, 2):  # chop experts never used (force-cash) -- skip training them
        rows = tr_idx[regime[tr_idx] == r]
        rows = purge_transition_rows(regime, rows, purge)
        cols = tops[r] if featset == "top20" else np.arange(x.shape[1])
        for si, seed in enumerate(seeds):
            clf = lgbm(seed)
            clf.fit(x[rows][:, cols], action[rows])
            clf.booster_.save_model(str(model_dir / f"{tag}_{REGIME_NAMES[r]}_seed{seed}.txt"))
            sub = np.flatnonzero(regime == r)
            probas[si][sub] = clf.booster_.predict(x[sub][:, cols])
        print(json.dumps({"trained": f"{tag}_{REGIME_NAMES[r]}", "n_rows": int(len(rows))}), flush=True)
    return probas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stageR", "control_feat", "val", "val_consensus", "oos"], required=True)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "models").mkdir(exist_ok=True)
    panel, ts, x, feat_cols, action, tp_moves, sl_moves, _d2, train_mask, val_mask, oos_mask = load_all()
    jm = load_jm(ts)
    tr_idx = np.flatnonzero(train_mask)
    v_idx = np.flatnonzero(val_mask)
    months = ts.dt.to_period("M").astype(str).to_numpy()
    seeds = draw_seeds()

    if args.stage == "stageR":
        out = {"seeds": seeds}
        for gate, col in GATES.items():
            regime = jm[col]
            res = {}
            for r in range(3):
                auc_tr = per_regime_auc(x, action, tr_idx, regime, r)
                auc_v = per_regime_auc(x, action, v_idx, regime, r)
                dev = np.abs(np.nan_to_num(auc_tr, nan=0.5) - 0.5)
                top = np.argsort(-dev)[:TOP_K]
                s_tr = np.sign(auc_tr[top] - 0.5)
                agree = float(np.mean(s_tr == np.sign(np.nan_to_num(auc_v[top], nan=0.5) - 0.5)))
                res[REGIME_NAMES[r]] = {"val_sign_agreement": round(agree, 3),
                                        "occupancy_train": round(float((regime[tr_idx] == r).mean()), 3),
                                        "n_train_rows": int((regime[tr_idx] == r).sum()),
                                        "top3": [feat_cols[i] for i in top[:3]]}
            n_pass = sum(res[n]["val_sign_agreement"] >= 0.60 for n in ("bear", "bull"))
            out[gate] = {"per_regime": res, "gate_pass": bool(n_pass >= 1)}
            print(json.dumps({gate: out[gate]}), flush=True)
        out["any_gate_pass"] = bool(any(out[g]["gate_pass"] for g in GATES))
        (OUT_DIR / "stageR.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({"any_gate_pass": out["any_gate_pass"]}, indent=2))
    elif args.stage == "control_feat":
        # single LGBM on all rows with the JM lam32 regime appended as a feature
        regime = jm["jm_lam32"]
        xr = np.column_stack([x, regime.astype(np.float32)])
        proba_sum = np.zeros((len(panel), 3))
        for seed in seeds:
            clf = lgbm(seed)
            clf.fit(xr[tr_idx], action[tr_idx])
            proba_sum[v_idx] += clf.booster_.predict(xr[v_idx])
            print(json.dumps({"control_feat_seed_done": seed}), flush=True)
        proba = proba_sum / N_SEEDS
        results = []
        for rule in ENTRY_RULES:
            side_state = np.zeros(len(panel), dtype=np.int64)
            side_state[v_idx] = side_state_from_proba(proba[v_idx], rule["threshold"])
            rres = replay(panel, side_state, tp_moves, sl_moves, val_mask)
            results.append({"rule": rule["name"], **{k: rres.get(k) for k in ("n_trades", "pnl_pct", "win_rate", "mdd_pct")}})
            print(json.dumps(results[-1]), flush=True)
        eligible = [r for r in results if (r["n_trades"] or 0) >= 15]
        best = max(eligible, key=lambda r: r["pnl_pct"]) if eligible else None
        out = {"results": results, "best_val_pnl": None if best is None else best["pnl_pct"]}
        (OUT_DIR / "control_feat.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({"control_feat_best_val_pnl": out["best_val_pnl"]}, indent=2))
    elif args.stage == "val":
        stager = json.loads((OUT_DIR / "stageR.json").read_text())
        if not stager.get("any_gate_pass"):
            print(json.dumps({"verdict": "REFUSED -- Stage R gate failed on both JM gates"}))
            return 1
        control_uncond = json.loads((CLOSED_LINE_DIR / "control.json").read_text())["best_val_pnl"]
        control_feat = json.loads((OUT_DIR / "control_feat.json").read_text())["best_val_pnl"]
        control_pnl = max(control_uncond, control_feat if control_feat is not None else -1e9)
        table = []
        for gate, col in GATES.items():
            if not stager[gate]["gate_pass"]:
                continue
            regime = jm[col]
            tops = {}
            for r in (0, 2):
                auc_tr = per_regime_auc(x, action, tr_idx, regime, r)
                dev = np.abs(np.nan_to_num(auc_tr, nan=0.5) - 0.5)
                tops[r] = np.argsort(-dev)[:TOP_K]
            for featset in FEATSETS:
                for purge in PURGES[gate]:
                    tag = f"{gate}_{featset}_p{purge}"
                    probas = train_expert_set(x, action, tr_idx, regime, featset, purge, tops, seeds,
                                              tag, OUT_DIR / "models")
                    proba_bag = sum(probas) / N_SEEDS
                    for bear_policy in BEAR_POLICIES:
                        for rule in ENTRY_RULES:
                            side_state = apply_policies(proba_bag, len(panel), v_idx, regime,
                                                        rule["threshold"], bear_policy)
                            rres = replay(panel, side_state, tp_moves, sl_moves, val_mask)
                            mon = {}
                            for m in sorted(set(months[v_idx])):
                                mon[m] = replay(panel, side_state, tp_moves, sl_moves,
                                                val_mask & (months == m)).get("pnl_pct", 0.0)
                            per_seed = []
                            for si in range(N_SEEDS):
                                ss = apply_policies(probas[si], len(panel), v_idx, regime,
                                                    rule["threshold"], bear_policy)
                                per_seed.append(replay(panel, ss, tp_moves, sl_moves, val_mask).get("pnl_pct", 0.0))
                            rec = {"gate": gate, "featset": featset, "purge": purge, "bear": bear_policy,
                                   "rule": rule["name"], "threshold": rule["threshold"],
                                   **{k: rres.get(k) for k in ("n_trades", "pnl_pct", "win_rate", "mdd_pct")},
                                   "monthly": mon, "n_pos_months": int(sum(v_ > 0 for v_ in mon.values())),
                                   "per_seed_val_pnl": [round(p, 2) for p in per_seed],
                                   "n_seeds_pos": int(sum(p > 0 for p in per_seed))}
                            table.append(rec)
                            print(json.dumps({k: rec[k] for k in ("gate", "featset", "purge", "bear", "rule",
                                                                  "n_trades", "pnl_pct", "n_pos_months", "n_seeds_pos")}), flush=True)
        def family_pos_frac(rec):
            fam = [r for r in table if (r["gate"], r["featset"], r["purge"]) ==
                   (rec["gate"], rec["featset"], rec["purge"])]
            return float(np.mean([(r["pnl_pct"] or 0) > 0 for r in fam]))
        eligible = [r for r in table if (r["n_trades"] or 0) >= 15 and (r["pnl_pct"] or 0) > 0
                    and r["n_pos_months"] >= 3 and (r["pnl_pct"] or 0) > control_pnl
                    and r["n_seeds_pos"] >= SEED_MIN_POS and family_pos_frac(r) >= FAMILY_MIN_POS_FRAC]
        best = max(eligible, key=lambda r: r["pnl_pct"]) if eligible else None
        out = {"seeds": seeds, "control_uncond_val_pnl": control_uncond, "control_feat_val_pnl": control_feat,
               "table": table, "n_eligible": len(eligible),
               "selected": None if best is None else {k: best[k] for k in
                    ("gate", "featset", "purge", "bear", "rule", "threshold", "pnl_pct", "n_trades",
                     "mdd_pct", "n_pos_months", "per_seed_val_pnl", "n_seeds_pos")},
               "selected_family_pos_frac": None if best is None else round(family_pos_frac(best), 2),
               "earns_oos_read": best is not None}
        (OUT_DIR / "val_results.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({"selected": out["selected"], "n_eligible": out["n_eligible"],
                          "earns_oos_read": out["earns_oos_read"]}, indent=2))
    elif args.stage == "val_consensus":
        base = json.loads((OUT_DIR / "val_results.json").read_text())
        if not (OUT_DIR / "val_results_base.json").exists():
            (OUT_DIR / "val_results_base.json").write_text(json.dumps(base, indent=2))
        stager = json.loads((OUT_DIR / "stageR.json").read_text())
        control_pnl = max(base["control_uncond_val_pnl"],
                          base["control_feat_val_pnl"] if base["control_feat_val_pnl"] is not None else -1e9)
        czz_named = jm["czz4"]
        czz_dir = np.where(czz_named == 2, 1, np.where(czz_named == 0, -1, 0)).astype(np.int64)
        table = list(base["table"])
        for rec in table:
            rec.setdefault("consensus", "off")
        for gate, col in GATES.items():
            if not stager[gate]["gate_pass"]:
                continue
            regime = jm[col]
            tops = {}
            for r in (0, 2):
                auc_tr = per_regime_auc(x, action, tr_idx, regime, r)
                dev = np.abs(np.nan_to_num(auc_tr, nan=0.5) - 0.5)
                tops[r] = np.argsort(-dev)[:TOP_K]
            for featset in FEATSETS:
                for purge in PURGES[gate]:
                    tag = f"{gate}_{featset}_p{purge}"
                    probas = [np.zeros((len(panel), 3)) for _ in seeds]
                    for r in (0, 2):
                        cols = tops[r] if featset == "top20" else np.arange(x.shape[1])
                        sub = v_idx[regime[v_idx] == r]
                        if not len(sub):
                            continue
                        for si, seed in enumerate(seeds):
                            booster = lgb.Booster(model_file=str(OUT_DIR / "models" / f"{tag}_{REGIME_NAMES[r]}_seed{seed}.txt"))
                            probas[si][sub] = booster.predict(x[sub][:, cols])
                    proba_bag = sum(probas) / N_SEEDS
                    for bear_policy in BEAR_POLICIES:
                        for rule in ENTRY_RULES:
                            side_state = apply_policies(proba_bag, len(panel), v_idx, regime,
                                                        rule["threshold"], bear_policy, czz_dir)
                            rres = replay(panel, side_state, tp_moves, sl_moves, val_mask)
                            mon = {}
                            for m in sorted(set(months[v_idx])):
                                mon[m] = replay(panel, side_state, tp_moves, sl_moves,
                                                val_mask & (months == m)).get("pnl_pct", 0.0)
                            per_seed = []
                            for si in range(N_SEEDS):
                                ss = apply_policies(probas[si], len(panel), v_idx, regime,
                                                    rule["threshold"], bear_policy, czz_dir)
                                per_seed.append(replay(panel, ss, tp_moves, sl_moves, val_mask).get("pnl_pct", 0.0))
                            rec = {"gate": gate, "featset": featset, "purge": purge, "bear": bear_policy,
                                   "consensus": "on", "rule": rule["name"], "threshold": rule["threshold"],
                                   **{k: rres.get(k) for k in ("n_trades", "pnl_pct", "win_rate", "mdd_pct")},
                                   "monthly": mon, "n_pos_months": int(sum(v_ > 0 for v_ in mon.values())),
                                   "per_seed_val_pnl": [round(p, 2) for p in per_seed],
                                   "n_seeds_pos": int(sum(p > 0 for p in per_seed))}
                            table.append(rec)
                            print(json.dumps({k: rec[k] for k in ("gate", "featset", "purge", "bear", "consensus",
                                                                  "rule", "n_trades", "pnl_pct", "n_pos_months",
                                                                  "n_seeds_pos")}), flush=True)

        def family_pos_frac2(rec):
            fam = [r for r in table if (r["gate"], r["featset"], r["purge"]) ==
                   (rec["gate"], rec["featset"], rec["purge"])]
            return float(np.mean([(r["pnl_pct"] or 0) > 0 for r in fam]))
        eligible = [r for r in table if (r["n_trades"] or 0) >= 15 and (r["pnl_pct"] or 0) > 0
                    and r["n_pos_months"] >= 3 and (r["pnl_pct"] or 0) > control_pnl
                    and r["n_seeds_pos"] >= SEED_MIN_POS and family_pos_frac2(r) >= FAMILY_MIN_POS_FRAC]
        best = max(eligible, key=lambda r: r["pnl_pct"]) if eligible else None
        sel_keys = ("gate", "featset", "purge", "bear", "consensus", "rule", "threshold", "pnl_pct",
                    "n_trades", "mdd_pct", "n_pos_months", "per_seed_val_pnl", "n_seeds_pos")
        out = {"seeds": seeds, "control_uncond_val_pnl": base["control_uncond_val_pnl"],
               "control_feat_val_pnl": base["control_feat_val_pnl"], "table": table,
               "n_eligible": len(eligible),
               "selected": None if best is None else {k: best.get(k) for k in sel_keys},
               "selected_family_pos_frac": None if best is None else round(family_pos_frac2(best), 2),
               "earns_oos_read": best is not None}
        (OUT_DIR / "val_results.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({"selected": out["selected"], "n_eligible": out["n_eligible"],
                          "earns_oos_read": out["earns_oos_read"]}, indent=2))
    else:
        prior = json.loads((OUT_DIR / "val_results.json").read_text())
        if not prior.get("earns_oos_read"):
            print(json.dumps({"oos": "REFUSED -- VAL gates failed; no OOS read earned"}))
            return 1
        sel = prior["selected"]
        regime = jm[GATES[sel["gate"]]]
        tops = {}
        for r in (0, 2):
            auc_tr = per_regime_auc(x, action, tr_idx, regime, r)
            dev = np.abs(np.nan_to_num(auc_tr, nan=0.5) - 0.5)
            tops[r] = np.argsort(-dev)[:TOP_K]
        o_idx = np.flatnonzero(oos_mask)
        tag = f"{sel['gate']}_{sel['featset']}_p{sel['purge']}"
        probas = [np.zeros((len(panel), 3)) for _ in seeds]
        for r in (0, 2):
            cols = tops[r] if sel["featset"] == "top20" else np.arange(x.shape[1])
            sub = o_idx[regime[o_idx] == r]
            for si, seed in enumerate(seeds):
                booster = lgb.Booster(model_file=str(OUT_DIR / "models" / f"{tag}_{REGIME_NAMES[r]}_seed{seed}.txt"))
                if len(sub):
                    probas[si][sub] = booster.predict(x[sub][:, cols])
        proba_bag = sum(probas) / N_SEEDS
        czz_dir = None
        if sel.get("consensus") == "on":
            czz_named = jm["czz4"]
            czz_dir = np.where(czz_named == 2, 1, np.where(czz_named == 0, -1, 0)).astype(np.int64)
        side_state = apply_policies(proba_bag, len(panel), o_idx, regime, sel["threshold"], sel["bear"], czz_dir)
        rres = replay(panel, side_state, tp_moves, sl_moves, oos_mask)
        mon = {}
        for m in sorted(set(months[o_idx])):
            mon[m] = replay(panel, side_state, tp_moves, sl_moves, oos_mask & (months == m)).get("pnl_pct", 0.0)
        per_seed = []
        for si in range(N_SEEDS):
            ss = apply_policies(probas[si], len(panel), o_idx, regime, sel["threshold"], sel["bear"], czz_dir)
            per_seed.append(round(replay(panel, ss, tp_moves, sl_moves, oos_mask).get("pnl_pct", 0.0), 2))
        out = {"stage": "oos", "selected": sel, "seeds": seeds, **rres, "monthly": mon,
               "per_seed_oos_pnl": per_seed, "n_seeds_pos_oos": int(sum(p > 0 for p in per_seed)),
               "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
               "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
               "adopted": bool((rres.get("pnl_pct") or 0) > 0)}
        (OUT_DIR / "oos_results.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
