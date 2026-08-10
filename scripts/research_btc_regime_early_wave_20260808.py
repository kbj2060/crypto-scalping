"""Why the frozen classifier under-uses early-wave information, and can it be fixed? (2026-08-08)

Observation that opened this line (lag audit of the frozen zigzag-only model, OOS):
    detector            total   Q1 (first fifth of a wave)   bars where czz05 is wrong
    frozen model        68.0    30.8                          36.0
    czz05 (slow)        60.8    15.2                          --
    czz01 (fastest in)  62.7    55.5                          61.8
The model's own fastest INPUT is nearly twice as good as the model early in a wave.  Total
agreement is dominated by the long stable middles of waves, so the training loss never asks for
early accuracy and the model does not deliver it.

Four studies, all measured on VAL (train <= 2025-08-31 with the standard 576-bar purge):

A  FRONTIER + ROUTING BOUND.  Agreement by wave quintile for causal zigzags across a wide
   threshold ladder.  If fine thresholds dominate Q1 and coarse ones dominate Q5, a
   position-conditioned combination has headroom; the "oracle-routed" detector (pick the best
   threshold per quintile using the TRUE position -- infeasible live) bounds how much.
   Also reports what fraction of Q1 bars fall BEFORE the theta=0.5% confirmation move, i.e. the
   part of the deficit that no theta=0.5%-based rule can ever recover.

B  POSITION-WEIGHTED TRAINING.  Retrain the frozen pipeline with per-sample weights that emphasise
   early-wave bars: uniform (baseline), linear, exponential(tau), early-only.  This tests the
   hypothesis directly -- if the model can be early but simply is not asked to be, reweighting
   should move Q1 without a catastrophic total loss.  (Weights use the retrospective wave position
   of TRAINING rows only, exactly the same epistemic status as the label, behind the same purge.)

C  CAUSAL TURN-SUSPECT ROUTING.  "Early in a wave" is not causally knowable, but "the fast zigzag
   disagrees with the slow one" is.  Measure that state's frequency and the per-detector agreement
   inside it, then build a routed detector: inside turn-suspect follow a fast-leaning decode,
   outside follow the frozen one.

D  EXCHANGE RATE.  Total vs Q1 agreement for every candidate, to see the achievable frontier
   rather than a single point.

PRE-REGISTERED: a candidate earns the ONE OOS read only if, on VAL, Q1 >= 35.8 (the frozen model's
30.8 plus 5pp) AND total >= 69.1 (frozen 70.1 minus at most 1pp) AND the standing eligibility
(coverage >= 50%, median run >= 8 bars).  Nothing else is adopted; this is a research round.
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
from audit_btc_regime_classifier_lag_20260808 import agree, by_quintile, dir_of, wave_position  # noqa: E402
from chart_btc_jm_regime_verification_20260808 import causal_zigzag  # noqa: E402
from ensemble_btc_regime_classifier_theta005_20260808 import logit, sigmoid  # noqa: E402
from refine_btc_regime_classifier_theta005_20260808 import (  # noqa: E402
    PANEL_PATH, PURGE, SCORE_SCALES, MIN_COVERAGE, MIN_MEDIAN_RUN,
    jump_decode_proba, summarize, to_named,
)
from test_statistical_jump_model_regimes_20260808 import zigzag_oracle  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    TRAIN_END, VAL_START, VAL_END, OOS_START, OOS_END,
)

OUT_DIR = ROOT / "tmp/regime_early_wave_20260808"
FROZEN_PATH = ROOT / "data/research/btc_regime_theta005_frozen_config_20260808.json"
THETA = 0.005
LADDER = [0.0005, 0.001, 0.002, 0.0035, 0.005, 0.008, 0.012, 0.020]
WEIGHT_SCHEMES = {"uniform": None, "linear": "linear", "exp_tau015": 0.15, "exp_tau030": 0.30,
                  "early_only_q1q2": "early"}
FROZEN_Q1, FROZEN_TOTAL = 30.8, 70.1
GATE_Q1, GATE_TOTAL = 35.8, 69.1


def main() -> int:
    argparse.ArgumentParser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(FROZEN_PATH.read_text())
    thetas = frozen["pipeline"]["1_features"]["thresholds"]
    seeds = frozen["pipeline"]["2_nowcaster"]["seed_bag"]
    lam = float(frozen["pipeline"]["4_decode"].split("lambda=")[1])

    panel = pd.read_csv(PANEL_PATH, low_memory=False, usecols=["timestamp", "close"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    ts = panel["timestamp"]
    close = panel["close"].to_numpy(dtype=np.float64)
    oracles = {t: zigzag_oracle(close, threshold=t)[0] for t in SCORE_SCALES}
    o_dir, pivots = zigzag_oracle(close, threshold=THETA)
    pos = wave_position(o_dir, pivots, len(close))

    train_mask = (ts <= TRAIN_END).to_numpy()
    tr_all = np.flatnonzero(train_mask)
    tr_idx = tr_all[:-PURGE]
    tr_idx = tr_idx[o_dir[tr_idx] != 0]
    y = (o_dir[tr_idx] == 1).astype(int)
    v_idx = np.flatnonzero(((ts >= VAL_START) & (ts <= VAL_END)).to_numpy())
    o_idx = np.flatnonzero(((ts >= OOS_START) & (ts <= OOS_END)).to_numpy())

    czz = {t: causal_zigzag(close, threshold=t).astype(np.int8) for t in LADDER}
    report: dict = {"frozen_reference": {"q1": FROZEN_Q1, "total": FROZEN_TOTAL},
                    "gate": {"min_q1": GATE_Q1, "min_total": GATE_TOTAL}}

    # ---------------- Study A: frontier + routing bound
    fro = {}
    for t in LADDER:
        d = czz[t]
        fro[f"czz_{t}"] = {"total": agree(d, o_dir, v_idx), **by_quintile(d, o_dir, pos, v_idx)}
    best_per_q = {}
    for q in ("Q1", "Q2", "Q3", "Q4", "Q5"):
        b = max(fro, key=lambda k: (fro[k][q] or 0))
        best_per_q[q] = {"threshold": b, "agreement": fro[b][q]}
    # oracle-routed upper bound: per-bar, take the quintile's best threshold (uses true position)
    routed = np.zeros(len(close), dtype=np.int8)
    for qi, q in enumerate(("Q1", "Q2", "Q3", "Q4", "Q5")):
        t = float(best_per_q[q]["threshold"].split("_")[1])
        m = (pos >= qi / 5) & (pos < (qi + 1) / 5)
        routed[m] = czz[t][m]
    routed_named = np.where(routed == 1, 2, np.where(routed == -1, 0, 1)).astype(np.int8)
    # how much of Q1 is before the theta=0.005 confirmation (unrecoverable by any 0.5% rule)
    d05 = czz[0.005]
    q1_idx = v_idx[(pos[v_idx] >= 0) & (pos[v_idx] < 0.2)]
    pre_conf = float(np.mean(d05[q1_idx] != o_dir[q1_idx])) * 100
    report["A_frontier"] = {"by_threshold": fro, "best_threshold_per_quintile": best_per_q,
                            "oracle_routed_bound": {"total": agree(dir_of(routed_named), o_dir, v_idx),
                                                    **by_quintile(dir_of(routed_named), o_dir, pos, v_idx)},
                            "q1_bars_where_czz05_still_wrong_pct": round(pre_conf, 1)}
    print(json.dumps(report["A_frontier"]["best_threshold_per_quintile"], indent=2), flush=True)
    print(json.dumps({"oracle_routed_bound": report["A_frontier"]["oracle_routed_bound"],
                      "q1_unrecoverable_by_czz05_pct": report["A_frontier"]["q1_bars_where_czz05_still_wrong_pct"]},
                     indent=2), flush=True)

    # ---------------- Study B: position-weighted training
    xm = np.column_stack([czz[t] for t in thetas]).astype(np.float32)
    tr_pos = np.nan_to_num(pos[tr_idx], nan=0.5)

    def weights(scheme):
        if scheme is None:
            return None
        if scheme == "linear":
            return (1.0 - tr_pos) * 2.0 + 0.1
        if scheme == "early":
            return np.where(tr_pos < 0.4, 3.0, 0.3)
        return np.exp(-tr_pos / float(scheme))

    b_res, b_states = {}, {}
    for name, sch in WEIGHT_SCHEMES.items():
        w = weights(sch)
        ps = []
        for s in seeds:
            clf = lgb.LGBMClassifier(objective="binary", n_estimators=700, learning_rate=0.05,
                                     num_leaves=63, min_child_samples=200, feature_fraction=0.8,
                                     bagging_fraction=0.8, bagging_freq=1, reg_lambda=1.0,
                                     random_state=s, n_jobs=-1, verbosity=-1)
            clf.fit(xm[tr_idx], y, sample_weight=w)
            ps.append(clf.predict_proba(xm)[:, 1])
        p = np.mean(ps, axis=0)
        st = to_named(jump_decode_proba(p, lam))
        d = dir_of(st)
        s_ = summarize(st, oracles, v_idx)
        b_res[name] = {"total": s_["agree"]["0.005"], "coverage_pct": s_["coverage_pct"],
                       "median_run_bars": s_["median_run_bars"], **by_quintile(d, o_dir, pos, v_idx)}
        b_states[f"weight_{name}"] = (st, p)
        print(json.dumps({f"B_{name}": b_res[name]}), flush=True)
    report["B_position_weighted"] = b_res

    # ---------------- Study C: causal turn-suspect routing
    p_base = b_states["weight_uniform"][1]
    fast = czz[0.001]
    slow = czz[0.005]
    suspect = (fast != slow) & (fast != 0) & (slow != 0)
    sus_v = v_idx[suspect[v_idx]]
    report["C_turn_suspect"] = {
        "state_frequency_pct": round(float(suspect[v_idx].mean()) * 100, 1),
        "inside_suspect": {"czz01": agree(fast, o_dir, sus_v), "czz05": agree(slow, o_dir, sus_v),
                           "frozen_model": agree(dir_of(b_states["weight_uniform"][0]), o_dir, sus_v)},
        "share_of_q1_bars_in_suspect_pct": round(float(suspect[q1_idx].mean()) * 100, 1)}
    print(json.dumps({"C_turn_suspect": report["C_turn_suspect"]}, indent=2), flush=True)

    c_res = {}
    for boost in (0.5, 1.0, 2.0):
        z = logit(p_base) + boost * np.where(suspect, fast.astype(float), 0.0)
        st = to_named(jump_decode_proba(sigmoid(z), lam))
        d = dir_of(st)
        s_ = summarize(st, oracles, v_idx)
        c_res[f"route_boost{boost:g}"] = {"total": s_["agree"]["0.005"], "coverage_pct": s_["coverage_pct"],
                                          "median_run_bars": s_["median_run_bars"],
                                          **by_quintile(d, o_dir, pos, v_idx)}
        b_states[f"route_boost{boost:g}"] = (st, sigmoid(z))
        print(json.dumps({f"C_{boost}": c_res[f'route_boost{boost:g}']}), flush=True)
    # fast zigzag as an extra INPUT feature at both scales, with position weighting off
    report["C_routing"] = c_res

    # ---------------- Study D: exchange rate + gate
    table = {}
    for k, v in b_res.items():
        table[f"B_{k}"] = v
    for k, v in c_res.items():
        table[f"C_{k}"] = v
    passing = {k: v for k, v in table.items()
               if (v["Q1"] or 0) >= GATE_Q1 and (v["total"] or 0) >= GATE_TOTAL
               and v["coverage_pct"] >= MIN_COVERAGE and v["median_run_bars"] >= MIN_MEDIAN_RUN}
    report["D_exchange_rate"] = {k: {"total": v["total"], "Q1": v["Q1"],
                                     "median_run_bars": v["median_run_bars"]} for k, v in table.items()}
    report["passing_gate"] = list(passing)
    best = max(passing, key=lambda k: passing[k]["Q1"]) if passing else None
    report["earns_oos_read"] = best is not None
    report["selected"] = best
    if best is not None:
        st = b_states[best.split("_", 1)[1] if best.startswith("B_") else best[2:]][0]
        d = dir_of(st)
        s_ = summarize(st, oracles, o_idx)
        report["oos_single_read"] = {"total": s_["agree"]["0.005"], "coverage_pct": s_["coverage_pct"],
                                     "median_run_bars": s_["median_run_bars"],
                                     **by_quintile(d, o_dir, pos, o_idx)}
        print(json.dumps({"OOS": report["oos_single_read"]}, indent=2), flush=True)

    (OUT_DIR / "early_wave_research.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("=== D exchange rate (VAL)", flush=True)
    for k, v in report["D_exchange_rate"].items():
        print(f"  {k:24} total {v['total']:6}  Q1 {v['Q1']:6}  run {v['median_run_bars']}", flush=True)
    print(json.dumps({"passing_gate": report["passing_gate"], "selected": best}, indent=2), flush=True)
    print(f"wrote {OUT_DIR / 'early_wave_research.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
