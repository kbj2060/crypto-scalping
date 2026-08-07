#!/usr/bin/env python3
"""The two non-RL allocation approaches from the same menu as
train_eval_sigma6_omega_rl_meta_controller_v2_cvar_20260801.py (user asked to run the remaining
options after the RL meta-controller showed high seed variance and no PnL edge over baseline):

(A) Rule-based conflict-resolution logic -- deterministic per-bar weight rules, selected on
    VAL_2025Q4 ONLY, then confirmed (not re-picked) on OOS_2026H1 to avoid the OOS-cherry-picking
    mistake this project has hit before.
(B) Supervised meta-model -- two HistGradientBoostingClassifier models (P(omega_delta>0),
    P(sigma6_delta>0) at each bar) trained on VAL_2025Q4 features only, applied causally
    (bar-by-bar, no future information) to OOS_2026H1 to derive a per-bar allocation weight.

Both reuse the identical bar-level state/reward data (build_bar_frame) and additive-dollar-PnL
combination convention already validated in the RL script and the original joint-portfolio script
-- only the WEIGHT-DECISION mechanism differs. Same caveats as before: research/diagnostic only,
VAL/OOS both already explored repeatedly in this project's history (not a genuinely blind
Fresh-Forward test per CLAUDE.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from train_eval_sigma6_omega_rl_meta_controller_20260801 import (  # noqa: E402
    build_bar_frame, run_baseline, STATE_COLS,
)

OUT_DIR = ROOT / "tmp/research_20260801/sigma6_omega_rule_and_meta_allocation"


def weighted_pnl(frame: pd.DataFrame, w_omega: np.ndarray, w_sigma6: np.ndarray) -> dict:
    equity, peak, mdd = 1.0, 1.0, 0.0
    om = frame["omega_delta"].to_numpy()
    s6 = frame["sigma6_delta"].to_numpy()
    for i in range(len(frame)):
        equity += w_omega[i] * om[i] + w_sigma6[i] * s6[i]
        peak = max(peak, equity)
        mdd = min(mdd, equity / peak - 1)
    return {"pnl_pct": (equity - 1) * 100, "mdd_pct": mdd * 100}


# ------------------------------------------------------------------ (A) rule-based

def rule_weights(frame: pd.DataFrame, rule: str) -> tuple[np.ndarray, np.ndarray]:
    n = len(frame)
    w_om, w_s6 = np.ones(n), np.ones(n)
    conflict = frame["conflict"].to_numpy() > 0
    if rule == "baseline":
        pass
    elif rule == "half_on_conflict":
        w_om[conflict] = 0.5
        w_s6[conflict] = 0.5
    elif rule == "skip_both_on_conflict":
        w_om[conflict] = 0.0
        w_s6[conflict] = 0.0
    elif rule == "omega_priority":  # trust the live-proven model, shrink the newer candidate
        w_s6[conflict] = 0.0
    elif rule == "sigma6_priority":  # trust the regime-gated candidate over the raw router
        w_om[conflict] = 0.0
    elif rule == "regime_tiebreak":
        bull, bear = frame["bull_prob"].to_numpy(), frame["bear_prob"].to_numpy()
        om_side, s6_side = frame["omega_side"].to_numpy(), frame["sigma6_side"].to_numpy()
        regime_side = np.where(bull >= bear, 1, -1)
        w_om[conflict] = np.where(om_side[conflict] == regime_side[conflict], 1.0, 0.0)
        w_s6[conflict] = np.where(s6_side[conflict] == regime_side[conflict], 1.0, 0.0)
    elif rule == "stability_scaled":  # shrink both proportional to regime instability during conflict
        stab = frame["stability"].to_numpy()
        w_om[conflict] = stab[conflict]
        w_s6[conflict] = stab[conflict]
    else:
        raise ValueError(rule)
    return w_om, w_s6


RULES = ["baseline", "half_on_conflict", "skip_both_on_conflict", "omega_priority",
         "sigma6_priority", "regime_tiebreak", "stability_scaled"]


def run_rules(frame_val: pd.DataFrame, frame_oos: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rule in RULES:
        w_om_v, w_s6_v = rule_weights(frame_val, rule)
        val_res = weighted_pnl(frame_val, w_om_v, w_s6_v)
        rows.append({"rule": rule, "val_pnl": val_res["pnl_pct"], "val_mdd": val_res["mdd_pct"]})
    df = pd.DataFrame(rows)
    print("\n=== (A) Rule-based conflict resolution -- selected on VAL_2025Q4 only ===")
    print(df.to_string(index=False))

    # selection: best VAL pnl among rules that also improve VAL mdd vs baseline (avoid picking a
    # rule that only wins by taking MORE risk than baseline)
    base_mdd = df.loc[df.rule == "baseline", "val_mdd"].iloc[0]
    candidates = df[(df.rule != "baseline") & (df.val_mdd >= base_mdd)]
    picked = candidates.sort_values("val_pnl", ascending=False).iloc[0]["rule"] if len(candidates) else "baseline"
    print(f"\nSelected on VAL (best pnl among rules with mdd >= baseline mdd): '{picked}'")

    oos_rows = []
    for rule in RULES:
        w_om_o, w_s6_o = rule_weights(frame_oos, rule)
        oos_res = weighted_pnl(frame_oos, w_om_o, w_s6_o)
        oos_rows.append({"rule": rule, "oos_pnl": oos_res["pnl_pct"], "oos_mdd": oos_res["mdd_pct"]})
    oos_df = pd.DataFrame(oos_rows)
    full = df.merge(oos_df, on="rule")
    print("\n=== All rules confirmed on OOS_2026H1 (for reference; only 'picked' rule is a fair test) ===")
    print(full.to_string(index=False))
    print(f"\n>>> VAL-selected rule '{picked}' on OOS: "
          f"{full.loc[full.rule == picked, ['oos_pnl','oos_mdd']].to_dict('records')[0]}")
    return full


# ------------------------------------------------------------------ (B) supervised meta-model

def run_meta_model(frame_val: pd.DataFrame, frame_oos: pd.DataFrame) -> dict:
    X_val = frame_val[STATE_COLS].to_numpy()
    y_om = (frame_val["omega_delta"].to_numpy() > 0).astype(int)
    y_s6 = (frame_val["sigma6_delta"].to_numpy() > 0).astype(int)

    clf_om = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200, random_state=0)
    clf_s6 = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200, random_state=0)
    clf_om.fit(X_val, y_om)
    clf_s6.fit(X_val, y_s6)

    def weights_from_model(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = frame[STATE_COLS].to_numpy()
        p_om = clf_om.predict_proba(X)[:, 1]
        p_s6 = clf_s6.predict_proba(X)[:, 1]
        w_om = np.clip(2.0 * p_om, 0.0, 1.5)
        w_s6 = np.clip(2.0 * p_s6, 0.0, 1.5)
        return w_om, w_s6

    w_om_v, w_s6_v = weights_from_model(frame_val)
    w_om_o, w_s6_o = weights_from_model(frame_oos)
    val_res = weighted_pnl(frame_val, w_om_v, w_s6_v)
    oos_res = weighted_pnl(frame_oos, w_om_o, w_s6_o)

    print("\n=== (B) Supervised meta-model (HistGradientBoosting, trained on VAL_2025Q4 only) ===")
    print(f"VAL (in-sample):  pnl={val_res['pnl_pct']:+.2f}% mdd={val_res['mdd_pct']:.2f}%")
    print(f"OOS (frozen model, never trained on): pnl={oos_res['pnl_pct']:+.2f}% mdd={oos_res['mdd_pct']:.2f}%")

    conflict_oos = frame_oos["conflict"].to_numpy() > 0
    if conflict_oos.sum() > 0:
        print(f"During OOS conflict bars (n={int(conflict_oos.sum())}): "
              f"mean omega w={w_om_o[conflict_oos].mean():.3f}, sigma6 w={w_s6_o[conflict_oos].mean():.3f}")
    return {"val": val_res, "oos": oos_res}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame_val = build_bar_frame("VAL_2025Q4")
    frame_oos = build_bar_frame("OOS_2026H1")

    base_val = run_baseline(frame_val)
    base_oos = run_baseline(frame_oos)
    print(f"Baseline fixed 1x-1x combo: VAL pnl={base_val['pnl_pct']:+.2f}% mdd={base_val['mdd_pct']:.2f}% | "
          f"OOS pnl={base_oos['pnl_pct']:+.2f}% mdd={base_oos['mdd_pct']:.2f}%")

    rule_df = run_rules(frame_val, frame_oos)
    rule_df.to_csv(OUT_DIR / "rule_based_results.csv", index=False)

    meta_res = run_meta_model(frame_val, frame_oos)
    pd.DataFrame([{"val_pnl": meta_res["val"]["pnl_pct"], "val_mdd": meta_res["val"]["mdd_pct"],
                   "oos_pnl": meta_res["oos"]["pnl_pct"], "oos_mdd": meta_res["oos"]["mdd_pct"]}]
                 ).to_csv(OUT_DIR / "meta_model_results.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
