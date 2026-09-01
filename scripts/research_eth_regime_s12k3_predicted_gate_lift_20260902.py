#!/usr/bin/env python3
"""Phase 3b: which label, ONCE MODELLED, gives the better evidence-signal gate? -- 2026-09-02.

Phase 2 measured conditioning value on the GROUND-TRUTH label (S12_K3 clearly better: 10/16 vs 3/14
both-window-positive cells). Phase 3 measured learnability and found the opposite ordering
(S12_K3 bal_acc 0.8550 / chop_P 0.8670 vs deployed 0.9108 / 0.9202). Neither number alone decides
anything, because what actually ships is the PRODUCT: the live gate fires on the model's PREDICTED
chop, so the realised benefit is (conditioning value of the label) x (how well the model finds it).

This script measures that product directly: train each label's GBM3 on TRAIN, predict the regime on
the evidence-signal window, and compute the chop-gated conditional lift improvement per signal-side,
VAL/OOS split -- exactly the Phase 2 statistic but with PREDICTED instead of true chop.

⚠️ NO ADDITIONAL REGIME-OOS SPEND. The evidence-signal window (2025-09-01~2026-02-17) sits INSIDE
the regime model's TRAIN range, so regime predictions there are IN-SAMPLE. That is the same
disclosed caveat the 2026-08-27 baseline carried: the chop/non-chop boundary is "best available",
not OOS-clean, and it applies EQUALLY to both arms so the COMPARISON between them stays fair --
which is the only thing this script claims. It is not a clean estimate of either arm's absolute
live gate quality.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from analyze_eth_confluence_oscillator_bottom_top_evidence_20260814 import (  # noqa: E402
    OOS_END as EV_OOS_END, event_study, load_zigzag_pivots,
)
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import (  # noqa: E402
    OOS_START as EV_OOS_START, VAL_END as EV_VAL_END, VAL_START as EV_VAL_START,
)
from live_evidence_signal_dashboard_20260823 import SIGNAL_ORDER  # noqa: E402
from research_eth_regime_label_conditional_lift_20260902 import (  # noqa: E402
    K_HORIZON, MIN_SEG_FIRES, build_evidence_frame, seg_lift,
)
from research_eth_regime_s12k3_label_train_20260902 import (  # noqa: E402
    GBM3_HP, GBM3_MODEL_PATH, SEED, deployed_label, load_frame, s12k3_label,
)
from research_eth_regime_scalping_label_geometry_20260902 import TRAIN_END, TRAIN_START  # noqa: E402

OUT_DIR = ROOT / "tmp/eth_regime_s12k3_predicted_gate_lift_20260902"


def main() -> None:
    payload = joblib.load(GBM3_MODEL_PATH)
    feat_cols, medians = payload["feature_cols"], payload["feature_medians"]

    reg = load_frame()
    ts_r = reg["timestamp"]
    tr_m = ((ts_r >= TRAIN_START) & (ts_r <= TRAIN_END)).to_numpy()
    x = reg[feat_cols].apply(pd.to_numeric, errors="coerce")
    for c in feat_cols:
        x[c] = x[c].replace([np.inf, -np.inf], np.nan).fillna(medians.get(c, 0.0))

    y_new, _, _ = s12k3_label(reg, tr_m)
    labels = {"DEPLOYED_RegimeEngine": deployed_label(reg), "S12_K3": y_new}

    pred_frames = {}
    for lname, y in labels.items():
        m = HistGradientBoostingClassifier(random_state=SEED, **GBM3_HP).fit(x[tr_m], y[tr_m])
        pred_frames[lname] = pd.DataFrame({"timestamp": ts_r, "pred": m.predict(x)})
        print(f"{lname}: fitted, predicted chop share over full series "
              f"{float((pred_frames[lname]['pred'] == 2).mean()):.3f}")

    frame = build_evidence_frame()
    pivots = load_zigzag_pivots()
    ts_e = frame["timestamp"]
    windows = {"VAL": ((ts_e >= EV_VAL_START) & (ts_e <= EV_VAL_END)).to_numpy(),
               "OOS": ((ts_e >= EV_OOS_START) & (ts_e <= EV_OOS_END)).to_numpy()}
    windows["POOLED"] = windows["VAL"] | windows["OOS"]
    pivot_pos = {s: frame.index[frame["timestamp"].isin(
        pivots.loc[pivots["pivot_type"] == s, "timestamp"])].to_numpy() for s in ("bottom", "top")}

    rows = []
    for lname, pf in pred_frames.items():
        merged = frame[["timestamp"]].merge(pf, on="timestamp", how="left")
        pred_chop = (merged["pred"] == 2).fillna(False).to_numpy()
        for wname, wmask in windows.items():
            seg = pred_chop & wmask
            for sname, _ in SIGNAL_ORDER:
                for side in ("bottom", "top"):
                    sig = frame[f"{side}_{sname}"].fillna(False).to_numpy()
                    l_all, n_all = seg_lift(sig, pivot_pos[side], wmask)
                    l_chop, n_chop = seg_lift(sig, pivot_pos[side], seg)
                    if not (np.isfinite(l_all) and np.isfinite(l_chop)) or l_all <= 0:
                        continue
                    rows.append({"label": lname, "window": wname, "signal": sname, "side": side,
                                 "n_all": n_all, "n_gated": n_chop,
                                 "lift_all": round(l_all, 3), "lift_gated": round(l_chop, 3),
                                 "improvement": round(l_chop / l_all - 1.0, 4)})

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "predicted_gate_lift.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)

    print("\n=== PREDICTED-chop gated lift improvement (the quantity that actually ships) ===")
    summ = df.groupby(["label", "window"]).agg(
        cells=("improvement", "size"), mean=("improvement", "mean"),
        median=("improvement", "median"), pos=("improvement", lambda s: int((s > 0).sum()))).reset_index()
    print(summ.round(4).to_string(index=False))

    print("\n=== per signal-side: both-window-positive count (the ss5.15 gate) ===")
    for lname in labels:
        sub = df[df["label"] == lname].copy()
        sub["ss"] = sub["signal"] + "." + sub["side"]
        p = sub.pivot_table(index="ss", columns="window", values="improvement")
        if "VAL" in p and "OOS" in p:
            both = ((p["VAL"] > 0) & (p["OOS"] > 0)).sum()
            print(f"  {lname:22s} {int(both)}/{len(p)} cells positive in BOTH windows "
                  f"| mean VAL {p['VAL'].mean():+.4f} OOS {p['OOS'].mean():+.4f}")
            print(p.round(3).sort_values("OOS", ascending=False).to_string())
    (OUT_DIR / "summary.json").write_text(json.dumps(summ.to_dict(orient="records"), indent=2))
    print(f"\nWrote {OUT_DIR}/predicted_gate_lift.csv")


if __name__ == "__main__":
    main()
