#!/usr/bin/env python3
"""Does collapsing S12_K3 to 2-class (trend vs chop) help? -- user question 2026-09-02
("추세와 chop으로 분류되나?").

MOTIVATION. The evidence-signal gate consumes ONLY the chop class; bull/bear never enters it.
Phase 3 showed the direction classes are the model's WEAKEST (S12_K3 OOS bull_R 0.8471 /
bear_R 0.8203 vs chop_R 0.8976), and that S12_K3 leans ~2x harder than the deployed label on the
direction-proxy features. So a large share of the model's difficulty is spent on a distinction the
gate discards. Collapsing bull+bear -> trend removes that burden; the question is whether chop
identification actually improves as a result.

⚠️ NO OOS SPEND. Accuracy is measured on the INTERNAL VAL slice (2026-04-01~06-30) with the model
fit on 2024-01-01~2026-03-31 -- the regime OOS (2026-07-01~08-19) is NOT touched (Phase 3 already
spent one look on it). Gate lift reuses the Phase 3b setup (evidence window, in-sample regime
predictions, disclosed) so the numbers stay comparable to that table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from analyze_eth_confluence_oscillator_bottom_top_evidence_20260814 import (  # noqa: E402
    OOS_END as EV_OOS_END, load_zigzag_pivots,
)
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import (  # noqa: E402
    OOS_START as EV_OOS_START, VAL_END as EV_VAL_END, VAL_START as EV_VAL_START,
)
from live_evidence_signal_dashboard_20260823 import SIGNAL_ORDER  # noqa: E402
from research_eth_regime_label_conditional_lift_20260902 import build_evidence_frame, seg_lift  # noqa: E402
from research_eth_regime_s12k3_label_train_20260902 import (  # noqa: E402
    GBM3_HP, GBM3_MODEL_PATH, SEED, deployed_label, load_frame, s12k3_label,
)
from research_eth_regime_scalping_label_geometry_20260902 import (  # noqa: E402
    TRAIN_END, TRAIN_START, _run_lengths,
)

SEL_END = pd.Timestamp("2026-03-31T23:55:00")     # fit slice ends here
IVAL_START = pd.Timestamp("2026-04-01T00:00:00")  # internal VAL -- NOT the regime OOS
OUT_DIR = ROOT / "tmp/eth_regime_s12k3_2class_vs_3class_20260902"


def chop_metrics(y: np.ndarray, pred: np.ndarray, chop_id: int) -> dict:
    cm = confusion_matrix((y == chop_id).astype(int), (pred == chop_id).astype(int), labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    runs = _run_lengths(pred)
    return {"chop_precision": round(float(tp / (tp + fp)), 4) if tp + fp else None,
            "chop_recall": round(float(tp / (tp + fn)), 4) if tp + fn else None,
            "chop_f1": round(float(2 * tp / (2 * tp + fp + fn)), 4) if tp else None,
            "pred_chop_share": round(float((pred == chop_id).mean()), 4),
            "label_chop_share": round(float((y == chop_id).mean()), 4),
            "pred_flip_rate": round(float(np.mean(pred[1:] != pred[:-1])), 4),
            "median_run_bars": float(np.median(runs)) if runs else 0.0}


def main() -> None:
    payload = joblib.load(GBM3_MODEL_PATH)
    feat_cols, medians = payload["feature_cols"], payload["feature_medians"]
    reg = load_frame()
    ts = reg["timestamp"]
    tr_full = ((ts >= TRAIN_START) & (ts <= TRAIN_END)).to_numpy()
    fit_m = ((ts >= TRAIN_START) & (ts <= SEL_END)).to_numpy()
    ival_m = ((ts >= IVAL_START) & (ts <= TRAIN_END)).to_numpy()
    print(f"fit {fit_m.sum():,} bars ({TRAIN_START.date()}~{SEL_END.date()}) | "
          f"internal VAL {ival_m.sum():,} bars ({IVAL_START.date()}~{TRAIN_END.date()}) "
          "| regime OOS NOT touched")

    x = reg[feat_cols].apply(pd.to_numeric, errors="coerce")
    for c in feat_cols:
        x[c] = x[c].replace([np.inf, -np.inf], np.nan).fillna(medians.get(c, 0.0))

    y_s12k3, _, _ = s12k3_label(reg, tr_full)
    base3 = {"DEPLOYED": deployed_label(reg), "S12_K3": y_s12k3}
    variants = {}
    for lname, y3 in base3.items():
        variants[f"{lname}_3class"] = (y3, 2)                       # chop id = 2
        variants[f"{lname}_2class"] = ((y3 == 2).astype(int), 1)    # 0=trend, 1=chop
    print()

    acc, gate_models = {}, {}
    for vname, (y, chop_id) in variants.items():
        m = HistGradientBoostingClassifier(random_state=SEED, **GBM3_HP).fit(x[fit_m], y[fit_m])
        acc[vname] = chop_metrics(y[ival_m], m.predict(x[ival_m]), chop_id)
        r = acc[vname]
        print(f"{vname:18s} internal-VAL  chop_P={r['chop_precision']:.4f} "
              f"chop_R={r['chop_recall']:.4f} F1={r['chop_f1']:.4f}  "
              f"pred_flip={r['pred_flip_rate']:.4f}  pred_chop_share={r['pred_chop_share']:.3f}")
        # separate fit on FULL train for the gate-lift arm (matches Phase 3b's setup)
        mg = HistGradientBoostingClassifier(random_state=SEED, **GBM3_HP).fit(x[tr_full], y[tr_full])
        gate_models[vname] = (pd.DataFrame({"timestamp": ts, "pred": mg.predict(x)}), chop_id)

    frame = build_evidence_frame()
    pivots = load_zigzag_pivots()
    ts_e = frame["timestamp"]
    windows = {"VAL": ((ts_e >= EV_VAL_START) & (ts_e <= EV_VAL_END)).to_numpy(),
               "OOS": ((ts_e >= EV_OOS_START) & (ts_e <= EV_OOS_END)).to_numpy()}
    windows["POOLED"] = windows["VAL"] | windows["OOS"]
    pivot_pos = {s: frame.index[frame["timestamp"].isin(
        pivots.loc[pivots["pivot_type"] == s, "timestamp"])].to_numpy() for s in ("bottom", "top")}

    rows = []
    for vname, (pf, chop_id) in gate_models.items():
        merged = frame[["timestamp"]].merge(pf, on="timestamp", how="left")
        gate = (merged["pred"] == chop_id).fillna(False).to_numpy()
        for wname, wmask in windows.items():
            for sname, _ in SIGNAL_ORDER:
                for side in ("bottom", "top"):
                    sig = frame[f"{side}_{sname}"].fillna(False).to_numpy()
                    l_all, _ = seg_lift(sig, pivot_pos[side], wmask)
                    l_g, n_g = seg_lift(sig, pivot_pos[side], gate & wmask)
                    if not (np.isfinite(l_all) and np.isfinite(l_g)) or l_all <= 0:
                        continue
                    rows.append({"variant": vname, "window": wname, "ss": f"{sname}.{side}",
                                 "n_gated": n_g, "improvement": round(l_g / l_all - 1.0, 4)})
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "gate_lift.csv", index=False)
    (OUT_DIR / "accuracy_internal_val.json").write_text(json.dumps(acc, indent=2))

    pd.set_option("display.width", 200)
    print("\n=== predicted-gate lift improvement (evidence window; in-sample regime, as Phase 3b) ===")
    summ = df.groupby(["variant", "window"]).agg(
        cells=("improvement", "size"), mean=("improvement", "mean"),
        pos=("improvement", lambda s: int((s > 0).sum()))).reset_index()
    print(summ.round(4).to_string(index=False))
    print("\n=== both-window-positive cells (ss5.15 gate) ===")
    for v in variants:
        sub = df[df["variant"] == v]
        p = sub.pivot_table(index="ss", columns="window", values="improvement")
        if "VAL" in p and "OOS" in p:
            print(f"  {v:18s} {int(((p['VAL'] > 0) & (p['OOS'] > 0)).sum())}/{len(p)}")
    print(f"\nWrote {OUT_DIR}/")


if __name__ == "__main__":
    main()
