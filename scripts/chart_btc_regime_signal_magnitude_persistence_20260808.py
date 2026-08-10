"""Why the regime-expert axis keeps failing: per-month signal-magnitude persistence (2026-08-08).

For the causal JM (k3 lam32) bear and bull regimes: take the top-20 features by train
|AUC-0.5| (the same selection every expert line uses), then plot BY CALENDAR MONTH the
mean signed edge  sign_train(AUC-0.5) * (AUC_month - 0.5)  among nonzero-TB-action bars.
Sign stability (bars mostly above zero) with magnitude collapsing toward/below zero in
many months IS the measured failure mechanism: experts tuned on any window inherit a
magnitude that does not persist quarter-to-quarter.  Diagnostic only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from train_eval_btc_regime_conditioned_entry_20260808 import auc_binary, load_all, per_regime_auc, TOP_K  # noqa: E402
from train_eval_btc_jm_regime_moe_20260808 import load_jm  # noqa: E402

OUT = ROOT / "tmp/jm_regime_verification_20260808/regime_signal_magnitude_by_month.png"
INK = "#1F2430"
C_BULL, C_BEAR = "#2563EB", "#D9542B"


def month_edge(x, action, idx, top, s_tr):
    a = action[idx]
    nz = a != 0
    if nz.sum() < 150:
        return np.nan
    y = (a[nz] == 1).astype(int)
    vals = []
    for j, f in enumerate(top):
        auc = auc_binary(x[idx, f][nz].astype(np.float64), y)
        if np.isfinite(auc):
            vals.append(s_tr[j] * (auc - 0.5))
    return float(np.mean(vals)) if vals else np.nan


def main() -> int:
    panel, ts, x, feat_cols, action, tp_moves, sl_moves, _d2, train_mask, val_mask, oos_mask = load_all()
    jm = load_jm(ts)
    reg = jm["jm_lam32"]
    tr_idx = np.flatnonzero(train_mask)
    months = ts.dt.to_period("M").astype(str).to_numpy()
    uniq_months = sorted(set(months[np.flatnonzero(np.isfinite(tp_moves))]))

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    results = {}
    for ax, (r, nm, color) in zip(axes, ((0, "bear", C_BEAR), (2, "bull", C_BULL))):
        auc_tr = per_regime_auc(x, action, tr_idx, reg, r)
        dev = np.abs(np.nan_to_num(auc_tr, nan=0.5) - 0.5)
        top = np.argsort(-dev)[:TOP_K]
        s_tr = np.sign(auc_tr[top] - 0.5)
        vals = []
        for m in uniq_months:
            idx = np.flatnonzero((months == m) & (reg == r) & np.isfinite(tp_moves))
            vals.append(month_edge(x, action, idx, top, s_tr))
        results[nm] = dict(zip(uniq_months, [None if np.isnan(v) else round(v, 4) for v in vals]))
        xs = np.arange(len(uniq_months))
        ax.bar(xs, np.nan_to_num(vals, nan=0.0), color=[color if (np.isfinite(v) and v > 0) else "#9AA0A6" for v in vals], alpha=0.85)
        for i, v in enumerate(vals):
            if not np.isfinite(v):
                ax.plot(i, 0, marker="x", color="#6B7280", markersize=5)
        ax.axhline(0, color=INK, linewidth=0.8)
        ax.set_title(f"JM-{nm}: monthly mean signed edge of train-selected top-20 features "
                     f"(gray = wrong-sign month, x = too few bars)", loc="left", fontsize=10, color=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for boundary, label in (("2025-09", "VAL start"), ("2026-01", "OOS start")):
            if boundary in uniq_months:
                bi = uniq_months.index(boundary) - 0.5
                ax.axvline(bi, color="#0E7C66", linewidth=1.0, linestyle="--")
                ax.annotate(label, xy=(bi, ax.get_ylim()[1]), fontsize=8, color="#0E7C66",
                            ha="left", va="top")
    axes[1].set_xticks(np.arange(len(uniq_months)))
    axes[1].set_xticklabels(uniq_months, rotation=60, fontsize=7)
    fig.suptitle("BTC 5m TB labels — within-regime signal magnitude is NOT month-persistent", fontsize=12, y=0.98)
    fig.savefig(OUT, dpi=130, bbox_inches="tight", facecolor="white")
    (OUT.parent / "regime_signal_magnitude_by_month.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
