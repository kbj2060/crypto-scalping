"""Equity/trade chart for the selected JM/czz regime-MoE config (2026-08-08).

Rebuilds the selected config's side_state from the saved seed models (same code path as
the --stage oos read in train_eval_btc_jm_regime_moe_20260808.py), then plots, for VAL
and OOS separately: price with entry markers and the causal fresh-forward equity curve.
Diagnostic visualization only -- numbers of record come from the stage outputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import lightgbm as lgb  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from core.causal_futures_backtest import simulate_single_position  # noqa: E402
from train_eval_btc_regime_conditioned_entry_20260808 import load_all, per_regime_auc, REGIME_NAMES, TOP_K  # noqa: E402
from train_eval_btc_jm_regime_moe_20260808 import (  # noqa: E402
    GATES, N_SEEDS, OUT_DIR, apply_policies, draw_seeds, load_jm,
)
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    HORIZON_BARS, MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE,
)

C_BULL, C_BEAR, INK = "#2563EB", "#D9542B", "#1F2430"


def main() -> int:
    prior = json.loads((OUT_DIR / "val_results.json").read_text())
    sel = prior["selected"]
    assert sel is not None, "no selected config"
    seeds = draw_seeds()
    panel, ts, x, feat_cols, action, tp_moves, sl_moves, _d2, train_mask, val_mask, oos_mask = load_all()
    jm = load_jm(ts)
    tr_idx = np.flatnonzero(train_mask)
    regime = jm[GATES[sel["gate"]]]
    tops = {}
    for r in (0, 2):
        auc_tr = per_regime_auc(x, action, tr_idx, regime, r)
        dev = np.abs(np.nan_to_num(auc_tr, nan=0.5) - 0.5)
        tops[r] = np.argsort(-dev)[:TOP_K]
    czz_dir = None
    if sel.get("consensus") == "on":
        czz_named = jm["czz4"]
        czz_dir = np.where(czz_named == 2, 1, np.where(czz_named == 0, -1, 0)).astype(np.int64)
    tag = f"{sel['gate']}_{sel['featset']}_p{sel['purge']}"

    fig, axes = plt.subplots(2, 2, figsize=(16, 8), gridspec_kw={"height_ratios": [3, 2], "hspace": 0.25})
    for col_i, (split_name, mask) in enumerate((("VAL 2025-09..12", val_mask), ("OOS 2026-01..03", oos_mask))):
        s_idx = np.flatnonzero(mask)
        probas = [np.zeros((len(panel), 3)) for _ in seeds]
        for r in (0, 2):
            cols = tops[r] if sel["featset"] == "top20" else np.arange(x.shape[1])
            sub = s_idx[regime[s_idx] == r]
            if not len(sub):
                continue
            for si, seed in enumerate(seeds):
                booster = lgb.Booster(model_file=str(OUT_DIR / "models" / f"{tag}_{REGIME_NAMES[r]}_seed{seed}.txt"))
                probas[si][sub] = booster.predict(x[sub][:, cols])
        proba_bag = sum(probas) / N_SEEDS
        side_state = apply_policies(proba_bag, len(panel), s_idx, regime, sel["threshold"], sel["bear"], czz_dir)
        dec = np.flatnonzero(mask & (side_state != 0) & np.isfinite(tp_moves) & np.isfinite(sl_moves))
        res = simulate_single_position(
            timestamps=ts, open_px=panel["open"].to_numpy(np.float64), high=panel["high"].to_numpy(np.float64),
            low=panel["low"].to_numpy(np.float64), close=panel["close"].to_numpy(np.float64),
            decision_indices=dec, scores=side_state[dec].astype(np.float64), tp_moves=tp_moves[dec],
            sl_moves=sl_moves[dec], upper_threshold=0.0, lower_threshold=0.0, horizon_bars=HORIZON_BARS,
            margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE)
        ledger, equity = res.ledger, res.equity
        h_ts = ts.to_numpy()[s_idx]
        axp = axes[0, col_i]
        axp.plot(h_ts, panel["close"].to_numpy()[s_idx], color=INK, linewidth=0.9)
        for _, row in ledger.iterrows():
            c = C_BULL if row["side"] == 1 else C_BEAR
            axp.axvline(row["entry_timestamp"], color=c, alpha=0.35, linewidth=0.8)
        axp.set_title(f"{split_name} — {len(ledger)} trades  PnL {(equity[-1] - 1) * 100:+.1f}%",
                      loc="left", fontsize=11, color=INK)
        axe = axes[1, col_i]
        eq_ts = ts.to_numpy()[s_idx] if len(equity) == len(s_idx) else np.arange(len(equity))
        axe.plot(eq_ts, equity, color="#0E7C66", linewidth=1.2)
        axe.axhline(1.0, color="#9AA0A6", linewidth=0.8)
        axe.set_title("fresh-forward equity", loc="left", fontsize=10, color=INK)
        for a in (axp, axe):
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
    fig.suptitle(f"BTC regime-MoE selected: {json.dumps({k: sel[k] for k in ('gate', 'featset', 'purge', 'bear', 'rule')})}"
                 + (f"  consensus={sel.get('consensus')}" if sel.get("consensus") else ""),
                 fontsize=10, y=1.0)
    out = OUT_DIR / "moe_selected_equity.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
