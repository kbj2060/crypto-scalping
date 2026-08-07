#!/usr/bin/env python3
"""Genuine leave-one-window-out validation for the freshly retrained BTC zig075
entry-quality model (tmp/causal_regen_20260516/
btc_omega4_3head_parent72_loose_entry_quality_20260708_zig075_freshdata_20260803).

The 2026-07-08 zig075 vintage was validated with a single VAL(Oct-Dec2025)/OOS(2026)
split and a downstream fast_param_search grid that picked the VAL-best leg config,
which then collapsed OOS (VAL +44.6%/-16.2%MDD/11tr -> OOS -47.3%/-47.3%MDD/33tr).
That is a classic VAL-overfit failure mode (see this project's Sigma6 leave-one-
window-out precedent: scripts/research_sigma6_regime_filter_leave_one_window_out_20260801.py).

This script does not retrain the network per fold (the model itself is trained
once, exactly like the reference Sigma6 script keeps its tape/backtest engine
fixed and only varies the window used for config selection vs scoring). Instead
it takes the single already-trained zig075 bundle's predictions across the full
genuinely-out-of-sample span (validation Oct 2025-Dec 2025 + oos Jan 2026-Aug 2026,
i.e. everything after the train/val split the network never got gradient updates
from) and asks: if you were only allowed to pick the quality_threshold using 4 of
5 rolling windows (majority-vote), does it still work on the 5th, truly-unseen
window?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_btc_20260708 as omega  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_btc_20260708 as zig  # noqa: E402

MODEL_DIR = ROOT / "tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_20260708_zig075_freshdata_20260803"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/btc_zig075_freshdata_leave_one_window_out_20260803"
LABEL_DIR = ROOT / "tmp/causal_regen_20260516/btc_zigzag_action_labels_freshforward_ext_20260802"
COST_MULT = 3.0
Q_VALUES = [0.65, 0.70, 0.75, 0.80, 0.85]

# 5 rolling ~2-month windows spanning the genuinely-unseen span: validation
# (2025-10-01..2025-12-31, part of btc_features_2025.csv the model was NOT
# trained on -- train_raw is only rows < SPLIT_TS) through oos (all of
# btc_features_2026.csv, now extended through 2026-08-01).
WINDOWS = [
    ("W1_2025Q4a", "2025-10-01", "2025-11-30"),
    ("W2_2025Q4b_2026Q1a", "2025-12-01", "2026-01-31"),
    ("W3_2026Q1b", "2026-02-01", "2026-03-31"),
    ("W4_2026Q2a", "2026-04-01", "2026-05-31"),
    ("W5_2026Q2b_Q3", "2026-06-01", "2026-08-01"),
]


def _q_tag(q: float) -> str:
    return f"q{int(round(float(q) * 100.0)):03d}"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = zig._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=LABEL_DIR,
        quality_mode="same_as_direction",
        quality_label_dir=None,
        quality_min_edge=0.0010,
        quality_max_mae=0.0100,
        quality_min_mfe_mae=1.20,
        quality_max_hold_bars=288,
    )
    val_raw = frames["val_raw"].reset_index(drop=True)
    oos_raw = frames["oos_raw"].reset_index(drop=True)
    fee, slip = omega._load_fee_slip()

    full_raw = pd.concat([val_raw, oos_raw], ignore_index=True)
    full_raw["timestamp"] = pd.to_datetime(full_raw["timestamp"])
    if not full_raw["timestamp"].is_monotonic_increasing:
        raise RuntimeError("val_raw + oos_raw concatenation is not time-ordered")

    # Per-threshold decisions across the full unseen span, built from the
    # already-trained bundle's saved prediction CSVs (val=oof-prefixed,
    # oos=non-oof-prefixed; parent._to_decisions needs the right oof flag).
    full_dec_by_q: dict[str, pd.DataFrame] = {}
    for q in Q_VALUES:
        tag = _q_tag(q)
        val_src = pd.read_csv(MODEL_DIR / f"validation_predictions_{tag}.csv", parse_dates=["timestamp"])
        oos_src = pd.read_csv(MODEL_DIR / f"oos_predictions_{tag}.csv", parse_dates=["timestamp"])
        if len(val_src) != len(val_raw) or len(oos_src) != len(oos_raw):
            raise RuntimeError(f"prediction/raw row-count mismatch for {tag}")
        val_dec = parent._to_decisions(val_src, oof=True).reset_index(drop=True)
        oos_dec = parent._to_decisions(oos_src, oof=False).reset_index(drop=True)
        full_dec_by_q[tag] = pd.concat([val_dec, oos_dec], ignore_index=True)

    windows = [(label, pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta("23h59min59s")) for label, s, e in WINDOWS]

    def window_metrics(tag: str, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
        mask = (full_raw["timestamp"] >= start) & (full_raw["timestamp"] <= end)
        idx = np.flatnonzero(mask.to_numpy())
        if len(idx) == 0:
            raise RuntimeError(f"empty window for {tag}: {start}..{end}")
        w_raw = full_raw.iloc[idx].reset_index(drop=True)
        w_dec = full_dec_by_q[tag].iloc[idx].reset_index(drop=True)
        m = omega._metrics(w_raw, w_dec, fee=fee, slip=slip, cost_mult=COST_MULT)
        return {"pnl": round(float(m["pnl"]), 2), "mdd": round(float(m["mdd"]), 2), "trades": int(m["trades"]), "wr": round(float(m["wr"]), 4)}

    # Precompute every (threshold, window) result once.
    grid_res: dict[str, dict[str, dict[str, Any]]] = {}
    for q in Q_VALUES:
        tag = _q_tag(q)
        grid_res[tag] = {label: window_metrics(tag, start, end) for label, start, end in windows}

    print("Per-window, per-threshold results (all genuinely post-training-cutoff data):")
    grid_df = pd.DataFrame(
        {label: {tag: f"pnl={r[label]['pnl']:+.2f} mdd={r[label]['mdd']:.2f} tr={r[label]['trades']}" for tag, r in grid_res.items()} for label, _, _ in windows}
    )
    print(grid_df.to_string())

    fold_rows: list[dict[str, Any]] = []
    for held_idx, (held_label, held_start, held_end) in enumerate(windows):
        selection_labels = [lbl for i, (lbl, _, _) in enumerate(windows) if i != held_idx]

        candidates = []
        for q in Q_VALUES:
            tag = _q_tag(q)
            wins = sum(1 for lbl in selection_labels if grid_res[tag][lbl]["pnl"] > 0.0)
            mean_pnl = float(np.mean([grid_res[tag][lbl]["pnl"] for lbl in selection_labels]))
            candidates.append((tag, wins, mean_pnl))
        # Majority agreement: threshold profitable on >=3 of the 4 selection windows.
        majority = [c for c in candidates if c[1] >= 3]
        pool = majority if majority else candidates
        pool.sort(key=lambda t: (-t[1], -t[2]))
        sel_tag, sel_wins, sel_mean_pnl = pool[0]

        held_r = grid_res[sel_tag][held_label]
        fold_rows.append(
            {
                "held_out": held_label,
                "selected_threshold": sel_tag,
                "selection_wins_of_4": sel_wins,
                "selection_mean_pnl_other4": round(sel_mean_pnl, 2),
                "used_majority_pool": bool(majority),
                "held_out_pnl": held_r["pnl"],
                "held_out_mdd": held_r["mdd"],
                "held_out_trades": held_r["trades"],
                "held_out_wr": held_r["wr"],
                "held_out_profitable": bool(held_r["pnl"] > 0.0),
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUT_DIR / "leave_one_window_out_results.csv", index=False)
    print()
    print(fold_df.to_string(index=False))

    n_pass = int(fold_df["held_out_profitable"].sum())
    print(f"\n{n_pass}/{len(fold_df)} folds: threshold selected from the OTHER 4 windows is profitable on the held-out window it never saw.")

    grid_df.to_csv(OUT_DIR / "per_window_per_threshold_grid.csv")
    summary = {
        "model_dir": str(MODEL_DIR),
        "windows": [{"label": l, "start": s, "end": e} for l, s, e in WINDOWS],
        "grid_results": grid_res,
        "fold_rows": fold_rows,
        "n_pass": n_pass,
        "n_folds": len(fold_df),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
