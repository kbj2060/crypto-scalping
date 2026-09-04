#!/usr/bin/env python3
"""Does swapping the chop-gate's regime source from GBM3 (argmax bull/bear/chop, used by every
sibling script in this lineage so far, including backtest_eth_evidence_signal_liquidation_
confluence_costgate_20260827.py) to GBM2 (2-class trend/chop, scripts/train_eth_regime_gbm2_
trend_chop_20260827.py -- trained same day specifically because GBM3 flip-flops too much for the
user's discretionary chop-fade use case) change the ONE positive result this evidence-signal x
regime x liquidation-confluence research line has produced so far: orthogonal_combo:bottom:
chop+near_or_mid_support passing a no-trade(0%) benchmark in 2/6 windows (val +0.66%, oos_q2
+0.47%, see docs/experiments/eth_evidence_signal_liquidation_confluence_20260827.md)?

Motivation: research_eth_evidence_signal_regime_model_comparison_20260827.py already found
orthogonal_combo/short_term_return_z are the only 2 (of 8) evidence signals whose chop-lift is
robust across GBM3-model / GBM2-model / GBM2-label regime sources -- orthogonal_combo being
exactly today's star confluence candidate makes this the natural next check, not a fishing
expedition. GBM2's own diagnostic Part B in research_eth_evidence_signal_liquidation_confluence_
20260827.py already used build_regime_frame_gbm2_LABEL (ground-truth debounced label, not a live-
causal model) for its lift-only diagnostic -- this script instead uses the actual GBM2 MODEL's
predict_proba() (payload["model"], same read-only-inference pattern as GBM3's _regime_labels()),
which is what a live/backtest DECISION gate must use to stay causal (the label's K=12 debounce is
a training-target construction detail, not something a real-time consumer can replicate without
seeing future bars).

GBM2's live-deployed hysteresis_config is k_bars=1/band=0.0 (2026-08-27 user override, "immediate
reaction" -- see model payload / eth_regime_gbm2_trend_chop_model_20260827 memory) -- this
literally disables the serving-side Schmitt-trigger/debounce, so predict_proba().idxmax(axis=1)
(raw argmax at the 0.5 boundary) reproduces the CURRENT LIVE dashboard's regime call exactly, with
no separate _apply_hysteresis() call needed.

Only the regime source changes -- CANDIDATES, VARIANTS, TP:SL/leverage/cost engine, liquidation-
confluence tertile logic (add_confluence, unchanged from the GBM3 script), and the 6 pre-registered
windows are all reused verbatim from backtest_eth_evidence_signal_liquidation_confluence_costgate_
20260827.py, so any difference in results is attributable ONLY to GBM3-vs-GBM2 chop classification.

CAVEAT (same as every sibling script): 2025q1..oos_q2 are inside GBM2's own TRAIN range
(2024-01-01~2026-06-30) -- in-sample regime classification, not a clean OOS split of the
classifier itself (identical caveat already disclosed for the GBM3 version).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
from backtest_eth_evidence_signal_chop_gated_costgate_20260827 import _compute_frame  # noqa: E402
from backtest_eth_evidence_signal_liquidation_confluence_costgate_20260827 import (  # noqa: E402
    CANDIDATES, VARIANTS, add_confluence, find_breakeven_bp, run_window_confluence,
)
from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12  # noqa: E402

OUT_DIR = ROOT / "tmp/eth_evidence_signal_liquidation_confluence_gbm2gate_20260827"
GBM3_REPORT_PATH = ROOT / "tmp/eth_evidence_signal_liquidation_confluence_costgate_20260827/report.json"
GBM2_MODEL_PATH = ROOT / "tmp" / "eth_regime_gbm2_trend_chop_20260827" / "model.joblib"


def _load_raw(base_csv: Path) -> pd.DataFrame:
    """Verbatim copy of _compute_frame's raw-loading steps (backtest_eth_evidence_signal_chop_
    gated_costgate_20260827.py) so the returned frame row-aligns 1:1 with _compute_frame(base_csv)'s
    output -- compute_signals() preserves row count/order, so this is the only way to recover the
    same `raw` _compute_frame used internally without duplicating its signal/ATR computation."""
    raw = pd.read_csv(base_csv, low_memory=False)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    return raw.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _regime_labels_gbm2(raw: pd.DataFrame) -> pd.Series:
    feats = _with_raw_state12(raw)
    payload = joblib.load(GBM2_MODEL_PATH)
    cols = payload["feature_cols"]
    med = pd.Series(payload["feature_medians"])
    for c in cols:
        if c not in feats.columns:
            feats[c] = med.get(c, 0.0)
    x = feats[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    proba = payload["model"].predict_proba(x)
    classes = list(payload["classes"])
    prob_df = pd.DataFrame(proba, columns=classes)
    return prob_df.idxmax(axis=1)  # "chop"/"trend" -- raw argmax == live (hysteresis k_bars=1/band=0 override)


def _compute_frame_gbm2(base_csv: Path) -> pd.DataFrame:
    fr = _compute_frame(base_csv)  # GBM3 regime_label computed and discarded -- cheap, avoids duplicating compute_signals/atr_pct
    raw = _load_raw(base_csv)
    assert len(raw) == len(fr), f"row-count mismatch: raw={len(raw)} fr={len(fr)} (cleanup steps diverged)"
    fr = fr.copy()
    fr["regime_label"] = _regime_labels_gbm2(raw).to_numpy()
    return fr


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building 2025/2026 frames with GBM2 (instead of GBM3) chop gate...")
    frames = {}
    for yr, base_csv in (("2025", gate.sweep.BASE_2025), ("2026", gate.sweep.BASE_2026)):
        fr = _compute_frame_gbm2(base_csv)
        fr = add_confluence(fr)
        frames[yr] = fr
        chop_share = (fr["regime_label"] == "chop").mean() * 100
        print(f"  {yr}: {len(fr)} rows, chop={chop_share:.1f}%, near_or_mid_support={fr['near_or_mid_support'].mean() * 100:.1f}%")

    report: dict[str, Any] = {"config": {"regime_source": "gbm2_model_raw_argmax"}, "results": {}}
    summary_rows = []
    for name, side in CANDIDATES:
        bcol = f"bottom_{name}"
        for variant in VARIANTS:
            key = f"{name}:{side}:{variant}"
            print(f"\n--- {key} ---")
            print(f"{'window':<8} {'n_trades':>8} {'wr':>7} {'return':>10} {'a_long':>9} {'a_short':>9} {'beats_bm':>9}  breakeven_bp")
            windows_out = {}
            for wname, wd in gate.WINDOW_DEFS.items():
                frame = frames["2025"] if wd["base_csv"] == gate.sweep.BASE_2025 else frames["2026"]
                res_std = run_window_confluence(frame, bcol, variant, start=wd["start"], end=wd["end"], roundtrip_cost=0.001)
                be = find_breakeven_bp(frame, bcol, variant, start=wd["start"], end=wd["end"])
                be_str = f"{be:.1f}bp" if be is not None else ">200bp"
                windows_out[wname] = {**res_std, "breakeven_bp": be}
                print(f"{wname:<8} {res_std['n_trades']:>8d} "
                      f"{res_std['wr'] * 100 if np.isfinite(res_std['wr']) else float('nan'):>6.1f}% "
                      f"{res_std['total_return'] * 100:>9.2f}% {res_std['always_long_return'] * 100:>8.2f}% "
                      f"{res_std['always_short_return'] * 100:>8.2f}%  {str(res_std['beats_benchmark']):>9}  {be_str}")
            report["results"][key] = windows_out
            no_trade_wins = sum(1 for w in windows_out.values() if w["total_return"] > 0)
            total_ret = sum(w["total_return"] for w in windows_out.values())
            print(f"SUMMARY {key}: no-trade-benchmark passes in {no_trade_wins}/{len(windows_out)} windows, "
                  f"sum(total_return)={total_ret * 100:.2f}%")
            summary_rows.append({"signal": name, "variant": variant, "no_trade_wins": no_trade_wins,
                                  "sum_total_return_pct": total_ret * 100})

    print("\n=== Cross-candidate summary (sum of total_return across 6 windows), GBM2 gate ===")
    print(f"{'signal':<24} {'ungated':>10} {'chop':>10} {'chop_confluence':>16} {'delta(conf-chop)':>18}")
    by_sig: dict[str, dict[str, float]] = {}
    for row in summary_rows:
        by_sig.setdefault(row["signal"], {})[row["variant"]] = row["sum_total_return_pct"]
    for name, _side in CANDIDATES:
        v = by_sig[name]
        delta = v["chop_confluence"] - v["chop"]
        print(f"{name:<24} {v['ungated']:>9.2f}% {v['chop']:>9.2f}% {v['chop_confluence']:>15.2f}% {delta:>+17.2f}%")

    print("\n=== GBM3 vs GBM2 gate, orthogonal_combo:bottom:chop_confluence, per-window total_return ===")
    if GBM3_REPORT_PATH.exists():
        gbm3 = json.loads(GBM3_REPORT_PATH.read_text())["results"]["orthogonal_combo:bottom:chop_confluence"]
        gbm2 = report["results"]["orthogonal_combo:bottom:chop_confluence"]
        print(f"{'window':<8} {'GBM3 ret':>10} {'GBM2 ret':>10} {'GBM3 n':>7} {'GBM2 n':>7}")
        for wname in gate.WINDOW_DEFS:
            g3, g2 = gbm3[wname], gbm2[wname]
            print(f"{wname:<8} {g3['total_return'] * 100:>9.2f}% {g2['total_return'] * 100:>9.2f}% {g3['n_trades']:>7d} {g2['n_trades']:>7d}")
    else:
        print(f"  (GBM3 report not found at {GBM3_REPORT_PATH}, skipping direct comparison)")

    out_path = OUT_DIR / "report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
