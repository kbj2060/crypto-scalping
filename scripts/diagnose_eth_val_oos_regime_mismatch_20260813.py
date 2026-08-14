#!/usr/bin/env python3
"""RESEARCH ONLY -- pure analysis, no training, no new backtest/simulation.

Investigates why VAL (2025-10-01..12-31) has repeatedly failed to represent OOS this session
(4 candidates: final_boss v2/v3, SLTP width recalibration, multi-slot MFE-gated capacity --
all improved on VAL and reversed on OOS). Reuses only already-computed data/features/ledgers:

  1. Volatility regime comparison across quarters (realized_vol_ratio, garman_klass_vol,
     bb_width, rogers_satchell_vol, parkinson_vol, ATR% via the same formula as
     eval_omega4_1_atr_safety_sltp_20260622._atr_pct) -- reads the exact same base+overlay CSVs
     research_eth_omega461_exit_sweep_20260721.py's load_frame reads.
  2. Trend/directional regime comparison (price drift, regime3_current_sensitive_wide24
     bull/bear/chop probability + dominant-regime share) across the same quarters.
  3. Side (long/short) asymmetry stability across quarters -- pulls from ALREADY-COMPUTED
     trade ledgers (no new simulation): the Phase1 robustness rolling walk-forward's 2025
     Q1-Q3 component x side sum_ret, the N=1 greedy-router baseline ledgers (VAL/OOS,
     tmp/eth_multislot_capacity_20260808/), and tonight's 5-seed MFE-gated ledgers
     (tmp/eth_multislot_mfe_gated_capacity_20260813/).
  4. Common-adverse-event finder for 2026-Q2 across the 5 MFE-gated seeds' OOS ledgers (why
     4/5 seeds hit the exact same -20.17% OOS MDD).

Findings written up in docs/experiments/eth_val_oos_regime_mismatch_investigation_20260813.md.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)


def _atr_pct(frame: pd.DataFrame, window: int) -> np.ndarray:
    """Inlined copy of eval_omega4_1_atr_safety_sltp_20260622._atr_pct (avoids importing that
    module here, which pulls in torch unnecessarily for this pure pandas/numpy task). Formula
    verified identical to the source via direct Read before inlining. atr_window=192 matches
    the live h48qual/zig075 COMPONENTS config in research_eth_omega461_exit_sweep_20260721.py."""
    high = pd.to_numeric(frame["high"], errors="raise").to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="raise").to_numpy(dtype=np.float64)
    close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    atr = pd.Series(tr).rolling(window=max(int(window), 1), min_periods=1).mean().to_numpy(dtype=np.float64)
    out = atr / np.maximum(close, 1.0e-12)
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite ATR percent")
    return out


BASE_2025 = ROOT / "data/splits/year_oos/training_features_2025.csv"
BASE_2026 = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
WIDE24_2025 = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2025_regime3_current_sensitive_hmm_wide24.csv"
WIDE24_2026 = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv"

PERIODS = [
    ("2025-Q1", "2025-01-01", "2025-03-31 23:59:59"),
    ("2025-Q2", "2025-04-01", "2025-06-30 23:59:59"),
    ("2025-Q3 (VAL 직전)", "2025-07-01", "2025-09-30 23:59:59"),
    ("VAL (2025-Q4)", "2025-10-01", "2025-12-31 23:59:59"),
    ("OOS Q1 2026", "2026-01-01", "2026-03-31 23:59:59"),
    ("OOS Q2 2026", "2026-04-01", "2026-06-30 23:59:59"),
    ("OOS tail (2026-07~)", "2026-07-01", "2026-07-20 23:59:59"),
]
VOL_COLS = ["realized_vol_ratio", "garman_klass_vol", "bb_width", "rogers_satchell_vol", "parkinson_vol", "atr_pct_192"]
REGIME_COLS = {
    "bull": "regime3_current_sensitive_wide24_bull_prob",
    "bear": "regime3_current_sensitive_wide24_bear_prob",
    "chop": "regime3_current_sensitive_wide24_chop_prob",
}


def load_frame(base_csv: Path, wide24_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(base_csv, low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    overlay = pd.read_csv(wide24_csv, low_memory=False)
    overlay["timestamp"] = pd.to_datetime(overlay["timestamp"])
    cols = [c for c in overlay.columns if c != "timestamp"]
    frame = frame.merge(overlay[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    return frame


def section_1_2_regime_stats() -> pd.DataFrame:
    print("loading 2025 full-year frame...", flush=True)
    f2025 = load_frame(BASE_2025, WIDE24_2025)
    print("loading 2026 frame...", flush=True)
    f2026 = load_frame(BASE_2026, WIDE24_2026)
    full = pd.concat([f2025, f2026], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    full["atr_pct_192"] = _atr_pct(full, 192)
    full["log_ret_5m"] = np.log(full["close"]).diff()

    rows = []
    for label, start, end in PERIODS:
        sub = full[(full["timestamp"] >= start) & (full["timestamp"] <= end)]
        if len(sub) == 0:
            continue
        close_start, close_end = sub["close"].iloc[0], sub["close"].iloc[-1]
        drift_pct = (close_end / close_start - 1.0) * 100.0
        ann_vol = sub["log_ret_5m"].std() * np.sqrt(288 * 365) * 100.0
        dom = sub[[REGIME_COLS["bull"], REGIME_COLS["bear"], REGIME_COLS["chop"]]].idxmax(axis=1)
        dom_share = dom.value_counts(normalize=True) * 100.0
        row = {
            "period": label, "start": start[:10], "end": end[:10], "n_bars": len(sub),
            "price_drift_pct": drift_pct, "ann_realized_vol_pct_from_5mret": ann_vol,
            "bull_dominant_share_pct": dom_share.get(REGIME_COLS["bull"], 0.0),
            "bear_dominant_share_pct": dom_share.get(REGIME_COLS["bear"], 0.0),
            "chop_dominant_share_pct": dom_share.get(REGIME_COLS["chop"], 0.0),
        }
        for c in VOL_COLS:
            if c in sub.columns:
                row[f"{c}_mean"] = sub[c].mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    print("\n=== 1+2. 가격 드리프트 / 실현변동성 / 레짐3 비중 ===")
    print(out.to_string(index=False))
    return out


def _side_stats(df: pd.DataFrame, label: str) -> None:
    df = df.copy()
    df["side_label"] = np.where(df["side"] == 1, "LONG", "SHORT")
    agg = df.groupby("side_label")["trade_return"].agg(["count", "mean", "sum", lambda x: (x > 0).mean()])
    agg.columns = ["n", "mean_ret", "sum_ret", "win_rate"]
    print(f"\n--- {label} ---")
    print(agg.to_string())
    longs, shorts = df[df["side"] == 1]["trade_return"], df[df["side"] == -1]["trade_return"]
    if len(longs) >= 2 and len(shorts) >= 2:
        t, p = stats.ttest_ind(shorts, longs, equal_var=False)
        print(f"Welch t (short vs long) = {t:.3f}, p = {p:.4f}")


def section_3_side_asymmetry() -> None:
    print("\n=== 3. 사이드(LONG/SHORT) 비대칭 안정성 ===")
    val = pd.read_csv(ROOT / "tmp/eth_multislot_capacity_20260808/ledger_val_n1.csv")
    oos = pd.read_csv(ROOT / "tmp/eth_multislot_capacity_20260808/ledger_oos_n1.csv")
    oos["entry_timestamp"] = pd.to_datetime(oos["entry_timestamp"])
    _side_stats(val, "N=1 baseline, VAL (2025-10-01~12-31), n=29")
    _side_stats(oos, "N=1 baseline, OOS FULL (2026-01-01~06-25), n=37")
    _side_stats(oos[oos["entry_timestamp"] < "2026-04-01"], "N=1 baseline, OOS Q1 2026 only")
    _side_stats(oos[oos["entry_timestamp"] >= "2026-04-01"], "N=1 baseline, OOS Q2 2026 only")

    mfe_val = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "tmp/eth_multislot_mfe_gated_capacity_20260813/ledger_val_n3_mfegated_seed*.csv")))], ignore_index=True)
    mfe_oos = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "tmp/eth_multislot_mfe_gated_capacity_20260813/ledger_oos_n3_mfegated_seed*.csv")))], ignore_index=True)
    mfe_oos["entry_timestamp"] = pd.to_datetime(mfe_oos["entry_timestamp"])
    _side_stats(mfe_val, "tonight's N=3 MFE-gated, VAL, 5 seeds pooled")
    _side_stats(mfe_oos, "tonight's N=3 MFE-gated, OOS FULL, 5 seeds pooled")
    _side_stats(mfe_oos[mfe_oos["entry_timestamp"] < "2026-04-01"], "tonight's N=3 MFE-gated, OOS Q1 2026, 5 seeds pooled")
    _side_stats(mfe_oos[mfe_oos["entry_timestamp"] >= "2026-04-01"], "tonight's N=3 MFE-gated, OOS Q2 2026, 5 seeds pooled")


def section_4_common_adverse_event() -> None:
    print("\n=== 4. 2026 Q2 공통 악재 이벤트 (5시드 OOS 렛저) ===")
    files = sorted(glob.glob(str(ROOT / "tmp/eth_multislot_mfe_gated_capacity_20260813/ledger_oos_n3_mfegated_seed*.csv")))
    dfs = []
    for f in files:
        seed = f.split("seed")[-1].replace(".csv", "")
        df = pd.read_csv(f)
        df["entry_timestamp"] = pd.to_datetime(df["entry_timestamp"])
        df["exit_timestamp"] = pd.to_datetime(df["exit_timestamp"])
        df = df[df["entry_timestamp"] >= "2026-04-01"].copy()
        df["seed"] = seed
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    losers = combined[combined["trade_return"] < -0.025].sort_values("entry_timestamp")
    print(losers[["seed", "entry_timestamp", "exit_timestamp", "side", "source_component", "reason", "trade_return"]].to_string(index=False))

    px = pd.read_csv(BASE_2026, usecols=["timestamp", "open", "high", "low", "close"])
    px["timestamp"] = pd.to_datetime(px["timestamp"])
    for label, start, end in [
        ("cluster A (mid-Apr bleed -> stop ~05-16)", "2026-04-15", "2026-05-18"),
        ("cluster B (06-04/05 flash drop)", "2026-06-03", "2026-06-06"),
        ("cluster C (06-17/18 drop)", "2026-06-16", "2026-06-19"),
    ]:
        sub = px[(px["timestamp"] >= start) & (px["timestamp"] <= end)]
        c0, cmin, cend = sub["close"].iloc[0], sub["close"].min(), sub["close"].iloc[-1]
        tmin = sub.loc[sub["close"].idxmin(), "timestamp"]
        print(f"\n{label}: close {c0:.2f}({sub['timestamp'].iloc[0]}) -> min {cmin:.2f}({tmin}) -> end {cend:.2f}({sub['timestamp'].iloc[-1]}), "
              f"drawdown={100*(cmin/c0-1):.2f}%")


def section_5_rolling_walk_forward_precedent() -> None:
    print("\n=== 5 참고. 2025 Q1-Q3 rolling walk-forward (기존 산출물, Phase1 robustness audit) ===")
    import json
    with open(ROOT / "tmp/causal_regen_20260516/omega4_6_1_phase1_robustness_20260707/result.json") as f:
        d = json.load(f)
    print(json.dumps(d["rolling_walk_forward_diagnostic"], indent=2))


if __name__ == "__main__":
    regime_out = section_1_2_regime_stats()
    regime_out.to_csv(ROOT / "tmp/eth_val_oos_regime_mismatch_20260813_regime_stats.csv", index=False)
    section_3_side_asymmetry()
    section_4_common_adverse_event()
    section_5_rolling_walk_forward_precedent()
