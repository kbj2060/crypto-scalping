#!/usr/bin/env python3
"""BTC port of this project's Homer evidence-signal Tier0 feature + trigger pipeline.

User request 2026-09-01: "비트코인도 증거신호와 V자급등락신호를 모두 그리드 스크리닝과
피쳐분석 진행해줘" (grid-screen + feature-analyze BTC's own evidence signals and V자급등락
signal too). This script builds the ONE shared foundational dataset every downstream
grid-screen/feature-analysis script reads from -- bar-by-bar Tier0 23 features + rsi, plus
boolean bottom_X/top_X trigger flags, for BTCUSDT's own 5m history.

Scope decisions (stated explicitly, not silent):
  - 5 of the 6 dashboard-deployed evidence signals are ported: orthogonal_combo,
    liquidity_sweep, short_term_return_z, taker_delta_z_climax, fib_extension_exhaustion.
    ALL are pure functions of a single asset's own OHLCV+taker_buy_base(+funding) --
    reused VERBATIM (formula copy-pasted, not reimplemented) from compute_signals()
    (live_evidence_signal_dashboard_20260823.py), just fed BTC's own frame as `df`.
  - smt_divergence is EXCLUDED. Its definition is inherently cross-asset ("ETH's own swing
    without BTC confirming"); for a BTC-primary run this needs a different confirming asset,
    and docs/eth_dashboard_multicoin_expansion_design_20260831.md section 9-d already flagged
    this exact question as unresolved/needs-user-decision. Not silently defaulted here.
  - demarker_extreme / kalman_deviation_meanrev (the 2 newest Homer candidate-pool triggers,
    added 2026-08-31 to compute_signals() by a concurrent session) are also EXCLUDED -- they
    are not among the 6 dashboard evidence-signal chips the user referred to as "증거신호",
    and are still under independent validation for ETH itself.
  - local_extreme (V_REBOUND's 9th trigger, the only non-precondition one: bar is the
    highest/lowest in a +-30min/+-6bar window) IS included -- it's asset-agnostic and trivial,
    and V자급등락's candidate pool is meaningless without it (it was the single largest/
    highest-hit-rate trigger for ETH, see live_eth_sweep_v_rebound_signal_20260829.py).
  - V_REBOUND's OUTCOME/label formula (fast_move_atr_mult>=1.5x/30min AND giveback<=0.20/60min)
    is NOT computed here -- decoupled from triggers by this project's own established
    convention (see live_eth_sweep_v_rebound_signal_20260829.py's docstring: "trigger and
    label are fully decoupled axes"). Downstream V_REBOUND scripts compute it themselves from
    this file's own open/high/low/close/atr columns.

Data sources (both confirmed to have full parity with ETH's own history depth, 2026-08-31
investigation -- BTC is NOT data-starved for this task, unlike the dashboard tail-risk work):
  - binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv (277,300 rows, 2023-12-31 -> 2026-08-20)
  - binance_data/funding_rate_other/BTCUSDT-fundingRate-2024-{01..12}.zip +
    2025-{01..12} + 2026-{01..07} (31 files, matches scripts/audit_btc_funding_source_20260803.py)

Run with the quant_ai conda env (compute_indicators transitively imports torch):
  ~/anaconda3/envs/quant_ai/bin/python3 scripts/build_btc_5m_evidence_signal_candidates_tier0_20260901.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analyze_eth_broad_evidence_signal_sweep_20260814 import add_broad_indicators  # noqa: E402
from analyze_eth_creative_reversal_evidence_signals_20260814 import add_creative_indicators  # noqa: E402
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import compute_indicators  # noqa: E402
from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402

KLINES_CSV = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
FUNDING_DIR = ROOT / "binance_data/funding_rate_other"
SWEEP_IMPL_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"
OUT_DIR = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901"

START = pd.Timestamp("2024-01-01", tz="UTC")  # matches sweep_impl's own START, and BTC funding's own start
LOCAL_EXTREME_W = 6  # +-30min, matches live_eth_sweep_v_rebound_signal_20260829.py::LOCAL_EXTREME_W
FUNDING_Z_WINDOW = 90
FUNDING_Z_MIN_PERIODS = 30

# NOTE: is_downside/sweep_penetration_atr/flow_aligned_delta_z (part of the ETH Tier0 lineage's
# 23-feature list) are candidate-direction-relative -- only meaningful once a downstream script
# picks a specific candidate row + which side (bottom/top) fired. This shared bar-wide file
# instead carries the raw ingredients (atr, sweep_level_low/high, delta_z) so each downstream
# per-signal script derives them itself, exactly the way live_eth_sweep_v_rebound_signal_
# 20260829.py::_multitrigger_rows does (penetration = (level - low) or (high - level) / atr;
# flow_aligned_delta_z = delta_z if is_downside else -delta_z).
BAR_WIDE_FEATURES = [
    "atr", "atr_percentile_864", "sweep_level_low", "sweep_level_high",
    "range_width_pct", "hour_utc", "weekday",
    "delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile",
]
NAMED_TRIGGERS = ["liquidity_sweep", "taker_delta_z_climax", "short_term_return_z",
                   "orthogonal_combo", "fib_extension_exhaustion"]  # smt_divergence excluded, see module docstring


def load_sweep_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_btc_20260901", SWEEP_IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_btc_funding() -> pd.DataFrame:
    frames = []
    for p in sorted(FUNDING_DIR.glob("BTCUSDT-fundingRate-*.zip")):
        with zipfile.ZipFile(p) as z:
            name = z.namelist()[0]
            with z.open(name) as f:
                frames.append(pd.read_csv(f))
    raw = pd.concat(frames, ignore_index=True)
    raw["calc_time"] = pd.to_datetime(raw["calc_time"].astype(np.int64), unit="ms", utc=True).dt.as_unit("us")
    raw = raw.drop_duplicates("calc_time").sort_values("calc_time").reset_index(drop=True)
    mean = raw["last_funding_rate"].rolling(FUNDING_Z_WINDOW, min_periods=FUNDING_Z_MIN_PERIODS).mean()
    std = raw["last_funding_rate"].rolling(FUNDING_Z_WINDOW, min_periods=FUNDING_Z_MIN_PERIODS).std()
    raw["funding_z"] = (raw["last_funding_rate"] - mean) / std.replace(0.0, np.nan)
    return raw[["calc_time", "funding_z"]]


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - 100 / (1 + rs)


def local_extreme_flags(low: np.ndarray, high: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Verbatim port of live_eth_sweep_v_rebound_signal_20260829.py::_multitrigger_rows' local_low/
    local_high loop -- bar is the lowest/highest in its own +-w window."""
    n = len(low)
    local_low = np.zeros(n, dtype=bool)
    local_high = np.zeros(n, dtype=bool)
    for i in range(w, n - w):
        seg_lo, seg_hi = low[i - w:i + w + 1], high[i - w:i + w + 1]
        if low[i] == seg_lo.min():
            local_low[i] = True
        if high[i] == seg_hi.max():
            local_high[i] = True
    return local_low, local_high


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep_impl = load_sweep_impl()

    raw = pd.read_csv(KLINES_CSV, usecols=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = (
        raw.dropna().sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        .loc[lambda d: d["timestamp"] >= START].reset_index(drop=True)
    )
    current_bar_start = pd.Timestamp.now(tz="UTC").floor("5min")
    raw = raw.loc[raw["timestamp"] < current_bar_start].reset_index(drop=True)

    funding_df = load_btc_funding()

    frame = compute_indicators(raw)
    frame = add_creative_indicators(frame)
    frame = add_broad_indicators(frame)

    ret3 = frame["close"] / frame["close"].shift(3) - 1.0
    ret3_mean = ret3.rolling(288, min_periods=288).mean()
    ret3_std = ret3.rolling(288, min_periods=288).std()
    frame["ret3_z"] = (ret3 - ret3_mean) / ret3_std.replace(0.0, np.nan)

    causal = sweep_impl.add_causal_columns(raw[["timestamp", "open", "high", "low", "close"]].copy())
    frame["sweep_level_low"] = causal["sweep_level_low"]
    frame["sweep_level_high"] = causal["sweep_level_high"]
    frame["atr"] = causal["atr"]
    frame["atr_percentile_864"] = frame["atr"].rolling(864, min_periods=864).rank(pct=True)
    frame["range_width_pct"] = (frame["sweep_level_high"] - frame["sweep_level_low"]) / frame["close"]
    frame["hour_utc"] = frame["timestamp"].dt.hour
    frame["weekday"] = frame["timestamp"].dt.weekday
    frame["rsi"] = rsi_wilder(frame["close"])

    sig = compute_signals(raw, btc_df=None, funding_df=funding_df)  # btc_df=None: smt_divergence never fires, by design

    for name in NAMED_TRIGGERS:
        frame[f"bottom_{name}"] = sig[f"bottom_{name}"].fillna(False).to_numpy()
        frame[f"top_{name}"] = sig[f"top_{name}"].fillna(False).to_numpy()

    local_low, local_high = local_extreme_flags(frame["low"].to_numpy(), frame["high"].to_numpy(), LOCAL_EXTREME_W)
    frame["bottom_local_extreme"] = local_low
    frame["top_local_extreme"] = local_high

    delta_z = frame["delta_z"].to_numpy()
    is_down_any = np.zeros(len(frame), dtype=bool)
    is_up_any = np.zeros(len(frame), dtype=bool)
    for name in NAMED_TRIGGERS + ["local_extreme"]:
        is_down_any |= frame[f"bottom_{name}"].to_numpy()
        is_up_any |= frame[f"top_{name}"].to_numpy()
    frame["any_bottom_trigger"] = is_down_any
    frame["any_top_trigger"] = is_up_any

    frame.to_csv(OUT_DIR / "btc_5m_evidence_signal_candidates_tier0.csv", index=False)

    trigger_counts = {}
    for name in NAMED_TRIGGERS + ["local_extreme"]:
        trigger_counts[name] = {
            "bottom": int(frame[f"bottom_{name}"].sum()),
            "top": int(frame[f"top_{name}"].sum()),
        }
    report = {
        "rows": int(len(frame)),
        "start": str(frame["timestamp"].min()),
        "end": str(frame["timestamp"].max()),
        "bar_wide_features": BAR_WIDE_FEATURES + ["rsi"],
        "named_triggers_ported": NAMED_TRIGGERS,
        "excluded": ["smt_divergence (cross-asset, unresolved partner choice)",
                     "demarker_extreme (not a dashboard chip, still under independent ETH validation)",
                     "kalman_deviation_meanrev (same)"],
        "trigger_fire_counts": trigger_counts,
        "any_bottom_trigger_rows": int(is_down_any.sum()),
        "any_top_trigger_rows": int(is_up_any.sum()),
        "nan_counts_bar_wide": {k: int(v) for k, v in frame[BAR_WIDE_FEATURES + ["rsi"]].isna().sum().items() if v > 0},
        "output": str(OUT_DIR / "btc_5m_evidence_signal_candidates_tier0.csv"),
    }
    (OUT_DIR / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
