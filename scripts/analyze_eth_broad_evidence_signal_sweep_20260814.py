#!/usr/bin/env python3
"""Broad evidence-signal sweep for ETH 5m -- NOT a trading algorithm. Extends the two prior
evidence studies (oscillator confluence, creative order-flow/Wyckoff signals) with two more
categories the user asked for beyond bottom/top calling: reversal signals grounded in more
literature, and trend-CONTINUATION/breakout signals (a different decision a discretionary
trader needs evidence for -- "should I expect this move to keep going", not just "is this a
turning point").

Category A -- more reversal evidence (reuses event_study/excess_move/load_zigzag_pivots from
analyze_eth_confluence_oscillator_bottom_top_evidence_20260814.py, same methodology):
  A1. mfi_divergence      -- Money Flow Index (volume-weighted RSI) divergence. Plain-momentum
                              divergence already failed this repo's own test (lift 1.24 bottom,
                              0.88 top, see eth_creative_reversal_evidence_signals_20260814.md) --
                              this checks whether volume-weighting rescues the divergence idea.
  A2. bollinger_pctb_extreme -- %b outside the bands, a distinct math family from Stochastic/%R
                              (std-dev based, not high-low range based).
  A3. liquidity_sweep      -- ICT/"smart money concepts" stop-hunt pattern: a wick pokes through
                              a prior N-bar swing high/low and closes back inside the old range.
  A4. hurst_gated_oscillator -- the original %R+SlowK oscillator confluence, gated by a
                              mean-reverting-regime filter (variance-ratio proxy for the Hurst
                              exponent, Lo & MacKinlay 1988 / the Hurst literature: H<0.5 =
                              mean-reverting). Direct test of this sub-project's own open
                              question: "a stronger/differently-tuned trend filter than ADX>25"
                              (see [[eth_oscillator_confluence_closed_20260814]]).
  A5. btc_lead_climax      -- cross-asset: BTC's own taker-volume climax (see
                              analyze_eth_creative_reversal_evidence_signals_20260814.py) within
                              the last 6 bars, on the premise (2025-2026 crypto-market-structure
                              commentary) that BTC leads and ETH/altcoins follow with a lag.

Category B -- trend-continuation / breakout evidence (NEW methodology: race-to-target). Ground
truth for reversal signals (zigzag pivots) doesn't fit a continuation question, so this uses a
directional race instead: does price reach +/-1x ATR in the PREDICTED direction before the
opposite 1x ATR move, within K bars? Compared against the SAME race computed over all bars in
the window for that direction (accounts for this window's own trend drift -- VAL was a strong
downtrend, so "down wins the race" has a higher baseline than "up wins the race" unconditionally;
a real continuation signal must beat ITS OWN direction's baseline, not a flat 50%).
  B1. squeeze_breakout     -- Bollinger-inside-Keltner squeeze (volatility multi-bar low) followed
                              by a close outside the Keltner channel (TTM-Squeeze-style setup;
                              2025 backtests report 5-10%+ moves after 10-15 period squeezes).
  B2. adx_di_cross         -- ADX crosses above 25 while +DI/-DI confirms direction (textbook
                              "trend just starting" signal; literature flags it as lagging/no
                              guarantee -- tested here rather than assumed).
  B3. donchian_breakout    -- close makes a new 96-bar high/low (classic turtle-style breakout).
  B4. btc_momentum_spillover -- BTC's own trailing 1h return in its own top/bottom decile,
                              predicting ETH's next-K-bar direction (cross-asset momentum lead).
  B5. donchian_trend_filtered -- B3 gated by the same mean-reverting/trending regime proxy as A4
                              (trending regime only) -- does filtering FOR trend improve a
                              continuation signal the way filtering FOR mean-reversion might help
                              a reversal signal?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analyze_eth_confluence_oscillator_bottom_top_evidence_20260814 import (  # noqa: E402
    K_HORIZONS,
    OOS_END,
    event_study,
    excess_move,
    load_zigzag_pivots,
)
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import (  # noqa: E402
    OOS_START,
    VAL_END,
    VAL_START,
    compute_indicators,
)

ETH_PATH = ROOT / "data" / "eth_5m_1year.csv"
BTC_PATH = ROOT / "data" / "btc_5m_1year.csv"
RAW_COLS = ["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"]


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=RAW_COLS, parse_dates=["timestamp"])
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _dmi(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    eps = 1e-12
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    up, down = high.diff(), -low.diff()
    pdm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    ndm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    pdi = 100.0 * pdm.ewm(span=period, adjust=False).mean() / (atr + eps)
    ndi = 100.0 * ndm.ewm(span=period, adjust=False).mean() / (atr + eps)
    dx = 100.0 * (pdi - ndi).abs() / (pdi + ndi + eps)
    return pdi, ndi, dx.ewm(span=period, adjust=False).mean()


def add_broad_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-12
    high, low, close, volume = frame["high"], frame["low"], frame["close"], frame["volume"]

    # A1: MFI divergence
    typical = (high + low + close) / 3.0
    raw_flow = typical * volume
    tp_diff = typical.diff()
    pos_flow = raw_flow.where(tp_diff > 0, 0.0).rolling(14, min_periods=14).sum()
    neg_flow = raw_flow.where(tp_diff < 0, 0.0).rolling(14, min_periods=14).sum()
    mfi = 100.0 - 100.0 / (1.0 + pos_flow / (neg_flow + eps))
    frame["mfi_roc_48"] = mfi - mfi.shift(48)
    frame["price_roc_48"] = close / close.shift(48) - 1.0

    # A2: Bollinger %b
    bb_mid = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std()
    frame["bb_pctb"] = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std + eps)
    bb_width = (4 * bb_std) / (bb_mid + eps)
    frame["bb_width_pctile"] = bb_width.rolling(864, min_periods=864).rank(pct=True)

    # B1: Keltner channel (needs atr_price from compute_indicators, already present)
    kc_mid = close.ewm(span=20, adjust=False).mean()
    frame["kc_upper"] = kc_mid + 1.5 * frame["atr_price"]
    frame["kc_lower"] = kc_mid - 1.5 * frame["atr_price"]
    frame["squeeze_on_prev"] = (frame["bb_width_pctile"] <= 0.10).shift(1).fillna(False)

    # A3: liquidity sweep / stop hunt
    swing_low_prior = low.rolling(48, min_periods=48).min().shift(1)
    swing_high_prior = high.rolling(48, min_periods=48).max().shift(1)
    frame["sweep_low"] = (low < swing_low_prior) & (close > swing_low_prior)
    frame["sweep_high"] = (high > swing_high_prior) & (close < swing_high_prior)

    # B2: DMI/ADX (already have frame["adx14"] from compute_indicators; also need +DI/-DI)
    pdi, ndi, _ = _dmi(high, low, close, 14)
    frame["pdi"], frame["ndi"] = pdi, ndi

    # B3: Donchian breakout
    frame["donch_high"] = high.rolling(96, min_periods=96).max().shift(1)
    frame["donch_low"] = low.rolling(96, min_periods=96).min().shift(1)

    # A4/B5: variance-ratio regime proxy (Lo & MacKinlay 1988; Hurst-family, simpler to compute)
    log_ret = np.log(close / close.shift(1))
    q = 12
    var_1 = log_ret.rolling(288, min_periods=288).var()
    var_q = log_ret.rolling(q).sum().rolling(288, min_periods=288).var()
    vr = var_q / (q * var_1 + eps)
    frame["trending_regime"] = vr > 1.1
    frame["mean_reverting_regime"] = vr < 0.9
    return frame


def add_btc_leadlag(eth_frame: pd.DataFrame) -> pd.DataFrame:
    btc = load_raw(BTC_PATH)
    delta = 2.0 * btc["taker_buy_base"] - btc["volume"]
    delta_z = (delta - delta.rolling(288, min_periods=288).mean()) / delta.rolling(288, min_periods=288).std().replace(0.0, np.nan)
    sell_climax = (delta_z <= -2.0).rolling(6, min_periods=1).max().astype(bool)
    buy_climax = (delta_z >= 2.0).rolling(6, min_periods=1).max().astype(bool)

    ret_12 = btc["close"] / btc["close"].shift(12) - 1.0
    hi_q = ret_12.rolling(864, min_periods=864).quantile(0.90)
    lo_q = ret_12.rolling(864, min_periods=864).quantile(0.10)

    btc_feats = pd.DataFrame(
        {
            "timestamp": btc["timestamp"],
            "btc_sell_climax_lead": sell_climax,
            "btc_buy_climax_lead": buy_climax,
            "btc_spillover_up": ret_12 >= hi_q,
            "btc_spillover_down": ret_12 <= lo_q,
        }
    )
    merged = eth_frame.merge(btc_feats, on="timestamp", how="left")
    for c in ("btc_sell_climax_lead", "btc_buy_climax_lead", "btc_spillover_up", "btc_spillover_down"):
        merged[c] = merged[c].fillna(False)
    return merged


# ---------------------------------------------------------------------------
# Category A: reversal (reuse event_study / excess_move)
# ---------------------------------------------------------------------------
def reversal_signals(frame: pd.DataFrame, side: str) -> dict:
    if side == "bottom":
        return {
            "A1_mfi_divergence": (frame["price_roc_48"] <= -0.01) & (frame["mfi_roc_48"] >= 5),
            "A2_bollinger_pctb_extreme_low": frame["bb_pctb"] <= 0.05,
            "A3_liquidity_sweep_low": frame["sweep_low"],
            "A4_hurst_gated_oscillator_low": (frame["p_fast"] <= 0.10) & (frame["p_slow"] <= 0.10) & frame["mean_reverting_regime"],
            "A5_btc_lead_sell_climax": frame["btc_sell_climax_lead"],
        }
    return {
        "A1_mfi_divergence": (frame["price_roc_48"] >= 0.01) & (frame["mfi_roc_48"] <= -5),
        "A2_bollinger_pctb_extreme_high": frame["bb_pctb"] >= 0.95,
        "A3_liquidity_sweep_high": frame["sweep_high"],
        "A4_hurst_gated_oscillator_high": (frame["p_fast"] >= 0.90) & (frame["p_slow"] >= 0.90) & frame["mean_reverting_regime"],
        "A5_btc_lead_buy_climax": frame["btc_buy_climax_lead"],
    }


def run_reversal(frame: pd.DataFrame, window_mask: np.ndarray, pivots: pd.DataFrame, side: str) -> pd.DataFrame:
    close = frame["close"].to_numpy()
    all_pos = np.flatnonzero(window_mask)
    pivot_pos = frame.index[frame["timestamp"].isin(pivots.loc[pivots["pivot_type"] == side, "timestamp"])].to_numpy()
    rows = []
    for name, mask in reversal_signals(frame, side).items():
        trigger_pos = np.flatnonzero(mask.fillna(False).to_numpy() & window_mask)
        for k_name, K in K_HORIZONS.items():
            stats = event_study(trigger_pos, pivot_pos, all_pos, K)
            move = excess_move(trigger_pos, pivot_pos, close, K)
            rows.append({"category": "A_reversal", "side": side, "signal": name, "horizon": k_name, **stats,
                         "excess_move_mean_pct": move["mean_pct"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Category B: continuation / breakout (race-to-target vs directional baseline)
# ---------------------------------------------------------------------------
def race_outcomes(positions: np.ndarray, direction: int, close: np.ndarray, high: np.ndarray,
                   low: np.ndarray, atr_pct: np.ndarray, K: int, mult: float = 1.0) -> np.ndarray:
    n = len(close)
    out = np.zeros(len(positions), dtype=np.int64)
    for i, pos in enumerate(positions):
        atr = atr_pct[pos]
        if not np.isfinite(atr) or atr <= 0 or pos + 1 >= n:
            continue
        entry = close[pos]
        target = entry * (1.0 + direction * mult * atr)
        stop = entry * (1.0 - direction * mult * atr)
        end = min(pos + K, n - 1)
        seg_high, seg_low = high[pos + 1: end + 1], low[pos + 1: end + 1]
        if len(seg_high) == 0:
            continue
        if direction > 0:
            t_hits, s_hits = np.flatnonzero(seg_high >= target), np.flatnonzero(seg_low <= stop)
        else:
            t_hits, s_hits = np.flatnonzero(seg_low <= target), np.flatnonzero(seg_high >= stop)
        t_first = t_hits[0] if len(t_hits) else np.inf
        s_first = s_hits[0] if len(s_hits) else np.inf
        if t_first == np.inf and s_first == np.inf:
            out[i] = 0
        elif t_first <= s_first:
            out[i] = 1
        else:
            out[i] = -1
    return out


def continuation_signals(frame: pd.DataFrame, direction: str) -> dict:
    if direction == "up":
        return {
            "B1_squeeze_breakout_up": frame["squeeze_on_prev"] & (frame["close"] > frame["kc_upper"]),
            "B2_adx_di_cross_up": (frame["adx14"] >= 25) & (frame["adx14"].shift(1) < 25) & (frame["pdi"] > frame["ndi"]),
            "B3_donchian_breakout_up": frame["close"] > frame["donch_high"],
            "B4_btc_momentum_spillover_up": frame["btc_spillover_up"],
            "B5_donchian_trend_filtered_up": (frame["close"] > frame["donch_high"]) & frame["trending_regime"],
        }
    return {
        "B1_squeeze_breakout_down": frame["squeeze_on_prev"] & (frame["close"] < frame["kc_lower"]),
        "B2_adx_di_cross_down": (frame["adx14"] >= 25) & (frame["adx14"].shift(1) < 25) & (frame["ndi"] > frame["pdi"]),
        "B3_donchian_breakout_down": frame["close"] < frame["donch_low"],
        "B4_btc_momentum_spillover_down": frame["btc_spillover_down"],
        "B5_donchian_trend_filtered_down": (frame["close"] < frame["donch_low"]) & frame["trending_regime"],
    }


def run_continuation(frame: pd.DataFrame, window_mask: np.ndarray) -> pd.DataFrame:
    close, high, low = frame["close"].to_numpy(), frame["high"].to_numpy(), frame["low"].to_numpy()
    atr_pct = frame["atr_pct"].to_numpy()
    all_pos = np.flatnonzero(window_mask)

    baseline_cache: dict[tuple[int, int], np.ndarray] = {}

    def baseline_rate(direction: int, K: int) -> float:
        key = (direction, K)
        if key not in baseline_cache:
            baseline_cache[key] = race_outcomes(all_pos, direction, close, high, low, atr_pct, K)
        outcomes = baseline_cache[key]
        decided = outcomes != 0
        return float((outcomes[decided] == 1).mean()) if decided.any() else float("nan")

    rows = []
    for dir_name, d in (("up", 1), ("down", -1)):
        for name, mask in continuation_signals(frame, dir_name).items():
            trigger_pos = np.flatnonzero(mask.fillna(False).to_numpy() & window_mask)
            for k_name, K in K_HORIZONS.items():
                outcomes = race_outcomes(trigger_pos, d, close, high, low, atr_pct, K)
                decided = outcomes != 0
                hit_rate = float((outcomes[decided] == 1).mean()) if decided.any() else float("nan")
                base = baseline_rate(d, K)
                lift = hit_rate / base if base and np.isfinite(base) and base > 0 else float("nan")
                rows.append({
                    "category": "B_continuation", "side": dir_name, "signal": name, "horizon": k_name,
                    "n_triggers": int(len(trigger_pos)), "n_decided": int(decided.sum()),
                    "precision": hit_rate, "baseline_rate": base, "lift": lift,
                    "timeout_rate": float((outcomes == 0).mean()) if len(outcomes) else float("nan"),
                })
    return pd.DataFrame(rows)


def main() -> None:
    eth_raw = load_raw(ETH_PATH)
    frame = compute_indicators(eth_raw).reset_index(drop=True)
    frame = add_broad_indicators(frame)
    frame = add_btc_leadlag(frame)
    pivots = load_zigzag_pivots()

    ts = frame["timestamp"]
    window_mask = (((ts >= VAL_START) & (ts <= VAL_END)) | ((ts >= OOS_START) & (ts <= OOS_END))).to_numpy()
    print(f"Study window: VAL {VAL_START.date()}..{VAL_END.date()} + OOS {OOS_START.date()}..{OOS_END.date()}, "
          f"{int(window_mask.sum())} bars, {len(pivots)} zigzag pivots")

    reversal_rows = pd.concat([run_reversal(frame, window_mask, pivots, s) for s in ("bottom", "top")], ignore_index=True)
    continuation_rows = run_continuation(frame, window_mask)

    pd.set_option("display.width", 170)
    print("\n\n########## CATEGORY A: MORE REVERSAL EVIDENCE ##########")
    for side in ("bottom", "top"):
        print(f"\n=== {side.upper()} ===")
        sub = reversal_rows[reversal_rows["side"] == side]
        for horizon in K_HORIZONS:
            print(f"\n-- {horizon} --")
            cols = ["signal", "n_triggers", "precision", "baseline_rate", "lift", "recall", "excess_move_mean_pct"]
            print(sub[sub["horizon"] == horizon][cols].to_string(index=False))

    print("\n\n########## CATEGORY B: CONTINUATION / BREAKOUT EVIDENCE ##########")
    for side in ("up", "down"):
        print(f"\n=== {side.upper()} ===")
        sub = continuation_rows[continuation_rows["side"] == side]
        for horizon in K_HORIZONS:
            print(f"\n-- {horizon} --")
            cols = ["signal", "n_triggers", "n_decided", "precision", "baseline_rate", "lift", "timeout_rate"]
            print(sub[sub["horizon"] == horizon][cols].to_string(index=False))

    out_dir = ROOT / "tmp" / "eth_broad_evidence_signal_sweep_20260814"
    out_dir.mkdir(parents=True, exist_ok=True)
    reversal_rows.to_csv(out_dir / "reversal_evidence_table.csv", index=False)
    continuation_rows.to_csv(out_dir / "continuation_evidence_table.csv", index=False)
    print(f"\nWrote tables to {out_dir}")


if __name__ == "__main__":
    main()
