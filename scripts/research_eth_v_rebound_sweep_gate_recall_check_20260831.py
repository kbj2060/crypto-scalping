#!/usr/bin/env python3
"""Ad-hoc diagnostic: did today's live V자반등 signal (KST 2026-08-31 00:00 -> now) call things
correctly, and separately, did any genuine V-shaped reversal happen WITHOUT a triggering
liquidity_sweep (user hypothesis: detection is entirely gated on sweep events, so non-sweep
V-shapes are structurally invisible to the current signal)?

Reuses (does not reimplement) the exact live signal's building blocks:
- add_causal_columns / compute_indicators / add_creative_indicators / add_broad_indicators
  (same functions scripts/live_eth_sweep_v_rebound_signal_20260829.py imports)
- the same sweep trigger condition (wick pierces causal 48-bar swing level, close reclaims)
- the same frozen v7b TabPFN train context

v7b outcome formula reproduced from docs/experiments/eth_liquidity_sweep_v_rebound_feature_plan_
20260829.md lines 1065-1075 (not re-derived/guessed):
  V자반등(1): within 30min (6 bars), best CLOSE reaches >=1.5x pre-sweep ATR(14) from the sweep
    extreme, AND over the full 60min (12 bars) window, giveback_ratio <= 0.20, where
    giveback_ratio = (peak - end_close) / (peak - sweep_extreme), peak = best HIGH/LOW (not
    close) reached anywhere in the 60min window.
  지지/횡보(0): 30min best-close move never even reaches 1.0x pre-sweep ATR.
  else: excluded/ambiguous (not scored here either, same as training).

2026-08-31 result (KST 00:00-06:36, 80 bars): 2 real sweep events, BOTH landed in the excluded/
ambiguous bucket (neither a clean V자반등 nor clean 지지/횡보). Meanwhile 3 non-sweep local
extremes met the strict V자반등 criteria cleanly (giveback 0.038-0.057). Small sample (one
morning) -- see docs/experiments/... writeup (memory: eth_v_rebound_sweep_gated_recall_gap_
20260831) for interpretation and the recommended longer-window follow-up before treating this as
a real base rate. TabPFN scoring for Part 1 needs TABPFN_TOKEN set (license auth) to run outside
the server -- ground-truth outcome check does not depend on the model at all.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path("/home/kbj20/crypto-scalping")
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analyze_eth_broad_evidence_signal_sweep_20260814 import add_broad_indicators  # noqa: E402
from analyze_eth_creative_reversal_evidence_signals_20260814 import add_creative_indicators  # noqa: E402
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import compute_indicators  # noqa: E402

IMPL_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"
TRAIN_CONTEXT_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/tabpfn_train_context_frozen_v7b_20260830.csv"

TIER0 = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday",
    "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile",
]
FEATURES = TIER0 + ["rsi"]

FAST_BARS = 6   # 30 min
FULL_BARS = 12  # 60 min
ATR_MULT = 1.5
CHOP_MULT = 1.0
T_SUSTAIN = 0.20

KST_MIDNIGHT_UTC = pd.Timestamp("2026-08-30 15:00:00", tz="UTC")  # 2026-08-31 00:00 KST


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_audit_20260831", IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fetch_klines(limit=1500) -> pd.DataFrame:
    resp = requests.get(
        "https://fapi.binance.com/fapi/v1/klines",
        params={"symbol": "ETHUSDT", "interval": "5m", "limit": limit}, timeout=20,
    )
    resp.raise_for_status()
    raw = resp.json()
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "qv", "trades",
            "taker_buy_base", "tq", "ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ("open", "high", "low", "close", "volume", "taker_buy_base"):
        df[c] = df[c].astype(float)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    now_ms = int(time.time() * 1000)
    if len(df) and int(df.iloc[-1]["close_time"]) >= now_ms:
        df = df.iloc[:-1].reset_index(drop=True)
    return df


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - 100 / (1 + rs)


def build_features(kl: pd.DataFrame, impl) -> pd.DataFrame:
    frame = compute_indicators(kl)
    frame = add_creative_indicators(frame)
    frame = add_broad_indicators(frame)
    ret3 = frame["close"] / frame["close"].shift(3) - 1.0
    ret3_mean = ret3.rolling(288, min_periods=288).mean()
    ret3_std = ret3.rolling(288, min_periods=288).std()
    frame["ret3_z"] = (ret3 - ret3_mean) / ret3_std.replace(0.0, np.nan)
    causal = impl.add_causal_columns(kl[["timestamp", "open", "high", "low", "close"]].copy())
    frame["sweep_level_low"] = causal["sweep_level_low"]
    frame["sweep_level_high"] = causal["sweep_level_high"]
    frame["atr"] = causal["atr"]
    frame["atr_percentile_864"] = frame["atr"].rolling(864, min_periods=864).rank(pct=True)
    frame["range_width_pct"] = (frame["sweep_level_high"] - frame["sweep_level_low"]) / frame["close"]
    frame["hour_utc"] = frame["timestamp"].dt.hour
    frame["weekday"] = frame["timestamp"].dt.weekday
    frame["rsi"] = rsi_wilder(frame["close"])
    return frame.reset_index(drop=True)


def realized_outcome(frame: pd.DataFrame, idx: int, is_down: bool) -> dict | None:
    n = len(frame)
    pre_atr = frame["atr"].iloc[idx - 1]
    if not np.isfinite(pre_atr) or pre_atr <= 0:
        return None
    extreme = frame["low"].iloc[idx] if is_down else frame["high"].iloc[idx]
    fast_slice = frame.iloc[idx + 1: idx + FAST_BARS + 1]
    full_slice = frame.iloc[idx + 1: idx + FULL_BARS + 1]
    if fast_slice.empty:
        return None
    partial = (idx + FULL_BARS) > (n - 1)

    if is_down:
        fast_move = fast_slice["close"].max() - extreme
        peak = full_slice["high"].max() if not full_slice.empty else fast_slice["high"].max()
    else:
        fast_move = extreme - fast_slice["close"].min()
        peak = full_slice["low"].min() if not full_slice.empty else fast_slice["low"].min()

    fast_mult = float(fast_move / pre_atr)
    end_price = float(full_slice["close"].iloc[-1]) if not full_slice.empty else float(fast_slice["close"].iloc[-1])
    denom = (peak - extreme) if is_down else (extreme - peak)
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        giveback = np.nan
    else:
        giveback = ((peak - end_price) / denom) if is_down else ((end_price - peak) / denom)

    if fast_mult >= ATR_MULT and np.isfinite(giveback) and giveback <= T_SUSTAIN:
        label = "V자반등"
    elif fast_mult < CHOP_MULT:
        label = "지지/횡보"
    else:
        label = "애매(제외권)"

    return {
        "fast_move_atr_mult": round(fast_mult, 3),
        "giveback_ratio": None if not np.isfinite(giveback) else round(float(giveback), 3),
        "outcome": label,
        "partial_window": bool(partial),
    }


def main() -> None:
    impl = load_impl()
    kl = fetch_klines(1500)
    frame = build_features(kl, impl)

    today = frame[frame["timestamp"] >= KST_MIDNIGHT_UTC]
    print(f"=== 대상 구간: {KST_MIDNIGHT_UTC} ~ {frame['timestamp'].iloc[-1]} UTC "
          f"(KST 2026-08-31 00:00 ~ 지금, {len(today)}개 5분봉) ===\n")

    low, high, close = frame["low"], frame["high"], frame["close"]
    level_low, level_high = frame["sweep_level_low"], frame["sweep_level_high"]
    is_down_sweep = level_low.notna() & (low < level_low) & (close > level_low)
    is_up_sweep = level_high.notna() & (high > level_high) & (close < level_high)

    sweep_idx = [i for i in today.index if bool(is_down_sweep.iloc[i]) or bool(is_up_sweep.iloc[i])]
    print(f"### Part 1: 오늘 실제 발생한 liquidity_sweep 이벤트 -- {len(sweep_idx)}건\n")

    part1_rows = []
    for idx in sweep_idx:
        is_down = bool(is_down_sweep.iloc[idx])
        outcome = realized_outcome(frame, idx, is_down)
        if outcome is None:
            continue
        row = {
            "idx": idx,
            "timestamp_utc": frame["timestamp"].iloc[idx].isoformat(),
            "direction": "downside(상승반등기대)" if is_down else "upside(하락반등기대)",
            **outcome,
        }
        part1_rows.append(row)
        status = "진행중(아직미확정)" if outcome["partial_window"] else outcome["outcome"]
        print(f"  {row['timestamp_utc']}  {row['direction']:20s}  "
              f"fast={outcome['fast_move_atr_mult']:.2f}x  giveback={outcome['giveback_ratio']}  -> {status}")

    try:
        from tabpfn import TabPFNClassifier
        train = pd.read_csv(TRAIN_CONTEXT_CSV)
        clf = TabPFNClassifier(device="cpu", random_state=20260829)
        clf.fit(train[FEATURES], train["label"].to_numpy())

        idx_list = [r["idx"] for r in part1_rows]
        if idx_list:
            feat_rows = frame.loc[idx_list, FEATURES]
            valid_mask = feat_rows.notna().all(axis=1)
            proba = clf.predict_proba(feat_rows[valid_mask])[:, 1] if valid_mask.any() else []
            proba_map = dict(zip(feat_rows[valid_mask].index, proba)) if valid_mask.any() else {}
        else:
            proba_map = {}

        print("\n### TabPFN 예측 vs 실현결과 (완결된 60분 창 + 명확한 라벨만 채점) ###")
        hit, n_scored = 0, 0
        for r in part1_rows:
            p = proba_map.get(r["idx"])
            print_line = f"  {r['timestamp_utc']}  "
            if p is None:
                print(print_line + "(피처 미완성 -- 채점 제외)")
                continue
            pred = "rebound" if p >= 0.5 else "continuation"
            if r["partial_window"]:
                print(print_line + f"proba={p:.3f} pred={pred}  (60분 창 진행중 -- 아직 정답 미확정)")
                continue
            if r["outcome"] == "애매(제외권)":
                print(print_line + f"proba={p:.3f} pred={pred}  (실현결과가 애매권이라 v7b 학습대상 자체가 아님 -- 채점 제외)")
                continue
            actual = "rebound" if r["outcome"] == "V자반등" else "continuation"
            correct = pred == actual
            hit += int(correct)
            n_scored += 1
            print(print_line + f"proba={p:.3f} pred={pred} actual={actual}  {'O 적중' if correct else 'X 오답'}")
        if n_scored:
            print(f"\n  적중률: {hit}/{n_scored} ({hit/n_scored:.1%})")
        else:
            print("\n  (채점 가능한 확정 사례 없음 -- 표본이 너무 적거나 전부 진행중/애매권)")
    except Exception as e:  # noqa: BLE001
        print(f"\nTabPFN 예측 스킵됨: {type(e).__name__}: {e}")

    print("\n\n### Part 2: 스윕 트리거 없이 발생한 V자형 반전 스캔 (재현율 갭 테스트) ###")
    print("(±30분 로컬 고점/저점을 후보로 잡고, 스윕이 없었던 지점에도 동일한 v7b 기준을 적용)\n")
    W = 6
    lo, hi = frame["low"].to_numpy(), frame["high"].to_numpy()
    n = len(frame)
    found_no_sweep = []
    for i in today.index:
        if i < W or i + W >= n:
            continue
        had_sweep = bool(is_down_sweep.iloc[i]) or bool(is_up_sweep.iloc[i])
        if had_sweep:
            continue
        window_lo, window_hi = lo[i - W:i + W + 1], hi[i - W:i + W + 1]
        is_local_low = lo[i] == window_lo.min()
        is_local_high = hi[i] == window_hi.max()
        for is_local, is_down in ((is_local_low, True), (is_local_high, False)):
            if not is_local:
                continue
            outcome = realized_outcome(frame, i, is_down)
            if outcome is None or outcome["partial_window"] or outcome["outcome"] != "V자반등":
                continue
            found_no_sweep.append({
                "timestamp_utc": frame["timestamp"].iloc[i].isoformat(),
                "type": "local_low(상승반전)" if is_down else "local_high(하락반전)",
                **outcome,
            })

    if found_no_sweep:
        for r in found_no_sweep:
            print(f"  {r['timestamp_utc']}  {r['type']:20s}  fast={r['fast_move_atr_mult']:.2f}x  "
                  f"giveback={r['giveback_ratio']}  -> V자반등 기준 충족 (스윕 없었음)")
        print(f"\n  스윕 없이 v7b 기준(1.5xATR/giveback<=0.20) 충족: {len(found_no_sweep)}건")
    else:
        print("  스윕 없이 v7b 기준을 충족한 로컬극값 반전 없음 (오늘 구간 한정, 표본 적음 주의)")


if __name__ == "__main__":
    main()
