#!/usr/bin/env python3
"""추세추종 기술적 셋업 10종(S급 4 + A급 6) raw rule 리프트 사전점검.
① MTF Trend+EMA Pullback ② VWAP Pullback+Price Structure ③ Breakout+Volume+OI/CVD
④ BOS+Retest+Volume/CVD ⑤ EMA Ribbon ⑥ Donchian Breakout ⑦ ADX+DMI+EMA
⑧ VWAP+Volume Profile ⑨ Bollinger Squeeze Breakout ⑩ Supertrend+EMA

딥러닝/라벨설계 이전 단계 — 규칙 로직 자체의 raw lift만 확인한다(Homer 후보풀의
"사전점검" 단계와 동급). 이 신호들은 방향성(롱/숏) 추세추종 베팅이라 zigzag pivot
방식(candidate_pool_raw_lift_check)이 아니라 breakout_continuation/trend_continuation
계열의 방향성 방법론을 따른다:

- 베이스라인 비-동어반복 원칙(research_eth_breakout_continuation_giveback_check_20260831.py):
  트리거 유무와 무관하게 "같은 방향 공식"을 윈도우 내 모든 봉에 적용한 것이 베이스라인.
  트리거 쪽과 베이스라인 쪽이 다른 규칙을 쓰면 안 된다(breakout v1의 20배 버그 원인).
- ATR 자기포함 버그 회피: 정방향 이동을 정규화할 때는 atr14_prior(=atr14.shift(1),
  트리거 봉 자신의 true range를 포함하지 않음)만 사용한다.
- 클러스터 디듑: first_bar_of_each_run으로 연속 발동봉을 독립 이벤트 1개로 축소.

데이터: binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv(OHLCV+taker_buy_base,
2023-12-31~현재, gap-free) + data/TOTAL_ETHUSDT_metrics_2024_2026.csv(실제 OI,
2024-01~현재 커버, merge_asof backward로 결합) — 5m klines 이외의 소스가 필요한
"진짜" 후보(예: microstructure_1m)는 이번 10종에 없음.

VAL 2025-09-01~2025-12-31, OOS 2026-01-01~2026-03-31(CLAUDE.md Fresh-Forward 기본
윈도우 그대로 — 이 스크립트는 최신 klines를 쓰므로 후보풀 스크립트처럼 데이터 부족으로
잘릴 필요가 없음). 이 단계는 벡터화된 forward-return 사전점검이며, bar-by-bar 인과적
워크포워드 시뮬레이션이 아니다 — Fresh-Forward 규칙이 promotion에 요구하는 그 검증이
아니라 "research/dev score" 등급의 사전 스크리닝이다(CLAUDE.md 문구 그대로).

자유선택 파라미터(다음 단계 재검증 없이 승계 금지): HIT_THRESHOLD_ATR=0.5,
RETEST_WINDOW=12/RETEST_ATR_BAND=0.25(④), VOL_Z_THRESHOLD=1.0, POC 재계산 간격=12봉·
윈도우=288봉(⑧), SUPERTREND_ATR_WINDOW=10/MULT=3.0(업계 표준값, 이 저장소 최초 사용).
이미 저장소 컨벤션인 값(Donchian 96, swing 48, Bollinger 20/864, Keltner 20/1.5,
ADX 임계 25)은 그대로 재사용.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KLINES = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
METRICS = ROOT / "data/TOTAL_ETHUSDT_metrics_2024_2026.csv"
OUT_DIR = ROOT / "tmp/eth_trend_signal_raw_lift_check_20260831"

VAL_START, VAL_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
OOS_START, OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31 23:59:59", tz="UTC")

HORIZONS = {"H12_1h": 12, "H48_4h": 48, "H96_8h": 96}
HIT_THRESHOLD_ATR = 0.5
ADX_THRESHOLD = 25.0
VOL_Z_THRESHOLD = 1.0
RETEST_WINDOW = 12
RETEST_ATR_BAND = 0.25
SUPERTREND_ATR_WINDOW = 10
SUPERTREND_MULT = 3.0
Z_95 = 1.959963984540054


def load_cvp():
    spec = importlib.util.spec_from_file_location("cvp_standalone_trendsig", ROOT / "core" / "cvp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wilson_ci(hits: int, n: int, z: float = Z_95) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n)))
    return (max(0.0, center - half), min(1.0, center + half))


def first_bar_of_each_run(idx: np.ndarray) -> np.ndarray:
    if len(idx) == 0:
        return idx
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], breaks + 1))
    return idx[starts]


# ════════════════════════════════════════════════════════════════
# 데이터 로딩
# ════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    kl = pd.read_csv(KLINES, usecols=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"])
    kl["timestamp"] = pd.to_datetime(kl["timestamp"], utc=True)
    kl = kl.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    mt = pd.read_csv(METRICS, usecols=["create_time", "sum_open_interest"])
    mt["timestamp"] = pd.to_datetime(mt["create_time"], utc=True)
    mt = mt.sort_values("timestamp").drop_duplicates("timestamp")[["timestamp", "sum_open_interest"]]
    mt = mt.rename(columns={"sum_open_interest": "oi"})

    df = pd.merge_asof(kl, mt, on="timestamp", direction="backward")
    return df.reset_index(drop=True)


# ════════════════════════════════════════════════════════════════
# 지표(전부 인과적 — 현재 봉 종가 시점까지 확정된 정보만 사용)
# ════════════════════════════════════════════════════════════════
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


def _supertrend_dir(df: pd.DataFrame, atr: pd.Series, atr_window: int, mult: float) -> pd.Series:
    hl2 = (df["high"] + df["low"]) / 2.0
    ub = (hl2 + mult * atr).to_numpy()
    lb = (hl2 - mult * atr).to_numpy()
    close = df["close"].to_numpy()
    atr_np = atr.to_numpy()
    n = len(df)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    direction = np.ones(n, dtype=int)
    for i in range(n):
        if not np.isfinite(atr_np[i]):
            continue  # stays NaN/direction=1 placeholder through ATR warmup
        if i == 0 or not np.isfinite(final_upper[i - 1]):
            # first finite-ATR bar (or first bar overall): nothing valid to ratchet from yet
            final_upper[i], final_lower[i], direction[i] = ub[i], lb[i], 1
            continue
        final_upper[i] = ub[i] if (ub[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]) else final_upper[i - 1]
        final_lower[i] = lb[i] if (lb[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]) else final_lower[i - 1]
        if direction[i - 1] == 1:
            direction[i] = -1 if close[i] < final_lower[i] else 1
        else:
            direction[i] = 1 if close[i] > final_upper[i] else -1
    return pd.Series(direction, index=df.index)


def _rolling_poc(df: pd.DataFrame, cvp, window: int = 288, step: int = 12) -> pd.Series:
    prices = ((df["high"] + df["low"] + df["close"]) / 3.0).to_numpy()
    vols = df["volume"].to_numpy()
    n = len(df)
    poc_vals = np.full(n, np.nan)
    for i in range(window, n, step):
        _, _, poc, _, _ = cvp._compute_volume_profile(prices[i - window:i], vols[i - window:i], n_bins=50)
        poc_vals[i:min(i + step, n)] = poc
    return pd.Series(poc_vals, index=df.index)


def add_indicators(df: pd.DataFrame, cvp) -> pd.DataFrame:
    high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]

    for span in (8, 13, 21, 34, 55, 200):
        df[f"ema{span}"] = close.ewm(span=span, adjust=False).mean()

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14, min_periods=14).mean()
    df["atr14_prior"] = df["atr14"].shift(1)

    df["pdi14"], df["ndi14"], df["adx14"] = _dmi(high, low, close, 14)

    df["donch_high96"] = high.rolling(96, min_periods=96).max().shift(1)
    df["donch_low96"] = low.rolling(96, min_periods=96).min().shift(1)
    df["swing_high48"] = high.rolling(48, min_periods=48).max().shift(1)
    df["swing_low48"] = low.rolling(48, min_periods=48).min().shift(1)

    bb_mid = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std()
    df["bb_upper20"] = bb_mid + 2 * bb_std
    df["bb_lower20"] = bb_mid - 2 * bb_std
    bb_width = (4 * bb_std) / (bb_mid + 1e-12)
    df["bb_width_pctile"] = bb_width.rolling(864, min_periods=864).rank(pct=True)
    df["squeeze_on_prev"] = (df["bb_width_pctile"] <= 0.10).shift(1).fillna(False)
    kc_mid = close.ewm(span=20, adjust=False).mean()
    df["kc_upper"] = kc_mid + 1.5 * df["atr14"]
    df["kc_lower"] = kc_mid - 1.5 * df["atr14"]

    typical = (high + low + close) / 3.0
    df["vwap288"] = (typical * volume).rolling(288, min_periods=288).sum() / volume.rolling(288, min_periods=288).sum()

    taker_delta = 2.0 * df["taker_buy_base"] - volume
    df["cvd_roll288"] = taker_delta.rolling(288, min_periods=288).sum()
    df["cvd_roll_roc48"] = df["cvd_roll288"] - df["cvd_roll288"].shift(48)
    df["volume_z288"] = (volume - volume.rolling(288, min_periods=288).mean()) / volume.rolling(288, min_periods=288).std()
    df["oi_roc48"] = df["oi"].pct_change(48)

    df["supertrend_dir"] = _supertrend_dir(df, df["atr14"], SUPERTREND_ATR_WINDOW, SUPERTREND_MULT)

    # 1h HTF 트렌드 — label="right" 즉 각 1h 버킷을 "그 버킷이 닫힌 실제 시각"으로 라벨링한 뒤
    # merge_asof(backward)로 결합 → 5m 봉 t에는 t 이전에 이미 닫힌 1h 봉의 EMA만 보임(룩어헤드 없음).
    h1_close = df.set_index("timestamp")["close"].resample("1h", label="right", closed="left").last().dropna()
    h1_ema50 = h1_close.ewm(span=50, adjust=False).mean()
    h1_ema200 = h1_close.ewm(span=200, adjust=False).mean()
    htf = pd.DataFrame({"timestamp": h1_close.index, "htf_ema50": h1_ema50.to_numpy(), "htf_ema200": h1_ema200.to_numpy()})
    df = pd.merge_asof(df.sort_values("timestamp"), htf.sort_values("timestamp"), on="timestamp", direction="backward")

    # 롤링 POC(1일=288봉, 12봉=1h 간격 재계산, 재계산 시점 이전 데이터만 사용 → 인과적)
    df["poc288"] = _rolling_poc(df, cvp, window=288, step=12)
    df["poc_prev12"] = df["poc288"].shift(12)

    return df


# ════════════════════════════════════════════════════════════════
# 10개 신호 정의 — 각각 (trigger_long, trigger_short) bool Series 반환
# ════════════════════════════════════════════════════════════════
def sig_mtf_trend_ema_pullback(df: pd.DataFrame):
    htf_up = df["htf_ema50"] > df["htf_ema200"]
    htf_down = df["htf_ema50"] < df["htf_ema200"]
    close, low, high, ema21, ema55 = df["close"], df["low"], df["high"], df["ema21"], df["ema55"]
    pullback_up = (low <= ema21) & (close > ema21) & (close.shift(1) > ema21.shift(1)) & (close > ema55)
    pullback_down = (high >= ema21) & (close < ema21) & (close.shift(1) < ema21.shift(1)) & (close < ema55)
    return (htf_up & pullback_up).fillna(False), (htf_down & pullback_down).fillna(False)


def sig_vwap_pullback_structure(df: pd.DataFrame):
    close, low, high, vwap = df["close"], df["low"], df["high"], df["vwap288"]
    touch_up = (low <= vwap) & (close > vwap) & (close.shift(1) > vwap.shift(1))
    touch_down = (high >= vwap) & (close < vwap) & (close.shift(1) < vwap.shift(1))
    roll_low = low.rolling(12, min_periods=12).min()
    roll_high = high.rolling(12, min_periods=12).max()
    higher_low = roll_low > roll_low.shift(12)
    lower_high = roll_high < roll_high.shift(12)
    return (touch_up & higher_low).fillna(False), (touch_down & lower_high).fillna(False)


def sig_breakout_volume_oicvd(df: pd.DataFrame):
    close = df["close"]
    level_high = df["high"].rolling(48, min_periods=48).max().shift(1)
    level_low = df["low"].rolling(48, min_periods=48).min().shift(1)
    volc = df["volume_z288"] > VOL_Z_THRESHOLD
    flow_up = (df["cvd_roll_roc48"] > 0) & (df["oi_roc48"] > 0)
    flow_down = (df["cvd_roll_roc48"] < 0) & (df["oi_roc48"] > 0)
    trig_long = (close > level_high) & volc & flow_up
    trig_short = (close < level_low) & volc & flow_down
    return trig_long.fillna(False), trig_short.fillna(False)


def sig_bos_retest_volume_cvd(df: pd.DataFrame):
    n = len(df)
    close = df["close"].to_numpy()
    low = df["low"].to_numpy()
    high = df["high"].to_numpy()
    swing_high = df["swing_high48"].to_numpy()
    swing_low = df["swing_low48"].to_numpy()
    atr_prior = df["atr14_prior"].to_numpy()
    volz = df["volume_z288"].to_numpy()
    cvdroc = df["cvd_roll_roc48"].to_numpy()

    trig_long = np.zeros(n, dtype=bool)
    trig_short = np.zeros(n, dtype=bool)

    break_up = np.isfinite(swing_high) & (close > swing_high) & (volz > VOL_Z_THRESHOLD) & (cvdroc > 0)
    break_down = np.isfinite(swing_low) & (close < swing_low) & (volz > VOL_Z_THRESHOLD) & (cvdroc < 0)

    for i in np.flatnonzero(break_up):
        level = swing_high[i]
        band = RETEST_ATR_BAND * atr_prior[i] if np.isfinite(atr_prior[i]) else np.nan
        if not np.isfinite(band):
            continue
        for j in range(i + 1, min(i + 1 + RETEST_WINDOW, n)):
            if close[j] < level:
                break
            if low[j] <= level + band:
                trig_long[j] = True
                break

    for i in np.flatnonzero(break_down):
        level = swing_low[i]
        band = RETEST_ATR_BAND * atr_prior[i] if np.isfinite(atr_prior[i]) else np.nan
        if not np.isfinite(band):
            continue
        for j in range(i + 1, min(i + 1 + RETEST_WINDOW, n)):
            if close[j] > level:
                break
            if high[j] >= level - band:
                trig_short[j] = True
                break

    return pd.Series(trig_long, index=df.index), pd.Series(trig_short, index=df.index)


def sig_ema_ribbon(df: pd.DataFrame):
    e8, e13, e21, e34, e55 = df["ema8"], df["ema13"], df["ema21"], df["ema34"], df["ema55"]
    stacked_up = (e8 > e13) & (e13 > e21) & (e21 > e34) & (e34 > e55)
    stacked_down = (e8 < e13) & (e13 < e21) & (e21 < e34) & (e34 < e55)
    trig_long = stacked_up & ~stacked_up.shift(1).fillna(False)
    trig_short = stacked_down & ~stacked_down.shift(1).fillna(False)
    return trig_long.fillna(False), trig_short.fillna(False)


def sig_donchian_breakout(df: pd.DataFrame):
    trig_long = df["close"] > df["donch_high96"]
    trig_short = df["close"] < df["donch_low96"]
    return trig_long.fillna(False), trig_short.fillna(False)


def sig_adx_dmi_ema(df: pd.DataFrame):
    strong = df["adx14"] > ADX_THRESHOLD
    strong_new = strong & ~strong.shift(1).fillna(False)
    trig_long = strong_new & (df["pdi14"] > df["ndi14"]) & (df["close"] > df["ema55"])
    trig_short = strong_new & (df["ndi14"] > df["pdi14"]) & (df["close"] < df["ema55"])
    return trig_long.fillna(False), trig_short.fillna(False)


def sig_vwap_volume_profile(df: pd.DataFrame):
    close, vwap = df["close"], df["vwap288"]
    cross_up = (close.shift(1) <= vwap.shift(1)) & (close > vwap)
    cross_down = (close.shift(1) >= vwap.shift(1)) & (close < vwap)
    poc_up = df["poc288"] > df["poc_prev12"]
    poc_down = df["poc288"] < df["poc_prev12"]
    return (cross_up & poc_up).fillna(False), (cross_down & poc_down).fillna(False)


def sig_bollinger_squeeze_breakout(df: pd.DataFrame):
    trig_long = df["squeeze_on_prev"] & (df["close"] > df["kc_upper"])
    trig_short = df["squeeze_on_prev"] & (df["close"] < df["kc_lower"])
    return trig_long.fillna(False), trig_short.fillna(False)


def sig_supertrend_ema(df: pd.DataFrame):
    st = df["supertrend_dir"]
    flip_up = (st == 1) & (st.shift(1) == -1)
    flip_down = (st == -1) & (st.shift(1) == 1)
    trig_long = flip_up & (df["close"] > df["ema200"])
    trig_short = flip_down & (df["close"] < df["ema200"])
    return trig_long.fillna(False), trig_short.fillna(False)


SIGNALS = {
    "01_mtf_trend_ema_pullback": sig_mtf_trend_ema_pullback,
    "02_vwap_pullback_structure": sig_vwap_pullback_structure,
    "03_breakout_volume_oicvd": sig_breakout_volume_oicvd,
    "04_bos_retest_volume_cvd": sig_bos_retest_volume_cvd,
    "05_ema_ribbon": sig_ema_ribbon,
    "06_donchian_breakout": sig_donchian_breakout,
    "07_adx_dmi_ema": sig_adx_dmi_ema,
    "08_vwap_volume_profile": sig_vwap_volume_profile,
    "09_bollinger_squeeze_breakout": sig_bollinger_squeeze_breakout,
    "10_supertrend_ema": sig_supertrend_ema,
}


# ════════════════════════════════════════════════════════════════
# 리프트 체크 하네스
# ════════════════════════════════════════════════════════════════
def main() -> None:
    print("[1/5] cvp.py 로딩(core/__init__ binance 의존성 우회)...")
    cvp = load_cvp()

    print("[2/5] 데이터 로딩+OI 병합...")
    df = load_data()
    print(f"  bars={len(df)}, {df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}")

    print("[3/5] 지표 계산(supertrend/POC는 순차 루프라 다소 소요)...")
    df = add_indicators(df, cvp)

    close = df["close"]
    windows = {
        "VAL": ((df["timestamp"] >= VAL_START) & (df["timestamp"] <= VAL_END)).to_numpy(),
        "OOS": ((df["timestamp"] >= OOS_START) & (df["timestamp"] <= OOS_END)).to_numpy(),
    }

    print("[4/5] 방향별/호라이즌별 forward-move 배열 + 비-동어반복 베이스라인 계산...")
    move_atr = {}
    move_bp = {}
    for side_sign in (1, -1):
        for hkey, H in HORIZONS.items():
            fwd = close.shift(-H)
            move_atr[(side_sign, H)] = (side_sign * (fwd - close) / df["atr14_prior"]).to_numpy()
            move_bp[(side_sign, H)] = (side_sign * (fwd - close) / close * 10000.0).to_numpy()

    baseline_stats = {}
    for window_name, mask in windows.items():
        idx_all = np.flatnonzero(mask)
        for side_sign in (1, -1):
            for hkey, H in HORIZONS.items():
                m = move_atr[(side_sign, H)][idx_all]
                valid = np.isfinite(m)
                mv = m[valid]
                hits = mv >= HIT_THRESHOLD_ATR
                baseline_stats[(window_name, side_sign, hkey)] = dict(
                    n=int(valid.sum()),
                    rate=float(hits.mean()) if valid.sum() else float("nan"),
                    mean=float(mv.mean()) if valid.sum() else float("nan"),
                    median=float(np.median(mv)) if valid.sum() else float("nan"),
                )

    print("[5/5] 신호별 리프트 계산...")
    rows = []
    for sig_name, fn in SIGNALS.items():
        trig_long, trig_short = fn(df)
        for side_label, trig_series, side_sign in (("long", trig_long, 1), ("short", trig_short, -1)):
            trig_np = trig_series.to_numpy()
            for window_name, mask in windows.items():
                idx = first_bar_of_each_run(np.flatnonzero(trig_np & mask))
                for hkey, H in HORIZONS.items():
                    m = move_atr[(side_sign, H)][idx]
                    bp = move_bp[(side_sign, H)][idx]
                    valid = np.isfinite(m)
                    n = int(valid.sum())
                    b = baseline_stats[(window_name, side_sign, hkey)]
                    if n == 0:
                        rows.append(dict(signal=sig_name, side=side_label, window=window_name, horizon=hkey,
                                          n_triggers=0, hit_rate=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                                          baseline_rate=b["rate"], lift=float("nan"),
                                          mean_atr_move=float("nan"), median_atr_move=float("nan"),
                                          baseline_mean_atr_move=b["mean"], mean_bp_move=float("nan"), low_n=True))
                        continue
                    mv = m[valid]
                    hits = mv >= HIT_THRESHOLD_ATR
                    rate = float(hits.mean())
                    ci_lo, ci_hi = wilson_ci(int(hits.sum()), n)
                    lift = rate / b["rate"] if b["rate"] and b["rate"] > 0 else float("nan")
                    rows.append(dict(
                        signal=sig_name, side=side_label, window=window_name, horizon=hkey,
                        n_triggers=n, hit_rate=rate, ci_lo=ci_lo, ci_hi=ci_hi,
                        baseline_rate=b["rate"], lift=lift,
                        mean_atr_move=float(mv.mean()), median_atr_move=float(np.median(mv)),
                        baseline_mean_atr_move=b["mean"], mean_bp_move=float(np.nanmean(bp[valid])),
                        low_n=n < 30,
                    ))

    scorecard = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scorecard.to_csv(OUT_DIR / "scorecard.csv", index=False)
    print(f"\nsaved: {OUT_DIR / 'scorecard.csv'} ({len(scorecard)} rows)")

    print("\n=== OOS H12_1h 헤드라인(신호 x side) ===")
    head = scorecard[(scorecard["window"] == "OOS") & (scorecard["horizon"] == "H12_1h")].copy()
    head = head.sort_values("lift", ascending=False)
    with pd.option_context("display.width", 160, "display.max_rows", None):
        print(head[["signal", "side", "n_triggers", "hit_rate", "baseline_rate", "lift",
                     "mean_atr_move", "median_atr_move", "low_n"]].to_string(index=False))


if __name__ == "__main__":
    main()
