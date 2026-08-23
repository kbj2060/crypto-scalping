#!/usr/bin/env python3
"""wide24 레짐 HMM 입력 피쳐 미니빌더 — ETH 2021-12~2023-12 히스토리 확장 (2026-08-23).

목적: scripts/experiment_regime3_current_hmm_wide24_20260529.py(STATE12_COLS+WIDE24_EXTRA_COLS
+ 라벨 함수)가 소비하는 캐노니컬 프레임 컬럼만 원시 아카이브에서 재구축한다.
state7_*/state12_* 파생은 HMM 런타임(_with_raw_state7/_with_raw_state12)이 계산하므로
여기서는 그 입력 컬럼까지만 만든다. _with_raw_state7 전체를 정독해 확인한 입력 목록:
  bb_width_z, chop_index, mtf_trend_1h, mtf_trend_4h, hma_slope, breakout_strength,
  dual_momentum, mean_reversion_z, net_taker_ratio, smart_money_flow,
  taker_acceleration, ofi_acceleration  (+close/high/low)
→ 과제의 19개 컬럼에 dual_momentum, smart_money_flow 2개를 추가로 생산한다(총 21개).

타임스탬프 컨벤션 (2026-08-23 실측):
  - 캐노니컬 training_features_2026_rebuilt.csv의 timestamp == klines bar OPEN time.
    (ETHUSDT-5m-api.csv timestamp(=open_time)와 동일 ts 조인 시 close 일치율 100%,
     open_time+5min 조인 시 0.3% — bar END 아님이 확정)
  - metrics(TOTAL_*_metrics, create_time=버킷종료 라벨·+5분 보정본)는 timestamp와
    같은 벽시계 값끼리 정확 조인(sum_open_interest_value 일치율 100%). 결측 버킷은
    merge_asof backward 9h 폴백 — scripts/fix_eth_canonical_2026_btc_metrics_
    contamination_20260823.py:194-205의 빌더 컨벤션 그대로.
  - 2021-2023 아카이브 klines는 open_time(ms, bar 시작)이므로 timestamp=open_time.

무결성 게이트(필수, 선실행): 2026 원시 소스(ETHUSDT-5m-api.csv + TOTAL_ETHUSDT_metrics_
2024_2026.csv + BTCUSDT-5m-api.csv)로 같은 빌더를 돌려, 검증된 캐노니컬 참조본
training_features_2026_rebuilt.csv과 두 프로브 구간(2026-01-05~01-15, 2026-08-01~08-15)
에서 전 컬럼 대조. 컬럼별로 행의 ≥99%가 상대오차<1e-6(값≈0이면 절대오차<1e-9)이어야
통과. 하나라도 실패하면 2021-2023 출력을 쓰지 않고 비정상 종료한다.
adx_14는 캐노니컬 파일에 없음(런타임 _adx 폴백 계산) — 게이트에서는 캐노니컬 자신의
high/low/close로 _adx를 재계산한 값을 참조로 쓴다.

참조본 측 결함 2건 (2026-08-23 게이트 1차 실행에서 발견·정밀 검증 — 오차 기준 완화가
아니라 참조 결함을 각각 별도 항등성 증명으로 대체, 전부 gate_report.json에 기록):
  [V1] dual_momentum: 캐노니컬 2026 파일은 2026-01-01부터 워밍업 없이 빌드돼
       shift(2016)=7일 창이 채워지는 2026-01-08 이전은 fillna(0)=0으로 저장돼 있다.
       → 직접 대조는 참조 워밍업 완료(ref 시작+2016bar) 이후 행으로 한정하고,
       별도로 "참조와 같은 2026-01-01 절단 시작으로 재빌드하면 캐노니컬과 일치"
       항등성 증명(양 구간 ≥99.9%)을 요구한다. 실측: 절단 재현 일치율 100%/100%.
  [V2] oi_change_rate/smart_money_flow: 캐노니컬 2026-07-12 이후 꼬리 구간은 metrics를
       비보정 라벨(버킷시작 라벨 = 1버킷 미래참조)로 조인한 다른 빈티지다
       (일별 스캔: 07-11까지 보정 컨벤션 일치 100%, 07-13부터 shift(-1) 항등 100%).
       추가로 2026-08-01~02 이틀은 참조본 자체의 값 스케일 결함(~1/319 배).
       → zone2 직접 대조는 참조 결함으로 판정 불능. 대신 "캐노니컬 == 내 시리즈의
       1bar 미래 시프트" 항등성 증명(스케일 결함일 제외, ≥99%)을 요구한다.
       내 빌드는 검증된 보정·인과 컨벤션(zone1 직접 대조 100% 통과)을 유지한다.

dual_momentum은 close_btc가 필요 — BTC 5m 아카이브(2021-12~2023-12)를
data.binance.vision 월별 zip에서 내려받아 tmp 출력 디렉토리에 캐시한다
(scripts/download_eth_klines_archive_2021_2023_20260823.py와 동일 패턴, data/ 미기록).
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tmp/eth_wide24_history_extension_20260823"
OUT_CSV = OUT_DIR / "eth_wide24_inputs_2021_2023.csv"
GATE_REPORT = OUT_DIR / "gate_report.json"

ETH_ARCHIVE = ROOT / "data/eth_5m_2021_2023_archive.csv"
METRICS_2123 = ROOT / "data/TOTAL_ETHUSDT_metrics_2021_2023.csv"
BTC_ARCHIVE = OUT_DIR / "btc_5m_2021_2023_archive.csv"  # 없으면 다운로드(캐시)

ETH_API_2026 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
BTC_API_2026 = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
METRICS_2426 = ROOT / "data/TOTAL_ETHUSDT_metrics_2024_2026.csv"
CANONICAL_2026 = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"

PROBE_ZONES = [("2026-01-05", "2026-01-15"), ("2026-08-01", "2026-08-15")]

PASS_COLS = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]
# engineering.py::_handle_missing:598-629 — diff 계열은 fillna(0)
DIFF_FILL_COLS = [
    "log_return", "hma_slope", "smart_money_flow", "oi_change_rate",
    "dual_momentum", "mean_reversion_z", "breakout_strength", "ofi_acceleration",
]
# engineering.py::_handle_missing:631-636 — ULTIMATE_FEATURE_COLS 나머지는 ffill().fillna(0)
FFILL_COLS = [
    "net_taker_ratio", "taker_acceleration", "volatility_z", "rsi", "macd_hist",
    "bb_width", "bb_width_z", "wick_ratio", "garman_klass_vol", "chop_index",
]
# mtf_trend_1h/4h는 ULTIMATE_FEATURE_COLS 밖 — 생성 시 인라인 fillna(0)만(engineering.py:302-303).
# adx_14는 캐노니컬 미저장 — wide24 런타임 _adx와 동일 수식으로 여기서 선계산(추가 fill 없음).
FEATURE_COLS = DIFF_FILL_COLS + FFILL_COLS + ["mtf_trend_1h", "mtf_trend_4h", "adx_14"]

BTC_URL = "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/5m/BTCUSDT-5m-{ym}.zip"
BTC_MONTHS = ["2021-12"] + [f"{y}-{m:02d}" for y in (2022, 2023) for m in range(1, 13)]
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
              "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def _ensure_btc_archive() -> pd.DataFrame:
    if BTC_ARCHIVE.exists():
        df = pd.read_csv(BTC_ARCHIVE)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    frames = []
    for ym in BTC_MONTHS:
        raw = urlopen(BTC_URL.format(ym=ym), timeout=120).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            with z.open(z.namelist()[0]) as f:
                first = f.read(64).decode("utf-8", errors="replace")
            with z.open(z.namelist()[0]) as f:
                has_header = not first.split(",")[0].strip().isdigit()
                df = pd.read_csv(f, header=0 if has_header else None)
                df.columns = KLINE_COLS
                frames.append(df)
        print(f"  BTC {ym}: {len(frames[-1])} rows", flush=True)
    btc = pd.concat(frames, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    btc["timestamp"] = pd.to_datetime(btc["open_time"], unit="ms")
    btc = btc[["timestamp", "close"]].reset_index(drop=True)
    btc.to_csv(BTC_ARCHIVE, index=False)
    return btc


def _adx_runtime(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """wide24 라벨 런타임 _adx 원문 복사 — scripts/experiment_regime3_current_hmm_wide24_20260529.py:117-127.
    (직접 import 시 features/__init__→core→binance 모듈 체인이 이 환경에 없어 실패하므로 원문 복제.)"""
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    up = high.diff()
    down = -low.diff()
    pdm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    ndm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    pdi = 100.0 * pdm.ewm(span=period, adjust=False).mean() / (atr + 1e-12)
    ndi = 100.0 * ndm.ewm(span=period, adjust=False).mean() / (atr + 1e-12)
    dx = 100.0 * (pdi - ndi).abs() / (pdi + ndi + 1e-12)
    return dx.ewm(span=period, adjust=False).mean()


def _calc_rma(x: pd.Series, n: int) -> pd.Series:
    # engineering.py:386-388
    return x.ewm(alpha=1 / n, adjust=False).mean()


def _calc_wma(s: pd.Series, period: int) -> pd.Series:
    # engineering.py:410-413
    weights = np.arange(1, period + 1)
    return s.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def build_features(kl: pd.DataFrame, btc: pd.DataFrame, met: pd.DataFrame) -> pd.DataFrame:
    """원시 klines(kl: timestamp=bar open)+BTC close+metrics에서 21개 대상 컬럼 계산.

    수식은 features/engineering.py 및 wide24 런타임에서 컬럼별 인용 라인 그대로 복제.
    """
    df = kl.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True).copy()

    # ── BTC 병합: engineering.py::_merge_data:201-207 (merge_asof backward) ──
    btc_r = btc[["timestamp", "close"]].rename(columns={"close": "close_btc"}).sort_values("timestamp")
    df = pd.merge_asof(df, btc_r, on="timestamp", direction="backward")

    # ── metrics 병합: 정확조인 + asof backward 9h 폴백 (fix_..._20260823.py:194-205) ──
    met_r = met[["create_time", "sum_open_interest_value"]].sort_values("create_time")
    exact = df[["timestamp"]].merge(met_r, left_on="timestamp", right_on="create_time", how="left")
    asof = pd.merge_asof(df[["timestamp"]], met_r, left_on="timestamp", right_on="create_time",
                         direction="backward", tolerance=pd.Timedelta("9h"))
    oi_val = exact["sum_open_interest_value"].fillna(asof["sum_open_interest_value"])
    df["_oi_exact"] = exact["sum_open_interest_value"].notna()
    df["sum_open_interest_value"] = oi_val.to_numpy()

    close, high, low, opn = df["close"], df["high"], df["low"], df["open"]

    # smart_money_flow: engineering.py:218 / oi_change_rate: engineering.py:221
    df["smart_money_flow"] = df["sum_open_interest_value"].pct_change().clip(-1, 1).fillna(0)
    df["oi_change_rate"] = df["sum_open_interest_value"].pct_change().clip(-1, 1).fillna(0)

    # net_taker_ratio: engineering.py:226-231
    quote_vol = df["quote_volume"].replace(0, np.nan)
    taker_buy = df["taker_buy_quote"]
    taker_sell = df["quote_volume"] - taker_buy
    df["net_taker_ratio"] = (taker_buy - taker_sell) / quote_vol
    # taker_acceleration: engineering.py:233-235
    short_ma = df["net_taker_ratio"].rolling(window=2, min_periods=1).mean()
    long_ma = df["net_taker_ratio"].rolling(window=20, min_periods=1).mean()
    df["taker_acceleration"] = short_ma - long_ma

    # log_return: engineering.py:253
    df["log_return"] = np.log(close / close.shift(1))

    # volatility_z: engineering.py:255-260 (ATR=RMA(TR,14), z창 288)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = _calc_rma(tr, 14)
    atr_mean = atr.rolling(window=288, min_periods=1).mean()
    atr_std = atr.rolling(window=288, min_periods=1).std().replace(0, 1e-8)
    df["volatility_z"] = (atr - atr_mean) / atr_std

    # rsi: engineering.py:262-263, 398-408 (Wilder RMA, +1e-8)
    delta = close.diff()
    avg_gain = _calc_rma(delta.clip(lower=0), 14)
    avg_loss = _calc_rma(-delta.clip(upper=0), 14)
    rs = avg_gain / (avg_loss + 1e-8)
    df["rsi"] = 100 - (100 / (1 + rs))

    # macd_hist: engineering.py:265-270
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    df["macd_hist"] = macd_line - macd_line.ewm(span=9, adjust=False).mean()

    # bb_width / bb_width_z: engineering.py:272-281 (bb_std는 ddof=0, z창 100)
    bb_mid = close.rolling(window=20, min_periods=1).mean()
    bb_std = close.rolling(window=20, min_periods=1).std(ddof=0)
    df["bb_width"] = ((bb_mid + 2 * bb_std) - (bb_mid - 2 * bb_std)) / (bb_mid + 1e-8)
    bbw_mean = df["bb_width"].rolling(window=100, min_periods=1).mean()
    bbw_std = df["bb_width"].rolling(window=100, min_periods=1).std().replace(0, 1e-8)
    df["bb_width_z"] = (df["bb_width"] - bbw_mean) / bbw_std

    # hma_slope: engineering.py:283-285, 415-424 (HMA n=20 → WMA 10/20/4)
    wma_half = _calc_wma(close, 10)
    wma_full = _calc_wma(close, 20)
    hma = _calc_wma(2 * wma_half - wma_full, 4)
    df["hma_slope"] = hma.diff() / (close + 1e-8)

    # wick_ratio: engineering.py:287-292
    body = np.abs(close - opn)
    rng = high - low
    df["wick_ratio"] = np.where(rng == 0, 0, (rng - body) / rng)

    # garman_klass_vol: engineering.py:294, 353-366 (rolling 20 mean, clip>=0, sqrt)
    h_ = high.clip(lower=low)
    o_ = opn.replace(0, np.nan)
    c_ = close.replace(0, np.nan)
    l_ = low.replace(0, np.nan)
    gk = 0.5 * (np.log(h_ / l_)) ** 2 - (2 * np.log(2) - 1) * (np.log(c_ / o_)) ** 2
    df["garman_klass_vol"] = gk.rolling(window=20, min_periods=1).mean().clip(lower=0) ** 0.5

    # mtf_trend_1h/4h: engineering.py:299-303 (EMA span 12/48의 pct_change)
    df["mtf_trend_1h"] = close.ewm(span=12, adjust=False).mean().pct_change().fillna(0)
    df["mtf_trend_4h"] = close.ewm(span=48, adjust=False).mean().pct_change().fillna(0)

    # chop_index: engineering.py:382, 426-439 (rolling 14, min_periods 미지정)
    atr_sum = tr.rolling(window=14).sum()
    high_max = high.rolling(window=14).max()
    low_min = low.rolling(window=14).min()
    df["chop_index"] = 100 * np.log10((atr_sum + 1e-8) / (high_max - low_min + 1e-8)) / np.log10(14)

    # dual_momentum: engineering.py:957-970 (shift 2016 = 7일)
    abs_mom = (close / close.shift(2016) - 1).fillna(0)
    btc_mom = (df["close_btc"] / df["close_btc"].shift(2016) - 1).fillna(0)
    rel_mom = abs_mom - btc_mom
    df["dual_momentum"] = pd.Series(
        np.where((abs_mom > 0) & (rel_mom > 0), 1.0,
                 np.where((abs_mom < 0) & (rel_mom < 0), -1.0, 0.0)), index=df.index).fillna(0)

    # mean_reversion_z: engineering.py:972-979 (창 288, -tanh(z/2))
    ma288 = close.rolling(288).mean()
    std288 = close.rolling(288).std()
    df["mean_reversion_z"] = pd.Series(-np.tanh(((close - ma288) / (std288 + 1e-8)) / 2), index=df.index).fillna(0)

    # breakout_strength: engineering.py:981-989 (창 144, clip ±1)
    box_high = high.rolling(144).max()
    box_low = low.rolling(144).min()
    strength = (close - (box_high + box_low) / 2) / ((box_high - box_low) + 1e-8)
    df["breakout_strength"] = pd.Series(np.clip(strength, -1, 1), index=df.index).fillna(0)

    # ofi_acceleration: engineering.py:154-156 (ewm span=5는 기본 adjust=True, 3-lag diff)
    ntr_smooth = df["net_taker_ratio"].ewm(span=5).mean()
    df["ofi_acceleration"] = ntr_smooth.diff(3).fillna(0)

    # adx_14: 캐노니컬 미저장 컬럼 — wide24 라벨 런타임과 동일 수식(_adx_runtime 원문 복제 사용)
    df["adx_14"] = _adx_runtime(high, low, close)

    # ── 결측 처리: engineering.py::_handle_missing:595-638 재현 (해당 컬럼만) ──
    df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    for col in DIFF_FILL_COLS:
        df[col] = df[col].fillna(0)
    df[FFILL_COLS] = df[FFILL_COLS].ffill().fillna(0)

    return df[["timestamp"] + PASS_COLS + FEATURE_COLS + ["_oi_exact"]]


def _load_klines_api(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def run_gate() -> tuple[bool, dict]:
    """2026 원시 소스로 빌더를 돌려 캐노니컬 참조본과 프로브 구간 대조."""
    eth = _load_klines_api(ETH_API_2026)
    btc = _load_klines_api(BTC_API_2026)[["timestamp", "close"]]
    met = pd.read_csv(METRICS_2426, usecols=["create_time", "sum_open_interest_value"])
    met["create_time"] = pd.to_datetime(met["create_time"])
    built = build_features(eth, btc, met)

    can = pd.read_csv(CANONICAL_2026, low_memory=False)
    can["timestamp"] = pd.to_datetime(can["timestamp"])
    # adx_14 참조: 캐노니컬 자신의 high/low/close로 런타임 _adx 재현(전 구간 계산 후 구간 절단)
    can["adx_14"] = _adx_runtime(pd.to_numeric(can["high"]), pd.to_numeric(can["low"]), pd.to_numeric(can["close"]))

    merged = can.merge(built, on="timestamp", suffixes=("_ref", "_new"), how="inner")

    # [V1] 참조 워밍업 경계: 캐노니컬은 2026-01-01 시작·무워밍업 빌드 → shift(2016) 창 미충족
    ref_warm_end = can["timestamp"].iloc[0] + pd.Timedelta(minutes=5 * 2016)
    # [V2] 캐노니컬 metrics 조인 컨벤션이 비보정(1버킷 미래)으로 바뀌는 경계 + 스케일 결함일
    ref_oi_vintage_switch = pd.Timestamp("2026-07-12")
    ref_oi_glitch_days = [pd.Timestamp("2026-08-01"), pd.Timestamp("2026-08-02")]
    oi_cols = {"oi_change_rate", "smart_money_flow"}

    def _cmp(ref: np.ndarray, new: np.ndarray) -> np.ndarray:
        both_nan = np.isnan(ref) & np.isnan(new)
        return (np.abs(new - ref) < np.maximum(1e-6 * np.abs(ref), 1e-9)) | both_nan

    report: dict = {"zones": {}, "vintage_findings": {}, "pass": True}
    check_cols = PASS_COLS + FEATURE_COLS
    for z0, z1 in PROBE_ZONES:
        zm = (merged["timestamp"] >= z0) & (merged["timestamp"] < z1)
        zrep = {}
        for col in check_cols:
            sel = zm.copy()
            note = None
            if col == "dual_momentum":
                sel &= merged["timestamp"] >= ref_warm_end  # [V1] 참조 워밍업 미완 행 제외
                if (zm & ~sel).any():
                    note = "V1: ref-warmup rows excluded"
            if col in oi_cols and pd.Timestamp(z0) >= ref_oi_vintage_switch:
                # [V2] 참조 꼬리 빈티지 결함 — 직접 대조 대신 1bar-미래 시프트 항등성 증명
                glitch = merged["timestamp"].dt.normalize().isin(ref_oi_glitch_days)
                # pct_change가 스케일 결함 경계 1bar 앞뒤로 번지므로 여유 1bar 포함 제외
                glitch = glitch | glitch.shift(1, fill_value=False) | glitch.shift(-1, fill_value=False)
                sel &= ~glitch
                ref = pd.to_numeric(merged.loc[sel, f"{col}_ref"], errors="coerce").to_numpy(dtype=float)
                new = merged[f"{col}_new"].shift(-1).loc[sel].to_numpy(dtype=float)
                note = "V2: ref tail uses uncorrected(1-bucket-future) metrics labels; shifted-identity proof (glitch days 08-01/02 excluded)"
            else:
                ref = pd.to_numeric(merged.loc[sel, f"{col}_ref"], errors="coerce").to_numpy(dtype=float)
                new = pd.to_numeric(merged.loc[sel, f"{col}_new"], errors="coerce").to_numpy(dtype=float)
            ok = _cmp(ref, new)
            rate = float(np.mean(ok)) if len(ok) else 0.0
            zrep[col] = {"match_rate": round(rate, 6), "n": int(len(ok))}
            if note:
                zrep[col]["note"] = note
            if rate < 0.99:
                report["pass"] = False
                bad = ~ok
                worst = float(np.nanmax(np.abs(new[bad] - ref[bad]))) if bad.any() else 0.0
                zrep[col]["max_abs_err"] = worst
        report["zones"][f"{z0}~{z1}"] = zrep

    # [V1] 항등성 증명: 참조와 같은 절단 시작(무워밍업)으로 dual_momentum 재현 시 캐노니컬과 일치해야 함
    cc = can.sort_values("timestamp").reset_index(drop=True)
    close_c = pd.to_numeric(cc["close"], errors="coerce")
    btc_c = pd.to_numeric(cc["close_btc"], errors="coerce")
    abs_mom = (close_c / close_c.shift(2016) - 1).fillna(0)
    rel_mom = abs_mom - (btc_c / btc_c.shift(2016) - 1).fillna(0)
    dm_trunc = np.where((abs_mom > 0) & (rel_mom > 0), 1.0, np.where((abs_mom < 0) & (rel_mom < 0), -1.0, 0.0))
    v1_rates = {}
    for z0, z1 in PROBE_ZONES:
        zmask = ((cc["timestamp"] >= z0) & (cc["timestamp"] < z1)).to_numpy()
        v1_rates[f"{z0}~{z1}"] = float((pd.to_numeric(cc.loc[zmask, "dual_momentum"]).to_numpy() == dm_trunc[zmask]).mean())
    v1_ok = all(r >= 0.999 for r in v1_rates.values())
    report["vintage_findings"]["V1_dual_momentum_ref_no_warmup"] = {
        "detail": f"canonical starts {cc['timestamp'].iloc[0]} with no warmup; shift(2016) rows before {ref_warm_end} stored as 0",
        "truncated_start_replication_match": v1_rates, "proof_pass": v1_ok,
    }
    report["vintage_findings"]["V2_oi_ref_tail_uncorrected_labels"] = {
        "detail": "canonical metrics join switches to uncorrected bucket-start labels (1-bucket future reference) from ~2026-07-12; "
                  "2026-08-01~02 additionally carry a value-scale glitch (~1/319x). Direct zone2 comparison invalid; "
                  "shifted-identity proof used instead (see zone2 oi_change_rate/smart_money_flow notes).",
    }
    if not v1_ok:
        report["pass"] = False
    return report["pass"], report


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/3] BTC 2021-2023 아카이브 확보")
    btc_hist = _ensure_btc_archive()

    print("[2/3] 무결성 게이트: 2026 원시 재구축 vs 캐노니컬 참조본")
    ok, report = run_gate()
    report["timestamp_convention"] = (
        "canonical timestamp == klines bar OPEN time (close 일치율 100% 실측); "
        "metrics create_time(버킷종료 +5분 보정 라벨)은 timestamp와 동일 벽시계 정확조인, "
        "결측버킷은 merge_asof backward 9h 폴백"
    )
    for zname, zrep in report["zones"].items():
        fails = {c: v for c, v in zrep.items() if v["match_rate"] < 0.99}
        print(f"  zone {zname}: {'PASS' if not fails else 'FAIL'}")
        for c, v in sorted(zrep.items(), key=lambda kv: kv[1]["match_rate"]):
            mark = "✗" if v["match_rate"] < 0.99 else "✓"
            print(f"    {mark} {c}: match {v['match_rate']*100:.3f}% (n={v['n']})")
    if not ok:
        print("게이트 실패 — 수식/빈티지 불일치. 2021-2023 출력을 쓰지 않고 중단.")
        return 1

    print("[3/3] 2021-2023 빌드")
    kl = pd.read_csv(ETH_ARCHIVE)
    kl["timestamp"] = pd.to_datetime(kl["open_time"], unit="ms")  # open_time=bar 시작 → timestamp 컨벤션 동일
    met = pd.read_csv(METRICS_2123, usecols=["create_time", "sum_open_interest_value"])
    met["create_time"] = pd.to_datetime(met["create_time"])
    out = build_features(kl, btc_hist, met)
    oi_exact_frac = float(out["_oi_exact"].mean())
    out = out.drop(columns=["_oi_exact"])
    out.to_csv(OUT_CSV, index=False)

    report["output"] = {
        "path": str(OUT_CSV),
        "rows": int(len(out)),
        "range": [str(out["timestamp"].min()), str(out["timestamp"].max())],
        "metrics_exact_join_frac": round(oi_exact_frac, 6),
        "nan_counts": {c: int(out[c].isna().sum()) for c in out.columns if out[c].isna().any()},
    }
    GATE_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"완료: {OUT_CSV} rows={len(out)} | gate_report={GATE_REPORT}")
    return 0


if __name__ == "__main__":
    ROOT_STR = str(ROOT)
    if ROOT_STR not in sys.path:
        sys.path.insert(0, ROOT_STR)
    sys.exit(main())
