#!/usr/bin/env python3
"""XRP port of this project's Homer evidence-signal Tier0 feature + trigger pipeline.

`build_btc_5m_evidence_signal_candidates_tier0_20260901.py`의 **자산 상수만** 바꾼 포팅이다.
로직/피쳐/트리거 정의는 한 줄도 재구현하지 않았다 -- 재구현하면 자산별 파라미터가 조용히
어긋난다(BTC 포팅에서 실제로 발생, docs/homer/evidence_signal_new_coin_port_protocol.md §5-4).

## 데이터 실사 (2026-09-03, 착수 전 확인)

    klines   binance_data/klines/XRPUSDT/XRPUSDT-5m-api.csv
             272,490행 · 2024-01-01 ~ 2026-08-04 · **5분 갭 0개** · taker_buy_base 있음
    funding  binance_data/funding_rate_other/XRPUSDT-fundingRate-*.zip (31개월, 2024-01~2026-07)
             BTC와 같은 디렉토리다.

⚠️klines가 ETH(08-28)보다 **24일 뒤처져 있다**(마지막 2026-08-04). HOLDOUT(2026-04-01~)은
  4개월분이 확보되므로 연구 파이프라인에는 충분하다. 라이브 스코어러/섀도우는 API에서 직접
  받으므로 영향 없다. 리포트에 명시한다.
⚠️펀딩은 2026-07까지라 마지막 ~4일은 funding_z가 NaN이다.

## 포팅 주의 (BTC에서 터진 것)

- `START`(2024-01-01)와 klines 시작이 **같으므로 인덱스 오프셋이 0**이다. 그래도 하류 코드는
  절대 `pos`를 다른 파일 인덱스로 쓰지 말고 **타임스탬프로 매핑**해야 한다
  (BTC는 108봉 어긋나 경제성게이트 결론이 통째로 뒤집혔다).
- `smt_divergence`는 교차자산 파트너가 미해결이라 BTC와 동일하게 **제외**한다.
- `compute_signals(btc_df=None)`이므로 smt는 발동하지 않는다(설계).
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

KLINES_CSV = ROOT / "binance_data/klines/XRPUSDT/XRPUSDT-5m-api.csv"
FUNDING_DIR = ROOT / "binance_data/funding_rate_other"
SWEEP_IMPL_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"
OUT_DIR = ROOT / "data/labels/xrp_5m_evidence_signal_candidates_20260903"

START = pd.Timestamp("2024-01-01", tz="UTC")  # matches sweep_impl's own START, and XRP funding's own start (2024-01, klines도 동일)
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
    spec = importlib.util.spec_from_file_location("sweep_impl_xrp_20260903", SWEEP_IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_xrp_funding() -> pd.DataFrame:
    frames = []
    for p in sorted(FUNDING_DIR.glob("XRPUSDT-fundingRate-*.zip")):
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

    funding_df = load_xrp_funding()
    # ⚠️2026-09-03: 펀딩 로더가 `.dt.as_unit("us")`를 쓰는데 klines CSV 파싱은 [ns]를 준다.
    # pandas 2.x의 merge_asof는 dtype 일치를 요구해 MergeError로 터진다(BTC 빌더가 만들어질
    # 당시엔 안 터졌다 -- 환경 차이). 여기서 klines 쪽 단위에 맞춘다.
    funding_df["calc_time"] = funding_df["calc_time"].astype(raw["timestamp"].dtype)

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

    frame.to_csv(OUT_DIR / "xrp_5m_evidence_signal_candidates_tier0.csv", index=False)

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
        "output": str(OUT_DIR / "xrp_5m_evidence_signal_candidates_tier0.csv"),
    }
    (OUT_DIR / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
