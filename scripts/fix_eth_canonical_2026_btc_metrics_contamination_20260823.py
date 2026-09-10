#!/usr/bin/env python3
"""ETH 캐노니컬 2026 파일의 BTC metrics 오염 수정 (2026-08-23, 사용자 지시로 즉시 수정).

배경: `docs/experiments/eth_binance_metrics_archive_backfill_and_canonical_divergence_20260823.md`
— training_features_2026_rebuilt.csv의 2026-01-20 00:05 ~ 07-12 00:00 구간(49,818행)에
BTCUSDT metrics가 병합돼 있음이 byte 대조로 확정됨(07-13 미커밋 재빌드의 다심볼 소스 혼선).
원래 09-30 차세대 재구축 때 일괄 수정 예정이었으나 사용자가 즉시 수정을 지시.

수정 내용:
1. 원시 3컬럼(sum_open_interest_value/sum_toptrader_long_short_ratio/count_long_short_ratio)을
   검증된 ETH 아카이브 참조본(data/TOTAL_ETHUSDT_metrics_2024_2026.csv, 버킷종료 라벨 보정본)
   값으로 교체 — 오염 창 내부만.
2. 파생 14컬럼을 features/engineering.py·features/high_order_state.py의 수식 그대로 전체
   시리즈에 대해 재계산, **확장 창**(오염 창 + 288bar(24h) 롤링 번짐 꼬리 = ~2026-07-13
   00:00)에만 덮어씀:
   - 직접: whale_retail_ratio, whale_conviction, smart_money_flow, squeeze_power, oi_change_rate
   - 2차: ofti, kel, mta_funding, funding_oi_divergence, oi_up_price_down, oi_up_price_up,
     crowded_long_unwind_risk, crowded_short_squeeze_risk
   - long/short_squeeze_risk: funding 성분은 무오염이므로 델타 패치
     (score = funding성분 + 0.2*clip(oi_change*10,0,1) 선형 구조 이용)
   - 3차(high_order_state): crowding_pressure, execution_quality
3. 검증 게이트(쓰기 전): 비오염 구간 2곳(1월 초순, 8월 초순)에서 재계산값이 기존 저장값과
   일치해야 함 — 수식 복제가 정확하다는 증명. 불일치가 크면 쓰지 않고 중단.

백업: 원본은 .bak_pre_btc_metrics_fix_20260823으로 보존. 쓰기는 temp+rename(원자적).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
REFERENCE = ROOT / "data/TOTAL_ETHUSDT_metrics_2024_2026.csv"
BACKUP = TARGET.with_name(TARGET.name + ".bak_pre_btc_metrics_fix_20260823")

WIN_START = pd.Timestamp("2026-01-20 00:05:00")   # 오염 창(원시 교체 구간)
WIN_END = pd.Timestamp("2026-07-12 00:00:00")
TAIL_BARS = 288                                    # 롤링(최대 288bar) 번짐 꼬리
RAW_COLS = ["sum_open_interest_value", "sum_toptrader_long_short_ratio", "count_long_short_ratio"]

# 검증용 비오염 구간(수식 복제 정확성 증명)
CLEAN_ZONES = [("2026-01-05", "2026-01-15"), ("2026-08-01", "2026-08-15")]


def safe_rolling_z(series: pd.Series, window: int) -> pd.Series:
    """engineering.py::_safe_rolling_z 재현 (funding_oi_divergence의 fallback 경로용은 아님 —
    funding_z_score 컬럼이 존재하므로 실제로는 미사용, 만약을 위해 보존)."""
    m = series.rolling(window, min_periods=1).mean()
    s = series.rolling(window, min_periods=1).std().replace(0, 1e-8)
    return ((series - m) / s).fillna(0.0)


def hos_zscore(series: pd.Series, window: int, min_periods: int = 20) -> pd.Series:
    """high_order_state.py::_safe_zscore 재현."""
    roll_mean = series.rolling(window=window, min_periods=min_periods).mean()
    roll_std = series.rolling(window=window, min_periods=min_periods).std()
    return ((series - roll_mean) / roll_std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def normalize_signed(series: pd.Series, scale: float) -> pd.Series:
    return pd.Series(np.tanh(series.astype(float) / max(float(scale), 1e-8)), index=series.index)


def recompute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """원시 컬럼(교체 후)으로부터 파생 컬럼들을 전체 시리즈로 재계산 — 수식은
    features/engineering.py(2026-08-23 시점)와 features/high_order_state.py 그대로."""
    out = {}
    oi_val = pd.to_numeric(df["sum_open_interest_value"], errors="coerce")
    toptr = pd.to_numeric(df["sum_toptrader_long_short_ratio"], errors="coerce")
    cnt_ls = pd.to_numeric(df["count_long_short_ratio"], errors="coerce")
    funding = pd.to_numeric(df["last_funding_rate"], errors="coerce").fillna(0.0)
    close = pd.to_numeric(df["close"], errors="coerce")

    # ── 직접 파생 (engineering.py:211~221) ──
    out["whale_retail_ratio"] = toptr / cnt_ls.replace(0, np.nan)
    out["whale_conviction"] = toptr.diff()
    out["smart_money_flow"] = oi_val.pct_change().clip(-1, 1).fillna(0)
    out["squeeze_power"] = oi_val * funding
    out["oi_change_rate"] = oi_val.pct_change().clip(-1, 1).fillna(0)

    # _handle_missing(engineering.py:599~)의 diff계열 fillna(0) 재현
    out["whale_conviction"] = out["whale_conviction"].fillna(0.0)
    out["whale_retail_ratio"] = out["whale_retail_ratio"].ffill().fillna(0.0)

    # ── ofti (engineering.py:541~546) ──
    amihud = pd.to_numeric(df["amihud_illiquidity_z"], errors="coerce")
    ofti_raw = out["smart_money_flow"] * out["whale_conviction"] * (amihud.abs() + 1.0)
    out["ofti"] = pd.Series(np.tanh(ofti_raw * 3.0), index=df.index).fillna(0)

    # ── kel (engineering.py:549~563) ──
    ROLL = 288
    gk = pd.to_numeric(df["garman_klass_vol"], errors="coerce")
    funding_pressure = funding.rolling(window=ROLL, min_periods=1).sum()
    kel_raw = out["oi_change_rate"] / (gk + 1e-6) * np.sign(funding_pressure)
    kel_mean = kel_raw.rolling(ROLL, min_periods=1).mean()
    kel_std = kel_raw.rolling(ROLL, min_periods=1).std().replace(0, 1e-8)
    out["kel"] = pd.Series(np.tanh((kel_raw - kel_mean) / kel_std * 0.5), index=df.index).fillna(0)

    # ── mta_funding (engineering.py:566~583) ──
    weighted_roc = (0.5 * pd.to_numeric(df["funding_roc_12"], errors="coerce")
                    + 0.3 * pd.to_numeric(df["funding_roc_48"], errors="coerce")
                    + 0.2 * pd.to_numeric(df["funding_roc_288"], errors="coerce"))
    funding_abs = np.abs(funding)
    mta_normalized = weighted_roc / funding_abs.clip(lower=1e-5)
    sq_mean = out["squeeze_power"].rolling(ROLL, min_periods=1).mean()
    sq_std = out["squeeze_power"].rolling(ROLL, min_periods=1).std().replace(0, 1e-8)
    squeeze_z = (out["squeeze_power"] - sq_mean) / sq_std
    out["mta_funding"] = ((mta_normalized * np.tanh(squeeze_z)).clip(-3, 3) / 3).fillna(0)

    # ── funding_oi_divergence 계열 (engineering.py:758~769) ──
    funding_z = pd.to_numeric(df["funding_z_score"], errors="coerce").fillna(0.0)
    oi_change = pd.to_numeric(out["oi_change_rate"], errors="coerce").fillna(0.0)
    price_ret = close.pct_change().fillna(0.0)
    out["funding_oi_divergence"] = pd.Series(
        (np.tanh(funding_z) * np.tanh(oi_change * 10.0) - np.tanh(price_ret * 50.0)), index=df.index
    ).clip(-2.0, 2.0) / 2.0
    oi_up = oi_change > 0.0
    out["oi_up_price_down"] = (oi_up & (price_ret < 0.0)).astype(float) * np.tanh(oi_change.abs() * 10.0)
    out["oi_up_price_up"] = (oi_up & (price_ret > 0.0)).astype(float) * np.tanh(oi_change.abs() * 10.0)
    out["crowded_long_unwind_risk"] = pd.Series(
        np.tanh(funding_z.clip(lower=0.0)) * out["oi_up_price_down"], index=df.index
    ).clip(0.0, 1.0).fillna(0.0)
    out["crowded_short_squeeze_risk"] = pd.Series(
        np.tanh((-funding_z).clip(lower=0.0)) * out["oi_up_price_up"], index=df.index
    ).clip(0.0, 1.0).fillna(0.0)

    # ── long/short_squeeze_risk: 델타 패치 (funding 성분 무오염, 0.2*oi_buildup만 선형 교체) ──
    old_oi_change = pd.to_numeric(df["oi_change_rate"], errors="coerce").fillna(0.0)
    old_buildup = np.clip(old_oi_change * 10, 0, 1)
    new_buildup = np.clip(oi_change * 10, 0, 1)
    out["long_squeeze_risk"] = pd.to_numeric(df["long_squeeze_risk"], errors="coerce") - 0.2 * old_buildup + 0.2 * new_buildup
    out["short_squeeze_risk"] = pd.to_numeric(df["short_squeeze_risk"], errors="coerce") - 0.2 * old_buildup + 0.2 * new_buildup

    # ── high_order_state: crowding_pressure / execution_quality ──
    funding_div = pd.to_numeric(df["funding_price_divergence"], errors="coerce").fillna(0.0)
    whale_ratio_n = out["whale_retail_ratio"].fillna(0.0)
    whale_conv_n = out["whale_conviction"].fillna(0.0)
    crowd_raw = (0.35 * funding_z
                 + 0.25 * hos_zscore(oi_change, 96)
                 + 0.25 * hos_zscore(whale_ratio_n, 96)
                 + 0.15 * hos_zscore(whale_conv_n, 96)
                 + 0.20 * out["long_squeeze_risk"].fillna(0.0)
                 - 0.20 * out["short_squeeze_risk"].fillna(0.0)
                 + 0.20 * funding_div)
    out["crowding_pressure"] = normalize_signed(crowd_raw, 1.7)

    cvp_poc = pd.to_numeric(df["cvp_poc_dist"], errors="coerce").fillna(0.0)
    cvp_cluster = pd.to_numeric(df["cvp_cluster_position"], errors="coerce").fillna(0.0)
    cvp_imb = pd.to_numeric(df["cvp_volume_imbalance"], errors="coerce").fillna(0.0)
    net_taker = pd.to_numeric(df["net_taker_ratio"], errors="coerce").fillna(0.0)
    trade_int = pd.to_numeric(df["trade_intensity"], errors="coerce").fillna(0.0)
    liq_vac = pd.to_numeric(df["liquidity_vacuum"], errors="coerce").fillna(0.0)  # 무오염 컬럼 재사용
    bb_z = pd.to_numeric(df["bb_width_z"], errors="coerce").fillna(0.0)
    wick = pd.to_numeric(df["wick_ratio"], errors="coerce").fillna(0.0)
    # 'vwap_dist'는 저장 안 되는 중간 컬럼 — engineering.py::_calc_vwap_dist(win=288) 그대로 재계산
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    typical_price = (high + low + close) / 3
    tp_vol = typical_price * volume
    cum_tp_vol = tp_vol.rolling(window=288, min_periods=1).sum()
    cum_vol = volume.rolling(window=288, min_periods=1).sum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    vwap_dist = ((close - vwap) / (vwap + 1e-8)).fillna(0.0)
    anchor_quality = -0.55 * cvp_poc.abs() - 0.20 * (cvp_cluster - 0.5).abs() + 0.20 * cvp_imb.abs()
    flow_quality = 0.25 * out["smart_money_flow"].fillna(0.0) - 0.25 * net_taker.abs() + 0.20 * hos_zscore(trade_int, 96)
    fric_quality = -0.35 * liq_vac - 0.20 * bb_z.abs() + 0.15 * wick - 0.15 * vwap_dist.abs()
    out["execution_quality"] = normalize_signed(anchor_quality + flow_quality + fric_quality, 1.4)

    for k in ("crowding_pressure", "execution_quality"):
        out[k] = pd.to_numeric(out[k], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.DataFrame(out, index=df.index)


def main() -> int:
    df = pd.read_csv(TARGET, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    orig_cols = df.columns.tolist()
    n_orig = len(df)
    print(f"target rows={n_orig} cols={len(orig_cols)}")

    ref = pd.read_csv(REFERENCE)
    ref["create_time"] = pd.to_datetime(ref["create_time"])
    ref = ref[["create_time"] + RAW_COLS].rename(columns={c: c + "_ref" for c in RAW_COLS})

    win_mask = (df["timestamp"] >= WIN_START) & (df["timestamp"] <= WIN_END)
    print(f"오염 창 행수: {win_mask.sum()}")

    # 1) 원시 컬럼 교체 (정확 조인 + 결측버킷은 merge_asof backward 9h — 빌더 컨벤션)
    merged = df[["timestamp"]].merge(ref, left_on="timestamp", right_on="create_time", how="left")
    asof = pd.merge_asof(df[["timestamp"]].sort_values("timestamp"), ref.sort_values("create_time"),
                         left_on="timestamp", right_on="create_time",
                         direction="backward", tolerance=pd.Timedelta("9h"))
    n_exact = merged.loc[win_mask, RAW_COLS[0] + "_ref"].notna().sum()
    print(f"원시 교체: 정확일치 {n_exact}/{win_mask.sum()}, 나머지는 asof-backward")
    for c in RAW_COLS:
        exact_vals = merged[c + "_ref"]
        fallback = asof[c + "_ref"]
        newvals = exact_vals.fillna(fallback)
        df.loc[win_mask, c] = newvals[win_mask].to_numpy()
    still_na = df.loc[win_mask, RAW_COLS].isna().sum().sum()
    if still_na:
        print(f"⚠️ 교체 후 NaN {still_na}개 — 중단"); return 1

    # 2) 파생 전체 재계산
    rec = recompute_derived(df)

    # 3) 검증 게이트: 비오염 구간에서 재계산 == 기존 저장값
    print("\n[검증] 비오염 구간 수식복제 정확성:")
    fail = False
    for z0, z1 in CLEAN_ZONES:
        zm = (df["timestamp"] >= z0) & (df["timestamp"] < z1)
        for col in rec.columns:
            a = pd.to_numeric(df.loc[zm, col], errors="coerce")
            b = rec.loc[zm, col]
            denom = a.abs().clip(lower=1e-6)
            bad = ((a - b).abs() / denom > 1e-4) & ~(a.isna() & b.isna())
            frac = bad.mean()
            if frac > 0.005:
                print(f"  ✗ {col} @{z0}: 불일치 {frac*100:.2f}%")
                fail = True
    if fail:
        print("검증 실패 — 아무것도 쓰지 않고 중단. 수식 재점검 필요."); return 1
    print("  ✓ 전 컬럼 통과(불일치율 ≤0.5%)")

    # 4) 확장 창(오염 창 + 288bar 꼬리)에 파생값 덮어쓰기
    ext_end_idx = df.index[df["timestamp"] <= WIN_END].max() + TAIL_BARS
    ext_end_ts = df.loc[min(ext_end_idx, len(df) - 1), "timestamp"]
    ext_mask = (df["timestamp"] >= WIN_START) & (df["timestamp"] <= ext_end_ts)
    print(f"\n파생 덮어쓰기 확장 창: {WIN_START} ~ {ext_end_ts} ({ext_mask.sum()}행)")
    changed_stats = {}
    for col in rec.columns:
        old = pd.to_numeric(df.loc[ext_mask, col], errors="coerce")
        new = rec.loc[ext_mask, col]
        changed = ((old - new).abs() > 1e-12).sum()
        changed_stats[col] = int(changed)
        df.loc[ext_mask, col] = new.to_numpy()
    print("컬럼별 변경 행수:", changed_stats)

    # 5) 백업 + 원자적 쓰기
    if not BACKUP.exists():
        shutil.copy2(TARGET, BACKUP)
        print(f"백업: {BACKUP.name}")
    assert df.columns.tolist() == orig_cols and len(df) == n_orig
    tmp = TARGET.with_suffix(".csv.tmp_fix")
    df.to_csv(tmp, index=False)
    tmp.replace(TARGET)
    print(f"✓ 수정 완료 → {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
