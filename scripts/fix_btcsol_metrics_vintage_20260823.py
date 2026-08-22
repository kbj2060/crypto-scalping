#!/usr/bin/env python3
"""BTC/SOL 캐노니컬 파일들의 metrics vintage 오염 수정 (2026-08-23).

배경: 전수 감사(docs/experiments/eth_binance_metrics_archive_backfill_and_canonical_divergence_20260823.md
§7 예정)에서 BTC/SOL 파일의 OI/롱숏비 컬럼이 아카이브 라벨-컨벤션 혼재(vintage 섞임)로
대량의 1-bar 어긋남을 갖고 있음이 확인됨:
- BTC 2024: 24.1%가 next-bucket(**1-bar 미래참조**), 2.5% prev / BTC 2026: 9.4% next
- SOL 2024: 2.3% prev / SOL 2026: 4.5% next
같은 심볼의 raw_frame/연도/결합/metrics4 파일 전부 동일 패턴(공통 소스에서 파생, 연도
파일은 결합본의 정확한 슬라이스임을 실측 확인).

수정 절차(ETH 수정과 동일 원칙):
1. 결합 features 파일: [게이트] 원본 raw로 파생 17컬럼을 재계산해 저장값과 일치함을
   증명(수식 복제 정확성) → raw 3컬럼을 검증된 아카이브 참조본(TOTAL_{SYM}USDT_metrics_
   2024_2026.csv, +5분 보정본)으로 전행 교체 → 파생 17컬럼 전행 재계산·덮어쓰기.
   (long/short_squeeze_risk는 델타패치 — funding 성분이 adaptive_squeeze 플래그 여부와
   무관하게 보존되는 구조 이용)
2. 연도 파일: 수정된 결합본에서 원본과 같은 구간으로 재슬라이스.
3. raw_frame: raw 3컬럼만 교체. metrics4 raw_frame: 3+2컬럼.
4. BTC metrics4 결합본: raw 5컬럼 교체 + 표준 17 + metrics4 파생 5(taker_vol_ratio_z/
   count_toptrader_ratio_z/toptrader_count_size_divergence/sig_whale/sig_oi_divergence,
   build_btc_features_metrics4_20260802.py 수식 그대로) 재계산.
5. 변형 파일(swingtransition/zigzag/regimeline/1h_full): [계보 게이트] 영향 컬럼이 원본
   결합본과 동일했음을 확인 후, 수정된 결합본에서 join 교체.

백업: 전부 .bak_pre_metrics_vintage_fix_20260823. 원자적 쓰기.
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
YEAR = ROOT / "data/splits/year_oos"

spec = importlib.util.spec_from_file_location(
    "ethfix", ROOT / "scripts/fix_eth_canonical_2026_btc_metrics_contamination_20260823.py")
ethfix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ethfix)
recompute_derived = ethfix.recompute_derived

RAW3 = ["sum_open_interest_value", "sum_toptrader_long_short_ratio", "count_long_short_ratio"]
RAW5 = RAW3 + ["sum_taker_long_short_vol_ratio", "count_toptrader_long_short_ratio"]
BAK_SUFFIX = ".bak_pre_metrics_vintage_fix_20260823"
PROBE = ("2025-03-01", "2025-03-20")   # 감사에서 양 심볼 모두 99.9% exact였던 구간


def load_ref(sym: str) -> pd.DataFrame:
    r = pd.read_csv(ROOT / f"data/TOTAL_{sym}_metrics_2024_2026.csv")
    r["create_time"] = pd.to_datetime(r["create_time"])
    return r


def save_atomic(path: Path, df: pd.DataFrame) -> None:
    bak = path.with_name(path.name + BAK_SUFFIX)
    if not bak.exists():
        shutil.copy2(path, bak)
    tmp = path.with_suffix(path.suffix + ".tmpfix")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def replace_raw(df: pd.DataFrame, ref: pd.DataFrame, raw_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    cols = ["create_time"] + [c for c in raw_cols if c in ref.columns]
    exact = df[["timestamp"]].merge(ref[cols], left_on="timestamp", right_on="create_time", how="left")
    asof = pd.merge_asof(df[["timestamp"]].sort_values("timestamp"), ref[cols].sort_values("create_time"),
                         left_on="timestamp", right_on="create_time",
                         direction="backward", tolerance=pd.Timedelta("9h"))
    stats = {}
    for c in raw_cols:
        newv = exact[c].fillna(asof[c])
        old = pd.to_numeric(df[c], errors="coerce")
        mask = newv.notna()
        stats[c] = {"replaced_diff": int(((old - newv).abs() > 1e-9)[mask].sum()),
                    "exact_hits": int(exact[c].notna().sum()), "na_left": int((~mask).sum())}
        df.loc[mask, c] = newv[mask].to_numpy()
    return df, stats


def fidelity_gate(df: pd.DataFrame, rec: pd.DataFrame, label: str) -> bool:
    zm = (df["timestamp"] >= PROBE[0]) & (df["timestamp"] < PROBE[1])
    if zm.sum() == 0:
        print(f"  [{label}] 게이트 구간 없음 — 스킵 불가, 실패 처리")
        return False
    ok = True
    for col in rec.columns:
        if col not in df.columns:
            continue
        a = pd.to_numeric(df.loc[zm, col], errors="coerce")
        b = rec.loc[zm, col]
        bad = (((a - b).abs() / a.abs().clip(lower=1e-6)) > 1e-4).mean()
        if bad > 0.005:
            print(f"  [{label}] 게이트 실패: {col} 불일치 {bad*100:.2f}%")
            ok = False
    return ok


def metrics4_extras(df: pd.DataFrame) -> pd.DataFrame:
    """build_btc_features_metrics4_20260802.py의 파생 수식 재현."""
    out = {}

    def z(s, window=288):
        m = s.rolling(window=window, min_periods=1).mean()
        sd = s.rolling(window=window, min_periods=1).std().replace(0, 1e-8)
        return ((s - m) / sd).fillna(0)

    taker = pd.to_numeric(df["sum_taker_long_short_vol_ratio"], errors="coerce")
    cnt_top = pd.to_numeric(df["count_toptrader_long_short_ratio"], errors="coerce")
    sum_top = pd.to_numeric(df["sum_toptrader_long_short_ratio"], errors="coerce")
    out["taker_vol_ratio_z"] = z(taker)
    out["count_toptrader_ratio_z"] = z(cnt_top)
    out["toptrader_count_size_divergence"] = out["count_toptrader_ratio_z"] - z(sum_top)

    ratio = pd.to_numeric(df["whale_retail_ratio"], errors="coerce").astype(float)
    conviction = pd.to_numeric(df["whale_conviction"], errors="coerce").astype(float)
    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    price_dir = np.sign(close.diff()).fillna(0.0)
    whale_strength = (ratio - 1.48) * 5.0
    whale_dir = whale_strength * (1.0 + conviction.abs())
    disagree = (price_dir * whale_dir) < 0
    sig = np.where(disagree, whale_dir.clip(-1, 1), (whale_dir * 0.3).clip(-1, 1))
    sig = pd.Series(sig, index=df.index)
    sig.iloc[0] = 0.0
    out["sig_whale"] = sig.fillna(0.0)

    oi_change = pd.to_numeric(df["oi_change_rate"], errors="coerce").astype(float)
    log_ret = pd.to_numeric(df["log_return"], errors="coerce").astype(float)
    trade_int = pd.to_numeric(df["trade_intensity"], errors="coerce").astype(float)
    active = oi_change.abs() > 0.002
    cs = active & (log_ret < -0.0005) & (oi_change > 0)
    cl = active & (log_ret > 0.0005) & (oi_change > 0)
    co = active & ~cs & ~cl
    s2 = pd.Series(0.0, index=df.index)
    s2[cs] = (0.5 * (oi_change * 100.0) * trade_int)[cs].clip(0, 1)
    s2[cl] = (-0.5 * (oi_change * 100.0) * trade_int)[cl].clip(-1, 0)
    s2[co] = np.sign(log_ret[co]) * 0.2
    out["sig_oi_divergence"] = s2.fillna(0.0)
    return pd.DataFrame(out, index=df.index)


def fix_features_file(path: Path, ref: pd.DataFrame, *, raw_cols: list[str], with_m4: bool) -> pd.DataFrame:
    print(f"\n=== {path.name} ===")
    df = pd.read_csv(path, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    n0, cols0 = len(df), df.columns.tolist()

    rec0 = recompute_derived(df)
    gates = [fidelity_gate(df, rec0, "표준17")]
    if with_m4:
        gates.append(fidelity_gate(df, metrics4_extras(df), "metrics4"))
    if not all(gates):
        raise SystemExit(f"{path.name}: 수식 복제 게이트 실패 — 중단")
    print("  게이트 통과 ✓")

    df, stats = replace_raw(df, ref, raw_cols)
    print("  raw 교체:", {k: v["replaced_diff"] for k, v in stats.items()})
    rec1 = recompute_derived(df)
    for c in rec1.columns:
        if c in df.columns:
            df[c] = rec1[c].to_numpy()
    if with_m4:
        m4 = metrics4_extras(df)
        for c in m4.columns:
            if c in df.columns:
                df[c] = m4[c].to_numpy()
    assert df.columns.tolist() == cols0 and len(df) == n0
    save_atomic(path, df)
    print("  ✓ 저장")
    return df


def main() -> int:
    for sym, prefix in [("BTCUSDT", "btc"), ("SOLUSDT", "sol")]:
        ref = load_ref(sym)

        fixed = fix_features_file(YEAR / f"{prefix}_features_2024_2026.csv", ref, raw_cols=RAW3, with_m4=False)

        # 연도 파일 재슬라이스
        for ypath in [YEAR / f"{prefix}_features_2024.csv", YEAR / f"{prefix}_features_2025.csv", YEAR / f"{prefix}_features_2026.csv"]:
            ydf = pd.read_csv(ypath, usecols=["timestamp"])
            ydf["timestamp"] = pd.to_datetime(ydf["timestamp"])
            t0, t1, n = ydf["timestamp"].min(), ydf["timestamp"].max(), len(ydf)
            sl = fixed[(fixed["timestamp"] >= t0) & (fixed["timestamp"] <= t1)].reset_index(drop=True)
            assert len(sl) == n, (ypath.name, len(sl), n)
            save_atomic(ypath, sl)
            print(f"  ✓ {ypath.name} 재슬라이스({n}행)")

        # raw_frame: raw 컬럼만
        rf = YEAR / f"{prefix}_raw_frame_2024_2026.csv"
        df = pd.read_csv(rf, low_memory=False)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df, stats = replace_raw(df, ref, RAW3)
        save_atomic(rf, df)
        print(f"  ✓ {rf.name} raw 교체:", {k: v['replaced_diff'] for k, v in stats.items()})

    # BTC metrics4 계열
    ref_btc = load_ref("BTCUSDT")
    m4_fixed = fix_features_file(YEAR / "btc_features_2024_2026_metrics4_20260802.csv", ref_btc, raw_cols=RAW5, with_m4=True)
    rfm4 = YEAR / "btc_raw_frame_metrics4_2024_2026.csv"
    df = pd.read_csv(rfm4, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    raw_present = [c for c in RAW5 if c in df.columns]
    df, stats = replace_raw(df, ref_btc, raw_present)
    save_atomic(rfm4, df)
    print(f"  ✓ {rfm4.name} raw 교체:", {k: v['replaced_diff'] for k, v in stats.items()})

    # 변형 파일: 계보 게이트 후 join 교체 (BTC 결합본 기준)
    fixed_btc = m4_fixed  # metrics4가 표준 17도 포함하므로 join 소스로 사용
    orig_btc = pd.read_csv((YEAR / "btc_features_2024_2026.csv").with_name("btc_features_2024_2026.csv" + BAK_SUFFIX), low_memory=False)
    orig_btc["timestamp"] = pd.to_datetime(orig_btc["timestamp"])
    AFFECTED = list(recompute_derived(orig_btc.head(600)).columns) + RAW3  # 컬럼명 목록만 사용
    for vname in ["btc_features_2024_2026_regimeline.csv", "btc_features_2025_swingtransition.csv",
                  "btc_features_2025_swingtransition_zigzag.csv", "btc_features_2026_swingtransition.csv",
                  "btc_features_2026_swingtransition_zigzag.csv", "btc_features_1h_full_2024_2026.csv"]:
        vpath = YEAR / vname
        vdf = pd.read_csv(vpath, low_memory=False)
        vdf["timestamp"] = pd.to_datetime(vdf["timestamp"])
        hit = [c for c in AFFECTED if c in vdf.columns]
        # 계보 게이트: 표본 2000행에서 영향컬럼이 원본 결합본과 동일한가
        probe = vdf.sample(n=min(2000, len(vdf)), random_state=7).merge(
            orig_btc[["timestamp"] + hit], on="timestamp", suffixes=("", "_orig"))
        lineage_ok = True
        for c in hit:
            a = pd.to_numeric(probe[c], errors="coerce")
            b = pd.to_numeric(probe[c + "_orig"], errors="coerce")
            if len(probe) and (((a - b).abs() > 1e-9).mean() > 0.01):
                lineage_ok = False
                print(f"  ⚠️ {vname}: {c} 계보 불일치 — 이 파일은 결합본 계보 아님, 스킵")
                break
        if not lineage_ok or len(probe) == 0:
            if len(probe) == 0:
                print(f"  ⚠️ {vname}: timestamp 겹침 없음 — 스킵")
            continue
        m = vdf[["timestamp"]].merge(fixed_btc[["timestamp"] + hit], on="timestamp", how="left")
        n_changed = 0
        for c in hit:
            newv = pd.to_numeric(m[c], errors="coerce")
            old = pd.to_numeric(vdf[c], errors="coerce")
            mask = newv.notna()
            n_changed += int(((old - newv).abs() > 1e-12)[mask].sum())
            vdf.loc[mask, c] = newv[mask].to_numpy()
        save_atomic(vpath, vdf)
        print(f"  ✓ {vname}: {len(hit)}컬럼 join 교체, 변경 {n_changed}셀")
    return 0


if __name__ == "__main__":
    sys.exit(main())
