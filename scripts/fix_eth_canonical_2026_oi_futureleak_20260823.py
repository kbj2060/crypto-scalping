#!/usr/bin/env python3
"""ETH 캐노니컬 2026 파일 — 2026-07-12 이후 metrics 1버킷 미래참조 + 07-20~08-02 원시값
스케일 결함 수정 (2026-08-23, 같은 날 3차 발견, 사용자 지시로 즉시 수정).

배경: 오전에 수정한 BTC-오염 창(01-20~07-12)의 fidelity 검증 과정에서, wide24 히스토리
확장용 미니빌더가 2026-07-13 이후 `oi_change_rate`가 참조 아카이브와 어긋난다는 걸 우연히
발견 → 직접 재검증한 결과 원시 3컬럼(sum_open_interest_value/sum_toptrader_long_short_ratio/
count_long_short_ratio) 자체가 **2026-07-12 00:05:00부터 파일 끝(2026-08-19 23:55:00)까지
1버킷 미래참조로 조인**돼 있었다(canon(t) == archive(t+1), 일별 스캔 leak_match 100%/
correct_match 0%). 추가로 **2026-07-20~08-02**는 raw `sum_open_interest_value` 자체가
아카이브 대비 ~1/300로 스케일 다운(day-median ratio 0.003~0.004)돼 있었다 — 별개 결함이나
같은 창 재교체로 동시 해소된다.

이 창은 RDE OOS 평가창(2026-07-01~08-19)과 거의 완전히 겹친다 — wide24 오버레이
oos_20260701_20260819도 재생성 대상.

수정 방법은 오전 BTC-오염 수정과 동일(원시 3컬럼 아카이브 참조본 교체 + 파생 14컬럼
`fix_eth_canonical_2026_btc_metrics_contamination_20260823.py::recompute_derived` 그대로
재사용 + 게이트). 이번엔 오염 창이 파일 끝까지라 "꼬리" 개념이 없다 — 창 자체가
2026-07-12 00:05:00 ~ 파일 끝.

게이트 구간(반드시 원시값 자체가 참조본과 ratio≈1임을 이번에 실측 확인한 구간만 사용):
2026-02-01~02-10, 2026-07-01~07-11 (둘 다 오전 수정으로 이미 검증된 창 내부).
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
REFERENCE = ROOT / "data/TOTAL_ETHUSDT_metrics_2024_2026.csv"
BACKUP = TARGET.with_name(TARGET.name + ".bak_pre_oi_futureleak_fix_20260823")
RAW_COLS = ["sum_open_interest_value", "sum_toptrader_long_short_ratio", "count_long_short_ratio"]

WIN_START = pd.Timestamp("2026-07-12 00:05:00")
WIN_END = None  # 파일 끝까지 (아래서 채움)

CLEAN_ZONES = [("2026-02-01", "2026-02-10"), ("2026-07-01", "2026-07-11")]

_spec = importlib.util.spec_from_file_location(
    "fix_btc_contam", ROOT / "scripts/fix_eth_canonical_2026_btc_metrics_contamination_20260823.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
recompute_derived = _mod.recompute_derived


def main() -> int:
    df = pd.read_csv(TARGET, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    orig_cols = df.columns.tolist()
    n_orig = len(df)
    win_end = df["timestamp"].max()
    print(f"target rows={n_orig} cols={len(orig_cols)}")
    print(f"오염 창: {WIN_START} ~ {win_end} (파일 끝)")

    ref = pd.read_csv(REFERENCE)
    ref["create_time"] = pd.to_datetime(ref["create_time"])
    ref = ref[["create_time"] + RAW_COLS].rename(columns={c: c + "_ref" for c in RAW_COLS})

    win_mask = (df["timestamp"] >= WIN_START) & (df["timestamp"] <= win_end)
    print(f"오염 창 행수: {win_mask.sum()}")

    # 1) 원시 컬럼 교체 (정확 조인 + 결측버킷은 merge_asof backward 9h — 기존 컨벤션)
    merged = df[["timestamp"]].merge(ref, left_on="timestamp", right_on="create_time", how="left")
    asof = pd.merge_asof(
        df[["timestamp"]].sort_values("timestamp"), ref.sort_values("create_time"),
        left_on="timestamp", right_on="create_time", direction="backward", tolerance=pd.Timedelta("9h"),
    )
    n_exact = merged.loc[win_mask, RAW_COLS[0] + "_ref"].notna().sum()
    print(f"원시 교체: 정확일치 {n_exact}/{win_mask.sum()}, 나머지는 asof-backward")
    before = {c: df.loc[win_mask, c].copy() for c in RAW_COLS}
    for c in RAW_COLS:
        exact_vals = merged[c + "_ref"]
        fallback = asof[c + "_ref"]
        newvals = exact_vals.fillna(fallback)
        df.loc[win_mask, c] = newvals[win_mask].to_numpy()
    still_na = df.loc[win_mask, RAW_COLS].isna().sum().sum()
    if still_na:
        print(f"⚠️ 교체 후 NaN {still_na}개 — 중단")
        return 1
    for c in RAW_COLS:
        n_changed = (before[c].to_numpy() != df.loc[win_mask, c].to_numpy()).sum()
        print(f"  {c}: 교체로 변경된 행 {n_changed}/{win_mask.sum()}")

    # 2) 파생 전체 재계산 (수식은 오전 스크립트와 완전 동일 — import로 재사용)
    rec = recompute_derived(df)

    # 3) 검증 게이트: 비오염(이미 검증됨) 구간에서 재계산 == 기존 저장값
    print("\n[검증] 수식복제 정확성 (오전 스크립트와 동일 함수 재사용, 원시값 ratio≈1 확인된 구간):")
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
        print("검증 실패 — 아무것도 쓰지 않고 중단.")
        return 1
    print("  ✓ 전 컬럼 통과(불일치율 ≤0.5%)")

    # 4) 오염 창 전체(파일 끝까지)에 파생값 덮어쓰기
    print(f"\n파생 덮어쓰기 창: {WIN_START} ~ {win_end} ({win_mask.sum()}행)")
    changed_stats = {}
    for col in rec.columns:
        old = pd.to_numeric(df.loc[win_mask, col], errors="coerce")
        new = rec.loc[win_mask, col]
        changed = ((old - new).abs() > 1e-12).sum()
        changed_stats[col] = int(changed)
        df.loc[win_mask, col] = new.to_numpy()
    print("컬럼별 변경 행수:", changed_stats)

    # 5) 백업 + 원자적 쓰기
    if not BACKUP.exists():
        shutil.copy2(TARGET, BACKUP)
        print(f"백업: {BACKUP.name}")
    assert df.columns.tolist() == orig_cols and len(df) == n_orig
    tmp = TARGET.with_suffix(".csv.tmp_fix2")
    df.to_csv(tmp, index=False)
    tmp.replace(TARGET)
    print(f"✓ 수정 완료 → {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
