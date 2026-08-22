#!/usr/bin/env python3
"""154피쳐 데이터셋의 BTC-metrics 오염 패치 (2026-08-23).

배경: 캐노니컬 2026 수정(fix_eth_canonical_2026_btc_metrics_contamination_20260823.py)과
wide24 오버레이 재생성(regen_wide24_overlays_post_metrics_fix_20260823.py) 이후, 그
캐노니컬로부터 빌드된 154피쳐셋(tmp/ilias_eth_154feature_dataset_20260821/)의 영향 컬럼을
패치한다. 금융ML 12피쳐는 OHLCV만 소비(무영향, 실측 확인).

패치 대상(2026 파일 + combined 파일의 2026-01-20 00:05 이후 행):
- 직접 15컬럼: 수정된 캐노니컬 2026에서 timestamp 조인으로 교체
- regime3 wide24 3컬럼: 재생성된 train 오버레이에서 조인
- combo 7컬럼: 컴포넌트 곱(a*b)으로 재계산 (COMBO_FEATURES json의 a/b 정의 사용)

검증: 오염 이전(01-01~01-19) 구간에서 교체값 == 기존값이어야 함(같은 원천 증명).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "tmp/ilias_eth_154feature_dataset_20260821"
CANON_2026 = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
OVERLAY = ROOT / "tmp/ilias_labellogic_recheck_20260821/train_2024_2026H1_regime3_current_states24_sticky090.csv"
COMBO_JSON = ROOT / "tmp/dc_engineered_feature_specs_20260820/dc_combo_feature_names_20260820.json"

WIN_START = pd.Timestamp("2026-01-20 00:05:00")

DIRECT = ["count_long_short_ratio", "crowded_long_unwind_risk", "crowded_short_squeeze_risk",
          "crowding_pressure", "execution_quality", "kel", "mta_funding", "ofti",
          "oi_up_price_down", "oi_up_price_up", "smart_money_flow", "squeeze_power",
          "sum_open_interest_value", "whale_conviction", "whale_retail_ratio"]
REGIME = ["regime3_current_sensitive_wide24_bull_prob", "regime3_current_sensitive_wide24_bear_prob",
          "regime3_current_sensitive_wide24_confidence"]
AFFECTED_SET = set(DIRECT)


def main() -> int:
    canon = pd.read_csv(CANON_2026, usecols=["timestamp"] + DIRECT, low_memory=False)
    canon["timestamp"] = pd.to_datetime(canon["timestamp"])
    ov = pd.read_csv(OVERLAY, usecols=["timestamp"] + REGIME)
    ov["timestamp"] = pd.to_datetime(ov["timestamp"])
    combos_all = json.loads(COMBO_JSON.read_text())
    combos = [c for c in combos_all if c["a"] in AFFECTED_SET or c["b"] in AFFECTED_SET]
    print(f"영향 combo {len(combos)}개:", [c["name"] for c in combos])

    for fname in ["ilias_eth_154feature_2026.csv", "ilias_eth_154feature_2024_2026H1_combined.csv"]:
        path = DS / fname
        df = pd.read_csv(path, low_memory=False)
        tcol = "timestamp" if "timestamp" in df.columns else df.columns[0]
        df[tcol] = pd.to_datetime(df[tcol])
        n_orig, cols_orig = len(df), df.columns.tolist()
        win = df[tcol] >= WIN_START
        print(f"\n{fname}: rows={n_orig}, 패치 대상 {win.sum()}행")

        m_canon = df[[tcol]].merge(canon, left_on=tcol, right_on="timestamp", how="left")
        m_ov = df[[tcol]].merge(ov, left_on=tcol, right_on="timestamp", how="left")

        # 검증: 오염 이전(2026-01-01~01-19) 구간에서 신규값 == 기존값
        pre = (df[tcol] >= "2026-01-01") & (df[tcol] < WIN_START)
        bad = []
        for c in DIRECT:
            if c not in df.columns:
                continue
            a = pd.to_numeric(df.loc[pre, c], errors="coerce")
            b = pd.to_numeric(m_canon.loc[pre, c], errors="coerce")
            frac = (((a - b).abs() / a.abs().clip(lower=1e-6)) > 1e-6).mean()
            if frac > 0.005:
                bad.append((c, frac))
        if bad:
            print(f"  ✗ 오염 이전 구간 원천 불일치: {bad} — 이 파일은 캐노니컬과 다른 원천, 중단")
            return 1
        print(f"  검증: 오염 이전 {pre.sum()}행에서 직접컬럼 원천 일치 ✓")

        changed = {}
        for c in DIRECT:
            if c not in df.columns:
                continue
            newv = pd.to_numeric(m_canon[c], errors="coerce")
            old = pd.to_numeric(df[c], errors="coerce")
            mask = win & newv.notna()
            changed[c] = int(((old - newv).abs() > 1e-12)[mask].sum())
            df.loc[mask, c] = newv[mask].to_numpy()
        for c in REGIME:
            if c not in df.columns:
                continue
            newv = pd.to_numeric(m_ov[c], errors="coerce")
            old = pd.to_numeric(df[c], errors="coerce")
            mask = win & newv.notna()
            changed[c] = int(((old - newv).abs() > 1e-12)[mask].sum())
            df.loc[mask, c] = newv[mask].to_numpy()
        for c in combos:
            name = c["name"]
            if name not in df.columns:
                continue
            a = pd.to_numeric(df[c["a"]], errors="coerce") if c["a"] in df.columns else pd.to_numeric(m_canon[c["a"]], errors="coerce")
            b = pd.to_numeric(df[c["b"]], errors="coerce") if c["b"] in df.columns else pd.to_numeric(m_canon[c["b"]], errors="coerce")
            newv = a * b
            old = pd.to_numeric(df[name], errors="coerce")
            changed[name] = int(((old - newv).abs() > 1e-12)[win].sum())
            df.loc[win, name] = newv[win].to_numpy()
        print("  변경 행수:", {k: v for k, v in changed.items() if v})

        assert df.columns.tolist() == cols_orig and len(df) == n_orig
        bak = path.with_name(path.name + ".bak_pre_btc_metrics_fix_20260823")
        if not bak.exists():
            shutil.copy2(path, bak)
        tmp = path.with_suffix(".csv.tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(path)
        print(f"  ✓ {fname} 패치 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
