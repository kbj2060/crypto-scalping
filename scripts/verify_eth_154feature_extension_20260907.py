#!/usr/bin/env python3
"""154피쳐 연장본 검증 -- **겹치는 구간이 정본과 같은가** (2026-09-07).

연장본(`tmp/ilias_eth_154feature_dataset_extended_20260907/`)은 재생성 오버레이를 물려
새로 빌드했다. 그것이 정본(`tmp/ilias_eth_154feature_dataset_20260821/`, 08-23 패치 2건
반영)과 **겹치는 구간(2024-01-01 ~ 2026-06-30)에서 같아야** 연장분을 이어 붙일 수 있다.

다르면: 연장분만 다른 방법론이 되므로 **못 쓴다**.
(08-20자 기존 연장본이 정확히 이 이유로 탈락했다 -- 4컬럼 최대 17.68% 행 불일치)

154컬럼 전부를 청크로 읽어 대조한다.
판정 임계는 **최대절대오차 1e-6** -- 금융ML 롤링 재계산의 부동소수 잡음(실측 ~3e-8)은
통과시키고 실제 계보 차이(1차 시도의 레짐 컬럼 0.58)는 잡는다. 1e-9 초과 컬럼도
참고로 전부 보고한다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "tmp/ilias_eth_154feature_dataset_20260821/ilias_eth_154feature_2024_2026H1_combined.csv"
EXT = ROOT / "tmp/ilias_eth_154feature_dataset_extended_20260907/ilias_eth_154feature_2024_2026H1_combined.csv"
OUT = ROOT / "tmp/eth_154feature_extension_verify_20260907"
TOL = 1e-9        # 보고 임계
FAIL_TOL = 1e-6   # 판정 임계 (부동소수 잡음 허용)
CHUNK = 25


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cols = pd.read_csv(CANON, nrows=0).columns.tolist()
    ecols = pd.read_csv(EXT, nrows=0).columns.tolist()
    print(f"[컬럼] 정본 {len(cols)} · 연장본 {len(ecols)} · 동일 순서 {cols == ecols}", flush=True)
    assert set(cols) == set(ecols), f"컬럼 집합 불일치: {set(cols) ^ set(ecols)}"

    feats = [c for c in cols if c != "timestamp"]
    bad, worst_all = [], 0.0
    for i in range(0, len(feats), CHUNK):
        grp = feats[i:i + CHUNK]
        a = pd.read_csv(CANON, usecols=["timestamp"] + grp)
        b = pd.read_csv(EXT, usecols=["timestamp"] + grp)
        a["timestamp"] = pd.to_datetime(a.timestamp); b["timestamp"] = pd.to_datetime(b.timestamp)
        if i == 0:
            print(f"[행] 정본 {len(a):,} [{a.timestamp.min()}..{a.timestamp.max()}]", flush=True)
            print(f"[행] 연장본 {len(b):,} [{b.timestamp.min()}..{b.timestamp.max()}]", flush=True)
        m = a.merge(b, on="timestamp", suffixes=("_A", "_B"))
        for c in grp:
            x = pd.to_numeric(m[c + "_A"], errors="coerce").to_numpy(float)
            y = pd.to_numeric(m[c + "_B"], errors="coerce").to_numpy(float)
            nan_mismatch = int((np.isnan(x) != np.isnan(y)).sum())
            ok = np.isfinite(x) & np.isfinite(y)
            d = np.abs(x[ok] - y[ok]) if ok.any() else np.array([0.0])
            worst_all = max(worst_all, float(d.max()))
            n_diff = int((d > TOL).sum())
            if n_diff or nan_mismatch:
                bad.append({"col": c, "n_diff": n_diff, "frac": n_diff / max(ok.sum(), 1),
                            "max_abs": float(d.max()), "nan_mismatch": nan_mismatch})
        print(f"  ...{min(i+CHUNK, len(feats))}/{len(feats)} 컬럼 대조 · 누적 불일치 {len(bad)}", flush=True)
        n_overlap = len(m)

    real = [r for r in bad if r["max_abs"] > FAIL_TOL or r["nan_mismatch"]]
    res = {"n_overlap_rows": int(n_overlap), "n_features": len(feats),
           "report_tol": TOL, "fail_tol": FAIL_TOL,
           "n_cols_over_report_tol": len(bad), "n_cols_over_fail_tol": len(real),
           "worst_abs_diff": worst_all, "over_report_tol": bad[:20], "over_fail_tol": real,
           "PASS": len(real) == 0}
    (OUT / "verify.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("\n" + "=" * 88, flush=True)
    print(f"겹치는 행 {n_overlap:,} · 컬럼 {len(feats)} · 허용오차 {TOL}", flush=True)
    print(f"1e-9 초과 {len(bad)}개 · 판정임계 1e-6 초과 {len(real)}개 "
          f"· 전체 최대오차 {worst_all:.3e}", flush=True)
    for r in bad[:10]:
        mark = "🔴" if r["max_abs"] > FAIL_TOL or r["nan_mismatch"] else "  부동소수"
        print(f"   {mark}{r['col']:<34} 다른행 {r['n_diff']:,} ({r['frac']:.2%}) "
              f"· 최대차 {r['max_abs']:.6g} · NaN불일치 {r['nan_mismatch']}", flush=True)
    print(f"\n⇒ {'✅PASS — 연장분을 이어 붙여도 된다' if res['PASS'] else '🔴FAIL — 이 연장본은 쓰면 안 된다'}",
          flush=True)
    return 0 if res["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
