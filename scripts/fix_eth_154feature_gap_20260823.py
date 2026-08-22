#!/usr/bin/env python3
"""154피쳐셋 2026-02-28 8시간 갭(96행) 수정 (2026-08-23).

원인(이번 조사로 확정): 154 빌드(08-21)가 base(캐노니컬, 갭 없음)와 당시의 구버전
ensemble/supervised wide24 사이드카를 inner 조인했는데, 그 사이드카가 02-28 16:05~23:55
구간이 비어 있었다(현재 사이드카는 이후 재생성돼 갭 없음 — 실측 확인). 즉 캐노니컬의
문제가 아니라 빌드 시점 사이드카 결손이 154셋에 각인된 것.

수정:
1. 결손 96행(2026-02-28 16:05 ~ 23:55)을 삽입 — VIF/직접 컬럼은 수정된 캐노니컬에서,
   regime3 3컬럼은 재생성된 train 오버레이에서, combo는 곱으로, 금융ML 12컬럼은 연속
   시계열 재계산에서 취함.
2. 금융ML 12컬럼은 **2026 전체를 연속 시계열로 재계산해 전 행 덮어씀** — 원본 빌드는
   갭이 있는 시계열 위에서 롤링을 돌려 갭 직후 윈도우 길이만큼 값이 미세 오염돼 있었기
   때문(갭을 건너뛴 롤링). 검증: 갭에서 먼 구간(1월)에서 재계산==기존값 확인.
3. combined 파일에도 동일 적용. manifest에 패치 기록 추가.

참고: 2026 파일이 06-30 00:00에 끝나는 것은 빌드 당시 문서화된 범위 규약(오버레이 자연
경계 수용)이라 결함이 아님 — 건드리지 않는다.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eth_dc_financial_ml_feature_construction_20260820 as finml  # noqa: E402

DS = ROOT / "tmp/ilias_eth_154feature_dataset_20260821"
CANON_2026 = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
OVERLAY = ROOT / "tmp/ilias_labellogic_recheck_20260821/train_2024_2026H1_regime3_current_states24_sticky090.csv"
SPEC = ROOT / "tmp/dc_engineered_feature_specs_20260820"

GAP_START = pd.Timestamp("2026-02-28 16:05:00")
GAP_END = pd.Timestamp("2026-02-28 23:55:00")

REGIME3 = ["regime3_current_sensitive_wide24_bull_prob", "regime3_current_sensitive_wide24_bear_prob",
           "regime3_current_sensitive_wide24_confidence"]


def main() -> int:
    vif112 = json.loads((SPEC / "dc_vif_clean_features_20260820.json").read_text())
    combos = json.loads((SPEC / "dc_combo_feature_names_20260820.json").read_text())
    finml_names = json.loads((SPEC / "dc_financial_ml_feature_names_20260820.json").read_text())

    canon = pd.read_csv(CANON_2026, low_memory=False)
    canon["timestamp"] = pd.to_datetime(canon["timestamp"])
    canon = canon[canon["timestamp"] <= "2026-06-30 23:55:00"].reset_index(drop=True)
    ov = pd.read_csv(OVERLAY, usecols=["timestamp"] + REGIME3)
    ov["timestamp"] = pd.to_datetime(ov["timestamp"])

    # 금융ML 12: 연속(갭 없는) 캐노니컬 2026 위에서 재계산
    fin = finml.build_financial_ml_features(canon)
    fin_df = pd.DataFrame({"timestamp": canon["timestamp"], **fin})

    p2026 = DS / "ilias_eth_154feature_2026.csv"
    df = pd.read_csv(p2026, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cols = df.columns.tolist()
    print(f"2026 파일: {len(df)}행")

    # 검증 1: 갭에서 먼 1월 구간에서 finml 재계산 == 기존값 (수식/원천 동일성 증명)
    jan = (df["timestamp"] >= "2026-01-10") & (df["timestamp"] < "2026-01-20")
    mj = df.loc[jan, ["timestamp"] + finml_names].merge(fin_df, on="timestamp", suffixes=("_old", "_new"))
    worst = 0.0
    for c in finml_names:
        rel = ((mj[c + "_old"] - mj[c + "_new"]).abs() / mj[c + "_old"].abs().clip(lower=1e-9))
        worst = max(worst, float(rel.quantile(0.99)))
    if worst > 1e-6:
        print(f"✗ finml 1월 재계산 불일치 p99={worst:.2e} — 중단")
        return 1
    print(f"  검증: 1월 finml 재계산 일치(p99 상대오차 {worst:.1e}) ✓")

    # 결손 96행 조립
    gap_ts = pd.date_range(GAP_START, GAP_END, freq="5min")
    assert not df["timestamp"].isin(gap_ts).any()
    gap_canon = canon[canon["timestamp"].isin(gap_ts)].set_index("timestamp")
    gap_ov = ov[ov["timestamp"].isin(gap_ts)].set_index("timestamp")
    gap_fin = fin_df[fin_df["timestamp"].isin(gap_ts)].set_index("timestamp")
    n_gap = len(gap_ts)  # 16:05~23:55 = 95행
    assert len(gap_canon) == n_gap and len(gap_ov) == n_gap and len(gap_fin) == n_gap, \
        (len(gap_canon), len(gap_ov), len(gap_fin), n_gap)

    new_rows = pd.DataFrame(index=gap_ts)
    for c in vif112:
        if c in REGIME3:
            new_rows[c] = gap_ov[c]
        elif c in gap_canon.columns:
            new_rows[c] = gap_canon[c]
        else:
            print(f"✗ VIF 컬럼 {c}가 캐노니컬에 없음 — 중단")
            return 1
    for c in combos:
        new_rows[c["name"]] = pd.to_numeric(new_rows.get(c["a"], gap_canon.get(c["a"])), errors="coerce") * \
                              pd.to_numeric(new_rows.get(c["b"], gap_canon.get(c["b"])), errors="coerce")
    for c in finml_names:
        new_rows[c] = gap_fin[c]
    new_rows = new_rows.reset_index().rename(columns={"index": "timestamp"})
    # 154 파일에 있는데 신규행에 없는 컬럼(라벨류 등) 확인
    missing_cols = [c for c in cols if c not in new_rows.columns]
    if missing_cols != []:
        print(f"  ⚠️ 154 파일의 비피쳐 컬럼 {missing_cols} — 캐노니컬에서 보충 시도")
        for c in missing_cols:
            if c in gap_canon.columns:
                new_rows[c] = gap_canon[c].to_numpy()
            else:
                print(f"✗ 보충 불가 컬럼 {c} — 중단")
                return 1
    new_rows = new_rows[cols]
    assert not new_rows.isna().any().any(), "신규 행에 NaN"

    # 병합: 96행 삽입 + finml 전 행 덮어쓰기
    out = pd.concat([df, new_rows], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    m = out[["timestamp"]].merge(fin_df, on="timestamp", how="left")
    n_fin_changed = 0
    for c in finml_names:
        old = pd.to_numeric(out[c], errors="coerce")
        newv = pd.to_numeric(m[c], errors="coerce")
        mask = newv.notna()
        n_fin_changed += int(((old - newv).abs() > 1e-12)[mask].sum())
        out.loc[mask, c] = newv[mask].to_numpy()
    print(f"  96행 삽입 → {len(out)}행, finml 덮어쓴 셀 {n_fin_changed}개")

    for path, frame in [(p2026, out)]:
        bak = path.with_name(path.name + ".bak_pre_gap_fix_20260823")
        if not bak.exists():
            shutil.copy2(path, bak)
        tmp = path.with_suffix(".csv.tmp")
        frame.to_csv(tmp, index=False)
        tmp.replace(path)
        print(f"  ✓ {path.name} 저장")

    # combined 파일: 2026 구간을 새 2026 파일로 교체
    pcomb = DS / "ilias_eth_154feature_2024_2026H1_combined.csv"
    comb = pd.read_csv(pcomb, low_memory=False)
    comb["timestamp"] = pd.to_datetime(comb["timestamp"])
    pre2026 = comb[comb["timestamp"] < "2026-01-01"]
    comb_new = pd.concat([pre2026, out[comb.columns.tolist()]], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    print(f"combined: {len(comb)} → {len(comb_new)}행 (+{len(comb_new)-len(comb)})")
    bak = pcomb.with_name(pcomb.name + ".bak_pre_gap_fix_20260823")
    if not bak.exists():
        shutil.copy2(pcomb, bak)
    tmp = pcomb.with_suffix(".csv.tmp")
    comb_new.to_csv(tmp, index=False)
    tmp.replace(pcomb)
    print(f"  ✓ {pcomb.name} 저장")

    # manifest 패치 기록
    mpath = DS / "manifest.json"
    manifest = json.loads(mpath.read_text())
    manifest["patched_20260823"] = {
        "btc_metrics_contamination_fix": "2026-01-20~07-12 OI/LS cols + derived + regime3 + combos replaced (see fix_eth_154feature_dataset_post_metrics_fix_20260823.py)",
        "gap_fix": "96 rows 2026-02-28 16:05~23:55 inserted (old sidecar defect); finml 12 cols recomputed on continuous series",
        "row_counts_after": {"2026": len(out), "combined": len(comb_new)},
    }
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("  ✓ manifest 갱신")
    return 0


if __name__ == "__main__":
    sys.exit(main())
