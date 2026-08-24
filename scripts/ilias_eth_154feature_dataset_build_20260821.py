#!/usr/bin/env python3
"""일리아스 이관용 최종 154피쳐 데이터셋 구축 -- 2024-01-01~2026-06-30 전 구간.

154 = VIF-clean 112(regime3_current_sensitive_wide24_{bull_prob,bear_prob,confidence} 3개
포함, chop_prob/entropy/margin은 이미 제거된 원본 그대로) + RIT조합 30 + financial-ML 12.
2024 regime3_current sidecar는 이번 세션에 재생성해 정식 경로에 저장 완료
(`eth_regime3_current_2024_training_data_compatibility_20260821` 메모리 참고, 안전성 검증됨).

2026은 06-30까지만 자른다(REGIME3_CURRENT 필터가 아니라 직접 날짜 필터 -- regime3_current를
쓰긴 하지만 이제 2024/2025/2026 전부 실데이터라 필터링 우회 사유가 없음, raw 커버리지 자체가
2026-07-20까지라 06-30 컷은 순수 사용자 지시).

combo/financial-ML 피쳐는 2024+2025+2026 전체를 하나로 이어붙인 연속 시계열 위에서 계산한다
(연도별로 따로 계산하면 각 연도 시작부에 불필요한 rolling-window 워밍업 NaN이 매번 생김)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import eth_dc_financial_ml_feature_construction_20260820 as finml  # noqa: E402

SCRATCH = Path("/tmp/claude-1000/-home-kbj20-crypto-scalping/7445be14-7df6-4085-bc4a-6a5de4e4597d/scratchpad")
OUT_DIR = ROOT / "tmp/ilias_eth_154feature_dataset_20260821"
DATE_END = "2026-06-30 23:55:00"

VIF_CLEAN_112 = json.loads((SCRATCH / "dc_vif_clean_features_20260820.json").read_text())
COMBO_FEATURES = json.loads((SCRATCH / "dc_combo_feature_names_20260820.json").read_text())
FINML_NAMES = json.loads((SCRATCH / "dc_financial_ml_feature_names_20260820.json").read_text())
assert len(VIF_CLEAN_112) == 112 and len(COMBO_FEATURES) == 30 and len(FINML_NAMES) == 12
FINAL_154 = sorted(VIF_CLEAN_112) + sorted(c["name"] for c in COMBO_FEATURES) + sorted(FINML_NAMES)
assert len(FINAL_154) == 154

REGIME_OVERLAY = {
    2024: ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2024_regime3_current_sensitive_hmm_wide24.csv",
    2025: ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2025_regime3_current_sensitive_hmm_wide24.csv",
    2026: ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv",
}
BASE_FILES = {
    2024: ROOT / "data/splits/year_oos/training_features_2024.csv",
    2025: ROOT / "data/splits/year_oos/training_features_2025.csv",
    2026: ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
}
REGIME3_CURRENT_COLS = [
    "regime3_current_sensitive_wide24_bull_prob", "regime3_current_sensitive_wide24_bear_prob",
    "regime3_current_sensitive_wide24_chop_prob", "regime3_current_sensitive_wide24_confidence",
    "regime3_current_sensitive_wide24_entropy", "regime3_current_sensitive_wide24_margin",
]


def _load_year(year: int) -> pd.DataFrame:
    base = pd.read_csv(BASE_FILES[year], low_memory=False)
    base["timestamp"] = pd.to_datetime(base["timestamp"])
    overlay = pd.read_csv(REGIME_OVERLAY[year], usecols=["timestamp"] + REGIME3_CURRENT_COLS, parse_dates=["timestamp"])
    merged = base.merge(overlay, on="timestamp", how="inner")
    if year in (2024, 2025) and len(merged) != len(base):
        # 2024/2025 overlay는 base와 완전일치해야 정상(전량 재생성/기존 검증됨) -- fail-loud
        raise RuntimeError(f"{year}: regime overlay merge lost rows ({len(merged)} vs {len(base)})")
    if year == 2026:
        # 2026 overlay 자체가 이미 51,746행(2026-01-01~06-30)으로 자연스럽게 끝남 -- 사용자가
        # 요청한 "6월까지" 컷과 거의 정확히 일치, base(57,601행, 07-20까지)와는 원래부터
        # 의도적으로 다른 범위라 완전일치를 요구하지 않는다(아래 DATE_END 필터로 한번 더 확정).
        print(f"  2026 overlay natural range: {overlay['timestamp'].min()}..{overlay['timestamp'].max()} "
              f"({len(overlay)} rows) -- base has {len(base)}, using overlay's natural intersection", flush=True)
    return merged


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("loading + merging regime overlay per year...", flush=True)
    frames = []
    for year in (2024, 2025, 2026):
        f = _load_year(year)
        if year == 2026:
            f = f[f["timestamp"] <= DATE_END].reset_index(drop=True)
        print(f"  {year}: {len(f)} rows [{f['timestamp'].min()}..{f['timestamp'].max()}]", flush=True)
        frames.append(f)

    full = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    dupe = full["timestamp"].duplicated().sum()
    if dupe:
        raise RuntimeError(f"duplicate timestamps after concat: {dupe}")
    print(f"concatenated: {len(full)} rows [{full['timestamp'].min()}..{full['timestamp'].max()}]", flush=True)

    missing_base = set(VIF_CLEAN_112) - set(full.columns)
    if missing_base:
        raise RuntimeError(f"VIF-clean base features missing from concatenated frame: {missing_base}")

    print("computing 30 combo features...", flush=True)
    for c in COMBO_FEATURES:
        full[c["name"]] = pd.to_numeric(full[c["a"]], errors="coerce") * pd.to_numeric(full[c["b"]], errors="coerce")

    print("computing 12 financial-ML features...", flush=True)
    finml_feats = finml.build_financial_ml_features(full)
    for name, arr in finml_feats.items():
        full[name] = arr

    missing_new = (set(FINAL_154) - set(full.columns))
    if missing_new:
        raise RuntimeError(f"engineered columns missing after construction: {missing_new}")

    final = full[["timestamp"] + FINAL_154].copy()
    if len(final.columns) != 155:  # timestamp + 154
        raise RuntimeError(f"expected 155 cols (timestamp+154), got {len(final.columns)}")

    print("\n=== completeness audit (154 features) ===", flush=True)
    nan_counts = final[FINAL_154].isna().sum()
    total_rows = len(final)
    worst = nan_counts.sort_values(ascending=False)
    print(f"total rows: {total_rows}")
    print(f"features with 0 NaN: {int((nan_counts == 0).sum())}/154")
    nonzero = worst[worst > 0]
    if len(nonzero):
        print(f"features with >0 NaN ({len(nonzero)}):")
        for name, cnt in nonzero.items():
            first_valid = final.loc[final[name].notna(), "timestamp"].min()
            print(f"  {name}: {cnt} NaN ({cnt/total_rows*100:.3f}%), first valid={first_valid}")
    else:
        print("  none -- fully complete")

    # save per-year + combined
    for year, sub in [(2024, final[final["timestamp"].dt.year == 2024]),
                       (2025, final[final["timestamp"].dt.year == 2025]),
                       (2026, final[final["timestamp"].dt.year == 2026])]:
        p = OUT_DIR / f"ilias_eth_154feature_{year}.csv"
        sub.to_csv(p, index=False)
        print(f"saved {p} ({len(sub)} rows)", flush=True)

    combined_path = OUT_DIR / "ilias_eth_154feature_2024_2026H1_combined.csv"
    final.to_csv(combined_path, index=False)
    print(f"saved {combined_path} ({len(final)} rows)", flush=True)

    manifest = {
        "feature_count": 154,
        "feature_list": FINAL_154,
        "date_range": [str(final["timestamp"].min()), str(final["timestamp"].max())],
        "total_rows": int(len(final)),
        "rows_per_year": {y: int((final["timestamp"].dt.year == y).sum()) for y in (2024, 2025, 2026)},
        "source_base_files": {str(y): str(p) for y, p in BASE_FILES.items()},
        "source_regime_overlay_files": {str(y): str(p) for y, p in REGIME_OVERLAY.items()},
        "combo_feature_definitions": COMBO_FEATURES,
        "financial_ml_feature_names": FINML_NAMES,
        "nan_audit": {name: int(cnt) for name, cnt in nan_counts.items() if cnt > 0},
        "note_2024_regime3_current": "regenerated 2026-08-21 from existing fitted joblib (no refit) -- see eth_regime3_current_2024_training_data_compatibility_20260821 memory",
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"saved {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
