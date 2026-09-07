#!/usr/bin/env python3
"""154피쳐 데이터셋을 **2026-08-19 까지 연장** (2026-09-07).

사용자: *"154피쳐 빌더 찾아서 8월까지 연장해줘"*

## 왜 필요한가
앵커 방향 축의 완화 기준(풀링) 검정에서 `three/perm20`(풀링 AUC 0.5611 [0.5013, 0.6224],
p=0.035)과 `wbin/perm20`(0.5322 [0.4975, 0.5671], p=0.025)이 처음으로 귀무를 넘었다.
그러나 Bonferroni(3검정, 0.0167)를 못 넘고 상위30% 선별 리프트는 유의하지 않다.
확증하려면 **안 쓴 기간**이 필요한데, 앵커는 2026-07-31 까지 있는 반면
154피쳐 CSV 가 **2026-06-30 에서 끝나** 계산이 안 된다. 이 파일이 그 벽을 없앤다.

## 🔴기존 연장본(2026-08-20자)을 재사용하면 안 된다
`tmp/ilias_eth_154feature_dataset_extended_20260820/` 이 이미 2026-08-19 까지 있지만,
그 뒤로 정본에 **세 번의 수정**이 들어갔다:
  - states24sticky090 재생성 (08-21)
  - BTC 메트릭 오염 수정 (08-23, `fix_eth_154feature_dataset_post_metrics_fix_20260823.py`)
  - 2026-02-28 8시간 갭 수정 (08-23, `fix_eth_154feature_gap_20260823.py`)
실측 대조: 겹치는 262,322행에서 **4개 컬럼이 다르다**
  crowding_pressure(17.68% 행) · count_long_short_ratio(17.67%) ·
  crowded_long_unwind_risk(2.87%) · crowded_short_squeeze_risk(2.76%)
⇒ 그대로 쓰면 2026-07~08 행만 다른 방법론으로 만들어진 셈이다.

## 🔴`DATE_END` 만 바꿔 돌려도 안 된다
빌더가 읽는 `REGIME_OVERLAY[2026]`
(`data/ensemble/supervised/.../training_features_2026_rebuilt_..._wide24.csv`)은
**08-20자 그대로**다. 08-23 재생성은 `tmp/ilias_labellogic_recheck_20260821/` 에 썼고,
정본은 그걸 **사후 패치**로 조인했다(`fix2` 의 `OVERLAY` 상수).
⇒ 재생성 오버레이를 빌더에 직접 물려야 정본과 같은 계보가 된다.

## 이 파일이 하는 일
1. 재생성 오버레이 두 조각을 **전 연도(2024/2025/2026) 공용** 단일 파일로 합친다
   (둘 다 08-23 03:33, 수정 반영):
     train_2024_2026H1_...  2024-01-01 ~ 2026-06-30 23:55  (262,609행)
     oos_20260701_20260819_...  2026-07-01 ~ 2026-08-19 23:55 (14,400행)
   ⚠️1차 시도는 2026 만 교체하고 2024/2025 는 `data/ensemble/...` 원본을 뒀는데
   **검증이 FAIL** 했다: 레짐 3컬럼이 210,481행(=2024+2025 행수 정확히 일치)에서
   최대 0.58 차이. 즉 정본의 레짐 컬럼은 **전 연도가** 08-21 states24/sticky0.90
   재생성본에서 왔다(정본 디렉터리의 `.bak_pre_states24sticky090_regen_20260821` 이 증거).
2. 빌드 모듈의 `REGIME_OVERLAY` 전체 / `DATE_END` / `OUT_DIR` 를 덮어쓰고 `main()` 호출
   -- `ilias_eth_154feature_dataset_extend_20260820.py` 와 같은 관용구, 구성 코드는 무수정.
3. **검증은 별도 스크립트**가 한다: 겹치는 구간(2024-01-01~2026-06-30)이
   정본과 1e-9 이내로 같아야 한다. 다르면 이 연장본은 못 쓴다.

## ⚠️빌더의 하드코딩된 스크래치 경로
빌드 모듈 29~35행이 **원 세션(7445be14…)의 스크래치패드**에서 피쳐 정의 JSON 3개를
**import 시점에** 읽는다. 그 디렉터리는 사라졌다(현재 세션 id 가 다르다).
모듈 속성 오버라이드로는 못 고친다 -- import 전에 평가되기 때문.
⇒ 래퍼가 **그 경로를 만들고 보관본을 복사**한다. 빌더는 한 줄도 안 건드린다
   (`tmp/dc_engineered_feature_specs_20260820/` 에 세 파일이 보관돼 있다).

⚠️출력이 ~680MB 다. 기존 정본을 덮어쓰지 않고 새 디렉터리에 쓴다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

SPECS = ROOT / "tmp/dc_engineered_feature_specs_20260820"
BUILD_SCRATCH = Path("/tmp/claude-1000/-home-kbj20-crypto-scalping/"
                     "7445be14-7df6-4085-bc4a-6a5de4e4597d/scratchpad")
SPEC_FILES = ("dc_vif_clean_features_20260820.json", "dc_combo_feature_names_20260820.json",
              "dc_financial_ml_feature_names_20260820.json")
RECHECK = ROOT / "tmp/ilias_labellogic_recheck_20260821"
TRAIN_OV = RECHECK / "train_2024_2026H1_regime3_current_states24_sticky090.csv"
OOS_OV = RECHECK / "oos_20260701_20260819_regime3_current_states24_sticky090.csv"
OUT_DIR = ROOT / "tmp/ilias_eth_154feature_dataset_extended_20260907"
MERGED_OV = OUT_DIR / "regime3_overlay_all_merged_20260907.csv"
DATE_END = "2026-08-19 23:55:00"


def restore_build_scratch() -> None:
    """빌더가 import 시점에 읽는 하드코딩 경로에 피쳐정의 JSON 3개를 복원."""
    import shutil
    BUILD_SCRATCH.mkdir(parents=True, exist_ok=True)
    for n in SPEC_FILES:
        src = SPECS / n
        if not src.exists():
            raise FileNotFoundError(f"피쳐정의 보관본 없음: {src}")
        dst = BUILD_SCRATCH / n
        if not dst.exists():
            shutil.copy2(src, dst)
    print(f"[스크래치] {BUILD_SCRATCH} 에 정의 {len(SPEC_FILES)}개 복원", flush=True)


def build_merged_overlay() -> Path:
    """2024-01-01 ~ 2026-08-19 전 구간 오버레이 (전 연도 공용)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(TRAIN_OV, parse_dates=["timestamp"])
    b = pd.read_csv(OOS_OV, parse_dates=["timestamp"])
    m = pd.concat([a, b], ignore_index=True).sort_values("timestamp")
    m = m.drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    # 5분 격자 연속성 (fix1 이 고친 2026-02-28 갭이 재생성본에 없는지 확인)
    d = m["timestamp"].diff().dropna()
    gaps = d[d != pd.Timedelta("5min")]
    print(f"[오버레이] {len(m):,}행 · {m.timestamp.min()} ~ {m.timestamp.max()} "
          f"· 5분 아닌 간격 {len(gaps)}건", flush=True)
    if len(gaps):
        for i, g in gaps.head(5).items():
            print(f"    갭 {m.timestamp[i-1]} → {m.timestamp[i]} ({g})", flush=True)
    m.to_csv(MERGED_OV, index=False)
    return MERGED_OV


def main() -> int:
    restore_build_scratch()
    ov = build_merged_overlay()
    import ilias_eth_154feature_dataset_build_20260821 as build_mod

    build_mod.DATE_END = DATE_END
    build_mod.OUT_DIR = OUT_DIR
    build_mod.REGIME_OVERLAY = {y: ov for y in (2024, 2025, 2026)}
    print(f"[빌드] DATE_END={DATE_END} · OUT_DIR={OUT_DIR}", flush=True)
    print(f"       REGIME_OVERLAY[2024/2025/2026] = {ov}  (전 연도 공용)", flush=True)
    return build_mod.main()


if __name__ == "__main__":
    raise SystemExit(main())
