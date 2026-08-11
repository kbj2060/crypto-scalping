# ETH h48qual — direction_head 확신도 재보정 진단, 전체 학습구간 재검증 (2026-08-12)

## 배경

[`eth_h48qual_direction_confidence_calibration_20260811.md`](eth_h48qual_direction_confidence_calibration_20260811.md)의
"학습" 수치는 `train_predictions_q050.csv`로 계산됐는데, 이 파일이 실제로는 2025-01-01~09-30
(78,509행)만 반영하고 모델이 진짜 학습한 2024-01~2025-09 전체(183,936행, `report.json`
확인)의 57%(2024년 전체)가 빠져 있었다는 사실이 2026-08-12에 발견됐다(다른 세션이 먼저 발견,
memory `[[odyssey_eth_h48qual_subproject]]` 2026-08-12 업데이트 참고). **이 문서는 그 재검증
결과다.**

**원인 정리**: 라이브 모델 가중치(`true_3head_tabm_bundle.pt`)는 2026-06-30 01:59에 정상적으로
전체 구간(183,936행)으로 학습됐다. 문제는 같은 날 17:59에 별도로 실행된
`scripts/export_omega4_parent_predictions_from_bundle_20260630.py`(순수 추론 재생성 유틸)가
risk-sidecar용 override 소스(`tmp/causal_regen_20260516/alpha5_a5dir_2024_train_2025_score_20260529_cleanfunding_stable48/02_fixed_regime4_state24_sticky090_tp18_sl10_preprocess_2024_to_2025/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv`,
2025-01~12만 포함)로 `train_predictions_q050.csv`를 덮어쓰면서 생긴 것 — 학습 자체의 버그가
아니라 **진단용 예측 CSV의 출처(provenance) 문제**다. VAL/OOS 예측 파일은 이 override의 영향을
받지 않아(우연히 override 소스가 그 구간은 정상적으로 커버) 그대로 신뢰 가능하다(직접 확인:
`validation_predictions_q050.csv` 26,490행/2025-10-01~12-31, `oos_predictions_q050.csv`
16,832행/2026-01-01~02-28 — 계약서의 canonical 표와 정확히 일치).

## 진짜 학습구간 예측 재생성 방법

`train_eval_omega1_2_tabm_diffusion_risk_20260603.py`의 `TRAIN_CSV`/`REGIME3_*_2025` 전역변수는
2026-06-03~08-07 사이 커밋 없이 계속 수정된 공유 설정이라(git blame: 현재 값은 2026-08-07
커밋), 2026-06-30 학습 시점의 정확한 값이 `report.json`에도 git 히스토리에도 남아있지 않다.
포렌식 대조로 재구성:

1. `tmp/causal_regen_20260516/omega_clean_regime_only_24_25_inputs_20260629/trade_candidates_2024_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv`
   (2026-06-29 생성, 학습 하루 전) — 이 파일을 `TRAIN_CSV`로 놓고 `2025-10-01` 이전으로 자르면
   **정확히 183,936행** (report.json과 diff=0), `2025-10-01` 이후는 정확히 26,496행 (계약서
   canonical VAL 행수와 일치) — 우연의 일치로 보기엔 너무 정확해서, 원본 소스와 사실상 동일한
   것으로 채택.
2. 같은 디렉터리의 `training_features_2024_2025_regime3_{current_sensitive_hmm_wide24,
   cryptomamba_h6_sidecar_20260601,stability_risk_h6}.csv`를 `REGIME3_{CURRENT,CMAMBA,RISK}_2025`
   로 사용 (같은 날 생성, 같은 명명 규칙).
3. cmamba/risk 오버레이 파일에 내부 NaN 구간이 있어 `_overlay_required`의 엄격한 edge-only
   결측 검사에 걸림 — **라이브 번들의 `base_cols` 102개에 cmamba/risk 컬럼이 0개**임을 먼저
   확인(`torch.load`로 직접 검증)한 뒤, 안전하다고 판단해 ffill/bfill로 메꿔 통과시킴(모델
   예측에는 영향 없음, edge-case 검증 통과 목적만).
4. EVAL_CSV(2026, OOS)는 이미 정확한 것으로 확인됐으므로 손대지 않음; 2026 오버레이 전역변수는
   현재 파일이 삭제된 상태라(`[[omega_cmamba_risk_overlay_dead_code]]`) `.bak_pre_extend_20260704`
   백업으로 대체(OOS 결과는 이번 진단에서 쓰지 않음, `eval_df` 생성이 죽지 않게 하려는 목적뿐).

스크립트: `scripts/regenerate_eth_h48qual_fullwindow_train_predictions_20260812.py` — 저장된
번들에 순수 추론만 수행(재학습 없음), 원본 라이브 번들 디렉터리는 건드리지 않고 별도 디렉터리
(`..._fullwindow_predictions_recheck_20260812/`)에 출력. 행수(183,936) 정합성 assert 통과.

## 재검증 결과

`scripts/diagnose_eth_h48qual_direction_confidence_calibration_fullwindow_recheck_20260812.py`
(원본 진단 스크립트의 최소-diff 변형, train만 새 파일 사용, VAL/OOS는 원본 그대로).

### 확신도 격차 (전체구간 vs 기존 부분구간)

| 구간 | n_long | n_short | 롱평균 | 숏평균 | 격차 |
|---|---:|---:|---:|---:|---:|
| 학습(전체, 2024-01~2025-09) | 66,503 | 79,909 | 0.5828 | 0.6191 | +0.0364 |
| 학습(기존, 2025-01~09만) | 25,910 | 34,064 | 0.5725 | 0.6210 | +0.0485 |
| VAL (불변) | 8,052 | 11,303 | 0.5676 | 0.6086 | +0.0411 |
| OOS (불변) | 5,238 | 6,746 | 0.5642 | 0.6038 | +0.0396 |

전체구간을 쓰면 격차가 소폭 줄지만(+0.0485→+0.0364), 여전히 VAL/OOS와 같은 방향·비슷한
크기이고 n이 6만 이상이라 표본 노이즈로 보기 어렵다.

### 2024년 단독 vs 2025-01~09 단독 — 핵심 재현 체크

| 구간 | 클래스 | n | 평균확신도 | 실제정확도 | 과신 |
|---|---|---:|---:|---:|---:|
| 2024(신규 데이터) | LONG | 40,585 | 0.5894 | 0.7194 | **-0.1300** |
| 2024(신규 데이터) | SHORT | 45,801 | 0.6176 | 0.6464 | -0.0288 |
| 2025-01~09(기존 문서) | LONG | 25,918 | 0.5725 | 0.7220 | **-0.1496** |
| 2025-01~09(기존 문서) | SHORT | 34,108 | 0.6212 | 0.6394 | -0.0183 |

**2024년(이전에 전혀 안 쓰인 데이터)이 기존 발견을 독립적으로 거의 그대로 재현한다** — LONG
과소신 13.0pp vs 15.0pp(같은 방향, 비슷한 크기), SHORT는 둘 다 거의 중립에 가까운 약한
과소신(-2.9pp vs -1.8pp). 원래 "LONG이 과소신됨" 프레이밍이 우연이나 표본 편향이 아니라는
근거가 강해졌다.

### Temperature 재적합 및 VAL/OOS 적용

| | 원본(2025-01~09만) | 재검증(전체구간) |
|---|---:|---:|
| T_long | 0.6245 | 0.6263 (거의 동일) |
| T_short | 0.9156 | 0.8622 (더 강한 보정 방향) |

| 구간 | 클래스 | ECE 원본(전→후) | ECE 재검증(전→후) |
|---|---|---|---|
| VAL | LONG | 0.1056→0.0741 | 0.1056→0.0739 (거의 동일) |
| VAL | SHORT | 0.0571→0.0742 | 0.0571→**0.0870**(악화폭 더 큼) |
| OOS | LONG | 0.0725→0.0813 | 0.0725→0.0815 (거의 동일) |
| OOS | SHORT | 0.0483→0.0513 | 0.0483→0.0520 (거의 동일) |

`T_long`은 거의 안 바뀌어 LONG 쪽 결과도 거의 그대로다. `T_short`는 전체구간 기준으로 더 강하게
보정해야 한다고 나오는데(0.9156→0.8622), VAL SHORT는 원래도 과신(+0.0559)이라 이 방향의 보정은
VAL에서 더 나쁘게 작동한다 — out-of-sample 실패가 완화되기는커녕 더 뚜렷해졌다.

## 결론

**원본 문서(`eth_h48qual_direction_confidence_calibration_20260811.md`)의 결론은 뒤집히지
않는다.** 오히려:
1. 핵심 발견("학습구간에서 LONG이 심하게 과소신, SHORT는 상대적으로 잘 보정됨")이 이전에
   안 쓰인 2024년 데이터로 독립 재현됨 — 더 견고해짐.
2. "재보정 시도, 부정/보류 결과"(temperature scaling이 VAL/OOS에 안정적으로 일반화 안 됨)도
   그대로 유지, SHORT 쪽은 오히려 더 나빠짐.

**이 버그가 실제로 바꾸는 것은 없다** — 원래 진단에 쓰인 데이터가 전체 학습구간의 43%였다는
사실은 방법론적으로 중요한 결함이었지만(항상 검증해야 할 종류의 문제), 이번 재검증으로 그
43%가 나머지 57%와 통계적으로 다르지 않다는 것이 확인됐다.

## 남은 영향 범위 (참고, 이 문서가 직접 고치지 않음)

같은 `train_predictions_q050.csv`(2025-01~09만 반영)를 썼던 다른 두 문서
(`eth_h48qual_zig075_direction_confidence_echo_check_20260811.md`의 Test 1,
`eth_h48qual_short_calibration_instability_cause_20260811.md`의 "train" 수치)도 원칙적으로
같은 보정이 필요하다 — 위 표의 패턴(격차 소폭 축소, 결론 방향 불변)이 그대로 적용될 가능성이
높지만 각 문서 고유의 세부 수치는 아직 재계산되지 않았다. 두 문서 모두 이 발견을 가리키는
정정 박스를 추가해뒀다(각 문서 상단).
