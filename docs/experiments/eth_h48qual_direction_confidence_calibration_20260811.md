# ETH h48qual — direction_head 확신도 클래스별 재보정 시도 (2026-08-11)

## 정정 (2026-08-12): "학습" 수치가 실제로는 2025-01~09만 반영했던 문제, 재검증 완료

아래 "결과"의 모든 "학습" 수치는 `train_predictions_q050.csv`로 계산됐는데, 이 파일은
2026-06-30 17:59에 `export_omega4_parent_predictions_from_bundle_20260630.py`가 risk-sidecar용
override 소스(`trade_candidates_...regime4_state24...csv`)로 재생성한 것으로, **2025-01-01~
09-30(78,509행)만 반영하고 2024년 전체(모델이 실제로 학습한 183,936행 중 57%)가 빠져 있었다**
(라이브 모델 가중치 자체는 정상적으로 2024-01~2025-09 전체로 학습됨 — `report.json`의
`label_quality_summary.train.rows=183936`으로 확인됨; 빠진 건 이 진단이 읽은 예측 CSV뿐).

**근본 원인**: `train_eval_omega1_2_tabm_diffusion_risk_20260603.py`의 `TRAIN_CSV`/`REGIME3_*_2025`
전역변수가 코드 커밋 없이(2026-06-03~08-07 사이 비커밋 상태로 계속 수정됨, `git blame` 확인)
자유롭게 바뀌는 공유 설정이라, 2026-06-30 학습 시점에 실제 사용된 파일이 지금은 무엇이었는지
`report.json`에 기록되어 있지 않다(이 자체가 재현성 결함). 포렌식 대조(행수 정확히 일치,
`git blame` 날짜, 디렉터리명 "24_25", overlay 파일 존재 여부)로
`tmp/causal_regen_20260516/omega_clean_regime_only_24_25_inputs_20260629/`(2026-06-29 생성 —
학습 하루 전)가 원본 소스와 사실상 동일함을 확인했다.

**재검증**: `scripts/regenerate_eth_h48qual_fullwindow_train_predictions_20260812.py`로 저장된
번들에 순수 추론(재학습 없음)만 다시 돌려 진짜 전체 구간(183,936행, 2024-01-01~2025-09-30,
`report.json`과 정확히 일치) `train_predictions_q050.csv`를 재생성했고,
`scripts/diagnose_eth_h48qual_direction_confidence_calibration_fullwindow_recheck_20260812.py`로
이 문서의 진단을 다시 돌렸다.

**결과: 핵심 결론은 뒤집히지 않는다.** 오히려 2024년 단독 구간이 기존 2025-01~09 단독 구간과
거의 동일한 패턴을 독립적으로 재현한다 — 2024: LONG 과신=-0.1300(n=40,585), SHORT 과신=-0.0288
(n=45,801) vs 2025-01~09: LONG 과신=-0.1496(n=25,918), SHORT 과신=-0.0183(n=34,108). 즉 "학습
구간에서 LONG이 심하게 과소신, SHORT는 거의 잘 보정됨"이라는 원래 발견이 이전에 안 쓰인 새
데이터(2024년)로 독립 재현된 것 — 우연이나 표본 편향이 아니라는 근거가 오히려 강해졌다.
전체구간 기준 확신도 격차는 소폭 축소(+0.0485→+0.0364, 여전히 n>6만 규모로 뚜렷)됐고,
temperature도 `T_long`은 거의 안 바뀜(0.6245→0.6263)지만 `T_short`은 더 강한 보정 방향으로
움직임(0.9156→0.8622) — VAL SHORT의 out-of-sample 실패는 오히려 더 뚜렷해짐(ECE 0.0571→0.0870,
원래 0.0571→0.0742보다 악화폭 큼). **"재보정 시도, 부정/보류 결과"라는 원래 결론은 그대로
유지되며, 근거는 더 단단해졌다.**

전체 수치·방법론: `docs/experiments/eth_h48qual_direction_confidence_calibration_fullwindow_recheck_20260812.md` 참고.

## 배경

[`eth_h48qual_zig075_direction_confidence_echo_check_20260811.md`](eth_h48qual_zig075_direction_confidence_echo_check_20260811.md):
`quality_head` 게이트가 사실상 `direction_head` confidence 필터(confidence-only top-K가 실제
게이트 결과를 4.3~6.5pp 이내로 재현). [`eth_zigzag_swing_shape_direction_asymmetry_check_20260811.md`](eth_zigzag_swing_shape_direction_asymmetry_check_20260811.md):
숏 스윙이 완결 전 되돌림이 롱보다 작다(실재하지만 효과크기 작음, `|r|≈0.14`) — confidence
격차(3~5pp)의 "지배적 원인이라기엔 부족", 나머지는 "모델 자체의 calibration일 가능성" — 이
문서가 그 미확인 부분을 직접 검증한다.

**중요한 설계 판단**: LONG/SHORT confidence를 무조건 같게 맞추는 접근은 기각했다 — 스윙 형태
진단이 확인한 "숏이 완결 전 되돌림이 실제로 더 적다"는 부분은 진짜 시장 정보일 수 있어서,
그것까지 지워버리면 실제 신호를 죽이는 것이다. 대신 **클래스별로 확신도가 실제 정확도와
일치하는지**(reliability/ECE)부터 확인하고, 진짜 과신/과소신이 있는 부분만 그 정도만큼
temperature scaling으로 고친다.

**메커니즘 주의(사전에 명시)**: `quality_for_action` 게이팅 공식(`trading_bot_modules/
omega4_6_1_live.py:174-178`)은 `quality_head`의 자체 softmax를 쓰지 `direction_head`의
`dir_confidence`를 코드상 직접 읽지 않는다 — Test 2(confidence-echo 문서)가 보인 상관(h48qual
ρ=0.18~0.43)은 경험적 상관일 뿐 코드 의존이 아니다. 즉 이 재보정은 `direction_head` 자체가
잘 보정됐는지 진단하는 것이며, 라이브 게이트 출력을 자동으로 바꾸지 않는다 — 실제 반영하려면
게이트 공식 자체를 바꾸거나 `quality_head`를 재설계해야 한다(이번 진단 범위 밖).

## 방법

데이터: 라이브 h48qual 번들(`true_3head_tabm_bundle.pt`, 2026-06-30 export, **재학습 없음**)의
저장 예측 + 정식 `zigzag_action` 참라벨(`build_wave3_action_labels_20260531.py` 산출, 실제
`direction_head` 학습 타겟) 시간축 조인. confidence-echo 문서와 동일 데이터 소스 —
**정합성 체크로 Test 1 숫자(n, 평균 confidence)를 소수점까지 재현 확인 후 진행**.

1. **클래스별 reliability**: LONG-예측/SHORT-예측 bar를 각각 confidence 10분위로 나눠 분위별
   평균 확신도 vs 실제 정확도(예측이 참라벨과 일치한 비율) 비교. ECE = 분위별 `|평균확신도-정확도|`의
   표본가중평균.
2. **클래스별 temperature scaling**: 저장된 softmax 확률에서 `log(p)`로 유사-logit 복원(softmax의
   shift-invariance로 정당화됨) → `T_class`를 학습구간에서 해당 클래스 NLL 최소화로 적합 →
   VAL/OOS에 적용해 ECE 개선 여부 확인. **argmax 불변 검증**(temperature scaling의 정의상 속성,
   코드에서 assert로 재확인) — `dir_action`은 안 바뀌고 `dir_confidence` 크기만 바뀐다.

스크립트: `scripts/diagnose_eth_h48qual_direction_confidence_calibration_20260811.py`.

## 결과

### 정합성 체크

학습/VAL/OOS 각각 n_long/n_short와 평균 confidence가 confidence-echo 문서 Test 1과 전부
소수점까지 일치(예: 학습 LONG n=25910, 평균=0.5725) — 같은 데이터, 같은 조인으로 확인됨.

### 클래스별 보정 상태

| 구간 | 클래스 | n | 평균확신도 | 실제정확도 | 과신(확신도−정확도) | ECE |
|---|---|---:|---:|---:|---:|---:|
| 학습 | LONG | 25,910 | 0.5725 | **0.7223** | **−0.1498** | 0.1498 |
| 학습 | SHORT | 34,064 | 0.6210 | 0.6390 | −0.0180 | 0.0191 |
| VAL | LONG | 8,052 | 0.5676 | 0.6731 | −0.1056 | 0.1056 |
| VAL | SHORT | 11,303 | 0.6086 | 0.5528 | **+0.0559** | 0.0571 |
| OOS | LONG | 5,238 | 0.5642 | 0.6367 | −0.0725 | 0.0725 |
| OOS | SHORT | 6,746 | 0.6038 | 0.5978 | +0.0059 | 0.0483 |

10분위 상세(학습구간): LONG은 전 분위에서 확신도 < 정확도(최저분위 0.42/0.60, 최고분위
0.74/0.85) — 일관된 과소신. SHORT는 분위별로 확신도 ≈ 정확도(최저 0.44/0.47, 최고 0.81/0.81) —
이미 상당히 잘 보정됨.

### 클래스별 temperature 적합(학습구간) 및 VAL/OOS 적용 결과

`T_long=0.6245`(강하게 sharpen 필요), `T_short=0.9156`(거의 보정 불필요, T=1에 근접).

| 구간 | 클래스 | ECE(보정 전 → 후) | 평균확신도(전 → 후) |
|---|---|---|---|
| VAL | LONG | 0.1056 → **0.0741**(개선) | 0.568 → 0.660 |
| VAL | SHORT | 0.0571 → **0.0742**(악화) | 0.609 → 0.627 |
| OOS | LONG | 0.0725 → **0.0813**(소폭 악화) | 0.564 → 0.654 |
| OOS | SHORT | 0.0483 → **0.0513**(소폭 악화) | 0.604 → 0.621 |

## 해석

**핵심 발견 — 원래 예상과 반대 방향**: 관찰된 confidence 격차(숏이 롱보다 3~5pp 높음)를 "숏이
과신됐다"로 읽기 쉽지만, 실제 보정 상태를 보면 **정반대에 가깝다** — 학습구간에서 LONG은
심각하게 과소신(확신도 57%, 실제 정확도 72%, 15pp 갭)인 반면 SHORT는 이미 거의 잘 보정돼
있다(확신도 62%, 정확도 64%, 2pp 갭). 즉 두 클래스의 원시 confidence 격차 중 상당 부분은
"SHORT가 과신됨"이 아니라 **"LONG이 과소신됨"**으로 설명된다 — 보정을 전제로 하면 오히려 LONG
쪽 정확도가 SHORT보다 높다(학습구간 72% vs 64%).

**하지만 이 패턴이 구간마다 다르게 불안정하다**: LONG의 과소신 방향은 학습·VAL·OOS 셋 다
일관되게 나타나지만(−15.0pp→−10.6pp→−7.3pp, 방향은 안정, 크기는 구간마다 다름) — **SHORT는
구간마다 부호 자체가 바뀐다**(학습 −1.8pp 과소신 → VAL +5.6pp 과신 → OOS +0.6pp 거의 중립).
SHORT가 학습구간에서는 잘 보정돼 있다가 하락장인 VAL에서 과신으로 바뀌는 것 — **이 세션
전체를 관통한 "학습구간과 평가구간의 레짐이 다르다"는 문제가 보정(calibration) 영역에서도
재발한 것**이다.

**Temperature scaling 결과: 학습구간에서 적합한 보정이 out-of-sample에서 안정적으로
일반화되지 않는다.** LONG 보정은 VAL에서는 개선(ECE 0.106→0.074)되지만 OOS에서는 오히려
소폭 악화(0.073→0.081) — 학습구간의 과소신 크기(15pp)가 OOS의 실제 과소신 크기(7.3pp)보다
훨씬 커서 과잉보정된 것으로 보인다. SHORT 보정은 VAL·OOS 둘 다 악화 — SHORT가 학습구간
기준으로는 "보정 불필요"에 가까웠는데, 그 미세한 보정치를 SHORT의 보정 상태가 완전히 달라진
(과신으로 바뀐) 평가구간에 적용하니 방향이 안 맞아서 생긴 결과다.

## 계약 문서에 미친 영향

**Direction_head confidence의 클래스별 재보정 시도는 부정적/보류 결과다** — 학습구간에서
적합한 temperature가 VAL·OOS에서 안정적으로 개선을 주지 못한다(LONG은 혼재, SHORT는 악화).
이건 이 세션에서 반복적으로 발견된 train/eval 레짐 불일치 문제가 calibration 영역에서도
동일하게 나타난 것으로 해석된다. 설령 이 보정이 안정적으로 작동했더라도, `quality_head`
게이트가 `dir_confidence`를 코드상 직접 소비하지 않으므로 **라이브 행동을 바꾸려면 게이트
설계 자체를 바꾸는 추가 작업이 필요**했을 것 — 이 경로의 실용적 가치는 애초에 제한적이었다.

유일하게 견고한 발견은 **LONG이 세 구간 모두에서 일관되게 과소신 상태**(방향은 안정, 크기는
구간별로 다름)라는 것 — 이건 "숏이 문제"라는 기존 프레이밍을 "롱이 저평가됨"으로 뒤집는
데이터이지만, 크기가 불안정해 정량적 보정치로 쓰기엔 근거가 약하다.

## 결과 (계약 문서 반영용)

`direction_head` confidence의 클래스별 재보정을 시도했다. 무조건 LONG/SHORT를 같게 맞추는
대신 클래스별 보정 상태(reliability/ECE)부터 확인 — 결과는 예상과 반대: 학습구간에서 SHORT는
이미 잘 보정돼 있고(과신 −1.8pp) LONG이 오히려 심하게 과소신(−15.0pp, 실제 정확도가 확신도보다
15pp 높음)이었다. 이 패턴은 LONG에 대해서는 세 구간(학습/VAL/OOS) 모두 방향이 일관되지만,
SHORT는 구간마다 부호가 바뀐다(학습 과소신 → VAL 과신) — calibration 영역에서도 train/eval
레짐 불일치가 재발한 것. 학습구간에서 적합한 temperature scaling은 VAL/OOS에 안정적으로
일반화되지 않는다(LONG 혼재, SHORT 악화) — **재보정 시도, 부정/보류 결과로 결론**. 참고로 이
보정은 `quality_head` 게이트가 `dir_confidence`를 직접 소비하지 않아 라이브 행동에 자동으로
반영되지 않는다는 점도 확인됨.
