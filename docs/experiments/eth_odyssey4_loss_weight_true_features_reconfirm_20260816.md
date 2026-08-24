# Odyssey4 quality/exit loss weight — 진짜 102피처 파이프라인 재확인, CLOSED (2026-08-16)

관련 상위 문서: `docs/experiments/eth_odyssey4_loss_weight_optuna_search_20260816.md`(C2,
`_prepare_frames_light()` 185피처 프록시 기준 원본 실험, 경계선 판정), `docs/experiments/
eth_odyssey4_layer_and_parameter_improvement_proposal_20260816.md`(§C2).

## 배경 — 왜 재확인했는가

같은 날 병행 세션(`from="general-purpose"`)이 `scripts/eth_odyssey4_true_feature_pipeline_20260816.py`
(`prepare_frames_true()`)로, 이 세션 전체(B1/A1/C1/C2/C3)가 써 온 `_prepare_frames_light()`
185피처 프록시 대신 진짜 라이브 배포 번들과 일치하는 102 base(+13 pos)=115피처 계약을 별도
경로로 복구했다는 메시지를 보내왔다(`omega._load_omega_frames()` 자체는 정상 동작함을 확인하고
`hard._build_frame()`의 라벨 fetch만 `_prepare_frames_light()`와 동일한 방식으로 우회, 결측
7컬럼은 수식 재현으로 채움 — 5/7 완전 일치, funding_roc 2개는 연도 경계 콜드스타트 오차
문서화). 이 세션의 기존 결과 중 **C2(loss weight Optuna 서치)만 "경계선"** 판정이었다
(A1/C1은 명확한 부정) — 프록시→진짜 피처 전환으로 판정이 뒤집힐 가능성이 가장 큰 항목이므로
가장 먼저 재확인했다.

`scripts/eth_odyssey4_true_feature_pipeline_20260816.py`를 임포트해서 스모크테스트로 독립
검증(`x_train.shape == (78568, 115)`, NaN 없음)한 뒤 재사용했다 — 병행 세션의 주장을 그대로
믿지 않고 직접 확인 후 사용.

## 실행

- 스크립트: `scripts/research_eth_odyssey4_loss_weight_nseed_confirm_true_features_20260816.py`
  (`research_eth_odyssey4_loss_weight_nseed_confirm_20260816.py`의 거의 동일한 복제본, 프레임
  준비만 `truepipe.prepare_frames_true(disable_tp_sl=False)`로 교체).
- baseline(`quality_loss_weight=0.80`/`exit_loss_weight=1.15`) vs 원본 C2 Optuna가 찾은 best
  (`quality_loss_weight=0.45108572002184927`/`exit_loss_weight=0.5978814568312127`) — 원본 C2와
  **완전히 같은 두 설정**을 비교, expert=bull, epochs=28, plain CE(A1 되돌림 반영), 캐노니컬
  CFG 나머지 동일.
- 시드: 원본 C2 재현과 동일한 방식으로 `secrets.randbelow` 추출 5개(고정간격 아님) —
  `[234740558, 264572419, 293140425, 784578921, 983935002]`.
- exit_head 데이터셋은 `max_samples=60000` 캡 적용(dev 15GB RAM 메모리 안전장치, 이 세션 전체
  공통).

## 결과 — N=5 시드, expert=bull, 진짜 102피처 파이프라인

| seed | baseline dir_bacc | best dir_bacc | Δ(best−baseline) |
|---:|---:|---:|---:|
| 234740558 | 0.5527 | 0.5524 | −0.0003 |
| 264572419 | 0.5697 | 0.5692 | −0.0005 |
| 293140425 | 0.5796 | 0.5765 | −0.0031 |
| 784578921 | 0.5708 | 0.5784 | +0.0076 |
| 983935002 | 0.5530 | 0.5539 | +0.0009 |

| 지표 | 값 |
|---|---:|
| baseline 평균 dir_bacc | 0.5651 |
| best 평균 dir_bacc | 0.5661 |
| 평균 Δ | **+0.0009** |
| Δ 표준편차 | **0.0040** |
| 개선된 시드 | **2/5** |
| 부호 일관성 | **False** |

모든 실행(baseline·best 둘 다, 5시드 전부)이 정확히 epoch 9에서 patience=8 조기종료로 멈췄다 —
원본 C2 프록시-피처 실행과 동일한 조기종료 시점 패턴.

## 원본(프록시 185피처) 대비 비교

| 지표 | 프록시(185피처, 원본 C2) | 진짜(115피처, 이 문서) |
|---|---:|---:|
| 평균 Δ | +0.0037 | **+0.0009**(1/4 수준으로 축소) |
| Δ 표준편차 | 0.0038 | 0.0040(거의 동일) |
| std/mean 비율 | ≈1.03 | **≈4.3** |
| 개선된 시드 | 4/5 | **2/5** |
| 부호 일관성 | False(1개 사실상 평탄) | False(3개 악화, 그 중 하나는 뚜렷) |

두 실행은 시드 값도 다르고(원본은 `[222496635, 248835011, 412226287, 843248084, 850919609]`,
이번은 `[234740558, 264572419, 293140425, 784578921, 983935002]`) 피처 파이프라인도 달라서
직접적인 "같은 시드 짝 비교"는 아니다 — 그러나 N=5 요약통계로 봤을 때 진짜 피처에서는 신호가
사라지는 게 아니라 **더 명백하게 노이즈로 보인다**: 평균 델타가 4배 이상 작아졌고, std가 평균의
4배를 넘어(원본은 std≈mean으로 이미 "경계선"이었는데, 이번엔 std가 mean을 훨씬 압도), 개선된
시드 비율도 4/5에서 2/5(과반 미만)로 떨어졌다.

## 판정 — **CLOSED, 반영 안 함 — 진짜 피처 파이프라인에서는 경계선조차 아니고 명확한 노이즈**

원본 C2가 "경계선, 승격 보류"였던 근거는 std(0.0038)가 mean(0.0037)과 거의 같아 이 저장소의
"std가 평균보다 훨씬 작아야 진짜 신호" 기준을 아슬아슬하게 못 넘긴 것이었다. 진짜 102피처
파이프라인에서는 그 아슬아슬함마저 사라진다 — std(0.0040)가 mean(0.0009)의 4배를 넘고,
개선된 시드가 과반에도 못 미친다(2/5). 이 결과는 A1(std가 mean의 3배 이상, 4/5 악화)이나
C1(사실상 완전 무효과)과 같은 급의 명확한 부정 신호로 격상됐다 — "경계선이라 판단 보류"가
아니라 "노이즈, 반영 안 함"으로 판정을 명확히 한다.

**결론**: `quality_loss_weight=0.80`/`exit_loss_weight=1.15`(캐노니컬 기본값)을 바꾸지 않는다.
프록시 파이프라인에서 관측된 경계선 신호는 진짜 라이브 피처 계약으로 전환하자 재현되지 않았다 —
이는 이 세션이 이미 반복적으로 확인해 온 패턴(단일/약한 신호가 조건을 바꾸면 사라짐)과 일관된
결과다.

## fresh-forward 규칙 준수

`fresh_forward_bar_by_bar=n/a`(분류기 학습, 내부 85/15 val split 기준 direction_balanced_accuracy
비교만 — VAL/OOS 포트폴리오 백테스트는 진행하지 않음, 명확한 부정 판정이라 백테스트까지 갈
이유가 없음), `trade_ledgers_used_as_input=false`, `saved_parent_exit_timestamps_used=false`,
`future_rows_used_for_entry=false`.

## 산출물

- 재확인 스크립트: `scripts/research_eth_odyssey4_loss_weight_nseed_confirm_true_features_20260816.py`.
- 결과 원장: `tmp/causal_regen_20260516/eth_odyssey4_loss_weight_nseed_confirm_true_features_20260816/report.json`.
- 재사용한 진짜 피처 파이프라인: `scripts/eth_odyssey4_true_feature_pipeline_20260816.py`(병행
  세션 산출물, 이 세션에서 스모크테스트로 독립 검증 후 재사용).
- 캐노니컬 스크립트(`scripts/train_eval_omega1_2_tabm_3head_20260603.py`)는 **미변경**.

## registry 반영

`docs/model_contracts/research_line_registry.json`에 `eth_odyssey4_loss_weight_true_features_reconfirm`
항목으로 등록 — scope: `_fit_expert_3head`의 `quality_loss_weight`/`exit_loss_weight`(진짜
102피처 파이프라인 기준), reason: 원본 C2(프록시 185피처)의 경계선 결과(std≈mean)가 진짜 라이브
피처 계약(115피처)으로 재확인 시 명확한 노이즈로 판정됨(std가 mean의 4배 초과, 개선 시드 2/5),
retest_guidance: 피처 파이프라인 차이로는 이 결론이 뒤집히지 않을 것 — 재시도하려면 다른 expert
(bear/chop)나 다른 탐색공간/목적함수 등 질적으로 다른 변경이 필요.
