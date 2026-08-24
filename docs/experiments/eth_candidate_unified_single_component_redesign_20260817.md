# eth_candidate: 단일 통합 컴포넌트 재설계 — 설계도 (2026-08-17, 구현 전 검토용)

## 명명 규칙 준수

[[odyssey_eth_h48qual_subproject]] 메모리 원칙에 따라 이후 후보는 `eth_candidate_*`로
명명한다 — "Odyssey5"/"Odyssey6" 명명 금지(이 저장소의 명시적 관례). 이 문서/이후 산출물은
`eth_candidate_unified_single_component_*` 접두어를 쓴다.

## 0. 왜 지금 이 문서가 필요한가

[[eth_odyssey4_single_vs_dual_component_contribution_20260817]]에서 기존 가중치 그대로
슬롯공유만 껐을 때 6개 창 중 5개에서 "둘 중 더 나은 단독"이 "조합"을 이긴다는 게 확인됐다.
사용자가 이를 근거로 "zig075 제거 + 단일 컴포넌트로 재설계·재학습"을 제안했다. 이건 큰
작업(GPU, 라벨 재설계, 이 리포의 승격 게이트)이라 착수 전에 설계도를 먼저 검토받는다.

## 1. 사용자 질문에 대한 답 — 현재 헤드별 오라클 라벨 (코드로 검증, 파일명 추측 아님)

`report.json`의 `label_contract.direction_label_dir`/`quality_label_dir`를 직접 읽어 확인했다
(디렉터리 이름만 보고 짐작한 이전 발언은 아래에서 정정한다):

| 컴포넌트 | direction 헤드 라벨 | quality 헤드 라벨 | exit 헤드 라벨(현재 배포 기준) |
|---|---|---|---|
| **h48qual** | `zigzag_action_labels_20260531`(진짜 ZigZag pivot 라벨) | **`sltp_h48_conservative_padded_to_zigzag_timestamps`** — direction과 **별개**, 48/384bar 삼중배리어(TP=max(0.6%,1.2·ATR96)/SL=max(0.4%,0.8·ATR96)) 기반 | liveATR 재라벨(현재 배포) — ATR배리어(min_tp=0.075 등) 기준 `near_barrier_resolution_exit`/`adverse_unreal_exit`/`mfe_giveback_exit`(giveback≥0.65). feature-배리어 버그는 이번 세션에 수정됨([[eth_odyssey4_exit_head_tpsl_feature_barrier_mismatch_20260817]]) |
| **zig075** | `zigzag_action_labels_20260531`(**h48qual과 동일 디렉터리, 동일 라벨**) | `quality_label_dir=None` → direction과 **동일**(`same_as_direction`, 별도 quality 라벨 없음) | **원본**(liveATR 미적용, 현재 배포) — h48qual의 원본과 바이트단위 동일한 오라클 세그먼트경계 라벨(양성 99.86%가 `terminal_window_exit`) |

**정정**: 이전 대화에서 "quality 헤드는 항상 direction과 같은 라벨을 공유한다"고 말씀드렸는데,
그건 구버전 학습스크립트(`train_eval_omega1_2_tabm_3head_20260603.py`)에 한정된 사실이었다.
실제 배포된 두 컴포넌트가 쓰는 현재 스크립트(`train_eval_omega4_3head_parent72_loose_entry_
quality_20260620.py`)는 quality 헤드에 **별도 라벨**(`quality_label_dir`)을 지원하고,
**h48qual은 실제로 이걸 쓰고 있다**(zig075는 안 씀). 정정합니다.

**설계에 대한 함의**: direction 라벨은 이미 두 컴포넌트가 공유 중이므로 "direction 라벨을
통합"하는 건 사실상 이미 돼 있다 — 통합 설계의 진짜 쟁점은 (a) quality 헤드 라벨(h48qual의
barrier기반 vs zig075의 same-as-direction 중 무엇을 쓸지, 또는 제3의 설계), (b)
quality_threshold 캘리브레이션(0.50 vs 0.75), (c) exit 헤드 라벨(둘 다 이미 실패로 확인된
설계 — 아래 3절)이다.

## 2. 범위 (Scope)

**포함**: L2~L9 전체를 대체하는 **단일 3-head TabM 부모모델**(direction+quality+exit) 하나로
h48qual+zig075를 통합. 아키텍처는 TabM 그대로 유지(이미 문헌+실험으로 확정된 축,
[[eth_odyssey_dl_rl_architecture_axis_closed_20260816]] — 대안 전부 종료, 재론 안 함).
피처셋도 기존 150피쳐 스크린 그대로(별도 축, [[eth_direction_nonmicro_research_20260817]]).

**제외(1차 버전)**: zig075 SHORT 진입거부, h48qual 레짐인지 exit가드 — 두 메커니즘 다
"단일 컴포넌트 순수 성능"을 먼저 보고 나중 단계에서 재도입 여부 결정(변수 하나씩 검증한다는
이 세션 전체의 원칙). 라이브/섀도우 배포는 이 문서 범위 밖 — 승격게이트 통과 전엔 연구용.

## 3. 핵심 설계 결정 3가지 (사용자 확인 필요)

### 3-A. Direction 라벨: 그대로 `zigzag_action` 유지 (변경 불필요, 이미 공유 중)

두 컴포넌트가 이미 같은 라벨을 쓰므로 재검토 불필요. 단, h48qual의 direction-head
**축 자체**는 이미 **2026-08-15에 N=5시드 무스킬로 완전종료**된 이력이 있다
([[odyssey_eth_h48qual_subproject]]). **이 재설계가 그 닫힌 축의 반복이 아님을 분명히 해야
한다** — 그 종료는 "h48qual 단독의 피처/구조 변형"이 대상이었지, "zig075와의 데이터/quality
통합"은 다른 축이다. 다만 direction 라벨 자체(zigzag_action)를 다시 흔드는 시도는 이미 닫힌
영역을 건드리는 것이므로 **이번 재설계에서는 direction 라벨을 고정하고 quality/exit만
재설계**하는 걸 원칙으로 제안한다.

### 3-B. Quality 라벨: h48qual의 barrier 기반 라벨을 기본으로 채택 (제안, 확인 필요)

근거: 오늘 확인한 [[eth_odyssey4_single_vs_dual_component_contribution_20260817]] 데이터에서
h48qual 단독은 최고치(2025Q1 +66.63%)와 최악치(2025Q2 −25.84%)를 모두 찍는 **고분산**,
zig075 단독(same-as-direction quality)은 참사 없이 **저분산**으로 나타났다 — 이건 "barrier
기반 quality가 더 우수하다"가 아니라 "무엇이 분산을 만드는지 아직 특정 안 됨"이라는 뜻일
수도 있다. 두 라벨 다 이미 실제로 학습·배포된 채로 대조군이 있으니, **처음부터 하나를
고정하지 말고 quality 라벨 A(h48_conservative barrier)/B(same-as-direction) 둘 다로 각각
학습해서 N≥5 시드로 비교**하는 걸 제안한다(아래 5단계 계획의 Phase 2).

### 3-C. Exit 라벨: 반드시 재설계 필요 — 기존 두 라벨 다 이미 실패로 확인됨

이번 세션에서 실증적으로 확인된 두 실패 라벨을 그대로 재사용하면 같은 실패가 반복된다:
- 원본(오라클 세그먼트경계, 양성 99.86%): 방향 품질과 무관 ([[eth_odyssey4_exit_head_passivity_root_cause_20260817]])
- liveATR 재라벨(`giveback_min=0.65`): 발동 시 평균 MFE 대비 97.6% 반납 후에야 청산, 76.6% 결국 손실 ([[eth_odyssey4_zig075_exit_head_threshold_review_20260817]])

**제안**: `giveback_min`을 0.65→0.2~0.3 수준으로 크게 낮추고(같은 ATR배리어 기반 라벨
구조는 유지, 새 자유변수 최소화 — 딱 이 상수 하나만 재보정), feature-배리어 버그 수정을
처음부터 반영(이번 세션에 이미 고친 버전을 베이스로 시작). **사전등록 검증 기준**: 새
exit_head가 발동한 거래의 평균 giveback비율이 낮아지고(목표: <50%), 발동 거래 승률이
50% 이상으로 개선되는지를 promotion 여부와 무관하게 반드시 측정 — PnL만 보고 판단하지
않는다(giveback_min 재보정만으로 PnL은 우연히 좋아 보일 수 있으나 근본 문제가 안 고쳐졌을
수 있음).

## 4. 검증/승격 기준 (사전등록, 실행 전 고정)

- **Fresh-Forward Rule** 준수: VAL/OOS-Q1/OOS-Q2 causal bar-by-bar만, 저장 렛저 재사용 금지.
- **Seed-Diversity Ensemble Promotion Gate**: N≥5 진짜 랜덤 시드(`np.random.SeedSequence`,
  고정간격 증분 금지), OOS 부호 일치 확인, 시드 리스트를 리포트에 기록.
- **Omega Artifact Integrity Promotion Gate**: `scripts/audit_omega_artifact_integrity_20260630.py`
  exit 0 + `promotion_pass=true` 필수. quality threshold 정확히 일치하는
  `train/validation/oos_predictions_qXXX.csv` 필수.
- **비교 기준선 3개**: (1) 현재 G0(h48qual+zig075 조합), (2) h48qual 단독(오늘 수치),
  (3) zig075 단독(오늘 수치) — 새 단일모델이 **이 셋 전부를 이겨야** 승격 후보 자격.
  하나라도 못 이기면 "재학습이 기존보다 낫다"는 애초 가설이 기각된 것으로 간주하고 그
  시점에 라인을 종료한다(이 리포의 "N=5시드 무스킬→완전종료" 관례와 동일한 규율 적용).
- exit_head 별도 기준: 3-C의 giveback/승률 기준.

## 5. 단계별 실행 계획

1. **Phase 0 (완료)**: 이 설계도 검토·승인.
2. **Phase 1 (완료, 결과는 5-1절)**: quality 라벨 A/B 두 갈래로 direction+quality 헤드만
   우선 학습(exit 제외, encoder 공유), 6개 표준창 전체 N=5시드 비교(VAL만 하려던 원안보다
   범위 확장 — 이유는 5-1절).
3. **Phase 2 (다음 단계, 미착수)**: Phase 1에서 확정된 quality 라벨 B(same_as_direction)로
   exit 헤드를 3-C 제안(giveback_min 재보정)으로 학습, N≥5시드.
4. **Phase 3**: 확정된 단일모델로 G0 동등 리플레이(실제 리스크사이드카 사이징 포함, 6개
   표준창), 4절의 3개 기준선과 비교.
5. **Phase 4**: Phase 3 통과 시에만 정식 승격 절차(Artifact Integrity Gate, Red Team류
   점검) — 이번 문서 범위 밖, 별도 진행.

## 5-1. Phase 1 실행 결과 (2026-08-17)

### 방법

`scripts/train_eval_omega4_3head_parent72_pinned102_20260727.py`(h48qual 102컬럼 계약에
피닝, 기존 확인된 feature-drift 함정 회피 — [[eth_odyssey4_zig075_direction_head_skill_formal_nseed_20260815]]와
동일 도구)로 두 quality 라벨 변형을 N=5 진짜 랜덤시드(`SeedSequence(20260817101)`, 같은
5개 시드를 두 변형에 공통 사용하는 paired 설계)로 학습 — 총 10회, CPU 시드당 약 4.5분.
direction_label_dir은 두 컴포넌트가 이미 공유 중인 `zigzag_action_labels_20260531`로 고정.

평가는 원래 VAL만 계획했으나, 트레이너 자체가 9개 threshold(0.45~0.85)를 한 번에 스윕해
주는데다 exit_head-free/고정사이징 백테스트가 저렴해서 **6개 표준창 전체**로 확장했다
(재현: `scripts/eval_eth_candidate_unified_phase1_quality_ab_20260817.py`). 트레이너 자체
내장 OOS 예측(`oos_predictions_qXXX.csv`)이 2026-02-28에서 끊겨있는 걸 발견해([[eth_h48qual_label_mismatch_discovered]]류의 "가정 말고 확인" 원칙 재적용),
기존 배포 h48qual/zig075에도 같은 문제를 이미 해결해준 검증된 선례
(`build_omega4_6_1_extended_parent_predictions_20260706.py`)를 그대로 재사용해 전체
2026 구간(BASE_2026 원본 + wide24 레짐오버레이)에서 직접 추론했다. exit_head/실제
리스크사이드카는 아직 없으므로(Phase 1 범위 밖), 고정 사이징(notional=0.45/leverage=2.0)
+ 순수 ATR TP/SL(현재 배포 floor와 동일 공식)만으로 백테스트 — 방향+quality 신호 자체의
품질만 격리해서 본다.

### 결과 — Variant B(same_as_direction)가 뚜렷한 승자

**Variant A(barrier, h48qual식)**: threshold≥0.80에서 **PnL이 정확히 0.0**(quality 점수가
0.80을 거의 못 넘어 거래 자체가 안 생김). 낮은 threshold(0.45~0.65)에서는 2025Q1~Q3에서만
강하고 VAL/OOS-Q1/OOS-Q2에서는 약하거나 마이너스 — 학습기간에 가까운 구간에만 강한
과적합 패턴.

**Variant B(same_as_direction, zig075식)**: threshold=0.80에서 **6개 창 전부 평균 PnL
양수**(2025Q1 +8.8%/2025Q2 +15.3%/2025Q3 +16.1%/VAL +9.2%/OOS-Q1 +7.6%/OOS-Q2 +0.3%).
시드 단위로 보면 **6개 창 중 5개에서 N=5시드 전부(5/5) 양수**(OOS-Q2만 3/5, 그것도
낙폭 작음 −7.48%~+7.41%) — 기존 h48qual 단독(2025Q2 참사)이나 zig075 단독(OOS-Q2 약세)
어느 쪽도 달성 못 한 전 창 일관성.

### 결론

**quality 라벨은 Variant B(same_as_direction)로 확정한다.** Variant A는 threshold를
어디에 둬도 문제(높으면 거래소멸, 낮으면 과적합)가 있다. Phase 2는 이 확정된 quality
라벨 위에서 exit 헤드 재설계(3-C 제안)로 진행한다.

**캐비어트**: (1) 아직 고정사이징+exit_head없음 조건이라 실전 성과 예측 아님(Phase 3에서
확정), (2) OOS-Q2가 유일하게 5/5가 아닌 3/5라 완전한 신호는 아님, (3) threshold=0.80 근방이
유독 좋아 보이는 게 진짜 신호인지 9개 threshold 중 하나가 우연히 맞은 것인지는 Phase 3의
독립적 재검증이 필요.

## 6. 리스크/열린 질문

- h48qual direction-head 축이 이미 한 번 무스킬로 닫혔다(3-A) — quality/exit 재설계만으로
  충분한 개선이 나올지는 불확실, Phase 1 결과가 사실상의 go/no-go 게이트.
  Phase 1이 두 quality 라벨 모두 3개 기준선을 못 이기면 라벨을 더 흔들지 않고 즉시 종료.
- zig075 SHORT 진입거부(CONFIRMED 유일 메커니즘)를 1차 버전에서 뺐을 때 그 보호효과를
  잃는 게 아닌지 — 2절에서 의도적으로 나중 단계로 미룸, Phase 3에서 필요시 재도입 검토.
- GPU 없는 dev 머신에서 전체 파이프라인(데이터셋 구축+학습)에 상당 시간 소요 예상
  (오늘 exit_head 재학습 1개 컴포넌트가 ~20분) — 시드 N≥5 × quality라벨 2갈래만 해도
  상당한 총 소요시간, 서버 위임 고려([[feedback_gpu_backlog_offload_to_server]]).

## 재현/근거

- 오늘 확인된 라벨 계보: 코드로 직접 확인, 재현 커맨드는 본문 1절 참조(각 컴포넌트
  `report.json`의 `label_contract` 필드).
- 관련: [[eth_odyssey4_single_vs_dual_component_contribution_20260817]],
  [[eth_odyssey4_exit_head_tpsl_feature_barrier_mismatch_20260817]],
  [[eth_odyssey4_zig075_exit_head_threshold_review_20260817]],
  [[odyssey_eth_h48qual_subproject]] (닫힌 direction-head 축, 명명규칙).
