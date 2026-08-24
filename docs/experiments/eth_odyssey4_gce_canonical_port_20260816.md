# Odyssey4 GCE(q=0.7) 캐노니컬 스크립트 이식 검증 — CLOSED, 되돌림 (2026-08-16)

관련 상위 문서: `docs/experiments/eth_odyssey4_layer_and_parameter_improvement_proposal_20260816.md`
(§A1), `docs/experiments/eth_odyssey4_tabm_layer_design_review_20260816.md`(§6, 최초 발견).

## 배경

레이어 감사가 "이미 검증된 GCE(q=0.7)가 캐노니컬 스크립트(`scripts/train_eval_omega1_2_tabm_3head_20260603.py`)엔
아직 안 들어가 있다"는 걸 이번 세션에서 가장 눈에 띄는 실행 가능 항목으로 지목했다. 근거는
`scripts/research_eth_candidate_faithful_tabm_batchensemble_regularizer_isolation_20260816.py`의
isolation 테스트 — GCE 단독이 plain CE를 이겼다(val bacc 0.5758 vs 0.5740). 단, 그 isolation
테스트는 **고정 40-epoch 예산, early stopping 없음, 단일 시드(260816), expert=bull** 프로토콜이었고,
캐노니컬 스크립트의 실제 학습 프로토콜(patience=8 early stopping, 실제 CFG)과는 다르다 — 그래서
"다른 스크립트에서 나온 결과를 그대로 이식하는 것이므로 이식 후 1회 확인은 필요"하다는 게 상위
제안 문서의 명시적 요구사항이었다.

## 데이터 파이프라인 주의사항 (중요)

`_prepare_frames()`(캐노니컬 스크립트 자체 함수)를 그대로 호출하면 dev/서버 양쪽에서
`FileNotFoundError`로 막힌다 — `hard._build_frame(year)`가 내부적으로 요구하는 vsnlstm/chronos_
uncertainty AI-context 피처 CSV 두 계열이 두 머신 어디에도 없다(모델 체크포인트 `.pt`/
`.joblib`만 남고 CSV는 사라짐, 마지막 정상 실행은 2026-06-08/06-18로 약 2.5개월 전). 이 세션과
독립적으로 병행 진행된 다른 세션(`eth_candidate_faithful_tabm_batchensemble`)도 동일한 블로커를
확인했고, 동일한 우회(`research_eth_candidate_faithful_tabm_batchensemble_cheap_gate_20260816.py`의
`_prepare_frames_light()` — `hard._build_frame`이 실제로 쓰는 건 `["timestamp","zigzag_action"]`
두 컬럼뿐이라, 이 죽은 AI-context 체인을 건너뛰고 `label_base._add_labels(year)`로 라벨만 직접
읽는다)를 이미 확립해뒀다. 이번 검증도 이 정확히 같은 헬퍼를 재사용했다(재도출하지 않음).

**⚠️ 이 문서(및 이 세션에서 만든 B1/C1/C2/C3 전부)의 모든 수치는 `feature_cols`=185개짜리
경량 프록시 파이프라인 기준이다 — 실제 라이브 배포 번들이 쓰는 102 base(+13 pos)=115차원과
다르다.** 이 축(트렁크/손실함수/split 구조가 분류지표를 움직이는가)을 보는 데는 충분하지만,
라이브 승격 근거로 쓰려면 진짜 102피처 파이프라인 복구가 별도로 필요하다
(`eth_omega4_quality_threshold_alpha67_pipeline_irreproducible_20260815`와 동일한 저장소 전체
데이터 유실 제약, 별도 세션에서도 독립적으로 재확인됨).

## 실행

1. **이식**: `scripts/train_eval_omega1_2_tabm_3head_20260603.py`에 `gce_loss(q=0.7)` 함수를
   추가(출처 명시: `research_eth_candidate_faithful_tabm_batchensemble_combo_regularizer_20260816.py`의
   `gce_loss` 그대로 복사)하고, `_fit_expert_3head`의 `loss_dir_k`/`loss_qual_k`(direction_head/
   quality_head 학습손실)만 `gce_loss`로 교체. `exit_head`(`loss_exit_k`)와 validation/early-stopping
   손실 계산은 전부 plain CE로 유지 — isolation 테스트와 정확히 같은 범위.
2. **검증**: `scripts/research_eth_odyssey4_gce_canonical_port_verification_20260816.py` —
   캐노니컬 스크립트의 진짜 학습 설정(`CFG` 그대로: k=8/hidden=192/layers=3/patience=8,
   quality_loss_weight=0.80/exit_loss_weight=1.15) 그대로, expert=bull(isolation 테스트와
   동일 scope로 공정 비교), epoch 예산 28(캐노니컬 `main()` 기본값), **N=5 진짜 무작위 시드**
   (`secrets.randbelow` 추출, 고정간격 아님): `[177534000, 190411884, 319942887, 515566228, 601204617]`.
   baseline_ce는 `research_eth_candidate_faithful_tabm_batchensemble_cheap_gate_20260816.py`의
   `_fit_one`(이미 존재하는 plain-CE 학습루프, 그대로 재사용)을, gce_ported는 그 함수의 손실
   함수만 `gce_loss`로 바꾼 쌍둥이 함수(`_fit_one_gce`)를 각각 호출.

## 결과 — N=5 시드, expert=bull

| seed | baseline dir_bacc | GCE dir_bacc | Δ(GCE−baseline) | 둘 다 epoch |
|---:|---:|---:|---:|---:|
| 177534000 | 0.5607 | 0.5561 | −0.0046 | 9 |
| 190411884 | 0.5135 | 0.5360 | **+0.0225** | 9 |
| 319942887 | 0.5675 | 0.5487 | −0.0188 | 9 |
| 515566228 | 0.5455 | 0.5364 | −0.0091 | 9 |
| 601204617 | 0.5666 | 0.5524 | −0.0141 | 9 |

| 지표 | 값 |
|---|---:|
| baseline 평균 dir_bacc | 0.5508 |
| GCE 평균 dir_bacc | 0.5459 |
| 평균 Δ | **−0.0048** |
| Δ 표준편차 | 0.0162 |
| 개선된 시드 | **1/5** |
| 부호 일관성 | **False** |

모든 실행(baseline·GCE 둘 다, 5시드 전부)이 정확히 epoch 9에서 patience=8 조기종료로 멈췄다 —
`eth_candidate_faithful_tabm_batchensemble` 축에서 이미 확인된 "best checkpoint가 항상 epoch 1"
패턴과 정확히 같은 모양이다.

## 판정 — **CLOSED, 반영 안 함, 이미 넣었던 이식은 되돌림**

isolation 테스트가 발견한 GCE의 개선은 **이 캐노니컬 스크립트의 실제 학습 프로토콜(patience=8
early stopping)로는 전이되지 않는다** — 5개 시드 중 4개가 악화, 평균 Δ=−0.0048(원래 isolation
테스트가 주장한 개선폭 +0.0018보다 절대값이 크고 부호가 반대), 부호 일관성 없음. std(0.0162)가
평균(−0.0048)보다 커서 엄밀한 "노이즈 아님" 기준(std < mean)은 통과 못 하지만, 4/5(80%) 악화
비율 자체는 이 저장소가 R+S+B 축(`eth_candidate_faithful_tabm_batchensemble`)에서 "bull 4/5 악화"를
유의미한 부정 신호로 판정한 선례와 같은 강도다.

**되돌린 이유**: 상위 제안 문서 자체가 "확인되면 승격 후보, 이식 후 1회 확인 필요"라고 명시했고,
그 확인이 실패했다 — isolation 테스트에서 나온 결과를 검증 없이 캐노니컬 스크립트에 방치하는 건
이 세션이 명시적으로 하지 말라고 한 일이다. `_fit_expert_3head`의 학습손실 두 줄을 plain CE로
원복해서 이 스크립트가 이 검증 이전과 byte-identical한 학습 동작을 하도록 되돌렸다. `gce_loss`
함수 자체는 (다른 후보 스크립트들이 여전히 참조하므로) 남겨뒀지만 `_fit_expert_3head`는 더 이상
호출하지 않는다.

## 왜 isolation 테스트와 다른 결론이 나왔을까 — 이후 확인됨 (2026-08-16, 병행 세션)

원래 이 절은 "GCE 손실의 절대 스케일이 CE보다 커서 patience 카운터가 다르게 작동할 수 있다"는
미검증 가설로 남겨뒀었다. 이후 병행 세션이 이 정확한 메커니즘을 문헌+실측으로 확정지었다
(user memory `feedback_modern_dl_training_checklist.md`, 2026-08-16 최종 갱신) — **GCE 손실은
수학적으로 유계(`L_q=(1-p^q)/q ∈ [0,1/q]`)라서 confidence가 오를수록 saturate하고, 그 결과
"8 epoch 연속 개선 없음"을 보는 patience 카운터가 GCE에서는 실제 accuracy가 아직 오르는
중에도 조기에 멈춘다**(GCE 논문 arXiv:1805.07836 자체도 자기 loss가 아니라 validation
accuracy로 체크포인트를 고름). 그 세션은 (1) plain class-balanced CE 기준 selection(GCE
학습손실과 분리) + (2) Prechelt strip/UP_4 기준(raw patience 대신) + (3) cosine LR(2e-4→2e-6)
+ (4) AdaBelief 옵티마이저를 결합한 별도 레시피로 N≥5시드 재확인해서, 그 결합 레시피 하에서는
`AdamW+GCE`/`AdaBelief+GCE`가 오히려 1위로 올라온다는 걸 보였다(가장 최근 통합 실험은
`baseline_R_only+AdaBelief+GCE`, selected_bacc 0.5749, 진짜 정점까지 격차 0.0014).

**이 결과가 위 CLOSED 판정을 뒤집지는 않는다 — 다른 질문에 대한 답이기 때문이다.** A1의
scope는 상위 제안 문서가 명시한 그대로 "GCE 손실만 이식, 나머지(옵티마이저·LR스케줄·조기종료
기준)는 캐노니컬 그대로 유지"였고, 그 정확한 조건에서 GCE는 이 세션의 N=5시드 검증대로
전이되지 않는다 — **이건 여전히 사실이다.** 병행 세션이 보인 건 "GCE + 다른 3가지 변경을
같이 결합하면" 이긴다는, 훨씬 큰 별도의 4-요소 실험이다. 이 실험을 A1의 일부로 재실행하는
건 상위 제안 문서가 정한 A1의 범위(1줄 손실함수 교체)를 벗어나고, from-scratch AdaBelief
구현·Prechelt strip 조기종료 재구현 등 상당한 신규 엔지니어링이 필요해서, 이 세션(B/A/C1/C2/C3
전체를 처리해야 하는) 시간 예산 안에서는 하지 않기로 판단했다 — **의도적 범위 결정이지 누락이
아니다.** 그 4-요소 결합 레시피 자체는 이미 `eth_candidate_faithful_tabm_batchensemble` 축에서
별도로 계속 진행 중이므로, 캐노니컬 스크립트 반영 여부는 그 축의 결론을 기다리는 게 맞다 —
지금 이 문서의 CLOSED 판정은 "GCE 단독 이식은 안 됨"으로 유지하고, "GCE+선정기준+LR+옵티마이저
결합은 유망해 보이지만 A1 범위 밖"이라는 걸 다음 세션을 위해 명시적으로 남긴다.

## fresh-forward 규칙 준수

`fresh_forward_bar_by_bar=n/a`(분류기 학습, 내부 85/15 val split 기준 direction_balanced_accuracy만
비교 — VAL/OOS 포트폴리오 백테스트는 이 축에서 진행하지 않음, 분류지표에서 이미 개선이
확인되지 않아 백테스트까지 갈 이유가 없음), `trade_ledgers_used_as_input=false`,
`saved_parent_exit_timestamps_used=false`, `future_rows_used_for_entry=false`.

## 산출물

- 이식/원복 diff: `scripts/train_eval_omega1_2_tabm_3head_20260603.py`(`gce_loss` 함수는 유지,
  `_fit_expert_3head` 학습손실은 plain CE로 원복).
- 검증 스크립트: `scripts/research_eth_odyssey4_gce_canonical_port_verification_20260816.py`.
- 결과 원장: `tmp/causal_regen_20260516/eth_odyssey4_gce_canonical_port_verification_20260816/report.json`.

## 데이터 파이프라인 추가 참고 (2026-08-16, 병행 세션 이후)

같은 날 병행 세션이 `scripts/eth_odyssey4_true_feature_pipeline_20260816.py`로 진짜 라이브
102피처(+13 pos=115) 파이프라인을 별도로 복구했다 — 위 "데이터 파이프라인 주의사항" 절이 말하는
프록시(185피처) 문제의 해결책이 이제 존재한다. 이 문서의 결과(4/5 악화, 평균 Δ=−0.0048)는
재실행하지 않았다 — GCE 단독 이식 판정은 부호 일관성 없음(1/5 개선)으로 이미 명확히 부정적이라
프록시→진짜 피처 전환으로 뒤집힐 가능성이 낮다고 판단, 재확인 우선순위를 C2(경계선 결과, std≈mean)
쪽에 먼저 배정했다(`docs/experiments/eth_odyssey4_loss_weight_optuna_search_20260816.md` 및
그 진짜-피처 재확인). 필요시 `research_eth_odyssey4_gce_canonical_port_verification_20260816.py`의
`gate._prepare_frames_light()` 호출을 `truepipe.prepare_frames_true()`로 교체해 재실행 가능.

## registry 반영

`docs/model_contracts/research_line_registry.json`에 `eth_odyssey4_gce_canonical_port` 항목으로
등록 대상 — scope: 캐노니컬 3-head TabM(`train_eval_omega1_2_tabm_3head_20260603.py`)의
direction/quality 학습손실, reason: isolation 테스트(고정epoch·무조기종료)에서 나온 개선이
실제 patience=8 조기종료 프로토콜/N=5시드로는 재현 안 됨(4/5 악화). retest_guidance: 조기종료
기준 자체를 바꾸거나(예: GCE 손실이 아닌 plain 지표로 체크포인트 선정) LR/옵티마이저를 함께
바꾸는 조합(병행 세션 E1/E2 참고)이면 다른 결과일 수 있음 — GCE 단독 재시도는 근거 없음.
