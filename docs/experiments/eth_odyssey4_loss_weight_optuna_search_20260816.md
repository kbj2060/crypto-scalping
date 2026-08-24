# Odyssey4 quality/exit loss weight Optuna 서치 — 경계선 결과 (2026-08-16)

관련 상위 문서: `docs/experiments/eth_odyssey4_layer_and_parameter_improvement_proposal_20260816.md`
(§C2), `docs/experiments/eth_odyssey4_gce_canonical_port_20260816.md`(A1, 이 실험이 재사용하는
캐노니컬 손실함수 상태 — GCE는 되돌려짐, plain CE가 현재 캐노니컬).

## 데이터 파이프라인 주의사항

이 실험은 `_prepare_frames_light()` 우회(`feature_cols`=185개, 진짜 라이브 102(+13pos)피처와
다름)를 쓴다 — `_prepare_frames()` 자체가 dev/서버 양쪽에서 죽은 vsnlstm/chronos 캐시 때문에
막혀있다(상세는 `docs/experiments/eth_odyssey4_gce_canonical_port_20260816.md`). **이 문서의
수치는 프록시 파이프라인 기준이며, 라이브 승격 근거로 쓰려면 진짜 102피처 파이프라인 복구가
별도로 필요하다.**

## 배경

`quality_loss_weight`(0.80)/`exit_loss_weight`(1.15)는 근거 주석 없이 고정된 상수 —
레이어 감사가 Tuning Playbook 분류법상 "nuisance 하이퍼파라미터"로 지목했다.

## 1단계 — Optuna TPE 서치(20trial, 단일시드)

- 스크립트: `scripts/research_eth_odyssey4_loss_weight_optuna_search_20260816.py`.
- 탐색공간: `quality_loss_weight`/`exit_loss_weight` 각각 선형 [0.3, 2.0], TPESampler(seed=20260816,
  이 세션의 N-HiTS/ModernTCN HP서치와 동일 컨벤션 재사용).
- 목적함수: **direction_balanced_accuracy 최대화**(combined val_loss가 아님 — 이 두 가중치가
  combined loss 항 자체를 리스케일하므로, 그걸 목적함수로 쓰면 "가중치를 줄여서 이기는" 자기참조적
  결과가 나올 수 있어 배제).
- 고정: expert=bull, 시드=260816(20trial 전부 동일), epochs=28/patience=8, plain CE(A1 되돌림
  반영).
- 결과: baseline(0.80/1.15) dir_bacc=0.5687. 최고 trial(0.451/0.598) dir_bacc=0.5733(+0.0046,
  단일시드).

## 2단계 — N≥5 시드 재현

- 스크립트: `scripts/research_eth_odyssey4_loss_weight_nseed_confirm_20260816.py`.
- 시드: `secrets.randbelow` 추출 5개(고정간격 아님) — `[222496635, 248835011, 412226287,
  843248084, 850919609]`.
- baseline(0.80/1.15) vs best(0.451/0.598), expert=bull, plain CE, 나머지 캐노니컬 CFG 동일.

### 결과

| seed | baseline dir_bacc | best dir_bacc | Δ(best−baseline) |
|---:|---:|---:|---:|
| 222496635 | 0.5444 | 0.5458 | +0.0015 |
| 248835011 | 0.5423 | 0.5423 | −0.0001 |
| 412226287 | 0.5728 | 0.5758 | +0.0030 |
| 843248084 | 0.5583 | 0.5682 | +0.0099 |
| 850919609 | 0.5578 | 0.5619 | +0.0042 |

| 지표 | 값 |
|---|---:|
| baseline 평균 dir_bacc | 0.5551 |
| best 평균 dir_bacc | 0.5588 |
| 평균 Δ | **+0.0037** |
| Δ 표준편차 | **0.0038** |
| 개선된 시드 | 4/5 |
| 부호 일관성 | False(1개 시드가 −0.0001로 사실상 평탄) |

## 판정 — **경계선, 승격 보류**

표준편차(0.0038)가 평균 델타(0.0037)와 거의 같다 — 이 저장소가 반복적으로 써 온 "std가 평균보다
훨씬 작아야 진짜 신호" 기준(예: A1의 std=0.0162 vs mean=−0.0048은 명확한 부정 신호, 이번 건 그
반대 극성이지만 크기 관계는 비슷하게 아슬아슬)을 정확히 충족하지 못한다. 다만:

- 4/5 시드가 개선, 나머지 1개는 "악화"라기보다 사실상 평탄(−0.00006, 반올림 이전 원값)이라 —
  "5개 중 4개는 확실히 개선, 1개는 무효과"에 더 가까운 모양이다.
- A1(전면 부정, 4/5 악화, std가 평균의 3배 이상)이나 C1(사실상 완전 무효과)과는 확실히 다른
  패턴 — 신호가 전혀 없다고 단정할 근거도 약하다.

**결론**: 이 정도 근거로는 캐노니컬 스크립트의 기본값(0.80/1.15)을 확신을 갖고 바꾸지 않는다.
근거가 애매한 상태에서 상수를 바꾸는 건 "실질적 변화 없음"을 "개선"으로 오독하는 것과 같은
위험을 가진다 — 정확히 이 저장소가 여러 차례 경고해 온 단일 관측/약한 신호 함정이다. 승격은
보류하고, 추가 신호(더 많은 시드, 다른 expert에서의 재현, 또는 실제 VAL/OOS 백테스트에서의
일관성)가 쌓이면 재검토한다.

## 진짜 102피처 파이프라인 재확인 (2026-08-16, 후속)

같은 날 병행 세션이 진짜 라이브 102(+13pos)피처 파이프라인(`scripts/eth_odyssey4_true_feature_
pipeline_20260816.py`)을 복구했고, 이 경계선 결과를 그 파이프라인으로 재확인했다 — 결과는
**더 명확한 부정(std가 mean의 4배 초과, 개선 시드 2/5로 하락)**, 승격 보류가 아니라 CLOSED로
격상됨. 전체 내용은 `docs/experiments/eth_odyssey4_loss_weight_true_features_reconfirm_20260816.md`
참고.

## fresh-forward 규칙 준수

`fresh_forward_bar_by_bar=n/a`(분류기 학습, 내부 85/15 val split 기준 direction_balanced_accuracy
비교만 — VAL/OOS 포트폴리오 백테스트는 진행하지 않음, 경계선 판정이라 백테스트까지 갈 확신이
없음), `trade_ledgers_used_as_input=false`, `saved_parent_exit_timestamps_used=false`,
`future_rows_used_for_entry=false`.

## 산출물

- Optuna 서치 스크립트: `scripts/research_eth_odyssey4_loss_weight_optuna_search_20260816.py`.
- N≥5시드 재현 스크립트: `scripts/research_eth_odyssey4_loss_weight_nseed_confirm_20260816.py`.
- 결과 원장: `tmp/causal_regen_20260516/eth_odyssey4_loss_weight_optuna_search_20260816/report.json`,
  `tmp/causal_regen_20260516/eth_odyssey4_loss_weight_nseed_confirm_20260816/report.json`.
- 캐노니컬 스크립트(`scripts/train_eval_omega1_2_tabm_3head_20260603.py`)는 **미변경** — 경계선
  판정이라 반영하지 않음.
