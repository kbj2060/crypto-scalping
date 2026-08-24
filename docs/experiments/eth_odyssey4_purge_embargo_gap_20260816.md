# Odyssey4 85/15 내부 split purge/embargo 갭 — cheap_gate 결과, CLOSED (2026-08-16)

관련 상위 문서: `docs/experiments/eth_odyssey4_layer_and_parameter_improvement_proposal_20260816.md`
(§C1), `docs/experiments/eth_odyssey4_gce_canonical_port_20260816.md`(A1, 이 실험이 재사용하는
캐노니컬 손실함수 상태).

## 데이터 파이프라인 주의사항

이 실험은 `_prepare_frames_light()` 우회(`feature_cols`=185개, 진짜 라이브 102(+13pos)피처와
다름)를 쓴다 — `_prepare_frames()` 자체가 dev/서버 양쪽에서 죽은 vsnlstm/chronos 캐시 때문에
막혀있다(`docs/experiments/eth_odyssey4_gce_canonical_port_20260816.md`에 상세 기록, 병행
세션도 독립 확인). **이 문서의 수치는 프록시 파이프라인 기준이며, 라이브 승격 근거로 쓰려면
진짜 102피처 파이프라인 복구가 별도로 필요하다.**

**추가 참고(2026-08-16, 병행 세션 이후)**: 같은 날 병행 세션이
`scripts/eth_odyssey4_true_feature_pipeline_20260816.py`로 이 복구를 실제로 해냈다. 이 문서의
cheap_gate 판정은 재실행하지 않았다 — early_stop_epoch 델타가 정확히 0(둘 다 epoch 1)이고
val_loss/bacc 델타가 무시할 수준이라는 결과가 애초에 "모델이 몇 epoch만에 조기종료되는 현재
memorization 패턴에서는 54-bar 갭이 드러날 여지가 거의 없다"는 구조적 설명과 정합적이므로,
프록시→진짜 피처 전환만으로 이 결론이 뒤집힐 개연성은 낮다고 판단했다. 우선순위는 C2(경계선
결과)의 진짜-피처 재확인에 먼저 배정했다.

## B2 — 갭 크기 실측

`scripts/diagnose_odyssey4_zigzag_pivot_confirmation_delay_20260816.py`로
`zigzag_action_labels_20260531`(h48qual·zig075가 실제 쓰는 라이브 라벨셋) 생성에 쓰인 pivot
로직(`scripts/build_zigzag_action_labels_v2_20260604.py`의 `_zigzag_pivots`, v1 파라미터로
재현: min_reversal_pct=0.01, atr_window=14, atr_multiplier=1.0, max_reversal_pct 무제한)을
그대로 재구현해서, 각 pivot이 "확정"되기까지(가격이 되돌아가 threshold를 넘기기까지) 걸린
bar 수를 직접 셌다(2024~2026, 4,961개 pivot):

| 구간 | median | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|
| 2024 | 12 | 41 | 60.4 | 119.0 | 311 |
| 2025 | 10 | 34 | 48.0 | 93.2 | 346 |
| 2026 | 9 | 41 | 55.3 | 131.1 | 432 |
| **전체 결합** | **11** | **38** | **54** | **105.4** | **432** |

즉 지그재그 라벨은 확정에 중앙값 11 bar, 95th percentile **54 bar**가 걸린다 — 85/15 split
경계 바로 앞 학습 샘플의 라벨이 검증쪽 미래 가격에 살짝 의존했을 가능성이 이론적으로 있다.
이 54 bar를 C1의 embargo 갭 크기로 채택.

## C1 — cheap_gate 실행

- 스크립트: `scripts/research_eth_odyssey4_purge_embargo_gap_cheap_gate_20260816.py`.
- `_fit_expert_3head`의 내부 85/15 direction/quality split(`train_idx=arange(split)`,
  `val_idx=arange(split,n)`)에 54-bar 갭을 추가 — `train_idx=arange(split-54)`로 train 끝을
  당겨서 경계 근처 54개 행을 purge, val_idx는 그대로.
- 단일 시드(260816), 단일 expert(bull), 캐노니컬 CFG 그대로(patience=8, k=8/hidden=192/
  layers=3, quality_loss_weight=0.80/exit_loss_weight=1.15). **손실함수는 plain CE** —
  A1(GCE 이식)이 이 정확히 같은 캐노니컬 스크립트에서 N=5시드 검증 결과 개선이 전이되지
  않아 되돌려졌으므로, 캐노니컬의 실제 손실(plain CE)을 그대로 반영.
- exit_head 데이터셋 split은 손대지 않음 — exit 라벨은 pivot 확정과 무관한 별도 실행기반
  라벨(`_build_exit_dataset_independent`)이라 B2의 forward-looking 문제가 적용 안 됨.

## 결과

| 갭 | early_stop_epoch | best_val_loss | dir_bacc | train_rows |
|---|---:|---:|---:|---:|
| 0(현재) | 1 | 3.1532 | 0.5687 | 66,782 |
| 54(B2 p95) | 1 | 3.1536 | 0.5677 | 66,728 |

| 지표 | Δ(gap54 − gap0) |
|---|---:|
| early_stop_epoch | **+0**(동일) |
| best_validation_loss | +0.0004(0.01% 상대) |
| direction_balanced_accuracy | −0.0010 |

## 판정 — **cheap_gate 미통과, CLOSED, N≥5 시드 본실험 진행 안 함**

상위 제안 문서의 명시적 게이팅 규칙: "갭 추가 전/후 val-loss 곡선·조기종료 시점이 실질적으로
달라지는가 — 안 달라지면 유효한 부정 결과로 문서화하고 끝낸다." 이번 결과는 **조기종료
시점이 완전히 동일(둘 다 epoch 1)**하고, val_loss/direction_balanced_accuracy 델타가 소수점
셋째~넷째 자리 수준으로 무시할 만하다 — 명백히 "실질적 변화 없음"에 해당한다.

두 설정 다 epoch 1에서 조기종료된 것 자체가 `eth_candidate_faithful_tabm_batchensemble` 축이
이미 광범위하게 확인한 "best checkpoint가 거의 항상 epoch 1(또는 그 근방)"이라는 이 저장소의
memorization 패턴과 일치한다 — 모델이 애초에 몇 epoch 만에 조기종료되는 상황에서는, 경계
근처 54개 행을 purge하든 안 하든 학습에 실질적 차이를 만들 여지가 거의 없다는 것도 이
결과를 설명하는 정합적인 해석이다.

**결론**: 85/15 내부 split의 zero-gap 정책이 이 캐노니컬 아키텍처/학습 프로토콜에서 낙관적
편향을 만든다는 증거를 찾지 못했다. 갭을 추가하지 않는다.

## fresh-forward 규칙 준수

`fresh_forward_bar_by_bar=n/a`(분류기 학습, 내부 85/15 val split 기준 비교만 — VAL/OOS
포트폴리오 백테스트는 진행하지 않음, cheap_gate 자체가 미통과라 상위 문서 규칙상 본실험으로
갈 이유가 없음), `trade_ledgers_used_as_input=false`, `saved_parent_exit_timestamps_used=false`,
`future_rows_used_for_entry=false`.

## 산출물

- B2 진단 스크립트: `scripts/diagnose_odyssey4_zigzag_pivot_confirmation_delay_20260816.py`.
- C1 cheap_gate 스크립트: `scripts/research_eth_odyssey4_purge_embargo_gap_cheap_gate_20260816.py`.
- 결과 원장: `tmp/causal_regen_20260516/eth_odyssey4_purge_embargo_gap_cheap_gate_20260816/report.json`.
- 캐노니컬 스크립트(`scripts/train_eval_omega1_2_tabm_3head_20260603.py`)는 **미변경** — 이
  실험은 후보 스크립트 안에서만 갭을 테스트했고, cheap_gate 미통과라 캐노니컬에 반영하지 않음.

## registry 반영

`docs/model_contracts/research_line_registry.json`에 `eth_odyssey4_purge_embargo_gap` 항목으로
등록 — scope: `_fit_expert_3head`의 85/15 direction/quality split에 B2 실측(p95=54bar) 기반
embargo 갭 추가, reason: cheap_gate(단일시드, early_stop_epoch 델타=0, val_loss/bacc 델타
무시할 수준)에서 실질적 변화 없음 확인, N≥5시드 본실험으로 확대할 근거 없음,
retest_guidance: 모델이 매번 epoch 1에서 조기종료되는 현재 memorization 패턴이 먼저 해소되지
않는 한(예: 병행 세션이 진행 중인 LR스케줄/옵티마이저 축) 이 갭의 효과가 드러날 여지 자체가
작다 — 조기종료 시점이 늦춰지는 변화(예: cosine LR, AdaBelief)가 먼저 검증된 뒤에만 재시도
근거가 생긴다.
