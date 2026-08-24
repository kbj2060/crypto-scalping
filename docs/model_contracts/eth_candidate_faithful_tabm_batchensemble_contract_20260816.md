# ETH h48qual/zig075 완전한 TabM(BatchEnsemble R+S+B) 후보 — 데이터 계약 (2026-08-16)

이 문서는 **공식 Odyssey 계보(Odyssey1~4)에 속하지 않는다** — 버전 번호는 확정된 성과가 있을
때만 올린다는 원칙에 따라(2026-08-16, 사용자 결정), 결과가 확정되기 전까지 "Odyssey5"로
명명하지 않는다.

## 상태

| 컴포넌트 | 상태 |
|---|---|
| **Step A(R+S+B)** | `CLOSED`. N≥5 시드 재현: direction_balanced_accuracy 기준 chop 5/5, bull 4/5 **일관된 악화** — 노이즈 아님(아래 정정 참고: 체크포인트 선정 버그 영향은 이 비교에선 무시 가능, 격차 0.0000~0.0008). |
| **Step B(piecewise-linear embedding)** | `CLOSED`, 중립. 최초 cheap_gate는 3개 expert 전부 −3.7~−5.1%p 악화로 나왔으나, **combined val_loss 기준 early stopping이 진짜 direction_balanced_accuracy 정점을 놓치는 버그였음이 확인됨**(임베딩 아키텍처에서만 발생). `quarter` 축소 config(109,836 파라미터, baseline보다 적음)로 3개 expert 전부 재검증한 결과: bull −0.0017, bear +0.0046, chop −0.0078, 평균 −0.0016 — **사실상 무승부(노이즈 수준)**. "결정적 악화"는 정정됐지만 채택할 근거도 없어 CLOSED. |
| **완전한 TabM(R+S+B+embedding) 후보 전체** | `CLOSED`. Step A 소폭 악화 + Step B 무승부 — 논문대로 완성해도 baseline을 이기지 못함. VAL 백테스트 진행 안 함.

**경과**: 사용자가 라이브 `ThreeHeadTabM` 구현이 TabM 논문(arXiv:2410.24210)과 실제로 일치하는지 물어서 코드를 직접 대조·검증한 뒤 시작됐다. 아래 "발견한 차이" 참고. Step A(단일시드) 결과는 부호 혼재(노이즈)였다가 N≥5 시드로 일관된 악화 확정 — 2026-08-16 한 차례 CLOSED 처리했으나, 사용자가 "닫지 말고 논문대로 계속 최적화하자"고 명시적으로 지시해 재개. 진행 중 발견: single-seed cheap_gate + N-seed 재현을 합친 66회 학습 전부 early-stop patience=8에 걸려 정확히 epoch 9에서 멈췄다 — baseline/완성판 가릴 것 없이 매번 best checkpoint가 epoch 1에서 나왔다는 뜻(이후 patience/epoch 완화 진단, LR sweep, GCE/ELR/mixup 정규화 테스트, 파라미터 축소 스윕으로 이어짐, 그 축소 스윕 과정에서 위 체크포인트 선정 버그 발견). 상세: `docs/experiments/eth_candidate_faithful_tabm_batchensemble_20260816.md`. |

## 범위

- 목적: `docs/experiments/eth_odyssey_dl_rl_architecture_research_20260816.md`가 "아키텍처
  교체는 시간낭비"라고 결론지었는데, 그 결론은 **"TabM을 논문대로 제대로 구현해서 다 써봤다"는
  전제** 위에 있었다. 코드 대조 결과 그 전제가 틀렸다는 게 드러났다 — 지금까지 시도한 TabM
  "대안"들(VSN/diffusion/Mamba/Transformer/TCN)은 전부 이 **불완전한 TabM 변형 위**에서
  시도된 것이다. 이 계약은 논문의 실제 설계를 충실히 구현해서, "TabM 자체를 제대로 썼을 때"의
  성능을 처음으로 측정한다.
- 이건 "8번째 새 아키텍처 시도"가 아니라 **이미 쓴다고 주장하는 걸 논문 그대로 완성**하는
  것이라, 다른 대안 아키텍처 시도들과 다른 근거 등급을 가진다 — 논문 자체 ablation이 이미
  일반 tabular 벤치마크에서 효과를 검증해뒀다(수치는 아래 "발견한 차이" 참고).
- 관련 문서: `docs/experiments/eth_odyssey_dl_rl_architecture_research_20260816.md`(이 후보의
  동기), `scripts/train_eval_omega1_2_tabm_3head_20260603.py`(라이브 h48qual/zig075가 실제로
  쓰는 `ThreeHeadTabM` 구현, 이식 원본이자 비교 기준선).

## 발견한 차이 (코드 대 논문 직접 대조, 2026-08-16)

TabM 논문(arXiv:2410.24210 HTML 원문 확인)의 핵심 수식:

```text
l_BE(X) = ((X ⊙ R) W) ⊙ S + B
```

`W`는 멤버 간 공유되는 선형층 가중치, `R`(사전곱셈)·`S`(사후곱셈)·`B`(bias)는 앙상블 멤버
`k`개별로 다른 학습 가능 파라미터다. 기본 "TabM" 변형은 이걸 **backbone의 모든 선형층마다**
적용한다(N개 블록이면 3N개 어댑터). 논문 자체 ablation(Figure 2 기준): 수치형 피처에 대한
piecewise-linear embedding을 추가하면 평균 약 +2~3% 상대 개선.

라이브 `ThreeHeadTabM`(`scripts/train_eval_omega1_2_tabm_3head_20260603.py:87`) 대조 결과:

| 구성요소 | 논문(TabM 기본형) | 라이브 구현 |
|---|---|---|
| 입력단 어댑터 | R+S+B | R(`input_scale`)+B(`input_bias`)만, **S 없음** |
| 이후 블록 어댑터(레이어당) | R+S+B | R(`expert_scale`)만, **S·B 둘 다 없음** |
| Residual 연결 | 논문에 없음(표준 MLP) | **있음**(`h = h + residual`) — 논문에 없는 걸 추가함 |
| 수치형 피처 embedding | piecewise-linear 권장(+2~3%) | **없음**(표준화된 raw float 직접 입력) |

즉 라이브 구현은 논문의 3가지 공식 변형(TabM/TabM_mini/TabM_packed) 중 어느 것과도 정확히
일치하지 않는 자체 변형이다 — R만 쓰는 점은 TabM_mini와 비슷하지만, TabM_mini는 R을 **첫
레이어에만** 적용하는데 라이브 구현은 **매 레이어에** 적용한다.

## 설계 — 2단계 ablation (섞어서 한 번에 테스트하지 않는다)

원인 분리를 위해 두 수정을 독립적으로 검증한다:

1. **Step A: BatchEnsemble 완성** — 모든 레이어에 S(사후곱셈)와 레이어별 B(bias)를 추가해
   R+S+B 세트를 완성한다(논문의 기본 "TabM" 변형과 동일 구조). Residual 연결은 그대로
   유지(제거하지 않음 — 이건 별도 변수라 이번 ablation 범위 밖). 수치형 embedding은 아직
   추가하지 않는다.
2. **Step B: + Piecewise-Linear Embedding** — Step A 위에 수치형 피처(102개 base, POS_COLS
   13개는 제외 — 진입시 0으로 채우는 상태 placeholder라 embedding 대상 아님) piecewise-linear
   embedding을 추가한다. Step A 결과가 유의미할 때만 착수.

두 단계 다 **k=8, hidden=192, layers=3, dropout=0.08 등 기존 CFG 불변** — 순수하게 어댑터
완전성/embedding 유무만 비교한다.

## 상태/데이터 계약

- 학습 데이터/라벨/split: `_prepare_frames()`(기존 스크립트) 그대로 재사용 — train
  (~2025-01~09), 그 안에서 85/15 내부 val split(기존 관례 그대로), 신규 라벨/피처 없음.
- 비교 대상: h48qual(3개 regime expert: bull/bear/chop) 먼저 — zig075는 h48qual 결과가
  유의미할 때만 확장.
- N≥5 시드 게이트: **적용 대상**(진짜 학습 모델). cheap_gate는 단일 시드로 먼저 방향성만
  확인 — 유의미하면 N≥5 시드 재현 필수.

## 필수 저비용 게이트 (cheap_gate)

전체 백테스트(replay 엔진 재실행)까지 가기 전에, **분류 지표 수준에서 먼저 확인**한다 —
`_fit_expert_3head`와 동일한 데이터·시드·하이퍼파라미터로 기존 아키텍처와 Step A 아키텍처를
각각 학습해서, held-out validation loss(direction/quality/exit 각각)와 direction balanced
accuracy를 나란히 비교한다. 방향/품질 신호 자체가 이미 무스킬로 확정된 상태라, **분류
지표에서조차 개선이 없으면 백테스트까지 갈 이유가 없다.**

## Red Team Gates

- [x] cheap_gate(분류지표) 먼저 통과 — **미통과** (단일시드 결과 부호 혼재).
- [x] N≥5 시드로 재현(진짜 무작위 시드, 고정간격 클러스터 아님) — **완료, 일관된 악화로 판명**
  (direction_balanced_accuracy: chop 5/5 악화, bull 4/5 악화, bear net 악화).
- [x] 개선이 통계적으로 유의미한지 확인 — 개선이 아니라 **악화가 유의미**(std가 평균보다 작은
  chop/bull에서 특히 뚜렷), 채택 기준 자체가 성립하지 않아 별도 유의성 검정 불필요.
- [ ] VAL 포트폴리오 백테스트 → OOS 단일터치 — **진행 안 함** (분류지표에서 악화로 판명돼 게이트
  실패, 계약 상 백테스트까지 갈 이유가 없음).
- [ ] `docs/model_contracts/research_line_registry.json`의 `falsification_audit` 취지에 맞게,
  개선처럼 보이는 결과를 라벨-순열(permutation) 대조군 없이 채택하지 않는다(외부 문헌
  조사가 지적한 "아키텍처 탐색은 무작위 데이터에서도 유의한 개선을 만들 수 있다"는 위험,
  arXiv:2604.15531).
- [x] **Step B 체크포인트 선정 버그 수정 후 3개 expert 재검증** — **완료, 무승부**(평균
  Δ=-0.0016). N≥5 시드 재현은 진행하지 않음 — 단일시드 효과 크기가 이미 노이즈 수준이라
  재현해도 결론이 바뀔 근거가 없음.

## 미해결 이슈

1. Residual 연결(논문에 없는 기존 추가사항)을 유지할지 제거할지 — 이번 ablation 범위 밖,
   Step A/B 결과를 보고 별도 결정.
2. Piecewise-linear embedding의 정확한 하이퍼파라미터(bin 수, embedding 차원)는 논문 기본값을
   그대로 따르지 않고 이 프로젝트 규모에 맞게 축소 조정 필요 — Step B 착수 시 결정.
3. h48qual/zig075 결과가 갈릴 가능성 — h48qual은 exit_head가 살아있고(9/14 VAL 관여) zig075는
   죽어있어서(0/86), 아키텍처 개선의 효과가 컴포넌트별로 다르게 나타날 수 있음.

## 다음 단계

1. Step A(BatchEnsemble 완성) 구현 → 단일시드 cheap_gate(분류지표) 실행.
2. 유의미하면 N≥5 시드 재현 → Step B(embedding) → VAL 백테스트 → OOS 단일터치.
3. 결과는 `docs/experiments/eth_candidate_faithful_tabm_batchensemble_<date>.md`에 기록.
