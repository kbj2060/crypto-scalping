# Odyssey4 ThreeHeadTabM 심층분석 + 신규 개선후보 (2026-08-16, 이어서)

상태: **완료 — 코드/체크포인트 직접 실측 기반. 신규 후보 제안, 아직 promotion 근거 아님.**

## 0. 목적과 범위

이 문서는 `docs/deep_learning_layer_design_and_training_reference_20260816.md`(DL 참고자료)를
바탕으로 Odyssey4(h48qual/zig075가 공유하는 `ThreeHeadTabM`,
`scripts/train_eval_omega1_2_tabm_3head_20260603.py`)를 재차 심층분석하고, **같은 날 이미 나온
`eth_odyssey4_tabm_layer_design_review_20260816.md`(1차 코드 대조 리뷰)에서 다루지 않은 두 가지
새 축**을 실측으로 파고든다:

1. **BatchEnsemble k=8 앙상블이 실제로 다양성을 갖는가** — 참고자료가 인용한 문헌 우려
   (arXiv:2601.16936, "BatchEnsemble 멤버가 거의 동일한 함수로 수렴")를 이 저장소 자체 체크포인트로
   최초로 직접 측정.
2. **학습 라벨 로직 자체의 설계** — 사용자 요청으로 `deep-learning` 스킬(데이터 처리/클래스 불균형/
   검증전략 원칙)을 적용해 zigzag_action/quality/exit 세 헤드의 실제 타겟이 무엇인지 코드 레벨로
   재확인.

1차 리뷰(`eth_odyssey4_tabm_layer_design_review_20260816.md`)가 이미 확인한 것(정규화 배치,
AdamW decoupled decay, gradient clipping, mean-of-k-loss 정합성, purge/embargo 부재, 초기화 공백
등)은 여기서 반복하지 않고 표로만 인용한다.

## 1. 사전 확인 — 서버 GPU 상태 (중복 실행 방지)

`bash scripts/ops/handoff.sh status server eth_nhits_moderntcn_direction_quality` 재확인 결과
**여전히 RUNNING**(pid=762310) — ModernTCN 아키텍처가 isolation(정규화기 비교) 단계를 마치고
`stage=hpsearch, n_trials=25`(Optuna류 HP서치) 단계로 진입한 상태를 로그로 직접 확인했다. 이
때문에 C3(공유 트렁크, `eth_candidate_shared_trunk_regime_experts_contract_20260816.md`)의 N≥5
시드 본실험은 **이번 세션에서도 착수하지 않았다** — 계약서에 명시된 대로 GPU가 빌 때까지 대기,
같은 GPU에 새 무거운 작업을 얹지 않는다([[feedback_gpu_backlog_offload_to_server]]).

## 2. 신규 발견 A — BatchEnsemble k=8 멤버 붕괴 실측

### 2.1 방법

`scripts/diagnose_eth_odyssey4_batchensemble_member_collapse_20260816.py`(신규 작성) — 실제 배포된
zig075 formal 5-seed 번들(`tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_
20260620_pinned102_zig075_formal5seed_20260815_seed946043153/true_3head_tabm_bundle.pt`)을 로드해
`ThreeHeadTabM.encode()` → `direction_head`/`quality_head`가 반환하는 **멤버별(batch,k=8,3) 로짓**을
그대로 뽑아, 진짜 OOS 데이터(`eth_odyssey4_true_feature_pipeline_20260816.py`의 115피처 파이프라인,
2026 OOS, n=16,897행)에 흘려서 측정했다. 학습/추가 튜닝 없음 — 순수 read-only 분석.

지표: (1) 멤버 쌍별 top-confidence 상관계수, (2) k=8개 멤버 전원이 argmax 클래스에 동의하는
unanimity 비율, (3) 같은 marginal class 분포를 가진 **독립** k개 사본이었다면 기대되는 unanimity
(베이스라인), (4) 멤버간 예측확률의 cross-k 표준편차.

### 2.2 결과

| 전문가 | 헤드 | 멤버쌍 평균상관 | 실측 unanimity | 독립 베이스라인 unanimity | 초과분 | cross-k prob std |
|---|---|---:|---:|---:|---:|---:|
| bull | direction | 0.9985 | 97.17% | 0.06% | +97.12%p | 0.0036 |
| bull | quality | 0.9988 | 97.00% | 0.05% | +96.95%p | 0.0035 |
| bear | direction | 0.9980 | 97.14% | 0.14% | +97.00%p | 0.0037 |
| bear | quality | 0.9977 | 96.95% | 0.17% | +96.77%p | 0.0039 |
| chop | direction | 0.9972 | 96.28% | 0.06% | +96.22%p | 0.0046 |
| chop | quality | 0.9977 | 96.73% | 0.07% | +96.66%p | 0.0043 |

k=8개 멤버가 **사실상 하나의 함수로 붕괴**해 있다 — 상관계수 0.997+, 서로 독립이었다면 우연으로도
0.1% 남짓만 일어날 만장일치가 실제로는 96~97%에서 일어난다. 이건 참고자료가 "문헌상 우려"로만
인용했던 BatchEnsemble collapse(arXiv:2601.16936)를, 이 저장소의 배포 아키텍처에서 **처음으로 직접
측정으로 확인**한 것이다.

### 2.3 원인 진단 — 왜 붕괴하는가 (재확인 포함)

**1차 가설(가중치 감쇠가 다양성을 깎는다)은 기각**: `input_scale`/`expert_scale`(R게이트, 근사
항등원 초기화 `randn()*0.03+1.0`)에 `weight_decay=2e-4`가 다른 모든 파라미터와 동일하게 적용되고,
이 파라미터들은 평균이 1.0에서 벗어나면 감쇠가 0쪽으로 끌어당긴다는 게 최초 가설이었다.

**실측으로 검증**: 신규 스크립트
`scripts/diagnose_eth_odyssey4_diversity_growth_fresh_run_20260816.py`로 캐노니컬 CFG 기본값
그대로(epochs 예산 28, patience=8) 신선한 학습을 1회 직접 돌려서 확인했다(주의: 디스크에 있던
"formal5seed" 번들들은 전부 `docs/experiments/eth_omega461_zig075_direction_head_skill_formal_
nseed_20260815.md`가 `--epochs 2`로 명시적으로 캡을 건 빠른 스킬-존재 테스트였다 — `epochs_ran=2`가
찍힌 걸 보고 처음엔 "2에폭만에 조기종료"로 오해했는데, 실제로는 인위적 캡이었다. 진짜 29에폭 예산
학습이 필요해서 새로 돌렸다):

| | 초기화 시점 | 9에폭 학습 후(`epochs_ran=9`, 실제 patience 기반 조기종료) |
|---|---:|---:|
| `input_scale` std(멤버간, 피처 평균) | 0.02840 | 0.02724 |
| `input_scale` 평균 | ~1.0 | 0.99436 |
| `expert_scale[0]` std(멤버간) | ~0.03 | 0.02847 |
| `expert_scale[1]` std(멤버간) | ~0.03 | 0.02889 |

**결론**: 평균은 1.0 근처에서 거의 안 움직였다(0.994~1.002) — weight_decay의 "0쪽으로 당기는 힘"은
이 스케일(2e-4, 9에폭)에서 무시할 만큼 작다는 뜻이라 1차 가설은 지지되지 않는다. **진짜 원인은 더
단순하다**: 멤버간 표준편차가 초기화 시점(~0.028~0.03)에서 학습 후(~0.027~0.029)로 **거의 그대로,
오히려 미세하게 줄었다.** 손실함수(`loss = loss_dir + qw*loss_qual + ew*loss_exit`, 전부
`mean(dim=1)`로 k에 대해 평균낸 뒤 역전파)에는 멤버간 다양성을 보상하거나 벌하는 항이 전혀
없다 — 공유 트렁크(`in_proj`/`blocks`) 입장에서 8개 멤버는 전부 "같은 과제를 푸는 협력자"일
뿐이고, 그래디언트는 8개 멤버 각각을 (공유 신호 최적화라는) 같은 방향으로 밀어붙인다. 다양성은
순수하게 무작위 초기화(`randn()*0.03`)가 준 것이 전부이고, 학습은 그걸 키우지도 깎지도 않는 채로
거의 그대로 둔다 — TabM/BatchEnsemble의 "묵시적 앙상블 정규화"라는 설계 의도가 **이 정확한
아키텍처·라벨·학습 레시피 조합에서는 사실상 작동하지 않고 있을 가능성**을 실측이 뒷받침한다.

### 2.4a cheap_gate 실행 결과 (2026-08-16, 이어서) — "다양성 붕괴 = 공짜로 줄여도 됨" 가설은 기각 방향

`scripts/research_eth_odyssey4_batchensemble_k_reduction_cheap_gate_20260816.py` — 단일 시드
(260816), expert=bull, plain CE, 진짜 115피처 파이프라인, k∈{1,2,4,8}을 동일 조건에서 비교했다.
4개 조건 전부 `epochs_ran=9`(patience=8로 조기종료, best checkpoint는 4개 다 epoch 1)로 동일한
조기종료 타이밍이라 비교 자체는 공정하다.

| k | 파라미터수 | direction_val_loss | direction_bacc | quality_bacc |
|---:|---:|---:|---:|---:|
| 1 | 99,694 | 0.8662 | 0.5617 | 0.5706 |
| 2 | 100,308 | 0.8715 | 0.5609 | 0.5537 |
| 4 | 101,536 | 0.8795 | 0.5664 | 0.5674 |
| **8(현재)** | 103,992 | 0.8729 | **0.5710** | 0.5662 |

k=8 대비 dir_bacc 델타: k=1 −0.0093, k=2 −0.0101, k=4 −0.0046 — **셋 다 음수, 즉 이 단일 시드에서는
k를 줄일수록 direction_balanced_accuracy가 일관되게(작지만) 나빠졌다.** "다양성이 이미 붕괴했으니
k를 줄여도 공짜"라는 원래 가설을 뒷받침하지 않는다 — 오히려 반대 방향의 예비 신호다. (val_loss
자체는 direction/quality 성분에서는 명확한 단조 경향이 없었고, k=2에서 유독 컸던 전체 val_loss
차이(+0.35)는 exit_val_loss 성분에 몰려 있었다(1.27 vs 나머지 0.98~1.01) — exit 미니배치 반복자의
별도 무작위성 때문일 가능성이 높고 k 자체의 체계적 효과로 보기 어렵다.)

**해석상 유의점**: 단일 시드이고, 이 저장소 자체 히스토리(`tabm_hp_low_signal_pattern` 등)가 반복
확인한 대로 이 정도 크기(0.005~0.01)의 델타는 시드 노이즈만으로도 나올 수 있는 범위다 — "확정적으로
나쁘다"고 결론내리기엔 이르지만, **적어도 "공짜로 줄여도 된다"는 기대를 뒷받침하는 방향은 전혀
아니다.** 이 축의 원래 동기(붕괴된 앙상블은 계산 낭비이니 줄이자)가 예비 결과로 이미 약화됐으므로,
N≥5시드 본실험에 자원을 쓸지는 사용자 판단으로 남긴다 — 붕괴 자체(§2.2)의 사실관계는 이 결과와
무관하게 그대로 유효하다.

### 2.4b R게이트 초기화 다양성 확대 cheap_gate (2026-08-16, 이어서) — "붕괴를 받아들이고 줄이기" 대신 "붕괴를 고치기"

k축소(§2.4a)가 실패 방향으로 나온 뒤, 반대 전략을 시도했다: k=8은 유지하되 `input_scale`/
`expert_scale`의 초기화 분산 자체를 키워서(현재 std=0.03 → 0.1/0.2) 학습이 다양성을 스스로 못
키운다는 §2.3의 발견을 "초기화로 앞당겨 준다." 같은 시드(260816), expert=bull, k=8·
same_as_direction 품질타겟 고정(이 축 하나만 격리), 학습 후 붕괴 진단(§2.1과 동일 지표)까지
같이 측정했다(`scripts/research_eth_odyssey4_batchensemble_wide_init_cheap_gate_20260816.py`).

| std(요청값) | 학습후 실측 gate std | dir_bacc | k=8·std=0.03 대비 | 학습후 멤버쌍 상관 | unanimity |
|---:|---:|---:|---:|---:|---:|
| 0.03(현재) | 0.0119 | **0.5716** | — | 0.9977 | 96.72% |
| 0.1 | 0.0338 | 0.5684 | −0.0032 | 0.9773 | 89.30% |
| 0.2 | 0.0745 | 0.5652 | −0.0063 | 0.9269 | 80.54% |

**절반의 성공**: 넓은 초기화는 실제로 학습을 버티고 살아남는다 — 학습후 gate std가 요청값에 거의
비례해서 커지고(0.0119→0.0338→0.0745), 멤버쌍 상관도 실질적으로 낮아진다(0.998→0.977→0.927,
unanimity 96.7%→80.5%) — §2.3에서 예측한 대로 "다양성은 초기화가 주는 만큼만 존재한다"가 다시
확인됐고, 이번엔 그 다양성을 초기화로 실제로 늘리는 데 성공했다. **하지만 direction_bacc는
넓힐수록 오히려 계속 나빠졌다**(단조 감소, −0.0032~−0.0063) — 다양성을 진짜로 늘렸는데도
정확도는 안 따라왔다. 다만 std=0.1의 델타(−0.0032)는 이번 세션에서 나온 세 후보(k축소
−0.005~−0.01, quality분리 −0.0140, 이번 −0.0032~−0.0063) 중 가장 작아서, 단일시드 노이즈 범위에
가장 가깝다.

**세 후보 종합**: 이번 세션에서 서로 다른 메커니즘으로 접근한 세 가지 시도(k 축소, quality 타겟
분리, R게이트 다양성 확대) **전부 단일시드 예비신호가 현재 배포 설정(k=8, std=0.03,
same_as_direction)보다 나쁜 방향**으로 나왔다 — 현재 설정이 이 세 축 모두에서 이미 로컬 최적점에
가깝다는 정황이 쌓이고 있다.

### 2.5 학습률 격리 cheap_gate (2026-08-17) — 넷째 후보, 다른 메커니즘이지만 결론은 비슷

세 후보(k축소/quality분리/다양성확대) 전부 정확히 같은 패턴을 보였다 — best checkpoint가 항상
1에폭째였고 그 뒤로는 계속 나빠졌다(전형적 memorization). 이 저장소 자체 연구
(`feedback_modern_dl_training_checklist`)가 이미 "lr=2e-4가 정점은 비슷하지만 훨씬 천천히
무너진다"는 걸 찾았지만, 그건 cosine+AdaBelief+GCE+Prechelt 묶음의 일부로만 테스트됐고(그 묶음
전체는 N=5시드에서 OLD 레시피에 패배), **실제 라이브 설정(k=8·std=0.03·same_as_direction) 위에
lr만 단독으로 격리 테스트된 적은 없었다.** 같은 시드(260816)·expert=bull, lr만 2e-3(현재)↔2e-4로
바꾸고 나머지는 전부 라이브 기본값 그대로, epoch 예산만 40으로 늘려 여유를 줬다
(`scripts/research_eth_odyssey4_lr_isolation_cheap_gate_20260816.py`).

| lr | epochs_ran | 선택된 checkpoint | selected dir_bacc | true peak dir_bacc(에폭) |
|---|---:|---:|---:|---:|
| 2e-3(현재) | 9 | epoch 1 | **0.5710** | 0.5710(epoch 1) — 선택=진짜 정점 |
| 2e-4(후보) | 10 | epoch 2 | 0.5432 | **0.5638(epoch 5)** — 선택≠진짜 정점 |

**둘로 나눠지는 결과**: (1) `selected_dir_bacc_delta=−0.0278` — 그대로 배포했다면 지금까지 나온
후보 중 가장 나쁨. (2) 하지만 원인이 분명하다 — lr=2e-4에서는 combined val_loss 조기종료 기준이
진짜 정점(에폭5, 0.5638)을 놓치고 에폭2(0.5432)에서 멈췄다. `true_peak_dir_bacc_delta`는
−0.0072로, 지금까지 후보 중 가장 작은 축에 든다(다양성확대 std=0.1의 −0.0032 다음). **새로운
발견**: `feedback_modern_dl_training_checklist`가 "combined val_loss 기준은 임베딩 없는 plain
아키텍처에서는 안전하다(gap 0.0000-0.0008)"고 확인한 건 **lr=2e-3 기준이었다** — 이번 결과는 같은
plain 아키텍처라도 **lr을 낮추면 그 안전함이 깨진다**는 걸 보여준다(gap 0.0206). 선정기준의
신뢰성이 아키텍처뿐 아니라 lr에도 의존한다는, 이 프로젝트에 없던 새 사실이다.

**결론**: `lr=2e-4` 단독은 진짜 정점 기준으로도 살짝 나쁘고(−0.0072), 그대로 배포하면 선정기준
붕괴 때문에 훨씬 나쁘다(−0.0278) — 이미 닫힌 "묶음 레시피가 OLD를 못 이긴다"는 결론과 같은
방향이지만, 이번엔 lr 하나만 격리해도 마찬가지라는 걸 처음으로 직접 확인했다. 넷째로 시도한
서로 다른 메커니즘(k, quality타겟, 다양성, lr)이 전부 현재 배포 설정보다 나쁜 방향으로 수렴하고
있다.

### 2.4 이게 실제로 문제인가 — 열린 질문

붕괴 자체는 확인됐지만, **이게 실제로 direction_balanced_accuracy나 PnL을 깎고 있는지는 별개
질문이고 아직 미검증**이다. 두 가지 가능성이 남아있다:
- (a) k=8 구조가 다양성 없이도 여전히 뭔가 유익한 정규화 효과(예: 학습궤적 중 매 스텝 8개 노이즈
  섞인 forward pass 평균이 일종의 데이터 증강처럼 작동)를 주고 있을 수 있다.
- (b) 다양성이 없으니 k=8은 사실상 계산 낭비이고, k를 크게 낮춰도(k=1 즉 평범한 단일 MLP까지) 정확도
  손실 없이 거의 동일한 결과를 얻을 가능성이 있다.

기존에 닫힌 축과 안 겹침: `feedback_modern_dl_training_checklist`의 "k=32로 올리는 것은 3/3 전문가
악화로 닫힘"(용량을 늘리는 방향)과 "R+S+B 완성형은 학습신뢰성 문제로 패배"(어댑터를 더 표현력있게
만드는 방향)는 둘 다 **다양성 메커니즘 자체를 늘리거나 완성하는** 시도였다. **k를 낮추는 방향(k=1,
2, 4)은 이 저장소 히스토리에서 한 번도 테스트된 적이 없다** — "다양성이 실제로 붕괴해있다"는 이번
실측이 이 미시도 방향에 처음으로 구체적 근거를 준다.

## 3. 신규 발견 B — 학습 라벨 로직 조사 (`deep-learning` 스킬 적용)

사용자 요청에 따라 `deep-learning` 스킬(데이터 처리 원칙: 클래스 불균형 처리, 검증전략, 파이프라인
정합성)의 렌즈로 direction/quality/exit 세 헤드가 실제로 무엇을 예측하도록 학습되는지 코드를 직접
추적했다.

### 3.1 direction_head/quality_head 타겟 — `zigzag_action`

`_add_labels()`(`scripts/train_omega1_direction_head_direction_only_20260602.py:177`)는 미리 계산된
`tmp/causal_regen_20260516/zigzag_action_labels_20260531/zigzag_action_labels_{year}.csv`를 그대로
읽는다. 생성 로직(`scripts/build_zigzag_action_labels_v2_20260604.py:_zigzag_pivots`, B2 진단
스크립트 `scripts/diagnose_odyssey4_zigzag_pivot_confirmation_delay_20260816.py`가 그대로
재구현+계측)은 **ATR 적응형 되돌림(reversal) 상태기계**다:

```
threshold(i) = clip(max(0.01, ATR14_pct(i) * 1.0), 0.01, ∞)   # v1 실측 파라미터, 상한 클램프 없음
현재 추세(상승/하락) 중 극값 대비 threshold 이상 되돌리면 그 극값 bar를 "확정된 피벗"으로 기록,
추세를 반전시킨다.
```

**핵심 특성 — 비인과적 확정(repainting)**: 피벗은 `pivot_idx`(극값이 실제로 찍힌 봉)가 아니라
`confirm_idx`(그 이후 되돌림이 threshold를 넘는 순간)에서야 "확정"된다. 즉 `pivot_idx` 근방 봉들의
`zigzag_action` 라벨이 맞는지는 **미래의 `confirm_idx` 봉이 와야 알 수 있다.** B2 진단이 실측한
확정 지연(confirm_idx − pivot_idx)의 분포: **p95=54봉, 최대값은 이보다 훨씬 큼**(메모리
`eth_odyssey4_layer_and_parameter_improvement_proposal_20260816`에 이미 기록).

이건 C1(purge/embargo 갭)이 이미 다룬 "분할 경계 근처 학습샘플 누출" 문제와 같은 근본원인이지만,
**더 넓은 함의가 있다**: 이 라벨은 구조적으로 "쉬운 다수"와 "어려운 소수"로 나뉜다 — 이미 확정된
추세 한가운데 있는 봉(다수)은 최근 가격 모멘텀만 봐도 라벨을 맞히기 쉽고, 진짜 가치있는 반전 직전
봉(소수)은 라벨 자체가 최대 수십~수백 봉 뒤의 정보로 사후 결정된 것이라 원래 어렵다. 이건 이미
`repo_label_methodology_meta_finding`이 zigzag류 라벨 전반에 대해 "방향-알파 테스트를 통과 못
한다"고 정리한 것과 **정확히 같은 실패 패턴을 메커니즘 레벨에서 재확인**한다 — 모델이 학습 초반
쉽게 배우는 것(train_bacc가 빠르게 오르는 것)의 정체가 "모멘텀 따라하기"이고, 그게 바로 §2에서 본
"학습이 몇 에폭 안 가 val_loss 최적점에 도달하는" 이유(과제 자체가 얕은 힌트로 대부분 풀리기
때문)라는 그림과 일치한다.

### 3.2 quality_head — 실질적으로 direction_head의 중복 (신규 확인, **정정됨** — 아래 3.2a 참고)

캐노니컬 학습 루프(`_fit_expert_3head`, 라인 279-284)를 직접 읽으면:

```python
loss_dir_k  = cross_entropy(out_dir["direction"], yb, ...)   # yb = zigzag_action
loss_qual_k = cross_entropy(out_dir["quality"],   yb, ...)   # 같은 yb!
```

**quality_head는 direction_head와 정확히 같은 타겟(`zigzag_action`)으로 학습된다** — 별도의
품질/기대수익 신호가 아니라, 손실가중치(`quality_loss_weight=0.80`)만 다른 **중복 헤드**다. 배포
번들 메타데이터(`report.json`)의 `"quality_target_rule"` 필드에는 `net_return_after_cost_min=0.001,
mae_max=0.01, mfe_mae_min=1.2, max_hold_bars=288` 같은 훨씬 정교한 트리플배리어류 품질 규칙이
정의돼 있고 `"quality_mode": "same_as_direction"`으로 명시돼 있다 — 즉 **더 풍부한 품질 타겟을 쓸
수 있는 인프라(라벨 계약)는 이미 존재하는데, 캐노니컬 스크립트가 실제로 쓰는 건 "direction과
동일"이라는 가장 단순한 모드뿐**이라는 뜻이다.

**메모리와의 불일치 플래그**: `h48qual_label_mismatch_discovered` 메모는 h48qual의 "진짜 레시피"를
"zigzag_action(방향)+48bar-ATR-barrier(품질)"로 기록하고 있다 — 이는 quality_head가 direction과
**별개의** 48봉 ATR 배리어 타겟을 쓴다는 뜻인데, 이번에 직접 읽은 캐노니컬 스크립트
(`train_eval_omega1_2_tabm_3head_20260603.py`, 이 문서 서두에서 "h48qual·zig075 둘 다 공유하는 정확히
같은 구조"라고 명시된 그 스크립트)는 quality_head를 direction과 동일한 타겟으로 학습한다. 둘 다 같은
세션의 서로 다른 메모/문서에서 나온 주장이라 **이 자체로 새 결론을 내리진 않지만, 다음 세션에서
"어느 쪽이 실제 배포 아티팩트를 만든 스크립트인가"를 먼저 확인해야 하는 미해결 불일치로 명시적으로
남긴다.**

**실행 가능한 신규 질문**: quality_head를 direction_head의 중복이 아니라, 배포 인프라에 이미
정의돼 있는 진짜 품질 배리어 타겟(`quality_target_rule`)으로 바꾸면 (a) 두 헤드가 서로 다른, 상호
보완적인 신호를 배우게 되어 공유 트렁크 표현이 더 풍부해질 수도 있고, (b) 반대로 이미
`eth_odyssey4_gce_canonical_port_20260816`(A1)이 "손실함수만 바꾸는 방향은 이미 N=5시드로 부정됨"과
같은 실패 패턴을 반복할 수도 있다 — **미검증, cheap_gate로 먼저 걸러야 할 신규 후보.**

### 3.2a 정정 (2026-08-16, 이어서) — 배포 아티팩트를 만드는 진짜 스크립트가 달랐다

§3.2는 `train_eval_omega1_2_tabm_3head_20260603.py`(MODEL_ID=`omega1_2_true_3head_tabm_20260603`)를
읽고 쓴 것인데, 배포 번들 `report.json`의 진짜 `model_id`는
**`omega4_3head_parent72_loose_entry_quality_20260620`**다 — 이 둘은 다른 스크립트다. 실제 소스는
`scripts/train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py`(`parent`로
`train_eval_omega1_2_tabm_3head_20260603.py`를 임포트해 `ThreeHeadTabM`은 재사용하되, quality 타겟
로직을 자체적으로 확장한 상위 스크립트)이고, 여기엔 `--quality-mode` 플래그가 있다
(`same_as_direction`/`hard_rule`/`quality_label_action`/`quality_label_hard_rule`/
`barrier_meta_action`/`risk_adjusted_barrier_meta_action`, **기본값은 `hard_rule`**) — 즉 direction과
완전히 별개인 품질 타겟 인프라가 이미 구현돼 있다.

**라이브 계약 문서로 확정**(`docs/model_contracts/odyssey4_eth_full_stack_architecture_20260814.md`
63행): "h48qual과 zig075는 같은 아키텍처, 독립적으로 학습된 가중치 — 라벨 계열이 다르다
(h48qual=`zigzagfix_06`+48bar ATR barrier quality; zig075=`zigzag_action`+동일 라벨의 quality,
`quality_mode=same_as_direction`)." 즉:

- **h48qual**: 이미 진짜 별도 품질 타겟(48봉 ATR 배리어)을 쓰고 있다 — §3.2의 우려는 h48qual에는
  해당 안 된다. `h48qual_label_mismatch_discovered` 메모의 기존 주장이 맞았다.
- **zig075**: **실제로 라이브에서 `quality_mode=same_as_direction`을 쓴다** — §3.2의 발견은
  틀리지 않았고, 격리 실험용 번들의 우연이 아니라 **zig075의 진짜 배포 설정**이었다. 이 부분만
  후보로 유효하다.

이 스크립트에 이미 구현된 `_quality_target_risk_adjusted_barrier_meta_action`(barrier 순전파
시뮬레이션, net_return_after_cost_min/mae_max/mfe_mae_min/max_hold_bars 임계값 사용, `report.json`에
기록된 기본값 그대로: 0.001/0.01/1.2/288봉)을 재사용해 zig075의 quality_head 타겟만 바꿔보는
cheap_gate를 실행했다(`scripts/research_eth_odyssey4_quality_target_separation_cheap_gate_20260816.py`).
새 라벨 로직을 만들 필요 없이 CLI 플래그 하나 차이라 구현 리스크가 낮다.

### 3.2b cheap_gate 실행 결과 — 이 후보도 기각 방향

단일 시드(260816), expert=bull, plain CE, 진짜 115피처 파이프라인, 두 조건 모두 동일 조기종료
타이밍(epoch 1)으로 공정 비교:

| 조건 | direction_bacc | direction_val_loss | quality_bacc(자기 타겟 기준, 비교불가) |
|---|---:|---:|---:|
| same_as_direction(현재 라이브) | **0.5710** | 0.8729 | 0.5662 |
| risk_adjusted_barrier(후보) | 0.5570 | 0.8808 | 0.5173 |

`dir_bacc_delta = −0.0140` — **k축소 후보(§2.4a, −0.005~−0.01)보다 더 큰 음수.** barrier 타겟은
매우 희소하다(active bar 중 12.84%만 통과, 클래스분포 [69664 CASH, 4441 long, 4463 short] vs
same_as_direction의 [9243, 36283, 33042]) — `compute_sample_weight(balanced)`로 클래스 가중은
줬지만, 극단적 희소성 자체가 quality_head를 거쳐 공유 트렁크로 들어가는 그래디언트를 더 노이즈있게
만들어 direction_head 쪽 표현까지 끌어내렸을 가능성이 있다(멀티태스크 간섭). 단일 시드라 확정은
아니지만, 이번 세션에서 나온 두 신규 후보(k축소, quality 타겟 분리) **둘 다 예비 결과가 현재
배포 설정보다 나쁜 방향**으로 나왔다 — 현재 설정이 이미 이 두 축에서는 로컬 최적점에 가깝다는
정황.

### 3.3 exit_head — 독립적으로 잘 설계된 타겟 (참고용, 문제 없음)

`_build_exit_dataset_independent`(`scripts/train_eval_omega1_2_tabm_exit_head_20260603.py:535`)는
direction/quality와 달리 **진짜 별개의 라벨**이다: 실제 체결가 시뮬레이션 → TP/SL 배리어까지
전진(`_continue_to_barrier_net`) → 여러 hold_offset 시점에서 MFE/MAE/미실현손익을 계산해 "지금
청산 vs 계속 보유"의 이진 라벨을 만든다. 미래 봉을 순차적으로 causal하게 걸어가며 만드는 구조라
zigzag_action 같은 사후 확정(repainting) 문제가 없다 — `giveback_exit_label_uniform_policy_
pattern_20260815` 메모가 지적한 "낮은 MFE 임계값이 정책을 단순화시킨다"는 별개의 이슈는 있지만, 이번
조사 범위(라벨이 direction과 중복인가/비인과적인가)에서는 exit_head가 유일하게 "문제 없음"으로
확인됐다.

## 4. 종합 — 우선순위별 신규 후보

| 우선순위 | 후보 | 근거 | 다음 단계 | 기존 축과의 관계 |
|---|---|---|---|---|
| 보류 | **BatchEnsemble k 축소(k=8→1/2/4)** | §2 실측: k=8 멤버 상관 0.997+, 다양성이 학습으로 안 자람. **§2.4a cheap_gate(단일시드) 완료: k=1/2/4 전부 dir_bacc가 k=8보다 −0.005~−0.01 낮음 — "공짜로 줄여도 된다"는 가설을 뒷받침 안 함** | N≥5시드로 확정할지 사용자 판단 대기(예비신호가 이미 채택 방향에 불리함) | k=32(증가) 실패와 다른 축, 붕괴 사실 자체는 유효·확인됨 |
| 보류 | **R게이트 초기화 다양성 확대(std 0.03→0.1/0.2)** | §2.4b cheap_gate(단일시드) 완료: 넓은 초기화는 실제로 학습 후에도 다양성을 유지시킴(멤버쌍 상관 0.998→0.927)을 확인했지만, dir_bacc는 단조 감소(−0.0032~−0.0063) — std=0.1의 델타는 이번 세션 세 후보 중 가장 작아 노이즈 경계에 가장 가까움 | std=0.1만 N≥5시드로 재확인할지 사용자 판단 대기 | k축소·quality분리와 독립적인 셋째 축, 예비신호 역시 불리하지만 가장 약함 |
| 보류 | **zig075 quality_head를 `risk_adjusted_barrier_meta_action`으로 분리** | §3.2a: h48qual은 이미 별도 타겟 사용 중(문제 없음), zig075만 실제로 same_as_direction. **§3.2b cheap_gate(단일시드) 완료: dir_bacc −0.0140, k축소보다도 더 나쁜 방향** — barrier 타겟이 매우 희소해(양성 12.84%) 멀티태스크 간섭을 일으켰을 가능성 | N≥5시드로 확정할지, 혹은 덜 엄격한 임계값(예: min_edge/max_mae 완화)으로 재시도할지 사용자 판단 대기 | A1(GCE, 손실함수 교체) 실패와는 다른 축(타겟 자체를 바꿈), 예비신호는 불리함 |
| 대기중 | **C3(공유 트렁크)** | 이미 구현/리뷰/sanity check 완료, 계약서 실행 명령 확정 | GPU 확보되는 즉시 실행(§1 확인 절차 그대로) | 기존 계획, 변경 없음 |
| 대기중 | **N-HiTS/ModernTCN 대안 아키텍처** | 서버에서 진행 중(hpsearch 단계) | 완료 대기, 결과만 확인 | 기존 계획, 변경 없음 |
| 낮음 | 명시적 가중치 초기화(Xavier/He) | 1차 리뷰(§3)가 이미 낮은 우선순위로 명시, 이번 실측도 이를 반박하지 않음 | 급하지 않음 | 기존 결론 유지 |
| **CLOSED** | **Rank-r BatchEnsemble 게이트(LoMETab, arXiv:2605.14365)** | §6.8 N≥5시드(진짜 랜덤) 확정: 정확도 델타(선택/정점 둘 다) 표준편차가 평균의 1.6~14배, 4개 비교 전부 `sign_consistent=False` — `tabm_hp_low_signal_pattern` 서명. **핵심 가설(다양성 증가)도 명시적으로 기각**: 진짜 정점 에폭에서 측정한 멤버쌍 상관 델타가 5시드 평균 −0.00003(r=2)/−0.00010(r=3), 개별 시드도 랜덤 부호로 흩어짐 — 측정 타이밍 문제가 아니라 실제로 다양성이 안 늘어난다 | 없음 — CLOSED | k축소·초기화확대(둘 다 r=1 안에서만 움직임)와도, R+S+B 완성형(닫힘, "고정 S 추가"였을 뿐)과도 다른 축이었으나 결론은 동일 |
| **CLOSED** | **손실함수 다양성항(NCL, Liu&Yao 1999)** | §7.4 N≥5시드 확정: λ=7에서 5/5 시드 전부 정확도 악화(−0.0497±0.0076, 표준편차가 평균의 15%뿐 — 이 조사 전체에서 가장 노이즈 아닌 결과)이면서 5/5 시드 전부 다양성 대폭 증가(상관 0.998→0.485) — **다양성 증가에는 유일하게 성공했지만 대가가 확정적으로 너무 크다.** λ=2는 시드마다 들쭉날쭉(3개 시드 무변화, 2개 시드 급격히 붕괴) — "안전한 작은 λ"는 존재하지 않음 | 없음 — CLOSED. **이로써 이 아키텍처에서 시도 가능한 6가지 다양성 완화 메커니즘(k축소·초기화확대·quality분리·lr격리·rank-r·NCL손실항) 전부 종결** — 남은 가설은 라벨(zigzag_action의 비인과적 확정) 자체뿐 | rank-r과 정반대 실패 양상(rank-r=다양성 자체가 안 늘어남, NCL=다양성은 늘지만 정확도가 확실히 깨짐) — 서로 다른 방향에서 같은 결론에 수렴 |

**이번 세션에서 실행하지 않은 이유**: 두 신규 후보(k축소, quality 타겟 분리) 모두 이 저장소의
표준 게이트(cheap_gate 단일시드 → N≥5시드 → fresh-forward VAL/OOS)를 거쳐야 promotion 근거가 되고,
`feedback_dl_needs_optimization_before_failure_verdict`/`feedback_modern_dl_training_checklist`가
반복 확인한 대로 **개별 조각이 단독으로 좋아 보여도 조합/N≥5시드에서 뒤집히는 패턴이 이 저장소에서
매우 흔하다** — 이 문서는 "무엇을 다음에 테스트할지"에 대한 근거를 실측으로 마련하는 것까지가
범위이고, 실제 게이트 통과는 다음 단계다.

## 5. 재사용 가능 산출물

- `scripts/diagnose_eth_odyssey4_batchensemble_member_collapse_20260816.py` — 임의의 3-head TabM
  번들에 대해 BatchEnsemble collapse를 측정하는 범용 진단 스크립트.
- `scripts/diagnose_eth_odyssey4_diversity_growth_fresh_run_20260816.py` — 캐노니컬 CFG로 신선한
  학습 1회를 돌려 다양성 파라미터의 초기화 대비 성장을 측정.
- `tmp/diag_batchensemble_collapse_oos_20260816.json` — §2.2 표의 원본 수치.

## 6. 신규 발견 C — Rank-r BatchEnsemble 게이트 (LoMETab, arXiv:2605.14365) — CLOSED, N≥5시드 확정 negative (2026-08-17)

사용자가 Odyssey4 아키텍처를 다이어그램(`docs/eth_odyssey4_tabm_architecture_20260817.html`)으로
정리한 뒤, "`deep-learning` 스킬로 개선안을 만들어달라"고 이어서 요청해 §2의 붕괴 진단을 다시 읽고
arXiv를 직접 조회(2026-08-17)해서 찾은 신규 후보. §2.4a(k 축소)·§2.4b(초기화 분산 확대) 둘 다
예비negative였던 이유를 설명하는 논문을 찾았고, 그 논문이 제안하는 정확한 수정은 이 저장소에서
아직 한 번도 테스트되지 않은 축이다. **미실행 — 구현 스펙과 테스트 계획까지만.**

### 6.1 §2.4b가 막혀 있었던 구조적 이유

§2.4b는 `input_scale`/`expert_scale`의 초기화 표준편차만 키웠다(0.03→0.1→0.2). 학습 후에도
다양성은 살아남았지만(멤버쌍 상관 0.998→0.927) direction_bacc는 단조로 나빠졌다. 우연이 아니라
구조적 필연일 수 있다 — 이 저장소의 게이트는 TabM 논문(arXiv:2410.24210) 수식
`l_BE(X)=((X⊙R)W)⊙S+B` 중 **R만** 구현하고 S를 생략한 버전이다(`eth_candidate_faithful_tabm_
batchensemble_20260816.md`가 이미 확인). S가 없으면 멤버 k의 유효 가중치는
`W_k = W ⊙ (r_k ⊗ 𝟙)`(𝟙=전부 1인 벡터로 고정) — **W의 각 행을 스칼라 하나로 균일하게 스케일하는
것**과 수학적으로 동일하다. 이 구조에서 σ만 키우면 스케일 강도만 세지고, 멤버가 표현할 수 있는
함수 공간(hypothesis class) 자체는 넓어지지 않는다 — 다양성은 늘어도 그 다양성의 "방향"엔 상한이
있다는 뜻이고, 이게 §2.4b에서 다양성은 늘었는데 정확도는 계속 떨어진 현상과 들어맞는다.

### 6.2 문헌 — LoMETab (arXiv:2605.14365, 2026-05-14, cs.LG)

arXiv API 직접 조회(`abs:"BatchEnsemble" AND abs:diversity`, 접근일 2026-08-17)로 찾음. 저자
Choi/Park/Kwon/Jeong. 초록 핵심(원문 인용):

> "We propose LoMETab, a rank-$r$ generalization of multiplicative implicit ensembles. LoMETab
> lifts the rank-1 BatchEnsemble/TabM modulation to a rank-$r$ identity-residual Hadamard family
> by parameterizing each member weight as $W_k = W \odot (1 + A_kB_k^\top)$ ... we prove that for
> $r \ge 2$ this generalization strictly enlarges BatchEnsemble's hypothesis class. Empirically ...
> LoMETab sustains higher pairwise KL than an additive low-rank ablation, and $(r,
> \sigma_{\mathrm{init}})$ provides broad control over pairwise KL ..."

§6.1에서 진단한 문제(R-only rank-1은 σ를 키워도 표현력 상한이 있다)를 이 논문이 **증명**하고
(r≥2에서 가설공간이 엄밀히 더 큼), **실측**으로도 확인한다(rank-r이 학습 후에도 pairwise KL
다양성을 더 잘 유지). 논문 스스로 "additive low-rank ablation"(단순히 파라미터를 더하는 방식)은
자신들의 곱셈적(Hadamard) rank-r 구성보다 다양성 유지가 약하다고 명시한다 — 이는 이 저장소가 이미
닫은 R+S+B 완성형(Step A, N=5시드 패배 확정)이 "그냥 파라미터를 더 넣는" 접근이었을 가능성과,
LoMETab이 더 원칙있는 다른 접근이라는 차별점을 논문 스스로 뒷받침한다.

### 6.3 이 저장소 메커니즘에 정확히 매핑

`ThreeHeadTabM.encode()`(`trading_bot_modules/odyssey_tabm_core.py:94-103`)의 게이트 2곳
(`input_scale`, 블록별 `expert_scale`)은 전부 R-only다: `xk = x ⊙ input_scale_k` → `h = in_proj(xk)`
(공유 W), 블록도 동일 패턴. LoMETab식 rank-r(r≥2)로 올리려면 **R쪽뿐 아니라 지금 없는 S쪽도 함께
rank-r로 새로 만들어야 한다** — r=1에서 S=𝟙(고정)이던 것을 r개의 (A열,B열) 쌍으로 확장:

```
h_k = Σ_{c=1}^{r} [ (x ⊙ A_k[:,c]) @ W ] ⊙ B_k[:,c]      # in_proj 위치, W는 그대로 공유
```

- `A_k ∈ R^{115×r}`(입력쪽), `B_k ∈ R^{192×r}`(출력쪽) — 멤버마다 독립.
- **필수 sanity check**: `r=1`이고 `A_k[:,0]`을 현재 `input_scale_k`와 같은 분포로, `B_k[:,0]=𝟙`
  (학습 안 함)으로 고정하면 현재 라이브 구조와 수치까지 정확히 같아져야 한다 — 구현 후 가장 먼저
  확인.
- 잔차 블록의 `expert_scale`도 같은 방식(입력쪽 A + 출력쪽 B)으로 확장. `W`(`in_proj`/블록
  `Linear`) 자체는 그대로 공유 — 바뀌는 건 게이트 파라미터화뿐이라 공유 가중치 용량은 그대로다.

파라미터 증가량은 작다: in_proj 게이트만 봐도 r=2에서 8×115×2(A)+8×192×2(B)≈4,912개 추가(기존
118,552 대비 +4.1%) — 이번 세션 다른 후보들(k=32가 +15%로 악화됐던 것)보다 훨씬 작은 증가폭이라
"용량 늘려서 악화" 패턴이 재현될 위험은 상대적으로 작다.

### 6.4 기존 후보와의 차별점

| 후보 | 무엇을 바꿨나 | 결과 |
|---|---|---|
| §2.4a k축소 | k(멤버 수)만 줄임, 게이트는 그대로 rank-1 | 예비negative |
| §2.4b 넓은 초기화 | rank-1 게이트의 초기화 σ만 키움 | 예비negative, §6.1이 구조적 이유 설명 |
| R+S+B 완성(닫힘) | r=1 그대로, S·B·수치형 임베딩을 "추가"만 함 | N=5시드 확정 negative |
| **rank-r 게이트(이번 제안)** | **r 자체를 1→2/3로 올려 가설공간을 논문이 증명한 방식으로 확장** | **미검증** |

k축소·초기화확대와는 독립 축(둘 다 r=1 안에서만 움직였음)이고, R+S+B 완성형과도 다르다(그건
"고정 S"를 추가한 것, 이건 R과 S를 함께 "rank-r 시리즈"로 일반화하는 것 — 논문이 이 차이가
다양성 유지력에서 실제로 갈린다고 실측으로 보여준다).

### 6.5 테스트 계획 (이 문서의 기존 cheap_gate와 동일 프로토콜)

1. **구현**: `RankRThreeHeadTabM` 신규 클래스, §6.3 수식 그대로.
2. **필수 sanity check**: `r=1, B고정` 조건에서 현재 `ThreeHeadTabM`과 동일 입력·동일 시드로
   forward 출력이 수치까지 일치하는지 먼저 확인(불일치 시 구현 버그, 결과 폐기 후 재구현).
3. **cheap_gate**: 단일 시드(260816), expert=bull, plain CE, 진짜 115피처 파이프라인(이 문서
   §2.4a/b/§3.2b와 동일 하네스, `research_eth_odyssey4_batchensemble_wide_init_cheap_gate_
   20260816.py` 구조 재사용), `r ∈ {1(재현 검증용), 2, 3}` × `σ_init=0.03`(고정 — 논문이 r과
   σ_init을 독립 축이라 하므로 1차는 r만 격리). §2.1과 동일한 붕괴 진단(멤버쌍 상관, unanimity)도
   학습 후 같이 측정 — "다양성이 실제로 늘었는가"와 "정확도가 따라왔는가"를 분리해서 봐야 §2.4b와
   같은 오독을 놓치지 않는다.
4. **판정**: r≥2가 dir_bacc를 개선하지 못하면(혹은 §2.4b처럼 다양성만 늘고 정확도가 안 따라오면)
   이 축도 §2.4b와 같은 결론(다양성 부족이 진짜 병목이 아니었다)으로 수렴 — 그 자체로 유의미한
   정보.
5. **승격 경로**: cheap_gate 통과 시에만 N≥5 시드 본실험, 그다음 VAL/OOS fresh-forward — 이
   저장소 표준 게이트를 그대로 따른다.

### 6.6 실패 시나리오 (사전 등록)

- rank-r이 다양성은 늘리지만(§2.1 지표 개선) direction_bacc는 §2.4b처럼 안 따라오는 경우 — §3.1이
  이미 지적한 라벨 자체의 근본 난이도(zigzag_action의 비인과적 확정/repainting)로 결론이 수렴할
  가능성. 이 경우 게이트 구조를 더 파는 것보다 라벨 재설계 쪽이 우선순위가 높아진다는 신호로
  기록해야 한다.
- 구현 복잡도가 실제 이득 대비 과한 경우(r=2~3에서 학습 속도가 느려지는데 개선폭이 노이즈 수준) —
  `tabm_hp_low_signal_pattern`(이 저장소 히스토리)이 반복 확인한 패턴이라 낮지 않은 확률로
  예상해야 한다.

### 6.7 구현 + cheap_gate 실행 결과 (2026-08-17)

**구현**: `scripts/research_eth_odyssey4_rankr_gate_cheap_gate_20260817.py` — `RankRGate`/
`RankRThreeHeadTabM` 신규 클래스. §6.3 수식을 그대로 구현하되, 논문의 `1+A_kB_k^T`(항등-잔차)
파라미터화 대신 더 단순한 `A_kB_k^T`(항등항 없이 r개 outer product 합) 형태를 썼다 — r=1에서
B를 1로 고정하면 현재 라이브 구조를 **그대로** 재현하게 만들기 위한 선택이며, "rank-r이 rank-1을
엄밀히 포함한다"는 논문의 핵심 성질에는 영향 없다(어느 파라미터화든 r=1은 r≥2 안에 중첩됨).

**필수 sanity check 통과**: r=1(B 고정) 조건에서 동일 가중치를 명시적으로 복사해 넣고 forward한
결과가 현재 라이브 `ThreeHeadTabM`과 **bit-for-bit 완전히 일치**(`max_abs_diff=0.000e+00`,
`allclose=True`, 3개 헤드 전부) — 구현 정합성 확인됨.

**cheap_gate**: 단일 시드(260816), expert=bull, plain CE, 진짜 115피처 파이프라인,
`r ∈ {1(=현재 라이브, B 고정 통제군), 2, 3}` × σ_init=0.03 고정. 결과:

| r | 파라미터수 | 선택된 체크포인트(val_loss 기준) dir_bacc | 진짜 정점 dir_bacc(에폭) | 선정 격차 | 붕괴진단(선택 체크포인트, 멤버쌍 상관/unanimity) |
|---:|---:|---:|---:|---:|---:|
| 1(통제군) | 103,992 | **0.5738**(에폭1) | 0.5738(에폭1) — 선택=정점 | 0.0000 | 0.9975 / 96.44% |
| 2 | 117,200 | 0.5611(에폭1) | 0.5705(에폭5) | **−0.0093** | 0.9975 / 96.49% |
| 3 | 125,800 | 0.5619(에폭1) | 0.5712(에폭4) | **−0.0093** | 0.9975 / 96.14% |

**선택된 체크포인트 기준 델타**: r=2 −0.0126, r=3 −0.0118 — 그대로 보면 이번 세션 다른 후보들과
비슷한 크기의 negative.

**하지만 진짜 정점 기준으로 다시 보면 그림이 달라진다**: r=2·r=3 둘 다 §2.5(lr=2e-4)와 똑같은
패턴을 보인다 — `combined val_loss` 조기종료가 에폭1에서 멈췄는데 진짜 정점은 에폭4~5에 있었다
(선정 격차 −0.0093, §2.5의 lr=2e-4 격차 −0.0206과 같은 방향·비슷한 자릿수). **정점 대 정점으로
비교하면 델타는 r=2 −0.0033, r=3 −0.0026** — 이번 문서에서 나온 모든 후보를 통틀어 가장 작은
축에 든다(§2.4b std=0.1의 −0.0032와 사실상 동률, §2.5 lr=2e-4의 진짜정점 델타 −0.0072보다도 작음).

**다양성 진단은 움직이지 않았다**: 멤버쌍 상관(0.9975)과 unanimity(96.1~96.5%)가 r=1→2→3에서
**소수점 4자리까지 거의 동일**하다 — §6.1이 예측한 "rank-r은 다양성 상한을 실제로 넓힌다"는
가설이 **이 측정 지점(선택된 체크포인트=에폭1)에서는 뒷받침되지 않았다.** 원인이 명확하다: 신규
r≥2 채널(A[:,:,1:], B[:,:,1:])은 거의 0 근처로 초기화됐고(§6.3 warm-start 설계), 이 저장소 전체가
반복 확인해 온 대로 체크포인트가 항상 에폭1 근방에서 선택된다 — **딱 1에폭만큼의 그래디언트
스텝으로는 신규 채널이 0 근처 초기화에서 벗어나 진짜 다양성을 만들 시간이 없었을 가능성이 높다.**
(진짜 정점(에폭4~5)에서의 다양성은 이번 실행에서 측정하지 않았다 — `_collapse_stats`를 "새
최적치 갱신" 시점에만 기록했는데, 에폭1 이후로는 val_loss가 계속 나빠져 그 시점이 다시 오지
않았다. 이 자체가 미해결 질문으로 남는다.)

**종합 판정**: 이 후보는 §2.4a(k축소)·§2.4b(초기화확대)·§3.2b(quality타겟분리)·§2.5(lr격리)에 이은
**다섯 번째 서로 다른 메커니즘**이고, 역시 현재 배포 설정을 못 이긴다 — 다만 실패의 성격이
다르다: 진짜 정점 기준 격차는 이번 조사 전체에서 가장 작은 축(노이즈 경계)이지만, 애초에
검증하려던 가설(다양성이 실제로 늘어나는가)이 이 실행에서는 **확인도 반박도 안 됐다**(측정
시점이 잘못됐을 가능성). N≥5시드로 넘어가거나, 에폭 예산을 늘려 진짜 정점에서 다양성을 재측정하는
후속 cheap_gate 둘 다 사용자 판단으로 남긴다 — 이번 세션 범위는 여기까지.

**재사용 가능 산출물**: `scripts/research_eth_odyssey4_rankr_gate_cheap_gate_20260817.py` —
`RankRGate`/`RankRThreeHeadTabM`(범용, `rank` 인자로 아무 rank나 테스트 가능), `sanity_check()`
(다른 아키텍처 변형 검증에도 재사용 가능한 패턴). 원본 리포트:
`tmp/causal_regen_20260516/eth_odyssey4_rankr_gate_cheap_gate_20260817/report.json`.

### 6.8 N≥5 시드 재확인 (2026-08-17) — CLOSED

사용자 요청("cheap gate 말고 제대로 테스트해서 재측정") 그대로, §6.7이 남긴 두 미해결 질문(선정
격차가 우연인지, 진짜 정점에서 다양성이 실제로 늘었는지)을 N=5 진짜 랜덤 시드
(`secrets.randbelow`, 고정 간격 아님 — Seed-Diversity Ensemble Promotion Gate)로 재확인했다.
`scripts/research_eth_odyssey4_rankr_gate_nseed_confirm_20260817.py` — cheap_gate와 동일 하네스에
`_collapse_stats`를 **매 에폭마다** 기록하도록 고쳐서, 선택된 체크포인트뿐 아니라 진짜 정점
에폭에서도 다양성을 읽을 수 있게 했다. 시드: `[147360281, 303419845, 783674062, 809896406,
824654821]`. expert=bull, plain CE, 진짜 115피처 파이프라인, r∈{1(통제군),2,3}, 총 15회 학습
(94분→39.5분 실측).

| 비교 | 선택된 체크포인트 델타 | 진짜 정점 델타 | 부호일치 | 진짜 정점 다양성(상관) 델타 |
|---|---:|---:|---:|---:|
| r=2 vs r=1 | −0.0065 ± 0.0106 (1/5 개선) | −0.0013 ± 0.0109 (2/5 개선) | **False** | **−0.00003 ± 0.0004** |
| r=3 vs r=1 | −0.0015 ± 0.0078 (1/5 개선) | −0.0008 ± 0.0111 (2/5 개선) | **False** | **−0.00010 ± 0.0003** |

**정확도**: 4개 비교(r=2/3 × 선택/정점) 전부 표준편차가 평균의 1.6~14배 — 이 저장소가 반복 확인해
온 `tabm_hp_low_signal_pattern`(시드간 잡음이 효과크기를 압도)과 정확히 같은 서명이고, 4개 전부
`sign_consistent=False`(시드마다 부호가 뒤집힘). 단일시드 cheap_gate가 보여준 "선택 기준은
나쁘지만 진짜 정점은 노이즈급"이라는 그림이 N=5에서도 그대로 유지됐다 — 진짜 정점 델타
(−0.0013/−0.0008)는 표준편차 절반도 안 되는, 통계적으로 0과 구분 안 되는 크기다.

**다양성 — 이번엔 확정적으로 답이 나왔다**: §6.7이 "측정 시점이 잘못됐을 수 있다"고 남겨둔
질문에 대한 답. 진짜 정점 에폭(시드·rank마다 1~9에폭으로 제각각이었음)에서 측정해도 멤버쌍 상관
델타는 **5개 시드 평균 −0.00003(r=2), −0.00010(r=3)** — 사실상 정확히 0이고, 개별 시드 델타도
±0.0002~0.0007 범위에서 무작위로 흩어져 있다(부호도 랜덤). **결론: rank를 1→2/3로 올려도 이
아키텍처·라벨·학습 레시피 조합에서는 앙상블 멤버 다양성이 늘지 않는다** — 측정 타이밍 문제가
아니라 실제로 그렇다는 게 이제 확정됐다.

**CLOSED 판정**: §2.4a(k축소)·§2.4b(초기화확대)·§3.2b(quality타겟분리)·§2.5(lr격리)에 이은 다섯
번째 메커니즘도 N≥5시드에서 현재 배포 설정을 못 이긴다(이번엔 정확도뿐 아니라 검증하려던 핵심
가설=다양성 증가까지 명시적으로 기각됨). **이 저장소의 "capacity/init을 만지는" 계열 다양성
완화책(k축소·초기화확대·rank-r) 전부가 소진됐다** — 남은, 아직 테스트 안 된 카테고리는 손실함수
자체에 멤버간 다양성을 명시적으로 보상하는 항을 넣는 것(negative correlation learning류)뿐이고,
이는 지금까지의 5개 후보와 질적으로 다른 축이라 재시도 가치가 있다면 그쪽이다 — 다만 §3.1이
지적한 라벨 자체의 비인과적 확정(repainting) 문제가 다양성 붕괴의 더 근본적인 원인일 가능성도
여전히 남아 있다(다양성이 "capacity를 늘려도" "손실 타이밍을 바꿔도" 전혀 안 늘어난다는 사실
자체가, 문제가 게이트 구조가 아니라 8개 멤버 전부에게 같은 얕은 힌트만 주는 라벨/피처 쪽에 있을
가능성을 오히려 강화한다).

원본 리포트: `tmp/causal_regen_20260516/eth_odyssey4_rankr_gate_nseed_confirm_20260817/report.json`.

## 7. 신규 발견 D — 손실함수 다양성항 (Negative Correlation Learning) — CLOSED, N≥5시드 확정 (2026-08-17)

§6.8이 남긴 "capacity/init 계열 완화책은 전부 소진, 남은 미시도 축은 손실함수에 명시적 다양성항을
넣는 것"이라는 결론을 그대로 이어받아, 사용자 요청("손실함수에 다양성항 넣는 거 구현해서
테스트해줘")으로 착수한 여섯 번째(그리고 마지막) 메커니즘.

### 7.1 문헌 (OpenAlex 직접 조회, 접근일 2026-08-17)

- **Krogh & Vedelsby (1994)**, "Neural network ensembles, cross validation, and active learning" —
  고전적 ambiguity 분해: 앙상블 오차 = 평균 멤버 오차 − ambiguity, ambiguity = mean_i(f_i − f_mean)².
  ambiguity를 키우는 것이 다양성 목표다.
- **Liu & Yao (1999)**, "Ensemble learning via negative correlation" (Neural Networks,
  doi:10.1016/s0893-6080(99)00073-8) — Negative Correlation Learning(NCL): 멤버 i의 손실에
  `p_i = (f_i−f_mean)·Σ_{j≠i}(f_j−f_mean)` 페널티를 더한다. 평균의 정의상 `Σ_j(f_j−f_mean)=0`이므로
  대수적으로 `p_i = −(f_i−f_mean)²` — 즉 NCL의 페널티는 정확히 Krogh-Vedelsby ambiguity의 음수다.
- **Wang, Chen & Yao (2010)**, "Negative correlation learning for classification ensembles" (IJCNN,
  doi:10.1109/ijcnn.2010.5596702) — 원래 NCL/ambiguity 분해는 회귀(제곱오차) 전용이었고 분류에는
  별도 적응이 필요함을 확인. 이 스크립트는 후속 딥러닝 NCL 연구들(Zhang+ 2019 CVPR, Shi+ 2018
  CVPR)이 쓰는 실용적 적응 방식 — ambiguity를 raw 출력이 아니라 **소프트맥스 확률**에 적용 — 을
  그대로 썼다.

### 7.2 구현

`scripts/research_eth_odyssey4_ncl_diversity_loss_cheap_gate_20260817.py` — 아키텍처는 **완전히
그대로**(`canon.ThreeHeadTabM` 무수정), direction_head의 멤버별 소프트맥스 확률에서 ambiguity를
계산해 손실에서 뺀다:

```
ambiguity = mean_batch[ mean_k( ||p_k − p_mean||² ) ]
total_loss = loss_dir + qw·loss_qual + ew·loss_exit − λ·ambiguity
```

λ=0이면 항이 정확히 0이 돼(0×무엇이든=0) 캐노니컬 손실과 완전히 같아진다 — 이번엔 아키텍처
자체를 안 건드려서 rank-r처럼 별도 bit-for-bit sanity check가 필요 없었다(스윕의 λ=0이 곧 통제군).
체크포인트 선정은 **항상 태스크 손실만**(ambiguity 제외) 기준 — λ가 달라져도 비교 가능하게.

### 7.3 cheap_gate — λ 스윕 (단일시드 260816, 9개 지점)

초기화 시점 ambiguity≈0.01, CE loss 스케일 1~3이라 λ가 몇 자릿수는 돼야 경쟁한다고 보고 로그
스케일로 스윕, 결과가 절벽이 아니라 매끄러운 dose-response라 λ=1~10 구간을 세분화했다:

| λ | 선택 dir_bacc | 선택 상관 | 정점 dir_bacc | 정점 상관 |
|---:|---:|---:|---:|---:|
| 0(통제) | 0.5710 | 0.9975 | 0.5710 | 0.9975 |
| 1 | 0.5726 | 0.9970 | 0.5726 | 0.9970 |
| 2 | 0.5726 | 0.9961 | 0.5726 | 0.9961 |
| 3 | 0.5691 | 0.9936 | 0.5691 | 0.9936 |
| 5 | 0.5504 | 0.9436 | 0.5504 | 0.9436 |
| 7 | 0.4945 | 0.5638 | 0.5262 | 0.4673 |
| 10 | 0.4638 | 0.4487 | 0.4847 | 0.4956 |
| 100 | 0.3906 | 0.5154 | 0.4675 | 0.7364 |
| 1000 | 0.4300 | 0.5523 | 0.4692 | 0.7428 |

**절벽이 아니라 매끄러운 트레이드오프**: λ≤3은 둘 다 거의 안 움직이고, λ=5부터 함께 움직이기
시작해 λ=7에서 상관이 0.997→0.56으로, 정확도가 0.571→0.49~0.53으로 같이 무너진다. λ≥10은 둘 다
더 나빠져서(과도한 힘이 학습 자체를 방해) λ=100/1000에서 상관이 부분적으로 재수렴하는데도
정확도는 회복 안 된다 — "공짜 구간"이 어디에도 없다.

### 7.4 N≥5 시드 확정 (경계 두 지점: λ=2 "무변화 구간", λ=7 "전환 구간")

시드 `[121889256, 241992079, 411267114, 631193301, 951559053]`(진짜 랜덤). λ=0을 각 시드 내
페어드 통제군으로 사용(같은 시드=같은 초기화, λ만 다름).

| 비교 | 선택 dir_bacc 델타 | 정점 dir_bacc 델타 | 정점 상관 델타 | 부호일치(정점 dir_bacc/상관) |
|---|---:|---:|---:|---:|
| λ=2 vs 0 | −0.0054 ± 0.0059 (1/5 개선) | −0.0086 ± 0.0094 (2/5 개선) | −0.169 ± 0.223 | False / **True(5/5 악화)** |
| λ=7 vs 0 | −0.0534 ± 0.0254 (0/5 개선) | −0.0497 ± 0.0076 (0/5 개선) | −0.513 ± 0.110 | **True(5/5 악화) / True(5/5 악화)** |

**λ=2 — 단일시드의 "무변화" 결론이 뒤집혔다**: 시드에 따라 완전히 다른 두 가지 일이 일어난다.
3개 시드는 상관이 거의 안 움직이고(−0.002~−0.011), 2개 시드는 λ=7급으로 크게 무너진다
(−0.42~−0.41). **λ=2가 "다양성 강제를 켜는" 문턱을 넘는지는 시드/초기화 궤적에 따라 달라진다** —
단일시드에서 관찰한 "λ≤3은 안전하다"는 결론은 일반화되지 않는다. 다만 방향 자체(0/5 개선, 상관은
5/5 하락)는 일관된다 — 안전한 λ가 아니라, 효과가 불규칙하게 나타나는 λ다.

**λ=7 — 이 조사 전체에서 가장 확실한(노이즈 아닌) 결과**: 정확도 델타 표준편차(0.0076)가 평균
(−0.0497)의 15%밖에 안 된다 — `tabm_hp_low_signal_pattern`이 반복 지적해 온 "표준편차가 평균을
압도하는" 노이즈 패턴과 정반대다. 5/5 시드 전부 정확도 악화, 5/5 시드 전부 다양성 대폭 증가
(상관 평균 0.998→0.485) — **다양성을 실제로 늘리는 데는 성공했지만, 그 대가로 정확도가
확실하고 재현 가능하게(노이즈가 아니라 진짜로) 나빠진다.**

### 7.5 최종 판정 — CLOSED, 그리고 6개 메커니즘 전체 종결

이 손실함수 다양성항은 **rank-r(§6)과 정반대의 실패 양상**을 보인다 — rank-r은 다양성 자체를
못 늘렸고(진짜 실패), NCL은 다양성을 확실히 늘렸지만 그 대가(정확도)가 이 저장소 기준으로
너무 크다(성공했지만 무의미한 트레이드오프). 어느 λ에서도 "다양성 증가 + 정확도 유지·개선"
조합은 나오지 않았다 — 이 축도 CLOSED.

**§2.4a(k축소)·§2.4b(초기화확대)·§3.2b(quality타겟분리)·§2.5(lr격리)·§6(rank-r)·§7(NCL 손실항)
— 이 아키텍처의 BatchEnsemble 붕괴에 대해 시도 가능한 6가지 서로 다른 메커니즘이 전부 종결됐다.**
capacity를 줄이거나(k축소) 늘리거나(rank-r) 초기화를 넓히거나(wide-init) 학습을 늦추거나
(lr격리) 보조타겟을 바꾸거나(quality분리) 손실에 직접 다양성을 강제하거나(NCL) — 어떤 손잡이를
당겨도 "다양성 증가"와 "정확도 유지"가 동시에 오지 않는다. 특히 rank-r(다양성 자체가 안 늘어남)과
NCL(다양성은 늘지만 정확도가 확실히 깨짐)이 서로 다른 방향에서 같은 결론에 도달했다는 점이
중요하다 — **이 붕괴는 게이트/손실 설계의 결함이 아니라, 8개 멤버 전부가 zigzag_action 라벨의
같은 얕은 힌트(§3.1의 비인과적 확정/repainting 문제)로 수렴하도록 만드는 라벨/과제 구조 자체의
결과일 가능성이 이제 가장 유력하다.** 이 축을 다시 열려면 게이트나 손실이 아니라 라벨 자체를
바꿔야 한다.

원본 리포트: cheap_gate
`tmp/causal_regen_20260516/eth_odyssey4_ncl_diversity_loss_cheap_gate_20260817/report.json`
(+ `report_gap_fill.json`), N=5시드
`tmp/causal_regen_20260516/eth_odyssey4_ncl_diversity_loss_nseed_confirm_20260817/report.json`.
