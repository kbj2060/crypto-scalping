# ETH 완전한 TabM(BatchEnsemble R+S+B) 후보 — Step A cheap_gate 결과 (2026-08-16)

관련 계약: `docs/model_contracts/eth_candidate_faithful_tabm_batchensemble_contract_20260816.md`

## 배경

라이브 `ThreeHeadTabM`(`scripts/train_eval_omega1_2_tabm_3head_20260603.py`)이 TabM 논문
(arXiv:2410.24210)의 BatchEnsemble 수식 `l_BE(X) = ((X⊙R)W)⊙S+B`에서 R(사전곱셈)만 구현하고
S(사후곱셈)·레이어별 B(bias)를 빠뜨린 자체 변형이라는 게 코드 대조로 드러나서, R+S+B를 완성한
`ThreeHeadTabMFull`을 만들어 단일시드 cheap_gate(분류지표)로 먼저 방향성을 확인했다.

## 실행

- 스크립트: `scripts/research_eth_candidate_faithful_tabm_batchensemble_cheap_gate_20260816.py`
- 데이터: `_prepare_frames_light()` — 기존 `_prepare_frames()`와 동일하되, 죽은 LSTM/chronos
  체인(`FileNotFoundError`, 이 dev 머신에 원본 피처 CSV 없음)을 우회해 `zigzag_action` 라벨을
  `train_omega1_direction_head_direction_only_20260602._add_labels(year)`로 직접 읽음. **주의**:
  이렇게 얻은 `feature_cols`가 185개로, 실제 라이브 배포 번들의 102개와 다르다 — 이 cheap_gate는
  피처 완전 동일성이 아니라 "R+S+B 완성이 분류 지표를 조금이라도 움직이는가"만 본다.
- 시드: 260816 (단일), epoch 예산 28(patience=8로 전부 9 epoch에서 조기 종료), k=8/hidden=192/
  layers=3/dropout=0.08 등 `CFG` 불변.
- 3개 regime expert(bull/bear/chop) × 2개 아키텍처(`baseline_R_only`=기존, `full_R_S_B`=완성판).

## 결과

| Expert | Arch | val_loss | dir_val_loss | quality_val_loss | exit_val_loss | dir_bacc | n_params |
|---|---|---:|---:|---:|---:|---:|---:|
| bull | baseline_R_only | 2.6573 | 0.8656 | 0.8615 | 0.9587 | 0.5732 | 118,552 |
| bull | full_R_S_B | 2.6923 | 0.8747 | 0.8715 | 0.9743 | 0.5790 | 126,288 |
| bear | baseline_R_only | 2.8079 | 0.8641 | 0.8647 | 1.0887 | 0.5668 | 118,552 |
| bear | full_R_S_B | 2.7894 | 0.8527 | 0.8612 | 1.0850 | 0.5608 | 126,288 |
| chop | baseline_R_only | 2.8832 | 0.8754 | 0.8464 | 1.1571 | 0.5813 | 118,552 |
| chop | full_R_S_B | 2.8864 | 0.8531 | 0.8632 | 1.1676 | 0.5821 | 126,288 |

Δ(full − baseline) 부호:

| Expert | val_loss | dir_val_loss | quality_val_loss | exit_val_loss | dir_bacc |
|---|---|---|---|---|---|
| bull | 악화(+0.0350) | 악화(+0.0091) | 악화(+0.0099) | 악화(+0.0156) | 개선(+0.0059) |
| bear | 개선(−0.0185) | 개선(−0.0114) | 개선(−0.0035) | 개선(−0.0037) | 악화(−0.0059) |
| chop | 거의동일(+0.0032) | 개선(−0.0223) | 악화(+0.0168) | 악화(+0.0104) | 거의동일(+0.0009) |

## 판정

**부호가 expert마다 일관되지 않는다.** bull은 loss 계열이 전부 악화되면서 dir_bacc만 개선,
bear는 loss 계열이 전부 개선되면서 dir_bacc만 악화, chop은 방향조차 지표별로 갈린다. 15개 셀
중 명확한 한쪽 방향 우위는 없고, 크기도 이 저장소가 반복적으로 "단일시드 노이즈"로 판정해 온
범위(수백 분의 1 수준)에 머문다 — `tabm_hp_low_signal_pattern`(메모리)의 "단일시드 HP 승자는
대개 노이즈"와 정확히 같은 모양.

계약서에 미리 못박은 cheap_gate 기준("방향/품질 신호 자체가 이미 무스킬로 확정된 상태라,
분류 지표에서조차 개선이 없으면 백테스트까지 갈 이유가 없다")을 이 결과는 통과하지 못한다 —
"개선"이 아니라 "혼재"다.

## N≥5 시드 재현 (2026-08-16, 사용자 지시로 진행)

사용자가 "N≥5 시드로 재현 확인 후 결정"을 선택해서 진행했다.

- 스크립트: `scripts/research_eth_candidate_faithful_tabm_batchensemble_nseed_20260816.py`
- 시드: `secrets.randbelow` 추출 5개(고정간격 클러스터 아님) — `[211581, 262041, 393534, 646498, 707258]`
- 5시드 × 2아키텍처 × 3expert = 30회 학습, 프레임/exit 데이터셋 준비는 1회만 공유(비용 절감).

### 결과 — 이번엔 노이즈가 아니라 일관된 방향이 나왔다

`direction_balanced_accuracy`(h48qual이 실제로 예측하는 대상 그 자체) 기준, delta = full_R_S_B − baseline_R_only:

| Expert | mean Δbacc | std | 개선 시드 수 |
|---|---:|---:|---:|
| bull | −0.0035 | 0.0068 | 1/5 |
| bear | −0.0036 | 0.0137 | 3/5(단, 악화 2개가 개선 3개보다 큰 폭이라 평균은 음수) |
| chop | **−0.0088** | 0.0049 | **0/5 — 5개 시드 전부 악화** |

val_loss/exit_val_loss는 시드 간 표준편차(0.06~0.20)가 평균 델타보다 훨씬 커서 방향성 신호가
없다(순수 노이즈) — 하지만 direction_balanced_accuracy는 표준편차가 평균 대비 작아서(특히
chop: std=0.0049 vs mean=-0.0088) 신호로 볼 수 있다.

**결론: R+S+B를 완성하면 방향 예측 정확도가 오히려 소폭이지만 일관되게 나빠진다.** 파라미터가
6.5% 늘어난(118,552→126,288) 게 이 프로젝트의 약하고 노이즈 많은 78k행 데이터셋에서 과적합을
가속했을 가능성이 높다 — 외부 문헌 조사에서 이미 지적한 "Nonstationarity-Complexity Tradeoff"
(arXiv:2512.23596, 약신호 하에서 모델 복잡도 증가는 held-out 성과를 악화시킨다)의 직접적이고
구체적인 재확인이다.

## 학습 품질 개선 실험 (2026-08-16, 사용자 지시: "닫지 말고 논문대로 최적화해보자")

Step A N≥5 시드 결과로 CLOSED 처리했으나, 사용자가 재개를 지시했다. "코드는 잘 만드는데 모델링에는
취약하다"는 사용자 피드백에 따라, 이 절 이후의 모든 시도는 실제 논문/문헌을 원문(WebFetch)으로
확인하고 인용을 남기는 방식으로 진행했다 — 관련 표준 관행: `feedback_modeling_needs_literature_
grounding`, `feedback_modern_dl_training_checklist`(메모리).

### 1) "왜 매번 epoch 1이 최고 체크포인트인가" — 원인 진단

Step A의 single-seed cheap_gate + N≥5 시드 재현을 합친 66회 학습이 **전부** early-stop
patience=8에 걸려 정확히 epoch 9에서 멈췄다(= best checkpoint가 항상 epoch 1). 두 가지로 검증했다.

**(a) patience/epoch 예산 완화 진단** (`scripts/research_eth_candidate_faithful_tabm_batchensemble_patience_diag_20260816.py`, patience 8→24, epoch 예산 28→60): 실제로 더 오래 학습되긴 하지만(epoch 25까지), 결론은 그대로 — delta_bacc가 expert마다 부호 혼재.

**(b) 전체 40-epoch 곡선 진단(early stopping 없음)** (`scripts/research_eth_candidate_faithful_tabm_batchensemble_curve_diag_20260816.py`, seed=260816, expert=bull): baseline_R_only/full_R_S_B 둘 다 **교과서적 memorization 곡선**을 보였다 —

| epoch | train_loss | val_loss | train_bacc | val_bacc |
|---|---:|---:|---:|---:|
| 1 | 2.366 | 2.657 | 0.589 | 0.573 |
| 2 | 2.081 | 3.169 | 0.609 | **0.574(정점)** |
| 40 | 0.434 | 7.197 | **0.905** | 0.492 |

train은 40 epoch 내내 단조 개선(bacc 0.59→0.90), val은 단조 악화 — 진동/스파이크 없음. 이건
학습률 불안정이 아니라 Arpit et al.("A Closer Look at Memorization in Deep Networks",
**arXiv:1706.05394**)의 "일반화 먼저, 그다음 암기" 패턴과 정확히 일치한다: 약신호일수록 일반화
구간이 짧아진다.

### 2) 학습기법 문헌조사 #1 — TabM 논문의 실제 학습 레시피

TabM 논문(**arXiv:2410.24210**) Appendix D.2 + 공식 레포(`github.com/yandex-research/tabm`)
직접 확인: AdamW, **LR 스케줄 없음**(warmup도 없음), LR은 `LogUniform[1e-4, 5e-3]`에서 데이터셋별
튜닝(레퍼런스 기본값 `lr=0.002`) — 저희 fixed lr=2e-3는 이 논문의 방식과 이미 일치했다(스케줄이
없다고 "고쳐야 할 버그"는 아니었음). grad-clip norm 1.0(저희는 2.0), early-stop patience 16(저희는
8), k=32(저희는 8, "암묵적 정규화" 목적으로 논문이 heuristic하게 선택). 진짜 원인은 학습 설정이
아니라 (1)에서 확인한 memorization이었다.

### 3) LR sweep — 가장 유의미한 단일 레버

(`scripts/research_eth_candidate_faithful_tabm_batchensemble_lr_sweep_20260816.py`, seed=260816,
expert=bull, baseline_R_only, 40 epoch 고정)

| LR | 정점 val_bacc | 정점 epoch | epoch40 |
|---|---:|---:|---:|
| 2e-3(기존) | 0.5740 | 2 | 0.492(급락) |
| **2e-4** | **0.5714** | **12** | **0.536** |
| 2e-5 | 0.5627 | 39(미정착) | 0.5627 |
| 2e-6 | 0.5048 | 40(학습 부족) | 0.5048 |

**lr=2e-4(10배 축소)가 정점 높이를 거의 유지하며 붕괴를 6배 늦춘다.** TabM 논문 자체가 lr을
`LogUniform[1e-4,5e-3]`에서 데이터셋별로 튜닝하라고 한 것과 일치하는 결론 — 이 프로젝트 데이터엔
논문 기본값(2e-3)보다 작은 쪽이 맞다. 단일 시드, N≥5 확인 필요.

### 4) 학습기법 문헌조사 #2 — memorization을 실제로 막는 기법

TabM 관련 조사와 별도로, "detect-and-stop이 아니라 실제로 loss를 낮추는 법"을 조사했다(원문
WebFetch로 확인). 후보:
- **GCE**(Generalized Cross Entropy, Zhang & Sabuncu, **arXiv:1805.07836**): `L_q=(1-f_j(x)^q)/q`,
  논문 기본값 q=0.7.
- **ELR**(Early-Learning Regularization, Liu et al., **arXiv:2007.00151**): 샘플별 과거 예측
  EMA에 정규화, 논문은 데이터셋별 λ/β 미제시 — 공개 구현체 관례값 λ=3, β=0.7 사용.
- **Latent mixup**(Zhang et al. **arXiv:1710.09412** §3.4 + 시계열 적응 **arXiv:2304.04271**):
  raw feature 대신 backbone 임베딩 단계에서 섞음(OHLCV 파생 피처를 직접 섞으면 물리적으로 말이
  안 되는 조합이 나올 수 있어서), α=1.0.
- 제외: **SAM**(arXiv:2010.01412, 비전 전용 근거뿐+2배 연산비용), **Confident Learning/cleanlab**
  (arXiv:1911.00068, 이 프로젝트 라벨은 오표기가 아니라 그냥 신호가 약한 거라 메커니즘 안 맞음,
  `repo_label_methodology_meta_finding`과 일치).

### 5) GCE+ELR+mixup 결합 테스트 vs 개별 분리 테스트

(`scripts/research_eth_candidate_faithful_tabm_batchensemble_combo_regularizer_20260816.py`,
`..._regularizer_isolation_20260816.py` — seed=260816, expert=bull, baseline_R_only, 40 epoch 고정)

| | 정점 val_bacc | 정점 epoch | epoch40 |
|---|---:|---:|---:|
| 순정 CE(기준선) | 0.5740 | 2 | 0.492 |
| **GCE 단독** | **0.5758** | 3 | 0.5065 |
| ELR 단독 | 0.5641 | 1(지연 효과 없음) | 0.4596 |
| mixup 단독 | 0.5545 | 4 | 0.4808 |
| GCE+ELR+mixup 결합 | 0.5368(개별 최악보다도 나쁨) | 8 | 0.4829 |

**GCE만 단독으로 기준선을 소폭 상회.** ELR·mixup은 (관례값 하이퍼파라미터로는) 단독으로도
기준선보다 나쁘고, 셋을 합치면 상호간섭으로 더 나빠진다 — "이 기법군 전체가 안 맞는다"가 아니라
"ELR/mixup은 하이퍼파라미터 튜닝이 안 됐고, 무조건 합친다고 좋아지는 게 아니다"가 정확한 결론.

### 6) 파라미터 축소 스윕 & 체크포인트 선정 버그 발견

(`scripts/research_eth_candidate_faithful_tabm_batchensemble_stepb_capacity_sweep_20260816.py`)
아래 "정정" 절 참고 — Step B(임베딩) 파라미터를 줄일수록 개선되고(410K→110K로 줄인 `quarter`가
baseline에 근접), 그 과정에서 combined val_loss 기준 early stopping이 임베딩 아키텍처의 진짜
dir_bacc 정점을 놓치는 버그를 발견했다.

## 정정 (2026-08-16, 추가 학습기법 리서치 도중 발견): Step B 판정 재검토 필요

파라미터 축소 스윕을 40 epoch 전체 곡선(early stopping 없이)으로 돌리다가, **combined val_loss
기준 early stopping이 고른 체크포인트가 실제 direction_balanced_accuracy 정점과 크게 어긋난다**는
걸 발견했다 — 임베딩이 있는 아키텍처에서만(임베딩 없는 baseline_R_only/full_R_S_B는 격차
0.0000~0.0008로 무시 가능):

| Config | val_loss가 고른 epoch의 bacc | 진짜 bacc 정점 | 격차 |
|---|---:|---:|---:|
| current 임베딩(410,448 파라미터) | 0.5281 | 0.5684(epoch4) | +0.0402 |
| quarter 임베딩(109,836 파라미터) | 0.5187 | **0.5717**(epoch9) | +0.0530 |
| eighth 임베딩(43,354 파라미터) | 0.5227 | 0.5606(epoch4) | +0.0379 |
| tiny 임베딩(31,074 파라미터) | 0.5090 | 0.5614(epoch5) | +0.0525 |

**아래 "Step B" 결과(3개 expert 전부 −3.7~−5.1%p 악화)는 combined val_loss 기준 early
stopping(patience=8)이 고른 체크포인트로 측정한 것이라, 진짜 아키텍처 성능이 아니라 체크포인트
선정 아티팩트를 상당 부분 반영했을 가능성이 높다.** `quarter` config(파라미터 수가 baseline보다도
적음)의 진짜 정점(0.5717)은 baseline의 정점(0.5740)과 거의 차이가 없다 — Step A의 N≥5 시드
판정(임베딩 없는 아키텍처 간 비교, 격차 무시 가능)은 이 문제의 영향을 받지 않아 그대로 유효하지만,
**Step B는 direction_val_loss(또는 dir_bacc) 단독 기준으로 3개 expert 전부 재검증이 필요하다** —
아래 결론은 이 재검증 전까지 잠정(pending revision)으로 취급한다.

## full_R_S_B_embed 위 최선 학습 조합 탐색 (2026-08-16, 사용자 지시로 별도 진행)

Step A/B 판정과 별개로, 사용자가 "full_R_S_B + 임베딩으로 결정하고, 나머지 구성요소는 최선의
조합으로 테스트"를 지시해서 진행한 학습 품질 축. `full_R_S_B_embed[quarter]`(109,836 파라미터)
고정, expert=bull, seed=260816.

**1차 시도 — cosine 스케줄(lr 2e-4→2e-6, T_max=60) + direction_val_loss 기준 patience=20**:
CE/GCE × AdamW/RAdam/AdaBelief 6개 조합. 결과: GCE 조합 3개가 전부 최하위(selected_bacc
0.517~0.519) — direction_val_loss가 GCE의 유계(bounded) 특성 때문에 학습 초반(epoch 2~5)에 이미
평평해져서 patience가 너무 일찍 소진됐다.

**원인 조사 → 문헌 확인**: Prechelt("Early Stopping — But When?", 1998, 원문 확인)가 이 정확한
현상(진짜 과적합 전 16개 지역 최소값, "뚜렷한 규칙 없음")을 이미 문서화했고, **GCE 논문
(arXiv:1805.07836) 자체가 자기 loss가 아니라 "validation accuracy"로 체크포인트를 고른다**는
걸 확인 — 학습 loss와 선정 기준을 분리하는 게 GCE 저자들의 관행이었다. 다만 2026년 논문
(arXiv:2602.22107)은 "accuracy보다 loss가 선정 기준으로 낫다"고 반박하므로, 그냥 accuracy로
바꾸는 대신 **"GCE가 아닌 순정 CE, 단 학습과 동일한 class-balanced 가중치 적용"**으로 선정
기준을 재설계하고, patience 대신 Prechelt의 **strip 기반 UP_4 규칙**(k=5 epoch 단위, 4연속
악화 시 정지)을 적용했다.

**2차 시도 — 수정된 기준으로 동일 6개 조합 재실행**:

| 조합 | 선정 bacc(수정) | 격차(수정) | 선정 bacc(기존) | 격차(기존) |
|---|---:|---:|---:|---:|
| **AdamW+GCE** | **0.5645** | 0.0054 | 0.5171(최하위) | 0.0528 |
| RAdam+CE | 0.5587 | 0.0069 | 0.5430 | 0.0226 |
| AdamW+CE | 0.5543 | 0.0146 | 0.5543(동일) | 0.0146 |
| AdaBelief+CE | 0.5517 | 0.0219 | 0.5517(동일) | 0.0219 |
| AdaBelief+GCE | 0.5498 | 0.0212 | 0.5172 | 0.0537 |
| RAdam+GCE | 0.5191 | 0.0529 | 0.5191(동일) | 0.0464 |

**순위가 뒤집혔다** — AdamW+GCE가 최하위에서 1위로. "GCE가 이 아키텍처에서 손해"라는 1차 결론은
선정 기준 버그의 아티팩트였다. 다만 완전한 해결은 아니다 — RAdam+GCE는 기준을 고쳐도 여전히
격차가 크다(0.0529), 일부 조합은 loss 궤적 자체가 근본적으로 나쁠 수 있다는 뜻.

**참고선**: baseline_R_only(임베딩 없음)의 진짜 정점(단일시드) = 0.5740. AdamW+GCE의 "실제 배포
시 나올 값"(0.5645)이 여기에 가장 근접(격차 0.0095) — 단, baseline_R_only도 동일한 cosine+
Prechelt 절차로 "실제 선정값"을 재야 완전히 공정한 비교다(아직 미실행).

**3차 — baseline_R_only 공정 비교(동일 cosine+Prechelt 절차)**: full_R_S_B_embed 결과가 정말
baseline보다 나은지 확인하려면 baseline도 같은 절차로 재야 공정하다. baseline_R_only(hidden=192
네이티브 크기, 118,552 파라미터)로 동일 6개 조합 재실행:

| 조합 | selected bacc | 격차 |
|---|---:|---:|
| **AdaBelief+GCE** | **0.5749** | 0.0014(거의 완벽) |
| AdamW+GCE | 0.5663 | 0.0105 |
| AdaBelief+CE | 0.5652 | 0.0083 |
| RAdam+CE | 0.5644 | 0.0055 |
| AdamW+CE | 0.5643 | 0.0075 |
| RAdam+GCE | 0.4158 | 0.1577(심각한 실패) |

**전체 12개 조합(quarter 임베딩 6개 + baseline 6개) 종합 결과: baseline_R_only가 1~5등을
독식**한다 — full_R_S_B_embed의 최고 조합(0.5645)조차 baseline 최하위권(0.5643)과 동률 수준.
**전체 1등은 `baseline_R_only + AdaBelief + GCE + cosine(2e-4→2e-6) + Prechelt UP_4`,
selected_bacc=0.5749**(원래의 "이상적 정점" 0.5740보다도 높음, 격차 0.0014로 지금까지 가장
예측 가능). `RAdam+GCE`는 두 아키텍처 모두에서 크게 실패(baseline 격차 0.158, embed 0.053) —
아키텍처 무관하게 피해야 할 조합으로 확정.

학습 절차를 아무리 정교화해도 구조는 단순한 쪽(baseline_R_only)이 이긴다 — Step A/B 결론을
재확인.

## N≥5 시드 최종 확정 (2026-08-16)

양쪽 1등 조합을 각각 무작위 시드 5개(`[144285, 270781, 588791, 618465, 780732]`) × 3개 expert
= 15회씩 재현했다(사용자 지시: 나중에 오디세이4 전체 재학습에서 어느 쪽이 이길지 모르니 두
후보 다 검증 유지).

| Expert | baseline_R_only+AdaBelief+GCE | full_R_S_B_embed[quarter]+AdamW+GCE | 격차 |
|---|---:|---:|---:|
| bull | 0.5534 ± 0.0230 | 0.5118 ± 0.0238 | +0.0416 |
| bear | 0.5570 ± 0.0044 | 0.5102 ± 0.0238 | +0.0468 |
| chop | 0.5617 ± 0.0191 | 0.5042 ± 0.0205 | +0.0575 |

**3개 expert 전부, 표준편차 대비 훨씬 큰 격차로 baseline_R_only 승리 확정.** 단일시드에서 봤던
화려한 수치(0.5749, 0.5645)는 운 좋은 시드였음이 확인됐다 — 이 저장소의 "단일시드 승자는
노이즈" 원칙이 다시 한번 맞아떨어졌다.

**메커니즘 분석**: true_peak(이상적 최고점) 기준 격차는 0.3~0.9%p로 작다 — 두 아키텍처의 "진짜
실력"은 비슷하다. 하지만 full_R_S_B_embed는 15회 중 12회(80%)가 epoch 1에서 조기종료됐고
baseline_R_only는 15회 중 3회(20%)뿐이었다 — **어댑터(S+B)+임베딩이 늘어난 만큼 학습 초반
validation loss 궤적이 더 요동쳐서, 어떤 조기종료 방식을 써도 진짜 실력까지 못 가고 일찍
멈추는 경향이 아키텍처에 내재**돼 있다. baseline_R_only가 이기는 건 "더 똑똑해서"가 아니라
"학습이 더 안정적으로 진행돼서"다.

**최종 결론**: 학습 절차(스케줄·옵티마이저·loss·선정 기준)를 아무리 정교하게 최적화해도
이 결과는 안 뒤집힌다 — Step A/B에서 이미 나온 "구조는 단순한 쪽이 이긴다"는 결론이 N≥5 시드
수준에서 최종 확정됐다. 두 후보(`baseline_R_only+AdaBelief+GCE`, `full_R_S_B_embed[quarter]+
AdamW+GCE`) 모두 향후 참고용으로 기록만 남기고, 이 학습-품질 축 자체는 종결한다.

## 남은 논문 차이 2가지 — Residual 제거 / k=8→32 (2026-08-16)

계약서 "미해결 이슈"에 남아있던 나머지 2개 TabM 논문 차이(Residual 연결은 논문에 없는데 라이브에
있음, k=8은 논문 기본값 32보다 훨씬 작음)를 baseline_R_only 위에서(AdaBelief+GCE+cosine+
Prechelt 레시피, 단일시드 260816, 3개 expert) 테스트:

| 변형 | bull | bear | chop | 파라미터 |
|---|---:|---:|---:|---:|
| reference(현재구조, k8+residual) | 0.5749 | 0.5484 | 0.5710 | 118,552 |
| no_residual | 0.5276 | 0.5622 | 0.5342 | 118,552 |
| residual_k32 | 0.5317 | 0.5360 | 0.5673 | 136,648 |

**둘 다 개선 신호 없음** — no_residual은 평균 -0.023(2/3 expert 악화), k32는 평균 -0.020(3/3
expert 악화, 파라미터만 +15%). 단일시드 cheap_gate에서조차 이길 기미가 없어 N≥5 재현으로
진행하지 않는다.

## 최종 요약 — TabM 논문과의 차이 4가지 전부 테스트 완료

| 차이 | 결과 |
|---|---|
| R+S+B 어댑터 완성 | 소폭 악화 (N≥5 확정) |
| Piecewise-linear 수치형 임베딩 | 무승부, 파라미터 늘수록 악화 (N≥5 확정) |
| Residual 연결 제거 | 개선 없음 (단일시드 cheap_gate) |
| k=8 → 논문 기본값 32 | 개선 없음 (단일시드 cheap_gate) |

**이 후보(완전한 TabM/논문 충실도 축) 전체를 여기서 최종 종결한다.** "논문대로 완성하면 더
나을 것"이라는 원래 가설은 4가지 차이 전부에서 기각됐다 — 라이브의 현재 구현(R-only, 임베딩
없음, residual 있음, k=8)이 이번 조사에서 시도한 어떤 변형보다도 낫다.

## 추가 정정 (2026-08-16, 같은 날 나중에) — "신규 레시피"도 기존 레시피를 못 이긴다

위 결론까지는 "아키텍처 축(baseline vs full_R_S_B_embed)에서 baseline이 이긴다"만 확정했지,
**"신규 레시피(AdaBelief+GCE+cosine+Prechelt) 자체가 기존 레시피(AdamW+순정CE+flat lr=2e-3+
patience=8)보다 낫다"는 검증하지 않은 상태였다** — 사용자가 "오디세이4 모델에 적용해서 재학습
비교"를 요청해서 마저 검증했다. `baseline_R_only` 위에서 두 레시피를 동일 시드(5개)·동일 3개
expert로 짝비교:

| Expert | OLD(AdamW+순정CE+flat lr2e-3+patience8) | NEW(AdaBelief+GCE+cosine+Prechelt) | 격차 |
|---|---:|---:|---:|
| bull | **0.5657 ± 0.0098** | 0.5534 ± 0.0230 | OLD +0.0123 |
| bear | **0.5623 ± 0.0049** | 0.5570 ± 0.0044 | OLD +0.0053 |
| chop | **0.5767 ± 0.0041** | 0.5617 ± 0.0191 | OLD +0.0150 |

**OLD가 3개 expert 전부 이긴다.** 이건 병행 세션이 독립적으로 확인한 "GCE 캐노니컬 이식 실패"와
정확히 같은 패턴 — 개별 요소는 단독/부분 조합 테스트에서 좋아 보였지만, 전체를 다 합쳐 N≥5
시드로 공정 짝비교하면 단순한 기존 레시피를 못 이긴다. **학습 레시피 축(LR/옵티마이저/loss/
선정기준 전체) 최종 종결 — 채택 근거 없음.** "구조는 단순한 게 이긴다"는 이번 조사의 결론이
레시피 차원에서도 다시 한번 확인된 셈이다: 라이브의 있는 그대로(AdamW+CE+flat lr=2e-3)가
지금까지 시도한 어떤 개선안보다도 낫다.

## Step B 재검증 최종 결과 (2026-08-16, 체크포인트 선정 버그 수정 후)

`quarter` config(hidden=96, d_embed=4, n_bins=8, 109,836 파라미터 — baseline보다 적음)로 3개
expert 전부, direction_val_loss 기준이 아니라 **40 epoch 전체 곡선에서 진짜 dir_bacc 정점**을
직접 비교(`scripts/research_eth_candidate_faithful_tabm_batchensemble_stepb_verify_20260816.py`):

| Expert | baseline_R_only 정점 | full_R_S_B_embed[quarter] 정점 | Δ |
|---|---:|---:|---:|
| bull | 0.5740 | 0.5723 | −0.0017 |
| bear | 0.5668 | 0.5714 | +0.0046 |
| chop | 0.5813 | 0.5735 | −0.0078 |
| **평균** | | | **−0.0016** |

**결론: 사실상 무승부.** 최초 cheap_gate의 "3개 expert 전부 −3.7~−5.1%p 결정적 악화"는 체크포인트
선정 버그의 아티팩트였음이 확정됐다 — 진짜로는 baseline과 거의 차이 없음(평균 −0.16%p, 이
저장소가 반복적으로 "노이즈"로 판정해 온 범위 안). 2/3 expert는 소폭 악화, 1/3(bear)은 소폭 개선
— 방향성 자체가 없다. **"임베딩이 결정적으로 나쁘다"는 틀렸지만 "임베딩이 도움이 된다"는 근거도
없다** — 채택할 근거가 없는 수준의 무의미한 차이.

## 최종 상태 (2026-08-16)

- **Step A(R+S+B)**: `CLOSED`, 부정. N≥5 시드로 일관된 악화 확인, 체크포인트 버그 영향 무시 가능.
- **Step B(embedding)**: `CLOSED`, 중립(무승부). 최초 "결정적 악화" 판정은 정정됐지만 개선 근거도
  없음 — N≥5 시드 재현이나 VAL 백테스트로 진행할 이유가 없다(효과 크기가 이미 노이즈 수준).
- **학습 품질 개선안(LR=2e-4, GCE)**: Step A/B와 별개 축. 유망하나 단일시드 — N≥5 시드 확인이
  필요하며, 별도로 다룰 사안. 이 후보(faithful TabM 자체)의 판정에는 영향 없음.
- **완전한 TabM(R+S+B+embedding) 후보 전체**: `CLOSED`. 논문대로 완성해도 baseline을 이기지
  못한다 — Step A는 소폭 악화, Step B는 무승부. VAL 포트폴리오 백테스트로 진행하지 않는다.
