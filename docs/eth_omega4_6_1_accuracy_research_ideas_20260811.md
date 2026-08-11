# Omega4.6.1 정확도 개선 리서치 아이디어 (2026-08-11)

zig075 jmredesign Step C 세션에서, "corrected TabM 백본 + 실제 헤드별 라벨"을 근거로 정확도를
끌어올릴 수 있는 방향을 조사한 결과. **구현 전 리서치 단계 문서** — 여기 있는 어떤 아이디어도
아직 코드로 옮기지 않았다.

## 0. 왜 이 순서로 접근했는가

h48qual 세션이 이번 주에 겪은 일(메모리 `h48qual-label-mismatch-discovered`,
`h48qual-standalone-replay-invalid`)이 정확히 이 문서가 피하려는 실수다: 실제 배포된 라벨/아키텍처를
확인하지 않고 "비슷해 보이는" 걸 다시 만들어서 며칠을 태웠다. 그래서 아래는 전부
`docs/model_contracts/omega4_6_1_full_architecture_blueprint_20260706.md`(실제 라이브 아키텍처)와
`docs/model_contracts/research_line_registry.json`(이미 시도되고 죽은 21개 리서치 라인)을 직접 읽고
검증한 사실 위에서만 제안한다.

## 1. 그라운딩: 실제 아키텍처 + 헤드별 라벨

- **백본**: TabM, k=8 (가중치 대부분 공유, 멤버별 소규모 파라미터로 앙상블 근사), hidden=192, 3
  layers, dropout=0.08. h48qual/zig075 각각 bull/bear/chop 3개 expert 서브네트워크 보유(HMM regime
  라우팅, end-to-end 학습 아님).
- **direction head**: 3-class(CASH/LONG/SHORT), 타깃 = `zigzag_action` — **h48qual과 zig075가
  동일**하게 공유.
- **quality head**: h48qual = 독립적인 48-bar ATR-relative barrier 규칙(`h48_conservative`,
  tp_mult=1.2, sl_mult=0.8, TP:SL=1.5); zig075 = `same_as_direction`(방향 확신도 자체가 quality).
  **두 컴포넌트가 서로 다른 quality 타깃을 쓴다** — 여기 안 섞는 게 중요.
- **exit head**: 이진(hold/exit), 포지션 보유 중에만 활성화, 13개 position-state 피처(side,
  hold_bars, unrealized move, MFE, MAE, giveback ratio, TP/SL까지 거리, notional, leverage,
  exposure, TP, SL) 입력, threshold=0.95 고정.
- **입력**: 102 base(96 기술/오더플로우/OU 피처 + 6 regime3-current) + 13 position-state = 115.

## 2. 이미 죽은 라인 (다시 제안하지 않을 것들)

`research_line_registry.json`의 21개 라인 중 이번 리서치와 직접 겹치는 것들:

| 시도 | 결과 |
|---|---|
| 44-feature kitchen-sink 필터 | in-sample AUC 0.9564, OOS 0.5166 — 순수 암기 |
| JEPA 학습 임베딩(자기지도) | 0/9 — 학습된 임베딩이 raw 입력보다 랭킹이 낮았음 |
| RL 기반 방향 예측 | 3 seed 전부 OOS negative, 모델군 바꿔도 동일 |
| zigzag를 엔트리 컴포넌트로 (curriculum/multi-task 포함) | 테스트한 모든 결합 방식이 PnL/MDD 둘 다 악화 |
| DVOL 피처 오버레이 | 0/9, 대부분 OOS 악화 |
| TP-first 3-way 라벨링 | 0/24, P(TP)-P(SL)가 실제 품질과 무관 |
| BTC panel 방향(cross-sectional) | rank score 거의 상수, 전부 negative |
| conformal(APS) abstention (h48qual 확률 위에) | 17개 중 하나로 실패 |
| 144-bar 모멘텀 규칙 | 처음엔 게이트 통과했지만 always_short baseline과 동일해서 철회 |
| 바리어/호라이즌 캘리브레이션 | 0/148 |

**공통 교훈**: (1) "재분류"(어느 방향이 맞는지 다시 고르기)는 이 프로젝트에서 계속 실패하고,
"필터링"(이 바가 나쁜지 스킵)만 반복적으로 작동해왔다 — h48qual의 진짜 가치가 바로 이거다. (2)
raw 피처보다 "똑똑한" 파생/학습 표현이 거의 항상 더 나쁘게 나왔다(JEPA, DVOL, TP-first 라벨 전부).
(3) always_long/short 벤치마크 없이 나온 "긍정적" 숫자는 신뢰하지 않는다. (4) 단일 시드 비교는
노이즈다(`tabm-hp-low-signal-pattern` 메모리, std 0.0009 > 전형적 HP 효과 크기).

## 3. 제안하는 방향 (4개, 각각 싼 falsification 테스트부터)

### 3-1. TabM 자체 앙상블 불일치(disagreement)를 quality/risk 신호로 노출

**근거**: `scripts/train_eval_omega1_2_tabm_3head_20260603.py:355-357`에서 k=8 멤버의 softmax를
`torch.softmax(...).mean(dim=1)`로 평균만 내고, 멤버 간 분산은 그 자리에서 버려진다. **이미
forward pass에서 다 계산돼 있는데 그냥 버리는 정보다** — 추가 학습이나 추가 forward pass가 필요
없다. Deep Ensembles의 인식적 불확실성(epistemic uncertainty) 문헌(Lakshminarayanan 2017)과
TabM 논문 자체가 "TabM ≈ deep ensemble 근사"라고 설계 의도를 명시하므로, k members 간 분산은
진짜 불확실성 신호일 개연성이 있다.

**이미 실패한 conformal abstention과 다른 점**: registry의 실패 사례는 확률값 위에 conformal
SET을 씌워 **경성 abstention**(거래 안 함)으로 썼다. 여기서는 (a) 방향을 다시 고르지 않고, (b)
경성 게이트도 아니고, L4 risk-sizing sidecar(이미 검증된 "사이징만 하는" 메커니즘)에 **연속값
피처**로 추가해서 사이징을 조정하는 용도로만 쓴다 — "필터링은 되는데 재분류는 안 된다"는 이미
확립된 교훈과 정확히 같은 방향.

**싼 falsification 테스트**: 새 모델 학습 없이, 기존 zig075/h48qual 체크포인트로 저장된 예측에서
k-차원 분산을 뽑아서(inference 코드에 한 줄 추가) 실현된 승/패와의 상관관계만 먼저 본다. 상관이
없으면 여기서 즉시 종료.

### 3-2. 수치 피처에 Piecewise-Linear Encoding (PLE) 적용

**근거**: Gorishniy et al. 2022 ("On Embeddings for Numerical Features in Tabular Deep
Learning")의 PLE는 각 수치 피처를 quantile 기반 구간으로 나눠 결정론적으로 인코딩한다 — GBDT의
축-정렬 분할(axis-aligned split) 능력을 미분 가능한 모델에 부여하는 게 핵심 동기다. **JEPA와는
메커니즘이 다르다**: JEPA는 자기지도로 "학습된" 표현이고 실패했지만, PLE는 학습되지 않는 결정론적
구간화라 같은 실패 모드(비지도 표현이 원 신호를 잃어버림)에 해당하지 않는다.

**싼 falsification 테스트**: 이번 세션에서 이미 검증한 15개 최종 피처만 대상으로, PLE 적용 전/후
5-seed VAL logloss를 비교. 15개뿐이라 학습 비용이 작다.

### 3-3. Exit head를 방향 재분류가 아니라 optimal-stopping 문제로 재구성

**근거**: registry의 실패 21개 중 **exit head를 직접 겨냥한 라인은 없다** — 전부 entry/direction
쪽이었다. Exit은 조건화 정보가 훨씬 풍부하다(포지션 보유 중, 실현/미실현 MFE·MAE 다 앎) — 이건
direction head보다 근본적으로 더 쉬운 문제다. 현재는 이진 분류(hold/exit, threshold=0.95
고정)인데, optimal-stopping 문헌(Becker/Cheridito/Jentzen 2019, "Deep Optimal Stopping")은
이걸 **연속 가치함수 회귀**(지금 청산했을 때 vs 한 스텝 더 들고 있을 때의 기대가치 차이)로 풀면
경성 임계값보다 안정적이라고 보고한다. `btc_advanced_rl_direction`의 실패는 direction에 대한
것이었지 exit/stopping에 대한 게 아니다 — RL/가치함수 계열이 이 프로젝트에서 전부 막힌 건 아니다.

**싼 falsification 테스트**: 현재 저장된 exit head 확률과 실제 forward-looking 최적 청산 시점을
비교해서, 0.95 경성 임계값이 레짐별로 얼마나 최적에서 벗어나 있는지부터 진단(재학습 없이 isotonic
recalibration만).

### 3-4. Quality threshold의 레짐별 재보정 (구조는 그대로, 임계값만 분리)

**근거**: 지금은 quality_threshold가 전역 상수(h48qual=0.50, zig075=0.75)다. 하지만 direction/
quality 자체는 bull/bear/chop expert로 이미 레짐 분기가 돼 있다 — **모델 구조는 안 건드리고
threshold만 레짐별로 따로 잡는 것**은 재분류가 아니라 순수 캘리브레이션(Platt
scaling/isotonic, subgroup별)이라 "필터링은 된다"는 교훈에 정확히 들어맞는다.

**싼 falsification 테스트**: 재학습 없이, 저장된 quality 확률 + 실현 결과로 레짐별 ROC를 그려서
전역 0.50/0.75가 세 레짐에 고르게 최적인지부터 확인.

## 4. 제안 우선순위

1. **3-1 (앙상블 불일치)** — 재학습 불필요, 저장된 예측만으로 상관관계 먼저 확인 가능. 가장 싸고
   가장 빠르게 죽이거나 살릴 수 있음.
2. **3-4 (레짐별 threshold)** — 마찬가지로 재학습 불필요.
3. **3-3 (exit optimal-stopping)** — 진단은 싸지만 실제 개선하려면 결국 재학습 필요.
4. **3-2 (PLE)** — 15개 피처 한정이라 싸지만, 유일하게 처음부터 재학습이 필요한 항목.

전부 재학습 없는 진단부터 시작해서, 상관관계/개선 여지가 실제로 보이는 것만 재학습으로
넘어가는 순서를 권한다 — registry의 21개 실패 라인 대부분이 "진단 없이 바로 풀 스케일 재학습"에서
비용을 태운 패턴이었다.
