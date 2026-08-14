# ETH Omega4.6.1 — Gittins Index Deep RL(QGI/DGN) 기반 exit_head 재정식화 (2026-08-14)

## 상태

**완료, VAL 게이트 자체 실패로 OOS 미실행(`REJECTED_VAL_GATE`)** — Odyssey2 문헌 스카우팅(#6)
5위(최종) 후보. 1~4위(대기압력·Risk-Controlled Post-Processing·Conformal Kelly·Selective
Conformal Risk Control)는 전부 부정 결과로 종결됐고, 이 문서는 그 큐의 마지막 항목이자 오늘 밤
유일하게 "기존 threshold/모델 재조합"이 아니라 **진짜 새 학습**(Deep Gittins Network)을 시도한
실험이다. 실패했지만, GBDT(#4)·TCN(#5)이 반복 관찰한 "컴포넌트 경제성 붕괴 vs 포트폴리오 개선"
패턴이 완전히 다른 학습 방법론(TD-부트스트랩 회귀)에서도 재현된다는 점과, 그 재현 메커니즘에 대한
새로운 가설(아래 "실패 메커니즘 진단" 절)을 확보했다는 점에서 진단 가치가 있다.

## 배경과 범위

`docs/experiments/eth_omega461_post_entry_literature_scouting_20260814.md`가 5위로 랭킹한 후보를
실행한다: Dhankhar, Mishra, Bodas, "Tabular and Deep Reinforcement Learning for Gittins Index"
(arXiv:2405.01157, v1 2024-05-02 / v4 2025-08-25)의 QGI(tabular)/DGN(deep) 알고리즘. h48qual의
exit_head **모델만** 이 방식으로 재정의한다 — direction_head/quality_head/encoder는 완전 동결,
zig075는 완전 동결(GBDT #4/TCN #5/오늘 밤 모든 exit_head 실험과 동일 원칙).

## 1단계: 논문 메커니즘 (WebFetch로 직접 확인)

arXiv:2405.01157 abs 페이지와 HTML 전문(`https://arxiv.org/html/2405.01157v4`)을 직접 fetch해
확인한 정확한 메커니즘.

### Retirement formulation (은퇴 정식화)

상태 x, 은퇴보상 M이 주어졌을 때:

```
Vr(x,M) = max{Q_M(x,1), Q_M(x,0)}
Q_M(x,0) = M                                                    (은퇴 = 고정 보상 M)
Q_M(x,1) = r(x,1) + γ·Σ_j p(j|x,1)·max{Q_M(j,1), Q_M(j,0)}      (계속 = 미래에도 같은 M을 walkaway 옵션으로 유지)
```

무차별점(indifference point) `M(x) = inf{M : Vr(x,M) = M}` — "지금 당장 M을 받고 은퇴하는 것"과
"이 상태부터 계속하되 언제든 M(x)로 은퇴할 수 있는 옵션을 유지하는 것"이 같은 가치가 되는 **가장
작은 M**. Gittins index는 `G(x) = M(x)·(1-γ)`. 이 M(x)는 "이 자원(arm)을 계속 쥐고 있는 것의
절대적 가치"를 하나의 스칼라로 압축한 것 — **자원 하나만의 값**이지, 경쟁하는 다른 arm의 상태를
참조하지 않는다(중요, 아래 "비동형성" 절에서 다시 다룸).

### QGI (tabular) vs DGN (deep) — 정확한 차이

- **QGI**: 참조상태 x마다 Q-테이블 `Q^x(s,1)`을 유지. 매 관측 전이 `(s_n→s_{n+1})`마다:
  `Q^x_{n+1}(s_n,1) = (1-α)Q^x_n(s_n,1) + α[r(s_n) + γ·max{Q^x_n(s_{n+1},1), M_n(x)}]` (Eq 3),
  `M_{n+1}(x) = M_n(x) + β(n)[Q^x_{n+1}(x,1) - M_n(x)]` (Eq 8, 느린 시간축). 논문이 강조하는
  핵심 효율성: **하나의 전이 샘플이 모든 참조상태 x에 대해 동시에 Q^x 테이블 행을 갱신**한다
  (r(s_n)과 전이 자체는 x와 무관, x는 오직 `max(...,M_n(x))`의 floor로만 개입하므로) — QWI 대비
  Q-테이블 크기 절반(action=0을 위한 저장 불필요).
- **DGN**: 연속·고차원 상태공간에서 x를 표로 열거하는 대신, **입력이 (s,x) 쌍인 신경망**
  `Q^x_θ(s,1)`을 학습(Section IV, 3-hidden-layer (64,128,64), ReLU). 학습 타겟(Eq 9):
  `Q^x_target(s_k,1) = r(s_k) + γ·max(Q^x_θ'(s'_k,1), M_n(x))` (θ'는 소프트 업데이트 타겟망).
  손실(Eq 10): `MSE = (1/B)·Σ_k Σ_x (target - Q^x_θ(s_k,1))²` — 미니배치 내 **여러 참조상태 x에
  대한 이중 합**. `M_{n+1}(x) = M_n(x) + β(n)[Q^x_θ(x,1) - M_n(x)]` (Eq 11, 여전히 느린 시간축).
  DGN은 action=0(은퇴)에 대한 경험을 저장하지 않아 replay buffer가 QWINN보다 작다.

### 논문이 실증한 응용: 배치 도착·미지 서비스시간 job 스케줄링

Section V: **K개 job이 시각 0에 전부 동시 도착**(배치, 순차 도착 아님), 단일 서버, preemptive-resume
(전환비용 없음). 상태 = job의 "나이"(이미 받은 서비스량) `s∈{1,...,N_max}`, 서비스시간 분포 미지
(hazard `ρ_s(i)`로 표현). **여기서 "은퇴"는 실제 행동이 아니라 G(x)를 계산하기 위한 내부 계산
장치다** — 실제 스케줄링 결정("지금 어느 job을 서빙할지")은 별도로, **매 순간 Gittins index가
가장 높은 job을 서빙**하는 index policy로 이뤄진다(Gittins 정리가 이 index policy의 전역 최적성을
보장). 목표는 평균 flowtime 최소화. 이산분포(Geometric/Binomial/Poisson)와 연속분포(Uniform/
Log-normal, quantum Δ로 이산화)에서 검증.

### 논문이 보고한 한계

- QWI는 학습률에 극도로 민감(4200개 하이퍼파라미터 조합 중 δ=0.02 이내 수렴 227/4200 vs QGI만
  비교 가능할 정도로 QWI/QWINN이 스케줄링 실험에서 "매우 낮은 성능"을 보여 restart-in-state와만
  비교).
- **DGN은 QGI(Theorem 1)와 달리 공식 수렴 증명이 없다** — 경험적 수렴만 보임.
- 낮은 ε(탐색)에서 QWI의 특정 상태 인덱스가 "비정상적으로 증가"하는 현상 보고(Appendix E1).
- Restless bandit(수동 arm도 상태 전이)은 명시적으로 범위 밖 — 논문은 "rested" bandit(수동
  arm은 상태 불변)만 다룸, Whittle index는 부록에서만 언급.

## 2단계: 이 프로젝트 재정식화 — 설계와 근거

| 논문 개념 | 이 프로젝트 매핑 | 근거/출처 |
|---|---|---|
| 자원(서버) | 공유 슬롯 1개 | h48qual·zig075가 경쟁(기존 구조) |
| 상태 s, x | 115차원 position-conditioned 벡터(102 base_cols + 13 pos_cols: pos_side/hold_bars/unrealized/mfe/mae/giveback/dist_to_tp/dist_to_sl/notional/leverage/exposure/tp/sl) | TabM/GBDT(#4)/TCN(#5) exit_head와 완전히 동일한 계약(`parent._exit_input_from_position_rows`, `parent.POS_COLS`) — 재사용, 재발명 안 함 |
| 행동(계속/은퇴) | hold vs exit(포지션 청산) | 기존 exit_head와 동일한 이진 결정의 재파라미터화 |
| 보상 r(s_t) | bar 단위 `exit_path_unrealized`(고정 참조 notional=0.45 기준 mark-to-market 미실현PnL)의 증분 | `research_eth_omega461_exit_head_liveatr_relabel_20260813._build_exit_dataset_entry_label_live_atr_barrier`가 이미 매 bar 계산해두는 값을 그대로 차분(신규 라벨링 없음) |
| 종단(episode 끝) | 각 후보의 TP/SL/timeout 배리어 도달 bar | 같은 함수의 `exit_path_hold_bars`/`exit_path_entry_i`로 candidate별 순차 경로 복원 |
| 은퇴가치 M(x) | 학습된 신경망의 대각선 자기평가 `Q_θ(x,x)` | 아래 "핵심 설계 결정" 참고 |

### 핵심 설계 결정과 근거 (완전히 새로 만들지 않고 기존 자산과 연결)

1. **데이터셋 재사용, 재구축 안 함**: `train_eval_omega461_gbdt_exit_head_liveatr_20260813._build_dataset`
   (seed=260813, max_candidates=1500)를 **무수정 import**로 호출 — GBDT(#4)·TCN(#5)과 바이트 단위
   동일 데이터셋(1,234,431행, 1,500 candidates, positive_rate 19.9%)을 재구성했고, 서버에서
   `dataset_reference_check`로 rows/positive_count/used_candidates 3개 지표 전부 정확히 일치함을
   확인했다(REFERENCE: `tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500/report.json`).
2. **은퇴가치 타겟을 기존 y-규칙의 단순 변환으로 만들지 않기로 결정**: 사용자 지시가 제안한
   "기존 `pos_giveback`/`pos_unrealized` 규칙(#8/#14가 재사용한 y-rule)을 은퇴가치로 변환"하는
   경로를 검토했으나, 그 규칙은 **이진 분류 라벨**(giveback≥0.65 또는 unrealized≤-0.010이면
   exit=1)이라 "얼마나 좋은가"라는 연속적 가치 정보를 담지 않는다 — 논문의 Eq 9-11은 명시적으로
   TD 부트스트랩 회귀 타겟을 요구하므로, 이 이진 라벨을 그대로 재사용하면 논문의 핵심 메커니즘
   (재귀적 자기일관성)을 버리게 된다. 대신 **같은 데이터셋의 bar-순차 구조**(`exit_path_hold_bars`가
   candidate별 forward-simulated 경로를 이미 순서대로 담고 있음, 직접 확인)에서 진짜
   `(s_t, r_t, s_{t+1})` TD 전이를 복원해 논문의 실제 학습 절차를 충실히 구현했다 — y-rule은
   비교용으로만 데이터셋에 남아있고(exit_diag를 통해), 학습 타겟으로는 쓰지 않았다.
3. **참조상태 x = 미니배치 자기 자신(대각 근사)**: 논문은 x가 전체 참조 테이블(job age 1..N_max,
   유한·이산)을 순회한다고 가정하지만, 이 프로젝트의 115차원 연속 상태공간에는 그런 유한 테이블이
   없다. 매 스텝 미니배치(B=256)의 상태 자신을 참조상태 집합으로 재사용해 `(B,B)` 쌍 전부에 대해
   Eq 9-10을 계산한다(에폭마다 다른 미니배치가 샘플링되므로 다양한 x가 누적적으로 커버됨) — 연속
   상태공간에 index류 방법을 확장하는 표준적 방식이며, 논문 자신의 "하나의 전이가 모든 x를
   갱신한다"는 효율성 성질의 자연스러운 연속화다.
4. **M(x)를 별도 3번째 네트워크 대신 타겟망의 대각선 읽기로 구현**: Eq 11의 M-업데이트(느린
   EMA)와 표준 DQN 타겟망(소프트 업데이트, Eq 9 부트스트랩 안정화용)은 둘 다 "온라인망 자기평가의
   느리게 추적되는 복사본"이라는 같은 종류의 장치다 — 이 둘을 병합해 `M_n(x) := Q_θ'(x,x)`로
   정의했다(타겟망 τ=0.01 Polyak 업데이트). 네트워크 3개(온라인 Q, 타겟 Q, 별도 M)를 유지하는
   대신 2개로 줄이는 명시적 단순화.
5. **레짐별 3개 전문가(bull/bear/chop)**: GBDT/TCN과 동일한 구조(`hard.EXPERT_NAMES`) — 통합
   신경망 1개가 아니라 이 프로젝트의 기존 라우팅 아키텍처와 일치시켜 `_predict_exit_prob_one`과
   같은 호출 지점(`train_eval_omega4_2_risk_sidecar_20260622._prepare_exit_runtime`)에 그대로
   꽂을 수 있게 했다. 가중치는 GBDT/TCN의 `compute_sample_weight(balanced)*route_prob`에서
   `balanced`(이진 분류의 클래스 불균형 보정, 회귀에는 해당 없음) 부분을 뺀 **route_prob만**
   (B,B) 손실 행렬의 sample축·reference축 양쪽에 외적으로 적용 — 회귀에 맞게 정직하게 축소했다.
6. **배포 모델은 타겟망**: 학습 종료 시점의 온라인망이 아니라 느리게 추적된 타겟망을 저장·배포 —
   더 안정적인 M(x) 추정치라는 판단(문서화된 선택).

### 비동형성 명시 (사용자 지시: "다를 수 있음 — 직접 검증할 것")

문헌 스카우팅 문서가 제기한 "이 프로젝트 구조가 논문의 배치 스케줄링 응용과 구조적으로 동형"이라는
가설을 직접 검증한 결과, **완전한 동형이 아니다**:

1. **배치 동시도착 vs 독립 트리거**: 논문의 K개 job은 시각 0에 전부 대기 중이고 전환비용 없이
   자유롭게 재개된다("rested" 가정 — 서빙 안 하는 job은 상태 불변). 이 프로젝트의 h48qual/zig075는
   "항상 대기 중인 job"이 아니라 **각자의 진입조건이 독립적으로 미래에 트리거되어야 하는 신호**다 —
   슬롯을 비운다고 곧바로 "다음 job"이 서빙되는 게 보장되지 않는다. 게다가 실제 청산/재진입에는
   수수료·슬리피지 비용이 있어 논문의 "무료 preemption" 가정과도 다르다.
2. **은퇴(계산 장치) vs 청산(실제 행동)**: 논문에서 "은퇴"는 G(x)를 계산하기 위한 **가상의 내부
   장치**(실제 스케줄링 결정은 "index가 가장 높은 job을 서빙"이라는 별도 index policy)이지, 그
   결정 자체가 아니다. 이 프로젝트에서는 이 구분이 무너진다 — 매 bar 실제로 내려야 하는 "hold vs
   exit" 결정과 "은퇴 가치를 계산하는 내부 절차"가 **같은 하나의 행동**으로 합쳐진다(공유계좌·
   단일슬롯 구조상 h48qual 포지션을 "은퇴"시키는 것이 곧 실제로 슬롯을 비우는 것이기 때문). 이
   합침은 명시적으로 도입한 단순화다.
3. **진짜 index policy(경쟁 arm 간 비교) 미구현**: Gittins 정리의 전역 최적성 보장은 "매 순간
   경쟁하는 모든 arm의 현재 index를 비교해 최댓값을 서빙"하는 데서 나온다. 이 실험은 h48qual의
   M(x)를 **zig075의 현재 상태와 비교하지 않고 고정 상수(threshold)와만 비교**한다 — zig075/entry
   로직을 건드리지 않는다는 이 서브프로젝트의 제약(h48qual exit_head만 재정의) 때문에 의도적으로
   생략했다. 즉 이 실험이 실제로 만든 것은 **"TD 부트스트랩으로 학습된 연속값 exit_head"**이지,
   논문이 증명하는 최적성을 보유한 진짜 Gittins index 정책은 아니다 — 이 구분은 아래 "실패
   메커니즘 진단"에서 실패 원인 가설과 직결된다.

## 3단계: 서버 학습 — GPU 여부 판단, 메모리 안전장치, 실제 로그

### GPU 필요성 판단

서버 확인 결과 `nvidia-smi`는 PATH에 없었지만 `torch.cuda.is_available()=True`(RTX 3070 Ti,
8.6GB 중 7.4GB 여유) — WSL2의 CUDA 런타임 경로가 살아있음을 직접 확인했다. 어제 밤 JM 전체재학습은
"가벼워서 CPU로도 개당 ~3.5분"이었지만, 이 실험은 **구조적으로 다르다** — DGN 학습 스텝마다
미니배치(B=256) 내 모든 (s,x) 쌍(B×B=65,536쌍)을 신경망에 통과시켜야 한다(Eq 10의 이중 합). dev
CPU 스모크테스트(B=16, 8스텝)로 파이프라인 정합성만 먼저 검증한 뒤(문제 없음 확인), B×B 스케일링이
O(B²)임을 근거로 실제 배치(B=256)에서 CPU 처리량을 외삽하면(16→256, 256배 연산량) 전문가당
수십~100분대, 3개 전문가 합산 2~5시간대로 예상돼 **GPU를 사용하기로 판단**했다 — 이 판단은 실측으로
뒷받침된다(아래 실제 로그, GPU에서 전문가당 25초 vs 외삽한 CPU 추정 수십 분).

### 메모리 안전장치 (서버 메모리 사고 전례 반영)

`scripts/run_jm_full_retrain_seed_robustness_20260813.sh`의 안전장치(순차 실행, 매 스테이지 후
`free -h` 로그, 4GB 미만이면 즉시 중단)를 학습 스크립트에 직접 코드화(`_mem_check`, 8회 호출:
시작/데이터셋빌드 후/표준화 후/전문가별×3). 학습 전 `handoff.sh status server`로 3개 섀도우봇
(`eth_exithead_asymmetric_shadow`, `eth_regime_aware_exit_guard_shadow`,
`eth-jmlam4-shadow.service`) 전부 RUNNING 확인, 학습 스크립트가 이들을 건드리는 코드 경로는 전혀
없음(별도 프로세스, 별도 파일).

### 실제 실행 로그

| 단계 | 소요시간 | 비고 |
|---|---:|---|
| 데이터셋 빌드(`_prepare_frames`+timescale checkpoint+후보1500개 forward 시뮬레이션) | 447.8초(~7.5분) | GBDT/TCN과 동일 비용(CPU-bound, 순수 feature 구성) |
| transitions 복원 + 표준화 | 수 초 | 1,234,431개 전이, candidate-level 90/10 분할(train 1,102,081 / val 132,350) |
| bull 전문가 학습(4000 스텝, B=256, GPU) | 25.1초 | final train_loss 8.5e-5, val_loss 7.0e-5 |
| bear 전문가 학습 | 24.8초 | final train_loss 7.8e-6, val_loss 2.2e-6 |
| chop 전문가 학습 | 24.8초 | final train_loss 8.5e-7, val_loss 9.9e-7 |
| **총 wall time** | **~8.9분** | GPU 처리량 ~160 steps/sec(B×B=65,536쌍/스텝) |

메모리: 서버 가용 메모리 17.8~23.3GB(31GB 중)로 안정, GPU 여유 6.4~7.4GB(8.6GB 중)로 안정 — 안전
중단 발동 0회. 데이터셋은 `dataset_reference_check`(rows/positive_count/used_candidates 3개 지표)
전부 정확히 일치. 산출물: `tmp/causal_regen_20260516/eth_omega461_gittins_retirement_exit_head_20260814/h48qual/gittins_retirement_bundle.pt`
(전문가별 state_dict + 스케일러 + 아키텍처 계약), `.../report.json`.

## 4단계: G0

`h48cons._evaluate_val` + `eth_omega461_multiwindow_confirmation_gate_20260814.run_portfolio_variant`를
그대로 재사용해 4개 기준값 전부 재현(오케스트레이터가 지정한 값과 정확히 일치):

| 지표 | 실제값 | 기준값 | 일치 |
|---|---:|---:|---|
| component_baseline_original | 5.45%/-11.62%/29 | 5.45%/-11.62%/29 | ✅ |
| component_tabm_liveatr | 9.23%/-7.59%/63 | 9.23%/-7.59%/63 | ✅ |
| VAL no_gate (asymmetric_tabm_liveatr) | 46.59%/-21.70%/35 | 46.59%/-21.70%/35 | ✅ |
| VAL with_gate | 77.31%/-21.76%/26 | 77.31%/-21.76%/26 | ✅ |
| OOS-Q1 no_gate | 93.27%/-15.48%/24 | 93.27%/-15.48%/24 | ✅ |
| OOS-Q1 with_gate | 67.25%/-15.48%/19 | 67.25%/-15.48%/19 | ✅ |

**G0 PASS.**

## 5단계: VAL 결과

### 임계값 그리드 선정 (never-trigger 진단 패스 기반, 사후선택 아님)

`retirement_threshold=-1e9`(절대 발동 안 함)로 컴포넌트 단독 VAL을 한 번 실행해 보유 bar마다 계산된
M(x) 19,438개를 수집: 분포는 `min≈-5e-6(사실상 0), p10=0.0261, p25=0.0349, p50=0.0458,
max=0.1427, mean=0.0496` — **우측으로 매우 치우친 분포이며 의미 있는 음수 꼬리가 없다**(중요,
아래 진단 참고). 그리드 = `{0.0(사전 경제적 앵커: "계속의 기대 한계가치가 0 이하면 은퇴"),
p10, p25, p50}` = `[0.0, 0.0261, 0.0349, 0.0458]`.

### 스윕 결과

기준선: VAL with_gate baseline **77.31%/-21.76%**(주의: `eth_omega461_relaxed_gate_rescoring_20260814.md`
표의 "54.88%/-31.11%"는 Odyssey2 실행 로그 #8이 발견한 대로 `baseline_both_original` 원장 값이며
이 실험의 올바른 비교 기준이 아니다 — `asymmetric_tabm_liveatr`의 진짜 with_gate 값 77.31%/-21.76%를
사용했다).

| threshold | 컴포넌트 PnL | 가드레일 | 포트폴리오 no_gate | 포트폴리오 with_gate | 원기준 | 완화기준 |
|---:|---:|---|---|---|---|---|
| 0.0000 | +13.95% | PASS | 36.82%/-24.34%/29 | 54.88%/-31.11%/22 | FAIL | FAIL |
| 0.0261 | +6.85% | PASS | -0.82%/-36.34%/34 | 6.72%/-42.04%/27 | FAIL | FAIL |
| 0.0349 | **-5.02%**(부호반전) | **FAIL** | 59.21%/-21.61%/36 | 84.35%/-25.27%/28 | FAIL | FAIL |
| 0.0458 | **-2.48%**(부호반전) | **FAIL** | 49.07%/-21.55%/44 | 72.61%/-22.40%/36 | FAIL | FAIL |

**`val_winner = None`** — 4개 그리드점 전부 (원기준 OR 완화기준) AND 가드레일을 동시에 통과하지
못한다. threshold=0.0은 컴포넌트 exit_head가 사실상 발동하지 않는(19,438개 관측 M값 중 음수가
거의 없으므로) **퇴화 모드**로, 포트폴리오 수치가 `baseline_both_original`(h48qual·zig075 둘 다
원본 exit_head)의 수치와 정확히 일치한다 — 우연이 아니라, 원본 exit_head도 이 VAL 구간에서 "거의
발동 안 함"이 이미 문서화된 사실이기 때문이다(`research_eth_omega461_exit_head_liveatr_relabel_20260813.py`
독스트링: "frozen SLTP + original exit_head, exit_head never fires"). 즉 threshold=0.0은 기존에
이미 알려진 baseline을 재현할 뿐 새로운 정보가 없다.

### 실패 메커니즘 진단

1. **급격한 상전이(phase transition), 점진적 조정 불가**: threshold를 0.0에서 10th 백분위수
   (0.0261)로 아주 조금만 올려도 컴포넌트 exit_head 발동 횟수가 2건 → 70건으로 35배 뛰고
   (거래수 30→81), 포트폴리오 no_gate PnL은 +36.82%→-0.82%로 붕괴한다. 25th/50th 백분위수로
   더 올리면 발동횟수가 130건, 209건까지 치솟아(전체 보유 bar 대비 압도적 비율) 컴포넌트가 거의
   즉시 청산하는 정책으로 변한다. **완만하게 조절 가능한 중간 지대가 관측되지 않았다** — M(x)
   분포 자체가 좁은 고값 구간(0.026~0.046)에 몰려있어(최솟값이 사실상 0), 이 구간을 스치는
   순간 대부분의 보유 bar가 동시에 "은퇴" 판정을 받는다.
2. **GBDT(#4)/TCN(#5)과 같은 컴포넌트-vs-포트폴리오 괴리가 재현된다**: threshold를 올릴수록
   포트폴리오 with_gate는 오히려 개선되는 것처럼 보이지만(84.35%, 72.61% — baseline 77.31%보다
   높음), 그 대가로 컴포넌트 단독 PnL이 부호반전한다(+9.23%→-5.02%/-2.48%) — 오늘 밤 확립된
   가드레일이 정확히 이 패턴을 걸러낸다. **학습 방법론이 완전히 다름에도(TD 부트스트랩 회귀 vs
   이진 분류/시퀀스 분류) 동일한 실패 계열이 재현된 것**은 이 프로젝트의 근본 제약(공유 슬롯
   재순환 상호작용)이 모델 종류가 아니라 **정책의 형태**(전역 상수 threshold 하나로 이진 결정)에
   있다는 가설을 강화한다.
3. **핵심 가설 — "진짜 index policy 부재"가 원인일 가능성**: 위 "비동형성" 절에서 명시했듯,
   이 실험은 h48qual의 M(x)를 zig075의 현재 상태와 **비교하지 않고** 고정 상수와만 비교한다.
   Gittins 정리의 최적성은 "경쟁 arm 간 index 비교"에서 나오는데, 이 단순화된 정책은 그 비교를
   생략했으므로 이론적으로 논문이 보장하는 최적성을 상속받지 못한다 — threshold를 넘는 순간
   "모든 보유 상태가 동시에 낮은 우선순위"로 취급되는 것은, 진짜 index policy라면 "그 순간
   zig075가 실제로 더 나은 대안을 갖고 있을 때만" 전환해야 할 결정을, 이 단순화가 "b48qual
   자신의 상태만으로" 내리기 때문일 수 있다(가설 — 직접 검증하지 않음, zig075 상태를 index
   비교에 포함시키는 것은 이 서브프로젝트 범위 밖). 이는 오늘 밤 문헌 스카우팅 1위 후보(대기압력,
   #7 — "반대 컴포넌트가 실제 대기 중일 때만" 개입하는 조건부 규칙)가 시도했던 것과 개념적으로
   같은 축이며, #7도 VAL 승리 후 OOS에서 반전돼 부정 결과였다는 점과 일치하는 관찰이다.

## 6단계: OOS

VAL 게이트(원기준 OR 완화기준) AND 가드레일을 통과하는 후보가 0개이므로, 이 프로젝트의 확립된
방법론(#9/#14/#15와 동일한 "VAL 기각 시 OOS 미개방")에 따라 **OOS-Q1/OOS-Q2는 실행하지 않았다**
(`oos_opened=false`).

## 최종 판정

`final_verdict = "REJECTED_VAL_GATE"`. 문헌 스카우팅(#6) 큐 전체(1~5위)가 이것으로 완전히
소진됐다.

## 생성/수정 파일

- `scripts/train_eval_omega461_gittins_retirement_exit_head_20260814.py`(서버 학습 — 신규,
  dev에서 작성 후 `handoff.sh push`로 서버 전송, 동일 사본이 dev에도 존재)
- `scripts/research_eth_omega461_gittins_index_exit_head_20260814.py`(dev 평가 — G0 + 은퇴가치
  주입 래퍼 + 이름바꾼 복사본(`replay_exit_variant_gittins`/`greedy_replay_gittins`) + VAL
  그리드 스윕 + OOS 단일터치 로직 전부 포함)
- `tmp/causal_regen_20260516/eth_omega461_gittins_retirement_exit_head_20260814/`(서버 산출물,
  `handoff.sh pull`로 dev에 회수: `h48qual/gittins_retirement_bundle.pt`, `report.json`)
- `tmp/causal_regen_20260516/eth_omega461_gittins_index_exit_head_20260814/`(dev 평가 산출물:
  `report.json`, 컴포넌트/포트폴리오 거래원장 CSV들, 정렬된 예측 CSV — 전부 diagnostic 전용)
- 본 문서, `docs/model_contracts/odyssey2_eth_live_injection_contract_20260813.md`(실행 로그
  #16 추가), `docs/model_contracts/odyssey2_eth_live_injection_data_resources_20260813.md`(신규
  파일 행 추가)

## 라이브 파일·섀도우봇 확인

- dev: `git diff --stat -- trading_bot.py trading_bot_modules/omega4_6_1_live.py trading_bot_modules/runtime_config.py` → 0줄. `.env` mtime 이번 세션 이전(4월) — 무접촉.
- server: 동일 3개 파일 `git diff --stat` → 0줄. `.env` mtime 이번 세션 이전(8/9) — 무접촉.
- 서버 섀도우봇 3개 학습·평가 실행 전후 모두 확인: `eth_exithead_asymmetric_shadow`(pid=62019
  RUNNING, 변동 없음), `eth_regime_aware_exit_guard_shadow`(pid=274710 RUNNING, 변동 없음),
  `eth-jmlam4-shadow.service`(systemd active, 변동 없음) — 전부 무사, 방해 없음.

## Fresh-Forward 체크리스트

`fresh_forward_bar_by_bar=true`(데이터셋 빌드는 기존 causal forward 배리어 시뮬레이션 무수정 재사용,
DGN 학습은 그 결과물의 이미 causal한 candidate별 순차 경로에 대한 오프라인 fitted-Q이지 신규
bar-by-bar 시뮬레이션이 아님, VAL/G0 리플레이는 전부 단일 causal forward pass).
`trade_ledgers_used_as_input=false`(원장은 출력 전용). `saved_parent_exit_timestamps_used=false`.
`future_rows_used_for_entry=false`. VAL 구간만 사용(2025-10-01~12-31), OOS는 열지 않았다.

## 정직한 결론

이 실험은 오늘 밤 유일하게 "기존 자산의 재조합"이 아니라 논문의 실제 알고리즘(TD 부트스트랩 +
자기참조 은퇴가치, Eq 9-11)을 충실히 구현한 시도였다. 데이터셋 재사용·인프라 재사용 면에서는
성공적이었고(G0 완벽 재현, 메모리/GPU 안전 판단 실측으로 검증됨), 학습 자체도 안정적으로 수렴했다.
그러나 VAL 게이트에서 실패했고, 그 실패는 우연한 노이즈라기보다 **구조적** — 학습된 은퇴가치가
좁은 고값 구간에 몰려 있어 단일 전역 임계값으로는 "점진적" 정책을 만들 수 없었고, 그 결과
GBDT/TCN과 동일한 "포트폴리오는 좋아 보이지만 컴포넌트 경제성이 무너지는" 패턴을 다른 경로로
재현했다. 가장 유력한 해석은, 이 구현이 논문의 진짜 강점(경쟁 arm 간 index 비교를 통한 전역 최적
스위칭 정책)을 상속받지 못하고 "값 기반으로 재파라미터화된 exit_head 분류기"로 축소됐기 때문이라는
것 — 이는 검증되지 않은 가설이지만, 문헌 스카우팅 문서가 사전에 경고했던 "논문이 트레이딩·
단일슬롯 사례를 직접 다루지 않아 재해석이 필요하다"는 우려가 실제로 실현된 사례로 읽는 것이
정직하다.
