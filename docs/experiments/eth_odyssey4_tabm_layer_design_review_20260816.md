# Odyssey4 ThreeHeadTabM 레이어 설계 평가 (2026-08-16)

상태: **완료. 코드 실측 기반 평가 — `docs/deep_learning_layer_design_and_training_reference_20260816.md`
참고자료의 각 항목을 실제 라이브 아키텍처 코드에 직접 대조했다.**

## 평가 대상

`scripts/train_eval_omega1_2_tabm_3head_20260603.py`의 `ThreeHeadTabM` (라인 87-122) —
h48qual·zig075 둘 다 이 정확히 같은 구조를 공유하며, `trading_bot_modules/odyssey_tabm_core.py`에
그대로 vendored돼 라이브로 돈다. Regime3 HMM(6개 피쳐 생성기)은 신경망 레이어가 아니라 별도
확률모델이라 이번 "레이어" 평가 범위에서 제외했다.

---

## 1. 구조 전체 흐름 (encode(), 라인 105-114)

```
xk = x·input_scale(R) + input_bias(B)     # 멤버별 near-identity 게이트, 라인 106
h  = in_proj(xk)                           # 공유 W, "stem" — 잔차 없음 (차원 불일치라 정상)
h  = Dropout(SiLU(norms[0](h)))            # 라인 108
for 2개 block:
    residual = h
    h = block(h · expert_scale[idx])       # R게이트 → 공유 W
    h = Dropout(SiLU(norms[idx+1](h)))     # 정규화 → 활성화 → 드롭아웃
    h = h + residual                       # 잔차는 정규화 이후에 더해짐
return h  # → direction_head/quality_head/exit_head 3개 선형층
```

k=8, hidden=192, layers=3(=stem 1 + block 2), dropout=0.08. **총 3단계 선형변환**으로 표준
기준으로는 매우 얕다.

## 2. 깊이(depth) — 참고자료 §1 대조

- **레이어 3개는 이 저장소 자체 문헌조사(TabReD 벤치마크)와 일치**: 표형 데이터에서 시간축 분할
  기준 단순 MLP/GBDT가 어텐션 기반 딥러닝을 이긴다는 근거가 있어(기존 메모리
  `eth_odyssey_dl_rl_architecture_axis_closed`), 얕은 구조 자체는 문제로 보지 않는다.
- **arXiv 모노그래프(§1) 프레임으로 보면**: 깊이 = 목적함수 최적화의 반복횟수. 3스텝은 그 관점에서
  "매우 적은 반복"이다 — 이게 부족한지는 수렴곡선을 직접 보기 전엔 판단 불가(현재 학습 스크립트는
  레이어별 목적함수 수렴을 모니터링하지 않음, val loss만 봄). **미검증 질문으로 남김**: layers=2나
  4로 바꿔봤을 때 어떻게 되는지 — 이건 기존에 닫힌 R+S+B 완성형 실험(같은 레이어 수에서 어댑터를
  더 표현력있게 만든 것)과는 **다른 용량 축**이라 그 결과를 여기 그대로 적용할 수 없다.
- **잔차연결 배치는 정상**: stem(in_proj)엔 잔차가 없는데, 입력 차원(n_features)과 hidden 차원이
  달라 그대로 더할 수 없기 때문 — 표준적이고 올바른 설계.

## 3. 초기화 — 참고자료 §1 대조 (구체적 공백 발견)

- **R게이트(`input_scale`, `expert_scale`)는 원리에 맞게 초기화됨**: `randn()*0.03+1.0` —
  근사 항등원(near-identity) 초기화, BatchEnsemble이 멤버들을 학습 초반엔 거의 동일하게
  시작해서 점진적으로 분화시키려는 의도와 정확히 일치. **긍정적으로 확인.**
- **공유 가중치(`in_proj`, `blocks`, 3개 헤드)는 전부 PyTorch 기본 초기화(Kaiming-uniform,
  a=√5)를 그대로 씀 — 명시적 초기화 스킴이 코드 어디에도 없다(grep 확인: `nn.init.*` 호출
  0건).** 참고자료가 인용한 Xavier(`σ=√(2/(n_in+n_out))`)도, He/Kaiming(ReLU 전용,
  `σ=√(2/n_in)`)도 명시적으로 선택된 게 아니다. 활성화가 ReLU가 아니라 SiLU라서 Xavier/He 둘 다
  완벽히 들어맞는 공식은 아니지만, **"의도적으로 고른 초기화가 없다"는 것 자체가 참고자료 §1의
  체크리스트 대비 명확한 공백**이다. 실전 영향은 불확실(PyTorch 기본값이 실무에서 크게 문제되는
  경우는 드묾) — 우선순위는 낮지만 명시적 공백으로 기록.

## 4. 정규화(Normalization) 배치 — 참고자료 §2 대조 (미묘하지만 확인됨)

Tuning Playbook의 규칙: `x + f(Norm(x))`(정규화가 먼저, pre-norm)가 `Norm(x + f(x))`(합 자체를
정규화, post-norm)보다 안정적. 실제 코드는 어느 쪽도 정확히 아니다 — **`h_new = residual +
Dropout(SiLU(Norm(Linear(residual·gate))))`**, 즉 정규화가 선형층의 **출력**에 적용되고, 그 결과가
잔차에 더해진다(합 자체는 정규화 안 함). 이건 원조 ResNet v1 스타일(conv→BN→활성화, 그 다음
더하기)에 가깝고, Playbook이 명시적으로 경고한 위험 패턴(`Norm(x+f(x))`)은 아니다 — **불안정
패턴은 피했지만, 참고자료가 추천하는 정석 pre-norm도 아닌 제3의 배치**. 실사용에서 불안정
징후(loss 급증, NaN)가 관측된 적은 없어서(로컬 sanity check들 전부 통과) 급한 문제는 아니지만,
학습이 불안정해지는 상황이 생기면 이 배치부터 pre-norm으로 바꿔보는 게 Playbook 체크리스트상 낮은
비용의 첫 시도가 된다.

BatchNorm이 아니라 LayerNorm을 쓴 것 자체는 **참고자료 §2가 명시적으로 지지하는 선택**(우리
5분봉 배치는 자기상관 있는 시계열이지 i.i.d. 샘플이 아님) — 다만 `DataLoader(shuffle=True)`로
배치를 랜덤 섞기 때문에 애초에 BatchNorm이었어도 시간적 자기상관 문제는 크지 않았을 것 — 이 경고가
실질적으로 발동하지 않는 상황이지만, LayerNorm 선택 자체는 맞다.

## 5. 손실함수 / 최적화 — 참고자료 §3 대조

- **mean-of-k-losses가 실제로 맞게 구현됨**(`loss_dir_k.mean(dim=1)`, 라인 290 등) — 기존
  "modern DL checklist" 메모리의 "verify mean-of-k-losses" 항목을 코드로 직접 확인, **정상.**
- **AdamW(decoupled weight decay) 사용** — 참고자료 §3이 명시한 정석 선택, 올바름.
- **LR 스케줄/워밍업 없음 — 이건 결함이 아니라 의도된 선택**: TabM 논문 자체가 고정 lr=2e-3,
  스케줄 없음을 쓴다는 게 이미 이 저장소 메모리에 기록돼 있음("don't add one reflexively"). 다만
  Tuning Playbook은 워밍업을 학습초반 불안정의 1순위 해결책으로 꼽으므로, 만약 나중에 학습
  불안정 징후가 생기면 시도해볼 첫 수단으로 남겨둔다.
- **Gradient clipping(norm=2.0, 라인 296)이 이미 존재** — Playbook의 디버깅 체크리스트 2순위
  조치가 사전에 반영돼 있음, 긍정적.
- **quality_loss_weight=0.80, exit_loss_weight=1.15는 근거 주석 없는 상수** — Tuning Playbook
  분류법으로는 이게 "nuisance 하이퍼파라미터"인데, 이 두 값에 대한 HP서치가 이뤄졌다는 증거를
  코드/문서 어디서도 못 찾음. **명시적 공백.**

## 6. 정규화(Regularization)/과적합 — 참고자료 §4 대조, 가장 실행 가능한 발견

- **Label smoothing 전혀 없음** (`cross_entropy` 호출 4곳 전부 `label_smoothing` 인자 없음,
  라인 274/279/285/308-310) — **그런데 이 프로젝트 자체 연구가 이미 GCE(q=0.7)가 이 정확히 같은
  3-head 분류 태스크에서 baseline CE보다 낫다는 걸 검증했다(val bacc 0.5758 vs 0.5740,
  `research_eth_candidate_faithful_tabm_batchensemble_regularizer_isolation_20260816.py`
  결과). 그런데 이 canonical 학습 스크립트엔 그게 전혀 반영이 안 돼 있다.** 이게 이번 평가에서
  가장 구체적이고 실행 가능한 발견 — 이미 검증된 개선을 캐노니컬 스크립트에 아직 안 넣은 상태.
- **EMA 가중치 없음** — checklist 항목, 미적용.
- **Purge/embargo 갭이 실제로 없음(재확인)**: `split = max(int(n*0.85), min(n-1,512))`,
  `train_idx=arange(split)`, `val_idx=arange(split,n)` (라인 239-241) — 인접 인덱스를 그냥
  자름, 갭 0. zigzag_action 라벨이 피벗 확정을 위해 미래 봉을 참조하는 구조라면, 분할 경계
  바로 앞 학습 샘플의 라벨이 검증셋 쪽 정보를 일부 담고 있을 가능성이 이론적으로 있다 — 이게
  실제로 early-stopping 판단(어느 epoch에서 멈출지)에 낙관 편향을 주는지는 미검증이지만,
  참고자료·이 프로젝트의 표준 관행(purge/embargo) 대비 명확한 이탈.
- **Early stopping(patience=8)은 잘 근거된 선택**: 참고자료 d2l 부분의 "early stopping의 이득은
  라벨 노이즈가 있을 때 집중된다"는 주장이 금융 수익률 기반 라벨에 정확히 들어맞음 — 이 부분은
  올바르게 돼 있다.

## 7. 용량 대 데이터 — 이미 나온 실험 결과를 이론으로 재해석 (2026-08-16 정정됨)

**정정**: 이 섹션은 원래 R+S+B 완성형 실패를 arXiv 모노그래프의 memorization/generalization
phase transition(용량-데이터 경계) 프레임으로 설명했는데, 병행 세션의 후속 N≥5시드 최종검증에서
그 설명이 틀렸다는 게 확인됐다 — 자세한 내용은 `feedback_modern_dl_training_checklist` 메모리와
`eth_odyssey4_layer_and_parameter_improvement_proposal_20260816.md`의 C3 정정을 참고. 아래는
정정된 내용이다.

기존에 닫힌 R+S+B 완성형 실험(파라미터 +6.5%)이 N=5시드 전부에서 `direction_balanced_accuracy`를
일관되게 악화시켰다는 결과 자체는 맞다(+0.042~0.058, 전 전문가). 다만 병행 세션이 cosine LR +
Prechelt UP_4 선정기준으로 재측정한 결과, **true-peak 정확도(이상적 상한)는 두 아키텍처가
0.003~0.009 차이로 사실상 동일했다** — 즉 용량이 늘어서 진짜 학습 상한이 나빠진 게 아니다. 실제
격차의 원인은 R+S+B(임베딩 포함)의 학습 초반 손실 곡선이 더 시끄러워서, 어떤 현실적인 조기종료
절차를 써도 진짜 잠재력을 거의 못 끌어내는(80%의 시드가 1에폭만에 조기종료) **학습 신뢰성
문제**였다. 그러니 "용량-데이터 경계 근처라 용량을 더 얹으면 암기로 간다"는 이론적 설명은 이
사례에는 적용되지 않는다.

실전 결론은 바뀌지 않는다(baseline_R_only가 여전히 이긴다, 그러니 그대로 유지) — 다만 **왜
이기는지**가 "용량이 적어서 일반화가 잘 됨"이 아니라 "구조가 단순해서 안정적으로 자기 상한까지
학습됨"이라는 게 정확한 설명이다. 이는 §6의 초기화·정규화배치 항목이 왜 중요한지도 재조명한다 —
만약 학습 안정성(정규화 배치, 초기화, LR 스케줄, 선정기준)을 먼저 고치면 더 복잡한 아키텍처도
자기 상한까지 안정적으로 도달할 수 있을지 모른다는 뜻이기도 하다. "다음에 아키텍처를 더 키우고
싶다면 유효 데이터부터 늘려야 한다"는 원래 결론도 근거가 약해졌다 — 대신 **먼저 학습 신뢰성부터
고쳐야 한다**(정정된 선정기준, cosine LR, 적절한 옵티마이저)는 게 더 정확한 우선순위다.

또한 3개 레짐 전문가(bull/bear/chop)는 별도 모델이지만 학습 데이터를 하드 분할하는 게 아니라
`route_w`(HMM 소프트 확률)로 가중치만 주는 방식(라인 230-234) — 완전한 데이터 분할보다는 덜
심각하지만, 그래도 각 전문가의 유효 표본수는 통합모델보다 작다. 레짐별 실효 표본수를 실제로
측정해본 적은 없는 것으로 보임 — 확인해볼 만한 저비용 진단.

---

## 종합 — 우선순위별 실행 가능 항목

| 우선순위 | 항목 | 근거 | 상태 |
|---|---|---|---|
| 높음 | GCE(q=0.7)를 canonical 스크립트에 반영 | 자체 연구로 이미 검증됨, 아직 미적용 — **정정(08-16): GCE 단독 이식은 부족, 선정기준(Prechelt UP_4)+cosine LR과 묶어야 함(§7, 개선제안 문서 A1 참고)** | 실행 가능, 4종 묶음으로 |
| 중간 | quality/exit loss 가중치(0.80/1.15) HP서치 | 근거 없는 상수, nuisance param | N-HiTS/ModernTCN 작업의 HP서치 인프라 재사용 가능 |
| 중간 | 내부 85/15 split에 purge/embargo 갭 추가 | 프로젝트 표준 관행 이탈 | 라벨 구조(zigzag 피벗의 forward-looking 정도) 먼저 확인 필요 |
| 낮음 | 레짐 전문가별 유효표본수 측정 | 용량-데이터 경계 진단 | 저비용, 그냥 세어보면 됨 |
| 낮음 | 공유 선형층에 명시적 초기화(Xavier/He 중 SiLU에 맞는 스케일) | 실전 영향 불확실 | 근거는 있으나 급하지 않음 |
| 참고 | LR 워밍업 부재 | 의도된 선택(TabM 논문과 일치), 결함 아님 | 불안정 징후 생기면 1순위 시도 |
| 참고 | pre-norm이 아닌 제3의 정규화 배치 | 위험 패턴은 아님, 정석도 아님 | 불안정 징후 생기면 2순위 시도 |

**가장 눈에 띄는 발견**: 이 저장소는 GCE/ELR/mixup을 이미 진짜로 테스트해서 GCE가 이긴다는 걸
알고 있는데, 그 결과가 아직 라이브 학습 스크립트로 안 들어갔다 — 새 아키텍처(N-HiTS/ModernTCN)
연구에 정성을 쏟는 동안, 이미 검증된 개선을 기존 TabM에 반영하는 더 작은 작업이 남아있었다.
