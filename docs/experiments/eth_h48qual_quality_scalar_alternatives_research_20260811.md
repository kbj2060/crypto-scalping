# ETH h48qual — quality_head 0~1 스코어 대안 연구 (2026-08-11, 구현 전 리서치 단계)

**추가 업데이트 (2026-08-11) — 0단계 진단 완료, candidate A·E 둘 다 닫힘**: 이 문서가 최우선으로
제안한 0단계 진단(`quality_for_action` vs realized-outcome 순위상관)을 돌렸다 —
`docs/experiments/eth_h48qual_quality_for_action_rank_correlation_20260811.md`. 결과: h48orig
(5시드)·h384(15시드) 어느 쪽도 신뢰할 만한 양의 순위상관 없음(h48orig는 오히려 음의 경향, 사이징
스코어의 역상관과 같은 방향이나 다중비교 보정 후 비유의). **아래 candidate A(temperature
scaling)는 이걸로 닫힌다** — 지킬 순위 자체가 없어서 스케일 보정이 무의미하다. 별도로,
`quality_head`를 분류에서 회귀로 바꾸는 전환도 실제로 시도됐다 — candidate E로
`docs/experiments/eth_h48qual_quality_head_regression_conversion_attempt_20260811.md`에서 검증
완료: 오라클 메커니즘은 유효하나 FINAL12/REL11(201개 풀 재스크리닝) 어느 쪽도 GBM 홀드아웃에서
실전 신호 없음 — 이것도 닫힘.

**추가 업데이트 (2026-08-11, 같은 날 더 나중) — candidate C도 닫힘**: 서버 GPU로 재학습한 5시드
번들에서 TabM k=8 멤버 불일치(Depeweg MI 분해의 `epistemic`)를 직접 뽑아 같은 순위상관 진단을
돌렸다 — `docs/experiments/eth_h48qual_ensemble_disagreement_rank_correlation_20260811.md`.
VAL/OOS 둘 다 신뢰할 만한 상관 없음(풀링 p=0.51/0.67). "quality_head의 자기 확률과 무관한 별도
신호원이라 앞선 실패를 상속하지 않는다"는 C의 핵심 가설이 기각됐다. 이제 A/C/E 셋 다 부정
결과로 닫혔고, 네 갈래 증거가 전부 "스칼라 추출 방법의 문제가 아니라 라벨/피쳐 자체에 신호가
없다"로 수렴한다. **남은 건 B(evidential, 이미 신중 취급이었는데 이번 결과로 더 회의적)뿐이거나,
`h48_conservative` 라벨/피쳐 조합 자체를 새 데이터소스로 교체하는 것이다.** 사용자는 후자를
선택했다 — 새 데이터소스 후보 리서치는 별도 문서
`docs/experiments/eth_h48qual_quality_new_data_source_research_20260811.md`(직교하는 축이라
candidate B와 달리 "같은 데이터에 다른 방법"이 아니라 "다른 데이터")로 분리했다. 아래 원문은
그대로 두고 각 절 앞에 업데이트 배지만 추가한다.

**구현 전 리서치 단계 문서** — 여기 있는 어떤 방법도 아직 코드로 옮기지 않았다. 재학습 없는 진단부터
검증하고, 그 결과가 나온 뒤에만 재학습이 필요한 방법으로 넘어가는 순서를 권한다.

## 질문

`quality_head`가 실제로 0~1 게이팅 값(`quality_for_action`)이 되는 과정이 지금은 "3-class
softmax를 학습하고, direction이 고른 클래스의 확률 하나만 인덱싱해서 꺼내 쓰는" 방식이다
(`trading_bot_modules/omega4_6_1_live.py:174-178`, `scripts/train_omega1_regime3_routed_expert_
direction_quality_20260602.py:_quality_for_action`). softmax(다중클래스) · sigmoid(직접 스칼라
회귀) 둘 다 아닌, 이 확률 추출 방식 자체를 개선할 더 나은 방법이 있는지 조사한다.

## 그라운딩 — 이미 시도되고 죽은 라인 (레포 감사)

`docs/model_contracts/research_line_registry.json`(21개 라인 전체)에는 확률 추출/캘리브레이션
방법 자체를 다룬 라인이 없다 — 전부 피쳐/라벨/아키텍처 축(JEPA, DVOL, TP-first 라벨 등)이다.
관련 있는 건 인접 스크립트들이다.

**캘리브레이션(Platt/isotonic/temperature) — `quality_head`/h48qual/zig075 자체엔 미시도, 인접
신호엔 시도됐고 대부분 실패/보류**:

- `scripts/train_eval_omega4_6_1_conformal_sizing_20260707.py`: L4 리스크 **사이징** 스코어(게이팅
  아님)에 Platt류 `sigmoid(a+b·z)` 재보정 — VAL에서 기각(52.04% vs 베이스라인 54.88%, MDD도 악화;
  `tmp/causal_regen_20260516/omega4_6_1_conformal_sizing_20260707/result.json`,
  `consistent_improvement: false`). OOS는 더 좋아 보였지만(+160.50% vs +145.34%) 사전등록된
  VAL-선택 규율 때문에 기각. 이 작업을 촉발한 진단 `scripts/diagnose_risk_sidecar_calibration_
  20260707.py`가 h48qual 원시 사이징 스코어의 OOS spearman(score, return) = **-0.406(p=0.036,
  역상관)**을 발견했다 — **`quality_for_action` 자체에는 이 진단이 아직 안 돌아갔다.**
- `scripts/alpha5_router_v5_ablation_20260520.py`: 자매 3-class 라우터에 클래스별 독립
  `IsotonicRegression` 적용 — **완전히 망가짐**: balanced_accuracy가 0.56→0.33으로 붕괴하고
  예측 거래수가 VAL/OOS 둘 다 0으로 떨어졌다. ECE/Brier(캘리브레이션 지표)는 오히려
  개선됐는데도 그렇다(`tmp/causal_regen_20260516/alpha5_router_v5_ablation_20260520/
  router5_ablation_summary.json`). **"캘리브레이션 지표는 좋아지는데 실제 결정은 망가지는"
  구체적 선례.**
- `scripts/calibrate_unified_direction_thresholds.py`: 이름과 달리 확률 재보정이 아니라 결정론적
  threshold 그리드서치(다른 파이프라인, h48qual 아님).
- `quant/live_30m_direction_quant.py:924-937`: temperature scaling이 **다른 파이프라인에서 이미
  라이브로 돌고 있다**(Brier 최적, grid {0.7..2.0}) — 메커니즘 자체가 레포에서 금지된 게 아니라
  quality_head엔 그냥 시도된 적이 없다는 뜻.

**Conformal prediction — h48qual에 직접 시도, 실패, 이미 계약서에서 금지됨**:
`scripts/research_conformal_abstention_eth_h48qual_20260809.py`(APS, Romano 2020)를 hard
abstention 게이트로 시도 — t=0.27/p=0.79로 실패(`docs/entry_exit_edge_root_cause_and_literature_
review_20260809.md:148`). `docs/model_contracts/odyssey_eth_h48qual_corrected_tabm_20260811_
contract.md`의 disagreement-exposure 가드레일이 이걸 다시 반복하지 말라고 이미 명시하고 있다.

**Evidential/Dirichlet deep learning — 이 레포 전체에서 어디서도 시도된 적 없음.** 유일한
"dirichlet" 매치는 무관한 sklearn `BayesianGaussianMixture`(`scripts/retrain_clean_regime_bgmm_
20260517.py:200`)뿐이다. **진짜 백지 영역.**

**Quantile regression — exit에서 시도(패배), entry-quality에서도 시도(실패)**:
exit head에 q10/q50/q90 continuation-value 회귀(`scripts/research_eth_omega461_distributional_
stopping_20260724.py`)는 VAL에서 순수 SLTP를 이겼지만(+70.47%) Stage-1 hazard 후보에 져서
OOS를 열어보지도 못했다. entry-quality 쪽 "quantile-regression forward-return skew"는 17개
실패 항목 중 하나였다(`docs/entry_exit_edge_root_cause_and_literature_review_20260809.md:160`).
**`quality_for_action`을 대체하는 용도로는 시도된 적 없다.**

**앙상블 불일치를 스코어로 — 아직 미구현이지만 이미 계획된 항목**: TabM의 k=8 멤버 분산은
`scripts/train_eval_omega1_2_tabm_3head_20260603.py:355-357`에서 이미 계산되고
(`.mean(dim=1)`) 버려지고 있다. Odyssey 계약의 3-1번 항목이 정확히 이 신호를 노출하자는
제안이다 — 단, L4 사이징 피쳐 후보로만, 게이트로는 절대 안 쓴다는 조건으로.

**메타라벨링(별도 모델) — 여러 번 시도, 대부분 약함. 단, quality_head 자체 설계는 이미
메타라벨링 패턴이다**: `scripts/train_eval_scalp_1m_meta_label_20260716.py`(소폭 열세),
`scripts/run_sigma3_metalabel_20260705.py`(베이스라인 못 이김), BTC v2 계열 0/12,544,
`scripts/build_btc_1h_zigzag_quality_meta_label_20260806.py`(상관 거의 0) — 전부 **별도
메타라벨 모델**로서 실패한 사례다. `quality_head`가 "direction이 고른 클래스가 맞는지 별도로
점수 매기는" 것 자체는 이미 메타라벨링 패턴이고, 이건 실패한 적이 없다 — h48qual은 검증된
필터로 확인돼 있다. 실패한 건 "메타라벨을 위한 새 모델/피쳐 조합"이지 메타라벨링이라는
개념 자체가 아니다.

## 0단계 — 진단 먼저 (재학습 없음, 최우선) — ✅ 완료, 결과: 순상관 없음

`diagnose_risk_sidecar_calibration_20260707.py`가 사이징 스코어에서 한 것과 똑같은 진단을
`quality_for_action` 자체에는 아직 아무도 돌리지 않았다. 저장된 VAL/OOS 예측에서
`spearman(quality_for_action, realized_outcome)`만 뽑아보면 재학습 없이 바로 답이 나온다:

- **역상관이거나 무상관이면**: 캘리브레이션(온도·isotonic·evidential 무엇이든) 문제가 아니라
  랭킹 자체가 깨진 것 — 사이징 스코어가 그랬던 것처럼. 이 경우 스코어 추출 방식을 바꿔도
  소용없고, 라벨/피쳐 문제(계약 문서 미해결 이슈 5, 6, 7)로 되돌아가야 한다.
  h48qual FINAL12는 direction/quality 각자 타겟 기준으로 relevance를 검증했으니 사이징
  스코어보다는 나을 가능성이 있지만, 확인 전엔 가정하지 않는다.
  `scripts/research_eth_omega461_quality_threshold_sweep_20260728.py`가 threshold **값**만
  스윕해서 VAL 승자(0.35)가 OOS에서 뒤집힌 전례(`confirmed=False`)가 있다 — 랭킹이 약하면
  threshold를 아무리 옮겨도 이런 일이 반복된다.
- **순상관이 있는데 스케일이 안 맞으면(과신/과소신)**: 아래 A(temperature scaling)가 바로 맞는
  처방.

**실제 결과(2026-08-11)**: h48orig(5시드)·h384(15시드) 둘 다 신뢰할 만한 양의 순위상관 없음 —
위 첫 번째 분기("역상관이거나 무상관")에 해당. 스코어 추출 방식을 바꿔도 소용없고, 라벨/피쳐
문제(계약 미해결 이슈 5, 6, 7)나 완전히 다른 신호원(C)으로 가야 한다는 뜻. 전체 수치:
`docs/experiments/eth_h48qual_quality_for_action_rank_correlation_20260811.md`.

## 후보 방법

### A. Temperature scaling — 저비용·저위험 보정 (Guo et al. 2017) — ❌ 닫힘 (0단계에서 순상관 없음)

logit을 스칼라 `T`로 나눈 뒤 softmax: `softmax(logits/T)`. **argmax를 절대 안 바꾼다** — 클래스
순위(어느 클래스가 1등인지)는 그대로 두고 확신도 크기만 조정하는 구조라, 0-1 손실 기준
정확도에 영향을 주지 않는다는 게 원논문의 핵심 성질이다. `alpha5_router_v5_ablation_20260520`이
망가진 이유(클래스별 독립 isotonic → 클래스간 정합성 붕괴 → argmax 왜곡)를 구조적으로 피해간다
— 문헌도 이 "클래스별 독립 보정이 다중클래스에서 정합성을 깬다"는 함정을 확인해준다(one-vs-rest
isotonic이 "종종 suboptimal", 클래스별 확률이 합이 1이 안 돼서 재정규화가 필요하고 그 과정에서
문제가 생긴다는 지적).

**왜 지금 관련 있는가**: `quality_threshold=0.50`은 원시(비보정) softmax 확률에 대고 그대로
비교되는 상수다. 최신 신경망은 체계적으로 과신하는 경향이 알려져 있다(Guo et al. 2017) — 만약
`quality_head`도 과신 상태라면 threshold=0.50이 의도한 것보다 더 느슨하게(혹은 레짐에 따라
다르게) 작동하고 있을 수 있다. 이건 레짐별로 다를 가능성이 커서, 레짐별 온도(스칼라 3개)가
리서치 문서 3-4번 항목("레짐별 quality threshold 재보정")이 제안하는 것보다 더 원리적인 해법일
수 있다 — 원시 threshold 값 자체를 레짐별로 다시 스윕하는 것보다, 확률을 먼저 레짐별로 보정한
뒤 threshold는 그대로 두는 쪽이 20260728 스윕처럼 VAL에 과적합해서 OOS서 뒤집힐 위험이 작다.

**싼 검증 순서**: 재학습 불필요. 저장된 VAL logit(있으면)이나 `quality_proba`에 스칼라 `T`
그리드서치로 NLL/ECE 최소화 지점 탐색 → `quality_for_action`의 realized-outcome 순위상관이
바뀌는지(안 바뀌어야 정상, 스케일만 이동) → OOS에서 threshold=0.50 교차 지점이 바뀌어 pnl이
개선되는지 확인. 0단계에서 순상관이 확인된 경우에만 진행.

### B. Evidential Deep Learning / Dirichlet 출력 (Sensoy et al. 2018) — 백지 영역, 재학습 필요

마지막 softmax 레이어를 비음수 evidence 출력(Softplus 등)으로 바꾸고, 이걸로 Dirichlet
분포 파라미터 `α_c = 1 + e_c`를 만든다. 기대 확률 `p̂_c = α_c / S`(`S = Σα_c`)가 `quality_proba`
자리를 대신하는데, 여기에 더해 `S`(Dirichlet 총 강도)에서 바로 "vacuity"(증거 부족도,
`K/S`, K=클래스 수) — 즉 단일 forward pass만으로 나오는 불확실성 신호를 공짜로 얻는다.

**단순 sigmoid 직접회귀(`NLinear(...,1)+sigmoid`, quality_head를 binary "좋은 거래인가" 회귀로
바꾸는 안) 대비 이론적 장점 3가지**:
1. 확률(`p̂_c`)과 증거량(`S`)이 분리된다 — sigmoid는 숫자 하나뿐이라 `p=0.5`가 "진짜 애매함"
   (aleatoric)인지 "본 적 없는 상황이라 모름"(epistemic)인지 구별을 못 한다.
2. CASH/LONG/SHORT 3-class 구조를 유지하면서도 `α_c/S`가 항상 합=1이라 재정규화가 필요 없다.
   Sigmoid로 3-class를 하려면 이진화(클래스 정보 손실)하거나 sigmoid 3개를 따로 놓아야
   하는데(one-vs-rest), 후자는 `alpha5_router_v5_ablation_20260520`을 무너뜨린 것과 같은
   구조(클래스별 독립 처리 → 정합성 붕괴 → 재정규화 필요)를 그대로 반복할 위험이 있다.
3. 앙상블/여러 forward pass 없이 단일 forward pass로 불확실성이 나온다.

**하지만 이 장점들이 실전에서 지켜지는지는 최근 문헌에서 활발히 반박되고 있다** — 원 논문
(Sensoy 2018)이 주장하는 성질이 후속 연구에서 계속 깨진다:
- NeurIPS 논문 "Are Uncertainty Quantification Capabilities of Evidential Deep Learning a
  Mirage?"는 EDL의 epistemic uncertainty가 데이터가 무한히 늘어도 0으로 수렴하지 않는 등
  신뢰할 수 없다고 결론짓는다.
- evidence scale 보정이 어렵고, **far-OOD 입력에서 오히려 과신하는 경향**이 보고된다 — 바로
  위 장점 1·3(증거 부족 = 낮은 확신)의 정반대 실패 사례.
- 표준 KL 정규화 항이 "예측 클래스의 evidence" 자체는 직접 억제하지 않는 구조적 허점이 있어,
  magnitude 기반 과신에 취약하다.
- in-distribution과 out-of-distribution 샘플의 불확실성 분포가 실제로는 많이 겹친다는 관찰도
  있다.

즉 이 방법도 이 레포에서 conformal abstention·isotonic이 겪은 것과 같은 패턴("원 논문에서는
그럴듯한데 후속 검증에서 약점이 계속 발견됨")이 될 위험이 있다. `quality_for_action`을
**대체**하는 목적이라면 실질 이점은 위 장점 2(재정규화 불필요) 정도로 좁혀질 수 있고,
불확실성 신호 자체를 목적으로 한다면 도입 전에 이미 이 프로젝트가 계획 중인 TabM k=8 앙상블
분산(Lakshminarayanan 2017 — 실전 검증이 더 많이 된 방법)과 반드시 직접 비교해야 한다.

**이 레포 맥락에서 주의할 점**: TabM의 k=8 앙상블이 이미 (아직 미구현이지만) 3-1번 항목으로
불일치 기반 epistemic/aleatoric 분해를 제공할 예정이다(Depeweg et al. 2018). Evidential DL의
핵심 장점(앙상블 없이도 불확실성을 얻는다)이 이 프로젝트에선 절반쯤 이미 있는 셈이라, 순수
추가가치는 "앙상블 분산보다 per-example 불확실성이 더 잘 보정돼 있는가"로 좁혀지는데, 위
반박 문헌을 감안하면 그 답이 "그렇다"일 가능성을 낙관하면 안 된다.

**싼 검증 순서**: temperature scaling과 달리 재학습이 필요하다(loss function 자체가 다름 —
표준 CE가 아니라 KL 정규화가 들어간 evidential loss, 게다가 위 문헌상 vanilla 2018 버전은
알려진 결함이 있어 후속 변형(Fisher-information 기반, density-aware 등)을 검토해야 할 수도
있다 — 구현 비용이 처음 생각보다 크다). 그래도 전체 재설계 전에, 단일 시드로 기존
`h48_conservative` 라벨 그대로 evidential-loss quality_head 하나만 학습해서 (a) VAL 캘리브레이션
지표(ECE), (b) `quality_for_action` 자리를 대신할 `p̂_{action}`의 realized-outcome 순위상관을
기존 softmax 버전과 **그리고 3-1번 항목의 앙상블 분산과** 함께 3자 비교 — 이 정도는 전체
파이프라인 통합 전에 싸게 죽이거나 살릴 수 있다.

### C. 앙상블 불일치 기반 스코어 — ❌ 닫힘 (검증 완료, 부정 결과)

3-1번 항목이 이 방향이었다: k=8 멤버 분산에서 뽑은 `epistemic` 신호. A/B와 다른 점은 이건
"quality_for_action이 얼마나 믿을 만한가"를 재는 **보조 신호**로 설계돼 있지, `quality_for_action`
자체를 대체하는 스칼라가 아니라는 것 — 계약서가 이미 이걸 하드 게이트가 아니라 L4 사이징
피쳐 후보로만 못 박아뒀다.

**검증 결과(2026-08-11)**: 서버 GPU로 v2와 동일 5시드를 번들 저장하도록 재학습, `.mean(dim=1)`
풀링 전 k=8 멤버 출력에서 Depeweg MI 분해로 `epistemic`을 뽑아 0단계와 동일한 방법론(게이트 전
`dir_action` 기준 시뮬레이션)으로 `spearmanr(epistemic, trade_return)` 확인. 정합성 체크(seed=
260620의 새 `dir_action`이 기존 v2 저장 예측과 100% 일치)로 파이프라인 검증됨. **VAL/OOS 어느
쪽도 신뢰할 만한 상관 없음**(풀링 rho=-0.039/p=0.505, rho=-0.031/p=0.671; 개별 시드 10개 전부
비유의). "quality_for_action과 다른 신호원이라 A/E의 실패 원인을 공유하지 않는다"는 가설이
기각됐다 — 전체 과정: `docs/experiments/eth_h48qual_ensemble_disagreement_rank_correlation_20260811.md`.

### D. 메타라벨링 전용 캘리브레이션 (Lopez de Prado, EF3M) — 참고, 게이팅 문제엔 안 맞음

원저자가 메타라벨 확률을 베팅 사이즈로 바꿀 때 권장하는 방법은 범용 ML 캘리브레이션이 아니라
베팅 동시성(concurrency) 분포에 가우시안 혼합을 피팅하는 EF3M 알고리즘이다. 이건 "이진
스코어를 얼마나 크게 베팅할지"로 바꾸는 문제에 맞는 도구지, 지금 `quality_head`처럼 "거래를
할지 말지"를 정하는 hard threshold 게이팅엔 자연스럽게 안 맞는다. 3-1/C의 사이징 피쳐가 실제로
쓰이게 되면(계약 미해결 이슈 상 아직 아님) 그때 일반 isotonic/Platt 대신 검토할 후보로만
남겨둔다 — 지금 질문(게이팅용 0~1 값)의 직접 답은 아니다.

## 하지 말아야 할 것

- **클래스별 독립 isotonic regression**: `alpha5_router_v5_ablation_20260520`에서 실증적으로
  붕괴(balanced_accuracy 0.56→0.33, 거래수→0)했고, 다중클래스 isotonic이 one-vs-rest
  정규화 때문에 구조적으로 이런 함정에 빠지기 쉽다는 게 문헌에서도 확인된다. ECE/Brier가
  좋아 보여도 신뢰하지 않는다.
- **Conformal abstention을 hard gate로**: 이미 h48qual에 직접 시도해서 실패했고
  (`research_conformal_abstention_eth_h48qual_20260809.py`), 계약 문서가 이미 금지하고 있다.
- **진단(0단계) 없이 바로 재학습 방법(B)으로 점프**: 이 문서가 인용한 21개 죽은 라인 대부분이
  "진단 없이 풀스케일 재학습"에서 비용을 태운 패턴이었다(`docs/eth_omega4_6_1_accuracy_research_
  ideas_20260811.md` 공통교훈 참고).

## 제안 우선순위 (2026-08-11 갱신 — 0단계·A·C·E 결론 반영)

1. ~~0단계 (진단, 최우선, 재학습 없음)~~ **완료** — 순상관 없음(h48orig 음의 경향 포함).
2. ~~A. Temperature scaling~~ **닫힘** — 0단계에서 지킬 순위 자체가 없어서 스케일 보정이 무의미.
3. ~~E. 직접 회귀 전환~~ **닫힘**(별도 문서) — 오라클 메커니즘은 유효하나 실전 피쳐로 학습 불가.
4. ~~C. 앙상블 불일치~~ **닫힘**(별도 문서) — "다른 신호원이라 앞선 실패를 상속하지 않는다"는
   핵심 가설이 실증적으로 기각됨. VAL/OOS 어느 쪽도 신뢰할 만한 상관 없음.
5. **B. Evidential/Dirichlet (백지 영역, 단일시드 재학습 필요) — 사실상 유일하게 남은 후보,
   그런데도 신중 취급**: 원 논문(Sensoy 2018)의 핵심 주장이 후속 문헌("Is EDL uncertainty a
   mirage?" 등)에서 반박되고 있고, A/C/E가 전부 실패했다는 건 이 라벨/피쳐 조합 자체에 학습
   가능한 신호가 없다는 뜻이라 evidential loss로 재학습해도 없는 신호를 만들어낼 가능성은 낮다.
   착수한다면 반드시 저비용 단일시드 파일럿으로 먼저 죽이거나 살릴 것.
6. **D. EF3M류 사이징 캘리브레이션**: 사이징 피쳐가 실제로 도입될 때까지 보류.

## 결과 (계약 문서 반영용 요약)

`quality_head`의 0~1 게이팅 값(`quality_for_action`)에 대한 대안 방법 조사 완료. 0단계 진단
(h48orig 5시드 + h384 15시드, 재학습 없음) 결과 어느 변형도 신뢰할 만한 양의 순위상관이 없다 —
h48orig는 오히려 음의 경향(사이징 스코어의 역상관 -0.406과 같은 방향, 다중비교 보정 후 비유의).
이걸로 temperature scaling(A)이 닫혔고, 직접 회귀 전환(E)도 실전 피쳐로는 신호가 없어 닫혔고,
TabM 앙상블 불일치(C)도 재학습 검증 결과 VAL/OOS 둘 다 신뢰할 만한 상관이 없어 닫혔다. 클래스별
독립 isotonic과 conformal 기반 hard gate는 이미 죽은 방향이라 배제.
**네 갈래 증거(always-short 대조, 회귀 전환, quality_for_action 순위상관, 앙상블 불일치
순위상관)가 전부 "스칼라 추출 방법이 아니라 라벨/피쳐 자체에 신호가 없다"로 수렴** — 남은 건
B(evidential, 문헌상 회의적)뿐이거나 `h48_conservative` 라벨/피쳐 조합 자체의 교체다.
