# ETH h48qual — quality_head 대체 메커니즘 리서치: 메타라벨링/셀렉티브 분류/컨포멀 관점 (2026-08-12, 구현 전 리서치 단계)

**문서 성격**: 이 문서는 순수 리서치/랭킹 문서다. 코드 변경, 학습, 승격 어느 것도 하지 않는다.
아래 "오늘 직접 재확인" 절의 결과 하나만 예외로, 기존에 이미 존재하던(재학습 아닌) 진단
스크립트를 이 문서를 준비하며 직접 실행해 수치를 검증했다 — 나머지는 전부 기존
`docs/experiments/*.md`/`docs/model_contracts/*.md`/코드를 읽고 인용한 것이다.

## 질문

`quality_head`는 구조적으로 이미 **메타라벨링/셀렉티브 분류(selective classification)/거부옵션
분류기(reject-option classifier)** 패턴이다 — 1차 모델(`direction_head`)이 방향을 고르면, 2차
모델(`quality_head`)이 "그 픽을 실행할지"를 별도로 판단한다. 이 세션에서 quality_head의 0~1
게이팅 값(`quality_for_action`)을 더 잘 뽑아내려는 4갈래 시도, 그리고 완전히 다른 레짐+피쳐
조합으로 재설계하려는 시도가 전부 부정 결과로 끝났다(아래 "그라운딩" 절). 이 문서는 "그럼
`quality_head`를 무엇으로 **교체**할 것인가"라는 다음 질문에 대해, 메타라벨링(López de Prado
2018)·셀렉티브 분류(Chow 1970; El-Yaniv & Wiener 2010)·컨포멀 예측(Vovk et al.; Romano et al.
2020)·불확실성 인지 게이팅 문헌을 **이 레포에서 실제로 관측된 실패 패턴**에 맞춰 순위를 매긴
후보 목록으로 답한다.

## 그라운딩 — 이미 확인된 것 (레포 감사 + 이 문서 작성 중 직접 재확인)

### A. `quality_head`가 실제로 하는 일 (코드로 검증)

`trading_bot_modules/omega4_6_1_live.py`를 직접 grep해 확인(2026-08-12):

```python
# line 177-178
qual_for_action = float(quality[action]) if action > 0 else float(quality[0])
final_action = action if (action != 0 and qual_for_action >= self.cfg.quality_threshold) else 0
# line 289-290
"h48qual": _Component(..., quality_threshold=0.50), ...
"zig075": _Component(..., quality_threshold=0.75), ...
```

`quality_head`는 `direction_head`와 마찬가지로 독립적인 3-class(CASH/LONG/SHORT) 분류기이고,
게이팅에 쓰이는 스칼라는 그 헤드의 raw 출력이 아니라 "quality 자신의 분포가 direction이 고른
클래스에 부여한 확률 하나"를 인덱싱한 파생값이다. 이건 López de Prado의 메타라벨링 프레임(1차
모델=방향, 2차 모델=베팅 여부)과 정확히 같은 모양이며, 레포 자체가 이미 이 이름을 붙여뒀다
(`docs/entry_exit_edge_root_cause_and_literature_review_20260809.md` §2.5, 아래 "구조적 원칙"
절 참고).

### B. `quality_for_action` 스칼라 자체를 개선하려던 4갈래 시도 — 전부 종료, 부정 결과

전체 과정은 `docs/experiments/eth_h48qual_quality_scalar_alternatives_research_20260811.md`(이
문서의 직접 선행 문서)에 있다. 이 문서는 그 결과를 전제로 삼는다 — 아래 표만 재확인.

| 후보 | 메커니즘 | 결과 | 근거 문서 |
|---|---|---|---|
| A. Temperature scaling (Guo et al. 2017) | logit을 스칼라 T로 나눠 확신도 크기만 재조정 | **닫힘** — 지킬 순위 자체가 없음 | 0단계 진단, 아래 |
| 0단계. `quality_for_action` vs 실현수익률 순위상관 | `dir_action` 기준 진입 시뮬레이션, spearman | h48orig **음의 경향**(OOS pooled ρ=-0.151, p=0.037, 다중비교 후 비유의); h384 약한 양(OOS ρ=+0.072, p=0.093, 비유의) | `eth_h48qual_quality_for_action_rank_correlation_20260811.md` |
| E. 분류→회귀 전환 | barrier가 이미 계산한 연속값(`tb_long/short_quality`)에 GBM 회귀 | 오라클은 always_short 압도(VAL/OOS 15/15시드, p<0.00001, 메커니즘 유효) — 그러나 실제 GBM 홀드아웃 R²는 FINAL12·REL11(201개 풀 재스크리닝 후) 둘 다 0 근처(대부분 마이너스) | `eth_h48qual_quality_head_regression_conversion_attempt_20260811.md` |
| C. TabM k=8 앙상블 불일치(Depeweg et al. 2018 MI 분해) | 풀링 전 멤버별 출력에서 `epistemic` 추출, 실현수익률과 순위상관 | VAL p=0.505, OOS p=0.671, 개별 시드 10개(5시드×2스플릿) 전부 비유의 | `eth_h48qual_ensemble_disagreement_rank_correlation_20260811.md` |
| B. Evidential/Dirichlet (Sensoy et al. 2018) | **유일하게 미시도** — 백지 영역이지만 후속 문헌("Are Uncertainty Quantification Capabilities of Evidential Deep Learning a Mirage?")이 원 논문 주장을 반박, 신중 취급 | 아래 후보 6에서 계승 | 같은 문서 |

네 갈래 실증 증거(0단계, A, C, E) + always-short 대조(FINAL12+h384 격리검증, OOS 15/15시드
always_short 승, p=0.00015)가 전부 "스칼라 추출 **방법**의 문제가 아니라 `h48_conservative`
배리어 라벨이 현재 확보한 피쳐(FINAL12, 201개 풀로 넓혀도)로는 실현 결과와 관계가 없다"로
수렴한다.

### C. 아키텍처 자체를 바꾸려던 시도 — JM 레짐 + curated 15피쳐(`final15`) — 다중시드+PnL 통제 실패

단일시드(260620)로는 direction_head confidence 격차가 +0.048→+0.0008로 거의 사라지는 인상적인
결과가 나왔으나:

| 검증 | 결과 |
|---|---|
| N=5 다양시드 confidence 격차 | 시드간 std(0.025~0.026) > 평균(0.008~0.016), 부호도 불안정(481003만 음수), 903174는 구버전과 거의 같은 크기(+0.044~0.050) 재현 |
| always-short 대조(단일시드, 5개 threshold×VAL/OOS) | 10/10 always_short 승 |
| 다중시드 PnL(N=5, q050) | model이 always_short을 이긴 칸: 10칸 중 2칸뿐(둘 다 VAL만, 어느 시드도 양쪽 스플릿 다 승 없음) |

근거: `docs/experiments/eth_zig075_final15_multiseed_pnl_validation_20260812.md`. 이건 이
레포의 Seed-Diversity Ensemble Promotion Gate(N≥5 진짜 다양 시드)가 정확히 막으려던 패턴이다.

### D. 인접 메커니즘 — 이미 시도되고 실패 (이 문서가 재제안하지 않을 것들의 근거)

- **Conformal(APS, Romano et al. 2020) 하드 abstention** — `direction_head` 자신의 확률에 직접
  적용해 하드 게이트로 시도(`scripts/research_conformal_abstention_eth_h48qual_20260809.py`).
  그날 밤 17개 아이디어 전체에 적용된 사전등록 falsification 게이트(effect-size t≥2.0,
  permutation≥0.90) 기준으로 **t=0.27, p=0.79 — 결정적 실패**(`docs/entry_exit_edge_root_cause_and_literature_review_20260809.md:148`,
  Part 4 표 1번 항목). 계약 문서의 "disagreement-exposure 신호를 하드 게이트로 쓰면 이미 실패한
  conformal-abstention 라인을 반복하는 것"이라는 가드레일이 이 실험을 가리킨다.
- **클래스별 독립 isotonic regression** — 자매 3-class 라우터(`alpha5_router_v5_ablation_20260520`)에서
  balanced_accuracy 0.56→0.33 붕괴, 거래수 VAL/OOS 둘 다 0. ECE/Brier는 오히려 개선됐는데도
  실제 결정이 무너진 구체적 선례.
- **Quantile regression(entry-quality용)** — "quantile-regression forward-return skew"가 2026-08-09
  17개 아이디어 중 14번으로 시도, 네거티브(`docs/entry_exit_edge_root_cause_and_literature_review_20260809.md:160`).
  Exit head에서도 시도(q10/q50/q90 continuation value)했으나 VAL에서 SLTP는 이겨도 Stage-1
  hazard 후보에 밀려 OOS를 열어보지 못함.
- **별도 메타라벨 모델(quality_head와 다른 모델/피쳐로 새로 학습)** — `scalp_1m_meta_label`(소폭
  열세), `sigma3_metalabel`(베이스라인 못 이김), BTC v2 계열(0/12,544), `btc_1h_zigzag_quality_meta_label`(상관
  거의 0). **주의**: 이건 quality_head라는 패턴 자체의 실패가 아니라 "메타라벨용 새 모델/피쳐
  조합"들의 실패다 — 이 구분은 아래 "구조적 원칙" 절에서 더 다룬다.
- **Portfolio-layer contextual bandit 게이트** (h48qual 내부가 아니라 ETH/SOL/BTC 슬롯 배분
  레이어, 2026-07-09) — 이진 take/skip 액션에 conservative contextual bandit(Contextual CQL,
  IQL, PAC-Bayesian offline bandit 문헌 참고)을 직접 적용한 선례가 **이미 이 레포에 있다**.
  `docs/model_contracts/portfolio_online_bandit_gate_native_20260709.md` +
  `docs/audits/portfolio_online_bandit_gate_native_redteam_20260709.md`: **promotion_pass=False**,
  OOS PnL **-18.03%**, skip rate **96.64%**(거의 전부 스킵으로 붕괴 — CQL류 비관적 추정이
  얇은 데이터에서 흔히 보이는 실패 모드), MDD -18~24%. Validation에서 하이퍼파라미터를 다시
  찾은 변형(`portfolio_online_bandit_gate_param_search_20260709.md`)은 OOS +4.05%/+7.92%로
  나아졌지만 **skip 0건**(OOS에서 사실상 전부 취함) — 이 설정에서 "게이팅"은 거의 아무 일도
  안 하고 있었다. 같은 날 LightGBM 지도학습 랭커 대조군(`portfolio_supervised_ranker_native_20260709.md`)도
  VAL +28.53%였지만 OOS는 +4.05%/+7.92%로 수렴 — 이 레이어의 이진 게이팅 문제 자체가 어떤
  방법으로도 아직 안정적으로 안 풀렸다는 뜻이다. **이건 quality_head가 아니라 다른 레이어(자산간
  슬롯 배분)에서 나온 결과라 직접적인 사망 선고는 아니지만, "이 레포 데이터로 이진 take/skip을
  강화학습/밴딧으로 학습시키면 무너진다"는 무시하기 힘든 선례다.** 아래 후보 7에서 이 선례를
  명시적으로 반영한다.

### E. 오늘 직접 재확인 — 게이트를 완전히 제거하면? (이 문서 작성 중 직접 실행)

이 세션이 한 번도 직접 답하지 않았던 질문("`quality_head` 게이트 없이 `direction_head`의 원본
픽(`dir_action`)만으로 거래하면 always-short/long을 이기는가")을 겨냥해 다른 세션이 작성해둔
`scripts/diagnose_eth_h48qual_ungated_direction_vs_always_short_20260812.py`(미커밋 상태로
남아 있던 것 확인)를 **이 문서를 준비하며 직접 실행**했다(`conda run -n quant_ai python ...` —
dev 환경 기본 python엔 torch가 없어 별도 conda env 필요했음). 라이브 번들 경로가
`tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630`로,
계약 문서가 인용하는 **실제 라이브 번들**(102피쳐, `true_3head_tabm_bundle.pt`)과 정확히 같은
경로임을 확인 — 즉 FINAL12 재현판이 아니라 진짜 라이브 가중치다.

| split | gated pnl | gated trades | ungated pnl | ungated trades | always_short pnl | always_long pnl | ungated beats always_short? |
|---|---:|---:|---:|---:|---:|---:|:---:|
| VAL | +4.51% | 29 | **+10.80%** | 35 | +10.86% | -9.65% | **False**(거의 동률, -0.06pp) |
| OOS | +12.01% | 9 | **-4.76%** | 26 | +13.53% | -14.53% | **False**(큰 격차, -18.3pp) |

**정합성 체크**: `gated_trades`(VAL 29, OOS 9)가 계약 문서 이슈 10 해소 절이 인용한 실제
라이브 가중치 always-short 대조의 거래수("9~29건")와 정확히 일치 — 같은 저장 예측을 정확히
불러왔다는 근거.

**해석**: 게이트를 완전히 제거해도(direction_head 원본을 그대로 거래) VAL은 always-short과
사실상 동률, OOS는 오히려 크게 짐. 즉 `quality_head`가 문제라서 걷어내면 나아지는 게 아니다 —
`direction_head` 자체가 이 VAL/OOS 구간(ETH 약 -32% 하락)에서 always-short 대비 검증된 방향
스킬을 아직 보여주지 못하고 있다. **이 결과는 단일 실행(1회, 5시드 아님)이라 이 레포의
N≥5 시드 표준에는 못 미친다** — 방향성 근거로는 쓸 수 있지만, "확정"으로 취급하면 안 된다.
아래 후보 9는 이 관측을 다음 우선순위 결정의 중심에 둔다.

### 구조적 원칙 — "필터형 vs 재추정형" (레포 자체 이론)

`docs/entry_exit_edge_root_cause_and_literature_review_20260809.md` §2.5가 이 세션이 경험적으로
발견한 것에 이름을 붙인다: **1차 모델(방향)과 2차 모델(베팅 여부)을 분리하면, 2차 모델은 "이미
나온 신호 중 뭘 거를지"라는 좁은 문제만 풀면 되므로 과적합 위험이 구조적으로 작다.** 반대로
BTC 3-way TP-first 메타라벨(경로 결과 확률을 직접 예측하려던 시도, 0/24 실패)이 실패한 이유도
같은 원리로 설명된다 — 그건 "필터링"이 아니라 노이즈 심한 라벨의 **결과 확률을 직접
재추정**하려 한 것이라 2차 모델의 안전한 문제 형태가 아니라 1차 모델과 똑같이 어려운 문제였다.

이 렌즈로 위 4갈래 실패를 다시 보면: **후보 E(분류→회귀 전환)가 실패한 것도 우연이 아니라 이
원칙이 예측한 그대로다** — "direction의 픽이 맞았는가"(필터형, 이산적)에서 "384bar 뒤 연속
수익이 얼마인가"(재추정형, 연속적, 더 많은 비트를 요구)로 문제를 슬쩍 바꿨기 때문이다. 이
원칙을 아래 모든 후보에 명시적으로 적용한다 — **재추정형으로 슬쩍 넘어가는 후보는 그것만으로
이미 경고 대상**이다.

## 후보 목록 (검증 비용 순)

### Tier 0 — 재학습 불필요, 기존 저장 예측 재사용

#### 후보 1. `direction_head` 자체의 네이티브 불확실성 스코어(margin/entropy) 직접 순위상관

**무엇인가**: `quality_head`를 완전히 우회하고, `direction_head`의 3-class softmax 자체에서
나오는 표준 셀렉티브 분류 스코어를 직접 실현수익률과 순위상관 검정한다 — (a) `dir_confidence`
= max(p_cash,p_long,p_short)(Chow's rule의 원래 기준, Chow 1970), (b) margin = 1등 확률 −
2등 확률(다중클래스 셀렉티브 분류·능동학습에서 흔히 쓰는 대안 기준), (c) 분포 엔트로피
`H(p) = -Σ p_c log p_c`(불확실성을 3-class 전체 형태로 재는 기준).

**왜 이미 실패한 4갈래와 다를 수 있는가**: 0단계/A/C/E 전부 `quality_for_action`(quality_head
자신의 확률) 아니면 TabM 멤버 분산을 실현수익률과 대조했다 — **`dir_confidence`/margin/entropy
자체를 실현수익률과 직접 대조한 적은 이 세션에 없다.** Confidence-echo 문서(Test 2)가
`quality_for_action`과 `dir_confidence`의 상관을 쟀지만(h48qual ρ=0.18~0.43 — 중간 정도, 완전
종속 아님), 이건 "quality_for_action이 dir_confidence를 얼마나 베끼는가"이지 "dir_confidence
자체가 수익과 상관있는가"가 아니다. 이건 이번 세션이 수집한 증거 사슬에 남은 마지막 빈틈이다.

**솔직한 기대치**: 낮음~중간. Confidence-echo 문서 Test 4가 이미 "confidence 단독 상위 K개가
실제 게이트 결과를 4.3~6.5pp 이내로 재현한다"는 걸 보였다 — `dir_confidence`가 `quality_for_action`과
**거의 같은 bar 집합**을 고른다는 뜻이고, 그 집합(`quality_for_action` 기준)은 이미 순상관이
없는 것으로 확인됐다. 그래서 이 테스트도 무상관으로 나올 확률이 높다고 본다 — 하지만 (i)
margin/entropy는 max-prob과 100% 동일하지 않고, (ii) 공짜이며, (iii) 음의 결과라도 "혹시 아직
안 써본 스칼라가 있지 않았나"라는 남은 의문을 깔끔하게 닫아서 후보 9(구조적 한계)의 근거를
강화한다.

**검증 비용**: 재학습 불필요 — 기존 예측 CSV에 이미 있는 `dir_p_cash`/`dir_p_long`/`dir_p_short`
컬럼에서 세 스칼라를 계산만 하면 됨.

**첫 테스트**: `scripts/diagnose_eth_h48qual_quality_for_action_rank_correlation_20260811.py`를
그대로 복사해 `quality_for_action` 컬럼만 `dir_confidence`/margin/entropy로 교체, h48orig(5시드)·h384(15시드)
양쪽에서 동일한 `dir_action` 기준 진입 시뮬레이션으로 재실행. 반나절 이내.

#### 후보 2. Trust Score — 피쳐공간 이웃 합치도 (Jiang, Kim, Guan, Gupta 2018, "To Trust Or Not To Trust A Classifier")

**무엇인가**: 모델의 softmax를 전혀 쓰지 않는다. 대신 FINAL12 피쳐공간에서, 각 후보 bar가
"자신이 예측된 클래스의 전형적 영역"에 얼마나 가까운지 대 "다른 클래스 영역"에 얼마나 가까운지
비율(TRAIN 셋의 실제(zigzag) 라벨로 클래스별 밀도 참조, k-최근접이웃 기반)로 신뢰도를 매긴다 —
원 논문의 정의: `trust_score = dist(x, 가장가까운 비-예측클래스) / dist(x, 가장가까운 예측클래스)`.

**왜 이미 실패한 4갈래와 다를 수 있는가**: 이 목록에서 **유일하게 어떤 모델의 확률분포에도
의존하지 않는** 후보다 — direction_head softmax도, quality_head softmax도, TabM 앙상블 분산도
아니다. 4갈래 실패는 전부 "모델이 자기 출력에 대해 뭐라고 말하는가"를 재는 방법들이었다 —
trust score는 순수 기하학(라벨된 이웃들과의 거리)이라 완전히 다른 추정기 계열이다.

**하지만**: FINAL12는 애초에 zigzag_action/h48_conservative에 대한 **분류 MI 관련성**으로
뽑힌 피쳐라, "이 피쳐공간의 기하학이 거래 성공 여부와 관계있다"는 가정 자체가 검증된 적 없다.
더 강력한 학습기(GBM, 201개 풀 포함)조차 이 피쳐공간에서 연속 수익률에 대해 R²≈0이었다는 게
이미 확인돼 있다(후보 E) — trust score가 더 약한(비모수, 국소거리 기반) 추정기이므로 같은
피쳐공간 정보 부족 문제를 다른 방식으로 재확인만 하고 끝날 위험이 실재한다.

**검증 비용**: 재학습 불필요 — TRAIN FINAL12+라벨로 sklearn 기반 kNN 인덱스만 만들면 됨(모델
재학습 없음).

**첫 테스트**: TRAIN 셋으로 클래스별(zigzag_action) kNN 인덱스 구축 → VAL/OOS 각 bar의 trust
score 계산 → 후보 1과 동일한 방법론(순위상관 vs 실현수익률, `dir_action` 기준)으로 검정.

#### 후보 3. 레짐별 quality threshold 재보정 (계약 미해결 이슈 4 / 연구문서 3-4번 항목 계승)

**무엇인가**: 지금은 `quality_threshold`가 전역 상수(h48qual=0.50)다. 저장된 `quality_for_action` +
`regime3_current_sensitive_wide24_{bull,bear,chop}_prob`(또는 라이브의 `_route_id()` 하드
라우팅)로 레짐별 ROC/threshold를 따로 그려서, 전역 0.50이 세 레짐에 고르게 최적인지 확인한다.

**왜 계약이 이걸 아직 열어뒀는가, 그리고 내가 왜 기대치를 낮추는가**: 계약 미해결 이슈 4는
이걸 "아직 반영 안 됨, 다를 가능성은 남아있음"으로 열어뒀다 — **전역** threshold 스윕(0.40~0.80)만
이미 실패했을 뿐(롱 승률 개선 없음, 0.55 이후 악화), 레짐 조건화 자체는 안 해봤기 때문이다.
하지만 이 문서를 준비하며 다른 두 문서를 연결해보면 낙관할 근거가 약하다: confidence-echo
문서가 발견한 핵심은 LONG/SHORT confidence 격차(게이트가 사실상 재현하는 편향의 근원)가
**레짐과 무관하게 학습 시점부터 동일하게 존재**한다는 것이다(학습구간이 오히려 VAL/OOS보다
격차가 크거나 같음). 레짐별로 threshold를 나누는 건 "이 편향이 레짐마다 다르게 나타난다"는
가설을 전제로 하는데, 그 가설은 가장 가까운 관련 진단에서 이미 반증됐다. 그래도 완전히 같은
질문은 아니라서(threshold 재보정 vs confidence 크기 자체) 값싸게 확인할 가치는 있다.

**검증 비용**: 재학습 불필요 — 기존 저장 컬럼 재슬라이스.

**첫 테스트**: VAL/OOS `quality_for_action` + 실현결과를 레짐 3분류로 나눈 뒤, 각 버킷에서
0.40~0.80 threshold 스윕 재실행 — 어느 한 버킷이라도 풀링 결과와 다른 최적점을 보이는지만 확인.

### Tier 0.5 — 재학습 불필요하지만 후보 1의 결과에 조건부

#### 후보 4. Adaptive Conformal Inference under Distribution Shift (Gibbs & Candès 2021)

**무엇인가**: 표준(split) 컨포멀 예측은 캘리브레이션셋과 테스트셋이 교환가능(exchangeable)하다고
가정한다. Gibbs & Candès(2021, "Adaptive Conformal Inference Under Distribution Shift")는 이
가정을 깨고, 워크포워드 진행 중 실제 커버리지 오차를 관측하며 임계값을 온라인으로 갱신한다 —
Barber et al.(2023, "Conformal Prediction Beyond Exchangeability")도 같은 방향의 이론적 근거를
준다.

**왜 이미 죽은 conformal 시도와 다른가**: 이미 실패한 conformal(APS, `research_conformal_abstention_eth_h48qual_20260809.py`)은
**정적** split conformal을 direction_head 확률 위에 하드 SET-abstention으로 씌운 것이다 —
교환가능성을 가정하고, 시간에 따라 적응하지 않는다. Adaptive conformal은 이 레포가 반복적으로
직접 발견한 실패 패턴(temperature scaling이 학습구간에서 적합해도 VAL/OOS로 일반화 안 됨 —
LONG은 세 구간 모두 방향은 일관되지만 크기가 다르고, SHORT는 구간마다 부호 자체가 뒤집힘,
`eth_h48qual_direction_confidence_calibration_20260811.md`)를 **정확히 겨냥**하는 메커니즘이다
— "레짐이 바뀌면 고정된 보정이 깨진다"는 문제 자체를 푸는 도구이지, 다른 종류의 컨포멀 하드
게이트가 아니다.

**정직한 한계**: 컨포멀/adaptive-conformal은 이미 어느 정도 신호가 있는 스코어의 **커버리지를
안정화**할 뿐, 무에서 신호를 만들지 못한다. 지금까지 시도된 모든 스코어(`quality_for_action`,
epistemic disagreement)가 순위상관 자체가 없었다 — 그러니 이 후보는 **후보 1(또는 후보 2)이
어느 스플릿에서든 명목상으로라도(다중비교 보정 전이라도) 양의 순위상관을 보여야만** 의미가
있다. 그 전제가 없으면 "이미 0인 신호의 커버리지를 안정화"하는 것뿐이라 무의미하다.

**검증 비용**: 재학습 불필요(저장된 확률에 적용), 다만 순차 온라인 갱신 로직은 레포에 없어
새로 짜야 함(진단 스크립트 수준, 모델 학습 아님).

**첫 테스트**: **조건부** — 후보 1/2가 먼저 도는 것이 선행 조건. 그중 하나라도 유의미한
방향(양의 상관)이 나오면, 그 스코어에 워크포워드 순서로 임계값을 온라인 갱신하며 VAL→OOS
전체를 흘려보고 고정 threshold 대비 커버리지/PnL 안정성이 개선되는지 확인.

### Tier 1 — 재학습 필요 (기존 피쳐/라벨, 신규 데이터 아님)

#### 후보 5. 필터형을 지키는 메타라벨링 재설계 — binary correct/incorrect (+ 선택: SelectiveNet식 결합학습)

**무엇인가 (5a, 별도 모델)**: `direction_head`의 `dir_action` 픽이 실제로 SL보다 TP에 먼저
닿았는가(라이브 `BASE_TEMPLATE` TP/SL 그대로)를 **이진** 타겟으로 삼아 별도 GBM(LightGBM —
모델 아키텍처 우선순위표 1순위)을 학습한다. `h48_conservative`(direction과 무관하게 독립
정의된 배리어 라벨)가 아니라 **direction의 실제 픽 자체의 정오** 라벨이라는 점이 다르다.

**무엇인가 (5b, 결합학습, 더 비쌈)**: 헤드 shape은 그대로 두고(TabM 트렁크 공유), quality_head의
loss만 독립 barrier CE에서 **coverage-제약 셀렉티브 리스크** 목적함수로 바꾼다 — SelectiveNet
(Geifman & El-Yaniv 2019, "SelectiveNet: A Deep Neural Network with an Integrated Reject
Option", ICML) 또는 Deep Gamblers(Liu et al. 2019, "Deep Gamblers: Learning to Abstain with
Portfolio Theory", NeurIPS)식으로, "선택된 부분집합의 리스크"를 direction_head와 **함께**
학습한다.

**왜 후보 E(분류→회귀 전환)와 다르며 성공 가능성이 있는가**: 필터형/재추정형 원칙으로 보면
E는 "384bar 뒤 연속 수익"(재추정형, 어려움)을 예측하려 했다 — R²≈0으로 실패. 이 후보는
"direction이 방금 고른 이 픽이 맞았나/틀렸나"(필터형, 이산)를 묻는다 — quality_head가 원래
하려던 일에 훨씬 가깝고, 정확히 이 논문 원칙이 "안전하다"고 예측하는 문제 형태다. **아직 아무도
이 특정 타겟(direction의 자기 픽 정오)으로 quality_head를 다시 학습한 적이 없다** — 지금까지는
전부 독립적으로 정의된 `h48_conservative` 라벨이었다.

**정직한 한계**: 이건 "새 피쳐-결과 관계"가 아니라 "새 라벨 정의"다. 이진 정오 라벨도 결국
같은 384bar/48bar 배리어 메커닉(SL-priority TP/SL 경주)에서 파생되므로, 201개 풀 재스크리닝이
이미 답한 "이 피쳐 우주에 384bar 앞선 결과에 대한 신호가 있는가"라는 질문의 변주에 가깝다.
확실히 다른 미탐사 구석이긴 하지만, 낙관할 근거는 "필터형이라 원칙적으로 더 쉽다"는 이론적
논거뿐이지 새로운 데이터 근거는 아니다.

**검증 비용**: 재학습 필요하지만 저렴(5a는 GBM 단일 적합, 몇 시간). 5b(TabM 결합 재학습)는
5a가 최소한의 신호도 못 보이면 시도하지 않는다.

**첫 테스트**: 기존 저장 TRAIN 예측(`dir_action` + 384bar/48bar 배리어 실현 결과, 이미 계산돼
있음, 새 라벨 생성 불필요)으로 이진 타겟 정의 → FINAL12(또는 이 타겟 기준 새 MI 스크리닝)로
LightGBM 단일시드 적합 → VAL holdout AUC/Brier 확인. 유의미한 AUC(0.5 근처 이상)가 안 나오면
5b로 넘어가지 않는다.

#### 후보 6. Evidential Deep Learning / Dirichlet quality_head (기존 candidate B 계승)

**무엇인가**: `eth_h48qual_quality_scalar_alternatives_research_20260811.md`의 candidate B를
그대로 계승한다 — Sensoy et al. 2018식 비음수 evidence 출력 → Dirichlet `α_c=1+e_c` →
`quality_proba` 대체 + 단일 forward pass로 vacuity(`K/S`) 무료 획득. 그 문서가 이미 매우
꼼꼼하게 분석했다(원 논문 대비 실전 장점 3가지, 후속 문헌의 반박 4가지, 신중 취급 결론).

**내가 추가하는 것 — 왜 우선순위를 더 낮춰야 하는가**: evidential DL의 vacuity와 TabM 앙상블
불일치(Depeweg MI 분해의 `epistemic`)는 **서로 다른 추정 메커니즘으로 같은 잠재량**(모델
자신의 인식적 불확실성)을 재는 두 가지 방법이다. 앙상블 불일치는 **이미 이 정확한 잠재량을
실증 검증했고 실패했다**(VAL p=0.505, OOS p=0.671). 두 추정기가 "어느 지점이 불확실한가"에
대해 실질적으로 다르게 판단할 특별한 이유가 없는 한(이 레포에 그런 근거는 없음), 사전확률은
"둘 다 같은 이유로 실패한다"에 더 가깝다. 기존 문서의 "신중 취급" 결론은 문헌 비판만
근거였는데, 이 논거를 더하면 신중 취급을 넘어 **이 목록에서 낮은 우선순위**로 두는 게 맞다 —
기존 결론을 뒤집는 게 아니라 다른 각도로 보강한다.

**검증 비용**: 재학습 필요, loss function 자체가 다름(표준 CE 아닌 KL 정규화 evidential loss).

**첫 테스트**: 기존 문서가 제안한 것 그대로 — 단일시드 파일럿으로 `h48_conservative` 라벨
그대로 evidential-loss quality_head 하나만 학습, ECE + `quality_for_action` 자리를 대신할
`p̂_action`의 순위상관을 기존 softmax 버전 **및** 이미 측정된 앙상블 불일치 epistemic과
3자 비교.

#### 후보 7. 게이트를 오프폴리시 가치추정/contextual bandit 문제로 재구성

**무엇인가**: take/skip을 contextual bandit으로 재정의 — 보상은 `BASE_TEMPLATE` 기준 실현
PnL. 첫 단계(학습 없음)는 기존 로그(`state, action, realized_return`)에 대한 **오프폴리시
가치추정**(doubly-robust, Dudík, Langford, Li 2011)만으로 "현재 게이트 vs 완전 언게이트 vs
confidence 상위-K vs always-short/long" 몇 개 정책의 기대가치를 추정하는 것 — 새 모델 학습이
전혀 아니다. Decision-focused learning(Elmachtoub & Grigas 2022, "Smart 'Predict, then
Optimize'")도 같은 방향(라벨 재추정이 아니라 최종 의사결정 보상을 직접 최적화)의 이론적 근거다.

**반드시 먼저 밝혀야 할 선례 (위 그라운딩 D절)**: 이 레포는 이미 **정확히 이 메커니즘**을
다른 레이어(ETH/SOL/BTC 슬롯 배분, `portfolio_online_bandit_gate_native_20260709`)에서
시도했다 — **promotion_pass=False**, OOS PnL -18.03%, **skip rate 96.64%**(거의 전부
스킵으로 붕괴). 이건 conservative Q-learning류가 얇은 "take" 팔 데이터에서 흔히 보이는
비관적 붕괴 패턴이다. 그 레이어는 OOS 결정 수가 1,668건(56건 실제 거래)으로 h48qual 게이트의
스플릿당 9~35건보다 **훨씬 데이터가 많았는데도** 이랬다 — h48qual 레이어에 그대로 적용하면
같은 문제가 더 심하게 재현될 것으로 예상하는 게 합리적이다.

**그럼에도 포함하는 이유**: (a) 다른 레이어/다른 보상구조라 완전한 사망 선고는 아니고, (b)
관측된 실패 모드(비관적 붕괴)는 알려진, 회피 가능한 구현 선택의 결과다(CQL류 비관 항 없이
plain doubly-robust 추정부터 시작하면 붕괴 위험이 줄 것), (c) 첫 단계(오프폴리시 **평가**,
학습 아님)는 사실상 공짜라서 "현재 게이트가 단순한 대안들 대비 가치추정 관점에서 합리적인
동네에 있는가"를 값싸게 스팟체크할 수 있다.

**검증 비용**: 오프폴리시 **평가** 단계는 재학습 불필요/저비용(기존 로그에 대한 산술). 실제
정책 **학습** 단계는 needs-retrain이며, 위 선례를 감안하면 고비용/저기대치로 분류한다 — 평가
단계에서 뚜렷한 격차가 안 보이면 학습 단계로 넘어가지 않는다.

**첫 테스트**: 기존 저장 `(dir_action/final_action, quality_for_action, realized_return)`으로
{현재 게이트, 완전 언게이트, confidence 상위-K 몇 개, always-short} 각각의 doubly-robust
오프폴리시 가치를 산출 — 새 모델 없이 순수 계산.

### Tier 2 — 신규 데이터 필요 (이미 확인된 인프라 벽과 동일)

#### 후보 8. 신규 데이터소스를 quality_head 학습 피쳐가 아니라 "게이팅 전용" 피쳐로 재프레이밍

**무엇인가**: `eth_h48qual_quality_new_data_source_research_20260811.md`가 이미 조사한 8개
후보(마이크로구조 toxicity/청산 캐스케이드/Polymarket/펀딩+basis/Deribit/온체인/라벨재구성/VPIN)를
"quality_head 학습 라벨을 대체하는 피쳐"가 아니라 "direction_head 출력과 무관하게 직접
게이팅/사이징에 쓰는 독립 상태 변수"로 다시 프레이밍한다 — 후보 3(레짐 조건화)을 레짐이 아닌
다른 외생 변수로 일반화한 것.

**정직하게 확인할 것 — 재프레이밍이 인프라 벽을 없애지 않는다**: 가장 유망했던 3개(마이크로구조·청산·Polymarket)는
라이브 duckdb 커버리지가 2026-05-03부터뿐(Polymarket은 4월 9일치)이라 **VAL(2025-10~12)/OOS(2026-01~02)와
전혀 안 겹친다** — "학습 피쳐로 쓰나 게이팅 피쳐로 쓰나" 이건 **사용 방식**의 문제가 아니라
**데이터 존재 여부**의 문제라 재프레이밍으로 우회되지 않는다. 온체인은 이미 닫힘(`CapMVRVCur`
corr(price)=0.95~0.97, 심각한 가격추세 오염). 펀딩-basis도 닫힘(라벨변형간 부호 불일치).
Deribit은 과거 시점 옵션체인 조회 API 자체가 없음.

**검증 비용/첫 테스트**: 이건 새로운 저비용 테스트가 아니다 — 계약 문서가 이미 "더 큰
작업(새 구간 causal inference)으로 재분류, 아직 미착수"라고 적어둔 바로 그 작업이 필요하다.
이 후보를 목록에 넣는 이유는 재프레이밍이 지름길이 아니라는 걸 명시적으로 확인해두기
위함이지, 단기 착수를 권하기 위함이 아니다.

## 교체가 아니라 범위 재정의 — 구조적 한계 인정 옵션

#### 후보 9. `quality_head`를 "포지티브 셀렉션 필터"로 다루는 걸 멈추고, `direction_head` 자체의 방향 엣지 존재를 서브프로젝트의 선결 질문으로 재정의

**논리**: 메타라벨링/셀렉티브 분류는 구조상 1차 모델이 이미 고른 것의 **부분집합**만 고를 수
있다(정밀도↑ 재현율↓의 트레이드오프) — 1차 모델에 없는 방향 스킬을 만들어낼 수 없다.
`direction_head`의 출력/그와 연관된 피쳐공간 위에서 작동하는 모든 방법(temperature scaling,
회귀 전환, 순위상관, 앙상블 불일치, JM-레짐+curated 피쳐, conformal 하드게이트)이 전부
실패했다. 그리고 오늘 직접 확인한 것: **게이트를 완전히 제거한 `direction_head` 원본조차
VAL/OOS 어느 쪽에서도 always-short를 확실히 이기지 못한다**(VAL 거의 동률, OOS -18.3pp 격패,
단일 실행). 이건 "quality_head가 나쁜 필터"라는 문제보다 한 단계 더 근본적이다 — **direction_head
자신이 이 하락장 구간에서 always-short(방향성 베타) 대비 검증된 스킬을 아직 보인 적이 없다.**
스킬이 없는 1차 신호 위에 아무리 정교한 게이팅/메타라벨링을 얹어도, 구조적으로 만들어낼 수
없는 걸 만들어낼 순 없다.

**"인정한다"가 구체적으로 뜻하는 것**:
1. `quality_head`-모양(quality_for_action을 더 잘 뽑는) 스칼라 추출 후보에 더 이상 엔지니어링
   자원을 배분하지 않는다 — 이건 이미 4/4 실패 후 이 세션이 실제로 택한 방향(새 데이터소스
   축)과 일치한다.
2. 지금 당장 결정이 필요하다면, `quality_head`를 "포지티브 셀렉션 필터"가 아니라 **거래빈도/분산
   완충재**로만 취급한다 — 게이트 통과율이 라이브 가중치 기준 0.7~2.5%(9~29건/스플릿)로
   극히 낮다는 게 이미 확인돼 있다. 다만 "거래를 줄이면 MDD가 준다"는 것도 아직 이 서브
   프로젝트에서 직접 검증된 적 없는 **가설**이다 — 주장하려면 별도의 MDD/tail-risk 전용
   검증이 필요하다. (참고: `docs/subagents/model_architect.md`의 "앙상블은 단순평균보다
   single-owner sleeve, priority, veto/gate 구조를 우선한다"는 원칙도 게이트를 완전히
   걷어내기보다 veto로 남기는 쪽을 지지한다.)
3. "이 하락장 구간에서 `direction_head`가 always-short/regime-beta 대비 진짜 방향 스킬을
   갖는가"를 **이 서브프로젝트가 이미 다른 모든 질문에 적용해온 것과 동일한 잣대**(N≥5 진짜
   다양 시드, always-short/short-only 대조, held-out 유의성)로 정식 검증하는 것을, "quality_head
   개선"과는 별개의, 더 크고 선행하는 리서치 질문으로 승격한다.

이건 "포기"가 아니라 이 프로젝트가 이미 반복적으로 증명한 규율(추세를 스킬로 착각하지 않기)을
한 단계 위 레이어(게이팅이 아니라 방향 자체)로 옮기는 것이다 — 증거가 계속 그쪽을 가리키고
있다.

## 하지 말아야 할 것 (이 문서가 재제안하지 않는 것)

- **Direction_head/quality_head 확률에 대한 conformal(APS) 하드 abstention** — 이미 직접
  시도, 결정적 실패(t=0.27, p=0.79). 후보 4는 이것과 메커니즘이 다르다(정적 vs 적응형)는 점을
  본문에서 명시했다.
- **클래스별 독립 isotonic regression** — 자매 라우터에서 실증적으로 붕괴(0.56→0.33, 거래수→0).
- **전역 reject-option threshold 재튜닝(Chow's rule 그대로)** — 0.40~0.80 스윕이 이미 실패,
  후보 3/4는 "레짐/시간에 따라 적응하는 임계값"이라 다른 문제이지만 순수 전역 재튜닝은
  재제안하지 않는다.
- **Quantile regression을 entry-quality에 재적용** — 17개 아이디어 중 14번으로 이미 네거티브.
- **quality_head와 별개로 완전히 새 모델/피쳐 조합의 메타라벨 모델을 처음부터 다시 만드는 것** —
  scalp_1m/sigma3/BTC v2/btc_1h 네 번 모두 약함/실패. 후보 5는 이것과 달리 **직접 direction의
  픽 정오를 타겟으로 삼는다**는 점에서 구분되지만, "그냥 다른 알고리즘을 FINAL12/201풀에 다시
  돌려본다"는 재제안하지 않는다.
- **Portfolio bandit 선례를 확인하지 않고 바로 h48qual 레이어에 RL/bandit을 학습시키는 것** —
  더 많은 데이터를 가진 레이어에서도 붕괴(skip rate 96.64%)했다. 후보 7의 오프폴리시 평가
  단계를 반드시 먼저 거친다.

## 제안 우선순위 요약

| 순위 | 후보 | 검증 비용 | 솔직한 기대치 | 첫 테스트 |
|---:|---|---|---|---|
| 1 | 1. `dir_confidence`/margin/entropy 직접 순위상관 | 재학습 불필요 (공짜) | 낮음~중간 — 그래도 남은 빈틈을 닫음 | 기존 rank-corr 스크립트 컬럼 교체 |
| 2 | 2. Trust score(피쳐공간 이웃) | 재학습 불필요 (kNN만) | 낮음~중간 — 유일하게 다른 추정기 계열 | TRAIN kNN 인덱스 + 순위상관 |
| 3 | 3. 레짐별 threshold 재보정 | 재학습 불필요 | 낮음 — 관련 진단이 레짐 무관성을 시사 | 레짐 3분류 재슬라이스 threshold 스윕 |
| 4 | 4. Adaptive conformal(분포 shift) | 재학습 불필요(조건부) | 1/2 성공에 완전히 종속 | 1/2가 신호를 보일 때만 착수 |
| 5 | 5. 필터형 메타라벨(binary correct/incorrect) | 재학습 필요(저렴) | 중간 — 이론적으로 가장 "안전한" 문제 형태 | GBM 단일시드 파일럿 + VAL AUC |
| 6 | 6. Evidential/Dirichlet(candidate B) | 재학습 필요 | 낮음 — 앙상블 불일치와 같은 잠재량 추정 | 단일시드 파일럿 + 3자 비교 |
| 7 | 7. 오프폴리시/bandit 재구성 | 평가는 공짜, 학습은 비쌈 | 낮음 — 포트폴리오 레이어에서 이미 붕괴 선례 | doubly-robust 오프폴리시 가치추정 |
| 8 | 8. 신규 데이터 재프레이밍 | 신규 데이터 필요 (같은 인프라 벽) | 판정 불가 — 벽이 안 없어짐 | 없음 (단기 착수 비권장) |
| — | 9. 구조적 한계 인정 / 범위 재정의 | 해당 없음 | — | direction_head 자체를 같은 잣대로 정식 검증 |

## 결론

가장 값싼 세 테스트(1, 2, 3)는 전부 재학습이 필요 없고 이 세션의 기존 검증된 방법론을 그대로
재사용한다 — 우선 이 셋부터 돌려서 "quality_head가 아니라 애초에 아무 스칼라도 이 정보로는 안
된다"는 가설을 완전히 닫는 게 다음 걸음으로 합리적이다. 하지만 이 문서에서 가장 중요한 발견은
개별 후보 순위가 아니라 **오늘 직접 재확인한 사실**이다 — 게이트를 완전히 없앤 `direction_head`
원본조차 이 VAL/OOS 구간에서 always-short를 확실히 이기지 못한다(단일 실행 기준). 후보 1~8
전부가 결국 `direction_head`의 출력이나 그와 연관된 피쳐공간 위에서 "더 나은 부분집합"을
고르려는 시도이고, 메타라벨링은 구조상 부분집합 선택 이상을 할 수 없다. 그래서 이 문서가
실제로 권하는 순서는: **(1)+(2)+(3)을 병렬로 값싸게 돌려 스칼라 추출의 마지막 빈틈을 닫는
동시에, (9)가 요구하는 "`direction_head` 자체가 always-short 대비 진짜 스킬을 갖는가"를 이
프로젝트의 표준 잣대(N≥5 다양 시드, always-short 대조)로 정식 검증하는 작업을 착수한다.** 후자가
음성으로 나오면, 후보 5~8 중 어느 것도 착수할 근거가 약해진다 — 메타라벨링이 아무리 정교해도
1차 모델에 없는 스킬을 만들 순 없기 때문이다.
