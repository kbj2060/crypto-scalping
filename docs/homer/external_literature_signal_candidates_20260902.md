# 호메로스 증거신호 확장 후보 — 외부 문헌 조사 (2026-09-02)

상태: 조사 완료 / 실행 전 (사용자 결정 대기). 진단·설계 문서이며 매매 알고리즘 아님.

## 요약 (결론 먼저)

현재 대시보드 증거신호 8개(`demarker_extreme`, `orthogonal_combo`, `short_term_return_z`,
`taker_delta_z_climax`, `smt_divergence`, `liquidity_sweep`, `kalman_deviation_meanrev`,
`fib_extension_exhaustion`)는 **전부 "가격/오실레이터/플로우가 극단값에 닿았다"는 같은 통계적
기반**을 쓴다. 2025~2026 문헌에서 이 저장소에 실제로 없는 축을 찾는 것이 조사 목표였다.

| 등급 | 후보 | 근거 문헌 | 데이터 | 판단 |
|---|---|---|---|---|
| ~~추천1~~ ❌ | ~~**Lee-Mykland 점프검정**~~ | Lee-Mykland(2008), Boudt et al. | klines만 | **2026-09-02 실행 → REJECTED(중복)** — lift는 통과(1h 2.80~3.17x)했으나 발동의 **78~96%가 `short_term_return_z` ±3봉 내**(독립성 기준선 6~9.5%의 10배). [실험문서](../experiments/eth_lee_mykland_jump_raw_lift_check_20260902.md) |
| ~~추천2~~ ❌ | ~~**VPIN 독성 극단**~~ | *Bitcoin wild moves*(RIBAF 2026) | klines만 | **2026-09-02 무방향 점프크기 라벨로 실행 → REJECTED** — VPIN(볼륨클럭) 1.28~1.43x로 통과선 미달, 대조군 volume(2.53x)·hl_range(2.50x)·|taker_dz|(2.27x)에 전 horizon 열세. ⚠️저장소 기존 `_vpin_approx`는 0.63~0.95x로 역방향. [실험문서](../experiments/eth_vpin_toxicity_jump_magnitude_raw_lift_check_20260902.md) |
| ~~추천3~~ ❌ | ~~**Corwin-Schultz 스프레드 급확대**~~ | 크립토 유동성프록시 비교연구, Frontiers(2026) | klines만 | **2026-09-02 실행 → 보류(진행 안 함)** — 교란우려는 반증(ρ 0.163)되고 축 신규성도 확인(겹침 22~43%)됐으나 lift 약·비대칭(p99 천장 1.76x)이고 **대조군 hl_range(3.40/2.99)에 전 구간 밀림**. [실험문서](../experiments/eth_corwin_schultz_spread_raw_lift_check_20260902.md) |
| 조건부 | 옵션 **스큐/리스크리버설/기간구조** | Grith et al.(Princeton BCF), Deribit RR 스큐 | **과거 데이터 수집 선행 필요** | registry가 **명시적으로 열어둔** 유일한 재개조건 |
| ~~저비용~~ ❌ | ~~라운드넘버 **오더플로우 불균형**~~ | *Buy-sell imbalances...*(FRL 2026) | klines만 | **2026-09-02 실행 → REJECTED(메커니즘 미재현)** — 오프셋 **플라시보 격자가 진짜 격자와 같거나 더 나음**, $50/$100 비대칭 부호도 반대. ⭐부수소득: **2026-08-14 A10의 1.79x가 라운드넘버 효과가 아니었음**(플라시보 2.04x). [실험문서](../experiments/eth_round_number_orderflow_raw_lift_check_20260902.md) |
| ❌ | 비추천 9종 | 아래 §4 | — | 사유별 정리 |

**핵심 논거 하나**: VPIN·Corwin-Schultz·Kyle λ는 이미 154피쳐셋에 구현돼 있고
`eth_dc_feature_engineering_redundancy_combination_finml_20260820`에서 **피쳐로는 실패**
(p=0.460)했다. 그러나 이 저장소 자신의 **DeMarker 선례**가 결정적이다 — `dem` 단독 AUC는
0.51(무의미)인데 **트리거**로 쓰니 HOLDOUT 0.7464(프로젝트 최고)였다. **피쳐 실패 ≠ 트리거
실패**이며, 그 종결라인의 retest_guidance도 "다른 라벨/피쳐 유니버스에 같은 파이프라인을
재사용하는 것은 금지된 재시도가 아니다"라고 명시한다.

---

## 0. 조사 필터 — 왜 대부분의 논문이 탈락하는가

문헌 후보를 이 저장소에 넣으려면 5개 관문을 전부 통과해야 한다. 조사에서 나온 대다수는
1번 또는 4번에서 죽는다.

1. **시간 해상도**: 5분봉 + 수시간 보유(현 라벨 H=8~72봉 = 40분~6h). 초·틱 단위 논문은 탈락.
2. **데이터 가용성**: raw L2 / microstructure_1m은 2026-09-14 / 09-30 게이트로 조기실행 금지
   (`eth_lob_raw_l2_early_peek_ofi_ic_20260824`). 온체인·ETF·스테이블코인은 각각 종결.
3. **중복 아님**: `docs/model_contracts/research_line_registry.json` 54개 종결라인 대조.
4. **횡단면 → 시계열 전이 불가**: 크립토 실현모멘트 논문 다수가 "코인 간 횡단면 주간
   리밸런싱"이다. 이 저장소는 단일자산 시계열이고, 횡단면 활용은
   `evidence_signal_quant_use` 서브프로젝트에서 이미 이벤트당 0.24bp vs 비용 10bp로 종결됐다.
5. **비용 10bp 벽**: 수수료 우대 가정 금지가 이 저장소 규칙
   (`feedback_no_fee_discount_assumptions...`).

---

## 1. 추천 Tier A — 지금 착수 가능

### A-1. Lee-Mykland 점프검정 (⭐최우선)

> **⛔ 2026-09-02 실행 완료 → REJECTED (중복).** 아래 설계·논거는 실행 전 원문 그대로 보존한다.
> 결과: 1h lift 2.80~3.17x로 통과선(1.8x)은 넘었으나 배포중 `short_term_return_z`(2.72~2.90x)와
> 신뢰구간 완전 중첩, 발동의 **78~96%가 그 신호 ±3봉 이내**. 고유 잔여분 n=19/26에 부호 모순,
> 품질필터 용도도 1h·4h에서 side별 부호 불일치. **아래 "주기성 보정" 논거는 반증됨** —
> 보정판이 무보정판보다 거의 모든 셀에서 낮았다(시간대 패턴 자체가 정보). 전문:
> [eth_lee_mykland_jump_raw_lift_check_20260902.md](../experiments/eth_lee_mykland_jump_raw_lift_check_20260902.md).
> 부수 소득으로 **"단기 수익률 극단값의 더 나은 정규화" 계열 전체가 닫혔다.**

**문헌**: Lee & Mykland(2008)의 비모수 점프검정 — 일중 수익률을 **인접 수익률의 강건
스케일 추정치(bipower variation)** 로 표준화한다. Boudt et al.의 일중 주기성 보정판이
24시간 시장에 특히 중요하다. 크립토 적용에서 "일중 수익률 예측 패턴 자체가 **대형 일중
점프의 존재 여부에 따라 달라진다**"는 결과가 반복 보고됐다(Wen-Bouri-Xu-Zhao, *Intraday
return predictability in the cryptocurrency markets*).

**정의(구현안)**:
```
L(t) = r(t) / BV(t),   BV(t) = (1/(K-2)) * Σ |r(i)| * |r(i-1)|   (직전 K봉, 점프에 강건)
r(t)를 Boudt 방식 일중 주기성 f(t)로 나눠 보정 → L*(t) = L(t)/f(t)
임계값: max|L*|의 Gumbel 근사 분포로 α=1% 수준 지정 (발동률이 통계적으로 통제됨)
```

**기존 신호와 뭐가 다른가** — `short_term_return_z`(3봉 수익률 롤링 z ≥±2.5)와 같은 계열로
보이지만 정규화가 근본적으로 다르다:
- `short_term_return_z`의 분모(롤링 표준편차)는 **점프 자신에 오염된다**. 변동성 클러스터링
  구간에서는 큰 움직임이 분모를 키워 z가 억제되고, 조용한 구간에서는 사소한 움직임도 발동한다
  — §5.9)절이 발견한 "저변동성 구간에서 문턱이 거래비용 밑으로 내려가는" 결함과 같은 뿌리다.
- LM의 분모(bipower variation)는 **점프에 강건**해서 이 오염이 구조적으로 없다.
- Gumbel 임계값은 **거짓발동률을 α로 고정**한다. 지금은 발동률이 문턱값의 부산물이다.
- 주기성 보정은 지금 `hour_sin/cos`·세션타이밍 피쳐가 *사후에* 하는 일을 **트리거 단계에서**
  한다(`fib_extension_exhaustion`·`smt_divergence` ablation에서 세션타이밍이 실제 기여로
  확인된 것과 같은 정보를 앞단에서 쓰는 셈).

**저장소 현황**: `lee_mykland` 없음. `enhanced_trading_engine.py:202-205`에 거친 BNS식
`(realized_var - bipower_var)/bipower_var` 비율 플래그가 있으나 임계값 무보정·주기성 무보정의
엔진 내부 상태값이고 증거신호가 아니다. **정식 구현은 부재.**

**첫 스텝**: raw lift 사전점검(`event_study` vs zigzag 피벗, 1h/4h/8h). 통과선은 후보풀 관행상
칼만 2.16/2.36·디마커 1.89/2.12, 탈락선은 VPOC 1.06/1.10·Renko 0.69/0.83.
**반드시 같이 측정할 것**: `short_term_return_z`와의 발동 겹침률(참고: smt_divergence↔
liquidity_sweep 6.0~9.5%가 "충분히 독립적"의 실측 기준).

---

### A-2. VPIN 독성 극단 (order-flow toxicity climax)

> **⛔ 2026-09-02 실행 완료 → REJECTED.** 아래 설계·논거는 실행 전 원문 그대로 보존한다.
> 사용자가 선택한 **무방향 점프크기 라벨**로 타깃을 새로 만들어 검정
> (`|excursion| ≥ K×ATR` within H, intrabar). 라벨 프레이밍 자체는 옳았고(문헌 주장과 정합)
> VPIN도 무정보는 아니다(1h 정밀도 44.0%, CI 41.8~46.2 vs baseline 30.8%). 그러나 **1.28~1.43x로
> 통과선 1.8x 미달**이고 사전등록 대조군 3종(volume 2.53 / hl_range 2.50 / |taker_dz| 2.27)에
> 전 horizon 열세. ⚠️저장소 154피쳐셋의 `_vpin_approx`(48봉 시간클럭)는 **0.63~0.95x로 역방향** —
> 사실상 "조용한 일방향 드리프트"를 재고 있다. ⚠️문헌 반증은 아님(논문은 틱단위, 여기 버킷은 ~29분).
> 전문: [eth_vpin_toxicity_jump_magnitude_raw_lift_check_20260902.md](../experiments/eth_vpin_toxicity_jump_magnitude_raw_lift_check_20260902.md).

**문헌**: *Bitcoin wild moves: Evidence from order flow toxicity and price jumps*
(Research in International Business and Finance, 2026, vol.81) — 고빈도 BTC 데이터 + VAR로
**VPIN이 미래 가격점프를 유의하게 예측**하고, VPIN과 점프크기 모두 양의 자기상관을 보인다
(비대칭정보 지속 + 모멘텀). 크립토에서 VPIN 적용은 아직 희소하다는 것이 논문 자신의 주장이다.

**정의**: 거래량 시계(volume clock)로 버킷을 자르고, 각 버킷의 |매수량−매도량|/총거래량을
n버킷 평균. 이 저장소는 `_vpin_approx(taker_buy, volume, window)`로 이미 구현
(`scripts/eth_dc_financial_ml_feature_construction_20260820.py:111`).

**`taker_delta_z_climax`와 뭐가 다른가**: taker는 **부호 있는** 순매수 z-score(방향 신호),
VPIN은 **부호 없는** 불균형 비율(정보거래자 존재 = 독성). 완전히 다른 양이다.

**⚠️ 설계상 반드시 짚어야 할 제약 두 가지**:
1. **피쳐로는 이미 실패했다**(154피쳐, p=0.460). 위 §요약의 DeMarker 선례가 반박근거지만,
   보장이 아니라 근거 있는 가설일 뿐이다.
2. **VPIN은 방향이 아니라 크기를 예측한다.** 현재 8개 신호의 라벨은 전부 방향성 MFE
   (top/bottom)다. 그대로 얹으면 논문이 주장하지 않은 것을 검정하게 된다. →
   라벨을 `다음 H봉 내 max|이동| ≥ K×ATR`(무방향)로 새로 설계하거나, 방향칩이 아닌
   **"독성/역선택 경고칩"** 으로 배치해야 한다. **이건 사용자 결정 사항이다** — 대시보드가
   재량보조 도구라는 점에서 경고칩 쪽도 실용가치가 있다.

---

### A-3. Corwin-Schultz 스프레드 급확대 (유동성 증발)

> **⛔ 2026-09-02 실행 완료 → 보류(진행 안 함).** 아래 설계·논거는 실행 전 원문 그대로 보존한다.
> ✅사전 교란우려(변동성 재포장)는 **반증**됐다 — CS와 hl_range/ATR의 Spearman이 0.163/0.173,
> 겹침도 22~43%로 Lee-Mykland(78~96%)보다 훨씬 독립적. 문헌 주장대로 진짜 다른 것을 측정한다.
> ❌그러나 **사전등록 대조군(순수 고가-저가 레인지)이 CS·AR은 물론 배포중 두 신호까지 전부
> 능가했다**(1h 3.40/2.99 vs CS 2.20/1.76). CS 설계가 레인지에서 변동성 성분을 빼내 스프레드만
> 남기는 것인데, **그 빼낸 성분이 정보의 대부분**이었다. 경계값 스윕(p98~p999)도 결론 불변.
> 전문: [eth_corwin_schultz_spread_raw_lift_check_20260902.md](../experiments/eth_corwin_schultz_spread_raw_lift_check_20260902.md).
> 유동성 상태축의 다음 관문은 klines 파생이 아니라 **raw L2(게이트 2026-09-14/09-30)** 다.

**문헌**:
- 크립토 유동성 측정 비교연구: **Corwin-Schultz(2012)와 Abdi-Ranaldo(2017)가 시계열 변동
  설명에서 다른 측정치를 능가**한다(관측빈도·거래소·고빈도 벤치마크 무관). 반대로 레벨
  추정에서는 Kyle-Obizhaeva와 Amihud가 낫다.
- *Microstructure alpha*(Frontiers in Blockchain, 2026): 12개 미시구조 피쳐 안정성선택에서
  **CS 스프레드가 3위(0.79)** — 실현변동성(0.84)·5분 모멘텀(0.83) 다음이고 **유동성 계열 중
  1위**. 반면 Amihud(0.51)·Kyle λ(0.56)는 5% 유의 미달.

**왜 새로운 축인가**: 8개 신호는 전부 가격·오실레이터·플로우 극단이다. **유동성 상태(state)축**은
`dalton_rule2_balance_edge` 제거 이후 비어 있다. CS 스프레드가 상위 백분위로 급확대되는 것은
"호가가 얇아졌다 = 같은 주문이 더 크게 밀린다"는 다른 종류의 정보다.

**⚠️ 같은 논문이 강한 반증도 제공한다**: 분단위 연속예측 + 5분 홀드 설계에서는 누수통제 후
LightGBM R² **−10.94%**(랜덤워크보다 나쁨), OLS도 유의하지 않았고, **비용 후 어떤 전략도
생존 못 함**(net Sharpe −52, 회전율 일 204배). → **연속예측으로 쓰면 안 된다.**
드문 이벤트 트리거(상위 1~2% 급확대)로만 쓰고, 그 이벤트 조건부로 TabPFN을 얹는 이
저장소 패러다임이라야 의미가 있다.

**저장소 현황**: `_corwin_schultz_spread` 구현됨(같은 파일 :75). `amihud_illiquidity_z`는 이미
캐노니컬 피쳐라 별도 후보 아님. Abdi-Ranaldo는 미구현(CS 실패시 대안).

---

## 2. Tier B — 조건부 (데이터 획득 선행)

### B-1. 옵션 스큐 / 리스크리버설 / IV 기간구조

**이게 특별한 이유**: registry `btc_dvol_feature_overlay`(DVOL 레벨/변화 오버레이, 0/9 실패로
종결)의 retest_guidance 원문이 이것이다 —
> "Option skew or term-structure data with independently verified availability is a strong
> differentiation."

즉 **저장소가 스스로 명시적으로 열어둔 유일한 재개 경로**다. DVOL *레벨*이 닫힌 것이지
*스큐*는 다른 양이다.

**문헌**: Grith-Almeida-Miftachov-Wang(Princeton BCF), *Option-Implied Risk Premia and
Cryptocurrency Market Regimes* — BTC 옵션 위험중립밀도 클러스터링 결과 **리스크프리미엄이
두 개의 변동성 레짐으로 갈리고**, 저변동성 레짐은 낮은 베이시스·낮은 분산위험프리미엄과
연결된다. Deribit 리스크리버설 스큐는 연도별로 IV와의 관계 부호가 뒤집힌다(2021 vs 2022).

**데이터 문제(이게 병목)**: 자체 collector `collect_deribit_option_gex_20260815.py`가 스트라이크별
`mark_iv`를 저장하지만 **2026-08-15 시작 = 약 18일치**라 TabPFN 학습에 못 쓴다. 과거 옵션체인
확보가 선행 프로젝트다(무료 경로: CryptoDataDownload의 Deribit 옵션 OHLCV CSV. 유료: Amberdata/
Tardis). **착수 우선순위 3위**로 두는 것이 합리적이다.

### B-2. VRP (분산위험프리미엄, DVOL² − 실현분산)

BTC의 연율화 위험중립/실물 월간분산이 0.72 / 0.58, **VRP 0.14로 S&P500(~0.02) 대비 훨씬 크다**.
DVOL 시간봉 히스토리는 `scripts/download_deribit_dvol_20260804.py`로 **이미 받고 있어서**
(`trading_bot_modules/btc_swing_transition_live.py:56` public REST) 즉시 계산 가능하다.
다만 DVOL 레벨축이 BTC에서 0/9로 닫혔으므로 기대치는 낮게 잡을 것 — B-1보다 싸지만 약하다.

---

## 3. 저비용 리파인 — 라운드넘버 오더플로우 불균형

> **⛔ 2026-09-02 실행 완료 → REJECTED (메커니즘 미재현).** 아래 설계·논거는 실행 전 원문 보존.
> 오프셋 **플라시보 격자**(반 칸 이동 = 기하학적으로 동등, 심리적으로 무의미)를 결정 기준으로 삼았고,
> STAGE 1에서 진짜 격자가 플라시보와 **전혀 분리되지 않았다**(플라시보 수열 = 진짜 수열의 반 바퀴 회전,
> 구조 없음의 서명). 논문이 예측한 비대칭 부호도 $50(+0.0007)과 $100(−0.0041)이 서로 반대.
> STAGE 2 트리거도 8종 전부 0.62~1.34x이고 거의 모든 대조에서 플라시보 ≥ 진짜.
> ⭐**부수 소득: 2026-08-14 A10의 1.79x/1.44x는 라운드넘버 효과가 아니었다** — 플라시보 격자가
> 바닥에서 2.04x로 진짜(1.79x)를 이긴다. lift는 `price_roc_48 ≤ −1%` 성분이 전부 만든 것이고
> `near_round`는 장식이었다(원인 귀속 정정). ⚠️논문 반증 아님 — 논문 자신이 "가격수준·거래규모가 낮은
> 코인일수록 뚜렷"이라 명시, $3,200 ETH는 가장 불리한 조건.
> 전문: [eth_round_number_orderflow_raw_lift_check_20260902.md](../experiments/eth_round_number_orderflow_raw_lift_check_20260902.md).

**이미 한 번 측정됐다**: `docs/experiments/eth_deep_evidence_signal_sweep_round2_20260814.md`의
A10 "라운드넘버 접근"(가격이 라운드 $50 레벨 근처 + 그쪽으로 추세) = **바닥 1.79x / 천장 1.44x**.
기각된 게 아니라 상위 8개에 못 들어 채택 안 된 것이다(참고: BTC-ETH 스프레드 바닥 1.70x가
"보류" 판정을 받은 수준).

**문헌이 제안하는 것은 다른 측정이다**: *Buy-sell imbalances on and around round numbers in
cryptocurrencies*(Finance Research Letters, 2026, 18개 코인 고빈도) — 발견은 가격 근접이 아니라
**오더플로우**다. 라운드넘버 **바로 아래에서 비정상적 매수압력**, **바로 위에서 매도압력**
(left-digit·threshold trigger·cluster undercutting 효과). 미국 주식시장 패턴이 크립토에도
존재한다는 것이 논문의 기여이고, 불균형은 **거래규모가 작은 코인일수록 강하다**(ETH엔 다소
불리한 조건부).

**할 일**: `taker_buy_base`가 있으므로 "라운드넘버 근접 × taker 불균형 부호"로 재측정.
비용 1~2시간. 1.79x → 2.0x 이상이면 후보풀 편입, 아니면 이 축을 정식 종결.

---

## 4. 비추천 — 문헌은 실재하나 이 저장소엔 부적합

| 후보 | 문헌 | 왜 안 되는가 |
|---|---|---|
| **Quarter-Hour Effect**(15분 경계 알고리즘 주기성) | arXiv:2607.09426 (2026-07, Binance 무기한 6종) | **10초 해상도**. 총수익 ~0.5bp = 테이커 수수료의 1/10, 왕복의 1/20. 논문 스스로 "단독 방향매매용이 아니라 실행타이밍/유동성공급용"이라고 결론. 5분봉 파이프라인에 얹을 수 없다 |
| **청산캐스케이드 분기비**(branching ratio λ) | arXiv:2608.03616 (2026-08) | **논문 결론 자체가 예측 실패**. 사전상태 λ 상승이 캐스케이드를 구분 못 하고(p=0.062), 증폭도 A∝1/(1−λ) 관계도 기각(검정력 ≥0.96). Hyperliquid 체결로그가 필요한데 사후에만 가용 |
| **Taker 흐름 분산압축 조기경보** | arXiv:2607.27070 (7개 캐스케이드에서 유일하게 생존한 EWS) | **이 저장소가 2026-08-25 이미 REJECTED**. TRAIN 4/4셀 생존에 raw IC −0.34~−0.44로 이례적으로 강해 보였으나, rolling volume과 Spearman **−0.59~−0.83 공선성** 발견 → 통제 후 붕괴(resid IC −0.01~−0.08). "저변동성 사전경보"가 아니라 **"거래량 저조기"의 재포장**이었음. 논문 자신도 "population-level precursor, not a per-event alarm"이라 인정 |
| **Kyle's lambda / Amihud** | Frontiers (2026) | 5% 유의 미달(선택확률 0.51/0.56). `amihud_illiquidity_z`는 이미 캐노니컬 피쳐, `kyle_lambda_48`은 154셋에서 실패 |
| **미시구조 연속예측(LightGBM)** | Frontiers (2026) | 누수통제 후 R² −10.94%, 랜덤워크보다 나쁨. 비용 후 net Sharpe −52. 이 저장소의 `eth_microstructure_panel_1h4h_direction_screen_20260823`("신호는 진짜, 수익화는 실패")과 정확히 같은 패턴의 독립 재현 |
| **횡단면 실현모멘트 / good-bad volatility** | JFQA (2024) Lee & Wang; 크립토 횡단면 semivariance 문헌 | **코인 간 횡단면 주간 리밸런싱** 설계다. 단일자산 5분 시계열로 전이할 근거가 없고, 이 저장소의 횡단면 활용은 `evidence_signal_quant_use`에서 0.24bp vs 10bp로 이미 종결. (단 `realized_semivar_ratio_96`은 154셋에 이미 있고 `realized_skewness`는 캐노니컬 피쳐 — 시계열 트리거화는 A-1/A-3 다음 순위의 잔여 아이디어로만 남긴다) |
| **CEX-DEX / Hyperliquid 펀딩 스프레드** | MDPI *Temporal Dynamics...* (2026, 26개 거래소) | 델타중립 **캐리**이지 방향신호가 아니다. 크로스거래소 펀딩스프레드(F4-C)는 이 저장소에서 N=37 무정보로 이미 측정, 다음 게이트 **2026-10-19** |
| **청산 자석효과 / 청산 히트맵** | 업계 자료 위주, 학술근거 약함 | 이미 별도 청산맵 트랙(v1_spliced 라이브 배포). 방향 A/B 4종·저항 비대칭 등 다수 REJECTED 이력 |
| **Fear & Greed 극단성 프리미엄** | arXiv:2602.07018 (2018~2026, N=2896) | 예측 대상이 **수익률이 아니라 스프레드/유동성**이다. 일봉 해상도. 저자 스스로 "F&G에 내장된 변동성 성분과 결정적으로 분리되지 않는다"고 인정하고, 사전등록 엔드포인트는 다중검정 보정 후 생존 못 함 |

---

## 5. 권고 실행 순서

| 순서 | 항목 | 비용 | 결정 필요? |
|---|---|---|---|
| ~~1~~ | ~~**A-1 Lee-Mykland**~~ | ✅**2026-09-02 완료 → REJECTED(중복)** | — |
| ~~2~~ | ~~**C-1 라운드넘버 오더플로우**~~ | ✅**2026-09-02 완료 → REJECTED(메커니즘 미재현)** | — |
| ~~3~~ | ~~**A-3 Corwin-Schultz**~~ | ✅**2026-09-02 완료 → 보류(진행 안 함)** | — |
| ~~4~~ | ~~**A-2 VPIN**~~ | ✅**2026-09-02 완료(무방향 라벨) → REJECTED** | — |
| 5 | **B-1 옵션 스큐** | 데이터 획득이 별건 프로젝트 | 예 |

1~3번은 GPU 불필요하고 기존 `event_study`/`load_zigzag_pivots` 하네스를 그대로 재사용한다.

## 6. 사전점검 통과 기준 (기존 후보풀 관행 그대로)

- **raw lift ≥ 1.8x** 를 실무 통과선으로 (통과 실적: 칼만 2.16/2.36, 디마커 1.89/2.12 /
  탈락 실적: VPOC 1.06/1.10, Renko 0.69/0.83, TPO 천장 1.08)
- **1h/4h/8h 세 horizon**에서 유지되는지, **양방향 대칭인지**(BTC-ETH·TPO가 비대칭으로 보류됨)
- **기존 8개와 발동 겹침률** (독립성 기준선: smt↔liquidity_sweep 6.0~9.5%)
- 통과한 것만 phase1 진단 → 라벨설계 → TabPFN (GPU를 먼저 태우지 않는다,
  `docs/homer/v_rebound_feeder_signal_protocol.md` 교훈)
- **새 라벨에는 §5.9)절 ATR 하한 점검 필수** — K×ATR 문턱이 저변동성 구간에서 거래비용(10bp)
  밑으로 내려가지 않는지. 짧은 horizon + 낮은 K가 위험군이다
- 경제성게이트를 돌린다면 **§5.8)절 방향뒤집기(direction-flip) 대조군을 그리드 전체에**
  처음부터 포함

---

## 참고 문헌

- [Bitcoin wild moves: Evidence from order flow toxicity and price jumps (RIBAF, 2026)](https://www.sciencedirect.com/science/article/pii/S0275531925004192)
- [The Quarter-Hour Effect: Periodic Algorithmic Trading and Return Predictability in Cryptocurrency Futures (arXiv:2607.09426)](https://arxiv.org/abs/2607.09426)
- [Where does the criticality live? Early-warning signals are event-heterogeneous across seven crypto-perpetual liquidation cascades (arXiv:2607.27070)](https://arxiv.org/abs/2607.27070)
- [Measuring the engine of a liquidation cascade: subcritical branching inside a first-order transition (arXiv:2608.03616)](https://arxiv.org/html/2608.03616)
- [Microstructure alpha: hierarchical learning and cross-asset transfer in cryptocurrency markets (Frontiers in Blockchain, 2026)](https://www.frontiersin.org/journals/blockchain/articles/10.3389/fbloc.2026.1811716/full)
- [Explainable Patterns in Cryptocurrency Microstructure (arXiv:2602.00776)](https://arxiv.org/abs/2602.00776)
- [Variance Decomposition and Cryptocurrency Return Prediction (JFQA, Lee & Wang)](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/variance-decomposition-and-cryptocurrency-return-prediction/9995E58095453CB44A3BC3C9C111969F)
- [Intraday return predictability in the cryptocurrency markets: Momentum, reversal, or both](https://www.sciencedirect.com/science/article/abs/pii/S1062940822000833)
- [Buy-sell imbalances on and around round numbers in cryptocurrencies (FRL, 2026)](https://www.sciencedirect.com/science/article/abs/pii/S106297692600058X)
- [Option-Implied Risk Premia and Cryptocurrency Market Regimes (Princeton BCF)](https://bcf.princeton.edu/published-papers/option-implied-risk-premia-and-cryptocurrency-market-regimes)
- [The Extremity Premium: Sentiment Regimes and Adverse Selection in Cryptocurrency Markets (arXiv:2602.07018)](https://arxiv.org/abs/2602.07018)
- [How to measure the liquidity of cryptocurrency markets? (CS/Abdi-Ranaldo 비교)](https://d-nb.info/1258712520/34)
- [Do higher-order realized moments matter for cryptocurrency returns? (Ahmed & Al Mafrachi)](https://www.sciencedirect.com/science/article/abs/pii/S105905602030294X)
- [DVOL — Deribit Implied Volatility Index](https://insights.deribit.com/exchange-updates/dvol-deribit-implied-volatility-index/)
- [Temporal Dynamics of Market Microstructure in Cryptocurrency Perpetual Futures (MDPI, 2026)](https://www.mdpi.com/2227-7072/14/5/103)
