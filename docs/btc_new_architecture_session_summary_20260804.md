# BTC 새 아키텍쳐 탐색 세션 종합 정리 (2026-08-04)

이 문서는 2026-08-04 하루 동안 진행한 "BTC 새 아키텍쳐 재설계" 세션 전체를 정리한다.
세부 결과는 각 스크립트/메모리에 남아있고, 여기서는 전체 흐름과 최종 결론만 압축한다.

관련 설계 문서: `docs/btc_panel_crossasset_architecture_design_20260804.md` (Rho1 상세 설계, 이제 CLOSED 상태)

> **교정 공지**: 이 문서의 최초 Stage 2 평가는 미래 score 분포를 사용한 threshold, 레버리지
> 비용 단위 불일치, 중첩 포지션, trade-level MDD를 bar-level MDD로 표기한 문제가 있었다.
> 아래 `0A` 절의 교정 재검증이 기존 Stage 2 PnL 수치와 방법론 주장을 대체한다.

---

## 0A. 평가 아키텍쳐 교정 + Rho2 재검증 (2026-08-04)

### 발견된 문제와 수정

| 문제 | 수정 |
|---|---|
| VAL+OOS 전체 score로 percentile threshold 계산 | 2026-01~03 CAL에서만 threshold/config 선택, 2026-04~07 TEST에 고정 적용 |
| `ret - 3×cost`로 수익에는 leverage 미적용 | `notional = margin_fraction × leverage = 0.30 × 3 = 0.90`, 가격수익과 비용 모두 동일 notional 적용 |
| 5분마다 신규 포지션을 열어 수천 개 거래가 중첩 | 한 번에 한 포지션만 허용, 보유 중 신호는 skip |
| trade PnL 누적을 `bar_level_mdd`로 표기 | 보유 중 close mark-to-market을 포함한 bar별 equity로 MDD 계산 |
| split 직전 row의 미래 label이 다음 split을 침범 | horizon 48bar purge 적용 |
| 입력은 `i-1`까지인데 target은 `close[i]` 기준 | bar `i`까지 관측 → `i+1` 시가 진입 → `i+48` 종가 target으로 통일 |
| pooled validation으로 BTC checkpoint 선택 | 학습은 60심볼 pooled, checkpoint는 BTC-only validation으로 선택 |
| `0.0833`을 무작위 baseline으로 잘못 표현 | 동일 split BTC TRAIN 평균 상수 baseline을 직접 계산 |
| DVOL hourly close를 candle timestamp부터 사용 | 문서화가 불명확하므로 `available_at = timestamp + 1h` 적용 |
| 고정 현재 유동성 top-60을 매 bar universe처럼 사용 | Rho2는 다운로드된 60개 안에서 trailing 30일 quote-volume top-40을 매 bar 재구성 |

공통 평가 엔진은 `core/causal_futures_backtest.py`, 회귀 테스트는
`test/test_causal_futures_backtest.py`, `test/test_btc_dvol_availability.py`에 둔다.

### 교정된 시간 순서

```
TRAIN (<2025-09-01)
  └─ 60심볼 pooled 학습
       ↓
MODEL VAL (2025-09~12)
  └─ BTC-only checkpoint 선택
       ↓
CAL (2026-01~03)
  └─ TP/SL + tail threshold 6개 중 1개 선택
       ↓ frozen config
TEST (2026-04~07)
  └─ 단 한 설정만 causal single-position account backtest
```

TEST 구간은 과거 Rho1 Stage 1 pinball 분석에서 이미 열람했으므로 이번 결과는 research evidence이며
pristine holdout 또는 promotion 근거는 아니다.

### Rho1 교정 재검증

- 학습 learning rate `3e-4 → 1e-4`, BTC-only validation checkpoint 선택.
- 분위수 head best BTC validation pinball: `0.25062` (epoch 8).
- rank head BTC 상수 baseline MSE: `0.08177`.
- rank head best BTC validation MSE: `0.07624` (약 6.8% 개선).
- 즉 기존의 "rank head가 상수와 동일"이라는 진단은 pooled checkpoint 선택 때문에 틀렸다.
- 그러나 CAL 6개 설정은 전부 음수. CAL best도 PnL `-25.49%`, 평균 `-0.1687%/trade`.
- CAL best를 고정한 TEST: **PnL `-32.67%`, bar-level MDD `-34.58%`, 262 trades,
  평균 `-0.1488%/trade`**.
- 리포트: `tmp/rho1_corrected_causal_20260804/report.json`.

### Rho2 — 실제 동시점 cross-symbol attention

Rho1은 여러 심볼 샘플을 공유 encoder로 학습했을 뿐, 추론 시 다른 심볼을 직접 보지 않았다.
이를 분리 검증하기 위해 Rho2를 구현했다.

```
BTC 최근 96×5m feature ── temporal Transformer ──┐
                                                  ├─ fused BTC state
시점 i top-40 심볼 feature ─ cross-symbol attention┘
                                                  ├─ absolute direction head
                                                  ├─ BTC cross-sectional rank auxiliary head
                                                  └─ monotone return-quantile head
```

- 스크립트: `scripts/train_eval_rho2_crosssymbol_causal_20260804.py`.
- absolute direction head를 진입 score로 사용한다. BTC가 횡단면 상위여도 시장 전체가 하락할 수
  있으므로 rank만으로 long/short를 결정하지 않는다.
- best MODEL VAL: direction BCE `0.69221`, rank MSE `0.08354`, pinball `0.002122`.
- CAL 6개 설정 전부 음수. CAL best: PnL `-25.01%`, 평균 `-0.0556%/trade`.
- 고정 TEST: **PnL `-39.06%`, bar-level MDD `-42.10%`, 408 trades,
  평균 `-0.1197%/trade`**.
- 리포트: `tmp/rho2_crosssymbol_causal_20260804/report.json`.

### Rho2 — ETH Tau1 1시간 피처 + Deribit DVOL 재검증

- BTC temporal encoder를 ETH Tau1의 38개 1시간 피처 계약으로 교체하고 `dvol_btc`를 추가해
  **39개 피처**로 만들었다. ETH 전용 외부자산 피처는 BTC에서는 ETH 수익률과 BTC-ETH 상대강도로
  대칭 변환했다.
- 1시간봉은 해당 시간 전체를 포함하므로 모든 Tau1/DVOL feature를 `timestamp + 1h`부터만
  5분 의사결정에 `merge_asof(backward)`로 제공했다. DVOL도 Deribit candle timestamp +1h 공개
  가정을 동일하게 적용했다.
- best MODEL VAL(epoch 2): loss `0.77783`, direction BCE `0.69233`, rank MSE `0.08331`,
  pinball `0.002192`.
- CAL의 6개 설정은 전부 음수이며, least-bad 설정(wide, tail 95/5)을 고정했다:
  CAL PnL `-33.86%`, bar-level MDD `-35.79%`, 227 trades.
- 고정 TEST: **PnL `-20.22%`, bar-level MDD `-22.02%`, 232 trades,
  평균 `-0.0950%/trade`**. 기존 Rho2 TEST보다 손실 폭은 작지만 수익성은 여전히 음수다.
- 리포트: `tmp/rho2_tau1_dvol_causal_20260804/report.json`.

### BTC-110 — causalfix + Regime3 + DVOL + 온체인 multi-branch 재검증

- 입력 계약은 **110개**: causalfix 원천 market-state 94개, Regime3 CURRENT 출력 4개,
  Deribit DVOL 6개, CoinMetrics 온체인 6개다. `regime3_pred_*`는 fail-fast으로 금지했다.
- 각 그룹을 32차원 branch encoder로 따로 인코딩한 뒤 late fusion하여 direction과 monotone
  return-quantile head를 예측한다. DVOL은 +1h, 온체인은 +1d 공개시점 규칙을 유지한다.
- best MODEL VAL: loss `0.73272`, direction BCE `0.69029`, pinball `0.002122`.
- CAL best(wide, tail 95/5): PnL `-36.53%`, bar-level MDD `-39.61%`, 449 trades.
- 고정 TEST: **PnL `-56.49%`, bar-level MDD `-57.09%`, 653 trades,
  평균 `-0.1257%/trade`**. 따라서 feature 축을 합쳐도 수익성은 없으며 promotion 불가다.
- 리포트: `tmp/btc_110branch_causal_20260804/report.json`.

### BTC-110 Event-TB — causal CUSUM + 3-class triple barrier 재검증

- 고정 48-bar forward-return 라벨을 폐기하고, close log-return의 causal CUSUM 이벤트만
  후보로 삼았다. 지그재그 pivot은 미래 확인 위험 때문에 학습/진입에서 사용하지 않았다.
- 이벤트 `t`에서 `t+1` 시가에 진입한다고 가정하고, 대칭 방향 장벽
  `max(0.6%, 1.2×ATR%)`과 48-bar vertical barrier로 `FLAT/SHORT/LONG` 라벨을 만들었다.
  상·하단을 같은 봉에서 동시에 건드리면 보수적으로 FLAT이다.
- 실행 청산은 동일 TP와 `max(0.4%, 0.8×ATR%)` SL을 적용했다. CAL에서 CUSUM
  multiplier(1.5/2.0/2.5)와 score threshold만 선택했다.
- CAL best: multiplier `2.0`, score threshold `0.20`, PnL `-15.88%`, 104 trades.
- 고정 TEST: **PnL `-23.61%`, bar-level MDD `-24.22%`, 220 trades,
  평균 `-0.1215%/trade`**. 이전 110 feature fixed-horizon 모델보다 손실은 줄었지만
  CAL과 TEST 모두 음수이므로 promotion 불가다.
- 리포트: `tmp/btc110_cusum_tb_causal_20260804/report.json`.

### BTC-110 Expectancy — calibrated dual TP-first head (CAL gate 실패)

- 3-class action 확률을 직접 진입 점수로 쓰던 문제를 분리하기 위해 long/short 각각의
  `TP-before-SL` 확률 head를 만들고, validation에서만 isotonic 보정을 맞췄다.
- 진입 점수는 `P(TP)×TP − (1−P(TP))×SL − round-trip cost`의 long/short 기대수익 중 큰
  방향으로 정의했다. CAL 선택 gate는 PnL 양수, 30 trade 이상, bootstrap 95% 하한 양수다.
- CUSUM 1.5/2.0/2.5와 expectancy floor 0~0.15%의 CAL 후보 12개는 **모두 음수**였다.
  최선도 CUSUM 2.5에서 PnL `-75.66%`, bootstrap 하한 `-0.1378%/trade`였다.
- 따라서 gate를 통과한 후보가 없어 TEST는 실행하지 않았다. 결과를 양수로 만들기 위해
  TEST를 선택/튜닝에 쓰지 않은 것이 핵심이다.
- 리포트: `tmp/btc110_expectancy_causal_20260804/report.json`.

### 교정 후 결론

1. 초기 평가의 `-0.42%~-0.48%/trade` 밀집은 레버리지 회계 버그의 영향을 크게 받았으므로
   모델 실패의 고유한 모양으로 해석하면 안 된다.
2. 올바른 BTC checkpoint 선택으로 Rho1 rank MSE는 상수 baseline을 이겼다. 하지만 rank 개선은
   절대 방향 또는 비용 초과 PnL로 이어지지 않았다.
3. Rho1 pooled transfer, Rho2 동시점 cross-symbol attention, Tau1+DVOL temporal contract,
   causalfix+Regime3+DVOL+온체인 110 feature model 모두 교정된 CAL과 TEST에서 음수다.
4. 따라서 **현재 다운로드 universe + 5분 OHLCV/funding/OI feature로 만든 패널 방향 모델은
   research NO-GO**다.
5. 현재 top-60 목록에는 2026-08-04 이전 폐지 심볼이 없으므로 survivorship bias는 완전히 해결되지
   않았다. 다만 두 모델의 손실 폭이 커서 이 한계가 promotion 결론을 뒤집을 근거는 없다.
6. 다음 BTC 실험은 같은 데이터 위의 모델/threshold 재튜닝이 아니라, 아직 사용하지 않은
   마이크로구조·청산 또는 point-in-time 옵션 스큐/기간구조 데이터가 준비된 뒤 시작한다.

---

## 0. 출발점

기존 BTC 리서치(causalfix_final 114~118col 프레임 위의 LightGBM quality-classifier 계열)가
2026-08-04 세션 초반에 이미 종합적으로 막다른 길로 확인됐다
(`project-btc-cusum-architecture-structural-redesign-closed-20260804`):
- 이벤트 게이트(CUSUM/zigzag/DC/dense) 무관 — 전부 0/9x 계열 실패
- 모델 패밀리(LightGBM/RandomForest/ExtraTrees/MLP) 무관 — 전부 0/9 실패
- 원인은 모델 성능이 아니라 **유효 표본 부족** + **causal 피처셋 자체의 신호 부재**로 추정

사용자가 "기존 아키텍쳐는 버리고 새로 설계해보자"고 요청 → 새 아키텍쳐 "Rho1" 설계.

---

## 1. Rho1 — 횡단면 패널 재설계 (CLOSED)

**핵심 아이디어**: BTC 혼자 학습하는 대신 유동성 상위 60개 USDT 무기한선물 심볼을 하나의
학습셋으로 풀링해서 유효 표본을 40~60배 늘리자는 것 (H2 가설). 별도로 H1 가설(횡단면 피처를
BTC 단일자산 모델에 컬럼으로 추가)도 같이 검증.

| 단계 | 내용 | 결과 |
|---|---|---|
| Stage 0 | 60개 심볼 5분봉/일별 metrics/월별 funding 다운로드 (2024-01-01~현재) | 완료, 커버리지 100%, 58,560개 원시파일 해시 등록 |
| Stage 0.5 (H1) | 횡단면 피처 24개(breadth/dispersion/funding·OI 순위)를 기존 LightGBM에 컬럼 추가 | **NO-GO**, OOS 대부분 악화, VAL+OOS 동시양수 0/9 |
| Stage 1 (H2) | 60개 심볼 풀링 트랜스포머(심볼 임베딩) 사전학습, 분포(quantile) 헤드 | BTC-only 대비 pinball -0.66%, EWMA 대비 -2.84%, bootstrap 유의 (단일 스플릿) |
| Stage 1 롤링검증 | 8개 겹치는 4개월 윈도우(2025-09~2026-08)로 재생 | **8/8 유의, 부호반전 0** — event gate 다음으로 이 프로젝트 두 번째로 이 기준 통과 |
| Stage 2 | 횡단면 순위 헤드 학습 + Fresh-Forward bar-by-bar 백테스트(CLAUDE.md 규칙) | 순위 헤드 val MSE가 무작위 기준선과 거의 동일(0.08319 vs 0.0833), 예측 score 사실상 상수(0.49~0.56) |
| Stage 2 백테스트 | TP/SL 2종 × 순위 백분위 임계값 3종 = 6개 설정 | **전부 VAL·OOS 동시 음수**, mean_net -0.42%~-0.48%/trade |

**결론**: Stage 1의 "8/8 롤링 안정"은 진짜였지만 **순수 분포 캘리브레이션**(pinball loss)이었을
뿐 방향 정보가 아니었다. Stage 2에서 방향을 추출하려던 순위 헤드가 사실상 아무것도 배우지
못했고, 실전 PnL 시뮬레이션은 전부 실패. **event gate와 정확히 같은 실패 모양**(rolling 안정성
통과 → 수익화 전부 실패)이라 CLOSED.

세부: `project-btc-rho1-panel-stage0-stage05-20260804.md`

---

## 2. Deribit DVOL — 옵션 내재변동성 축 (CLOSED)

**핵심 아이디어**: 지금까지 쓴 데이터는 전부 spot/perp 시장 정보였다. **완전히 독립된 시장(옵션)**이
매기는 변동성 가격(DVOL, VIX 스타일 30일 선행 내재변동성)을 넣어보자는 것.

- **데이터 확보**: BTC/ETH DVOL 시간봉, 2024-01-01~현재, **결측 0건**, 무료 공개 API
  - 페이지네이션 버그 발견+수정: Deribit API의 `continuation`이 forward cursor가 아니라
    구간 **끝에서부터 거꾸로** 채워주는 방식 — 순진하게 forward로 취급하면 에러 없이 최근
    42일치만 받고 조용히 멈춤. `end_timestamp`를 거꾸로 걸어가도록 고쳐서 해결.
  - 옵션 스큐/기간구조(전체 옵션체인)는 무료로 재구성 불가 — 유료 벤더(Tardis.dev/Amberdata)
    필요, 보류.
- **저비용 반증 테스트**: 피처 7개(레벨, BTC-ETH 스프레드, 30일 백분위, 24h/168h 변화율,
  vol risk premium) 추가 → **0/9**, 그런데 top-20 중요도에 7개 진입(패널 축의 6개보다 많음)
  — 모델이 무시한 게 아니라 적극적으로 썼는데도 OOS 8개 비교셀 중 6개 악화.

**결론**: 완전히 독립적인 새 시장 데이터조차 같은 벽에 부딪혔다. 이걸로 **세 번째 독립 데이터
축**(모델 패밀리 무관 → 횡단면 패널 → 옵션시장 데이터)이 동일한 패턴을 보였다는 점이
중요하다 — "빠진 데이터가 있다"는 가설의 신빙성이 낮아지고, "라벨링/평가 아키텍쳐 자체가
문제"라는 가설로 무게중심이 옮겨감.

세부: `project-btc-deribit-dvol-data-acquired-20260804.md`

---

## 3. 라벨링 패러다임 재검토 (완료, 재확인만 됨)

DVOL마저 실패한 뒤, "새 데이터"가 아니라 **triple-barrier quality-classifier 라벨링/평가 방식
자체**에 숨은 편향이 있는지 두 가지를 점검했다 (`scripts/diagnose_btc_labeling_paradigm_20260804.py`).

### H_selection — "argmax 선택 편향" 가설: 기각 (더 나쁜 소식)
long_q/short_q 중 예측값이 높은 쪽을 고르는 방식이 "승자의 저주"처럼 진짜 신호를 왜곡할
거라 예상했는데, 실제로는 그보다 근본적이었다:

- 예측 품질과 실현 gross 수익률의 **Spearman 순위상관이 사실상 0**
  (long: 0.0016, short: -0.0303 — 오히려 약한 역상관)
- quintile로 쪼개봐도 단조 관계 없음(long), 혹은 약하게 반대 방향(short: 예측이 가장 낮은
  구간이 실현수익 +0.053%로 제일 높고, 가장 높은 구간이 -0.002%로 제일 낮음)

즉 "괜찮은 신호를 선택 메커니즘이 낭비한다"가 아니라 **"애초에 순위 매길 신호가 없다"**는 걸
threshold sweep보다 더 엄격한 방식(quintile 분석)으로 한 번 더 확인한 셈이다.

### H_cost — 학습/평가 비용 불일치: 실재하지만 결론을 못 뒤집음
- 학습 라벨: `long_q = long_ret - 0.07% - (SL 청산시 0.3%)`
- 평가(백테스트): `net = long_ret - 0.42%` (청산 이유 무관하게 고정)
- 두 값 사이에 **일관되게 0.15~0.25%p/trade의 회계 불일치**가 있음 (진짜 방법론 이슈, 이 세션의
  모든 백테스트 스크립트가 이 관례를 그대로 썼음)
- 다만 표본이 충분한 threshold(n=361)에서는 가벼운 비용을 적용해도 이미 음수(-0.163%) →
  이 불일치가 결론을 뒤집지는 못함. 표본이 너무 얇은(n=15) threshold에서만 gross 양수(+0.348%)가
  가벼운 비용 하에서도 양수(+0.178%)로 남았지만, 신뢰할 표본 크기가 아님.

**결론**: 숨은 편향을 찾아 신호를 되살리려 했지만, 오히려 "신호가 없다"는 게 이전보다 더
엄격한 기준(quintile, Spearman)으로 재확인됐다. 다만 비용 회계 불일치는 향후 세션의 백테스트
스크립트를 쓸 때 반드시 고쳐야 할 실제 버그로 남겨둔다.

---

## 4. 종합 결론

이번 세션에서 causalfix_final(98~118col) 프레임 + triple-barrier quality-classifier
아키텍쳐는 다음 축을 전부 소진했다:

1. 이벤트 게이트/라벨 기하학 (CUSUM/zigzag/DC/dense) — 0/9x
2. 모델 패밀리 (LightGBM/RF/ExtraTrees/MLP/트랜스포머) — 0/9~0/10
3. 횡단면 패널 데이터(60개 코인 풀링) — 방향/PnL 0/9, 0/6
4. 독립적 옵션시장 데이터(DVOL) — 0/9
5. 선택 메커니즘/비용 회계 진단 — 편향 없음, 신호 부재 재확인

**이 조합(causalfix_final 프레임 + triple-barrier quality-classifier)과 현재 다운로드 universe의
패널 방향 모델은 research NO-GO다.** 다만 "무엇을 더 시도해도 안 된다"는 보편 명제가 아니라,
교정된 평가에서 검증한 입력·목적함수·실행 계약의 범위로 결론을 제한한다. 같은 데이터 위에서
모델이나 threshold만 바꾸는 시도는 더 이상 하지 않는 게 맞다.

### 아직 안 닫힌 문 (다음에 열어볼 수 있는 것)
- **마이크로구조/청산 데이터**: 2026년 9~10월경 축적 완료 예정 (기존 시간 게이트)
- **온체인(블록체인) 데이터**: Glassnode/CryptoQuant — 이번에 "진짜 온체인 vs Binance
  파생상품 지표"를 구분만 해뒀고, 무료 티어의 실제 가용성(히스토리 깊이)은 아직 확인 안 함
- **옵션 스큐/기간구조**: DVOL(단일 지수)은 실패했지만, 스큐·기간구조는 유료 벤더가 있어야
  하는 별개 질문이라 아직 검증 안 됨
- **라벨/아키텍쳐 패러다임 자체를 바꾸는 것**: quality-regression이 아닌 완전히 다른 목적함수
  (예: 분포 예측만 하고 방향은 아예 포기, 혹은 이 프로젝트가 아직 안 써본 강화학습 재구성 등) —
  다만 이건 "새로운 데이터 없이 같은 프레임 위에서 다르게 접근"하는 것이라 우선순위는 낮음

이 프로젝트의 기존 방침([[feedback-btc-keep-iterating-despite-failures-20260802]])대로 BTC
자체를 완전히 닫는 건 아니다 — 이번에 소진된 건 "causalfix_final + triple-barrier
classifier"라는 **특정 조합**이지, BTC 리서치 전체가 아니다.

---

## 5. BTC-110 Path Utility 라벨 — CAL gate 실패 (2026-08-05)

- 기존 CUSUM·zigzag·고정 horizon·triple-barrier의 first-touch 방향 라벨을 재사용하지 않고,
  진입 다음 봉부터 24시간(288개 5분봉)의 경로를 이용하는 3-class 라벨을 만들었다.
  - `long_utility = long_MFE - 1.25 × long_MAE - 0.14%`
  - `short_utility = short_MFE - 1.25 × short_MAE - 0.14%`
  - 한쪽 utility가 `0.20%` 이상이고 반대편보다 클 때만 LONG/SHORT, 나머지는 FLAT.
- 라벨의 미래 MFE/MAE는 supervised target 구성에만 사용했다. 실시간 추론·진입에는 110개
  causal feature만 사용하고, `t+1 open`에 들어가 고정 TP `1.20%`, SL `0.80%`, 최대 24시간으로
  fresh-forward bar-by-bar 실행했다.
- 아키텍처는 label 효과를 분리하기 위해 `94→64` market branch와 `16→32` context branch,
  `96→64` residual fusion, `64→3 [FLAT/SHORT/LONG]`으로 고정했다.
- VAL early stopping 후 CAL에서 score threshold `0.05/0.10/0.15/0.20`만 비교했다. 4개 모두
  음수(PnL `-63.84%`, `-63.08%`, `-61.72%`, `-61.67%`)였고, 최고 threshold `0.20`도 675 trades,
  win rate `39.11%`, 평균 `-0.1382%/trade`, bar-level MDD `-62.63%`였다.
- 따라서 CAL 양수 및 30 trades 이상 gate를 통과한 후보가 없어 TEST는 실행하지 않았다. 이
  결과는 이전에 본 TEST로 라벨·임계값을 선택하지 않았으며, promotion 근거가 아니다.
- 구현: `scripts/train_eval_btc110_path_utility_causal_20260805.py`; 리포트:
  `tmp/btc110_path_utility_causal_20260805/report.json`.

## 6. BTC-110 Temporal Causal TCN — CAL gate 실패 (2026-08-05)

- Path Utility 라벨·24시간 보유·TP/SL·split을 바로 위 실험과 완전히 고정하고, 단일시점
  MLP만 시간 모델로 교체했다. 입력은 현재를 끝으로 하는 24개 5분봉 × 110 causal feature다.
- market 94개는 `94→48` projection 후 kernel 3, dilation `1/2/4`의 left-padding-only causal
  TCN 세 block으로 처리했다. 현재 16개 context는 별도 `16→32` branch이고, `80→64` residual
  fusion 뒤 `FLAT/SHORT/LONG` 3-class head를 사용했다. 미래 입력 변경이 과거 TCN output을
  바꾸지 않는 unit test도 통과했다.
- VAL CE는 epoch 1의 `1.14064`가 최선이고 이후 `1.36527 → 1.55677 → 1.75249`로 악화되어,
  temporal backbone이 더 빨리 과적합했다.
- CAL score threshold `0.05/0.10/0.15/0.20`은 모두 음수였다. 최선인 `0.10`도 PnL
  **`-52.81%`**, 669 trades, win rate `40.81%`, 평균 `-0.1083%/trade`, bar-level MDD
  `-56.02%`로, 직전 MLP의 최선 `-61.67%`보다 덜 나빴을 뿐 양수 gate에 한참 못 미쳤다.
- 통과 후보가 없어 TEST는 실행하지 않았다. 구현:
  `scripts/train_eval_btc110_temporal_tcn_causal_20260805.py`; 리포트:
  `tmp/btc110_temporal_tcn_causal_20260805/report.json`.

## 7. Tau1 분리형 라벨 계약 확정 (2026-08-05)

이후 BTC 후보는 하나의 방향 classifier가 아니라, 독립적인 5분 tactical Leg A와 1시간
trend-continuation Leg B로 구성한다. 두 레그의 미래 경로는 **학습 target**을 만드는 데만
사용하며, 진입 시점에는 확정된 과거 feature와 causal trend scan만 사용한다.

| 항목 | Leg A (5분 tactical) | Leg B (1시간 continuation) |
|---|---|---|
| 진입 | decision 뒤 다음 5분봉 open | 완료된 1시간봉 뒤 다음 5분봉 open |
| 후보 방향 | long/short 양측 경로 비교 | 과거 1시간 log-price 회귀의 beta 부호 |
| 방향 gate | 없음 | 3/6/12/24/36/48h max-|t|, 이전 720h의 60분위 이상 (최소 168h) |
| 최대 보유 | 8시간 | `min(288시간, 4 × 선택 윈도우)` |
| 청산 | TP `max(2%, 8×ATR14)`, SL `max(1%, 4×ATR14)` | hard stop `4×1h ATR`; +`4×ATR` 이후 `8×ATR` trailing giveback |
| 양성 라벨 | 비용 차감 후 우세 방향 순수익 >= 0.40%, 반대편 대비 >= 0.50%p | 후보 방향 비용 차감 순수익 >= 1.00% |
| 비용 | 왕복 0.14% | 왕복 0.14% |

Leg B 확장 계약으로 재생성한 라벨은 8,781개 중 LONG 1,512개(17.22%), SHORT 1,458개
(16.60%), FLAT 5,811개(66.18%)다. 양성 라벨 순수익 중앙값은 4.45%, 보유시간 중앙값은
192시간이다. 이 숫자는 target의 분포 설명일 뿐, 모델 성과나 live PnL 근거가 아니다.

구현과 unit test:
`scripts/build_btc_leg_a_tactical_labels_20260805.py`,
`scripts/build_btc_tau1_continuation_labels_20260805.py`,
`test/test_btc_tau1_continuation_labels.py`.

## 8. Tau1 분리형 모델 선택 계약 (학습 전 고정, 2026-08-05)

### 시간 분할과 purge

| 역할 | 기간 | 용도 |
|---|---|---|
| train | 데이터 시작 ~ 2025-08-31 | 가중 cross-entropy 학습 |
| checkpoint validation | 2025-09-01 ~ 2025-10-31 | epoch/early stopping만 선택 |
| calibration | 2025-11-01 ~ 2025-12-31 | 진입 확률 임계값만 선택 |
| OOS | 2026-01-01 ~ 2026-03-31 | 고정 후보의 단 한 번 fresh-forward 평가 |

각 split의 끝에는 해당 Leg의 최대 target horizon(각각 8시간, 288시간)을 purge한다. sequence는
그 시점과 이전 row만 사용한다. DVOL은 공개 시각에서 +1시간, on-chain은 공개 시각에서 +1일
지연시켜 forward-fill하며, 발표 전 값을 사용하면 즉시 실패한다.

### 공통 학습 규칙

- 입력: `99 causalfix + DVOL 6 + on-chain 6 = 111`개 causal feature와 별도 Regime3의 24개
  causal input. Regime3 예측값/미래 state는 입력 금지다.
- target: 각 레그의 확정 3-class label. label 생성의 미래 price path는 loss 계산에만 쓰고 model
  input 또는 fresh-forward decision에 조인하지 않는다.
- loss: train split에서만 계산한 class count의 inverse-square-root weight를 평균 1이 되도록
  정규화한 weighted cross-entropy. validation/calibration/OOS의 class 비율은 weight에 쓰지 않는다.
- checkpoint: 최대 40 epoch, checkpoint-validation weighted CE 최저값, patience 5. 동일 loss면
  더 이른 epoch을 채택한다. CAL/OOS 수익률로 epoch·레이어·loss를 선택하지 않는다.

### 실행 후보 선택

- Leg A: `max(P(LONG), P(SHORT))`가 {0.45, 0.50, 0.55, 0.60, 0.65} 이상이고, 선택 방향 확률이
  `P(FLAT)`보다 {0.00, 0.05, 0.10} 이상 큰 후보만 만든다.
- Leg B: causal trend gate를 먼저 통과하고 gate 방향의 확률이
  {0.45, 0.50, 0.55, 0.60, 0.65} 이상인 후보만 만든다. 반대 방향으로 뒤집지 않는다.
- 각 레그는 calibration fresh-forward에서 순수익 양수, Leg A 50건 이상 또는 Leg B 20건 이상,
  그리고 trade PnL bootstrap 95% 하한이 0보다 큰 후보만 적격이다. 적격 후보 중 평균 순수익이
  가장 높은 하나를 고정한다. 적격 후보가 없으면 그 레그는 OOS에서 비활성화한다.
- OOS는 위에서 고정한 checkpoint·임계값만 사용하며, bar-by-bar 순차 실행한다. 저장 ledger,
  미래 label, 과거 exit timestamp는 입력으로 사용하지 않는다.

### 구현 상태

`scripts/btc_tau1_dual_leg_architecture_20260805.py`에 학습과 평가를 아직 호출하지 않는
두 네트워크 계약을 구현했다. 99 causalfix 필드는 원본 causalfix parquet의 114개 열에서
시장 원시값·OHLCV·`mtf1h_ts_t_value`를 제외해 exact count를 검증한다. Regime3의 24개 raw
입력은 동결된 `regime3_current_sensitive_hmm_wide24_2024.joblib`의 `feature_cols`와 일치한다.
따라서 111개 market input(99+6+6)과 24개 Regime input의 소스가 각각 고정되어 있다.

- Leg A: `48×111 → Linear(111,48) → causal TCN(dilation 1/2/4) → last 48`,
  `Regime(24→16→4)`, `52→32→3`.
- Leg B: 완료 시점으로 timestamp를 이동한 `192×111` 1시간 sequence,
  `Linear(111,40) → GRU(40)`, `Regime(24→16→4)`, `44→24→3`.
- 테스트: `test/test_btc_tau1_dual_leg_architecture.py`는 Leg A/B 3-class 출력, TCN의
  prefix causality, split 경계를 검증한다. 이 구현은 checkpoint, threshold, OOS를 선택하지
  않으며 학습 성과를 주장하지 않는다.

## 9. Tau1 Leg B checkpoint/CAL — gate 실패 (2026-08-05)

- Leg B GRU는 train 5,428개, checkpoint-validation 501개로 학습했다. weighted CE 최저는
  epoch 4의 `0.94822`였고, 이후 5회 개선이 없어 epoch 9에서 early stopping했다.
- 이 checkpoint의 CAL(2025-11~12) 평가는 저장 label/exit ledger를 읽지 않고, 각 1시간 결정
  시점의 확정 feature·causal trend gate·다음 5분봉 진입만으로 처음부터 순차 실행했다.
- 확률 임계값 0.45에서 8건, 누적 순수 price return `-6.96%`, 평균 `-0.87%`, bootstrap 5%
  평균 하한 `-3.62%`였다. 0.50/0.55는 각 1건 음수, 0.60 이상은 거래가 없었다.
- 최소 20건·누적 양수·bootstrap 5% 하한 양수 조건을 만족한 후보가 없으므로 **OOS를 실행하지
  않았다.** 이는 fresh-forward CAL gate의 정상적인 NO-GO 결과이며 promotion 근거가 아니다.

구현: `scripts/train_btc_tau1_dual_leg_20260805.py`,
`scripts/eval_btc_tau1_leg_b_fresh_forward_20260805.py`; report:
`tmp/btc_tau1_leg_b_fresh_forward_20260805/report.json`.

## 10. Tau1 Leg A checkpoint/CAL — NaN 버그 수정 후 게이트 실패 (2026-08-05)

섹션 9까지는 Leg B만 평가됐고 Leg A는 `leg_a_console.log`가 0바이트인 채로 방치돼 있었다.
재학습을 시도해 원인을 찾았다.

**버그**: `train_btc_tau1_dual_leg_20260805.py`의 `ready` 마스크가 윈도우 **끝 행**의
finite 여부만 확인하고, 윈도우 내부(과거 47개 행, Leg A는 48×111 시퀀스)에 NaN이 있는지는
확인하지 않았다. `btc_unified_raw_panel_20260804.parquet`의 `mtf1h_ts_opt_L`
(RAW_EXCLUDE에서 누락되어 111개 market field에 포함되어 있었음), DVOL, 온체인, state7/12
컬럼에 히스토리 전반에 걸쳐 산발적 NaN이 있어(끝 행 근처 index 271796까지 존재, 워밍업
구간에만 국한되지 않음) 윈도우 중간에 NaN이 섞여 forward pass가 즉시 NaN을 내고
`RuntimeError("no checkpoint saved")`로 죽었다. Leg B는 우연히 이 문제를 피했는데,
`hourly_completed_features()`가 시간봉으로 리샘플하면서 `.dropna()`를 이미 호출하기
때문이다.

**수정**: (1) `load_feature_frame()`에서 market/regime 컬럼에 causal forward-fill 적용
(미래값 사용 없음), (2) `ready` 마스크를 윈도우 끝 행이 아니라 **윈도우 전체**가 finite인
행만 통과하도록 롤링 체크로 교체. 두 수정 모두 `scripts/btc_tau1_dual_leg_architecture_20260805.py`
/ `scripts/train_btc_tau1_dual_leg_20260805.py`에 반영했다.

**재학습**: best checkpoint는 epoch 1 (checkpoint-validation weighted CE `1.11919`),
이후 5epoch 연속 악화로 조기종료 — Sigma3 문서가 이미 기록한 "neural은 즉시 과적합" 패턴과
동일하다.

**CAL 평가** (`scripts/eval_btc_tau1_leg_a_fresh_forward_20260805.py`, Leg B와 동일하게
저장 ledger 미사용·bar-by-bar 순차 실행): probability threshold {0.45..0.65} × flat-margin
{0.00,0.05,0.10} 15개 조합 전부 평균 순수익 음수(-0.06%~-0.25%/trade). 거래 건수는
87~211건으로 Leg B(8건)와 달리 표본은 충분하다. 최소 50건·누적 양수·bootstrap 5% 하한 양수
게이트를 통과한 후보가 없어 **OOS는 실행하지 않았다.**

**결론**: Leg A(5분 tactical)도 게이트 실패로, Tau1 이중 레그 둘 다 이제 평가가 끝났다 —
Leg B(1시간, 표본 8건)와 Leg A(5분, 표본 87~211건) 모두 CAL에서 막혔다. Leg A는 표본이
충분해 노이즈로 보기 어렵다. 이걸로 causalfix+Regime3+DVOL+온체인 조합 위에서 시도한 라벨/
아키텍처가 (섹션 0A의 4개 + 섹션 5~6의 2개 + 섹션 9~10의 2개 =) 8개로 늘었고 전부 실패다.
섹션 4의 "다음은 마이크로구조/청산 또는 옵션 스큐 데이터가 준비된 뒤" 결론이 다시 한번
재확인됐다.

구현: `scripts/eval_btc_tau1_leg_a_fresh_forward_20260805.py`; report:
`tmp/btc_tau1_leg_a_fresh_forward_20260805/report.json`.
