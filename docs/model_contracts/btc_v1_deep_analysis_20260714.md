# BTC v1 Deep Analysis - 2026-07-14

Status: `analysis_reference_not_promotion_artifact`

이 문서는 2026-07-14 현재 `trading_bot.py`에 연결된 BTC v1을 지도학습 라벨부터
5분봉 매매 결정, 포지션 종료, risk sizing, OOS 성능까지 역추적한 분석 기준서다.
모델이나 라이브 설정은 변경하지 않았다.

## 1. 결론 요약

BTC v1은 단순한 3-class 분류기가 아니다. 실제 전략은 아래 다섯 층의 결합이다.

```text
5분봉 causal feature
  -> regime3 hard router (bull / bear / chop)
  -> TabM direction head + quality head
  -> q055 entry gate
  -> ATR TP/SL + trade-level HGB risk sizing + side scale
  -> 5분마다 TP/SL/learned-exit 검사
```

현재 모델의 핵심 평가는 다음과 같다.

1. **독립 BTC 모델이라고 부르기에는 ETH 설계 의존성이 크다.** 라벨 레시피, feature
   naming, parent 구조를 ETH에서 복제했고, cross asset은 ETH를 사용한다.
2. **방향 라벨은 hindsight zigzag 구간, quality 라벨은 4시간 triple barrier지만 실제
   포지션은 수일에서 수주까지 유지될 수 있다.** 학습 target과 실행 lifecycle의 시간축이
   맞지 않는다.
3. **보고서상 train 78,624 bar와 direction/quality fit 30,000 bar를 구분해야 한다.**
   Direction/quality fit 구간은 2025-01-01 00:00부터 2025-04-15 03:55까지이며, shared
   model은 별도로 만든 exit-state 12,000 rows도 함께 학습한다.
4. **quality head의 OOS 분리력은 약하다.** 현재 2026-01-01..07-12 prediction artifact에서
   final action의 3-class balanced accuracy는 33.11%, active signal의 exact-side precision은
   28.54%다. confidence가 높은 quintile에서도 precision lift가 없다.
5. **최종 수익은 parent 전체보다 short exposure 비대칭에 크게 의존한다.** risk sidecar의
   leverage mapping은 실제로 2.0 고정이고, 후단 scale map이 long 0.5, short 2.5를 적용한다.
   결과적으로 기본 leverage 2.0은 long 1.0, short 5.0까지 갈 수 있다.
6. **현재 기준 BTC solo 성능은 292%가 아니다.** 최신 v1 checkpoint 기준 OOS extended는
   +10.52%, MDD -16.46%다. 292.19%는 과거 ETH/SOL/BTC 동시 portfolio replay 결과이며,
   pipeline 재생성 후 같은 전체 기간 결과도 +62.63%로 바뀌었다.
7. **BTC v2의 첫 목표는 더 복잡한 parent가 아니다.** 먼저 event label, 실제 exit horizon,
   immutable artifact, OOF prediction, untouched test window를 바로잡아야 한다.

## 2. 현재 라이브 계약

Source of truth: `docs/model_contracts/live_model_v1_checkpoint_20260714.md`

| 항목 | BTC v1 현재 값 |
|---|---|
| component | `h48qual` only |
| parent | `btc_omega4_3head_parent72_loose_entry_quality_20260708_h48qual_20260708` |
| risk sidecar | `btc_omega4_2_trade_risk_sidecar_20260708_h48qual_q055_20260708` |
| quality threshold | `0.55` (`q055`) |
| duration threshold 원값 | `0.00541154875` |
| 현재 duration gate | OFF, runtime threshold `-999.0` |
| scale map | `h48qual_L=0.5`, `h48qual_S=2.5` |
| BTC 추가 notional multiplier | 없음, implicit `1.0x` |
| leverage cap | `5.0` |
| notional cap | `1.8` |
| exit threshold | `0.95` |
| portfolio cap | `uncapped` |

`duration_gate=off`는 `ou_halflife` feature 자체를 제거하지 않는다. Runtime은 여전히
`ou_halflife`가 존재하고 finite인지 검사하며 parent 입력에도 사용한다. 단지
`halflife <= duration_threshold` post-entry veto가 항상 false가 되도록 threshold를
`-999.0`으로 바꾼다.

또한 BTC의 `1.0x`는 낮은 exposure를 뜻하지 않는다. 별도 multiplier가 없을 뿐이며,
risk sidecar 뒤에서 short scale 2.5가 적용된다. 선택된 sidecar leverage가 2.0 고정이므로
long은 약 1.0 leverage, short는 cap인 5.0 leverage로 변환된다.

실주문 plumbing은 연결돼 있지만 `.env`의 `BINANCE_ACCOUNT_ENABLED=False` 때문에
`BinanceFuturesExecutionAdapter._ready()`가 false다. 현재는 의사결정 경로만 active이고
실주문은 fail-closed 상태다.

## 3. 데이터와 causal warmup

BTC feature pipeline은 다음 순서다.

```text
BTCUSDT 5m kline
+ BTC daily OI / top-trader metrics
+ BTC funding
+ ETHUSDT 5m cross reference
  -> btc_raw_frame_2024_2026.csv
  -> FeatureEngineer.process()
  -> btc_features_2024_2026.csv
  -> year slice: btc_features_2024.csv / 2025.csv / 2026.csv
  -> frozen regime3 current-wide24 overlay
```

FeatureEngineer를 2024-2026 통합 frame에 먼저 적용하고 나중에 연도별로 자르므로,
2025-01-01과 2026-01-01의 rolling feature에는 각 시작일 이전의 과거 bar가 들어간다.
즉 feature warmup은 존재하며 미래 bar를 warmup에 쓰지 않는다.

현재 분석 기준 split은 다음과 같다.

| 용도 | 구간 | rows |
|---|---|---:|
| label/prediction train frame | 2025-01-01..09-30 | 78,624 |
| validation | 2025-10-01..12-31 | 26,496 |
| extended OOS | 2026-01-01..07-12 16:50 | 55,499 |
| direction/quality fit input | 2025-01-01..04-15 03:55 | 30,000 |
| separate exit-head fit frame | early train segments, capped sample | 12,000 |

마지막 두 행의 차이가 중요하다. Trainer 기본값 `--max-train-rows=30000`이 적용돼
direction/quality head는 Jan-Sep 전체가 아니라 첫 30,000 bar만 사용했다. 이 30,000 bar
안에서 다시 앞 85%를 optimizer train, 뒤 15%를 internal validation으로 사용한다.
동일 shared TabM은 별도 exit-state 12,000 rows도 같이 학습한다. Report의 78,624 train
label 통계는 direction/quality optimizer에 실제 투입된 row 수가 아니라 전체 train frame
통계다.

## 4. 지도학습 라벨

### 4.1 Direction head

Direction target은 `CASH=0`, `LONG=1`, `SHORT=2`인 confirmed-pivot zigzag segment label이다.

주요 파라미터:

- reversal: 1%
- minimum wave: 8 bars, 즉 최소 40분
- transition buffer: 2 bars
- ATR window: 14
- ATR multiplier: 1.0
- transition buffer는 CASH
- future path는 offline label 생성에만 사용

2025 전체 기준 CASH는 5.72%뿐이다. 따라서 direction head가 배우는 문제는 "거래할지
말지"보다 hindsight trend segment의 long/short 방향을 매 bar에 복원하는 문제에 가깝다.
같은 wave 안의 수십 개 5분봉이 거의 동일한 target으로 반복되므로 effective independent
sample 수는 raw row 수보다 훨씬 작다.

| split | CASH | LONG | SHORT | active ratio |
|---|---:|---:|---:|---:|
| train Jan-Sep 2025 | 4,347 | 39,100 | 35,177 | 94.47% |
| validation Q4 2025 | 1,671 | 13,408 | 11,417 | 93.69% |
| current OOS 2026-01-01..07-12 | 3,699 | 26,049 | 25,751 | 93.33% |

### 4.2 Quality head

현재 quality target은 parent report에 적힌 generic hard rule이 아니다.
`quality_mode=quality_label_action`이므로 실제 target은 별도
`h48_conservative` triple-barrier label의 action을 그대로 사용한다.

실제 contract:

- signal bar 다음 bar open 진입
- horizon 48개 5분봉, 즉 4시간
- past-only ATR: rolling 96, min periods 24, 1-bar shift
- TP: `max(0.6%, 1.2 * ATR)`
- SL: `max(0.4%, 0.8 * ATR)`
- 같은 bar에서 TP와 SL이 모두 닿으면 SL 우선
- label quality cost: `2 * (fee 0.05% + slip 0.02%) * 3 = 0.42%`
- adverse excursion penalty와 SL penalty를 차감한 long/short quality 중 양수인 최선 side
- 둘 다 양수가 아니면 CASH

이 label builder가 validation distribution만 보고 자동 선택한 설정은 `h24_runner`였다.
하지만 BTC v1은 ETH 구조를 맞추기 위해 `h48_conservative`를 수동 채택했다. 즉 현재
quality horizon은 BTC validation utility로 선택된 결과가 아니다.

현재 2026 quality label 분포는 CASH 23,947, LONG 14,912, SHORT 16,640으로 active ratio가
56.85%다. Parent report 안의 OOS 52,350-row 통계는 7월 연장 전 snapshot이며 현재
55,499-row artifact와 동일한 snapshot이 아니다.

### 4.3 라벨과 실행의 핵심 불일치

Quality target은 4시간 안의 outcome을 묻지만 live execution에는 fixed time exit가 없다.
과거 exact replay diagnostic에서 실제 hold는 다음과 같았다.

| side | trades | median hold | max hold | side compound PnL |
|---|---:|---:|---:|---:|
| LONG | 12 | 32.96시간 | 422.5시간 | -3.82% |
| SHORT | 18 | 151.42시간 | 573.0시간 | +27.56% |

이 표는 2026-07-08의 30-trade old-snapshot ledger 진단이며 최신 +10.52% 성능 원장이
아니다. 그래도 4시간 label과 실제 lifecycle 간 시간축 불일치를 확인하는 데는 충분하다.
최장 573시간은 약 23.9일이다.

## 5. Feature contract

Parent bundle의 입력은 base 147개와 position 13개, 합계 160개다. Entry 때 position
feature는 모두 0으로 채우고, open position의 exit head 평가 때만 실제 position state를
넣는다.

Feature group은 대략 다음과 같다.

- raw level: OHLCV, quote volume, trades, taker volume
- derivatives/microstructure: OI, top-trader ratio, funding, taker imbalance
- technical/path: RSI, MACD, Bollinger width, HMA, wick, volatility, VWAP, breakout
- statistical: Hurst, GARCH, OU, jump, EVT
- cross asset: ETH reference return/volume/correlation/beta features
- route state: regime3 bull/bear/chop probabilities, confidence, entropy, margin

### 5.1 Cross-asset naming inversion

BTC raw builder는 ETHUSDT를 cross reference로 읽은 뒤, 기존 ETH-primary
`FeatureEngineer` contract를 재사용하기 위해 ETH 값을 `close_btc`, `volume_btc`,
`quote_volume_btc`에 저장한다.

따라서 BTC model 안에서는 다음과 같은 의미 역전이 생긴다.

- `close` = BTC
- `close_btc` = ETH
- `btc_ret_*` = 실제로 ETH return 계열
- `eth_btc_ret_spread_*` = 실제로 BTC minus ETH return spread
- `btc_lead_eth_follow_gap_*` = 이름과 반대 방향의 cross relation을 포함

수학적으로 cross feature는 작동할 수 있지만 contract 이름이 실제 asset semantics와
다르다. 새 BTC 모델에서는 alias나 묵시적 보정을 추가하지 말고 `primary_*`, `cross_eth_*`
같은 정확한 이름으로 artifact contract를 새로 정의해 fail-fast 해야 한다.

### 5.2 비정상 raw level과 drift

Bundle에는 `open/high/low/close`, OI level, top-trader ratio처럼 stationarity가 약한 raw
level이 그대로 들어 있다. 실제 저장 scaler 기준 2026 OOS median의 standardized shift는
다음이 컸다.

| feature | OOS median shift vs saved fit scaler |
|---|---:|
| `sum_toptrader_long_short_ratio` | -4.56 sigma |
| `high` | -2.95 sigma |
| `open` | -2.95 sigma |
| `close` | -2.95 sigma |
| `low` | -2.94 sigma |

`ou_halflife`의 현재 robust-IQR median shift는 0.12 IQR로 위 raw levels보다 작았다.
따라서 이번 snapshot에서 `ou_halflife` 하나만 제거한다고 drift 문제가 해결되지는 않는다.
가격 level 대신 return, normalized distance, rolling rank를 사용하고 train/live drift를
feature group 단위로 감시해야 한다.

## 6. Parent architecture와 실제 학습

각 regime expert는 동일한 3-head TabM이다.

| 설정 | 값 |
|---|---:|
| TabM members `k` | 8 |
| hidden | 192 |
| layers | 3 |
| dropout | 0.08 |
| batch size | 2,048 |
| learning rate | 0.002 |
| weight decay | 0.0002 |
| quality loss weight | 0.80 |
| exit loss weight | 1.15 |
| trained epochs | bull/bear/chop 모두 4 |

세 expert는 각자 regime row만 따로 학습하지 않는다. 모든 row를 사용하되 bull/bear/chop
route probability를 sample weight로 곱한다. Inference에서는 마지막 bar의 route를 hard
selection해 expert 하나만 사용한다. 즉 training은 soft-weighted, inference는 hard-routed다.

또 하나의 artifact contract 문제가 있다. `train_predictions_q055.csv`와
`validation_predictions_q055.csv` column prefix는 `_oof_`지만 실제 trainer는 OOF fold
prediction을 만들지 않는다. 첫 30,000 row로 fit한 동일 model이 train 전체를 rescore한다.
따라서 risk sidecar의 train prediction을 진정한 OOF prediction으로 간주하면 안 된다.

Parent report는 quality threshold ranking을 OOS PnL 순으로 저장한다. 현재 q055는 우연히
validation PnL 최고 threshold이기도 하지만, report 설계 자체는 OOS를 ranking에 노출한다.
새 모델 선택에서는 OOS column을 selection table에서 제거해야 한다.

## 7. Prediction 진단

아래 수치는 current q055 prediction CSV와 현재 direction/quality label을 timestamp로 직접
join해 재계산했다. 거래 성과가 아니라 classifier diagnostic이다.

| split | direction balanced acc | final quality balanced acc | final active rate | active exact precision |
|---|---:|---:|---:|---:|
| train frame | 57.90% | 35.43% | 9.79% | 38.96% |
| validation | 54.22% | 33.88% | 7.47% | 31.75% |
| current OOS | 49.97% | 33.11% | 17.25% | 28.54% |

Current OOS q055 final action:

- active bars: 9,576 / 55,499
- long bars: 1,258
- short bars: 8,318
- long exact precision: 18.76%
- short exact precision: 30.02%
- active signal 중 short 비율: 86.86%

월별 short 비중도 고정적이지 않다. 특히 2026-04은 active 2,464개 중 short 2,422개,
2026-05는 active 2,476개 중 short 2,320개다. 이는 OOS에서 quality gate가 validation보다
2.3배 자주 열리고 거의 short gate로 변했다는 뜻이다.

OOS active signal을 `quality_for_action` confidence quintile로 나눴을 때 exact precision은
낮은 quintile부터 29.02%, 29.50%, 28.77%, 27.89%, 27.52%였다. 높은 confidence가 더 높은
정확도로 이어지지 않는다. 현재 `0.55`는 확률 calibration threshold라기보다 특정
snapshot에서 선택된 score cutoff로 보는 편이 정확하다.

주의할 점은 9,576 active bar가 9,576건의 진입 후보라는 뜻은 아니라는 것이다. 포지션이
열려 있는 동안 새 entry는 무시되고 exit만 평가한다. 긴 hold와 single-position contract
때문에 최신 final replay의 실제 거래는 31건뿐이다.

## 8. Exit lifecycle

Live exit 순서는 다음과 같다.

1. raw directional price move가 ATR TP에 도달했는지 검사
2. raw directional price move가 ATR SL에 도달했는지 검사
3. originating component의 learned exit probability 계산
4. probability가 0.95 이상이면 `exit_head`, 아니면 계속 hold

Entry 시 barrier:

- ATR window: 192 bars, 16시간
- TP: `clip(max(7.5%, 12 * ATR), max=22%)`
- SL: `clip(max(4.0%, 6 * ATR), max=12%)`
- barrier는 price move이며 notional/leverage로 다시 곱하지 않는다
- fixed max hold와 cooldown은 0, 즉 비활성

Exit head train set은 12,000 in-position rows다.

| label | rows |
|---|---:|
| hold | 11,441 |
| exit positive | 559 (4.66%) |
| terminal-window exit | 558 |
| MFE giveback exit | 1 |

Exit positive의 99.8%가 zigzag segment 마지막 3 bars에서 생겼다. 모델은 실행 utility가
아니라 hindsight zigzag 종료 시점을 맞추도록 학습됐으며 class imbalance도 크다. 과거 BTC
final replay에서는 learned exit가 실제 청산을 한 번도 만들지 않았다. 현재 lifecycle의
실질적인 exit owner는 넓은 ATR TP/SL이고, 그 결과 수일에서 수주 hold가 가능하다.

## 9. Risk sidecar와 exposure

Risk sidecar는 side-split HGB 두 개로 entry 시점의 `net_per_notional`을 예측한다.
입력은 raw market feature가 아니라 parent/router/decision 출력 29개다.

Risk training label은 46 trades뿐이다.

- long 25, short 21
- target median: -4.03% per notional
- target p25: -4.15%
- target p75: +7.54%
- validation replay: 16 trades
- old OOS replay: 30 trades

선택 mapping은 margin을 score에 따라 바꾸지만 leverage는 `min=2.0`, `max=2.0`으로
고정이다. Report는 `dynamic_leverage=true`와
`require_dynamic_leverage_mapping=true`를 선언하지만 실제 selected mapping은 dynamic하지
않다. Runtime contract 검사도 boolean flag만 확인하고 min/max 차이는 검증하지 않는다.

더 중요한 점은 selection rule의 validation MDD guard가 `>= -8%`인데 선택된 variant의
validation MDD는 ledger 기준 -11.31%, full replay 기준 -13.73%라는 것이다. Guard를
통과한 candidate가 없을 때 fallback으로 선택됐고 `full_replay_selection_applied=false`다.
즉 sidecar는 자체 승격 기준을 만족한 모델이 아니다.

실제 performance asymmetry는 후단 scale map에서 만들어진다.

```text
sidecar margin_fraction: 약 0.22..0.36
sidecar leverage: 2.0 fixed

LONG:  leverage = 2.0 * 0.5 = 1.0
       notional ~= 0.22..0.36

SHORT: leverage = min(2.0 * 2.5, 5.0) = 5.0
       notional ~= 1.10..1.80
```

따라서 같은 margin prediction에서도 short notional은 long의 약 5배다. 과거 old-snapshot
replay에서도 short compound +27.56%, long -3.82%였다. 현재 BTC v1의 양의 PnL은
균형 잡힌 long/short alpha보다 short regime와 비대칭 sizing에 더 의존한다.

새 risk model은 46건으로 dynamic sizing을 학습시키기보다, 충분한 OOF trade sample이
쌓일 때까지 fixed conservative margin을 baseline으로 두는 편이 통계적으로 더 정직하다.

## 10. 5분봉 매매 결정의 정확한 의미

매 5분 완료 bar마다 flat 상태이면 다음을 수행한다.

1. feature/state가 finite이고 bundle contract와 정확히 일치하는지 확인
2. frozen regime3 overlay로 bull/bear/chop expert 선택
3. direction argmax가 CASH면 종료
4. 선택 action의 quality probability가 0.55 미만이면 종료
5. ATR192로 TP/SL price barrier 계산
6. HGB sidecar로 margin fraction과 leverage 계산
7. side scale과 leverage/notional cap 적용
8. duration gate는 현재 OFF이므로 `ou_halflife` veto를 적용하지 않음
9. entry를 생성하고 이후 각 5분 bar에서 TP/SL/exit head를 평가

Futures Risk Sizing Contract는 다음 식을 따른다.

```text
notional = margin_fraction * leverage
PnL = directional_price_move * notional
```

TP/SL은 price line이다. Notional을 계산한 뒤 TP/SL price move에 leverage를 다시 곱하면
double counting이다. 현재 runtime의 TP/SL 비교는 raw move를 사용하므로 이 부분은 맞다.

## 11. 성능 수치의 provenance

### 11.1 현재 BTC v1 기준

2026-07-14 live checkpoint, pipeline 재생성 후, cost multiplier 1.0,
`duration_gate=off`, BTC 추가 multiplier 1.0x:

| split | PnL | MDD | MTM MDD | trades | WR |
|---|---:|---:|---:|---:|---:|
| validation 2025 Q4 | +6.69% | -12.11% | -12.97% | 16 | 31.25% |
| OOS 2026-01-01..07-12 | +10.52% | -16.46% | -20.48% | 31 | 35.48% |
| OOS Q1 2026 | +6.21% | -16.46% | -20.48% | 20 | 35.00% |

이 수치는 현재 v1 비교 baseline으로 사용한다. 다만 raw result는
`/tmp/btc_sol_lowcost_tuning_sweep_results.json`에 있고 실행 script도 scratchpad였으므로,
장기 재현 가능한 promotion artifact로는 부족하다. 다음 공식 candidate는 workspace 안에
script, report, ledger, config hash를 모두 저장해야 한다.

### 11.2 2026-07-08 old snapshot

`btc_final_scale_map_20260708/report.json`의 수치는 현재 pipeline 이전 진단이다.

| config | validation | OOS extended |
|---|---:|---:|
| no duration gate | +7.45% / MDD -11.93% | +22.69% / MDD -15.88% |
| selected duration gate | +12.39% / MDD -6.49% | +29.23% / MDD -10.65% |

이 값은 구조 분석에는 쓸 수 있지만 현재 live expected PnL로 인용하면 안 된다.

### 11.3 292.19%와 BTC v1의 차이

`portfolio_concurrent_3asset_CURRENT_BASELINE_20260712.md`의 292.19%는 다음 조건의 portfolio
수치다.

- ETH, SOL, BTC 동시 독립 포지션
- duration gate 전체 OFF
- ETH notional multiplier 1.5x
- BTC/SOL multiplier 1.0x
- portfolio notional uncapped
- shared cash compounding

즉 BTC solo alpha가 아니다. 더구나 이후 current-formula로 data/feature/prediction을
재생성한 `portfolio_concurrent_3asset_fresh_window_confirmation_20260713.md`에서는 동일한
full extended OOS 결과가 +62.63%, MDD -46.08%로 바뀌었다. 292.19%는 당시 snapshot에서
계산된 causal replay였지만 현재 pipeline의 재현 가능한 기준 성능은 아니다.

또한 이 portfolio chain은 같은 Jan-Jun OOS를 보고 cap, gate, ETH multiplier를 반복
조정했다. Bar-by-bar 계산이 causal이어도 연구자가 OOS 결과를 보고 다음 설정을 선택하면
selection-level OOS leakage가 생긴다. 292% 또는 62.63%를 expected live return으로 보면
안 되는 이유다.

## 12. Artifact integrity 문제

Parent report는 2026-07-08 생성 당시 OOS rows를 52,350으로 기록한다. 현재 같은 directory의
`oos_predictions_q055.csv`는 2026-07-13 rescore로 in-place overwrite돼 55,499 rows다.

따라서 현재 directory에는 다음 두 snapshot이 섞여 있다.

- `report.json`: old cutoff metadata와 old OOS result
- `oos_predictions_q055.csv`: extended current-formula prediction

Omega integrity audit가 exact q055 path와 runtime timestamp 일치를 검사하더라도, report와
prediction이 동일 run에서 생성됐다는 불변성까지 보장하지는 않는다. 새 artifact는 기존
directory를 overwrite하지 않고 run ID 아래 immutable하게 저장해야 한다.

필수 manifest:

- source data hash와 timestamp range
- feature code/version hash
- label config hash
- bundle hash
- exact train/validation/OOS prediction hash
- sidecar hash와 exact parent prediction tag
- replay config와 cost convention
- report/ledger 생성 command

## 13. 주요 실패 원인 우선순위

### P0 - Evaluation contract

- 현재 Jan-Jul 2026은 여러 번 본 OOS다. 다음 upgrade selection에 다시 사용하면 안 된다.
- `_oof`로 명명된 train prediction이 실제 OOF가 아니다.
- Report와 prediction CSV가 in-place rescore로 서로 다른 snapshot이다.
- 최신 v1 metric의 raw runner가 `/tmp` scratchpad에 남아 있어 durable reproduction이 약하다.

### P1 - Label/execution mismatch

- direction target은 hindsight wave의 매-bar 복제다.
- quality target horizon은 4시간인데 실제 median hold는 33시간/151시간이다.
- quality label은 cost3, 현재 final performance는 cost1 convention이다.
- learned exit target은 거의 전부 hindsight segment terminal window다.
- fixed time exit가 없어 label horizon 밖의 risk를 parent가 책임지지 못한다.

### P1 - Model/data drift

- 실제 parent fit은 2025 첫 30,000 bars에 제한된다.
- raw price와 top-trader level이 OOS에서 3-4.6 sigma 이동했다.
- ETH cross feature 이름과 실제 semantics가 뒤집혀 있다.
- Validation 대비 OOS active rate가 7.47%에서 17.25%로 증가하고 short에 편향됐다.

### P1 - Risk overfit

- risk HGB 학습 표본은 46 trades뿐이다.
- dynamic leverage contract지만 선택 mapping은 2.0 fixed다.
- selected mapping이 validation MDD guard를 위반한다.
- 최종 PnL은 learned risk보다 hand-selected short 2.5 scale에 크게 의존한다.

## 14. BTC v2 설계에 대한 직접적 요구사항

다음 candidate는 아래 순서로 설계해야 한다.

1. **Sparse event target**
   - 5분마다 중복 label을 만들지 않는다.
   - causal change point 또는 volatility-adjusted event에서만 entry candidate를 만든다.
   - 같은 trend 안의 중복 sample을 제거하고 purged embargo를 적용한다.

2. **Direction과 utility 분리**
   - direction은 BTC return/path 기준으로 학습한다.
   - quality는 선택된 side의 실제 execution utility를 예측한다.
   - 서로 다른 binary probability를 직접 비교해 side를 고르지 않는다.

3. **Label과 exit horizon 통일**
   - 4시간 label이면 4시간 또는 사전 정의된 hazard/time-exit contract를 둔다.
   - 수일 runner를 허용하려면 multi-horizon target과 lifecycle model을 별도로 설계한다.
   - TP/SL/timeout/giveback label이 실제 replay와 동일한 price/cost contract를 써야 한다.

4. **Stationary BTC-native feature contract**
   - raw OHLC level을 return, normalized range, rolling rank로 교체한다.
   - ETH cross feature는 `cross_eth_*`로 명시한다.
   - feature rename은 새 artifact version에서 fail-fast하며 legacy alias를 live path에 넣지 않는다.

5. **True walk-forward OOF**
   - risk/quality calibration 입력은 purged walk-forward OOF prediction만 사용한다.
   - 동일 row를 fit한 model의 prediction을 OOF라고 부르지 않는다.
   - threshold와 side scale은 validation fold aggregate로만 선택한다.

6. **단순한 risk baseline부터 시작**
   - OOF trades가 충분하기 전에는 fixed margin/fixed leverage를 baseline으로 둔다.
   - dynamic mapping을 쓰면 leverage min/max가 실제로 달라야 하고 모든 MDD guard를 통과해야 한다.
   - `notional = margin_fraction * leverage`와 price-move TP/SL contract를 유지한다.

## 15. 다음 candidate 승격 조건

BTC v2는 최소한 다음 조건을 모두 만족하기 전에는 live 후보가 아니다.

- untouched future test window 사전 등록
- train/validation/test 날짜와 embargo 명시
- `fresh_forward_bar_by_bar=true`
- `trade_ledgers_used_as_input=false`
- `saved_parent_exit_timestamps_used=false`
- `future_rows_used_for_entry=false`
- immutable report/prediction/bundle/sidecar hash 일치
- Omega artifact integrity audit `promotion_pass=true`
- validation fold별 PnL/MDD/trade count 공개
- long/short 별 성능과 exposure 공개
- confidence calibration curve와 threshold stability 공개
- max hold, median hold, exit reason 분포 공개
- cost1뿐 아니라 fee/slippage stress 결과 공개
- BTC v1 +10.52%/-16.46%와 동일한 current pipeline에서 비교

## 16. 근거 파일

현재 live와 성능 기준:

- `docs/model_contracts/live_model_v1_checkpoint_20260714.md`
- `docs/model_contracts/btc_sol_lowcost_tuning_sweep_20260713.md`
- `trading_bot.py`
- `trading_bot_modules/omega4_6_1_live.py`
- `trading_bot_modules/binance_execution.py`

Data/label/parent:

- `scripts/build_btc_raw_frame_20260708.py`
- `scripts/build_btc_features_20260708.py`
- `scripts/split_btc_features_by_year_20260708.py`
- `scripts/build_omega1_2_triple_barrier_labels_btc_20260708.py`
- `scripts/pad_h48_quality_labels_to_zigzag_timestamps_btc_20260708.py`
- `scripts/train_eval_omega4_3head_parent72_loose_entry_quality_btc_20260708.py`
- `tmp/causal_regen_20260516/btc_zigzag_action_labels_20260708/zigzag_action_label_audit.json`
- `tmp/causal_regen_20260516/btc_omega1_2_triple_barrier_labels_20260708/report.json`
- `tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_20260708_h48qual_20260708/report.json`

Risk/final replay:

- `scripts/train_eval_omega4_2_risk_sidecar_btc_20260708.py`
- `scripts/apply_final_scale_map_btc_20260708.py`
- `tmp/causal_regen_20260516/btc_omega4_2_trade_risk_sidecar_20260708_h48qual_q055_20260708/report.json`
- `tmp/causal_regen_20260516/btc_final_scale_map_20260708/report.json`
- `docs/model_contracts/btc_omega4_6_1_full_stack_20260708_contract.md`

Portfolio 292% provenance:

- `docs/model_contracts/portfolio_concurrent_3asset_CURRENT_BASELINE_20260712.md`
- `docs/model_contracts/portfolio_concurrent_3asset_fresh_window_confirmation_20260713.md`

후속 negative upgrade research:

- `docs/model_contracts/btc_v2_upgrade_research_20260714.md`
