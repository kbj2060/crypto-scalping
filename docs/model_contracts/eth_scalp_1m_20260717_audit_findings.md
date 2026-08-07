# ETH 1분 스캘핑 연구 전수감사 및 문제제기

- 감사 대상: `docs/model_contracts/eth_scalp_1m_20260717_contract.md`
- 감사일: 2026-07-17 KST
- 감사 상태: `completed`
- 원계약 상태: `research`
- 최종 판정: `research_invalidated_pending_causal_rebuild`
- Live 또는 promotion 사용 가능 여부: `불가`

## 1. 감사 목적

이 문서는 `eth_scalp_1m_20260716` 연구선에서 수행한 데이터 구축, feature 생성,
label 생성, 모델 학습, walk-forward, maker 체결 시뮬레이션, position sizing,
block bootstrap 및 Red Team Gate가 실제로 연구 계약을 충족하는지 전수 검토한 결과다.

감사 범위는 다음과 같다.

- 원계약 문서 전체 133줄
- 관련 `scripts/` 파일 28개
- `data/ensemble/reports/scalp_1m*.json` 리포트 27개
- `data/training_features_1m.csv`의 feature schema와 timestamp 범위
- Binance ETH 1분봉 및 BTC 5분봉 timestamp 의미
- baseline OOS maker 거래 원장 11,339개 signal
- validation/OOS/walk-forward timestamp 중복
- maker fill, TP/SL, portfolio, exposure-cap 회계 구현
- 실험 0번부터 15번까지의 개별 유효성

감사 과정에서는 `data/live/microstructure.duckdb`를 포함한 DuckDB를 수정하지 않았고,
원계약과 기존 리포트도 수정하지 않았다.

## 2. 최종 결론

연구 아이디어와 실험 범위는 폭넓고, 한계를 문서화하려는 노력도 확인된다. 그러나
현재 저장된 수익률, hit rate, max drawdown, walk-forward 양성 fold 수는 live 기대성과나
모델 승격의 근거로 사용할 수 없다.

전체 연구선을 무효화하는 공통 원인은 다음 네 가지다.

1. BTC 5분봉의 종가와 거래량이 확정되기 전에 ETH 1분 feature에 들어가는
   semantic look-ahead가 있다.
2. clean이라고 분류한 walk-forward fold 7이 confidence threshold 선택용 validation과
   21,600개 1분 행 직접 겹친다.
3. maker 체결과 exposure-capped portfolio 시뮬레이터가 실제 주문 및 손익 발생 순서를
   정확하게 처리하지 않는다.
4. 동일한 2026-05-31~2026-07-12 OOS를 architecture, label, threshold, sizing 선택에
   반복 사용했기 때문에 더 이상 untouched OOS가 아니다.

따라서 현재 신뢰 가능한 예상 수익률은 `알 수 없음`이다. 100%/100%, 20%/20%,
1%/5%를 포함한 모든 sizing 후보는 live 후보가 아니다.

## 3. 심각도 기준

| 심각도 | 의미 |
|---|---|
| P0 | 전체 성과나 causal validity를 무효화하는 문제 |
| P1 | 특정 리포트, 실험 또는 리스크 수치를 무효화하는 문제 |
| P2 | 재현성, 통계적 강도, live parity를 크게 약화시키는 문제 |

## 4. P0: BTC 5분봉 semantic look-ahead

### 4.1 원인

`scripts/build_features_1m_20260716.py`는 BTC 5분봉에서 다음 컬럼을 읽는다.

- `timestamp`
- `close`
- `volume`
- `quote_volume`

Binance kline CSV에서 `timestamp`는 봉 시작시각이다. 반면 `close`, `volume`,
`quote_volume`은 5분봉이 종료된 뒤에만 확정된다. 하지만 builder는 availability
timestamp를 5분 뒤로 이동하지 않고 그대로 `FeatureEngineer.process()`에 전달한다.

`features/engineering.py::_merge_data()`는 BTC row를 동일 timestamp 또는 과거 timestamp의
ETH row에 backward-asof로 붙인다. backward-asof라는 연산 방향은 맞지만, 입력 timestamp가
event time이 아니라 bar open time이므로 실제 정보 가용성을 보장하지 못한다.

예를 들어 BTC `15:00` 5분봉의 close는 `15:05`에 확정된다. 현재 구현에서는 이 close가
다음 ETH 의사결정에 사용될 수 있다.

| ETH signal row | 가정된 진입 | 사용된 BTC close | 문제 |
|---|---|---|---|
| 15:00 | 15:01 | 15:00~15:05 봉 close | 약 4분 미래 정보 |
| 15:01 | 15:02 | 동일 봉에서 파생된 rolling state | 미래 정보 전파 |
| 15:02 | 15:03 | 동일 봉에서 파생된 rolling state | 미래 정보 전파 |
| 15:03 | 15:04 | 동일 봉에서 파생된 rolling state | 미래 정보 전파 |
| 15:04 | 15:05 | close 확정과 주문 의사결정이 동시, latency 미반영 | 실거래 가용성 불명확 |

### 4.2 영향을 받는 feature

baseline의 119개 feature 중 다음 15개 BTC 파생 feature가 실제 입력에 포함된다.

1. `btc_corr_60`
2. `eth_btc_ratio_change`
3. `btc_ret_1`
4. `btc_ret_3`
5. `btc_ret_6`
6. `btc_ret_12`
7. `btc_ret_z_48`
8. `eth_btc_ret_spread_12`
9. `eth_btc_ret_spread_48`
10. `eth_btc_beta_residual_z`
11. `btc_lead_eth_follow_gap_3`
12. `btc_breakout_eth_lag_dir`
13. `btc_volume_impulse_z`
14. `btc_eth_volume_rank_spread`
15. `btc_impulse_x_eth_beta`

### 4.3 기존 truncate audit가 검출하지 못한 이유

`scripts/verify_scalp_1m_no_lookahead_20260717.py`는 BTC 원천 행을
`timestamp <= cutoff`으로 자른다. cutoff 시점에 시작한 BTC 5분봉 row는 그대로 남아 있고,
그 row의 완성된 close 역시 남는다. 따라서 full build와 truncated build가 같게 나와
141개 feature 0 mismatch가 발생한다.

이 검사는 미래 row 참조나 centered rolling 같은 계산형 look-ahead는 검출할 수 있지만,
bar open timestamp와 정보 availability timestamp가 다른 semantic timing 오류는 검출하지
못한다. 따라서 기존 `PASS`는 no-lookahead 전체 검증이 아니라 계산형 truncation 검증에만
한정해야 한다.

### 4.4 독립적인 후속 실증

후속 `DeepScalp-PnL v1` 연구에서도 같은 현상이 확인됐다.

- BTC 파생 feature 포함 시 validation +134.0%
- 개발 OOS +77.3%
- `btc_lead_eth_follow_gap_3`, `eth_btc_ret_spread_12`, `btc_ret_3`의 next-1m raw IC가
  약 0.22~0.23
- 모든 BTC 파생 feature 제거 후 과도한 성과 소멸
- 비용 차감 최종 정책은 validation과 개발 OOS에서 CASH 선택

이는 `docs/deepscalp_pnl_v1_20260717.md`에 별도로 기록돼 있다.

### 4.5 영향 범위

다음 실험은 모두 같은 `training_features_1m.csv`와 `ULTIMATE_FEATURE_COLS`를 사용하므로
공통 오염의 영향을 받는다.

- HGB baseline
- B1/B2
- confidence threshold
- realistic maker fill
- 모든 walk-forward
- uniqueness weighting
- meta-label
- TabM
- DP label model
- trendscan label model
- Omega-style router
- threshold 0.70
- GRU
- short-horizon model
- portfolio/exposure sizing
- block bootstrap

## 5. P0: walk-forward clean fold 주장 오류

### 5.1 실제 중복 행 수

원시 ETH 1분봉 timestamp로 직접 계산한 결과는 다음과 같다.

| 구간 | 행 수 |
|---|---:|
| single-window validation | 44,640 |
| single-window OOS | 60,480 |
| walk-forward fold 7 | 61,921 |
| fold 7과 validation 중복 | 21,600 |
| walk-forward fold 8 | 82,081 |
| fold 8과 validation/OOS 중복 | 82,081 |

fold 7은 `2026-04-02~2026-05-15`, validation은 코드상
`timestamp > 2026-04-30`이므로 2026-04-30 00:01 이후 fold 7 구간이 validation과 겹친다.

따라서 timestamp가 직접 겹치지 않는 fold는 1~6뿐이다. 원계약의 “folds 1–7 have zero
overlap with the threshold-selection window” 주장은 사실과 다르다.

### 5.2 threshold의 시간방향 문제

threshold 0.55는 2026년 5월 validation에서 선택한 뒤 2025년과 2026년 초 fold에
소급 적용됐다. folds 1~6의 가격 행은 threshold 선택에 직접 사용되지 않았지만,
해당 시점에 실제 이용 가능하지 않았던 미래 hyperparameter를 사용했다.

따라서 folds 1~6은 다음과 같이 분류해야 한다.

- causal historical production walk-forward: 아님
- threshold와 timestamp가 직접 겹치지 않는 retrospective stability test: 가능
- 독립적인 final promotion test: 아님

### 5.3 threshold 0.70의 OOS 선택

threshold 0.70은 validation sweep으로 선택되지 않았다.
`scripts/reduce_scalp_1m_trade_frequency_20260717.py`가 기존 OOS 전체에서 threshold 0.55부터
0.90까지 거래 수와 PnL을 비교한 뒤 0.70을 operating point로 채택했다.

이후 동일한 정책을 historical folds에 적용한 결과는 retrospective diagnostic이며,
새로운 untouched validation이 아니다.

## 6. P0: 동일 OOS 반복 사용과 선택 편향

2026-05-31~2026-07-12 OOS는 다음 연구 의사결정에 반복 사용됐다.

1. baseline 성과 확인
2. HGB와 B1/B2 비교
3. confidence 및 maker 방식 선택
4. meta-label 평가
5. TabM 평가
6. DP label 평가
7. trendscan label 평가
8. Omega-style 평가
9. threshold 0.70 선택
10. GRU 평가
11. short-horizon 설계와 평가
12. slot cap 선택
13. exposure cap 선택
14. block bootstrap 입력 생성
15. baseline이 feature ceiling에 가깝다는 결론

개별 스크립트가 OOS row를 직접 학습에 넣지 않았더라도, 연구자가 OOS 결과를 보고
다음 architecture, label, threshold, sizing을 결정하면 연구선 전체 관점에서 OOS는
development set으로 소비된다.

특히 short-horizon 실험은 전체 데이터의 DP oracle hold-time을 보고 5분 horizon과
TP/SL scale을 정한 뒤 동일 OOS에서 평가했다. 이 결과는 untouched OOS가 아니다.

원계약의 “14 independent architecture/label/frequency challengers” 표현도 부정확하다.
모든 실험이 같은 feature artifact, BTC 누수, TP/SL 평가, maker simulator 및 OOS를
공유하므로 통계적으로 강하게 종속된다.

## 7. P1: maker 체결 시뮬레이션 문제

### 7.1 동일 1분봉에서 fill과 TP의 순서를 알 수 없음

현재 simulator는 LONG의 경우 low가 limit 아래로 내려가면 fill됐다고 판단한다. 이후
동일한 fill bar의 high와 low로 즉시 TP/SL을 검사한다.

LONG bar에서 다음 순서가 가능하다.

1. high가 TP에 먼저 도달
2. 가격이 하락
3. low가 buy limit에 도달해 체결

이 경우 TP 가격은 체결 전에 발생했으므로 실제 거래의 TP가 아니다. 그러나 현재
1분 OHLC simulator는 이 거래를 TP 성공으로 기록할 수 있다. SHORT도 반대 방향으로
같은 문제가 있다.

### 7.2 OOS 원장 재계산 결과

baseline OOS 거래 원장 11,339개 signal을 원시 ETH OHLC 및 TP/SL과 다시 대조한 결과다.

| 항목 | 결과 |
|---|---:|
| 전체 signal | 11,339 |
| filled | 8,075 |
| 원장과 기존 simulator outcome mismatch | 0 |
| fill bar에서 TP 가격 관측 | 886 |
| fill 중 fill-bar TP 비율 | 10.97% |
| fill bar에서 SL 가격 관측 | 321 |
| filled no-touch | 117 |
| 보수적 OHLC ordering으로 outcome 변경 | 174 |
| 기존 decimal-return 합 | 3.73906464 |
| 보수적 ordering 및 horizon-close 적용 합 | 3.55050617 |

보수적 재계산은 fill bar의 SL은 유효하게 처리하되, fill bar의 TP는 순서가 불명확하므로
다음 bar 이후에 다시 발생해야 유효한 것으로 처리했다. TP/SL no-touch는 0이 아니라
horizon close로 청산했다.

이 결함만으로 기존 edge가 완전히 사라지지는 않지만 현재 구현을 “realistic maker fill”로
표현하기에는 부족하다.

### 7.3 touch-to-full-fill 가정

limit 가격을 한 번 touch하면 주문 전체가 maker fee로 체결된다고 가정한다. 다음 항목이
모델링되지 않는다.

- queue position
- 주문 전방 대기수량
- 체결 가능한 실제 수량
- partial fill
- 주문 제출 latency
- cancel latency
- tick size와 step size
- market impact
- 계정 크기에 따른 capacity
- fill 직후 adverse selection
- TP/SL taker exit slippage
- API 오류와 재시도

특히 100% account notional을 하루 수십 회에서 수백 회 진입하는 정책에서는 계정 크기와
order-book depth가 fill rate에 직접 영향을 준다.

### 7.4 no-touch time exit 오류

filled 이후 TP와 SL을 모두 건드리지 않은 거래를 horizon close 수익으로 청산하지 않고
`realized_move=0.0`으로 처리한다. 실제 close가 entry보다 불리하거나 유리한 경우가 모두
사라지고 수수료만 차감된다.

### 7.5 pending order 상태 누락

실제 limit order는 최대 3분 동안 pending일 수 있다. 현재 simulator는 미래 bar를 보고
filled 또는 unfilled를 즉시 확정하며, pending 주문이 다른 signal, 반대 주문, slot,
exposure에 미치는 영향을 처리하지 않는다.

## 8. P1: PnL 단위 오류

`backtest_maker()`는 각 거래의 decimal return을 단순 합산한 뒤 이를 `total_pnl_pct`라고
저장한다. 100을 곱하지 않으므로 이름과 단위가 일치하지 않는다.

baseline의 `3.73906464`는 다음 의미다.

- +3.739%가 아님
- decimal returns의 합 3.73906464
- percentage-point 합으로는 +373.906464pp
- 겹치는 각 거래가 독립적으로 100% notional을 사용한다는 비현실적 가정
- 실제 portfolio return을 의미하지 않음

따라서 원계약의 “OOS +3.74%”는 단위상 잘못됐고, +373.9% portfolio return으로
바꾸는 것도 잘못이다. 정확한 표현은 “8,075개 겹치는 거래의 decimal net return
단순합 3.7391”이다.

## 9. P1: 비용 스트레스 Gate 오류

원계약은 fee/slippage 1x/2x/3x ranking이 보고됐다고 체크했다. 실제 scalp 리포트에는
maker와 taker 수수료 비교만 있고 realistic maker fill에 대한 1x/2x/3x stress가 없다.

baseline OOS의 거래당 평균 수익으로 직접 계산하면 다음과 같다.

| 비용 가정 | round-trip 비용 | 평균 gross | 평균 net |
|---|---:|---:|---:|
| 0x | 0bp | 11.1304bp | 11.1304bp |
| 1x | 6.5bp | 11.1304bp | +4.6304bp |
| 2x | 13.0bp | 11.1304bp | -1.8696bp |
| 3x | 19.5bp | 11.1304bp | -8.3696bp |

break-even 비용은 현재 모델 비용의 약 1.71배다. 2x 비용에서 평균 net edge가 이미
음수이므로 비용 안정성 Gate는 PASS가 아니라 FAIL이다.

## 10. P1: 초기 baseline 및 tuning replay 오류

`scripts/train_eval_scalp_1m_hgb_20260716.py::backtest_replay()`와
`scripts/tune_scalp_1m_levers_20260716.py::backtest_replay()`는 predicted action이 label과
같으면 TP, 다르면 무조건 SL로 처리한다.

label이 LONG이더라도 predicted SHORT가 실제로 반드시 SL을 맞는 것은 아니다. predicted
SHORT도 TP를 맞을 수 있고, 양쪽 모두 no-touch일 수 있으며, 서로 다른 순서로 barrier가
발생할 수 있다.

이 오류는 다음 리포트에 직접 영향을 준다.

- 초기 A/B1/B2 backtest
- wide horizon
- naive confidence threshold
- naive maker/taker fee comparison

최종 maker simulator는 LONG과 SHORT를 별도로 계산하므로 이 특정 오류는 피하지만,
동일 bar ordering과 touch-to-fill 문제는 남는다.

## 11. P1: exposure-capped 회계 오류

### 11.1 미래 손익의 조기 equity 반영

`run_exposure_capped()`는 미래 OHLC로 exit와 realized move를 계산한 뒤, 실제 exit 시각까지
기다리지 않고 signal 처리 시점에 다음 계산을 수행한다.

```text
notional = per_trade_pct * equity
equity = equity + notional * net
```

그 결과 concurrent 설정에서는 아직 종료되지 않은 거래의 미래 손익이 다음 거래 sizing에
사용된다. equity curve도 exit 순서가 아니라 signal 순서가 된다.

직접적인 영향이 큰 설정은 다음과 같다.

- 5% per trade / 20% max exposure
- 2% per trade / 20% max exposure
- 5% per trade / 10% max exposure
- 1% per trade / 5% max exposure

20%/20%처럼 한 번에 한 포지션만 허용하는 설정은 다음 신규 진입 전에 기존 포지션이
종료되므로 조기 반영이 sizing에 미치는 영향이 제한적이다. 그러나 공통 BTC 누수,
maker fill, pending order, MDD 문제가 남으므로 해당 결과 역시 유효한 promotion 근거가
아니다.

### 11.2 exposure 단위가 실제 notional이 아님

open position에는 실제 notional 금액이 아니라 `per_trade_pct`가 저장된다. equity가
변해도 기존 포지션 notional의 현재 equity 대비 비율을 재계산하지 않는다. 서로 다른
equity 시점에 열린 포지션을 단순 fraction 합으로 비교하므로 exposure cap이 정확한
현재 notional cap이 아니다.

### 11.3 MDD가 mark-to-market이 아님

portfolio simulator는 settlement 시점에만 equity를 변경한다. 포지션 보유 중의
미실현손익, maintenance margin, liquidation proximity가 equity curve에 들어가지 않는다.
따라서 저장된 MDD는 실제 account MDD를 과소평가할 수 있다.

### 11.4 exact worst-loss bound가 아님

원계약은 `MAX_TOTAL_EXPOSURE_PCT`가 정확한 worst single-event loss bound라고 주장하지만
다음 조건이 모델링되지 않았다.

- margin fraction과 leverage의 분리
- stop gap과 slippage
- short position의 100% 초과 손실 가능성
- liquidation 및 maintenance margin
- exchange negative-balance 처리
- 시점별 notional 변화

따라서 max exposure는 주문 허용 정책일 수는 있지만 손실의 수학적 보장은 아니다.

## 12. P1: block bootstrap의 한계와 구현 의미

block bootstrap은 오염된 feature로 생성한 signal과 optimistic maker outcome을 그대로
입력으로 사용한다. bootstrap은 샘플 순서를 재배열할 뿐 다음 문제를 고치지 못한다.

- BTC look-ahead
- maker touch-to-fill bias
- OOS model-selection bias
- 실제 queue와 partial fill
- 미관측 crash regime
- account capacity와 market impact
- mark-to-market drawdown

또한 `collect_daily_multipliers()`는 cap 값과 관계없이 `open_exit_time` 하나만 유지하므로
항상 한 포지션만 연다. cap=5는 최대 5개 동시 포지션이 아니라 단일 포지션을 equity의
1/5로 sizing하는 정책이다. 결과가 exposure 20%/20%와 일치하는 이유도 이 때문이다.

bootstrap 결과는 “관측된 유리한 sample을 재표본화하면 손실 시나리오가 거의 나오지
않는다”는 진단으로만 사용할 수 있다. tail-risk가 작다는 증거로 사용하면 안 된다.

## 13. P2: split 날짜의 자정 해석

Pandas에서 문자열 `2026-04-30`은 `2026-04-30 00:00:00`이다. 현재 split은 다음과 같다.

```text
train: timestamp <= 2026-04-30 00:00
validation: 2026-04-30 00:00 < timestamp <= 2026-05-31 00:00
OOS: 2026-05-31 00:00 < timestamp <= 2026-07-12 00:00
```

따라서 문서가 자연어로 표현한 “train through 2026-04-30”이나 “OOS through 2026-07-12”와
실제 calendar-day 의미가 다르다. 모든 split은 `[start, next_start)` 형식의 명시적인
timestamp로 바꿔야 한다.

## 14. P2: label purge와 alternative-label 문제

### 14.1 baseline과 walk-forward purge

20분 triple-barrier label은 train boundary 직전 row에서 test 구간 OHLC를 사용한다.
main baseline과 main walk-forward는 이 row들을 purge하지 않는다. 영향 행 수가 전체
training에 비해 작더라도 causal contract 위반이며, “작아서 무시”할 문제가 아니다.

weighted/purged ablation이 비슷한 결과를 냈다는 사실은 BTC semantic leak과 threshold
contamination을 해결하지 않으므로 baseline 전체의 무결성을 입증하지 못한다.

### 14.2 DP trajectory label

DP label은 전체 2024~2026 가격 배열의 끝에서 시작해 역방향 value function을 계산한다.
`v_flat[i+1]`이 데이터 끝까지의 최적 미래가치를 포함하므로 training row의 target이
validation/OOS 가격경로에 의존할 수 있다.

MAX_AGE=60은 한 포지션 보유시간만 제한하며 flat state의 미래 opportunity chain을
전체 데이터 끝에서 분리하지 않는다. DP label은 각 training fold 내부에서 별도로
계산해야 한다.

### 14.3 trendscan label

trend-scan threshold 14는 전체 데이터의 t-stat 분포를 보고 선택했다. 또한 label은
최대 60분 forward price를 사용하지만 train boundary에 60분 purge가 없다.

### 14.4 short-horizon label

5분 horizon, ATR lookback, TP/SL bounds는 전체 데이터 DP oracle의 hold-time과 수익률을
보고 결정했다. 이후 동일 OOS에서 평가했으므로 OOS가 label 설계에 사용됐다.

## 15. P2: architecture 실험별 추가 문제

### 15.1 Meta-label

purged 5-block OOF는 held-out block 전후의 모든 다른 training block을 사용한다. 과거
block 예측을 만들 때 더 미래의 training block도 모델 fit에 들어간다. 전체가 외부
validation 이전이라는 점에서 직접적인 validation leak는 아니지만, time-series causal
OOF는 아니며 regime 변화에 낙관적일 수 있다.

### 15.2 TabM

- PyTorch seed가 고정되지 않음
- NumPy seed가 고정되지 않음
- 단일 학습 run만 수행
- HGB에서 선택한 threshold 0.55를 그대로 사용
- architecture별 threshold calibration이 없음
- train/validation boundary label purge가 없음

따라서 HGB 대비 약간 낮은 한 번의 결과만으로 TabM이 열등하다고 확정할 수 없다.

### 15.3 GRU

GRU는 internal validation을 early stopping에 쓰지만 scaler mean/std를 `TRAIN_END`까지의
전체 train, 즉 internal validation을 포함해 fit한다. internal validation이 scaler
추정에 들어가므로 early-stopping validation이 완전히 분리되지 않았다.

추가로 다음 문제가 있다.

- PyTorch/NumPy seed 미고정
- 단일 run
- HGB threshold 고정
- timestamp gap을 확인하지 않고 인접 row를 30분 window로 간주
- 공통 BTC semantic leak

### 15.4 Omega-style router

meta, duration, risk model과 router가 동일한 single-pass OOF rows를 사용한다. router는
meta/duration/risk model의 해당 training rows에 대한 in-sample 예측을 입력받는다.
router-level nested OOF가 없으므로 stacking training score가 과적합될 수 있다.

또한 unfilled order를 실제 0손익이 아니라 `-ROUND_TRIP_FEE`로 risk target에 넣어
실제 execution contract와 다르게 학습한다.

## 16. P2: B2 microstructure 실험의 범위 문제

B2는 동일 기간의 B1과 비교했다는 점은 적절하다. 그러나 다음 이유로 “price + raw
order-book model”로 해석하면 안 된다.

- 입력은 `microstructure_1m` derived signals 중심임
- raw order-book depth snapshot 전체를 직접 학습하지 않음
- `kelly_mult`, `signal_bias`, `eai` 같은 규칙 파생값 포함
- staleness, connection, age, warmup health 컬럼은 join 전에 제외
- 약 70일 overlap, OOS 약 12일
- 동일 BTC semantic leak 포함

따라서 B2는 “기존 live microstructure rule outputs를 포함한 짧은-window overlay”에
가깝다. 순수 microstructure alpha의 증분을 입증하지 못한다.

## 17. P2: 재현성 및 artifact integrity

현재 exposure script에는 50%/50% 설정이 있지만 저장된 exposure JSON에는 해당 row가
없다. 반대로 원계약의 100%/100% row는 exposure JSON에 없고 별도 chart/bootstrap
artifact에서 가져왔다.

확인된 파일 시각은 다음과 같다.

| 파일 | 시각 |
|---|---|
| `scalp_1m_exposure_capped_20260717.json` | 2026-07-17 15:13 KST |
| 원계약 문서 | 2026-07-17 17:16 KST |
| `simulate_exposure_capped_scalp_1m_20260717.py` | 2026-07-17 17:20 KST |

현재 report에는 다음 provenance가 없다.

- git commit 또는 source-code hash
- 입력 feature/label CSV hash
- config snapshot
- Python/sklearn/PyTorch 버전
- feature schema hash
- model artifact
- scaler artifact
- seed 및 deterministic 설정
- report를 생성한 명령

현재 source로 기존 JSON을 정확히 재생성할 수 있다는 보장이 없다.

## 18. 실험 0~15 개별 판정

| # | 실험 | 판정 | 핵심 근거 |
|---:|---|---|---|
| 0 | HGB baseline | 무효 | BTC semantic leak, maker 체결 오류, PnL 단위 및 portfolio 오류 |
| 1 | B2 microstructure | 무효 | BTC 누수, 짧은 OOS, 규칙 파생 입력, raw order-book 전체 비교 아님 |
| 2 | Wide 60분 | 무효 | correctness 기반 잘못된 replay, 60분 boundary purge 없음 |
| 3 | Confidence + naive fee | 무효 | 잘못된 replay 사용, 최종 maker threshold와 다른 선택 |
| 4 | Realistic maker fill | 무효 | BTC 누수, same-bar ordering, touch-to-full-fill, no-touch 오류 |
| 5 | Walk-forward | 무효 | fold 7 직접 중복, 미래 threshold 소급 적용, purge 없음 |
| 6 | Uniqueness weighting | 진단용 | weighting 아이디어는 타당하지만 공통 누수와 fold 문제가 유지됨 |
| 7 | Meta-label | 진단용 | 양방향 time-series OOF, 공통 누수 및 execution 문제 |
| 8 | TabM | 진단용 | 단일 seed, HGB threshold 고정, purge 없음, 공통 누수 |
| 9 | DP trajectory | 무효 | 전체 데이터 역방향 DP로 train target이 미래 가치에 의존 |
| 10 | Trend scanning | 진단용 | 전체 데이터 threshold 선택, 60분 purge 없음, 공통 누수 |
| 11 | Omega-style | 무효 | in-sample stacked router, unfilled target 불일치, 공통 누수 |
| 12 | Threshold 0.70 | 무효 | OOS 결과를 본 뒤 선택하고 historical fold에 소급 적용 |
| 13 | GRU | 무효 | internal-val scaler leak, seed 없음, threshold 고정, BTC 누수 |
| 14 | Short horizon | 무효 | full-sample DP oracle로 설계한 뒤 같은 OOS 재사용 |
| 15 | Block bootstrap | 진단용 | 오염된 outcome 재표본화, 실제 tail/execution/selection bias 복원 불가 |

## 19. 원계약 주장별 정정 판정

| 원계약 주장 | 판정 | 정정 내용 |
|---|---|---|
| causal 1-minute features | FAIL | BTC 5분봉 availability leak 존재 |
| empirical no-lookahead PASS | FAIL | 계산형 truncation만 확인, semantic timing 검출 실패 |
| folds 1~7 clean | FAIL | fold 7 validation과 21,600행 중복 |
| 14 independent challengers | FAIL | 같은 OOS, feature, simulator를 공유하는 종속 실험 |
| realistic maker fill | FAIL | queue, partial fill, same-bar order, latency 미반영 |
| OOS +3.74% | FAIL | decimal-return 합 3.7391을 %로 잘못 표기 |
| 1x/2x/3x cost stress PASS | FAIL | 실제 리포트 없음, 2x에서 평균 edge 음수 |
| exact worst loss = max exposure | FAIL | leverage, gap, short loss, liquidation 미모델링 |
| 1%/5% +26.7%, MDD 0.17% | FAIL | 조기 미래손익 반영 및 mark-to-market 누락 |
| feature set ceiling에 근접 | 입증 안 됨 | 공통 누수와 OOS 반복 사용으로 결론 불가 |
| live 준비 완료 | 해당 없음 | 원계약도 미완료로 명시함 |

## 20. 유지할 가치가 있는 부분

문제와 별개로 다음 연구 관행은 유지할 가치가 있다.

- 원계약 상태를 `research`와 미승격으로 명시
- B1/B2를 동일 기간에 비교해 history 길이 통제
- forward label을 model input과 분리하려고 설계
- uniqueness weighting과 purge를 별도 실험
- maker entry와 taker exit 비용 구분
- unconstrained concurrent capital 문제가 있음을 자체 발견
- fold 8 overlap을 발견하고 제외
- model artifact와 live execution이 없음을 공개
- DuckDB를 read-only로 사용
- 후속 DeepScalp 연구에서 BTC 누수를 발견하고 과도한 성과를 폐기
- 최종 DeepScalp causal model이 CASH를 선택한 결과를 억지로 거래시키지 않음

따라서 기존 연구 기록은 architecture와 label 아이디어의 development log로 보존할 수
있다. 다만 성과 수치는 promotion evidence가 아니라 invalidated research score로
재분류해야 한다.

## 21. 필수 재검증 절차

### 21.1 데이터 availability contract 재구축

1. ETH 1분봉 decision timestamp를 bar close availability 기준으로 정의한다.
2. BTC 5분봉 timestamp를 `open time + 5분 + 수집 latency`로 이동한다.
3. 가능하면 BTC 1분봉을 사용해 time resolution을 일치시킨다.
4. metrics, funding, microstructure도 event time과 availability time을 분리한다.
5. 모든 asof join에 `source_timestamp`, `availability_timestamp`, `age`를 저장한다.
6. negative age와 availability violation이 하나라도 있으면 fail-fast한다.
7. causal base model에서는 기존 BTC 파생 feature 15개를 우선 모두 제외한다.
8. causal BTC artifact가 준비된 뒤 BTC branch를 독립 ablation으로 추가한다.

### 21.2 microstructure input 정리

1. raw market state와 rule-derived state를 구분한다.
2. 순수 딥러닝 연구에서는 `kelly_mult`, `signal_bias`, `eai`, shadow regime tag를 제외한다.
3. staleness, connection, warmup, age 정보를 모델 입력 또는 hard availability gate로 유지한다.
4. USDT와 USDC는 사용자 지시에 따라 symbol feature 없이 합칠 수 있지만, source별
   timestamp와 coverage 통계는 보존한다.
5. order-book snapshot은 spread, depth, imbalance, microprice, capture latency를 포함한다.

### 21.3 split 및 label contract

1. split은 `[start, next_start)` 형식으로 정의한다.
2. 20분 label은 train boundary 앞 20분 purge한다.
3. 60분 label은 60분 purge한다.
4. fill lookahead와 execution horizon까지 고려한 추가 embargo를 검토한다.
5. DP label은 각 training fold 내부에서만 다시 계산한다.
6. label parameter 선택은 inner validation 안에서만 한다.
7. outer test 결과를 본 뒤 label을 변경하면 새 outer test를 확보한다.

### 21.4 nested walk-forward

각 outer fold에서 다음 순서를 지킨다.

1. outer test 이전 데이터만 확보
2. training 내부 inner train/validation 생성
3. feature set, architecture, threshold, label, cost config를 inner validation에서 선택
4. 선택 완료 후 outer train 전체 재학습
5. outer test를 한 번 평가
6. outer test 결과를 다음 실험 선택에 사용하지 않음
7. 모든 outer fold 설정과 code/data hash 저장

과거 folds 1~6은 historical development diagnostics로 재사용할 수 있지만 final promotion
test로 다시 사용할 수 없다.

### 21.5 event-driven execution simulator

주문 상태는 최소 다음 상태를 가져야 한다.

```text
signal_created
order_submitted
pending
partially_filled
filled
cancel_requested
cancelled
tp_submitted
sl_submitted
closed_tp
closed_sl
closed_time
rejected
```

처리할 데이터와 비용은 다음과 같다.

- 주문 제출 timestamp
- exchange acknowledgement timestamp
- limit price와 tick rounding
- requested quantity와 step rounding
- queue ahead estimate
- partial fills
- cancel race
- fill price 및 maker/taker 여부
- TP/SL trigger와 실제 exit price
- exit slippage
- funding settlement crossing
- network/API failure
- account size별 market impact

1분 OHLC만 사용할 경우 fill bar TP는 보수적으로 무효 처리하거나, 더 작은 tick/trade
데이터로 실제 순서를 복원해야 한다.

### 21.6 portfolio 및 futures sizing

다음 항목을 명시적으로 분리한다.

```text
notional = margin_fraction * leverage
margin_fraction = notional / leverage
account_pnl = price_move * notional
```

portfolio replay는 다음 원칙을 따라야 한다.

1. signal, submit, fill, mark, exit, funding event를 timestamp 순으로 처리
2. exit 시점에 realized PnL settlement
3. 매 bar mark-to-market equity 계산
4. available margin 기준 신규 sizing
5. pending order와 open position을 별도 집계
6. long/short gross 및 net exposure 집계
7. maintenance margin과 liquidation price 계산
8. gap과 stop slippage 반영
9. equity가 변할 때 기존 notional의 현재 exposure 비율 재계산

### 21.7 비용 및 capacity stress

최소 다음 시나리오를 동일 signal에 적용한다.

| 시나리오 | 목적 |
|---|---|
| 0x cost | gross alpha 확인 |
| 1x cost | base assumption |
| 1.5x cost | moderate stress |
| 2x cost | promotion 필수 stress |
| 3x cost | severe stress |
| queue fill 25/50/75/100% | passive fill 민감도 |
| exit slippage 0/1/2/5bp | stop/TP execution 민감도 |
| latency 0/250/500/1000ms | live latency 민감도 |
| account notional grid | market capacity와 impact |

### 21.8 통계 검증

1. per-trade가 아니라 일별 account return을 기본 단위로 사용한다.
2. model 비교는 같은 날짜의 paired difference로 수행한다.
3. seed ensemble을 사용한다.
4. bootstrap CI를 보고한다.
5. 여러 threshold, architecture, label 실험에 multiple-testing correction을 적용한다.
6. probability calibration과 realized net-PnL bucket을 생성한다.
7. 월별, 요일별, session별, volatility regime별 결과를 보고한다.
8. PnL concentration과 worst contiguous block을 보고한다.
9. OOS를 본 뒤 변경한 모든 선택을 research log에 기록한다.

### 21.9 artifact integrity

각 실험 artifact에는 다음을 저장한다.

- model id
- git commit
- dirty-worktree 여부
- script SHA-256
- feature CSV SHA-256
- label CSV SHA-256
- DuckDB snapshot signature 또는 read timestamp
- feature 목록과 순서
- feature schema version
- train/validation/test exact timestamp
- purge와 embargo
- model hyperparameters
- random seeds
- package versions
- scaler parameters
- model artifact
- prediction artifact
- execution config
- risk config
- report 생성 명령
- promotion eligibility

source, config, report 중 하나라도 일치하지 않으면 재현 및 promotion을 fail-fast해야 한다.

## 22. 새로운 final OOS 원칙

2026-07-12까지의 데이터는 현재 연구선에서 반복 사용됐으므로 모두 development data로
분류한다. 2026-07-17 이후 새로 수집되는 데이터만 진짜 fresh-forward shadow가 될 수 있다.

권장 순서는 다음과 같다.

1. causal feature 및 execution contract를 먼저 동결
2. 동결 후 parameter를 변경하지 않음
3. 2026-07-18 이후 shadow signal과 실제 주문 가능성 기록
4. passive order는 initially paper/shadow로 queue와 fill calibration
5. 최소 여러 volatility regime와 연속 월을 확보
6. final OOS를 한 번 평가
7. gate 통과 전 real order enable 금지

## 23. Promotion Gate

다음 조건이 모두 충족되기 전에는 promotion할 수 없다.

- BTC 및 모든 cross-asset feature의 availability audit PASS
- label purge/embargo PASS
- nested outer walk-forward PASS
- 완전히 새로운 forward OOS PASS
- event-driven execution replay PASS
- 2x 비용에서 양의 net edge
- mark-to-market MDD와 liquidation stress PASS
- artifact/code/data hash 일치
- model/scaler/prediction artifact 저장
- live feature parity PASS
- shadow fill calibration PASS
- account sizing과 leverage contract 승인
- 실주문 enable에 대한 별도 사용자 결정

## 24. 최종 상태 선언

현재 `eth_scalp_1m_20260716`의 상태는 다음과 같이 해석해야 한다.

```text
model_status = research_invalidated_pending_causal_rebuild
promotion_pass = false
live_candidate = false
expected_live_pnl = unknown
headline_backtest_returns_valid = false
btc_semantic_lookahead_found = true
walkforward_clean_folds_claim_valid = false
maker_execution_validated = false
portfolio_accounting_validated = false
fresh_untouched_oos_available = false
```

기존 연구는 architecture 및 label 탐색 기록으로 보존할 수 있다. 그러나 원계약의
headline 성과, 7/7 clean fold, 1%/5% sizing, 100% sizing, Red Team PASS 및 feature ceiling
결론은 causal rebuild와 새로운 forward OOS가 완료될 때까지 사용할 수 없다.
