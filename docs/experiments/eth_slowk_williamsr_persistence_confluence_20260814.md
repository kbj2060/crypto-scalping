# Slow %K × Williams %R 지속성 컨플루언스 오실레이터 설계 — 2026-08-14

상태: **CLOSE (부정적 결과)**. Fresh-forward VAL/OOS 백테스트 결과 모든 arm이 벤치마크 대비 대폭 열위. 피처 스크리닝에서도 forward-return과의 관계가 거의 0에 가까움. 아래 8~9절 참조. 이 문서는 지표 조합 기반 진입 알고리즘의 도출 근거, 정확한 스펙, 실행 결과를 기록한다.

**리비전 노트**: 최초 버전은 "Williams %R과 Fast Stochastic %K는 수학적으로 동일(%K=100+%R)하므로 원안이 지표 1개짜리 조건과 같다"고 지적하고 4계열 지표 앙상블(DCOE)로 대체했었다. 사용자가 **Slow Stochastic %K**(raw %K의 3봉 SMA)를 의도했음을 명확히 하여 재설계한다. 이 버전이 유효 설계다.

## 1. 왜 이 조합이 겹말이 아닌가

```
raw %R(t)     = -100 * (HH14 - C) / (HH14 - LL14)          # 원시, 무평활
fastK(t)      = 100 + %R(t)                                  # = raw %K(14), %R과 동일 정보
slowK(t)      = SMA( fastK, 3 )                               # Slow Stochastic %K(14,3)
```

`raw %R`(≡ fastK)과 `slowK`는 **같은 원천 시계열의 서로 다른 시간축 버전**이다 — fastK는 현재 봉, slowK는 최근 3봉 평균. 상관은 높지만(보통 0.85~0.95) 동일하지 않다. 따라서:

```
fastK(t) 극값이지만 slowK(t)는 아직 극값 아님  → 방금 시작된 스파이크, 지속성 미확인
fastK(t)와 slowK(t) 둘 다 극값                → 극값이 최소 수 봉간 유지됨, 노이즈 필터 통과
```

**"둘 다 80/20 돌파"는 지표 중복이 아니라 "현재값 AND 최근 3봉 평균"이라는 지속성(persistence) 필터다.** 이것이 원안의 진짜 가치이며, 이 설계는 이를 그대로 살리고 세 가지 창의적 요소로 보강한다.

## 2. 핵심 설계 3가지

### 2.1 고정 80/20 대신 적응형 분위수 임계값

고정 레벨은 추세장에서 상시 발화하고 횡보장에서 거의 발화하지 않는다. fastK, slowK 각각을 **rolling percentile-rank**로 변환:

```
p_fast(t) = percentile_rank( fastK(t) | fastK(t-W..t-1) )
p_slow(t) = percentile_rank( slowK(t) | slowK(t-W..t-1) )
W = 864 bar (5분봉 3일)
```

과매수/과매도 기준을 "최근 분포 기준 상/하위 q%"로 자동 적응시킨다 (기본 q=0.10). 완전 causal이라 fresh-forward 규칙과 호환된다.

지속성 필터 조건:
```
과매도 확정: p_fast(t) ≤ q  AND  p_slow(t) ≤ q
과매수 확정: p_fast(t) ≥ 1-q  AND  p_slow(t) ≥ 1-q
```

### 2.2 스토캐스틱 스프레드 — 내장된 조기 반전 신호 (핵심 창의 요소)

fastK와 slowK는 이미 "빠른 선/느린 선" 관계이므로, **별도 지표를 추가하지 않고** 이 관계 자체에서 MACD와 동일한 구조의 조기 신호를 뽑아낼 수 있다:

```
spread(t) = fastK(t) - slowK(t)     # MACD의 (fast EMA - slow EMA)와 같은 구조
```

- 과매도 존에서 `spread`가 음(-)에서 양(+)으로 전환 = raw %R이 자신의 3봉 평균보다 먼저 반등하기 시작 = slowK가 아직 못 따라온 조기 반전 신호.
- 이는 "존에서 빠져나오는 첫 신호"를 레벨 교차보다 먼저 포착한다 (fastK가 slowK를 상향 돌파하는 시점 ≈ 스토캐스틱 %K/%D 크로스와 동일 개념이지만, 여기선 fastK 자체가 %D 역할의 slowK와 크로스).

### 2.3 진입 트리거: 지속성 확정 + 스프레드 반전, 레벨이 아니라 re-cross

%R 단독 백테스트 문헌에서 단순 OB/OS 레벨 진입(profit factor ~1.8)보다 **존 이탈 재진입** 트리거(profit factor ~4.5)가 일관되게 우월했다. 두 조건을 결합:

```
1단계 — 지속성 확정 (t-1 시점): p_fast(t-1) ≤ q AND p_slow(t-1) ≤ q   (과매도 존에 "머물렀음"을 확인)
2단계 — 스프레드 반전 트리거 (t 시점): spread(t-1) < 0 AND spread(t) ≥ 0   (fastK가 slowK를 상향 돌파)

LONG 진입: 1단계가 최근 N bar(예 N=6) 내 성립 + 2단계가 t에서 발생
SHORT 진입: 대칭 (과매수 존 + spread(t-1) > 0 AND spread(t) ≤ 0)
```

레벨에 진입하지 않고, "극값이 지속된 뒤 방향 전환의 증거(fastK가 slowK를 앞지름)"가 나온 시점에 진입 — 낙하하는 칼을 받지 않는 구조.

### 2.4 레짐 게이트: fade 모드 vs follow 모드

문헌 공통 결론: 과매수/과매도 역추세 매매는 횡보장 전용이고, 추세장에서 극값은 반전이 아니라 지속 신호다.

```
trend(t) = (ADX_14 > 25) and (|EMA48_slope| > 0.5 * ATR14 / 48)

fade 모드 (횡보, trend=False): 위 2.3의 LONG/SHORT 트리거를 그대로 역추세 신호로 사용
follow 모드 (추세, trend=True): 극값을 fade하지 않는다.
  상승추세 & 과매도 지속성+spread 반전 → LONG (pullback 매수만 허용)
  하락추세 & 과매수 지속성+spread 반전 → SHORT
  역추세 방향(상승추세에서 SHORT 등)의 트리거는 무시
```

## 3. 청산·사이징 (Futures Risk Sizing Contract 준수)

```
leverage = 3 (고정)
strength(t_entry) = |p_fast - 0.5| + |p_slow - 0.5|          # 0~1, 지속성 강도
margin_fraction = base_mf * min(1, strength / 0.8)
notional = margin_fraction * leverage

tp_price_move = 1.6 * ATR14 / price     # 초기값, VAL 재보정 대상
sl_price_move = 1.0 * ATR14 / price
time_exit     = 48 bar (4시간)
take_profit   = tp_price_move * notional
stop_loss     = sl_price_move * notional
```

notional 도출 후 TP/SL 가격선에 leverage를 다시 곱하지 않는다 (contract 원문 참조).

## 4. 검증 계획 (리포 규칙 준수)

- Fresh-forward bar-by-bar: VAL 2025-09-01~2025-12-31, OOS 2026-01-01~2026-03-31.
- `fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false` 명시.
- 벤치마크: max(always_long, always_short) 대비 초과 성과 필수 (drift-as-skill 방지).
- 수수료 0.1% round-trip 포함. 문헌상 수수료가 단순 지표 전략 수익의 상당분을 잠식함.
- Ablation 필수: (a) 원안(고정 80/20 레벨 AND, 트리거 없음) vs (b) 본 설계 full vs (c) 적응형 분위수 제거(고정 80/20 복귀) vs (d) 스프레드 반전 트리거 제거(레벨 진입으로 대체) vs (e) 레짐 게이트 제거. 각 요소의 기여를 분리.
- 파라미터(q, W, N, θ 관련값, ATR 배수)는 VAL에서만 튜닝, OOS는 1회 평가.

## 5. 파라미터 초기값 요약

| 파라미터 | 값 | 비고 |
|---|---|---|
| %R / fastK n | 14 | |
| slowK 평활 | SMA 3 | Slow Stochastic %K(14,3) |
| percentile 윈도 W | 864 bar | 3일 |
| extreme 분위수 q | 0.10 | VAL 튜닝 대상 |
| 지속성 확정 lookback N | 6 bar | 30분, VAL 튜닝 대상 |
| ADX 추세 기준 | 25 | |
| TP/SL ATR 배수 | 1.6 / 1.0 | |
| time exit | 48 bar | |

## 6. 참고 문헌·자료

- Technical Analysis Meets Machine Learning: Bitcoin Evidence — arxiv.org/html/2511.00665v1 (수수료 반영 시 단순 지표 전략 대부분 buy&hold 미달)
- Predicting Market Trends with Enhanced Technical Indicator Integration — arxiv.org/abs/2410.06935
- A forest of opinions: multi-model ensemble-HMM voting framework for regime shift detection — aimspress.com/article/id/69045d2fba35de34708adb5d (레짐 감지 개념 차용)
- QuantifiedStrategies Williams %R backtest — quantifiedstrategies.com/williams-r-trading-strategy/ (failure-swing/재진입 트리거의 우월성 근거)
- Fidelity indicator guide — %R = fast stochastic 역수 관계 확인

## 7. 향후 확장 (선택, 이번 스코프 아님)

원 검토에서 도출했던 "탈상관 가중 다계열 앙상블"(MFI, Bollinger %b, taker imbalance 추가) 아이디어는 이 2-지표 설계가 VAL/OOS에서 검증된 뒤, 신호 빈도가 부족하거나 추가 알파가 필요할 때 확장안으로 남겨둔다.

## 8. 실행 결과 — 독립 전략 백테스트 (2026-08-14)

스크립트: `scripts/backtest_eth_slowk_williamsr_persistence_confluence_20260814.py` (신규, `core/causal_futures_backtest.py::simulate_single_position` + `purged_decision_mask` 재사용). G0 셀프체크(합성 데이터로 %K=100+%R 항등식, TP/SL/notional 산식 검증) 통과 후 실행.

`fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false`.

**윈도 편차 (명시)**: OOS 기본 종료일 2026-03-31이나 `data/eth_5m_1year.csv`가 2026-02-17까지만 존재하여 OOS를 2026-01-01~2026-02-17로 절단함. 이후 데이터 확보 시 재실행 필요.

레버리지 3, margin_fraction=0.30 고정 (설계 문서 3절의 strength-비례 사이징 대신 원 신호의 순수 엣지를 먼저 검증하기 위해 고정값 사용 — 사이징은 신호가 검증된 뒤에나 의미가 있음), roundtrip cost 0.1%.

### VAL (2025-09-01~2025-12-31) — 5-arm ablation

이 구간은 ETH가 강하게 하락(always_long -32.09%, **always_short +47.26%**)한 추세장이었다.

| arm | trades | return | MDD | win rate | PF | Sharpe(일) | 벤치마크 초과 |
|---|---|---|---|---|---|---|---|
| a_original (고정 80/20, 트리거 없음) | 3568 | -97.36% | -97.39% | 37.8% | 0.52 | -1.45 | ✗ |
| b_full (제안 설계) | 595 | -43.42% | -43.80% | 35.5% | 0.47 | -0.69 | ✗ |
| c_no_adaptive | 1028 | -62.05% | -62.39% | 36.2% | 0.51 | -0.77 | ✗ |
| d_no_spread_trigger | 598 | -41.33% | -41.70% | 37.8% | 0.51 | -0.69 | ✗ |
| e_no_regime_gate | 2272 | -91.00% | -91.06% | 36.7% | 0.50 | -1.10 | ✗ |

**5개 arm 전부 always_short(+47.26%)는 물론 always_long(-32.09%)에도 못 미친다.** 원안(a)은 과다매매(3568건)와 수수료 누적으로 사실상 계좌가 소진(-97%)됐다. 레짐 게이트(b, e 비교)는 손실을 91%→43%로 줄였지만 부호를 뒤집지는 못했다 — ADX>25 기준이 이 구간의 완만하지만 지속적인 하락 추세를 "추세"로 충분히 인식하지 못해 fade(역추세 매수)가 하락장에서 반복적으로 발화한 것으로 보인다.

### OOS (2026-01-01~2026-02-17, 절단됨) — VAL 선정 arm 단일 확인

VAL 기준 벤치마크 대비 손실폭이 가장 작았던 `d_no_spread_trigger`를 선정해 OOS 1회 평가:

| | trades | return | MDD | win rate | PF | 벤치마크(long/short) |
|---|---|---|---|---|---|---|
| d_no_spread_trigger | 257 | -20.00% | -21.28% | — | 0.53 | -34.38% / **+52.40%** |

OOS도 같은 패턴(강한 하락 추세, always_short 압도적 우위, 전략은 손실)이 반복됐다. VAL에서 이미 탈락한 candidate라 OOS는 참고용.

### 결론

**이 알고리즘 계열은 독립 매매 전략으로 채택하지 않는다 (CLOSE).** 지속적 하락 추세 구간에서 역추세 오실레이터가 구조적으로 불리하다는, 문헌에서 이미 경고한 실패 모드가 그대로 재현됐다. 레짐 게이트로 완화는 되지만 제거하지는 못했다.

## 9. 실행 결과 — ML 피처화 가능성 스크리닝 (2026-08-14)

스크립트: `scripts/verify_eth_slowk_williamsr_confluence_feature_relevance_20260814.py` (신규). `scripts/verify_eth_defillama_onchain_direction_relevance_20260812.py`의 오염도 검사(spearman vs price/time, `CONTAMINATION_THRESHOLD=0.561`) + IC/MI 스크리닝 패턴을 재사용. **탐색적 스크리닝이며 promotion 근거 아님** — forward return은 스크리닝 타깃으로만 쓰였고 트레이딩 입력이 아니다 (repo의 다른 `verify_*_relevance` 스크립트와 동일한 역할 구분).

후보 피처 8개: `confluence_p_fast`, `confluence_p_slow`, `confluence_spread`, `confluence_spread_z`, `confluence_score`(±0.5 중심 연속 점수), `confluence_persistence_bars`, `confluence_score_x_fade`, `confluence_score_x_follow`(레짐 상호작용 항).

**오염도**: 8개 전부 `|spearman vs price|`, `|spearman vs time|` 모두 0.02 미만 — 오실레이터 특성상 자연히 detrend되어 있어 오염 없음 (통과).

**Forward-return 관련성** (spearman IC, h=12/48/96bar):

| feature | h12(1h) | h48(4h) | h96(8h) |
|---|---|---|---|
| confluence_score_x_follow | +0.0154 | +0.0121 | **+0.0206** |
| confluence_score | +0.0112 | +0.0070 | +0.0127 |
| confluence_p_fast | -0.0136 | -0.0074 | -0.0126 |
| confluence_spread | -0.0132 | -0.0053 | -0.0045 |
| 나머지 | 모두 \|IC\| < 0.016 | | |

mutual_info(부호 분류)는 대부분 0.000이고 최댓값도 0.004 수준으로 사실상 무신호.

### 해석

모든 IC가 **|0.02| 미만**으로, 이 서브프로젝트의 다른 통과 피처(예: h48qual MFE 회귀 헤드 spearman VAL +0.28/OOS +0.39, [[odyssey_eth_h48qual_subproject]])보다 한 자릿수 이상 약하다. 8개 중 유일하게 방향성이 일관되고(모든 horizon 양수) horizon이 길어질수록 커지는 것은 **레짐(추세) 상호작용 항 `confluence_score_x_follow`** 뿐 — "극값 지속 + 추세 방향 일치"일 때만 약한 신호가 남는다는 8절의 백테스트 결론과 정합적이다.

**결론**: 이 오실레이터 조합을 raw feature로 그대로 Omega 피처 패널에 추가하는 것은 권장하지 않는다 — IC가 너무 약해 100+ 피처 앙상블에서 노이즈에 묻힐 가능성이 높고, 이미 ADX 등 추세 피처가 존재해 `x_follow` 상호작용 항의 한계 정보량도 낮을 것으로 예상된다. 굳이 시도한다면 raw `confluence_p_fast`/`p_slow`가 아니라 **`confluence_score_x_follow`(추세 상호작용 항) 단일 컬럼만** 후보로 좁히고, 기존 100+ 피처 대비 ablation(넣기 전/후 VAL R²·MI 변화)으로 실제 한계 기여도를 확인해야 한다 — 이번 스크리닝은 그 전 단계의 예비 필터일 뿐 최종 판단이 아니다.

## 10. 산출물

- `scripts/backtest_eth_slowk_williamsr_persistence_confluence_20260814.py`
- `scripts/verify_eth_slowk_williamsr_confluence_feature_relevance_20260814.py`
- `tmp/eth_slowk_williamsr_persistence_confluence_20260814/report.json` (5-arm VAL + OOS 지표, 파라미터)
- `tmp/eth_slowk_williamsr_persistence_confluence_20260814/{val,oos}_ledger_*.csv` (arm별 거래 원장)
- `tmp/eth_slowk_williamsr_persistence_confluence_20260814/feature_contamination.csv`, `feature_relevance.csv`
