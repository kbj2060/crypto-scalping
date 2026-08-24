# ETH 분포적 회귀(distributional regression) 라벨 빌드 (2026-08-19)

**상태: 라벨 빌드 완료(진단용 분포 통계 포함), 학습/평가 미실시 — 기존 direction 라벨
계약과 스키마가 달라 아직 어떤 학습 스크립트로도 소비되지 않음**

## 배경

`entry_exit_edge_external_labeling_literature_review_20260819.md` Part A에서 선정한
재테스트 후보 3개 중 ③번(마지막). Michańków(2025) "Forecasting Probability Distributions
of Financial Returns with Deep Neural Networks"(arXiv:2508.18921) 레시피 — 이산 배리어/
threshold 없이 각 시점의 미래 실현 로그수익률(연속값)을 라벨로 삼고, 모델이 그 위에
Normal/Student-t/skewed-Student-t 분포 모수를 custom NLL loss로 직접 추정(CRPS/LPS로
평가)한다. "배리어 선택"이라는 이 저장소가 반복 실패해온 축 자체를 제거하는 게 핵심
매력이다.

후보①(DC)/②(CUSUM)과 같은 원시 bar(`data/splits/year_oos/training_features_{year}.csv`)를
쓰지만, 방법론이 근본적으로 달라 아래 3가지는 이번 후보에 해당 없다:
- **이벤트 샘플링**: "어느 시점을 볼지 고르는" 개념 자체가 없음 — 고정 horizon 전방수익률
  방식이라 매 bar가 라벨을 가진다(all_bars와 동일 사고방식).
- **배리어 자동튜닝**: 배리어가 없으니 튜닝할 것도 없음.
- **이산화**: 이 라벨은 discretize하지 않는다 — 그 자체가 이 방법론의 매력이라 여기서
  다시 클래스로 뭉개면 의미가 없음.

## 사용자 질문에 대한 답 — "헤드별로 라벨을 정하고 있나?"

정확히는 아니다. 후보①②는 direction 헤드의 라벨만 바꿔치기했고, quality 헤드는
`same_as_direction` 모드로 그 라벨을 재사용, exit 헤드는 라벨 파일 없이 기존 파이프라인이
OHLCV+BASE_TEMPLATE로 즉석 시뮬레이션한다 — "헤드별 라벨 정의"가 아니라 "direction만
후보를 바꾸고 나머지는 기존 메커니즘을 물려받는다"가 정확한 서술이다.

이번 후보③은 그 direction 헤드 자체의 **출력 형태**를 classification(zigzag_action
{0,1,2})에서 regression(분포 모수)으로 바꿔야 해서, 라벨 스키마도 근본적으로 달라진다.
quality/exit 헤드는 이번에도 건드리지 않았다 — 다만 quality는 향후 이 방식에서 "예측
분포의 불확실성(분산/왜도)"로 자연스럽게 대체될 수 있어 보이며, 이는 학습 스크립트
설계 단계의 논의 대상이지 이번 라벨 빌드의 범위가 아니다.

## 구현

신규: [scripts/build_eth_distributional_regression_return_labels_20260819.py](../../scripts/build_eth_distributional_regression_return_labels_20260819.py)

핵심 로직은 단순하다 — horizon별 causal forward log-return:
```python
log_close = np.log(full["close"].to_numpy())
fwd[: n - h] = log_close[h:] - log_close[: n - h]   # bar t 라벨 = t+h 시점까지의 로그수익률
```
horizon 4개(12/24/48/96bar)를 동시에 계산했다 — 24bar는 후보①②의 `calibrate_barriers`가
두 이벤트 샘플링 방식 모두에서 반복 선택한 max_hold와 일치시켜 비교 가능하게 했고, 나머지는
참고용. 어떤 horizon/분포족이 맞는지는 라벨 빌드 단계가 아니라 학습 단계에서 결정할 문제라
이번 스크립트는 선택하지 않는다.

3개년 concat 처리 이유는 후보①②보다 오히려 더 명확하다 — per-year로 쪼개면 매년 12월 말
bar들이 다음 해 1월 데이터로 계산 가능한 forward return을 갖고 있는데도 그냥 버려진다
(워밍업/배리어절단 같은 부수 효과가 아니라 실질적 정보 손실). concat 후 연도별로 재분할하되,
2026년(가장 마지막 연도) 끝부분은 horizon만큼 자연스럽게 NaN으로 남긴다(실측: h=96 기준
마지막 96행 NaN, 정확히 일치 확인).

## 결과 — horizon별 분포 진단 (편집 전 realized log-return)

| 연도 | horizon | mean | std | skew | kurtosis(초과) |
|---|---:|---:|---:|---:|---:|
| 2024 | 12bar(1h) | +0.0043% | 0.688% | **-0.86** | **+25.1** |
| 2024 | 24bar(2h) | +0.0086% | 0.965% | -0.90 | +22.2 |
| 2024 | 48bar(4h) | +0.0173% | 1.362% | -0.54 | +16.0 |
| 2024 | 96bar(8h) | +0.0347% | 1.924% | -0.36 | +10.9 |
| 2025 | 12bar | -0.0014% | 0.768% | -0.49 | +18.7 |
| 2025 | 24bar | -0.0027% | 1.092% | -0.28 | +12.1 |
| 2025 | 48bar | -0.0054% | 1.568% | -0.33 | +8.3 |
| 2025 | 96bar | -0.0107% | 2.269% | -0.36 | +5.7 |
| 2026(~07) | 12bar | -0.0097% | 0.664% | -0.01 | +11.3 |
| 2026(~07) | 24bar | -0.0194% | 0.926% | +0.09 | +7.3 |
| 2026(~07) | 48bar | -0.0391% | 1.291% | -0.10 | +4.7 |
| 2026(~07) | 96bar | -0.0782% | 1.817% | -0.13 | +3.8 |

**세 가지 관찰**:
1. **초과 첨도가 전 구간에서 매우 크다**(3.8~25.1, Normal 분포 기준선=0) — 특히 짧은
   horizon일수록 극단적이다. Michańków 논문이 Normal 대신 Student-t/skewed-Student-t를
   제안하는 이유가 이 데이터에서도 그대로 확인된다.
2. **왜도가 해마다 부호와 크기가 다르다**(2024 강한 음의 왜도 -0.86~-0.36, 2025 약한 음
   -0.49~-0.28, 2026 사실상 대칭 -0.01~+0.09) — 고정된 하나의 왜도 모수로는 이 비정상성을
   못 담는다. 다만 이건 이 방법론의 약점이 아니라 정확히 이 방법론이 풀려는 문제다 — 모델이
   입력 조건부로 분포 모수(왜도 포함)를 매번 다르게 예측하도록 설계됐기 때문.
3. **평균의 부호가 해마다 바뀐다**(2024 양(+), 2025/2026 음(-)) — `eth_btc_regime_shift_
   reopening_candidates_20260819`(메모리)가 기록한 "VAL/OOS 단방향 약세장"과 방향이
   일치한다. 별도 검증 없이 raw 라벨 통계에서 같은 레짐 신호가 재확인된 셈이다.

전체 `report.json`: `tmp/eth_distributional_regression_return_labels_20260819/report.json`.

## 스모크 테스트 — 해당 없음(설계상)

후보①②와 달리 이 라벨은 `_read_labels()`나 다른 기존 소비 스크립트로 읽히도록 설계되지
않았다(스키마 자체가 다름을 스크립트 docstring에 명시). 대신 출력 CSV 구조를 직접
점검했다 — 2026년 파일 마지막 행(2026-07-20 00:00)에서 4개 horizon 전부 NaN, h=96
기준 정확히 마지막 96행만 NaN인 것을 확인해 causal 방향과 NaN 처리가 의도대로 동작함을
검증했다.

## 스코프 경고

이 문서의 분포 진단(첨도/왜도/평균)은 **"realized return이 정규분포가 아니다"라는 사실을
보여줄 뿐, 방향 예측 edge가 있다는 근거가 아니다.** 애초에 이건 라벨(정답값)의 형태에
대한 진단이지 모델 예측 성능이 아니다. `docs/label_methodology_survey_20260815.md`가
기록한 대로 이 저장소의 40개 이상 선행 라벨 방법론이 전부 "학습 가능하나 방향 edge
없음"으로 수렴했다 — edge 판정은 별도의 학습 + Fresh-Forward 평가 단계의 몫이며, 이
후보는 아직 그 단계 근처에도 가지 않았다(학습 스크립트 자체가 없다).

## 다음 단계 (미착수, 범위가 가장 큼)

후보①②는 기존 TabM 학습 스크립트에 `--direction-label-dir`만 바꾸면 됐지만, 이 후보는
**direction_head의 출력층/손실함수 자체를 새로 설계**해야 해서 나머지 두 후보보다 훨씬
큰 작업이다:
1. TabM 백본 위에 분포 모수(예: skewed-Student-t의 μ,σ,ν,λ) 출력 헤드 설계.
2. NLL loss 구현 + CRPS/LPS 평가 지표 구현(문헌은 트레이딩 백테스트가 아니라 이 지표들로만
   검증했음 — 이 저장소는 실제 백테스트까지 요구하므로 그 연결고리도 새로 만들어야 함).
3. 예측 분포에서 실제 방향 결정(LONG/SHORT/CASH)을 뽑아내는 규칙 설계(예: 부호, 특정
   분위수 임계값, 혹은 quality 대체용 분산 기반 게이팅).
4. horizon 4개 중 실제로 쓸 것 선택(또는 멀티-horizon 헤드).

## Cheap-gate 착수 — 정보량 체크 (2026-08-20)

전체 분포모수 헤드+NLL loss+CRPS평가+백테스트("나머지 두 후보보다 훨씬 큰 작업")를 새로 짜기
전에, 158개 캐노니컬 피쳐가 연속 forward log-return 타겟과 조금이라도 관계가 있는지부터
확인(DC/CUSUM 개별피쳐 정보량 체크와 동일 원칙). 스크립트:
`scripts/eth_distributional_regression_feature_information_content_20260820.py`. TRAIN
(2024-01~2025-08)/VAL(2025-09~12)/OOS(2026-01~03) 3-split × 4horizon(h12/24/48/96bar) ×
158개 중 최고|IC| permutation-null(N=2000, 벡터화).

**1단계 — raw IC 스캔**: 12/12칸 전부 순열귀무 통과(empirical_p=0.000, 이 세션에서 가장 높은
통과율). 그러나 **"최고IC" 피쳐가 TRAIN/VAL/OOS 3개 split 전부 겹치지 않음**(12칸 전부
서로 다른 승자: mtf_trend_1h/cvd_288/hour_cos/fibonacci_level/vwap_dist_288/regime3_bear_
prob/low/sum_open_interest_value) — 이 세션 내내 봐온 "다중비교 중 매번 다른 우연 승자"
패턴과 동일.

**⚠️ 시도했다 폐기한 체크(방법론 실수, 투명하게 기록)**: "1주(2016bar) 롤링평균 대비
초과수익률"로 국소추세를 제거해보니 12칸 전부 `dual_momentum`이 압도적 1위(|IC|
0.076→0.247, 호라이즌 길수록 증가)로 나와 처음엔 "진짜 신호"로 보였다. 그러나 검증 결과
`dual_momentum`도 정확히 같은 2016bar(`features/engineering.py:958`) lookback을 쓰고,
직접 확인하니 `dual_momentum` vs raw타겟 rho≈0(±0.01~0.02)인데 vs 내가 뺀 롤링평균
rho=+0.84~0.86 — **내가 우연히 dual_momentum과 똑같은 윈도우로 "추세제거"를 해서 만든
순환논리(기계적 아티팩트)였다.** 진짜 발견이 아니므로 폐기, 이 절 전체를 기록으로만 남김
(같은 실수를 반복하지 않기 위해).

**2단계 — 158개 전부의 split간 부호일관성**(1위 하나가 아니라 전체 대조, 이 세션의 정확한
기준): |IC|>=0.02 & TRAIN·VAL·OOS 3개 전부 동일부호인 피쳐가 h12=12개/h24=10개/h48=12개/
h96=16개 존재. 그러나 **h24~h96에서 `open/high/low/close`(원시가격레벨)가 반복 등장** —
`low`는 이미 별도로 가격오염 체크에서 rho=1.0000(동시점 close와 사실상 동일)으로 확인됨,
동일 계열인 open/high/close도 같은 우려. 이 넷을 빼면 h48/h96에 남는 건 **거의 전부
변동성 클러스터링 계열**(`volatility_z`/`atr_pct_rank_288`/`realized_vol_ratio`/
`garch_vol_z`/`garman_klass_vol`/`rogers_satchell_vol`/`parkinson_vol`/
`bb_width_pct_rank_288`/`compression_score`) — 부호가 양(변동성↑→미래수익률↑, `compression_
score`만 음=저변동성↔양의 관계, 개념적으로 일관)이고 TRAIN→VAL→OOS·h48→h96 갈수록 |IC|가
단조증가(예 `atr_pct_rank_288`: 0.037→0.060→0.074, 세 split 전부).

**해석 — 이 세션에서 가장 유력한 후보지만 아직 미확정**: 변동성 스파이크 이후 되돌림(잘 알려진
"vol spike → mean reversion" 패턴)일 가능성과, VAL/OOS 자체가 단방향 약세장이라 변동성이
하락과 구조적으로 동시발생(leverage effect)하는 것을 방향신호처럼 잘못 읽는 것일 가능성을
아직 구분 못 함 — 3개 대형 split 단위 부호일관성만으로는 이 둘을 못 가른다(하위기간별
안정성 체크, 추세통제 등 추가 검증 필요). raw open/high/low/close처럼 명백히 오염된 건
아니지만, 그렇다고 이 시점에서 "확인된 신호"라고 부를 근거도 아직 없다.

## 변동성 클러스터링 패턴 집중검증 — 통계적으론 최강, 경제적으론 기각 (2026-08-20 후속)

사용자 지시("1번, 집중검증")로 두 체크 수행.

**체크A(월별 하위기간 안정성)**: TRAIN/VAL/OOS를 30개 월별 블록(2024-01~2026-06)으로 쪼개
블록별 IC 개별계산(`scripts/eth_distributional_regression_volatility_pattern_verification_20260820.py`).
**5개 변동성피쳐×2호라이즌(h48/h96) 전부 66.7~83.3%가 집계부호와 일치**(n=30 이항검정 기준
대부분 p<0.01) — 3개 대형 split 일관성보다 훨씬 엄격한 기준을 통과.

**체크B(후행추세 통제 partial correlation)**: 158피쳐/dual_momentum 어느 창과도 안 겹치는
신규 60bar(5시간) 후행수익률을 통제변수로 넣고 partial correlation 계산. **raw와 partial이
거의 동일**(예 TRAIN h48 volatility_z: 0.0342→0.0342 불변, OOS는 거의 모든 피쳐에서 소수점
셋째자리까지 불변) — leverage effect(하락 중 변동성 상승)로 설명되는 관계가 아님을 시사,
①단계의 dual_momentum 순환논리 실수와 달리 이번엔 통제변수 자체가 어느 피쳐 정의와도 안
겹침을 사전확인 후 진행.

**최종관문(체크C) — 실제 비용반영 벤치마크 백테스트**: 이 세션 전체의 최종판정 기준을 동일
적용(TRAIN median을 임계값+TRAIN IC부호로 고정, VAL/OOS 불변적용, 왕복10bp,
`scripts/eth_distributional_regression_volatility_backtest_20260820.py`). **결과: 5피쳐×2
호라이즌=10조합 중 9개 기각(VAL 증분 대부분 큰 폭 음수: −1.5~−12.3bp), 1개만
3-split전부양수(`garch_vol_z`+h48: TRAIN+0.7/VAL+0.5/OOS+2.3bp)이나 크기가 극히 작고 10개
중 1개라 이 세션에서 반복된 "다중비교 중 우연한 생존자" 패턴과 구분 안 됨.**

**결론**: 이 패턴은 통계적으로는 이 세션 전체에서 가장 견고했다(하위기간 안정성+추세통제
둘 다 통과, dual_momentum류 아티팩트 아님을 직접 확인). 그러나 **단순 임계값 기반 매매규칙으로
전환하면 실제 비용 반영시 경제성이 없다** — ETF플로우(IC유의했던 VAL h7d가 백테스트 대패)/
spot-perp베이스(TRAIN자체도 손익분기)에 이은 이 세션 3번째 "통계적 IC는 진짜, 경제적으론
무의미" 사례. IC 크기(raw 0.03~0.08) 자체가 왕복10bp 비용을 이기기엔 원래 작았다는 게
가장 근본적인 설명으로 보인다(더 정교한 포지션사이징 규칙을 시도 안 해봄 -- 남은 유일한
미탐색 변형, 우선순위는 낮음: IC크기 자체가 작아 정교화해도 비용을 넘길 가능성이 낮다고 판단).

**전체 판정**: ③분포적회귀 후보의 cheap-gate 완료 — raw IC(노이즈패턴)/변동성클러스터링
(통계적 생존, 경제적 기각) 둘 다 확인. 전체 분포모수헤드+NLL loss+CRPS평가+백테스트 연결(고비용
전체구축)에 투자할 근거 없음. registry 등록.

## 참고

- `docs/experiments/eth_directional_change_labels_20260819.md` — 후보①(DC)
- `docs/experiments/eth_cusum_triple_barrier_labels_20260819.md` — 후보②(CUSUM)
- `docs/entry_exit_edge_external_labeling_literature_review_20260819.md` — 후보 선정 근거,
  A.2절에서 이 논문을 다룸
- `docs/label_methodology_survey_20260815.md` — 40+ 선행 라벨 방법론 메타발견
- 메모리: `eth_tabm_label_logic_retest_initiative_20260819`,
  `eth_btc_regime_shift_reopening_candidates_20260819`(약세장 방향 교차확인)
