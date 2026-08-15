# 이벤트 기반 라벨 생성 엔진 설계 (2026-08-15)

## 배경

기존 라벨 스크립트들(`scripts/build_*label*.py`, `scripts/build_*barrier*.py` 등)은
대부분 특정 자산·특정 실험 하나를 위해 매직넘버(고정 %, 고정 bar 수)를 하드코딩한
1회성 스크립트다. 이 문서는 오디세이 등 특정 서브프로젝트에 종속되지 않고, 금융 ML
라벨링 문헌에서 검증된 기법들을 하나로 결합한 범용 라벨 생성 로직을
`core/event_label_engine.py`에 새로 구현한 결과를 정리한다.

기존 `core/causal_event_labels.py`, `utils/generate_oracle_labels.py`와는 독립적인
구현이다 — 대체가 아니라, 문헌 원리에서 새로 설계한 별도 엔진이며 두 방식을 실제
모델 실험에서 비교해볼 수 있도록 나란히 존재한다.

## 설계 원칙 및 근거 문헌

| # | 구성요소 | 핵심 아이디어 | 근거 |
|---|---|---|---|
| 1 | 동적 변동성 배리어 | 배리어 폭 = `mult * vol`, vol은 EWMA(log-return) / ATR / **`return_dispersion`**(N-bar 누적수익 표준편차, 2026-08-15 추가) 중 선택 — 고정 % 대신 로컬 변동성에 비례, 레짐 변화에 자동 적응 | Lopez de Prado (2018); `return_dispersion`은 이 저장소에서 실제 진단된 "단일-bar ATR 노이즈" 버그(BTC race 라벨·zigzag corrected-vol 리빌드, 두 곳에서 독립 확인)를 반영 — [label_methodology_survey_20260815.md](label_methodology_survey_20260815.md) 참고 |
| 2 | 이벤트 샘플링 | 매 bar 대신 CUSUM 필터 또는 Directional-Change(intrinsic time)로 "정보량 있는" 시점만 샘플링 | AFML Ch.2.5.2.1 (CUSUM); Tsang et al., DC intrinsic time; crypto 적용: Razmi & Barak, *Adaptive Crypto Trading Using Directional Change and Meta-Learning* (2024/2025) |
| 3 | Triple-Barrier | PT/SL(수평)+max-hold(수직) 3중 배리어, 먼저 닿는 배리어로 라벨 결정. 표준 3-way와 메타라벨링(이진)을 side 인자 하나로 통합 | Lopez de Prado (2018); *Algorithmic crypto trading using information-driven bars, triple barrier labeling and deep learning*, Financial Innovation (2025) — BTC/ETH, CUSUM+triple-barrier가 next-bar 예측 대비 거래비용 반영 후에도 일관되게 우수 |
| 4 | Trend-Scanning | 여러 전방구간에 OLS를 적합, `\|t-value\|` 최대 구간을 선택 — 고정 보유기간을 미리 정하지 않고 통계적으로 가장 뚜렷한 추세를 스스로 찾음 | Lopez de Prado, AFML 강의노트 / *Machine Learning for Asset Managers* |
| 5 | 메타라벨링 | 1차(side: 방향)와 2차(그 방향이 맞을 확률/신뢰도)를 분리 — "언제 내 신호를 믿을지"를 별도로 학습 | Lopez de Prado (2017, Guggenheim/Cornell); 2025 BTC/ETH 딥러닝 연구에서 평균 Sharpe 0.48(16개 자산, 2000–2025) 보고 |
| 6 | 표본 uniqueness 가중치 | 배리어 구간이 겹치는 라벨은 서로 독립이 아님 → concurrency 기반 가중치(1/동시발생수) + 절대수익 기여도 가중치로 보정 | AFML Ch.4 (Sample Uniqueness, Sequential Bootstrap) |
| 7 | Purged K-Fold + Embargo | 배리어 구간이 test와 겹치는 train 표본 제거 + test 이후 embargo 구간 추가 제외 — 라벨이 미래로 걸쳐 있어 생기는 CV 누수 방지 | AFML Ch.7 |

### 의도적으로 구현하지 않은 기법

- **GA/베이지안 최적화로 pt/sl/max_hold 탐색** (*Enhanced Genetic-Algorithm-Driven Triple
  Barrier Labeling*, MDPI *Mathematics* 2024, crypto pair trading) — 핵심 동기(매직넘버
  배제)만 `calibrate_barriers()`의 단순 grid search로 반영. 이 저장소 규모에서 GA까지
  도입할 정도로 배리어 탐색이 병목이라는 근거가 아직 없어 보류.
- **Multi-scale Granger-causality + MAML 적응형 라벨링** (AEDL, *Adaptive Event-Driven
  Labeling*, MDPI *Applied Sciences* 2025) — 프레임워크 자체가 별도 연구 과제 수준의
  복잡도라 이번 범위에서 제외. 필요해지면 이 엔진의 `side` 산출 단계를 교체하는 형태로
  얹을 수 있다.

## 아키텍처

`core/event_label_engine.py` 단일 모듈, 의존성은 `numpy`/`pandas`/`numba`(전부 이미 설치됨).
입력은 `columns=[timestamp, open, high, low, close, volume]` OHLCV DataFrame 하나뿐이며
자산·타임프레임에 무관하다.

```
generate_labels(df, LabelEngineConfig, side=None) -> DataFrame
  ├─ ewma_volatility / atr_volatility          변동성 추정
  ├─ cusum_filter / directional_change_events   이벤트 샘플링 (numba)
  ├─ apply_triple_barrier                       표준 3-way 또는 메타(0/1) 라벨 (numba)
  ├─ trend_scanning_labels                      연속형 추세 라벨 (numba)
  └─ sample_uniqueness_weights + return_attribution_weights → weight 컬럼

calibrate_barriers(df, event_idx, vol, ...) -> TripleBarrierConfig   # grid search
purged_kfold_splits(event_idx, t1_idx, n_bars, ...) -> Iterator[(train_mask, test_mask)]
```

출력 라벨 DataFrame 컬럼: `event_idx, event_time, t1_idx, t1_time, bars_held, side,
touch_type, realized_ret, vol_at_entry, label, trend_tstat, trend_slope, trend_horizon,
weight_uniqueness, weight_return_attr, weight`.

## 실데이터 검증 결과

`data/eth_5m_1year.csv` (224,353 bar, 2023-12-31 ~ 2026-02-17, 실제 5분봉) 전체에 대해
`python3 core/event_label_engine.py` 실행:

| event_method | 이벤트 수 | 실행시간 | 라벨 분포(−1/+1) | 평균 보유 | timeout 비율 | uniqueness 가중치 평균 |
|---|---|---|---|---|---|---|
| cusum | 82,013 | 1.75s | 0.505 / 0.495 | 6.0 bar | 0.2% | 0.420 |
| directional_change | 17,475 | 0.37s | 0.508 / 0.492 | 6.5 bar | 0.4% | 0.823 |
| all_bars | 224,332 | 1.52s | 0.502 / 0.498 | 6.8 bar | 0.5% | 0.166 |

- 세 샘플링 방식 모두 pt_mult=sl_mult=2.0에서 라벨이 거의 완벽히 50/50으로 갈리고
  timeout(방향성 없이 시간초과)은 1% 미만 — 배리어 캘리브레이션이 합리적임을 시사.
  (참고: 이 대칭성은 배리어 자체가 대칭이기 때문에 나오는 정상적인 결과이며, 방향
  예측이 쉽다는 의미가 아니다 — 표준 3-way 라벨은 side 정보 없이 "어느 배리어가
  먼저 닿았는가"만 묻는다.)
- `all_bars`(매 bar 라벨링)는 uniqueness 가중치 평균이 0.166으로 가장 낮음 — 배리어
  구간이 극심하게 겹친다는 뜻이며, 바로 이 문제 때문에 CUSUM/DC 같은 이벤트 샘플링이
  필요하다는 문헌의 주장이 이 데이터에서도 정량적으로 확인됨.
- 메타라벨링 2-pass 데모(1차 CUSUM+trend-scan → `\|t\|>1.0`인 이벤트만 side 확정 → 2차
  메타라벨): 82,013개 중 82,010개에서 side 확정, 적중률(label==1) 56.6% — 표준
  3-way의 baseline(약 49.5~50.5%)보다 높아 trend-scan side가 완전 무작위는 아님을
  보여주지만, 이 수치 자체를 실거래 엣지로 해석하지 않는다(비용/슬리피지 미반영,
  단일 (pt,sl,horizon) 설정 기준 — 저장소의 Fresh-Forward 규칙에 따라 향후 실제 모델
  적용 시 별도로 causal walk-forward 검증 필요).
- `calibrate_barriers()`는 이 CUSUM 이벤트셋에서 `pt_mult=1.5, sl_mult=1.5,
  max_hold=24`를 최적으로 선택.
- `purged_kfold_splits()` 5-fold 실행: fold 0–3은 654~859개 표본을 purge(주로 test
  직후 embargo 구간 때문), 마지막 fold는 0개 — 마지막 fold는 뒤에 이어지는 fold가
  없어 embargo 대상 자체가 없으므로 정상 동작(버그 아님, 직접 재현해 확인함).

## 사용법

```python
from core.event_label_engine import generate_labels, LabelEngineConfig

# 표준 3-way 라벨
labels = generate_labels(df, LabelEngineConfig(event_method='cusum'))

# 메타라벨링 2-pass
primary = generate_labels(df, cfg)
side = np.sign(primary['trend_tstat']).where(primary['trend_tstat'].abs() > 1.0, 0)
meta = generate_labels(df, cfg, side=side)
```

## 참고문헌

- Lopez de Prado, *Advances in Financial Machine Learning* (2018) — Triple-Barrier / Meta-Labeling / Sample Uniqueness / Purged K-Fold 원전.
  [mlfinpy Labelling docs](https://mlfinpy.readthedocs.io/en/latest/Labelling.html) ·
  [Triple Barrier 해설](https://quantstrategy.io/blog/the-triple-barrier-method-revolutionizing-how-we-label/)
- [Trend Scanning — mlfinlab 문서](https://random-docs.readthedocs.io/en/latest/implementations/labeling_trend_scanning.html)
- [Meta-Labeling — Wikipedia](https://en.wikipedia.org/wiki/Meta-Labeling)
- *Algorithmic crypto trading using information-driven bars, triple barrier labeling and deep learning*, Financial Innovation (2025).
  [ideas.repec.org](https://ideas.repec.org/a/spr/fininn/v11y2025i1d10.1186_s40854-025-00866-w.html)
- [Directional-change intrinsic time — Wikipedia](https://en.wikipedia.org/wiki/Directional-change_intrinsic_time) ·
  [A Modern Paradigm for Algorithmic Trading (2025)](https://arxiv.org/pdf/2501.06032) ·
  [Razmi & Barak, Adaptive Crypto Trading Using Directional Change and Meta-Learning (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5017215)
- *Enhanced Genetic-Algorithm-Driven Triple Barrier Labeling Method ... Cryptocurrency Markets*, Mathematics/MDPI (2024).
  [mdpi.com](https://www.mdpi.com/2227-7390/12/5/780)
- *Adaptive Event-Driven Labeling: Multi-Scale Causal Framework with Meta-Learning*, Applied Sciences/MDPI (2025).
  [mdpi.com](https://www.mdpi.com/2076-3417/15/24/13204)
- [Bagging in Financial Machine Learning: Sequential Bootstrapping — Hudson & Thames](https://hudsonthames.org/bagging-in-financial-machine-learning-sequential-bootstrapping-python/)
- [Machine Learning Blueprint (Part 4): Label Concurrency — MQL5](https://www.mql5.com/en/articles/19850)

## 코드와의 관계

이 문서에 기록한 모든 수치는 `core/event_label_engine.py`를 직접 실행해 얻은 결과다
(스크립트 하단 `if __name__ == '__main__':` 블록 참고). 저장된 원장이나 과거 replay가
아닌, 이 모듈이 그 자리에서 계산한 값이다.

## 저장소 기존 라벨 히스토리와의 대조 (2026-08-15 추가)

이 엔진을 설계한 뒤, 저장소에 이미 존재하는 40개 이상의 라벨 방법론(triple-barrier/
zigzag/meta-labeling/trend-scan/DP-oracle 전 계열)을 배경 조사로 훑어 이 설계와
대조했다. 전체 결과와 방법론별 색인은 [label_methodology_survey_20260815.md](label_methodology_survey_20260815.md)에
정리했다. 요지:

- **위 "동적 변동성 배리어"(`return_dispersion`)와 "메타라벨링"(OOF 경고), 그리고
  `trend_scanning_labels()`의 "실시간 feature로 쓰지 말 것" 경고**는 전부 이 조사에서
  나온 실제 버그 사례를 근거로 이번에 추가/보강한 것이다.
- 조사된 40개 이상의 방법론 중 방향/진입 alpha를 검증까지 통과한 사례는 **없다** — 라벨
  자체는 학습 가능해도(예: 분류기 AUC 0.9+) 수익화로 이어지지 않는 패턴이 반복
  확인됐고, 진단된 원인은 거의 항상 배리어 공식이 아니라 feature information content
  부재였다. 이 엔진은 그 문제를 해결하지 않는다 — 더 나은 라벨 구성·표본가중치·CV
  위생을 제공할 뿐이다.
- `core/causal_event_labels.py`는 실사용 소비자가 2개뿐이고 둘 다 실패해, 이 엔진을
  그 확장이 아니라 독립 구현으로 설계한 결정이 사후적으로도 타당했다.
- 실모델 검증: h48qual TabM parent의 `direction_head` out-of-sample 예측(train 제외,
  validation+oos만)을 이 엔진의 메타라벨링 경로에 넣어본 결과 적중률 49.13%(동전던지기
  수준) — 엔진의 메타라벨링 배관이 실제 모델 출력으로 정상 동작함을 확인했을 뿐,
  h48qual에 새 방향 edge가 있다는 뜻은 아니다(기존 결론과 일치). 스크립트:
  `scripts/diagnose_eth_h48qual_dirhead_metalabel_via_event_label_engine_20260815.py`.
