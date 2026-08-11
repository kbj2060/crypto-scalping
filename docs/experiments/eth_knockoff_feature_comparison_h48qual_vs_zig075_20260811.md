# ETH h48qual vs zig075 — Knockoff 기반 피쳐선택 교차비교 (2026-08-11)

h48qual(품질/배리어-레이스) 컴포넌트 튜닝 세션에서, zig075(지그재그 방향) 컴포넌트가 LGBM 재학습 +
단변량 Mann-Whitney AUC 스크리닝(`dir_edge>=0.02` 고정 임계값, FDR 통제 없음, 중복성 체크 없음)으로
63개 피쳐 생존을 주장한 것과, 우리 h48qual 쪽 mRMR 결과를 공정 비교하기 위해 Model-X Knockoff
Filter(Candès/Fan/Janson/Lv 2018, JRSS-B)를 양쪽 후보 풀에 동일하게 적용했다.

## 방법

- `knockpy.KnockoffFilter(fstat='randomforest', ksampler='gaussian')` — FDR 상한을 통계적으로
  보장하면서 비선형 관계도 포착. 셔플-null보다 엄격한 대조군(Ledoit-Wolf 공분산 추정으로 진짜
  피쳐간 상관구조를 보존한 knockoff 변수를 만듦).
- zig075의 원래 프레이밍(tradeability: action!=0 / direction: long vs short)을 그대로 재사용해
  동일 서브태스크로 비교.
- zig075 풀(138개)은 그쪽 소스 파일과 TRAIN 윈도우(2024-06-01~2025-06-30)를 그대로 재구성.
  h48qual 풀(145개)은 우리 쪽 TRAIN(<2025-10-01), SL=0.55% 라벨 기준.

## 결과 (FDR-통제 게이트, 8개 조합 전부)

| | fdr=0.10 | fdr=0.20 |
|---|---|---|
| h48qual-tradeability | 32/145 | 35/145 |
| h48qual-direction | 38/145 | 46/145 |
| zig075-tradeability | 0/138 | 7/138 |
| zig075-direction | 34/138 | 36/138 |

**핵심**: zig075의 "63개 생존" 주장이 실제로 테스트한 타겟은 위 표의 zig075-direction과 동일하다
(zigzag long vs short). FDR 통제 + 중복성 인지 방법을 같은 풀·같은 라벨에 적용하면 **34~36개만
생존** — 주장한 63개의 약 55%. 나머지 ~45%는 (a) FDR 미통제 상태에서의 거짓양성이거나 (b) 상관된
변형 피쳐 중 중복분일 가능성이 높다 (다만 knockoff도 `vwap_dist_24/96/288`처럼 상관 변형을 여러 개
같이 살려두는 경우가 있어, 순수 중복성보다 FDR 미통제 쪽이 더 큰 원인으로 보인다).

부가 발견: zig075의 "tradeability"(스윙포인트 여부) 신호는 fdr=0.10에서 0/138 — 이 라벨은 현재
기술적 피쳐 우주로는 거의 예측 불가능해 보인다. h48qual의 tradeability(배리어-레이스 품질)는 같은
기준에서 32/145(22%)가 생존해 뚜렷한 대조를 이룬다.

## zig075 쪽 knockoff 게이트 → mRMR 압축 (참고용, 우리가 대신 수행)

knockoff 게이트(tradeability@0.20 ∪ direction@0.20 = 39개) → mRMR 랭킹 top20 → |r|>0.5 하드
중복제거 → **7개 최종 생존**:

```
cvp_regime, funding_roc_288, ou_halflife, vwap_dist_24, funding_roc_48,
breakout_strength, regime3_current_sensitive_wide24_chop_prob
```

이 중 `cvp_regime, funding_roc_288, ou_halflife, breakout_strength` 4개는 h48qual FINAL13과도
겹친다 — 완전히 다른 라벨·게이트·TRAIN 윈도우에서 나온 결과인데도 수렴한다는 건 이 4개가 라벨별
노이즈가 아니라 실제 전이 가능한 신호라는 근거다.

## zig075 세션에 참고 제안

1. **중복 경고**: 지금 138풀에 `whale_retail_ratio`, `squeeze_power` 등 raw(비-detrend) 형태로
   들어있는데, 우리 쪽 h48qual 분석에서 이 계열 9개 피쳐(`funding_pressure`, `m7_vae_error`,
   `last_funding_rate`, `squeeze_power`, `long_squeeze_risk`, `funding_abs`, `whale_retail_ratio`,
   `count_long_short_ratio`, `sum_toptrader_long_short_ratio`)가 raw 상태로는 가격추세와
   corr(close) 0.25~0.62 수준으로 오염돼 있다는 걸 확인했다. `funding_pressure`→diff1,
   나머지는 rolling(288) 평균제거(dt288) 버전을 쓰면 이 오염이 사라진다.
2. **FDR 미통제 우려**: 고정 임계값(`dir_edge>=0.02`) 대신 knockoff 같은 FDR-통제 방법을 최소
   교차검증으로라도 돌려보길 권한다 — 63개 중 실제로 몇 개가 통계적으로 방어 가능한지 이 리포트가
   1차 답을 준다(약 34~36개).
3. **중복성 체크 부재**: 63개 리스트에 상관 0.8~0.99대 근접-중복 쌍이 섞여있을 가능성이 높다
   (h48qual 쪽에서도 `funding_roc_288`/`funding_pressure_diff1`이 r=0.996로 사실상 동일 피쳐였던
   걸 이번에 처음 발견했다). 배포용 최종 셋을 만들 때는 상관행렬 감사를 한 번 거치는 걸 권한다.

## 관련 산출물

`knockoff_comparison.py`, `knockoff_resume.py`, `knockoff_h48qual_names.py`,
`zig075_knockoff_mrmr.py`, `h48qual_knockoff_mrmr.py` — h48qual 세션의 scratchpad에 있다.
scratchpad는 세션별로 격리돼 있어 zig075 세션에서 직접 열람은 안 되지만, 스크립트 자체는
`data/splits/year_oos/eth_features_2024_2026_analysis.csv` 등 리포지토리 파일과 zig075 세션이
쓴 것과 동일한 소스(`data/ensemble/supervised/eth_regime3_current_hmm_jmredesign_20260810_*`,
`tmp/causal_regen_20260516/zigzag_action_labels_20260531/*`)만 참조하므로, 필요하면 이 문서의
로직 설명을 보고 zig075 세션 쪽에서 동일하게 재현할 수 있다.
