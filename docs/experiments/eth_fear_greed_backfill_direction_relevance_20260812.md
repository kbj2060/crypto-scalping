# ETH h48qual — Fear & Greed Index 과거 백필 방향 relevance 확인 (2026-08-12)

**⚠ 정정 노트 (2026-08-12, 후속 DefiLlama 실험에서 발견)**: 아래 MI 수치는 `mutual_info_classif`를
일별 forward-fill 데이터에 직접 적용한 것인데, 후속 실험(`eth_defillama_onchain_direction_relevance_20260812.md`)에서
이 추정량이 288-bar 중복 블록 구조 앞에서 **완전히 무관한 랜덤 daily 시리즈에도 실제 데이터와
거의 동일한 값을 반환하는 degenerate 현상**이 합성 데이터로 재현·확정됐다. 이 문서의 3개 MI값
(0.0191/0.0103/0.0447)은 서로 달라서 완전히 같은 케이스는 아니지만, 안전을 위해 **MI 수치는
참고용으로만 취급**하고 **아래 오염도 체크(Pearson/Spearman, 문제없음)와 LightGBM 홀드아웃
비교(문제없음, 결정적 근거)에 결론을 의존**할 것 — 결론 자체(부정 결과)는 홀드아웃 비교만으로도
이미 충분히 뒷받침되므로 바뀌지 않는다.

## 배경

신규 탐색 축 스카우팅 (a)-2 후보. alternative.me의 Fear & Greed Index는 무료 공개 API로
2018-02부터 일별 히스토리를 제공한다. 이 서브 프로젝트가 이미 F4-C 수집기로 F&G를 수집
중이지만(`scripts/run_f4c_altdata_collector.py`, 2026-08-10부터 라이브 전방향만) 그건 3개월
뒤 사전등록 이벤트 스터디로만 계획돼 있고, 과거 TRAIN/VAL/OOS 백필은 별개 — 다른 8개 신규
데이터소스 후보를 막았던 "라이브 duckdb가 2026-05 이후만 커버" 벽이 이 API엔 적용 안 된다.

## 방법

- 스크립트: `scripts/verify_eth_fear_greed_backfill_direction_relevance_20260812.py`
- `https://api.alternative.me/fng/?limit=0` 전체 히스토리(3,110일, 2018-02-01~2026-08-11)
  1회 호출, 일별 값을 5분봉에 as-of forward-fill 조인.
- 파생 2개(가격추세 오염 완화 표준 관례와 동일 정신): `fng_diff1`(전일대비), `fng_ma7_dev`
  (7일 이동평균 대비 편차).
- **표준 절차**: 신규 raw-level 피쳐는 학습 전 `corr(price)`/`corr(시간순번)` 오염도부터 확인
  (배제 기준 0.561) — 원값·두 파생 전부 체크.
- TRAIN(2024-06~2025-09)/VAL(2025-10~12)/OOS(2026-01~02)는 direction-only 재스크리닝과 동일한
  zig075 소스 패널 기준.
- 검증: 튜닝 없는 단일 LightGBM fit으로 FINAL12(패널가용 8개) 단독 vs +F&G(오염도 통과분) 대조.

## 결과

| 컬럼 | corr(price) | corr(시간순번) | MI(zigzag_action, TRAIN) | 판정 |
|---|---:|---:|---:|---|
| `fng_value`(원값) | +0.385 | +0.011 | 0.0191 | 통과(단 0.561 기준 대비 여유는 크지 않음) |
| `fng_diff1` | -0.019 | +0.025 | 0.0103 | 통과, 오염 거의 없음 |
| `fng_ma7_dev` | -0.043 | -0.005 | 0.0447 | 통과, 셋 중 MI 최고 |

**오염도는 셋 다 통과**하지만 **MI 자체가 낮다**(0.01~0.045, FINAL12 최상위 `cvp_regime`의
0.41과 비교하면 한 자릿수 작음) — 스카우팅 문서가 사전에 예상한 대로 일별 값이 5분봉 288개에
반복되는 구조적 정보량 한계가 그대로 나타남.

가벼운 홀드아웃 비교:

| 구성 | VAL balanced_acc | VAL macro_f1 | OOS balanced_acc | OOS macro_f1 |
|---|---:|---:|---:|---:|
| FINAL12(패널가용 8개) 단독 | 0.469 | 0.448 | 0.466 | 0.446 |
| FINAL12 + F&G(3개) | 0.461 | 0.440 | 0.461 | 0.437 |

**VAL/OOS 둘 다, 두 지표 다 오히려 소폭 악화**(-0.5~0.9pp) — F&G 추가가 도움이 안 되는 정도가
아니라 살짝 해가 되는 방향.

## 결론

**부정 결과, 스카우팅 문서의 사전 기대치(낮음)와 일치.** 오염도 체크는 통과했지만 relevance
자체가 약하고, 실제 홀드아웃에서는 오히려 소폭 악화를 보였다 — 일별 해상도 외부 지수를
5분봉 방향 예측에 단순 forward-fill로 조인하는 방식은 이 자산/타임프레임에서 추가 가치가
없다. 이 후보는 여기서 닫는다. 시간외 해상도가 더 높은 대체 심리지표(예: 소셜 볼륨, 펀딩
기반 유사지표)가 있다면 다른 후보지만, 이 API의 일별 F&G 자체는 재시도 근거가 약함.

## 산출물

`tmp/eth_fear_greed_backfill_20260812/` — `fear_greed_daily_raw.csv`,
`contamination_and_mi_report.json`, `holdout_comparison.json`.
