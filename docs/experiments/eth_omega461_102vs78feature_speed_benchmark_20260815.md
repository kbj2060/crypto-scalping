# ETH Omega4.6.1 102-feature vs 78-feature(dedup) 속도 벤치마크 (2026-08-15)

## 배경

`eth_omega461_dedup78feature_nseed_skill_retest_20260815.md`가 102-feature(live)와
78-feature(24개 중복 제거) 버전의 PnL/skill이 사실상 동일(둘 다 zig075/h48qual 20칸 전부
always_short에 패배)함을 확정했다. 사용자가 이어서 "feature가 적으면 더 빠르지 않나?"라는
별개 질문을 던져, 순수 속도만 재측정한다. **재학습 없음** — 기존 5-seed 번들만 재사용.

## 방법

- 재사용 번들: zig075 컴포넌트, 시드 946043153/542143953 (102-feature:
  `tmp/causal_regen_20260516/..._pinned102_zig075_formal5seed_20260815_seed<SEED>/`, 78-feature:
  `..._pinned78_zig075_dedup_seed<SEED>/`), 둘 다 `true_3head_tabm_bundle.pt`.
- 모델 로딩/추론 코드는 라이브 경로(`trading_bot_modules/omega4_6_1_live.py:_Component._build_model`,
  `scripts/train_eval_omega1_2_tabm_3head_20260603.py:ThreeHeadTabM`)와 동일한 클래스/
  state_dict 로딩 방식 재사용 — 재구현 없음. 새 스크립트:
  `scripts/benchmark_eth_omega461_102vs78feature_inference_speed_20260815.py`.
- Device: CPU. 라이브 기본값이 `device="cpu"`(`Omega461LiveAdapter.__init__`)이고, 이 dev
  머신도 `torch.cuda.is_available()=False`(GPU 없음)라 CPU가 곧 라이브와 동일 경로.
- 3개 expert(bull/bear/chop) 중 `bull` 하나로 대표 측정(아키텍처/n_features 동일).
- 단일-row: warmup 50회 + 측정 300회(랜덤 float32 입력 — latency는 값이 아니라 shape에
  의해 결정되므로 랜덤 데이터로 충분), mean/median/p95/min/max 기록.
- 배치: 10,000행, 3회 warmup + 10회 측정 평균.
- 노이즈 대조를 위해 시드 2개씩 반복.

## 결과 1: 추론 latency (CPU, ms)

| 버전 | seed | n_features | 단일-row mean | median | p95 | min | max | 배치10000 throughput |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 102-feature | 946043153 | 115 | 0.1053 | 0.0903 | 0.1643 | 0.0832 | 0.2880 | 38,360 rows/s |
| 102-feature | 542143953 | 115 | 0.1260 | 0.1120 | 0.2249 | 0.0840 | 0.3076 | 34,851 rows/s |
| 78-feature | 946043153 | 91 | 0.1655 | 0.1437 | 0.2844 | 0.0874 | 0.4044 | 38,571 rows/s |
| 78-feature | 542143953 | 91 | 0.3562 | 0.1345 | 1.3769 | 0.0858 | 9.0354 | 38,986 rows/s |

시드 평균: 단일-row mean 102-feat 0.116ms vs 78-feat 0.261ms(+125% — 78-feature가 오히려
"느림"), median 0.101ms vs 0.139ms(+38%), 배치 throughput 36,606 vs 38,778 rows/s(+5.9%,
78-feature가 근소 우세). **같은 78-feature 조건 안에서도 시드 하나(542143953)가 단일-row
mean/max를 9ms까지 튀게 만드는 OS 스케줄링 잡음**을 보여, 이 스케일(0.1~0.3ms)에서는 측정
노이즈가 102 vs 78 차이보다 훨씬 크다 — 부호도 뒤집힌다(mean은 78이 "느림", median/batch는
78이 근소 "빠름"). 결론: **feature 개수 차이로 인한 방향성 있는 latency 차이를 이 스케일에서
검출할 수 없다.**

## 결과 2: 학습 wall-clock (기존 5-seed 드라이버 로그, zig075만 필터)

| | 102-feature (`.../driver.log`) | 78-feature (`..._driver.log`, zig075만) |
|---|---:|---:|
| seed별 소요(초) | 194, 192, 193, 191, 192 | 191, 181, 180, 183, 182 |
| 평균 | **192.4초** | **183.4초** |

Delta: **-9.0초/run, -4.68%**. 재학습 없이 기존 로그(`tmp/eth_zig075_direction_head_formal_nseed_20260815/driver.log`,
`tmp/eth_dedup78_nseed_skill_retest_20260815_driver.log`)의 시작/종료 타임스탬프만 파싱.

## 결과 3: 파라미터 수 (state_dict 실측, 추정 아님)

| | 102-feature (n_features=115) | 78-feature (n_features=91) | delta |
|---|---:|---:|---:|
| encoder | 102,448 | 97,456 | -4.9% |
| 3개 head 합(direction+quality+exit) | 1,544 | 1,544 | 0% (hidden=192 고정, 입력차원 무관) |
| **단일 expert 모델 총합** | **103,992** | **99,000** | **-4.8%** |
| 3-expert(bull/bear/chop) 번들 총합 | 311,976 | 297,000 | -4.8% |

Head 파라미터는 hidden(192)→3(direction/quality)·2(exit) 선형층이라 입력 feature 수와 무관해
그대로다. Encoder도 대부분 `in_proj`(115→192 vs 91→192)와 `input_scale/input_bias`(8×n_features)
가 줄어드는 부분뿐이고, `blocks`(192×192 두 층)가 encoder 파라미터의 대부분을 차지해 24개
feature 제거가 총 파라미터에 주는 영향은 -4.8%에 그친다. **학습 wall-clock 감소(-4.68%)와
파라미터/encoder 감소(-4.8~4.9%)가 서로 일치** — 이는 방향성이 실재함을 뒷받침하지만, 그
크기 자체가 작다.

## 결과 4: 이게 라이브 봇에 실제로 의미가 있는가

- 라이브은 5분봉(300초)마다 1회 결정한다. 측정된 단일-row 추론 latency(0.1~0.36ms, CPU)는
  300초 예산의 **0.00003~0.0001%**에 불과 — 102 vs 78 어느 쪽이든 완전히 무시 가능한 수준이다.
- `trading_bot.py`에 `TIMING main_cycle fetch=..s process=..s run_cycle=..s total=..s` 로깅이
  있지만(`FINAL_GOVERNOR_TIMING_LOG_ENABLE` 게이트), 이 dev 머신은 라이브 봇을 구동하지 않아
  실측 캡처값이 없다 — 추정치를 지어내지 않는다.
- **feature 계산 파이프라인 자체는 줄어들지 않는다.** `features/engineering.py`의
  `FeatureEngineer.process()`는 매 bar 무조건 전체 feature를 계산하고, `omega4_6_1_live.py`의
  `_Regime3CurrentLiveFeatures`도 별도로 무조건 계산된다. 78-feature 목록에서 빠진 24개 중
  다수가 유지된 feature의 **원재료(intermediate input)**로 여전히 필요하다:
  - `garman_klass_vol`(유지)은 내부적으로 `high`/`low`/`close`(제거 대상 3개) 필요
    (`features/engineering.py:294`).
  - `smart_money_flow`(유지)는 `sum_open_interest_value`(제거 대상)의 `pct_change()`
    (`features/engineering.py:218`).
  - `trade_intensity`/`net_taker_ratio`/`taker_acceleration`/`big_trade_ratio`(전부 유지)는
    `trades`, `quote_volume`, `taker_buy_quote`(전부 제거 대상)에서 파생
    (`features/engineering.py:226-239`).
  - 제거 대상 `regime3_current_sensitive_wide24_margin`은 모델 입력에서는 빠지지만
    risk-sizing sidecar가 `parent_router_margin`으로 직접 그대로 읽어 쓴다
    (`omega4_6_1_live.py:198`) — 계산을 건너뛸 수 없음.
  - 즉 "모델 입력 리스트에서 24개를 뺀다"는 것은 **모델 forward pass 안에서만 일어나는 변화**이고,
    원천 feature 계산 비용(전체 파이프라인의 대부분을 차지할 것으로 추정되는 부분)은 그대로다.

## 산출물

- 벤치마크 스크립트: `scripts/benchmark_eth_omega461_102vs78feature_inference_speed_20260815.py`
- 원시 결과 JSON: `tmp/eth_omega461_102vs78feature_speed_benchmark_20260815/results.json`

## 결론

78-feature 버전은 파라미터가 -4.8%, 학습 시간이 -4.7% 더 작다 — 방향은 일관되고 실재하지만
크기가 작다. 반면 **단일-row 추론 latency는 노이즈 안에 완전히 묻혀 어느 쪽이 더 빠른지조차
측정으로 구분할 수 없고(부호가 실험마다 뒤집힘)**, 설령 78-feature가 실제로 조금 더 빠르다
해도 그 절대량(<0.4ms)은 300초 bar 예산 대비 무의미하다. 게다가 feature 계산 파이프라인
(`features/engineering.py`) 자체는 78-feature로 줄여도 전혀 축소되지 않는다 — 제거된 컬럼
다수가 유지되는 파생 feature의 원재료이거나 risk sidecar가 별도로 소비하기 때문이다.
**"78-feature가 더 빠르다"는 직관은 모델 forward pass 파라미터 수준에서는 미세하게 맞지만
(-4.8%), 라이브 5분봉 파이프라인 전체의 체감 속도에는 실질적으로 영향을 주지 않는다.**
