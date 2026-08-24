# ETH 1·2·3등급 특수데이터 결합 방향 모델 (tier123 combined direction model)

- 일자: 2026-08-24
- 상태: 사전등록 → 실행 (TRAIN/VAL만, 동결창 미터치)
- 요청: 사용자 — "1,2,3 등급 데이터들로 매매 방향을 결정하는 공식이나 학습모델 만들어줘"
- registry id: `eth_tier123_combined_direction_model_20260824`

## 1. 배경과 질문

2026-08-23 특수데이터 인벤토리에서 방향예측 관점 등급을 매겼다:
- **1등급** = 통계신호 확인(수익화 미달): `microstructure_1m`의 taker-flow 계열
- **2등급** = 미결론/수집중: GEX, OI·롱숏비, BTC/SOL 마이크로, raw L2, 펀딩스프레드
- **3등급** = 개별 기각: fear_greed, basis, ETF플로우, 스테이블코인, TSMOM, tail_risk

기존 확정 증거(`eth_trend_dl_multivariate_probe_20260823`):
- h=12(1h): 결합이득 0 — taker_buy_ratio 단일피쳐가 전부 (ridge 0.98x)
- h=48(4h): **ridge 선형결합만 생존** (IC +0.0747 = 최고단일 1.63x, z=3.33) — 계수 실체는
  "리테일 순매수 + 큐 안정 → 4h 지속". LGBM은 음수IC 붕괴, MLP는 300파라미터조차 신호 85% 파괴
  (용량 그래디언트 4단계 완전단조). 경제성은 gross 1.5~3bp/트레이드로 전패.

**이번 질문**: 위 결합에 2등급(OI·롱숏비 메트릭스)과 3등급(fear_greed)을 추가하면
(a) IC가 1등급-단독 ridge보다 더 오르는가, (b) 경제성이 3개 비용 시나리오
(10bp 테이커관행 / **6.2bp 메이커실측**(08-24 1차) / 2.5bp 낙관=USDC-M 0% 가정)에서
어디까지 가는가. 그리고 사용자가 쓸 수 있는 **명시적 공식**(계수 bp/1σ 표)을 산출한다.

## 2. 데이터 소스 (실측 확인 완료, 2026-08-24)

| 등급 | 소스 | 커버리지 | 비고 |
|---|---|---|---|
| 1 | `data/live/microstructure.duckdb::microstructure_1m` | 2026-05-03~08-24 02:29 | **오늘 서버 재동기화본**(dev 정체 함정 해소 직후). ts는 KST tz-aware → `tz_convert('UTC')` 필수 |
| 2 | `data/TOTAL_ETHUSDT_metrics_2024_2026.csv` | 2024-01-01~2026-08-22, 5m, 99.99% | 08-23 무결성 대수술 검증 참조본, +5분 종료라벨 보정 완료. 자체수집 `oi_lsratio.duckdb`(08-22~, 2일)를 동일 필드로 대체 |
| 3 | alternative.me F&G API (`limit=0` 전체 히스토리) | 2018~현재, 일별 | 08-12 백필 검증 스크립트와 동일 규약(그날 값은 그날 자정 UTC부터 유효 = causal) |
| — | `binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv` | ~2026-08-20, UTC-naive | 타깃 수익률·가격 소스(패널 스크린과 동일) |

**제외(사유 명기)**: tail_risk_1m(07-18 이전 청산피드 결함 + 청산축 ≥09-15 게이트 사전등록
오염 방지), GEX(Tier0 문턱 미달), cross_exchange_funding_spread(108행),
raw L2/OFI(09-14 게이트), BTC/SOL microstructure(07-14 시작 — TRAIN 절반 상실, 후속 arm
후보로만 기록), basis/ETF/스테이블/주간TSMOM(개별 기각 + 5m 시계열 소스 부재),
whale_position_score(07-18 이전 100% NULL + 컨펌필터 결정적 기각).

## 3. 피쳐 정의 (16개)

롤링 z는 전부 **window=288(5m 기준 24h), `min_periods=274`(95% 허용)** — 같은 날
스캘핑 스크린에서 확인된 min_periods=window(완전무결 요구) 설계결함의 재발 방지.

- **1등급 (9, 기존 패널과 동일)**: obi, taker_buy_ratio, spoofing_score, nif_whale,
  nif_retail, shadow_toxicity_score, shadow_queue_collapse (원시값),
  eai_z, oi_delta_pct_z (z288)
- **2등급 (5, 메트릭스 자체 5m 그리드에서 변환 후 조인)**:
  - oi_chg_12 = sum_open_interest.pct_change(12) (1h OI 변화율)
  - top_acct_lsr_z = z288(count_toptrader_long_short_ratio)
  - top_pos_lsr_z = z288(sum_toptrader_long_short_ratio)
  - global_lsr_z = z288(count_long_short_ratio)
  - taker_vol_ratio_z = z288(sum_taker_long_short_vol_ratio)
- **3등급 (2, 08-12 규약)**: fng_value(레벨), fng_diff1(전일 대비 변화)

조인: klines bar_close_time(=timestamp+5m, UTC-naive) 기준 merge_asof backward —
micro tolerance 5m, metrics tolerance 30m(미게시 버킷은 직전값, 인과-안전), F&G는 당일
date 조인. 신규(2·3등급) 피쳐는 spearman(feature, close) **|ρ|≥0.5 오염 배제**(패널 관례).

## 4. 스플릿·모델·판정 기준 (실행 전 고정)

- **TRAIN 2026-05-03~07-31 / VAL 2026-08-01~08-16** (기존 프로브와 동일 경계).
  ⚠️ 이 VAL은 이미 다중 소비됨(패널·프로브·MLP arm·whale 게이트·스캘핑 스크린 ≥5회) —
  오늘 수치는 전부 **research/dev score**. 실판정은 **동결창 08-17~09-30, 09-30 이후
  단일터치**(오늘 절대 안 봄).
- **모델**: Ridge(alpha=1.0) 고정 — 프로브와 동일 HP, 튜닝 없음(VAL 추가소비 최소화).
  결정론적이라 시드 게이트 해당 없음(다중시드 평균 주장 아님).
- **Arm 구조**: (A) 1등급 9피쳐 = 프로브 재현 베이스라인 / (B) 1+2등급 14피쳐 /
  (C) 1+2+3등급 16피쳐 = 사용자 요청 본체. h={12, 48}, primary는 **arm C, h=48**.
- **통계 기준**: arm C h=48이 TRAIN/VAL IC 부호일치 AND VAL circular-shift |z|≥3
  (N_PERM=2000, 패널과 동일 널) AND **VAL IC ≥ arm A VAL IC**(등급 추가가 결합이득을
  실제로 만드는가 — 이게 이 실험의 고유 질문).
- **경제성**: 패널 economic_gate 구조 재사용 — score를 TRAIN std로 정규화한 score_z가
  |score_z|≥1일 때 sign(score) 방향 진입, h bar 비중첩 홀드, 왕복비용 {10, 6.2, 2.5}bp,
  각 시나리오에서 max(always_long, always_short) 대비 순증분 보고. 승격 관련 판단은
  6.2bp(실측) 기준.
- **공식 산출**: arm C h=48 계수를 bp/1σ 표로 제시 + 진입규칙 명문화.
- **신선도 가드(스크립트 내장)**: micro max ts ≥ 08-17, metrics max ≥ 08-17,
  F&G max ≥ 08-16 미달 시 즉시 abort — dev 사본 정체 함정의 제도화된 방지선.

성공/실패와 무관하게 registry 등록, 09-30 단일터치 계획 명기.

---

## 5. 결과 (2026-08-24 실행, TRAIN/VAL만 — 동결창 미터치)

스크립트: `scripts/research_eth_tier123_combined_direction_model_20260824.py`

### 5.0 실행 중 확인된 데이터 사실 2건

1. **microstructure_1m에는 5~6월 통짜 결측일 14일이 실재** (05-12~14, 05-16, 05-24,
   05-30, 06-06, 06-14~16, 06-19, 06-23~24, 07-26 등) — dev 정체가 아니라 서버 원본
   자체의 수집기 다운타임(145,646행/기대 161,289행=90.3%). 통짜 결측일 하나가 이후
   24h의 z-피쳐 워밍업까지 무효화해 TRAIN 조인트 커버리지 73%. 기존 "05-03부터 99.4~100%
   건강" 감사는 존재하는 행의 값-존재율 기준이었고 행 자체의 부재는 별개였음.
   커버리지 가드를 조인트 90% → 컬럼별 60%(파국 탐지)로 교정(정체 탐지는 max-ts assert가
   담당).
2. **fng_value(레벨)는 오염 배제 자동 발동** — TRAIN spearman(fng_value, close)=+0.87.
   F&G 레벨은 사실상 가격 추세의 재표현. fng_diff1(ρ=−0.056)만 잔존 → arm C는 15피쳐.
   2등급 5종은 전부 오염 통과(|ρ|≤0.089).

### 5.1 통계 결과 (circular-shift z, N=2000)

| arm | h | TRAIN IC (z) | VAL IC (z) | 판정 |
|---|---|---:|---:|---|
| A_tier1 (9) | 12 | +0.0479 (+5.51) | +0.0376 (+2.11) | — |
| A_tier1 (9) | 48 | +0.0337 (+3.29) | **+0.0190 (+0.90)** | 기존 프로브 수치 재현 실패 → §6 |
| B_tier12 (14) | 12 | +0.0757 (+4.82) | +0.0775 (+2.63) | — |
| B_tier12 (14) | 48 | +0.0818 (+2.72) | −0.0079 (−0.15) | 부호반전 |
| C_tier123 (15) | 12 | +0.0830 (+5.31) | +0.0775 (+2.52) | z<3 |
| **C_tier123 (15)** | **48** | **+0.1161 (+3.53)** | **+0.0017 (+0.03)** | **PRIMARY FAIL** |

**Primary(arm C, h=48): FAIL_STATS.** TRAIN IC가 arm A(+0.0337)→C(+0.1161)로 3.4배
부풀지만 VAL에서 완전 소멸(+0.0017) — 2등급 롱숏비 피쳐들이 TRAIN 구간(5~7월 상승장)의
레짐 특이 패턴에 과적합한 전형. h=48에서 등급 추가는 결합이득이 아니라 과적합만 만든다.

h=12는 상대적으로 나음: arm B/C VAL IC +0.0775가 arm A(+0.0376)의 2.1배로 부호일치
유지 — 그러나 z=2.52<3(사전등록 기준 미달)이고 아래 경제성도 실측비용에서 음수.

### 5.2 경제성 (진입 |score_z|≥1, h bar 비중첩, max(always) 대비 증분)

| arm/h | split | n | gross bp/tr | hit | inc@10bp | inc@6.2bp | inc@2.5bp |
|---|---|---:|---:|---:|---:|---:|---:|
| C h=48 | TRAIN | 302 | +16.97 | 55.3% | +4.79% | +16.27% | +27.44% |
| C h=48 | VAL | 52 | **−1.60** | 57.7% | −7.20% | **−5.22%** | −3.30% |
| C h=12 | TRAIN | 1095 | +4.85 | 53.1% | −71.40% | −29.79% | +10.72% |
| C h=12 | VAL | 216 | +5.08 | 55.1% | −11.68% | **−3.47%** | +4.52% |
| A h=48 | VAL | 75 | −3.06 | 42.7% | −10.96% | −8.11% | −5.34% |

실측 메이커비용 6.2bp 기준 전 조합 음수. h=12 arm C만 가상의 2.5bp(USDC-M 0% 프로모션
가정, 미확인)에서 +4.52% — 단 이 조합은 통계 기준(z≥3)도 미달이고 VAL은 다중소비
상태라 근거로 쓸 수 없음. 관찰 기록만.

### 5.3 산출된 공식 (arm C h=48, bp of 4h fwd return per +1σ — **검증 실패 딱지 포함**)

top_acct_lsr_z +18.21 / global_lsr_z −17.15 / fng_diff1 +9.11 / oi_chg_12 −4.57 /
taker_vol_ratio_z −3.13 / shadow_queue_collapse −2.77 / top_pos_lsr_z −2.07 /
nif_retail +1.74 / eai_z +1.68 / nif_whale +1.32 / taker_buy_ratio +1.15 /
toxicity +0.90 / obi +0.52 / oi_delta_pct_z +0.42 / spoofing +0.20
(절편 −5.45bp = TRAIN 드리프트, 방향규칙은 중심화 score 사용)

**이 공식을 지배하는 상위 2계수(상위트레이더 계좌 롱숏비↑, 글로벌 롱숏비↓)가 바로 VAL에서
붕괴한 성분이다** — 표는 사용자 요청에 따라 산출하되, 이 계수로 실매매하면 안 된다는 것이
본 실험의 실증 결과다.

## 6. 부수 발견 — 어제 추세 프로브의 h48 "통과"가 stale dev 사본 산물로 판명 (별도 정정 전파)

arm A가 프로브의 h48 ridge VAL IC +0.0747/z=3.33을 재현하지 못해 조사 → 프로브(08-23
실행)는 dev `microstructure.duckdb`의 130시간 동기화 공백(08-11 21:37~08-17) 상태에서
돌아 **VAL 16일 중 앞 11일만 평가**했던 것. 동일 스크립트를 완전본으로 재실행하면 h48
ridge **+0.0330/z=1.64(0.72x) → FAIL**. VAL을 공백 이전으로 클립하면 +0.0707/z=3.29로
구수치 복원(귀속 확정). MLP arm 재실행에서 용량 그래디언트("DL 비권고")는 재현.
정정 전문: [eth_candidate_trend_dl_multivariate_probe_20260823.md](eth_candidate_trend_dl_multivariate_probe_20260823.md) §정정.
같은 stale 사본으로 돌았던 1h/4h 패널 스크린도 재실행해 §7에 기록.

## 7. 최종 판정

- **REJECTED (사전등록 기준 FAIL_STATS)** — 1·2·3등급 결합은 h=48에서 TRAIN 과적합만
  생산, h=12에서 z<3 + 실측비용 미달. "약신호 결합으로 새 축 안 열림" 메타발견의 재확인이며,
  이번엔 이종 정보원(파생상품 포지셔닝 지표)을 더해도 마찬가지임을 보임.
- 09-30 동결창(08-17~09-30) 단일터치: 사전등록대로 arm C h=48 고정계수(§5.3, 결정론적
  재적합으로 재현 가능)를 1회 평가해 종결 확인 예정 — 단 VAL FAIL로 사전확률 극히 낮음.
  프로브 정정으로 취소된 구 h48 선형 생존체의 09-30 계획을 이것이 승계.
- 시드 게이트: ridge 결정론적 — 해당 없음(다중시드 평균 주장 아님).

