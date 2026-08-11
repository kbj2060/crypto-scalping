# ETH h48qual — quality_head 신규 데이터소스 리서치 (2026-08-11, 구현 전 리서치 단계)

**구현 전 리서치 단계 문서** — 여기 있는 어떤 데이터소스도 아직 수집·조인·학습 파이프라인에
연결되지 않았다. 전부 아이디어와 검증 계획이다. 학습이나 구현으로 넘어가기 전에, "제안
우선순위" 절이 권하는 순서(가장 싼 진단부터, 재학습 없이 죽이거나 살리기)를 따르길 권한다.
`docs/model_contracts/odyssey_eth_h48qual_corrected_tabm_20260811_contract.md`은 이 문서로 인해
변경되지 않았다.

## 질문

계약 문서와 이번 세션에 완료된 4갈래 독립 진단(호라이즌 재설계, always-short 대조, 분류→회귀
전환, 앙상블 불일치 순위상관)이 전부 같은 결론으로 수렴했다: `quality_head`(타겟
`h48_conservative`, TP/SL 트리플배리어)가 게이트 통과 후 숏으로 쏠리는 문제는 스칼라 추출
방법(온도보정·isotonic·회귀·앙상블 불일치)의 문제가 아니라, **현재 확보한 ~201개 컬럼 리서치/
프로덕션 피쳐 패널 자체에 "이미 direction_head가 고른 이 zigzag_action 진입이 SL 전에 TP를
칠지"를 예측할 신호가 없다**는 것. 회귀 전환 시도(`eth_h48qual_quality_head_regression_
conversion_attempt_20260811.md`)는 12개(FINAL12)와 201개(REL11 재스크리닝) 양쪽에서 GBM 홀드아웃
R²가 0 근처였다 — 방법론이 아니라 정보량 자체의 문제라는 뜻이다.

이 문서는 그 결론을 받아들이고 "같은 패널에 다른 방법을 쓰는" 대신 **패널 자체를 넓히는** 방향의
후보를 조사한다. `quality_scalar_alternatives_research` 문서(스칼라 추출 방법 축)와는 직교하는
별도 축이다 — 거기서 유일하게 남은 candidate B(evidential/Dirichlet)도 "같은 데이터에 다른 loss"일
뿐이라 이 문서의 결론과 무관하게 그 자체로는 이 문제를 못 푼다.

## 그라운딩 — 이미 시도되고 죽은 라인 (레포 감사)

### quality_head 자체에 대한 4갈래 진단 (이번 세션, 전부 종료 — 재확인 불필요)

1. **호라이즌 스윕**(`eth_h48qual_quality_horizon_sweep_20260811.md`) — 48→384bar. 라벨 내부
   일관성은 개선(방향일치 89.5%→92.1%, specificity 34.2%→65.1%)했지만 게이트 편향은 못 고침.
2. **always-short 대조**(`eth_h48qual_quality_trend_bias_h48orig_control_20260811.md` +
   `eth_h48qual_final12_h384_isolated_tuning_sweep_20260811.md`) — h48orig 5시드, h384 15시드,
   라이브 실제 가중치(102피쳐) 전부에서 always-short이 모델을 이김. direction_head 원본은 숏
   53~59%로 균형인데 게이트 통과 후 75~92%로 쏠림.
3. **분류→회귀 전환**(`eth_h48qual_quality_head_regression_conversion_attempt_20260811.md`) —
   오라클 게이트는 always-short을 15/15 시드로 압도(메커니즘 유효)하지만 FINAL12·REL11(201개 풀
   재스크리닝) 둘 다 GBM 홀드아웃 R²≈0. **이 문서가 근거로 삼는 핵심 증거**: 지금 피쳐 우주엔
   이 라벨과 실전에서 학습 가능한 관계가 없다.
4. **앙상블 불일치 순위상관**(`eth_h48qual_ensemble_disagreement_rank_correlation_20260811.md`) —
   TabM k=8 멤버 불일치(Depeweg MI 분해)도 실현 순수익률과 무상관(VAL/OOS p=0.51/0.67). "다른
   신호원이라 앞선 실패를 안 물려받는다"는 가설도 기각.

### `research_line_registry.json`의 관련 라인 (21개 중 발췌 — 전문은 원본 참고)

- **`global_technical_indicator_search`**: "기존 기술지표 피쳐 우주... 반복 탐색이 소진됨...
  진짜 새로운 인과적 원천 데이터라야 정보집합이 바뀐다." → 아래 모든 후보는 기존 OHLCV/오더북
  파생 기술지표 재조합이 아니라 **원천이 다른 데이터**여야 한다는 게이트.
- **`global_funding_carry_contrarian`**: "펀딩-only 변형은 소진됨... 알려진 지연시간을 가진 새
  원천 데이터라야 함." → 단일거래소 펀딩비 자체(FINAL12에 이미 3개: `funding_pressure_diff1`,
  `funding_roc_48`, `mta_funding`)는 이미 죽은 축.
- **`global_macro_tradfi_overlay`**: "무료 캘린더 신호는 검증 실패; 유료 데이터의 증분은
  미입증." → Fear&Greed 류의 저정보량 매크로 심리 지표를 제안할 때는 회의적으로 취급해야 함.
- **`btc_dvol_feature_overlay`**: "DVOL 레벨/변화량 오버레이, BTC quality classifier — 0/9
  구성, 대부분 OOS 악화." retest_guidance: **"옵션 스큐나 기간구조 데이터로, 독립적으로
  검증된 가용성이 있다면 강한 차별점"** — 즉 DVOL(집계 IV 지수) 자체는 죽었지만 옵션체인의
  다른 파생물(스큐·기간구조)은 레지스트리가 스스로 "아직 안 죽었다"고 명시한 유일한 옵션계열
  후보.
- **`eth_overnight_generic_feature_entry_filter_20260809`** (2026-08-09, 17개 아이디어 전부
  음성): conformal abstention, path-signature, 144bar 모멘텀, 캘린더, taker buy/sell OFI, **BTC
  lead-lag 모멘텀**, **BTC+SOL 바스켓 대비 상대강도**, 6개 스킵필터 계열(모멘텀/OFI, 실현변동성/
  ATR, 결합, **Hawkes 가격점프 클러스터링**, Kaufman Efficiency Ratio, 44피쳐 종합), 실제
  라이브 코드 복제 2-모델 우선순위 라우터, quantile-regression 스큐 — 전부 실패. **레포 검증
  결과**: `scripts/research_btc_lead_eth_entry_20260809.py`, `scripts/research_cross_sectional_
  relative_strength_eth_20260809.py`가 실제로 ETH 대상으로 존재해 이름만이 아니라 실행까지
  확인됨. `scripts/research_hawkes_jump_clustering_skip_filter_eth_20260809.py`도 확인 —
  **단, 이 Hawkes는 `is_jump = abs(bar_ret) > threshold`로 가격 5분봉 리턴 자체에서 정의된
  점프 지표(순수 기술지표 파생)이지, 아래 후보 2가 제안하는 실제 청산 이벤트 스트림
  (`tail_risk_interceptor.py`의 `@forceOrder` 데이터)에 피팅한 Hawkes와는 원천 데이터가
  다르다** — 같은 통계 도구(Hawkes self-exciting process)를 쓴다는 이유로 이미 죽은 라인과
  혼동하면 안 되지만, 혼동 소지가 있어 명시적으로 구분해둔다.

### FINAL12/REL11이 이미 커버한 카테고리 (재확인, Step 1 audit)

| 카테고리 | FINAL12/REL11에 있는 피쳐 | 결론 |
|---|---|---|
| 펀딩비(단일거래소) | `funding_pressure_diff1`, `funding_roc_48`, `mta_funding`, `funding_roc_288`, `funding_abs_dt288` | 이미 있고 실패 — 새로 "펀딩비 추가"는 무의미 |
| 고래/탑트레이더 포지셔닝(파생상품) | `sig_whale_dt288`, `sum_toptrader_long_short_ratio_dt288` | 이미 있고 실패 — "고래 추적 추가"는 무의미 |
| 오더플로우/CVD | `cvd_288`(REL11) + 위 `eth_overnight...` 라인의 taker OFI | 이미 있고 실패 |
| 변동성 추정치 | `garman_klass_vol`, `hurst_288`, `realized_skewness`(FINAL12/REL11 공통), `parkinson_vol`(dedup 패배) | 이미 있고 실패 |
| 레짐 | `regime3_current_sensitive_wide24_chop_prob`, `cvp_regime` | 이미 있고 실패 |

**결론: 이 5개 카테고리를 "새 아이디어"로 재제안하지 않는다** — 사용자 지시사항과 동일하게,
이들은 이미 시도되고 실패한 풀에 포함돼 있다. 아래 후보는 전부 이 5개 카테고리 밖의 원천이다.

## 데이터 인프라 감사 (레포 실사, 2026-08-11 기준)

**범례**: **(a)** 이미 연결됨, quality_head엔 미사용 · **(b)** 부분 인프라 존재, 확장 필요 ·
**(c)** 신규 인프라 필요(스크래퍼/유료API 등)

| # | 데이터소스 | 상태 | 증거(레포 실사) | 현재 실제 용도 | quality_head 연결 |
|---|---|---|---|---|---|
| 1 | 청산 이벤트 스트림 | **(a)** | `tail_risk_interceptor.py:37` `_FORCE_ORDER_WS_URL = "wss://fstream.binance.com/market/ws/{symbol}@forceOrder"`; `:39` DB경로 `data/live/tail_risk.duckdb`; `:186-190` `_db_insert(bucket_ts, long_1m, short_1m, liq_event_count_1m)` — 1분 버킷으로 영구 저장 | Hawkes 프로세스 기반 "사후 요격기" — 포지션 진입 후 캐스케이드 리버설 타이밍 개입(`intercept()`) | 없음(grep 확인, quality_head/FINAL12/REL11 어디에도 청산 파생 피쳐 없음) |
| 2 | MicrostructureScanner L2(상위 20단계)+aggTrade | **(a)** 일부 / **(b)** 일부 | `microstructure_scanner.py:25-29` `@depth20@100ms`, `@aggTrade`, OI/펀딩 REST poll; `:91` DB경로 `data/live/microstructure.duckdb` | whale flow·toptrader ratio 파생은 FINAL12에 있음(`sig_whale_dt288` 등); `shadow_toxicity_score`/`shadow_queue_collapse`/`shadow_absorption_score`/`spoofing_score`는 `:512-521` 등에서 계산되지만 `integrated_overlay.py`/`news_shock_guard.py`/`playbook_meta_controller.py:66-67` 같은 **오버레이·가드 모듈에서만** 소비됨 | toxicity/queue/absorption/spoofing 4종은 `pipeline/feature_contract.py` grep 0건 — quality_head 학습 피쳐 계약에 없음 |
| 3 | F4-C altdata(거래소간 펀딩 스프레드 Binance/OKX + Fear&Greed + 바이낸스 공지) | **(a)** 수집 중, 소비처 0 | `scripts/run_f4c_altdata_collector.py` 전문 확인; `data/research/altdata.duckdb`(1,585,152 bytes)와 `altdata_collector_cron.log` 둘 다 **2026-08-10 01:00 최종수정**(이 세션 하루 전, 실측 활성) — 로그에 ETH/BTC/SOL 펀딩 스프레드 실제 값 기록 확인 | 격리 리서치 DB, "WS-D의 D4 수집 건강 감시에 추후 편입 예정"(주석) | 레포 전체 grep 결과 수집 스크립트 자신 외 어떤 `.py`도 `altdata`를 참조하지 않음 — 완전 미사용 |
| 4 | Polymarket 예측시장(Gamma API + CLOB) | **(a)** | `polymarket_engine.py:26-28` 실제 엔드포인트; `:23-31` 날짜기반 슬러그 자동 해석(`POLYMARKET_SLUG_LOOKAHEAD_DAYS=5`); `features/integrated_overlay.py:91-115` `build_polymarket_overlay_features`가 `gap`/`momentum_1m`/`mode_prob`/`tail_up_prob`/`tail_down_prob`/`direction_pressure` 등 계산 | `news_shock_guard.py`의 리스크 감축 가드(포지션 진입 후 "샷" 대응)로만 사용 — **이름과 달리 텍스트 뉴스가 전혀 아님** | quality_head 학습 피쳐엔 없음. 실제 시장 만기/유동성 깊이는 **미검증**(라이브 조회 필요) |
| 5 | Deribit DVOL(BTC+ETH, 시간봉) | **(b)** — DVOL 레벨 자체는 이미 죽음 | `scripts/download_deribit_dvol_20260804.py:99` `for currency in ["BTC", "ETH"]` — 무료 공개 REST, 인증 불필요 | 레지스트리 `btc_dvol_feature_overlay`: BTC quality classifier에 0/9, 대부분 OOS 악화 | ETH quality_head엔 시도된 적 없음(BTC 전용 실패 사례). 옵션체인 스큐/OI/GEX는 미존재 |
| 6 | CoinMetrics 온체인(Community 무료 tier) | **(b)** — BTC 전용, ETH 미다운로드 | `scripts/download_coinmetrics_onchain_20260804.py:32` `"assets": "btc"` 하드코딩, 7개 지표(`AdrActCnt/CapMVRVCur/FlowInExNtv/FlowOutExNtv/SplyExNtv/HashRate/TxCnt`), 일봉 | `tmp/btc_dense_nogate_quality_onchain_20260804.csv` 존재(실행 흔적) | **결과 문서/report.json을 못 찾음 — 결과 미상, 검증 불가로 취급**(fabrication 금지 원칙에 따라 "실패"로도 "성공"으로도 단정하지 않음) |
| 7 | 정식 VPIN / L2 20단계 초과 depth | **(c)** | "vpin_lite" 형태 프록시(`abs(taker_buy-taker_sell)/총거래량`)가 `enhanced_trading_engine.py`, `backtest_macro_micro_playbook.py`, `backtest_param_ensemble.py`, `optimize_duckdb_quant_formula.py` 등에 존재 — 전부 Easley/López de Prado/O'Hara(2012)식 고정-거래량-버킷 bulk classification이 아닌 **단순화된 근사** | 위 스크립트들은 h48qual/quality_head와 무관해 보이는 별도 전략 모듈들 | h48qual/quality_head에 연결됐다는 증거 0건(미검증) |
| 8 | 텍스트 뉴스/소셜 감성(NLP) | **(c)** | `news_shock_guard.py` 전문 확인 — 실제로는 `build_micro_overlay_features`(마이크로구조) + `build_tail_overlay_features`(청산 기반 테일리스크) + `build_polymarket_overlay_features`(예측시장) 3종의 합성일 뿐, 텍스트 인풋 전무. repo 전역 twitter/reddit/nlp grep도 무관한 매치만 | 이름이 오해를 유발하는 기존 모듈일 뿐 | 존재하지 않음 — 완전 신규 구축 필요 |
| 9 | 거래소간 가격 basis(펀딩 제외 순수 가격차) | **(c)**에 가까운 **(b)** | F4-C가 펀딩 스프레드는 수집하지만 가격 basis는 미수집. 단, 동일 스크립트가 이미 `ccxt`로 Binance+OKX 양쪽을 호출 중이라(`run_f4c_altdata_collector.py:69-77`) mark/index price 호출 추가는 코드 확장 수준 | 없음 | 없음 |

**"오더플로우 96 피쳐" 특정 확인 시도**: 계약 문서가 언급하는 "102 base(기술적/오더플로우/OU
96 + regime3-current 6)"의 정확한 96개 목록을 정의하는 단일 파일을 찾지 못했다(`pipeline/
feature_contract.py`엔 관련 매치 없음) — **미검증**. 다만 위 표의 행 2에서 확인했듯,
MicrostructureScanner가 실시간으로 계산하는 `shadow_toxicity_score`/`shadow_queue_collapse`/
`shadow_absorption_score`/`spoofing_score`(L2 20단계+aggTrade 파생)는 오버레이/가드 모듈에서만
참조되고 quality_head 학습 피쳐 계약에서는 grep 0건이었다 — 즉 "96 오더플로우"가 이 신호들을
포함하는지 여부와 무관하게, **적어도 이 4개 필드는 지금 quality_head 학습에 들어가지 않는다**는
것은 확인됐다.

## 후보 (전부 미검증 — 아이디어 단계, 구현 전. 재학습 없음)

### 후보 1. 기존 마이크로구조 toxicity/queue/absorption/spoofing 필드를 quality_head 피쳐로 승격 — 🔓 열린 후보, 검증 안 됨

**인프라: (a)** — 이미 라이브로 계산되고 `data/live/microstructure.duckdb`에 영구 저장 중. 새
수집기·새 API 불필요, 과거 재구성만 필요.

**왜 이 갭에 맞는가**: `quality_head`의 질문은 정확히 "이미 direction이 고른 이 진입이 팔로우
스루할지 반전할지"이고, 이건 정의상 **진입 시점의 주문흐름 질**에 관한 질문이다. Easley,
López de Prado, O'Hara(2012, *Review of Financial Studies*)의 flow toxicity(VPIN) 개념은
바로 이 질문 — "지금 흐름에 정보우위 트레이더가 많은가"를 겨냥해 설계됐다. FINAL12/REL11의
기존 오더플로우 피쳐(`cvd_288`, whale/toptrader 파생)는 전부 **방향성** 신호(누가 사는지
파는지)인 반면, toxicity/queue_collapse/absorption/spoofing은 **질(quality)** 신호(그 흐름이
얼마나 일방적이고 흡수되지 않는지)라 라벨의 성격과 더 가깝다. `_compute_shadow_toxicity`가
간단한 근사치(OBI-taker_ratio 불일치 + 거래크기 버스트)이긴 하나, 이미 계산되고 있으니 정식
VPIN(후보 8)보다 먼저 공짜로 테스트할 수 있다.

**가장 싼 검증 스텝**: 기존 `quality_for_action` 0단계 진단과 동일한 패턴 — 재학습 없이,
저장된 h48orig/h384 시드들의 dir_action 타임스탬프에 `data/live/microstructure.duckdb`의
`shadow_toxicity_score`/`shadow_queue_collapse`/`shadow_absorption_score`/`spoofing_score`를
조인해서 `spearmanr(신호, realized_outcome)` 확인. 상관 없으면 즉시 폐기, GBM/TabM 재학습 불필요.

**미검증/캐비어트**: (1) `data/live/microstructure.duckdb`의 실제 과거 커버리지가 h48qual
TRAIN/VAL/OOS 구간(2025-01~2026-03 근방)을 얼마나 채우는지 확인 안 됨 — 라이브 스캐너가 언제부터
계속 켜져 있었는지에 달림. (2) toxicity 근사치 자체가 조악해서(mismatch+burst 가중합) 신호가
약할 수 있음 — 이건 후보 8(정식 VPIN)로 가는 조건부 다음 단계일 뿐, 이 후보의 실패가 후보 8의
실패를 의미하진 않음.

### 후보 2. 청산 캐스케이드 / Hawkes 강도 파생 피쳐 — 🔓 열린 후보, 검증 안 됨

**인프라: (a)** — `tail_risk_interceptor.py`가 이미 Binance `@forceOrder`(실제 강제청산 이벤트)
스트림을 수신해 Hawkes 자기여기과정으로 캐스케이드를 모델링하고 1분 버킷으로
`data/live/tail_risk.duckdb`에 저장 중. **그라운딩 절에서 확인했듯, 이건 레지스트리가 이미
닫은 "Hawkes 가격점프 클러스터링"(가격 리턴 기반)과 원천 데이터가 다르다** — 실제 청산
이벤트라는 별개의 raw source.

**왜 이 갭에 맞는가**: Brunnermeier & Pedersen(2009, *Review of Financial Studies*)의
유동성-펀딩 스파이럴 이론이 정확히 이 메커니즘을 설명한다 — 레버리지 청산이 발생하면 한
방향으로 기계적 가격압력이 생기고, 그 압력이 얼마나 빨리 흡수되는지가 이후 팔로우스루 대
반전을 가른다. 크립토 무기한선물 시장에서 청산 캐스케이드가 단기 추세/반전에 미치는 영향은
업계에서 광범위하게 관찰·논의되는 현상이다(정확한 학술 인용은 이 문서에서 특정하지 않음 —
확인된 것만 인용한다는 원칙에 따름). direction_head가 막 진입을 고른 시점에 "그 방향으로
청산 캐스케이드가 방금 일어났는가/일어나는 중인가"는 방향 신호(FINAL12에 이미 있음)와는
완전히 다른 축의 정보 — **타이밍과 팔로우스루 가능성**에 관한 정보다.

**가장 싼 검증 스텝**: 후보 1과 동일 패턴. `data/live/tail_risk.duckdb`의 `liq_event_count_1m`/
`long_1m`/`short_1m`을 direction과 같은 방향/반대 방향으로 나눠 진입 시점 전후 짧은 윈도우로
집계, `quality_for_action` 0단계와 동일한 순위상관 진단. 재학습 없음.

**미검증/캐비어트**: `data/live/tail_risk.duckdb`의 과거 커버리지도 후보 1과 동일한 문제 —
TRAIN/VAL/OOS 구간을 얼마나 채우는지 미확인. 또한 청산 캐스케이드는 본질적으로 **희소 이벤트**라
(평상시엔 0에 가까움) 대부분의 bar에서 피쳐가 0이 되어 유효 표본이 작을 수 있음 — 진단 단계에서
바로 드러날 것.

### 후보 3. Polymarket 예측시장 피쳐를 quality_head 학습 피쳐로 승격 — 🔓 열린 후보, 검증 안 됨

**인프라: (a)** — `polymarket_engine.py` + `features/integrated_overlay.py:91-115`의
`build_polymarket_overlay_features`가 이미 `direction_pressure`, `confidence`, `uncertainty`,
`tail_bias` 등을 계산 중.

**왜 이 갭에 맞는가**: Wolfers & Zitzewitz(2004, *Journal of Economic Perspectives*)가 정리한
예측시장의 핵심 성질 — 분산된 참가자들의 신념을 단일 가격으로 집계 — 은 OHLCV/오더북/펀딩
파생물과 정보원 자체가 다르다(같은 거래소 spot/perp 데이터에서 유도된 게 아님). ETH/BTC
가격목표 Polymarket 시장의 실시간 가격변화(`gap`, `momentum_1m`)는 "군중이 지금 이 방향
지속을 얼마나 믿는가"에 대한 시장기반 신뢰도이며, 이는 quality_head가 근사하려는 것(이 방향이
팔로우스루할 확률)과 개념적으로 가장 가까운 후보다.

**가장 싼 검증 스텝**: 동일 순위상관 패턴. 단, 과거 Polymarket 스냅샷을 TRAIN/VAL/OOS 구간에
맞춰 재구성할 수 있는지가 선행 조건(현재 `polymarket_engine.py`는 라이브 조회 구조로 보이며
과거 재현 가능 여부는 미확인).

**미검증/캐비어트**: (1) 슬러그가 날짜기반 자동생성이고 조회창이 5일(`POLYMARKET_SLUG_
LOOKAHEAD_DAYS=5`)이라 시장 자체는 단기(일~수일) 만기로 추정되나, 정확한 정산조건과 48~384bar
(4h~32h) 호라이즌과의 정합성은 **미검증**. (2) 실제 유동성 깊이(스프레드, 체결가능 물량)가
과거 전체 구간에서 충분했는지 **미검증** — Polymarket의 크립토 가격목표 시장은 대형 정치/스포츠
이벤트 시장 대비 유동성이 낮을 수 있음. (3) 과거 재현(historical replay) 가능 여부 자체가
확인 안 됨 — 이게 안 되면 이 후보는 사실상 (c)로 격상됨.

### 후보 4. 거래소간 포지셔닝 괴리 — 펀딩 스프레드 + 가격 basis — ❌ 닫힘 (검증 완료, 2026-08-11)

**검증 결과**: `docs/experiments/eth_h48qual_basis_candidate4_rank_correlation_20260811.md`.
펀딩 스프레드는 라이브 확인 결과 OKX `funding-rate-history` 공개 엔드포인트가 `since`를 사실상
무시하고 최근 ~1개월치만 반환 — VAL/OOS 백필 불가로 애초에 검증 불가능. 가격 basis는 OKX
1시간봉(2025-01~현재) 다운로드해 검증 — 오염도는 낮았으나(`corr(price)` 최대 0.30, candidate 6
절차로 확인) 순위상관이 라벨변형 간 일관되지 않음(`|basis|`가 h384 OOS에서만 강한 음의 상관
p&lt;0.0001, h48orig OOS는 반대 방향 비유의) — candidate 6의 `AdrActCnt`와 같은 부호-불안정
패턴으로 기각. **결론: 두 축 다 닫힘.**

원문(제안 당시 논리)은 아래에 남긴다.

**인프라: (a)** 펀딩 스프레드 부분 / **(b)** 가격 basis 부분 — F4-C(`run_f4c_altdata_
collector.py`)가 Binance-OKX 펀딩비 스프레드(ETH/BTC/SOL)를 이미 매일 수집 중(2026-08-10
활성 확인). 가격 basis(마크/인덱스 가격차)는 미수집이나 같은 `ccxt` 클라이언트로 호출 한 줄
추가 수준.

**왜 이 갭에 맞는가**: FINAL12/REL11의 펀딩 관련 피쳐(`funding_pressure_diff1`,
`funding_roc_48`, `mta_funding`, `funding_roc_288`, `funding_abs_dt288`)는 전부 **단일 거래소
(바이낸스) 레벨/변화율**이다 — 이미 시도되고 실패한 축(그라운딩 절 참고). **거래소간 스프레드**는
다른 정보를 담는다: 바이낸스와 OKX의 트레이더 구성이 다르므로(리테일 대 기관 비중 등), 스프레드
확대는 한쪽 거래소에 편향된 레버리지 쏠림 — 즉 "이 방향 포지셔닝이 얼마나 일방적인가"의 대리
지표가 될 수 있다. 가격 basis(순수 가격차)는 차익거래 압력의 실시간 지표로, 방향성보다는
"지금 가격이 여러 시장에서 얼마나 일치하는가"를 측정해 팔로우스루의 안정성과 관련 있을 수 있다.

**가장 싼 검증 스텝**: 두 가지 문제 해결이 선행돼야 함 — (1) 현재 F4-C가 하루 1회 수집이라
실시간(5분봉) 피쳐로 못 씀. 코드 자체는 REST 폴링이라 수집 주기를 5분~1시간으로 올리는 건
간단한 설정 변경. (2) **과거 데이터가 없음** — 2026-08-10부터 수집 시작이라 TRAIN 구간
(2025년) 백필이 안 됨. Binance/OKX 양쪽 다 펀딩비 히스토리 REST 엔드포인트가 보통 존재하므로
과거 백필 가능성은 있으나 **미검증**(실제 조회 안 해봄).

**미검증/캐비어트**: Fear&Greed Index는 이 후보와 같은 F4-C 배치로 수집되지만, 이미 상당 부분
가격 파생 지표(변동성·모멘텀·거래량의 합성)라 그라운딩 절의 `global_macro_tradfi_overlay`
(무료 캘린더/매크로 신호 실패)와 유사한 회의를 적용해야 한다 — 별도 후보로 승격하지 않고 이
후보의 부속 실험으로만 취급, 우선순위 낮음. 과거 백필 가능성이 확인 전까지는 이 후보 전체가
사실상 (c)에 가까울 수 있음.

### 후보 5. Deribit 옵션체인 스큐 / OI / GEX 프록시 — ⏸ 인프라 미비로 보류 (2026-08-11 확인)

**라이브 확인(2026-08-11)**: `get_book_summary_by_currency`는 **현재 스냅샷만** 반환(과거 시점
파라미터 없음), `get_instruments(expired=true)`는 만기 계약의 메타데이터(계약명 등)만 줄 뿐
과거 가격/IV 시계열을 안 준다 — Deribit 공개 REST API에 "특정 과거 시점의 옵션체인"을 조회하는
엔드포인트가 없다. 정식 히스토리를 만들려면 계약별 과거 trade/chart 데이터를 개별 수집해 재구성
해야 하는데, 이건 "가장 싼 검증 스텝"의 범위를 벗어나는 별도 엔지니어링 프로젝트다. VAL/OOS
구간 백필이 사실상 안 되므로 보류 — 아래 원문(제안 당시 논리)은 그대로 둔다.

**인프라: (b)** — `download_deribit_dvol_20260804.py`가 무료 공개 Deribit REST(`get_
volatility_index_data`, 인증 불필요)로 BTC+ETH DVOL을 이미 받고 있어, 같은 API 계열(예:
`get_book_summary_by_currency`, 개별 인스트루먼트 ticker)로 확장하는 건 새 계약/과금이
필요 없는 코드 확장.

**왜 이 갭에 맞는가**: 레지스트리 `btc_dvol_feature_overlay`가 스스로 명시: "DVOL(집계
지수) 오버레이는 0/9로 죽었지만, **옵션 스큐나 기간구조 데이터는 독립적으로 검증된 가용성이
있다면 강한 차별점**"이라고 retest_guidance에 적어뒀다 — 즉 이 자체가 레지스트리에 남아있는
몇 안 되는 "아직 안 죽은" 명시적 후보. DVOL은 "평균적으로 향후 30일 변동성이 얼마일 것 같은가"
하나의 숫자인 반면, 25-delta put-call 스큐는 옵션시장 참가자들의 **방향성 꼬리위험 프라이싱**을
담는다 — Xing, Zhang, Zhao(2010, *Journal of Financial and Quantitative Economics*)가 주식옵션
스큐가 향후 주식수익률에 정보력이 있음을 보인 것과 같은 논리가 여기서도 성립할 수 있다: 옵션
시장은 perp/spot 흐름과 다른 참가자군(정교한 변동성 트레이더, 마켓메이커)이 지배하므로 정보원
자체가 다르다. dealer gamma exposure(GEX) 프록시는 "가격이 지금 핀(pin)될지 가속될지"에 대한
메커니즘적 가설을 제공한다(딜러가 숏감마면 헤지가 가격변동을 증폭 — Barbon & Buraschi의
"Gamma Fragility" 워킹페이퍼 계열 논의) — quality_head가 답하려는 "팔로우스루 대 반전" 질문과
메커니즘적으로 직결된다.

**가장 싼 검증 스텝**: 전체 히스토리 백필 전에 최근 며칠치 book-summary 스냅샷만 받아 (1) ETH
옵션의 실제 유동성/미결제약정이 스큐·GEX를 계산할 만큼 충분한지, (2) 값이 퇴화(거의 상수)되지
않고 실제로 변동하는지부터 싸게 확인.

**미검증/캐비어트**: ETH 옵션의 Deribit 미결제약정·거래량이 BTC 대비 훨씬 얇다는 건 업계에서
잘 알려진 사실이나 이 감사에서 라이브로 직접 확인하지는 않았다(**미검증**) — 스큐/GEX 계산이
얇은 유동성에서 노이즈만 만들 위험이 있음, 1단계 스냅샷 점검이 바로 이걸 걸러낼 것.

### 후보 6. ETH 온체인 순유입 + 스테이블코인 발행 — ❌ 닫힘 (검증 완료, 2026-08-11, 부정 결과)

**검증 결과**: `scripts/download_coinmetrics_onchain_eth_20260811.py`로 ETH 6개 지표(스테이블코인
발행량 지표는 결국 미시도 — 아래 참고) 다운로드 후 0단계 순위상관 진단 완료 —
`docs/experiments/eth_h48qual_onchain_candidate6_rank_correlation_20260811.md`. `CapMVRVCur`가
4개 조합(h48orig/h384 × VAL/OOS) 전부 방향 일관 + h384 양쪽 Bonferroni 생존으로 강한 신호처럼
보였으나, **오염도 직접 측정 결과 `corr(price)=0.95~0.97`로 심각하게 오염**(FINAL12 dedup의
배제 기준 0.561보다 훨씬 심함) — MVRV 비율이 사실상 가격 그 자체를 재현한 것으로, 전일대비/
7일변화율로 detrend하자 신호가 완전히 붕괴(4개 조합 전부 비유의). `SplyExNtv`도 동일 기제로
오염(0.82~0.87). 나머지 4개 지표(AdrActCnt/FlowInExNtv/FlowOutExNtv/TxCnt)는 오염은 약하나
애초에 신뢰할 신호가 없었음(무상관 또는 스플릿간 부호반전). **결론: 무료tier 6개 지표 중
진입-레벨 신호 없음 — 강해 보인 신호는 드리프트-베타였다.** 스테이블코인 발행량(당초 제안에
있었던 지표)은 6개 지표가 전부 무신호로 닫히면서 착수하지 않음 — 파생 메커니즘이 부분적으로
겹치는 축이라(순유입과 마찬가지로 "매수여력 프록시") 추가 시도 우선순위 낮음.

원문(제안 당시 논리, 인프라는 (b) 표기 그대로 유지)은 아래에 남긴다.

**인프라: (b)** — `download_coinmetrics_onchain_20260804.py`가 BTC 전용으로 하드코딩
(`"assets": "btc"`)이나 CoinMetrics Community API는 멀티에셋이라 ETH로 확장은 파라미터 변경
수준. 스테이블코인 발행량 관련 정확한 무료tier 지표명(USDT/USDC 공급 메트릭)은 이 감사에서
확인하지 않음(**미검증**).

**왜 이 갭에 맞는가**: FINAL12의 `sig_whale_dt288`/`sum_toptrader_long_short_ratio_dt288`은
**파생상품 포지셔닝**(바이낸스 선물 대형계좌의 롱숏비율)이지 **현물 커스터디 이동**이 아니다 —
서로 다른 행위자(파생상품 레버리지 트레이더 대 실제 자산 이동)가 서로 다른 시간축에서 움직이는
별개의 메커니즘이다. 거래소 순유입(`FlowInExNtv`/`FlowOutExNtv`, 이미 CoinMetrics 무료tier에
있는 지표)은 매도 압력의 선행지표로, 스테이블코인 신규발행은 신규 매수여력의 선행지표로
업계에서 흔히 논의된다 — 단, Griffin & Shams(2020, *Journal of Finance*) "Is Bitcoin Really
Untethered?"가 테더 발행-BTC가격 관계에 대한 반박도 상당히 존재함을 보여주듯, 이 인과관계
자체가 학계에서 다투어지는 주제임을 인지하고 접근해야 한다 — 과신 금지.

**가장 싼 검증 스텝**: ETH로 파라미터만 바꿔 다운로드(수 분), TRAIN/VAL/OOS 구간에 일 단위로
조인 후(≥24시간 인과적 지연 명시), 기존 `rescreen_eth_h48qual_quality_regression_*` 스크립트와
동일한 방법론(Spearman + `mutual_info_regression`)으로 재학습 없이 relevance부터 확인.

**미검증/캐비어트**: 일봉 vs 5분봉의 극심한 해상도 불일치 — 48~384bar(4h~32h) 배리어 대비
정보가 하루 지연으로만 갱신되므로, 빠르게 변하는 신호가 아니라 **느린 컨텍스트 피쳐**로만
쓸 수 있음. 유사한 BTC 온체인 시도의 실행 흔적(`tmp/btc_dense_nogate_quality_onchain_
20260804.csv`)은 있으나 결과 문서를 못 찾아 **선례 결과 자체가 검증 불가** — 이걸 "이미
실패한 축"으로도 "아직 안 죽은 축"으로도 단정하지 않는다.

### 후보 7. 엔트리 시점 competing-risks / hazard 라벨 재구성 — 🔓 열린 후보, 검증 안 됨 (라벨축 — 부차적)

**인프라**: 해당없음 — 새 원천 데이터가 아니라 **기존 데이터를 다르게 라벨링**하는 방법론
후보. 하지만 사용자가 명시적으로 "대안적 라벨 철학"을 조사 범위에 포함시켰으므로 기록한다.

**⚠ 먼저 밝혀야 할 것 — 이 레포에서 hazard/competing-risk 모델링 자체는 새롭지 않음**:
`scripts/research_eth_omega461_competing_risk_rescue_20260724.py`(TP/SL/timeout을 12/48/384bar
+ 우측중도절단까지 포함한 7-class competing-risk, bootstrap 앙상블), `scripts/eval_omega1_2_
stop_loss_hazard_veto_20260604.py`, `scripts/train_regime3_transition_hazard_20260530.py`가
이미 존재한다. **단, 이 세 라인은 전부 다른 질문에 답한다**: `competing_risk_rescue`와
`stop_loss_hazard_veto`는 **이미 포지션을 연 뒤**(direction/quality 후보 에피소드) 그 포지션을
구제/청산할지를 다루는 **exit-side** hazard이고, `regime3_transition_hazard`는 레짐 전환
자체를 예측하는 것이다. `research_eth_omega461_distributional_stopping_20260724.py`(quantile
continuation-value 회귀)는 VAL에서 순수 SLTP를 이겼지만(+70.47%) Stage-1 hazard 후보에 내부
경쟁에서 져서 OOS를 열어보지도 못했다(scalar_alternatives 문서 인용). **이 후보가 제안하는 건
다르다**: 포지션을 열기 **전**, `h48_conservative`의 고정 48/384bar 3-way 배리어 분류 자체를
TP-hit-time과 SL-hit-time의 competing-risks 분포로 바꾸는 것 — exit_head의 입력(13개 `pos_*`,
이미 포지션이 열려 MFE/MAE를 관측한 상태)이 전혀 없는 시점의 문제다.

**왜 이 갭에 맞는가**: 지금 라벨은 미래 경로 전체를 고정 호라이즌 하나의 3-way 판정으로
뭉갠다 — TP를 5bar 만에 쳤는지 47bar 만에 쳤는지, SL을 살짝 비껴갔는지 크게 빗나갔는지 등
경로의 형태 정보가 전부 버려진다. Fine & Gray(1999, *JASA*)의 competing-risks 서브분포 hazard나
Lee, Zame, Yoon, van der Schaar(2018, AAAI) "DeepHit"류의 discrete-time survival 프레이밍은
TP/SL 각각의 hazard를 매 시점마다 추정해 훨씬 조밀한 학습신호(각 bar가 "아직 안 끝남" 관측치를
제공)를 준다 — 3-way 분류보다 유효 표본이 훨씬 많아질 수 있다.

**핵심 리스크(자체 반박)**: 이 문서 자신의 그라운딩 절이 지적하듯, 회귀 전환(candidate E)도
"메커니즘은 유효하나(오라클 게이트 15/15 시드 압도) 실전 신호가 없다"로 닫혔다 — 라벨을
연속값으로 바꿔도 정보량 자체가 없으면 실패한다는 게 이미 한 번 증명됐다. Hazard 프레이밍도
근본적으로 "같은 원천 데이터에 다른 통계적 표현"이라, **이 후보 단독으로는 후보 1~6 같은
새 원천 데이터 없이 문제를 풀 가능성이 낮다** — evidential DL(candidate B)이 "다른 신호원(C)도
실패했으니 다른 loss만으론 안 된다"는 논리로 회의적으로 취급된 것과 정확히 같은 이유다.
**따라서 이 후보는 단독 후보가 아니라, 후보 1~6 중 하나가 0단계 진단을 통과했을 때 그 신호를
가장 잘 살리는 라벨 표현으로 재고려하는 2차 후보로 취급한다.**

**가장 싼 검증 스텝**: 순수 재라벨링 + 저비용 베이스라인(신경망 재설계 전) — 기존 캐노니컬
배리어 빌더(`scripts/build_omega1_2_triple_barrier_labels_20260619.py`)의 bar별 TP/SL 터치
데이터로 discrete-time hazard 라벨만 새로 구성하고, FINAL12만으로 `lifelines`/`scikit-survival`
류의 가벼운 Cox/discrete-hazard 베이스라인의 concordance를 확인 — TabM 재설계 전에 통계
패키지 수준에서 싸게 죽이거나 살릴 수 있다.

### 후보 8. 정식 L2/L3 오더북 딥쓰 + 학술적 VPIN — 🔓 열린 후보, 검증 안 됨 (고비용, 조건부)

**인프라: (c)** — 현재 depth 캡처는 `@depth20@100ms`(상위 20단계 스냅샷)뿐, add/cancel/modify
메시지 단위의 완전한 L2/L3 큐 재구성이 없다. 정식 VPIN(고정 거래량 버킷 + bulk trade
classification, Easley/López de Prado/O'Hara 2012)도 없음 — 존재하는 건 여러 곳의 "vpin_lite"
근사치(위 인프라 감사 표 참고)뿐이고 quality_head 연결 증거는 0건.

**왜 이 갭에 맞는가**: 후보 1과 같은 논리(flow toxicity가 팔로우스루/반전과 개념적으로 직결)지만
더 엄밀한 버전. 상위 20단계보다 깊은 큐 동역학(주문 취소/추가 패턴)은 스푸핑이나 진짜 흡수력을
더 정확히 구별할 수 있고, 정식 VPIN은 근사치보다 이론적 기반이 탄탄하다.

**왜 우선순위가 낮은가(자체 판단)**: 이건 이 문서에서 가장 비싼 인프라 후보다 — 새 과금 API는
아니지만(Binance 무료 diff-depth 스트림 자체는 공짜) 라이브 오더북 리플리카 유지, 고정거래량
버킷 클럭 구현, 그 결과를 과거 기간 전체에 대해 재구성하는 엔지니어링 비용이 상당하다. **후보
1의 조악한 근사치가 0단계 진단에서 아무 상관도 못 보이면, 정식 VPIN을 굳이 새로 지을 이유가
약해진다** — 반대로 조악한 근사치에서 약한 신호라도 보이면, "제대로 만들면 더 강해질 수 있다"는
근거가 생긴다. 그래서 이 후보는 **후보 1의 결과에 조건부**로 취급한다.

**가장 싼 검증 스텝**: 후보 1을 먼저 끝낸 뒤에만 착수. 그 전에는 검증 스텝 자체가 없음(선행조건
미충족).

## 하지 말아야 할 것

Step 1 감사에서 확인된, 다시 제안하면 안 되는 것들:

- **일반 기술지표 재조합**(`global_technical_indicator_search`, 반복 소진) — 위 8개 후보 전부
  기존 OHLCV/오더북 파생이 아닌 원천이 다른 데이터거나(1~6, 8) 순수 라벨 재구성(7)이어야
  한다는 게이트를 통과시켰다.
- **단일거래소 펀딩비 레벨/변화율 추가**(`funding_pressure_diff1`, `funding_roc_48`,
  `funding_roc_288`, `mta_funding`, `funding_abs_dt288` — 이미 FINAL12/REL11에 있고 실패) —
  거래소간 **스프레드**(후보 4)는 다른 축이라 구분된다.
- **고래/탑트레이더 롱숏비율 추가**(`sig_whale_dt288`, `sum_toptrader_long_short_ratio_dt288` —
  이미 FINAL12에 있고 실패) — 온체인 커스터디 순유입(후보 6)은 파생상품 포지셔닝이 아닌 별개
  메커니즘이라 구분된다.
- **CVD/taker OFI 추가**(REL11 `cvd_288`, `eth_overnight_generic_feature_entry_filter_20260809`
  라인에서 실패) — 정식 VPIN(후보 8)이나 마이크로구조 toxicity(후보 1)는 방향성 OFI가 아니라
  흐름의 "질"을 재는 것이라 구분되나, 이 구분이 실제로 유효한지는 후보 1/8의 0단계 진단이
  스스로 증명해야 한다.
- **변동성 추정치 추가**(`garman_klass_vol`, `hurst_288`, `realized_skewness`, `parkinson_vol` —
  이미 FINAL12/REL11 dedup에 포함/패배) — DVOL(Deribit 집계 IV)도 이미 BTC에서 0/9로 실패한
  같은 범주. 옵션 **스큐**(후보 5)는 방향성 꼬리위험 프라이싱이라 집계 변동성 지수와 다른 축.
- **BTC-ETH lead-lag 모멘텀, BTC+SOL 바스켓 상대강도**(`eth_overnight_generic_feature_entry_
  filter_20260809`에서 실제로 ETH 대상 스크립트까지 확인된 실패) — 재제안 금지.
- **가격 리턴 기반 Hawkes 점프 클러스터링**(`research_hawkes_jump_clustering_skip_filter_
  eth_20260809.py`, 이미 실패) — 실제 청산 이벤트 스트림에 피팅하는 후보 2와는 원천이 다름을
  이미 위에서 명시적으로 구분해뒀다.
- **Conformal abstention을 hard gate로**, **클래스별 독립 isotonic** — 계약 문서가 이미 금지,
  레지스트리 근거 있음(다른 축이지만 재확인).
- **무료 매크로/캘린더 오버레이를 단독 승격 후보로**(`global_macro_tradfi_overlay`, 실패) —
  Fear&Greed는 후보 4의 부속 실험으로만 남겨두고 단독 후보로 격상하지 않는다.
- **`quality_head`를 회귀로 재전환하거나 스칼라 추출 방법을 다시 시도하는 것**(candidate A/C/E,
  전부 별도 문서에서 닫힘) — 이 문서의 범위 밖. 새 원천 데이터가 실제로 신호를 보인 뒤에만
  그 데이터에 대해 회귀/분류 선택을 다시 고려한다.
- **완전 신규 유료 데이터 계약을 검증 없이 먼저 사는 것** — `global_macro_tradfi_overlay`의
  교훈("유료 데이터 증분은 미입증")을 옵션체인(후보 5)·온체인(후보 6) 등 모든 유사 후보에
  동일하게 적용: 항상 무료/이미 있는 데이터로 먼저 진단하고, 유료·신규 인프라는 그 진단이
  긍정적일 때만 고려한다.

## 우선순위 갱신 (2026-08-11, 실제 착수 후 재조정)

원래 우선순위(아래 원문)는 "인프라가 이미 연결돼 있으니 조인만 하면 된다"는 가정으로 짰는데,
실제 착수해보니 그 가정이 후보 1~3·5에서 깨졌다:

- **후보 1·2(마이크로구조/청산)**: 라이브 duckdb 커버리지가 2026-05-03 이후뿐이라 기존
  VAL(2025-10~12)/OOS(2026-01~02)와 전혀 안 겹침 — "조인만" 하는 게 아니라 새 구간에 대한
  causal inference부터 다시 필요(더 큰 작업으로 재분류, 아래 미결 상태 유지).
- **후보 3(Polymarket)**: 커버리지가 9일(2026-04-21~30)뿐이라 표본도 부족하고 1·2의 구간과도
  안 겹침 — 사실상 (c) 수준으로 재분류.
- **후보 5(Deribit 스큐/GEX)**: 과거 특정 시점 옵션체인을 조회하는 API가 없음이 확인돼 보류로
  재분류(위 후보 5 절 참고).
- **후보 6(온체인)**: 유일하게 원래 가정대로(다운로드 스크립트로 원하는 과거 구간을 직접 확보)
  진행 가능했고, 실제로 검증까지 완료 — **부정 결과로 닫힘**(위 후보 6 절 참고). 강해 보였던
  `CapMVRVCur` 신호는 가격추세 오염으로 판명.

**교훈**: "이미 연결된 데이터"(인프라 (a))가 실제로는 "다운로드 스크립트로 원하는 과거 구간을
받는" 것(인프라 (b))보다 항상 싸다고 가정하면 안 된다 — 라이브 모니터링 수집은 켜진 시점부터만
쌓이므로, 과거 특정 구간이 필요한 진단(이 프로젝트의 VAL/OOS 대조 전부)에는 오히려 목적 다운로드
스크립트 쪽이 더 직접적일 수 있다. 남은 미검증 후보(4·7·8)와 보류 후보(1·2·3·5)는 이 교훈을
반영해 재평가가 필요하다 — 아직 착수 안 함.

## 제안 우선순위 (원문 — 착수 전 순서, 위 갱신 참고)

가장 싼 검증(재학습 없음, 이미 있는 데이터 조인만)부터, 가장 비싼 것(신규 인프라 구축)을
나중에 배치. 각 단계는 이전 단계 결과와 독립적으로 착수 가능(단, 후보 8은 후보 1에 조건부).

1. **후보 1 — 마이크로구조 toxicity/queue/absorption/spoofing 필드**: 인프라(a), 조인만
   하면 되는 가장 싼 진단. 최우선.
2. **후보 2 — 청산 캐스케이드 피쳐**: 인프라(a), 후보 1과 동급으로 싸다. 단, 이벤트 희소성으로
   유효표본이 작을 위험을 진단 단계에서 바로 확인.
3. **후보 3 — Polymarket 피쳐**: 인프라(a)지만 과거 재현 가능성이 선행조건으로 걸려있어 1·2보다
   한 단계 더 확인이 필요. 재현 가능이 확인되면 1·2와 동급 우선순위로 승격.
4. **후보 6 — ETH 온체인 순유입/스테이블코인**: 인프라(b), 다운로드 자체는 몇 분이면 되는
   파라미터 변경 수준이라 싸다. 단, 일봉 해상도라 신호가 약할 가능성을 미리 감안.
5. **후보 4 — 거래소간 펀딩 스프레드/가격 basis**: 인프라(a)/(b) 혼합이나 **과거 백필 가능성이
   미확인**이라 4위로 내림 — 백필이 안 되면 사실상 새로 데이터가 쌓이길 기다려야 해서 다른
   후보 대비 검증 착수 자체가 늦어짐.
6. **후보 5 — Deribit 옵션체인 스큐/OI/GEX**: 인프라(b), API는 무료·검증된 패턴이나 신규 계산
   로직(스큐·GEX 산출)과 ETH 옵션 유동성 리스크 때문에 6위.
7. **후보 7 — Competing-risks/hazard 라벨 재구성**: 새 데이터가 아니라 라벨 축이라 단독으로는
   약함(자체 반박 절 참고) — 후보 1~6 중 하나가 살아난 뒤 그 신호를 위한 라벨 표현으로 재고려.
8. **후보 8 — 정식 L2/L3 딥쓰 + 학술 VPIN**: 인프라(c), 가장 비싼 엔지니어링 비용. 후보 1의
   결과에 조건부 — 후보 1이 완전히 무상관이면 이 후보의 착수 근거가 약해진다.

## 결과 (계약 문서 반영용 요약)

`quality_head`(`h48_conservative` 라벨)의 실전 무신호 문제에 대해 "같은 ~201컬럼 피쳐 패널에
다른 스칼라 추출/loss를 쓰는" 축(온도보정·회귀전환·앙상블불일치·evidential — 전부 별도 문서에서
닫히거나 회의적)과 직교하는 **새 원천 데이터** 축을 조사했다. 레포 실사 결과 청산 이벤트 스트림
(`tail_risk_interceptor.py`), 마이크로구조 toxicity/queue/absorption/spoofing 파생값
(`microstructure_scanner.py`), 거래소간 펀딩 스프레드+Fear&Greed(F4-C `run_f4c_altdata_
collector.py`, 실측 활성 확인), Polymarket 예측시장(`polymarket_engine.py`)이 **이미 라이브로
연결·수집되고 있으나 quality_head 학습 피쳐 계약에는 전혀 쓰이지 않는다**는 걸 확인했다(전부
grep으로 0건 검증) — 이들이 재학습 없이 가장 싸게 검증 가능한 1순위 후보다. Deribit 옵션체인
스큐/GEX(DVOL 자체는 이미 BTC에서 0/9로 실패했으나 스큐/기간구조는 레지스트리가 스스로 "아직
안 죽은 차별점"으로 남겨둠)와 ETH 온체인 순유입/스테이블코인 발행(CoinMetrics, 현재 BTC 전용을
확장)은 부분 인프라만 있어 중간 비용. 엔트리 시점 competing-risks/hazard 라벨 재구성은 이
레포에 이미 존재하는 exit-side hazard 연구(`research_eth_omega461_competing_risk_rescue_
20260724.py` 등, 전부 포지션을 이미 연 뒤의 질문이라 이 문서의 범위와 다름)와 명시적으로 구분한
별개 후보이나, 새 원천 데이터 없이 단독으로는 회귀전환 실패와 같은 이유로 성공 가능성이 낮아
2차/조건부 후보로 취급한다. 정식 L2/L3 오더북 딥쓰+학술적 VPIN 구축은 가장 비싼 신규 인프라라
가장 낮은 우선순위이며 마이크로구조 toxicity 근사치(1순위 후보)의 결과에 조건부다.
**모든 후보는 미검증·미구현 상태다.** 다음 단계는 학습이나 아키텍처 변경이 아니라, 1~3순위
후보에 대해 재학습 없는 순위상관 진단(기존 `quality_for_action` 0단계와 동일한 방법론)부터
돌리는 것을 권한다.

**추가 업데이트 (2026-08-11, 실제 착수 후)**: 후보 1·2·3은 라이브 duckdb 커버리지가 VAL/OOS
구간과 안 겹쳐(2026-05-03 이후만 존재) "재학습 없이 조인만"이라는 전제가 깨져 더 큰 작업으로
재분류됨(위 "우선순위 갱신" 참고). 후보 5(Deribit)는 과거 시점 옵션체인 조회 API가 없어 보류.
**후보 6(온체인)은 유일하게 원래 계획대로 진행돼 검증까지 완료 — 부정 결과.** 강한 신호처럼
보였던 `CapMVRVCur`(4개 조합 전부 방향 일관, h384 양쪽 Bonferroni 생존)이 실제로는 `corr(price)
=0.95~0.97`의 심각한 가격추세 오염이었고(FINAL12 dedup 배제 기준 0.561보다 훨씬 심함),
detrend(전일대비/7일변화율)하자 신호가 완전히 붕괴했다. 나머지 5개 지표는 무상관이거나 스플릿간
부호 불안정. 상세: `docs/experiments/eth_h48qual_onchain_candidate6_rank_correlation_20260811.md`.
**신규 절차**: 이 발견으로 새 raw-level 피쳐 후보는 학습 전에 `corr(price)`/`corr(시간순번)`
오염도부터 확인하는 걸 이 리서치 라인의 표준 절차로 추가한다 — 남은 후보(4·7·8, 그리고 인프라
문제가 풀리면 1·2·3·5도)에 전부 적용해야 한다.

**추가 업데이트 (2026-08-11, 후보 4도 검증)**: 새로 세운 오염도 체크 절차를 후보 4(거래소간
펀딩 스프레드+가격 basis)에 처음부터 적용해 검증 완료 — 부정 결과. 펀딩 스프레드는 OKX
`funding-rate-history` 공개 엔드포인트의 보존 기간이 짧아(약 1개월, `since` 사실상 무시)
애초에 VAL/OOS 백필이 안 돼 검증 불가능함을 라이브로 확인. 가격 basis(OKX 1시간봉 다운로드)는
오염도는 낮았으나(`corr(price)` 최대 0.30) 순위상관이 라벨변형 간 일관되지 않아(`|basis|`가
h384 OOS에서만 강한 음의 상관, h48orig OOS는 반대 방향) 기각 — **오염도 체크를 통과해도 변형 간
일관성이라는 별도의 신뢰성 기준을 통과해야 한다**는 게 재확인됨. 상세:
`docs/experiments/eth_h48qual_basis_candidate4_rank_correlation_20260811.md`. 지금까지 착수한
6개 후보(1·2·3·5·6·4) 전부 인프라 문제로 막히거나 부정 결과로 닫혔다 — 남은 미착수 후보는
7(라벨 재구성, 새 데이터 아님)과 8(정식 VPIN, 가장 비싼 인프라, 6·8 모두 후보 1 결과에
조건부였는데 그 후보 자체가 보류라 근거 약화)뿐이다.
