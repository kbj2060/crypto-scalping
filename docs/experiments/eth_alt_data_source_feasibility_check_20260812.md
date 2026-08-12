# ETH h48qual — 대체데이터 5종(CoinGlass/Dune/DefiLlama/LunarCrush/Santiment) 접근성 확인 (2026-08-12)

## 배경

퀀트펀드 데이터소스 리서치([앞선 대화 응답] 참고)가 짚은 5개 대체데이터 후보를 사용자 지시로
재검토. 이 서브 프로젝트의 표준 절차("새 데이터소스는 인프라 가용성부터 확인 후 검증 착수")에
따라, 실제 API를 직접 호출/조사해 과거 TRAIN(2024-06~2025-09)/VAL(2025-10~12)/OOS(2026-01~02)
구간 백필이 가능한지부터 확인했다.

## 접근성 결과

| 소스 | 무료 여부 | 과거 백필 범위 | 판정 |
|---|---|---|---|
| **DefiLlama** | 완전 무료, API키 불필요 | 2017~2018년부터 전체 커버 | ✅ **검증 완료** — 별도 문서, 부정 결과 |
| **Santiment** | 무료(계정 불필요, 익명 조회 가능) | **최근 약 12개월 롤링만**(실측: 2025-08-12~2026-07-13), 그 이전은 유료+SAN 토큰 스테이킹 필요 | TRAIN(2024-06~2025-09) 대부분을 못 채움 — 정식 MI/R² 게이트 방법론과 안 맞음 |
| **Dune Analytics** | 계정 가입 시 무료 티어 제공(API 호출 포함) | 온체인 SQL 쿼리라 이론상 전체 블록체인 역사 커버 가능(계정 만들어야 실측 확인 가능) | **계정 가입 필요 — 내가 자율적으로 할 수 없음, 사용자 결정 필요** |
| **LunarCrush** | 실시간/최근 데이터는 무료 계정 가능하나 **과거 시계열은 Pro 유료 티어부터**(2026-08-12 오전 기록, 오늘 오후 재확인 실패) | **5분봉 자체가 API에 없음**(시간봉이 최고해상도) — 시간봉의 실제 커버 시작일은 계정 없이 확인 불가 | 5분봉 목적으로는 결제와 무관하게 폐기(아래 추가업데이트) |
| **CoinGlass** | **무료 티어 자체가 없어짐**(Hobbyist $29/월부터) | 인터벌별 상이 — 일봉만 all-time, 5분봉은 최대 60일(Professional $699/월)뿐 | **결제 없이 요금제/해상도만 조사 완료 — 유료라도 프로젝트 TRAIN~OOS 5분봉 백필은 구조적으로 불가능(아래 추가업데이트)** |

## 실제 검증한 것 — DefiLlama

완전 무료 조건을 충족한 유일한 소스라 바로 검증까지 진행. ETH 체인 TVL/DEX거래량/수수료·매출
3종(+detrend 파생) 대상 오염도 체크 + LightGBM 홀드아웃 비교 — **부정 결과**(FINAL12 단독
대비 VAL/OOS 둘 다 소폭 악화). 부수적으로 일별 forward-fill 데이터의 `mutual_info_classif`
degenerate 버그를 발견·확정(Fear&Greed 실험에도 소급 caveat 추가). 전체 과정:
`docs/experiments/eth_defillama_onchain_direction_relevance_20260812.md`.

## 진행 못 한 3개 — 사용자 결정 필요

1. **Dune Analytics**: 무료 티어가 있지만 계정 가입(이메일 인증 등)이 필요해 내가 자율적으로
   진행할 수 없다. 온체인 SQL 쿼리 특성상 이론적으로는 이 프로젝트에서 가장 유망한 후보(전체
   블록체인 역사가 인덱싱돼 있어 커스텀 지표를 원하는 기간만큼 백필 가능) — 계정을 만들어
   API 키를 공유해주시면 바로 검증 착수 가능.
2. **CoinGlass**: 청산/OI 히스토리는 이 서브 프로젝트가 진작부터 원했던 데이터(2026-08-11
   새 데이터소스 리서치의 최우선 후보 1번, 라이브 duckdb가 2026-05 이후만 커버해 막혔던 바로
   그 데이터)인데, 지금은 월 $29 이상 결제해야 과거 데이터에 접근 가능하다. 결제 승인이
   필요한 사안이라 진행 전 확인이 필요하다.
3. **LunarCrush**: 과거 시계열이 유료라 후순위 — Santiment(무료지만 최근 12개월만)로 감성
   데이터의 대략적인 방향성만 참고용으로 확인해볼 수는 있으나, 이 프로젝트의 TRAIN/VAL/OOS
   표준 구간과 안 맞아 정식 검증으로는 부적합.

## 권장

- **Dune 계정 가입**이 비용 없이 가장 큰 문을 여는 선택지 — 사용자가 가입 후 API 키를 공유하면
  즉시 착수 가능.
- **CoinGlass 결제**는 청산/OI 데이터가 이 프로젝트가 가장 오래 원했던 데이터인 만큼 값어치는
  있어 보이나, 결제 여부는 사용자 판단.
- 둘 다 보류한다면, 무료로 시도 가능한 다음 후보는 이 문서 밖의 다른 축(post-OOS 구간 재검증
  등, 계약 문서의 열린 항목)으로 넘어가는 것을 권장.

## 추가 업데이트 (2026-08-12, 사용자 요청으로 CoinGlass 요금제·해상도 실사)

**결제 없이** CoinGlass 공식 API 문서(`docs.coinglass.com`)와 공식 GitHub 레퍼런스
(`coinglass-official/coinglass-api-skills`)를 조사. `futures/liquidation`,
`futures/open-interest`, `futures/long-short-ratio` 세 카테고리 전부에서 바이트 단위로 동일한
"플랜×인터벌→최대 lookback" 표를 확인했다 — 이 서브 프로젝트가 원했던 청산·OI 백필 대상
엔드포인트 전부에 동일 정책이 적용된다는 뜻.

### 요금제

| 플랜 | 월 요금 | endpoint 수 | rate limit | 인터벌 하한 |
|---|---|---|---|---|
| Hobbyist | $29 | 80+ | 30/분 | 4h(그 아래 인터벌 전부 "Not available") |
| Startup | $79 | 130+ | 80/분 | 30m |
| Standard | $299 | 150+ | 300/분 | 제한 없음(1m까지) |
| Professional | $699 | 160+ | 1200/분 | 제한 없음(1m까지) |
| Enterprise | 협의 | custom | custom | custom |

### 인터벌별 최대 lookback (청산/OI/long-short ratio 공통, 공식 레퍼런스 원문 확인)

| 플랜 | 5m | 1h | 4h | 12h | 1d |
|---|---|---|---|---|---|
| Hobbyist | 불가 | 불가 | 180일 | 360일 | all-time |
| Startup | 불가 | 180일 | 180일 | 360일 | all-time |
| Standard ($299) | **30일** | 360일 | 360일 | 360일 | all-time |
| Professional ($699) | **60일** | 720일 | 720일 | 720일 | all-time |

### 이 프로젝트 구간(TRAIN 2024-06~09, VAL 2025-10~12, OOS 2026-01~02)에 대입

오늘(2026-08-12) 기준 TRAIN 시작(2024-06-01)은 802일 전, VAL 시작(2025-10-01)은 315일 전, OOS는
223~165일 전이다(`date -d` 직접 계산, 아래 전부 동일 방식으로 검증).

- **5분봉은 어떤 유료 등급으로도 안 됨.** Standard 30일(2026-07-13~), Professional 60일
  (2026-06-13~) 둘 다 OOS 끝(165일 전)에도 못 미친다. 월 $699를 내도 최근 두 달치 5분봉만
  나온다.
- **시간봉대도 부족.** Professional의 최대치(1h~12h, 720일)조차 2024-08-22까지만 닿아 TRAIN
  시작(2024-06-01)보다 82일(~2.7개월) 모자란다. Hobbyist/Startup의 360일 상한(6h~12h)은
  2025-08-17까지만 닿아 TRAIN 486일 중 44일(9%)만 건지고(VAL·OOS는 완전 커버), Hobbyist의
  4h/180일 상한은 2026-02-13까지만 닿아 OOS 58일 중 15일(26%)만 건진다.
- **일봉(all-time)만 전체 구간 커버** — 가장 싼 Hobbyist($29)도 여기엔 제한이 없다. 단, 이건
  이 프로젝트가 이미 두 번(DefiLlama, CoinMetrics 온체인) 겪은 "일봉 forward-fill 컨텍스트
  피쳐" 패턴과 동일한 리스크(`mutual_info_classif` degenerate 버그
  [[feedback_forward_fill_mutual_info_degenerate]], 가격추세 오염
  [[feedback_raw_feature_price_trend_contamination]])를 그대로 물려받는다.

### 쓸 수 있는 피쳐(엔드포인트 카테고리)

Liquidation(pair/coin 히스토리, heatmap 3종, max pain), Open Interest(OHLC, 증거금유형/
스테이블코인별, 거래소별), Funding Rate, Long/Short Ratio(글로벌·top trader 계정, 포지션,
taker buy/sell), Order Book(L2 heatmap, large order), Whale Positions(Hyperliquid 전용),
Taker Buy/Sell(CVD, net flow), Spot, Options, On-chain, ETF, 기술지표(RSI/MACD/BB 등). 이
프로젝트가 원래 원했던 건 이 중 Liquidation·Open Interest 히스토리뿐이다(candidate 1·2 —
`tail_risk.duckdb`/`microstructure.duckdb`가 2026-05-03 이후만 커버해 막혔던 부분의 백필용,
[[odyssey_eth_h48qual_subproject]] 참고).

### 결론 — 돈을 낸다고 풀리는 문제가 아님

5분봉(라이브 파이프라인 표준 해상도) 기준으로는 $699/월 최상위 등급도 최근 60일치만 주므로
TRAIN/VAL/OOS 백필 자체가 원천적으로 불가능하고, 시간봉대로 타협해도 TRAIN 시작 쪽 2~3개월이
항상 빈다. 유일하게 전체 구간이 나오는 일봉 해상도는 이미 이 서브 프로젝트가 두 번 실패한
저해상도 포워드필 컨텍스트 피쳐 패턴과 동일한 리스크를 안고 간다. **CoinGlass 결제는
권장하지 않음** — candidate 1·2(청산/마이크로구조)의 진짜 해법은 이미 켜져 있는 라이브 duckdb가
시간이 지나며 VAL/OOS급 구간을 자연히 축적하길 기다리거나(신규 window로 재정의), 저해상도
일봉 대체 피쳐로 격을 낮춰 재시도(알려진 오염 리스크 감안)뿐이다.

출처: [CoinGlass Pricing](https://www.coinglass.com/pricing), [OI OHLC History](https://docs.coinglass.com/reference/oi-ohlc-histroy), [Liquidation History](https://docs.coinglass.com/reference/liquidation-history), [Endpoint Overview](https://docs.coinglass.com/reference/endpoint-overview), 공식 GitHub 레퍼런스 `coinglass-official/coinglass-api-skills`의 `futures/{liquidation,open-interest,long-short-ratio}/references/plans-interval-history-length.md` (raw 파일 직접 확인, 2026-08-12).

## 추가 업데이트 (2026-08-12, 사용자 요청으로 LunarCrush API 해상도·피쳐 실사)

CoinGlass와 같은 방식으로 조사했으나, LunarCrush 웹사이트(`lunarcrush.com/pricing`,
`/developers/*`, FAQ 2종)가 전부 클라이언트사이드 렌더링 SPA라 WebFetch·Wayback 스냅샷
전부 빈 셸만 반환됨 — **요금제 정확한 $ 티어표는 오늘 재확인 못 함**(아래 명시). 대신 공식
API 스펙 저장소(`github.com/lunarcrush/api`, 정적 markdown이라 접근 가능)에서 기술 스펙은
확정적으로 확인했다.

### 확인됨 — 5분봉은 애초에 상품으로 없음 (CoinGlass와는 다른 종류의 한계)

공식 v4 API README 전체를 grep한 결과, Topic/Category/Creator/Posts/Coins/Stocks/NFTs
**7개 시계열 엔드포인트 전부**가 `bucket` 파라미터로 `hour` 또는 `day` 두 값만 받는다 —
`5m`/`1m`/`15m` 같은 분단위 옵션이 스펙 자체에 없다. CoinGlass는 "더 내면 더 세밀한
인터벌"이었지만, LunarCrush는 **돈과 무관하게 분단위 데이터가 제품에 없다**. 소셜 포스트
집계라는 데이터 특성상 애초에 5분 단위로 의미가 있기 어려운 지표라는 점과도 부합한다.

### 최고 해상도(시간봉)의 실제 커버리지 — 계정 없이 확인 불가

`bucket=hour`를 명시하면 문서상 "full historical data available in hourly aggregation"
(전체 과거 데이터를 시간봉으로)라고 되어 있어 이론상으론 TRAIN(2024-06~) 이전까지도 커버할
가능성이 있다. 단, ETH(`ethereum`) 토픽이 실제로 언제부터 데이터가 쌓였는지, 이 "전체
히스토리" 접근이 무료 계정에도 열리는지 유료 등급부터인지는 **API 키 없이는 확인 불가** —
Dune Analytics(계정 가입 필요)와 같은 종류의 막힘이다.

### 쓸 수 있는 피쳐 (Topic/Coins Time Series 스키마, 공식 문서로 확인됨)

`galaxy_score`(가격기술지표+소셜심리 복합 proprietary 점수), `alt_rank`(상대성과 순위),
`sentiment`(0~100%, interaction 가중 긍정비율), `social_dominance`(전체 소셜볼륨 대비 비중),
`interactions`/`posts_active`/`posts_created`/`contributors_active`/`contributors_created`
(볼륨류), `spam`(스팸포스트 수) — 전부 OHLC/market_cap/volume_24h와 한 응답에 번들로 나옴.
별도로 Creators(인플루언서별), Posts(개별 포스트 상호작용 시계열), Categories 엔드포인트도
있다.

### 요금제 — 오늘은 확정 못 함

공식 pricing 페이지, `/developers/pricing`, FAQ 2종, 3rd-party 리뷰(rate limit로 실패),
Wayback 스냅샷까지 시도했으나 전부 SPA 빈 셸만 반환돼 정확한 티어명·월요금·크레딧 한도를
오늘 재확인하지 못했다. 검색엔진 요약에서 "크레딧 기반, 최소 $1/일에 크레딧 2,000개 포함,
추가 크레딧 $0.0005/개"라는 문구를 봤으나 1차 소스 대조를 못 해 **미검증으로 취급**한다. 이
문서 상단 표의 기존 기록("과거 시계열은 Pro 유료 티어부터")은 오늘 재확인도 반박도 못 했다 —
기존 기록 유지, 갱신 아님.

### 결론

5분봉 목적으로는 **결제 여부와 무관하게 폐기** — 애초에 API에 없는 해상도라 어떤 등급을 사도
안 나온다. 시간봉 대체 피쳐로 격을 낮추면 이론상 전체 구간 커버 가능성이 있으나(문서상
"full historical data"), 실제 ETH 토픽 커버 시작일과 무료/유료 게이팅 둘 다 계정을 만들어
API 키로 직접 호출해봐야 확인되는 상태다 — Dune과 동일하게 **계정 가입이 다음 단계의
전제조건**이라 사용자 결정이 필요하다.

출처: [LunarCrush API v4 GitHub](https://github.com/lunarcrush/api) (README.md 원문 직접 확인, 2026-08-12), [LunarCrush Pricing](https://lunarcrush.com/pricing/)(SPA, 오늘 콘텐츠 접근 실패), [LunarCrush Developers](https://lunarcrush.com/developers/api/endpoints)(SPA, 동일).
