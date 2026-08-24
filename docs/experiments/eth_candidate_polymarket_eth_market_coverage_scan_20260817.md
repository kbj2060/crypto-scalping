# ETH 방향예측 — 폴리마켓 ETH 마켓 커버리지 스캔 (2026-08-17)

## 배경

사용자가 폴리마켓/코인글래스 청산히트맵/레딧/X 4개 소스로 ETH 방향예측 축을 설계해달라고
요청. 사전 조사에서 (1) 코인글래스는 실현청산(`tail_risk_1m`, Binance `@forceOrder`)과는 다른
선행적 예상청산클러스터 데이터라는 점, (2) 폴리마켓은 `polymarket_engine.py` +
`features/integrated_overlay.py::build_polymarket_overlay_features`가 이미 존재하지만
`features/news_shock_guard.py`의 리스크가드 용도로만 쓰였을 뿐 방향예측 피쳐로 스크린된 적이
없다는 점, (3) 레딧/X는 리포에 텍스트 수집기가 전혀 없다는 점을 확인했다. 이 문서는 그중 Tier
B(폴리마켓)의 첫 단계 — **모델링 이전에 ETH 관련 마켓이 실제로 상시 존재하고 트레이딩 가능한
유동성/스프레드를 갖는지** — 를 Gamma API(`gamma-api.polymarket.com`)와 CLOB API
(`clob.polymarket.com`)에 대한 실제 호출로 검증한 결과다. 결제·계정가입 불필요, 전부 공개
엔드포인트.

## 마켓 타입별 커버리지 (2026-08-17 기준, `tag_slug=ethereum&closed=false` 100건)

| 타입 | 예시 제목 | 동시 존재 개수 | 24h거래량[중앙값] | 유동성[중앙값] |
|---|---|---|---|---|
| **daily_updown** | "Ethereum Up or Down on August 17?" | 1~2개(정확히 24h 주기로 생성/소멸) | $18,180 | $23,723 |
| daily_above_ladder | "Ethereum above ___ on August 17?" (11~16개 행사가 사다리) | 7개(향후 며칠치 동시 오픈) | $2,228 | $106,317 |
| daily_price_bucket | "Ethereum price on August 17?" (11개 구간버킷, 기존 `news_shock_guard`가 이미 탭하는 타입) | 7개 | $393 | $35,486 |
| range_ladder_monthly | "What price will Ethereum hit in August/2026?" | 3개 | $30,079 | $651,392 |
| **subdaily_updown** | "Ethereum Up or Down - August 17, 9:45AM-9:50AM ET" (시간 단위~15분 단위) | **79개** | **$4** | $10,999 |

## 핵심 발견 1 — daily_updown이 가장 깨끗한 후보

- **해상도/커버리지**: "Ethereum Up or Down on [날짜]" 마켓은 최소 2025-06-27부터 오늘까지
  매일 끊김없이 생성 확인(슬러그 패턴 `ethereum-up-or-down-on-{month}-{day}[-{year}]`로
  2025-09-01, 2025-12-31, 2026-01-01, 2026-03-31, 2026-06-01 전부 존재 — 즉 이 프로젝트의
  Fresh-Forward VAL(2025-09-01~12-31)/OOS(2026-01-01~03-31) 구간 전체를 커버).
- **판정소스가 깨끗함**: 리졸루션이 "Binance ETH/USDT 1분봉 종가, 전일 12:00 ET → 당일 12:00
  ET 비교"로 명시돼 있음 — 우리가 이미 쓰는 것과 동일한 거래소/페어라 크로스소스 베이시스
  리스크가 없음.
- **유동성 곡선**: 마켓 생성 직후(24h 남음)엔 스프레드 10%(bestBid 0.49/bestAsk 0.59,
  유동성 $1,652)로 얇지만, 리졸루션 임박 시점엔 스프레드 1%(0.80/0.81, 유동성 $23,723)까지
  좁혀짐 — "빠른 선행신호"보다 "장중 누적확률의 연속 업데이트"로 더 유용할 가능성.

## 핵심 발견 2 — 과거 시계열 API 백필이 원천적으로 불가능

CLOB `prices-history` 엔드포인트를 마켓 나이별로 테스트(오늘 살아있는 마켓, 1주 전, 1개월 전,
2개월 전, 3개월 전, 6개월 전, 11.5개월 전 종료 마켓 각각의 토큰ID로 `interval=max`와 명시적
`startTs/endTs` 둘 다 시도):

- **아직 열려있는(오늘) 마켓**: 178개 포인트(5분 해상도, 최근 구간만) 반환 — 정상 동작.
- **닫힌(리졸브된) 마켓은 나이 불문 전부 0포인트** — 7일 전 마켓도, 11.5개월 전 마켓도
  동일하게 빈 배열. 파라미터 방식(interval vs 명시적 타임스탬프)도 무관.

→ **닫힌 마켓의 장중 확률경로는 폴리마켓 공개 API로 사후에 복원할 수 없다.** 이미 알려진
"`polymarket.duckdb`가 2026-04-21~30 9일치뿐"인 상황과 정확히 같은 구조 — 라이브로 수집하지
않은 구간은 영구 소실이다. VAL/OOS 구간에 대한 정식 백테스트는 API로 불가능하며, **지금부터
수집을 재개해야만 몇 주~몇 달 후 forward-collected 구간으로 검증 가능**하다(청산피드의
09-15 게이트와 동일한 시간축 제약).

## 핵심 발견 3 — 현재 라이브 파이프라인은 사실상 죽은 신호를 흘려보내는 중

`append_polymarket_snapshot_to_duckdb` 호출부가 코드베이스에 없어 `polymarket_markets_10s_json`
테이블이 4월 30일 이후 갱신되지 않는다. `build_polymarket_overlay_features(row)`는 순수
패스스루 변환(자체 fetch 없음)이라, 지금 `news_shock_guard`에 들어가는 `row`는 사실상 빈
값/기본값(`mode_prob=0, tail_up=0, tail_down=0, entropy=1.0` 등)일 가능성이 높다 —
`direction_pressure≈tanh(0)=0`, `confidence≈0`으로 리스크가드가 상시 무신호 상태로
운영되고 있을 것으로 추정된다(라이브 로그 직접 확인은 안 했음, 코드경로상의 추정).

## 핵심 발견 4 (부수 발견) — Tier A 코인글래스는 이미 5일 전에 조사 완료, 비권장 결론

이번 스캔 도중 `docs/experiments/eth_alt_data_source_feasibility_check_20260812.md`에서
CoinGlass가 이미 결제 없이 요금제/해상도까지 실사됐음을 확인했다: **5분봉은 최고가($699/월
Professional)로도 최근 60일치만 제공돼 TRAIN/VAL/OOS 백필이 구조적으로 불가능**, 시간봉대도
TRAIN 시작 쪽 2~3개월이 항상 비고, 전체구간이 나오는 건 일봉뿐인데 이건 이미 이 프로젝트가
두 번 실패한 저해상도 forward-fill 오염 패턴과 같은 리스크를 진다. 결론은 **"CoinGlass 결제는
권장하지 않음"**. 즉 앞서 제안한 로드맵에서 Tier A("코인글래스 청산 히트맵")는 벤더 경유로는
막다른 길이고, 유일하게 살아있는 청산 관련 리드는 기존 `tail_risk_1m`(Binance 실현청산) 기반
컨트래리언 스파크뿐이다(09-15 게이트).

## 판정 및 다음 단계

| 항목 | 판정 |
|---|---|
| ETH 마켓 상시 존재 | ✅ daily_updown/daily_above_ladder/daily_price_bucket 전부 매일 생성 확인 |
| 트레이딩 가능한 유동성/스프레드(리졸루션 임박 시) | ✅ 1% 스프레드, $23K+ 유동성 (daily_updown 기준) |
| VAL/OOS 과거 구간 백필 | ❌ **불가능** — 닫힌 마켓 히스토리 API가 항상 빈 배열 반환 |
| subdaily_updown(시간~15분 단위) | ❌ 24h거래량 중앙값 $4 — 노이즈, 후보에서 제외 권고 |
| 현재 라이브 파이프라인 상태 | ⚠️ 데이터 4월 이후 끊김, `news_shock_guard`가 상시 무신호로 추정 |

**권고**: (1) 백필이 원천 불가능하므로, 이 축을 살리려면 오늘부터 `append_polymarket_snapshot_to_duckdb`
호출부를 즉시 복구해 수집을 재개해야 한다(더 늦출수록 forward-collected 구간만 영구히 짧아짐,
비용은 거의 0 — 이미 만들어진 함수 연결뿐). (2) `daily_updown`을 기존 `daily_price_bucket`과
별도의 새 피쳐군으로 추가 — 지금까지 방향예측용으로 스크린된 적이 없다는 게 핵심 갭.
(3) `subdaily_updown`은 제외. (4) 승격 검증은 forward-collected 데이터가 GEX 프로토콜 수준
표본(Tier1 ≥20일)에 도달한 뒤에만 가능 — 그 전까지는 수집 인프라 구축 단계로 취급.

## 조사 방법 (재현용)

- Gamma API: `GET /events?tag_slug=ethereum&closed=false&limit=100&order=volume24hr&ascending=false`
- 개별 마켓 조회: `GET /events?slug=ethereum-up-or-down-on-{month}-{day}[-{year}]`
- CLOB 가격이력: `GET /prices-history?market={token_id}&interval=max&fidelity={n}` 및
  `startTs/endTs` 명시 버전 둘 다 테스트
- 전부 무인증 공개 엔드포인트, 접근일 2026-08-17
