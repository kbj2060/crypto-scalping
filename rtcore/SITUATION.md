# rtcore 상황·미시 지표 — 무엇을 코어가 세어 미나 (2026-11-21)

원본은 server.py 의 미시 루프(`collect_micro_ref`, 1초) → 같은 틱 끝의 `compute_situation`(SITUATION_EVERY_TICKS = 1) → `market_context_payload`.
코어는 이미 받는 웹소켓 원천(체결·최우선 호가·마크·청산·1초 수급)만으로 되는 몫을 옮긴다. REST·수집기 DB·래스터 파일이 필요한 몫은 «남음»이다.

## 1. 옮긴 것 — `situation` 메시지 (src/situation.rs)

`Event::Situation` = `situation.classify` 결과 + `computed_at` + `missing`. 1초마다 세고 결과가 바뀌었을 때만 민다. 스냅숏 `situation` 칸에는 마지막 결과가 실린다.
흐름 엔진 종목(footprint.rs SPECS)만 센다. 옛 받는 쪽은 모르는 type 이라 건너뛴다. 새 칸은 모두 serde default 라 뒤로 호환된다.

| classify 입력 | 원본 원천 | 코어 원천 | 같은가 |
|---|---|---|---|
| 봉 델타(매수 − 매도) | 풋프린트 봉 칸 | 같은 풋프린트 봉 칸 | 같음(칸 합 차례만 다름 · 아래) |
| 고가·저가·종가 | REST 5분 klines(60초 캐시) → 없으면 풋프린트 칸 고저 · 1초 링 마지막 체결가 · 봉 VWAP | `@trade` 로 센 5분봉 고가·저가·마지막 체결가 → 없으면(복원으로 메운 봉) 원본과 같은 대체 | klines 는 같은 체결로 만든다 — 빠진 체결이 없으면 같다. 원본은 캐시가 60초 늦어 방금 닫힌 봉에 대체값을 쓴다. 코어는 곧바로 캔들을 쓴다(일부러) |
| 청산 롱·숏 USD | 봇 DB 5분 집계(`compute_liquidation_5m_history`) | 바이낸스 `@forceOrder` 5분 누적(supply.rs liq5m) | 같은 사건, 다른 저장 경로 — 실서버 대조는 아직 안 함 |
| mid | REST bookTicker 중간값(매초) | `@bookTicker` 중간값 | 원천만 다름 |
| 펀딩·베이시스·Δ·임계 | `@markPrice@1s` 링 7,200초 → `deriv_from_ring` | 같은 사건(Mark), 같은 링·정리 규칙(MarkRing) | 같음 |
| BTC 같은 창 | REST BTC 캔들 | btcusdt `@trade` 로 센 캔들(btcusdt 를 받을 때만) | 캔들 행과 같은 조건 |
| prev_dir(슈미트) | 직전 결과의 dir(ok=false 면 0) | 같음 | 같음 |
| veto(12h 추세) | `trend_veto_rows`(5분봉 720개 SMA144·ATR ewm) | **없음 → 0** · `missing` | 남음 — 5분봉 720개 워밍업이 필요하다(REST klines, 또는 코어가 1분봉·5분봉을 2.5일치 들기) |
| book{obi, persist_share} | 호가 래스터 파일 요약(`_heatmap_read`) | **없음 → null** · `missing` | 남음 — 래스터 수집기 파일 |
| oi_delta | 바이낸스 OI hot DB | 없음 | classify 가 안 쓴다 |

일치 시험(tests/situation_parity.rs): 정답지는 `rtcore/tools/gen_situation_fixture.py` 가 만든다. 원본 `situation.classify`·`micro_ref.deriv_from_ring` 은 import 해서 부른다.
`_situation_inputs` 는 make_app 안에 닫힌 함수라 import 가 안 되어 글자 그대로 옮겨 적었다. server.py 를 고치면 그 사본도 맞춰야 한다.
마크 링 넣기·정리는 웹소켓 루프 안이라 옮겨 적었다. 합성 37경우 1,290걸음(ok 944: 하락 344·횡보 77·상승 523 · 거부 봉 181 · 베이시스 임계 있음 242 · BTC 라벨 698 · lead≠0 41)을 JSON 값 비트 단위로 대조했다 — 어긋남 0.
경우 종류에는 강한 직접 반전(flip), 캔들 없음과 가격 0 인 초(zero_sec), 델타 .5 동률(ties), 임계 표본이 59→60 을 넘는 순간(grow), 정수 초 수신(링 7,200초 경계 동률)이 들어 있다.
양성 대조: `python3 rtcore/tools/mutate_situation.py`(27 변형). 못 잡는 두 변형은 이렇다.
- 베이시스 주도 경계 `>=`→`>`: |Δ베이시스| 가 임계와 비트까지 같아야 갈린다.
- 봉 13→12: classify 는 걸러진 봉의 마지막 7개만 쓴다. 7개 넘는 구멍 봉이 있어야 갈린다 — 사실상 등가다.

알려진 차이: 봉 칸 합(델타·VWAP)은 가격 키 순서로 더한다. 원본은 칸이 생긴 순서로 더한다. 마지막 비트가 갈려도 cvd 는 정수로 반올림된다. 거부 봉 판정은 경계 1ulp 안에서만 갈린다. 정답지는 칸을 키 순서로 넣어 이 차이를 대조 밖에 둔다.

## 2. 미시 루프(`collect_micro_ref` payload) — 칸별

| 칸 | 원천 · 주기 | REST/DB | 코어 |
|---|---|---|---|
| mid · spread | REST bookTicker 1초 | REST | 받는 `@bookTicker` 로 바로 셀 수 있다 — 남음(이번 조각은 situation 만) |
| qi(→ agree) | bookTicker 수량 → `mref.qi` | REST | 셀 수 있다. 다만 agree 는 OFI 가 필요하다 → 남음 |
| agree · trigger · imb40 · sweep | 호가 래스터 파일 `read_window` 12초 → `raster_flow`·`sweep_depth` | 래스터 파일 | 남음 — 래스터 수집기(rtcollect) 몫 |
| vol60 | 1초 링 c[4]+c[5] 마지막 60초(진행 초 뺌) | 없음 | 셀 수 있다 |
| vol60_pct · act | vol60 의 시별 분위(`baseline_from_tape` — 테이프 DB 1시간마다) | 테이프 DB | 남음(기준선 읽기) |
| sr | 청산맵(`load_liquidation_map`) | 파일·REST | 남음 |
| liq60 | 바이낸스 청산 사건 60초 합 | 없음 | 셀 수 있다(supply.rs liq 링) |
| (시장 맥락 flow_scales 등) | 테이프 DB 시별 분모 | 테이프 DB | 남음 |

`compute_situation` 의 나머지(카드30 모델·`_flow_read_ctx`·`fr.read` 융합·장부 쓰기·`vol_mult`)는 REST 5분봉 하루치·모델 파일·Bybit DB 가 필요하다 → 남음(rtmodel·rtserve 쪽).

## 3. rtserve 두 엔드포인트의 코어 몫 — 결론

**`/api/oi-5m`(바이낸스 OI 1초): 코어 범위 밖이다.** 바이낸스는 OI 를 웹소켓으로 안 준다. 원본은 REST `/fapi/v1/openInterest` 폴링을 수집기 `live_binance_ctx_collector` 하나만 한다(IP 한도 2,400/분 — coin_flow.py 641~645). 그 수집기가 hot SQLite `oi_1s` 에 쓰고, 서버는 그 표를 0.5초마다 꼬리만 읽는다(`collect_oi_1s`). 5분 봉은 `oi_5m_buckets` 가 같은 표에서 센다.
→ 쓰기는 rtcollect(수집기)가, 읽기·5분 봉은 rtserve 가 hot SQLite 를 직접 맡는다. OKX 합산 몫(`okx_oi_5m`)은 코어가 이미 받는 OKX OI 사건(`Event::Oi` · supply.rs okx_oi)으로 셀 수 있다 — 필요하면 Liq5m 처럼 5분봉 [처음, 끝]을 밀면 된다(지금은 안 함).

**`/api/liquidation-5m-history`(OKX·Bybit·HL 합산): 둘은 되고 하나는 안 된다.**
- 바이낸스 몫: 이미 `Liq5m` 이다(원본 메모리 `liq_5m` 과 같은 칸 · quote_parity 대조).
- OKX: 원본은 메모리 `okx_liq_events` 를 5분봉으로 더한다(`bar > okx_fp["first_bar"]` 인 봉만). 코어는 같은 OKX 청산 사건(`Event::Liq` venue=okx)을 받는다. 다만 지금 링은 660초뿐이다(1초 수급용). liq5m 과 같은 5분 누적 하나와 «첫 봉» 경계를 더하면 된다.
- Bybit: 원본은 Bybit 청산 DB(`bybit_liq_events` · 수집 시작 bstart 뒤 온전한 봉만)를 읽는다. 코어도 Bybit 청산 사건을 받는다(BYBIT_SYMBOLS 만). 그러나 경계가 «코어가 받기 시작한 뒤»라 재시작 직후 봉이 원본과 다르다. 재시작 복원(restore.rs read_bybit_liq)이 660초만 메우기 때문이다. 같게 하려면 복원 창을 144봉으로 늘려야 한다.
- HL(Hyperliquid, `_with_hl_liq`): 코어가 받지 않는 원천이다 → rtcollect/rtserve 몫.
- 결론: rtserve 가 코어 `Liq5m`(바이낸스)에 OKX·Bybit 5분 누적을 코어에서 받고, HL 은 자기가 읽어 합치는 구조가 맞다. 코어에 OKX·Bybit 5분 누적을 더하는 것은 다음 조각이다(지금은 안 함).
