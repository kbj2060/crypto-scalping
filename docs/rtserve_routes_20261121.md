# 응답 서버(rtserve · 꾸러미 R6) — 라우트 전수 조사와 첫 조각 (2026-11-21)

원본 = 파이썬 대시보드 응답 서버(aiohttp · 127.0.0.1:8787 · `dashboard/server.py` 5,148줄 + `dashboard/orders.py` 핸들러). 행 번호는 2026-11-21 브랜치
`claude/rust-migration-w2-20261120`(6bda3c48) 판. graft 색인은 낡아 행이 수백 줄 어긋나 있다 — 행은 전부 파일을 직접 읽어 확인했다.
Rust 판 = `rtserve/`(새 크레이트). 설계 문서 `docs/rtcore_design_20261120.md` §11 R6.

## 0. 요약

- 등록부(server.py 4932~5025) = **84개**: 화면 문서·정적 6(`/` · `/dashboard/live` 셋 · sw.js/manifest · add_static) + **`/api/*` 78**(GET 61 · POST 17) + SSE 2(그 78 안).
- `/api/*` 78 을 «무엇을 읽어 답하나»로 나누면: 파일·SQLite 만 21 · 파일 쓰기/설정(POST·푸시) 6 · 저장소+계산(REST 없음) 7 · 프로세스 메모리 상태 10 · 요청 때 외부 REST/모델 20 · 주문·계좌 14.
- **첫 조각(main c12b7cc8 · PR 69)**: 공통 동작 전부(미들웨어 둘 · 정적 파일 · index 302 · 버스터 · pwa · 404/405/500) + 엔드포인트 12(파일·SQLite 만 읽는 것).
  정답지 = 원본 함수를 진짜 aiohttp 객체(라우터·미들웨어·Response.prepare)로 돌려 뜬 연결 바이트 · 요청 113건(압축 바이트 포함) 일치 · 의도한 차이 1종 3건(§5).
- **둘째 조각(2026-11-21 · 미커밋)**: 나머지 파일·SQLite 9(state · breakout-detector · gex · chip-score · notify-center · news · macro-spikes · bubble-tape · pred-record)
  + 쓰기 4(acct-notes·acct-wrong POST · push/subscribe·unsubscribe) + 설정 읽기 2(push/config·devices) + client-timing + order_request_log(요청 111 · 쓴 파일 대조 184 일치)
  + rtcore 허브 구독 → `/api/supply-1s` · `/api/footprint` · SSE `/api/stream`(수급) · `/api/events`(rtcore 시험 정답지의 파이썬 행과 비트 일치). 양성 대조 §6.
- **셋째 조각(2026-11-21 · 미커밋)**: 저장소+계산 7 중 2(session-alerts · liquidation-5m-signal · DuckDB 묶음) + swr 캐시(원본과 같은 ttl·max_stale) + 처리 시간 벤치(§4 셋째 조각 표).
- **넷째 조각(2026-11-22 · 미커밋)**: 주문·계좌 14 프록시(rtorder 문 · 진짜 rtorder 실행 파일로 route 정답지 42건 바이트 일치) + `/api/events` acct_seq(rtorder `/internal/acct` 1초)
  + `/api/oi-5m`(hot oi_1s + 코어 Micro.okx_oi5m · 요청 7) + `/api/liquidation-5m-history`(코어 Liq5m + hot OKX·HL · 요청 12) + 모델 5(dir4h · dir-short · vol-forecast · open-rev · dir-signals = rtmodel::handlers ·
  가짜 REST 녹화본 · dir 5벌 payload 비트 + 요청 545 같은 열 · 본문 글자 4 · QX4 배선 · 밴·낡음). §4 넷째 조각 · §5 #23~#33.
- **다섯째 조각(2026-11-22 · 미커밋)**: 병행 가동 준비(§8: `supervisor_rtserve.sh` · `RTSERVE_READONLY` · REST 기본 꺼짐 · `tools/shadow_http.py`) + 일봉·4시간 풋프린트 4 라우팅
  (api3.rs → fpd.rs · 요청 42 불일치 0 · 400 갈래 asset·bad_date·bad_key·from_to·range) + flow/heatmap «체결 대조 반올림» 확인(§7 1번 -- 코어 원값 칸은 필요 없음).
- **여섯째 조각(2026-11-22 · 미커밋)**: `/api/flow/heatmap`(raster_read.rs 읽기 + heatmap.rs 계산 · 서버 판 numpy 2.2.5 정답지 비트 일치) + 서버 병행 대조 원인 가르기(§8 표)
  + 결함 고침: supply-1s `oi` 가 늘 [] 이던 것(원본도 hot oi_1s 를 읽는다 -- §5 #13 정정) · vol-forecast QX4I(Deribit IV 칸) 붙임(rtmodel 새 함수).
- **옮김 합계(2026-11-22 · 여섯째 조각 뒤)**: `/api/*` 78 중 **60** + 화면·정적 6 = **66 / 84**. 남은 것 = §7.
  (다시 셈 -- 묶음별: 파일 21/21 · 쓰기·설정 6/6 · 저장소+계산 6/7 · 메모리 상태 7/10 · 요청 때 REST·모델 5/20 · 주문 14/14 = 59. 앞 조각 보고의 «33 → 54 → 60/84»는 하나 덜 센 값이었다.)
- **아홉째 조각(2026-11-22 · 미커밋 · §10)**: 요청 때 REST 14주소 표 · 바이낸스 한 문(`model::binance_get` = 밴·가중치 1,200 넘은 분 거절·간격 200ms · rtmodel::gate::Gate 재사용) ·
  `/api/trend` · `/api/cycle-path` · `/api/swing-lines` 옮김(정답지 33/33). 합계 66 → **69 / 84**(+ /api/situation). sr-watch 는 청산맵 계산에 막힘 · position-profile 은 다음.
- 인증·세션: **원본 서버 안에 없다.** 127.0.0.1 에만 묶이고, 공개 경로는 Cloudflare 터널(`dashboard/scripts/start_cloudflare_tunnel.sh`) + Cloudflare Access 가 막는다
  (`dashboard/scripts/supervise_server.sh:31~32` 주석 · `docs/rtcore_browser_route_20261120.md:22`). «인증 없이 열면 `/` 는 302, 그 밖은 404» 로 보인 것은
  원본 라우터 그대로다(`/` = `HTTPFound("/dashboard/live/")`, 모르는 주소 = `404: Not Found`). rtserve 도 인증을 새로 만들지 않고 루프백이 아니면 시작을 거절한다.

## 1. 공통 동작 (원본 위치 → rtserve)

| 동작 | 원본 | 내용 | rtserve |
|---|---|---|---|
| 미들웨어 순서 | 1381 | `[order_request_log, static_asset_headers, json_compress_etag]`(앞이 바깥) | 같은 순서(`lib.rs handle`) |
| `json_compress_etag` | 1325 | GET·200·`application/json`·본문 있음 → ETag 없으면 `W/"body-<sha256 앞16>"` + `Cache-Control` 기본 no-cache · If-None-Match(쉼표로 갈라 공백 벗긴 글자 = `*` 또는 같음)면 304(ETag·Cache-Control 만) · 압축 켬 | 같음 |
| 압축 | aiohttp 3.13/3.14 | **문턱 없음**(모든 JSON·index.html·ASSET_FILES). Accept-Encoding 소문자에 «deflate» 글자가 있으면 deflate(zlib 머리) 먼저, 없으면 «gzip» — 부분 글자 일치라 브라우저(`gzip, deflate, br, zstd`)는 **deflate** 를 받는다. C zlib 수준 6. 4096 바이트는 «압축을 실행기로 넘기는» 문턱(`_zlib_executor_size`)일 뿐 | flate2 + C zlib(같은 수준·같은 머리) → 압축 바이트까지 같음 |
| `static_asset_headers` | 1287 | 요청 경로 마지막 이름이 ASSET_FILES(app.js·styles.css·fp_daily.js·liq_profile.js·bubble_map.js)면 no-cache+압축 · `.ttf` 면 1년 immutable · `.html` 이면 no-cache | 같음 |
| NOCACHE | 1044 | `{"Cache-Control": "no-cache"}` — 거의 모든 JSON 핸들러가 붙임 | `Resp::json`·`Resp::bytes` 기본 |
| JSON 직렬화 | aiohttp `json_response` | `json.dumps` 기본(", "·": "·ensure_ascii·float repr) · `application/json; charset=utf-8` | `rtcollect::py_dumps_default`(path 의존성) |
| `json_response()`(명시 ETag) | 1270 | `/api/state` 한 곳만(`make_etag("state", 파일 서명)`) | 다음 조각(`/api/state` 는 화면이 안 부른다) |
| 정적 파일 | 5025 `add_static("/dashboard/live/", show_index=False)` | FileResponse: Content-Type = Python 3.11 `MimeTypes()` 기본표(js = `application/javascript` · ttf·woff = octet-stream) · `Etag "<mtime_ns 16진>-<크기 16진>"` · `Last-Modified` = ceil(st_mtime) · `Accept-Ranges: bytes` · If-None-Match(약한 비교)·If-Modified-Since → 304 · 폴더 = 403 `403: Forbidden` · 없는 파일 = **본문 없는 404**(청크·octet-stream · ASSET 이름이면 압축된 빈 흐름) · 폴더 밖(`..`·심볼릭 링크) = 404 | 같음. 🔴신뢰 경계: 정규화 뒤 폴더 밖이면 404 · 절대 경로·NUL 거절 · 심볼릭 링크로 나가는 것도 404 — Rust 전용 시험 10경로 |
| 캐시 버스터 | 3593·3625 | index.html 의 `<ASSET>?v=…` 를 ASSET_FILES **내용** sha256 앞 12자로 바꿈 · 매 요청 다시 해시 | 같음 |
| 꺼진 코인 탭 | 3609 | `<button … data-asset="x">` 의 x 가 `DASHBOARD_ASSETS`(env · 기본 eth,sol,xrp)에 없으면 ` hidden>` | 같음(정규식 대신 손으로 · 같은 결과) |
| pwa | 3635 | sw.js·manifest = FileResponse + no-cache | 같음 |
| 예외 응답 | aiohttp web_protocol | `HTTPException` 은 미들웨어를 안 지나고 `"{status}: {reason}"` text/plain · 400 은 상태 줄 글자도 reason(`400 unsupported_asset`) · 405 는 `Allow: GET,HEAD` · 잡히지 않은 예외 = 500 `500 Internal Server Error\n\nServer got itself in trouble` + 연결 닫음 | 같음(상태 줄 글자 포함) |
| 요청 기록 `order_request_log` | 1352 | POST `/api/manual-*` 만: KST 시각·경로·쿼리(**키마다 첫 값**)·상태·error·ms·UA 200자·IP(CF-Connecting-IP → X-Forwarded-For → 접속 IP) → `data/live/order_requests.jsonl` 한 줄 덧붙이기. 예외(404·405)도 상태와 함께 적힌다 | **옮김**(`api2.rs order_request_log` · 넷째 조각부터 rtorder 로 넘긴 요청은 rtorder 가 적고 rtserve 는 안 적는다 -- 502(문 닫힘)·404·405 만 rtserve 가 적는다) |
| M0 `client_timing` | 5102 | POST `/api/client-timing` 본문 1KB·아는 키·숫자만 → 로그 한 줄 · 0.2초 상한 | **옮김**(로그 = `Config.log` · 실행 파일은 표준 출력) · 허용 숫자 칸 = 원본 10칸 + rt_* 9칸(main 4ff90f88 · 정답지도 같은 19칸) + 🔴머지 대기 2칸(rt_draw_s · rt_back_n · 원본 PR 전 -- 정답지 밖 · Rust 시험 rt_timing_keys) |
| swr 캐시 | 1460~1500 | 키별 ttl·max_stale · 묵은 값 먼저 주고 뒤에서 갱신 | **안 옮김**(이 조각의 읽기는 µs) — §5 |
| 시작·종료 | 5030~5043 · main 5123 | 시작: http 세션 · SSE 발행 · 흐름(코인별 WS) · 미시 루프 6종 · 손절 감시 · 대기 정리 · 사용자 스트림 · 멈춤 감지 / 종료는 역순 · `shutdown_timeout=2.0`(SSE 때문) | 시작 = 묶기만 · Ctrl+C 끝 · 백그라운드 루프 없음(§3) |

### SSE 두 개

| 주소 | 주기 | 프레임 | Rust 몫 |
|---|---|---|---|
| `/api/events`(3651) | `EVENT_POLL_SECONDS=1.0`(1036) | 접속 즉시 `data: {state, tickers, assets}` · 그 뒤 큐(maxsize 1 · 넘치면 옛 것 버림)에서 `data: {state(바뀐 때만 sse_state_view 3칸, 아니면 null), tickers, assets, acct_seq}` · 20초 조용하면 `: keepalive` | 내용은 rtcore 허브(티커 = Prices) + `dashboard_state.json`(rtduck market_data 가 씀) + 계좌 순번(rtorder 사용자 스트림) — rtserve 가 이어 붙인다(다음 조각) |
| `/api/stream`(3130) | `STREAM_TICK_S=0.25`(973) | 연결마다 루프 · `?supply=1` 이면 매 틱 `event: supply` + `supply_1s_payload`(4181 · 연결별 커서 8종) · situation 계산 시각이 바뀌면 `event: situation` | rtcore 허브 `/rt` 가 이미 같은 수급 칸을 민다(Supply1s) — 화면이 허브로 옮기면 이 SSE 는 지운다. situation 은 계산 엔드포인트 조각 |

## 2. 라우트 전수 표

원천: M = 이 프로세스 메모리(수집 루프가 채움) · F = 파일 · S = SQLite · D = DuckDB · R = 요청 때 외부 REST · C = 계산(pandas·numpy·모델).
화면 주기 = `dashboard/live/app.js` 의 `tick()`(500ms 허브 · 숨으면 건너뜀) 안 자체 게이트 · `docs/rtcore_display_latency_map_20261120.md` 보완(그 문서의 행 번호는 낡았다).
«Rust 대응물»: 이미 있는 것. 순서 = 제안(1 이 먼저).

### 2-1. 파일·SQLite 만 (21) — rtserve 가 직접 읽는다

| 주소 | 핸들러(행) | 읽는 것 | 캐시 | 화면 주기 | Rust 대응물(쓰는 쪽) | 상태 |
|---|---|---|---|---|---|---|
| `/api/acct-notes` GET | 2866 → acct_notes.load | F `data/live/acct_notes.json` | — | 처음 한 번 | — | **옮김** |
| `/api/acct-wrong` GET | 2878 → acct_notes.wrong_load | F `data/live/acct_wrong.json` | — | 탭 열 때 | — | **옮김** |
| `/api/risk-radar` | 2799 → risk_radar.py:20 | F `risk_radar_latest.json` + 계수 json | swr 10/120 | 30초 | rtmodel `bin/rt_radar` 가 씀 | **옮김** |
| `/api/sr-lean` | 2792 → sr_lean.py:22 | F `sr_lean/ethusdt_state.json`·jsonl 꼬리 256KB | swr 1/30 | 2초(ETH) | rtmodel `bin/rt_sr_lean` | **옮김** |
| `/api/regime-band` | 2785 → regime_band.py:22 | F `regime_band_forward.jsonl` 꼬리 32KB | swr 60/7200 | 5분 | rtmodel REMAINING #5(크론) | **옮김** |
| `/api/liq-tab` | 2808 | F `liq_tab_{asset\|all}.json` 바이트 그대로 | mtime 메모 | 탭 10분 | 안 옮김(rtcollect REMAINING: liq_tab 크론) | **옮김** |
| `/api/iceberg` | 2958 | F `iceberg/{eth,xrp}_state.json`·`_events.jsonl` | — | 30초 | rtcollect `iceberg` | **옮김** |
| `/api/ob5m` | 2982 | S `ob5m/{a}usdt.sqlite`(bars·live·marks) | swr 4/600 | 5초 | rtcollect `ob5m` | **옮김** |
| `/api/treasury-futures` | 2110 → 596 | S `data/hot/treasury_futures.sqlite` | swr 60/600 · h=24 300/1800 | 60초 · 5분 | rtcollect `treasury_futures` | **옮김** |
| `/api/xyz100` | 2121 → 622 | S `data/hot/hl_xyz.sqlite` | since 없을 때 swr 5/60 | 비교 화면 1초 · 아니면 60초 | rtcollect `xyz` | **옮김** |
| `/api/macro-today` | 2117 → 646 | S hl_xyz + `macro_5m.sqlite` | swr 30/600 | 60초 | rtcollect `xyz`·`macro_5m` | **옮김** |
| `/api/macro-calendar` | 4836 → 1150 → worker_state.py:31 | F `macro_calendar_state.json`(90분 신선도) | swr 6h/600 | 6시간 | 안 옮김(거시 달력 워커) | **옮김** |
| `/api/macro-spikes` | 2130 → 712 | S treasury + hl_xyz · `statistics.stdev` · 뉴욕 시각(CME 개장) | swr 30/120 | 30초(비교 화면) | — | **옮김**(statistics.stdev 정확 유리수 = num-bigint · 미국 서머타임 규칙) |
| `/api/news` | 2096 → 729 | S `data/hot/news_rss.sqlite` · `html.unescape`·정규식 | swr 15/300 | 30초 | 안 옮김(뉴스 수집기) | **옮김**(html.unescape = `rtcollect::html`(뉴스 수집기와 한 표) · news_key 정규식 다섯 줄 손으로) |
| `/api/chip-score` | 2088 → 303·321 | F `chip_score_YYYYMM.jsonl` 두 달 | swr 60/600 | 5분 | — (쓰는 쪽 = 서버 안 chip_score_loop · §3) | **옮김**(흐름 코인 = FLOW_SPECS 열쇠 ∩ DASHBOARD_ASSETS) |
| `/api/notify-center` | 2134 → 336 | F verdict_calendar·push_sent_log·liqmap_score_status + 상수 PUSH_KINDS(scripts/live_push_notifier:131) | — | 5분 | — | **옮김**(PUSH_KINDS 를 옮겨 적음 -- 알림 데몬 모듈은 import 때 환경 파일을 읽어 불러 쓰지 않았다 · 바뀌면 둘 다) |
| `/api/state` | 3647 | F `dashboard_state.json` + 명시 ETag | json_cache | 안 부름(SSE 가 같은 원천) | rtduck market_data | **옮김**(명시 ETag `W/"state-…"` · ensure_ascii=False 압축형) |
| `/api/breakout-detector` | 4456 | F `{a}_breakout_detector_state.json` | swr 60/600 | **안 부름** | (워커 꺼짐 · 질문지 20) | **옮김**(원본이 지금 내는 그대로: 파일 없음/낡음 → available false · 지울지는 사용자) |
| `/api/gex` | 4715 → 1208 | F deribit_gex·cboe_option·block_trades state | swr 60/600 | 60초 | rtcollect `cboe_chain`(반쯤) | **옮김** |
| `/api/bubble-tape` | 1107 → 1071 | S 체결 테이프 hot 셋(바이낸스·OKX·Bybit) | 자체 1초 | 1초(bubble_map.js) | rtcollect 테이프 셋 · rtcore 테이프 | **옮김**(503 본문 error = sqlite3 예외 이름) |
| `/api/pred-record` | 2890 → pred_record.py:187 | F `pred_record/*.jsonl` 꼬리 8MB + **M** `_xs_latest` | swr 30/600 | 60초 | rtmodel `bin/rt_dirpred`(기록) | **옮김**(xsec.latest 는 늘 null -- §5 #12) |

### 2-2. 파일 쓰기·설정 (6) — rtserve

| 주소 | 핸들러 | 하는 일 | 비고 |
|---|---|---|---|
| `/api/acct-notes` POST | 2869 → acct_notes.save | 날짜·글 검증 → `acct_notes.tmp` 에 쓰고 rename | **옮김** · 400 = ValueError 글자 |
| `/api/acct-wrong` POST | 2881 → wrong_add·wrong_del | 한 줄 더하기(id = max(기존, 지금 ms−1)+1 · 공백 접기)·지우기 → `acct_wrong.tmp` → rename | **옮김** |
| `/api/push/config` · `/devices` GET | 4856·4891 | env VAPID 공개키 + 구독 파일 | **옮김** · 비밀키는 «있는지»만 든다(값은 안 든다) |
| `/api/push/subscribe` · `/unsubscribe` POST | 4868·4879 | 구독 파일(json.dump indent=2) → `.json.tmp` → rename | **옮김** · 같은 기기 다시 구독 = 이름·시각 유지 |

`/api/push/test`(4904 · 웹푸시 발송 = 외부 접속)는 2-5 묶음 -- 파이썬에 둔다. 쓰기 핸들러는 `Config.write_lock` 하나로 읽기-고치기-쓰기를 한 줄로 세운다(원본은 이벤트 루프 하나라 저절로 한 줄).

### 2-3. 저장소 + 계산(요청 때 REST 없음) (7) — rtserve(계산은 Rust 로 다시)

| 주소 | 핸들러 | 읽는 것 · 계산 | 화면 | rtserve |
|---|---|---|---|---|
| `/api/footprint-daily` · `/cells` | 3861·3868 → footprint_daily.py | D/F lake + S hot 테이프·맥락 → DailyFootprint | fp_daily.js 60초·범위 바뀔 때 | **옮김**(계산 fpd.rs = 다른 에이전트 d1a40656 · 라우팅 api3.rs · swr 30/600 · 칸 키 800 넘으면 오래된 절반 정리) |
| `/api/footprint-4h` · `/cells` | 3898·3904 | 같은 원천 h4 | 〃 | **옮김**(4h cells 는 원본처럼 범위 검사 없음) |
| `/api/liquidation-5m-signal` | 4751 | D tail_risk(rtduck market_data 가 씀) + 신호식 | 60초 | **옮김**(DuckDB 1.5.1 묶음 · swr 1/0) |
| `/api/flow/heatmap` | 4692 → 4482·4523 | F 호가 래스터 + numpy 행 집계 + **M** 풋프린트(체결 대조) | 1초 | **옮김**(여섯째 조각 · raster_read.rs · heatmap.rs) |
| `/api/session-alerts` | 4840 | F 거시 달력 + compute_macro_event_alert | 30초 | **옮김**(달력은 macro-calendar 와 같은 6시간 캐시 · 경보는 매번 계산) |

### 2-4. 프로세스 메모리 상태 (10) — rtcore 허브/상태와 잇기

| 주소 | 핸들러 | 원본 상태(채우는 루프) | rtcore 대응물 | rtserve |
|---|---|---|---|---|
| `/api/events` SSE | 3651 | latest_event_*(publish_dashboard_events 1초 · 티커 REST) | Prices(티커) · 계좌 순번은 rtorder | **옮김**(첫 프레임·1초 발행·20초 keepalive · acct_seq = rtorder /internal/acct -- §5 #14) |
| `/api/stream` SSE | 3130 | footprint_state·oi_1s·okx_*·spot_*·liq_events_by·situation_state | Supply1s(밀기) — 화면이 허브로 가면 지움 | **수급만 옮김**(🔴situation 이벤트 없음 -- §5 #15) |
| `/api/supply-1s` | 4341 | 같은 supply_1s_payload | Supply1s(실서버 그림자 일치 · 설계 문서 진행 기록 5·6) | **옮김**(oi 는 [] -- §5 #13) |
| `/api/footprint` | 3971 | flows[a].footprint_state + fp_dhl(S) | Footprint(봉 623·칸 1,637 일치) | **옮김**(agg 늘 false · dhl 코어 판 -- §5 #16) |
| `/api/situation` | 4721 | situation_state·fo_state·mp_state(collect_micro_ref 1초 · compute_situation) | 없음 — 계산 조각 | 다음 |
| `/api/sr-watch` | 2041 | sr_watch(sr_break_loop 10초) | 없음 · 화면이 안 부름(SR 근접 알림 워커만) | 다음 |
| `/api/oi-5m` | 4363 | S oi_1s + M oi_1s·okx_oi_5m | Micro.okx_oi5m·okx_from | **옮김**(바이낸스 = hot oi_1s 그대로 · 현재 OI = hot 마지막 행 -- §5 #25) |
| `/api/liquidation-5m-history` | 4413 | M liq_5m·okx_liq_events·bybit + S OKX(first_bar 앞)·HL 청산 + D tail_risk | Liq5m(.okx·.bybit) | **옮김**(코어에 없는 코인 = 빈 봉 -- §5 #26) |
| `/api/market-context` | 2698 | M micro_state·mark_ring·mid_ring·depth25 + S/D 수집기 + R Deribit | Mark·Book · 나머지 계산 조각 | 다음 |
| `/api/client-timing` POST | 5102 | 로그 한 줄 | — | **옮김** |

### 2-5. 요청 때 외부 REST·모델 (20) — 순서 맨 뒤 · 대부분 rtmodel 서빙을 붙이거나 워커 파일로 바꾼 뒤

`/api/market-history`(3682 · 5분봉 REST + trend_veto) · `/api/footprint-week`(4164 · S + REST 5분봉) · `/api/footprint-daily/xcvd`(3834 · 현물·OKX·Bybit REST)
· `/api/footprint-daily/heat` · `/api/footprint-4h/heat`(1시간봉 REST) · `/api/opt-smile`(4730 · Deribit) · `/api/chart-markers`(4738 · 안 옮김 — 사용자 결정)
· `/api/liquidation-map`(4755) · `/api/liquidation-map/tiers`(4828) · `/api/position-profile`(4766) · `/api/cycle-path`(4801) · `/api/trend`(2724 · 1d 120봉)
· `/api/swing-lines`(2743) · `/api/dir-signals`(2777 · Coinbase·바이낸스·FRED) · `/api/vol-forecast`(3072 · rtmodel `qx4`) · `/api/dir4h`(2915 · rtmodel `dir4h`)
· `/api/dir-short`(2945 · rtmodel `lgb`·`dirshort`) · `/api/open-rev`(3109) · `/api/macro-trend`(2106 · FRED·DefiLlama) · `/api/push/test`(4904 · 웹푸시 발송).
🔴개발기에서는 이 묶음의 REST 를 부르면 안 된다(서버와 IP 한도 공유) — 대조는 응답을 녹화한 가짜 서버로만.
**넷째 조각에서 옮김 5**: `/api/dir4h` · `/api/dir-short` · `/api/vol-forecast` · `/api/open-rev` · `/api/dir-signals`(`src/model.rs` → `rtmodel::handlers` · §4 넷째 조각). 나머지 15 는 §7.

### 2-6. 주문·계좌 (14) — rtorder intake 붙이기

`/api/binance-account`(orders.py:778 · swr 10/600 · 사용자 스트림이 순번을 올리면 바로) · `/api/manual-entry/preview`(1633) · `/submit`(1669 · busy_or)
· `/status`(2591 · 3초/진행 중 1.5초) · `/wait-action`(2610) · `/cancel`(2654) · `/limit-amend`(2548) · `/api/manual-bracket/disarm`(2050) · `/rearm`(2082) · `/tp`(2260)
· `/tp-legs`(2215) · `/add`(2345) · `/modify`(2442) · `/api/manual-exit/submit`(2002).
Rust: `rtorder/src/intake.rs` `Desk::api_*`(접수 핸들러가 HTTP 와 떨어진 함수로 이미 있다 · 정답지 = 원본 핸들러 대조 139건) — rtserve 가 쿼리를 넘기고 응답을 내면 된다.
주문 경로 공통: `DASHBOARD_MANUAL_EXEC_ENABLED`(꺼지면 403) · `confirm=1` · busy 409 · order_request_log.
**넷째 조각: 14개 전부 rtorder 문(따로 뜨는 프로세스 · `RTSERVE_RTORDER` 기본 127.0.0.1:8791 · 루프백만)으로 그대로 넘긴다**(`src/order.rs` · 규칙 = rtorder/SHADOW.md §7):
메서드·경로·쿼리 원문 · User-Agent · CF-Connecting-IP · Content-Type · X-Forwarded-For(화면이 안 보냈으면 접속 주소) · 본문 → 받은 상태 줄·Content-Type·Cache-Control·Allow·본문 그대로(ETag·압축은 rtserve 미들웨어 = 원본도 바깥 미들웨어).
시간 제한 15초(rtorder 칸 기다림 5초보다 길게). 접수 기록(order_requests.jsonl)은 rtorder 가 적는다 -- rtserve 는 넘긴 요청을 안 적고, 문이 닫혀 502 를 낸 요청·404·405 만 적는다(같은 줄이 두 번 생기지 않게).

### 2-7. 삭제됨

| 주소 | 언제 | 근거 |
|---|---|---|
| `GET /api/manual-exit/preview` | 2026-11-21 PR 61(fbf7cd92) | 플로팅 버튼 제거 — 사용자 «sltp 만» |
| `POST /api/manual-exit/submit-all` | 〃 | 〃(server.py:5017 주석) |
| `GET /api/box-hold` | 2026-11-21 PR 57 | 박스 유지 서빙 끔(server.py:3123 주석 · `dashboard/box_hold.py` 와 joblib 은 파일로 남아 있다) |
| `/data/live/*` 정적 | 2026-10-19 | 보안 수정 — 다시 열지 말 것(5026 주석) |

### 2-8. 화면 문서·정적 (6)

`/`(302) · `/dashboard/live` · `/dashboard/live/` · `/dashboard/live/index.html`(셋 다 dashboard_index) · `/dashboard/live/{sw.js|manifest.webmanifest}` · add_static — **전부 옮김**.

## 3. 백그라운드 루프 — 누구 몫인가 (rtserve 는 떠안지 않는다)

| 원본(시작 훅) | 하는 일 | 몫 |
|---|---|---|
| `start_flows`(1732 → coin_flow.make_coin_flow) | 코인별 바이낸스 체결·현물·OKX WS + OI 1초 → footprint_state·oi_1s | **rtcore**(feed·footprint·supply · 실서버 그림자 일치) |
| `collect_force_orders`(1816) | `!forceOrder@arr` WS → liq_events·liq_5m | **rtcore**(Liq·Liq5m) |
| `collect_mark_price`(1854) | `@markPrice@1s` → mark_ring | **rtcore**(Mark) |
| `collect_micro_ref`(1893) | 1초: bookTicker REST·래스터·청산맵 → micro_state + compute_situation | rtcore(Book 은 받음) + 상황 계산 = 다음 «계산» 조각(rtserve 또는 rtcore) |
| `chip_score_loop`(2005) | 10초: 청산 급증 배지 → 15분 뒤 되돌림 jsonl | rtcore(배지 원천이 Liq) — 결정 필요 |
| `sr_break_loop`(2044) | 10초: 청산맵 지지·저항 뚫림 → 텔레그램 · sr_watch | 범위 밖(알림) + 청산맵 계산 조각 |
| `pred_record_loop`(2840) | 5분봉+20초: dir_short·dir4h 계산·기록 + xsec | **rtmodel** `bin/rt_dirpred`(이미 있음) |
| `publish_dashboard_events`(3524) | 1초: 상태 파일 서명·티커 REST·계좌 원장 | rtserve(SSE 조립) + rtcore(티커) |
| `order_api.start_bracket_watcher`·`start_wait_cleanup`·`start_user_stream` | 손절·익절 감시 · 재시작 정리 · listenKey | **rtorder**(runner·감시·사용자 스트림) |
| `start_loop_watch`(5054) | M0 멈춤 감지 | 파이썬 몫(Rust 는 asyncio 루프가 없다) |
| `start_http_session` | 공용 aiohttp 세션 | 해당 없음 |

## 4. rtserve 첫 조각

- **HTTP 스택**: hyper 1(+hyper-util 의 tokio 어댑터 · http-body-util). 라우트가 정해진 글자 목록이라 axum·tower 의 라우터·미들웨어 틀이 할 일이 없고, 원본과 헤더·상태 줄·청크 전송까지 바이트를 맞추려면 응답을 손에 쥐는 쪽이 짧다. tokio 1.54 는 형제 크레이트와 같은 판.
- **설정**: `RTSERVE_ROOT`(필수 · 저장소 뿌리) · `RTSERVE_ADDR`(기본 127.0.0.1:8790 · 루프백 아니면 종료코드 2) · `DASHBOARD_ASSETS`(원본과 같은 뜻 · 기본 eth,btc,sol,xrp,sndk,soxl,skhynix,mu) · 둘째 조각부터 `RTSERVE_HUB`(rtcore 허브 주소) · `VAPID_PUBLIC_KEY`·`VAPID_PRIVATE_KEY`(원본과 같은 이름 · 비밀키는 있는지만).
- **옮긴 엔드포인트 12**: acct-notes·acct-wrong(GET) · risk-radar · sr-lean · regime-band · liq-tab · iceberg · ob5m · treasury-futures · xyz100 · macro-today · macro-calendar.
  원본 구조대로 «워커가 쓴 파일을 읽어 돌려줌»만 한다(rtmodel 은 쓰는 쪽이라 의존하지 않는다).
- **정답지**: `rtserve/tools/gen_http_fixture.py`. 원본 함수 본문을 옮겨 적고(대시보드 서버 모듈은 import 하지 않는다 — import 만으로 수집기·키를 끌어온다)
  별도 모듈(sr_lean·risk_radar·regime_band·acct_notes·worker_state·data_store)은 그대로 불러, 진짜 `web.Application`(라우터·미들웨어) + `make_mocked_request` +
  바이트를 모으는 가짜 전송으로 연결 바이트를 뜬다. 서버를 띄우지 않고 소켓도 안 연다. 시계 고정(time.time·datetime.now). 파일 트리 = 합성(개인 값 없음).
  시나리오 4(정적·API·파일 낡음/깨짐/빔·꼬리 경계) · 요청 113 · 그중 304 재요청 자동 생성.
- **시험**: `cargo test` — `http_parity`(Rust 를 127.0.0.1:0 에 띄워 날 HTTP 로) · `static_traversal_blocked`(Rust 만 · 10경로) · `round_like_python`(파이썬 round 27,784개 비트) · `loopback_only` · 단위 3.

### 둘째 조각 (2026-11-21)

- **정답지**: `rtserve/tools/gen_http2_fixture.py`(첫 생성기의 가짜 전송·시계 고정·미들웨어를 가져다 씀 · order_request_log 미들웨어 포함 · POST 본문은 aiohttp StreamReader 로).
  시나리오 3(api2 · api2_variants = 없음·깨짐·낡음 · writes = 쓰기 45걸음) · 요청 111 · 쓰기 시나리오는 요청마다 메모·오답·구독·주문 기록 파일 바이트 + 남은 `.tmp` + print 줄까지.
  원본 상수(FLOW_SPECS·MARKET_SYMBOLS·테이프 칸 폭·고래 경계)는 정답지가 원본 모듈에서 읽어 실어 Rust 상수와 대조한다(`consts_match_original`). statistics.stdev 7벌 비트(`stdev_exact`).
- **허브 구독**(`src/hub.rs`): `RTSERVE_HUB`(예 `ws://127.0.0.1:8788/rt`)에 붙어 흐름 코인 심볼을 구독(끊기면 1·2·4…30초). 받는 규칙은 proto.rs 그대로(스냅숏 통째 · Supply1s::merge · 풋프린트 칸 덮어쓰기+drop+창).
  응답은 원본 supply_1s_payload·api_footprint 의 창(660초)·커서(since*)·키 순서. 시험(`tests/hub_live.rs`)은 127.0.0.1 의 진짜 rtcore 허브에 rtcore 정답지 합성 체결을 넣고,
  rtserve 가 가운데서 붙어(앞 절반 스냅숏 · 뒤 밀기) 낸 값이 정답지 `final`(파이썬 행)과 비트까지 같은지 본다: 수급 619행·4,952필드 · 풋프린트 144봉·374칸(dhl 142 = 코어 판과 같음).
- 고친 것(첫 조각 결함): `DASHBOARD_ASSETS` 기본값·허용 목록이 옛 값(eth,sol,xrp · 5코인)이었다 → 원본 지금 값(eth,btc,sol,xrp,sndk,soxl,skhynix,mu · MARKET_SYMBOLS 9코인).

### 셋째 조각 (2026-11-21)

- **옮김 2**: `/api/session-alerts`(달력 = 6시간 캐시 · 경보 = 매번) · `/api/liquidation-5m-signal`(tail_risk DuckDB · 0·0.4·0.8·1.6초 다시 읽기 · 30분 봉 · pandas 합 = numpy 짝 합).
  DuckDB = `duckdb =1.10501.0`(rtduck 와 같은 1.5.1 · bundled · parquet) -- 처음 빌드 15분(-j 2). 정답지 `tools/gen_http3_fixture.py` 는 파이썬 duckdb(1.5.5)로 합성 tail_risk 파일을 만들고
  원본 계산 함수를 그대로 부른다 · 요청 9 일치 · 표가 없는 파일은 Rust 만 시험(`liq5m_db_read_error`: 2.8초 다시 읽기 · `db_read_error:` 접두).
- **swr**(`src/swr.rs`): 원본과 같은 키별 ttl·max_stale 로 붙였다 -- regime-band 60/7200 · sr-lean 1/30 · risk-radar 10/120 · ob5m 4/600 · treasury 60/600 · treasury24 300/1800 ·
  xyz100(since 없음) 5/60 · macro-today 30/600 · macro-calendar 6h/600 · news 15/300 · macro-spikes 30/120 · chip-score 60/600 · gex 60/600 · breakout 60/600 · pred-record 30/600 ·
  liquidation-5m-signal 1/0 · bubble-tape(since 없음) 1/0. 캐시 없는 것 그대로: iceberg · notify-center · acct · liq-tab(mtime 메모 = 같은 바이트) · state(서명 메모 = 같은 바이트).
  단위 시험: 신선·묵음(옛 값 + 뒤에서 한 번)·갱신 실패 때 옛 값 유지·로그·ttl+max_stale 넘으면 기다려 새로·만드는 중 예외는 부른 쪽으로.
- **못 한 것 5**: footprint-daily·cells · footprint-4h·cells · flow/heatmap (§7).
- **공유 함수로 바꿈**(코디네이터 지시 · rtcollect PR 73): numpy 합 = `rtcollect::ob5m::np_sum`(8,192칸 묶음 규칙 · rtserve 의 두 자리는 9칸·60칸 이하라 묶음에 안 닿는다) · html.unescape = `rtcollect::html::unescape`(rtserve 의 표 사본 삭제 · news 정답지 그대로 통과).

#### 한 요청 처리 시간 (rtserve 만 · 2026-11-21 개발기 Ryzen 5 7500F · release · `cargo run --release --example bench -- 300`)

잰 구간 = `rtserve::handle` 한 번(라우터 → 핸들러 → 미들웨어 → 압축) · 소켓·hyper 제외 · 입력 = 시험 정답지의 합성 트리 · 요청마다 300건 ·
«계산» = 캐시 없이 매번(새 Config) · «캐시 맞음» = swr 신선 값(직렬화·압축은 매번 한다). 파이썬 쪽은 재지 않았다(개발기에서 서버를 띄우지 않는다).
bubble-tape 의 «캐시 맞음»은 ttl 1초라 300건 반복 중 만료가 섞였다.

| 주소 | 시나리오 | 계산 중앙 | 계산 상위1% | 캐시 맞음 중앙 | 캐시 맞음 상위1% | 건수 | 응답 바이트(압축 전→후) |
|---|---|---|---|---|---|---|---|
| `/dashboard/live/` | static | 209µs | 276µs | 212µs | 279µs | 300 | 506→267 |
| `/dashboard/live/app.js` | static | 88µs | 156µs | 87µs | 156µs | 300 | 23670→2988 |
| `/api/acct-notes` | api | 21µs | 47µs | 21µs | 33µs | 300 | 179→125 |
| `/api/risk-radar` | api | 40µs | 71µs | 15µs | 32µs | 300 | 205→153 |
| `/api/sr-lean` | api | 70µs | 134µs | 18µs | 39µs | 300 | 786→272 |
| `/api/regime-band` | api | 30µs | 66µs | 12µs | 27µs | 300 | 436→189 |
| `/api/liq-tab` | api | 9µs | 23µs | 9µs | 25µs | 300 | 40→48 |
| `/api/iceberg` | api | 25µs | 56µs | 25µs | 71µs | 300 | 272→175 |
| `/api/ob5m` | api | 396µs | 596µs | 46µs | 94µs | 300 | 1407→341 |
| `/api/treasury-futures` | api | 333µs | 402µs | 34µs | 51µs | 300 | 749→231 |
| `/api/xyz100` | api | 278µs | 397µs | 45µs | 64µs | 300 | 1062→306 |
| `/api/macro-today` | api | 280µs | 473µs | 31µs | 69µs | 300 | 1023→304 |
| `/api/macro-calendar` | api | 17µs | 39µs | 8µs | 16µs | 300 | 185→146 |
| `/api/sr-lean` | api_big | 20.22ms | 27.50ms | 5.18ms | 7.25ms | 300 | 248286→18056 |
| `/api/regime-band` | api_big | 761µs | 836µs | 14µs | 28µs | 300 | 151→120 |
| `/api/state` | api2 | 22µs | 36µs | 22µs | 42µs | 300 | 116→112 |
| `/api/breakout-detector` | api2 | 18µs | 33µs | 9µs | 20µs | 300 | 208→162 |
| `/api/gex` | api2 | 62µs | 108µs | 21µs | 32µs | 300 | 972→343 |
| `/api/chip-score` | api2 | 17µs | 34µs | 6µs | 12µs | 300 | 84→66 |
| `/api/notify-center` | api2 | 94µs | 181µs | 94µs | 175µs | 300 | 2013→592 |
| `/api/news` | api2 | 298µs | 361µs | 65µs | 93µs | 300 | 2722→737 |
| `/api/macro-spikes` | api2 | 15.09ms | 17.43ms | 21µs | 33µs | 300 | 207→140 |
| `/api/bubble-tape` | api2 | 2.09ms | 2.39ms | 1.11ms | 1.22ms | 300 | 27152→3345 |
| `/api/pred-record` | api2 | 2.41ms | 3.92ms | 58µs | 96µs | 300 | 2716→568 |
| `/api/session-alerts` | calc | 29µs | 67µs | 17µs | 40µs | 300 | 236→151 |
| `/api/liquidation-5m-signal` | calc | 8.44ms | 12.56ms | 24µs | 42µs | 300 | 239→160 |



### 다섯째 조각 (2026-11-22)

- **일봉·4시간 풋프린트 4 라우팅**(`api3.rs` · 원본 3714~3751 손잡이 · 3861~3912 엔드포인트를 Read 로 확인):
  `?asset=` 은 원본처럼 **소문자로 안 바꾼다**(없으면 eth · fpd::ASSETS 밖 = 400 asset) · since/from/to = `_iso_day`(나쁘면 400 bad_date · 빈 글자도) · 4h = `_iso_h4`(400 bad_key) ·
  from·to 없음 = 400 from_to(나쁜 키 검사가 먼저 -- 원본 순서) · row = parse_row(행 달러 → 칸) · 일봉 cells 의 ValueError(`Err("range")`) = 400 range · 그 밖 예외 = 500 ·
  swr 키·시간 원본 그대로(fp_daily_days_ · fp_daily_cells_{a}_{from}_{to}_{row} · fp_4h_days_ · fp_4h_cells_ · 30/600) · 일봉 cells 앞에서 저장 값이 800 넘으면 «fp_daily_cells_·fp_4h_cells_·position_profile:» 중 오래된 절반 지움(`Swr::prune_oldest_half`).
  원천: lake = `DATA_LAKE` 또는 data/lake · 소급본 data/footprint_daily · hot 테이프 data/hot/binance_tape.sqlite(live_trade_tape_collector:61) · 맥락 binance_ctx.sqlite. Fpd 는 코인마다 처음 요청 때 하나(`Config.fpd`).
  시험 `tests/fpd_http.rs`: gen_fpd_fixture.py 정답지의 첫 걸음(eth·xrp 각 호출 11 -- 오류 range 포함)을 HTTP 로 + since 1 + 400 갈래 10 = **요청 42 · 불일치 0**.
  🔴정답지(tests/fixtures/gen/fpd · 59MB)는 커밋 안 하는 생성물이다 -- 없으면 fpd_parity 처럼 이 시험도 «gen_fpd_fixture.py 먼저»로 멈춘다.

### 넷째 조각 (2026-11-22)

- **주문·계좌 14 프록시**(`src/order.rs`): 시험 `tests/order_proxy.rs` = rtorder/tests/door_bin.rs 방식 그대로 진짜 rtorder 실행 파일(그림자 · 가짜 거래소 `rtorder::fakesrv` · 127.0.0.1:0)을 띄우고
  rtserve 를 거쳐 route 정답지(rtorder/tests/fixtures/route_fixture.json) 중 바깥을 안 부르는 사례 **42건 · 불일치 0**(상태·Content-Type·Cache-Control·Allow·본문 -- 벽시계 칸은 door_bin 과 같게 ms0) ·
  rtorder 가 받은 UA·IP(X-Forwarded-For 넣기 · CF 우선) · 거래소 쓰기 0 · POST 로 status = 405 · `/internal/acct` seq · rtorder 를 죽이면 주문 주소 502 · 다른 주소(acct-notes)는 200 · acct_seq 못 읽음.
- **/api/events acct_seq**: 발행기가 1초마다 rtorder `GET /internal/acct`(연결 0.5초 · 읽기 1초)의 seq 를 읽는다 · 못 읽으면 마지막 값.
- **/api/oi-5m**(`hub.rs`): 원본 api_oi_5m(4363~4411)·oi_5m_buckets(983~1019)를 Read 로 확인 · bars 1~288(기본 48 · 숫자 아니면 48) · swr `oi_5m:{sym}:{bars}` 5/0 · OKX = 코어 Micro 의 okx_oi5m([봉, Δ, 끝] = 원본 okx_d)
  first_bar «뒤» 봉만 합산(OKX 가 있으면 OKX 를 모르는 봉은 뺀다 -- 원본 그대로). 정답지 `tools/gen_http4_fixture.py`(원본 두 함수를 옮겨 적고 합성 oi_1s · 수집기 공백 → gap 1).
- **/api/liquidation-5m-history**(`hub.rs`): 원본 핸들러(4413~4473)·liq_5m_payload(537)·merge_bybit_liq(833)·okx_liq_hot_bars(856)·merge_okx_liq(878)·merge_hl_liq(907)·hl_whale_liq_events(504)를 Read 로 확인.
  바이낸스 봉·OKX(first_bar 뒤)·Bybit(수집 시작 봉 뒤) = 코어 Liq5m(스냅숏 통째 · 밀기 = 바뀐 봉 덮어쓰기 · okx·bybit 실리면 통째) · OKX first_bar «앞» = hot `okx_ctx.sqlite` okx_liquidations 봉 합(swr `okx_liq_hot:{inst}` 60) ·
  HL = hot `hl_positions*.sqlite`(코인별 원본 표 HL_LIQ_BY_ASSET · swr `hl_whale_liq_events:{asset}` 30 · 깊이 288봉 고정). 더하는 순서(OKX → Bybit → HL)·반올림(봉 합은 반올림 안 함 · 거래소 몫 칸은 round)·키 순서 원본 그대로.
  시험 `tests/http4_parity.rs`: 원본 함수로 뜬 바이트 == rtserve(Liq5m 은 두 번에 나눠 밀어 합치기도 봄) -- 요청 12(bars 12·144·288·999·abc·0·-3 · SOL hot 만 · XRP OKX 없음·Bybit 만 · 400 · gzip). oi-5m 7 과 합쳐 **19 · 불일치 0**.
- **모델 5**(`src/model.rs` → `rtmodel::handlers`): 원본 dir4h_cached·api_dir4h(2916~2945) · dir_short_cached·api_dir_short(2947~2975) · qx4_payload·api_vol_forecast(3049~3126) · api_open_rev(3128~3140) · api_dir_signals(2796~2802) ·
  fetch_binance_json(1372~1392 · graft 로 찾고 동작은 aiohttp 로 문구 확인) · d4._get(live_eth_dir4h_20261015.py:279).
  코인 표(dir4h = ETH·SOL·XRP · dir-short = ETH+BTC·SOL·XRP · vol = eth·xrp·btc·sol · 밖이면 `unsupported_asset`) · swr 원본 그대로(dir4h·dir_short·vol4h 300/1800 · open_rev 120/900 · dir_signals 3600/86400) ·
  예외 → `{"ok": false, "reason": "<종류>: <글>"}`(dir4h·dir-short·open-rev 는 원본처럼 200자에서 자름) · 모델 JSON 은 mtime 이 바뀌면 다시 읽고, 읽은 모델이 «낡음»(rtmodel::model_stale: 원본 joblib(data/live)이 JSON 보다 새것이거나 없음)이면
  값을 내지 않고 `ModelStale: …` 오류 · 밴 파일(data/live/binance_ban_until)이 있으면 보내지 않음 · 418·429 는 같은 파일에 적음(rtmodel::ban).
  설정: `RTSERVE_MODELS`(기본 data/live/rtmodel = rt_dirpred 전환 때 RT_DIR_MODELS) · `RTSERVE_REST`(기본 원본 FAPI) · `FRED_API_KEY`(원본 이름 · 값은 찍지 않음). REST = ureq(rtmodel 실행 파일과 같은 판 · 새 크레이트 없음).
  시험 `tests/model_parity.rs`(127.0.0.1 가짜 REST 가 rtmodel 정답지 녹화본을 돌려줌 · 바깥 접속 0):
  ① dir4h SOL·XRP + dir-short BTC·SOL·XRP 첫 회차 = **payload 비트·키 순서 같음 5벌 · 요청 545개 순서·쿼리 같음**(dirrest_fixture) ② vol-forecast HGB(eth·xrp · 1000봉 × 10쪽 endTime)·open-rev(실수 시계)·dir-signals(FRED 키) **본문 글자 4 같음**(signals_fixture) ·
  ③ 지원 밖 3 · 모델 없음 2 = REST 0 ④ QX4(qx4_fixture 첫 벌 4코인 봉을 쪽으로) = 요청 72 같은 열 · 본문 = 같은 녹화본으로 돌린 rtmodel 값(QX4 의 파이썬 비트 대조는 rtmodel/tests/qx4.rs · MLP 1ulp 차이 있음) ·
  ⑤ 밴 = 안 보냄 2(BinanceBanned 문구 · HTTPServiceUnavailable binance_ip_ban) ⑥ 낡음 2(joblib 새것 · 원본 없음).
- **못 한 것**: ETH dir-short(호가 깊이 sqlite 75열판 · rt_dirpred 와 같은 `dirshort::book_sqlite` 를 부르지만 rtserve 시험 정답지 없음) · ETH dir4h(정답지 모델 없음) · QX4I(Deribit 옵션 IV 칸 -- rtmodel 에 받기가 없다) ·
  open-rev 의 다른 4벌·dir-signals 키 없음 벌(rtmodel 시험이 라이브러리로 봄) · 벤치(넷째 조각 주소는 안 쟀다).

## 5. 의도한 차이 · 원본에서 본 이상한 점

| # | 무엇 | 원본 | rtserve |
|---|---|---|---|
| 1 | 🔴압축을 켠 정적 파일(ASSET_FILES)의 **304 뒤에 빈 zlib 흐름(deflate 8 · gzip 20바이트)** 이 본문 길이 표시 없이 나간다 | aiohttp StreamWriter 가 304 에도 압축기 flush 를 쓴다 → keep-alive 연결이면 다음 응답 앞에 쓰레기 바이트 | 안 보냄(hyper). 시험이 이 3건만 «의도한 차이»로 센다. 브라우저가 app.js·styles.css 를 재검증할 때마다 생기므로 화면 쪽 간헐 오류의 후보 — 파이썬에서 고칠지는 사용자 결정 |
| 2 | swr 캐시 | 1~60초(또는 6시간) 묵은 값 · 갱신 실패 때 max_stale 까지 옛 값 | 셋째 조각부터 **같은 ttl·max_stale 로 같은 동작**(`src/swr.rs`). 남은 시점 차이: ① 캐시가 프로세스마다라 파이썬과 rtserve 를 나란히 돌리면 둘이 다른 순간에 만든 값을 낸다(묵은 값이 ttl 만큼 서로 어긋날 수 있다) ② 원본은 이벤트 루프 하나라 «만드는 중» 요청이 같은 잠금에 줄을 서지만 rtserve 는 스레드 풀이라 줄 서는 순서가 다를 수 있다(결과는 같은 값) ③ 묵은 값을 낸 뒤 갱신이 끝나는 순간은 원본도 rtserve 도 그 계산 시간만큼 늦다 -- 다음 요청부터 새 값 |
| 3 | 아이스버그 events.jsonl 에 **깨진 줄·빈 줄** 하나 | ValueError 를 안 잡아 **500** | 같게 500(정답지 대조) — 고칠지는 사용자 결정 |
| 4 | ob5m·macro-calendar·treasury 의 예외 | 안 잡아 500(SQLite 표 없음·JSON 깨짐·0 나누기) | 같게 500 |
| 5 | 없는 정적 파일 | `404: Not Found` 가 아니라 **본문 없는** 404(청크 · octet-stream) — StaticResource 가 FileResponse 를 만든 뒤 stat 실패 | 같게 |
| 6 | 브라우저는 **deflate** 를 받는다 | Accept-Encoding 부분 글자 비교 + deflate 먼저 | 같게 |
| 7 | JSON 오류 문구 | `reason` 에 파이썬 JSONDecodeError 문구 | serde 문구(다름) — 화면은 reason 을 그대로 보여 주기만 한다. 정답지에는 이 갈래를 넣지 않았다 |
| 8 | HEAD | aiohttp 는 HEAD 에 JSON 미들웨어(ETag·압축)를 안 건다 | 같은 갈래로 처리하나 정답지 대조 안 함 |
| 9 | 안 옮긴 FileResponse 기능 | Range(206) · If-Match/If-Unmodified-Since(412) · 미리 압축한 `.gz`/`.br` 형제 | 없음(화면이 안 쓴다 · `ponytail:` 주석) |
| 10 | regime_band 의 시각 | `pd.Timestamp(글자, tz="UTC")` 는 꼴을 넓게 받는다 | 크론이 쓰는 꼴(`YYYY-MM-DD[ T]HH:MM[:SS[.f]]`)만 · 그 밖은 깨진 줄 |
| 11 | 🔴pred-record 코인 열쇠 순서 | `set(coins) \| {eth,btc,sol,xrp}` 를 **집합 순서(문자열 해시 무작위)**로 돌며 띠만 있는 코인을 더한다 → 기록 없이 띠만 있는 코인이 둘 이상이면 열쇠 순서가 프로세스마다 다르다(본문 ETag 도 재시작마다 바뀜) | 기록 있는 코인(이름 순) 뒤 eth·btc·sol·xrp 순으로 고정 · 정답지는 그런 코인을 하나만 둔다 |
| 12 | pred-record `xsec.latest` | 서버 메모리 `_xs_latest`(pred_record_loop 가 5분마다 채움) | 늘 null -- 다음: rt_dirpred 가 스냅숏을 파일로 쓰게 |
| 13 | supply-1s `oi` | 🔴정정(2026-11-22 · coin_flow.py collect_oi_1s 641~665 Read): 원본도 REST 가 아니라 **수집기 hot oi_1s 꼬리**를 0.5초마다 읽는 링이다 | 전엔 늘 [] (결함 -- 병행 대조 `.oi[길이]` 173 대 0) → **고침**: 부를 때마다 hot 꼬리(지금−690초 뒤)에서 초 → 마지막 값 · 지금−660 앞 버림 · 가장 늦은 초−660·sinceOi 뒤 (`hub.rs oi_rows` · 시험 http4_parity::supply_oi_from_hot) |
| 14 | events `acct_seq` · tickers | acct_seq = 사용자 스트림이 계좌를 새로 읽을 때마다 +1 · tickers = REST ticker/price · ts = 받은 시각 | acct_seq = 넷째 조각부터 rtorder `/internal/acct` 의 seq(1초마다 · 못 읽으면 마지막 값) · 가격 = 허브 마지막 체결가 · ts = 발행 시각 |
| 15 | 🔴/api/stream 의 `event: situation` | 상황 계산이 바뀔 때마다 | 안 낸다(상황 계산은 파이썬 몫) -- 화면은 스트림이 살면 /api/situation 폴링을 끄므로, 상황을 옮기기 전에는 화면을 rtserve 스트림으로 돌리면 안 된다 |
| 16 | footprint `agg`·창 | aggTrades 로 메운 봉 = true · 창 288봉까지 | 코어가 agg 를 싣지 않아 늘 false · 코어 스냅숏이 144봉이라 bars>144 도 144봉까지 · dhl 은 코어 판(테이프 가지가 파이썬과 다른 봉 -- rtcore footprint.rs 머리 주석) |
| 17 | supply-1s meta 나이 | 응답 시각 기준 tradeAge·oiAge | 코어가 보낸 시각 기준(받은 그대로) |
| 18 | events 큐가 찼을 때 | 옛 프레임을 버리고 새 것 | 새 프레임을 버린다(다음 초에 다시 온다 · `ponytail:`) |
| 19 | 첫 조각 #1(304 뒤 빈 zlib) | 그대로 | 그대로 안 보냄(파이썬 수정 여부는 사용자 결정 대기) |
| 20 | liquidation-5m-signal 의 `db_read_error:` 뒤 글자 | 파이썬 duckdb 예외 글자 | duckdb-rs 예외 글자(다름) -- 화면은 warmed_up=false 만 본다 |
| 21 | session-alerts 에 시간대 없는 시각 하나 | TypeError(naive − aware)를 안 잡아 **500** -- 달력 하나가 경보 전체를 막는다 | 같게 500(정답지 대조) -- 고칠지는 사용자 결정 |
| 22 | /api/footprint 의 OKX 합산 | **확인함(실제 파일 Read · 3971~4036행): 원본은 바이낸스 선물만**(`venues: ["binance-perp"]` · 11-02 사용자 결정) | rtcore 의 OKX 합산 커밋은 되돌려졌다(PR 91 · main 9aefd9e5 · proto `okxFrom`·`Micro.okx_bars` 없음) · rtserve 도 venues 를 늘 ["binance-perp"] 로 |
| 23 | 🔴주문 주소인데 rtorder 문이 닫혀 있음 | (원본은 한 프로세스라 해당 없음) | **502** `{"ok": false, "error": "rtorder_unavailable"}`(원본 주문 핸들러의 오류 꼴 · Cache-Control no-cache) · 다른 주소는 영향 없음 · 이 요청은 rtserve 가 접수 기록에 status 502 · error rtorder_unavailable 로 적는다. 둘째 조각 정답지(주문 주소가 없던 때의 404)는 tests/http2_parity.rs 가 이 값으로 바꿔 본다 |
| 24 | 🔴supply-1s·footprint 의 `symbol` | 원본 `FlowSpec.symbol` = **"ETHUSDT"**(대문자) | 둘째 조각부터 소문자("ethusdt")로 내고 있었다(허브 시험이 symbol 을 안 봤다) → **고침**(대문자) |
| 25 | oi-5m `openInterest`(지금 OI) | 수집 루프의 1초 링 마지막 값 | hot oi_1s 의 마지막 행(수집기 쓰기 주기만큼 늦을 수 있다 · 링이 비면 원본도 마지막 봉 끝 값) |
| 26 | liquidation-5m-history: 코어가 Liq5m 을 안 보낸 코인(흐름 엔진 밖 HYPE 등 · 허브 끊김) | 서버 시작 때 hot liquidations 로 되살린 메모리(+ @forceOrder 전 종목) · 그 전 몇 초만 tail_risk DuckDB(compute_liquidation_5m_history) | 빈 봉(원본에서 «기록 시작 = 지금»과 같은 모양 `{"warmed_up": true, "bars": [], …}`) -- 원본의 DuckDB 갈래는 시작 직후 몇 초뿐이라 안 옮겼다(`ponytail:` 는 아님 · 필요하면 rtcore 가 그 코인도 Liq5m 을 밀게) |
| 27 | 모델 핸들러의 적중 기록(pred_record_safe) | 화면 요청으로 계산할 때도 pred_record/{코인}.jsonl 에 덧붙임 | 안 적는다 -- rt_dirpred 가 5분마다 적는다(두 프로세스가 같은 파일에 같은 결정 시각을 쓰지 않게) |
| 28 | 모델 핸들러의 REST | 화면 요청과 5분 루프가 같은 swr 캐시·같은 REST 캐시 | rtserve 와 rt_dirpred 가 따로 받는다(5분에 REST 한 벌 더 · 첫 요청은 코인마다 31일치) -- 줄이려면 rtserve 가 rt_dirpred 의 마지막 payload 파일(dir4h_<코인>.json)을 읽게 |
| 29 | 모델 핸들러 예외 문구 | 파이썬 예외 글(requests HTTPError 의 URL 포함 문장 · JSONDecodeError 등) | 같은 종류 이름 + Rust 글(`HTTPError: 503` 등 -- 다름) · 같은 글: 지원 밖·artifact_missing·밴 두 문구(BinanceBanned … request not sent · HTTPServiceUnavailable: binance_ip_ban)·HTTPBadGateway: vol_upstream_error / open_rev_upstream_error |
| 30 | 밴 중 상태 코드 | 코디네이터 지시는 «503» 이었으나 원본을 읽어 보니 fetch_binance_json 의 503(HTTPServiceUnavailable)은 핸들러의 except 가 잡아 **200 `{"ok": false, "reason": "HTTPServiceUnavailable: binance_ip_ban"}`** 로 나간다(dir4h·dir-short 은 requests 가드의 BinanceBanned) | 원본 그대로 200 + 그 문구 |
| 31 | 모델이 낡음 | 원본은 joblib 을 바로 읽어 새 모델 값을 낸다 | rtserve 는 내보낸 JSON 만 읽는다 -- joblib 이 더 새것이거나 없으면 값을 안 내고 `{"ok": false, "reason": "ModelStale: 모델 낡음: …"}` (재학습 cron 뒤 export_*.py 를 다시 돌릴 때까지) |
| 32 | vol-forecast 의 QX4I 칸(ETH·BTC: iv_atm·sig4h_iv_*·log_rv_iv) | Deribit 옵션 체결로 붙임(실패면 칸만 빠짐) | 여섯째 조각부터 붙임(`rtmodel::handlers::vol_forecast_qx4i` · Deribit 받기 = 모델 주소와 같은 바깥 받기 · RTSERVE_REST 없으면 안 보냄 → 칸만 빠짐) |
| 34 | 🔴REST 기본 | (해당 없음) | 다섯째 조각부터 `RTSERVE_REST` 가 없으면 **꺼짐** -- 모델 주소 5 = `{"ok": false, "reason": "RestOff: RTSERVE_REST unset -- request not sent"}`. 실전 전환 때 원본 FAPI 주소를 준다 |
| 35 | 병행 중 쓰기 | (해당 없음) | `RTSERVE_READONLY=1`: 메모·오답·푸시 구독·해지 POST = 503 `{"ok": false, "error": "readonly_shadow"}` · 접수 기록 안 적음(§8) |
| 36 | 풋프린트 lake 자리 | data_store.LAKE = DATA_LAKE 또는 data/lake(**data/lake 가 없으면 ~/backups/… 백업 lake**) | DATA_LAKE 또는 data/lake 만(백업 갈래 안 옮김 · 서버엔 data/lake 가 있다) |
| 33 | vol-forecast QX4 이어 받기 중 일부 실패 | `refresh_rows` 가 False → HGB | 같음 · 다만 QX4 쪽 «낡음»은 HGB 로 떨어지지 않고 오류(옛 값·다른 모델 값을 안 냄) |

## 6. 양성 대조

`python3 rtserve/tools/mutate_check.py <작업 폴더>` — 규칙 하나씩 틀리게 바꾼 사본으로 `cargo test` 를 돌려 실패해야 «잡힘». 원본 `rtserve/` 바이트는 끝에 대조.
2026-11-21 결과: **46/46 잡힘**(공통 동작 22 · 엔드포인트 22 · 파이썬 의미 2) · 무변경 사본 통과 · 원본 바이트 그대로.
첫 판에서 못 잡은 5개로 정답지를 보강했다: 정적 `.html` 파일 · 같은 초·같은 쪽에 grp 만 다른 사건 · stale 경계(100초 전) · 꼬리 32KB 가 줄 머리에서 정확히 잘리는 파일.
절대 경로 검사를 빼는 변이는 동등 변이(폴더 밖 검사가 같은 404)라 목록에서 뺐다.
둘째 조각 변이 38개를 더했다(api2·text·py 30 · hub 8). **둘째 조각 결과(2026-11-21): 전체 84/84 잡힘**(첫 조각 46 포함) · 무변경 통과 · 원본 바이트 그대로. 첫 판 못 잡음 6(급변 문턱 · 버블 반올림 자리 · 적중 구멍 문턱 · 순위 동점 · 구독 옛 시각 · IP 머리 순서) → 정답지에 그 갈래를 넣어 보강.
**셋째 조각 결과(2026-11-22): 전체 95/95 잡힘**(셋째 조각 변이 12 = 세션 경보 3 · 청산 게이지 4 · swr 3 · 풋프린트 venues 1 · client-timing 머지 대기 칸 1 · 지운 «숫자 참조» 변이 1 은 rtcollect::html 로 옮겨 감). 고친 것: 청산 게이지 봉 길이 변이를 못 잡아 정답지 시계를 11:28 로(30분·15분 봉이 갈림) ·
무변경 사본이 «링크 실패»(duckdb 묶음 디버그 정보 + 메모리 상한)·«허브 시험 기다림 부족»(부하에서 3초 안에 밀기가 다 안 닿음)으로 떨어져 사본은 debug=0 · ulimit 12GB · 허브 시험은 «칸이 스냅숏과 같아질 때까지 30초»로.

**넷째 조각 결과(2026-11-22): 넷째 조각 변이 21/21 잡힘**(`④` 이름 · 주문 프록시 6 · oi-5m 3 · 청산 원 7 · 모델 핸들러 5) · 무변경 통과 · 원본 바이트 그대로.
도구 바꿈(코디네이터 지시): 빌드 --release 만 · 산출물은 rtserve/target 하나(새 빌드 폴더 없음 · 사본은 새 수정 시각으로 복사해 `cargo clean` 없이 다시 빌드) · 변이마다 시험 하나 · 끝나면 사본·링크 지움.
첫 판 못 잡음 4: X-Forwarded-For(rtorder 도 같은 127.0.0.1 을 접속 주소로 써 같아 보였다 → 시험이 127.0.0.2 에서 한 번 더 보냄) · oi 공백 300→600(정답지 공백이 1시간뿐 → 봉 하나 빈 600초 간격을 넣음) ·
oi 기본 48→24(ETH 는 OKX 가 있어 20봉만 남아 같았다 → SOL 기본 요청을 더함) · «현재 OI = 마지막 봉 끝» 은 동등 변이(hot 마지막 행 = 마지막 봉 끝 행)라 뺐다. 앞 조각 변이 95개는 이번에 다시 돌리지 않았다.
🔴같은 target 을 쓰니 끝난 뒤 원본 시험 실행 파일이 지운 사본 경로(CARGO_MANIFEST_DIR)를 가리킨 채 남았다(http4_parity 가 정답지 NotFound) -- 도구가 끝에 원본 src/lib.rs·tests/*.rs 수정 시각을 올려 다시 빌드되게 했다.

**다섯째 조각 결과(2026-11-22): 변이 8/8 잡힘**(`⑤` · 읽기 전용 2 · REST 꺼짐 1 · 풋프린트 라우팅 5) · 무변경 통과 · 원본 바이트 그대로(두 번째 판). 첫 판 못 잡음 2:
읽기 전용 «접수 기록 안 적음»(시험 트리에 data/live 가 없어 적기가 조용히 실패 → 폴더를 만들어 봄) · 4h cells 의 «bad_key 먼저»(from 만 나쁜 사례가 없었다 → 더함).

**여섯째 조각 결과(2026-11-22): 변이 8/8 잡힘**(`⑥` · 백분위 f32 · 재깔림 합 차례 · 블록 문턱 · 접근 문턱 · 구멍 ts 0 · agg 첫 최대 · supply oi 660초) · 무변경 통과. 첫 판 못 잡음 2(agg 첫 최대 = 정답지에 |값| 같은 부호 다른 칸이 없었다 → 주기 3 칸 · supply oi 자르기 = 가장 늦은 초가 오래된 사례 → 더함).

**아홉째 조각 결과(2026-11-22): 변이 17/17 잡힘**(`⑨` · 가중치 문턱 2 · trend 6 · cycle-path 6 · swing 4 -- 무변경 통과 · 원본 바이트 그대로). 첫 판 못 잡음 1: swing «깨짐 > → >=»(종가가 스윙 고점과 꼭 같은 봉이 없었다 → SNDK 봉을 더함).

## 7. 다음 조각 순서

1. ~~일봉·4시간 풋프린트 4~~ -- 다섯째 조각에서 옮김.
2. ~~flow/heatmap~~ -- 여섯째 조각에서 옮김(아래는 다섯째 조각 때 적은 확인):
   🔴«체결 대조가 풋프린트 칸의 반올림 전 값을 쓴다» 문제 -- **실제로는 갈리지 않는다(코어 원값 칸 필요 없음)**. 원본 `_heatmap_rows`(4523~4627 Read): 체결 대조 값은 `fill_ask[i] += v[0]` · `fill_bid[i] += v[1]`
   (메모리 풋프린트 칸 = 봉·가격칸별 테이커 매수·매도 **기초자산 수량 합**) → `filled = Σ min(drop, fill)` → 응답에 나가는 것은 `offtouch_leave_share = round(1 − filled/drop, 3)` 하나뿐(fill 배열 자체는 안 나감).
   수량 단위가 바이낸스 수량 칸(ETH·BTC 0.001 · SOL 0.01 · XRP 0.1)의 배수라 칸 합의 참값은 소수 셋째 자리까지의 십진수다 -- 반올림 전 값과 round(x, 3)의 차는 부동소수 덧셈 잡음뿐.
   실측(개발기 data/live/footprint_eth.json · 12봉 · 칸 값 472개): **0.001 의 배수가 아닌 값 0/472 · |원값 − round(원값,3)| 최대 3.96e-11 · 합 3.61e-10**.
   → filled 차이 ≤ 3.6e-10(그 창 전체) · drop(창 안 잔량 감소, 보통 수백~수천 ETH)으로 나누면 비율 차 ~1e-12 이하 -- round(·,3) 경계(±5e-4)에서 그 거리 안에 들 확률은 응답당 ~1e-9 수준.
   (래스터와 같은 시각 풋프린트가 개발기에 없어 offtouch_leave_share 자체를 두 값으로 나란히 계산하진 못했다 -- 위 상한이 근거.) 수량 칸이 0.001 보다 작은 종목이 생기면 다시 본다(지금 흐름 엔진 종목엔 없음).
   남은 일: 래스터 read_window(.f32 시각 파일 · 절대 격자 · agg 접기)·row_stats·vol_scores·approach_ratio·heatmap_obi·np.percentile(97)·nanmedian 을 numpy float32 규칙(축 0 줄이기 = 차례 합 · cumsum · argmax 첫 값)으로 옮기고
   정답지는 원본 함수를 합성 .f32 파일에 돌려 뜬다. 원본 quirk(캐시된 payload 를 제자리에서 update)는 그대로. 래스터 read_window(.f32 시각 파일)·row_stats·vol_scores·approach_ratio(numpy f32) + 체결 대조(원본은 풋프린트 칸 «반올림 전» 값 -- 허브는 반올림한 칸만 민다:
   rtcore 가 칸 원값을 싣거나 대조를 «모름»으로 내는 결정 필요). 원본 quirk: 캐시된 payload 를 제자리에서 `update` 한다(접근·변동 신호가 다음 1초 캐시 값에도 남는다).
3. 메모리 상태 남은 것: situation · sr-watch · market-context -- rtcore 상황 계산 먼저.
4. 요청 때 REST 남은 15(§2-5): market-history · footprint-week · footprint-daily/xcvd·heat · footprint-4h/heat · opt-smile · chart-markers(안 옮김) · liquidation-map·tiers · position-profile · cycle-path · trend · swing-lines · macro-trend · push/test.

남은 주소 집계(2026-11-22 · 다섯째 조각 뒤): 등록부 84 = 옮김 66(화면·정적 6 + `/api/*` 60 · 여섯째 조각 뒤) · 남음 18(`/api/*`: 저장소+계산 0 · 메모리 상태 3(situation·sr-watch·market-context -- situation·market-context 는 다른 에이전트 ctx.rs) · 요청 때 REST 15(위 4번 -- xcvd·heat 셋·push/test 포함)).
4. 계좌·주문 14(rtorder 계좌 알림 뒤) · 5. 요청 때 REST·모델 · 6. 나란히 가동.

### 여섯째 조각 (2026-11-22)

- **`/api/flow/heatmap`**(api3.rs 조립 · 원본 api_flow_heatmap 4692~4732 Read): symbol 화이트리스트 8(server.py:396 · 소문자로) · cols 10~900(기본 300) · agg 1~60(기본 3) · 숫자 아니면 400 bad_cols_or_agg ·
  swr `heatmap_{sym}_{cols}_{agg}_{rows}` 1초 · rows 모드면 접근행동(`approach_` 120/600) → rows 에 · 변동 신호 자(`volsig_ref_` 300/600) · 지금 값(`volsig_` 1초) → summary 에(실패는 삼킴 -- 원본 try/except).
  원본 quirk(캐시된 payload 를 제자리에서 update): rtserve 는 캐시 사본에 매번 덧붙인다 -- 접근행동이 한 번 성공한 뒤 실패하는 1초 안에서만 다를 수 있다.
- **`src/raster_read.rs`**(read_window 327~407 · 형식 = rtcollect/src/raster.rs): 상황 카드(ctx*.rs)도 이 함수를 쓴다(코디네이터에 함수 모양 알림).
- **`src/heatmap.rs`**: row_stats(185~224) · approach_ratio(227~276) · vol_scores(279~324) · heatmap_obi(1136~1147) · _heatmap_read(4482~4521) · _heatmap_rows(4523~4627) · _volsig_ref(4629) · _volsig_read(4649) · _approach_read(4672).
  numpy 2.2.5 규칙을 서버 판 venv 로 실측해 맞췄다: np.percentile(f32) = q·지표·가중치 전부 f32(5,000/5,000) · 축 0 줄이기 = 행 차례 f32 · `win[:, sel]` 은 열 우선(F)이라 `up.sum()` = 열 우선 차례 8192칸 묶음 짝 합(60/60 -- 행 우선이면 37/60) ·
  np.arange(0, 2.5001, 0.05, f32) = i·0.05f32 · nanmedian = 가운데 둘의 f32 평균 · NEP 50(f32 ⊕ 파이썬 실수 = f32: `dm < 0.35` 는 0.35f32 와 비교).
  체결 대조 = 허브 풋프린트 칸(반올림 값 -- §7-2 실측으로 갈림 없음) · 흐름 엔진 없는 심볼 = fill_source null.
- **정답지** `tools/gen_heatmap_fixture.py`(서버 판 venv · 원본 수집기 모듈의 함수를 그대로 부름 · server.py 쪽은 옮겨 적음 · 커밋 안 하는 생성물 tests/fixtures/gen/raster):
  합성 래스터 6시간(원본 _RasterWriter · 공백 = 무효 행 · mid NaN · 벽) + 손 갈래(시 중간 구멍 ts 0 · 잘린 마지막 행 · 마법 글자 틀림 · 칸 수 다름 · |값| 같고 부호 다른 주기 3 칸).
  시험 `tests/heatmap_parity.rs`: read_window 11 · heatmap 본문 8(rows·이미지 · 풋프린트 있음·없음 · agg 60 · 빈 창) · 접근행동 3 · 변동 신호 3(자 배열 비트 + 지금 값) · HTTP 6(400 셋 포함) = **다름 0**.
- **못 한 것(이번 조각)**: 2번(sr-watch · position-profile) · 3번(요청 때 REST 15 -- 표·가중치 계량기) · 4번(상황 카드 붙이기 -- CTX.md 에 «붙이는 쪽이 부를 함수와 사용법»이 아직 없다).

## 8. 서버 병행 가동(2026-11-22 · 실행은 코디네이터)

- **래퍼** `scripts/ops/supervisor_rtserve.sh`(_supervise.sh 방식 · 잠금 data/live/.supervisor_rtserve.lock · 로그 logs/supervisor/rtserve_YYYYMMDD.log):
  실행 파일 tmp/rtbin/rtserve · `RTSERVE_ROOT`=저장소 · `RTSERVE_ADDR`=127.0.0.1:8790(8787 이면 거절) · `RTSERVE_HUB`=ws://127.0.0.1:8788/rt ·
  점-env 를 프로세스 환경에만(대시보드 감독 스크립트와 같은 줄 · /api/push/config 의 VAPID 공개키가 같아지게).
- **병행 고정**: `RTSERVE_READONLY=1`(메모·오답·푸시 구독·해지 POST = **503 `{"ok": false, "error": "readonly_shadow"}`** · 접수 기록 order_requests.jsonl 도 안 적음 -- 파이썬과 같은 파일을 둘이 쓰지 않게) ·
  `RTSERVE_REST` 지움(모델 주소 5 = `{"ok": false, "reason": "RestOff: RTSERVE_REST unset -- request not sent"}` · 밴 파일에도 안 적음) · 주문 14 는 rtorder 문 그대로(안 떠 있으면 502).
  🔴바뀐 기본: `Config::new` 의 REST 뿌리가 빈 값(= 꺼짐)이 됐다 -- 실전 전환 때는 `RTSERVE_REST=<원본 FAPI 주소>` 를 줘야 모델 주소가 계산한다.
  시험 `tests/model_parity.rs::shadow_rest_off_readonly`(503 넷 · RestOff · 주문 502 · client-timing 안 막음 · 파일 안 생김).
- **대조 도구** `rtserve/tools/shadow_http.py`(표준 라이브러리 · 루프백만 · GET 만 · 쪽마다 기본 2요청/초 · `--selftest`):
  옮긴 주소 35종 × 코인(eth·sol·xrp 중 코인별 주소) · 모델 5는 `--models` 때만 · 주문·계좌 14 는 안 보냄(파이썬 쪽이 계좌 REST 를 부른다) · SSE 2 는 안 봄.
  주소마다 «같음 n/N · 다른 칸(횟수) · 두 값 예 · 처리 시간(파이썬·rtserve 중앙·상위 1%)» · JSON 은 칸별(키 순서 · 목록 길이 · 값과 형 -- 1.0 대 1 도 다름) · 아니면 바이트.

빼는 칸(이름이 같으면 어느 깊이든):

| 칸 | 왜 |
|---|---|
| generated_at · computed_at · fetched_at · updated_at · server_time | 응답·계산을 만든 순간(두 서버가 다른 순간에 만든다) |
| now · now_sec · now_ms | 요청 순간의 시계 |
| age · age_s · age_sec · age_ms · tradeAge · oiAge | 지금 − 파일/자료 시각(같은 파일이라도 요청 순간이 달라 끝자리가 다르다) |
| elapsed_ms | 처리 시간 |

안 빼는 것(일부러): `stale`(나이에서 나오지만 경계에서만 갈린다 -- 갈리면 보고 싶다) · `updated`(파일에 적힌 시각 -- 같아야 한다) · 봉 시각 `ts`.
매초 움직이는 원천(supply-1s · footprint · bubble-tape · xyz100 · oi-5m 의 진행 봉)은 «거의 같은 순간»이라도 다를 수 있다 -- 다른 칸 이름이 진행 봉·마지막 초에만 몰리면 정상, 닫힌 봉에서 갈리면 결함.

### 서버 병행 대조 1회차 원인 가르기(2026-11-22 · 코디네이터 실행 · 같음 135/189 · 주소 63)

| 주소 | 다른 칸 | 원인 | 고칠 곳 |
|---|---|---|---|
| supply-1s ×3 | `.oi[길이]` 173~175 대 0 | **rtserve 결함**: 원본 oi_1s 링도 hot oi_1s 꼬리(collect_oi_1s 641~665)인데 rtserve 는 «코어 범위 밖»이라며 [] | **고침**(hub.rs oi_rows · 시험 supply_oi_from_hot) |
| supply-1s ×3 | `.okxMeta.errors` 0 대 1 | **코어 값**: rtcore OkxMeta.errors(접속 실패 횟수)가 1 · 원본 okx_state["errors"] 는 0 -- 서버 rtcore 가 옛 판(PR 94 전)이라 그 판의 OKX 첫 접속 실패를 센 값으로 본다 | rtcore(새 판으로 다시 대조) |
| supply-1s eth | `.seconds[][]` ×193 · `.partial.bn[]` | 시점: 마지막 초가 1초 어긋남(두 요청 사이 초가 넘어감) -- 행 전체가 한 칸 밀린 모양 | 없음 |
| supply-1s sol | `.bybitLiq[길이]` 0 대 1(1/3) | 시점: 창 끝(지금−3초) 경계의 한 건 -- 같은 규칙(4249~4254 Read) | 없음 |
| footprint ×6 | `.bars[].agg` true 대 false | 코어 범위(§5 #16): 원본은 재시작 뒤 aggTrades 로 메운 봉을 agg=true 로 -- rtcore 는 agg 를 싣지 않는다 | rtcore(싣는다면 proto) |
| footprint ×6 | `.levels[][]` 값 · `.dhl[]` · dhl 이 한쪽에만 | **코어 값**: agg=true 봉(원본 = REST aggTrades 백필 · rtcore = 자기 복원)과 테이프 가지 dhl 이 다름 -- 서버 rtcore 옛 판 · 같은 봉 원천이 달라서 | rtcore(새 판으로 다시 · 그래도 다르면 원본 백필 봉 vs 코어 복원 봉) |
| liquidation-5m-history eth | 진행 봉 `long_usd`·`events` 4 대 6·`long_min_px` null 대 1676.16 | **코어 값**: 같은 봉 바이낸스 강제청산 이벤트가 rtcore 에 2건 더(원본 liq_5m = @forceOrder WS 1835~1860 + 시작 때 hot liquidations 되읽기 1803~1833) | rtcore(Liq5m 이벤트 원천·중복 · 옛 판 여부) |
| liquidation-5m-history ?bars=144 eth·sol | `okx_detail.long_usd`·`n` | 기동 시점: 두 프로세스의 OKX first_bar 가 달라 같은 봉을 한쪽은 메모리(라이브 체결)·한쪽은 hot(bk_px 금액)으로 센다(merge_okx_liq 878~904) | 없음(같은 원천 규칙 · 시작 시각이 다를 뿐) |
| liquidation-5m-history ?bars=144 xrp | `long_usd` 끝자리 | 합 차례: 원본 Bybit 몫 = bybit_liquidations ts_ms 순(SQL) · rtcore = 받은 차례 | rtcore(맞추려면 ts_ms 순으로 더하기 · 화면 영향 없음) |
| oi-5m ?bars=12 ×3 | 진행 봉 `.bars[][]`(1/3) | 시점: 진행 봉 OKX 끝 OI(Micro 밀기 주기)·swr 5초 | 없음 |
| pred-record | `.xsec.latest` 값 대 null · `asof_ms` | 아직 안 넘어간 원천(§5 #12 · rt_dirpred 전환 전 xsec_latest.json 없음) · asof_ms = 벽시계 | rt_dirpred 전환 · asof_ms 는 빼는 칸에 더함 |
| macro-calendar | `updated_utc` 16:08 대 16:38 · `stale_min` | 캐시 시점: 둘 다 swr 6시간 -- 파이썬은 16:20 에 읽어 둔 값, rtserve 는 늦게 떠서 16:38 판 파일을 읽음(값을 만든 순간의 나이) | 없음 |
| gex | `stale_min` · `block_trades.flow_by_coin.BTC[]` | 캐시 시점: swr 60초 · 블록 거래 상태 파일은 20초마다 바뀐다 -- 다른 순간의 파일 | 없음 |
| risk-radar(1/3) · macro-today `recv_sec`(2/3) | 분 경계 · 받은 시각 | 캐시 시점(swr 10·30초 사이 파일 갱신) | 없음(recv_sec 은 원천 칸이라 안 뺌) |
| liquidation-5m-signal eth | `bar_elapsed_sec` | 벽시계(지금 − 봉 시작) | 빼는 칸에 더함 |

빼는 칸에 더한 것(벽시계 몫이 확실한 것만): `asof_ms`(pred_record.summary 에 요청 순간 ms) · `bar_elapsed_sec`. 대조 주소에 footprint-daily · footprint-4h · flow/heatmap(rows) 둘을 더했다.
처리 시간: 거의 전 주소에서 rtserve 가 1/2~1/5 · 예외 `/dashboard/live/`(rtserve 10.4ms 대 3.6ms -- 버스터 sha256 을 요청마다 다시 계산하는 것으로 보임 · 다음에 mtime 메모).

## 9. 일곱째 조각 (2026-11-22)

### 9-1. 🔴래스터 읽기 할당 한도(raster_read.rs)
원본 read_window(327~407 Read): 머리 n_bins 로 `np.zeros((total, n_bins))` · 행 bin_lo 로 `np.zeros((total, g_hi − g_lo))` -- 깨진 파일이면 numpy 가 **MemoryError** 를 던져 그 요청만 실패한다
(_heatmap_read 는 잡지 않아 → 500 · 원본 api_flow_heatmap 의 접근행동·변동 신호는 try/except 로 삼킴 · 상황 카드 micro 틱은 collect_micro_ref 의 예외 갈래).
Rust 는 할당 실패 = 프로세스 abort 라 **`MAX_CELLS` = 6,400만 칸**(정상 최대 = 히트맵 900열 × 60초 접기 = 54,000행 × N_BINS 240 × 가격 이동 여유 5배 ≈ 6,480만)을 넘으면 할당 전에 멈춘다:
`try_read_window` → `Err("MemoryError: …")` (히트맵 본체 = 500 · 접근행동 = 빈 맵 → 원본처럼 뒤까지 건너뜀) · `read_window`(함수 모양 그대로) = 빈 창(n_bins 0 -- 상황 카드처럼 «못 읽음 = 없음»).
시험 `heatmap_parity::broken_raster_does_not_abort`: bin_lo ±20억 행 · 머리 n_bins 65535 × 54,000행 -- 죽지 않고 Err/500/빈 창.

| 곳 | 크기를 정하는 것 | 한도 · 근거 |
|---|---|---|
| raster_read `raw`·`out` | 파일 머리 n_bins · 행 bin_lo · cols×agg | **MAX_CELLS**(위) |
| heatmap `dm`·`bkt`(접근행동) · `up`(변동 신호) · `fill_*` | 창 칸 수(위 한도 안) | 창 × 50 거리대 · 600행 × 고른 칸 |
| heatmap 요청 cols·agg | 쿼리 | 원본처럼 10~900 · 1~60 으로 자름 |
| fpd `rebin` 의 `vec![0; n]` | 칸 키의 (최대 − 최소)/row -- 소급본 parquet·hot 테이프 가격 | **`REBIN_MAX` 100만 칸**(원본 footprint_daily.rebin 193~201 `np.zeros(n)` = MemoryError → cells 는 ValueError 가 아니라 **500** · 시험 fpd_parity handles_match_server) |
| fpd 읽기 행 | SQL 시각 범위(날·4시간 창) | 시각 범위가 상한 |
| hub `fp_bars`·`liq`·`okx_oi5m` | 코어 메시지(같은 기계의 rtcore) | 288봉·600봉 자르기 |
| hub `oi_rows` | hot oi_1s 지금−690초 뒤 | 초마다 한 행 → ≤ 700 |
| order `parse`(rtorder 응답) | rtorder 본문 길이 | 루프백 rtorder · 15초 · ponytail: 길이 상한 없음 |

### 9-2. 상황 카드 붙이기(hub.rs · CTX.md «붙이는 쪽 사용법»)
`/api/situation` · SSE `event: situation`(computed_at 이 바뀔 때 · `situation_run` 이 hub 시계로 매초) · ETH 만(원본 2489~2490).
재료: 코어 Situation(스냅숏·밀기) · Micro(vol60·liq60·streams·flow → `ctx_x::apply_flow`) · 코어 @bookTicker(Book 사건 · REST 0 -- 결정) · 래스터 12초(`raster_read::read_window`) · evidence.obi·persist = 히트맵 rows 요약(swr 5초).

빠진 칸(«없음» 모양 -- 원천이 rtserve 에 아직 없음):

| 칸 | 원본 원천 | 지금 | 다음 |
|---|---|---|---|
| now.evidence.veto | REST klines 720봉 trend_veto_rows | 0 | RTSERVE_REST 조각(5분봉 하루와 함께) |
| vol_mult · vol_mult_by | candles_day(REST klines 300) → vol_size_mult | 빈 목록 → null | 같음 |
| x.move60_p75 · range30_pct · card30(dir_p·reach_p) | candles_day | 빈 봉(계산 안 됨) | 같음 + rtmodel::card30 |
| x.z60h·z30h·*_all(기준선) | hot 테이프 14일 flow_hour_scales(1시간마다) | **붙임**(아래) | — |
| x.venues 의 Bybit | bybit_bars(hot bybit 테이프 · 20초) | **붙임** | — |
| micro.sr(청산맵 레벨) | load_liquidation_map(REST 90일) | null | 청산맵 조각 |
| micro baseline_from_tape(vol60_pct) | hot 테이프 7일 | **붙임** | — |
| 장부(flow_read_fused.jsonl) | 봉마다 한 줄 | 안 씀(병행 중 READONLY · 경로 상수 미확인) | 전환 때 |
| 원본 collect_micro_ref 예외 → now {ok false, reason} | | 호가가 오기 전엔 micro 틱을 건너뜀(available false) | — |
| inp.bars(close·oi_delta) | 코어 상황 입력 봉 | 빈 목록(코어 Situation 에 실리지 않음) | rtcore |

시험 `http4_parity::situation_before_core`(코어 전 = «계산 전» · computed_at 0.0).
**통합 시험(일곱째 조각 둘째 판)** `tests/sit_parity.rs`: 정답지 `tools/gen_sit_fixture.py`(서버 판 venv · gen_ctx_x_fixture.py 의 정의부를 읽어 실행 -- 파일은 안 고침 ·
원본 situation.classify · flow_read.read · vol_size_mult 를 그대로 · situation_payload 2734~2738 · SSE 3175~3179 옮겨 적음) -- 40걸음(완결 봉 부족 16 · ok 24 · OKX 시작 · 수신 상태 바뀜):
코어 Situation·Micro(flow = rtcore flow_nets_tick·flow_delta 로 만든 실제 밀기 · streams) 를 Live::apply 로 넣고 situation_once → HTTP 본문·SSE 줄 **40/40 글자 같음** · 양성 대조(코어 값 한 칸 틀림 → 다름).
원본 쪽 «없음»으로 맞춘 재료: candles_day·candles·bybit·flow_scales·micro payload(코어 호가 없음)·imb40 링·inp.bars(코어 Situation 에 없음)·veto 0·obi/persist null.
이 시험에서 고친 결함: `vol_mult_by` 를 빈 dict 로 냈다 -- 원본은 흐름 코인마다 열쇠(situation_by 2738) → 코인마다 `vol_size_mult([])` 모양으로.

**파일만 읽는 재료 붙임**(`hub.rs sit_materials` · 원본 collect_micro_ref 1926~1954 · compute_situation 2470~2472 Read):
기준선 주기(`MicroRef::baseline_due/baseline_set` = 1시간 · 없으면 60초 · 실패면 있던 값 · 60초 뒤 다시) 때 `baseline_from_tape`(hot binance_tape 7일 → vol60_pct) ·
flow_scales(`tape_flow_rows`+`flow_hour_scales_from_bars` · 14일 · since = size_bands_since(479~490: 표 sol·xrp 1790487335 · 아니면 meta size_bands_changed) · 빈 결과면 있던 값) ·
flow_scales_all(`flow_hour_scales_combos` · binance·okx(okx_tape · ETH-USDT-SWAP)·bybit(bybit_tape · ETHUSDT) · 조합마다 빈 결과면 있던 값) · Bybit 봉(`bybit_bars` 26시간 · 닫힌 지 10초 넘은 봉 · 20초마다).
1초 틱 안(spawn_blocking)에서 돈다 -- 무거운 SQL 은 주기마다만(원본과 같은 주기). 🔴이 셋의 «주기·유지» 갈래는 시험이 없다(함수 값은 ctx_x_parity·ctx_micro_parity 가 원본과 대조).

### 9-3. `/dashboard/live/` 버스터 메모
`asset_buster` = 파일마다 (수정 시각 ns · 크기)가 그대로면 지난 값(ponytail 주석). http_parity 113 그대로 통과.

### 서버 병행 대조 2회차 원인 가르기(2026-11-22 02:5x KST · 새 코어 · 같음 163/213 · 주소 71) -- 여덟째 조각

**대조 도구 시점 맞춤**(`tools/shadow_http.py`): 두 요청을 `threading.Barrier` 로 같은 순간에 떠나게 · 원천 시각 칸(`STAMP_KEYS` = updated·updated_utc·updated_at·generated_at·asof·computed_at·last_update_ms·file_ts --
맨 위와 한 단계 안 dict · 뉴스는 항목 `seen_ms` 최대)이 두 쪽 다르면 본문을 견주지 않고 «시점 다름 n»으로 따로 센다(같음 분모에서 뺀다) · `--selftest` 에 stamp 3건.

| 주소 | 다른 칸 | 원인(코드로 확인) | 고칠 곳 |
|---|---|---|---|
| macro-calendar | updated_utc 16:08 대 17:38 | 원천 시각: 둘 다 swr `macro_calendar` 6시간/600(rtserve api3 `macro_calendar_cached` = 원본 load_macro_calendar 키·시간) -- 파이썬은 16:20 에 읽어 둔 판을 6시간 낸다 · rtserve 는 17:52 기동이라 17:38 판 | 없음(도구가 «시점 다름»으로) |
| news | items 한 칸 밀림 | 원천 시각: rtserve 쪽에 새 항목 1건(seen_ms 1795283542 대 …173) -- 둘 다 swr 15/300 · 같은 정렬·중복 제거(첫 조각 정답지 통과) | 없음(seen_ms 로 «시점 다름») |
| ob5m eth(1/3) | events_known · tilt_pct · walls 길이 | 시점: 진행 봉(live 줄 · 워커 5초마다 갈아 끼움)을 두 쪽이 다른 회차에 읽음 · swr 4/600 같음 | 없음 |
| supply-1s sol | `.oi[][]` 창 시작 7초 · 길이 161 대 160 | 시점(구조적): 원본 링은 **행이 도착할 때만** «지금−660» 앞을 지운다(collect_oi_1s 656~659 · 수집기 flush 10초) -- 지움이 최대 flush 만큼 늦어 앞 몇 초를 더 낸다. rtserve 는 부를 때의 벽시계로 지운다 · 같은 순간이어도 flush 지연만큼 창 앞끝이 다를 수 있다 | 맞추려면 «마지막으로 새 행을 본 시각 − 660» 을 들고 있어야 한다(ponytail -- 화면은 누적선이라 앞 몇 초 차이는 안 보임) |
| pred-record | recent[][] 한 칸 밀림 | 원천 교대: rt_dirpred 가 막 넘어간 회차 | 없음 |
| footprint ×6 | `.bars[].levels[][]` 값 · `.agg` · `.dhl` 키·값 | **코어(rtcore) 값** -- 아래 | rtcore |

**footprint 차이 -- rtcore 가 원본 어느 줄과 다른가**(dashboard/coin_flow.py · Read 로 본 줄):
1. **재시작 뒤 빈 구간 채우기**: 원본 `footprint_backfill`(442~) 이 REST **aggTrades** 로 봉을 메우고(요청당 가중치 20 · 최신 봉부터 · 예산 1200), 그 체결을 `footprint_add(…, agg=True)` 로 넣는다 →
   ① 그 봉은 `agg_bars` 에 들어가 응답 `agg: true`(footprint_add 381~382) ② aggTrade 한 줄 = 주문 하나로 **총량과 크기 구간(고래·리테일)을 같이** 센다(373~380 `if order or agg`).
   rtcore 는 REST 없이 자기 복원(테이프·스냅숏)으로 채운다 → 같은 봉의 칸 값(특히 고래·리테일 열)·agg 표시가 다르다. 코어를 막 다시 띄운 직후 봉에서 몰려 나는 것이 이것.
2. **실시간 체결의 크기 구간**: 원본은 `@trade` 를 총량으로 바로 넣고(369~372) 주문이 닫힐 때 구간을 따로 넣는다(같은 규칙: 가격·방향·ms 로 되묶기 · 343~353) -- rtcore 가 같은 되묶기를 하는지(복원 봉 밖에서도 값이 다르면 여기).
3. **dhl**(봉 안 누적 델타 저·고·마감): 원본 api_footprint 의 fp_dhl(서버 · 테이프 hot 에서) vs 코어 판(rtcore footprint.rs 머리 주석 «테이프 가지가 파이썬과 다른 봉») -- sol 은 dhl 키가 파이썬에만(코어가 그 봉 dhl 을 안 셈).
→ 전환 전에 정하려면: rtcore 가 재시작 공백을 aggTrades 로 메울지(REST 가중치 · 밴 가드) 아니면 «복원 봉은 다르다»를 받아들일지(사용자 결정).
- (여덟째 조각 뒤) 결정: rtcore 는 재시작 공백을 REST aggTrades 로 메우지 않고 복원 봉에 `agg: true` 를 싣는다(코어 몫) → `shadow_http.py` 는 /api/footprint 에서 **한쪽이라도 agg 인 봉 시각을 두 쪽에서 빼고** 나머지만 글자 대조 · 뺀 수는 «복원 봉 n»(불일치 아님) · 자체 시험 2(한쪽만 agg 도 뺌 · agg 아닌 봉 차이는 그대로 잡힘).

## 10. 아홉째 조각 (2026-11-22) -- 요청 때 REST

### 10-1. 원본이 부르는 REST 표(주소마다 · server.py 줄은 Read 로 확인 · 가중치는 바이낸스 문서 규칙: klines limit [1,100)=1 · [100,500)=2 · [500,1000]=5 · >1000=10)

공통: 바이낸스 호출은 거의 전부 `fetch_binance_json`(1531~1551 · 밴 가드는 호스트가 binance.com 일 때만 · 밴 중이면 안 보냄: error_reason 있으면 503 `binance_ip_ban`, 없으면 None ·
200 아니면 418/429 기록 뒤 502(reason) 또는 None · 연결 오류·시간 초과는 안 잡음 → 핸들러 따라 500). Deribit·OKX·Bybit 도 이 함수를 지나지만 가드·기록은 안 탄다.
requests 경로(macro_trend)는 server.py:171 의 binance_ban_guard 가 Session.request 를 감싼다. swr = 1482~1517 · STALE_GRACE_SECONDS 600(1040).
🔴원본에는 «가중치 머리를 보고 쉬기»·«요청 사이 간격»이 없다(같은 IP 의 주문과 한도를 나눠 쓴다).

| 주소 | 바깥 요청(한 번 만들 때) | 가중치 | 캐시(swr 키 · ttl/max_stale) | 실패 | 계산 · 메모리 상태 | rtserve |
|---|---|---|---|---|---|---|
| `/api/trend` (2743~2760) | fapi klines 1d limit 120 ×1 | 2(원본 주석 «1»은 틀림) | `trend_{a}_1d` 300/600 | 모든 예외 → 200 `{"ok": false, "reason": "<종류>: <글>"}` · 코인 밖 400 unsupported_asset | trend_rule.py 81줄(math) · 상태 없음 | **옮김** |
| `/api/cycle-path` (4820~4845) | fapi klines {1m·5m·15m} startTime limit 1500 ×1 | 10 | `cycle_path:{SYM}:{start}` 20/0 · 키 상한 없음 | HTTPException → 200 `{"ok": false, "reason": <reason>, "rows": []}` · 연결 오류 500 · 400 unsupported_symbol·bad_start | 한 줄 변환 · 상태 없음 | **옮김** |
| `/api/swing-lines` (2762~2794) | klines 1h·30m limit 1000 ×2 | 10 | `swing_{a}` 60/600 | 한쪽 실패 = 그 tf null + partial · 둘 다 = 200 ok false | swing_lines.py 57줄(numpy max·min 같음 비교뿐) | **옮김** |
| `/api/opt-smile` (4739~4755) | Deribit get_book_summary_by_currency(ETH·BTC·USDC) ×1 · 6초 | 바이낸스 아님 | `opt_smile_{CUR}` 60/600 · 🔴실패 payload 도 캐시 | 200 ok false deribit_unavailable·eth_btc_xrp_only | front_smile 35줄(np.polyfit) | 다음 |
| `/api/market-history` (3701~) | eth·btc: ETH·BTC 5m limit 1500 ×2(evidence 캐시) · 그 밖: 5m limit 1000 ×1(3초) | 20 / 5 | `evidence_signal` 60/0 · `market_history:{a}` 60/600 | 502 텍스트 evidence_signal_upstream_error·market_history_upstream_error · 밴 503 · 400 unsupported_market_history_asset | trend_veto(pandas rolling·ewm -- ctx_rest.rs 에 이미 있음) · vwap_rows · 형성 봉 | 다음 |
| `/api/liquidation-map` (4774 → 3461) | klines 1h limit 48 + 5m limit 460 | 3 | `liquidation_map:{a}` 60/600 + liq_hist_memo | 502 JSON `{"error": "liquidation_map_upstream_error", "detail": …}` · 밴 503 텍스트 | live_liquidation_map_20260824.py 725줄(numpy·pandas) + 로컬 OI 3 DB | **옮김**(src/liqmap.rs · 11-28) |
| `/api/liquidation-map/tiers` (4847 → 3359) | 1h limit days×24+2 · 1500 넘으면 endTime 으로 앞쪽(1d 1회 … 365d 6회) | 1~60 | `liquidation_tiers:{a}:{d}` 60(365d 300)/600 | 502 JSON · days 밖 = 조용히 1 | compute_tier_profile | **옮김**(src/liqmap.rs · 11-28) |
| `/api/position-profile` (4785 → 3418·3388) | 첫 적재: openInterestHist 5m limit 500 ×18 + klines 5m limit 1500 ×6 · 증분: 각 1(5분봉 닫힌 뒤 · 60초 간격) | ≥60 + oiHist(모름) | `position_profile:{a}:{wins}:{rng}` 1/600 · pos_books(코인 장부) · 실패 백오프 60초 | 502 JSON position_profile_upstream_error · 404 flow_off · 400 bad_range | position_profile.py 204줄(numpy exp·pairwise 합) · **흐름 메모리(oi_1s·footprint sec·bars)** | 다음 |
| `/api/sr-watch` (2060) | 직접 없음 -- sr_break_loop(2063~2105) 이 10초마다 load_liquidation_map 을 불러 채운 메모리 | (청산맵 몫) | 메모리 | — | sr_breaks·sr_watch_view + footprint sec | **막힘**: 청산맵 먼저 |
| `/api/footprint-daily/heat` · `/api/footprint-4h/heat` (3928·3914 → 3753) | klines 1h limit 500 ×1(둘이 공유) | 5 | `fp_daily_k1h_{a}` 600/3600 · 바깥 `fp_*_heat_{a}_{from}_{to}_{row}` 300/1800 | 받기 실패 = 빈 heat 를 300초 캐시 | footprint_daily heat_at → 청산맵 내부 함수 · 소급 parquet | 다음 |
| `/api/footprint-daily/xcvd` (3853) | 현물 klines·OKX 3·Bybit 2·fapi fundingRate·OKX funding = 8 | 현물 2(?)·fundingRate(모름) | `fp_xcvd_{a}` 3600/21600 + 디스크 누적 data/live/xcvd/{a}.json | 전부 실패 = RuntimeError(차가우면 500) | dict 합치기 75줄 · 🔴파일 쓰기(병행 중 READONLY 와 부딪힘) | 다음 |
| `/api/footprint-week` (4183 → 4161) | klines 5m limit 1000 · 4000행까지 쪽(보통 4회) | 20 | `fp_week_k5_{a}` 600/3600 · `fp_week_{a}` 300/1800 | 받기 실패 = candles [] 를 300초 캐시 | hot SQLite·duckdb 테이프 재구성이 본체 | 다음 |
| `/api/macro-trend` (2125 → macro_trend.payload) | FRED 20계열(3번 시도) + DefiLlama 1 + fapi 1d limit 1200 ×2 | 20 | `macro_trend` 3600/21600 · 실패 payload 도 캐시 | 200 ok false | macro_trend.py 434줄(pandas) · 키 필요 | 제안: 파이썬 워커가 파일로 |
| `/api/push/test` POST (4923) | 구독마다 웹푸시 POST(새 세션 · 가드 밖) | — | — | VAPID 키 없음 503 | RFC8291 암호화 | 알림 묶음과 함께 |

### 10-2. 바이낸스 한 문(`model::binance_get`)
- rtserve 의 바이낸스 선물 REST(모델 5 · trend · cycle-path · 앞으로 옮길 것)는 전부 `model::fetch_binance`/`rest_get` → `binance_get` 하나를 지난다:
  ① 밴 파일(cfg 시계 · 418/429 기록 = rtmodel::gate::Gate 안의 ban_note) ② 앞 응답 `x-mbx-used-weight-1m` > `RT_REST_WEIGHT_LIMIT`(기본 1,200) 이면 **그 분 끝 +0.5초까지 보내지 않고 거절**
  (Gate 는 쉬며 기다리지만 화면 요청을 1분 붙잡지 않게 거절 → 원본 «못 받음» 갈래: fetch 꼴 = `HTTPServiceUnavailable: binance_weight_hold` · 모델 rest_get 꼴 = `BinanceWeightHold: …`)
  ③ 요청 사이 최소 간격 `RT_REST_GAP_MS`(기본 200ms · 문 하나 잠금 안에서 · 시험은 0).
- 보내기·가중치 머리 읽기는 `rtmodel::gate::Gate`(rtmodel 안 고침). 🔴Gate 의 시계 상자가 `Send` 가 아니라 스레드마다 하나씩 두고, 공유 상태(마지막 보낸 시각·거절 끝)는 Config 안 잠금 하나.
  바뀐 점: 모델 주소의 받기 시간 제한 10초(fetch_binance_json) → 20초(Gate 하나) · 바이낸스 요청이 한 줄로 선다(원본 연결 32개 동시).
- 바깥 받기 중 **이 문 밖**: `ext_get`(dir-signals 의 Coinbase·바이낸스 현물·FRED · QX4I Deribit -- 원본도 밴 가드 밖 · 현물 가중치는 선물과 다른 한도).
- 시험 `tests/rest_parity.rs::weight_over_limit_holds_rest_of_minute`: 가짜 REST 가 1100 을 알리면 다음 요청도 보냄(양성) · 1300 을 알린 뒤 trend·cycle-path·trend·open-rev 4요청 = **가짜 서버가 받은 요청 0** · 응답은 못 받음 갈래 글자.
  `rest_off_sends_nothing`: RTSERVE_REST 없음 = trend `RestOff: …` · cycle-path `{"ok": false, "reason": "RestOff: …", "rows": []}`.

### 10-3. 옮긴 주소
- `/api/swing-lines`(swing_lines.swing_state 17~44 · 앞뒤 50봉 피벗 · 한쪽 실패 = null + partial `"<tf>: <종류>"` · 둘 다 = RuntimeError) · `/api/trend`(`src/rest.rs` · trend_rule.trend_payload 22~68 를 그대로: 합 = 파이썬 3.11 sum(왼쪽부터 -- 3.12 부터 보정 합이라 정답지는 3.11 venv) · ln·sqrt = libm) · `/api/cycle-path`(int() = 공백·부호·숫자 사이 밑줄 · 1490·7450분 경계 · 주문 심볼 = MANUAL_EXEC_SYMBOLS(env DASHBOARD_MANUAL_EXEC_SYMBOL)).
- 정답지 `tools/gen_rest_fixture.py`(서버 판 venv 3.11.15 · 원본 핸들러 본문 옮겨 적음 · trend_rule 은 원본 모듈 · 생성물 tests/fixtures/gen/rest/rest.json) -- 요청 33(swing 8: 정상·1h 실패·둘 다 실패·60봉(피벗 없음)·30m 실패·종가 = 스윙 고점(안 깨짐)·400·캐시 · trend 10: 정상·모자람·502·0 종가·크기 상한·같은 종가·400·빈 asset 캐시 · cycle 15: 1m/5m/15m 경계 ±1ms·소문자·502·400 다섯·밑줄 숫자)
  → 시험 `rest_parity::rest_endpoints_match_original` 본문 바이트·상태·보낸 요청 열 **33/33 같음**.
- 알려진 차이: 연결 실패 글자(원본 aiohttp ClientConnectorError 문장 · rtserve `ClientError: <ureq 글>`) · cycle-path 연결 실패는 둘 다 500.
- 원본 결함(그대로 둠 · 사용자 결정): cycle-path 15m × 1500봉 = 15.6일 -- start 가 그보다 오래면 선이 지금까지 안 온다.

### 10-3b. 청산맵(2026-11-28 · `src/liqmap.rs`)
- `/api/liquidation-map`(load_liquidation_map 3469~3516 전부: 레벨 + heatmap_bins + 5분 경계 밀도 이력 144 + liq_hist_memo + 청산 막대 tier_profile + levels_price·known_until) · `/api/liquidation-map/tiers`(1·7·30·60·90·180·365일 · 1,500 넘으면 endTime 쪽 나누기 · 60일 이상 감쇠 끔).
- 계산 = rtmodel::regime_band(`prepare_hl`(감쇠 끔 갈래 더함 · `prepare` 는 그대로 240h) · `bins` · `side_levels` · `own_alive`/Common 칸 pub) -- 같은 원본을 한 곳만. liq_oi_usd = hot binance_ctx·okx_ctx·bybit_ctx.
- 정답지 `tools/gen_liqmap_fixture.py`(원본 모듈 import + 적재 함수 옮겨 적음) → `tests/liqmap_parity.rs` 22요청 본문·상태·보낸 요청 열 같음 · 양성 대조(사본 변이) 6/6 잡음(memo 마지막 경계 변이는 한 번 호출 시나리오라 같은 값 -- 등가).
- 알려진 차이: RTSERVE_REST 없음 = 503 `rest_off`(원본에 없는 갈래) · 밴·가중치 = 503 글자(원본 미포착 HTTPServiceUnavailable 과 같음).
- 남음: heat 둘(FPD h4_heat_payload·소급 parquet) · `/api/sr-watch`(sr_break_loop = 텔레그램 보냄 + 코인별 1초 체결가 -- 병행 중 중복 알림 결정 필요).

### 10-4. 할당 한도(§9-1 에 더함)

| 곳 | 크기를 정하는 것 | 한도 · 근거 |
|---|---|---|
| rest klines 응답 | 바이낸스 응답 길이 | limit 120·1500 행(요청이 정함) · ponytail: 본문 길이 상한 없음(ureq 읽기 그대로) |
| swr `cycle_path:{SYM}:{start}` 키 | 쿼리 start(ms) | 원본처럼 상한 없음(30일 안 정수 · 화면은 순환 첫 체결 하나) -- ponytail |
