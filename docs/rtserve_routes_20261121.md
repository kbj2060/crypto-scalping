# 응답 서버(rtserve · 꾸러미 R6) — 라우트 전수 조사와 첫 조각 (2026-11-21)

원본 = 파이썬 대시보드 응답 서버(aiohttp · 127.0.0.1:8787 · `dashboard/server.py` 5,148줄 + `dashboard/orders.py` 핸들러). 행 번호는 2026-11-21 브랜치
`claude/rust-migration-w2-20261120`(6bda3c48) 판. graft 색인은 낡아 행이 수백 줄 어긋나 있다 — 행은 전부 파일을 직접 읽어 확인했다.
Rust 판 = `rtserve/`(새 크레이트). 설계 문서 `docs/rtcore_design_20261120.md` §11 R6.

## 0. 요약

- 등록부(server.py 4932~5025) = **84개**: 화면 문서·정적 6(`/` · `/dashboard/live` 셋 · sw.js/manifest · add_static) + **`/api/*` 78**(GET 61 · POST 17) + SSE 2(그 78 안).
- `/api/*` 78 을 «무엇을 읽어 답하나»로 나누면: 파일·SQLite 만 21 · 파일 쓰기/설정(POST·푸시) 6 · 저장소+계산(REST 없음) 7 · 프로세스 메모리 상태 10 · 요청 때 외부 REST/모델 20 · 주문·계좌 14.
- **이번 조각에서 옮김**: 공통 동작 전부(미들웨어 둘 · 정적 파일 · index 302 · 버스터 · pwa · 404/405/500) + 엔드포인트 12(파일·SQLite 만 읽는 것).
  정답지 = 원본 함수를 진짜 aiohttp 객체(라우터·미들웨어·Response.prepare)로 돌려 뜬 연결 바이트 · 요청 113건(압축 바이트 포함) 일치 ·
  의도한 차이 1종 3건(§5) · 양성 대조 §6.
- 인증·세션: **원본 서버 안에 없다.** 127.0.0.1 에만 묶이고, 공개 경로는 Cloudflare 터널(`dashboard/scripts/start_cloudflare_tunnel.sh`) + Cloudflare Access 가 막는다
  (`dashboard/scripts/supervise_server.sh:31~32` 주석 · `docs/rtcore_browser_route_20261120.md:22`). «인증 없이 열면 `/` 는 302, 그 밖은 404» 로 보인 것은
  원본 라우터 그대로다(`/` = `HTTPFound("/dashboard/live/")`, 모르는 주소 = `404: Not Found`). rtserve 도 인증을 새로 만들지 않고 루프백이 아니면 시작을 거절한다.

## 1. 공통 동작 (원본 위치 → rtserve)

| 동작 | 원본 | 내용 | rtserve |
|---|---|---|---|
| 미들웨어 순서 | 1381 | `[order_request_log, static_asset_headers, json_compress_etag]`(앞이 바깥) | 같은 순서(`lib.rs handle`) · order_request_log 는 다음 조각(POST `/api/manual-*` 만 건드림) |
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
| 요청 기록 `order_request_log` | 1352 | POST `/api/manual-*` 만: KST 시각·경로·쿼리·상태·error·ms·UA·IP(CF-Connecting-IP 우선) → `data/live/order_requests.jsonl` | 다음 조각(주문 접수와 같이) |
| M0 `client_timing` | 5102 | POST `/api/client-timing` 본문 1KB·아는 키·숫자만 → 로그 한 줄 · 0.2초 상한 | 다음 조각(로그 한 줄이라 쉬움 · 전환이 끝날 때까지 둔다 — 질문지 25) |
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
| `/api/macro-spikes` | 2130 → 712 | S treasury + hl_xyz · `statistics.stdev` · 뉴욕 시각(CME 개장) | swr 30/120 | 30초(비교 화면) | — | 다음(정확 유리수 stdev·뉴욕 서머타임 표 필요) |
| `/api/news` | 2096 → 729 | S `data/hot/news_rss.sqlite` · `html.unescape`·정규식 | swr 15/300 | 30초 | 안 옮김(뉴스 수집기) | 다음(HTML 엔티티 표 전체 필요) |
| `/api/chip-score` | 2088 → 303·321 | F `chip_score_YYYYMM.jsonl` 두 달 | swr 60/600 | 5분 | — (쓰는 쪽 = 서버 안 chip_score_loop · §3) | 다음(404 판정에 흐름 코인 목록 필요) |
| `/api/notify-center` | 2134 → 336 | F verdict_calendar·push_sent_log·liqmap_score_status + 상수 PUSH_KINDS(scripts/live_push_notifier:131) | — | 5분 | — | 다음(PUSH_KINDS 를 한 곳에 둘 방법) |
| `/api/state` | 3647 | F `dashboard_state.json` + 명시 ETag | json_cache | 안 부름(SSE 가 같은 원천) | rtduck market_data | 다음 |
| `/api/breakout-detector` | 4456 | F `{a}_breakout_detector_state.json` | swr 60/600 | **안 부름** | (워커 꺼짐 · 질문지 20) | 지움 후보 |
| `/api/gex` | 4715 → 1208 | F deribit_gex·cboe_option·block_trades state | swr 60/600 | 60초 | rtcollect `cboe_chain`(반쯤) | 다음 |
| `/api/bubble-tape` | 1107 → 1071 | S 체결 테이프 hot 셋(바이낸스·OKX·Bybit) | 자체 1초 | 1초(bubble_map.js) | rtcollect 테이프 셋 · rtcore 테이프 | 다음 |
| `/api/pred-record` | 2890 → pred_record.py:187 | F `pred_record/*.jsonl` 꼬리 8MB + **M** `_xs_latest` | swr 30/600 | 60초 | rtmodel `bin/rt_dirpred`(기록) | 다음(상대 강도 latest 가 메모리) |

### 2-2. 파일 쓰기·설정 (6) — rtserve

| 주소 | 핸들러 | 하는 일 | 비고 |
|---|---|---|---|
| `/api/acct-notes` POST | 2869 → acct_notes.save | 날짜·글 검증 → 원자적 쓰기 | 400 = ValueError 글자 |
| `/api/acct-wrong` POST | 2881 → wrong_add·wrong_del | 한 줄 더하기·지우기 | 〃 |
| `/api/push/config` · `/devices` GET | 4856·4891 | env VAPID 공개키 + 구독 파일 | 비밀값 = 환경변수(값은 옮기지 않는다) |
| `/api/push/subscribe` · `/unsubscribe` POST | 4868·4879 | 구독 파일 쓰기 | 〃 |

### 2-3. 저장소 + 계산(요청 때 REST 없음) (7) — rtserve(계산은 Rust 로 다시)

| 주소 | 핸들러 | 읽는 것 · 계산 | 화면 |
|---|---|---|---|
| `/api/footprint-daily` · `/cells` | 3842·3849 → footprint_daily.py | D/F lake + S hot 테이프·맥락 → DailyFootprint | fp_daily.js 60초·범위 바뀔 때 |
| `/api/footprint-4h` · `/cells` | 3879·3885 | 같은 원천 h4 | 〃 |
| `/api/liquidation-5m-signal` | 4751 | D tail_risk(rtduck market_data 가 씀) + 신호식 | 60초 |
| `/api/flow/heatmap` | 4673 → 4463·4504 | F 호가 래스터(rtcollect `raster`) + numpy 행 집계 + **M** 풋프린트(체결 대조) | 1초 |
| `/api/session-alerts` | 4840 | F 거시 달력 + compute_macro_event_alert | 30초 |

### 2-4. 프로세스 메모리 상태 (10) — rtcore 허브/상태와 잇기

| 주소 | 핸들러 | 원본 상태(채우는 루프) | rtcore 대응물 |
|---|---|---|---|
| `/api/events` SSE | 3651 | latest_event_*(publish_dashboard_events 1초 · 티커 REST) | Prices(티커) · 계좌 순번은 rtorder |
| `/api/stream` SSE | 3130 | footprint_state·oi_1s·okx_*·spot_*·liq_events_by·situation_state | Supply1s(밀기) — 화면이 허브로 가면 지움 |
| `/api/supply-1s` | 4341 | 같은 supply_1s_payload | Supply1s(실서버 그림자 일치 · 설계 문서 진행 기록 5·6) |
| `/api/footprint` | 3971 | flows[a].footprint_state + fp_dhl(S) | Footprint(봉 623·칸 1,637 일치) |
| `/api/situation` | 4721 | situation_state·fo_state·mp_state(collect_micro_ref 1초 · compute_situation) | 없음 — 계산 조각 |
| `/api/sr-watch` | 2041 | sr_watch(sr_break_loop 10초) | 없음 · 화면이 안 부름(SR 근접 알림 워커만) |
| `/api/oi-5m` | 4344 | S oi_1s + M oi_1s·okx_oi_5m | Oi 사건 · OI 5분 묶음은 아직 |
| `/api/liquidation-5m-history` | 4394 | M liq_5m + S OKX·Bybit·HL 청산 + D tail_risk | Liq5m 밀기 · Liq 사건 |
| `/api/market-context` | 2698 | M micro_state·mark_ring·mid_ring·depth25 + S/D 수집기 + R Deribit | Mark·Book · 나머지 계산 조각 |
| `/api/client-timing` POST | 5102 | 로그 한 줄 | — (rtserve 가 그대로) |

### 2-5. 요청 때 외부 REST·모델 (20) — 순서 맨 뒤 · 대부분 rtmodel 서빙을 붙이거나 워커 파일로 바꾼 뒤

`/api/market-history`(3682 · 5분봉 REST + trend_veto) · `/api/footprint-week`(4164 · S + REST 5분봉) · `/api/footprint-daily/xcvd`(3834 · 현물·OKX·Bybit REST)
· `/api/footprint-daily/heat` · `/api/footprint-4h/heat`(1시간봉 REST) · `/api/opt-smile`(4730 · Deribit) · `/api/chart-markers`(4738 · 안 옮김 — 사용자 결정)
· `/api/liquidation-map`(4755) · `/api/liquidation-map/tiers`(4828) · `/api/position-profile`(4766) · `/api/cycle-path`(4801) · `/api/trend`(2724 · 1d 120봉)
· `/api/swing-lines`(2743) · `/api/dir-signals`(2777 · Coinbase·바이낸스·FRED) · `/api/vol-forecast`(3072 · rtmodel `qx4`) · `/api/dir4h`(2915 · rtmodel `dir4h`)
· `/api/dir-short`(2945 · rtmodel `lgb`·`dirshort`) · `/api/open-rev`(3109) · `/api/macro-trend`(2106 · FRED·DefiLlama) · `/api/push/test`(4904 · 웹푸시 발송).
🔴개발기에서는 이 묶음의 REST 를 부르면 안 된다(서버와 IP 한도 공유) — 대조는 응답을 녹화한 가짜 서버로만.

### 2-6. 주문·계좌 (14) — rtorder intake 붙이기

`/api/binance-account`(orders.py:778 · swr 10/600 · 사용자 스트림이 순번을 올리면 바로) · `/api/manual-entry/preview`(1633) · `/submit`(1669 · busy_or)
· `/status`(2591 · 3초/진행 중 1.5초) · `/wait-action`(2610) · `/cancel`(2654) · `/limit-amend`(2548) · `/api/manual-bracket/disarm`(2050) · `/rearm`(2082) · `/tp`(2260)
· `/tp-legs`(2215) · `/add`(2345) · `/modify`(2442) · `/api/manual-exit/submit`(2002).
Rust: `rtorder/src/intake.rs` `Desk::api_*`(접수 핸들러가 HTTP 와 떨어진 함수로 이미 있다 · 정답지 = 원본 핸들러 대조 139건) — rtserve 가 쿼리를 넘기고 응답을 내면 된다.
주문 경로 공통: `DASHBOARD_MANUAL_EXEC_ENABLED`(꺼지면 403) · `confirm=1` · busy 409 · order_request_log.

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
- **설정**: `RTSERVE_ROOT`(필수 · 저장소 뿌리) · `RTSERVE_ADDR`(기본 127.0.0.1:8790 · 루프백 아니면 종료코드 2) · `DASHBOARD_ASSETS`(원본과 같은 뜻).
- **옮긴 엔드포인트 12**: acct-notes·acct-wrong(GET) · risk-radar · sr-lean · regime-band · liq-tab · iceberg · ob5m · treasury-futures · xyz100 · macro-today · macro-calendar.
  원본 구조대로 «워커가 쓴 파일을 읽어 돌려줌»만 한다(rtmodel 은 쓰는 쪽이라 의존하지 않는다).
- **정답지**: `rtserve/tools/gen_http_fixture.py`. 원본 함수 본문을 옮겨 적고(대시보드 서버 모듈은 import 하지 않는다 — import 만으로 수집기·키를 끌어온다)
  별도 모듈(sr_lean·risk_radar·regime_band·acct_notes·worker_state·data_store)은 그대로 불러, 진짜 `web.Application`(라우터·미들웨어) + `make_mocked_request` +
  바이트를 모으는 가짜 전송으로 연결 바이트를 뜬다. 서버를 띄우지 않고 소켓도 안 연다. 시계 고정(time.time·datetime.now). 파일 트리 = 합성(개인 값 없음).
  시나리오 4(정적·API·파일 낡음/깨짐/빔·꼬리 경계) · 요청 113 · 그중 304 재요청 자동 생성.
- **시험**: `cargo test` — `http_parity`(Rust 를 127.0.0.1:0 에 띄워 날 HTTP 로) · `static_traversal_blocked`(Rust 만 · 10경로) · `round_like_python`(파이썬 round 27,784개 비트) · `loopback_only` · 단위 3.

## 5. 의도한 차이 · 원본에서 본 이상한 점

| # | 무엇 | 원본 | rtserve |
|---|---|---|---|
| 1 | 🔴압축을 켠 정적 파일(ASSET_FILES)의 **304 뒤에 빈 zlib 흐름(deflate 8 · gzip 20바이트)** 이 본문 길이 표시 없이 나간다 | aiohttp StreamWriter 가 304 에도 압축기 flush 를 쓴다 → keep-alive 연결이면 다음 응답 앞에 쓰레기 바이트 | 안 보냄(hyper). 시험이 이 3건만 «의도한 차이»로 센다. 브라우저가 app.js·styles.css 를 재검증할 때마다 생기므로 화면 쪽 간헐 오류의 후보 — 파이썬에서 고칠지는 사용자 결정 |
| 2 | swr 캐시 | 1~60초(또는 6시간) 묵은 값 · 갱신 실패 때 max_stale 까지 옛 값 | 캐시 없음 — 매 요청 새로 읽는다(더 신선). 갱신 실패 때 «옛 값으로 버티기»도 없다 |
| 3 | 아이스버그 events.jsonl 에 **깨진 줄·빈 줄** 하나 | ValueError 를 안 잡아 **500** | 같게 500(정답지 대조) — 고칠지는 사용자 결정 |
| 4 | ob5m·macro-calendar·treasury 의 예외 | 안 잡아 500(SQLite 표 없음·JSON 깨짐·0 나누기) | 같게 500 |
| 5 | 없는 정적 파일 | `404: Not Found` 가 아니라 **본문 없는** 404(청크 · octet-stream) — StaticResource 가 FileResponse 를 만든 뒤 stat 실패 | 같게 |
| 6 | 브라우저는 **deflate** 를 받는다 | Accept-Encoding 부분 글자 비교 + deflate 먼저 | 같게 |
| 7 | JSON 오류 문구 | `reason` 에 파이썬 JSONDecodeError 문구 | serde 문구(다름) — 화면은 reason 을 그대로 보여 주기만 한다. 정답지에는 이 갈래를 넣지 않았다 |
| 8 | HEAD | aiohttp 는 HEAD 에 JSON 미들웨어(ETag·압축)를 안 건다 | 같은 갈래로 처리하나 정답지 대조 안 함 |
| 9 | 안 옮긴 FileResponse 기능 | Range(206) · If-Match/If-Unmodified-Since(412) · 미리 압축한 `.gz`/`.br` 형제 | 없음(화면이 안 쓴다 · `ponytail:` 주석) |
| 10 | regime_band 의 시각 | `pd.Timestamp(글자, tz="UTC")` 는 꼴을 넓게 받는다 | 크론이 쓰는 꼴(`YYYY-MM-DD[ T]HH:MM[:SS[.f]]`)만 · 그 밖은 깨진 줄 |

## 6. 양성 대조

`python3 rtserve/tools/mutate_check.py <작업 폴더>` — 규칙 하나씩 틀리게 바꾼 사본으로 `cargo test` 를 돌려 실패해야 «잡힘». 원본 `rtserve/` 바이트는 끝에 대조.
2026-11-21 결과: **46/46 잡힘**(공통 동작 22 · 엔드포인트 22 · 파이썬 의미 2) · 무변경 사본 통과 · 원본 바이트 그대로.
첫 판에서 못 잡은 5개로 정답지를 보강했다: 정적 `.html` 파일 · 같은 초·같은 쪽에 grp 만 다른 사건 · stale 경계(100초 전) · 꼬리 32KB 가 줄 머리에서 정확히 잘리는 파일.
절대 경로 검사를 빼는 변이는 동등 변이(폴더 밖 검사가 같은 404)라 목록에서 뺐다.

## 7. 다음 조각 순서

1. **나머지 파일·SQLite 읽기 9**: state · gex · bubble-tape · pred-record(상대 강도 latest 를 rt_dirpred 가 파일로 쓰게) · chip-score · notify-center · news(엔티티 표) · macro-spikes(정확 stdev·뉴욕 시각) · breakout-detector(지울지 결정). + POST 쓰기 6(acct·push).
2. **메모리 상태 = rtcore 허브/상태와 잇기**: rtserve 가 rtcore `/rt` 를 구독(rtorder `coreval::subscribe` 와 같은 방식)해 footprint·supply-1s·oi-5m·liquidation-5m-history 를 같은 JSON 으로. SSE 두 개는 화면이 허브로 옮기면 지우고, 그 전까지는 rtserve 가 같은 프레임을 낸다(`/api/events` 의 티커 = rtcore Prices).
3. **계좌·주문 = rtorder intake 붙이기**: 14 주소 → `Desk::api_*` · order_request_log · client-timing · busy_or · exec 스위치. 그림자 모드(rtorder `shadow.rs`)로 먼저.
4. **계산 엔드포인트**: footprint-daily/4h·cells · flow/heatmap(래스터 행 집계) · session-alerts · situation · market-context · liquidation-5m-signal.
5. **요청 때 REST·모델**(맨 뒤): rtmodel 서빙을 붙이거나(dir4h·dir-short·vol-forecast) 워커가 파일로 쓰게 바꾼다. 가짜 서버 녹화본으로만 대조.
6. 나란히 가동: 파이썬 8787 과 rtserve 8790 을 같은 서버에서 띄우고 같은 요청을 둘에 보내 본문 대조(읽기 전용 엔드포인트만) → 서버 처리 시간 중앙값 비교(§2 전환 규칙).
