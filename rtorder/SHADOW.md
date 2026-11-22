# rtorder 서버 그림자 가동에 필요한 것 (2026-11-21)

가동 여부·시각은 사용자 결정이다. 이 문서는 목록만 적는다. 그림자 = 쓰기(주문·취소·레버리지·익절·고아 정리·텔레그램)를 **보내지 않고** «안 보낸 쓰기» 목록(`그림자(안 보냄): …` 줄)에 적는다. 그래도 어딘가에서 쓰기가 새면 `http::Binance` 의 그림자 막이가 보내기 전에 panic 한다.

## 1. 빌드 두 가지

| 빌드 | 키 | 할 수 있는 것 |
|---|---|---|
| 기본 `cargo build --release` | 읽지 않음(코드가 없음) | 허브 구독·감시 틀만 돈다. 계좌를 못 읽으니(서명 요청이 «키 없음») 무장 판정이 안 된다. 사용자 스트림도 안 뜬다(키가 없으면 원본처럼 안 띄움). |
| `cargo build --release --features live` + `RTORDER_MODE` 비움(=shadow) | `BINANCE_API_KEY`·`BINANCE_SECRET_KEY` 를 **환경 변수**에서 읽음(점-env 는 안 읽는다) | 계좌·사용자 스트림까지 읽고, 쓰기는 그림자. **의미 있는 그림자는 이쪽이다.** 실전송은 `RTORDER_MODE=live` 와 `RTORDER_LIVE_CONFIRM=실주문을-보낸다` 가 둘 다 있어야 켜진다. 그림자에서는 둘 다 넣지 않는다. |

## 2. 환경 변수

| 이름 | 필수 | 뜻 · 기본 |
|---|---|---|
| `RTORDER_DIR` | 예 | Rust 전용 상태 폴더. 파이썬 폴더(`order_requests.jsonl` 이 있는 `data/live`)는 시작을 거절한다(무장 파일을 서로 덮지 않게). |
| `DASHBOARD_MANUAL_EXEC_ENABLED` | 예(`1`) | 원본 실행 스위치. 꺼져 있으면 원본 기동 훅처럼 시작 정리·사용자 스트림을 안 띄우고, 감시 틱도 아무것도 안 한다. |
| `RTORDER_LISTEN` | 아니오 | 루프백 HTTP 문(기본 `127.0.0.1:8791` · 루프백만 · 묶기 실패 = 시작 안 함 -- 7절). |
| `RTORDER_STREAM` | 권장(`1`) | 사용자 데이터 스트림 + 체결 뒤 계좌 갱신기. 스위치와 키가 있어야 뜬다. |
| `RTORDER_HUB` | 사실상 필수 | rtcore 허브 ws 주소. 감시 5분봉·1초 체결가는 허브에서만 받는다(REST 대체 없음). 허브가 없으면 손절 판정 자료가 비어 판정을 못 한다. 호가·1분 변동성도 허브 값을 먼저 쓴다(나이 한도를 넘으면 REST). |
| `RTORDER_BAN_FILE` | 권장 | 기본은 `RTORDER_DIR/binance_ban_until`. **파이썬과 같은 파일**(`data/live/binance_ban_until`, `scripts/binance_ban_guard.BAN_FILE`)을 가리키게 해야 같은 IP 의 418·429 차단을 서로 지킨다. 형식이 같다(초 단위 실수 한 줄). |
| `BINANCE_API_KEY`·`BINANCE_SECRET_KEY` | live 빌드에서 | 위 1절. |
| `DASHBOARD_MANUAL_EXEC_SYMBOL` | 아니오 | 기본 `ETHUSDC`. |
| `DASHBOARD_SIZING_CAP_EQUITY_X` | 아니오 | 원본과 같은 배수 상한 덮어쓰기. |
| `RTORDER_PERIOD` | 아니오 | 감시 주기 초(기본 2 = 원본). |
| `RTORDER_BASE`·`RTORDER_WS` | 아니오 | 기본 = 실거래소(`https://fapi.binance.com` · `wss://fstream.binance.com/private/ws/`). 시험에서만 바꾼다. |
| `TELEGRAM_BOT_TOKEN`·`TELEGRAM_CHAT_ID` | 아니오 | 그림자에서는 어차피 보내지 않는다(`TELEGRAM …` 줄로만 남음). 파이썬은 점-env 에서 읽지만 rtorder 는 환경 변수만 본다. |
| `RTORDER_TELEGRAM_BASE`·`RTORDER_MAX_TICKS`·`RTORDER_MODE`·`RTORDER_LIVE_CONFIRM` | 아니오 | 시험용 · 실전송용. 그림자에서는 넣지 않는다. |

## 3. 읽는 파일 · 쓰는 파일 (전부 `RTORDER_DIR` 안 · 밴 파일만 예외)

| 파일 | 읽기 | 쓰기 |
|---|---|---|
| `manual_bracket_state.json` | 매 틱 | 무장·정리·재시도 시각(임시 파일 → rename) |
| `manual_limit_wait_<심볼>.json`·`manual_limit_wait_<심볼>_<wid>.json` | 시작 정리(복구) | 복구가 끝나면 지움. 그림자에서는 걸지 못했으니 남긴다. |
| `order_requests.jsonl` | -- | 접수 기록(POST /api/manual-* 한 줄씩 · 원본과 같은 칸) |
| `.rtorder` | 시작 | 시작할 때 쓰는 표식(이 폴더는 rtorder 것) |
| 밴 파일(`RTORDER_BAN_FILE`) | 요청마다 | 418·429 를 받으면 |

**빈 폴더로 그림자를 켜면 무장이 하나도 없다.** 그림자에서는 문으로 들어온 진입이 거래소에 나가지 않아 체결·무장이 생기지 않는다. 파이썬과 같은 판정을 보려면 파이썬 `data/live/manual_bracket_state.json` 의 **사본**을 주기적으로 `RTORDER_DIR` 에 넣어야 한다. 어느 주기로 넣을지는 사용자 결정이다. 원본 파일을 rtorder 가 직접 쓰게 하면 안 된다.

## 4. 거래소로 나가는 읽기 요청 (2026-11-21 서버 실측 뒤 다시 · 그림자는 가중치를 줄여 돈다)

서버 실측: 응답 머리 `x-mbx-used-weight-1m` 를 15초 간격으로 4번 잰 값이 1431 · 248 · 592 · 960 이었다(분 창 안에서 쌓였다가 매 분 0 으로 돌아간다). 지금 분당 최고는 **1,431 이상**으로, 한도 2,400 의 60% 다. 그래서 그림자는 아래처럼 줄여 돈다. **실전송 모드는 원본 그대로다**(손절 청산을 늦추지 않게).

| 그림자만 바뀐 것 | 원본(실전송) | 그림자 |
|---|---|---|
| 감시 틱 계좌 캐시 수명 | 10초 | 60초(`RTORDER_SHADOW_ACCOUNT_TTL`) |
| 체결 뒤 계좌 다시 읽기 최소 간격 | 3초 | 30초 |
| 계좌 한 번 읽기 | 25요청 · 가중치 77(userTrades 11 × 5 포함) | 14요청 · 가중치 22(userTrades 건너뜀) |
| 거래소 사용 가중치가 1,800 을 넘은 분 | -- | 그 분이 끝날 때까지 REST 0(«RateLimitPause» -- 계좌는 «못 읽음 = 모름»으로 무장 그대로) |
| 허브 끊김(`feed.stale`) 중 거래소 직접 조회 · **무장 코인마다**(2026-11-22) | 감시 틱(2초)마다 `ticker/price` 1 + 마감 5분봉이 바뀔 때만 `klines 5m limit=2` 1 = 틱당 ≤2 요청 · 분당 약 31 요청 · 가중치 약 31(둘 다 1) · 1,800 쉼 없음 | 실제 요청은 60초에 한 번(ticker 1 + klines ≤1 = 분당 ≤2 · 가중치 ≤2) · 나머지 틱은 «대체했을 것» 로그 줄(60초에 한 번) |

- userTrades 를 건너뛰어도 되는 이유: 원본 `fetch_account` 가 userTrades 로 만드는 것은 두 가지뿐이다. 하나는 왕복 원장(`trades` -- 화면 손익·수수료 표시)이고, 다른 하나는 열린 포지션의 `entry_at`(보유 시간 표시 → 계획의 `hold_remaining_min`·`trade_plan` 설명 칸)이다. 감시 판정과 청산 계획의 수량·가격·손절선에는 쓰지 않는다. 건너뛰면 `entry_at` 은 원본이 왕복을 못 찾았을 때와 같은 `updated_at` 이 되고, 계좌에 `trades_skipped: true` 가 붙는다.
- 가중치 = 바이낸스 선물 공개 문서 값이다. 계좌 읽기 22 = time 1 + account 5 + positionRisk 5 + 심볼 11 × openOrders 1.

**그림자가 더하는 분당 가중치 상한** (같은 IP 의 파이썬 몫과 별개)

| 상태 | 계산 | 분당 상한 |
|---|---|---|
| 무장 없음 · 체결 없음 | 감시 틱이 계좌를 안 읽는다 · 스트림 연결 때 1번(22) | 사실상 0(연결한 분만 22) |
| 무장 있음 · 체결 없음 | 감시 틱 60초 캐시 1번 | **22** |
| 체결이 몰림(무장 있음) | 22 + 체결 뒤 30초 간격 2번 44 | **66** |
| + 손절선에 닿은 칸이 있음(그림자는 30초마다 다시 조립) | 칸 하나당 2번 × (새 계좌 22 · 스트림 계좌를 쓸 수 있으면 0 + premiumIndex 1 + commissionRate 20 은 1시간에 1번) | 칸 하나당 **약 46** 더 |
| + 허브 끊김(무장 코인마다) | 60초에 ticker/price 1 + klines 5m 1 | 코인당 **2** 더 |
| 거래소 사용 가중치 > 1,800 인 분 | 그 분 나머지 | **0** |

전에는 무장이 있으면 분당 약 462 를 더했다. 지금은 22~66 이다. 실측 최고 1,431 에 더해도 약 1,500(한도의 62%)이다.
`--release` 시험 `tests/shadow_rate.rs`: 가짜 거래소와 2초마다 체결 알림을 보내는 가짜 스트림으로 무장 하나 있는 그림자를 **5분** 돌린다. 분 창별 가중치가 모두 90 미만이고, userTrades 0 · 쓰기 0 이다. 1,900 을 알린 분에는 그 분 경계까지 요청이 0 이다. 계량 줄은 분마다 남는다.
- 계량 줄(서버 로그): 매 분 `rtorder 요청 계량 · 지난 60초 · 요청 N · 가중치 W(추정) · 거래소 사용 가중치(마지막 응답) U · 쉰 요청 S · 그림자`.

허브(`RTORDER_HUB`)가 끊기면 호가·변동성이 REST 로 떨어진다. 청산 조립 때만 묻기 때문에 늘어나는 양은 작다. 감시 5분봉·1초 체결가는 2026-11-22부터 허브가 10초 넘게 비면 무장 코인만 거래소에서 직접 읽는다(위 표 · RULES.md «허브 끊김 → 거래소 직접 조회»).

### listenKey 를 파이썬과 같은 키로 둘이 만들면
- 바이낸스 선물은 계좌에 살아 있는 listenKey 가 있으면 POST 에 **같은 값**을 돌려주고 유효 시간을 60분 늘린다. PUT 도 같은 키를 늘린다. 그래서 둘은 같은 listenKey 로 각자 웹소켓을 연다.
- 위험은 DELETE 하나뿐이다. 한쪽이 지우면 같은 키를 쓰는 다른 쪽 연결도 끊긴다(파이썬 orders.py user_stream_loop 주석도 같은 이유로 DELETE 를 안 부른다).
- rtorder 는 DELETE 를 부르지 않는다. 그림자 막이(`http.rs guard`)도 2026-11-21 부터 listenKey 는 POST·PUT 만 받고 DELETE 는 보내기 전에 panic 한다. 그림자가 파이썬 스트림을 끊을 길은 없다.
- PUT 연장은 파이썬에도 해롭지 않다(같은 키가 더 오래 산다). 24시간마다 거래소가 연결을 끊으면 둘이 각자 다시 붙는다.
- 그래도 끄려면 `RTORDER_SHADOW_STREAM=0` 이다. listenKey 를 만들지 않고 계좌 REST 만 쓴다.

### 띄우는 법 · 띄운 뒤 볼 것 (서버)
- 빌드(개발기): `cd rtorder && cargo build --release --features live -j 2 --bin rtorder` → `rtorder/target/release/rtorder` 를 서버의 `<저장소>/tmp/rtbin/rtorder` 로 복사.
- 띄우기: `bash scripts/ops/supervisor_rtorder.sh &` · 파이썬 무장 사본: `nohup bash scripts/ops/rtorder_shadow_copy_brackets.sh >> logs/supervisor/rtorder_copy.log 2>&1 &`
- 확인 목록:
  1. 프로세스: `pgrep -af tmp/rtbin/rtorder` 가 하나, 감독자 `pgrep -af supervisor_rtorder` 가 하나.
  2. 포트: `ss -ltnp | grep 8791` 이 `127.0.0.1:8791` 만(0.0.0.0 이면 즉시 내린다).
  3. 로그 첫 줄: `logs/supervisor/rtorder_YYYYMMDD.log` 에서 «SUPERVISOR starting …» 다음 줄이 **«모드: 그림자 · 쓰기 막힘»**, 이어서 «rtorder 문 http://127.0.0.1:8791» · «rtorder 시작 · 그림자 · 폴더 /mnt/data2/rtcollect_shadow/rtorder …».
  4. 안 보낸 쓰기: 같은 로그의 «그림자(안 보냄): …» 줄(취소·익절·청산 GTX 등). 그림자 막이 panic 이 보이면(«그림자 모드: 쓰기 요청 금지») 바로 내리고 알린다 -- 막이가 일한 것이지만 어딘가 쓰기 경로가 샌 것이다.
  5. `curl -s http://127.0.0.1:8791/internal/acct` → `{"seq": N, "user_stream": {"connected": true, "last_event_at": …, "error": null}}`(스트림 켬이면 connected true · 끔이면 false·seq 0).
  6. 상태 폴더: `/mnt/data2/rtcollect_shadow/rtorder/` 에 `.rtorder` 표식 · (사본 스크립트가 돌면) `manual_bracket_state.json`.
  7. 가중치: 로그의 «rtorder 요청 계량» 줄의 가중치가 위 표 상한 안인지(무장 있음 22 · 체결 몰림 66) · «거래소 사용 가중치» 가 1,800 을 넘은 분에는 «쉰 요청» 이 0 보다 크다.
- 내리기: 감독자 pid 를 먼저 끝내고 rtorder 에 SIGTERM(종료 = 받기 멈춤 → 도는 작업 취소 → 답 중인 요청 2초 · 종료코드 0).

## 5. 키가 필요한 자리

- 서명 GET: 계좌 읽기 전부(`/fapi/v2/account`·`positionRisk`·`userTrades`·`openOrders`·`openAlgoOrders`) · `commissionRate`.
- 키만(서명 없음): `listenKey` POST·PUT.
- 쓰기(주문·취소·레버리지·algoOrder): 그림자에서는 보내지 않는다. 보내려 하면 panic.
- 키가 없는 빌드(기본)에서는 위 요청이 모두 원본의 «키 없음»과 같은 결과가 된다. 계좌는 `ok=false` 로 오고, 감시는 «못 읽음 = 모름»으로 무장을 그대로 둔다.

## 6. 배치 (2026-11-21 결정: rtorder 는 응답 서버와 **따로 둔 프로세스**)

근거는 두 가지다(조정자 결정). 설계 §4-3 이 주문 연결을 화면 응답과 같은 실행 줄에 두지 말라고 한다. 그리고 손절 감시가 화면 재배포나 응답 서버 장애와 같이 멈추면 안 된다. 127.0.0.1 한 겹의 비용(1ms 미만)은 거래소 왕복(약 41ms)보다 작다.

한 프로세스 안에서는 다음이 모두 **하나**를 같이 쓴다.
- `DeskCell` 하나: 감시 틱, 시작 정리, 접수 핸들러, 접수가 띄운 작업. 무장 칸과 대기 칸이 메모리에서도 하나다. 원본처럼 감시가 청산 중이면 같은 코인 접수는 409 를 받는다.
- 계좌 한 곳(`account::SharedAccount`): 감시 swr 10초, 접수의 새로 읽기, 체결 뒤 갱신기, 화면 `?fresh=1`(1초 안 받은 값은 재사용)이 같은 캐시와 같은 last_good 을 쓴다.
- `Front` 하나: 주문 잠금. 감시는 이 잠금이 잡혀 있으면 손절 발동을 다음 틱으로 미룬다(D9).

## 7. 루프백 HTTP 문 (`src/door.rs` · `RTORDER_LISTEN`)

- 주소는 `RTORDER_LISTEN` 이고 기본값은 `127.0.0.1:8791` 이다. 루프백이 아니면 설정 오류(종료코드 2)로 시작하지 않는다.
  - 묶기에 실패해도 시작하지 않는다(종료코드 2). 같은 주소로 둘째 rtorder 가 떠서 같은 계좌를 두 번 감시하는 일을 막는다.
  - 시작할 때 표준출력 첫 줄에 `rtorder 문 http://<주소>` 를 찍는다.
- 주문·계좌 주소 14개는 `route::Front::call` 그대로다. 본문 글자, 상태, Content-Type, Cache-Control, Allow 가 원본 aiohttp 와 바이트까지 같다.
- `GET /internal/acct` 는 `{"seq": <acct_seq>, "user_stream": {"connected", "last_event_at", "error"}}` 를 낸다. 응답 서버의 `/api/events` 가 1초마다 읽는다.
- 그 밖의 주소는 404 이고 본문은 `404: Not Found` 다.
- 접수 기록은 rtorder 가 `RTORDER_DIR/order_requests.jsonl` 에 직접 적는다. 원본 order_request_log 와 칸·순서가 같다.
  - at 과 ms 는 rtorder 가 잰다. at 은 KST 밀리초다.
  - ua 와 ip 는 응답 서버가 넘긴 머리에서 읽는다.
  - 폴더에는 `.rtorder` 표식을 둔다. 표식이 있으면 접수 기록이 있어도 파이썬 폴더로 보지 않는다.
- 접수가 내놓은 작업(진입 따라가기, 청산 따라가기)은 이 프로세스 안에서 돈다. 감시와 같은 `DeskCell` 위다.
- 연결과 작업은 감시와 같은 실행 줄에서 돌지만 서로 막지 않는다(FuturesUnordered). 클라이언트가 반쯤 보낸 요청을 쥐고 있거나, 끊거나, 쓰레기를 보내도 감시 틱은 그대로 돈다(`tests/door_bin.rs`).

### 응답 서버가 프록시할 때 넘겨야 하는 것

| 넘길 것 | 왜 |
|---|---|
| 메서드 · 경로 · **쿼리 원문**(풀지 말고 그대로) | 같은 키가 여럿이면 첫 값을 쓴다. `+` 와 `%XX` 는 rtorder 가 푼다 |
| `User-Agent` | 접수 기록 ua(200자) |
| `CF-Connecting-IP` | 접수 기록 ip 1순위(Cloudflare 를 거친 실제 접속 주소) |
| `X-Forwarded-For` | ip 2순위. 화면이 보낸 값이 없으면 **응답 서버가 자기 접속 주소(request.remote)를 넣어서** 넘긴다 -- 안 넣으면 기록 ip 가 늘 127.0.0.1 이 된다 |
| 응답의 상태 · `Content-Type` · `Cache-Control` · `Allow` · 본문 | 그대로 돌려준다. ETag 와 압축은 응답 서버 미들웨어 몫이다(원본도 바깥 미들웨어였다) |

- rtorder 문이 닫혀 있으면(죽음, 재시작 중) 응답 서버가 502 를 낸다. 손절 감시는 rtorder 안에 있어 응답 서버와 상관없이 돈다.
- 시간 제한: 요청은 칸을 5초까지 기다리고, 넘으면 409 already_working 이다(`route::DESK_WAIT_SEC`). 프록시 시간 제한은 이보다 길게(10초 이상) 잡는다.

### 한쪽이 죽으면 · 재시작 복구
- 손절은 거래소에 걸려 있지 않다. 비상 스탑은 2026-11-21 사용자 결정으로 없앴다. 손절은 감시 틱(2초)뿐이다. **rtorder 가 죽어 있는 동안에는 손절이 없다.** 거래소에 남는 것은 익절 지정가(dbtp)와 걸려 있던 진입 지정가(dbwt)뿐이다.
- 응답 서버가 죽으면 화면과 주문 접수만 멈춘다. 감시, 사용자 스트림, 도는 작업은 계속 돈다.
- rtorder 가 죽으면 무장 파일과 지정가 대기 복구 파일은 디스크에 남는다. 칸의 phase, 도는 작업, 주문 잠금은 사라진다(원본과 같다). supervisor 가 다시 띄우면 실행 스위치가 켜져 있을 때 이 순서로 돈다. ① 남은 dbwt·dbpg 를 지운다(지운 결과는 eth 대기 칸에 «재시작으로 취소»로 남는다). ② 복구 파일로 무장한다. ③ 감시를 시작한다. ④ 사용자 스트림이 붙자마자 계좌를 한 번 읽는다.
  - 청산 도중 죽었으면 무장이 남아 있다. 가격이 아직 손절선 너머면 다음 틱에 다시 쏜다.
  - 진입 따라가기 도중 죽었으면 걸려 있던 GTX 는 ①에서 지워진다. 이미 체결된 몫은 복구 파일 선으로 무장된다.
- 파이썬 서버의 실전송과 rtorder 실전송을 같은 계좌에서 동시에 켜면 손절이 두 번 나간다. 그림자일 때만 둘을 같이 돌릴 수 있다.

### 의도한 차이 (원본과 다르게 둔 것)
- 접수 요청 하나가 칸(DeskCell)을 끝까지 쥔다. 조립 REST 동안 다른 주문 요청은 줄을 서고, 5초를 넘으면 409 를 받는다(`route::DESK_WAIT_SEC`).
- 감시 틱은 2026-11-21 부터 칸 사본에서 돈다. 손절 청산이 도는 동안에도 접수는 원본처럼 바로 돈다(위 «틱이 칸을 쥔다» 차이는 없어졌다 -- RULES.md 마지막 절).
- 종료는 원본 순서와 같다(R5·R6). 받기를 멈추고, 도는 작업과 감시 틱을 취소하고, 답 중인 요청을 2초 기다린다.

## 8. «그림자 기록» -- 파이썬 쪽 (2026-11-21 구현: order-shadow-log 판 `live/execution/order_shadow.py` · 조정자 검토 뒤 배포)

구현은 아래 제안서와 같고, 다른 점은 다음과 같다.
- 기본 켜짐이고, `ORDER_SHADOW_LOG=0` 이면 끈다. 경로는 orders.register 가 `LIVE_DIR/order_shadow.jsonl` 로 정한다(chmod 600 · 50MB 가 넘으면 `.1` 로 한 번 돌린다).
- 이벤트 루프에서는 JSON 한 줄을 만들어 `queue.SimpleQueue` 에 넣기만 한다. 파일 쓰기, 돌리기, 무장 파일·복구 파일 읽기(start·end 의 file·limits)는 뒤의 스레드 하나가 맡는다.
- 응답 원문은 aiohttp 가 `json()` 하며 받아 둔 바이트(`response._body`)를 쓴다. 새로 `.read()` 하지 않으므로 기존 `.json()` 동작이 그대로다(정상·빈 본문 EmptyResponse·400·JSON 아님·끊김 5꼴로 기록 켬/끔 결과 같음을 확인).
- 작업(진입·청산)은 제 chain 을 따로 쓰고, `parent` 에 접수 요청 chain 을 적는다(접수 기록 줄의 `chain` 과 잇는다). 감시 틱은 `shadow_bracket_tick` 이 감싸서, 요청이 나갔거나 무장 파일을 쓴 틱만 남긴다.
- 체결 알림은 chain `"*"` 로 적는다. exchangeInfo 원문(수 MB) 대신 그 심볼의 격자를 `input filters` 로 적는다. 1초 가격은 `[[초, 가격]…]` 으로 적는다.
- 가림: 서명은 대상(target)과 예외 문구 둘 다에서 지우고, 키 머리는 적지 않고, listenKey 는 `<listenKey>` 로 바꾼다.
- 시험: 파이썬 주문 시험 19개 파일 254개는 그대로 통과했고, 새 시험 7개를 더해 261 통과. rtorder 정답지 생성기 셋(desk·flow·route)을 새 판 파이썬으로 다시 돌리면 desk·flow 는 글자까지 같고, route 는 server_ms 숫자만 다르다(159/159).

### 제안서 원문(참고)

파일은 새로 하나 쓴다: `data/live/order_shadow.jsonl`. 한 줄에 한 기록이고, 모양은 `{"rec": ..., "chain": ..., "t": time.time(), "mono": time.monotonic(), ...}` 이다. 쓰기는 한 곳에서만 한다.

```python
# live/execution/live_manual_peg_execute_20260912.py (모듈 위쪽 · 기존 _ORDER_T 옆)
SHADOW_LOG = Path(__file__).resolve().parents[2] / "data" / "live" / "order_shadow.jsonl"
_SHADOW_ON = os.getenv("ORDER_SHADOW_LOG") == "1"            # 기본 끔 -- 켜는 것은 사용자
_CHAIN: contextvars.ContextVar[str | None] = contextvars.ContextVar("chain", default=None)

def shadow_rec(rec: str, **kw) -> None:
    if not _SHADOW_ON:
        return
    try:                                                        # 기록 실패가 주문 흐름을 막으면 안 된다
        row = {"rec": rec, "chain": kw.pop("chain", None) or _CHAIN.get(), "t": time.time(), "mono": time.monotonic(), **kw}
        with SHADOW_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_redact(row), ensure_ascii=False, default=str) + "\n")
    except Exception:  # noqa: BLE001
        pass

def new_chain(kind: str) -> str:
    c = f"{kind}-{time.time_ns():x}"
    _CHAIN.set(c)                                               # create_task 는 맥락을 복사한다 -- 집행 작업까지 같은 chain
    return c
```

| # | 자리 | 넣을 줄 |
|---|---|---|
| 1 | orders.py:965 `entry_then_bracket` 첫 줄 | `shadow_rec("start", op="entry", plan=copy.deepcopy(plan), limit=limit, slot=copy.deepcopy(state), file=bracket_load(), limits=sorted(p.name for p in LIVE_DIR.glob("manual_limit_wait*.json")))` |
| 2 | orders.py `api_manual_exit_submit` 의 create_task 바로 앞(2045 근처) | `shadow_rec("start", op="exit", plan=..., rearm=fraction < 1, slot=copy.deepcopy(st), file=bracket_load())` |
| 3 | orders.py:1128 `bracket_tick` 첫 줄 | `new_chain("tick")` -- start 줄은 틱 끝(13)에서 «요청이 나갔거나 파일이 바뀐 틱»일 때만 앞당겨 적는다(2초마다 쌓이지 않게: 틱 안 기록을 목록에 모았다가 조건이 맞을 때만 쓴다) |
| 4 | 실행 모듈:101 `signed` 의 `session.request` | 보내기 직전 `t_send = time.time()` · 응답에서 `raw = await response.read()`를 먼저 하고 그다음 기존 `payload = await response.json()`(본문은 캐시되어 동작이 같다) · `shadow_rec("wire", method=method, target=path + "?" + _strip_sig(params_qs), status=response.status, body=raw.decode("utf-8", "replace"), sent_at=t_send)` · 예외 갈래에서는 `error=f"{type(exc).__name__}: {exc}"` |
| 5 | `maker_price`(bookTicker) · 계좌 `_clock_offset`(/time) · `load_filters`(exchangeInfo) | 4 와 같은 칸(서명 없음) |
| 6 | orders.py:1134 swr_cached("binance_account") 뒤 | `shadow_rec("input", what="account", value=acct)` |
| 7 | orders.py:1151 load_market_history 뒤 | `shadow_rec("input", what="bars", args=[asset], value=bars)` |
| 8 | orders.py:1002 `stop_touched` · 1160 sec_prices 뒤 | `shadow_rec("input", what="sec", args=[asset], value=[r for r in rows if r[0] > since])` |
| 9 | orders.py:1195·1250 assemble_exit_plan 뒤 | `shadow_rec("input", what="exit_plan", args=[pside, frac, fresh, asset, only_symbol], value={"plan": plan, "err": err})` |
| 10 | `limit_reprice` 의 호가·필터 · `basis_for` 기준 호가 | `what="book"` / `what="filters"` |
| 11 | 실행 모듈:49 `wake_fills` | `shadow_rec("input", what="wake")` |
| 12 | `api_manual_wait_action`·`api_manual_limit_amend`·`api_manual_entry_cancel` 끝(응답 만들기 직전) | `shadow_rec("input", what="handler", chain=<그 칸의 chain>, args=[이름, dict(request.query)], value={"status": s, "body": body})` -- 칸 dict 에 `chain` 을 넣어 두면(1·2 에서) 핸들러가 꺼내 쓴다 |
| 13 | `entry_then_bracket` 끝(finally) · `settle` 뒤 · `bracket_tick` 끝 | `shadow_rec("end", slot=..., file=bracket_load(), limits=..., raised=repr(exc) or None)` |
| 14 | server.py:1371 `order_request_log` 의 row | `"chain": getattr(request, "_chain", None)` -- 접수 핸들러가 `request["chain"] = new_chain("api")` 로 남긴다 |

**가릴 칸 (`_redact`)**: 파일에 키·서명·listenKey 가 절대 남지 않게 한다.
- `target`: 서명 쿼리에서 `signature=` 와 그 뒤를 잘라 낸다(`_strip_sig`). `timestamp`·`recvWindow` 는 남긴다(재생이 시계 차이를 본다).
- 머리: `X-MBX-APIKEY` 는 적지 않는다(row 에 headers 칸 자체를 두지 않는다).
- listenKey: `/fapi/v1/listenKey` 응답 본문의 `listenKey` 값과 웹소켓 주소 끝은 `"<listenKey>"` 로 바꾼다.
- 텔레그램 토큰·채팅 번호: 주문 경로 기록에는 들어오지 않는다(보내기를 기록하지 않는다).
- 계좌 값(잔고·증거금·포지션 수량·진입가·미실현·원장 trades·income): **정확한 재생에는 진짜 값이 필요하다**(수량 반올림·상한·청산 수량이 이 값으로 정해진다). 그래서 값을 가리는 대신 두 벌로 나눈다.
  - 서버 원본 파일: chmod 600, git 밖(`data/live` 는 이미 무시 목록), 서버 밖으로 복사하지 않는다. 재생도 서버에서 돌린다.
  - 개발기로 가져올 사본: `tools/redact_shadow.py`(새로 만들 것)로 금액 칸 전부(`balance*`·`available*`·`margin*`·`wallet*`·`unrealized*`·`assets`·`trades`·`income*`)를 지우고, 서명 응답 본문은 상태 코드만 남긴다. 이 사본은 재생이 «구조(요청 순서·핸들러 응답)»만 맞춰 보고 금액이 드는 비교는 건너뛴다.
- `ua`·`ip`(14번)는 접수 기록에 이미 있다. 그림자 기록에는 넣지 않는다.

크기 어림(RULES.md 와 같다): 따라가기 120초 ≈ 요청 80 · 무장 1회 ≈ 5 · 감시 발동 ≈ 10 → 하루 접수 수십 건이면 수 MB.

## 9. 주문 전환(실전송) 전 점검표 · 전환·되돌리기 순서 (2026-11-21 · 전환 자체는 사용자 승인 사항 -- 여기는 문서까지)

### 통과해야 하는 것
| # | 점검 | 통과 기준 | 보는 법 |
|---|---|---|---|
| 1 | 실기록 재생 | 서버 `data/live/order_shadow.jsonl`(+ `.1`) 전부 **어긋난 작업 0**. 조립 구간 왕복은 따로 세고, 잘린 chain(재시작)은 빼고 센다. 진입(따라가기)·지정가 대기·감시 진입·청산(부분·전량)·감시 발동 청산·분할 익절 조각·화면 입력(대기 취소·선 옮김)이 **각 1건 이상** 들어 있어야 한다 | `tmp/rtbin/rtorder replay data/live/order_shadow.jsonl` -- 마지막 줄 «… 어긋난 작업 0». 다르면 «첫 어긋남»의 위치(칸 이름)를 보고 `--show-values` 는 서버 안에서만 |
| 2 | 그림자 가동 중 막이·사고 | panic 0(«그림자 모드: 쓰기 요청 금지» · «재생: 바깥 주소» 줄 0) · 감독자 재기동 0 · 418·429 0 | `grep -c` 로 `logs/supervisor/rtorder_*.log` 확인. 시간 기준 대신 사건 수 기준(사용자 «기다리지 말고 바로바로»): 아래 3번이 실제 사건 3건 이상을 덮으면 된다 |
| 3 | 감시 판정 대조 | 파이썬 서버가 실제로 한 감시 행동(정리·발동 청산·분할 조각)마다 같은 시각(±1틱)에 그림자 로그에 같은 «그림자(안 보냄)» 줄이 있다. 그림자만 한 행동도 0 | 파이썬 서버 로그 «bracket …» 줄 ↔ 그림자 로그 «그림자(안 보냄): CLEAR/GTX …» 줄 (무장 사본 스크립트가 돌고 있어야 같은 무장을 본다) |
| 4 | 가중치 | 그림자 계량 줄이 상한 안(4절) · 거래소 사용 가중치 최고가 전환 뒤에도 한도의 80% 아래 | «rtorder 요청 계량» 줄 |
| 5 | 시험 | `cd rtorder && cargo test --release --features live -j 2` 전부 통과 · 정답지 생성기(desk·flow·route·http·py_shadow)를 그때의 main 판 파이썬으로 다시 돌려 차이 0 | 개발기 |
| 6 | 접수 문 | 응답 서버(rtserve)가 주문·계좌 14주소를 rtorder 로 넘기는 경로에서 응답 바이트가 파이썬과 같다(route 정답지 · door_bin 시험과 같은 사례를 프록시로) | 개발기 + 서버에서 한 번 |
| 7 | 허브 가격 수신 | 그림자 가동 내내 «rtorder 허브 가격 못 받음» 줄 0 · «rtorder 허브 연결 끝 · 닫힘 …» 줄이 재기동 때 말고는 없음 · 전환 직전 `curl 127.0.0.1:8791/internal/acct` 의 `feed.stale=false`·`feed.connected=true`·`feed.symbols` 에 무장 코인의 화면 심볼(…USDT 1초가)과 주문 심볼(…USDC 호가)이 모두 있고 나이가 몇 초 안 | 2026-11-22 서버 그림자에서 실제로 걸렸다 -- 허브 연결당 상한 8 < 구독 10 이라 Close(Policy)로 거절당해 **떠 있고 panic 0 인데 감시 자료가 내내 비어 있었다**(전에는 1초마다 다시 붙었다 · 지금은 첫 메시지 뒤에만 1초로 되돌리고 1·2·4…30초로 늘림 · 닫힘 사유를 같은 사유 1분에 한 줄) |

### 전환 순서 (파이썬 → rtorder) -- 손절 감시가 **겹치지도(두 번 청산) 비지도(손절 없음) 않게**
감시가 겹치면 두 번 청산되고, 비면 그동안 손절이 없다(거래소 비상 스탑을 없앴다). 겹침이 더 위험하므로 **짧게 비우는 쪽**으로 한다. 빈 시간 목표는 30초 안이다.
0. **조건:** 도는 작업이 없을 때만 한다. 진입 칸이 idle 이고, 대기 칸에 busy 가 없고, 청산이 진행 중이지 않아야 한다(`/api/manual-entry/status`). 따라가기·지정가 대기는 넘겨받지 못한다 -- 새 쪽의 시작 정리가 남은 GTX 를 지운다. 열린 포지션·무장은 있어도 된다(무장 파일로 넘긴다). 가격이 모든 손절선에서 충분히 먼 때를 고른다(빈 시간 동안 닿지 않게).
1. 사용자에게 «전환 중 주문 누르지 않기»를 알린다. 응답 서버가 프록시라면 manual-* 를 잠깐 503 으로 막는다.
2. 파이썬 감시 끔: 대시보드를 `DASHBOARD_MANUAL_EXEC_ENABLED=0` 으로 다시 띄운다(파이썬 감시 틱·사용자 스트림·시작 정리가 스위치를 본다). ← 빈 시간 시작
3. 무장 넘기기: `data/live/manual_bracket_state.json` 을 rtorder 폴더로 마지막으로 복사(사본 스크립트를 멈추고 cp 한 번 · 바이트 같은지 cmp). `manual_limit_wait_*.json` 이 있으면 같이(0번 조건이면 없어야 한다).
4. rtorder 를 실전송으로 다시 띄운다: live 빌드 + 감독 스크립트의 그림자 고정 줄을 사용자 승인 아래 바꾼 판(`RTORDER_MODE=live` · `RTORDER_LIVE_CONFIRM=실주문을-보낸다` · 그림자 전용 줄임 해제 -- 계좌 캐시 10초·userTrades 포함) · 첫 줄 «모드: 실전송». ← 첫 감시 틱(2초)에서 빈 시간 끝
5. 확인: 로그에 «rtorder 시작 · 실전송» → 무장 N개 → 첫 틱 계좌 읽기 · `/internal/acct` connected · 응답 서버 manual-* 를 rtorder 로 열기 · 사용자에게 알림.
6. 사본 스크립트는 다시 켜지 않는다(이제 rtorder 가 주인). 파이썬 대시보드는 화면만(실행 스위치 0).

### 되돌리기 (rtorder → 파이썬) -- 같은 원칙(겹치지 않게 · 짧게 비움)
0. 같은 조건: 도는 작업이 없을 때.
1. 접수를 막는다(응답 서버 manual-* 503).
2. rtorder 를 그림자로 다시 띄운다(감독 스크립트 원래 판 -- 그림자 고정). 쓰기가 막힌다. ← 빈 시간 시작
3. rtorder 폴더의 `manual_bracket_state.json` 을 파이썬 `data/live/` 로 복사한다(그 사이 rtorder 가 바꾼 무장을 잃지 않게 -- 먼저 파이썬 쪽 파일을 따로 보관).
4. 파이썬 대시보드를 `DASHBOARD_MANUAL_EXEC_ENABLED=1` 로 다시 띄운다. ← 파이썬 첫 감시 틱에서 빈 시간 끝 · 접수를 파이썬으로 연다.
5. 사본 스크립트를 다시 켠다(그림자 비교로 돌아감).

### 알아 둘 것
- 🔴**위험 -- 실전송 중 허브가 비면(2026-11-22 사용자 결정으로 처리: 거래소 직접 조회로 감시를 잇고 알린다).** 감시 5분봉·1초 체결가의 1차 원천은 허브다. `feed.stale`(10초 넘게 못 받음 · 한 번도 못 받음)이면 **무장 코인만** 감시 틱이 거래소에서 직접 읽는다: 5분봉 = `klines 5m limit=2` 의 마감 봉(원본 `load_market_history` 와 같은 마감 봉 규약) · 1초가 자리 = `ticker/price`(그 순간 마지막 체결가) 틱마다 1점. 한 틱은 한 원천만 쓰고, 허브가 돌아오면 다음 틱부터 허브 값이다. 알림(텔레그램 · 그림자는 로그 줄만): 들어갈 때 «허브 가격 끊김 N초 · 거래소 직접 조회로 감시 계속 · 무장 코인 …» · 10분마다 다시 · 돌아오면 «허브 가격 돌아옴». **남는 위험: 닿음 판정이 원본보다 늦거나 놓칠 수 있다** -- 원본 1초 수급 링은 초마다 마지막 체결가(1초 안 꼬리는 원본도 못 본다)인데 대체는 2초마다 한 점이라, 2초 사이에 손절선을 찍고 되돌아온 가격은 못 본다(원본은 그 초의 마지막 체결가가 선 너머면 본다). 닿는 시각은 최대 약 2초 + 왕복(수십~수백 ms) 늦다. 빨리 닿는 쪽 오판은 없다(같은 마지막 체결가). 허브와 거래소가 **둘 다** 안 되면 자료가 비어 hold(조용한 정지) -- 그때는 «🔴허브·거래소 둘 다 가격을 못 읽음» 알림을 60초에 한 번 보낸다(나머지 틱은 «직접 조회 실패» 로그 줄).
- 거래소에 걸린 익절 지정가(dbtp)·대기 지정가(dbwt)는 두 쪽이 같은 표식을 쓰므로 그대로 넘어간다. 새 쪽의 시작 정리는 dbwt·dbpg 만 지운다(익절은 안 지움).
- 사용자 스트림 listenKey 는 같은 값을 공유한다. 어느 쪽도 DELETE 하지 않는다.
- 빈 시간을 0 으로 하려면 «감시 손잡이 파일» 같은 실행 중 전환 스위치를 파이썬에 넣어야 한다(지금은 재시작으로만 끈다 · 서빙 코드라 사용자 결정).
