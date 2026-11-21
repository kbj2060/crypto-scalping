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
| 밴 파일(`RTORDER_BAN_FILE`) | 요청마다 | 418·429 를 받으면 |

**빈 폴더로 그림자를 켜면 무장이 하나도 없다.** 접수 문(웹 서버 R6)이 아직 없어 rtorder 스스로는 무장하지 않는다. 파이썬과 같은 판정을 보려면 파이썬 `data/live/manual_bracket_state.json` 의 **사본**을 주기적으로 `RTORDER_DIR` 에 넣어야 한다. 어느 주기로 넣을지는 사용자 결정이다. 원본 파일을 rtorder 가 직접 쓰게 하면 안 된다.

## 4. 거래소로 나가는 읽기 요청 (파이썬 서버가 같은 계좌·같은 IP 로 같은 일을 하므로 그대로 두 배가 된다)

| 언제 | 요청 | 횟수 | 비고 |
|---|---|---|---|
| 감시 틱: 무장이 하나라도 있고 스위치가 켜져 있을 때 | 계좌 읽기 1회 = 25요청(`/fapi/v1/time` · `/fapi/v2/account` · `/fapi/v2/positionRisk` · 심볼 11개 × `/fapi/v1/userTrades`·`/fapi/v1/openOrders`) · 무게 약 50 | 10초 캐시라 **분당 6회 = 150요청 · 무게 약 300** | 무장이 없으면 0. 파이썬도 같은 조건에서 같은 양이다. |
| 체결 알림 뒤 계좌 갱신기 | 계좌 읽기 1회(위와 같음) | 연결 때 1회 + 체결마다(0.3초 묶음) · **최소 3초 간격 → 연속 체결이면 분당 최대 20회(500요청 · 무게 약 1,000)** | 파이썬 `account_refresher` 와 같은 규칙. 둘이 같은 알림을 받으니 체결마다 두 배가 된다. |
| 사용자 스트림 | `POST /fapi/v1/listenKey`(키만 · 서명 없음) | 연결마다 1회(보통 하루 1회 · 실패하면 5→10→…→300초 간격) · `PUT` 30분마다 | 같은 계좌의 listenKey 는 거래소가 하나로 돌려준다(파이썬과 공유됨). 닫기(DELETE)는 부르지 않는다(다른 쪽 연결이 끊긴다). |
| 손절선 닿음 → 청산 계획 조립 | 새 계좌(스트림 계좌를 쓸 수 있으면 0, 아니면 25요청) · 호가(`/fapi/v1/ticker/bookTicker`, 허브 값이 2초를 넘었을 때만) · 필터(`/fapi/v1/exchangeInfo`, 하루 캐시) · 1분봉(`/fapi/v1/klines`, 허브 변동성이 없을 때만) · 펀딩(`/fapi/v1/premiumIndex`) · 수수료(`/fapi/v1/commissionRate`, 서명 · 1시간 캐시) | 그림자에서는 청산을 «못 함»으로 받아 **30초마다 다시** 조립한다 → 닿아 있는 칸 하나당 분당 약 2회 | 파이썬은 실제로 청산해 포지션이 사라지므로 보통 한두 번에 끝난다. rtorder 는 다음 틱에 포지션이 없어진 것을 보고 정리로 넘어간다(정리도 그림자). |
| 시작 정리(`wait_cleanup`) | 없음 | 0 | 그림자에서는 남은 대기 주문 조회·취소를 보내지 않고 목록에만 적는다. |

허브(`RTORDER_HUB`)가 끊기면 호가·변동성이 REST 로 떨어진다. 청산 조립 때만 묻기 때문에 늘어나는 양은 작다. 감시 5분봉·1초 체결가는 REST 로 대신하지 않는다.

## 5. 키가 필요한 자리

- 서명 GET: 계좌 읽기 전부(`/fapi/v2/account`·`positionRisk`·`userTrades`·`openOrders`·`openAlgoOrders`) · `commissionRate`.
- 키만(서명 없음): `listenKey` POST·PUT.
- 쓰기(주문·취소·레버리지·algoOrder): 그림자에서는 보내지 않는다. 보내려 하면 panic.
- 키가 없는 빌드(기본)에서는 위 요청이 모두 원본의 «키 없음»과 같은 결과가 된다. 계좌는 `ok=false` 로 오고, 감시는 «못 읽음 = 모름»으로 무장을 그대로 둔다.

## 6. 바깥에서 읽을 것 (응답 서버 rtserve)

- `app::run(cfg, mk, stop, hub)` 에 넘긴 `Arc<userstream::Hub>` 를 같이 쥐면 된다. `hub.acct_seq()` 는 원본 `/api/events` 의 `acct_seq` 와 같다(체결 알림 뒤 계좌를 다시 읽는 데 성공할 때마다 +1, 처음은 0). `hub.us` 는 원본 `user_stream` 의 connected·last_event_at·error 이다(seq 는 `acct_seq()` 로 읽는다).
- rtorder 를 **다른 프로세스**로 돌리면 이 값은 보이지 않는다. 같은 프로세스에 넣거나, 이 값을 내보낼 길(파일이나 소켓)을 따로 정해야 한다.
