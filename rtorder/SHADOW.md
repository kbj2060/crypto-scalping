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

## 7. 주문·계좌 주소 14개를 응답 서버가 부르는 법 (`route::Front::call`)

- 부르는 쪽이 할 일은 넷이다. ① 받은 메서드·경로·쿼리(받은 순서 그대로 `(키, 값)` 목록)를 넘긴다. ② 돌아온 `Reply` 의 status·content_type·headers·body 를 그대로 내보낸다. ③ `Reply::task` 를 `runner::run_task` 로 띄운다. ④ `Reply::log` 가 있으면 at(KST 밀리초)·ms·ua(200자)·ip(CF-Connecting-IP → X-Forwarded-For → 접속 주소)를 더해 `order_requests.jsonl` 에 한 줄 덧붙인다.
- 주소 표, 405·404 글자, 주문 잠금 409, confirm·실행 스위치, 본문 글자(파이썬 json.dumps), 200 의 no-cache, server_ms 는 모두 rtorder 안에서 처리한다.
- `Front` 하나, `DeskCell` 하나, `Hub` 하나를 프로세스에 하나씩 두고 모든 요청이 같이 쓴다.
- 요청 하나가 `DeskCell` 을 끝까지 쥔다. 그래서 조립(REST 왕복, 보통 수백 ms) 동안에는 다른 주문 요청이 기다린다. 잠금 안 주소는 기다리지 않고 409 로 돌아온다. 원본은 await 사이에 다른 핸들러가 돌았다.

## 8. 프로세스 배치에 필요한 사실 (결정은 조정자 몫)

### 공통 사실
- 손절은 거래소에 걸려 있지 않다. 비상 스탑은 2026-11-21 사용자 결정으로 없앴다. 손절은 서버의 감시 틱(2초)뿐이다. **감시가 돌지 않는 동안에는 손절이 없다.** 거래소에 남는 것은 익절 지정가(dbtp)와 걸려 있던 진입 지정가(dbwt)뿐이다.
- 무장(`manual_bracket_state.json`)과 지정가 대기 복구 파일(`manual_limit_wait_*.json`)은 디스크에 남는다. 진입·청산·대기 칸의 phase, 도는 작업(PEG 따라가기, 청산 따라가기), 주문 잠금은 메모리에만 있어 프로세스가 죽으면 사라진다. 원본 파이썬도 같다.
- 재시작 복구(`app::run`)는 실행 스위치가 켜져 있을 때 이 순서로 돈다. ① 남은 대기·따라가기 주문(dbwt·dbpg)을 지운다. ② 복구 파일로 무장한다. ③ 감시를 시작한다. ④ 사용자 스트림이 붙자마자 계좌를 한 번 읽는다.
  - 청산 도중 죽었으면 무장이 남아 있다. 가격이 아직 손절선 너머면 다음 틱에 다시 쏜다.
  - 진입 따라가기 도중 죽었으면 걸려 있던 GTX 는 ①에서 지워진다. 이미 체결된 몫은 ②의 복구 파일 선으로 무장된다(복구 파일이 없으면 무장 없이 남는다 -- 원본과 같다).
- 파이썬 서버의 실전송과 rtorder 실전송을 같은 계좌에서 동시에 켜면 손절이 두 번 나간다. 그림자일 때만 둘을 같이 돌릴 수 있다.
- 지금 `app::run` 의 감시 `Watcher` 와 접수용 `Desk` 는 서로 다른 객체다. 무장은 같은 파일이라 공유되지만, 메모리 칸(진입·청산 칸의 working 여부)은 공유되지 않는다. 원본은 같은 dict 를 본다(감시 청산 중이면 접수가 409). **어느 배치든 붙이기 전에** `app::run` 이 `DeskCell` 하나를 받아 감시 틱도 `desk.w` 로 돌게 바꿔야 한다. 이 조각에서는 바꾸지 않았다.

### A. 같은 프로세스 (rtserve 가 rtorder 라이브러리를 링크해 `app::run` 을 tokio 작업으로 띄움)

| 넘길 것 | 길 |
|---|---|
| acct_seq · 스트림 상태 | `Arc<Hub>` 를 같이 쥔다(`hub.acct_seq()` · `hub.us`) -- 복사 없음 |
| 계좌 캐시 | `Wired.cache`(`Arc<Mutex<Option<(시각, 계좌)>>>`). 지금은 `app::run` 안에서 만든다 -- 바깥에서 넘겨받게 한 줄 고쳐야 /api/binance-account 가 감시와 같은 캐시를 본다 |
| 대기·진입·청산 칸 | `Arc<DeskCell>` 하나(위 공통 사실의 고칠 것) |
| 주문 잠금 | `Front` 하나 |

- 한쪽이 죽으면: 응답 서버 코드의 panic 은 그 요청 작업만 죽는다(tokio 작업 단위 · panic=unwind 일 때). 프로세스가 죽거나 OOM·SIGKILL 이면 **화면과 손절 감시가 같이 멈춘다** -- 지금 파이썬과 같은 위험이다.
- 응답 서버 재배포(재시작)마다 감시도 재시작한다. 재시작 동안(보통 수 초) 손절이 없고, 메모리 칸·도는 작업도 잃는다.
- 사용자 스트림·REST 연결을 하나로 쓴다. 따로 둘 때보다 요청 수가 늘지 않는다.

### B. 따로 (rtorder 를 별도 실행 파일로 · rtserve 는 127.0.0.1 로 넘김)

| 넘길 것 | 길 |
|---|---|
| 주문·계좌 14주소 | rtorder 가 127.0.0.1 HTTP 로 `Front::call` 을 그대로 내고 rtserve 가 프록시한다(본문 글자 그대로). 접수 HTTP 서버는 아직 없다 -- hyper 한 겹(rtserve 와 같은 판)이 새로 필요하다 |
| acct_seq | rtorder 의 작은 주소(예: GET /api/acct-seq) 를 rtserve 가 SSE 1초 주기마다 읽거나, rtorder 가 바뀔 때 미는 연결 하나 |
| 계좌 캐시 | rtorder 안에만 둔다. /api/binance-account 는 프록시라 사본이 없다 |
| 대기 칸 | rtorder 안에만(프록시) |

- rtserve 가 죽으면 손절 감시는 계속 돈다. 화면·주문 접수만 멈춘다.
- rtorder 가 죽으면 손절 감시가 멈춘다. rtserve 의 주문 주소는 502 가 된다. supervisor 재기동이 필요하다(재시작 복구는 위와 같다).
- 화면 배포와 주문 경로 재시작이 갈라진다. 화면을 자주 고쳐도 감시는 재시작하지 않는다.
- 비용: 프록시 왕복(127.0.0.1, 1ms 미만) 한 번이 더해진다. acct_seq 가 주기 읽기면 화면이 최대 1초 늦게 안다(원본도 SSE 1초 주기라 같다).

## 9. «그림자 기록» 제안서 -- 파이썬에 넣을 기록 코드의 모양 (고치지 않았다 · 서빙 코드라 사용자 결정)

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
